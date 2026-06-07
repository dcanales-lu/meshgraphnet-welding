"""Training orchestration for the MeshGraphNet welding surrogate.

Run headless (e.g. inside a RunPod docker container)::

    uv run python -m src.training.train --config config.json
    uv run python -m src.training.train --epochs 200 --noise_std 2.0 --hidden_dim 128

Pipeline wired here:

* load :class:`~data.graph_builder.WeldingGraphDataset` (processes raw ``.npz``
  simulations and writes ``stats.pt``);
* **simulation-level** train/val/test split (``training.utils``);
* PyG ``DataLoader`` over the train fold with **training-noise injection** baked
  into the fold transform (noise applied before normalization, train only);
* AdamW + LR schedule (``onecycle`` / ``cosine`` / ``plateau`` / ``none``),
  MSE on **normalized** ΔT targets;
* validation = full **autoregressive rollout** RMSE (``training.rollout``), not
  single-step loss — this is the checkpoint-selection metric;
* **early stopping** on rollout RMSE (``--early_stop_patience``, in validation
  units) for long, open-ended epoch budgets;
* ``best_model.pt`` (selection metric) + ``last_model.pt`` (full optimizer/
  scheduler state, written atomically every epoch) saved under ``checkpoints/``.
  ``--resume`` continues from ``last_model.pt`` — for preemptible (spot) pods.
  Resume is intended for ``plateau``/``cosine``; ``onecycle``'s fixed
  total-steps curve does not resume cleanly.

For a long RunPod run::

    uv run python -m src.training.train --config config.runpod.json
    uv run python -m src.training.train --config config.runpod.json --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from data.graph_builder import WeldingGraphDataset
from models.meshgraphnet import MeshGraphNet, MeshGraphNetConfig
from simulation.thermal_solver import SimulationResult
from training.rollout import export_rollout, run_autoregressive_rollout
from training.utils import (
    TrainingConfig,
    make_split_datasets,
    split_by_simulation,
)

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """All training/model/data hyperparameters (JSON- and CLI-configurable)."""

    # data / io
    data_root: str = "data"
    checkpoint_dir: str = "checkpoints"
    rollout_pred_dir: str = "data/output/rollout_pred"  # ParaView rollout exports
    num_export_sims: int = 2                            # val sims to export after training

    # optimization
    epochs: int = 50
    batch_size: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0              # 0 disables gradient clipping
    scheduler: str = "onecycle"        # "onecycle" | "cosine" | "plateau" | "none"

    # early stopping (in *validation* units; counts validations w/o improvement)
    early_stop_patience: int = 0       # 0 disables early stopping
    early_stop_min_delta: float = 0.0  # min rollout-RMSE (K) drop to count as better

    # resume from <checkpoint_dir>/last_model.pt (intended for plateau/cosine)
    resume: bool = False

    # training-noise injection (kelvin; applied to raw temperature, train only)
    noise_std: float = 5.0

    # model
    hidden_dim: int = 128
    num_processing_steps: int = 8
    num_mlp_layers: int = 2
    activation: str = "relu"
    use_layer_norm: bool = True
    aggregation: str = "sum"
    use_generic: bool = False  # enable GENERIC structure-preserving thermal head

    # simulation-level split
    val_fraction: float = 0.20
    test_fraction: float = 0.0
    split_seed: int = 0

    # loop / runtime
    val_every: int = 5
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    progress_bar: bool = True

    # -- (de)serialization --------------------------------------------------
    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        valid = {f.name for f in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            log.warning("Ignoring unknown config keys: %s", sorted(unknown))
        return cls(**{k: v for k, v in raw.items() if k in valid})

    def save_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    # -- derived configs ----------------------------------------------------
    def model_config(self) -> MeshGraphNetConfig:
        return MeshGraphNetConfig(
            node_in_dim=16, edge_in_dim=3, out_dim=1,
            hidden_dim=self.hidden_dim,
            num_mlp_layers=self.num_mlp_layers,
            num_processing_steps=self.num_processing_steps,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            aggregation=self.aggregation,
            use_generic=self.use_generic,
        )

    def data_config(self) -> TrainingConfig:
        return TrainingConfig(
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            split_seed=self.split_seed,
            noise_std=self.noise_std,
        )

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _val_simulations(dataset: WeldingGraphDataset, sim_ids: List[int]) -> Dict[str, SimulationResult]:
    """Load the ground-truth :class:`SimulationResult` for each validation sim.

    Validation scores the model with full autoregressive rollouts, which need
    the raw simulations (trajectory + ground truth), not the per-snapshot graphs.
    """
    raw_dir = Path(dataset.raw_dir)
    results = {}
    for sim_id in sim_ids:
        fname = dataset._raw_files[sim_id]
        results[fname] = SimulationResult.load_npz(raw_dir / fname)
    return results


def _build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Return ``(scheduler, mode)`` per the configured policy.

    ``mode`` controls *when* the scheduler is stepped:

    * ``"batch"`` — once per optimizer step (``onecycle``);
    * ``"epoch"`` — once per epoch (``cosine``);
    * ``"plateau"`` — on the validation metric, validation epochs only
      (``plateau`` / :class:`ReduceLROnPlateau`); the natural pairing for early
      stopping on a large epoch budget;
    * ``None`` — no scheduler.
    """
    if cfg.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr,
            total_steps=max(cfg.epochs * steps_per_epoch, 1),
        )
        return sched, "batch"
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        return sched, "epoch"
    if cfg.scheduler == "plateau":
        # Halve the LR after the val metric stalls for roughly half the
        # early-stopping window, so the LR drops a couple of times before we give
        # up. Falls back to a small fixed patience when early stopping is off.
        plateau_patience = max(1, cfg.early_stop_patience // 2) if cfg.early_stop_patience > 0 else 5
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=plateau_patience,
        )
        return sched, "plateau"
    if cfg.scheduler in ("none", "", None):
        return None, None
    raise ValueError(f"Unknown scheduler '{cfg.scheduler}'.")


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def _train_one_epoch(model, loader, optimizer, scheduler, sched_mode, device,
                     grad_clip, progress_bar, epoch, epochs) -> float:
    model.train()
    total, n = 0.0, 0

    iterator = loader
    if progress_bar:
        try:
            from tqdm import tqdm

            iterator = tqdm(loader, desc=f"epoch {epoch}/{epochs}", unit="batch", leave=False)
        except ImportError:
            pass

    for batch in iterator:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)
        loss = F.mse_loss(pred, batch.y)        # MSE on normalized ΔT
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None and sched_mode == "batch":
            scheduler.step()

        total += float(loss.detach()) * batch.num_graphs
        n += batch.num_graphs
        if progress_bar and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.3e}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    # Epoch-/plateau-level scheduler stepping happens centrally in ``train()``.
    return total / max(n, 1)


def _atomic_save(obj, path: Path) -> None:
    """Write a checkpoint via temp-file + ``os.replace`` so a preempted pod never
    leaves a half-written ``last_model.pt`` (the resume anchor) on disk."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def _validate_rollout(model, val_results, normalizer, device) -> Dict[str, float]:
    """Mean autoregressive-rollout RMSE over the validation simulations."""
    per_sim = {}
    for name, result in val_results.items():
        rollout = run_autoregressive_rollout(model, result, normalizer, device=device)
        per_sim[name] = rollout.rmse
    per_sim["mean"] = float(np.mean(list(per_sim.values()))) if per_sim else float("nan")
    return per_sim


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig) -> dict:
    """Run the full training loop. Returns a history dict."""
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    log.info("Loading dataset from %s ...", cfg.data_root)
    dataset = WeldingGraphDataset(root=cfg.data_root)   # processes raw -> processed, writes stats.pt
    normalizer = dataset.make_normalizer()
    stats = dataset.get_norm_stats()

    split = split_by_simulation(
        dataset, val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction, seed=cfg.split_seed,
    )
    log.info("%s", split.summary())

    subsets = make_split_datasets(dataset, cfg.data_config(), normalizer, split=split)
    train_loader = DataLoader(
        subsets["train"], batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_results = _val_simulations(dataset, split.val_sims)
    if not val_results:
        log.warning("No validation simulations; checkpointing on train loss instead.")

    # --- Model / optim ---
    model = MeshGraphNet(cfg.model_config()).to(device)
    log.info("MeshGraphNet on %s | %d params | %d processing steps",
             device, model.num_parameters(), cfg.num_processing_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler, sched_mode = _build_scheduler(optimizer, cfg, len(train_loader))

    # Persist config + normalization stats so the checkpoint is self-contained.
    cfg.save_json(ckpt_dir / "config.json")
    torch.save(stats, ckpt_dir / "stats.pt")

    # --- Loop state (restored from last_model.pt when --resume) ---
    history = {"train_loss": [], "val_rmse": [], "epoch": []}
    best_metric = float("inf")
    best_epoch = -1
    epochs_since_improve = 0     # validations without a min_delta improvement
    start_epoch = 1
    metric_name = "val_rollout_rmse" if val_results else "train_mse"

    last_path = ckpt_dir / "last_model.pt"
    if cfg.resume and last_path.exists():
        ckpt = torch.load(last_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_metric = ckpt.get("best_metric", best_metric)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        epochs_since_improve = ckpt.get("epochs_since_improve", 0)
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        log.info("Resumed from %s at epoch %d (best %s %.4f @ epoch %d).",
                 last_path, start_epoch, metric_name, best_metric, best_epoch)
    elif cfg.resume:
        log.warning("--resume set but %s not found; starting from scratch.", last_path)

    # --- Loop ---
    stopped_early = False
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, sched_mode, device,
            cfg.grad_clip, cfg.progress_bar, epoch, cfg.epochs,
        )

        do_val = val_results and (epoch % cfg.val_every == 0 or epoch == cfg.epochs)
        if do_val:
            val = _validate_rollout(model, val_results, normalizer, device)
            metric = val["mean"]
            history["val_rmse"].append(metric)
            history["epoch"].append(epoch)
            log.info("epoch %3d/%d | train_mse %.4e | val_rollout_rmse %.4f K%s",
                     epoch, cfg.epochs, train_loss, metric,
                     "  <- best" if metric < best_metric else "")
        else:
            metric = train_loss if not val_results else None
            log.info("epoch %3d/%d | train_mse %.4e", epoch, cfg.epochs, train_loss)

        history["train_loss"].append(train_loss)

        # Central scheduler stepping (per-batch onecycle already stepped above).
        if scheduler is not None:
            if sched_mode == "epoch":
                scheduler.step()
            elif sched_mode == "plateau" and metric is not None:
                scheduler.step(metric)

        # Checkpoint + early-stopping bookkeeping. ``metric`` is only set on
        # validation epochs (or every epoch when there is no val fold).
        if metric is not None:
            improved = metric < best_metric - cfg.early_stop_min_delta
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_since_improve = 0
                _atomic_save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(cfg),
                        "epoch": epoch,
                        "metric": best_metric,
                        "metric_name": metric_name,
                    },
                    ckpt_dir / "best_model.pt",
                )
            else:
                epochs_since_improve += 1

        # Persist full resume state every epoch (atomic; survives preemption).
        _atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "config": asdict(cfg),
                "epoch": epoch,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "epochs_since_improve": epochs_since_improve,
                "history": history,
            },
            last_path,
        )

        # Early stopping: stop once the val metric has stalled for `patience`
        # validations (only meaningful when we actually validate).
        if (cfg.early_stop_patience > 0 and metric is not None
                and epochs_since_improve >= cfg.early_stop_patience):
            log.info("Early stopping at epoch %d: no %s improvement (> %.3g K) "
                     "for %d validations. Best %.4f K @ epoch %d.",
                     epoch, metric_name, cfg.early_stop_min_delta,
                     epochs_since_improve, best_metric, best_epoch)
            stopped_early = True
            break

    history["best_metric"] = best_metric
    history["best_epoch"] = best_epoch
    history["stopped_early"] = stopped_early
    history["checkpoint_dir"] = str(ckpt_dir)

    # --- Export best-model rollouts to ParaView-ready XDMF/H5 (pred vs FEM) ---
    if val_results and cfg.num_export_sims > 0:
        best_path = ckpt_dir / "best_model.pt"
        if best_path.exists():
            model.load_state_dict(
                torch.load(best_path, weights_only=False)["model_state"]
            )
        export_dir = Path(cfg.rollout_pred_dir)
        names = list(val_results)[: cfg.num_export_sims]
        log.info("Exporting %d rollout prediction(s) to %s ...", len(names), export_dir)
        exported = []
        for name in names:
            stem = Path(name).stem
            rollout = run_autoregressive_rollout(
                model, val_results[name], normalizer, device=device
            )
            export_rollout(rollout, export_dir, stem)
            log.info("  %s -> rollout RMSE %.4f K", stem, rollout.rmse)
            exported.append(stem)
        history["exported_rollouts"] = exported

    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    log.info("Done. Best %s = %.4f at epoch %d. Checkpoints in %s",
             "val_rollout_rmse" if val_results else "train_mse",
             best_metric, best_epoch, ckpt_dir)
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Argparse with one flag per config field (CLI overrides JSON/defaults)."""
    p = argparse.ArgumentParser(description="Train the MeshGraphNet welding surrogate.")
    p.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    # `from __future__ import annotations` makes f.type a string (e.g. "int"), so
    # dispatch on the annotation name rather than the type object.
    type_map = {"int": int, "float": float, "str": str}
    for f in fields(TrainConfig):
        flag = f"--{f.name}"
        if f.type == "bool":
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument(flag, type=type_map.get(f.type, str), default=None)
    return p


def config_from_args(argv: Optional[List[str]] = None) -> TrainConfig:
    """Build a :class:`TrainConfig` from CLI args, layered over an optional JSON."""
    args = _build_arg_parser().parse_args(argv)
    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    for f in fields(TrainConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = config_from_args(argv)
    train(cfg)


if __name__ == "__main__":
    main()
