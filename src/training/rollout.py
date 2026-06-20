"""Autoregressive rollout evaluation for the MeshGraphNet welding surrogate.

At inference the model is given only the **initial** temperature field ``T^0``
and the **scheduled** Goldak trajectory; it must predict the entire temperature
evolution autoregressively, feeding its own prediction back as the next input
state. This module runs that loop and scores it against the FEM ground truth.

Key design points
-----------------
* **Feature parity with training.** Inputs are built with the exact same
  :func:`data.graph_builder.build_graph_sequence` used to train the model, so the
  node/edge feature layout, the analytical Goldak field, the co-moving relative
  coordinates and the welding speed are computed identically.
* **Only temperature is rolled.** The trajectory is known in advance, so every
  *time-dependent* feature (``q_Goldak``, ``[dx', dy']``, ``speed``) is correct
  for all steps up front. During the loop we overwrite **only** the temperature
  column (column 0) with the model's running prediction — exactly the "dynamic
  feature recomputation" requirement, done vectorized and ahead of time because
  the schedule is deterministic.
* **No absolute positioning in the network.** The model is called as
  ``model(x, edge_index, edge_attr)``; ``x`` carries only relative/physical
  features (12-d, no coordinates). Coordinates are used solely to *compute*
  ``q_Goldak`` / ``[dx', dy']`` (inside ``build_graph_sequence``), never fed in.
* **Normalization.** Node features are standardized with the saved
  :class:`NormalizeTransform` constants (``stats.pt``); the predicted normalized
  ``ΔT`` is mapped back to kelvin with ``NormalizeTransform.inverse_y``. Edge
  features are passed raw, matching the training contract (the normalizer never
  touched ``edge_attr``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from data.graph_builder import NODE_FEATURE_NAMES, NormalizeTransform, build_graph_sequence
from simulation.thermal_solver import SimulationResult

#: Column of the node-feature vector holding temperature (rolled autoregressively).
TEMPERATURE_INDEX = NODE_FEATURE_NAMES.index("T")

NormalizerLike = Union[NormalizeTransform, dict, str, Path]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RolloutResult:
    """Predicted vs. ground-truth temperature history and rollout metrics."""

    predicted_temperature: np.ndarray      # (S, N)
    ground_truth_temperature: np.ndarray   # (S, N)
    times: np.ndarray                      # (S,)
    per_step_rmse: np.ndarray              # (S,) RMSE over nodes at each step
    rmse: float                            # scalar RMSE over all nodes & steps

    coords: np.ndarray                     # (N, 2)
    cells: np.ndarray                      # (M, 3)
    boundary_masks: dict
    source_position: np.ndarray            # (S, 2)
    source_tangent: np.ndarray             # (S, 2)
    source_normal: np.ndarray              # (S, 2)
    source_power: np.ndarray               # (S,)
    metadata: dict

    def to_simulation_result(self) -> SimulationResult:
        """Wrap the *predicted* field as a :class:`SimulationResult`.

        This lets the prediction reuse the existing visualization tooling
        (``save_xdmf`` for ParaView, ``save_npz`` for ML formats).
        """
        return SimulationResult(
            coords=self.coords,
            cells=self.cells,
            times=self.times,
            temperature=self.predicted_temperature,
            boundary_masks=self.boundary_masks,
            source_position=self.source_position,
            source_tangent=self.source_tangent,
            source_normal=self.source_normal,
            source_power=self.source_power,
            metadata=self.metadata,
        )

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Save a self-contained prediction/ground-truth comparison ``.npz``."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            predicted_temperature=self.predicted_temperature,
            ground_truth_temperature=self.ground_truth_temperature,
            times=self.times,
            per_step_rmse=self.per_step_rmse,
            rmse=np.array(self.rmse),
            coords=self.coords,
            cells=self.cells,
            source_position=self.source_position,
        )
        return path


# ---------------------------------------------------------------------------
# Normalizer coercion
# ---------------------------------------------------------------------------
def _as_normalizer(normalizer: NormalizerLike, device: torch.device) -> NormalizeTransform:
    """Coerce a NormalizeTransform / stats dict / stats.pt path to a normalizer
    with all constant tensors placed on ``device``."""
    if isinstance(normalizer, NormalizeTransform):
        import copy
        nt = copy.copy(normalizer)   # shallow copy — tensors reassigned below, original stays on CPU
    elif isinstance(normalizer, dict):
        nt = NormalizeTransform(normalizer)
    else:  # path to stats.pt
        nt = NormalizeTransform(torch.load(Path(normalizer), weights_only=False))

    # Move constants onto the target device so the loop stays device-resident.
    nt.x_mean = nt.x_mean.to(device)
    nt.x_std = nt.x_std.to(device)
    nt.y_mean = nt.y_mean.to(device)
    nt.y_std = nt.y_std.to(device)
    nt.mask = nt.mask.to(device)
    return nt


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_autoregressive_rollout(
    model: nn.Module,
    result: SimulationResult,
    normalizer: NormalizerLike,
    device: Optional[Union[str, torch.device]] = None,
    sim_id: int = 0,
    save_path: Optional[Union[str, Path]] = None,
) -> RolloutResult:
    """Roll the model forward over a full simulation and score it vs. FEM.

    Parameters
    ----------
    model:
        Trained :class:`~models.meshgraphnet.MeshGraphNet` (any ``nn.Module``
        with ``forward(x, edge_index, edge_attr) -> (N, 1)``).
    result:
        The evaluation :class:`SimulationResult` (ground truth + metadata +
        scheduled trajectory). Must carry solver metadata (Goldak/BC params).
    normalizer:
        A :class:`NormalizeTransform`, a stats dict, or a path to ``stats.pt``.
    device:
        Torch device; defaults to the model's device.
    sim_id:
        Tag passed through to graph construction (cosmetic).
    save_path:
        If given, save the comparison ``.npz`` there.

    Returns
    -------
    RolloutResult
    """
    if not result.metadata:
        raise ValueError(
            "SimulationResult has no metadata; regenerate it with the current "
            "solver so the trajectory/Goldak parameters are available."
        )

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    nt = _as_normalizer(normalizer, device)
    mask = nt.mask.bool()

    was_training = model.training
    model.eval()
    model.to(device)

    # --- Precompute the full input-feature stack (training-identical) ---
    # graphs[t].x already contains the correct time-dependent features for step
    # t (Goldak field, co-moving coords, speed); column 0 holds the *ground
    # truth* T^t, which we overwrite with the rolled prediction during the loop.
    graphs = build_graph_sequence(result, sim_id=sim_id)
    n_steps = len(graphs)  # = S - 1
    feats = torch.stack([g.x for g in graphs]).to(device)        # (S-1, N, 12)
    edge_index = graphs[0].edge_index.to(device)
    edge_attr = graphs[0].edge_attr.to(device)                   # raw, unnormalized

    gt = torch.as_tensor(result.temperature, dtype=torch.float32, device=device)

    # --- Autoregressive loop (only temperature is fed back) ---
    T_cur = gt[0].clone()                       # (N,) start from true T^0
    history = [T_cur.clone()]
    for t in range(n_steps):
        x = feats[t].clone()
        x[:, TEMPERATURE_INDEX] = T_cur          # overwrite baseline with rolled T

        # Standardize node features (masked continuous columns only).
        x_norm = x.clone()
        x_norm[:, mask] = (x[:, mask] - nt.x_mean[mask]) / nt.x_std[mask]

        # Predict normalized ΔT, then de-normalize to physical kelvin.
        dT_norm = model(x_norm, edge_index, edge_attr)           # (N, 1)
        dT = nt.inverse_y(dT_norm).squeeze(-1)                   # (N,)

        T_cur = T_cur + dT
        history.append(T_cur.clone())

    pred = torch.stack(history)                  # (S, N)

    # --- Metrics (vectorized) ---
    pred_np = pred.detach().cpu().numpy()
    gt_np = result.temperature
    diff = pred_np - gt_np
    per_step_rmse = np.sqrt(np.mean(diff ** 2, axis=1))          # (S,)
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if was_training:
        model.train()

    rollout = RolloutResult(
        predicted_temperature=pred_np,
        ground_truth_temperature=gt_np,
        times=result.times,
        per_step_rmse=per_step_rmse,
        rmse=rmse,
        coords=result.coords,
        cells=result.cells,
        boundary_masks=result.boundary_masks,
        source_position=result.source_position,
        source_tangent=result.source_tangent,
        source_normal=result.source_normal,
        source_power=result.source_power,
        metadata=result.metadata,
    )

    if save_path is not None:
        rollout.save_npz(save_path)

    return rollout


def export_rollout(
    rollout: RolloutResult,
    out_dir: Union[str, Path],
    name: str,
    save_ground_truth: bool = True,
) -> Path:
    """Export a rollout to ParaView-ready XDMF/H5 (+ a comparison ``.npz``).

    Writes, under ``out_dir``:

    * ``<name>_pred.xdmf`` / ``.h5`` — the **predicted** temperature history;
    * ``<name>_gt.xdmf`` / ``.h5`` — the FEM **ground truth** (for side-by-side
      comparison in ParaView), unless ``save_ground_truth`` is False;
    * ``<name>_rollout.npz`` — both fields + per-step / global RMSE.

    Uses :meth:`SimulationResult.save_xdmf`, which co-locates the HDF5 payload
    next to the ``.xdmf`` so the pair loads cleanly in ParaView.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout.to_simulation_result().save_xdmf(out_dir / f"{name}_pred")
    if save_ground_truth:
        gt = SimulationResult(
            coords=rollout.coords,
            cells=rollout.cells,
            times=rollout.times,
            temperature=rollout.ground_truth_temperature,
            boundary_masks=rollout.boundary_masks,
            source_position=rollout.source_position,
            source_tangent=rollout.source_tangent,
            source_normal=rollout.source_normal,
            source_power=rollout.source_power,
            metadata=rollout.metadata,
        )
        gt.save_xdmf(out_dir / f"{name}_gt")
    rollout.save_npz(out_dir / f"{name}_rollout")
    return out_dir


def evaluate_rollouts(
    model: nn.Module,
    results: Dict[str, SimulationResult],
    normalizer: NormalizerLike,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, RolloutResult]:
    """Convenience: run :func:`run_autoregressive_rollout` over several sims.

    Returns a ``{name: RolloutResult}`` map; the aggregate mean RMSE is available
    via ``np.mean([r.rmse for r in out.values()])``.
    """
    return {
        name: run_autoregressive_rollout(model, res, normalizer, device=device)
        for name, res in results.items()
    }
