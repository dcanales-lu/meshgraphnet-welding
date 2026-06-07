"""Tests for the training orchestration (``training.train``)."""

from __future__ import annotations

import json

import torch

import training.train as train_mod
from models.meshgraphnet import MeshGraphNet
from training.train import TrainConfig, config_from_args, train
from tests.test_graph_builder import _make_sim


def _make_dataset(root, n_sims=4):
    raw = root / "raw"
    raw.mkdir(parents=True)
    params = [(2000, 0.02), (2600, 0.015), (1800, 0.025), (2200, 0.018)]
    for i in range(n_sims):
        power, speed = params[i % len(params)]
        _make_sim(t_end=0.2, dt=0.05, power=power, speed=speed).save_npz(raw / f"sim_{i:03d}")


def _tiny_config(tmp_path) -> TrainConfig:
    return TrainConfig(
        data_root=str(tmp_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        rollout_pred_dir=str(tmp_path / "rollout_pred"),
        epochs=2,
        batch_size=2,
        lr=1e-3,
        noise_std=2.0,
        hidden_dim=16,
        num_processing_steps=2,
        num_mlp_layers=1,
        val_fraction=0.25,
        test_fraction=0.25,
        val_every=1,
        scheduler="onecycle",
        progress_bar=False,
        device="cpu",
    )


def test_train_runs_and_checkpoints(tmp_path):
    _make_dataset(tmp_path, n_sims=4)
    cfg = _tiny_config(tmp_path)

    history = train(cfg)

    ckpt = tmp_path / "checkpoints"
    assert (ckpt / "best_model.pt").exists()
    assert (ckpt / "config.json").exists()
    assert (ckpt / "stats.pt").exists()
    assert (ckpt / "history.json").exists()

    assert len(history["train_loss"]) == cfg.epochs
    assert len(history["val_rmse"]) >= 1
    assert history["best_epoch"] >= 1
    assert history["best_metric"] < float("inf")


def test_checkpoint_is_loadable_for_inference(tmp_path):
    _make_dataset(tmp_path, n_sims=4)
    cfg = _tiny_config(tmp_path)
    train(cfg)

    payload = torch.load(tmp_path / "checkpoints" / "best_model.pt", weights_only=False)
    assert payload["metric_name"] == "val_rollout_rmse"
    saved_cfg = TrainConfig(**payload["config"])
    model = MeshGraphNet(saved_cfg.model_config())
    model.load_state_dict(payload["model_state"])  # state dict matches the config

    stats = torch.load(tmp_path / "checkpoints" / "stats.pt", weights_only=False)
    assert "x_mean" in stats and "y_std" in stats


def test_config_json_roundtrip(tmp_path):
    cfg = _tiny_config(tmp_path)
    path = tmp_path / "cfg.json"
    cfg.save_json(path)
    reloaded = TrainConfig.from_json(path)
    assert reloaded.hidden_dim == cfg.hidden_dim
    assert reloaded.noise_std == cfg.noise_std
    assert reloaded.scheduler == cfg.scheduler


def test_cli_overrides_layer_over_json(tmp_path):
    base = TrainConfig(epochs=50, hidden_dim=64, use_layer_norm=True)
    cfg_path = tmp_path / "cfg.json"
    base.save_json(cfg_path)

    # CLI flags must override JSON values; unspecified fields keep JSON values.
    cfg = config_from_args(
        ["--config", str(cfg_path), "--epochs", "5", "--no-use_layer_norm"]
    )
    assert cfg.epochs == 5            # overridden
    assert cfg.hidden_dim == 64       # from JSON
    assert cfg.use_layer_norm is False  # boolean override


def test_unknown_scheduler_raises(tmp_path):
    _make_dataset(tmp_path, n_sims=3)
    cfg = _tiny_config(tmp_path)
    cfg.scheduler = "not_a_scheduler"
    try:
        train(cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_plateau_scheduler_runs_and_writes_last(tmp_path):
    _make_dataset(tmp_path, n_sims=4)
    cfg = _tiny_config(tmp_path)
    cfg.scheduler = "plateau"
    cfg.early_stop_patience = 5  # not reached in 2 epochs

    history = train(cfg)

    assert (tmp_path / "checkpoints" / "last_model.pt").exists()
    assert len(history["train_loss"]) == cfg.epochs
    assert history["stopped_early"] is False


def test_early_stopping_triggers(tmp_path, monkeypatch):
    _make_dataset(tmp_path, n_sims=4)
    cfg = _tiny_config(tmp_path)
    cfg.epochs = 20
    cfg.val_every = 1
    cfg.scheduler = "plateau"
    cfg.early_stop_patience = 3

    # Constant val metric: improves once (inf -> 5.0), then never again.
    monkeypatch.setattr(train_mod, "_validate_rollout", lambda *a, **k: {"mean": 5.0})

    history = train(cfg)

    # epoch1 improves; validations 2,3,4 don't -> stop at epoch 4 (well before 20).
    assert history["stopped_early"] is True
    assert len(history["train_loss"]) == 4
    assert history["best_epoch"] == 1
    assert history["best_metric"] == 5.0


def test_last_checkpoint_carries_resume_state(tmp_path):
    _make_dataset(tmp_path, n_sims=4)
    cfg = _tiny_config(tmp_path)
    cfg.scheduler = "plateau"
    train(cfg)

    last = torch.load(tmp_path / "checkpoints" / "last_model.pt", weights_only=False)
    for key in ("model_state", "optimizer_state", "scheduler_state",
                "epoch", "best_metric", "best_epoch", "epochs_since_improve",
                "history"):
        assert key in last
    assert last["epoch"] == cfg.epochs
    assert last["scheduler_state"] is not None


def test_resume_skips_completed_epochs(tmp_path, monkeypatch):
    _make_dataset(tmp_path, n_sims=4)

    cfg = _tiny_config(tmp_path)
    cfg.scheduler = "plateau"
    cfg.epochs = 2
    train(cfg)
    assert torch.load(
        tmp_path / "checkpoints" / "last_model.pt", weights_only=False
    )["epoch"] == 2

    # Count epoch executions on the resumed run: only epochs 3 and 4 should run.
    calls = {"n": 0}
    real_epoch = train_mod._train_one_epoch

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(train_mod, "_train_one_epoch", counting)

    cfg2 = _tiny_config(tmp_path)
    cfg2.scheduler = "plateau"
    cfg2.epochs = 4
    cfg2.resume = True
    history = train(cfg2)

    assert calls["n"] == 2  # resumed: only the 2 remaining epochs executed
    assert torch.load(
        tmp_path / "checkpoints" / "last_model.pt", weights_only=False
    )["epoch"] == 4
    assert len(history["train_loss"]) == 4  # 2 carried over + 2 new


def test_resume_without_checkpoint_starts_fresh(tmp_path):
    _make_dataset(tmp_path, n_sims=4)
    cfg = _tiny_config(tmp_path)
    cfg.scheduler = "plateau"
    cfg.resume = True  # no last_model.pt yet -> warn and start from scratch

    history = train(cfg)

    assert len(history["train_loss"]) == cfg.epochs
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()
