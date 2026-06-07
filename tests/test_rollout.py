"""Tests for the autoregressive rollout engine (``training.rollout``)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data.graph_builder import (
    NODE_FEATURE_NAMES,
    WeldingGraphDataset,
    build_graph_sequence,
)
from models.meshgraphnet import MeshGraphNet, MeshGraphNetConfig
from simulation.thermal_solver import SimulationResult
from training.rollout import TEMPERATURE_INDEX, run_autoregressive_rollout
from tests.test_graph_builder import _make_sim


def _dataset_with_one_sim(tmp_path):
    """Process a single tiny simulation and return (result, normalizer)."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    result = _make_sim(t_end=0.3, dt=0.05)
    result.save_npz(raw / "sim_000")
    dataset = WeldingGraphDataset(root=tmp_path)
    return result, dataset.make_normalizer()


# ---------------------------------------------------------------------------
# Stub models for exact, model-independent verification of the loop.
# ---------------------------------------------------------------------------
class _OracleModel(nn.Module):
    """Returns the normalized true ΔT for each step, in call order."""

    def __init__(self, normalized_dT_seq):
        super().__init__()
        self.seq = normalized_dT_seq
        self.i = 0

    def forward(self, x, edge_index, edge_attr):
        out = self.seq[self.i]
        self.i += 1
        return out


class _RecordingModel(nn.Module):
    """Records each input x and returns a constant normalized prediction.

    The default ``normalized_fill`` is chosen so the *physical* ΔT is zero
    (``inverse_y(fill) == 0``), keeping the rolled temperature pinned at T^0.
    """

    def __init__(self, normalized_fill: float = 0.0):
        super().__init__()
        self.seen = []
        self.fill = float(normalized_fill)

    def forward(self, x, edge_index, edge_attr):
        assert x.shape[1] == len(NODE_FEATURE_NAMES)  # no absolute coords appended
        self.seen.append(x.clone())
        return torch.full((x.size(0), 1), self.fill)


def test_rollout_shapes_and_initial_state(tmp_path):
    result, normalizer = _dataset_with_one_sim(tmp_path)
    cfg = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2)
    model = MeshGraphNet(cfg)

    out = run_autoregressive_rollout(model, result, normalizer, device="cpu")

    S, N = result.temperature.shape
    assert out.predicted_temperature.shape == (S, N)
    assert out.per_step_rmse.shape == (S,)
    assert np.isfinite(out.rmse)
    # The first frame is the true initial condition -> zero error.
    assert np.allclose(out.predicted_temperature[0], result.temperature[0])
    assert out.per_step_rmse[0] == 0.0


def test_oracle_reproduces_ground_truth(tmp_path):
    """Feeding the true normalized ΔT each step must recover the FEM field."""
    result, normalizer = _dataset_with_one_sim(tmp_path)

    # True physical increments -> normalize the same way the dataset target was.
    gt = result.temperature
    y_mean = float(normalizer.y_mean)
    y_std = float(normalizer.y_std)
    seq = []
    for t in range(gt.shape[0] - 1):
        dT_phys = torch.tensor(gt[t + 1] - gt[t], dtype=torch.float32).reshape(-1, 1)
        seq.append((dT_phys - y_mean) / y_std)  # normalized ΔT the oracle "predicts"

    out = run_autoregressive_rollout(_OracleModel(seq), result, normalizer, device="cpu")

    assert np.allclose(out.predicted_temperature, gt, atol=1e-2)
    assert out.rmse < 1e-2


def test_only_temperature_is_rolled_dynamic_features_preserved(tmp_path):
    """The loop overwrites only column T; q / co-moving / speed stay as built."""
    result, normalizer = _dataset_with_one_sim(tmp_path)
    # Predict normalized value that de-normalizes to ΔT = 0, so T stays at T^0.
    fill = -float(normalizer.y_mean) / float(normalizer.y_std)
    rec = _RecordingModel(normalized_fill=fill)

    out = run_autoregressive_rollout(rec, result, normalizer, device="cpu")

    # With zero predicted ΔT, temperature never changes -> stays at T^0 every step.
    T0 = torch.tensor(result.temperature[0], dtype=torch.float32)
    graphs = build_graph_sequence(result)
    xm = normalizer.x_mean
    xs = normalizer.x_std
    m = normalizer.mask.bool()
    ti = TEMPERATURE_INDEX
    other = [c for c in range(len(NODE_FEATURE_NAMES)) if c != ti]
    for t, x_seen in enumerate(rec.seen):
        # Temperature column (de-normalized) is the rolled state (constant T^0 here).
        temp_phys = x_seen[:, ti] * xs[ti] + xm[ti]
        assert torch.allclose(temp_phys, T0, atol=1e-1)
        # Every non-temperature column must equal the freshly-built features for
        # step t (recomputed Goldak field, co-moving coords, speed, BC, params).
        # Compare in NORMALIZED space (all O(1)) to avoid the huge dynamic range
        # of physical columns like q_Goldak (~1e8).
        gx = graphs[t].x.clone()
        gx[:, m] = (gx[:, m] - xm[m]) / xs[m]
        assert torch.allclose(x_seen[:, other], gx[:, other], atol=1e-5)


def test_rollout_export_roundtrip(tmp_path):
    result, normalizer = _dataset_with_one_sim(tmp_path)
    model = MeshGraphNet(MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2))

    out = run_autoregressive_rollout(
        model, result, normalizer, device="cpu", save_path=tmp_path / "rollout"
    )

    # Comparison npz written and reloadable.
    npz = tmp_path / "rollout.npz"
    assert npz.exists()
    data = np.load(npz)
    assert data["predicted_temperature"].shape == result.temperature.shape

    # Predicted field re-wraps as a SimulationResult for visualization tooling.
    sim = out.to_simulation_result()
    assert isinstance(sim, SimulationResult)
    assert sim.temperature.shape == result.temperature.shape
    p = sim.save_npz(tmp_path / "pred_sim")
    assert SimulationResult.load_npz(p).metadata  # metadata carried through


def test_rollout_restores_model_training_mode(tmp_path):
    result, normalizer = _dataset_with_one_sim(tmp_path)
    model = MeshGraphNet(MeshGraphNetConfig(hidden_dim=16, num_processing_steps=1))
    model.train()
    run_autoregressive_rollout(model, result, normalizer, device="cpu")
    assert model.training  # eval() inside rollout is restored afterwards
