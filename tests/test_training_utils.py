"""Tests for training utilities: simulation-level split + noise injection."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.graph_builder import WeldingGraphDataset
from training.utils import (
    TemperatureNoiseInjection,
    TrainingConfig,
    TransformedSubset,
    WindowedSubset,
    collate_windows,
    dynamic_temperature_noise,
    make_split_datasets,
    simulation_ids_from_dataset,
    split_by_simulation,
)
from tests.test_graph_builder import _make_sim


# ---------------------------------------------------------------------------
# Lightweight stub exposing the WeldingGraphDataset `_index` contract so the
# splitter can be tested without running the FEM solver.
# ---------------------------------------------------------------------------
class _StubDataset:
    def __init__(self, sims_to_snaps: dict):
        self._index = []
        for sim, n in sims_to_snaps.items():
            for t in range(n):
                self._index.append((f"sim_{sim}", sim, t))

    def __len__(self):
        return len(self._index)


# ---------------------------------------------------------------------------
# 1. Simulation-level split
# ---------------------------------------------------------------------------
def test_split_is_simulation_level_and_disjoint():
    ds = _StubDataset({i: (i % 4) + 2 for i in range(10)})  # 10 sims, varied len
    ids = simulation_ids_from_dataset(ds)
    split = split_by_simulation(ds, val_fraction=0.2, test_fraction=0.2, seed=0)

    # Folds partition all graph indices exactly once.
    all_idx = split.train_indices + split.val_indices + split.test_indices
    assert sorted(all_idx) == list(range(len(ds)))

    # Simulations are disjoint across folds.
    assert set(split.train_sims).isdisjoint(split.val_sims)
    assert set(split.train_sims).isdisjoint(split.test_sims)
    assert set(split.val_sims).isdisjoint(split.test_sims)

    # Every graph's simulation lies in the fold it was assigned to.
    for idx in split.val_indices:
        assert ids[idx] in set(split.val_sims)
    for idx in split.test_indices:
        assert ids[idx] in set(split.test_sims)

    # No simulation is split across folds: all snapshots of a sim share a fold.
    for sim in np.unique(ids):
        sim_graphs = set(np.where(ids == sim)[0].tolist())
        in_train = sim_graphs <= set(split.train_indices)
        in_val = sim_graphs <= set(split.val_indices)
        in_test = sim_graphs <= set(split.test_indices)
        assert in_train or in_val or in_test


def test_split_fraction_counts_and_reproducibility():
    ds = _StubDataset({i: 3 for i in range(10)})
    s0 = split_by_simulation(ds, 0.2, 0.2, seed=0)
    assert (len(s0.train_sims), len(s0.val_sims), len(s0.test_sims)) == (6, 2, 2)

    # Same seed -> identical split; different seed -> generally different.
    s0b = split_by_simulation(ds, 0.2, 0.2, seed=0)
    assert s0.val_sims == s0b.val_sims
    s1 = split_by_simulation(ds, 0.2, 0.2, seed=123)
    assert (s0.val_sims, s0.test_sims) != (s1.val_sims, s1.test_sims)


def test_split_small_corpus_nudges_to_nonempty():
    ds = _StubDataset({0: 4, 1: 4, 2: 4})  # 3 sims, tiny fractions
    split = split_by_simulation(ds, val_fraction=0.15, test_fraction=0.15, seed=0)
    # Each fold should get at least one simulation (and training stays non-empty).
    assert len(split.train_sims) >= 1
    assert len(split.val_sims) == 1
    assert len(split.test_sims) == 1


# ---------------------------------------------------------------------------
# 2. Noise injection
# ---------------------------------------------------------------------------
def _toy_graph(n=600, node_dim=16):
    x = torch.randn(n, node_dim)
    x[:, 0] = 300.0 + 50.0 * torch.randn(n)  # temperature-like column
    y = torch.randn(n, 1)
    return x, y


def test_noise_target_correction_invariant():
    """T̃ + ΔT_target == T + ΔT (i.e. the true next-step temperature is preserved)."""
    x, y = _toy_graph()
    torch.manual_seed(0)
    out = TemperatureNoiseInjection(sigma=5.0)(Data(x=x.clone(), y=y.clone()))

    # T̃ + ỹ == T + y  <=>  the +η/−η cancel exactly.
    assert torch.allclose(out.x[:, 0] + out.y[:, 0], x[:, 0] + y[:, 0], atol=1e-3)
    # The temperature perturbation equals minus the target adjustment.
    # (Tolerance accounts for float32 round-off: recovering eta from the ~300 K
    # temperature baseline loses ~1e-4 precision; the target y is O(1) and keeps
    # full precision, so dT and -dy differ at the float32 cancellation level.)
    dT = out.x[:, 0] - x[:, 0]
    dy = out.y[:, 0] - y[:, 0]
    assert torch.allclose(dT, -dy, atol=1e-2)
    # Only the temperature column is perturbed.
    assert torch.allclose(out.x[:, 1:], x[:, 1:])


def test_noise_statistics():
    x, y = _toy_graph(n=20000)
    torch.manual_seed(1)
    out = TemperatureNoiseInjection(sigma=4.0)(Data(x=x.clone(), y=y.clone()))
    eta = out.x[:, 0] - x[:, 0]
    assert abs(float(eta.mean())) < 0.2          # zero-mean
    assert abs(float(eta.std()) - 4.0) < 0.3     # std ~ sigma


def test_noise_disabled_paths_are_identity():
    x, y = _toy_graph()
    for t in (
        TemperatureNoiseInjection(sigma=0.0),         # sigma off
        TemperatureNoiseInjection(sigma=5.0, enabled=False),  # disabled
    ):
        out = t(Data(x=x.clone(), y=y.clone()))
        assert torch.equal(out.x, x)
        assert torch.equal(out.y, y)


def test_noise_is_reproducible_with_generator():
    x, y = _toy_graph()
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    o1 = TemperatureNoiseInjection(3.0, generator=g1)(Data(x=x.clone(), y=y.clone()))
    o2 = TemperatureNoiseInjection(3.0, generator=g2)(Data(x=x.clone(), y=y.clone()))
    assert torch.allclose(o1.x, o2.x) and torch.allclose(o1.y, o2.y)


# ---------------------------------------------------------------------------
# 3. End-to-end integration with the real dataset
# ---------------------------------------------------------------------------
def test_make_split_datasets_train_noisy_val_clean(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for i, (power, speed) in enumerate([(2000, 0.02), (2600, 0.015), (1800, 0.025)]):
        _make_sim(t_end=0.2, power=power, speed=speed).save_npz(raw / f"sim_{i:03d}")

    dataset = WeldingGraphDataset(root=tmp_path)  # no transform
    cfg = TrainingConfig(val_fraction=0.34, test_fraction=0.34, noise_std=5.0)
    normalizer = dataset.make_normalizer()
    subsets = make_split_datasets(dataset, cfg, normalizer)

    # Folds cover the whole dataset.
    total = len(subsets["train"]) + len(subsets["val"]) + len(subsets["test"])
    assert total == len(dataset)
    assert isinstance(subsets["train"], TransformedSubset)

    # Training fold injects fresh noise -> two accesses differ.
    a, b = subsets["train"][0], subsets["train"][0]
    assert not torch.allclose(a.x, b.x)

    # Validation fold has no noise -> deterministic across accesses.
    v1, v2 = subsets["val"][0], subsets["val"][0]
    assert torch.allclose(v1.x, v2.x)

    # Normalization is applied (temperature column no longer at the ~300 K scale).
    assert v1.x[:, 0].abs().mean() < 50.0

    # The PyG DataLoader batches the subsets.
    batch = next(iter(DataLoader(subsets["train"], batch_size=2, shuffle=False)))
    assert batch.x.shape[1] == 12 and batch.y.shape[1] == 1


# ---------------------------------------------------------------------------
# 3. Dynamic (temperature-proportional) noise
# ---------------------------------------------------------------------------
def test_dynamic_temperature_noise_scales_with_excess():
    gen = torch.Generator().manual_seed(0)
    # Node 0: melt-pool excess 1400 K; node 1: far-field (T == T_inf).
    T = torch.tensor([1700.0, 300.0])
    T_inf = torch.tensor([300.0, 300.0])
    n = 8000
    Tg = T.expand(n, 2).contiguous()
    Tig = T_inf.expand(n, 2).contiguous()

    out = dynamic_temperature_noise(Tg, Tig, beta=0.03, floor=0.0, generator=gen)
    std = (out - Tg).std(dim=0)

    assert abs(std[0].item() - 0.03 * 1400) < 5.0   # ~42 K near the melt pool
    assert std[1].item() < 1.0                       # ~0 K in the cold far-field
    assert (out - Tg).mean().abs().item() < 2.0      # zero-mean perturbation

    # No-op when both knobs are off.
    assert torch.equal(dynamic_temperature_noise(Tg, Tig, beta=0.0, floor=0.0), Tg)


# ---------------------------------------------------------------------------
# 4. Windowed dataset for push-forward training
# ---------------------------------------------------------------------------
def test_windowed_subset_validity_and_collate(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for i in range(3):
        _make_sim(t_end=0.3, dt=0.05).save_npz(raw / f"sim_{i:03d}")
    ds = WeldingGraphDataset(root=tmp_path)
    sims = sorted({si for (_, si, _) in ds._index})

    k = 3
    win = WindowedSubset(ds, sims, k=k)
    assert len(win) > 0

    for w in win:
        assert len(w) == k
        # Every window is K consecutive snapshots of a *single* simulation.
        assert len({int(g.sim_id) for g in w}) == 1
        times = [float(g.time) for g in w]
        assert times[0] < times[1] < times[2]

    # A window must not start within (K-1) steps of a simulation's end, so a sim
    # with `m` graphs contributes exactly `m-(K-1)` windows.
    from collections import Counter
    per_sim = Counter(si for (_, si, _) in ds._index)
    expected = sum(max(m - (k - 1), 0) for m in per_sim.values())
    assert len(win) == expected

    # collate -> K step-aligned batches, node-count-consistent across steps.
    batch = collate_windows([win[0], win[1 % len(win)]])
    assert len(batch) == k
    n_nodes = batch[0].num_nodes
    assert all(b.num_nodes == n_nodes for b in batch)
