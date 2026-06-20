"""Tests for the PyG graph data pipeline (``data.graph_builder``)."""

from __future__ import annotations

import numpy as np
import torch

from data.graph_builder import (
    NUM_NODE_FEATURES,
    NORMALIZE_MASK,
    NormalizeTransform,
    WeldingGraphDataset,
    build_edges,
    build_graph_sequence,
)
from simulation.thermal_solver import (
    Dirichlet,
    GoldakParams,
    LinearTrajectory,
    MaterialProperties,
    Robin,
    SimulationResult,
    SolverConfig,
    TransientThermalSolver,
    rectangular_plate,
)


def _make_sim(
    t_end: float = 0.3,
    dt: float = 0.05,
    power: float = 2000.0,
    speed: float = 0.02,
    c_f: float = 3e-3,
    c_r: float = 6e-3,
    b: float = 3e-3,
) -> SimulationResult:
    """Small straight-weld simulation used across the tests.

    Process parameters are exposed so tests can build genuinely different
    simulations (needed to exercise per-feature normalization of the global
    process-parameter columns).
    """
    width, height = 0.10, 0.05
    mesh = rectangular_plate(width, height, nx=12, ny=6)
    bcs = {
        "left": Dirichlet(value=300.0),
        "right": Robin(h_conv=15.0, T_inf=300.0, emissivity=0.2),
        "bottom": Robin(h_conv=15.0, T_inf=300.0),
        "top": Robin(h_conv=15.0, T_inf=300.0),
    }
    solver = TransientThermalSolver(
        mesh,
        MaterialProperties(),
        GoldakParams(power=power, efficiency=0.8, c_f=c_f, c_r=c_r, b=b),
        LinearTrajectory((0.02, height / 2), (0.08, height / 2), speed=speed),
        bcs,
        SolverConfig(dt=dt, t_end=t_end, verbose=False),
    )
    return solver.run()


def test_metadata_roundtrip(tmp_path):
    """Solver embeds self-contained metadata that survives save/load."""
    result = _make_sim()
    assert "goldak" in result.metadata
    assert "boundary_specs" in result.metadata
    path = result.save_npz(tmp_path / "sim")
    reloaded = SimulationResult.load_npz(path)
    assert reloaded.metadata["goldak"]["c_r"] == result.metadata["goldak"]["c_r"]
    assert reloaded.metadata["boundary_specs"]["left"]["type"] == "dirichlet"
    assert reloaded.metadata["boundary_specs"]["right"]["type"] == "robin"


def test_build_edges_bidirectional_no_self_loops():
    mesh = rectangular_plate(0.1, 0.05, 4, 2)
    edge_index = build_edges(mesh.t.T)
    assert edge_index.shape[0] == 2
    # No self loops.
    assert not np.any(edge_index[0] == edge_index[1])
    # Bidirectional: even count and the reversed set matches the forward set.
    assert edge_index.shape[1] % 2 == 0
    fwd = set(map(tuple, edge_index.T.tolist()))
    rev = set((j, i) for (i, j) in fwd)
    assert fwd == rev


def test_graph_sequence_shapes_and_target():
    result = _make_sim()
    graphs = build_graph_sequence(result)
    S = result.temperature.shape[0]
    N = result.coords.shape[0]

    assert len(graphs) == S - 1
    g0 = graphs[0]
    assert g0.x.shape == (N, NUM_NODE_FEATURES)
    assert g0.edge_index.shape[0] == 2
    assert g0.edge_attr.shape == (g0.edge_index.shape[1], 3)
    assert g0.y.shape == (N, 1)
    assert g0.pos.shape == (N, 2)
    assert torch.all(torch.isfinite(g0.x))

    # Target is the next-step temperature increment.
    expected = result.temperature[1] - result.temperature[0]
    assert torch.allclose(g0.y.squeeze(1), torch.tensor(expected, dtype=torch.float32))

    # Feature column 0 is the current temperature.
    assert torch.allclose(
        g0.x[:, 0], torch.tensor(result.temperature[0], dtype=torch.float32)
    )


def test_edge_attr_distance_consistency():
    result = _make_sim()
    g = build_graph_sequence(result)[0]
    dxy = g.edge_attr[:, :2]
    dist = g.edge_attr[:, 2]
    assert torch.allclose(dist, torch.linalg.norm(dxy, dim=1), atol=1e-6)
    assert torch.all(dist > 0)  # no zero-length edges


def test_goldak_feature_peaks_near_source():
    result = _make_sim()
    g = build_graph_sequence(result)[2]
    q = g.x[:, 1].numpy()
    assert np.all(q >= 0.0)
    hottest = int(np.argmax(q))
    # The max Goldak input must sit near the instantaneous source position.
    src = result.source_position[2]
    dist = np.linalg.norm(result.coords[hottest] - src)
    assert dist < 0.01


def test_bc_onehot_is_valid_and_dirichlet_overrides():
    result = _make_sim()
    g = build_graph_sequence(result)[0]
    onehot = g.x[:, 6:9]
    # Exactly one node type per node.
    assert torch.allclose(onehot.sum(dim=1), torch.ones(g.num_nodes))
    # The left edge is Dirichlet -> those nodes flagged dirichlet, zero conv values.
    left_mask = torch.tensor(result.boundary_masks["left"])
    assert torch.all(g.x[left_mask, 7] == 1.0)
    assert torch.all(g.x[left_mask, 9:12] == 0.0)
    # Robin nodes carry the convection coefficient.
    robin_nodes = g.x[:, 8] == 1.0
    assert torch.all(g.x[robin_nodes, 9] == 15.0)


def test_no_absolute_coordinates_in_features():
    """Sanity: positions are stored separately, never as a node feature column."""
    result = _make_sim()
    g = build_graph_sequence(result)[1]
    coords = torch.tensor(result.coords, dtype=torch.float32)
    for c in range(NUM_NODE_FEATURES):
        col = g.x[:, c]
        assert not torch.allclose(col, coords[:, 0]) and not torch.allclose(
            col, coords[:, 1]
        )


def test_dataset_processing_and_normalization(tmp_path):
    # Two raw simulations under <root>/raw.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    # Two simulations with different process parameters (better generalization
    # coverage and exercises normalization of the global-parameter columns).
    s1 = _make_sim(t_end=0.2, power=2000.0, speed=0.02, c_f=3e-3, c_r=6e-3)
    s2 = _make_sim(t_end=0.3, power=2600.0, speed=0.015, c_f=2.5e-3, c_r=7e-3)
    s1.save_npz(raw_dir / "sim_000")
    s2.save_npz(raw_dir / "sim_001")

    dataset = WeldingGraphDataset(root=tmp_path)
    expected = (s1.temperature.shape[0] - 1) + (s2.temperature.shape[0] - 1)
    assert len(dataset) == expected
    assert (tmp_path / "processed" / "stats.pt").exists()

    sample = dataset[0]
    assert sample.x.shape[1] == NUM_NODE_FEATURES

    norm = dataset.make_normalizer()
    xs = torch.cat([norm(dataset[i]).x for i in range(len(dataset))], dim=0)
    col_std = xs.std(dim=0, unbiased=False)

    # Columns that always vary in these sims must be ~zero-mean / unit-std.
    varying = torch.tensor([0, 1, 2, 3, 9, 10, 11])  # T, q, dx', dy', h, eps, T_inf
    assert torch.all((col_std[varying] - 1.0).abs() < 1e-1)
    assert torch.all(xs[:, varying].mean(dim=0).abs() < 1e-1)
    # No column is over-amplified; one-hot/constant columns stay <= ~unit std.
    assert torch.all(col_std < 1.0 + 1e-1)

    # Inverse target transform round-trips.
    y_norm = norm(dataset[0]).y
    y_phys = norm.inverse_y(y_norm)
    assert torch.allclose(y_phys, dataset[0].y, atol=1e-3)


def test_dataset_skips_reprocessing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    _make_sim(t_end=0.2).save_npz(raw_dir / "sim_000")

    WeldingGraphDataset(root=tmp_path)
    stats_path = tmp_path / "processed" / "stats.pt"
    first_mtime = stats_path.stat().st_mtime_ns

    # Re-instantiation must not reprocess (stats.pt untouched).
    WeldingGraphDataset(root=tmp_path)
    assert stats_path.stat().st_mtime_ns == first_mtime
