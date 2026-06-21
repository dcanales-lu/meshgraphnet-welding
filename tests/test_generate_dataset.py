"""Tests for the diversified dataset generator (``simulation.generate_dataset``).

Most tests are fast (mesh construction, geometry predicates, trajectory
confinement, simulation specification). A single small end-to-end test runs the
solver on a coarse mesh and confirms the produced ``.npz`` is consumable by the
graph pipeline — the contract that actually matters downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.graph_builder import NUM_NODE_FEATURES, build_graph_sequence
from simulation.generate_dataset import (
    GEOMETRY_KINDS,
    _GEOMETRY_BUILDERS,
    _path_inside,
    build_simulation,
    main,
    make_holed,
    make_lshape,
    make_rectangular,
    sample_trajectory,
)
from simulation.thermal_solver import GoldakParams, SimulationResult, TransientThermalSolver

H_EL = 6.0e-3  # coarse mesh keeps these tests fast


def test_rectangular_has_only_outer_markers():
    geom = make_rectangular(np.random.default_rng(0), 0.06, 0.05, H_EL)
    assert geom.kind == "rect"
    assert set(geom.mesh.boundaries) == {"left", "right", "bottom", "top"}
    assert geom.mesh.p.shape[1] > 0 and geom.mesh.t.shape[1] > 0
    assert geom.removed == []


@pytest.mark.parametrize("builder,kind", [(make_lshape, "lshape"), (make_holed, "hole")])
def test_cut_geometries_tag_inner_boundary(builder, kind):
    geom = builder(np.random.default_rng(1), 0.07, 0.06, H_EL)
    assert geom.kind == kind
    # The cut introduces interior boundary facets tagged "inner".
    assert "inner" in geom.mesh.boundaries
    assert len(geom.removed) >= 1


def test_lshape_removes_corner_nodes():
    """No surviving node should sit strictly inside the removed top-right block."""
    geom = make_lshape(np.random.default_rng(2), 0.08, 0.06, H_EL)
    _, (x0, y0, x1, y1) = geom.removed[0]
    pts = geom.mesh.p
    # Centroid-based removal leaves a jagged ~element-sized edge, so only nodes
    # *well* inside the removed block must be gone (margin > one element).
    m = 1.5 * H_EL
    deep_inside = (
        (pts[0] > x0 + m) & (pts[0] < x1 - m) & (pts[1] > y0 + m) & (pts[1] < y1 - m)
    )
    assert not deep_inside.any()


def test_is_solid_predicate_respects_circle_and_box():
    geom = make_holed(np.random.default_rng(3), 0.07, 0.06, H_EL)
    shape, (cx, cy, r) = geom.removed[0]
    assert shape == "circle"
    # Hole centre is not solid; a point well outside everything is.
    assert not bool(geom.is_solid(np.array([cx, cy]), margin=0.0))
    assert not bool(geom.is_solid(np.array([-1.0, -1.0])))


@pytest.mark.parametrize("kind", GEOMETRY_KINDS)
def test_sampled_trajectory_stays_inside_material(kind):
    """Across many seeds, the weld path never leaves the solid region."""
    goldak = GoldakParams()
    margin = 1.1 * max(goldak.b, goldak.c_f)
    for seed in range(25):
        rng = np.random.default_rng(seed)
        geom = _GEOMETRY_BUILDERS[kind](rng, 0.07, 0.06, H_EL)
        traj, t_end, traj_kind = sample_trajectory(geom, rng, speed=6.0e-3, goldak=goldak)
        assert t_end > 0.0
        assert traj_kind in {"straight", "diagonal", "sinusoid", "arc"}
        assert _path_inside(traj, t_end, geom, margin)


def test_build_simulation_returns_solver_and_provenance():
    solver, prov = build_simulation(np.random.default_rng(7), h_el=H_EL, t_cool_max=2.0)
    assert isinstance(solver, TransientThermalSolver)
    assert prov["geometry"]["kind"] in GEOMETRY_KINDS
    assert prov["trajectory"]["kind"] in {"straight", "diagonal", "sinusoid", "arc"}
    assert 0.01 <= prov["trajectory"]["dt"] <= 0.05
    assert prov["boundary"]["h_conv"] > 0.0
    assert solver.cfg.cool_to_relaxed is True


def test_build_simulation_is_reproducible():
    """Same seed -> identical sampled spec (provenance)."""
    _, a = build_simulation(np.random.default_rng(11), h_el=H_EL, t_cool_max=2.0)
    _, b = build_simulation(np.random.default_rng(11), h_el=H_EL, t_cool_max=2.0)
    assert a == b


def test_build_simulation_bc_stratification():
    """Forced strata populate radiation / Dirichlet deterministically."""
    _, prov = build_simulation(
        np.random.default_rng(3), h_el=H_EL, t_cool_max=2.0,
        bc_strata={"radiation": True, "dirichlet": True},
    )
    assert prov["boundary"]["emissivity"] > 0.0
    assert prov["boundary"]["dirichlet_edge"] is not None
    _, prov2 = build_simulation(
        np.random.default_rng(3), h_el=H_EL, t_cool_max=2.0,
        bc_strata={"radiation": False, "dirichlet": False},
    )
    assert prov2["boundary"]["emissivity"] == 0.0
    assert prov2["boundary"]["dirichlet_edge"] is None


def test_end_to_end_npz_consumable_by_graph_pipeline(tmp_path, monkeypatch):
    """`main` writes named .npz files that build valid (N,12) graphs."""
    # Shrink the sampling ranges so this plumbing test stays fast (the real
    # corpus uses large plates / long welds; here we only exercise the pipeline).
    import simulation.generate_dataset as gd
    monkeypatch.setattr(gd, "WIDTH_RANGE", (0.040, 0.050))
    monkeypatch.setattr(gd, "HEIGHT_RANGE", (0.040, 0.050))
    monkeypatch.setattr(gd, "SPEED_RANGE", (8.0e-3, 10.0e-3))
    main(
        [
            "--num_train", "1",
            "--num_val", "1",
            "--element_size", "7e-3",
            "--t_cool_max", "1.0",
            "--workers", "1",
            "--data_root", str(tmp_path),
        ]
    )
    raw = tmp_path / "raw"
    train = raw / "sim_train_001.npz"
    val = raw / "sim_val_001.npz"
    assert train.exists() and val.exists()

    result = SimulationResult.load_npz(train)
    assert "generation" in result.metadata
    assert "goldak" in result.metadata and "boundary_specs" in result.metadata
    graphs = build_graph_sequence(result, sim_id=0)
    assert len(graphs) == result.temperature.shape[0] - 1
    g0 = graphs[0]
    assert g0.x.shape[1] == NUM_NODE_FEATURES
    assert g0.y.shape[1] == 1
    assert g0.edge_attr.shape[1] == 3
    assert bool(g0.x.isfinite().all())
