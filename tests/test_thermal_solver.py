"""Tests and runnable example for the 2D transient thermal welding solver.

Run the fast unit tests with::

    uv run pytest tests/test_thermal_solver.py -q

Run the full example (exports .npz, XDMF and a PNG snapshot to data/raw/)::

    uv run python tests/test_thermal_solver.py
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.thermal_solver import (
    GoldakParams,
    LinearTrajectory,
    MaterialProperties,
    Robin,
    SimulationResult,
    SolverConfig,
    TransientThermalSolver,
    goldak_flux,
    rectangular_plate,
)


def _nearest_node(coords: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.sum((coords - point) ** 2, axis=1)))


def _build_solver(**cfg_overrides) -> TransientThermalSolver:
    """A small rectangular-plate straight-weld setup used by several tests."""
    width, height = 0.10, 0.05
    mesh = rectangular_plate(width, height, nx=30, ny=15)
    material = MaterialProperties()
    goldak = GoldakParams(power=2000.0, efficiency=0.8, c_f=3e-3, c_r=6e-3)
    trajectory = LinearTrajectory(
        start=(0.02, height / 2), end=(0.08, height / 2), speed=0.01
    )
    bcs = {edge: Robin(h_conv=15.0, T_inf=300.0) for edge in ("left", "right", "bottom", "top")}
    cfg = SolverConfig(dt=0.05, t_end=0.5, snapshot_every=1, verbose=False)
    for key, val in cfg_overrides.items():
        setattr(cfg, key, val)
    return TransientThermalSolver(mesh, material, goldak, trajectory, bcs, cfg)


def test_goldak_flux_front_rear_shape():
    """The 2D Goldak flux peaks at the source centre and is rear-elongated."""
    params = GoldakParams(c_f=2e-3, c_r=6e-3, b=3e-3)
    pos = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    thickness = 5e-3

    q_center = goldak_flux(np.array([0.0]), np.array([0.0]), pos, tangent, normal, params, thickness)
    q_front = goldak_flux(np.array([3e-3]), np.array([0.0]), pos, tangent, normal, params, thickness)
    q_rear = goldak_flux(np.array([-3e-3]), np.array([0.0]), pos, tangent, normal, params, thickness)

    assert q_center[0] > 0.0
    # A point 3 mm ahead sits ~1.5 c_f away (steep falloff); the same distance
    # behind sits only ~0.5 c_r away, so the rear flux is much larger.
    assert q_rear[0] > q_front[0]


def test_straight_weld_runs():
    """Solver runs, produces well-shaped finite output, and heats the plate."""
    solver = _build_solver()
    result = solver.run()

    n_snap = result.temperature.shape[0]
    assert result.temperature.shape == (n_snap, solver.N)
    assert result.coords.shape == (solver.N, 2)
    assert result.cells.shape[1] == 3
    assert np.all(np.isfinite(result.temperature))

    # Temperature must rise well above ambient near the torch.
    T_final = result.temperature[-1]
    assert T_final.max() > MaterialProperties().T_ambient + 100.0

    # The hottest node should be close to the instantaneous source position.
    hot_node = int(np.argmax(T_final))
    dist = np.linalg.norm(result.coords[hot_node] - result.source_position[-1])
    assert dist < 0.015  # within ~2.5 * c_r of the source


def test_thermal_field_trails_behind_source():
    """With c_r > c_f the hot zone trails behind the moving torch (rear-hot)."""
    solver = _build_solver(t_end=0.6, dt=0.05)
    result = solver.run()

    T_final = result.temperature[-1]
    pos = result.source_position[-1]
    tangent = result.source_tangent[-1]
    d = 5e-3
    rear_node = _nearest_node(result.coords, pos - d * tangent)
    front_node = _nearest_node(result.coords, pos + d * tangent)
    assert T_final[rear_node] > T_final[front_node]


def test_dirichlet_boundary_is_held():
    """A Dirichlet edge stays pinned at its prescribed value."""
    from simulation.thermal_solver import Dirichlet

    width, height = 0.10, 0.05
    mesh = rectangular_plate(width, height, nx=20, ny=10)
    bcs = {
        "left": Dirichlet(value=300.0),
        "right": Robin(h_conv=15.0, T_inf=300.0),
        "bottom": Robin(h_conv=15.0, T_inf=300.0),
        "top": Robin(h_conv=15.0, T_inf=300.0),
    }
    solver = TransientThermalSolver(
        mesh,
        MaterialProperties(),
        GoldakParams(),
        LinearTrajectory((0.02, height / 2), (0.08, height / 2), 0.01),
        bcs,
        SolverConfig(dt=0.05, t_end=0.3, verbose=False),
    )
    result = solver.run()
    left_mask = result.boundary_masks["left"]
    assert np.allclose(result.temperature[-1][left_mask], 300.0, atol=1e-6)


def test_npz_roundtrip(tmp_path):
    """save_npz / load_npz reproduces all fields exactly."""
    result = _build_solver(t_end=0.2, dt=0.05).run()
    path = result.save_npz(tmp_path / "sim")
    assert path.exists()

    loaded = SimulationResult.load_npz(path)
    assert np.allclose(loaded.temperature, result.temperature)
    assert np.array_equal(loaded.cells, result.cells)
    assert np.allclose(loaded.source_position, result.source_position)
    assert set(loaded.boundary_masks) == set(result.boundary_masks)
    for marker, mask in result.boundary_masks.items():
        assert np.array_equal(loaded.boundary_masks[marker], mask)


def test_xdmf_export(tmp_path):
    """XDMF time-series export writes the .xdmf + .h5 pair."""
    pytest.importorskip("h5py")
    result = _build_solver(t_end=0.2, dt=0.05).run()
    path = result.save_xdmf(tmp_path / "sim")
    assert path.exists()
    assert path.stat().st_size > 0


def _demo():
    """Richer run exporting .npz, XDMF and a PNG snapshot into data/raw/."""
    from pathlib import Path

    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    width, height = 0.12, 0.06
    mesh = rectangular_plate(width, height, nx=60, ny=30)
    solver = TransientThermalSolver(
        mesh,
        MaterialProperties(),
        GoldakParams(power=2500.0, efficiency=0.8, c_f=3e-3, c_r=6e-3, b=3e-3),
        LinearTrajectory((0.02, height / 2), (0.10, height / 2), speed=0.008),
        {e: Robin(h_conv=20.0, T_inf=300.0, emissivity=0.3) for e in ("left", "right", "bottom", "top")},
        SolverConfig(dt=0.05, t_end=8.0, snapshot_every=2, verbose=True),
    )
    result = solver.run()

    npz_path = result.save_npz(out_dir / "plate_straight_weld")
    xdmf_path = result.save_xdmf(out_dir / "plate_straight_weld")
    print(f"peak temperature: {result.temperature.max():.1f} K")
    print(f"snapshots: {result.temperature.shape[0]}  nodes: {solver.N}")
    print(f"saved: {npz_path}\n       {xdmf_path}")

    try:
        import matplotlib.pyplot as plt

        T_final = result.temperature[-1]
        fig, ax = plt.subplots(figsize=(8, 4))
        tpc = ax.tripcolor(
            result.coords[:, 0], result.coords[:, 1], result.cells, T_final, shading="gouraud"
        )
        ax.plot(*result.source_position[-1], "wo", markersize=6, label="torch")
        ax.set_aspect("equal")
        ax.set_title("Final temperature field [K]")
        fig.colorbar(tpc, ax=ax)
        ax.legend()
        png_path = out_dir / "plate_straight_weld.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        print(f"       {png_path}")
    except Exception as exc:  # pragma: no cover - visualization is optional
        print(f"(skipped PNG: {exc})")


if __name__ == "__main__":
    _demo()
