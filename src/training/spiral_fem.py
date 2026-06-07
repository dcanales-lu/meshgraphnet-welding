"""FEM **ground-truth** simulation on the spiral path, for comparison with the
MeshGraphNet autoregressive rollout (:mod:`training.spiral_rollout`).

This runs the real :class:`~simulation.thermal_solver.TransientThermalSolver`
along the *identical* constant-speed Archimedean spiral — reusing
:func:`training.spiral_rollout.build_spiral_trajectory` so the FEM and the
surrogate see exactly the same path — on the same plate/mesh and process
parameters as the inference stress test. It exports a ParaView-ready XDMF/H5
time series to ``data/output/spiral_fem.{xdmf,h5}`` (the fixed, co-located
format) and, if the rollout prediction (``data/output/spiral_rollout.npz``) is
present, reports the global FEM-vs-surrogate RMSE.

Run::

    uv run python -m src.training.spiral_fem
    uv run python -m src.training.spiral_fem --max_steps 25 --quiet   # quick timing

The default geometry / Goldak / BC values **mirror**
``training.spiral_rollout`` so the two outputs are directly comparable.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from simulation.thermal_solver import (
    GoldakParams,
    MaterialProperties,
    Robin,
    SolverConfig,
    Trajectory,
    TransientThermalSolver,
    rectangular_plate,
)
from training.spiral_rollout import AMBIENT, build_spiral_trajectory

log = logging.getLogger("spiral_fem")


class ArrayTrajectory(Trajectory):
    """Trajectory backed by precomputed per-step arrays (queried on the time grid).

    The thermal solver evaluates the moving source only at on-grid times
    ``t_n = n·dt``, so indexing by ``round(s/dt)`` reproduces the *exact*
    positions / tangents / normals used by the rollout — guaranteeing the FEM
    ground truth and the surrogate prediction follow the identical weld path.
    """

    def __init__(self, pos: np.ndarray, tan: np.ndarray, nor: np.ndarray, dt: float):
        self.pos = np.asarray(pos, dtype=float)
        self.tan = np.asarray(tan, dtype=float)
        self.nor = np.asarray(nor, dtype=float)
        self.dt = float(dt)
        self.n = len(self.pos)

    def _idx(self, s: float) -> int:
        return int(min(max(round(s / self.dt), 0), self.n - 1))

    def position(self, s: float) -> np.ndarray:
        return self.pos[self._idx(s)].copy()

    def tangent(self, s: float) -> np.ndarray:
        return self.tan[self._idx(s)].copy()

    def normal(self, s: float) -> np.ndarray:
        return self.nor[self._idx(s)].copy()


def build_solver(args) -> tuple:
    """Construct the FEM solver on the spiral path; returns ``(solver, n_frames)``."""
    mesh = rectangular_plate(args.plate_size, args.plate_size, args.resolution, args.resolution)
    theta_max = 2.0 * np.pi * args.turns
    max_radius = args.plate_size / 2.0 - args.margin
    coil_spacing_b = max_radius / theta_max
    center = (args.plate_size / 2.0, args.plate_size / 2.0)
    pos, tan, nor, times = build_spiral_trajectory(
        center, coil_spacing_b, theta_max, args.speed, args.dt
    )

    n_frames = len(times)
    if args.max_steps > 0:
        n_frames = min(n_frames, args.max_steps + 1)

    traj = ArrayTrajectory(pos, tan, nor, args.dt)
    material = MaterialProperties(thickness=args.thickness, T_ambient=AMBIENT)
    goldak = GoldakParams(
        power=args.power, efficiency=args.efficiency,
        a=args.goldak_b, b=args.goldak_b, c_f=args.c_f, c_r=args.c_r, f_f=args.f_f,
    )
    bcs = {m: Robin(h_conv=args.h_conv, T_inf=AMBIENT, emissivity=0.0) for m in mesh.boundaries}
    cfg = SolverConfig(
        dt=args.dt, t_end=(n_frames - 1) * args.dt, snapshot_every=1, verbose=not args.quiet
    )
    return TransientThermalSolver(mesh, material, goldak, traj, bcs, cfg), n_frames


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spiral_fem",
        description="FEM ground-truth spiral simulation (mirrors spiral_rollout) for comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # geometry / trajectory (mirror spiral_rollout defaults)
    p.add_argument("--plate_size", type=float, default=0.2)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--turns", type=float, default=3.0)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--speed", type=float, default=0.01)
    p.add_argument("--dt", type=float, default=0.05)
    # process / Goldak
    p.add_argument("--power", type=float, default=2500.0)
    p.add_argument("--efficiency", type=float, default=0.8)
    p.add_argument("--goldak_b", type=float, default=3.0e-3)
    p.add_argument("--c_f", type=float, default=3.0e-3)
    p.add_argument("--c_r", type=float, default=6.0e-3)
    p.add_argument("--f_f", type=float, default=0.6)
    p.add_argument("--thickness", type=float, default=5.0e-3)
    p.add_argument("--h_conv", type=float, default=15.0)
    # io / runtime
    p.add_argument("--output_dir", type=Path, default=Path("data/output"))
    p.add_argument("--output_name", type=str, default="spiral_fem")
    p.add_argument("--max_steps", type=int, default=0,
                   help="cap number of solver steps (0 = full spiral; use for timing)")
    p.add_argument("--quiet", action="store_true", help="suppress the solver progress bar")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    # skfem logs a few INFO lines per Picard iteration; silence over long runs.
    logging.getLogger("skfem").setLevel(logging.WARNING)
    args = build_arg_parser().parse_args(argv)
    solver, n_frames = build_solver(args)
    n_steps = n_frames - 1
    log.info(
        "FEM spiral: %d nodes | %d steps | dt %.3f s | t_end %.2f s | net power %.0f W%s",
        solver.N, n_steps, args.dt, n_steps * args.dt, solver.goldak.net_power,
        "  [TIMING RUN]" if args.max_steps > 0 else "",
    )

    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0
    per_step = elapsed / max(n_steps, 1)
    log.info(
        "FEM finished in %.1f s (%.3f s/step) | T range [%.1f, %.1f] K",
        elapsed, per_step, float(result.temperature.min()), float(result.temperature.max()),
    )

    if args.max_steps > 0:
        # Timing run: extrapolate to the full spiral and stop (don't pollute outputs).
        full = len(build_spiral_trajectory(
            (args.plate_size / 2.0, args.plate_size / 2.0),
            (args.plate_size / 2.0 - args.margin) / (2.0 * np.pi * args.turns),
            2.0 * np.pi * args.turns, args.speed, args.dt,
        )[3]) - 1
        log.info(
            "ESTIMATE: full spiral = %d steps -> ~%.1f min at %.3f s/step "
            "(Picard cost grows once the pool forms, so treat as a lower bound).",
            full, full * per_step / 60.0, per_step,
        )
        return

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    xdmf = result.save_xdmf(out_dir / args.output_name)
    npz = result.save_npz(out_dir / args.output_name)
    log.info("Saved ParaView time-series: %s (+ .h5)", xdmf)
    log.info("Saved FEM npz:              %s", npz)

    # Optional comparison vs the surrogate rollout (same path/mesh/steps).
    pred_npz = out_dir / "spiral_rollout.npz"
    if pred_npz.exists():
        pred = np.load(pred_npz)["temperature"]
        if pred.shape == result.temperature.shape:
            diff = pred - result.temperature
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            per_step_rmse = np.sqrt(np.mean(diff ** 2, axis=1))
            log.info(
                "FEM vs surrogate: global RMSE = %.3f K | final-frame RMSE = %.3f K "
                "(%d frames x %d nodes)",
                rmse, float(per_step_rmse[-1]), diff.shape[0], diff.shape[1],
            )
        else:
            log.warning(
                "spiral_rollout.npz shape %s != FEM %s; skipping RMSE "
                "(re-run spiral_rollout with matching args).",
                tuple(pred.shape), tuple(result.temperature.shape),
            )


if __name__ == "__main__":
    main()
