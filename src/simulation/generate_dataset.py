"""Diversified dataset generator for the MeshGraphNet welding surrogate.

This is the *fan-out* layer on top of :mod:`simulation.thermal_solver`. Where
the solver runs **one** transient welding simulation, this module samples a
whole **corpus** of physically-plausible-but-varied simulations and writes each
one as a self-contained ``.npz`` (the raw contract consumed by
:mod:`data.graph_builder`).

Why diversify
-------------
A MeshGraphNet only generalizes over the regimes it has seen. We therefore
randomize, within realistic welding bounds, every axis the network must learn
to be invariant / responsive to:

* **Geometry** — rectangular plates, **L-shaped** plates (re-entrant corner),
  and plates with one or more **circular cut-outs**. This forces the model to
  learn boundary behaviour on shapes whose edges are *not* axis-aligned simple
  rectangles, exercising the relative-coordinate / edge-displacement inductive
  bias.
* **Trajectory** — straight, diagonal, sinusoidal, and circular-arc weld paths.
  All curved paths are :class:`~simulation.thermal_solver.ParametricTrajectory`
  objects with *analytic* derivatives so the Goldak orientation kinematics
  (tangent/normal frame) stay smooth.
* **Process parameters** — gross power, arc efficiency (hence net power
  ``eta*P``), travel speed, and the Goldak ellipsoid semi-axes.
* **Boundary conditions** — randomized convection coefficient ``h`` and ambient
  temperature ``T_inf`` (Robin response), with an occasional fixed-temperature
  (Dirichlet) edge so the ``bc_dirichlet`` node-type feature is populated too.

Storage
-------
Each simulation is saved as ``data/raw/sim_train_NNN.npz`` /
``sim_val_NNN.npz``. The solver already embeds Goldak/thickness/BC metadata; we
additionally stamp a ``"generation"`` block (geometry + trajectory descriptors,
seed) into ``metadata`` for full provenance.

CLI
---
::

    uv run python -m src.simulation.generate_dataset --num_train 20 --num_val 5

See ``--help`` for resolution / time-stepping / reproducibility knobs.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from skfem import MeshTri

from simulation.thermal_solver import (
    Dirichlet,
    GoldakParams,
    LinearTrajectory,
    MaterialProperties,
    ParametricTrajectory,
    Robin,
    SolverConfig,
    TransientThermalSolver,
    Trajectory,
)

# ---------------------------------------------------------------------------
# Sampling ranges (SI units, kelvin). Realistic mild-steel arc-welding regimes.
# ---------------------------------------------------------------------------
#: Gross arc power [W]; net power deposited is ``efficiency * power``.
POWER_RANGE = (1500.0, 3500.0)
EFFICIENCY_RANGE = (0.70, 0.90)
#: Travel speed [m/s] (3-10 mm/s).
SPEED_RANGE = (3.0e-3, 10.0e-3)

#: Goldak ellipsoid semi-axes [m].
B_RANGE = (2.0e-3, 3.5e-3)        # transverse half-width (along normal)
C_F_RANGE = (2.0e-3, 3.5e-3)      # front semi-axis (along tangent)
C_R_FACTOR_RANGE = (1.6, 2.4)     # rear semi-axis = factor * c_f (trailing lobe)
F_F_RANGE = (0.5, 0.7)            # front heat fraction

#: Material / boundary randomization.
THICKNESS_RANGE = (4.0e-3, 8.0e-3)
T_AMBIENT_RANGE = (290.0, 320.0)
H_CONV_RANGE = (5.0, 40.0)
#: Emissivity choices (weighted toward 0 — radiation is costly and small at
#: the cool boundaries, but a few nonzero samples populate the feature).
EMISSIVITY_CHOICES = (0.0, 0.0, 0.0, 0.2, 0.35)
#: Probability that one outer edge is held at a fixed (Dirichlet) temperature.
DIRICHLET_PROB = 0.30

#: Plate bounding-box dimensions [m].
WIDTH_RANGE = (0.050, 0.085)
HEIGHT_RANGE = (0.040, 0.065)

#: Geometry mix (weights must sum to 1).
GEOMETRY_KINDS = ("rect", "lshape", "hole")
GEOMETRY_WEIGHTS = (0.50, 0.25, 0.25)

#: Trajectory mix.
TRAJECTORY_KINDS = ("straight", "diagonal", "sinusoid", "arc")


# ---------------------------------------------------------------------------
# Geometry: mesh + analytic solid-region predicate (for trajectory validation)
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    """A meshed plate plus the analytic description of its solid region.

    ``removed`` lists the regions cut out of the bounding box, each as
    ``("rect", (x0, y0, x1, y1))`` or ``("circle", (cx, cy, r))``. The
    :meth:`is_solid` predicate (used to keep weld paths inside material) tests
    membership in the bounding box minus those regions, with a safety
    ``margin``.
    """

    kind: str
    mesh: MeshTri
    width: float
    height: float
    removed: List[Tuple[str, tuple]] = field(default_factory=list)

    def is_solid(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of points that lie in the (eroded) solid region.

        ``pts`` is ``(..., 2)``. A point is solid if it is at least ``margin``
        inside the outer box and at least ``margin`` away from every cut-out.
        """
        x = pts[..., 0]
        y = pts[..., 1]
        ok = (
            (x >= margin)
            & (x <= self.width - margin)
            & (y >= margin)
            & (y <= self.height - margin)
        )
        for shape, p in self.removed:
            if shape == "rect":
                x0, y0, x1, y1 = p
                inside = (
                    (x > x0 - margin)
                    & (x < x1 + margin)
                    & (y > y0 - margin)
                    & (y < y1 + margin)
                )
                ok &= ~inside
            elif shape == "circle":
                cx, cy, r = p
                ok &= np.hypot(x - cx, y - cy) > r + margin
        return ok


def _tensor_mesh(width: float, height: float, h_el: float) -> MeshTri:
    """Uniform right-triangle tensor mesh sized by target element edge ``h_el``."""
    nx = max(8, int(round(width / h_el)))
    ny = max(8, int(round(height / h_el)))
    return MeshTri.init_tensor(
        np.linspace(0.0, width, nx + 1), np.linspace(0.0, height, ny + 1)
    )


def _filter_triangles(mesh: MeshTri, keep: np.ndarray) -> MeshTri:
    """Drop triangles where ``keep`` is False and re-index surviving nodes."""
    t = mesh.t[:, keep]
    used = np.unique(t)
    remap = np.full(mesh.p.shape[1], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return MeshTri(mesh.p[:, used], remap[t])


def _tag_boundaries(
    mesh: MeshTri, width: float, height: float, has_inner: bool
) -> MeshTri:
    """Attach ``left/right/bottom/top`` outer markers (+ ``inner`` if cut)."""
    tol = min(width, height) * 1e-6
    bnd = {
        "left": lambda x: np.isclose(x[0], 0.0, atol=tol),
        "right": lambda x: np.isclose(x[0], width, atol=tol),
        "bottom": lambda x: np.isclose(x[1], 0.0, atol=tol),
        "top": lambda x: np.isclose(x[1], height, atol=tol),
    }
    if has_inner:
        # Any boundary facet not on the outer box edges (cut-out / hole rims).
        def _inner(x, w=width, h=height, tol=tol):
            on_outer = (
                np.isclose(x[0], 0.0, atol=tol)
                | np.isclose(x[0], w, atol=tol)
                | np.isclose(x[1], 0.0, atol=tol)
                | np.isclose(x[1], h, atol=tol)
            )
            return ~on_outer

        bnd["inner"] = _inner
    return mesh.with_boundaries(bnd)


def make_rectangular(rng: np.random.Generator, w: float, h: float, h_el: float) -> Geometry:
    mesh = _tag_boundaries(_tensor_mesh(w, h, h_el), w, h, has_inner=False)
    return Geometry("rect", mesh, w, h, removed=[])


def make_lshape(rng: np.random.Generator, w: float, h: float, h_el: float) -> Geometry:
    """L-shaped plate: cut the top-right rectangle out of the bounding box."""
    cx = float(rng.uniform(0.50, 0.62) * w)
    cy = float(rng.uniform(0.50, 0.62) * h)
    base = _tensor_mesh(w, h, h_el)
    centroids = base.p[:, base.t].mean(axis=1)  # (2, M)
    cut = (centroids[0] >= cx) & (centroids[1] >= cy)
    mesh = _tag_boundaries(_filter_triangles(base, ~cut), w, h, has_inner=True)
    return Geometry("lshape", mesh, w, h, removed=[("rect", (cx, cy, w, h))])


def make_holed(rng: np.random.Generator, w: float, h: float, h_el: float) -> Geometry:
    """Plate with one or two circular cut-outs in the upper band."""
    base = _tensor_mesh(w, h, h_el)
    centroids = base.p[:, base.t].mean(axis=1)
    n_holes = int(rng.integers(1, 3))
    removed: List[Tuple[str, tuple]] = []
    keep = np.ones(base.t.shape[1], dtype=bool)
    r = float(rng.uniform(0.08, 0.13) * min(w, h))
    for _ in range(n_holes):
        cx = float(rng.uniform(0.30, 0.70) * w)
        cy = float(rng.uniform(0.55, 0.72) * h)  # upper band; weld goes below
        keep &= np.hypot(centroids[0] - cx, centroids[1] - cy) > r
        removed.append(("circle", (cx, cy, r)))
    mesh = _tag_boundaries(_filter_triangles(base, keep), w, h, has_inner=True)
    return Geometry("hole", mesh, w, h, removed=removed)


_GEOMETRY_BUILDERS: dict = {
    "rect": make_rectangular,
    "lshape": make_lshape,
    "hole": make_holed,
}


# ---------------------------------------------------------------------------
# Trajectories (analytic-tangent parametric paths confined to a safe corridor)
# ---------------------------------------------------------------------------
def _corridor(geom: Geometry, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """A horizontal solid band ``(xa, xb, yc, half_height)`` for the weld path.

    Cut-outs live in the *upper* band for ``lshape``/``hole`` geometries, so a
    band in the lower part of the plate is guaranteed solid across the width.
    """
    w, h = geom.width, geom.height
    xa, xb = 0.14 * w, 0.86 * w
    if geom.kind == "rect":
        yc = float(rng.uniform(0.35, 0.65) * h)
        half = min(yc, h - yc) - 0.06 * h
    elif geom.kind == "lshape":
        cy = geom.removed[0][1][1]
        yc = float(rng.uniform(0.18, 0.42) * h)
        half = min(yc, cy - yc) - 0.05 * h
    else:  # hole — stay well below the holes
        yc = float(rng.uniform(0.16, 0.34) * h)
        half = yc - 0.05 * h
    return xa, xb, yc, max(half, 0.01 * h)


def _straight(xa, xb, yc, speed) -> Tuple[Trajectory, float]:
    traj = LinearTrajectory((xa, yc), (xb, yc), speed)
    return traj, (xb - xa) / speed


def _diagonal(xa, xb, yc, half, speed) -> Tuple[Trajectory, float]:
    d = 0.6 * half
    start = np.array([xa, yc - d])
    end = np.array([xb, yc + d])
    length = float(np.linalg.norm(end - start))
    return LinearTrajectory(start, end, speed), length / speed


def _sinusoid(xa, xb, yc, half, speed, rng) -> Tuple[Trajectory, float]:
    amp = float(rng.uniform(0.4, 0.85) * half)
    n_waves = float(rng.uniform(1.0, 2.5))
    span = xb - xa
    k = 2.0 * np.pi * n_waves / span
    # Parametrize by along-x speed so the torch sweeps the corridor end-to-end.
    vx = speed
    t_end = span / vx

    def pos(s, xa=xa, yc=yc, amp=amp, k=k, vx=vx):
        x = xa + vx * s
        return np.array([x, yc + amp * np.sin(k * (x - xa))])

    def dpos(s, xa=xa, amp=amp, k=k, vx=vx):
        x = xa + vx * s
        return np.array([vx, amp * k * vx * np.cos(k * (x - xa))])

    return ParametricTrajectory(pos, dpos), t_end


def _arc(xa, xb, yc, half, speed, rng) -> Tuple[Trajectory, float]:
    """Shallow circular arc bulging upward, endpoints on the corridor line."""
    phi = float(rng.uniform(0.30, 0.55))            # half sweep angle [rad]
    half_chord = 0.5 * (xb - xa) * float(rng.uniform(0.8, 1.0))
    R = half_chord / np.sin(phi)
    sag = R * (1.0 - np.cos(phi))
    if sag > 0.9 * half:  # too tall for the corridor -> flatten
        phi = 0.30
        R = half_chord / np.sin(phi)
        sag = R * (1.0 - np.cos(phi))
    cx = 0.5 * (xa + xb)
    cy = yc - R * np.cos(phi)                        # center below the line
    omega = speed / R                                # angular speed [rad/s]
    theta0 = 0.5 * np.pi + phi                        # sweep top down to +x side
    t_end = 2.0 * phi / omega

    def pos(s, cx=cx, cy=cy, R=R, omega=omega, theta0=theta0):
        th = theta0 - omega * s
        return np.array([cx + R * np.cos(th), cy + R * np.sin(th)])

    def dpos(s, R=R, omega=omega, theta0=theta0):
        th = theta0 - omega * s
        return np.array([R * omega * np.sin(th), -R * omega * np.cos(th)])

    return ParametricTrajectory(pos, dpos), t_end


def _path_inside(traj: Trajectory, t_end: float, geom: Geometry, margin: float) -> bool:
    s = np.linspace(0.0, t_end, 240)
    pts = np.array([traj.position(float(si)) for si in s])
    return bool(geom.is_solid(pts, margin=margin).all())


def sample_trajectory(
    geom: Geometry, rng: np.random.Generator, speed: float, goldak: GoldakParams
) -> Tuple[Trajectory, float, str]:
    """Pick and build a weld path that stays inside the material.

    Tries the chosen path kind; if it would carry the torch outside the solid
    region (margin = Goldak core size) it falls back to a straight path along
    the corridor, which is solid by construction.
    """
    xa, xb, yc, half = _corridor(geom, rng)
    margin = 1.1 * max(goldak.b, goldak.c_f)
    kind = str(rng.choice(TRAJECTORY_KINDS))

    if kind == "straight":
        traj, t_end = _straight(xa, xb, yc, speed)
    elif kind == "diagonal":
        traj, t_end = _diagonal(xa, xb, yc, half, speed)
    elif kind == "sinusoid":
        traj, t_end = _sinusoid(xa, xb, yc, half, speed, rng)
    else:
        traj, t_end = _arc(xa, xb, yc, half, speed, rng)

    if not _path_inside(traj, t_end, geom, margin):
        traj, t_end = _straight(xa, xb, yc, speed)
        kind = "straight"
    return traj, t_end, kind


# ---------------------------------------------------------------------------
# Per-simulation sampling
# ---------------------------------------------------------------------------
def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def build_boundary_conditions(
    geom: Geometry, rng: np.random.Generator, t_ambient: float
) -> Tuple[dict, float, float, Optional[str]]:
    """Robin convection on every boundary; optionally one Dirichlet outer edge.

    A single randomized ``(h_conv, T_inf, emissivity)`` triple is shared across
    markers so the network sees a coherent Robin response per simulation.
    """
    h_conv = _u(rng, *H_CONV_RANGE)
    emissivity = float(rng.choice(EMISSIVITY_CHOICES))
    robin = Robin(h_conv=h_conv, T_inf=t_ambient, emissivity=emissivity)

    markers = ["left", "right", "bottom", "top"]
    if "inner" in geom.mesh.boundaries:
        markers.append("inner")
    bcs: dict = {m: robin for m in markers}

    dirichlet_edge = None
    if rng.random() < DIRICHLET_PROB:
        dirichlet_edge = str(rng.choice(["left", "right", "bottom", "top"]))
        bcs[dirichlet_edge] = Dirichlet(value=t_ambient)
    return bcs, h_conv, emissivity, dirichlet_edge


def build_simulation(
    rng: np.random.Generator, h_el: float, target_steps: int
) -> Tuple[TransientThermalSolver, dict]:
    """Sample one fully-specified simulation; return solver + provenance dict."""
    # Geometry.
    w = _u(rng, *WIDTH_RANGE)
    h = _u(rng, *HEIGHT_RANGE)
    kind = str(rng.choice(GEOMETRY_KINDS, p=GEOMETRY_WEIGHTS))
    geom = _GEOMETRY_BUILDERS[kind](rng, w, h, h_el)

    # Process parameters.
    power = _u(rng, *POWER_RANGE)
    efficiency = _u(rng, *EFFICIENCY_RANGE)
    speed = _u(rng, *SPEED_RANGE)
    c_f = _u(rng, *C_F_RANGE)
    goldak = GoldakParams(
        power=power,
        efficiency=efficiency,
        a=_u(rng, *B_RANGE),
        b=_u(rng, *B_RANGE),
        c_f=c_f,
        c_r=c_f * _u(rng, *C_R_FACTOR_RANGE),
        f_f=_u(rng, *F_F_RANGE),
    )

    # Material / ambient.
    t_ambient = _u(rng, *T_AMBIENT_RANGE)
    material = MaterialProperties(
        thickness=_u(rng, *THICKNESS_RANGE), T_ambient=t_ambient
    )

    # Boundary conditions.
    bcs, h_conv, emissivity, dirichlet_edge = build_boundary_conditions(
        geom, rng, t_ambient
    )

    # Trajectory + time horizon.
    traj, t_end, traj_kind = sample_trajectory(geom, rng, speed, goldak)
    dt = float(np.clip(t_end / target_steps, 0.01, 0.05))
    cfg = SolverConfig(dt=dt, t_end=t_end, verbose=False)

    solver = TransientThermalSolver(geom.mesh, material, goldak, traj, bcs, cfg)

    provenance = {
        "geometry": {
            "kind": geom.kind,
            "width": w,
            "height": h,
            "removed": [[s, list(p)] for s, p in geom.removed],
            "num_nodes": int(geom.mesh.p.shape[1]),
            "num_cells": int(geom.mesh.t.shape[1]),
        },
        "trajectory": {"kind": traj_kind, "speed": speed, "t_end": t_end, "dt": dt},
        "boundary": {
            "h_conv": h_conv,
            "emissivity": emissivity,
            "dirichlet_edge": dirichlet_edge,
        },
    }
    return solver, provenance


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate_split(
    split: str,
    count: int,
    out_dir: Path,
    base_seed: int,
    h_el: float,
    target_steps: int,
    save_xdmf: bool,
) -> List[Path]:
    """Generate ``count`` simulations for one split, with a clean tqdm bar."""
    try:
        from tqdm import tqdm

        bar = tqdm(range(count), desc=f"{split:5s}", unit="sim")
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        bar = range(count)

    written: List[Path] = []
    for i in bar:
        # Per-sim seed keeps the corpus reproducible *and* resumable per index.
        offset = (1 if split == "train" else 2) * 100_000
        seed = base_seed + offset + i
        rng = np.random.default_rng(seed)
        solver, provenance = build_simulation(rng, h_el, target_steps)
        result = solver.run()
        result.metadata["generation"] = {"split": split, "seed": seed, **provenance}

        stem = f"sim_{split}_{i + 1:03d}"
        path = result.save_npz(out_dir / stem)
        if save_xdmf:
            result.save_xdmf(out_dir / stem)
        written.append(path)

        if hasattr(bar, "set_postfix"):
            g = provenance["geometry"]
            bar.set_postfix(
                geom=g["kind"],
                path=provenance["trajectory"]["kind"],
                nodes=g["num_nodes"],
                snaps=int(result.times.shape[0]),
            )
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Generate a diversified welding FEM dataset (raw .npz files).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_train", type=int, default=20, help="number of training simulations")
    p.add_argument("--num_val", type=int, default=5, help="number of validation simulations")
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="dataset root; raw files are written to <data_root>/raw",
    )
    p.add_argument("--seed", type=int, default=0, help="base RNG seed for reproducibility")
    p.add_argument(
        "--element_size",
        type=float,
        default=2.5e-3,
        help="target mesh element edge length [m] (smaller = finer/slower)",
    )
    p.add_argument(
        "--target_steps",
        type=int,
        default=80,
        help="approximate number of time steps per simulation (sets dt)",
    )
    p.add_argument(
        "--save_xdmf",
        action="store_true",
        help="also write XDMF time-series alongside each .npz (for ParaView)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    raw_dir = args.data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating dataset -> {raw_dir.resolve()}\n"
        f"  train={args.num_train}  val={args.num_val}  "
        f"element_size={args.element_size * 1e3:.2f} mm  "
        f"~steps={args.target_steps}  seed={args.seed}"
    )
    t0 = time.perf_counter()
    written: List[Path] = []
    for split, count in (("train", args.num_train), ("val", args.num_val)):
        if count <= 0:
            continue
        written += generate_split(
            split,
            count,
            raw_dir,
            args.seed,
            args.element_size,
            args.target_steps,
            args.save_xdmf,
        )
    elapsed = time.perf_counter() - t0
    print(
        f"\nDone: wrote {len(written)} simulations to {raw_dir.resolve()} "
        f"in {elapsed:.1f}s ({elapsed / max(len(written), 1):.1f}s/sim)."
    )


if __name__ == "__main__":
    main()
