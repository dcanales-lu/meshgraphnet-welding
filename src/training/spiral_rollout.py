"""Pure-inference stress test: autoregressive rollout on an unseen spiral weld.

This script exercises the trained MeshGraphNet on a trajectory it has **never
seen** — an Archimedean spiral expanding from the plate centre — *without ever
running the FEM solver*. There is no ground truth: the network is given only the
initial (ambient) temperature field and the scheduled torch motion, and it must
predict the entire thermal history autoregressively.

How it reuses the training contract
------------------------------------
Rather than re-deriving the node-feature assembly (and risking drift from the
training layout), we construct a synthetic :class:`SimulationResult` whose
``source_position / tangent / normal`` follow the spiral and whose
``temperature`` is an **ambient placeholder** (only ``T^0`` is consumed — the
rollout overwrites every later step with its own prediction). We then call
:func:`training.rollout.run_autoregressive_rollout`, which performs exactly the
requested loop for every step:

    (a) evaluate the analytical Goldak field at the new source position,
    (b) rotate each node's relative displacement into the local (t̂, n̂) frame,
    (c) assemble the 16-d node features (T, q_Goldak, co-moving coords, process
        params, one-hot interior/boundary flags + Robin values),
    (d) normalize, forward through the GNN, de-normalize ΔT, T^{t+1}=T^t+ΔT,
    (e) feed the temperature back for the next step.

Because the trajectory is deterministic, those time-dependent features are
computed once up front (vectorized) and only the temperature column is rolled —
mathematically identical to per-step recomputation.

Run::

    uv run python -m src.training.spiral_rollout
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from skfem import Basis, ElementTriP1

from models.meshgraphnet import MeshGraphNet
from simulation.thermal_solver import GoldakParams, SimulationResult, rectangular_plate
from training.rollout import run_autoregressive_rollout
from training.train import TrainConfig

log = logging.getLogger("spiral_rollout")

#: Baseline ambient temperature [K] (initial field + Robin sink temperature).
AMBIENT = 293.15


# ---------------------------------------------------------------------------
# 1. Geometry / static graph context
# ---------------------------------------------------------------------------
def boundary_node_masks(mesh) -> Tuple[dict, int]:
    """Per-marker boolean node masks (matches the solver's P1 dof convention)."""
    basis = Basis(mesh, ElementTriP1())
    n = basis.N
    masks = {}
    for marker in mesh.boundaries:
        mask = np.zeros(n, dtype=bool)
        mask[basis.get_dofs(marker).flatten()] = True
        masks[marker] = mask
    return masks, n


# ---------------------------------------------------------------------------
# 2. Analytical Archimedean spiral trajectory
# ---------------------------------------------------------------------------
def build_spiral_trajectory(
    center: Tuple[float, float],
    coil_spacing_b: float,
    theta_max: float,
    speed: float,
    dt: float,
    dense: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constant-speed samples of ``γ(θ) = c + b·θ·[cosθ, sinθ]``.

    The spiral is sampled densely in ``θ``, re-parametrized by **arc length** so
    the torch advances at a constant welding ``speed`` (``Δs = speed·dt`` per
    step), and the unit tangent / normal are taken from the analytic derivative

        dγ/dθ = b·[cosθ − θ·sinθ,  sinθ + θ·cosθ],   n̂ = R₊₉₀ t̂.

    Returns ``(positions, tangents, normals, times)`` each of length ``S``.
    """
    theta = np.linspace(0.0, theta_max, dense)
    x = center[0] + coil_spacing_b * theta * np.cos(theta)
    y = center[1] + coil_spacing_b * theta * np.sin(theta)

    seg = np.hypot(np.diff(x), np.diff(y))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])

    step = speed * dt
    n_steps = int(total // step) + 1
    s_targets = np.arange(n_steps) * step
    s_targets = s_targets[s_targets <= total]

    theta_k = np.interp(s_targets, arc, theta)
    px = center[0] + coil_spacing_b * theta_k * np.cos(theta_k)
    py = center[1] + coil_spacing_b * theta_k * np.sin(theta_k)
    pos = np.column_stack([px, py])

    tx = coil_spacing_b * (np.cos(theta_k) - theta_k * np.sin(theta_k))
    ty = coil_spacing_b * (np.sin(theta_k) + theta_k * np.cos(theta_k))
    tnorm = np.hypot(tx, ty)
    tnorm[tnorm == 0.0] = 1.0
    tan = np.column_stack([tx / tnorm, ty / tnorm])
    nor = np.column_stack([-tan[:, 1], tan[:, 0]])  # rotate tangent +90° (solver convention)

    times = s_targets / speed  # == arange(S) * dt
    return pos, tan, nor, times


# ---------------------------------------------------------------------------
# Synthetic SimulationResult carrying the spiral schedule (no FEM)
# ---------------------------------------------------------------------------
def make_spiral_result(args) -> SimulationResult:
    """Build a self-contained :class:`SimulationResult` for the spiral rollout.

    ``temperature`` is an **ambient placeholder** — only ``T^0`` is used as the
    rollout's starting field; later steps are produced by the network.
    """
    mesh = rectangular_plate(args.plate_size, args.plate_size, args.resolution, args.resolution)
    masks, n = boundary_node_masks(mesh)
    coords = mesh.p.T.copy()
    cells = mesh.t.T.copy()

    center = (args.plate_size / 2.0, args.plate_size / 2.0)
    theta_max = 2.0 * np.pi * args.turns
    max_radius = args.plate_size / 2.0 - args.margin
    coil_spacing_b = max_radius / theta_max
    pos, tan, nor, times = build_spiral_trajectory(
        center, coil_spacing_b, theta_max, args.speed, args.dt
    )
    s = len(times)

    goldak = GoldakParams(
        power=args.power, efficiency=args.efficiency,
        a=args.goldak_b, b=args.goldak_b, c_f=args.c_f, c_r=args.c_r, f_f=args.f_f,
    )
    goldak_md = {
        "power": goldak.power, "efficiency": goldak.efficiency, "net_power": goldak.net_power,
        "a": goldak.a, "b": goldak.b, "c_f": goldak.c_f, "c_r": goldak.c_r,
        "f_f": goldak.f_f, "f_r": goldak.f_r,
    }
    boundary_specs = {
        m: {"type": "robin", "h_conv": args.h_conv, "T_inf": AMBIENT, "emissivity": 0.0}
        for m in mesh.boundaries
    }

    return SimulationResult(
        coords=coords,
        cells=cells,
        times=times,
        temperature=np.full((s, n), AMBIENT, dtype=np.float64),  # placeholder; only [0] used
        boundary_masks=masks,
        source_position=pos,
        source_tangent=tan,
        source_normal=nor,
        source_power=np.full(s, goldak.net_power),
        metadata={
            "goldak": goldak_md,
            "thickness": args.thickness,
            "T_ambient": AMBIENT,
            "boundary_specs": boundary_specs,
            "generation": {
                "kind": "spiral_inference",
                "turns": args.turns,
                "speed": args.speed,
                "dt": args.dt,
                "coil_spacing_b": coil_spacing_b,
                "center": list(center),
                "num_nodes": int(n),
                "num_steps": int(s),
            },
        },
    )


# ---------------------------------------------------------------------------
# 3. Model / normalizer loading
# ---------------------------------------------------------------------------
def load_model(checkpoint: Path, device: torch.device) -> MeshGraphNet:
    """Rebuild the MeshGraphNet from the checkpoint's own config and load weights."""
    payload = torch.load(checkpoint, weights_only=False)
    model_cfg = TrainConfig(**payload["config"]).model_config()
    model = MeshGraphNet(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    log.info(
        "Loaded %s | %d params | hidden=%d | %d hops",
        checkpoint, model.num_parameters(), model_cfg.hidden_dim,
        model_cfg.num_processing_steps,
    )
    return model


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spiral_rollout",
        description="Autoregressive MeshGraphNet rollout on an unseen spiral weld path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # geometry / trajectory
    p.add_argument("--plate_size", type=float, default=0.2, help="square plate side [m]")
    p.add_argument("--resolution", type=int, default=64, help="elements per side (nx=ny)")
    p.add_argument("--turns", type=float, default=3.0, help="number of full spiral turns")
    p.add_argument("--margin", type=float, default=0.02, help="keep-out from plate edge [m]")
    p.add_argument("--speed", type=float, default=0.01, help="welding speed [m/s]")
    p.add_argument("--dt", type=float, default=0.05, help="time step [s]")
    # process / Goldak
    p.add_argument("--power", type=float, default=2500.0, help="gross arc power [W]")
    p.add_argument("--efficiency", type=float, default=0.8, help="arc efficiency [-]")
    p.add_argument("--goldak_b", type=float, default=3.0e-3, help="Goldak half-width b [m]")
    p.add_argument("--c_f", type=float, default=3.0e-3, help="Goldak front semi-axis [m]")
    p.add_argument("--c_r", type=float, default=6.0e-3, help="Goldak rear semi-axis [m]")
    p.add_argument("--f_f", type=float, default=0.6, help="Goldak front heat fraction [-]")
    p.add_argument("--thickness", type=float, default=5.0e-3, help="plate thickness [m]")
    p.add_argument("--h_conv", type=float, default=15.0, help="Robin convection coeff [W/m^2K]")
    # io / runtime
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    p.add_argument("--stats", type=Path, default=Path("data/processed/stats.pt"),
                   help="normalization stats (falls back to checkpoints/stats.pt)")
    p.add_argument("--output_dir", type=Path, default=Path("data/output"))
    p.add_argument("--output_name", type=str, default="spiral_rollout")
    p.add_argument("--device", type=str, default="cpu")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device)

    stats_path = args.stats if args.stats.exists() else Path("checkpoints/stats.pt")
    if not stats_path.exists():
        raise FileNotFoundError(
            f"No normalization stats found at {args.stats} or checkpoints/stats.pt. "
            "Run training first to produce stats.pt."
        )

    model = load_model(args.checkpoint, device)

    result = make_spiral_result(args)
    gen = result.metadata["generation"]
    log.info(
        "Spiral: %.1f turns | %d nodes | %d steps | speed %.1f mm/s | dt %.3f s | net power %.0f W",
        args.turns, gen["num_nodes"], gen["num_steps"], args.speed * 1e3, args.dt,
        result.metadata["goldak"]["net_power"],
    )

    log.info("Rolling out autoregressively on %s (no FEM ground truth) ...", device)
    rollout = run_autoregressive_rollout(model, result, str(stats_path), device=device)

    pred = rollout.predicted_temperature
    log.info(
        "Rollout done: %d frames | predicted T range [%.1f, %.1f] K (ambient %.2f K)",
        pred.shape[0], float(pred.min()), float(pred.max()), AMBIENT,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_sr = rollout.to_simulation_result()
    xdmf_path = pred_sr.save_xdmf(out_dir / args.output_name)
    npz_path = pred_sr.save_npz(out_dir / args.output_name)
    log.info("Saved ParaView time-series: %s (+ .h5)", xdmf_path)
    log.info("Saved prediction npz:       %s", npz_path)


if __name__ == "__main__":
    main()
