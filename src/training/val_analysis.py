"""Thermal-history post-processing for held-out *validation* rollouts.

Consumes a single-pass rollout archive written by ``training.rollout.export_rollout``
(under ``<rollout_pred_dir>/<stem>_rollout.npz``) carrying ``predicted_temperature``
and ``ground_truth_temperature`` ``(S, N)`` arrays on the sim's own mesh, and
produces — mirroring :mod:`training.spiral_analysis` so the paper figures share a
look — for each sim:

1. A **difference-field XDMF/H5** time series (``T_fem``, ``T_pred``, signed
   ``error``) for ParaView.
2. A **thermal-history** PNG + CSV at a few probe nodes.

Unlike the spiral (fixed 0.2 m plate, known probe coordinates), validation sims
have arbitrary geometry, so probes are picked *generically* by the per-node GT
peak excess over ambient: the hottest weld node, two intermediate quantiles, and
the coolest corner. This spans the field on any mesh.

Run::

    uv run python -m src.training.val_analysis \
        --rollout data/output/rollout_pred_enthsrc_pf/sim_train_001_rollout.npz \
        --prefix val_enthsrc_sim001
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.spiral_analysis import write_multifield_xdmf  # reuse XDMF writer

log = logging.getLogger("val_analysis")

AMBIENT_K = 293.15


def select_probes(
    gt: np.ndarray, coords: np.ndarray, n: int = 4
) -> List[Tuple[str, int, np.ndarray]]:
    """Pick ``n`` probe nodes spanning the *weld-affected* field, log-geometrically.

    The peak-excess distribution is dominated by far-field nodes that barely heat,
    so plain quantiles land on flat ambient traces. Instead we target geometrically
    spaced fractions of the maximum peak excess — ``[100%, ..., 3%]`` for ``n=4`` —
    and snap each to the node whose peak is closest. This walks melt pool → HAZ →
    near-field → far-field, the informative range for a weld. Returns
    ``(label, node_index, xy)`` like :func:`spiral_analysis.nearest_nodes`.
    """
    peak = gt.max(axis=0) - AMBIENT_K  # per-node peak excess over ambient
    max_exc = float(peak.max())
    fractions = np.geomspace(1.0, 0.03, n)
    picks: List[Tuple[str, int]] = []
    used: set[int] = set()
    for f in fractions:
        target = f * max_exc
        order = np.argsort(np.abs(peak - target))
        idx = next((int(i) for i in order if int(i) not in used), int(order[0]))
        used.add(idx)
        picks.append((f"{f*100:.0f}%·peak", idx))
    return [(lbl, idx, coords[idx]) for lbl, idx in picks]


def plot_thermal_histories(times, gt, pred, probes, png_path: Path, title: str) -> None:
    """Per-probe FEM-vs-surrogate temperature history grid (paper style)."""
    n = len(probes)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (label, idx, xy) in zip(axes.flat, probes):
        ax.plot(times, gt[:, idx], "-", color="C0", lw=1.8, label="FEM")
        ax.plot(times, pred[:, idx], "--", color="C3", lw=1.6, label="surrogate")
        rmse = float(np.sqrt(np.mean((pred[:, idx] - gt[:, idx]) ** 2)))
        ax.set_title(
            f"{label}  @({xy[0]*1e3:.0f},{xy[1]*1e3:.0f}) mm  | "
            f"peak {gt[:, idx].max():.0f} K | RMSE {rmse:.0f} K",
            fontsize=10,
        )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("T [K]")
        ax.axhline(AMBIENT_K, color="0.6", lw=0.8, ls=":", label="ambient")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_history_csv(times, gt, pred, probes, csv_path: Path) -> None:
    cols = [times]
    header = ["time_s"]
    for label, idx, _ in probes:
        safe = label.replace(" ", "_").replace("=", "").replace(".", "")
        cols += [gt[:, idx], pred[:, idx]]
        header += [f"{safe}_fem_K", f"{safe}_pred_K"]
    np.savetxt(csv_path, np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="val_analysis",
        description="Thermal histories + difference field for a held-out val rollout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rollout", type=Path, required=True,
                   help="path to a *_rollout.npz from export_rollout")
    p.add_argument("--output_dir", type=Path, default=Path("data/output"))
    p.add_argument("--prefix", type=str, default=None,
                   help="output filename prefix (default: rollout stem)")
    p.add_argument("--n_probes", type=int, default=4)
    p.add_argument("--no_xdmf", action="store_true", help="skip the ParaView XDMF export")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)
    if not args.rollout.exists():
        raise FileNotFoundError(f"Missing {args.rollout}")
    prefix = args.prefix or args.rollout.stem.replace("_rollout", "")
    out = args.output_dir

    d = np.load(args.rollout)
    pred = d["predicted_temperature"]
    gt = d["ground_truth_temperature"]
    times = d["times"]
    coords = d["coords"]
    cells = d["cells"]
    error = pred - gt
    log.info(
        "%s: %d frames x %d nodes | global RMSE %.1f K | max |error| %.1f K",
        prefix, gt.shape[0], gt.shape[1],
        float(np.sqrt(np.mean(error ** 2))), float(np.abs(error).max()),
    )

    if not args.no_xdmf:
        xdmf = write_multifield_xdmf(
            out / f"{prefix}_comparison", coords, cells, times,
            {"T_fem": gt, "T_pred": pred, "error": error},
        )
        log.info("Saved difference-field time series: %s (+ .h5)", xdmf)

    probes = select_probes(gt, coords, n=args.n_probes)
    for label, idx, xy in probes:
        log.info(
            "  probe %-10s node %5d @(%.0f,%.0f) mm | FEM peak %.0f K | surrogate peak %.0f K",
            label, idx, xy[0] * 1e3, xy[1] * 1e3,
            float(gt[:, idx].max()), float(pred[:, idx].max()),
        )
    png = out / f"{prefix}_thermal_history.png"
    csv = out / f"{prefix}_thermal_history.csv"
    plot_thermal_histories(
        times, gt, pred, probes, png,
        title=f"Held-out validation thermal history ({prefix}) — FEM (solid) vs MeshGraphNet (dashed)",
    )
    write_history_csv(times, gt, pred, probes, csv)
    log.info("Saved thermal-history plot: %s", png)
    log.info("Saved thermal-history csv:  %s", csv)


if __name__ == "__main__":
    main()
