"""Post-processing for the spiral FEM-vs-surrogate comparison.

Consumes the two prediction archives written under ``data/output/`` —
``spiral_fem.npz`` (FEM ground truth) and ``spiral_rollout.npz`` (MeshGraphNet
autoregressive rollout) — which share the *same mesh and timeline*, and produces:

1. **Difference-field time series** ``data/output/spiral_comparison.xdmf`` (+ ``.h5``)
   carrying three per-timestep nodal fields for ParaView:
   ``T_fem``, ``T_pred`` and ``error = T_pred - T_fem`` (signed, kelvin).
2. **Thermal histories** at several probe points (plate centre, mid/outer spiral
   radii, and a cool corner): a comparison plot
   ``data/output/spiral_thermal_history.png`` and a tidy
   ``data/output/spiral_thermal_history.csv``.

No solver or model is run here — it is fast array post-processing.

Run::

    uv run python -m src.training.spiral_analysis
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("spiral_analysis")

#: Probe points (label, x, y) in metres, for the default 0.2 m plate centred at
#: (0.1, 0.1). Chosen to span the weld: centre (spiral start), increasing radii
#: the torch sweeps, and a cool far corner.
DEFAULT_PROBES: List[Tuple[str, float, float]] = [
    ("center", 0.100, 0.100),
    ("r=0.04 (+x)", 0.140, 0.100),
    ("r=0.07 (+x)", 0.170, 0.100),
    ("outer (+y)", 0.100, 0.165),
    ("cool corner", 0.030, 0.030),
]


def _load(npz_path: Path):
    d = np.load(npz_path)
    return d["temperature"], d["coords"], d["cells"], d["times"]


def write_multifield_xdmf(
    path: Path, coords: np.ndarray, cells: np.ndarray, times: np.ndarray,
    fields: Dict[str, np.ndarray],
) -> Path:
    """Write a multi-field XDMF/H5 time series (ParaView), h5 co-located.

    ``fields`` maps an attribute name to an ``(S, N)`` nodal array. Mirrors the
    fix in :meth:`SimulationResult.save_xdmf`: meshio writes the ``.h5`` to the
    CWD by bare basename, so we move it next to the ``.xdmf`` afterwards.
    """
    import meshio

    path = Path(path).with_suffix(".xdmf")
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.column_stack([coords, np.zeros(len(coords))])

    origin = Path.cwd()
    with meshio.xdmf.TimeSeriesWriter(str(path)) as writer:
        writer.write_points_cells(points, [("triangle", cells)])
        for i, t in enumerate(times):
            writer.write_data(float(t), point_data={k: v[i] for k, v in fields.items()})

    h5_name = Path(writer.h5_filename).name
    produced = origin / h5_name
    target = path.with_name(h5_name)
    if produced.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(produced), str(target))
    return path


def nearest_nodes(coords: np.ndarray, probes) -> List[Tuple[str, int, np.ndarray]]:
    """Map each (label, x, y) probe to its nearest mesh node index."""
    out = []
    for label, x, y in probes:
        idx = int(np.argmin(np.hypot(coords[:, 0] - x, coords[:, 1] - y)))
        out.append((label, idx, coords[idx]))
    return out


def plot_thermal_histories(
    times, fem, pred, probes_resolved, png_path: Path,
) -> None:
    """Per-probe FEM-vs-surrogate temperature history grid."""
    n = len(probes_resolved)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, (label, idx, xy) in zip(axes.flat, probes_resolved):
        ax.plot(times, fem[:, idx], "-", color="C0", lw=1.8, label="FEM")
        ax.plot(times, pred[:, idx], "--", color="C3", lw=1.6, label="surrogate")
        rmse = float(np.sqrt(np.mean((pred[:, idx] - fem[:, idx]) ** 2)))
        ax.set_title(f"{label}  @({xy[0]*1e3:.0f},{xy[1]*1e3:.0f}) mm  | RMSE {rmse:.0f} K",
                     fontsize=10)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("T [K]")
        ax.axhline(293.15, color="0.6", lw=0.8, ls=":", label="ambient")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Spiral weld thermal history — FEM (solid) vs MeshGraphNet (dashed)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_history_csv(times, fem, pred, probes_resolved, csv_path: Path) -> None:
    cols = [times]
    header = ["time_s"]
    for label, idx, _ in probes_resolved:
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        cols += [fem[:, idx], pred[:, idx]]
        header += [f"{safe}_fem_K", f"{safe}_pred_K"]
    arr = np.column_stack(cols)
    np.savetxt(csv_path, arr, delimiter=",", header=",".join(header), comments="")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spiral_analysis",
        description="Difference field + thermal histories for the spiral comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir", type=Path, default=Path("data/output"))
    p.add_argument("--prefix", type=str, default="spiral",
                   help="output filename prefix (<prefix>_comparison, <prefix>_thermal_history)")
    p.add_argument("--fem", type=Path, default=None, help="default <output_dir>/spiral_fem.npz")
    p.add_argument("--pred", type=Path, default=None,
                   help="default <output_dir>/spiral_rollout.npz")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    args = build_arg_parser().parse_args(argv)
    out = args.output_dir
    fem_npz = args.fem or out / "spiral_fem.npz"
    pred_npz = args.pred or out / "spiral_rollout.npz"
    for pth in (fem_npz, pred_npz):
        if not pth.exists():
            raise FileNotFoundError(f"Missing {pth}; run spiral_fem / spiral_rollout first.")

    fem, coords, cells, times = _load(fem_npz)
    pred, coords_p, _, times_p = _load(pred_npz)
    if fem.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch FEM {fem.shape} vs surrogate {pred.shape}; "
            "re-run both with identical args."
        )

    error = pred - fem
    log.info(
        "Loaded %d frames x %d nodes | global RMSE %.1f K | max |error| %.1f K",
        fem.shape[0], fem.shape[1], float(np.sqrt(np.mean(error ** 2))),
        float(np.abs(error).max()),
    )

    # 1) Difference-field XDMF (T_fem, T_pred, signed error) for ParaView.
    xdmf = write_multifield_xdmf(
        out / f"{args.prefix}_comparison", coords, cells, times,
        {"T_fem": fem, "T_pred": pred, "error": error},
    )
    log.info("Saved difference-field time series: %s (+ .h5) [fields: T_fem, T_pred, error]", xdmf)

    # 2) Thermal histories at probe points.
    probes_resolved = nearest_nodes(coords, DEFAULT_PROBES)
    for label, idx, xy in probes_resolved:
        log.info(
            "  probe %-14s node %5d @(%.0f,%.0f) mm | FEM peak %.0f K | surrogate peak %.0f K",
            label, idx, xy[0] * 1e3, xy[1] * 1e3, float(fem[:, idx].max()),
            float(pred[:, idx].max()),
        )
    png = out / f"{args.prefix}_thermal_history.png"
    csv = out / f"{args.prefix}_thermal_history.csv"
    plot_thermal_histories(times, fem, pred, probes_resolved, png)
    write_history_csv(times, fem, pred, probes_resolved, csv)
    log.info("Saved thermal-history plot: %s", png)
    log.info("Saved thermal-history csv:  %s", csv)


if __name__ == "__main__":
    main()
