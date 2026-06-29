"""Build a 3-panel figure of representative training-dataset samples.

Picks three FEM simulations from ``data/raw/`` with **different geometries**
(rectangular / L-shaped / holed) and **different weld trajectories**
(straight|diagonal / sinusoid / arc), and renders, for each, the temperature
field at the end of the weld phase (``tripcolor`` on the FEM mesh) with the full
torch trajectory overlaid. Used as the "training dataset" slide of the congress
deck.

Run::

    uv run python scripts/make_dataset_samples_fig.py

Output: ``docs/paper/dataset_samples.png``.

Reads the ``.npz`` arrays directly (no package import needed): ``coords (N,2)``,
``cells (M,3)``, ``temperature (S,N)``, ``source_position (S,2)``,
``source_power (S,)`` and the JSON ``metadata`` (geometry/trajectory descriptors).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

RAW_DIR = Path("data/raw")
OUT_PNG = Path("docs/paper/dataset_samples.png")

# One sim per geometry; each must use a *different* trajectory kind. Order =
# panel order (left -> right). Filled deterministically by scan_samples().
GEOMETRY_ORDER = ("rect", "lshape", "hole")


def read_meta(path: Path) -> dict:
    """Cheaply read only the JSON metadata of a sim .npz."""
    with np.load(path, allow_pickle=True) as d:
        return json.loads(str(d["metadata_json"])) if "metadata_json" in d else {}


def scan_samples() -> list[tuple[Path, str, str]]:
    """Pick (path, geometry, trajectory) for one rect, one lshape, one hole,
    each with a distinct trajectory kind. Deterministic.

    For the holed plate we deliberately pick the sim with the **most nodes**
    (largest plate at the fixed 3 mm element size) so the circular cut-out is
    resolved by the most elements and looks round, not jagged.
    """
    files = sorted(RAW_DIR.glob("sim_train_*.npz"))
    # geometry -> list of (path, trajectory, num_nodes)
    by_geom: dict[str, list[tuple[Path, str, int]]] = {g: [] for g in GEOMETRY_ORDER}
    for f in files:
        gen = read_meta(f).get("generation", {})
        g = gen.get("geometry", {}).get("kind")
        t = gen.get("trajectory", {}).get("kind")
        n = int(gen.get("geometry", {}).get("num_nodes", 0))
        if g in by_geom and t is not None:
            by_geom[g].append((f, t, n))

    chosen: list[tuple[Path, str, str]] = []
    used_traj: set[str] = set()
    # Resolve the holed plate FIRST (roundest hole = most nodes), then the others
    # with trajectories distinct from those already used.
    for g in ("hole", "lshape", "rect"):
        cands = by_geom[g]
        if not cands:
            raise RuntimeError(f"No training sim found for geometry '{g}'.")
        if g == "hole":
            f, t, _ = max(cands, key=lambda c: c[2])           # most nodes
        else:
            fresh = [c for c in cands if c[1] not in used_traj]
            pool = sorted(fresh or cands, key=lambda c: (-c[2], c[0].name))
            f, t, _ = pool[0]                                  # prefer big & fresh-traj
        used_traj.add(t)
        chosen.append((f, g, t))
    # Display order left->right: rectangular, L-shaped, holed.
    order = {"rect": 0, "lshape": 1, "hole": 2}
    chosen.sort(key=lambda c: order[c[1]])
    return chosen


def weld_end_index(source_power: np.ndarray, temperature: np.ndarray) -> int:
    """Snapshot at the end of the weld phase (last step with the source on);
    fallback = step of peak spatial-max temperature."""
    on = np.flatnonzero(source_power > 0.0)
    if on.size:
        return int(on[-1])
    return int(np.argmax(temperature.max(axis=1)))


_GEOM_LABEL = {"rect": "Rectangular", "lshape": "L-shaped", "hole": "Holed"}
_TRAJ_LABEL = {
    "straight": "straight", "diagonal": "diagonal",
    "sinusoid": "sinusoidal", "arc": "circular-arc",
}


def main() -> None:
    samples = scan_samples()
    print("Selected samples:")
    for f, g, t in samples:
        print(f"  {f.name:24s} geometry={g:7s} trajectory={t}")

    # Shared color scale across panels (so the colorbar is comparable).
    fields = []
    for f, _, _ in samples:
        with np.load(f, allow_pickle=True) as d:
            coords = d["coords"]; cells = d["cells"]
            temperature = d["temperature"]; spos = d["source_position"]
            spower = d["source_power"]
        idx = weld_end_index(spower, temperature)
        removed = read_meta(f).get("generation", {}).get("geometry", {}).get("removed", [])
        fields.append((coords, cells, temperature[idx], spos, idx, removed))
    vmin = min(float(fld[2].min()) for fld in fields)
    vmax = max(float(fld[2].max()) for fld in fields)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    im = None
    for ax, (f, g, t), (coords, cells, Tfield, spos, idx, removed) in zip(axes, samples, fields):
        tri = Triangulation(coords[:, 0], coords[:, 1], cells)
        # ParaView "Surface With Edges" look: smooth thermal field + visible mesh.
        im = ax.tripcolor(tri, Tfield, shading="gouraud", cmap="jet",
                          vmin=vmin, vmax=vmax, rasterized=True)
        ax.triplot(tri, color="k", lw=0.22, alpha=0.35)        # FEM discretization
        # Analytic circular cut-out(s) the staircase mesh approximates — makes the
        # hole read as a discretized circle, not an irregular blob.
        for shape, p in removed:
            if shape == "circle":
                cx, cy, r = p
                ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                        ec="black", lw=1.3, ls="--", alpha=0.85))
        # Full torch trajectory + current torch position at the snapshot.
        ax.plot(spos[:, 0], spos[:, 1], color="k", lw=1.8, alpha=0.85)
        ax.plot(spos[:, 0], spos[:, 1], color="w", lw=0.7, alpha=0.9)
        ax.plot(spos[idx, 0], spos[idx, 1], marker="o", color="white",
                markersize=7, markeredgecolor="black", markeredgewidth=0.9)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{_GEOM_LABEL[g]} plate, {_TRAJ_LABEL[t]} weld", fontsize=12)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Temperature [K]")
    # No suptitle: the slide title + bullets carry the caption (saves vertical
    # space so the slide does not overflow).
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
