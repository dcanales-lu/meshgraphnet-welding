r"""Single-panel 'maximum-principle violation' figure for the slides.

Standalone version of the right panel of ``ablation_figure.png``: minimum
predicted spiral temperature vs. epoch for GENERIC ON vs OFF, with the physical
floor (293 K). ON stays at the floor; OFF cools sub-ambient at every checkpoint
(unphysical). Larger fonts so it reads well when enlarged on a slide.

Reads the committed sweep CSVs next to ``scripts/ablation/``.

Run::

    uv run python scripts/make_minT_fig.py

Output: ``docs/paper/ablation_minT.png``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / "ablation"
OUT = HERE.parent / "docs" / "paper" / "ablation_minT.png"

AMB = 293.15  # ambient = Robin bath = initial field -> physical minimum T [K]
C_ON, C_OFF = "#1b7837", "#b2182b"


def load(name: str):
    ep, tmin = [], []
    with open(SWEEP_DIR / f"spiral_{name}_sweep.csv") as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            tmin.append(float(r["Tmin_K"]))
    order = np.argsort(ep)
    return np.array(ep)[order], np.array(tmin)[order]


def main() -> None:
    ep_on, tmin_on = load("genon")
    ep_off, tmin_off = load("genoff")

    plt.rcParams.update({"font.size": 15, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(AMB, color="k", ls=":", lw=1.8, label=f"physical floor {AMB:.0f} K")
    ax.plot(ep_on, tmin_on, "o-", color=C_ON, lw=2.4, ms=8, label="GENERIC ON")
    ax.plot(ep_off, tmin_off, "s--", color=C_OFF, lw=2.4, ms=8, label="GENERIC OFF")
    # shade the unphysical (sub-ambient) region
    ax.axhspan(ax.get_ylim()[0] if False else 120, AMB, color=C_OFF, alpha=0.06)
    ax.set_xlabel("epoch")
    ax.set_ylabel("minimum predicted $T$ [K]")
    ax.set_title("Maximum-principle violation\n(sub-ambient $=$ unphysical)", fontsize=15)
    ax.legend(frameon=False, loc="center right")
    ax.set_ylim(130, 305)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    plt.close(fig)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
