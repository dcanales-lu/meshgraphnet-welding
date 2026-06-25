"""Build the GENERIC-enthalpy ablation deliverables (table + figures).

Ablation: **GENERIC-enthalpy ON vs OFF, both push-forward K=2**, identical
split/budget (split_seed=0, val_fraction=0.2, max_val_sims=20, batch 16,
max_windows 40000, 80 epochs, plateau, 5 K noise). Row ON = headline
(`checkpoints_enthsrc_pf`, 1,375,238 params); row OFF = pure MeshGraphNet
(`checkpoints_genoff_pf`, 1,291,777 params).

Produces (under ``data/output/``):
  1. ``ablation_table.md``                 — the 2-row results table.
  2. ``ablation_figure.png``               — spiral-RMSE-vs-epoch (★ best-val) +
                                             min-T-vs-epoch (293 K physical floor).
  3. ``compare_spiral_fem_on_off.png``     — FEM vs ON(ep60) vs OFF(ep70), 5 spiral probes.
  4. ``compare_val_sim{001,006}_fem_on_off.png`` — FEM vs ON vs OFF, held-out val sims.

Data sources
------------
* Spiral sweep numbers: ``spiral_gen{on,off}_sweep.csv`` (next to this script;
  the committed record). Regenerate with, for each checkpoint epoch NN::

    uv run python -m src.training.spiral_rollout  --checkpoint <ckpt> --stats <stats> \
        --output_name spiral_genXX_epNN --device cuda
    uv run python -m src.training.spiral_analysis --pred data/output/spiral_genXX_epNN.npz \
        --prefix spiral_genXX_epNN     # logs "global RMSE … | max |error| …"

* val_rmse per epoch: training ``history.json`` (embedded below; tiny, and
  ``checkpoints_*`` is gitignored).
* 3-curve comparisons: read the spiral/val rollout ``.npz`` from ``data/output/``
  (regenerable, gitignored). Skipped with a notice if absent.

Run::

    uv run python scripts/ablation/build_ablation_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

AMB = 293.15  # ambient = Robin bath = initial field → physical minimum temperature [K]
C_FEM, C_ON, C_OFF = "k", "#1b7837", "#b2182b"

# val_rollout_rmse per epoch (held-out single-pass; from each run's history.json)
VAL = {
    "on":  {10: 46.4, 20: 44.2, 30: 39.4, 40: 39.7, 50: 39.7, 60: 34.5, 70: 40.5, 80: 37.4},
    "off": {10: 78.97, 20: 64.15, 30: 71.69, 40: 47.80, 50: 58.20, 60: 69.85, 70: 46.86, 80: 55.02},
}
BEST_EP = {"on": 60, "off": 70}     # val-selected (best_model.pt) epoch
PARAMS = {"on": 1_375_238, "off": 1_291_777}


def load_sweep(name: str) -> dict:
    rows = {}
    with open(HERE / f"spiral_{name}_sweep.csv") as f:
        for r in csv.DictReader(f):
            rows[int(r["epoch"])] = dict(
                spiral=float(r["global_rmse_K"]), maxerr=float(r["max_abs_err_K"]),
                tmin=float(r["Tmin_K"]),
            )
    eps = sorted(rows)
    return dict(
        ep=np.array(eps),
        spiral=np.array([rows[e]["spiral"] for e in eps]),
        maxerr=np.array([rows[e]["maxerr"] for e in eps]),
        tmin=np.array([rows[e]["tmin"] for e in eps]),
        val=np.array([VAL[name.replace("gen", "")][e] for e in eps]),
    )


def summary(name: str, d: dict) -> dict:
    s = d["spiral"]
    bi = list(d["ep"]).index(BEST_EP[name.replace("gen", "")])
    return dict(
        val_best=d["val"][bi], spiral_best_val=s[bi], maxerr_best_val=d["maxerr"][bi],
        spiral_min=s.min(), spiral_max=s.max(), spiral_std=float(s[1:].std()),  # ex-ep10 warmup
        tmin_worst=d["tmin"].min(), undershoot=AMB - d["tmin"].min(),
    )


def build_table(don, doff, son, soff) -> None:
    def row(label, d, st):
        return (f"| {label} | {PARAMS[d]:,} | {st['val_best']:.1f} | {st['spiral_best_val']:.1f} | "
                f"{st['spiral_min']:.0f}-{st['spiral_max']:.0f} ({st['spiral_std']:.1f}) | "
                f"{AMB - st['undershoot']:.0f} (-{st['undershoot']:.0f}) |")
    md = [
        "## Ablation: GENERIC-enthalpy ON vs OFF, both push-forward K=2",
        "",
        "| config | params | val RMSE (held-out, K) | spiral RMSE @best-val (K) | "
        "spiral range ep20-80 [sigma] (K) | min T / undershoot (K) |",
        "|---|---|---|---|---|---|",
        row("**ON** (enthalpy GENERIC + enriched source)", "on", son),
        row("**OFF** (pure MeshGraphNet)", "off", soff),
        "",
        f"Physical floor = {AMB:.0f} K (ambient = Robin bath = initial). ON respects the "
        "maximum principle (min T within ~6 K of the floor); OFF undershoots to 140-190 K "
        "(up to ~150 K of unphysical sub-ambient cooling) at every checkpoint.",
        "",
        "Held-out single-pass val histories: sim001 8.1 K (ON) vs 36.2 K (OFF); "
        "sim006 36.0 K (ON) vs 63.4 K (OFF).",
    ]
    (OUT / "ablation_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md), "\n")


def build_ablation_figure(don, doff, son, soff) -> None:
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(don["ep"], don["spiral"], "o-", color=C_ON, lw=2, label="GENERIC ON")
    ax1.plot(doff["ep"], doff["spiral"], "s--", color=C_OFF, lw=2, label="GENERIC OFF")
    ax1.scatter([BEST_EP["on"]], [son["spiral_best_val"]], s=180, marker="*", color=C_ON, edgecolor="k", zorder=5)
    ax1.scatter([BEST_EP["off"]], [soff["spiral_best_val"]], s=180, marker="*", color=C_OFF, edgecolor="k", zorder=5)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("spiral global RMSE (K)")
    ax1.set_title("Spiral accuracy & variance\n(* = val-selected checkpoint)")
    ax1.legend(frameon=False)
    ax2.axhline(AMB, color="k", ls=":", lw=1.5, label=f"physical floor {AMB:.0f} K")
    ax2.plot(don["ep"], don["tmin"], "o-", color=C_ON, lw=2, label="GENERIC ON")
    ax2.plot(doff["ep"], doff["tmin"], "s--", color=C_OFF, lw=2, label="GENERIC OFF")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("minimum predicted T (K)")
    ax2.set_title("Maximum-principle violation\n(sub-ambient = unphysical)")
    ax2.legend(frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "ablation_figure.png", dpi=160); plt.close(fig)
    print("Saved", OUT / "ablation_figure.png")


# --------------------------- 3-curve comparison figures ---------------------------
def _nearest(coords, x, y):
    return int(np.argmin(np.hypot(coords[:, 0] - x, coords[:, 1] - y)))


def _peak_probes(gt, n=4):
    peak = gt.max(axis=0) - AMB
    used, picks = set(), []
    for f in np.geomspace(1.0, 0.03, n):
        order = np.argsort(np.abs(peak - f * peak.max()))
        idx = next((int(i) for i in order if int(i) not in used), int(order[0]))
        used.add(idx); picks.append((f"{f*100:.0f}%·peak", idx))
    return picks


def _grid(probes, times, curves, suptitle, png, coords):
    n = len(probes); ncols = 2 if n > 1 else 1; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    gt = curves[0][3]
    for ax, (label, idx) in zip(axes.flat, probes):
        for clabel, color, ls, T in curves:
            if clabel == "FEM":
                ax.plot(times, T[:, idx], ls, color=color, lw=2.0, label="FEM")
            else:
                rmse = float(np.sqrt(np.mean((T[:, idx] - gt[:, idx]) ** 2)))
                ax.plot(times, T[:, idx], ls, color=color, lw=1.6, label=f"{clabel} (RMSE {rmse:.0f} K)")
        xy = coords[idx]
        ax.set_title(f"{label}  @({xy[0]*1e3:.0f},{xy[1]*1e3:.0f}) mm | FEM peak {gt[:, idx].max():.0f} K", fontsize=9)
        ax.set_xlabel("time [s]"); ax.set_ylabel("T [K]")
        ax.axhline(AMB, color="0.6", lw=0.8, ls=":"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.suptitle(suptitle, fontsize=13); fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("Saved", png)


def build_comparison_figures() -> None:
    need = [OUT / "spiral_fem.npz", OUT / "spiral_genon_ep60.npz", OUT / "spiral_genoff_ep70.npz"]
    if not all(p.exists() for p in need):
        print("[skip] spiral .npz missing — regenerate the rollouts (see module docstring).")
    else:
        def lt(p):
            d = np.load(p, allow_pickle=True); return d["temperature"], d["coords"], d["times"]
        T_fem, coords_s, times_s = lt(need[0])
        T_on, _, _ = lt(need[1]); T_off, _, _ = lt(need[2])
        probes = [("center", 0.100, 0.100), ("r=0.04", 0.140, 0.100), ("r=0.07", 0.170, 0.100),
                  ("outer", 0.100, 0.165), ("cool corner", 0.030, 0.030)]
        sp = [(lbl, _nearest(coords_s, x, y)) for lbl, x, y in probes]
        _grid(sp, times_s,
              [("FEM", C_FEM, "-", T_fem), ("GENERIC ON (ep60)", C_ON, "-", T_on),
               ("GENERIC OFF (ep70)", C_OFF, "--", T_off)],
              "Spiral weld — FEM vs GENERIC ON (best ep60) vs GENERIC OFF (best ep70)",
              OUT / "compare_spiral_fem_on_off.png", coords_s)

    for sim in ("001", "006"):
        pon = OUT / f"rollout_pred_enthsrc_pf/sim_train_{sim}_rollout.npz"
        poff = OUT / f"rollout_pred_genoff_pf/sim_train_{sim}_rollout.npz"
        if not (pon.exists() and poff.exists()):
            print(f"[skip] val sim{sim} rollout .npz missing.")
            continue
        don_, doff_ = np.load(pon), np.load(poff)
        gt, coords_v, times_v = don_["ground_truth_temperature"], don_["coords"], don_["times"]
        _grid(_peak_probes(gt, 4), times_v,
              [("FEM", C_FEM, "-", gt), ("GENERIC ON", C_ON, "-", don_["predicted_temperature"]),
               ("GENERIC OFF", C_OFF, "--", doff_["predicted_temperature"])],
              f"Held-out val sim_{sim} — FEM vs GENERIC ON vs OFF",
              OUT / f"compare_val_sim{sim}_fem_on_off.png", coords_v)


def main() -> None:
    don, doff = load_sweep("genon"), load_sweep("genoff")
    son, soff = summary("on", don), summary("off", doff)
    build_table(don, doff, son, soff)
    build_ablation_figure(don, doff, son, soff)
    build_comparison_figures()


if __name__ == "__main__":
    main()
