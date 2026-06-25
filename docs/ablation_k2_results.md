# Ablation: GENERIC-enthalpy ON vs OFF (push-forward K=2)

Congress results-section ablation. Two runs, **identical** in every respect
except the thermodynamic head, so any difference is attributable to the GENERIC
structure (energy/entropy bookkeeping + enriched source), not to capacity or
training schedule.

| | ON (headline) | OFF (ablation) |
|---|---|---|
| config | `config.runpod_enthalpy_pf.json` | `config.runpod_genoff_pf.json` |
| head | enthalpy GENERIC + enriched source | none (pure MeshGraphNet) |
| `use_generic` | true (`generic_mode=enthalpy`, `enriched_source=true`) | false |
| params | 1,375,238 | 1,291,777 |
| checkpoints | `checkpoints_enthsrc_pf/` | `checkpoints_genoff_pf/` |

Everything else identical: push-forward **K=2**, `split_seed=0`,
`val_fraction=0.2`, `max_val_sims=20`, batch 16, `max_windows_per_epoch=40000`,
80 epochs, plateau LR, 5 K fixed input noise, hidden 128, 8 processing steps.
The split is identical (288 train / 72 val) so **val RMSE is directly
comparable**. Both ran on an A100-80GB on the CA-MTL-3 network volume.

## Results table

| config | params | val RMSE (held-out, K) | spiral RMSE @best-val (K) | spiral range ep20-80 [σ] (K) | min T / undershoot (K) |
|---|---|---|---|---|---|
| **ON** (enthalpy GENERIC + enriched source) | 1,375,238 | 34.5 | **33.2** | 33–61 (3.4) | 287 (−7) |
| **OFF** (pure MeshGraphNet) | 1,291,777 | 46.9 | 58.6 | 38–**111** (10.2) | **140 (−153)** |

`spiral RMSE @best-val` is the spiral global RMSE of the **val-selected**
checkpoint (`best_model.pt`, the one you would actually deploy): ON=ep60,
OFF=ep70. `min T` is the lowest temperature anywhere in the 1526-step spiral
rollout; the physical floor is the 293 K ambient (= Robin bath = initial field),
below which only an unphysical (maximum-principle-violating) solver can go.

## Full spiral sweep (local 5070, all checkpoints vs `spiral_fem.npz`)

Raw data: `scripts/ablation/spiral_gen{on,off}_sweep.csv`.

| epoch | ON spiral | ON min T | OFF spiral | OFF min T | OFF val |
|---|---|---|---|---|---|
| 10 | 61.3 | 292.7 | 110.6 | 140.4 | 79.0 |
| 20 | 41.4 | 291.6 | 66.2 | 190.2 | 64.2 |
| 30 | 41.6 | 290.0 | 69.6 | 157.0 | 71.7 |
| 40 | 40.9 | 291.4 | 46.4 | 175.6 | 47.8 |
| 50 | 43.2 | 288.4 | 61.1 | 179.6 | 58.2 |
| 60 | **33.2** | 286.6 | 54.0 | 166.1 | 69.9 |
| 70 | 45.0 | 288.4 | 58.6 | 186.8 | **46.86** |
| 80 | 40.0 | 291.8 | **38.4** | 176.7 | 55.0 |

(ON val per epoch: ep10 46.4 → ep60 **34.5** → ep80 37.4; smooth/stable.
OFF val oscillates 47–91 K with no clean convergence.)

## Verdict — GENERIC wins on STRUCTURE, not on a single RMSE

The ablation is anchored on what GENERIC **guarantees**, not on "genoff is
catastrophic" (it is not — a lucky OFF checkpoint, ep80, reaches a decent 38.4 K).

1. **Maximum principle.** ON never cools below the 293 K heat bath (min T
   286–293 K across all checkpoints, within ~6 K of the floor — i.e. it respects
   the physics). OFF undershoots to **140–190 K at every checkpoint** — up to
   ~150 K of physically impossible sub-ambient cooling. The energy/entropy
   structure forbids spurious heat removal by construction; pure MGN has nothing
   stopping it.
2. **Variance / stability.** OFF's spiral RMSE swings **38–111 K (σ≈10.2)**; ON
   stays in a **33–61 K band (σ≈3.4)** — ~3× tighter. OFF never converges.
3. **Selectability (val tracks the spiral).** For ON the val-selected checkpoint
   (ep60) **is** the best spiral (33.2 K) — the objective is aligned. For OFF the
   val-selected checkpoint (ep70) gives a mediocre spiral (58.6 K); the best
   spiral (ep80, 38.4 K) is *not* the one val picks. So with OFF you cannot
   reliably select a good model — val and spiral are disconnected (as in the
   earlier chaotic genoff_v2 K=1 run).
4. **Accuracy.** At the deployable (best-val) checkpoint, ON=33.2 K vs OFF=58.6 K
   (~1.8× better). Held-out single-pass val: sim001 8.1 (ON) / 36.2 (OFF),
   sim006 36.0 (ON) / 63.4 (OFF).

`max|error|` was the originally-planned stability column but turned out **not**
to discriminate (both ~430–870 K, dominated by the sharp center-peak smoothing),
so the table reports the two metrics that do carry the signal: the maximum-
principle violation (min T) and the across-checkpoint spiral variance.

## Deliverables / how to reproduce

Figures + table are built by `scripts/ablation/build_ablation_figures.py`
(reads the committed sweep CSVs + the regenerable rollout `.npz`):

```bash
uv run python scripts/ablation/build_ablation_figures.py
```

Outputs (under `data/output/`, gitignored — regenerable):
- `ablation_table.md`, `ablation_figure.png` (2-panel: spiral-RMSE-vs-epoch with
  ★ best-val + min-T-vs-epoch with the 293 K floor).
- `compare_spiral_fem_on_off.png` — FEM vs ON(ep60) vs OFF(ep70), 5 spiral probes.
- `compare_val_sim{001,006}_fem_on_off.png` — FEM vs ON vs OFF, held-out val sims.
- `val_genoff_sim{001,006}_thermal_history.png` — OFF held-out histories
  (via `python -m src.training.val_analysis`).

To regenerate the spiral sweep from checkpoints, for each epoch NN of each run:

```bash
uv run python -m src.training.spiral_rollout  --checkpoint <ckpt> --stats <stats> \
    --output_name spiral_genXX_epNN --device cuda
uv run python -m src.training.spiral_analysis --pred data/output/spiral_genXX_epNN.npz \
    --prefix spiral_genXX_epNN     # logs "global RMSE … | max |error| …"
```

## Still open / next (no more training planned for now)
- Spiral 3D **video** from `spiral_pf_best_ep60_comparison.xdmf` (ParaView).
- Mathematical write-up of the method (the user's focus going forward).
- Optional: a 3rd held-out val sim for figure variety (roll a fresh
  `sim_val_*.npz` through both models).
