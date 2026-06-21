# Training-set redesign — physics coverage for the spiral generalization claim

*Written 2026-06-21. Goal: a scientifically defensible training corpus whose ONLY
held-out axis is the trajectory topology, so the spiral generalization is the
clean wow result for the paper/congress.*

## Scientific framing (what stays held out)

**Contribution:** MGN+GENERIC trained on *single-pass* welds (straight, diagonal,
sinusoid, arc) generalizes to an unseen, self-crossing **3-turn spiral** that
re-heats already-hot zones. Therefore:

- **HELD OUT (do NOT add to training):** spiral / multi-pass / re-heating
  trajectories. This is the generalization claim.
- **COVERED (must bracket the spiral test):** everything else — plate size, weld
  speed, boundary conditions, cooling/relaxation, process params. A reviewer must
  not be able to attribute failure to domain-size or BC extrapolation; only the
  trajectory is novel.

Today the spiral is OOD on 4 axes at once (trajectory + plate size + speed +
truncated cooling). Fixing the non-trajectory axes **strengthens** the claim.

## Measured gaps (current 125-sim corpus vs the 0.2 m, 10 mm/s spiral)

| Axis | Train (current) | Spiral | Status |
|---|---|---|---|
| Trajectory | straight/diag/sinusoid/arc | spiral 3 turns | hold out ✅ |
| Plate size | ≤0.085×0.065 m | 0.20×0.20 m | gap (2.4–3×) |
| Weld speed | 2–8 mm/s | 10 mm/s | gap (25% over) |
| Cooling | 65% of peak excess dissipated; ends +1363 K | — | truncated |
| Peak T | up to 9059 K (unphysical) | 3301 K | dynamic range too wide |
| Power, η, thickness, h_conv, T_amb, Goldak | — | — | already inside ✅ |

## Design decisions (agreed)

- **Plate size: 0.08–0.25 m per side** (full bracket; 0.20 m interior). Keep
  element size ≈ **3.0 mm** (matches the spiral's 0.2/64 ≈ 3.1 mm) so the per-step
  diffusion stencil matches inference. ⇒ meshes ~750 (0.08 m) to ~7000 (0.25 m)
  nodes. **~10× heavier than now → RunPod.**
- **Weld speed: 2–12 mm/s** (brackets 10 mm/s).
- **Cooling: near-field relaxation criterion.** After the source switches off,
  keep stepping until `max(T − T_inf) < 0.07 · peak_excess` (peak fully relaxed)
  OR a `t_cool_max ≈ 30 s` cap. Do NOT target whole-plate ambient (a 0.2 m plate
  needs ~770 s to equilibrate — infeasible and unnecessary; the far-field echo is
  fixed by seeing full *local* relaxation, not bulk equilibration).
- **Corpus: ~300 train / 60 val** (360 sims).
- **Dynamic-range cap:** keep peak T physical (steel boils ~3130 K). The current
  9000 K source-node spikes pollute normalization. Investigate a phase-change /
  enthalpy latent-heat sink or a source-density cap; secondary but worth doing.
- **BC coverage (stratified, not purely random):** guarantee representation of
  the full `h_conv` range (5–40), radiation on/off (emissivity 0 vs >0), Dirichlet
  edges (enough samples), and the `T_ambient` range — with the spiral's pure-Robin
  convective regime well inside.

## Implementation checklist

- `src/simulation/generate_dataset.py`:
  - `WIDTH_RANGE`/`HEIGHT_RANGE` → `(0.08, 0.25)`; `SPEED_RANGE` → `(2e-3, 12e-3)`.
  - element size → ~3.0 mm (the `--element_size` default / corridor logic).
  - stratified BC sampling (cycle Dirichlet/radiation flags so coverage is
    guaranteed across 360 sims rather than left to chance).
  - **parallelize** generation across CPU cores (sims are independent) — essential
    at 360 large sims; current driver is serial.
- `src/simulation/thermal_solver.py`:
  - cooling-relaxation stopping criterion (step with source off until peak excess
    < threshold or `t_cool_max`), replacing the fixed cooling-tail length.
  - (optional) latent-heat / peak-T cap for physical dynamic range.
- Regenerate dataset on RunPod (many-core box; parallel FEM), then retrain
  GENERIC-full (`config.generic_full_n5.json` recipe) on the new corpus on a
  cloud GPU.
- Re-run the spiral eval — now an honest single-axis (trajectory-only)
  generalization test.

## Why this is the right move

We spent the session on architectural fixes (noise, push-forward, GENERIC energy
& full). They helped stability but hit a ~65–93 K spiral floor. The audit shows a
large part of that floor is **train/test coverage mismatch**, not architecture.
Closing the non-trajectory gaps (a) likely lowers the floor and (b) makes the
remaining spiral error cleanly attributable to trajectory generalization — the
exact story the paper needs.

**Compute note (revised — laptop is fine for generation):** the dev laptop is a
Ryzen AI 9 HX 370 (12 physical / 24 logical cores, 63 GB RAM). FEM generation is
embarrassingly parallel (independent sims); the current driver is **serial**,
which is the only real bottleneck. Parallelized across the 12 cores, the full
360-sim corpus (mean ~3000-node meshes, ~900 steps) generates in **~5–8 h
overnight, locally** (memory trivial). **Action: parallelize `generate_dataset`
(`ProcessPoolExecutor`) — this is the enabling change.**

**Training** is the heavier part: ~3000–7000-node graphs won't fit batch 16 in
8 GB VRAM → drop to batch ~4 (~6 GB), ~6–10× slower/epoch (~20–30 min). Feasible
locally but slow; cloud GPU (see `RUNPOD.md`) is optional here, decided later.
