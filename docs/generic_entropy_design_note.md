# Design note — Full (entropy-aware) GENERIC vs the current energy-only version

*Written 2026-06-20 while the energy-only GENERIC vs non-physics A/B was launching.
Decision record so we don't re-derive this later.*

## What we have today (`src/models/meshgraphnet.py`)

The `GenericThermalHead` is a **degeneracy-only / energy-conserving** relaxation of
GENERIC, **not** a full one:

- State vector **z = temperature T**. Potentials (uniform nodal volumes):
  `E = Σ ρCp·T_i·V_i` (→ **∇E is a constant vector**), `S = Σ ρCp·ln T_i·V_i`.
- The dissipative increment is the network's raw `dT_tilde` made energy-conserving
  by an **analytical projection** `P = I − ĝE ĝEᵀ` (`energy_conserving_projection`)
  → per-graph mean-removal, so `∇E·(P dT_tilde) = 0` ⇒ `M·∇E = 0` (degeneracy) ✅.
- Plus an analytical external exchange (Goldak source + Newton boundary cooling)
  with two learnable softplus gains.

**What it does NOT do:** there is **no explicit SPSD Onsager `M`** (no `WWᵀ`, no
Cholesky, no learned operator) and **no guarantee of entropy production**
(`∇Sᵀ M ∇S ≥ 0`). The second-law half of GENERIC is absent.

## The question

Is it worth building a *real* GENERIC with entropy considered (SPSD `M`,
entropy non-decreasing), or do we keep the energy-only version?

## Why entropy could actually fix our far-field "echo" (real mechanism)

Energy conservation alone only says "what rises somewhere falls elsewhere"; it
**does not forbid creating spurious structure**. It permits **entropy-decreasing,
anti-diffusive** redistributions — moving heat from cold to hot. Our pathology
(cool-corner "thermal echoes": heat appearing in a cold zone against the
gradient) is exactly that.

A genuinely **SPSD `M`** (so `dS/dt ≥ 0`) **forbids this by construction**: the
dissipative term can only move heat **hot → cold**. So there is a concrete
physical mechanism by which the entropy-aware version would suppress the echo
that the energy-only version cannot touch. Not a guarantee — but a legitimate
argument, not an aesthetic one.

## How to implement it *tractably* (the important part)

**Not** as a dense `M = WWᵀ` (that's N×N — intractable for thousands of nodes).
The elegant, correct form on a graph is a **Laplacian of learned non-negative
conductances** (a learned nonlinear heat equation):

1. Learn per-edge conductances `w_ij = softplus(MLP(edge_feats, T_i, T_j)) ≥ 0`.
2. Use `M = L_w = D − W` (graph Laplacian) → **SPSD by construction**, and
   `M·1 = 0`. Since `∇E` is uniform, `M·∇E = 0` ⇒ **degeneracy is free**.
3. Evolve the **energy** by `ė = L_w · μ` with the thermodynamic force
   `μ_i = ∂S/∂e_i = 1/T_i` (the entropy gradient); then `Ṫ = ė /(ρCp)`.

Sign/consistency check (per node): `ė_i = Σ_j w_ij (μ_i − μ_j) = Σ_j w_ij(1/T_i − 1/T_j)`.
For a hot node (small `1/T_i`) next to a cold node (large `1/T_j`), `ė_i < 0`
→ **hot loses energy** ✅. And `dS/dt = μᵀ L_w μ = ½ Σ w_ij (μ_i−μ_j)² ≥ 0` ✅.

Properties obtained **by construction**: SPSD, degeneracy `M·∇E=0`, entropy
production `dS/dt ≥ 0`, monotone hot→cold flow. It also naturally captures
**temperature-dependent conductivity** via `w_ij(T)`. Cost: a message-passing
operation, ~1 day of careful work + tests. Structurally far stronger than the
current projection + source/sink.

## Why NOT now (discipline)

1. **We haven't measured the energy-only version yet.** Building the full one
   before knowing whether the simple one moves the spiral is building blind.
2. **Track record this session is humbling:** every "more sophisticated"
   intervention (cosine LR, push-forward K=3) lost to the simple plateau /
   single-step baseline (champion spiral RMSE = 65 K). Healthy skepticism.
3. **Physical caveat — phase change.** The code assumes **constant `ρCp`**. Latent
   heat lives in `Cp(T)`, which changes `E`, `S` and `μ = ∂S/∂e`. A rigorous full
   GENERIC consistent with our welding physics would need **variable `Cp(T)`** —
   extra complexity the current version ignores.

## Recommendation / decision gate

- **Finish the cheap A/B** (energy-only GENERIC vs non-physics baseline; K=1
  single-step + dynamic noise, 8 processing steps, 100 epochs;
  `config.generic_baseline.json` / `config.generic_physics.json`). Judge on the
  **spiral** (global RMSE, cool-corner echo, drift), not val.
- **Then decide:**
  - Energy-only **helps directionally** → the entropy version is well-motivated;
    build the **learned-conductance graph-Laplacian** form above (and consider
    `Cp(T)` for phase change).
  - Energy-only **does nothing / hurts** → GENERIC likely isn't the lever here;
    don't spend the day — pivot (more/closer data, the 1D FDM sandbox).

**One-liner:** the entropy version is the *correct* version and has a real
mechanism to kill the echo, but earn the right to build it by first checking
whether energy conservation alone does anything.
