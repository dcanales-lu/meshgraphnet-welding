# The welding surrogate — final theory and model structure (consolidated)

*Single reference pulling together the full design: the physics, the GENERIC
framework, the four lessons that shaped the architecture, and the final model
(enriched-source enthalpy GENERIC). Companion derivations:
`generic_thermodynamics_congress.md` (physics why) and
`enthalpy_generic_implementation.md` (discrete scheme). Math in LaTeX.*

> **Status (2026-06-23):** the final architecture below is the design we
> converged to. Parts I–IV are settled (theory + code). The **enriched source**
> (§III.4) is the current best hypothesis for the remaining "cold peak" and is
> **training / pending spiral validation** — flagged explicitly in §V.

---

## Part I — Problem and framework

### I.1 The physics
Transient heat conduction in a thin plate with a moving Goldak arc and
convective/radiative boundaries:

$$
\rho\,c_p(T)\,\partial_t T = \nabla\!\cdot\!(k(T)\nabla T) + Q(\mathbf x,t),
\qquad
\text{Robin/Dirichlet boundaries.}
$$

Crucially, the metal **melts and boils** → strong **latent heat** → $c_p(T)$
varies by orders of magnitude. Energy is the **enthalpy**
$h(T)=\rho\int_{T_0}^T c_p^{\text{app}}\,d\tau$, with apparent heat capacity

$$
c_p^{\text{app}}(T)=c_p^{\text{sens}}+L_f\frac{df_\ell}{dT}+L_v\frac{df_v}{dT}
\quad(\text{fusion + vaporization bumps}).
$$

### I.2 GENERIC (pure dissipative thermal)
Evolution of a state $\mathbf z$ with energy $E$, entropy $S$, antisymmetric $L$
and SPSD $M$: $\dot{\mathbf z}=L\,\partial_z E+M\,\partial_z S$, with degeneracies
$L\,\partial_z S=0$, $M\,\partial_z E=0$ giving $\dot E=0$ (1st law) and
$\dot S=\partial_z S^\top M\,\partial_z S\ge 0$ (2nd law). Heat conduction has **no
reversible part** ($L=0$):

$$
\dot{\mathbf z}=M\,\partial_z S + \text{(external source/cooling)}.
$$

The two design questions that decide consistency: **what is the state $\mathbf z$**,
and **is $M\partial_z E=0$ the right energy conservation**.

---

## Part II — The four lessons (why the architecture is what it is)

**Lesson 1 — Degeneracy alone is not enough.** The first head conserved energy by
projecting the decoder increment onto $\nabla E^\perp$ (mean-removal
$P=I-\tfrac{1}{N}\mathbf 1\mathbf 1^\top$). It enforces $M\nabla E=0$ but has **no
entropy production** and no learned operator — it removes global drift but cannot
shape diffusion. *(This is the `energy` mode; superseded.)*

**Lesson 2 — A learned SPSD operator gives the 2nd law.** Replace the projection
by a graph Laplacian of learned non-negative symmetric conductances
$M=L_w=D-W$, $w_{ij}=w_{ji}\ge0$, acting on $\mu=1/T$. Then SPSD, degeneracy
($L_w\mathbf 1=0$), entropy production ($\mu^\top L_w\mu\ge0$) and hot→cold are
**all guaranteed by construction**. *(This is the `full` mode.)*

**Lesson 3 — Conserve ENERGY, not temperature.** The `full` head evolved
**temperature** directly, $\Delta T=L_w\mu$, implicitly assuming $E\propto T$
(constant $c_p$). With latent heat that is **the wrong conserved quantity**:
$\sum_i\Delta T_i=0$ is *temperature-sum* conservation, off from energy by the
$c_p^{\text{app}}(T)$ factor that blows up at the melt pool — so the peak is
mis-handled. **Fix:** evolve the **enthalpy** and recover $T$ through the
latent-heat curve (the classical *enthalpy method*): $\Delta h=L_w\mu$,
$T=h^{-1}(h)$. (Note $\mu=1/T$ is correct in both — it is $\partial S/\partial e$
by definition, valid for any $c_p$.)

**Lesson 4a — Numerical conditioning: work in $H$ [K], not $h$ [J/m³].** Raw
enthalpy $h\sim10^9$ makes $\partial T/\partial h=1/(\rho c_p)\sim2\times10^{-7}$,
which drives parameter gradients below Adam's $\varepsilon=10^{-8}$ → **training
stalls** (observed: loss frozen, diffusion never develops). Use
**temperature-equivalent enthalpy** $H(T)=h/(\rho c_p^{\text{sens}})=\int
(c_p^{\text{app}}/c_p^{\text{sens}})\,d\tau$ [K]: a constant rescale (so energy
conservation is unchanged), but increments are $O(\mathrm K)$ and gradients
$O(1)$. Confirmed: gradients $10^{-13}\to10^{-4}$, loss unstuck from epoch 1.

**Lesson 4b — The peak is limited by the SOURCE, not the energy law.** Both
structure-preserving heads (`full`, `enthalpy`) run **cold at the active peak**,
while the *unconstrained* MeshGraphNet nails it (but is unstable). Reason: the
dissipative operator can **only cool** the hottest node (hot→cold by
construction); the **only thing heating the peak is the external source**, which
was a single global scalar $\mathrm{softplus}(g)\,q$ — too rigid to inject enough.
**Fix:** enrich the external term (the *non-conservative* exchange, which the
GENERIC guarantees do **not** constrain) with a per-node learned modulation. The
dissipation stays structure-preserving; only the source gains expressivity. *(This
is `enthalpy` + `enriched_source` — the final model.)*

---

## Part III — The final model: enriched-source enthalpy GENERIC

State rolled by the network: temperature $T$ (so rollout/data pipeline are
unchanged). Inside the head, per step:

### III.1 Enthalpy curve (fixed, no learning)
$$
H(T)=\int_{T_0}^{T}\frac{c_p^{\text{app}}(\tau)}{c_p^{\text{sens}}}\,d\tau\;[\mathrm K],
\qquad T=H^{-1}(H)\;\text{(monotone, invertible)} .
$$
Tabulated once from `MaterialProperties` (identical to the FEM), stored as
buffers, queried by a differentiable piecewise-linear `_interp1d`. Away from phase
change $H\approx T$; at melting/boiling the curve plateaus (latent heat).

### III.2 Learned conductances (symmetric, ≥0)
$$
w_{ij}=e^{\alpha}\,\mathrm{softplus}\!\big(\mathrm{MLP}(\mathbf z_i+\mathbf z_j,\,
(\mathbf z_i-\mathbf z_j)^2,\,T_i+T_j,\,|T_i-T_j|)\big)=w_{ji}\ge0,
$$
$\mathbf z$ = processor node latents, $\alpha$ = learnable global scale.

### III.3 Dissipative energy change (structure-preserving)
$$
\boxed{\;\Delta H_i^{\text{diss}}=(L_w\boldsymbol\mu)_i=\sum_{j\sim i}w_{ij}(\mu_i-\mu_j),
\quad \mu_i=1/T_i\;}
$$
Guarantees (by construction): $\sum_i\Delta H_i^{\text{diss}}=0$ (**1st law** —
genuine energy conservation, since $H\propto$ energy); $\Delta S\propto
\mu^\top L_w\mu=\tfrac12\sum w_{ij}(\mu_i-\mu_j)^2\ge0$ (**2nd law**); hot→cold.

### III.4 Enriched external exchange (the source — non-conservative, flexible)
$$
\boxed{\;\Delta H_i^{\text{ext}}
=\mathrm{softplus}\!\big(g_q+\mathrm{MLP}_q(\mathbf z_i,q_i,T_i)\big)\,q_i
+\mathrm{softplus}\!\big(g_c+\mathrm{MLP}_c(\mathbf z_i,T_i,T_{\infty,i})\big)\,r_i(T_{\infty,i}-T_i)\;}
$$
Per-node learned gains (zero-initialized ⇒ start at the scalar head) modulate the
Goldak heating ($\ge0$) and the Newton cooling on Robin nodes ($r_i$). This is the
**only** non-structure-preserving term, and GENERIC permits it: the degeneracy /
2nd law apply to the **dissipative** operator, not to the external energy exchange
with the source/environment.

### III.5 Update and temperature recovery
$$
H_i^{\text{new}}=H(T_i)+\Delta H_i^{\text{diss}}+\Delta H_i^{\text{ext}},
\qquad
\Delta T_i=H^{-1}(H_i^{\text{new}})-T_i .
$$
The latent-heat plateau means a large $\Delta H$ yields a small $\Delta T$ at the
melt pool — exactly as in the FEM. The head returns the normalized $\Delta T$.

### III.6 What this buys (the thesis)
- **Stability** (energy conserved + $dS/dt\ge0$): no autoregressive blow-ups — the
  failure mode of the unconstrained net.
- **Correct peak physics** (enthalpy method): no constant-$c_p$ distortion.
- **Enough peak heating** (enriched source): the structure-preserving dissipation
  cools the peak; a flexible source can now counter it.
- **Far-field consistency**: the SPSD operator forbids entropy-decreasing,
  anti-diffusive moves → mitigates spurious cold-zone "echoes".

---

## Part IV — Code map

| concept | code (`src/`) |
|---|---|
| apparent heat capacity / latent heat | `simulation/thermal_solver.py::MaterialProperties.cp_apparent` |
| enthalpy head | `models/meshgraphnet.py::EnthalpyGenericThermalHead` |
| conductances, Laplacian dissipation | inherited from `FullGenericThermalHead` |
| enthalpy table + `_interp1d` | `EnthalpyGenericThermalHead.__init__` / module helper |
| enriched source MLPs | `EnthalpyGenericThermalHead` (`enriched_source=True`) |
| degeneracy-only head (lesson 1) | `GenericThermalHead` (`generic_mode="energy"`) |
| temperature head (lesson 2–3) | `FullGenericThermalHead` (`generic_mode="full"`) |
| config switches | `MeshGraphNetConfig.generic_mode`, `.enriched_source` |
| training config | `config.generic_enthalpy_src_v2.json` |

Heads are selected in `MeshGraphNet.__init__`; conductance heads bypass the
decoder. Normalization constants ride in `state_dict` via
`set_normalization`/buffers, so checkpoints reproduce the exact physics.

---

## Part V — Status, guarantees vs claims, open validation

**Guaranteed by construction (proven):** SPSD dissipation, energy conservation
($\sum\Delta H^{\text{diss}}=0$), entropy production ($dS/dt\ge0$), hot→cold flow,
latent-heat-consistent $T(H)$. Unit-tested in `tests/test_meshgraphnet.py`.

**Empirical, established on the v2 dataset (held-out, single-pass):** strong
generalization across geometry / plate size / process / BC — e.g. held-out val
sims at $\sim$25–48 K global RMSE, far-field $\sim$1 K (no echo). This already
supports a paper.

**Empirical, OPEN (the spiral, multi-pass OOD):** the unconstrained net is
accurate-but-unstable (diverges on extreme sims); the temperature `full` head is
stable-but-caps-the-peak (≈88 K); the `enthalpy` head is stable and improving
(≈108 K @ep25, still descending) but ran **cold at the peak**. The
**enriched-source enthalpy** model (§III) is the hypothesis that should deliver
*stable + correct peak* simultaneously — **training now, pending spiral
validation**. If it beats the baselines on the spiral while preserving the
structure, that is the headline result.

**Caveat to revisit:** the dataset still has a soft-cap tail (~11% of sims peak
>4000 K, up to 7569 K — unphysical for metal). Tightening the vaporization cap in
`MaterialProperties` and regenerating would clean the tail; the enthalpy head's
$H(T)$ table auto-syncs (same `MaterialProperties`). Deferred until after the
enriched-source result, to keep the A/B on identical data.
