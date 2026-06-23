# Enthalpy-state GENERIC head — implementation mathematics

*Companion to `generic_thermodynamics_congress.md` (which gives the physics
"why"). This note derives the **discrete numerical scheme actually coded** in
`EnthalpyGenericThermalHead` (`src/models/meshgraphnet.py`): variables, units,
the enthalpy lookup table, the differentiable inverse map, the per-step update,
and the discrete proofs of the two laws. Equations in LaTeX for compilation.*

---

## 1. Variables and units

Per mesh node $i$ (tributary volume $V_i$, taken uniform):

| symbol | meaning | units |
|---|---|---|
| $T_i$ | temperature | $\mathrm{K}$ |
| $h_i$ | **volumetric enthalpy** (energy density) | $\mathrm{J\,m^{-3}}$ |
| $\mu_i=1/T_i$ | thermodynamic force $\partial S/\partial e$ | $\mathrm{K^{-1}}$ |
| $w_{ij}$ | learned edge conductance | $\mathrm{J\,m^{-3}\,K}$ |
| $q_i$ | Goldak volumetric power (feature `q_goldak`) | $\mathrm{W\,m^{-3}}$ |
| $\rho$ | density $=7850$ | $\mathrm{kg\,m^{-3}}$ |
| $c_p^{\text{app}}(T)$ | apparent heat capacity (eq. 3) | $\mathrm{J\,kg^{-1}K^{-1}}$ |

The network state that is *rolled autoregressively* remains $T$ (so the rollout
loop and data pipeline are unchanged); the **head internally converts
$T\!\to\!h$, updates energy, and converts back $h\!\to\!T$**, outputting the
temperature increment $\Delta T$.

---

## 2. The enthalpy curve $h(T)$ (built once, no learning)

Define the volumetric enthalpy relative to a reference temperature $T_0$:

$$
h(T)\;=\;\rho\int_{T_0}^{T} c_p^{\text{app}}(\tau)\,d\tau ,
\tag{1}
$$

with the apparent heat capacity carrying the latent heats of fusion ($L_f$) and
vaporization ($L_v$):

$$
c_p^{\text{app}}(T)\;=\;c_p^{\text{sens}}
   \;+\;L_f\,\frac{df_\ell}{dT}\;+\;L_v\,\frac{df_v}{dT},
\tag{2}
$$

$$
\frac{df_\ell}{dT}=\frac{1}{s_\ell\sqrt\pi}\,e^{-((T-T_m)/s_\ell)^2},
\qquad
\frac{df_v}{dT}=\frac{1}{s_v\sqrt\pi}\,e^{-((T-T_b)/s_v)^2},
\tag{3}
$$

i.e. normalized Gaussian bumps centred on melting $T_m$ and boiling $T_b$ (this is
exactly `MaterialProperties.cp_apparent` in the solver, so the surrogate's
$T\!\leftrightarrow\!h$ map is **identical to the FEM ground truth**).

**Temperature-equivalent rescaling (numerical conditioning — important).** The
raw volumetric enthalpy is $h\!\sim\!10^9\,\mathrm{J\,m^{-3}}$, so the inverse map
has slope $\partial T/\partial h = 1/(\rho c_p^{\text{app}})\sim 2\times10^{-7}$.
That factor multiplies every gradient flowing back to the conductances/gains,
pushing them **below Adam's $\varepsilon=10^{-8}$ floor** — training stalls (the
diffusion never develops; observed empirically). We therefore work in
**temperature-equivalent enthalpy** (units kelvin):

$$
H(T)\;=\;\frac{h(T)}{\rho\,c_p^{\text{sens}}}
   \;=\;\int_{T_0}^{T}\frac{c_p^{\text{app}}(\tau)}{c_p^{\text{sens}}}\,d\tau .
\tag{4}
$$

Away from any phase change $c_p^{\text{app}}=c_p^{\text{sens}}$ so $H\approx
T-T_0$ (unit slope); across melting/boiling the slope spikes (the plateau). This
is a **constant rescale of $h$**, so it changes nothing thermodynamically —
energy conservation is identical ($\sum_i\Delta H^{\text{diss}}_i=0 \Leftrightarrow
\sum_i\Delta h^{\text{diss}}_i=0$) — but now increments are $O(\mathrm K)$ and
gradients are $O(1)$, well above $\varepsilon$. All quantities below ($\Delta
h^{\text{diss}}$, $\Delta h^{\text{ext}}$, the source/cooling gains) are read in
these $H$-units.

**Discretisation (code).** On a uniform grid $T^{(0)}<\dots<T^{(G-1)}$
($G=4096$, from $200$ to $8000\,\mathrm{K}$ to cover the hottest sims):

$$
H^{(g)} \;=\; \sum_{m\le g}\frac{c_p^{\text{app}}\!\big(T^{(m)}\big)}{c_p^{\text{sens}}}\,\Delta T^{(m)} ,
\qquad \Delta T^{(m)}=T^{(m)}-T^{(m-1)} .
\tag{5}
$$

Since $c_p^{\text{app}}>0$, $H^{(g)}$ is **strictly increasing** $\Rightarrow$
invertible. The arrays $(T^{(g)},H^{(g)})$ are stored as model buffers (they ride
in `state_dict`, so checkpoints reproduce the exact curve).

---

## 3. Differentiable interpolation (`_interp1d`)

We need $h=\mathcal H(T)$ and its inverse $T=\mathcal H^{-1}(h)$, both
differentiable for back-propagation. With a monotone grid $x^{(g)}\!\to f^{(g)}$,
for a query $x$ we locate the bracket $x^{(g-1)}\le x< x^{(g)}$ (binary search,
`torch.searchsorted`) and linearly interpolate:

$$
\mathrm{interp}(x)\;=\;f^{(g-1)}
   +\frac{x-x^{(g-1)}}{x^{(g)}-x^{(g-1)}}\,\big(f^{(g)}-f^{(g-1)}\big).
\tag{5}
$$

This is piecewise-linear, so $\partial(\mathrm{interp})/\partial x$ equals the
local slope — finite and well defined. Queries are clamped to the grid range
(flat extrapolation) for safety.

---

## 4. One forward step (the discrete update)

Given the (de-normalized) nodal temperatures $T_i$ and the processor latents,
the head computes, per directed edge $s\!\to\!r$ and node:

**(a) Learned symmetric conductances** ($w_{ij}=w_{ji}\ge0$ from symmetric edge
features; identical to the "full" head):

$$
w_{ij}=e^{\alpha}\,\mathrm{softplus}\!\big(\text{MLP}(\,\mathbf z_i\!+\!\mathbf z_j,\,
(\mathbf z_i\!-\!\mathbf z_j)^2,\,T_i\!+\!T_j,\,|T_i\!-\!T_j|\,)\big),
\tag{6}
$$

with $\mathbf z$ the node latents and $\alpha=$ `log_cond_scale` a learnable
global scale.

**(b) Dissipative ENERGY change** — the SPSD graph Laplacian acting on
$\mu=1/T$:

$$
\boxed{\;\Delta h_i^{\text{diss}}
   =(L_w\boldsymbol\mu)_i
   =\sum_{j\sim i} w_{ij}\,(\mu_i-\mu_j)\;}
\qquad \mu_i=\frac1{T_i}.
\tag{7}
$$

In code this is a `scatter`-add over edges: edge $s\!\to\!r$ contributes
$w_{sr}(\mu_r-\mu_s)$ to receiver $r$; summed over the bidirectional edge set this
is exactly row $r$ of $L_w\boldsymbol\mu$.

**(c) External ENERGY exchange** (source adds energy, boundary cooling removes
it; non-negative `softplus` gains $g_q,g_c$):

$$
\Delta h_i^{\text{ext}}
   =\underbrace{\mathrm{softplus}(g_q)\,q_i}_{\text{Goldak heating}}
   +\underbrace{\mathrm{softplus}(g_c)\,r_i\,(T_{\infty,i}-T_i)}_{\text{Newton cooling (Robin nodes }r_i)} .
\tag{8}
$$

**(d) Enthalpy update and inverse map to temperature:**

$$
h_i^{\text{new}}=\mathcal H(T_i)+\Delta h_i^{\text{diss}}+\Delta h_i^{\text{ext}},
\qquad
T_i^{\text{new}}=\mathcal H^{-1}\!\big(h_i^{\text{new}}\big),
\tag{9}
$$

$$
\boxed{\;\Delta T_i \;=\; T_i^{\text{new}}-T_i\;}
\tag{10}
$$

which is the (normalized) quantity the head returns. **The key nonlinearity is
in (9):** at the melt pool $\mathcal H^{-1}$ is nearly flat (the enthalpy
plateau), so a large $\Delta h$ yields a small $\Delta T$ — latent heat absorbs
the energy, exactly as in the FEM.

---

## 5. The two laws, discretely

**1st law (energy).** The graph Laplacian has zero column sums,
$\mathbf 1^\top L_w=\mathbf 0$, hence the dissipative part conserves total
enthalpy per graph:

$$
\sum_i \Delta h_i^{\text{diss}}
   =\mathbf 1^\top L_w\boldsymbol\mu
   =0 .
\tag{11}
$$

Because the conserved quantity is now **energy** $h$ (not temperature), (11) is
the *genuine* first law even with latent heat — the defect of the temperature
formulation (§3.2 of the physics note) is removed. The external term (8) changes
the total energy on purpose (it is the exchange with source/environment).

**2nd law (entropy).** With $S=\sum_i s(h_i)V_i$ and $\partial S/\partial h_i=
1/T_i=\mu_i$, the dissipative entropy production is a quadratic form in the SPSD
Laplacian:

$$
\Delta S^{\text{diss}}\;\propto\;\boldsymbol\mu^\top L_w\boldsymbol\mu
   =\tfrac12\sum_{i,j} w_{ij}\,(\mu_i-\mu_j)^2\;\ge\;0 ,
\tag{12}
$$

since $w_{ij}\ge0$. **Hot $\to$ cold:** for a hot node $i$ ($\mu_i$ small) beside
colder neighbours ($\mu_j$ larger), $\Delta h_i^{\text{diss}}=\sum_j w_{ij}
(\mu_i-\mu_j)<0$ — it loses energy. ∎

(These are checked numerically in
`tests/test_meshgraphnet.py::test_enthalpy_generic_structure_and_latent_heat` and
`::test_enthalpy_latent_heat_damps_temperature_rise`.)

---

## 6. Gain initialisation (numbers that matter)

The gains carry the $dt/(\rho c_p)$-type scale. For the source, a per-step
deposit $\Delta h^{\text{ext}}\!\sim\!q\,\Delta t$ with $q\!\sim\!10^{10}\,
\mathrm{W\,m^{-3}}$ and $\Delta t\!\approx\!0.047\,\mathrm s$ gives
$\Delta h\!\sim\!5\times10^{8}\,\mathrm{J\,m^{-3}}$, i.e.
$\Delta T\!\sim\!\Delta h/(\rho c_p)\!\sim\!10^2\,\mathrm K$ — the right per-step
heating. We therefore initialise

$$
\mathrm{softplus}(g_q)\approx 0.025\;(\sim\!\Delta t),
\qquad
\mathrm{softplus}(g_c)\approx 10^{-3},
$$

and let training tune them (and $\alpha$). The conductance scale $e^{\alpha}$ is
inherited from the "full" head (init $\sim\!500$); both are validated by a smoke
test on real graphs (finite, physical $\Delta T$, $\sum\Delta h^{\text{diss}}\!=\!0$).

---

## 7. What changed vs the temperature ("full") head — one line

Only the **conserved variable**: the SPSD Laplacian now acts on, and conserves,
**energy** $h$ (with $T=\mathcal H^{-1}(h)$), instead of acting on and conserving
**temperature**. Conductances, the entropy gradient $\mu=1/T$, the degeneracy,
the second law and the source/cooling structure are all identical. The single
change repairs the thermodynamic inconsistency under phase change.
