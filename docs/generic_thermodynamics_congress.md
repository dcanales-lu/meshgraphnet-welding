# Thermodynamic consistency of the GENERIC welding surrogate

*A step-by-step physical and mathematical note for the congress talk. Two
formulations are compared: (i) the GENERIC head we implemented first, which is
**thermodynamically inconsistent** for welding because it ignores latent heat,
and (ii) the **energy/enthalpy-state** GENERIC we will build next, which is
consistent. Written at an introductory (first-year) level; equations are in
LaTeX so this `.md` can be compiled (e.g. `pandoc`/KaTeX).*

---

## 0. The physical problem

We model transient heat conduction in a thin metal plate with a moving welding
arc (Goldak source) and convective/radiative boundaries. The temperature field
$T(\mathbf{x},t)$ obeys the nonlinear heat equation

$$
\rho\,c_p(T)\,\frac{\partial T}{\partial t}
   \;=\; \nabla\!\cdot\!\big(k(T)\,\nabla T\big) \;+\; Q(\mathbf{x},t),
\tag{0.1}
$$

where $\rho$ is the density, $c_p(T)$ the specific heat, $k(T)$ the thermal
conductivity and $Q$ the volumetric heat source (the Goldak field). The boundary
loses heat by convection/radiation (Robin) or is held fixed (Dirichlet).

**Key physical fact for welding:** the metal *melts* (and locally *boils*). A
phase change absorbs a large amount of energy — the **latent heat** — *at almost
constant temperature*. This makes $c_p$ strongly temperature-dependent. As we
will see, this single fact is what breaks our first GENERIC formulation.

---

## 1. Energy, enthalpy and latent heat (the physics that matters)

Define the **volumetric enthalpy** (energy stored per unit volume, measured from
a reference temperature $T_{\text{ref}}$):

$$
h(T) \;=\; \rho \int_{T_{\text{ref}}}^{T} c_p(\tau)\,d\tau .
\tag{1.1}
$$

For a material **without** phase change, $c_p$ is roughly constant and (1.1) is a
straight line, $h \approx \rho\,c_p\,(T-T_{\text{ref}})$, i.e. **energy is
proportional to temperature**.

For a **welding** material, melting (and boiling) inject extra stored energy over
a narrow temperature band. We model this with the **apparent heat capacity**

$$
\boxed{\;c_p^{\text{app}}(T) \;=\; \underbrace{c_p^{\text{sens}}}_{\text{sensible}}
   \;+\; L_f\,\frac{d f_\ell}{dT}\;+\;L_v\,\frac{d f_v}{dT}\;}
\tag{1.2}
$$

where $L_f, L_v$ are the latent heats of fusion and vaporization, and
$f_\ell(T), f_v(T)$ are the liquid and vapor fractions that rise smoothly from
$0$ to $1$ across the melting / boiling ranges. The derivatives
$df_\ell/dT,\,df_v/dT$ are sharp bumps centred on the phase-change temperatures
(in the code these are normalized Gaussians, so $\int c_p^{\text{app}}\,dT$
recovers the latent heats). **This is exactly the `cp_apparent` function in
`thermal_solver.py`.**

The consequence is the **enthalpy–temperature curve** $h(T)$:

```
 h(T)                                   .-- boiling plateau (L_v absorbed)
   |                                ___/
   |                          _____/
   |  melting plateau  ___---/      (L_f absorbed at ~T_melt)
   |             ___---
   |        ___-/
   |   ___-/   (nearly linear, slope = rho*c_p, away from phase changes)
   +------------------------------------------------ T
```

**Read this curve carefully — it is the whole point:** near $T_{\text{melt}}$ a
*large* change in enthalpy $h$ (energy) produces an *almost zero* change in
temperature $T$, because the energy goes into the phase change, not into heating.
Equivalently, the slope $dh/dT=\rho\,c_p^{\text{app}}$ becomes enormous there.

---

## 2. GENERIC in a nutshell (dissipative thermal case)

GENERIC ("General Equation for the NonEquilibrium Reversible–Irreversible
Coupling") writes the evolution of a state $\mathbf{z}$ as

$$
\frac{d\mathbf{z}}{dt}
  \;=\; \underbrace{L(\mathbf{z})\,\frac{\partial E}{\partial \mathbf{z}}}_{\text{reversible}}
   \;+\; \underbrace{M(\mathbf{z})\,\frac{\partial S}{\partial \mathbf{z}}}_{\text{dissipative}},
\tag{2.1}
$$

with two potentials — total energy $E(\mathbf{z})$ and entropy $S(\mathbf{z})$ —
and two operators, $L$ (antisymmetric, reversible) and $M$ (symmetric positive
semi-definite, dissipative). The framework requires the **degeneracy
conditions**

$$
L\,\frac{\partial S}{\partial \mathbf{z}} = 0,
\qquad
M\,\frac{\partial E}{\partial \mathbf{z}} = 0 .
\tag{2.2}
$$

From these one proves the two laws of thermodynamics automatically:

$$
\frac{dE}{dt}=0 \ \text{(1st law, isolated system)},
\qquad
\frac{dS}{dt}=\Big(\tfrac{\partial S}{\partial \mathbf{z}}\Big)^{\!\top}
M\,\Big(\tfrac{\partial S}{\partial \mathbf{z}}\Big)\ge 0\ \text{(2nd law).}
\tag{2.3}
$$

**Pure heat conduction is purely dissipative** (there is no reversible flow of
heat), so $L=0$ and

$$
\frac{d\mathbf{z}}{dt} \;=\; M(\mathbf{z})\,\frac{\partial S}{\partial \mathbf{z}}.
\tag{2.4}
$$

The two questions that decide whether the model is *physically consistent* are:

1. **What is the state $\mathbf{z}$?** (temperature, or energy?)
2. **Is $M\,\partial E/\partial\mathbf{z}=0$ the *correct* energy conservation?**

Our first implementation gets both subtly wrong.

---

## 3. The GENERIC we implemented first — and why it is inconsistent

### 3.1 Choices we made

We used **temperature as the state**, $\mathbf{z}=\mathbf{T}=(T_1,\dots,T_N)$ on
the mesh nodes (tributary volumes $V_i$, assumed equal), and the **constant-$c_p$
ideal-solid potentials**

$$
E(\mathbf{T}) = \sum_i \rho\,c_p\,T_i\,V_i,
\qquad
S(\mathbf{T}) = \sum_i \rho\,c_p\,\ln(T_i)\,V_i .
\tag{3.1}
$$

Their gradients are

$$
\frac{\partial E}{\partial T_i} = \rho\,c_p\,V_i \;=\;\text{const},
\qquad
\mu_i \;\equiv\; \frac{\partial S}{\partial e_i}
   \;=\; \frac{\partial S/\partial T_i}{\partial E/\partial T_i}
   \;=\; \frac{\rho\,c_p\,V_i/T_i}{\rho\,c_p\,V_i}
   \;=\; \frac{1}{T_i}.
\tag{3.2}
$$

The dissipative operator is a **graph Laplacian of learned conductances**
$M = L_w = D - W$, with $w_{ij}=w_{ji}\ge 0$ produced by the network. Because a
graph Laplacian has zero row/column sums, $L_w\mathbf{1}=0$, the degeneracy
$M\,\partial E/\partial\mathbf{z}=0$ holds (since $\partial E/\partial\mathbf{z}$
is the constant vector). We then wrote the **temperature increment** directly as

$$
\boxed{\;\Delta T_i \;=\; (L_w\,\boldsymbol{\mu})_i
   \;=\; \sum_j w_{ij}\,(\mu_i-\mu_j)\;}
\qquad (\text{plus the external source/cooling}).
\tag{3.3}
$$

Per graph this gives $\sum_i \Delta T_i = \mathbf{1}^\top L_w\boldsymbol{\mu}=0$,
which we *called* "energy conservation", and $\sum_i \mu_i\,\Delta T_i=
\boldsymbol{\mu}^\top L_w\boldsymbol{\mu}\ge 0$, "entropy production". On paper,
both laws hold. **So what is wrong?**

### 3.2 The flaw: it conserves *temperature*, not *energy*

The thermodynamically correct dissipative law (2.4) evolves the **energy**:

$$
\frac{de_i}{dt} \;=\; (L_w\,\boldsymbol{\mu})_i ,
\qquad
\sum_i \frac{de_i}{dt}=0 \;\Rightarrow\; \text{energy is conserved.}
\tag{3.4}
$$

Temperature is then recovered from energy through the enthalpy curve (1.1):

$$
\frac{dT_i}{dt}
  \;=\; \frac{1}{\rho\,c_p^{\text{app}}(T_i)}\,\frac{de_i}{dt}
  \;=\; \frac{(L_w\boldsymbol{\mu})_i}{\rho\,c_p^{\text{app}}(T_i)} .
\tag{3.5}
$$

Compare (3.5) with what we actually did, (3.3): **we dropped the division by the
apparent heat capacity $\rho\,c_p^{\text{app}}(T_i)$.** Two cases:

- **Constant $c_p$** (no phase change): $c_p^{\text{app}}$ is a constant, the
  division is a global scale, and it can be absorbed into the learned $w_{ij}$.
  Then (3.3) and (3.5) agree — our formulation is fine.

- **Variable $c_p$** (welding!): $c_p^{\text{app}}(T_i)$ varies by **orders of
  magnitude** between cold metal and the melt pool. Dropping it means

$$
\sum_i \Delta T_i = 0
\quad\text{(temperature-sum conserved)}
\;\neq\;
\sum_i \rho\,c_p^{\text{app}}(T_i)\,\Delta T_i = \sum_i \Delta e_i = 0
\quad\text{(energy conserved).}
\tag{3.6}
$$

**So "$\sum_i\Delta T_i=0$" is the wrong conservation law.** Our head conserves
the *sum of temperatures*, but physics conserves the *sum of energies*, and the
two differ exactly by the factor $c_p^{\text{app}}(T)$ that blows up at the melt
pool.

### 3.3 Why this hurts precisely at the weld peak

At the melt pool $c_p^{\text{app}}$ is huge (latent heat). Physically (3.5) then
says $dT/dt$ should be **small** there — incoming energy is swallowed by the
phase change, not turned into temperature. Our scheme (3.3) has **no such
damping**: it pushes the same "$L_w\mu$" straight into $\Delta T$, so it
over-moves temperature at the hottest nodes and mis-attributes energy across the
mesh. This is consistent with what we observe empirically: the GENERIC head
**distorts / caps the peak**, while the unconstrained network fits it better.

> **One-line summary of the flaw:** we built GENERIC on $E\propto T$ and evolved
> $T$ directly, but welding has $E = h(T)$ with a *strongly nonlinear* $h$ (latent
> heat), so our energy bookkeeping is wrong exactly where the physics is richest.

---

## 4. The physically consistent GENERIC (energy / enthalpy state)

The fix is conceptually simple: **use energy (enthalpy) as the state, not
temperature.** Temperature becomes a *derived* quantity.

### 4.1 State, potentials, gradients

Let the state be the nodal **enthalpy** $\mathbf{z}=\mathbf{h}=(h_1,\dots,h_N)$,
with the physical map $h_i = h(T_i)$ from (1.1) (monotonically increasing, hence
invertible: $T_i = h^{-1}(h_i)$, the **enthalpy method**). Then

$$
E(\mathbf{h}) = \sum_i h_i\,V_i ,
\qquad
\frac{\partial E}{\partial h_i} = V_i = \text{const},
\tag{4.1}
$$

$$
S(\mathbf{h}) = \sum_i s\big(h_i\big)\,V_i,
\qquad
\mu_i \equiv \frac{\partial S}{\partial h_i}
   = \frac{ds}{dh}\bigg|_{h_i}
   = \frac{1}{T(h_i)} .
\tag{4.2}
$$

The last equality is the **definition of temperature** in thermodynamics,
$dS = dE/T$ at constant volume, so $\partial S/\partial E = 1/T$ **holds for any
$c_p(T)$, including latent heat.** (This is why $\mu=1/T$ was *not* the error in
§3 — the error was evolving $T$ instead of $E$.)

### 4.2 Evolution and the two laws

With $M=L_w$ the same SPSD graph Laplacian of learned conductances,

$$
\boxed{\;
\frac{dh_i}{dt} \;=\; (L_w\,\boldsymbol{\mu})_i
   \;=\; \sum_j w_{ij}\Big(\tfrac{1}{T_i}-\tfrac{1}{T_j}\Big)
\;}
\tag{4.3}
$$

and the temperature is recovered **after** the energy update by inverting the
enthalpy curve:

$$
T_i \;=\; h^{-1}(h_i).
\tag{4.4}
$$

Now the guarantees are physically meaningful:

- **1st law (energy):** $\displaystyle \sum_i \frac{dh_i}{dt}
  = \mathbf{1}^\top L_w\boldsymbol{\mu}=0$ — the **total enthalpy (energy) is
  conserved**, because the Laplacian conserves the quantity it acts on, and that
  quantity is now *energy*.
- **2nd law (entropy):** $\displaystyle \frac{dS}{dt}
  = \boldsymbol{\mu}^\top L_w \boldsymbol{\mu}
  = \tfrac12\sum_{ij} w_{ij}\,(\mu_i-\mu_j)^2 \ge 0$ — entropy never decreases.
- **Heat flows hot $\to$ cold:** a hot node ($\mu_i=1/T_i$ small) beside cold
  neighbours ($\mu_j$ large) has $dh_i/dt<0$: it **loses energy**. ✓
- **Latent heat handled correctly:** at the melt pool, a large $dh_i/dt$ produces
  almost no $dT_i$ because $T=h^{-1}(h)$ is nearly flat there (the plateau in the
  $h(T)$ curve). The phase change absorbs the energy, exactly as in the FEM. ✓

### 4.3 Where conductances come from (closing the loop with the heat equation)

To check (4.3) reduces to Fourier conduction, note
$\nabla T = -T^2\,\nabla(1/T) = -T^2\,\nabla\mu$, so the conductive flux is

$$
\mathbf{q} = -k\,\nabla T = k\,T^2\,\nabla\mu .
\tag{4.5}
$$

Discretising the divergence of $\mathbf q$ on the graph gives exactly the
Laplacian form $L_w\boldsymbol\mu$ with edge conductances

$$
w_{ij} \;\sim\; k\,T_i\,T_j \,/\, \ell_{ij}^2 ,
\tag{4.6}
$$

i.e. the network's job is to learn these (temperature- and geometry-dependent)
conductances. The $T^2$-type factor in (4.6) is also *why working in $\mu=1/T$
needs large conductances* — a purely numerical remark, but reassuring that the
form is the right one.

---

## 5. Side-by-side summary (the slide)

| | **Flawed GENERIC (v1)** | **Consistent GENERIC (v2)** |
|---|---|---|
| State $\mathbf z$ | temperature $T$ | enthalpy / energy $h$ |
| Energy | $E=\sum \rho c_p T_i V_i$ (assumes $E\propto T$) | $E=\sum h_i V_i$ (exact) |
| Update | $\Delta T_i=(L_w\mu)_i$ | $\Delta h_i=(L_w\mu)_i,\ T_i=h^{-1}(h_i)$ |
| "Conservation" | $\sum_i\Delta T_i=0$ (temperature sum — **wrong**) | $\sum_i\Delta h_i=0$ (**energy** — correct) |
| Latent heat | ignored ($c_p$ const) → melt pool mishandled | built into $h(T)$ → melt pool correct |
| $\mu=1/T$ | correct | correct |
| 2nd law $dS/dt\ge0$ | holds | holds |
| Valid when | no phase change | always (welding included) |

**Take-home for the congress:** GENERIC *guarantees* the laws of thermodynamics
*only for the energy you give it*. If you feed it temperature while the material
has latent heat, you conserve the wrong thing — the structure is exact but the
**physics is mis-specified**. Moving the state from $T$ to the enthalpy $h$ (the
classic *enthalpy method* of phase-change heat transfer) restores consistency:
energy is truly conserved, entropy truly increases, and the melt-pool plateau
emerges for free.

---

## 6. How this maps onto the GNN surrogate

In the network (`models/meshgraphnet.py`, `FullGenericThermalHead`):

- **v1 (current):** the message-passing processor produces edge conductances
  $w_{ij}=\mathrm{softplus}(\text{MLP})$; the head outputs
  $\Delta T = L_w\mu + \text{(source+cooling)}$ directly, with $\mu=1/T$ read from
  the (de-normalized) temperature feature.
- **v2 (to build):** keep the conductance head, but (a) carry the **enthalpy**
  $h$ as the rolled state, (b) update $\Delta h = L_w\mu + \text{external}$, and
  (c) map back $T=h^{-1}(h)$ with the *same* fixed, latent-heat-aware curve
  $h(T)$ used by the FEM solver (a cheap 1-D monotone interpolation, no extra
  learning). The external source then injects **energy** (W·dt), not a temperature
  increment, which is also more physical.

Everything else (SPSD Laplacian, degeneracy, entropy production) is inherited
unchanged — only the *variable we conserve* is corrected.

---

## 7. One-paragraph script for the talk

> "We first imposed GENERIC structure directly on temperature: energy
> $\propto T$, entropy $\propto\ln T$, and a learned positive-semidefinite graph
> Laplacian for diffusion. That guarantees energy conservation and the second law
> — but only for a material whose energy is proportional to temperature. Welding
> is not such a material: melting and vaporization store large latent heat, so
> energy is the *enthalpy* $h(T)$, a strongly nonlinear function. Conserving
> $\sum_i T_i$ is then not conserving energy, and the mismatch is largest exactly
> at the melt pool — which is why the structure-preserving head distorts the peak.
> The fix is the classical enthalpy method: take energy (enthalpy) as the state,
> evolve it with the same SPSD operator, and recover temperature by inverting the
> enthalpy curve. Energy is then genuinely conserved, entropy genuinely increases,
> and the phase-change plateau appears naturally."
