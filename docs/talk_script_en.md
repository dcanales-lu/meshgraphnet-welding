# Talk script — CMN 2026 (Gijón)
**"Towards Structure-Preserving Deep Learning for Welding: A GENERIC-based Graph Network Approach"**

**Target: ~12 minutes.** Estimated total below ≈ **11:40**, leaving a small buffer.
Pace assumed ≈ **120–125 words/min** (comfortable for a mostly non-native audience).
Style: simple scientific English, short sentences, first-person plural ("we").

**Delivery tips**
- Don't rush. The buffer is there so you can breathe and point at the figures.
- On the **video slide (17)**: start the animation, then stop talking for ~10 s and let people watch.
- The estimates assume you read at a steady pace; if you are ahead, slow down on slides 10, 14, 18 (the key messages).
- Numbers in **[brackets]** after each slide are the time budget for that slide; the running total is on the right.

---

## Slide 1 — Title  · [0:25] · *(00:25)*
> Good morning, everyone, and thank you for being here. My name is **[D. Canales]**, and I will present our work on **structure-preserving deep learning for welding simulation**: a graph-network surrogate based on the GENERIC framework. This is joint work between Universidad Loyola and the University of Zaragoza.

*(Transition: "Let me start with the motivation.")*

---

## Slide 2 — Computational welding simulation  · [0:42] · *(01:07)*
> Welding is simulated with the finite element method, and it is a **multiphysics** problem. The thermal field drives the mechanics, which gives residual stresses and distortion, and it also drives the metallurgy, the final microstructure. These simulations are very accurate, but a single nonlinear transient run can take **minutes to hours**. So if we want to explore many designs, optimize, or quantify uncertainty, the cost quickly becomes prohibitive. This is the motivation for a **fast surrogate**.

---

## Slide 3 — Learned surrogates and their challenges  · [0:42] · *(01:49)*
> A natural idea is a **learned surrogate**. Mesh-based models such as **MeshGraphNet** are attractive: they work directly on the simulation mesh and roll the solution forward in time. But a purely data-driven model has two limitations. **First**, without physics, the error accumulates over long rollouts, so the model becomes unstable and generalizes poorly. **Second**, it needs a lot of training data, which is expensive to produce. Our idea is to **put physics into the model**, so it is stable and needs less data.

---

## Slide 4 — The pipeline  · [0:34] · *(02:23)*
> This is the overall pipeline, and it is also the structure of the talk. We start from **FEM data**, turn each simulation into **graphs**, process them with a **MeshGraphNet**, and then, instead of a standard decoder, we use a thermodynamically consistent **GENERIC head**. We **roll** the prediction forward in time, and we train with a **push-forward loss**. Let me go through each block.

---

## Slide 5 — Governing physics and the FEM ground truth  · [0:38] · *(03:01)*
> The physics is **transient heat conduction** with a moving arc source. The key feature of welding is the **latent heat** of melting and boiling, which we fold into an **apparent heat capacity**. Integrating it gives the **enthalpy**, and this will be important later. We solve this with standard finite elements and backward Euler in time, which gives a semidiscrete system that we advance step by step. Each FEM run is our **ground truth**.

---

## Slide 6 — The training dataset  · [0:30] · *(03:31)*
> For the data, we generated **360 simulations** covering a wide range: different plate geometries, sizes, powers, speeds, and boundary conditions. We deliberately hold out **only one thing, the weld trajectory**, so that any generalization can be attributed to the path, and not to the other parameters. We split **by simulation**, so no snapshot leaks between training and validation.

---

## Slide 7 — From FEM snapshots to graphs  · [0:38] · *(04:09)*
> Each pair of consecutive snapshots becomes **one graph**. Mesh nodes are graph nodes, mesh edges are graph edges. We use **only relative features**: each node carries its temperature, the local heat source, and a few process and boundary values; each edge carries only the **relative displacement** between its nodes. There are no absolute coordinates, so the model cannot memorize positions, it must learn local physics. The target is the temperature increment, **delta T**.

---

## Slide 8 — The MeshGraphNet: Encoder–Processor  · [0:37] · *(04:46)*
> The network is a MeshGraphNet. An **encoder** lifts the node and edge features into a latent space. Then a **processor** applies eight message-passing steps: each step updates the edges, sums the messages at each node, and updates the nodes, with residual connections. After eight steps, every node has seen its neighborhood up to eight edges away. Normally a decoder would map this to delta T, but here we **replace it with the GENERIC head**.

---

## Slide 9 — GENERIC: from the general form to pure dissipation  · [0:42] · *(05:28)*
> So what is GENERIC? It is a framework that splits any dynamics into a **reversible** part and a **dissipative** part, driven by two potentials, the **energy** and the **entropy**. Heat conduction has no reversible part, it is purely dissipative, so only the second term survives. This leaves two choices: **what is the state**, and **what is the operator M**. Since the conserved quantity is energy, we take the state to be the **nodal energy**, the enthalpy times the volume, not the temperature.

---

## Slide 10 — The operator M: chain rule and the two laws  · [1:00] · *(06:28)*
> Now, why this works. One important point: the **state is a vector**, one energy per node, while the **energy E is a scalar**, the sum of them. If we differentiate the energy and the entropy along the dynamics, the **chain rule** tells us we need two gradients. Because the state *is* the energy, the energy gradient is simply **one**, and the entropy gradient is **one over temperature**. Then we choose the operator M to be a **graph Laplacian** built from **learned, non-negative conductances**. This single choice gives both laws for free: the Laplacian has zero row sums, so **energy is conserved**, the first law; and it is positive semi-definite, so **entropy never decreases**, the second law. And heat flows from hot to cold by construction.

*(This is the key slide. If you are behind schedule, this is the one to deliver clearly and not skip.)*

---

## Slide 11 — What the head computes each step: ΔT  · [0:45] · *(07:13)*
> So what does the head actually compute each step? It works **through energy**, in three steps. **First**, neighbors exchange energy through the Laplacian, plus the external source from the arc. **Second**, we update the stored energy. **Third**, we recover the temperature by inverting the enthalpy curve. This last step is where the **latent heat** lives: near the melt pool the curve is almost flat, so a large change in energy gives almost no change in temperature, exactly as in the real physics. The output is the temperature increment.

---

## Slide 12 — Inference: autoregressive rollout  · [0:28] · *(07:41)*
> At test time, we give the model only the **initial temperature** and the known torch path, and it predicts the whole history step by step, feeding its own prediction back. **Only the temperature is fed back**; everything else is known in advance. Over long horizons, errors can accumulate, and that is exactly what training must control.

---

## Slide 13 — Push-forward training  · [0:31] · *(08:12)*
> And that is the **push-forward** loss. Instead of training on a single step, we **unroll** the model a few steps, feed each prediction back, and compare to the truth over the whole window, with the gradient flowing through all the steps. This aligns training with the long rollout. We use **two steps**, K equals two, because it clearly outperformed a single step.

*(Transition: "Now, the results.")*

---

## Slide 14 — Ablation: structure ON vs OFF  · [0:40] · *(08:52)*
> We compare two models that are **identical except for the head**: our GENERIC model, "ON", against a plain MeshGraphNet, "OFF". On **held-out validation**, unseen welds similar to training, the GENERIC model is better. And on the **spiral**, a much harder out-of-distribution case, it is also better, with about **two times lower error**. Importantly, the plain model **violates the maximum principle**, it predicts temperatures well below ambient, which is unphysical, while ours stays at the physical floor.

---

## Slide 15 — Held-out validation (case 1)  · [0:20] · *(09:12)*
> Here are the temperature histories at a few probe points for a held-out weld. **Black** is the FEM ground truth, **green** is our model, **red dashed** is the plain one. Our model tracks every probe almost exactly.

---

## Slide 16 — Held-out validation (case 2)  · [0:14] · *(09:26)*
> A second case shows the same picture: the GENERIC model stays close to the truth, while the unconstrained one drifts, especially in the far field.

---

## Slide 17 — The Archimedean spiral (video)  · [0:45] · *(10:11)*
> Finally, the hardest test: a **spiral weld**. This trajectory is **never seen in training**, and we roll out more than fifteen hundred steps.
>
> *(Start the animation. Pause ~10 seconds and let it play.)*
>
> On the left is the FEM ground truth, in the middle our surrogate, and on the right the error. The surrogate follows the moving melt pool and the cooling arms very closely, and the error stays small.

---

## Slide 18 — Spiral results: histories and stability  · [0:40] · *(10:51)*
> The probe histories confirm it: our model, in green, stays close to the FEM, in black, across the whole spiral, while the plain model, in red, drifts. On the right we show **stability across training**: the plain model swings a lot and cools below ambient at every checkpoint, the unphysical region; our model stays in a tight band and respects the physical floor. So the structure does not only improve accuracy, it makes the model **stable and reliable**.

---

## Slide 19 — Conclusions  · [0:50] · *(11:41)*
> To conclude. The **enthalpy-state GENERIC** formulation is a sound and thermodynamically consistent way to build a welding surrogate. Energy is the natural variable, so **phase change is handled exactly**. The structure **guarantees the first and second laws and the maximum principle by construction**, at every step, and it also **regularizes training**, giving stable, selectable models, where the plain network is accurate but fragile. This is ongoing work: we are looking at the training set, at the number of push-forward steps, at generalization beyond the trajectory, and at the extension to 3D. **Thank you very much. I am happy to take any questions.**

---

### Timing summary
| Block | Slides | Budget |
|---|---|---|
| Intro / motivation | 1–4 | ~2:23 |
| Method (physics → data → graphs → MPNN) | 5–8 | ~2:23 |
| GENERIC head | 9–11 | ~2:27 |
| Rollout + training | 12–13 | ~0:59 |
| Results | 14–18 | ~2:39 |
| Conclusions | 19 | ~0:50 |
| **Total** | **19** | **≈ 11:41** |

**Word count ≈ 1,420** → ≈ 11.3 min at 125 wpm, ≈ 11.8 min at 120 wpm. Comfortably under 12.

> If you need to **save ~1 minute** (running over on the day): shorten slide 16 to one sentence, drop the second example wording on slide 11, and trim the "ongoing work" list on slide 19 to two items.
