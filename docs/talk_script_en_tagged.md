# Talk script (tagged for AI audio) — CMN 2026

Tagged version of `talk_script_en.md` for text-to-speech "podcast" generation.
Each sentence starts with an intent tag that guides the delivery/tone.

**Tags used**
- From your example: `[informative]`, `[explanation]`, `[reminder]`, `[instruction]`, `[approval]`
- Natural additions: `[emphasis]` (key points), `[transition]` (signposts).
  If your tool rejects them, find-and-replace `[emphasis]` → `[informative]` and
  `[transition]` → `[instruction]`, or just delete them.

**How to use:** the spoken content is only the tagged sentences. The `<!-- Slide N -->`
markers are just for your reference (HTML comments are not read aloud); delete them
if you paste plain text. You can also generate the audio slide by slide.

---

<!-- Slide 1 — Title -->
[informative] Good morning, everyone, and thank you for being here. [informative] My name is D. Canales, and I will present our work on structure-preserving deep learning for welding simulation: a graph-network surrogate based on the GENERIC framework. [informative] This is joint work between Universidad Loyola and the University of Zaragoza.

<!-- Slide 2 — Computational welding -->
[informative] Welding is simulated with the finite element method, and it is a multiphysics problem. [explanation] The thermal field drives the mechanics, which gives residual stresses and distortion, and it also drives the metallurgy, the final microstructure. [informative] These simulations are very accurate, but a single nonlinear transient run can take minutes to hours. [explanation] So if we want to explore many designs, optimize, or quantify uncertainty, the cost quickly becomes prohibitive. [emphasis] This is the motivation for a fast surrogate.

<!-- Slide 3 — Learned surrogates -->
[transition] A natural idea is a learned surrogate. [informative] Mesh-based models such as MeshGraphNet are attractive: they work directly on the simulation mesh and roll the solution forward in time. [explanation] But a purely data-driven model has two limitations. [explanation] First, without physics, the error accumulates over long rollouts, so the model becomes unstable and generalizes poorly. [explanation] Second, it needs a lot of training data, which is expensive to produce. [emphasis] Our idea is to put physics into the model, so it is stable and needs less data.

<!-- Slide 4 — Pipeline -->
[transition] This is the overall pipeline, and it is also the structure of the talk. [explanation] We start from FEM data, turn each simulation into graphs, process them with a MeshGraphNet, and then, instead of a standard decoder, we use a thermodynamically consistent GENERIC head. [explanation] We roll the prediction forward in time, and we train with a push-forward loss. [transition] Let me go through each block.

<!-- Slide 5 — Physics + FEM -->
[informative] The physics is transient heat conduction with a moving arc source. [explanation] The key feature of welding is the latent heat of melting and boiling, which we fold into an apparent heat capacity. [reminder] Integrating it gives the enthalpy, and please remember this, because it will be important later. [informative] We solve this with standard finite elements and backward Euler in time, which gives a semidiscrete system that we advance step by step. [informative] Each FEM run is our ground truth.

<!-- Slide 6 — Dataset -->
[informative] For the data, we generated 360 simulations covering a wide range: different plate geometries, sizes, powers, speeds, and boundary conditions. [explanation] We deliberately hold out only one thing, the weld trajectory, so that any generalization can be attributed to the path, and not to the other parameters. [informative] We split by simulation, so no snapshot leaks between training and validation.

<!-- Slide 7 — Graphs -->
[explanation] Each pair of consecutive snapshots becomes one graph. [informative] Mesh nodes are graph nodes, and mesh edges are graph edges. [explanation] We use only relative features: each node carries its temperature, the local heat source, and a few process and boundary values; each edge carries only the relative displacement between its nodes. [emphasis] There are no absolute coordinates, so the model cannot memorize positions; it must learn local physics. [informative] The target is the temperature increment, delta T.

<!-- Slide 8 — Encoder-Processor -->
[informative] The network is a MeshGraphNet. [explanation] An encoder lifts the node and edge features into a latent space. [explanation] Then a processor applies eight message-passing steps: each step updates the edges, sums the messages at each node, and updates the nodes, with residual connections. [explanation] After eight steps, every node has seen its neighborhood up to eight edges away. [transition] Normally a decoder would map this to delta T, but here we replace it with the GENERIC head.

<!-- Slide 9 — GENERIC general -->
[transition] So what is GENERIC? [explanation] It is a framework that splits any dynamics into a reversible part and a dissipative part, driven by two potentials, the energy and the entropy. [explanation] Heat conduction has no reversible part; it is purely dissipative, so only the second term survives. [explanation] This leaves two choices: what is the state, and what is the operator M. [emphasis] Since the conserved quantity is energy, we take the state to be the nodal energy, the enthalpy times the volume, not the temperature.

<!-- Slide 10 — Operator M / chain rule -->
[transition] Now, why this works. [emphasis] One important point: the state is a vector, one energy per node, while the energy E is a scalar, the sum of them. [explanation] If we differentiate the energy and the entropy along the dynamics, the chain rule tells us we need two gradients. [explanation] Because the state is the energy, the energy gradient is simply one, and the entropy gradient is one over temperature. [explanation] Then we choose the operator M to be a graph Laplacian built from learned, non-negative conductances. [emphasis] This single choice gives both laws for free: the Laplacian has zero row sums, so energy is conserved, the first law; and it is positive semi-definite, so entropy never decreases, the second law. [explanation] And heat flows from hot to cold by construction.

<!-- Slide 11 — What the head computes -->
[transition] So what does the head actually compute each step? [explanation] It works through energy, in three steps. [explanation] First, neighbors exchange energy through the Laplacian, plus the external source from the arc. [explanation] Second, we update the stored energy. [explanation] Third, we recover the temperature by inverting the enthalpy curve. [emphasis] This last step is where the latent heat lives: near the melt pool the curve is almost flat, so a large change in energy gives almost no change in temperature, exactly as in the real physics. [informative] The output is the temperature increment.

<!-- Slide 12 — Rollout -->
[explanation] At test time, we give the model only the initial temperature and the known torch path, and it predicts the whole history step by step, feeding its own prediction back. [emphasis] Only the temperature is fed back; everything else is known in advance. [explanation] Over long horizons, errors can accumulate, and that is exactly what training must control.

<!-- Slide 13 — Push-forward -->
[explanation] And that is the push-forward loss. [explanation] Instead of training on a single step, we unroll the model a few steps, feed each prediction back, and compare to the truth over the whole window, with the gradient flowing through all the steps. [explanation] This aligns training with the long rollout. [emphasis] We use two steps, K equals two, because it clearly outperformed a single step.

<!-- Slide 14 — Ablation table -->
[transition] Now, the results. [explanation] We compare two models that are identical except for the head: our GENERIC model, "ON", against a plain MeshGraphNet, "OFF". [informative] On held-out validation, unseen welds similar to training, the GENERIC model is better. [informative] And on the spiral, a much harder out-of-distribution case, it is also better, with about two times lower error. [emphasis] Importantly, the plain model violates the maximum principle: it predicts temperatures well below ambient, which is unphysical, while ours stays at the physical floor.

<!-- Slide 15 — Val case 1 -->
[explanation] Here are the temperature histories at a few probe points for a held-out weld. [informative] Black is the FEM ground truth, green is our model, red dashed is the plain one. [emphasis] Our model tracks every probe almost exactly.

<!-- Slide 16 — Val case 2 -->
[informative] A second case shows the same picture: the GENERIC model stays close to the truth, while the unconstrained one drifts, especially in the far field.

<!-- Slide 17 — Spiral video -->
[transition] Finally, the hardest test: a spiral weld. [emphasis] This trajectory is never seen in training, and we roll out more than fifteen hundred steps. [instruction] Let me play the animation. [informative] On the left is the FEM ground truth, in the middle our surrogate, and on the right the error. [explanation] The surrogate follows the moving melt pool and the cooling arms very closely, and the error stays small.

<!-- Slide 18 — Spiral histories + stability -->
[explanation] The probe histories confirm it: our model, in green, stays close to the FEM, in black, across the whole spiral, while the plain model, in red, drifts. [explanation] On the right we show stability across training: the plain model swings a lot and cools below ambient at every checkpoint, the unphysical region; our model stays in a tight band and respects the physical floor. [emphasis] So the structure does not only improve accuracy; it makes the model stable and reliable.

<!-- Slide 19 — Conclusions -->
[transition] To conclude. [emphasis] The enthalpy-state GENERIC formulation is a sound and thermodynamically consistent way to build a welding surrogate. [explanation] Energy is the natural variable, so phase change is handled exactly. [emphasis] The structure guarantees the first and second laws and the maximum principle by construction, at every step, and it also regularizes training, giving stable, selectable models, where the plain network is accurate but fragile. [informative] This is ongoing work: we are looking at the training set, at the number of push-forward steps, at generalization beyond the trajectory, and at the extension to 3D. [approval] Thank you very much. I am happy to take any questions.
