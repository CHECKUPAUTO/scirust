# SciRust — "Research → Functions" Roadmap

> **Status: 80/80 ✅** — all candidate papers (#1–#80) are delivered (tested,
> honest oracle/gradient check, 8 green gates), from certifiable to N-D LLM, from
> optimizers to quantization, from sequence models to full
> verification and verifiable inference.
>
> Real research papers selected for their **fit with scirust's fundamentals**
> (bit-exact determinism, certifiability, testable pure Rust) and
> translated into **concrete functions**. Each entry: reference →
> target function/module → status → effort.
>
> Statuses: ✅ delivered (tested, 8 green gates) · 🔨 in progress · 📋 planned.
> Effort: S (hours) · M (days) · L (week) · XL (months).
>
> Golden rule (cf. [GROWTH_PLAN](GROWTH_PLAN.md)): **no** entry moves
> to ✅ without a test (gradient check for an op, oracle/soundness for a guarantee)
> and without the 8 green gates. No stub, no half-implementation.

## Tier 1 — Certifiable + deterministic (the differentiators)

| # | Paper | scirust function | Module | Status | Effort |
|---|-------|------------------|--------|--------|--------|
| 1 | Gowal et al., *On the Effectiveness of Interval Bound Propagation* (2018) | `IbpMlp::certify(box) -> box` + `certified_robust` : intervals propagated layer by layer; **proven output bound** (soundness tested) | `nn::ibp` | ✅ | S |
| 2 | Zhang et al., *CROWN* (NeurIPS 2018) ; Wang et al., *β-CROWN* (NeurIPS 2021, arXiv:2103.06624) | `crown_bounds` — output bounds via linear relaxation (back-substitution) **tighter than IBP**; soundness + tighter-than-IBP tested; exposed in `certify` (displays IBP **and** CROWN) | `nn::ibp` | ✅ | L |
| 3 | Demmel & Nguyen, *Algorithms for Efficient Reproducible Floating-Point Summation* (ACM TOMS 2020) | `reproducible_sum`/`_mean`/`_dot` : **bit-identical sum regardless of order / thread count** (canonical sorting + exact expansion) | `reproducible` | ✅ | M |
| 4 | Katz et al., *Reluplex* (CAV 2017, arXiv:1702.01135) ; *Marabou* (CAV 2019) | `reluplex_verify`/`reluplex_unstable_count` : **complete** SMT-style verification — search for the **satisfiability** of a counterexample by **case-splitting ReLU phases**, but **lazily** : a neuron whose pre-activation interval stays on one side of 0 over the box is **stable** (phase forced, never split); only **unstable** neurons are split (`2^instables` leaves vs the eager MILP's `2^hiddens`); each leaf (a complete ReLU pattern) is affine, counterexample sought via exact 2D LP (shared with #31); oracle: **agreement with MILP** (two exact methods, ray sweep) + real counterexample (SAT) + **splits fewer** than all neurons (bound elimination) + determinism. Small network (2 inputs, 1 hidden layer) | `nn::ibp` | ✅ | XL |
| 5 | *DiFR: Inference Verification Despite Nondeterminism* (2025, arXiv:2511.20621) | `scirust_runtime::difr::difr_verify` : verifies an inference output **despite FP nondeterminism** — recomputes a **canonical reference** via `reproducible_dot` (products + sum in f64, order-independent) and accepts iff the claimed output lies in a **sound FP error envelope** (`γ·Σ\|terms\|` propagated through layers, ReLU 1-Lipschitz); oracle: accepts an f32 computation in a **different summation order**; envelope **sound** (1000 random orders all accepted) and **tight** (~ppm of scale ⇒ useful control); **rejects falsification** (beyond the envelope) + determinism. Extends the bit-exact `proof` certificates (which would reject an honest output across different hardware) | `scirust_runtime::difr` | ✅ | L |

## Tier 2 — N-D LLM stack (quick wins, gradient-checkable)

| # | Paper | scirust function | Module | Status | Effort |
|---|-------|------------------|--------|--------|--------|
| 6 | Zhang & Sennrich, *Root Mean Square Layer Normalization* (NeurIPS 2019) | `NdRmsNorm` (+ `rmsnorm` op) ; `NdLlamaBlock` | `autodiff::nd`, `nn::nd_layers` | ✅ | S |
| 7 | Shazeer, *GLU Variants Improve Transformer* (2020, arXiv:2002.05202) | `NdSwiGLU` (+ `sigmoid`/SiLU op) ; `NdLlamaBlock` | `autodiff::nd`, `nn::nd_layers` | ✅ | S |
| 8 | Su et al., *RoFormer / RoPE* (2021) | `rope` op (gradient-checked, relative position property tested) + `NdMultiHeadAttention::with_rope` | `autodiff::nd`, `nn::nd_layers` | ✅ | M |
| 9 | Milakov & Gimelshein, *Online normalizer for softmax* (2018) ; Dao, *FlashAttention-2* (arXiv:2307.08691) | Tiled FlashAttention + online-softmax with **backward** — **already present** | `nn::transformer::flash_attention` | ✅ | M |
| 10 | Leviathan et al., *Speculative Decoding* (ICML 2023) ; Chen et al., *Speculative Sampling* (2023) | `generate_speculative` (greedy variant): output **exactly** = target greedy, fewer forwards. + `generate_greedy` | `nn::nd_decoder` | ✅ | M |
| 11 | Ainslie et al., *GQA* (2023) ; Shazeer, *MQA* (2019) | `NdMultiHeadAttention::new_gqa` (`num_kv_heads`, sharing via broadcast `bmm`; MQA = 1) | `nn::nd_layers` | ✅ | M |

## Tier 3 — Optimizers

| # | Paper | scirust function | Module | Status | Effort |
|---|-------|------------------|--------|--------|--------|
| 12 | Loshchilov & Hutter, *Decoupled Weight Decay (AdamW)* (2017, arXiv:1711.05101) | decoupled `weight_decay` in `AdamConfig` + `NdAdam::with_lr_wd` | `nn::nd_optim` | ✅ | S |
| 13 | Chen et al., *Symbolic Discovery of Optimization Algorithms (Lion)* (2023) | `NdLion` (sign-based, half memory, deterministic) | `nn::nd_optim` | ✅ | S |
| 14 | Jordan et al., *Muon* (2024) | `NdMuon` (momentum + Newton-Schulz orthogonalization) + `newton_schulz_orthogonalize` | `nn::nd_optim` | ✅ | M |

## Tier 4 — Quantization (the bit-exact int8 thesis)

| # | Paper | scirust function | Module | Status | Effort |
|---|-------|------------------|--------|--------|--------|
| 15 | Frantar et al., *GPTQ* (2022) ; Lin et al., *AWQ* (2023) ; Xiao et al., *SmoothQuant* (2022, arXiv:2211.10438) | **SmoothQuant** (`smoothquant_scales`/`apply_smoothquant`) + per-channel int8 + **GPTQ** (`quantize_gptq`/`gptq_hessian`: 2nd-order error feedback via inverse calibration Hessian; `gptq` CLI) + **AWQ** (`awq_quantize`/`awq_act_scale`: activation-aware per-channel scaling by search; `awq` CLI). All three tested < round-to-nearest | `quantization` | ✅ | L |

## Tier 5 — Scientific computing bridge (unique fusion: solvers + autograd + symbolic)

| # | Paper | scirust function | Module | Status | Effort |
|---|-------|------------------|--------|--------|--------|
| 16 | Chen et al., *Neural ODEs* (NeurIPS 2018, arXiv:1806.07366) | `rk4_integrate` + `NeuralOde`: backprop **through** the RK4 solver on the N-D tape | `nn::neural_ode` | ✅ | M |
| 17 | Raissi, Perdikaris & Karniadakis, *PINNs* (J. Comp. Phys. 2019) | `nn::pinn` (`Pinn1D`, `solve_harmonic`): **PDE residual in the loss** — solves the boundary value problem `u''=−u`, `u(0)=0`, `u(π/2)=1` (solution `sin x`); `u''` by finite differences in the input (shared network, exact param grads via reverse autodiff); verified vs analytic solution (max error ≈ 0.004); `pinn` CLI | `nn::pinn` | ✅ | L |

## Tier 6 — Alternative architectures & trust

| # | Paper | scirust function | Module | Status | Effort |
|---|-------|------------------|--------|--------|--------|
| 18 | Gu & Dao, *Mamba* (2023, arXiv:2312.00752) | `selective_scan` + `NdMamba` — S6 *selective scan* (input-dependent Δ, B, C; diagonal A; `exp(Δ·A)` discretization), deterministic linear-time recurrence unrolled on the tape (new `exp` op) ⇒ **gradient check**; reference match + training; `mamba` CLI | `nn::nd_layers` | ✅ | XL |
| 19 | Abadi et al., *Deep Learning with Differential Privacy (DP-SGD)* (2016) | `clip_gradients` + `add_noise` (gaussian **seeded**) + `dp_protect` + moments accountant (Rényi DP) — **already present** | `dp` | ✅ | M |
| 20 | Frantar & Alistarh, *SparseGPT* (2023) ; Sun et al., *Wanda* (2023) ; Frankle & Carbin, *Lottery Ticket* (2019) | `prune_wanda` (activation-aware) + magnitude/structured/Lottery-Ticket already present | `pruning` | ✅ | M |

## Tier 7 — New papers (cycle 2, 14/06 research)

Found during a second search; chosen for their fit with the
fundamentals (certifiable, deterministic, implementable, testable).

| # | Paper | scirust function | Module | Status | Effort |
|---|-------|------------------|--------|--------|--------|
| 21 | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (2021, arXiv:2107.07511) | `nn::conformal` : `conformal_quantile`, `ConformalRegressor`, `ConformalClassifier` — guaranteed coverage *without distributional assumptions*; tests: empirical coverage ≈ 1−α (regression + classification). `scirust conformal` CLI. | `nn::conformal` | ✅ | M |
| 22 | Defazio et al., *The Road Less Scheduled (Schedule-Free)* (2024 ; MLCommons AlgoPerf winner) | `NdScheduleFree` : optimizer **without LR schedule** (Polyak averaging `x`, separate eval point); deterministic; `lm --opt schedule-free` CLI | `nn::nd_optim` | ✅ | M |
| 23 | Pagliardini et al. (Apple), *The AdEMAMix Optimizer* (2024, arXiv:2409.03137) | `NdAdEMAMix` : **two gradient EMAs** (fast β1 + slow β3, α mixing); deterministic; `lm --opt ademamix` CLI | `nn::nd_optim` | ✅ | M |
| 24 | Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* (2024) | `NdSoap` — Adam in Shampoo's **eigenbasis** (`L=E[GGᵀ]`, `R=E[GᵀG]`; deterministic **Jacobi** eigensolver `jacobi_eigenvectors`); convergence + determinism tested; `lm --opt soap` CLI | `nn::nd_optim` | ✅ | L |
| 25 | Yang et al., *Gated Delta Networks / DeltaNet* (2024, arXiv:2412.06464) | `delta_rule` + `NdDeltaNet` — **recurrent linear attention** (delta rule: `S_t = S_{t-1} + β_t(v_t − S_{t-1}k_t)k_tᵀ`), linear time, causal, deterministic; unrolled on the tape (new `cat0` op) ⇒ **gradient check**; reference match + training; `deltanet` CLI | `nn::nd_layers` | ✅ | L |

## Tier 8 — Verified candidates (cycle 3, 15/06 research) — verification & certified robustness

> ~55 real papers (verified arXiv) found in research, translated into concrete
> scirust functions, chosen for their fit with the fundamentals (certifiable,
> deterministic, testable, pure Rust). All 📋 (candidates to implement), at the same
> standard as the ✅ (test/oracle + 8 gates). Extend IBP/CROWN (#1-2).

| # | Paper | Proposed scirust function | Module | Status | Effort |
|---|-------|---------------------------|--------|--------|--------|
| 26 | Zhang et al., *GCP-CROWN : General Cutting Planes for BaB Verification* (NeurIPS 2022, arXiv:2208.05740) | `verify_robustness`/`BabResult` : **complete** **branch-and-bound** verifier — bounds class margins via DeepPoly; if all > 0 the box is **robust**, otherwise probes the center for a concrete **counterexample**, otherwise **splits** the box (input domain, widest axis) and recurses; as boxes shrink DeepPoly becomes exact ⇒ **decides** (Robust / Unsafe+counterexample / Unknown beyond tolerance); oracle: Robust **sound** (5000 pts), **certified radius > DeepPoly alone** (extra sampled sound region), Unsafe = real counterexample, deterministic; exposed via `certify` CLI. (Split of the **input domain**; unstable-ReLU splitting and GCP-CROWN cutting planes are not implemented.) | `nn::ibp` + CLI | ✅ | XL |
| 27 | Cohen, Rosenfeld & Kolter, *Certified Robustness via Randomized Smoothing* (ICML 2019, arXiv:1902.02918) | `nn::smoothing::SmoothedClassifier::certify` : smoothed classifier, **proven L2 radius** `σ·Φ⁻¹(pₐ)` via seeded gaussian noise + **Clopper-Pearson** bound (`betai`/`lgamma`) + probit `Φ⁻¹` (Acklam); oracle: radius = exact distance to the halfspace (σ-independent) + soundness/abstention + determinism; `certify` CLI (IBP/CROWN + smoothing) | `nn::smoothing` | ✅ | M |
| 28 | Singh et al., *DeepPoly : An Abstract Domain for Certifying NN* (POPL 2019) | `deeppoly_certify`/`IbpMlp::certify_deeppoly` : **relational** abstract domain — each neuron keeps an **affine-in-the-inputs** lower/upper bound (back-substitution), **asymmetric** ReLU relaxation (upper bound = chord `(u/(u−l))(y−l)`, lower bound `λy` with minimal area `λ=1 if u>−l else 0`) ⇒ tighter than IBP **at any depth** (vs `crown_bounds` limited to 2 layers); oracle: **sound** (4000 pts ∈ box, 3-layer MLP) + **tighter than IBP** (`relu(x)+relu(−x)=\|x\|`: DeepPoly **exact** [0,1] vs IBP [0,2]) + determinism; exposed in `certify` CLI | `nn::ibp` | ✅ | L |
| 29 | Gehr et al., *AI² : Abstract Interpretation for NN* (IEEE S&P 2018) | `Zonotope`/`certify_zonotope` : propagation via **zonotopes** (DeepZ) — exact affine, relaxed ReLU `λx+μ±μ` (1 generator/unstable neuron); oracle: exact affine + **soundness** (sampling) + **tighter than IBP** under correlation (`relu(x)−relu(x)`: zono [−0.5;0.5] vs IBP [−1;1]); exposed in `certify` CLI | `nn::ibp` | ✅ | L |
| 30 | Zhang et al., *CROWN-IBP : Stable & Efficient Verified Training* (ICLR 2020, arXiv:1906.06316) | `nn::crown_ibp::CrownIbpMlp` : **certified training** — **differentiable** IBP propagation on the tape (`\|W\|=relu(W)+relu(−W)`, ReLU-interval via `relu`) ⇒ **robust logits** (true class at its lower bound, others at their upper bound), loss = cross-entropy on these logits ⇒ network **provably** robust; oracle: tape IBP ≡ reference `IbpMlp` verifier + sound (2000-pt sampling) + **certified radius grows** (robustly-trained network certifies a much larger ℓ∞ radius than the accuracy-only network, both classifying correctly) + determinism | `nn::crown_ibp` | ✅ | L |
| 31 | Tjeng, Xiao & Tedrake, *Evaluating Robustness with MILP* (ICLR 2019, arXiv:1711.07356) | `milp_min_margin`/`milp_verify_robustness` : **exact** verification of a small ReLU network (2 inputs, 1 hidden layer) via the MILP formulation — the ReLU **activation patterns** are the binary variables; enumerate them and solve each LP **exactly** (margin `logitₜ−logⱼ` affine per pattern, minimized over the box ∩ activation halfspaces by 2D vertex enumeration) ⇒ exact **global minimum**; `>0` ⇒ robust, else exact counterexample; oracle: **= brute force** (fine grid, bound ≤ any sample) + real counterexample + **bound ≥ DeepPoly** (DeepPoly sound) and strictly tighter at some radii + determinism | `nn::ibp` | ✅ | L |
| 32 | Leino, Wang & Fredrikson, *Globally-Robust Neural Networks (GloRo)* (ICML 2021) | `nn::lipschitz` : `spectral_norm` (power iteration) + `spectral_normalize` (**1-Lipschitz** layer) + `GloroClassifier` (proven L2 radius `margin/(√2‖W‖₂)`); oracle: known spectral norms + **sound** radius (worst perturbation does not flip) + **conservative** (≤ exact distance to the boundary) + determinism | `nn::lipschitz` | ✅ | M |

## Tier 9 — Uncertainty, calibration & conformal (beyond split-conformal #21)

| # | Paper | Proposed scirust function | Module | Status | Effort |
|---|-------|---------------------------|--------|--------|--------|
| 33 | Romano, Patterson & Candès, *Conformalized Quantile Regression (CQR)* (NeurIPS 2019, arXiv:1905.03222) | `ConformalQuantileRegressor` : conformalizes a quantile regressor (score `Eᵢ=max(q_lo−y, y−q_hi)`, finite correction `Q`) ⇒ **adaptive** intervals `[q_lo−Q, q_hi+Q]`, marginal coverage ≥ 1−α; oracle: exact score semantics + coverage (band) + varying width (strong vs weak noise region) + determinism; `conformal` CLI (split + CQR) | `nn::conformal` | ✅ | M |
| 34 | Romano, Sesia & Candès, *Classification with Valid & Adaptive Coverage (APS)* (NeurIPS 2020, arXiv:2006.02544) | `AdaptivePredictionSets` : **classification** sets by cumulative score `s(x,c)` (mass of classes at least as probable as c); set `{c : s≤q̂}`, marginal coverage ≥ 1−α + **adaptive size** (easy→small, ambiguous→large); oracle: exact score + coverage + adaptivity + determinism | `nn::conformal` | ✅ | M |
| 35 | Angelopoulos et al., *RAPS : Regularized Adaptive Prediction Sets* (ICLR 2021, arXiv:2009.14193) | `AdaptivePredictionSets::calibrate_raps` : penalty `λ·max(0, rank−k_reg)` added to the score ⇒ **smaller** sets than APS at equal coverage; oracle: mean RAPS size < APS with coverage ≥ 1−α | `nn::conformal` | ✅ | M |
| 36 | Bates et al., *Risk-Controlling Prediction Sets (RCPS)* (J. ACM 2021, arXiv:2101.02703) | `hoeffding_ucb` + `rcps_select` : control of a bounded **risk** (beyond coverage) — smallest λ whose concentration bound (Hoeffding) is ≤ α ⇒ `R(λ̂)≤α` w.p. 1−δ (PAC); oracle: exact selection + test risk ≤ α on fresh data | `nn::conformal` | ✅ | M |
| 37 | Angelopoulos et al., *Learn then Test* (arXiv:2110.01052) | `learn_then_test` : control of **multiple risks** via Hoeffding p-values + family-wise correction (Bonferroni); FWER ≤ δ verified by simulation; deterministic | `nn::conformal` | ✅ | M |
| 38 | Gibbs & Candès, *Adaptive Conformal Inference* (NeurIPS 2021, arXiv:2106.00170) | `AdaptiveConformal` : **online** conformal — level αₜ adapted by feedback `αₜ₊₁=αₜ+γ(α−errₜ)` ⇒ coverage ≈ 1−α **under drift** (where static conformal collapses); oracle: exact update rule + coverage maintained under variance shift + determinism | `nn::conformal` | ✅ | M |
| 39 | Guo et al., *On Calibration of Modern NN (Temperature Scaling)* (ICML 2017, arXiv:1706.04599) | `nn::calibration` : `temperature_scale` (golden-section on NLL) + `expected_calibration_error` + `nll`; post-hoc recalibration **without changing accuracy**; oracle: ECE decreases (tested, deterministic); `calibrate` CLI | `nn::calibration` | ✅ | S |
| 40 | Lakshminarayanan et al., *Deep Ensembles* (NeurIPS 2017, arXiv:1612.01474) | `nn::ensemble::DeepEnsemble` : N seeded ReLU MLPs (tape + NdAdam); `predict→(mean, std)` = estimate + **epistemic uncertainty**; oracle: ensemble MSE ≤ member mean (Jensen) + std **≫ out-of-distribution** + determinism | `nn` | ✅ | M |

## Tier 10 — Optimizers (beyond Adam/Lion/Muon/SF/AdEMAMix/SOAP)

| # | Paper | Proposed scirust function | Module | Status | Effort |
|---|-------|---------------------------|--------|--------|--------|
| 41 | Gupta, Koren & Singer, *Shampoo* (ICML 2018, arXiv:1802.09568) | `NdShampoo` : **Kronecker** preconditioner (`L^{-1/4} G R^{-1/4}`, inverse roots via `inverse_pth_root`/`jacobi_eigenvectors`); matrices → preconditioned update, vectors → diagonal Adagrad; inverse-root oracle (`A^{-1/2}²·A≈I`) + convergence + determinism tested; `lm --opt shampoo` CLI | `nn::nd_optim` | ✅ | L |
| 42 | Shazeer & Stern, *Adafactor* (ICML 2018, arXiv:1804.04235) | `NdAdafactor` : **factorized** 2nd-order moments (row/column sums `V[i,j]=R[i]·C[j]/ΣR`, sublinear memory) + RMS clipping of the update + β2 scheduling; exact rank-1 reconstruction + convergence (band) + determinism tested; `lm --opt adafactor` CLI | `nn::nd_optim` | ✅ | M |
| 43 | You et al., *LAMB* (ICLR 2020, arXiv:1904.00962) | `NdLamb` : Adam with **per-layer trust** (ratio `‖θ‖/‖r‖` per tensor); convergence (band) + determinism tested; `lm --opt lamb` CLI | `nn::nd_optim` | ✅ | M |
| 44 | Liu et al., *Sophia* (arXiv:2305.14342) | `NdSophia` : **clipped** 2nd order — `θ←θ−lr·clip(m/max(γ·h,eps),ρ)` with `h` = EMA of the **diagonal Hessian** estimated by **Hutchinson** (`ĥ=v⊙Hv`, `v∈{±1}` seeded) via Hessian-vector product in **finite differences** (`Hv≈(∇L(θ+εv)−∇L(θ))/ε`, exact for a quadratic); Newton-like steps in curved directions, sign-like bounded in flat ones; oracle: **converges on ill-conditioned quadratic** (curvatures 4 vs 0.25, cond. 16) + bit-exact determinism. Like SAM (2 gradients/step ⇒ orchestrated by the caller), library **outside the `lm --opt` loop** | `nn::nd_optim` | ✅ | L |
| 45 | Zhang et al., *Lookahead* (NeurIPS 2019, arXiv:1907.08610) | `NdLookahead` : **slow/fast weights** wrapper around Adam (`k` fast steps then `φ←φ+α(θ−φ); θ←φ`); deterministic; convergence + determinism tested; `lm --opt lookahead` CLI | `nn::nd_optim` | ✅ | S |
| 46 | Mishchenko & Defazio, *Prodigy* (arXiv:2306.06101) | `NdProdigy` : Adam **without learning rate** — estimates `d≈‖x₀−x*‖` online (global correlation `⟨g,x₀−x⟩`) and uses it as the step; oracle: `d` adapts to scale + loss reduced + determinism; `lm --opt prodigy` CLI | `nn::nd_optim` | ✅ | M |
| 47 | Foret et al., *Sharpness-Aware Minimization (SAM)* (ICLR 2021, arXiv:2010.01412) | `NdSam` : `ascent` (perturb toward `θ+ρ·g/‖g‖`, local worst case) then `descent` (restores θ, SGD step on perturbed gradient); oracle: perturbation = `ρ·g/‖g‖` (‖ε‖=ρ) + convergence (band ∝ lr·ρ) + determinism; library (2 gradients/step ⇒ outside `lm --opt` loop) | `nn::nd_optim` | ✅ | M |
| 48 | Zhao et al., *GaLore* (ICML 2024, arXiv:2403.03507) | `NdGalore`/`galore_subspace` : **low-rank projection of gradients** — Adam runs in the dominant subspace `PᵀG` (top-`r` singular vectors, reuses `jacobi_eigenvectors`) ⇒ compressed optimizer states `r×max(m,n)`; oracle: `P` orthonormal + optimal orthogonal projection (Pythagoras) + low-rank gradient reconstructed exactly + **convergence on low-rank target with compressed state** + below rank ⇒ residual + determinism; `lm --opt galore` CLI | `nn::nd_optim` | ✅ | M |
| 49 | Xie et al., *Adan* (arXiv:2208.06677) | `NdAdan` : **adaptive Nesterov** momentum (3 EMAs: gradient, differences, squared look-ahead term); convergence + determinism tested; `lm --opt adan` CLI | `nn::nd_optim` | ✅ | M |

## Tier 11 — Efficient sequence models (beyond Mamba/DeltaNet/Flash/RoPE/GQA)

| # | Paper | Proposed scirust function | Module | Status | Effort |
|---|-------|---------------------------|--------|--------|--------|
| 50 | Dao & Gu, *Mamba-2 / Structured State-Space Duality* (ICML 2024, arXiv:2405.21060) | `ssd_dual`/`NdMamba2` : the SSD **duality** — restricting `A` to a per-step **scalar** makes the recurrence `Hₜ=aₜHₜ₋₁+xₜBₜᵀ`, `yₜ=HₜCₜ` **exactly equal** to a masked quadratic form `Y=(L⊙CBᵀ)X`, `L[i,j]=∏_{j<k≤i}aₖ` (cumulative decay = prefix-sum of `a_log` via triangular matmul, masked **before** `exp` ⇒ no overflow); oracle: **dual form ≡ sequential recurrence** (the duality) + **gradient check** (x, B, C, a_log) + `NdMamba2` trains (MSE↓) + determinism | `nn::nd_layers` | ✅ | L |
| 51 | Gu, Goel & Ré, *S4 : Structured State Spaces* (ICLR 2022, arXiv:2111.00396) | `s4_scan`/`NdS4` : diagonal **LTI** SSM (S4D) — `Ā=exp(Δ⊙A)`, `B̄=Δ⊙B`, `h_t=Ā⊙h_{t−1}+B̄⊙x_t`, `y_t=Σ_n C⊙h_t` unrolled on the tape; **HiPPO** diagonal init `A[:,j]=−(j+1)`; oracle: **gradient check** (x, a_log, B, C, log_dt) + training (MSE↓) + determinism | `nn::nd_layers` | ✅ | L |
| 52 | Smith, Warrington & Linderman, *S5* (ICLR 2023, arXiv:2208.04933) | `s5_scan`/`s5_parallel_scan`/`NdS5` : diagonal **MIMO** SSM — a **single shared** `n`-dim state driven by all inputs via `B`, read via `C` (vs S4D's per-channel SISO); recurrence `hₜ=Ā⊙hₜ₋₁+xₜB`, `yₜ=hₜC`; **associative parallel scan** (Hillis-Steele, combine `(a₁,u₁)∘(a₂,u₂)=(a₂a₁, a₂u₁+u₂)`, fixed doubling order ⇒ deterministic); oracle: **parallel scan ≡ sequential recurrence** (with time-varying `aₜ`, which proves the associativity that licenses parallelization) + `s5_scan` ≡ MIMO reference + **gradient check** (x, Ā, B, C) + `NdS5` trains (MSE↓) + determinism | `nn::nd_layers` | ✅ | L |
| 53 | Peng et al., *RWKV* (EMNLP Findings 2023, arXiv:2305.13048) | `rwkv_wkv` + `NdRwkv` : **WKV** temporal mixing (per-channel exp. decay + bonus, normalized) unrolled on the tape (new `div` op); oracle: recurrent ≡ explicit formula + gradient check (k,v,decay,bonus) + training + determinism; `rwkv` CLI | `nn::nd_layers` | ✅ | L |
| 54 | Sun et al., *RetNet : Retentive Network* (arXiv:2307.08621) | `retention` + `NdRetention` : linear attention recurrence with decay γ (`S_t=γS_{t-1}+kₜᵀvₜ`, `o_t=q_tS_t`) unrolled on the tape; **oracle: recurrent form ≡ parallel form** `(QKᵀ⊙D)V` + gradient check + training; `retnet` CLI | `nn::nd_layers` | ✅ | L |
| 55 | Yang et al., *Gated Linear Attention (GLA)* (ICML 2024, arXiv:2312.06635) | `gated_linear_attention` + `NdGla` : **gated** linear attention — **input-dependent** per-channel forget gate `αₜ=σ(·)` (`S_t=diag(αₜ)S_{t-1}+kₜᵀvₜ`), unrolled on the tape; reference match + gradient check (q,k,v,α) + training; `gla` CLI | `nn::nd_layers` | ✅ | L |
| 56 | Poli et al., *Hyena* (ICML 2023, arXiv:2302.10866) | `hyena_long_conv`/`NdHyena` : **attention-free** operator — **implicit long convolutions** (filter **generated** by an MLP from a positional encoding + learnable `exp(−γ·t̄)` window ⇒ long filter with few parameters) interleaved with **data-dependent multiplicative gating** (`z=x1⊙(h1*v)`, `z=x2⊙(h2*z)`); causal conv = `Σ_τ h[τ]⊙(Sτ·u)` (constant shift matrices ⇒ differentiable without a scatter op); oracle: conv ≡ hand-written reference + **gradient check** (u, h) + training (MSE↓) + determinism | `nn::nd_layers` | ✅ | L |
| 57 | Beck et al., *xLSTM* (NeurIPS 2024, arXiv:2405.04517) | `slstm_scan`/`mlstm_scan`/`NdXlstm` : extended LSTM — **scalar sLSTM** (**exponential** input gate `iₜ=exp(ĩₜ)` + normalizer state `nₜ`, `hₜ=oₜ⊙cₜ/nₜ`; `tanh=2σ(2x)−1`, output bounded in (−1,1) ⇒ stable without stabilizer) and **matrix mLSTM** (covariance memory `d×d` via outer products `vₜᵀkₜ`, query read, denominator `max(\|nₜ·qₜ\|,1)` **exact** via `\|a\|=relu(a)+relu(−a)`, `max(a,1)=relu(a−1)+1`) unrolled on the tape; oracle: mLSTM ≡ reference recurrence + **gradient check** (sLSTM: 4 gates; mLSTM: q,k,v,iₜ,fₜ) + training (MSE↓) + determinism | `nn::nd_layers` | ✅ | L |
| 58 | Qin et al., *HGRN : Hierarchically Gated RNN* (NeurIPS 2023, arXiv:2311.04823) | `hgrn` + `NdHgrn` : linear RNN with per-channel leaky integration, **lower-bounded** forget gate `f=lb+(1−lb)σ(·)` (the bound `lb` sets the minimal memory horizon, increasing per layer); reference match + gradient check (c,f) + training; `hgrn` CLI | `nn::nd_layers` | ✅ | M |
| 59 | Press, Smith & Lewis, *ALiBi* (ICLR 2022, arXiv:2108.12409) | `alibi_slopes` + `alibi_bias` (attention bias **linear in distance**, slopes `2^(−8h/H)`) + `NdMultiHeadAttention::with_alibi`; oracle: geometric slopes + linear/causal/Toeplitz bias + decreasing softmax weights (∝ exp(−slope·dist)) + deterministic attention | `nn::nd_layers` | ✅ | S |
| 60 | Peng et al., *YaRN* (arXiv:2309.00071) | `nn::yarn` : `yarn_frequencies`/`rope_yarn` — RoPE context extension via **NTK-by-parts interpolation** (keeps high frequencies, interpolates low ones) + `yarn_attention_scale` temperature; oracle: **relative position** property preserved + low-frequency angle brought back in distribution at `s·L` + `scale=1` ≡ plain RoPE | `nn::yarn` | ✅ | M |

## Tier 12 — Efficient decoding & inference (beyond speculative #10)

| # | Paper | Proposed scirust function | Module | Status | Effort |
|---|-------|---------------------------|--------|--------|--------|
| 61 | Cai et al., *Medusa* (ICML 2024, arXiv:2401.10774) | `MedusaHeads`/`generate_medusa` : **multiple decoding heads** (head `j` predicts the token at +`j+2` from the hidden state) ⇒ multi-token draft from a single forward, verified (accepted prefix + greedy correction); oracle: output **exactly = greedy** for arbitrary heads + determinism + trained heads ⇒ blocks accept >1 token (forwards < 2·n) while staying exact | `nn::nd_decoder` | ✅ | M |
| 62 | Li et al., *EAGLE* (ICML 2024, arXiv:2401.15077) | `EagleHead`/`generate_eagle` : **features-level** speculative decoding — head `(feature, embed(token)) → next feature` autoregressed, mapped by the frozen LM head ⇒ draft; verified (prefix + correction); oracle: output **exact = greedy** for an arbitrary head + determinism + trained head (MSE on features) ⇒ blocks accept >1 token (forwards < 2·n) while staying exact | `nn::nd_decoder` | ✅ | M |
| 63 | Kwon et al., *PagedAttention / vLLM* (SOSP 2023, arXiv:2309.06180) | `nn::paged_attention::PagedKvCache` : **paged** KV-cache (blocks from a pool, block table), attention indexed **via the table**; oracle: gather **bit-identical** under fragmentation (interleaved decoy blocks) + paged `attention` **bit-identical** to the contiguous cache + block accounting/empty case | `nn::paged_attention` | ✅ | M |

## Tier 13 — Quantization & compression (beyond GPTQ/AWQ/SmoothQuant #15)

| # | Paper | Proposed scirust function | Module | Status | Effort |
|---|-------|---------------------------|--------|--------|--------|
| 64 | Tseng et al., *QuIP#* (ICML 2024, arXiv:2402.04396) | `quantize_quip`/`nearest_e8`/`random_hadamard_transform` : **incoherence** via randomized Hadamard transform (seeded ±1 signs + FWHT, orthogonal ⇒ spreads the outliers, shrinks the dynamic range the `2^bits` levels must cover, at equal budget) + **E8 lattice** codebook (Conway-Sloane decoder `D8 ∪ (D8+½)`, denser than the cubic grid at equal density ⇒ ~14 % packing gain); oracle: RHT orthogonal (round-trip) + reduces outlier range + E8 valid & **< cubic grid** on average (lattice gain) + end-to-end **< scalar RTN** at 2-bit on outlier weights + determinism | `quantization` | ✅ | L |
| 65 | Shao et al., *OmniQuant* (ICLR 2024, arXiv:2308.13137) | `omniquant_quantize` : **learnable weight clipping** (LWC) — per-output-channel cut factor `γ∈(0,1]` that shrinks the range to `γ·max\|w\|` (deterministic grid search **including γ=1=RTN**); oracle: **< RTN** on heavy-tailed weights (at least one channel clips) + **never worse** than RTN + determinism | `quantization` | ✅ | L |
| 66 | Kim et al., *SqueezeLLM* (arXiv:2306.07629) | `SqueezeLlmCodebook` : **non-uniform** quantization by **sensitivity-weighted** k-means (diag. Hessian proxy); quantile init + deterministic Lloyd; oracle: weighted error **< RTN** (gaussian, 3 bits, <0.85×) + exact round-trip + determinism; library | `quantization` | ✅ | M |
| 67 | Dettmers et al., *SpQR* (arXiv:2306.03078) | `SpqrOutliers` : **sparse-quantized** — keeps the outlier fraction (largest quantif errors) in fp, the rest in low-bit dense; oracle: heavy-tail error divided (1 % outliers ⇒ error < 0.3×) + exact outlier reconstruction + determinism; library | `quantization` | ✅ | M |
| 68 | Hooper et al., *KVQuant* (NeurIPS 2024, arXiv:2401.18079) | `kvquant_kv` : **KV-cache** quant — **per-channel** keys (channel outliers), **per-token** values; oracle: attention error **< per-tensor** (<0.5×, keys with channel outliers) + per-channel resolves small columns + determinism | `quantization` | ✅ | M |
| 69 | Ma et al., *BitNet b1.58* (arXiv:2402.17764) | `ternary_quantize` + `ternary_matmul` : **ternary {−1,0,1}** weights (absmean scale, ~1.58 bit/weight); **multiplication-free** matmul (sum/diff/skip); **oracle**: = dequantized product (bit-exact for the sum-of-signs form); `bitnet` CLI | `quantization` | ✅ | M |
| 70 | Egiazarian et al., *AQLM : Additive Quantization* (ICML 2024, arXiv:2401.06118) | `quantize_aqlm`/`AqlmResult` : **additive** multi-codebook quantization — each weight group ≈ **sum** of one codeword per codebook (M codebooks of K words, vectors of dim g); codebooks **learned** by residual k-means then **alternating optimization** (greedy re-encoding + LS readjustment of each codebook; AQLM's beam search simplified to greedy residual assignment); oracle: reconstruction **< 0.7× scalar RTN** at equal ~2-bit budget on structured weights (the **vector** codebooks capture inter-dimension structure that scalar RTN cannot) + round-trip (non-divisible length) + determinism | `quantization` | ✅ | L |
| 71 | Dettmers et al., *LLM.int8()* (NeurIPS 2022, arXiv:2208.07339) | `int8_mixed_matmul` : mixed matmul — **outlier** feature columns (>threshold) kept in fp32, the rest in **int8**; oracle: error vs fp **< 0.5×** plain int8 (activations with outliers) + reduces to pure int8 without outliers + determinism | `quantization` | ✅ | M |
| 72 | Hu et al., *LoRA* (ICLR 2022, arXiv:2106.09685) | `LoraLinear` : **low-rank** adaptation (`W` frozen + `ΔW = (α/r)·A·B`, only `A`,`B` trained); `B=0` at init ⇒ = base; gradient check on `A`,`B`; N-D tape layer | `nn::nd_layers` | ✅ | M |
| 73 | Liu et al., *DoRA* (ICML 2024, arXiv:2402.09353) | `nn::dora::DoraLinear` : LoRA decomposed **magnitude/direction** `W'=m⊙(W₀+BA)/‖W₀+BA‖_col` (W₀ frozen; m, A, B trained); closed-form backward of the per-column normalization; oracle: init `B=0, m=‖W₀‖_col` ⇒ **W'=W₀ exact** + **gradient check** (finite differences vs analytic) + recovers a DoRA target (loss ÷100) + determinism | `nn::dora` | ✅ | M |
| 74 | Dettmers et al., *QLoRA / NF4* (NeurIPS 2023, arXiv:2305.14314) | `nf4_quantize`/`nf4_dequantize` + `NF4_LEVELS` : 4-bit **NormalFloat** type (16 levels = normal quantiles, absmax scale); **oracle**: error < uniform int4 on gaussian weights (+ exact round-trip + determinism) | `quantization` | ✅ | M |

## Tier 14 — Scientific computing, determinism & audit (beyond Neural ODE/PINN/reproducible)

| # | Paper | Proposed scirust function | Module | Status | Effort |
|---|-------|---------------------------|--------|--------|--------|
| 75 | Li et al., *Fourier Neural Operator (FNO)* (ICLR 2021, arXiv:2010.08895) | `nn::fno` : `FnoSpectralConv1d`/`NdFno` — learned operator in the **Fourier domain**. Real DFT = **fixed cos/sin matrices** (deterministic matmul, differentiable on the tape, no FFT nor complex type); keep the `modes` low frequencies, multiply each mode by a learned **complex weight** `R_k=Ar_k+iAi_k` (channel mixing via `bmm`), then DFT⁻¹; `σ(spectral+local)` block. Oracle: **exact** reconstruction of a band-limited signal (DFT⁻¹∘DFT) + **gradient check** (v, Ar, Ai) + **learns the derivation operator** (`d/dx↔×ik`, exactly representable) and **generalizes** to an unseen phase (test MSE <0.02) + determinism | `nn::fno` | ✅ | L |
| 76 | Lu et al., *DeepONet* (Nature Mach. Intell. 2021, arXiv:1910.03193) | `nn::deeponet::DeepONet` : **operator** learning `G(u)(y)=Σ b_k(u)·t_k(y)` (fixed cosine trunk + linear branch = POD-DeepONet, convex); oracle: learns the **antiderivative** and **generalizes to unseen functions** (test MSE < 0.01, ≪ baseline) + determinism | `nn::deeponet` | ✅ | L |
| 77 | Liu et al., *KAN : Kolmogorov-Arnold Networks* (arXiv:2404.19756) | `nn::kan::KanLayer` : **learnable edge activations** (FastKAN RBF basis, Li 2024) `y_j=Σᵢφᵢⱼ(xᵢ)`; output linear in coefficients ⇒ convex fitting (GD); oracle: fits a non-linear additive target `sin(2x₀)+x₁²` (MSE<0.02, ≪ linear model) + localized basis + determinism; library | `nn::kan` | ✅ | L |
| 78 | Mironov, *Rényi Differential Privacy* (CSF 2017, arXiv:1702.07476) | `dp::gaussian_rdp`/`rdp_to_dp`/`rdp_gaussian_epsilon` : **RDP** of the gaussian mechanism `α/(2σ²)` + Mironov conversion `ε=RDP+ln(1/δ)/(α−1)` optimized over α; strengthens DP-SGD (#19); oracle: exact RDP/conversion + **much tighter** than basic composition + monotonicity | `dp` | ✅ | M |
| 79 | Kirchenbauer et al., *A Watermark for LLMs* (ICML 2023, arXiv:2301.10226) | `nn::watermark` : `is_green`/`apply_green_bias`/`detect_z` — **seeded** green/red partition (hash of `(seed,prev,token)`) + logit bias + **z-test** detection `(g−γn)/√(nγ(1−γ))` **without model access**; oracle: green fraction ≈ γ + bias on green tokens only + watermarked text detected (z≫8) vs natural (z≈0) + wrong seed not detected + determinism | `nn::watermark` | ✅ | M |
| 80 | *ZK-based Verifiable ML* (survey arXiv:2502.18535 ; zkSNARK eval arXiv:2402.02675) | `scirust_runtime::vinfer` : **verifiable inference** — model (full linear layer over `GF(2³¹−1)`) **committed** via hash; verification of a batched output `Y` by **Freivalds** over the field (`W·(X·r)=Y·r`, `r` drawn by **Fiat-Shamir** from `hash(commitment,X,Y)`) ⇒ **compact** argument (`O(out·in+in·b)` vs recomputation `O(out·in·b)`), **sound** (false `Y` passes with probability ≤ (1/p)^k); oracle: accepts a correct inference; **1000 falsifications all rejected**; commitment **binds** the model; Fiat-Shamir **binds** the output (output of other inputs rejected); deterministic. Cryptographic soundness (**not** zero-knowledge: the verifier holds the weights; the full zk-SNARK hiding the weights remains out of scope) | `scirust_runtime::vinfer` | ✅ | XL |

---

## Attack order

**✅ Delivered / present** (tested + 8 green gates): certified IBP (#1) · **CROWN
(#2)** · reproducible summation (#3) · N-D RoPE (#8) · RMSNorm + SwiGLU +
`NdLlamaBlock` (#6, #7) · FlashAttention online-softmax (#9) · exact speculative
decoding (#10) · GQA/MQA (#11) · AdamW + Lion (#12, #13) · Muon (#14) · Neural ODE
(#16) · DP-SGD (#19) · pruning Wanda + magnitude/lottery (#20) · **SmoothQuant +
GPTQ + AWQ (#15)** · **conformal prediction (#21)** · **Schedule-Free (#22)** ·
**AdEMAMix (#23)** · **SOAP (#24)** · **DeltaNet (#25)** · **Mamba (#18)** ·
**PINN (#17)**. → **18/20 + #21 + #22 + #23 + #24 + #25**.

**Heavy bets** (planned, milestone-tracked): SMT/Marabou (#4) · DiFR (#5).

**Verified candidate pool (Tier 8-14, #26-#80, ~55 papers, 15/06 research)**:
next implementations, by estimated tractability order —
*quick wins*: Lookahead (#45), ALiBi (#59), temperature scaling (#39),
LoRA (#72) ; *sequence models* (reuse tape + `cat0`/`exp`): Mamba-2
(#50), GLA (#55), RetNet (#54), RWKV (#53), S4/S5 (#51/#52), HGRN (#58) ;
*optimizers* (reuse `jacobi_eigenvectors`): Shampoo (#41), Sophia (#44),
Adafactor (#42), LAMB (#43), Adan (#49), Prodigy (#46), SAM (#47) ;
*quantization* (oracle < RTN): QuIP# (#64), AQLM (#70), BitNet b1.58 (#69),
SqueezeLLM (#66), SpQR (#67), KVQuant (#68), NF4 (#74), LLM.int8 (#71) ;
*certifiable* (extends IBP/CROWN): randomized smoothing (#27), GCP-CROWN BaB
(#26), CROWN-IBP (#30), MILP (#31), DeepPoly/AI² (#28/#29), Lipschitz (#32) ;
*conformal/uncertainty*: CQR (#33), APS/RAPS (#34/#35), RCPS+LtT (#36/#37), ACI
(#38), deep ensembles (#40) ; *decoding*: Medusa (#61), EAGLE (#62),
PagedAttention (#63) ; *scientific*: FNO (#75), DeepONet (#76), KAN (#77) ;
*audit/privacy*: Rényi-DP accountant (#78), LLM watermark (#79), ZK
inference proof (#80). All at the same standard: test/oracle + 8 gates before ✅.

Each item respects the fundamentals: autograd op ⇒ **gradient check**;
guarantee (bound, privacy, reproducibility) ⇒ **oracle/soundness test**;
determinism preserved (seeded PCG, fixed order); 8 green gates.
