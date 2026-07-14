<img width="1273" height="671" alt="image" src="https://github.com/user-attachments/assets/c36c292c-5893-44c2-94d9-3895ec0749e8" />






# SciRust 🦀

> A pure-Rust deep learning framework — SIMD CPU kernels, reverse-mode
> autograd, batch normalization, convolutions, and data parallelism.
> (A portable wgpu GEMM is wired behind the optional `wgpu` feature, tested
> against the CPU oracle on software Vulkan — see docs/GPU.md.)
> No C++, no Python, no FFI — just Rust from top to bottom.

## Why?

Existing Rust ML libraries either wrap libtorch (`tch`), wrap ONNX runtime,
or are research toys. SciRust is the middle path: a real framework with
real ops, but written entirely in Rust so you can read every line of compute,
modify it, and trust it.

It's not the fastest framework on the planet. It's the one you can **fully
understand**, modify safely, and extend without crossing language boundaries.

## Positioning

A research artifact: a pure-Rust deep-learning and scientific-computing stack built and
validated from scratch — a runtime plus a transpiler layer — rather than a wrapper
over libtorch or ONNX. The guiding discipline is that every primitive is accepted only
after its output matches a reference oracle, with reproducibility measured rather than
assumed (in several cases bit-for-bit). SciRust is not a production competitor to PyTorch,
Burn, or candle; it is a framework you can read, modify, and trust, with its claims backed
by measurements.

## Validated capabilities

Every result below is reproduced by code in this repository and documented in the
technical report ([`paper/SciRust-technical-report.md`](paper/SciRust-technical-report.md)).

- **Deep-learning core + reverse-mode autodiff** — 1718 passing workspace tests (0 failures; measured 2026-06-19); an MLP reaches 97.70% on MNIST.
- **N-D autograd stack, research-backed and gradient-checked** — a complete causal **decoder LM** (token + positional embeddings, causal attention, fused softmax cross-entropy) that trains end-to-end and overfits a sequence *exactly*; LLaMA-family layers (**RMSNorm**, **SwiGLU**, **RoPE**, **grouped/multi-query attention**); a **LoRA** low-rank adapter (frozen base + trainable `B·A`, gradient-checked); deterministic optimizers (**Adam, AdamW, Lion, Muon, Schedule-Free, AdEMAMix, SOAP** — Adam in Shampoo's eigenbasis, with a from-scratch Jacobi eigensolver — plus **Lookahead, LAMB, Adan**); **exact speculative decoding** and **FlashAttention** (online softmax); a **DeltaNet** delta-rule linear-attention layer, a **Mamba** selective state-space layer (S6 input-dependent scan), a **RetNet** retention layer (recurrent form proven equal to the parallel form), a **GLA** gated-linear-attention layer (data-dependent forget gate), and an **HGRN** gated-linear-RNN token mixer, all linear-time recurrences unrolled on the tape; a **Neural ODE** (backprop through an RK4 solver); a **Physics-Informed Neural Network** that solves a boundary-value problem with the PDE residual in the loss (recovers `sin x` to ~4 decimals). Every op is validated by a finite-difference gradient check. See [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md).
- **Certifiable & reproducible** — **Interval Bound Propagation** and **CROWN** (linear-relaxation back-substitution, provably *tighter* than IBP) give *provable* output bounds and robustness certificates for ReLU MLPs, shown side by side (`scirust certify`); **conformal prediction** gives distribution-free prediction sets with a *guaranteed* coverage level (`scirust conformal`); **temperature scaling** recalibrates over-confident probabilities (lowers the expected calibration error without touching accuracy, `scirust calibrate`); **order-independent floating-point reductions** are bit-identical regardless of thread count; **Wanda** pruning, **SmoothQuant**, **GPTQ** (second-order error-feedback weight quantization — beats round-to-nearest on calibration error, `scirust gptq`), **AWQ** (activation-aware search-based per-channel scaling, `scirust awq`), **BitNet b1.58** (ternary `{−1,0,+1}` weights, ~1.58 bit/weight, with a verified multiplication-free matmul, `scirust bitnet`), and **NF4** (QLoRA's 4-bit NormalFloat — quantile-matched levels that beat uniform 4-bit on Gaussian weights) extend the deterministic int8 path.
- **Optional GPU compute (wgpu and CUDA)** — a real WGSL `f32` GEMM behind the `wgpu` feature, validated against the CPU oracle on a software Vulkan adapter (Mesa lavapipe) in CI and plumbed into the autograd tape and Conv2d's im2col GEMMs. A separate `cuda` feature exposes the current bf16/cuBLASLt backend in `scirust-cuda`; it is dynamically loaded, reports `BackendError::Unavailable` when the CUDA runtime/device is absent, and requires a CUDA-capable host for device execution. A historical run of the CUDA lineage reached ~63 TFLOPS on an NVIDIA Jetson Thor; hosted CI does not reproduce that hardware measurement — see `scirust_complete_audit_report.md` §5.
- **Deterministic inference runtime** — bit-exact forward (a 64-bit output fingerprint identical across thread counts and processes), bounded latency (p99/p50 ~1.15), and architecture-agnostic reconstruction from a plain-text manifest plus an SRT1 weight file.
- **Certified-deterministic multi-thread training** — `DataParallelTrainer::train_batch_threaded` runs workers across N OS threads yet reduces gradients in a fixed worker order, so the aggregate is **bit-identical for 1/2/4/8 threads** and equal to the sequential path (float addition isn't associative — the reduction order is pinned). CI-tested, including a real autograd backward. To our knowledge, SciRust is the only **self-contained** DL framework (100 % auditable Rust stack, zero FFI in the compute path) that simultaneously offers CI-tested bit-identical multi-thread training (1/2/4/8 threads == sequential), a deterministic int8 pipeline for embedded targets, and audit artifacts (inference fingerprints, hash-chained logs, manifest-based reconstruction). Closest related work: RepDL (Microsoft, 2025, [arXiv:2510.09180](https://arxiv.org/abs/2510.09180)) provides **cross-platform** bitwise reproducibility for a float32 subset of PyTorch via correctly-rounded operations — a stronger guarantee on that axis for f32, but as an overlay on a C++/Python TCB, without low-precision support and without audit artifacts. SciRust's integer and fixed-point paths are bit-exact cross-platform; its sanitized f32 path is deterministic within a single architecture.
- **Deterministic int8 quantization for embedded** — weight-only int8 is lossless and 4x smaller; a fully-integer calibrated pipeline reproduces the float model bit-for-bit; a true integer convolution and a portable QSR1 / QModel artifact; an aarch64 NEON int8 kernel ~10x faster and bit-exact against the scalar reference; separable depthwise + pointwise convolutions in deterministic int8.
- **Symbolic regression** — a hybrid genetic-gradient engine recovers closed-form laws (structure and constants) from data, fitting constants with the framework's own symbolic differentiation.
- **Evolutionary optimization** (`scirust-evo`) — NSGA-II recovers the ZDT1 Pareto front to within ~1e-3; the simplified single-objective optimizers are honest about their limits (see the report).
- **Industrial & automotive monitoring** (`scirust-signal`, `scirust-opcua`, `scirust-mqtt`, `scirust-pdm`, `scirust-mlops`, `scirust-func-safety`, `scirust-integration`) — signal processing (FFT, windows, bearing BPFO/BPFI/BSF diagnostics, order analysis, **radar** — pulse-compression (LFM chirp / Barker codes, plus **polyphase / CAZAC codes** — Frank, P3, P4 and Zadoff-Chu — the perfect-periodic-autocorrelation and low-probability-of-intercept (LPI) waveforms that exist beyond the length-13 Barker limit, with the exact periodic-autocorrelation-is-an-impulse oracle Barker cannot meet; matched filtering with exact energy/peak-to-sidelobe oracles), the **ambiguity function** (joint delay-Doppler response with the LFM range-Doppler coupling ridge), **Doppler processing** (range-Doppler map — stationary vs. moving targets separated into zero- vs. non-zero-Doppler bins), **MTI clutter cancellers** (binomial pulse cancellers with an exact DC null), **CFAR detection** (cell-averaging & ordered-statistic, the latter robust to interfering targets that mask CA-CFAR; false-alarm rate validated statistically — plus a **2-D CFAR** over the range-Doppler map with **connected-component clustering** of detections into amplitude-weighted target centroids, then **target tracking** — α–β constant-velocity track filters and a nearest-neighbour multi-target tracker with track birth/death, plus a full constant-velocity **Kalman filter** (adaptive gain, live covariance, innovation likelihood) and an **IMM** (Interacting Multiple Model) estimator that blends a quiet and an agile model by Markov mode probabilities to follow manoeuvring targets with less lag than any fixed filter, a **planar coordinated-turn IMM** — a general n-state linear Kalman filter (Cholesky measurement update) blending constant-velocity and constant-turn-rate models to track turning targets in the (x, y) plane, following the arc instead of overshooting it — an **extended Kalman filter** that tracks a Cartesian state directly from raw **polar range/bearing** radar measurements (nonlinear observation linearised by its Jacobian, azimuth innovation wrapped across the ±π boundary), a **multi-target tracker** of per-target EKFs associated by a **statistical NIS (normalised-innovation-squared) validation gate** — a χ²-thresholded Mahalanobis distance that adapts to each track's own uncertainty, with track birth/death — and a **probabilistic data association filter (PDAF)** that tracks through clutter by soft-combining every gated measurement weighted by its association probability (with an innovation-spread covariance term) instead of a brittle hard nearest-neighbour pick — closing the OpenRadar detection→track pipeline), **micro-Doppler analysis** (a Hann-windowed spectrogram of the slow-time return plus ridge / bulk-Doppler / bandwidth / cadence descriptors — the time-frequency micro-motion signature behind non-cooperative target recognition, recovering a rotating scatterer's body Doppler and rotation cadence from a synthetic oracle), **detection statistics** (Swerling I probability of detection `P_fa^{1/(1+SNR)}` and Albersheim's equation for the steady-target case — both `P_d`↔SNR directions — quantifying the fluctuation loss that complements the CFAR threshold), the **radar range equation** (a monostatic link budget — `1/R⁴` received power, `k_B·T·B·F` noise, delivered SNR versus RCS/range, and the maximum detection range that closes with the Swerling required SNR — the radar analog of the EO/IR range budget), **clutter amplitude statistics** (Rayleigh, Weibull and log-normal distributions — the spiky-clutter models CFAR thresholds are designed against, with a self-contained error function; Weibull with shape 2 recovers Rayleigh), and **PRF ambiguities & blind speeds** (unambiguous range `c/(2·PRF)` and velocity `λ·PRF/4`, MTI blind speeds `n·λ·PRF/2`, and range/velocity folding — with the invariant range–velocity ambiguity product `cλ/8`, the pulse-Doppler dilemma), and **stepped-frequency synthetic wideband ranging** (fine range resolution `c/(2·N·Δf)` synthesised by an inverse DFT of a burst of frequency-stepped narrowband pulses — wideband resolution without wideband hardware, as in ISAR/GPR), and **space-time adaptive processing (STAP)** for airborne radar — a joint angle-Doppler adaptive filter `w = R⁻¹s/(sᴴR⁻¹s)` that nulls the ground-clutter ridge `f_d = β·f_s` (clutter collapses onto a 1-D ridge in the 2-D angle-Doppler plane, so a slow mover buried in clutter Doppler but off the ridge is separable), delivering `SINR = P·sᴴR⁻¹s` — deeply notched on the ridge, near the full `NM` coherent gain off it — the flagship technique for detecting slow movers no angle-only or Doppler-only filter can pull out of clutter, and **synthetic-aperture radar (SAR) azimuth compression** — the imaging mode, focusing a point target's quadratic slow-time phase history (an azimuth chirp `exp(−j·2π·(x−x₀)²/(λR))`) by matched filtering to synthesise a `λR/D` aperture, giving the celebrated range-independent `D/2` cross-range resolution (finer for a *smaller* antenna), with the closed-form aperture / Doppler-bandwidth `2v/D` / chirp-rate `2v²/(λR)` relations and point-target focusing and two-target resolution demonstrated against oracles, **array processing** (ULA steering vectors, delay-and-sum beamforming, DOA estimation, high-resolution **MVDR/Capon** — resolving two sources closer than the array beamwidth where the conventional beamformer merges them, via a from-scratch complex Hermitian inverse — **MUSIC** subspace direction finding, built on a from-scratch complex-Hermitian eigensolver (cyclic Jacobi), gridless **ESPRIT** — reading the angles straight off the eigenvalues of the subarray-subspace rotation, resolving off-grid sources without a spectral scan, via a from-scratch complex eigensolver (Hessenberg reduction + shifted QR) — and **amplitude-comparison monopulse** — single-dwell sum/difference angle estimation with the exact `tanh(θ·θ_s/σ²)` discriminator, giving off-boresight angle accuracy far finer than the beamwidth — and **phase-comparison (interferometric) angle estimation** — off-boresight angle read from the phase difference `Δφ = 2π·d·sin θ/λ` across a two-element baseline (`θ = arcsin(Δφ·λ/(2π·d))`), with the resolution vs. unambiguous-field-of-view `arcsin(λ/2d)` trade-off that governs baseline choice, and the phase wrap/alias behaviour a wide baseline induces) and **FMCW / mmWave processing** (beat-frequency ranging, range resolution, and the range-Doppler cube built from raw beat chirps by two FFTs — fast-time range and slow-time Doppler — the automotive/OpenRadar model), and an extensible **denoising & noise-detection** toolkit — linear/rank/transform/variational/adaptive families behind a `Denoiser` trait (Savitzky-Golay, Hampel, Wiener, Haar/Daubechies-4/6/8 wavelet shrinkage with universal or per-level SURE thresholds, spectral subtraction, iterative and **exact** (Condat, KKT-certified) total variation, Kalman RTS smoothers — local-level auto-tuned by innovation whiteness, and local linear-trend — plus LMS/RLS adaptive line enhancers), a noise characterizer that classifies Gaussian/impulsive/periodic/colored/baseline disturbances, and a detect-then-denoise auto pipeline with a residual whiteness self-check that flags when information leaks into the "noise"), OPC-UA PLC connectivity with 8 simulated sensors, MQTT SparkPlug B publishing, predictive maintenance (Health Index, RUL, CUSUM, 4 fault detectors), industrial MLOps (drift, shadow deploy, signed OTA), ISO 26262 functional safety (ASIL A-D, fault injection, degraded mode, hash-chained audit log), integration kit (unified Backend/Pipeline/Config/templates), and a dedicated `scirust-industrial` CLI (discover, test, gen-config, scaffold, run, doctor — plus the vertical demos below).
- **State estimation, navigation, water & OT security** (`scirust-estimation`, `scirust-nav`, `scirust-water`, `scirust-ids`, `scirust-func-safety`) — Kalman / IMM / **UD square-root** filters (covariance positive-semidefinite by construction), **GNSS/INS fusion** and **TDOA** multilateration, acoustic **leak correlation** and water-hammer physics (Joukowsky/Korteweg), OT **firmware attestation** and **PLC ladder integrity** (Stuxnet write-set detection) on a tamper-evident hash chain, and a **GMP golden-batch** comparator (DTW alignment + hash-chained 21 CFR Part 11 audit). All reachable from the `scirust-industrial` CLI: `nav-tdoa`, `nav-fusion`, `track-imm`, `track-ud`, `water-leak`, `water-surge`, `ot-firmware`, `ot-plc`, `golden-batch`. Validated on x86 and natively on a **Jetson AGX Thor** via [`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md).
- **Process safety / Safety Instrumented Systems** (`scirust-reliability`, `scirust-sis`) — IEC 61511/61508 `PFDavg`/`PFH`/SIL for the full 1oo1/1oo2/2oo2/2oo3/1oo3 MooN voting family (validated against a published NTNU worked example, not just hand derivation), a full SIF loop model (sensors → logic solver → final elements, summed `PFDavg`), fault injection showing e.g. that 2oo3 tolerates one failed channel while 2oo2 does not, deterministic cause-and-effect matrices, and proof-test-interval sizing by numerically inverting `PFDavg` (reusing `scirust-solvers::roots::bisection`) — the direct answer to Triton/Trisis-style unauditable SIS logic, exposed as MCP tools (`sis_verify_sif_loop`, `sis_size_proof_test_interval`).
- **Pattern detection** (`scirust-vision`, `scirust-audio`, `scirust-graph`, `scirust-sequential`, `scirust-multivariate`, `scirust-unsupervised`, `scirust-seasonal`, `scirust-nlp-advanced`) — computer vision (CNN, HOG, LBP, Canny, Otsu, NMS, plus **optronics / precision-optics imaging** — Gaussian **and diffraction-limited Airy PSFs** (with the Rayleigh resolution limit `1.22·λ/D`), MTF with the MTF50 resolution metric, **Richardson–Lucy (spatial) and Wiener (frequency-domain) deconvolution** for EO/IR image restoration, ray optics: ABCD ray-transfer matrices with Gaussian-beam propagation via the complex q-parameter for optical-train design, **image-domain small-target CFAR detection** — the EO/IR analogue of the radar CFAR, pulling small hot targets out of a spatially varying thermal background (local guard/training-ring statistics, intensity-weighted sub-pixel centroids) to feed the tracking chain from an IR focal plane, and **thermal-infrared radiometry & sensor sensitivity** — Planck's law, Stefan–Boltzmann `σT⁴` and Wien's peak-wavelength law, band-integrated thermal contrast, the **NETD** (noise-equivalent temperature difference) and **MRTD** (minimum resolvable temperature difference, the NETD-over-MTF trade-off) metrics that spec a thermal imager, and **Beer–Lambert atmospheric transmission & the EO/IR range budget** — path transmittance `e^{−αR}`, extinction from meteorological visibility (Koschmieder), the contrast-transmission law, and the target ΔT required to detect through an attenuating path at range), audio (MFCC, chroma, pitch YIN, onset detection), graph patterns (subgraph isomorphism, motif discovery, community detection, betweenness), sequential (HMM, CRF, DTW, KMP/Boyer-Moore), multivariate (PCA, ICA, K-Means++, MDS, CCA), unsupervised (autoencoder, isolation forest, DBSCAN, LOF, GMM), seasonal (STL, ACF/PACF, Mann-Kendall, CUSUM), NLP (NER, LDA, TextRank, MinHash, NaiveBayes).
- **Algorithm creation** (`scirust-automl`, `scirust-synthesis`, `scirust-algogen`, `scirust-codetrans`, `scirust-rl-algo`, `scirust-scaffold`) — AutoML (Bayesian optimization, GP surrogate, model selection, ensembles), program synthesis (30+ ops, sketch-based, bottom-up/top-down/GP/beam search), algorithm generation (sort/search/graph/DP/DaC, complexity analysis), code transformation (AST, pattern matching, 20 optimization rules, refactoring, Rust→Python/C transpilation), RL-based discovery (REINFORCE, Actor-Critic, Q-Learning, MCTS, meta-learning), scaffolding (DSL, code gen, 16 templates, docs).
- **General-purpose linear algebra & optimization** (`scirust-solvers`) — beyond LU/QR/Cholesky/conjugate-gradient: a general dense **symmetric eigendecomposition** (Householder + implicit QL, Wilkinson shift), a general dense **SVD** (one-sided Jacobi), a **randomized SVD** (Halko-Martinsson-Tropp, seeded `SplitMix64` for bit-reproducible projections), restarted **GMRES** and **BiCGSTAB** for nonsymmetric matrix-free systems with a Jacobi preconditioner, **Anderson acceleration** for fixed-point iterations, and a bound-constrained **spectral projected gradient** optimizer — all deterministic (fixed iteration budgets, sequential orthogonalization), all from scratch.
- **Agent connectivity & safe OT/IT discovery** (`scirust-mcp`, `scirust-discovery`) — a [Model Context Protocol](https://modelcontextprotocol.io) server exposing SciRust's solvers, dev tools, discovery, the regulated-industry vertical primitives, and now the **simulation environments** (`sim_epidemic` runs an SIR outbreak; `sim_battery_discharge` runs the Thévenin 1-RC cell plant; `sim_grid_stability` analyzes the swing-equation operating point and transient; `sim_hvac_zone` runs the 2R2C building-zone thermal model; `sim_pharmacokinetics_oral` runs the oral one-compartment drug model with its C_max/t_max/AUC oracles; `sim_stiff_robertson` integrates the canonical stiff Robertson kinetics with the implicit Rosenbrock solver where an explicit method blows up) as standard MCP tools callable by any agent (the in-house `scirust-sciagent` SLM, Claude, ChatGPT, a script) with a SHA-256 hash-chained audit log per call; and consent-scoped, protocol-native OT/IT asset discovery (OPC-UA UACP handshake, Modbus Read Device Identification, mDNS/DNS-SD, BACnet/IP Who-Is/I-Am, SNMPv1 sysDescr, EtherNet/IP ListIdentity — never a generic port scan, following the IEC 62443 zone/conduit model and NIST SP 800-82 doctrine) so an agent can find what industrial hardware is actually reachable before driving it. See [`docs/DOMAIN_ROADMAP.md`](docs/DOMAIN_ROADMAP.md) for the researched regulated-industry verticals this connector layer is meant to unlock.
- **Regulated-industry vertical primitives** (`scirust-grid`, `scirust-biomed`, `scirust-maritime`, `scirust-fab`, `scirust-agtech`, `scirust-fatigue`, `scirust-tolerance`, `scirust-sis`) — one primitive per researched domain in `docs/DOMAIN_ROADMAP.md`, each with a worked-example or reference-implementation verification and an honest "not delivered" boundary rather than a guessed formula: power-grid **WLS state estimation** with bad-data detection and mho-characteristic **distance-relay** logic (`scirust-grid`); a closed-loop dosing **PID + insulin-on-board + Control-Barrier-Function safety filter** stack, explicitly not a clinical device (`scirust-biomed::control`) — whose glucose plant (`dG/dt = -a·(G-G_b)-k·u`) also implements `scirust_sim::System` behind an optional `sim` feature, so the shared simulation engine integrates the vertical's own model directly (the "reverse direction" of the sim layer, verified against the closed-form `G(t) = G* + (G0-G*)·e^{-a·t}`); **COLREG encounter classification**, **CPA/TCPA** collision-risk assessment, and weighted-pseudo-inverse **thrust allocation** for dynamic positioning (`scirust-maritime`); **EWMA run-to-run** recipe control and **PCA-based T²/SPE** fault detection (`scirust-fab`); a reproducible **yield-map** cleaning pipeline (global+local outlier filters, IDW) and the verified ISO 25119-2 risk-parameter model (`scirust-agtech`); **ASTM E1049 rainflow** cycle counting (ported and verified against an independent reference implementation) plus **Palmgren-Miner** damage (`scirust-fatigue`); **inertial tolerancing** (Pillet) — inertia `I=√(δ²+σ²)`, the inertial capability index `Cpi` alongside `Cpm/Cpk/Pp`, 1D tolerance-chain analysis & allocation (worst-case / statistical / weighted / guaranteed-`Cpk` / cost-optimal, cross-checked against arXiv:1002.0270 Table 2), the inertial piloting chart, acceptance sampling via the non-central-χ² law, lot mixing, **surface/modal form tolerancing** (surface inertia as the RMS of point inertias, DCT modal decomposition with the `Σ Iₖ²=m·I_S²` partition identity, arXiv:1002.0251), **3D small-displacement-torsor tolerancing** (normal deviation `e=T·n+R·(OM×n)`, best-fit torsor + form residual, surface inertia `I_S²=θ̄ᵀHθ̄+tr(HΣ)` as the statistical combination of location and orientation, arXiv:1002.0253), **minimum-cost synthesis under several requirements at once** (convex Lagrangian dual, KKT-verified, + cost–quality Pareto frontier), and the measurement-and-analysis layer competitors ship around these methods — **crossed Gage R&R by ANOVA** (AIAG MSA: EV/AV/PV variance components, %study-R&R, ndc, 10 %/30 % verdict), **statistical tolerance intervals** (ISO 16269-6 two-sided Howe and one-sided Natrella factors), **GeoFactor/dual sensitivity** splitting each contributor's pull on the assembly *mean* from its pull on the *variance* (as 3DCS/CETOL do), **ISO 22514-2 distribution fitting** (Normal/Lognormal/Rayleigh/Weibull best-fit + percentile capability), **advanced GD&T** (virtual/resultant condition, datum shift from MMB, two-tier composite position), and **exact Cp / large-sample Cpk confidence intervals**, plus the process-quality reporting layer around them — **variables acceptance sampling** (ISO 3951 / MIL-STD-414 Form-`k`: two-point `(n, k)` design from AQL/RQL and the closed-form OC `Φ(√n(z_p−k))`, known- and unknown-`σ` methods), **Six-Sigma yield accounting** (DPMO/DPU, throughput and rolled-throughput yield, yield↔sigma-level↔DPMO with the 1.5σ shift), and **data-driven root-cause attribution** (a least-squares variance-transmission decomposition of measured assembly variation onto co-measured components, with fitted sensitivities, the exact `Σcⱼ=R²` share identity, and the unexplained remainder that flags an unmodeled cause), plus the go/no-go and reliability primitives that round out a shop-floor quality toolkit — **attributes acceptance sampling** (ISO 2859-1: binomial OC `P(D≤c)` and two-point `(n,c)` design), **stress-strength interference** for assembly-fit reliability (`R=P(strength>stress)=Φ((μ_S−μ_L)/√(σ_S²+σ_L²))`, the clearance-fit probability a random hole/shaft pair assembles — which a worst-case min/max stack cannot give), and a **rational-subgroup capability study** (AIAG/ISO 22514-2: within-subgroup `σ̂=R̄/d₂=s̄/c₄` driving short-term `Cp/Cpk` vs the overall spread driving long-term `Pp/Ppk`, measured from a designed study rather than assumed as a 1.5σ rule); and the drawing-and-economics layer around it — **ISO 286 limits & fits** (standard tolerance grades `ITn` from the factor `i=0.45∛D+0.001D`, shaft fundamental deviations `d–h`, and clearance/transition/interference classification of a hole/shaft pairing), **double and sequential (Wald SPRT) acceptance sampling** (smaller average sample for the same risks, with OC/ASN and the straight accept/reject boundary lines), and the **Taguchi quadratic loss** (the `E[L]=k·I²` identity that makes inertial tolerancing a direct minimiser of quality loss, plus the economic manufacturing tolerance `Δ=Δ₀√(A/A₀)`) (`scirust-tolerance`); and channel-bypass reconfiguration for nuclear reactor-trip **MooN voting** (`scirust-sis::reactor_trip`). Each is wired into `scirust-mcp` as a callable tool.
- **Multi-domain simulation environments** (`scirust-sim`) — the unified "here is a system, step it through time, let an agent interact with it" layer the platform previously lacked: a deterministic time-stepping engine (a `System` trait `y' = f(t, y)` with both fixed-step RK4 **and an error-controlled adaptive Dormand–Prince 5(4)** integrator — the `ode45` scheme, choosing the step automatically so it reproduces `e^{-t}` to 1e-8 in <300 steps where fixed RK4 would need ~1700 — plus a `SecondOrderSystem` trait + **symplectic** semi-implicit Euler, the tests showing the two-body orbit staying closed at a step size where explicit Euler visibly spirals out), a gym-style **`Environment` trait** (`reset` / `step(action) → observation, reward, done`) with the classic **cart-pole** (reference physics constants, seeded bit-replayable episodes) and a deterministic **grid world** — and, behind the optional **`rl` feature**, an `RlEnv` adapter that presents these as a `scirust-learning::rl::Env` so the existing tabular/PPO/deep agents train on them unchanged (a tabular Q-learner provably finds the grid-world shortest path); a seeded **SplitMix64** generator validated against the published reference vectors, and oracle-tested ready-made models in sixteen domains: **mechanics** (spring–mass–damper vs the underdamped closed form, full nonlinear pendulum with energy conservation at large amplitude, projectile with linear drag vs its exact solution, and the **chaotic double pendulum** — energy conserved to 1e-6 along a chaotic orbit, with a demonstrated sensitive dependence on initial conditions where a 1e-8 perturbation grows to O(1)), **orbital** (two-body Kepler: energy and angular momentum conserved to 1e-9, circular orbit closing after exactly one Kepler period), **epidemiology** (SIR/SEIR: population conserved to round-off, epidemic threshold at R₀ = 1, the exact transcendental final-size relation), **ecology** (Lotka–Volterra first integral conserved, logistic closed form), **chemistry** (consecutive reactions vs the Bateman solution, reversible reaction relaxing to K = k_f/k_r, and the canonical **stiff** Robertson benchmark integrated — via the optional `stiff` feature's bridge to `scirust-stiff` — against its reference solution where explicit RK4 blows up), **thermal** (Newton cooling, 1-D heat rod validated on the *discrete* eigenmode decay rate and the maximum principle), **electrical** (RC charging, series RLC vs the damped-oscillator closed form + passivity, and the nonlinear **Van der Pol** oscillator — the library's first *limit cycle*, where trajectories from inside and outside converge onto the same self-sustaining periodic orbit of amplitude ≈ 2), **stochastic/queueing** (GBM and Ornstein–Uhlenbeck sampled by their *exact* transition laws, an M/M/1 discrete-event queue matching L = ρ/(1−ρ), W = 1/(μ−λ) and Little's law), **pharmacokinetics** (oral one-compartment Bateman absorption and IV two-compartment biexponential disposition, the adaptive-integrated AUC recovering the exact `dose/k₁₀`), **rigid-body dynamics** (torque-free Euler equations: kinetic energy and |L|² conserved, symmetric-top precession in closed form, and the intermediate-axis / Dzhanibekov instability), and — turning the industrial verticals' physics into runnable plants — a **Thévenin 1-RC battery** with self-heating (`scirust-bms`: exact coulomb counting, RC/thermal steady states), a **2R2C building-zone** thermal model (`scirust-hvac`: exact linear steady state), and the synchronous-machine **swing equation** (`scirust-grid`: equilibrium `asin(P_m/P_max)`, small-signal frequency, conserved transient energy), plus — the optoelectronics pillar of the radar/optronics track — the single-mode **semiconductor-laser rate equations** (carrier/photon densities: the lasing threshold, the linear light–current law `s = Γ·τ_p·(J − J_th)`, and relaxation-oscillation ringing measured against `f_r = √(g₀·s_ss/τ_p)/2π`) and its detector counterparts, a **photodiode** (optical power → photocurrent → RC-limited voltage: the spectral responsivity `η·q·λ/(h·c)`, the −3 dB bandwidth `1/(2π·R_L·C_j)`, and the exponential step response) and a high-sensitivity **avalanche photodiode** (the McIntyre excess-noise factor `F(M) = k·M + (1−k)(2−1/M)` and the shot-vs-thermal SNR trade-off with its optimal avalanche gain — the lidar/rangefinder receiver). Pure Rust, zero dependencies, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`, explicit seeds everywhere (bit-reproducible runs), Miri-gated in CI — 118 tests + doctests by default, more behind the optional `rl` (RL agents) and `stiff` (stiff plants) features.

## What's in it?

```
✓ Reverse-mode autograd        ✓ Conv2d / MaxPool2d / BatchNorm1d
✓ SIMD CPU kernels (AVX2/SSE2/NEON) ✓ Deterministic int8 quantization
✓ Adam / SGD optimizers        ✓ Data parallelism (1 tape per thread)
✓ Lazy graph compilation       ✓ MNIST IDX reader + DataLoader
✓ safetensors persistence      ✓ Pure Rust, no FFI
```

## Quick start (60 seconds)

No code to copy. Install the unified `scirust` CLI and run a command:

```bash
git clone https://github.com/Memorithm/scirust && cd scirust
cargo install --path scirust-cli      # provides the `scirust` binary

scirust help                          # list every command, grouped
scirust info                          # capabilities & determinism guarantees
scirust quickstart                    # train a demo classifier (deterministic) → 4/4
scirust lm                            # train a tiny causal decoder LM (N-D tape) → exact recall
scirust lm --opt lion                 # …with a different deterministic optimizer (adam|adamw|lion)
scirust certify --eps 0.02            # prove a ReLU MLP's output bounds over an L∞ box (IBP)
scirust conformal --alpha 0.1         # conformal intervals with a guaranteed coverage level
scirust som train                     # train the ownership model; accuracy vs baseline
scirust evo                           # minimize a function with a seeded genetic algorithm
scirust diff "x^2 + 3*x"              # symbolic derivative → ((2 * x) + 3)
scirust solve "x^2 - 4"               # real roots → { -2, 2 }
scirust integrate "sin(x)" 0 3.14159  # definite integral (Romberg) → 2
scirust linsolve "2,1;1,3" "3,5"      # solve A·x = b → [0.8, 1.4]
scirust ode "y" 1 0 1                 # dy/dt=y, y(0)=1 → y(1) ≈ e
scirust eval "2*x + 1" x=3            # evaluate → 7
scirust analyze src/main.rs           # ownership analysis of a real Rust file
scirust analyze src/main.rs --sarif   # same, as SARIF 2.1.0 for CI code scanning
scirust verify emit  model.qsr1 model.proof    # seal an inference certificate
scirust verify verify model.proof model.qsr1   # re-check it bit-for-bit

# Industrial monitoring (scirust-industrial CLI)
scirust-industrial discover --simulated                  # browse available PLC sensors
scirust-industrial gen-config --template automotive --stations 3
scirust-industrial scaffold --name line3-monitor --template automotive
scirust-industrial run --config config.json --cycles 100
scirust-industrial doctor --config config.json           # diagnose integration issues

# Vertical demos (deterministic, real crate APIs)
scirust-industrial nav-tdoa                 # TDOA: locate an emitter from arrival-time differences
scirust-industrial nav-fusion --outage 10   # GNSS/INS fusion through a 10-step GNSS outage
scirust-industrial track-imm                # IMM filter swings onto the maneuver model
scirust-industrial track-ud                 # UD square-root Kalman ≡ textbook Kalman, covariance PSD
scirust-industrial water-leak               # acoustic leak correlation (locate a known leak)
scirust-industrial water-surge              # Joukowsky surge + Korteweg wave speed
scirust-industrial ot-firmware              # firmware attestation: clean vs tampered image
scirust-industrial ot-plc                   # PLC integrity + Stuxnet critical-write detection
scirust-industrial golden-batch             # GMP golden-batch comparator (DTW + audit chain)

# MCP server — connect any agent (the in-house SLM, Claude, ChatGPT, a script) to SciRust
cargo run -p scirust-mcp --bin scirust-mcp  # JSON-RPC 2.0 over stdio, see scirust-mcp/README.md
```

`scirust quickstart` prints a decreasing loss and reaches 4/4 on a
non-linearly-separable task — proof the autograd tape, Adam, and the layers
work together. Same seed ⇒ identical numbers, every run.

No `cargo install`? Run any command in place with
`cargo run -p scirust-cli -- <command>`.

## Library API (for embedding)

The CLI is a thin layer over the crates; embed them directly when you need
full control. The 2→8→2 classifier the quickstart trains, in code:

```rust
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

let mut rng = PcgEngine::new(42);
let mut model = Sequential::new()
    .add(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
    .add(ReLU::new())
    .add(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));
let loss_fn = CrossEntropyLoss::new();
let mut opt = Adam::new(0.05);

let tape = Tape::new();
let x = tape.input(Tensor::from_vec(vec![0.0, 1.0], 1, 2));
let y = tape.input(Tensor::from_vec(vec![0.0, 1.0], 1, 2)); // one-hot
let logits = model.forward(&tape, x);
let loss = loss_fn.forward(&tape, logits, y);
tape.backward(loss.idx());
opt.step(&model.parameter_indices(), &tape);
model.sync(&tape);
```

Add a single crate to your own `Cargo.toml`:

```toml
[dependencies]
scirust-core = { path = "path/to/scirust-core" }
```

> GPU note: `scirust-gpu` ships a deterministic CPU reference backend (always
> built; the bit-tolerant oracle) and a real portable **wgpu GEMM** behind the
> optional `wgpu` feature, validated against that oracle on a software Vulkan
> adapter (Mesa lavapipe) in CI. Without the feature/adapter the `wgpu` path
> returns `BackendError::Unavailable` — never fabricated output. It is also
> plumbed into the autograd tape (`WgpuEngine` + `Var::matmul_gpu`) and into
> Conv2d's im2col GEMMs, forward and backward, validated end-to-end on lavapipe.
> The optional `cuda` feature similarly uses the real bf16/cuBLASLt backend. It
> compiles without a local toolkit through dynamic loading and returns
> `BackendError::Unavailable` when its runtime or device is absent; device parity
> still requires a CUDA-capable runner. Next: keep activations in VRAM across
> layers — see `docs/GPU.md` (P2.2).

## Architecture

```
scirust-core/          Core compute, autograd, layers (~12k loc)
scirust-simd/          SIMD CPU kernels (AVX2, SSE2, NEON)
scirust-gpu/           CPU reference + real wgpu/CUDA dispatch (optional features)
scirust-signal/        Signal processing: FFT, windows, bearing diagnostics, order analysis
scirust-opcua/         OPC-UA connector: trait + 8 simulated industrial sensors
scirust-mqtt/          MQTT publishing: SparkPlug B payloads, severity classification
scirust-pdm/           Predictive maintenance: Health Index, RUL, CUSUM, fault detectors
scirust-mlops/         Industrial MLOps: drift detection, shadow deployment, signed OTA
scirust-func-safety/   Functional safety: ASIL A-D, fault injection, degraded mode, audit log
scirust-integration/   Integration kit: Backend, Pipeline, config, code templates
scirust-som/           Ownership Model: real-Rust analyzer + Transformer pipeline
scirust-mcp/           Model Context Protocol server: exposes solvers/tools/discovery to any agent
scirust-discovery/     Safe OT/IT asset discovery: OPC-UA/Modbus/mDNS/BACnet/SNMP/EtherNet-IP, signed scope, audit log
scirust-reliability/   IEC 61508 PFDavg/PFH/SIL for the MooN voting family (incl. general M-of-N)
scirust-sis/           IEC 61511 Safety Instrumented Systems: SIF loops, cause-and-effect, reactor-trip bypass, audit log
scirust-grid/          Power-grid analytics: frequency/RoCoF/synchrophasors, WLS state estimation, distance relay
scirust-biomed/        ECG signal analysis + closed-loop dosing control primitives (PID, IOB, CBF-QP safety filter)
scirust-maritime/      COLREG encounter classification, CPA/TCPA collision risk, DP thrust allocation
scirust-fab/           Semiconductor-fab process control: EWMA run-to-run, PCA-based T²/SPE fault detection
scirust-agtech/        Precision-agriculture yield-map cleaning pipeline + ISO 25119-2 risk-parameter model
scirust-fatigue/       Structural fatigue: ASTM E1049 rainflow counting, Palmgren-Miner damage
scirust-tolerance/     Inertial tolerancing (Pillet): inertia I=√(δ²+σ²), Cpi/Cpm/Cpk, chain allocation, piloting chart, acceptance sampling, lot mixing, surface/modal + 3D-torsor form tolerancing, multi-requirement cost optimization, non-normal (Cornish-Fisher/Clements) + GD&T position/ISO-1101 tolerancing, Monte-Carlo simulation, correlated/nonlinear chains, sensitivity ranking, discrete-process allocation, short/long-term drift, Gage R&R/MSA (ANOVA), statistical tolerance intervals (ISO 16269-6), GeoFactor/dual mean-vs-variance sensitivity, ISO 22514-2 distribution fitting + percentile capability, advanced GD&T (virtual/resultant condition, datum shift, composite position), Cp/Cpk confidence intervals, variables acceptance sampling (ISO 3951/MIL-STD-414), Six-Sigma DPMO/rolled-throughput-yield/sigma-level reporting, data-driven root-cause attribution (least-squares variance transmission), attributes acceptance sampling (ISO 2859-1 binomial OC), stress-strength interference / assembly-fit reliability, rational-subgroup capability study (within Cp/Cpk vs overall Pp/Ppk via d₂/c₄), ISO 286 limits & fits (IT grades, shaft deviations d–h, clearance/transition/interference classification), double + sequential (Wald SPRT) acceptance sampling, Taguchi quadratic loss / cost-of-non-quality (E[L]=k·I², economic tolerance)
examples/              Quickstart, MNIST training, industrial_monitor, benchmarks
```

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — Train a 2-class classifier in 50 lines
- [`docs/MNIST.md`](docs/MNIST.md) — Real MNIST training with data parallelism
- [`docs/GPU.md`](docs/GPU.md) — Portable wgpu compute (status, testing, roadmap)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — How the autograd tape works
- [`docs/REFERENCE.md`](docs/REFERENCE.md) — Exhaustive command/binary/API reference
- [`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md) — Functional acceptance protocol: `scripts/test-protocol.sh` runs every gate, every crate's oracle tests, and a cross-process determinism check in one command
- [`docs/sbom/`](docs/sbom/) — CycloneDX SBOM (reproducible, regenerated in CI & attached to releases)
- [`docs/GROWTH_PLAN.md`](docs/GROWTH_PLAN.md) — Vision, non-negotiable fundamentals, and the ambitious growth plan
- [`docs/RELEASING.md`](docs/RELEASING.md) — Release process & branch-protection runbook
- [`SECURITY.md`](SECURITY.md) — Supply-chain posture, SBOM, accepted advisories
- [`scirust-som/README.md`](scirust-som/README.md) — Ownership Model (real-Rust analyzer)
- [`scirust-mcp/README.md`](scirust-mcp/README.md) — MCP server: exposed tools, audit log, how to connect an agent
- [`scirust-discovery/README.md`](scirust-discovery/README.md) — Safe OT/IT discovery: protocol doctrine, scope authorization, sources
- [`scirust-sis/README.md`](scirust-sis/README.md) — IEC 61511 SIS: voting architectures, SIF loops, Triton/Trisis motivation, sources
- [`docs/DOMAIN_ROADMAP.md`](docs/DOMAIN_ROADMAP.md) — Researched regulated-industry verticals where determinism/auditability is a documented differentiator

## Status

| Feature | Status |
|---|---|
| MLP training | ✅ Stable |
| CNN (Conv2d + MaxPool) | ✅ Stable |
| BatchNorm | ✅ Stable |
| Dropout | ✅ Stable |
| Data parallelism (CPU multithread) | ✅ Stable |
| Transformer (MHA, Encoder, Decoder) | ✅ Stable |
| GQA & KV-Cache | ✅ Stable (GQA + `infer_step` with cache, 6 tests) |
| RoPE embeddings | ✅ Stable |
| RNN / LSTM | ✅ Stable (`nn/lstm.rs`, `forward_sequence`, 7 tests) |
| Flash Attention | ✅ Stable (`nn/transformer/flash_attention.rs`, 4 tests vs dense-attention oracle) |
| Conv2dTranspose | ✅ Stable (`nn/conv2d_transpose.rs`, 7 tests) |
| Mixed precision (fp16) | ✅ Stable (`autodiff/mixed_precision.rs`, 3 tests) |
| Checkpointing (save/resume training) | ✅ New |
| DataLoader (batching, shuffle, prefetch) | ✅ New |
| ONNX export | ✅ New |
| Automatic Mixed Precision (AMP) | ✅ New |
| Differential Privacy (DP-SGD) | ✅ New |
| Model pruning (magnitude, structured, LTH) | ✅ New |
| Distributed training (all-reduce) | ✅ New |
| TensorBoard / CSV logging | ✅ New |
| Neural Architecture Search (NAS) | ✅ New |
| Advanced optimizers (RMSprop, AdamW, LAMB) | ✅ New |
| Fused ops (matmul+SiLU, matmul+GELU, etc.) | ✅ New |
| HPC im2col (cache-aware) | ✅ New |
| SOM — real-Rust ownership analyzer (`som-analyze`) | ✅ New (type-aware Copy/move; see `scirust-som/README.md`) |
| Portable GPU compute + autograd + Conv2d (`scirust-gpu`, feature `wgpu`) | ✅ New (WGSL GEMM, `Var::matmul_gpu` + Conv2d fwd/bwd, oracle-validated on lavapipe) |
| CUDA Tensor-core GEMM (`scirust-gpu`, feature `cuda`) | ✅ Opt-in bf16/cuBLASLt backend; graceful `Unavailable` without runtime/device; hardware CI still needed |

> **GPU scope (honest).** A portable wgpu GEMM is wired behind the optional
> `wgpu` feature, tested against the CPU oracle on a software Vulkan adapter
> (Mesa lavapipe) in CI, plumbed into the autograd tape (`WgpuEngine` +
> `Var::matmul_gpu`) and Conv2d's im2col GEMMs (forward and backward), with a
> VRAM-resident matmul-chain API (`GpuChain`) that keeps intermediates on the
> device across GEMMs. It is opt-in, so the default bit-exact guarantee is
> unaffected. Still to do: make tape residency transparent and move im2col onto
> the GPU (P2.2).
> **CUDA** is implemented as an opt-in bf16/cuBLASLt backend. Dynamic loading
> keeps no-CUDA builds usable and makes `CudaBackend` return `Unavailable`
> when the runtime/device is absent. Hardware parity and performance validation
> still require a CUDA runner; earlier drafts live in `archive/scirust-gpu/`.
> See [`docs/GPU.md`](docs/GPU.md) and
> [`docs/INDUSTRIAL_ROADMAP.md`](docs/INDUSTRIAL_ROADMAP.md).


## Package layout: framework library vs. bundled agent

The `scirust` package exposes the framework as a **library** (`src/lib.rs`), re-exporting
the member crates under `scirust::{core, simd, symbolic, learning, solvers}`. The
deep-learning and scientific-computing capabilities described here live in those crates.

The repository also bundles a small **experimental autonomous-agent binary**,
`openclaw-u` (`src/main.rs`, run with `cargo run --features openclaw --bin openclaw-u`). It is *not* a
component of the framework, is not required to build or use it, and can be ignored
entirely. Parts of the repository were developed with the assistance of **SoulLink**, a
separate agent system maintained outside this repository; like OpenClaw-U, it is not part
of the framework.








## scirust-sciagent — Deterministic SLM for Rust

The `scirust-sciagent` crate provides a from-scratch transformer trained on Rust source code (672MB from crates.io + The Stack v2).

**Architecture**: GQA + SwiGLU + RoPE + RMSNorm, from 106K to 7B params.

**Configs**:
| Config | Params | Vocab | Layers | Seq Len |
|--------|--------|-------|--------|---------|
| debug  | 106K   | 256   | 2      | 128     |
| small  | 1.6M   | 8192  | 4      | 256     |
| 350M   | 350M   | 32768 | 24     | 8192    |
| 7B     | 7B     | 32768 | 40     | 8192    |

**Pretrained**: `small` checkpoint (2000 steps, loss 9.01→8.90) at `/tmp/scirust_small_2k/final/`. See `scirust-sciagent/README.md`.

```bash
cargo run --release -p scirust-sciagent --bin sciagent -- --model small \\
  --checkpoint /tmp/scirust_small_2k/final ask "fn main()" --max-tokens 100
```

## License

Dual-licensed: [PolyForm Noncommercial 1.0.0](LICENSE.md) for noncommercial and personal use; commercial license required for any commercial use. See [LICENSING.md](LICENSING.md).
