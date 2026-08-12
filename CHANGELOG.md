# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/);
semantic versioning starts from the next tagged release.

## [Unreleased]

### Added — Extension of adjacent ECC domains in SciRust
- **Bilinear Pairings**: implementation of line evaluation, the Miller algorithm, and reduced Weil and Tate pairings with zero dynamic allocation (zero-allocation hot paths).
- **Identity-Based Encryption & Commitments**: Boneh-Franklin IBE primitives and bilinear accumulators/commitments.
- **Post-Quantum Isogenies**: implementation of Vélu's formulas for curves and points of odd-order subgroups, and isogeny-graph exploration by BFS.
- **Cayley-Dickson Hypercomplex Curve Algebras**: Octonion (`Oct8Fp`) and Sedenion (`Sedenion16Fp`) types over $F_p$ with non-commutative/non-associative multiplication and geometric encryption.
- **Hybrid Simulator & CCOS Traceability**: modeling of the Shor filter resonance on `DenseStateVector` and immutable logging in `CcosAuditChain` via SHA-256 fingerprints.

### Closed — Correctness '26 paper effort: archiving (not submitted in 2026)
- **User decision (2026-07-11)**: no submission this year. The
  `paper/correctness26/` draft is archived as-is (status banner +
  note freezing the figures at commits `0c2f1bf`/`014795f` dated 2026-07-10),
  `paper/PAPER_PLAN.md` gains a §7 "Final status — ARCHIVED" listing what
  remains valid (draft, related work, all `[CI]` claims re-tested at every
  commit) and what needs refreshing before a future submission (re-measure
  O1, bibliography, submission TODOs, section on the portable f32 path
  added since).
- **Raw evidence archived in `docs/evidence/`**: the 22 "dead guards"
  mining reports per repository (sealed `Report-SHA256`, previously only in
  the session's ephemeral `/tmp`) + `SHAS.txt` of the cloned commits; and
  the O1 bench outputs — two x86-64 runs and the two complete Jetson AGX
  Thor protocols (transcribed from the operator's terminal outputs, original
  bundles kept on the Jetson), each file with its provenance note. The 4
  cross-platform fingerprints (`0x60daf62c…`, `0x9bf7c3f3…`, `0xd5b8e15f…`,
  `0x7e99a9d0…`) are re-verifiable there.

### Added — real-data validation #3: noisy speech (VoiceBank+DEMAND)
Third real domain (§9.4 of the TSHF report) — **an honest mapping of a limit**
of the generalist toolkit rather than a success.
- **Attributed real fixture** `scirust-signal/tests/data/voicebank_demand.csv`
  (CC-BY-4.0): a VoiceBank utterance (`p232_022`, 16 kHz) + its version
  corrupted by **real recorded DEMAND noise** (global SNR ~7 dB), decoded
  offline from the Hugging Face parquet. Provenance + license at the top of
  the file.
- **Test** `tests/real_data_audio.rs` + example `denoise_real_speech`: on the
  waveform-SNR metric, the generalist toolkit **does not improve** speech —
  `denoise_auto` classifies it `Colored` and over-processes (aggressive
  per-level wavelet → **−8.9 dB**, non-stationary speech is smoothed).
  Actionable finding pinned: **for speech, call `stft_wiener_auto` directly**
  (noise-floor tracking; far less destructive, −0.04 dB) rather than the
  auto routing. Note: waveform SNR is a harsh metric for speech (the domain
  metrics are segmental SNR / PESQ / STOI) — "≈ neutral" is a floor, not the
  full picture.

### Added — real-data validation #2: bearing vibration (CWRU)
Extends real-data validation to a **different domain** (§9.4 of the TSHF
report) and stresses the classifier's robustness to legitimate periodic
features (#586) outside the ECG.
- **Attributed real fixture** `scirust-signal/tests/data/cwru_bearing.csv`:
  12 kHz drive-end accelerometer signals from the Case Western Reserve
  University Bearing Data Center — healthy bearing (record 97) and 0.007"
  outer-race fault (record 130), decoded offline from the MATLAB format.
  Provenance at the top of the file.
- **Integration test** `tests/real_data_vibration.rs` + example
  `classify_real_bearing`: the outer-race fault produces impacts that
  **reach the impulsive gate** (kurtosis 5.02 > 4 AND crest 5.15 > 5) — a
  naive classifier would take them for impulsive noise and a spike
  suppressor **would destroy the fault signature**. The energy-envelope
  periodicity veto (`periodic_impulse_train`) recognizes the BPFO impact
  train as a legitimate repeated feature: verdict **non-`Impulsive`**,
  signature preserved. Same robustness as the QRS ECG case, on a totally
  different physical domain.

### Added — `denoise::remove_baseline`: signal-preserving zero-phase detrend
Motivated by the real ECG validation (#579): baseline drift masks the ECG's
own low-frequency content, and the rigid Tikhonov detrend of the `Baseline`
path erodes it along with the drift.
- **`remove_baseline(signal, sample_rate, cutoff_hz)`**: 2nd-order
  Butterworth high-pass applied **forward-backward** (`filtfilt_sos`,
  zero-phase → effective 4th order) at an explicit cutoff in Hz. Unlike the
  Tikhonov detrend (soft, implicit cutoff that creeps into the signal), the
  cutoff is sharp and deliberately placed: for an ECG, `cutoff_hz = 0.5`
  removes the drift while preserving the ST segment (ANSI/AAMI EC11 / AHA
  value). Deliberately gentle order 2 (at the very low normalized cutoffs
  ~0.003 that drift removal requires, a higher order would place the poles
  pathologically close to `z = 1`). Graceful degradation.
- **Methodology finding (real data)**: the DC-inclusive SNR metric is
  *unsuitable* for evaluating drift removal — confounded by the legitimately
  removed DC/baseline, it caps every method at ~+1 dB. Measured correctly
  (morphology recovery without drift), the physiological high-pass recovers
  the ECG **> +10 dB better than raw and beats the Tikhonov detrend**
  (`tests/real_data_ecg.rs`, example `denoise_real_ecg`).

### Changed — classifier robust to legitimate periodic features (`detect::classify`)
Motivated by the real ECG validation (#579): QRS complexes were read as
"impulsive noise". The impulsive gate now requires **aperiodicity** of the
peaks.
- **`periodic_impulse_train`**: autocorrelation (via FFT, O(n log n)) of the
  energy envelope `core² − mean` of the high-pass residual; a normalized peak
  > 0.30 on the repetition lags `[8, n/3]` (≥ 3 periods) signals a
  **legitimate periodic train** (QRS, engine knock, repeated transient) →
  veto of the `Impulsive` verdict. Threshold calibrated by measurement: real
  QRS (record 100) ≈ 0.3–0.4 at the rhythm lag (~73 bpm, robust to heart-rate
  variability); aperiodic impulses (salt-and-pepper, electrode pops) ≈ 0.2 →
  still `Impulsive`.
- Effect on real data: an ECG dominated by its QRS is no longer labeled
  `Impulsive` (no more routing to a spike suppressor); when real
  wideband/impulsive noise is mixed in, the periodicity is masked and the
  verdict stays `Impulsive` — correct discrimination, not binary.
- Test impulse fixtures made **aperiodic** (Bernoulli placement) — real
  impulsive noise is aperiodic, a realism gain; new test
  `periodic_spike_train_is_a_legitimate_feature` + assertion on real ECG
  (`qrs_complexes_are_not_mislabeled_as_impulsive_noise`). Zero regression
  (467 tests).

### Added — denoising validation on **real data** (MIT-BIH ECG + real noise)
Response to §9.4 of the TSHF report (real-data validation), with ground truth.
- **Attributed real fixture** `scirust-signal/tests/data/ecg_mitbih.csv`
  (ODC-BY): MIT-BIH ECG record 100 (lead II) + **real recorded** noise from
  the Noise Stress Test Database (`ma` wideband muscle artifact, `bw`
  baseline drift), 4096 samples at 360 Hz, decoded from WFDB 212 format.
  Provenance and license at the top of the file.
- **Deterministic integration test** `tests/real_data_ecg.rs` (real noise
  added at controlled SNR, no RNG): the VST detector returns **`Identity`**
  on real ECG at every SNR and for both noises (no false positive —
  conservativeness validated on real data); the wavelet removes the muscle
  artifact (+0.5 to +1.1 dB); `denoise_auto` gains > +2 dB at 0 dB;
  fixture-shape guardrail (QRS structure present).
- **Example** `examples/denoise_real_ecg.rs`: full tables (classifier
  verdict, `denoise_auto` method, SNR improvement per method) over both
  noises × three SNRs, with an honest synthesis of the observations —
  including the limits revealed *by the real data*: at high SNR the QRS
  complexes are seen as impulsive (cautious routing to Hampel, near no-op);
  baseline drift masks the ECG's own low-frequency content (detrend =
  modest gain).

### Added — `denoise::streaming::StreamingNlm`: causal non-local means
- **`StreamingNlm`**: causal, sample-by-sample counterpart of batch
  non-local means (`nlm::nlm1d`), for self-similar signals (periodic
  shapes, repeated transients, piecewise-constant) on real-time/embedded
  streams. `delay()` = `search_half + patch_half`; state = a circular
  buffer of `2·delay+1` samples.
  - **Bit-exact batch equivalence**: at a given σ (the calibration parameter
    the batch estimates globally), once the buffer is full, the output is
    bit-for-bit `nlm1d(signal, patch_half, search_half, h)[i − delay]` on the
    batch interior — the batch `sum_sq_diff` kernel is reused (exposed
    `pub(crate)`), and the weight rule (noise compensation `2σ²`, reference
    term at weight 1, NaN quarantine → weight 0) is byte-for-byte the
    batch's.
  - σ = **calibration parameter** (measured offline on a representative
    capture, like `StreamingVst`); bandwidth `h ≤ 0` → `0.8·σ`; non-positive
    effective bandwidth → pass-through (constants preserved). Graceful
    degradation (finite warm-up, NaN quarantined). Object-safe and
    composable in `StreamingVst<StreamingNlm>` (causal NLM in the
    stabilized domain).

### Added — `denoise::streaming`: causal VST for embedded (`StreamingVst<D>`)
- **`StreamingVst<D: StreamingDenoiser>`**: causal, sample-by-sample
  counterpart of the VST pipeline (`vst::vst_denoise`), for
  signal-dependent noise on a real-time/embedded stream. Wraps any
  streaming denoiser `D` with a pointwise forward transform and a
  bias-corrected inverse; `delay()` = `D`'s (VST stages add none), memory
  `O(D + W)`, deterministic arithmetic.
  - **Pointwise-inverse kernels** (Identity / Anscombe / GAT): output
    **bit-identical** to batch `vst_denoise` around `D`'s batch counterpart,
    shifted by `delay()`, on the interior (pinned by test and example).
  - **Smearing kernels** (signed log / signed root / Box-Cox): Duan inverse
    over a **sliding window** of recent residuals
    (`DEFAULT_RESIDUAL_WINDOW`, configurable) — causal and locally
    adaptive, `O(W)` per sample; non-finite residuals discarded, fallback to
    the naive inverse while no residual exists.
  - Transform type = **calibration parameter** (identified offline by
    `detect_noise_model` on a representative recording); `Identity` makes
    the wrapper transparent. Graceful degradation (degenerate parameters →
    `D` on the raw stream; finite warm-up; `residual_window = 0` clamped to
    1).
  - Refactor `vst.rs`: scalar helpers `forward_scalar` /
    `inverse_corrected_pointwise_scalar` / `inverse_naive_scalar` extracted
    as the single shared source of truth for batch/streaming (batch outputs
    unchanged).
  - Example `examples/vst_streaming_embedded.rs`: Poisson photon-counting
    stream, denoised continuously (8.96 → 19.16 dB), bit-exact batch
    equivalence verified, bounded memory footprint displayed.

### Added — `denoise` round 5: GAT, replayable §9 protocol, 2-D VST, multilingual docs
Extends the execution of the TSHF program (round 4); every choice is
calibrated by measurement (addendum 2 of the report).

- **`VstKind::Gat { gain, sigma }`**: generalized Anscombe for the mixed
  Poisson-Gaussian sensor model `x = gain·p + n` (Murtagh-Starck-Bijaoui
  1995), exact closed-form unbiased inverse (Mäkitalo-Foi 2013); gain=1,
  σ=0 ≡ Anscombe (tested). Measured: +1.54 to +2.87 dB depending on
  calibration.
- **`examples/vst_protocol.rs`**: the report's §9 protocol, replayable and
  deterministic (P1-P5). Measured answers to the open questions: materiality
  crossing ≤ 2 % noise at ×10 dynamic range (P4a); **≈ ×3 dynamic range of
  levels** at 30 % noise — at ×2 the VST loses −0.77 dB (P4b); carrier
  collapse +5.17 dB (3 cycles) → −0.93 dB (40 cycles) (P5).
- **Selector tightened by measurement**: `detect_noise_model`'s dynamic
  range gate widened from ×2 to **×3** (`DETECT_MIN_RANGE`), aligned with
  the P4b crossing — "never degrade".
- **"Fast carriers" limitation documented and pinned**: a pointwise φ does
  not commute with the spectrum (harmonics clipped by the internal
  denoiser); module doc + sentinel test.
- **`scirust_vision::denoise::{vst_denoise2d, vst_denoise2d_auto}`**: 2-D
  image VST; measured internal partner = **2-D NLM** (+5.4 dB Poisson, +3.0
  dB GAT); 2-D VisuShrink loses under stabilization (1-D negative result
  transposed); 2-D median is bitwise invariant (Prop. 2 of the report
  confirmed); auto conservative (Identity verdict → copy).
- **Multilingual docs**: the TSHF block (vst/multichannel/compand +
  GAT/2-D/protocol) translated into the six languages AR/DE/ES/JA/KO/ZH.

### Added — `denoise` round 4: execution of the TSHF report recommendations
All codable recommendations of the report
`docs/project-notes/TSHF_RESEARCH_2026-07-16.md` (§12 and roadmap) are
executed; every quantified acceptance gate of the report was measured before
integration.

- **`denoise::vst` (Phase 1)**: variance-stabilizing transforms with
  bias-corrected inverse — Anscombe + exact unbiased inverse
  (Mäkitalo-Foi 2011, closed form), signed log + Duan smearing (1983),
  signed root, Box-Cox(λ); conservative selector `detect_noise_model`
  (Theil-Sen on log σ vs log level, default = identity); generic
  `vst_denoise` and `vst_denoise_auto`. Wired as a conditional pre/post
  stage of `denoise_auto`. **Gates passed**: low-count Poisson +5.02 dB
  (criterion ≥ +1 dB) and the corrected inverse beats the naive by +3.90
  dB; strong multiplicative +4.88 dB; gentle regime +0.04 dB (null gain
  predicted, never a loss); retransformation bias 0.015 vs 0.268 (naive)
  at λ = 4. Measured choice: internal denoiser `stft_wiener_auto` (the
  VisuShrink wavelet violated "never degrade" by −1.0 dB on level-correlated
  signals).
- **`denoise::multichannel` (Phase 2)**: `wiener_spatial` — joint spatial
  Wiener (real-vector equivalent of widely-linear quaternionic filtering,
  Took-Mandic) — **passes the gate**: +2.48 dB and +3.67 dB vs its diagonal
  restriction on the two correlated fixtures; `vector_median` (Astola 1990)
  — **fails the gate** (0/2): −1.81 dB on synchronized impulses (E5b of the
  report reproduced) and −2.02 dB even on desynchronized impulses — the
  report's §12.4 conjecture is falsified and documented; the operator is
  kept as a reference with its verdict. Reproducible report:
  `phase2_gate_report()`.
- **`denoise::compand` (reco 3)**: `soft_clip` / `soft_clip_robust`
  (tanh/atan/softsign) — bounded soft clipping for display and robust
  features, without inverse by design (E2/E4: ×10-×100 amplification,
  Jensen bias).
- **Phases 3-5 not triggered** (report conditions not met): octonion — no
  demonstrated need at 8 coupled channels (Phase 2 even falsified the
  vector median); SIMD of φ/φ⁻¹ — O(n) cost negligible before the internal
  denoisers; GPU — unchanged volumes. Addendum added to the report.

### Added — `denoise` round 3: multilingual docs, SIMD, BM3D-1D, and the TSHF research program
- **Multilingual documentation**: the denoising section (8.1.1) added to
  the six translations `Documentation_{AR,DE,ES,JA,KO,ZH}.md` (code
  identifiers kept verbatim, known limitation included).
- **SIMD-ization of the NLM kernels** (1-D and 2-D): auto-vectorizable
  restructuring (precomputed mirror buffer → patch distances on contiguous
  slices, independent accumulators); reference scalar path kept and pinned
  at 1e-12 relative; measurement harness
  `examples/denoise_kernel_timing.rs`.
- **`denoise::collab`**: BM3D-style 1-D collaborative patch filtering
  (Dabov et al. 2007 — similarity grouping, 2-D Haar patch×group, 2.7σ
  hard thresholding, weighted aggregation) — `collab1d`, `collab1d_auto`.
- **TSHF research program**
  (`docs/project-notes/TSHF_RESEARCH_2026-07-16.md`): skeptical
  investigation of "Transformed-Scalar Hypercomplex Filters" — mathematical
  analysis (φ/embedding separability, median invariance, quasi-arithmetic
  means), six blocks of reproducible falsification experiments
  (`examples/tshf_experiments.rs`: 1/Γ not injective; identity wins on
  additive noise; quantified retransformation bias; vector median beaten by
  the per-channel one on correlated impulses) and extensive literature
  review (Anscombe 1948 → Mäkitalo-Foi 2013, homomorphic 1968,
  Kolmogorov-Nagumo 1930, Alfsmann-Göckler 2007). **Verdict: TSHF family
  rejected**; viable subset identified (VST module with corrected inverse)
  with a roadmap and quantified acceptance criteria.

### Added/Changed — `denoise`: classifier refinements, new methods, 2-D denoising
Second batch of the anti-noise filter: measured classifier corrections, new
families, and 2-D extension — parallel implementation (3 agents: STFT,
NLM/blocks, 2-D vision) + central integration, every threshold calibrated by
quantified probes.

**Classifier (`detect::classify`) — the 3 errors observed at the bench fixed:**
- Baseline gate **0.6 → 0.45**: a drift at power equal to the signal (0 dB,
  ts ≈ 0.49) is now detected and relaxed (+15.5 dB at the bench, vs 0
  before); AR(0.9) stays far below the gate (ts ≈ 0.05).
- **Robust edge score** (max of the median-prefiltered derivative over its
  MAD): step recordings are no longer "relaxed" as drifts — their bench
  destruction (up to −19 dB at 20 dB input) is eliminated; a kurtosis test
  would be blind below ~30 dB (1/n dilution of a single peak).
- LowNoise gate **1 % → 5 %** of RMS: recordings with almost no wideband
  floor (spectral slope measured on leak skirts, not noise) get the
  Savitzky-Golay touch-up instead of the wavelet machinery (−15.8 dB → ±0).
- Documented limitation: a tone below ~5 % of fs is indistinguishable from a
  legitimate component (any smooth/residual split statistic is a pure
  function of f/fs) — call `remove_mains_hum_iir` explicitly.

**Tonal guard of `wavelet_denoise_leveldep`**: each band is screened
(kurtosis < −0.75 **and** σ_j > 2.5×median ⇒ band filled by a tone); flagged
bands borrow the σ of a healthy fine band and go through BayesShrink — a
sustained tone survives (7.3 dB → ~11 dB instead of 7.3 → 0.0 before).

**New methods:** `nlm1d`/`nlm1d_auto` (1-D non-local means, Buades 2005),
`wavelet_denoise_neighblock` (block thresholding, Cai-Silverman 2001),
`stft_mmse_lsa` (Ephraim-Malah 1985, E1 by series + continued fraction),
`stft_wiener_tracked_ms` (minimum statistics with adaptive smoothing, Martin
2001), `StreamingStftWiener` (**real-time** short-term Wiener, one-frame
latency).

**2-D denoising (`scirust-vision::denoise`)**: `median2d`,
`wavelet_denoise2d` (separable Mallat pyramid on the now-public 1-D filter
banks, exact round-trip at 1e-9), `nlm2d` — +6.2 dB PSNR wavelets, +32 dB
median on salt-and-pepper.

**Consolidation:** batch rank filters rewritten **on the streaming engine**
(incremental sorted window: O(w) per sample, batch↔streaming equivalence by
construction, pinned bit-for-bit against the naive definition); **odd**
cycle-spinning step (the degenerate n_shifts | n case disappears: +3 dB on
odd edge with 8 shifts); `Serialize/Deserialize` on
`AutoResult`/`BestResult`; denoising sections in
`docs/translations/Documentation_EN.md`/`docs/REFERENCE.md`.

At the bench: automatic pipelines win 5 of 7 noise types (cascade: white
+13.0 and non-stationary +11.7) and never destroy clean or step references
again. Check: full `scirust-signal` suite + `scirust-vision` green;
`fmt` / `clippy -D warnings` clean.

### Added — `scirust-signal`: anti-noise filter overhaul (`denoise`)
Large extension of the denoising toolbox, produced by parallel
implementation (4 agents) then integration and **adversarial review** (5
axes, 3 refuters per finding); all confirmed findings were fixed before
merge. **382 unit tests + 7 integration tests + 2 doctests** green;
`fmt` / `clippy -D warnings` clean.

**New denoisers (`transform`, `iir`, `stft`, `streaming`):**
- **Translation-invariant wavelets** (`cycle_spin`, `wavelet_denoise_ti`,
  Coifman-Donoho 1995): averaging over circular shifts — removes
  pseudo-Gibbs artifacts around transients.
- **Scale-dependent thresholds** (`wavelet_denoise_leveldep`,
  Johnstone-Silverman 1997): σ_j estimated per band — the right tool for
  colored noise.
- **BayesShrink** (`wavelet_denoise_bayes`, Chang-Yu-Vetterli 2000).
- **Zero-phase IIR notch** (`rbj_notch`, `filtfilt_sos`, `notch_iir`,
  `remove_mains_hum_iir`, `BiquadState`): no ringing or spectral leakage,
  accurate even when the interferer falls between two FFT bins.
- **Short-term (STFT) Wiener** (`stft_wiener`, `stft_wiener_dd` with
  decision-directed a-priori SNR Ephraim-Malah, `stft_wiener_auto`,
  `stft_wiener_tracked` with min-statistics floor tracking): gains
  re-estimated per frame for **non-stationary** noise.
- **Streaming denoisers** (`StreamingDenoiser` + `StreamingMovingAverage`,
  `StreamingMedian`, `StreamingHampel`, `StreamingEma`, `StreamingKalman`):
  causal sample-by-sample versions for edge/embedded.

**Selection & pipeline (`mod`, `detect`, `cascade`):**
- **Multi-stage cascade** (`denoise_cascade`, `denoise_cascade_auto`):
  detect → process → re-detect for mixed noise (impulses + sector + floor),
  with anti-loop protection and an **accept/abort guard** — a wideband stage
  is validated only if what it removed is noise-typed (flat spectrum),
  otherwise it is aborted (it was eating a signal tone).
- **Tournament selection** (`denoise_best`): reference-free score (residual
  whiteness minus over/under-denoising penalty) over a per-family
  preselection.
- **Harmonic multi-line detection** (`detect_lines`, `harmonic_stack`,
  `SpectralLine`) and `denoise_auto` v2: line peeling + harmonic IIR notch.
- Measurement bench `examples/denoise_benchmark.rs` (methods × noise types ×
  SNR) and non-regression guard `tests/denoise_integration.rs`.

**Fixes from the review:**
- `harmonic_stack`: bounded harmonic index (`k ≤ 12`), tolerance on
  *distinct* indices and required low fundamental — eliminates false
  families (a 7 Hz signal residue and a 137 Hz interferer are no longer
  "harmonics" via `f0 = 3.5`).
- Periodic router (`notch_detected_lines`): protects the signal's own tone
  (`signal_dominant_freq`) instead of notching it, covers the whole detected
  family (`harmonic_span`, not the simple count), and handles lines near
  Nyquist honestly (brick-wall wrap).
- Cascade: cumulative-whiteness progression criterion
  (self-contradictory) replaced by the accept/abort guard above.
- Periodic `denoise_best`: removal of the line enhancer (it returned the
  tone, not the denoised signal, and fooled the score).
- Streaming rank denoisers: total order `f64::total_cmp` — a single NaN no
  longer corrupts the sorted window (which silently degenerated to even
  size).
- Cycle-spinning `n_shifts` raised from 8 (degenerate on power-of-two
  lengths) to **15** (odd) everywhere; added the `WaveletSure`
  wrapper/catalog.

### Added — radar & optronics: massive batch of 10 self-contained modules (blocks 40-49)
Ten independent radar and EO/IR capabilities, each oracle-tested, produced
in parallel (isolated worktree agents) then integrated and verified
centrally. **`scirust-signal` (radar):**
- **`cfar_variants`** — CFAR greatest-of / smallest-of / trimmed-mean
  (`go_cfar`, `so_cfar`, `tm_cfar`): GO removes false alarms on a clutter
  edge, SO resolves two close targets, the trimmed mean censors an
  interferer.
- **`binary_integration`** — M-of-N binary integration (`binomial_pmf`,
  `binomial_sf_ge`, `integrated_pfa`, `integrated_pd`, `optimal_m`): strong
  false-alarm probability reduction via the binomial law.
- **`crt_prf`** — multi-PRF range de-ambiguation by the Chinese remainder
  theorem (`egcd`, `mod_inverse`, `crt_pair`, `resolve_range`,
  `combined_ambiguity`).
- **`costas`** — Costas frequency-hopping arrays (`welch_costas`,
  `is_costas`, `max_coincidence`, `primitive_root`): ideal "thumbtack"
  ambiguity.
- **`propagation`** — two-ray propagation factor (ground multipath)
  `F = 2|sin(2π·h_a·h_c/(λR))|`, interference lobes, power in `F⁴`, first
  null.
- **`dbs`** — Doppler beam sharpening (`azimuth_doppler`, `doppler_gradient`,
  `dbs_azimuth_resolution`, `sharpening_ratio`): transverse resolution by
  Doppler gradient in the real beam.
**`scirust-vision` (optronics):**
- **`nuc`** — two-point non-uniformity correction of an IR focal plane
  (`two_point_coeffs`, `apply_nuc`, `fixed_pattern_noise`).
- **`lidar`** — time-of-flight and CW-phase laser ranging
  (`range_from_time_of_flight`, `time_of_flight`, `range_resolution`,
  `range_from_phase`, unambiguous ranges).
- **`centroid`** — subpixel centroiding (weighted, thresholded, windowed
  center of gravity) for EO/IR pointing.
- **`zernike`** — Zernike wavefront aberrations (defocus, astigmatism, coma,
  spherical), RMS error, and Maréchal Strehl `exp(−(2π·σ)²)`.
Check: `scirust-signal` **317 tests**, `scirust-vision` **95 tests** (+72
oracles); `fmt` / `clippy -D warnings` clean on both crates.

### Added — `scirust-signal`: radar — measurement accuracy, Cramér–Rao bounds (`radar::accuracy`) — block 39
The theoretical precision floor of radar estimators (delay/ranging,
Doppler/velocity, monopulse angle): the **Cramér–Rao bounds**, all in
`1/√SNR`.
- **`rms_bandwidth_lfm(B)`** = `B/√12` (RMS bandwidth of a flat spectrum);
  **`rms_duration_rect(T)`** = `T/√12`; **`delay_crlb(SNR, β_rms)`** =
  `1/(2π·β_rms·√(2·SNR))`; **`range_crlb`** = `(c/2)·σ_τ`.
- **`doppler_crlb(SNR, T_rms)`** = `1/(2π·T_rms·√(2·SNR))`;
  **`velocity_crlb`** = `(λ/2)·σ_fd`; **`angle_crlb(SNR, θ₃dB, k_m)`** =
  `θ₃dB/(k_m·√(2·SNR))` (monopulse).
- Oracles: exact closed forms; **∝ 1/√SNR** (×4 SNR ⇒ ÷2); ranging
  `= (c/2)·delay` and velocity `= (λ/2)·Doppler`; a wider band **sharpens
  range**, a longer integration time **sharpens velocity**, a steeper
  monopulse slope sharpens angle; guards (SNR/band/duration ≤ 0 ⇒ `+∞`, no
  NaN). 7 tests (273 total for the crate); `fmt`/`clippy -D warnings`
  clean.

### Added — `scirust-vision`: optronics — atmospheric turbulence and adaptive optics (`turbulence`) — block 38
The [`atmosphere`] module models the **attenuation** of contrast along the
path; this module models the **blur** that index turbulence adds (strength
given by the structure constant `Cn²`), which limits long-range EO/IR
imaging and makes a laser beam scintillate.
- **`fried_parameter(Cn², λ, L)`** = `(0.423·k²·Cn²·L)^(−3/5)` — the Fried
  parameter `r₀` (coherence length, `k=2π/λ`); **`seeing_angle(λ, r₀)`** ≈
  `0.98·λ/r₀`, the seeing blur that replaces the diffraction limit `λ/D` as
  soon as `D > r₀`.
- **`strehl_ratio(D, r₀)`** = `[1 + (D/r₀)^(5/3)]^(−6/5)` — the fraction of
  peak intensity preserved without correction; **`greenwood_frequency(v,
  r₀)`** = `0.426·v/r₀`, the bandwidth of an adaptive-optics loop;
  **`degrees_of_freedom(D, r₀)`** = `(D/r₀)²`; **`rytov_variance(Cn², λ,
  L)`** = `1.23·Cn²·k^(7/6)·L^(11/6)`, the scintillation (intensity
  twinkling) in weak turbulence.
- Oracles: `r₀` matches the closed form and **∝ λ^(6/5)** (decreases with
  Cn² and L); seeing = `0.98·λ/r₀` and **exceeds diffraction** `λ/D` when
  `D ≫ r₀` (→ 0 without turbulence); Strehl → 1 for `D ≪ r₀`,
  `= 2^(−6/5)` at `D = r₀`, monotonically decreasing, bounded (0, 1];
  Greenwood ∝ v and ∝ 1/r₀; Rytov ∝ Cn², ∝ L^(11/6), ∝ k^(7/6); guards
  (degenerate inputs safe, no NaN). 7 tests (67 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — polyphase / CAZAC codes (`radar::polyphase`) — block 37
The compression waveforms beyond Barker. Barker codes are optimal but stop
at length 13; **polyphase codes** use several phase values, exist at every
length, and — for Frank and Zadoff-Chu — have a **perfect periodic
autocorrelation** (one impulse, zero sidelobe), the sought property when a
code is repeated every PRI. P3/P4 codes (sampled LFM) trade a little of
that perfection for Doppler tolerance and a "noisy" phase that makes them
the canonical **LPI** (low probability of intercept) waveforms.
- **`frank_code(n)`** — Frank code, length `N²`, phase `2π·i·k/n`;
  **`p3_code(L)`** = `exp(j·π·n²/L)`; **`p4_code(L)`** =
  `exp(j·(π·n²/L − π·n))`; **`zadoff_chu(L, u)`** — perfect CAZAC sequence
  for any `u` coprime with `L`.
- **`periodic_autocorrelation(code)`** — `R[τ] = Σ code[n]·conj(code[(n+τ)
  mod L])`.
- Oracles: Frank structure (length `N²`, unit modulus, exact phase);
  **Frank has a perfect periodic autocorrelation** (0 off-zero, `N²` at
  zero); **Zadoff-Chu is CAZAC** (perfect for even/prime length,
  non-coprime root rejected); P3/P4 = **sampled LFM phases** (unit modulus,
  exact formula); the **aperiodic autocorrelation peak = length** (reuses
  `cross_correlate`); Frank's **periodic** autocorrelation **beats
  Barker-13** (zero sidelobes vs a real one); guards. 7 tests (266 total
  for the crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — SAR imaging, azimuth compression (`radar::sar`) — block 36
The **imaging** mode of radar. A real antenna of length `D` has a transverse
resolution `λR/D`, coarse at long range; **SAR** refines it by synthesizing
a long aperture from the platform's motion: passing a point target at the
minimum distance `R₀`, the range history traces a parabola `R(x) ≈ R₀ +
(x−x₀)²/(2R₀)`, imprinting a quadratic phase (an **azimuth chirp**) on the
slow-time signal. Matched filtering of this chirp — the pulse compression of
`radar::matched_filter`, but in azimuth — focuses the target into a sharp
peak.
- **`synthetic_aperture_length(λ, R, D)`** = `λR/D` (synthesized aperture);
  **`azimuth_resolution(D)`** = `D/2`, the transverse resolution
  **independent of range**; **`azimuth_doppler_bandwidth(v, D)`** = `2v/D`;
  **`azimuth_chirp_rate(v, λ, R)`** = `2v²/(λR)`.
- **`azimuth_history(R, x₀, λ, positions)`** — the slow-time phase history
  `exp(−j·2π·(x−x₀)²/(λR))`; **`azimuth_reference(R, λ, positions)`** — the
  reference chirp; **`focus_azimuth(signal, reference)`** — azimuth
  compression by correlation (reuses `cross_correlate`).
- Oracles: closed forms (resolution `D/2` independent of R, aperture ∝ R,
  bandwidth `2v/D`); chirp rate ∝ `v²` and ∝ `1/R`; the azimuth history
  **is a linear FM chirp** (constant second phase difference) and matches
  the parabola; the **matched filter focuses a point target** exactly at
  its azimuth position (correlation peak); **two separated targets
  resolved** into two peaks; guards. 7 tests (259 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — space-time adaptive processing (STAP) (`radar::stap`) — block 35
The clutter filter of airborne radars. Under a moving platform, ground
clutter folds onto an angle-Doppler **ridge** `f_d = β·f_s` (with
`f_s = (d/λ)·sin θ`): a slow target buried in clutter in range and Doppler
is nevertheless **separated in the 2-D plane** — it is off-ridge. A filter
that adapts jointly over the `N` antenna elements **and** the `M` CPI
pulses places a null along the ridge while keeping unit gain on the target:
something no 1-D filter (angle-only or Doppler-only) can do.
- **`space_time_steering(f_s, f_d, N, M)`** — space-time steering vector
  `s = b(f_d) ⊗ a(f_s)` (Kronecker product, length `NM`);
  **`spatial_frequency(θ, d)`** = `(d/λ)·sin θ`; **`clutter_ridge_doppler(f_s,
  β)`** = `β·f_s`, the clutter ridge.
- **`clutter_covariance(N, M, patches, β, σ_n²)`** — interference+noise
  covariance `R = σ_n²·I + Σ P_c·s_c s_cᴴ` (clutter patches on the ridge).
- **`adaptive_weights(R, s)`** = `R⁻¹s/(sᴴR⁻¹s)` (MVDR/SMI weights);
  **`optimal_sinr(R, s, P)`** = `P·sᴴR⁻¹s`, the output SINR — deep on the
  ridge, close to the full coherent gain `NM` off-ridge. Reuses the complex
  matrix inverse from `radar::doa`.
- Oracles: steering vector = **Kronecker product** (`|s|²=NM`,
  factorization); white noise ⇒ weights = **matched filter** `s/NM`, unit
  gain, SINR `P·NM/σ²`; **clutter notch** (endoclutter target strongly
  attenuated, off-ridge target preserved); the weight **nulls** the
  co-Doppler clutter patch while keeping the target; the **SINR minimum
  falls exactly on the ridge**; ridge/spatial-frequency relations; guards.
  7 tests (252 total for the crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — interferometric goniometry (phase-comparison) (`radar::interferometer`) — block 34
Where amplitude monopulse ([`radar::monopulse`]) reads the angle from the *ratio*
of two squinted beams, a **phase interferometer** reads it from the *phase
difference* between two elements separated by a baseline `d`: a plane wave at
angle `θ` reaches the far element with a path delay `d·sin θ`, i.e. a phase
advance `Δφ = 2π·d·sin θ/λ`; measuring `Δφ` and inverting gives
`θ = arcsin(Δφ·λ/(2π·d))`.
- **`phase_difference(θ, d, λ)`** = `2π·d·sin θ/λ`; **`angle_from_phase(Δφ, d, λ)`**
  = `arcsin(Δφ·λ/(2π·d))` (argument bounded to [−1, 1]); **`phase_from_signals(near,
  far)`** = `arg(far·conj(near))`, the phase actually observed by the receiver.
- **`unambiguous_angle(d, λ)`** = `arcsin(λ/2d)` — the unambiguous field: a wide
  baseline sharpens the measurement but wraps the phase sooner (resolution/
  ambiguity trade-off); **`wrap_phase`** brings a phase back into `(−π, π]`.
- Oracles: phase **zero at boresight** and **odd**; the estimate **exactly inverts
  the phase** in the unambiguous field; phase recovery from element voltages; a
  **wide baseline shrinks the unambiguous field**; **fold-over (aliasing)** outside
  the field (phase > ±π misread); guards (degenerate baseline/λ).
  7 tests (245 total for the crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — wideband stepped-frequency ranging (`radar::stepped_frequency`) — block 33
Fine range resolution without wideband hardware: a burst of `N` narrowband
pulses at frequencies `fₙ = n·Δf` samples the reflectivity
in frequency, and an **inverse DFT** synthesizes from it a high-
resolution range profile.
- **`synthetic_bandwidth(N, Δf)`** = `N·Δf`; **`range_resolution(N, Δf)`** =
  `c/(2·N·Δf)` (set by the synthesized bandwidth); **`max_unambiguous_range(Δf)`**
  = `c/(2·Δf)` (unambiguous window).
- **`range_profile(measurements)`** — magnitude of the inverse FFT of the per-step
  complex samples (power-of-two length); **`range_bins(N, Δf)`** — the
  distance of each bin.
- Oracles: bandwidth/resolution formulas (resolution that **improves with the
  synthesized bandwidth**, window that widens as Δf decreases); a **point
  scatterer localizes exactly on its bin** (sharp profile at the right
  distance); **two separated scatterers** resolved into two peaks; guards
  (non-power-of-two length → empty). 5 tests (238 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — monopulse direction finding (`radar::monopulse`) — block 32
Precision angle measurement of tracking radars: from a **single** pulse, two
beams squinted on either side of boresight form a **sum** channel `Σ = A + B`
and a **difference** channel `Δ = A − B`, whose ratio `Δ/Σ` gives the
off-boresight angle — accuracy far finer than the beamwidth.
- **`beam_voltage(θ, θ₀, σ)`** — gain (voltage) of a Gaussian beam.
- **`monopulse_ratio(θ, squint, σ)`** = `Δ/Σ` (beams at ±squint) = exactly
  `tanh(θ·squint/σ²)`; **`monopulse_slope(squint, σ)`** = `squint/σ²`, the slope
  of the discriminator at boresight; **`estimate_angle(ratio, squint, σ)`** =
  `atanh(ratio)·σ²/squint`, the inversion.
- Oracles: ratio **zero at boresight** and **odd** (the sign gives the side);
  monotonic and bounded to (−1, 1); **equal to the closed-form tanh**; the
  estimate **exactly inverts** the ratio (angle recovered); **linearization** near
  boresight (`ratio ≈ k_m·θ`); guards. 6 tests (233 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — PRF ambiguities and blind speeds (`radar::prf`) — block 31
The two hard limits that sampling at the pulse repetition frequency (PRF)
imposes on a pulsed-Doppler radar.
- **`unambiguous_range(prf)`** = `c/(2·PRF)` — unambiguous range;
  **`unambiguous_velocity(λ, prf)`** = `λ·PRF/4` — unambiguous velocity;
  **`max_doppler(prf)`** = `PRF/2` (Nyquist).
- **`blind_speed(n, λ, prf)`** = `n·λ·PRF/2` — blind speeds (Doppler = n·PRF,
  cancelled by the MTI canceller along with the clutter);
  **`velocity_from_doppler(f_d, λ)`** = `λ·f_d/2`.
- **`fold_range(range, prf)`** and **`fold_velocity(v, λ, prf)`** — folding a true
  range / velocity back to its measured (aliased) value.
- Oracles: unambiguous range ∝ 1/PRF; **range·velocity ambiguity product =
  cλ/8 invariant** of the PRF (the pulsed-Doppler dilemma); the blind speeds
  are uniformly spaced multiples (`v_blind(1) = 2·v_ua`); at the Nyquist
  Doppler the velocity equals `v_ua`; range folding wraps beyond `R_ua`;
  velocity folding aliases beyond `±v_ua` and stays within the interval.
  6 tests (227 total for the crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — clutter amplitude statistics (`radar::clutter`) — block 30
The amplitude laws that CFAR thresholds target: over calm sea the clutter is
**Rayleigh**, but over rough sea or terrain it becomes **spiky** (heavy tail
that a Rayleigh threshold underestimates, inflating the false-alarm rate).
- **Rayleigh** — `rayleigh_pdf/cdf/quantile` (homogeneous clutter, noise-like).
- **Weibull** — `weibull_pdf/cdf/quantile`: the reference spiky-clutter model,
  with the shape `c` tuning the tail (`c=2` = Rayleigh, `c=1` = exponential,
  `c<2` spikier).
- **Log-normal** — `lognormal_pdf/cdf` (very spiky clutter), via a standalone
  **error function** `erf` (A&S 7.1.26 approximation).
- Oracles: `erf` on known values (0, 1, odd symmetry); Rayleigh CDF
  monotone 0→1, inverse quantile, unit-integral PDF; **Weibull(c=2, b=σ√2) =
  Rayleigh(σ)** exactly; the Weibull quantile inverts and a **weaker shape is
  spikier**; valid log-normal CDF (median e^μ → 0.5), unit integral; negative-
  support guards. 6 tests (221 total for the crate); `fmt`/`clippy -D warnings`
  clean.

### Added — `scirust-signal`: radar — radar equation / range budget (`radar::range_equation`) — block 29
The other half of the link budget (the *required* SNR coming from
`radar::swerling`): the SNR a radar *delivers* on a target of given RCS at a
given range, and the **maximum detection range**. Radar counterpart of the
EO/IR range budget (block 25).
- **`RadarLink`** — groups the radar/system parameters (peak power, gain,
  wavelength, bandwidth, noise figure, temperature, losses; SI/linear units).
- **`noise_power`** = `k_B·T·B·F`; **`received_power(rcs, range)`** — monostatic
  equation `P_r = P_t·G²·λ²·σ / ((4π)³·R⁴·L)`; **`snr_at_range(rcs, range)`**;
  **`max_range(rcs, snr_min)`** = `[P_t·G²·λ²·σ / ((4π)³·N·L·SNR_min)]^{1/4}`,
  closing the loop with the required SNR from `swerling`.
- Oracles: received power in **1/R⁴**; noise power = `k_B·T·B·F`; SNR ∝ σ and
  ∝ 1/R⁴; **max-range ↔ SNR consistency** (the SNR delivered at `max_range`
  equals `snr_min`); range ∝ σ^{1/4}; **integration with Swerling** (a higher
  `P_d`, via Albersheim, shortens the range). 6 tests (215 total for the
  crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — Swerling detection statistics (`radar::swerling`) — block 28
Complement of the CFAR (which sets the threshold for a given false-alarm rate):
the **probability of detection** `P_d` as a function of signal-to-noise ratio,
according to the fluctuation of the target's RCS (**Swerling** cases).
- **`single_pulse_threshold(pfa)`** — single-pulse quadratic threshold
  `V_T = −ln(P_fa)`.
- **`swerling1_pd(snr, pfa) = P_fa^{1/(1+SNR)}`** — single-pulse Swerling I
  `P_d` (slowly fluctuating Rayleigh target); **`swerling1_required_snr`** its
  inverse (linear SNR required for a `P_d`).
- **`albersheim_snr(pd, pfa, n_pulses)`** — **Albersheim** equation (stable
  non-fluctuating target): SNR (dB) required after non-coherent integration of
  `n` pulses, `−5·log₁₀N + (6.2 + 4.54/√(N+0.44))·log₁₀(A + 0.12·A·B + 1.7·B)`;
  **`albersheim_pd`** its inverse (`P_d` from an SNR in dB).
- Oracles: threshold = false-alarm law; Swerling I → `P_fa` without signal, →1 at
  high SNR, monotone, round-trip inversion; Albersheim **round-trip**
  forward/inverse at 1e-6 over four points, `P_d` grows with SNR and with
  P_fa, integration **lowers the required SNR**; the Swerling I **fluctuation
  loss** exceeds the stable target (> 3 dB at `P_d = 0.9`). 5 tests (209 total
  for the crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — micro-Doppler analysis (`radar::micro_doppler`) — block 27
The **time-frequency** signature of a target's micro-motions (rotor blades,
propeller, gait) — the basis of non-cooperative target recognition (NCTR).
- **`spectrogram(signal, win_len, hop)`** — Hann-windowed STFT on the crate's
  power-of-two FFT (magnitude spectrum per frame); **`bin_frequencies(win_len,
  fs)`** — signed Doppler frequencies of the bins.
- Descriptors: **`ridge`** (dominant frequency per frame = instantaneous
  Doppler), **`mean_doppler`** (body Doppler, mean of the ridge),
  **`doppler_bandwidth`** (peak-to-peak spread) and **`cadence`** (repetition
  frequency of the micro-motion, via the autocorrelation peak of the ridge
  beyond the main lobe). No dependency.
- Oracles (synthetic rotating-scatterer signal, instantaneous frequency
  `f_b + f_max·cos(2π f_rot t)`): the ridge mean **recovers the body Doppler**
  `f_b`; the bandwidth **reflects the micro-motion** (≈ 2·f_max) and is zero
  for a pure tone; the cadence **recovers the rotation frequency** `f_rot`; a
  pure tone has a flat ridge at its frequency; guards (non-power-of-two
  window, zero hop, too-short signal, empty ridge). 6 tests (204 total for the
  crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — probabilistic data association filter PDAF (`radar::pda`) — block 26
Fidelity upgrade of tracking in **cluttered environments**: where `radar::mtt`
associates each track to a single measurement by a hard nearest-neighbor choice
(which a closer false echo can divert), the **PDAF** keeps all the
measurements in the gate, weighted by their association probability.
- **`PdaFilter`** — single-target PDAF on the constant-velocity Cartesian state
  `[x, vₓ, y, v_y]` with position measurements `(x, y)`. Each frame: prediction,
  then update by the combined innovation `ν̄ = Σ βᵢ νᵢ` over the in-gate
  measurements, with `β₀` the probability of non-detection (parametric PDA:
  `b = λ·|2πS|^{1/2}·(1 − P_D·P_G)/P_D`). The covariance carries the
  **innovation spread** term `K(Σβᵢ νᵢνᵢᵀ − ν̄ν̄ᵀ)Kᵀ` which inflates it according to
  the association ambiguity. Reuses the dense matrix utilities of `radar::imm2d`.
  No dependency.
- Oracles: without clutter (λ=0) and one measurement per frame, the PDAF reduces
  to a Kalman filter and **tracks a constant-velocity target**; **tracks a
  target through dense clutter** (noisy true measurement + 5 false echoes per
  frame); a detection-free frame **coasts and inflates the covariance** (β₀=1);
  β₀ **drops when a measurement falls on the prediction** and equals 1 on an
  empty frame. 4 tests (198 total for the crate); `fmt`/`clippy -D warnings`
  clean.

### Added — `scirust-vision`: optronics — atmospheric transmission and range budget (`atmosphere`) — block 25
The link that turns the sensor's intrinsic sensitivity (NETD) into a **range
budget**: the atmosphere between target and sensor attenuates the contrast, so
what is detectable depends on the path.
- **`transmittance(α, R) = e^{−αR}`** — **Beer–Lambert** law; **`optical_depth`**
  `α·R`; **`extinction(absorption, scattering)`** additive.
- **`extinction_from_visibility(V) = 3.912/V`** — **Koschmieder** law (2 %
  contrast threshold); **`extinction_from_transmittance(τ, R) = −ln(τ)/R`**
  (inverse).
- **`apparent_contrast(C₀, α, R) = C₀·e^{−αR}`** — contrast transmission law;
  **`required_delta_t(NETD, α, R) = NETD/τ`** — the target ΔT needed to pierce
  the path at range `R`, which grows with distance.
- Oracles: unit transmittance at zero range and monotone decay (closed form
  `e^{−αR}`); **multiplicative Beer–Lambert** over segments
  `τ(R₁+R₂)=τ(R₁)τ(R₂)`; optical depth ↔ transmittance inverses with
  extinction round-trip; Koschmieder visibility **reaches the 2 % threshold**;
  additive extinction and closed-form apparent contrast; the required ΔT
  **grows with range** (= NETD at zero range). 6 tests (60 total for the
  crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-vision`: optronics — IR radiometry and NETD/MRTD sensitivity (`radiometry`) — block 24
The *radiometric* counterpart of the `optics` module (which covers the spatial
PSF/MTF response): the physics that sets the smallest temperature difference an
EO/IR sensor can see.
- **Radiometry** — Planck's law (`planck_radiance`, `planck_radiance_dt`),
  Stefan–Boltzmann exitance `M = σT⁴` (`radiant_exitance`) and its derivative
  `4σT³`, Wien's displacement law `λ_peak = b/T` (`peak_wavelength`), and
  in-band integrated radiance / **thermal contrast** by quadrature
  (`band_radiance`, `thermal_contrast`).
- **Sensitivity** — **`netd`** (noise-equivalent temperature difference: the
  ΔT giving a signal equal to the detector noise)
  `NETD = 4F²√Δf / (π√A_d·τ_o·D*·(∂L/∂T)_band)`, and **`mrtd`** (minimum
  resolvable temperature difference) `MRTD = k·NETD/MTF`, a thermal-
  sensitivity / resolution trade-off that combines the NETD with the MTF of
  `optics`.
- Oracles: the Planck integral over the whole spectrum × π **recovers σT⁴**;
  exitance and derivative at the closed forms (and finite difference); the Wien
  peak **shifts as 1/T** and the Planck curve peaks there; analytical ∂L/∂T =
  finite difference; thermal contrast is positive and **grows with
  temperature**; the NETD **follows its scaling laws** (∝ F², 1/D*, 1/contrast,
  1/√A_d); the MRTD **diverges when the MTF vanishes**. 7 tests (54 total for
  the crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-vision`: optronics — CFAR detection of small targets (`detect`) — block 23
Bridge between optronics and tracking: the **image-side** CFAR detector, the
EO/IR analogue of the radar CFAR, which extracts small hot targets from a
variable thermal background and reduces them to centroids ready to feed the
tracking chain.
- **`cfar_mask(image, guard, train, k)`** — detection mask: for each pixel,
  estimates the local background over a **ring of training cells** around a
  **guard band** (so that a target cannot corrupt its own background estimate)
  and marks the pixel if it exceeds the local mean by `k` local standard
  deviations. Since the threshold follows the *local* statistics, a target is
  found on a dark sky as on a bright pedestal.
- **`detect_targets(image, guard, train, k)`** / **`TargetDetection`** — groups
  the thresholded pixels by connected components (`connected_components`) and
  reduces them to **intensity-weighted** centroids (sub-pixel position, peak
  amplitude, pixel count).
- Oracles: detects a point target on a flat background; no detection on a
  uniform background; detects a target **on a bright pedestal** (the CFAR
  follows the local level); the weighted centroid is **sub-pixel**; resolves
  two separated targets; a higher `k` threshold **does not grow** the false
  alarms (noisy background). 6 tests (47 total for the crate); `fmt`/`clippy
  -D warnings` clean.

### Added — `scirust-signal`: radar — multi-target tracking with NIS gate (`radar::mtt`) — block 22
Closing the loop of the tracking chain: an end-to-end **multi-target** tracker on
**polar** measurements, one `RadarEkf` per track, with **statistical gate**
association.
- **`RadarMultiTracker`** / **`RadarTrack`** — each track is an extended Kalman
  filter (block 21) fed with range/azimuth; the association gates each
  (track, measurement) pair by the **normalized innovation squared** (NIS,
  Mahalanobis distance) compared to a χ² quantile (2 d.o.f.), so the gate
  tightens or widens according to each track's own uncertainty instead of a
  fixed radius. Each frame: prediction of all tracks, NIS gating, greedy
  nearest-neighbor association, update of the matched tracks, coasting of the
  others, birth of a track for each unassociated measurement
  (polar→Cartesian), death beyond `max_misses`.
- **`RadarEkf::nis(...)`** — new method exposing `yᵀ·S⁻¹·y`, the gating
  statistic, without mutating the filter.
- Oracles: tracks a single target (position at truth, stable id); keeps two
  separated targets distinct; the **NIS gate rejects clutter** (the true
  track coasts uncontaminated, the clutter spawns its own track);
  birth then death of a lost track; the NIS is **small on target and large
  off target**; empty frames inert. 6 tests (194 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — polar extended Kalman filter (`radar::ekf`) — block 21
Tracking from **real radar measurements**: the previous trackers assume a
Cartesian position already available, whereas a radar provides range and
azimuth (polar measurement, a nonlinear function of the state).
- **`RadarEkf`** — **extended** Kalman filter on the Cartesian state
  `[x, vₓ, y, v_y]`: the prediction stays linear (constant velocity, hence
  exact), and the correction linearizes the polar observation
  `h(x) = (√(x²+y²), atan2(y, x))` via its Jacobian around the current estimate.
  The azimuth innovation is **wrapped** into `(−π, π]` so that a target
  crossing the ±π boundary is tracked without discontinuity. Range and azimuth
  each have their own measurement variance.
- Reuses the dense matrix utilities of `radar::imm2d` (product, transpose,
  Cholesky), now `pub(super)`. No dependency.
- Oracles: `wrap_pi` maps into the principal interval; the EKF **recovers a
  Cartesian trajectory from polar measurements** (position and velocity at
  truth); **tracks a target crossing the azimuth fold** (along −x) without
  losing lock; the update reduces the position variance; the predicted
  measurement matches the state; guard at the origin (undefined azimuth →
  inert update). 6 tests (188 total for the crate); `fmt`/`clippy -D warnings`
  clean.

### Added — `scirust-signal`: radar — 2-D coordinated-turn IMM (`radar::imm2d`) — block 20
Tracking **maneuvering** targets in the plane: where the 1-D IMM (block 19)
approximates a maneuver by inflating the process noise of a constant-velocity
model, this block adds a true **coordinated-turn** model that rotates the
velocity vector at a constant angular rate ω.
- **`KalmanLinear`** — general linear Kalman filter with `n` states / `m`
  measurements (dense matrices, Cholesky-factorization update, Gaussian
  innovation likelihood). Reusable well beyond tracking.
- **`cv_model_2d`** / **`ct_model_2d`** — the two planar motion models on the
  Cartesian state `[x, vₓ, y, v_y]`: quasi-constant-velocity and coordinated
  turn at fixed rate ω (degenerates toward CV when ω→0).
- **`Imm2D`** — Interacting Multiple Model estimator over a bank of these
  models (typically CV + one or two CT at ±ω): in a straight line the CV model
  wins, as soon as the turn starts the matching CT model takes over and the
  tracking follows the arc instead of cutting across it. No dependency.
- Oracles: CT degenerates toward CV when ω→0; the linear Kalman recovers the
  2-D constant velocity; the CT filter **tracks a circular trajectory much
  better than a CV filter** (error < half); the IMM **selects the turning model
  and beats a CV-only filter on a maneuver** (CT-mode probability rising);
  valid mode probabilities; Cholesky solves a known system and rejects a
  non-positive-definite matrix; empty-bank guard. 7 tests (182 total for the
  crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — Kalman filtering / IMM (`radar::kalman`) — block 19
Scaling up tracking: the fixed-gain α–β filter is complemented by an adaptive
Kalman filter and a multi-model estimator for maneuvering targets.
- **`KalmanCV`** — constant-velocity Kalman filter on the state `(p, v)` with
  explicit `2×2` covariance: continuous white-acceleration process-noise model
  `Q = q·[[dt³/3, dt²/2],[dt²/2, dt]]`, adaptive gain, state variance and
  innovation likelihood exposed.
- **`Imm`** — **Interacting Multiple Model** estimator: a bank of Kalman
  filters (typically a "calm" low-noise model and an "agile" high-noise model)
  mixed at each frame by Markovian mode probabilities. In steady flight the
  calm model dominates (smooth, low variance); as soon as the maneuver starts
  the agile model's likelihood prevails and takes over, following the maneuver
  with far less lag than a fixed model. No dependency (2-D state, all in
  closed-form `2×2`).
- Oracles: Kalman recovers constant velocity, the update reduces the variance
  toward a steady state, the likelihood is maximal at the prediction; the IMM
  mode probabilities form a valid distribution and **favor the calm model on a
  stable target**; the IMM **beats a calm-only filter through a maneuver**
  (smaller error, rising agile-model probability); empty-bank guard. 7 tests
  (175 total for the crate); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar — ESPRIT direction finding (`radar::esprit`) — block 18
**Gridless** angle-of-arrival estimation: complement of MUSIC which, instead of
sweeping a spectrum, reads the angles directly from the eigenvalues.
- **`esprit_doa(snapshots, spacing, num_sources)`** — exploits the rotational
  invariance of a uniform linear array: the two subarrays (first and last `M−1`
  sensors) see the same wavefronts up to a phase shift `e^{jμ}`, with `μ =
  2π·spacing·sin θ` the phase step of the steering vector. The matrix `Ψ`
  relating their signal subspaces (least squares `E₁Ψ = E₂`) is similar to
  `diag(e^{jμ_k})`; its eigenvalues give `sin θ_k = μ_k / (2π·spacing)`,
  returned angles sorted.
- Reuses the Hermitian eigensolver of `radar::music` for the subspace, plus a
  **complex eigensolver written from scratch** (Hessenberg reduction by Givens
  rotations then Wilkinson's **shifted QR algorithm**) for the non-Hermitian
  matrix `Ψ`. No dependency.
- Oracles: the eigenvalues of a triangular matrix are its diagonal and those of
  a rotation stay on the unit circle; ESPRIT recovers a single source and
  **resolves two off-grid sources** to better than 0.5°; guards (array < 2
  sensors, singular system → empty). 5 tests (168 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-vision`: optronics — Wiener deconvolution (`optics`) — block 17
Image restoration in the **frequency domain**, complement of the spatial
Richardson–Lucy deconvolution.
- **`wiener_deconvolution(blurred, psf, nsr)`** — Wiener filter
  `F̂ = 𝔉⁻¹[ conj(H)/(|H|² + nsr)·G ]`: `G`/`H` are the transforms of the blurred
  image and of the PSF (mean-preserving, centered at the origin with circular
  wrap-around), `nsr` the noise-to-signal ratio that regularizes the inverse.
  Built on a **separable 2-D FFT** (FFT rows then columns via `scirust-signal`,
  power-of-two dimensions).
- Oracles: a known circular convolution is **inverted exactly** by Wiener with
  vanishing regularization (scene→blur→restoration round-trip at 1e-5); a Dirac
  PSF is the identity; guards (non-power-of-two dimensions, PSF larger than the
  image → empty image). 3 tests (41 total for the crate); `fmt`/`clippy -D
  warnings` clean.

### Added — `scirust-sim`: optoelectronics — avalanche photodiode APD (`apd`) — block 16
The high-sensitivity optoelectronic receiver (lidar, laser ranging), which
completes the photodiode: an avalanche gain `M` amplifies the signal but adds
**excess noise** (McIntyre factor).
- **`excess_noise_factor(M, k)`** — McIntyre factor `F(M) = k·M + (1−k)(2−1/M)`.
- **`Apd`** / **`ApdParams`** — primary / multiplied / signal current, excess
  noise, shot-noise variances (`2q·I·M²·F·B`) and thermal
  (`4k_B·T·B/R_L`), and the signal-to-noise ratio `SNR = I_s²/(σ²_shot +
  σ²_thermal)`.
- Oracles: `F(1) = 1` for any `k`, `F → 2−1/M` (k=0) and `F = M` (k=1),
  monotone; currents and variances at the closed forms; the **SNR goes through
  an optimum** (gain 50 better than 1 — thermal-limited — and than 1000 —
  excess-noise-limited); rejection of invalid parameters (gain < 1, `k ∉
  [0,1]`). 4 tests (118 total for the crate); `fmt`/`clippy -D
  warnings`/`miri` clean.

### Added — `scirust-sim`: optoelectronics — photodiode / photodetector (`photodiode`) — block 15
The optoelectronic receiver, complement of the laser (transmitter): optical
power → photocurrent → voltage limited by an `RC`, like `System` of
`scirust-sim`.
- **`Photodiode`** / **`PhotodiodeParams`** — state `[v]`:
  `v' = (I_ph − v/R_L)/C_j` with `I_ph = ℛ·P_opt + I_dark`.
- **`responsivity(η, λ)`** — spectral sensitivity `ℛ = η·q·λ/(h·c)` (A/W),
  linear in wavelength.
- Closed-form quantities: `photocurrent`, `steady_state_voltage` (`I_ph·R_L`),
  `time_constant` (`R_L·C_j`), `bandwidth` (`1/(2π·R_L·C_j)`).
- Oracles: the sensitivity follows the closed form and grows linearly with λ;
  closed-form photocurrent / steady-state voltage / bandwidth; the dark
  current sets the floor without light; the step response charges with the
  `RC` time constant (`v(τ) = v_ss(1−1/e)`, then `v_ss` at 10τ); rejection of
  invalid parameters. 5 tests (114 total for the crate); `fmt`/`clippy -D
  warnings`/`miri` clean.

### Added — `scirust-vision`: optronics — Airy PSF / diffraction limit (`optics`) — block 14
Continuation of the precision image-processing strand: the **Airy PSF**, the
response of a diffraction-limited system (circular aperture), which completes
the Gaussian PSF.
- **`airy_psf(size, first_null)`** — normalized Airy PSF (`Σ = 1`), intensity
  `[2·J₁(v)/v]²` with the first dark ring at `first_null` pixels (reuses
  `scirust_special::bessel_j`).
- **`rayleigh_resolution(λ, D)`** — Rayleigh angular resolution `θ = 1.22·λ/D`.
- **`airy_first_null(λ, D, f, pitch)`** — radius of the first ring in pixels on
  a focal plane `1.22·λ·f/(D·pitch)`, the argument of `airy_psf`.
- Oracles: normalized, symmetric PSF with a central peak; the first dark ring
  falls exactly at radius `first_null` (zero of J₁); the closed forms of
  Rayleigh and of the ring radius. 3 tests (38 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-sim`: optoelectronics — semiconductor laser rate equations (`laser`) — block 13
Third optronics pillar — **optoelectronics** (device dynamics) — via the
single-mode rate equations of the semiconductor laser coupling the carrier
density `n` and the photon density `s`, as `System` of `scirust-sim`.
- **`SemiconductorLaser`** / **`LaserParams`** — state `[n, s]`:
  `n' = J − n/τ_n − g₀(n−n_t)s`, `s' = Γg₀(n−n_t)s − s/τ_p + Γβ·n/τ_n`.
- Closed-form quantities (limit `β→0`): `threshold_density`
  (`n_th = n_t + 1/(Γg₀τ_p)`), `threshold_pump` (`J_th = n_th/τ_n`),
  `steady_state_photon_density` (linear light-current law
  `s_ss = Γτ_p(J−J_th)`), `steady_state_carrier_density` (gain clamping at
  `n_th`), `relaxation_frequency` (`f_r = √(g₀s_ss/τ_p)/2π`).
- Oracles: threshold and linear L-I curve (closed forms); turn-on converges
  to the closed stationary state; below threshold the laser stays off; a
  small perturbation rings at the relaxation oscillation frequency (period
  measured on the trajectory); spontaneous emission `β>0` ignites the turn-on
  without a photon seed; rejection of invalid parameters. 7 tests (109 total
  for the crate); `fmt`/`clippy -D warnings`/`miri` clean.

### Added — `scirust-vision`: ray optics — ABCD matrices + Gaussian beams (`beams`) — block 12
Second optronics block: the design of the **optical train** (lenses, mirrors,
free space) by **ABCD transfer matrices**, and the propagation of a **Gaussian
beam** via the complex parameter `q`.
- **`RayMatrix`** — 2×2 matrix `[[a,b],[c,d]]` acting on a ray `(y, θ)`:
  constructors `identity` / `free_space` / `thin_lens` / `curved_mirror` /
  `flat_interface`, composition `then` (product `next·self`), `determinant`
  (`= n_in/n_out`), `apply`.
- **Gaussian beams** — `rayleigh_range` (`z_R = πw0²/λ`), `beam_radius`
  (`w(z) = w0√(1+(z/z_R)²)`), `radius_of_curvature`, `divergence` (`λ/(πw0)`),
  `gouy_phase`, plus the `q` parameter: `q_at_waist`, `propagate_q`
  (`q' = (Aq+B)/(Cq+D)`), `beam_radius_from_q`, `radius_from_q`.
- Oracles: unit determinant for lossless elements (and `n1/n2` for a plane
  diopter); a collimated ray focuses at `f`; the imaging condition zeroes `B`
  and gives the magnification `−si/so`; beam geometry (waist `w0`, √2 at
  `z_R`, divergence, Gouy phase); the free-space `q` reproduces `w(z)` and
  `R(z)`; a lens reforms a waist at the predicted Gaussian plane
  `s' = f·z_R²/(f²+z_R²)`. 8 tests (37 total for the crate);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-vision`: optronics — PSF, MTF and deconvolution (`optics`) — block 11
First block of the **optronics / precision optics / image processing** strand:
image quality and restoration of an EO/IR imager, in the existing vision
crate.
- **`gaussian_psf`** — normalized Gaussian point spread function (PSF)
  (`Σ = 1`), odd size.
- **`apply_psf`** — direct optical blur (image ⊛ PSF convolution), reuses
  `convolve2d`.
- **`line_spread`** / **`mtf`** — line spread function (LSF) then **modulation
  transfer function** (MTF): normalized modulus of the LSF's DFT
  (`MTF[0] = 1`), frequencies in cycles/pixel, direct DFT without power-of-two
  constraint.
- **`mtf50`** — resolution metric: frequency where the MTF drops to 0.5
  (interpolated), the key figure of a precision optic.
- **`richardson_lucy`** — Richardson–Lucy deconvolution (iterative spatial
  restoration, convolutions only; conserves flux, stays positive).
- Oracles: the PSF is normalized / symmetric / peaked at center; the MTF of a
  Gaussian follows the closed form `exp(−2π²σ²f²)` and decays monotonically;
  the `MTF50` follows `√(ln2/2)/(πσ)`; Richardson–Lucy is the identity for a
  Dirac PSF and re-concentrates a blurred point (peak restored, flux
  conserved). 7 tests (29 total for the crate); `fmt`/`clippy -D warnings`
  clean.

### Added — `scirust-signal`: tracking — α–β filter + multi-target tracker (`radar::track`) — block 10
Tenth block of the radar domain: the **temporal layer** that associates the
detections of a frame with the existing tracks and smooths each state. Closes
the detection → track chain.
- **`AlphaBeta`** — scalar α–β filter for a constant-velocity state `(x, v)`:
  `predict` (`x + v·dt`), `update` (predicts then corrects by `α·residual` /
  `(β/dt)·residual`), `coast` (advances without a measurement). Unbiased with
  zero steady-state position error on a ramp.
- **`critically_damped_gains(θ)`** — critical gains `α = 1 − θ²`, `β = (1 − θ)²`.
- **`Track`** — 2-D track (one `AlphaBeta` per range/Doppler coordinate)
  consuming a `Detection`, with hits/misses lifecycle and amplitude.
- **`MultiTracker`** — nearest-neighbor multi-target tracker: at each `step`,
  prediction of all tracks, greedy association of the detections to the
  nearest predicted track within a gate, update of the associated tracks,
  extrapolation of the others, birth of tracks for orphan detections, death
  after `max_misses` extrapolated frames.
- Oracles: the critical gains follow the closed form; the α–β filter tracks a
  constant-velocity ramp without lag (exact position and velocity); the
  extrapolation advances by one velocity step; the tracker tracks a single
  target (exact estimated velocity), keeps two distant targets separate
  (stable ids), creates then removes a lost track. 6 tests (163 total);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: 2-D detection — 2-D CFAR + clustering (`radar::detect`) — block 9
Ninth block of the radar domain: the **detection stage** that turns the
range-Doppler map into a short list of targets — exactly the chain of the
reference projects (OpenRadar): 2-D CFAR → clustering.
- **`ca_cfar_2d`** — cell-averaging CFAR on a **power** map
  `power[distance][doppler]`: for each tested cell, the threshold is
  `α · mean(training cells)` over the square window of half-width
  `train + guard` minus the guard region `(2·guard+1)²`, with `α` from the same
  `ca_cfar_alpha` as the 1-D detector (`N = (2(train+guard)+1)² − (2·guard+1)²`).
  Boolean detection mask; edges never marked.
- **`Detection`** / **`cluster_detections`** — grouping of the detections into
  targets by connected-component labeling (8-connectivity), amplitude-weighted
  centroid (fractional bin coordinates), peak amplitude and cell count; sorted
  by descending peak.
- Oracles: the 2-D CFAR detects a point target on a flat floor (a single
  detection) and holds the design **false-alarm rate** on 2-D exponential
  noise; the clustering separates two distant blobs (exact weighted centroids,
  strongest first), merges diagonally touching cells; the CFAR-2-D →
  clustering chain localizes two targets; guards (empty mask / inconsistent
  shapes / too-small map). 6 tests (157 total); `fmt`/`clippy -D warnings`
  clean.

### Added — `scirust-signal`: subspace direction finding MUSIC (`radar::music`) — block 8
Eighth block of the radar domain: the **MUSIC subspace method** (MUltiple SIgnal
Classification), which closes the *angle* path. Where the MVDR still refines
the beamformer, MUSIC decomposes the covariance into signal / noise subspaces
and exploits the orthogonality of each source steering vector to the noise
subspace — resolution limited by the number of snapshots and the SNR, not by
the array aperture.
- **`music_spectrum`** — spectrum `P(θ) = 1/‖Eₙᴴ·a(θ)‖²`: `Eₙ` is the noise
  subspace (eigenvectors of the `M − num_sources` smallest eigenvalues of the
  covariance). Peaks at the source directions. `num_sources` bounded to
  `1..=M-1`.
- Relies on an **in-place complex Hermitian eigensolver** (complex cyclic
  Jacobi rotations: phase annihilation then real Jacobi rotation on the 2×2
  block), without any new dependency — reusable for ESPRIT.
- Oracles: the eigensolver reconstructs `A = V·diag(λ)·Vᴴ`, `Vᴴ V = I`, real
  eigenvalues with correct trace; MUSIC peaks at the single source; **two
  sources at 6°** (below the ~11° beam) are **resolved** (null at the median);
  degenerate inputs (empty / single-element array) → null spectrum. 4 tests
  (151 total); `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: FMCW / mmWave radar (`radar::fmcw`) — block 7
Seventh block of the radar domain: the **FMCW** model (frequency-modulated
continuous wave) — that of automotive / mmWave radars (TI, OpenRadar). Instead
of compressing a coded pulse by matched filtering, the FMCW *mixes* the echo
with the transmitted ramp; range and velocity fall out of two FFTs of the
beat signal.
- **`beat_frequency_to_range`** — range from the beat frequency
  `R = f_b·c / (2·slope)` (guard on non-finite / negative slope).
- **`range_resolution`** — range resolution `ΔR = c / (2·B)`.
- **`range_profile`** — range profile of a ramp: fast-time FFT of the complex
  beat (power-of-two guard). A target at range `R` peaks at the bin of its
  beat frequency.
- **`range_doppler`** — **range-Doppler cube** from the raw beat ramps:
  fast-time FFT (range) of each ramp then slow-time FFT (Doppler) of each
  range bin; `N×M` map `[distance][doppler]`, Doppler bin 0 = stationary
  target. Does not overlap `doppler::range_doppler_map` (which assumes the
  pulses already range-compressed): here one starts from the raw beat and
  performs both FFTs.
- Oracles: the range profile of a beat tone peaks at the right bin with
  coherent amplitude integration `N`; the range→beat→range round-trip loops; a
  moving target localizes at `(distance, doppler)` with an amplitude peak
  `N·M`, a stationary target stays at Doppler 0; non-power-of-two /
  irregular inputs → empty. 6 tests (147 total); `fmt`/`clippy -D warnings`
  clean.

### Added — `scirust-signal`: high-resolution direction finding MVDR/Capon (`radar::doa`) — block 6
Sixth block of the radar domain: **high-resolution** DOA, which separates
sources closer than the beamwidth of the array.
- **`covariance`** — spatial covariance matrix `R = (1/T)·Σ x·xᴴ` (M×M
  Hermitian) of the array snapshots.
- **`mvdr_spectrum`** — **MVDR / Capon** beamformer:
  `P(θ) = 1/(aᴴ(θ)·R⁻¹·a(θ))`, diagonal loading for stability. Far more
  resolving than the conventional (Bartlett) beamformer. Relies on an
  **in-place complex matrix inverse** (Gauss-Jordan with partial pivoting),
  without any new dependency.
- Oracles: the MVDR peaks at the source direction; **two sources at 6°**
  (within the ~11° beam of a 10-element array) are **resolved** by the MVDR —
  the midpoint is a null between two peaks — whereas the Bartlett merges
  them; the covariance is indeed Hermitian. 3 tests (141 total);
  `fmt`/`clippy -D warnings` clean. MUSIC (noise subspace by
  eigendecomposition) will come in a later block.

### Added — `scirust-signal`: antenna array processing / radar direction finding (`radar::beamform`) — block 5
Fifth block of the radar domain: the **multi-channel** step — estimation of the
direction of arrival (DOA) on a uniform linear array (ULA), the *angle* path
that completes the range-Doppler chain (direction finding of the two reference
projects).
- **`steering_vector`** — steering vector `a(θ)` of a ULA (`exp(j·2π·d·m·
  sin θ)`): unit magnitude, all-ones at broadside (`θ = 0`).
- **`beamform_spectrum`** — conventional (delay-and-sum / Bartlett) beamformer:
  average power `|aᴴ(θ)·x|²` over the snapshots, a spatial spectrum whose
  peaks give the source directions.
- **`estimate_doa`** — angle of the spectrum peak (single-source DOA
  estimate).
- Oracles: a plane wave from `θ0` → beamformer peak at `θ0` (at grid
  resolution); two separated sources each stand out above an empty direction.
  No new dependency. 4 tests (138 total); `fmt`/`clippy -D warnings` clean.
  The high-resolution estimators (MVDR/Capon, MUSIC) will reuse these steering
  vectors in a later block.

### Added — `scirust-signal`: ambiguity function + MTI radar — block 4
Fourth block of the radar domain: waveform analysis and rejection of
stationary clutter.
- **`radar::ambiguity::ambiguity`** — ambiguity surface `|χ(τ, ν)|` (joint
  delay-Doppler response) of a waveform, computed by cross-correlation of the
  Doppler-modulated waveform with the original. Exposes the **range-Doppler
  coupling** (diagonal ridge of the LFM chirp). Oracles: peak at the origin =
  energy and global maximum; the zero-Doppler cut **equals the autocorrelation**
  (matched-filter cut); the LFM ridge is **sheared** (the peak delay varies
  monotonically with Doppler).
- **`radar::mti::mti_canceller`** — MTI canceller with `order` pulses
  (cascaded first differences, binomial weights `[1,−1]`, `[1,−2,1]`, …).
  Response **exactly null at DC** → stationary clutter removed, moving target
  passed with gain `|1−e^{−j2πf}|^order`. Oracles: constant clutter → exact
  zero; moving tone passed with the binomial gain; clutter removed / target
  kept.
- Reuses `cross_correlate` from block 1, no dependency. 8 tests (134 total);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar Doppler processing (`radar::doppler`) — block 3
Third block of the radar domain: the **range-Doppler map**, the surface on
which the CFAR detects (common core of the two reference projects).
- **`doppler_spectrum`** — slow-time FFT of a range bin (one complex sample
  per pulse); bin 0 is zero Doppler (stationary target / clutter).
- **`range_doppler_map`** — from a stack of `M` range-compressed pulses,
  slow-time FFT per range bin → `N×M` magnitude map
  `[distance][doppler]`. Separates moving targets from stationary clutter.
- Oracles: a **stationary** target falls in Doppler bin 0 with coherent
  integration (magnitude = M); a **moving** target (phase ramp of k₀ cycles
  over the M pulses) falls in bin k₀ (up to the FFT sign) and **not** at zero
  Doppler; rejection of non-power-of-2 / irregular inputs.
- Built on the crate's existing FFT, no dependency. 3 tests (126 total);
  `fmt`/`clippy -D warnings` clean.

### Added — `scirust-signal`: radar CFAR detection (`radar::cfar`) — block 2
Second block of the radar domain: **constant false-alarm rate detection**, the
detection link of the processing chains of the two reference projects
(OpenRadar, AERIS/plfm_radar).
- **`ca_cfar`** — cell-averaging CFAR: threshold `α · mean(reference cells)`
  with the closed-form factor `α = N·(P_fa^{−1/N} − 1)`, guard cells around
  the tested cell. Returns a detection mask.
- **`os_cfar`** — order-statistic CFAR (k-th smallest reference cell),
  **robust to interfering targets** and to clutter edges that would mask the
  CA-CFAR; factor `α` found by bisection on
  `P_fa(α) = ∏ (N−i)/(N−i+α)`.
- Oracles: CFAR identity `(1 + α/N)^{−N} = P_fa`; **false-alarm rate held
  statistically** (exponential noise, 20 000 cells, empirical ≈ P_fa); the
  OS-CFAR detects a target that the CA-CFAR masks because of an interferer in
  the window; `α_os` correctly inverts the P_fa formula.
- No new dependency. 6 tests (123 total in the crate); `fmt`/`clippy -D
  warnings` clean.

### Added — `scirust-signal`: radar signal processing (`radar` module) — block 1
First block of a **radar/optronics** domain (useful for defense systems of the
Safran/Sagem type), extending the existing crates: pulse compression, the core
of the range processing of a pulsed-Doppler radar.
- **`radar::waveform`** — waveform generation: **LFM** chirp (linearly swept
  frequency band, unit amplitude, tunable time-bandwidth product) and **Barker
  phase codes** (lengths 2–13).
- **`radar::matched_filter`** — **matched filter / pulse compression**:
  complex cross-correlation, echo delay estimation at the peak, peak-to-
  sidelobe ratio.
- Exact oracles: the chirp's autocorrelation peak **equals the energy** of the
  pulse and its main lobe compresses (width ≈ fs/B ≪ duration); the
  **Barker-13** autocorrelation has a peak-to-sidelobe ratio = **13**
  (defining property); the matched filter **localizes a delayed echo** at the
  correct delay; the chirp's instantaneous frequency indeed sweeps −B/2 →
  +B/2.
- Built on the existing `Complex`/FFT, no new dependency. 8 oracle tests (117
  total in the crate); `fmt`/`clippy -D warnings` clean.
### Added — `scirust-sim`: Van der Pol oscillator (`electrical::VanDerPol`)
The library's first **limit cycle**, natural complement of the chaotic
double pendulum: the two flagship behaviors of nonlinear dynamics.
- **`electrical::VanDerPol`** — self-sustained oscillator
  `x'' - μ·(1 - x²)·x' + x = 0`, state `[x, v]`, implements `System`. The
  nonlinear damping injects energy when `|x| < 1` and removes it when
  `|x| > 1`, so every trajectory (except the unstable fixed point at the
  origin) converges to **the same** stable periodic orbit — unlike a linear
  oscillator whose amplitude depends on the initial conditions.
- Oracles: two trajectories (starting *inside*, near the origin, and
  *outside*) join the same cycle of **amplitude ≈ 2** (classical result);
  `μ = 0` restores the harmonic oscillator (`x(t) = cos t`, energy
  conserved); sign of `dE/dt = μ·(1 - x²)·v²` (pumping/dissipation per the
  unit band). At large `μ` it becomes stiff (integrable via the `stiff`
  feature).
- 4 additional tests (102 total by default, +2 doctests). `fmt`/`clippy
  -D warnings` clean; heavy runs skipped under Miri (crate convention).

### Added — `scirust-biomed`: glycemic dynamics exposed as `System` (feature `sim`)
First example of the **reverse** direction of the simulation layer: instead
of `scirust-sim` redeclaring a vertical's physics, the vertical exposes its
own model through the shared trait.
- **`control::sim::GlucoseSystem`** (behind the optional `sim` feature) —
  wraps the affine glycemic model `dG/dt = -a·(G − G_b) − k·u`
  (`control::GlucoseModel`, the CBF filter's *plant*) with a constant
  insulin rate, and implements `scirust_sim::System`. The `scirust-sim`
  engine (RK4, Dormand–Prince) therefore integrates the vertical's
  physiological model directly, without `scirust-sim` having to redeclare
  it.
- Closed-form oracle: `G(t) = G* + (G0 − G*)·e^{−a·t}` with the steady
  state `G* = G_b − (k/a)·u`. The tests compare the numerical trajectory to
  this exact solution, verify the relaxation to `G_b` at u=0 and the
  vanishing of the derivative at equilibrium.
- Feature **off by default** (default build unchanged; `scirust-sim` pulled
  only under `sim`, no cycle because `scirust-sim` depends on no vertical).
  Dedicated CI steps (`test`/`clippy -D warnings --features sim`), like the
  `rl`/`stiff` features. 44 tests with the feature (+3, +1 doctest);
  `fmt`/`clippy` clean.

### Added — `scirust-mcp`: MCP tool `sim_stiff_robertson` (implicit stiff integrator)
Exposes Robertson's **stiff** kinetics via the implicit Rosenbrock-W(2,3)
integrator of `scirust-sim` (`stiff_bridge` bridge to `scirust-stiff`):
- **`sim_stiff_robertson`** — parameterizable rate constants (canonical
  defaults `k₁=0.04`, `k₂=3·10⁷`, `k₃=10⁴`, nine orders of magnitude apart),
  configurable initial state and horizon. Returns the final concentrations
  `[a, b, c]`, the total mass `a+b+c` (preserved linear invariant) and the
  fraction converted to C. An explicit method (RK4) would blow up on the
  fast initial transient; the implicit one does not.
- `scirust-mcp` now enables `scirust-sim`'s `stiff` feature (pulls
  `scirust-stiff` into the server build only). 6 `sim_*` tools total; +2
  tool tests (Hairer & Wanner oracle at t=0.4: a≈0.9851, c≈0.0149;
  validation), registry assertion. `fmt`/`clippy -D warnings` clean.

### Added — `scirust-sim`: chaotic double pendulum (`mechanics::DoublePendulum`)
The first **chaotic** system of the simulation library, the canonical
example of deterministic chaos:
- **`mechanics::DoublePendulum`** — two masses `m1`/`m2` on two rigid rods
  `l1`/`l2`, state `[θ1, ω1, θ2, ω2]`, accelerations in standard Lagrangian
  form, implements `System` like the other mechanics models.
- `energy` method (kinetic + potential): first integral of motion,
  **conserved to 1e-6** along a chaotic orbit by the adaptive
  Dormand–Prince integrator at tight tolerance — the test oracle.
- **Sensitive dependence on initial conditions** test: two trajectories
  differing by 1e-8 (integrated identically, so physical divergence, not
  numerical) drift apart by O(1) — an amplification > 1e6× that only a
  chaotic system produces.
- 3 additional tests (98 total by default, +2 doctests). `fmt`/`clippy
  -D warnings` clean; heavy runs skipped under Miri (crate convention).

### Added — `scirust-mcp`: MCP tools `sim_hvac_zone` and `sim_pharmacokinetics_oral`
Two new `scirust-sim` simulators exposed as MCP tools (typed JSON schema,
SHA-256 audit log per call), with no extra feature or dependency:
- **`sim_hvac_zone`** — **2R2C** building zone (`scirust-hvac`) driven by a
  constant outdoor temperature and HVAC power: returns the exact linear
  steady state (air `t_out + Q·(R_aw+R_wo)`, wall `t_out + Q·R_wo`), the
  heat-loss conductance `1/(R_aw+R_wo)` (W/K), and the air/wall
  temperatures reached after `duration_s`.
- **`sim_pharmacokinetics_oral`** — first-order oral absorption,
  **one-compartment** body (`scirust-sim`, Bateman curve): returns the peak
  plasma concentration C_max and its time t_max, the elimination half-life
  `ln(2)/k_e`, the **exact** total exposure AUC(0..∞)
  `= F·dose/(V·k_e)`, and the concentration at the end of the horizon.
- 5 `sim_*` tools total (3 → 5); +4 tool tests, updated registry
  assertions. `fmt`/`clippy -D warnings` clean.

### Changed — `scirust-rl-algo`: `AlgoEnv` unified on the shared `Env` trait
The algorithm-discovery crate defined its own environment trait
`AlgoEnv` (with `reset`/`step`/`available_actions`), duplicating
the `scirust_learning::rl::Env` trait already present in a direct dependency.
This duplication is removed:
- **`AlgoEnv` becomes a sub-trait** of `scirust_learning::rl::Env`
  (`AlgoEnv: Env<State = AlgoSearchState, Action = AlgoAction>`). The three
  methods `reset`/`step`/`available_actions` now come from `Env`;
  `AlgoEnv` keeps only its specifics: `observe`, `reward` (correctness/
  efficiency/simplicity decomposition) and `is_terminal`.
- `AlgoSearchEnv` now implements `Env` then `AlgoEnv`; the shared
  tabular / policy-gradient agents of `scirust-learning` apply
  directly to the algorithm-search environment, without a duplicated
  environment abstraction.
- Internal crate change: no public method signature modified,
  47 green tests, `clippy -D warnings` and `fmt` clean.

### Added — a priori proof extended to sin/cos/ln (installment 118)
- **`scirust-core::formal_proof`** (extended): generic toolbox for
  rounding-error propagation `(value, error)` (standard IEEE model,
  always bounded by triangle inequality), replaying the actual
  operation sequence of `sin_poly`/`cos_poly`/`ln_f64_core`. `sin`/`ln`
  lower-bounded via an extracted factor (Jordan for sin, `atanh(s)≥s` for ln,
  structural argument justifying that a single evaluation at the boundary suffices);
  `cos` lower-bounded directly (`cos(0)=1`, same scheme as `exp`); `ln` handled
  in 2 cases (Sterbenz for the x≈1 case — `m−1` without any rounding error).
  Margins obtained: sin ×7.2·10⁷, cos ×4.8·10⁷, ln ×1.4·10⁵ below the
  correctly-rounded threshold. `erf` remains out of the a priori scope (series
  converging on `|y|<4`, non-monotone terms at the start of the series — documented).
- 764 tests (+5), clippy and fmt clean. TCP test on physically
  separate hardware (Jetson + x86-64): infrastructure already complete (installment 117-C),
  remains out of reach of this session (no access to external hardware).

### Added — `scirust-mcp`: `scirust-sim` simulations exposed as MCP tools
An agent can now launch a `scirust-sim` simulation with a simple
MCP tool call (typed JSON schema, SHA-256 audit log per call like
the other tools), without writing any integration code:
- **`sim_epidemic`** — SIR epidemic: returns R0, the infected fraction at the peak
  and the peak day, the final attack rate.
- **`sim_battery_discharge`** — Thévenin 1-RC + thermal battery plant
  (`scirust-bms`) at constant current: final SoC, terminal voltage and
  temperature, steady-state temperature, polarization time constant.
- **`sim_grid_stability`** — machine–grid swing equation
  (`scirust-grid`): existence of a synchronism point, equilibrium angle
  `asin(P_m/P_max)`, small-signal electromechanical frequency, and — if a
  disturbance is provided — verdict on the transient's return to equilibrium.
- `scirust-mcp` now depends on `scirust-sim`; 6 tool tests + the
  registry assertions. `fmt`/`clippy -D warnings` clean.

### Added — `scirust-sim`: industrial vertical plants (battery, HVAC, grid)
The `scirust-bms`/`scirust-hvac`/`scirust-grid` verticals provided
the physics and estimators but no step-by-step simulator; three new
`scirust-sim` modules implement the corresponding `System` trait, tested
against oracle and dependency-free:
- **`scirust-sim::battery`** — Thévenin 1-RC model (SoC, polarization
  overvoltage, self-heating thermal). **Exact** coulomb counting
  (linear invariant, bit-exact RK4), closed-form RC relaxation toward
  `I·R₁`, steady-state temperature `T_amb + P·R_th`, terminal voltage.
- **`scirust-sim::hvac`** — **2R2C** building zone (air + wall mass);
  exact, linear steady state `T_air = T_ext + Q·(R_aw+R_wo)`,
  biexponential relaxation toward the outside when `Q = 0`.
- **`scirust-sim::grid`** — machine–grid **swing equation** (single
  machine–infinite bus), `SecondOrderSystem`. Equilibrium `δ* = asin(P_m/P_max)`,
  small-signal electromechanical frequency `√((ω_s/2H)·P_max·cos δ*)`, transient
  energy conserved without damping, decay toward `δ*` with
  damping, loss-of-synchronism detection (`P_m > P_max`).
- 11 additional tests (94 total, +2 doctests; 97 with the `rl` feature).
  `fmt`/`clippy -D warnings` clean; `cargo miri test` green (heavy runs
  skipped under Miri, `scirust-stiff` convention).

### Added — axis 3 (block-channel) of the QRD-RLS brief: `BlockQrdRls`, block absorption via Householder
The third axis of the Gentleman/McWhirter brief, previously documented as
deliberately deferred, is now delivered — with an honest scope and a
measured benchmark result that does **not** go in the direction hoped for by the brief.

- **`BlockQrdRls`** (new module `block_qrd_rls`) — absorbs a block of `B`
  new samples in a single QR reduction via Householder
  reflectors (Golub & Van Loan, Alg. 5.1.1) on the augmented
  `(n+B)×n` system, instead of `B` sequential Givens rotations. Each
  reflector is applied to the remaining columns of the factor *and* to the `n_out`
  columns of the right-hand side. Zero added external dependencies (dense loops
  written by hand, no BLAS-3 GEMM — see below).
- **Scope clarified in the module docs**: two distinct ideas
  hide behind "multichannel block-channel FQRD-RLS" in the literature.
  (1) block processing of several time samples — that is what
  is delivered here. (2) the "fast" QRD-RLS algorithms with order
  recursion (Cioffi–Kailath, in `O(n)` instead of `O(n²)` per sample) —
  a completely different derivation, **not delivered**, which would require its
  own from-scratch cross-validation rather than a generalization of
  Gentleman's block size. Moreover, even within scope (1), the
  real BLAS-3 gain (compact WY representation, `Q = I − Y·T·Yᵀ`, two
  matrix-matrix products) is **not** implemented — each reflector is
  applied column by column (one dot product + one `axpy` per remaining
  column, BLAS-2 in form), explicitly documented as such rather than
  oversold.
- **Recency weighting inside a block** — derived then verified,
  not assumed: `B` sequential calls to `update()` each scale
  **all** of the existing factor by `√λ` (see the docs of
  `squared_givens`), so `B` grouped samples must reproduce an
  existing factor scaled by `λ^(B/2)` overall, the oldest
  sample of the block by `λ^((B-1)/2)`, the newest by `λ⁰ = 1`.
  Two cross-oracle tests confirm that this construction reproduces
  exactly the sequential absorption: `update_block(..., block_size=1)`
  matches `GivensQrdRls::update` to 1e-6 over 1000 steps, and grouping the same
  stream into blocks of 5 matches one-by-one processing to 1e-6 over 400
  samples. Also cross-verified in MIMO against `SquaredGivensRls`
  (`n_in=3, n_out=2`, blocks of 8, 250 blocks) and on a drifting system.
  5 new tests (60 total on `scirust-estimation`).
- **Measured, not assumed** (x86_64 container, Intel Xeon @2.80GHz, 4 cores,
  `cargo run --bin bench_rls --release`) — the honest result: grouping into
  blocks helps *relative to itself* (`B=64` vs `B=1`: 5.0× faster at
  n=4; 9.0× at n=16; 21.2× at n=64), but **never beats** sequential `SquaredGivensRls`,
  even at `B=64`:

  | n | SquaredGivensRls | BlockQrdRls B=1 | BlockQrdRls B=64 |
  |---|---|---|---|
  | 4 | 34.5 ns | 256.1 ns | 51.2 ns |
  | 16 | 272.8 ns | 3 559.4 ns | 395.1 ns |
  | 64 | 3 447.5 ns | 184 834.7 ns | 8 726.6 ns |

  Explanation: Householder reflectors reintroduce the `√` and the
  `÷` that Gentleman's substitution had eliminated from the hot path of
  `SquaredGivensRls`. The brief hoped for a block-channel throughput gain; the
  measurement says this gain does not exist at these sizes with this formulation,
  and that the real BLAS-3 restructuring (compact WY) — or a port to
  real SIMD kernels — would still be needed to hope to beat axis 1.
  Kept despite this negative result: it is a correct, tested
  implementation of block-channel as requested, and the measurement itself is the
  honest answer to the question posed by the brief.

### Added — fluids & thermo, installment 5: subcritical IF97 regions 3a/3b, backward `p(h,s)` equations
The last two work items explicitly requested on the IF97 backward equations:
- **`scirust-thermo::backward::region3_{v,t}_{ph,ps}`** — the official
  backward equations **`v(p,h)`, `T(p,h)`, `v(p,s)`, `T(p,s)`** of
  region 3, dispatched on the fitted subregions **3a/3b** (boundary
  `h_3ab(p)` for `(p,h)` queries, critical entropy for `(p,s)`).
  Unlike `region3_from_tp` (density bisection, restricted to the
  supercritical regime because `p(ρ)` is not monotone below the critical point), these
  closed-form correlations are valid on **all** of the region-3 domain,
  subcritical included — no density to solve for. Notable discovery while
  writing the verification test: below the critical point, region 3
  is bounded **below by the B23 boundary** and not by the
  saturation curve — `B23(T) < Psat(T)` in this narrow band (623.15 K to
  647.096 K), which grants region 3 a "vapor-like" branch (3b)
  even subcritical, in addition to the classic liquid branch (3a).
- **`scirust-thermo::backward::region{1,2,3}_p_hs`** — official backward
  equation **`p(h,s)`** for regions 1, 2 (2a/2b/2c dispatch via the
  `hab_s(s)` boundary and the threshold `s ≥ 5.85 kJ/(kg·K)`) and 3 (dispatch
  3a/3b via the critical entropy) — pressure directly from the
  thermodynamic state (h,s), without bisection or prior knowledge of T.
- Unchanged methodology: the 14 coefficient groups (32 to 46 terms
  each) were extracted programmatically from the reference Python package
  `iapws`, scanned for any non-integer exponents (none this
  time — the lesson of the previous installment held), then verified in pure Python
  against the 33 official numerical examples of the
  Supp-Tv(ph,ps)3-2014, Supp-PHS12-2014 and Supp-phs3-2014 publications before writing
  the Rust.

### Added — square-root-free QRD-RLS (Gentleman 1973) + McWhirter systolic decomposition
Three research axes proposed to harden/speed up MIMO RLS; two
delivered with proof, a third explicitly deferred rather than rushed.

- **`GivensQrdRls`** — QRD-RLS reference via sequential Givens
  rotations (`√`-based), the **information** form (root of `P⁻¹`), dual
  of the **covariance** form (`QrRls`/Potter) already in the crate. Each
  rotation is an exact orthogonal transformation ⇒ stable by
  construction, without re-symmetrization. Cross-verified against `VectorRls`
  (weights to 1e-6 over 1500 random steps) — a second independent oracle
  for the same least-squares solution.
- **`SquaredGivensRls`** (Gentleman, *"Least squares computations by Givens
  transformations without square roots"*, 1973) — the same recursion without
  any square root: each triangular row is stored as a weight
  `d_i` and a normalized vector `t_i` (`t_i[i]=1`), and the full substitution
  of the Givens computation makes all the `√` disappear (full derivation at
  the head of the module). Bonus: the `√d_i` scale of each row cancels in
  the normal equations, so weight extraction by back substitution
  requires **neither `√` nor division** (unit diagonal). Native MIMO
  (`n_in`/`n_out`), zero heap allocation.
  **Verified, not just plausible**: the reconstructed physical `R`
  (`√d_i·t_i`) matches the `√`-based `R` of `GivensQrdRls` to 1e-6 over 1500
  steps — proof that the square-root-free derivation is exact, not only
  numerically lucky; MIMO version cross-verified against `RlsFilter`.
  Two derivation bugs were caught *by these very tests* before any
  commit: the `R(0)` initialization convention (root of the **information**,
  hence `1/√delta`, not `√delta`) and the weight `d_in` of the incoming residual, which
  **changes at every row** and must be propagated rather than reset —
  exactly the kind of silent error that a derivation "copied from memory of a paper"
  would have let through without an oracle to detect it.
  **Measured** (x86_64 container, Intel Xeon @2.80GHz, 4 cores, `cargo run
  --bin bench_rls --release`): faster than `VectorRls` at all
  sizes (43.3 ns vs 75.0 ns at n=4; 313.9 ns vs 808.0 ns at n=16; 4 402.0 ns
  vs 12 797.4 ns at n=64 — 1.7×–2.9×), and faster than `QrRls` (square
  root) at all sizes too (1.7×–2.2×) — consistent with the
  literature's promise (root eliminated, half as many multiplications).
- **`squared_givens::systolic`** — McWhirter's triangular array
  (*"RLS minimization using a systolic array"*, 1983), made verifiable:
  two pure functions `boundary_cell`/`internal_cell` with
  **nearest-neighbor-only** communication (no cell reads another
  cell's column), which reproduce `SquaredGivensRls::update` **bit for bit** over
  400 random steps — proof that the update genuinely decomposes
  without any data-race risk. Honestly presented as a
  **reference software model** of the flow structure (the natural anchor
  point for a future GPU/FPGA port with wavefront scheduling), not
  as a claim of hardware parallelism realized on CPU.
- **Block-channel axis (multichannel FQRD-RLS, BLAS-3) deliberately not delivered**:
  a real block-channel throughput gain requires a real blocked GEMM (BLAS-3
  level) hooked into the Givens updates — which would break the
  crate's deliberate boundary (zero dependency beyond `serde`) by coupling it to
  `scirust-core`/`scirust-simd`. Documented here as a concrete plan rather than
  silently abandoned: processing `B` samples per block via `B`
  sequential Givens/Squared-Givens passes is already the correct
  block-recursive algorithm (amortizes call overhead, not a real GEMM); the
  real BLAS-3 throughput would require calling on the workspace's existing SIMD
  kernels for the internal updates — out of scope for this batch.
- 6 new tests (55 total on `scirust-estimation`); fmt/clippy
  `-D warnings` clean.

### Measured — scirust RLS vs padasip, same machine (protocol run on Jetson)
The protocol `scripts/bench-rls-padasip.py` + `cargo run --bin bench_rls
--release` ran on a **single machine** — the only mode of comparison
allowed (see the "claims backed by measurements" discipline).

**Machine**: Jetson, L4T R38.4 (`generic` board), kernel `6.8.12-tegra`,
aarch64, 14 cores (`uname -a` / `/etc/nv_tegra_release` / `nproc` — exact
model not captured by these commands; Thor class according to L4T R38 and
the repository's existing Jetson references).

| n | `scirust::VectorRls` | padasip `FilterRLS` | ratio |
|---|---|---|---|
| 4 | 59.7 ns | 8 636.2 ns | **144.7×** |
| 16 | 532.3 ns | 10 640.4 ns | **20.0×** |
| 64 | 7 792.0 ns | 73 349.3 ns | **9.4×** |

The ratio tightens markedly with `n` — expected and
honest behavior, not an artifact: at small `n` the fixed per-call cost of
the Python/NumPy interpreter (object creation, dispatch) dominates; as
`n` grows, NumPy's vectorized BLAS amortizes that cost and makes up
ground on the scalar loop-by-loop Rust implementation. On this same
machine, `QrRls` reproduces the advantage already observed in the x86_64 container (faster
than `VectorRls` from n=16: 433.5 ns vs 532.3 ns at n=16, 6 278.9 ns vs
7 792.0 ns at n=64 — no symmetrization pass). `RlsFilterConst`, on the
other hand, does **not** show here the clear advantage seen in the container (34.0 ns at
n=4, but 7 894.2 ns at n=64, roughly on par with the heap version) —
raw figures recorded without smoothing, the unrolling/vectorization
gap between aarch64 and x86_64 targets remains to be investigated rather than
over-interpreted on a single run.

### Added — RLS level 3: directional forgetting, multi-reference cancellation, `QrRlsConst`, reconditioning
The complete batch validated — principled anti-windup and the closure with the
denoising pipeline:
- **`DirectionalRls`** (directional forgetting, Kulhavý / Cao-Schwartz): forgets
  only in the **excited direction** (rank-1 cut of the information
  matrix `R` along the regressor, matched update of `P` via
  Sherman-Morrison, O(n²)/sample, zero allocation). **Discriminating windup
  test**: 2 000 excited steps on a single direction at λ=0.9 — standard
  RLS sees its orthogonal covariance explode (> 10⁵⁰, λ⁻ᵏ); the
  directional one keeps it **bounded at its initial value**, then re-adapts
  healthily when excitation returns. λ=1 ≡ growing-window RLS (tested
  at 1e-8); drift tracking verified.
- **`reference_noise_cancel` + `wavelet_rls_rts_smooth_multiref`**
  (`scirust-signal::denoise::pipeline`): **convolutive multi-reference**
  noise cancellation — `MimoFirRls` learns online the reference-sensor → primary FIR paths,
  the a priori error IS the cleaned signal;
  chained as stage 0 of the Wavelet–RLS–RTS pipeline. Tests: 2-reference
  convolutive interference removed (> +20 dB vs raw); the
  multi-reference chain beats the blind pipeline by > 6 dB in the presence
  of interference + wideband noise.
- **`QrRlsConst<const N>`**: Potter's square root **on the stack**,
  `core`-only — the ultimate hardened embedded filter (PSD by construction +
  zero heap + compile-time unrolling). **Bit-identical** to the heap `QrRls`
  (aligned accumulation order, tested bit-for-bit over 500 steps).
- **Long-horizon reconditioning**: `QrRls::recondition` /
  `QrRlsMimo::recondition` (re-factorization `S ← chol(S·Sᵀ)`, preserves `P`
  up to rounding, restores triangularity) and
  `DirectionalRls::recondition` (exact `P ← R⁻¹` via the crate's `Mat::inverse`)
  + `consistency_error()` diagnostic. The local Cholesky factorization
  is **verified against the `scirust-solvers` oracle** (deliberate
  dev-dependency: the crate's prod dependencies remain serde-only).
- **`scripts/bench-rls-padasip.py`**: the Python half of the
  cross-library comparison protocol (to be run on the same machine as
  `bench_rls`, e.g. the Jetson) — no cross-library figure is
  claimed until both halves have run on a single host.
- 49 `scirust-estimation` + 93 `scirust-signal` tests green;
  fmt/clippy `-D warnings` clean.

### Added — fluids & thermo, installment 3: IF97 region 5, real Rankine, Hardy Cross ↔ Colebrook
The three remaining "possible follow-ups" of installment 2 are delivered:
- **`scirust-thermo::steam::region5`** — IAPWS-IF97 region 5 (very
  high temperature steam, 1073.15 < T ≤ 2273.15 K, ≤ 50 MPa: gas turbine
  exhaust, ultra-supercritical heat recovery boilers).
  Same ideal Gibbs + residual structure as region 2 (coefficients
  extracted from the same reference package `iapws`, verified in pure Python).
  Oracles: the **official numerical examples of the IF97 publication**
  for region 5 (v, h, u, s, cp, cv, w — six values to 1e-8); physical
  join with region 2 verified at the 1073.15 K boundary.
- **`scirust-thermo::cycles::rankine_real`** — **irreversible** Rankine
  cycle: turbine and pump at real isentropic efficiency
  (`RankineCycleReal`, with the ideal efficiency joined for direct
  comparison). The turbine's real outlet state is located directly
  from its real enthalpy (direct quality inside the dome, or deterministic
  bisection on h if still superheated) — without needing the
  heavy per-subregion IF97 "backward" T(p,h) equations. Verified:
  η_t=η_p=1 reproduces exactly the ideal cycle; at 85 %/85 % the
  real efficiency drops below the ideal and — non-trivial physical fact —
  the exhaust steam becomes drier (higher quality) than at
  the ideal, because a less efficient expansion leaves more enthalpy in
  the steam; first law verified exactly on the real cycle.
- **`scirust-fluids::network::hardy_cross_darcy`** — direct coupling of
  Hardy Cross to `pipe::friction_factor`: `PhysicalPipe` (real diameter,
  length, roughness) instead of a precomputed resistance; at
  each outer iteration, the Darcy friction factor is
  recomputed at the current Reynolds number (laminar/Colebrook-White/mixed,
  exactly as in `scirust-fluids::pipe`), successive substitution
  until convergence — the standard method of real network solvers
  for the (weak) dependence of f on Re. Verified: a wider pipe
  carries more flow, exact continuity preserved, and the
  Darcy-Weisbach head loss recomputed **from the physical dimensions**
  closes the loop to 1e-6 (end-to-end physical validation, not
  only the internal consistency of one iteration).
- Summary: scirust-fluids 57 tests (+3), scirust-thermo 63 tests (+6),
  clippy `-D warnings` clean, rustfmt applied.

### Added — MIMO RLS, the notch above: multi-output QR-RLS, spatio-temporal FIR, auto-λ, Kalman oracle
Improvements to the MIMO RLS filter reusing the repository's building
blocks:
- **`QrRlsMimo`**: the square-root form (Potter factor, PSD by
  construction) extended to multiple outputs — the factor `S` depends only
  on the inputs, so a single recursion is shared across all outputs
  (`O(n_in² + n_out·n_in)`/sample, zero allocation). Tests: row 0
  **bit-identical** to the scalar `QrRls`; 1e-6 equivalence with
  `RlsFilter` on a 2-output system.
- **Crossed RLS ≡ Kalman oracle**: at λ=1 the RLS *is* a Kalman filter with
  static state (`F=I, Q=0, H_k=u_kᵀ, R=1`). New test replaying the same
  trajectory in the crate's `KalmanFilter` (generic matrix path with
  explicit inversion, rebuilt each step with the current `H`) and requiring
  agreement to 1e-8 over 300 steps — two independent implementations of the
  same estimator cross-validate each other.
- **`MimoFirRls`**: the real **spatio-temporal** MIMO adaptive filter —
  delay lines per input channel, regressor stacked on a core `RlsFilter`,
  identified FIR kernel exposed per (output, input) pair. Test: a 2×2
  convolutional coupling with 3 coefficients (crosstalk/echo) is identified
  to 1e-3 on white noise. This is the temporal dimension the instantaneous
  filter was missing.
- **`tune_lambda`**: forgetting-factor choice by **innovation whiteness** —
  each candidate λ is scored by the autocorrelation test `±1.96/√N`, and the
  largest λ the diagnostic does not reject wins (the parsimony rule of
  `denoise::adaptive::kalman_smooth_auto`, reapplied to identification).
  Tests: static system → keeps λ=1; drifting system → rejects λ=1.
- 41 `scirust-estimation` tests green; fmt/clippy `-D warnings` clean.

### Added — a priori formal proof, reproducible FP8, inter-machine TCP (installment 117)
- **`scirust-core::formal_proof`** (new): **a priori** proof (error bounds
  derived analytically, not tested point by point) of correct rounding for
  `exp`/`tanh`/`sigmoid` — Lagrange remainder (Taylor) + Higham's γ_k
  theorem (Horner), in exact rational arithmetic (`num-rational`). Binary
  `proof_formal_bounds`: relative error bound ≈ 2⁻⁴⁷·⁰⁷, margin ≈ 4.4×10⁶
  below the 2⁻²⁵ threshold. Complements (without replacing) the exhaustive
  a posteriori verification of installment 115-A; `sin`/`cos`/`ln`/`erf`
  remain out of scope (cores canceling near zero — documented).
- **`scirust-core --bin proof_fp8_training`**: witness training in
  **FP8 E4M3 with stochastic rounding** (same recipe as the installment-116
  bf16 witness) — `f32_to_fp8_stochastic` (new, `lowprec.rs` refactored
  into `fp8_pre_round`/`fp8_finish` shared with the existing RNE variant).
  Loss trajectory and final FP8 codes under a fingerprint contract,
  bit-reproducible cross-platform (QEMU-validated before commit).
- **`proof_tcp_multihost`** (+ `scripts/proof-tcp-multihost.sh`):
  fixed-tree all-reduce over **TCP between physically separate machines**
  (not just 127.0.0.1) — each rank regenerates its input locally (Philox
  seed+rank) and rank 0 recomputes the reference in-process to compare
  bit-for-bit with the result received over the network: a **self-verifying**
  proof, with no fingerprint to harvest beforehand. Validated
  multi-process (3 and 8 ranks) and in real cross-architecture (one rank
  under `qemu-aarch64` emulation talking TCP with native x86-64 ranks).
- CI gap filled: `lowprec` and `tree_allreduce` were never executed by the
  QEMU `cross-check-aarch64` job (validated manually only) — added, along
  with `formal_proof` and `proof_formal_bounds`.
- 759 tests (+6), clippy and fmt clean.
### Added — Riemann ζ and 5 more discrete distributions (3rd pass of the probabilities installment)
- **`scirust-special::riemann_zeta`/`riemann_zeta_tail`** — ζ(s) for s > 1
  via Euler–Maclaurin at fixed budget (direct sum of the first 19 terms,
  smallest first, + tail at m = 20 with 10 Bernoulli terms), ~1e-15
  relative, validated against `scipy.special.zeta` and the identities ζ(2) = π²/6,
  ζ(4) = π⁴/90, behavior at the pole ζ(s) ~ 1/(s−1) + γ. The **tail**
  `Σ_{j≥m} j^(−s)` is exposed separately: it is what yields an
  O(1) zeta-distribution survival function with no cancellation `ζ − partial sum`.
- **`scirust-stats::discrete`, 5 distributions**: **`Zeta`** (Zipf with infinite support,
  `scipy.stats.zipf` — head summed directly, Euler–Maclaurin tail ⇒
  usable quantile even in the heavy-tailed regime s ≤ 2 where the
  mean diverges); **`PoissonBinomial`** (successes of n **heterogeneous** Bernoulli
  — system reliability, defects per batch; mass vector
  precomputed by the exact O(n²) convolution recurrence, homogeneous case
  = binomial, tested); **`Multinomial`** and **`MultivariateHypergeometric`**
  (vector-valued, dedicated slice API outside the univariate trait: ln_pmf/pmf,
  mean, covariance for the multinomial, deterministic sequential conditional
  sampling — cascaded binomials / hypergeometrics on the remaining urn;
  2-category case = univariate distributions, tested); SciPy oracles
  1.17.1 and exact fraction 280/2001 (`multivariate_hypergeom`).
  45 tests + doctest on the crate, clippy 0 warnings.

### Added — real TCP all-reduce + reproducible bf16-SR training (installment 116)
- **`tcp_tree_all_reduce_rank`** (+ `WireState` trait, little-endian
  serialization of Vec<f32>/Vec<ExactAcc>): fixed-tree all-reduce over
  **real TCP sockets** — same absorption order as the in-process engine,
  hence identical bits (tested under jitter, n ∈ {3, 8}, FixedOrderSum and ExactSum).
  Multi-process/multi-machine ready.
- **`scirust-core --bin proof_lowprec_training`**: control training
  **bf16 with stochastic rounding** (f32 masters, forward copies bf16
  quantized by counter-based SR-Philox, portable graph, Adam) — the
  loss trajectory AND the final bf16 codes under a fingerprint contract,
  bit-reproducible cross-platform (validated under QEMU before commit). Integrated into the
  proof script and the QEMU CI job.

### Added — RLS hardening: zero-allocation, const-generic, square-root QR-RLS, measured benchmarks
The 4 items of the plan validated after the Gemini text review — every
claim in this batch is backed by a test or a measurement:
- **Zero-allocation `update()`** (`RlsFilter`, `VectorRls`): the
  intermediates (`P·u`, a priori error) live in persistent buffers
  (`#[serde(skip)]`, lazy resizing after deserialization); the
  gain is folded on the fly — no more heap allocation per sample
  (the old loop did 4). `RlsFilter::update` now returns
  `&[f64]` (internal view) instead of an allocated `Vec`.
- **`RlsFilterConst<const N_IN, const N_OUT>`** (`rls_const`): fully
  stack-based variant, `core`-only (extractable to `no_std` for
  embedded), compiler-known dimensions ⇒ real unrolling/vectorization.
  **Bit-identical** arithmetic to the heap version — verified by a
  test comparing weight trajectories bit-for-bit over 500 steps.
- **`QrRls`** (`qr_rls`): **square-root** RLS — propagates the factor `S`
  (`P = S·Sᵀ`, Potter rank-1 update, the method family of the in-house
  `UdFilter`). The positive semi-definiteness of the covariance holds **by
  construction** (`xᵀSSᵀx = ‖Sᵀx‖² ≥ 0`), not by forced re-symmetrization —
  the honest answer to the standard RLS divergence risk (no
  claim beyond that: the estimate still depends on excitation, documented).
  Tests: weight-level equivalence (1e-6) with standard RLS on healthy data;
  stress 100 000 steps, λ=0.9, nearly collinear inputs → P finite,
  diagonal ≥ 0, 2×2 leading principal minors ≥ 0; tracking of a drifting system.
- **Measured benchmarks** (`--bin bench_rls`, release, x86_64 CI container —
  figures tied to this machine, to be re-measured elsewhere): ns/update
  `VectorRls` / `QrRls` / `RlsFilterConst`: n=4 → 40 / 47 / 34; n=16 →
  633 / 476 / 326; n=64 → 10 017 / 6 740 / 8 451. Measured findings: the
  const-generic variant is ~2× faster at n=16 (real unrolling), and
  QR-RLS **beats** standard RLS from n=16 on (no symmetrization pass).
  padasip comparison not performed here (installation failure in the
  container) — open point, no cross-library claims.

### Added — fluids & thermo, installment 2: full IF97 (Rankine), convection, networks
Announced continuation of the previous installment — the three "possible follow-up"
work items of the LIVESTATE are delivered:
- **`scirust-thermo::steam` — IAPWS-IF97 regions 1 and 2** (in addition to the
  existing region 4): complete Gibbs equations giving v, h, u, s,
  cp, cv and the speed of sound for **compressed liquid** (region 1,
  273.15–623.15 K, up to 100 MPa) and **superheated vapor**
  (region 2, up to 1073.15 K, bounded by the saturation line and the
  **B23** region 2/3 parabola, also implemented). The 34 + 9 + 43
  coefficients were extracted **programmatically** from
  the reference implementation (`iapws` package) — zero manual
  transcription. Saturated states `saturated_liquid`/`saturated_vapor`.
  Oracles: **official IF97 verification tables 5 and 15**
  reproduced to 1e-8 at the six points (all properties), B23 pair,
  classic steam tables at 100 °C, phase-equilibrium consistency
  (g_f ≈ g_g, h_fg ≈ T·s_fg to ~1e-4, the deviation of the regional fits).
- **`scirust-thermo::cycles::rankine_ideal` — complete Rankine cycle**
  on IF97 properties: isentropic pump (v·Δp), isobaric boiler,
  isentropic expansion (wet exhaust by quality determination inside the dome,
  or superheated by deterministic bisection), isobaric condenser.
  Efficiency, work, heats, exit quality. Oracles: classic
  Cengel example (3 MPa/350 °C/75 kPa → η ≈ 26.0%, x₄ ≈ 0.886),
  exact first law, Carnot bound, physical meaning of the condenser
  pressure.
- **`scirust-thermo::convection`** — external and natural convection:
  laminar flat plate (0.664, = exact Colburn analogy) and mixed
  (continuous junction at Re = 5×10⁵ verified), **Churchill–Bernstein**
  (cylinder in cross-flow), **Ranz–Marshall** (sphere, exact
  conduction limit Nu = 2), **Churchill–Chu** (vertical plate and
  horizontal cylinder in natural convection), Rayleigh number.
  Validity ranges enforced.
- **`scirust-fluids::network` — Hardy Cross method** for
  looped pipe networks: head-loss law
  h = r·|Q|^{n−1}·Q (n = 2 Darcy–Weisbach, 1.852 Hazen–Williams),
  loop corrections preserving node continuity exactly,
  deterministic sweep, flow reversal handled by the signed law.
  Oracles: analytical distributions (2 and 3 pipes in parallel,
  loop closure to ~1e-8), Hazen–Williams exponent, degenerate
  inputs rejected.
- Summary: scirust-fluids 54 tests (+6), scirust-thermo 57 tests (+18),
  clean clippy `-D warnings`, rustfmt applied.

### Added — full correct-rounding proof, fixed-tree all-reduce, low precisions (installment 115)
- **Correct rounding over 100% of the f32 domain** for the 7 portable
  transcendentals: the 465 faulty inputs identified in installment 114 (outputs
  verified by the arbitrary-precision oracle) are served by
  **exception tables** consulted before the analytic path — a result of
  the RLIBM class (CR over the whole domain), obtained by exhaustive verification;
  the machine-checked formal proof of the bounds remains the next step
  (documented). `oracle` category in the certificate; dense/exhaustive
  fingerprints re-collected.
- **`scirust-core::tree_allreduce`**: **fixed-tree** all-reduce,
  transport-agnostic — children absorbed in tree order
  (out-of-order held back) ⇒ timing-independent result; with
  `ExactSum` (Kulisch), also topology-independent and correctly
  rounded. Demonstrated under adversarial jitter (Philox) on n ∈ {2,3,5,8,16}.
  The "multi-node fixed-tree reduction" milestone of the GROWTH_PLAN.
- **`scirust-core::lowprec`**: reproducible bf16/f16/FP8 (OCP E4M3/E5M2) —
  bit-manipulated RNE conversions (portable by construction, exhaustive
  roundtrips 65 536 + 256 codes, exact midpoints), **counter-based
  stochastic rounding** (Philox: reproducible, order-independent,
  unbiased), `gemm_bf16_exact` (exact products, fixed-order accumulation).
  Explicitly out of scope of RepDL.

### Added — 4 additional discrete distributions (`scirust-stats::discrete`, continuation of the lottery installment)
- Fills the gaps vs SciPy listed as "possible follow-ups" in PR #280:
  **`NegativeBinomial`** (failures before the r-th success, `scipy.stats.nbinom`
  convention, real r allowed — Pólya parametrization for
  overdispersed count regression; closed-form CDF via the regularized
  incomplete beta I_p(r, k+1), direct survival without `1 − cdf`),
  **`BetaBinomial`** (binomial with Beta(a, b)-distributed p — overdispersed
  proportions; a = b = 1 recovers the discrete uniform, tested),
  **`Zipfian`** finite over ranks 1..=n (`scipy.stats.zipfian`;
  generalized harmonic normalization summed smallest-terms-first in fixed
  order; s = 0 = uniform; the infinite-support zeta would require Riemann ζ
  and is deliberately not approximated), and **`Skellam`**
  (difference of two Poissons — full ℤ support, hence a clean `i64` API outside
  the u64 trait; pmf/cdf/sf by deterministic convolution with fixed truncation
  on the `scirust-special` base rather than via Bessel I_k, ~1e-12 vs SciPy;
  deterministic sampling = difference of two inverse-CDF Poisson draws).
- Validation: hard-coded SciPy 1.17.1 oracles in the tests (pmf/cdf/sf/ppf,
  moments), invariants Σ pmf = 1, Skellam symmetry at equal rates,
  cdf + sf = 1 on both ℤ tails, r = 1 ⇒ shifted geometric.
  40 unit tests + doctest in total on the crate, clean clippy.

### Added — fluid mechanics & thermodynamics (`scirust-fluids`, `scirust-thermo`)
- **`scirust-fluids`** — deterministic fluid mechanics (pure Rust, zero
  dependency, `forbid(unsafe_code)`, validated inputs → typed `FluidsError`):
  - `dimensionless`: Reynolds, Prandtl, Mach, Froude, Weber, Péclet,
    Strouhal, Nusselt;
  - `pipe`: Darcy friction factors (laminar 64/Re,
    **implicit Colebrook–White** solved by deterministic Newton, Haaland,
    Swamee–Jain), continuous dispatch over the whole Reynolds domain
    (critical zone = documented interpolation), Darcy–Weisbach
    head losses (Δp and head), minor losses, hydraulic
    diameter;
  - `bernoulli`: dynamic/total pressures, Pitot, Torricelli, Bernoulli
    equation between two stations, Venturi and orifice flow metering;
  - `external`: Stokes drag, standard sphere drag curve
    (Clift–Gauvin, Re ≤ 3×10⁵), **terminal fall velocity**
    (deterministic bisection on the force balance);
  - `boundary_layer`: Blasius flat plate (δ, δ*, θ, c_f) and
    turbulent correlations in the 1/7 power law;
  - `compressible`: speed of sound, isentropic ratios (T₀/T, p₀/p,
    ρ₀/ρ, A/A*), **normal shock relations** (M₂, p₂/p₁, ρ₂/ρ₁, T₂/T₁,
    p₀₂/p₀₁);
  - `channel`: Manning equation, critical and normal depths
    (deterministic bisection), specific energy, hydraulic jump
    (Bélanger).
  49 oracle tests: Moody diagram, NACA 1135 tables (exact
  shock fractions at M=2: p₂/p₁ = 4.5, ρ₂/ρ₁ = 8/3), Blasius
  constants, standard drag curve, Colebrook residual ≤ 1e-10, ISA.
- **`scirust-thermo`** — deterministic thermodynamics (same guarantees,
  typed `ThermoError`):
  - `ideal_gas`: calorically perfect ideal gas (state, cp/cv,
    work/heat of isothermal, isobaric, isochoric,
    adiabatic, polytropic processes, entropy variation);
  - `cycles`: Carnot (efficiency + refrigerator/heat-pump COP), Otto,
    Diesel, Brayton (air standard);
  - `heat_transfer`: conduction resistances (plane wall, cylindrical
    shell), convection, radiation (exact CODATA σ), **LMTD**,
    **NTU-effectiveness** (counter-flow and parallel-flow, exact C_r = 0/1
    limits), Dittus–Boelter with **enforced** validity range;
  - `psychro`: ASHRAE moist air (saturation pressure
    **Hyland–Wexler** over ice and liquid water, humidity ratio, dew
    point by deterministic bisection, enthalpy, specific volume);
  - `steam`: water/vapor saturation line **IAPWS-IF97 region 4**
    (p_sat(T) and T_sat(p), mutually inverse closed forms).
  40 oracle tests: official IF97 verification tables (35/36,
  1e-8 agreement), ASHRAE psychrometric tables, classic
  cycle efficiencies, Incropera NTU tables; cross-consistency
  IF97 ↔ Hyland–Wexler (< 0.5%) and zero-entropy Carnot cycle.
- Answer to the observation "scirust offers no solutions to
  fluid mechanics and thermodynamics problems":
  these two crates lay the foundation (exact reference correlations
  and relations) on which CFD/process verticals can build.

### Added — multi-channel RLS + composite Wavelet–RLS–RTS pipeline (integration of PR #278)
- **`scirust-estimation::rls`** (`RlsFilter`, `VectorRls`): multi-channel
  **recursive least squares** adaptive filter (weight matrix `n_out × n_in`
  learned online, gain `k = P·u/(λ + uᵀPu)`, forgetting factor `λ`,
  inverse covariance `P` with **forced symmetrization** `P=(P+Pᵀ)/2` against
  positive-definiteness drift), deterministic `f64`, serializable. Fills the
  crate's "online identification" gap: the Kalman family estimates the state
  of a *known* model, the RLS **learns** the model. Taken as-is from
  PR #278 (developed on a Jetson), with its convergence tests.
- **`denoise::pipeline`** (`wavelet_rls_rts_smooth`, `_1d`): the composite
  pipeline `x̂ = M_RTS·[(I−Δ_RLS)·Wᵀ·𝒯_τ(W·s)]` of PR #278, rebased onto the
  framework's periodized DWT — gaining the Db4/Db6/Db8 bases, arbitrary
  lengths (reflection padding) and robust σ estimation on
  the true fine band (correction of the original's fixed-offset window).
  Soft thresholding applies to *all* coefficients (approximation band
  included), faithful to the original design: the systematic amplitude bias
  thus introduced is precisely what the RLS stage learns and
  corrects — verified by a discriminating test (pipeline > thresholding alone,
  the mutant without stages 2-3 fails); `delta_norm` computed in O(n) with no
  n×n matrix. `scirust-signal` now depends on `scirust-estimation` (no
  cycle: the latter depends only on serde).
- Replaces PR #278 (`denoise.rs` vs `denoise/` module conflict and
  threshold/DWT duplication with the framework merged since); 122 tests
  accumulated on the two crates + 1 doctest; clean fmt/clippy `-D warnings`.

### Added — exact TV (Condat), db6/db8 wavelets, SURE threshold, Kalman with trend (`scirust-signal::denoise`, batch 3)
- **`total_variation_exact`**: **exact** 1-D TV denoising via Condat's
  direct algorithm (IEEE SPL 2013) — the unique global minimizer of
  `½‖x−y‖² + λ·TV(x)` in a single O(n) sweep, without iteration or tolerance.
  Optimality is **proved via the KKT conditions in the tests** (the running
  sum `sᵢ = Σ(xⱼ−yⱼ)` stays within `[−λ,+λ]`, touches `±λ` exactly at
  jumps of the corresponding sign, ends at 0 — strictly convex objective ⇒
  KKT ⇔ global optimum), on 6 varied inputs (steps, sinusoid, pure
  noise, short signal, tiny/huge λ); objective verified ≤ that of
  the existing IRLS approximation; huge λ ⇒ exact flattening to the mean.
- **Daubechies-6 and Daubechies-8 wavelets** (`Wavelet::{Db6, Db8}`, 3 and 4
  vanishing moments): constants derived by spectral factorization and
  **pinned independently by test** to the identities that define them
  (`Σh=√2`, `‖h‖=1`, double-shift orthogonality, vanishing moments of the
  quadrature mirror to ~1e-10); perfect multi-level reconstruction for the
  four bases.
- **Level-wise SURE threshold** (`wavelet_denoise_sure`, SureShrink
  Donoho-Johnstone 1995): minimizes Stein's unbiased risk estimator
  `SURE(t) = m − 2·#{|uᵢ|≤t} + Σmin(uᵢ²,t²)` band by band (prefixes of
  squares over sorted magnitudes, candidates capped at the universal threshold), with
  the "hybrid" fallback to the universal threshold in bands that are too sparse.
  Verified: beats the universal threshold in SNR on a dense signal (two tones) where
  VisuShrink over-smooths.
- **`kalman_trend_smooth`**: Kalman/RTS smoother with **local trend**
  (2-D level+slope state, F=[[1,1],[0,1]]). Where the level-only model
  trades lag against noise on a ramp signal, the trend model
  follows it without bias: discriminating test — a clean ramp is reproduced to
  <1e-3 where the level model (same variances) does >100× worse;
  SNR gain verified on noisy trending signal.
- Module + crate re-exports; 86 unit tests + 1 doctest in total;
  clean `cargo fmt`/`clippy -D warnings`; zero dependency beyond
  `scirust-core`/serde.

### Added — multi-domain simulation environments (`scirust-sim`)
- **`scirust-sim`** — the unified "here is a system, step it forward in
  time, let an agent interact with it" layer the platform was missing:
  - **Deterministic engine**: `System` trait (`y' = f(t, y)`, same in-place
    form as the `scirust-solvers::ode::dopri5` closures) + fixed-step RK4
    (`simulate` → `Trajectory`, linear invariants preserved to
    rounding); `SecondOrderSystem` trait + **symplectic Euler**
    (`simulate_second_order`) — the test shows the two-body orbit staying
    closed where explicit Euler visibly spirals outward.
  - **Agent-in-the-loop layer**: gym-style `Environment` trait
    (`reset` / `step(action) → observation, reward, done`, mirror of
    `scirust-learning::rl::Env`), `run_episode`, **CartPole** (constants of
    the reference implementation, episodes bit-replayable by seed) and
    deterministic **GridWorld**.
  - **Eight domains, all tested against oracle**: mechanics
    (mass-spring-damper vs underdamped closed form, nonlinear
    pendulum with energy conservation at large amplitude, projectile with
    linear drag vs exact solution), orbital (two-body Kepler:
    energy and angular momentum conserved to 1e-9, circular orbit closed
    after exactly one period), epidemiology (SIR/SEIR: population
    conserved to rounding, epidemic threshold at R₀ = 1, exact
    transcendental final-size relation), ecology (Lotka–Volterra
    first integral conserved, logistic closed form), chemistry (consecutive
    reactions vs Bateman solution, reversible reaction relaxing toward
    K = k_f/k_r), thermal (Newton cooling, 1-D rod validated
    on the decay rate of the *discrete* eigenmode and the maximum
    principle), electrical (RC charge, series RLC vs closed form + passivity),
    stochastic/queues (GBM and Ornstein–Uhlenbeck sampled
    by their *exact* transition laws, M/M/1 queue by discrete
    events recovering L = ρ/(1−ρ), W = 1/(μ−λ) and Little's law).
  - **SplitMix64** with explicit seed (published reference vectors verified),
    zero dependency, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`,
    no panic on malformed input (`SimError`), 66 tests + doctest,
    added to the CI's Miri gate.

### Added — the 4 remaining work items of the mapping (installment 114)
- **`scirust-core::philox`** — **counter-based** RNG Philox4x32-10 (Salmon
  et al., SC'11), clean-room from the paper and **validated against the
  published test vectors** of Random123 (+ independent Python
  implementation). Output = pure function of (key, counter) ⇒ dropout/init/
  shuffle parallelizable over any thread partition while
  remaining bit-identical (the JAX-style "order-independent randomness",
  common RepDL/scirust gap). Pure integer arithmetic ⇒ portable by
  construction. 6 tests (published KATs, partition invariance, 4 threads
  bit-exact, statistics, fingerprint contract).
- **`scirust-core::exact_acc`** — **exact** Kulisch accumulator for
  f32 products (704-bit fixed point covering the whole product
  range): `dot_exact`/`gemm_exact` **correctly rounded** (a single
  rounding operation), order-independent, with **associative fusion**
  (multithread bit-exact regardless of partitioning) — the answer to the
  "reproducible and parallelizable GEMM" gap (ReproBLAS class, in fact
  stronger: exact sum). Verified bit-for-bit against the Shewchuk reference
  (two independent constructions of the same rounded real); catastrophic
  cancellation and subnormals handled. 6 tests.
- **`NdVar::rope_portable`** (N-D tape) — RoPE whose frequencies
  (portable `exp`/`ln`) and rotations (portable `sin`/`cos`, Payne–Hanek)
  call no libm: bit-exact cross-platform positional encoding of transformers,
  forward and backward. **`fft_portable`/`ifft_portable`**
  (scirust-signal) — twiddle factors via `sincos_small_f64`
  (Cody–Waite + portable polynomials, new small-argument f64 API of
  `portable_f32`): bit-identical cross-platform spectral analysis.
  Fingerprint contracts committed for both.
- **Correct-rounding certification** (`portable_f32::certify` + `--certify`/`--eval`
  modes of the proof binary): for every f32 input
  (exhaustive sweep 7 × 2³²), an interval certificate proves that the
  published result is THE correctly rounded f32; the internal evaluator
  is re-validated against the shipped function on every input. Non-certified
  inputs are decided offline in arbitrary precision
  (`scripts/verify-certify-offline.py`: Decimal 60 digits, midpoints
  compared in exact rationals — no double rounding). Results of the
  campaign: see LIVESTATE, installment 114.

### Added — Adaptive family, Db4 wavelets, spectral subtraction (`scirust-signal::denoise`, continuation)
- **`Adaptive` family delivered** (`denoise::adaptive`) — the fifth family of the
  taxonomy is no longer "reserved":
  - **`kalman_smooth`**: local-level Kalman filter (random walk +
    white noise) followed by the **Rauch-Tung-Striebel** smoother — bidirectional
    estimation without phase lag.
  - **`kalman_smooth_auto`**: automatic variance tuning by a **sparsity rule on
    the whiteness of the innovations** — a well-specified filter
    produces white innovations, but on a non-random-walk signal
    whiteness grows without bound with `q` (the filter ends up following everything);
    one therefore takes the smallest `q` whose whiteness stays within tolerance of the
    maximum: the smoothest model the diagnostic does not reject. Verified:
    ~17 dB SNR gain on noisy sinusoid where the naive argmax gave only
    ~1.4; follows a level step without smoothing it away.
  - **`lms_line_enhancer` / `rls_line_enhancer`**: adaptive line
    enhancers (predictor on the delayed signal — normalized NLMS, RLS with exponential
    forgetting). Extract a periodic line from broadband noise **without
    external reference or a priori frequency**, and track it if it drifts.
    RLS verified convergent on short recording (~2·taps samples).
- **Daubechies-4 wavelets** (`Wavelet::Db4`): the DWT is refactored into a
  **generic periodized** orthonormal filter bank (`Wavelet::{Haar, Db4}`,
  quadrature mirror `g[j]=(−1)^j·h[K−1−j]`); `wavelet_denoise_with` exposes the
  basis choice, `wavelet_denoise` remains the backward-compatible Haar wrapper.
  Tests: tap orthonormality, **perfect reconstruction** single- and
  multi-level (< 1e-10) for both bases, and Db4 > Haar in SNR on smooth
  signal (2 vanishing moments ⇒ fewer blocking artifacts).
- **Spectral subtraction** (`spectral_subtraction`): subtraction in
  **power** with over-subtraction and spectral floor
  (Berouti-Schwartz-Makhoul 1979, the refinement of Boll's method — not
  its magnitude rule): per-bin gain `√(max(floor², 1 − over·P_n/|X|²))`,
  noisy phase preserved — the classic speech-enhancement front end.
- **NLMS guard rail**: `lms_line_enhancer` refuses `mu ≥ 2` (mean-square
  stability limit — beyond it the output diverged toward ±∞) via
  no-op pass, the module's convention for out-of-range parameters.
- `catalog()` now genuinely covers the five families (`KalmanAuto` added);
  `Denoiser` wrappers for auto Kalman and ALE; crate-level re-exports.
- **Multi-agent adversarial review passed on the diff** (4 dimensions ×
  contradictory verification, mutation testing); confirmed test gaps
  are filled by *discriminating* tests: the RTS backward pass
  is verified by the absence of phase lag (cross-correlation peak at lag 0
  — a mutant reduced to the causal filter fails at lag 8), the spectral
  subtraction `over`/`floor` parameters are pinned by an exact
  identity (huge `over` ⇒ output ≡ `floor·x`), and every `catalog()` entry
  is run with wrapper wiring verification (a `taps`/`delay` transposition
  would compile silently).
- 77 unit tests + 1 doctest in total on the crate; clean `cargo fmt`/`clippy -D
  warnings`; still zero dependency beyond `scirust-core`/serde.

### Added — 100% portable training + portable tanh/sigmoid (installment 113)
- **`Var::{exp_portable, ln_portable, matmul_portable}`**: opt-in autodiff
  primitives whose forward AND backward call no libm nor per-architecture SIMD
  kernel — bit-exact cross-platform (backward: exp from the stored output;
  ln = g⊙1/x IEEE division; matmul via the portable GEMM and transposes).
  `CrossEntropyLoss::new_portable()` switches its internal log-softmax onto it
  (portable loss + gradient; gradient ≡ libm path at 1e-6, frozen fingerprint).
- **`scirust-core --bin proof_portable_training`**: 100% portable witness
  training (MLP 32×16×10, 30 Adam steps, PCG data/init) whose loss trajectory
  and **final weights** are compared to committed fingerprints — identical
  weights bit for bit on any conforming machine. Integrated into
  `scripts/proof-portable-f32.sh` and the QEMU aarch64 CI job.
- **`tanh_f32` / `sigmoid_f32`** in `portable_f32` (shared `exp_f64` core,
  stable forms on both sides, analyzed saturations, exact odd tanh): faithfully
  rounded (≤ 1 ulp vs libm f64 oracle on 200,000 points), contract/dense/
  exhaustive contracts committed, proof binary extended to 4 functions. First
  batch of the gap mapping (AUDIT_REPDL §postscript): unblocks portable
  LSTM/GRU and GELU-tanh.
- **`sin_f32` / `cos_f32`** portable with argument reduction by **Payne &
  Hanek in pure integer arithmetic** (u128) — exact for any finite f32 up to
  3.4×10³⁸. The 448 bits of 2/π are generated by us (π via Chudnovsky,
  verification by recomposition — no table copied from a libm). Quadrant + 128
  bits of signed fraction; correctly rounded i128 → f64 conversion ⇒ fidelity
  maintained even in the worst reduction cases of the f32 format. Oracle ≤ 1 ulp
  vs libm f64 on 200,000 points over all magnitudes; bit-exact parities;
  contract/dense/exhaustive contracts committed (proof binary: 6 functions).
  Unblocks: portable RoPE (transformers), portable FFT, positional encodings.
- **`erf_f32` / `gelu_f32`** portable — **batch 1 complete**. erf: f64
  Maclaurin series with deterministic relative stopping, saturation |x| ≥ 4,
  small-argument shortcut preserving ±0; GELU **exact**
  (x/2·(1+erf(x/√2)) via the f64 core, without intermediate cast). Accuracy
  verified against an **independent** reference table (series in 60-digit
  Decimal — not libm). Contract/dense(/exhaustive for erf) contracts committed;
  proof binary: 8 sweeps. The portable path now offers exp, ln, tanh, sigmoid,
  sin, cos, erf, GELU — strictly more than RepDL's transcendentals — all
  faithfully rounded and bit-exact cross-platform by construction.

### Added — extensible denoising & noise detection (`scirust-signal::denoise`)
- **New `denoise` module**: a denoising toolbox designed to be *exhaustive by
  families* rather than by enumeration. A closed taxonomy (`DenoiserFamily`:
  Linear / Rank / Transform / Variational / Adaptive) and a uniform `Denoiser`
  trait: adding a method = choosing its family and implementing the trait.
  Current coverage, each routine validated by a measured SNR gain on a synthetic
  signal with a known clean reference:
  - **Linear** (`linear`): moving average, Gaussian smoothing,
    **Savitzky-Golay** (least-squares polynomial fit, preserves peaks — tested
    exact on a polynomial), exponential moving average.
  - **Rank / order** (`rank`): **median** filter, **Hampel** (impulse rejection
    via MAD, decision filter), α-trimmed mean, plus `impulse_mask` which
    explicitly labels which samples are noise.
  - **Transform** (`transform`): ideal FFT low-pass/high-pass, **notch** and
    50/60 Hz + harmonics removal, **Wiener** filter (white noise),
    **wavelet denoising** (multi-level Haar, universal VisuShrink threshold
    `σ√(2 ln N)` with robust MAD σ, soft/hard thresholding).
  - **Variational** (`variational`): **Tikhonov** smoother (L2, tridiagonal
    system solved by Thomas) and **Total Variation** (L1, lagged-diffusivity
    IRLS, preserves edges — verified better than Tikhonov on a noisy step).
- **Noise/information detection & separation** (`denoise::detect`): `classify`
  characterizes noise without naming it via a fixed set of descriptors (robust
  MAD σ; kurtosis/crest factor of the residual; spectral flatness; periodicity
  via the **Fisher g test** insensitive to the number of bins; trend strength;
  `1/f` color slope) and a decision tree → `NoiseType`
  (Gaussian / Impulsive / Periodic / Colored / Baseline / LowNoise). `separate`
  decomposes the signal into information + noise **then falsifies the
  separation** with a **whiteness test** of the residual (autocorrelation vs the
  `±1.96/√N` band): `leaked_structure` flags whether information leaked into the
  noise — the guarantee that makes the separation verifiable and not merely
  plausible.
- **'Universal denoiser' pipeline** `denoise_auto`: detects → chooses the
  suited family → applies (Hampel for impulsive, notch for tonal, trend removal
  for drift, Wiener/wavelets for broadband), and `catalog()` provides a default
  set of denoisers covering each family.
- 27 unit tests + 1 doctest; zero dependency outside `scirust-core`/serde;
  `cargo fmt`/`clippy -D warnings` clean.

### Added — aarch64 proof in CI + portable softmax in the tape (panel 112, continued)
- **CI: the `cross-check-aarch64` job now runs aarch64 code**
  (qemu-user + gcc-aarch64-linux-gnu): `portable_f32` tests + the proof binary
  in standard mode on `aarch64-unknown-linux-gnu`. QEMU implements IEEE-754
  faithfully: every CI run really verifies the committed contract's bit-for-bit
  x86↔ARM identity (commands validated locally: 13/13 tests + verdict=PASS
  under qemu). Closes the open point "CI aarch64 = check only" tracked since
  panel 108.
- **`Var::softmax_portable()`** (+ `Tensor::softmax_portable`,
  `Op::SoftmaxPortable` in reverse.rs and parallel.rs): row-wise softmax whose
  forward goes through `portable_f32::softmax_f32` and whose backward computes
  the Jacobian **from the stored output** — no libm call in the node, so both
  forward AND gradient are bit-exact cross-platform. Opt-in: `Var::softmax`
  (libm) is unchanged. Tests: forward bit-identical to the portable reference,
  gradient equivalent to the libm node (≤ 1e-5), gradient fingerprint frozen
  (cross-platform training contract).

### Added — executable cross-platform proof of the portable f32 path (panel 112)
- **`scirust-core --bin proof_portable_f32`**: self-verifying proof binary —
  recomputes locally the point goldens, the FNV-1a fingerprints of the f32
  bit-space sweeps of `exp_f32`/`ln_f32` (contract step 65,537, dense step 257,
  **exhaustive step 1** — all 2³² entries — with `--full`) and the
  softmax/GEMM composites, then compares everything against the `PROOF_*`
  constants **committed in the repository** (computed on x86-64). Exit code 0 ⇔
  verdict=PASS ⇔ the machine reproduces the x86-64 results bit for bit. Canonical
  lines outside `#` context: their SHA-256 must be identical between machines.
- **`scripts/proof-portable-f32.sh`**: wrapper following the repository
  convention (timestamped evidence bundle `proof-portable-f32-<UTC>/`:
  platform.txt, report.txt, canonical.sha256; stays on the machine,
  `.gitignore`d). Protocol documented in `docs/TEST_PROTOCOL.md` (x86_64 Debian
  panel + Jetson/aarch64 panel).
- The `portable_f32` proof contract is now public (`PROOF_*`,
  `sweep_fingerprint`, `proof_softmax_fingerprint`, `proof_gemm_fingerprint`)
  and shared between the unit tests and the binary; dense and exhaustive
  fingerprints added to the contract.

### Added — RepDL coverage audit and gap closure (panel 111)
- **`docs/audits/AUDIT_REPDL_2026-07-10.md`**: element-by-element functional
  coverage audit of [microsoft/RepDL](https://github.com/microsoft/RepDL)
  (MIT, arXiv:2510.09180) against SciRust — 18/23 items already covered, 2 by
  composition, 1 not applicable by design, 3 closed below. Copyright risk
  analysis: **no RepDL code in the repository** (documentary citations only),
  clean-room approach recorded in writing (audit §3).
- **`Adam::with_amsgrad()`** (`scirust-core`): AMSGrad variant (Reddi, Kale &
  Kumar, ICLR 2018) — the denominator uses the historical maximum of the
  (bias-corrected) 2nd moment, never decreasing. Convergence oracle + test of
  the anti-spike property (AMSGrad steps < 10% of Adam steps after a gradient
  spike).
- **`scirust_runtime::hash`**: hex SHA-256 fingerprints of `f32` slices, of
  tensors (shape included) and of `state_dict` (sorted keys ⇒ independent of
  insertion order); little-endian encoding of IEEE-754 bits ⇒ identical on any
  platform for bit-identical data. The *attestation* tool of reproducibility
  (two machines, same hash), cryptographic complement to the FNV-1a
  fingerprints. 5 tests.
- **`reproducible::{exp_via_f64, ln_via_f64}`** (`scirust-core`): f32
  transcendentals by promotion to `f64` — same technique class as RepDL, honest
  documentation of the guarantee (faithfully rounded; correctly rounded outside
  the table-maker's dilemma; cross-platform identity likely but not proven).
  Fidelity test ≤ 0.5 ulp on 8,000 points.
- **`scirust-core::portable_f32`** — the **portable f32 path**: `exp_f32`,
  `ln_f32`, `softmax_f32`, `dot_f32`, `gemm_f32` in pure Rust **without libm**,
  using only basic IEEE-754 operations in fixed order ⇒ results **bit-exact
  cross-platform by construction** (the 'cross-platform f32' axis on which
  RepDL was stronger, achieved here without an external TCB). exp: k·ln 2
  reduction (hi/lo split) + degree-13 Taylor; ln: mantissa normalization + atanh
  series; softmax: max-subtract + portable exp + `reproducible_sum`
  (bit-wise equivariant under permutation); dot/gemm: exact f64 products,
  sequential f64 accumulation. Guarantees stated without over-promising:
  faithfully rounded (≤ 1 ulp, verified against the libm f64 oracle on 200,000
  points); correct rounding *proven* = future work; x87/i586 caveat documented.
  13 tests including bit-for-bit goldens and FNV fingerprints of a full sweep of
  the f32 bit space (step 65,537) — the portability contract to run on ARM;
  identical fingerprints in debug/release. Clean-room implementation (public
  mathematical methods; no fdlibm/musl/RepDL code consulted).

### Changed — CHECKUPAUTO actor replaced by TAREK ZEKRITI
- **Attribution**: the `authors` field of `scirust-burn-bridge` changes from
  "CheckupAuto" to "Tarek Zekriti"; the local git identity of commits is now
  TAREK ZEKRITI \<zekrititarek@gmail.com\>.
- **GitHub URLs/slugs**: all `CHECKUPAUTO/*` references (26 files — the
  `repository` of the Cargo.tomls, README, LICENSE.md, RELEASING, CycloneDX
  SBOM, technical reports ×8 languages, protocol scripts, scirust-rsi docs,
  scirust-som SARIF URI) point to `Memorithm/*`, the org that actually hosts
  the repositories.
- **Brand also replaced (2nd pass, on user confirmation)**: contact emails
  `contact@checkupauto.fr` → `zekrititarek@gmail.com` (LICENSE, LICENSING,
  SECURITY, brochure, headers of the reports ×8 languages) and the SPDX
  identifier `LicenseRef-CheckupAuto-Dual` → `LicenseRef-TarekZekriti-Dual`
  (LICENSE + root Cargo.toml, scirust-burn-bridge, scirust-license;
  `deny.toml` and the SBOM referenced nothing there).

### Added — Correctness '26 submission draft (`paper/correctness26/`)
- **Venue decided**: Correctness '26 (10th Int. Workshop on Software
  Correctness for HPC Applications, SC26 Chicago), deadline July 23, 2026,
  notification September 1st. Evaluation platform: **Jetson AGX Thor** (user
  decision). JOSS ruled out (non-OSI PolyForm license, user decision not to
  relicense).
- **Complete draft**: `main.tex` (ACM sigconf, ~8 pages: 'determinism-as-
  evidence' intro, related work with an honest RepDL pivot, three numerical
  regimes + σ invariant, bit-reproducible training T1-T4, inference-as-audit-
  artifact, deterministic edge int8, σ gate + 'dead guards' negative study 22
  repositories/9.16 M LOC, measured determinism cost with the x86-64/Thor table
  and the cross-platform bit-for-bit identity of the fingerprints, limitations,
  claims → evidence table in `table*`); `references.bib` (7 references,
  metadata verified on arXiv/publisher on 2026-07-10, no invented reference);
  `README.md` (latexmk/Overleaf build, submission TODO). Structural check
  performed: balanced environments/braces, all citations and refs resolved.
  Every claim of the paper is backed by the table of `paper/PAPER_PLAN.md` —
  no claim without an executable witness.

### Added/Changed — README honesty, 'dead guards' empirical study, paper positioning
- **Honesty fix (uniqueness claims)**: the claim "No mainstream framework
  ships this guarantee tested" (README) and its FR equivalents
  (`docs/INDUSTRIAL_ROADMAP.md`, the investor dossier, historical 0.14.0
  entry of this file — rectified, not rewritten) are **falsified by RepDL**
  (Microsoft, 2025, arXiv:2510.09180: bit-for-bit **cross-platform**
  reproducibility of an f32 subset of PyTorch via correct rounding). Replaced
  by the exact formulation: to our knowledge, SciRust is the only
  **self-contained** DL framework (100% auditable Rust stack, zero FFI in the
  computation path) simultaneously offering multi-thread bit-identical training
  tested in CI, deterministic embedded int8 and audit artifacts; RepDL is
  stronger on the cross-platform f32 axis, as an overlay on a C++/Python TCB,
  without low precision or audit pieces.
- **`epsilon-audit --mine <dir>`** (crate `scirust-sigma`, public module
  `mine`, std-only): multi-language mining (Rust, C/C++/CUDA/OpenCL,
  WGSL/GLSL/Metal/compute shaders) of "dead epsilon guards" — f32 literals
  below `f32::MIN_POSITIVE` (M1, FTZ/DAZ flush) or below `1/f32::MAX`
  (M2, inversion to `inf`). Documented typing heuristics (suffix/line; bare C
  literal = double, never counted; f32 shaders by default), comparison to the
  threshold on the value **rounded to f32** (materialization semantics),
  `test*/`/`bench*/`/vendor exclusions, detection of fast-math/FTZ flags in
  build files, deterministic Markdown+TSV report sealed with SHA-256. 27 unit
  tests on synthetic fixtures (real M1, real M2, benign f64, test exclusion,
  range bounds, comments/strings). Read-only, exit 0 (analytical).
- **Empirical study `docs/DEAD_GUARDS_STUDY.md`**: campaign on **22 public
  repositories** (llama.cpp, ggml, candle, burn, pytorch, tensorflow,
  onnxruntime, OpenBLAS, eigen, cutlass, ndarray, nalgebra, faer-rs, tract,
  wgpu, glam, ncnn, MNN, tvm, whisper.cpp, stable-diffusion.cpp, wonnx — SHAs
  recorded, 0 clone failure), **9,160,848 lines** scanned, 14 raw candidates,
  full manual review → **0 confirmed dead guard** (14 BENIGN: `approx` test
  tolerances of ndarray, deliberate subnormal constants of naga's WGSL lexer).
  Verdict **NO-GO** (rule: ≥ 3 confirmed in ≥ 2 repositories) — negative result
  honestly recorded; in return, the FTZ threat model is confirmed (9/22
  repositories enable fast-math/FTZ in their builds). No issue/PR opened, no
  external contact.
- **Paper material (Lot 3)**: `paper/RELATED_WORK.md` (citable section —
  Goldberg/Monniaux/ReproBLAS; PyTorch deterministic mode/EasyScale/RepDL with
  pivot paragraph; arXiv:2410.09172 for the sanitized path) and
  `paper/PAPER_PLAN.md` (title + 2 variants; venues: correctness/reproducibility
  workshop recommended — JOSS blocked by the non-OSI PolyForm license as-is;
  section plan; **claims → evidence table** T1-P1 mapping each claim to its
  exact test with command; answers to anticipated weaknesses).
- **Closure of the plan's TODO-EVIDENCE (S2, R4, O1)** — decisions recorded:
  - **S2**: the `epsilon-audit --check` gate is wired in CI (new
    `epsilon-audit` job in `.github/workflows/ci.yml`) — no f32 guard below
    σ_sanitized can enter `scirust-gpu/src` anymore without breaking the build.
  - **R4**: new CI test
    `forward_fingerprint_is_thread_count_invariant`
    (`scirust-runtime/tests/fingerprint_thread_invariance.rs`) — the 64-bit
    forward fingerprint (MLP 784-256-10, 100% integer synthetic batches,
    portable) is bit-identical under rayon pools of 1/2/4/8 threads, and stable
    across re-executions. `rayon` added as a dev-dependency of
    `scirust-runtime` (already in the lockfile — zero new download).
  - **O1**: 'cost of determinism' bench
    (`scirust-core/src/bin/bench_reduction_overhead.rs`) — worker-order-frozen
    reduction (pattern of `train_batch_threaded`) vs arrival-order accumulation
    (channel), ±1e16 magnitudes making the order observable, bit-for-bit
    fingerprints per repetition. x86 measurement (4 cores, release,
    dim=100,352, 30 reps): **the frozen order is faster** (0.76×–0.93× of the
    baseline time) and bit-stable; the 'arrival' baseline produced 3 distinct
    fingerprints at 8 threads (real observed non-determinism). Wall-clock ⇒
    protocol, never CI; Jetson panel via `scripts/bench-o1-jetson.sh` (platform
    report recorded, opt-in clock pinning `--pin-clocks`, 3 runs, native Q3
    NEON + R4 fingerprint tests, timestamped and git-ignored evidence bundle),
    documented in `docs/TEST_PROTOCOL.md`. **Measured Jetson panel (AGX Thor,
    14 cores, MAXN, 3×30 reps)**: frozen order ≈ free up to 2 threads
    (0.93–0.99×), ~1–3% at 4, ~6–11% at 8; and the frozen-reduction
    fingerprints are **bit-identical x86_64 ↔ aarch64** — cross-platform
    reproducibility of the pattern, measured. Script fixes: cargo env under
    sudo (secure_path); `--lib` on the Q3 filter (the tail only showed the last
    test target).

### Added — `scirust-sigma`: structural σ bounds ('zero lid') + epsilon audit
New leaf crate **without external dependency** (`std` only) that names and
encodes the numerical invariant, until now implicit, of the determinism
contract:

- **σ = zero lid per regime.** Each deterministic numerical path
  (`scirust-gpu/src/deterministic.rs`) has a smallest representable positive:
  exact integer `1`, fixed-point Q15.16 `2⁻¹⁶`, fixed-point Q31.32 `2⁻³²`, f32
  *sanitized* `f32::MIN_POSITIVE`, raw f32 / raw f64 = smallest subnormal.
  Named constants (`SIGMA_SANITIZED_F32`, `SIGMA_RAW_F32`, `SIGMA_RAW_F64`,
  `SIGMA_Q15_16`, `SIGMA_Q31_32`), `sigma_f32`/`sigma_f64`,
  `guard_denominator_f32/f64`, and the **central invariant**
  `is_valid_guard_f32` (an anti-zero guard under σ is *dead* on the sanitized
  path: `sanitize_f32` overwrites it). Edge behaviors (0, negative, NaN, regime
  without f32 σ) defined and tested — 12 unit tests at bit-for-bit values
  (`to_bits()`).
- **Alignment test** (`tests/sanitize_alignment.rs`): asserts, without coupling
  the crate to `scirust-gpu`, that the `sanitize_f32` threshold
  (= `f32::MIN_POSITIVE`) is bit-identical to `SIGMA_SANITIZED_F32`. Breaks if
  one moves without the other.
- **`epsilon-audit` binary** (std-only; `sha2` already in the lockfile seals
  the report): homegrown lexical scanner (outside comments/strings) that
  classifies the ~14,400 floating literals `< 1.0` of the workspace into A
  (algorithm, do not migrate) / B (zero guard, σ target) / C (test) / D
  (convergence) / U (unclassified), and produces `docs/EPSILON_AUDIT.md`
  (deterministic report, SHA-256 sealed).
- **CI `--check` gate**: exit ≠ 0 if an f32 guard below σ_sanitized remains
  outside tests in `scirust-gpu/src`. Exit 0 on the current tree (no dead guard
  on the sanitized path; 686 gpu/src literals inspected). Safety: preventive
  control of a class of defects (dead guard → silent `Inf`/`NaN`) invisible in
  human review, without added supply-chain surface, sealed artifact, strictly
  read-only binary.

### Added — neighboring crates: CUSUM chart (`scirust-spc`) and GUM expanded uncertainty (`scirust-metrology`)
Two additions in the crates neighboring tolerancing, each verified by a
Monte-Carlo cross-check embedded in its tests:

- **`scirust-spc::cusum`**: tabular two-sided **CUSUM control chart**.
  Cumulative sums `Cᵢ⁺ = max(0, Cᵢ₋₁⁺ + (xᵢ−μ₀) − K)` / `Cᵢ⁻` with reference
  value `K = k·σ` and decision interval `H = h·σ`, drift signal, and **ARL** via
  the Siegmund approximation (`b = h+1.166`), combined two-sided
  `1/ARL = 1/ARL⁺ + 1/ARL⁻`. Complements EWMA as a memory detector of small
  sustained drifts. *Cross-check*: Monte-Carlo false-alarm rate (N(0,1)) vs
  the ARL₀ ≈ 168 for `k=0.5, h=4`.
- **`scirust-metrology::expanded`**: GUM **expanded uncertainty**. Effective
  degrees of freedom of Welch–Satterthwaite `ν_eff = u_c⁴ / Σ(uᵢ⁴/νᵢ)`,
  coverage factor `k = t_{(1+p)/2}(ν_eff)` (Student quantile via the
  Cornish–Fisher expansion, exact when `ν_eff→∞ ⇒ k→1.96`), expanded
  uncertainty `U = k·u_c` and coverage interval. Closes the GUM loop after
  `combined_uncertainty`. *Cross-check*: `t` quantiles vs tables
  (t₀.₉₇₅(10)=2.228, …).

### Added — plan & economics: ISO 286 fits, double/sequential sampling, Taguchi loss (`scirust-tolerance`)
Three modules that border dimensioning: the standardized fits table, the
sampling plans with reduced average size, and the cost of non-quality linked to
inertia. Each module is verified by fuzzing cross-check against an
**independent reference**:

- **`fits`**: **ISO 286 limits and fits**. Standard tolerance `ITn` from the
  factor `i = 0.45·∛D + 0.001·D` (µm) and the grade multipliers (IT5–IT18),
  fundamental **shaft** deviations `d, e, f, g, h` (verified ISO formulas), and
  **fit classification** hole/shaft in the basic-hole system H (max/min
  clearance, clearance / uncertain / interference category). *Cross-check*:
  identity 'clearance range = IT_hole + IT_shaft', independent recomputation of
  the `IT` by the factor-`i` formula, monotonicity in grade.
- **`sequential`**: **double and sequential sampling** (Wald SPRT). Double plan
  `(n1,c1,r1,n2,c2)` with binomial OC `Pa(p) = P(d1≤c1) + Σ P(d1=k)·P(d2≤c2−k)`
  and average sample number `ASN = n1 + n2·P(c1<d1<r1)`; SPRT with two straight
  boundaries `d = s·n ∓ h` (accept / reject / continue). *Cross-check*: OC and
  ASN of the double plan vs direct Monte-Carlo; SPRT OC guarantee at the two
  design points.
- **`taguchi`**: **Taguchi loss and cost of non-quality**. Quadratic loss
  `L = k(y−T)²`, coefficient `k = A/Δ²` calibrated on the cost at the limit, and
  the identity `E[L] = k·(σ²+δ²) = k·I²` — the exact reason why inertial
  tolerancing directly minimizes Taguchi loss. Smaller-is-better /
  larger-is-better variants and **economic tolerance** `Δ = Δ₀·√(A/A₀)`.
  *Cross-check*: Monte-Carlo of the quadratic loss vs `k·I²`; economic
  tolerance equilibrium.

Wired into `scirust-mcp` (`tolerance_fits`, `tolerance_sequential`,
`tolerance_taguchi`). Global fuzz: **118,476 checks / 0 errors** over 29
modules.

### Added — workshop: attribute sampling, stress-strength interference, capability study by subgroups (`scirust-tolerance`)
Three modules that complete the workshop quality toolbox: accepting a lot
without measuring, quantifying the reliability of a random **fit**, and running
a real **capability study** on rational subgroups. Each module is verified by
fuzzing cross-check against an **independent reference**:

- **`attributes`**: **attribute sampling** (ISO 2859-1 / MIL-STD-105). Simple
  plan `(n, c)` — accept if defective ≤ `c` — with exact binomial operating
  curve `Pa(p) = Σ_{d≤c} C(n,d) pᵈ(1−p)ⁿ⁻ᵈ` (stable recurrence), with
  **two-point design** (sweep of increasing `c`, smallest `n` holding the
  producer point) and average outgoing quality (AOQ). *Cross-check*: direct
  Monte-Carlo of the acceptance rule vs the binomial OC; the designed plans
  hold the two nominal points.
- **`interference`**: **stress-strength interference** and fit reliability.
  Reliability `R = P(S > L) = Φ(β)`, `β = (μ_S−μ_L)/√(σ_S²+σ_L²)`
  (reliability index), and bore/shaft **fit** analysis: clearance
  `C = bore − shaft ∼ N(μ_h−μ_s, σ_h²+σ_s²)`, `P(clearance > 0)` (free fit) vs
  `P(clearance < 0)` (interference) — the probability that a randomly drawn
  pair assembles as intended, which a worst-case min/max does not give.
  *Cross-check*: Monte-Carlo of `P(S>L)` vs the closed form; clearance
  partition identities.
- **`subgroup`**: **capability study on rational subgroups** (MSA AIAG /
  ISO 22514-2). Separates **within-subgroup** dispersion (short-term,
  `σ̂ = R̄/d₂ = s̄/c₄` via control-chart constants) which carries the
  **capability** indices `Cp`/`Cpk`, from the **overall** dispersion
  (long-term) which carries the **performance** indices `Pp`/`Ppk`: a large
  `Cp` with a small `Pp` signals a stable but drifting process. *Cross-check*:
  independent recomputation of the overall σ; agreement of the `R̄/d₂` and
  `s̄/c₄` estimators; `Cp` identity.

Wired into `scirust-mcp` (`tolerance_attributes_plan`, `tolerance_interference`,
`tolerance_subgroup_capability`). Global fuzz: **113,336 checks / 0 errors**
over 26 modules.

### Added — process quality: variables sampling, Six Sigma, cause attribution (`scirust-tolerance`)
Three modules that extend the measurement & analysis layer toward the **quality
control** that competing suites (Minitab, Q-DAS) offer around dimensioning:
accepting a lot on measurements, quantifying the yield of a multi-stage
process, and tracing data back to cause. Each module is verified by fuzzing
cross-check against an **independent reference**:

- **`variables`**: **variables sampling** (ISO 3951 / MIL-STD-414, `k` method).
  Accepts the lot when the normalized distance `Q = (limit−x̄)/σ ≥ k`;
  closed-form operating curve `Pa(p) = Φ(√n_eff·(z_p−k))` with `z_p = −Φ⁻¹(p)`,
  and **two-point design** `√n = (z_{1−α}+z_{1−β})/(z_aql−z_rql)`,
  `k = (z_aql·z_{1−β}+z_rql·z_{1−α})/(z_{1−α}+z_{1−β})` from (AQL, RQL, α, β).
  `σ` known and `s` unknown methods (sample inflated by `1+k²/2`); maximum
  allowable standard deviation for a centered lot `MSD = (USL−LSL)/(2k)`,
  pendant to the inertial budget `I_max` on measurements. *Cross-check*: direct
  Monte-Carlo of the acceptance rule vs the closed-form OC; `MSD` identity.
- **`sixsigma`**: **Six-Sigma yield accounting**. DPU, DPMO, throughput yield
  `Y = e^(−DPU)`, **rolled throughput yield** `RTY = ∏ Yᵢ` (the probability
  that a part crosses *all* steps without rework, invisible on a single
  capability), normalized yield `RTY^(1/steps)`, and the
  yield↔sigma-level↔DPMO conversions `Z = Φ⁻¹(Y)+shift` with the Motorola
  `1.5σ` shift (hence "6σ ⇒ 3.4 DPMO"). *Cross-check*: round-trips vs the
  independent normal tail; `RTY` vs explicit product; `−ln Y = DPU` of Poisson.
- **`attribution`**: **data-driven cause attribution**. Fits the measured
  assembly to the co-measured components by least squares `y ≈ β₀ + Σ βⱼxⱼ`
  and decomposes the explained variance by the exact identity (OLS with
  constant) `Σⱼ βⱼ·Cov(xⱼ,y) = Var(ŷ) = R²·Var(y)`: **empirical**
  sensitivities `βⱼ` (to compare to the design `αⱼ`), signed shares
  `cⱼ = βⱼ·Cov(xⱼ,y)/Var(y)` (sum to `R²`, Pratt measure) and the
  **unexplained remainder** `1−R²` that betrays a cause outside the measured
  set. *Cross-check*: identity `Σcⱼ = R²`; recovery of the generating
  coefficients; `c = corr²` with a single regressor.

Wired into `scirust-mcp` (`tolerance_variables_plan`, `tolerance_six_sigma`,
`tolerance_attribution`). Global fuzz: **111,926 checks / 0 errors** over 23
modules.

### Added — the measurement & analysis layer of inertial tolerancing (`scirust-tolerance`)
Six modules that bring the crate to the level of competing products (Minitab,
Q-DAS, 3DCS, CETOL) on what they do *around* dimensioning: qualifying the
measurement device, bounding statistically, separating the levers, fitting a
distribution, deepening GD&T and quantifying the uncertainty of an index. Each
module is verified by fuzzing cross-check against an **independent reference**:

- **`msa`**: **crossed Gage R&R by ANOVA** (MSA AIAG). Decomposition of the
  model `yᵢⱼₖ = μ + Partᵢ + Opⱼ + (Part·Op)ᵢⱼ + εᵢⱼₖ` into the variance
  components repeatability `EV` / reproducibility `AV` / part `PV`, with `%R&R`
  (study variation), `%contribution`, `%tolerance = 6σ_GRR/(USL−LSL)`,
  `ndc = ⌊1.41·σ_PV/σ_GRR⌋` and the AIAG verdict (10%/30% bands). *Cross-check*:
  sum-of-squares decomposition identity; constructions with null
  repeatability/reproducibility.
- **`interval`**: **statistical tolerance intervals** (ISO 16269-6). Howe's
  two-sided factor `k = z_{(1+p)/2}·√(ν(1+1/n)/χ²_{ν,α})` and Natrella's
  one-sided (closed form); both tend (slowly, in `1−1.645√(2/ν)`) toward the
  normal quantile. *Cross-check*: Monte-Carlo coverage of the true content.
- **`sensitivity::dual_contributions`**: **GeoFactor / dual sensitivity** — for
  each contributor, geometric magnification `|αᵢ|`, share of the assembly
  **mean** `αᵢδᵢ` (sum to `δ_Y`) *and* share of the **variance** `αᵢ²σᵢ²/σ_Y²`
  (sum to 1), in the manner of 3DCS/CETOL: distinguishes a dimension to
  **recenter** from a dimension to **tighten**, which the variance share alone
  masks.
- **`distfit`**: **distribution fitting** (ISO 22514-2). Normal / Lognormal /
  Rayleigh / Weibull families (median-rank regression), best fit by maximum
  likelihood, and **capability by percentiles**
  `Cp = (USL−LSL)/(X₀.₉₉₈₆₅−X₀.₀₀₁₃₅)`. *Cross-check*: `cdf∘quantile`
  round-trip; the Normal recovers the classic `Cp`; parameter recovery.
- **`position` (advanced GD&T)**: **virtual / resultant condition** (`VC`
  internal `MMC−t`, external `MMC+t`), **datum shift** (slippage from the MMB)
  and two-stage **composite position** (PLTZF/FRTZF). *Cross-check*:
  monotonicity and bounds of the envelopes vs the actual size of the feature.
- **`capability` (capability CI)**: **exact** (χ²) confidence interval on `Cp`
  and **large-sample** (Bissell) on `Cpk`. *Cross-check*: Monte-Carlo coverage
  of the `Cp` CI.

Wired into `scirust-mcp` (`tolerance_gage_rr`, `tolerance_statistical_interval`,
`tolerance_dual_sensitivity`, `tolerance_distribution_fit`, `tolerance_gdt`,
`tolerance_capability_ci`). Global fuzz: **98,858 checks / 0 errors**.

### Added — transpiler: **MATLAB `range(v)`** — statistical range, proven against real Octave (Phase 2, increment 49)
`range(v) = max(v) − min(v)` (sample range), composed from the already-verified
reduction `Max`/`Min` nodes — no new SIR node, std-only. Type
inference recognizes `range` as a reduction (vector argument).

- `range(v)`: vector → scalar (max−min difference).

One oracle case (`range(v)` on a 7-element vector). **Oracle 140/140** (200
trials each); **97 unit tests** (1 new).
*Non-vacuity*: replacing the subtraction with an addition (`max+min` instead of
`max−min`) makes the `range` case diverge — the composition is indeed load-bearing.

### Added — transpiler: **MATLAB `fftshift` / `ifftshift`** — spectrum centering, proven against real Octave (Phase 2, increment 48)
FFT companions: `fftshift(v)` brings the zero-frequency component to the
center (swapping the two halves = `circshift` by `⌊n/2⌋`) and `ifftshift(v)` does the inverse
(`circshift` by `⌈n/2⌉`) — **exact** inverses for even **and
odd** lengths. New SIR nodes `Fftshift`/`Ifftshift` (vector→vector) and deterministic
preamble helpers built on `np::circshift`, reusing the already-proven modular
reindexing.

- `fftshift(v)`, `ifftshift(v)`: real vector → vector (same length).
- Apply naturally to a real magnitude spectrum: `fftshift(abs(fft(x)))`.

Three oracle cases (`fftshift`/`ifftshift` in **odd** length to distinguish
`⌊·⌋`/`⌈·⌉`, plus `fftshift(abs(fft(x)))` — routed FFT + complex abs + shift).
**Oracle 139/139** (200 trials each); **96 unit tests** (1 new).
*Non-vacuity*: making `ifftshift` use `⌊n/2⌋` (instead of `⌈n/2⌉`) makes
the odd-length `ifftshift` case diverge while `fftshift` stays green — the
floor/ceiling distinction is indeed load-bearing.

### Added — transpiler: **MATLAB `fft` / `ifft`** routed to `scirust-signal`, proven against real Octave (Phase 2, increment 47)
First **signal-processing routing** on the MATLAB side: `fft(x)` (complex DFT
of a real vector), `ifft(c)` (inverse DFT) and `abs(fft(x))` (magnitude spectrum)
emit the verified FFT kernel of `scirust-signal` rather than re-deriving it —
reusing exactly the complex machinery (`Fft`/`Ifft`/`ComplexArray`/`ComplexAbs`)
already proven on the Python side.

- `fft(x)`: real vector → complex vector (full N-point spectrum).
- `ifft(c)`: complex vector → complex vector (inverse DFT, `1/N`).
- `abs(z)` on a complex spectrum → real array of magnitudes (routed
  separately from the element-wise real `abs`).

The oracle harness now serializes Octave's complex results as
interleaved `(re, im)` to align with the Rust `ComplexArray` output (an
`ifft(fft(x))` that Octave reduces to real is padded with zero imaginary
parts). Three oracle cases proven against real Octave (compiled via cargo with
`scirust-signal`). **Oracle 136/136** (200 trials each); **95 unit tests**
(1 new).
*Non-vacuity*: routing `fft` to `rfft` (half-spectrum) makes the three FFT
cases diverge (lengths 10/16 and 5/8, round-trip crash) — the `fft` routing is
indeed load-bearing.

### Added — transpiler: **MATLAB `sec` / `csc` / `cot`** — reciprocal trigonometry, proven against real Octave (Phase 2, increment 46)
Completes the trigonometry: the reciprocal functions `sec = 1/cos`, `csc = 1/sin`,
`cot = 1/tan`, each applying the base trig function (scalar or element by
element) then taking the inverse via the new `reciprocal` helper (`1.0 / e`,
scalar or broadcast).

- `sec(x)`, `csc(x)`, `cot(x)`: scalar or vector (element-wise).

Four oracle cases (scalar `sec`/`csc`/`cot` on ranges away from poles, plus
element-wise `sec(flip(v))`). **Oracle 133/133** (200 trials each);
**94 unit tests** (1 new).
*Non-vacuity*: routing `sec` to `sin` (instead of `cos`) makes the two `sec`
cases diverge while `csc`/`cot` stay green — the reciprocal mapping is indeed
load-bearing.

### Added — transpiler: **MATLAB `asind` / `acosd` / `atand`** — inverse trigonometry in degrees, proven against real Octave (Phase 2, increment 45)
Completes the degree-based trig family: `asind`/`acosd`/`atand` apply
the inverse `asin`/`acos`/`atan` (result in radians, scalar or element-wise)
then convert the angle to **degrees** (`× 180/π`, via the `scale_by_const`
helper shared with `rad2deg`).

- `asind(x)`, `acosd(x)`, `atand(x)`: scalar or vector (element-wise).
- Domains: `asind`/`acosd` on `[-1, 1]`; `atand` on all reals.

Four oracle cases (scalar `asind`/`acosd`/`atand`, plus element-wise
`atand(flip(v))`). **Oracle 129/129** (200 trials each); **93 unit tests** (1 new).
*Non-vacuity*: replacing the `180/π` factor by `90/π` for the inverse degree
trig makes the four cases diverge while `rad2deg` (proper factor) stays green —
the conversion factor is indeed load-bearing.

### Added — transpiler: **MATLAB `sind` / `cosd` / `tand`** — degree-based trigonometry, proven against real Octave (Phase 2, increment 44)
Trigonometry with a **degree** argument: `sind`/`cosd`/`tand` convert the argument
to radians (`× π/180`, scalar or by broadcast) then apply `sin`/`cos`/`tan`
(scalar or element-wise). The conversion logic is factored into a
`scale_by_const` helper shared with `deg2rad`/`rad2deg`.

- `sind(x)`, `cosd(x)`, `tand(x)`: scalar or vector (element-wise).

*Honest boundary*: the MATLAB special cases (exact zero / exact `Inf` at
multiples of 90°) are **not** replicated — the simple definition `f(x·π/180)` is
used (the oracle draws random angles that never reach those points).

Four oracle cases (scalar `sind`/`cosd`/`tand`, plus element-wise `cosd(flip(v))`).
**Oracle 125/125** (200 trials each); **92 unit tests** (1 new).
*Non-vacuity*: replacing the `π/180` factor by `π/90` for degree trig makes
the four `sind`/`cosd`/`tand` cases diverge while `deg2rad` (proper factor)
stays green — the conversion factor is indeed load-bearing.

### Added — transpiler: **MATLAB `circshift(v, k)`** — circular shift, proven against real Octave (Phase 2, increment 43)
Modular reindexing: `circshift(v, k)` circularly shifts the vector by `k`
positions (`result[i] = v[(i−k) mod n]`, length unchanged), with `k` rounded to
the nearest integer and reduced modulo `n` — so **any sign / any magnitude**
is valid. New SIR node `Circshift { arr, k }` and deterministic preamble helper
`np::circshift` (arithmetic via `rem_euclid`, safe for negative shifts).

- `circshift(v, k)`: `v` vector, `k` integer scalar (literal or variable) →
  shifted vector (same length).

Two oracle cases (positive `circshift(v, 2)`, negative `circshift(v, -3)`). **Oracle
121/121** (200 trials each); **91 unit tests** (1 new).
*Non-vacuity*: reversing the shift direction (`i + k` instead of `i − k`) makes
the two `circshift` cases diverge (positive AND negative) — the reindexing
direction is indeed load-bearing.

### Added — transpiler: **MATLAB `gradient(v)`** — numerical gradient with unit spacing, proven against real Octave (Phase 2, increment 42)
Numerical differentiation: `gradient(v)` returns a vector of **the same length**
as the input, by **centered** differences in the interior `(v[i+1] − v[i−1])/2`
and **one-sided** differences at the two ends (`v[1] − v[0]`, `v[n−1] − v[n−2]`),
with unit spacing — exactly MATLAB/Octave's `gradient`. New SIR node `Gradient`
(vector→vector, like `diff`) and deterministic preamble helper `np::gradient`.

- `gradient(v)`: `v` vector → vector of numerical derivatives (same length).
- Edge cases: `gradient([x]) = [0]`; `gradient([]) = []`.

One oracle case (`gradient(v)` on a 7-element vector). **Oracle 119/119**
(200 trials each); **90 unit tests** (2 new — routing + structure of the
centered/one-sided formula).
*Non-vacuity*: dividing the centered difference by `3` instead of `2` makes
the `gradient` case diverge (from the first interior index) while `diff` stays
green — the central factor is indeed load-bearing.

### Added — transpiler: **MATLAB `log2` / `asinh` / `acosh` / `atanh`** — base-2 log and inverse hyperbolic trigonometry, proven against real Octave (Phase 2, increment 41)
Completes the elementary vocabulary: the **base-2 logarithm** `log2` and the
three hyperbolic inverses **arc-sine** `asinh` / **arc-cosine** `acosh` /
**arc-tangent** `atanh`, each a unary function applying in scalar **or**
element-wise form (same proven mechanism as `sin`/`asin`), mapped 1:1 onto the
corresponding `f64` method.

- `log2(x)`, `asinh(x)`, `acosh(x)`, `atanh(x)`: scalar or vector (element-wise).
- Domains: `log2` on `(0, ∞)`; `asinh` on all reals; `acosh` on `[1, ∞)`;
  `atanh` on `(-1, 1)`.

Five oracle cases (scalar `log2`, `asinh`, `acosh`, `atanh` on ranges within
the domain, plus element-wise `atanh(flip(v))`). **Oracle 118/118** (200 trials
each); **89 unit tests** (1 new).
*Non-vacuity*: routing `atanh` to the `asinh` method makes the two `atanh`
cases diverge (scalar AND element-wise) while `log2`/`asinh`/`acosh` stay
green — the name→method mapping is indeed load-bearing.

### Added — transpiler: **MATLAB `tan` / `asin` / `acos`** — elementary and inverse trigonometry, proven against real Octave (Phase 2, increment 40)
Completes the trig vocabulary: the **tangent** `tan` and the inverses
**arc-sine** `asin` / **arc-cosine** `acos`, each a unary function that
applies in scalar **or** element-wise form (same proven mechanism as
`sin`/`cos`/`atan`), mapped 1:1 onto the corresponding `f64` method.

- `tan(x)`, `asin(x)`, `acos(x)`: scalar or vector (element-wise).
- Domains: `asin`/`acos` on `[-1, 1]`; `tan` finite away from the poles `±π/2`.

Four oracle cases (scalar `tan`, `asin`, `acos` on ranges within the
domain, plus element-wise `asin(flip(v))`). **Oracle 113/113** (200 trials
each); **88 unit tests** (1 new).
*Non-vacuity*: routing `asin` to the `acos` method makes the two `asin` cases
diverge (scalar AND element-wise) while `acos` and `tan` stay green —
the name→method mapping is indeed load-bearing.

### Added — transpiler: **MATLAB `norm(v, p)`** — general finite vector p-norm, proven against real Octave (Phase 2, increment 39)
`norm` now accepts a **second form** `norm(v, p)` computing the vector p-norm
`(Σ |vᵢ|^p)^{1/p}` for any **finite** `p` (`norm(v, 1)` = sum of absolute
values, `norm(v, 2)` = Euclidean norm, etc.). Composed from already-verified
nodes: element-wise `abs`, broadcast `.^ p`, fixed-order sum, then scalar power
`^(1/p)`. The one-argument form `norm(v)` (2-norm) is unchanged.

- `norm(v, p)`: `v` vector, `p` finite scalar → p-norm.
- Type inference recognizes the first argument of `norm(v, p)` as a
  vector (like `polyval`), with `p` remaining scalar.

*Honest boundary*: the norms `p = Inf`/`-Inf` and the matrix (spectral) norm
are distinct quantities and remain **rejected**.

Two oracle cases (literal `norm(v, 1)`, `norm(v, p)` with `p` fuzzed in `[1, 5]`).
**Oracle 109/109** (200 trials each); **87 unit tests** (1 new).
*Non-vacuity*: replacing the exponent `1/p` by `2/p` (numerator `1`→`2`) makes
the two p-norm cases diverge while the 2-norm `norm(v)` (separate path) stays
green — the reciprocal exponent is indeed load-bearing.

### Added — transpiler: **MATLAB `logspace(a, b, n)`** — logarithmic vector constructor, proven against real Octave (Phase 2, increment 38)
Sibling of `linspace`: `logspace(a, b, n)` produces `n` logarithmically spaced
points, `10^a .. 10^b`. Defined exactly as `10 .^ linspace(a, b, n)`,
it thus inherits `linspace`'s **exact endpoints** (`y(end) = 10^b`) and the
`logspace(a, b, 1) = [10^b]` case. New SIR node `Logspace { a, b, n }` (same
type/scan rules as `Linspace`) and deterministic preamble helper `np::logspace`
built on `np::linspace`.

- `logspace(a, b, n)`: `a`, `b` scalars, `n` integer count (literal or
  `length(x)`) → vector of `n` values `10^a..10^b`.

*Honest boundary*: the MATLAB special case "if `b == pi`, points between
`10^a` and `pi`" is **not** replicated — the simple `10.^linspace` definition
is used (the oracle draws random bounds that never equal `pi`).

One oracle case (`logspace(a, b, 6)`). **Oracle 107/107** (200 trials each);
**86 unit tests** (1 new). *Non-vacuity*: replacing the base `10` by `2`
in `np::logspace` makes the `logspace` case diverge alone (wrong base) while
all the others stay green — base 10 is therefore indeed load-bearing.

### Added — transpiler: **MATLAB `mod` / `rem` element-wise on arrays** proven against real Octave (Phase 2, increment 37)
`mod` and `rem` become **vectorized**. The new `lower_modrem` helper
assembles `a − b·floor(a/b)` (resp. `a − b·fix(a/b)`) by delegating each
arithmetic step to `ew_or_broadcast`, so that the same logic covers scalars,
two vectors, or a scalar↔vector broadcast (with `floor`/`fix` applied
element-wise when the quotient is an array).

- `mod(a, b)` / `rem(a, b)` now accept **vectors** (element-wise)
  and scalar↔vector mixtures (broadcast), in addition to scalars.

Two oracle cases (`mod(cumsum(v), 3)`, `rem(cumsum(v), 3)` in broadcast).
**Oracle 106/106** (200 trials each); **85 unit tests** (1 new).
*Non-vacuity*: making `rem` use `floor` (instead of `fix`) makes
the `rem` cases with negative dividend diverge (scalar AND element-wise) while
`mod` stays green — the `floor`/`fix` distinction is therefore indeed
load-bearing.

### Added — transpiler: **MATLAB `deg2rad` / `rad2deg` + element-wise `sign`** proven against real Octave (Phase 2, increment 36)
Angle conversions and vector `sign`:

- **`deg2rad(x)`** / **`rad2deg(x)`** — degree↔radian conversion (multiplication
  by `π/180` resp. `180/π`), scalar or broadcast over a vector (reuses
  `ScalarBin`/`ScalarBroadcast`, no new primitive).
- **`sign(v)`** — **element-wise** form of `sign` (new `ArraySign` node,
  `-1/0/+1` per element); scalar `sign(x)` remains the `Sign` node.
  Unambiguous dispatch on the argument type (unary).

Three oracle cases: `deg2rad(x)` (scalar), `rad2deg(cumsum(v))` (broadcast),
`sign(cumsum(v))` (element-wise). **Oracle 104/104** (200 trials each);
**84 unit tests** (1 new; an old test updated because `sign` now accepts
a vector). *Non-vacuity*: swapping the `>`/`<` branches of `ArraySign` makes
the case diverge (|Δ|=2, RED).

### Added — transpiler: **MATLAB `atan2` / `hypot` / `max` / `min` element-wise on arrays** proven against real Octave (Phase 2, increment 35)
The two-argument math functions become **vectorized**: they now dispatch on
the operand types (scalar∘scalar → scalar, array∘array → element-wise,
scalar↔array mixture → broadcast), reusing the `EwBinFn`/`BroadcastFn` nodes
(created for `.^`). The new helper `lower_math2` centralizes this logic for
`atan2`/`hypot`/`max`/`min`.

- `atan2(y, x)`, `hypot(a, b)`, `max(a, b)`, `min(a, b)` now accept
  **vectors** (element-wise) and scalar↔vector mixtures (broadcast),
  in addition to scalars; the operand order is preserved for `atan2`.

Three oracle cases: `atan2(cumsum(v), flip(v))` (element-wise),
`hypot(cumsum(v), 2)` (broadcast), `max(…)−min(…)` on arrays — since the
operands are built by builtins returning arrays, inference flows naturally
(no dead code). **Oracle 101/101** (200 trials each); **83 unit tests**
(1 new; an old test updated because `hypot` now accepts a vector).
*Non-vacuity*: swapping the element-wise operand order makes `atan2` diverge
(RED) while `max`/`min`, being commutative, stay green.

### Added — transpiler: **MATLAB `expm1` / `log1p`** proven against real Octave (Phase 2, increment 34)
Two math functions **accurate near zero**, mapped onto the exact IEEE methods
`f64::exp_m1` / `f64::ln_1p` and integrated into the `MATH_FNS` pattern
(scalar *and* element-wise):

- **`expm1(x)`** = `exp(x) − 1` without loss of precision for `x` near 0.
- **`log1p(x)`** = `ln(1 + x)` without loss of precision for `x` near 0.

Two oracle cases (scalar `expm1`; element-wise `log1p`). **Oracle
98/98** (200 trials each); **82 unit tests** (1 new). *Non-vacuity*:
mapping `expm1` to `exp` instead of `exp_m1` shifts the result by 1 and turns
the oracle RED.

### Added — transpiler: **MATLAB `conv` + `polyval`** proven against real Octave (Phase 2, increment 33)
Two signal-processing / numerical classics, wired via deterministic preamble
helpers:

- **`conv(a, b)`** — full linear convolution (length `len(a)+len(b)−1`,
  fixed accumulation order hence bit-reproducible); **both** operands are
  inferred as vectors.
- **`polyval(p, x)`** — Horner evaluation of the polynomial with coefficients
  `p` (descending degree) at the scalar point `x`; `p` is inferred as a vector,
  `x` remains scalar.

Two oracle cases. **Oracle 96/96** (200 trials each); **81 unit tests**
(1 new). *Non-vacuity*: replacing the Horner recurrence `acc·x + p[i]` by
`acc + x·p[i]` makes `polyval` diverge and turns the oracle RED.

### Added — transpiler: **MATLAB `kron` + `cumtrapz`** proven against real Octave (Phase 2, increment 32)
Two additional vector operations, wired via deterministic preamble helpers:

- **`kron(a, b)`** — Kronecker product of two vectors (`out[i·n+j] =
  a[i]·b[j]`, length `len(a)·len(b)`); **both** operands are inferred
  as vectors.
- **`cumtrapz(v)`** — **cumulative** trapezoidal integral with unit spacing
  (first element `0`, same length).

Two oracle cases. **Oracle 94/94** (200 trials each); **80 unit tests**
(1 new). *Non-vacuity*: swapping the nesting order of `kron`'s loops
reorders the output and turns the oracle RED.

### Added — transpiler: **MATLAB `diag` (overloaded) + `trapz`** proven against real Octave (Phase 2, increment 31)
Two additions, including MATLAB's **`diag` overload**, disambiguated by the
**operand type** (never guessed):

- **`diag(A)`** where `A` is a matrix → **extraction** of the diagonal (vector;
  new `DiagExtract` node).
- **`diag(v)`** where `v` is a vector → **construction** of a diagonal matrix
  (reuses the existing `Diag` node, already proven on the Python side via
  `np.diag`).
- **`trapz(v)`** — trapezoidal integration with unit spacing (`Σ ½·(v[i−1]+v[i])`).

Three oracle cases, exercising both `diag` paths naturally:
`diag(A' * A)` (extraction — diagonal of the Gram matrix) and
`diag(cumsum(v))` (construction), plus `trapz(v)`. **Oracle 92/92** (200 trials
each); **79 unit tests** (2 new). *Non-vacuity*: removing the `½` factor
from `trapz` doubles the result and turns the oracle RED.

### Added — transpiler: **MATLAB `trace(A)` + `cross(a, b)`** proven against real Octave (Phase 2, increment 30)
Two classic operations, wired via deterministic preamble helpers:

- **`trace(A)`** — sum of a matrix's diagonal (scalar); `A` inferred
  as a matrix from the intrinsic.
- **`cross(a, b)`** — cross product of two 3D vectors; **both**
  operands are inferred as vectors.

Two oracle cases. **Oracle 89/89** (200 trials each); **77 unit tests**
(1 new). *Non-vacuity*: flipping a sign in one component of `cross`
makes the case diverge (|Δ|≈1) and turns the oracle RED.

### Added — transpiler: **MATLAB transposition operator `A'` / `A.'`** proven against real Octave (Phase 2, increment 29)
Adds the postfix **transposition** operator — ubiquitous in MATLAB — routed
to the SIR `Transpose` node (already proven on the Python side via `A.T`).

- **`A'`** (conjugate transpose) and **`A.'`** (plain transpose) — identical
  for real matrices. The lexer recognizes `'` (postfix, never a string
  since the subset has no character strings), the parser applies it postfix
  (binding tighter than `^`), the lowering requires a matrix.
- `A'` **proves** that its operand is a matrix (new matrix-proof),
  hence the hint-free inference.

Two oracle cases: `B = A'` (plain transposition) and `C = A' * A` (Gram
matrix, composing transposition and matrix product). **Oracle 87/87** (200
trials each); **76 unit tests** (1 new). *Non-vacuity*: parsing `A'` as
the identity (without transposing) makes `A` lose its matrix type and breaks
both cases (RED).

### Added — transpiler: **MATLAB matrix product `A*b` / `A*B`** routed to `scirust-solvers`, proven against real Octave (Phase 2, increment 28)
Completes MATLAB linear algebra: the `*` operator (**matrix** multiplication
in MATLAB, distinct from the element-wise `.*`) routes to the verified
`matvec` / `matmul` kernels of `scirust-solvers` when the left operand is
a **matrix** (inferred from `det`/`inv`/`eig`/`\`).

- **`A * x`** (matrix × vector) → `matvec`.
- **`A * B`** (matrix × matrix) → `matmul` (matrix output).
- Otherwise, `*` remains scalar multiplication or scalar↔array broadcast;
  `A * b` with two arrays remains rejected (pointing to `.*`), and the
  unsupported matrix forms (scalar×matrix, `matrix/`) are clearly rejected.

**Safe** disambiguation: routing only happens if the matrix operand is
already typed as a matrix by another operation (never guessed). The oracle
cases exercise it naturally, without dead code: **residual `r = A*(A\b) ≈ b`**
(matvec) and **`C = A*inv(A) ≈ I`** (matmul). **Oracle 85/85** (200 trials
each); **75 unit tests** (1 new). *Non-vacuity*: transposing the matrix in
`matvec` makes both the MATLAB residual and the Python `A @ b` case (shared
emitter) diverge and turns the oracle RED.

### Added — transpiler: **MATLAB `linspace(a, b, n)`** — vector constructor, proven against real Octave (Phase 2, increment 27)
First **array construction** of the MATLAB front-end (until now arrays
came from parameters or transformations). `linspace(a, b, n)` produces `n`
regularly spaced points from `a` to `b` inclusive, with **exact endpoints**
(like MATLAB, which forces `y(end) = b`) and the `linspace(a, b, 1) = [b]`
case.

- `a`, `b` are scalars; `n` is an **integer** (literal or integer expression
  like `length(x)`), lowered via the existing integer-index path.
- Wired via a deterministic preamble helper.

One oracle case (fixed length 6, `a`/`b` drawn at random). **Oracle 83/83**
(200 trials each); **74 unit tests** (1 new). *Non-vacuity*: using the step
`(b−a)/n` instead of `(b−a)/(n−1)` makes the interior points diverge and
turns the oracle RED (200/200, |Δ|≈0.13).

### Added — transpiler: **MATLAB `var` / `std` / `median`** — reduction statistics, proven against real Octave (Phase 2, increment 26)
Three statistical reductions (vector → scalar), aligned exactly with
Octave's convention (verified empirically before wiring):

- **`var(v)`** — **sample** variance, normalized by **`N−1`** (like MATLAB's
  default, not `N`); `0` for `N < 2`.
- **`std(v)`** — sample standard deviation (`√var`).
- **`median(v)`** — median (middle value; average of the two central values
  for even length).

Wired via deterministic preamble helpers; the argument is inferred as a vector
(added to `is_reduction`). Three oracle cases (var+std, even median, odd
median). **Oracle 82/82** (200 trials each); **73 unit tests** (1 new).
*Non-vacuity*: normalizing `var` by `N` instead of `N−1` makes the case
diverge (200/200, |Δ|≈0.89) and turns the oracle RED.

### Added — transpiler: **MATLAB `cumprod` / `cummax` / `cummin` / `flip`** — vector → vector functions, proven against real Octave (Phase 2, increment 25)
Four additional native functions (a vector in, a vector out), on the same
model as `cumsum`/`diff`/`sort` (deterministic preamble helpers, argument
inferred as vector):

- **`cumprod(v)`** — cumulative product in fixed left→right order.
- **`cummax(v)`** / **`cummin(v)`** — running maximum / minimum.
- **`flip(v)`** — reversed vector.

Four oracle cases. **Oracle 79/79** (200 trials each); **72 unit tests**
(the vector-builtins test now covers all seven functions).
*Non-vacuity*: replacing `>` by `<` in `cummax` makes the case diverge
(200/200, |Δ|=∞) and turns the oracle RED.

### Added — transpiler: **MATLAB `cumsum` / `diff` / `sort`** — vector → vector functions, proven against real Octave (Phase 2, increment 24)
Three unambiguous native functions (a vector in, a vector out), wired via new
deterministic preamble helpers:

- **`cumsum(v)`** — cumulative sum in **fixed left→right order** (hence
  bit-reproducible), same length.
- **`diff(v)`** — consecutive differences `v[i+1] − v[i]` (length `n−1`).
- **`sort(v)`** — **ascending** sort (MATLAB's `sort`).

The argument is inferred as a vector from the intrinsic. Three oracle cases.
**Oracle 76/76** (200 trials each); **72 unit tests** (1 new; the tests'
`sig_of` helper now targets the top-level `pub fn` so as not to confuse a user
function with a same-named preamble helper, e.g. `np::cumsum`). *Non-vacuity*:
reversing the subtraction order of `diff` (`v[i−1] − v[i]`) negates each
element and turns the oracle RED (200/200, |Δ|≈2.4).

### Added — transpiler: **MATLAB element-wise power `.^` on arrays** proven against real Octave (Phase 2, increment 23)
First **vectorized** two-argument operation — the idiom at the heart of MATLAB.
Adds reusable infrastructure (`SirExpr::EwBinFn` for array∘array,
`SirExpr::BroadcastFn` for scalar↔array, `MathFn2::Powf` variant) that will
serve again for the element-wise forms of `max`/`min`/`atan2`/`hypot`.

- **`v .^ w`** (two arrays) → `np::ew2(v, w, |x, y| x.powf(y))`.
- **`v .^ 2`** (array base, scalar exponent) → `np::map1(v, |x| x.powf(2))`.
- **`2 .^ v`** (scalar base, array exponent) → `np::map1(v, |x| (2).powf(x))`
  — operand order is preserved (`.^` is not commutative).
- **`^` on an array** (matrix power `mpower`) remains **refused** with a
  diagnostic pointing to `.^`.

`f64::powf` reproduces Octave's `.^` (verified, including integer exponents). Three
oracle cases. **Oracle 73/73** (200 trials each); **71 unit tests** (2
new). *Non-vacuity*: reversing the broadcast order of `2 .^ v` (computing
`v .^ 2`) makes the case diverge (200/200, |Δ|≈0.83) and turns the oracle RED.

### Added — transpiler: **MATLAB `max(a,b)` / `min(a,b)` (2-arg) + `power(a,b)`** proven against real Octave (Phase 2, increment 22)
Reuses the binary math node (`MathFn2`) for the two-argument forms of
`max`/`min`, distinguished from the one-argument **reduction** by the **number
of arguments**:

- **`max(a, b)`** / **`min(a, b)`** (two scalars) → `f64::max` / `f64::min`.
  The one-argument form remains the reduction over a vector; type inference
  no longer marks the operands of the two-argument form as arrays
  (guard `args.len() == 1` on the reduction proof).
- **`power(a, b)`** → functional form of `a ^ b` (shares the lowering of
  `^`: an integer exponent folds onto `powi`).

Scalar operands. Two oracle cases. **Oracle 70/70** (200 trials each);
**69 unit tests** (2 new). *Non-vacuity*: swapping `max`/`min` in the
two-argument form reverses the sign of the range `max−min` and turns the oracle
RED (200/200, |Δ|≈6.2).

### Added — transpiler: **MATLAB `atan2` / `hypot`** — two-argument math functions, proven against real Octave (Phase 2, increment 21)
Adds to the SIR a reusable **binary scalar math node**
(`SirExpr::ScalarBinFn` + `MathFn2`), emitted as `(l).method(r)`, and wires the
first two two-argument functions on the MATLAB side:

- **`atan2(y, x)`** — four-quadrant arctangent (`f64::atan2`), the argument
  order is significant.
- **`hypot(a, b)`** — Euclidean length `√(a²+b²)` without overflow
  (`f64::hypot`).

Scalar operands in this subset. Verified empirically against Octave
(four-quadrant cases included) before wiring. Two oracle cases. **Oracle
68/68** (200 trials each); **67 unit tests** (1 new). *Non-vacuity*:
reversing the argument order of `atan2` makes the case diverge (200/200,
|Δ|≈0.22) and turns the oracle RED — while `hypot`, being symmetric, stays
green (the test therefore proves the position, not just the presence).

### Added — transpiler: **MATLAB `round` / `fix` / `mod` / `rem` / `sign`** — rounding and modular arithmetic, proven against real Octave (Phase 2, increment 20)
Widening of the MATLAB scalar vocabulary, each function aligned exactly
with Octave's semantics (verified empirically before wiring):

- **`round(x)`** — round to nearest, **ties away from zero**
  (`f64::round`). This is *deliberately* different from NumPy's banker's rounding
  (`round half to even`), so `round` is wired only on the MATLAB path.
- **`fix(x)`** — truncation **toward zero** (`f64::trunc`).
- **`mod(a, b)`** — modulo, result with the **sign of the divisor**;
  lowered as `a - b·floor(a/b)`.
- **`rem(a, b)`** — remainder, result with the **sign of the dividend**;
  lowered as `a - b·fix(a/b)`.
- **`sign(x)`** — `-1 / 0 / +1` with **`sign(0) = 0`** (MATLAB semantics,
  distinct from `f64::signum`); new node `SirExpr::Sign` emitted as a bound
  `if/else` (the argument is evaluated only once).

`round`/`fix` also work element-wise (like the other math intrinsics);
`mod`/`rem`/`sign` are scalar in this subset.
Four oracle cases. **Oracle 66/66** (200 trials each); **66 unit
tests** (3 new). *Non-vacuity*: swapping the `+1`/`-1` branches of
`sign` makes the case diverge (200/200, |Δ|=2) and turns the oracle RED.

### Added — transpiler: **MATLAB `norm` / `dot` / `eig`** — norms, dot product and eigenvalues, proven against real Octave (Phase 2, increment 19)
Continuing broad and safe MATLAB coverage, reusing already
verified kernels (no new primitive to prove from scratch). Three intrinsics,
all unambiguous:

- **`norm(v)`** — Euclidean norm (2-norm) of a **vector**, lowered as
  `sqrt(sum(v .* v))` from existing SIR nodes (restricted to a vector;
  the `norm` of a matrix is the spectral norm, a different quantity —
  refused with a diagnostic).
- **`dot(a, b)`** — dot product, routed to the `np::dot` reduction at **fixed
  order** (bit-reproducible). Type inference now marks **both**
  operands as vectors.
- **`eig(A)`** — eigenvalues (ascending order) of a **symmetric** matrix,
  routed to `scirust_solvers::eigen_symmetric`. Octave's `eig` returns
  increasing real eigenvalues for a symmetric input, so this
  route is proven on symmetric inputs (via `SymMatrix` in the oracle);
  `A` is inferred as a matrix from the intrinsic.

Three oracle cases: `norm(v)`, `dot(a,b)`, `eig(A)`. **Oracle 62/62** (200
trials each); **63 unit tests** (3 new). *Non-vacuity*: replacing
`v .* v` by `v + v` in `norm` makes the case diverge (99/200, |Δ|≈1.4) and turns
the oracle RED.

### Added — transpiler: **MATLAB linear algebra — `det` / `inv` / `\` (solve)** routed to `scirust-solvers`, proven against real Octave (Phase 2, increment 18)
MATLAB coverage gains the linear algebra of the numerical core, reusing the
verified kernels of `scirust-solvers` already wired on the Python side. (1) **`det(A)`**
and **`inv(A)`** become MATLAB intrinsics (scalar determinant,
2-D matrix inverse). (2) **`A \ b`** — the left-division operator, the
idiomatic way of solving `Ax = b` in MATLAB — is lexed (`\`), parsed
(`MBinOp::LDiv`) and lowered into `LinSolve` (LU factorization). (3) **Matrix
parameter inference**: the arguments of `det`/`inv` and the left side of `\`
prove that a parameter is a **matrix** (`infer_param_ty` now tests
the matrix-proof before the array-proof); `\` requires a matrix on the left and a
vector on the right (clear diagnostic otherwise). The Octave oracle now serializes
matrix outputs in **row-major** order (`r.'`) and vectors in **column**
order to align with the `A \ b` semantics. Three oracle cases: `det(A)`,
`inv(A)` (matrix output), `A \ b` (solve). **Oracle 59/59** (200 trials
each); **60 unit tests** (3 new). *Non-vacuity*: serializing `inv`
column-major makes the non-symmetric case diverge and turns the oracle RED.

### Added — transpiler: **MATLAB multi-output `[a,b] = f(…)` + widened MATLAB vocabulary** proven against real Octave (Phase 2, increment 17)
Toward broad and safe MATLAB coverage. (1) **Multi-output functions**:
`function [o1, o2, …] = f(x) … end` transpiles to a `pub fn … -> (T0, T1, …)`
(tuple return), reusing the `RetTy::Tuple`/`ReturnTuple` machinery from the
Python side (increment 16). Lexer extended (`[`/`]` with depth tracking), parser
(output list between brackets), lowering (`ReturnTuple` of the output
variables). (2) **MATLAB intrinsics aligned with Python**: new math
functions `log`/`log10`/`floor`/`ceil`/`sinh`/`cosh`/`atan` and reductions
`prod`/`mean`/`max`/`min` (reductions also count as array proof
for parameter inference). The Octave oracle now captures multiple
outputs (`[o1,…] = f(args)`). Four oracle cases: `[s,d]=sumdiff`,
`[n,ss]=normstats`, `[lo,mu,hi]=stats3` (min/mean/max), `mathx` (log/floor/atan).
**Oracle 56/56** (200 trials each); **57 unit tests** (4 new).
Non-vacuity re-verified: reversing the order of the outputs makes the
multi-output cases diverge and turns the oracle RED.

### Added — transpiler: **general tuple returns** (`return a, b`) proven against real NumPy (Phase 2, increment 16)
Completes the tuple story on the **production** side: a function can return
multiple values (`def minmax(x): return np.min(x), np.max(x)` → `-> (f64, f64)`).
New features: `RetTy` (simple or tuple return, outside the `Copy` `Ty` lattice),
`SirStmt::ReturnTuple`, Python parsing of `return e0, e1, …`, and oracle
serialization of tuple returns (field-by-field printing `r.0`, `r.1`, …). The
tuple elements are restricted to **scalars** in this subset; a
mixed simple/tuple return is refused, and a function returning a tuple cannot
be called as a value (clear diagnostic). Three oracle cases: `addsub`
(a+b, a-b), `minmax` (min, max), `stats3` (min, mean, max). **Oracle 52/52**
(200 trials each); **53 unit tests**. Non-vacuity re-verified: reversing
the order of the tuple elements at emission makes the three cases diverge (|Δ|≈5)
and turns the oracle RED.

### Added — transpiler: **widened numeric vocabulary** (log/floor/ceil/sinh/…, prod/mean/max/min) proven against real NumPy (Phase 2, increment 15)
Seven new elementary math functions (scalar or array) — `np.log`
(→ `ln`), `np.log10`, `np.floor`, `np.ceil`, `np.sinh`, `np.cosh`, `np.arctan`
(→ `atan`) — and four reductions — `np.prod`, `np.mean` (desugared as
`sum(a)/len(a)`, without a new node), `np.max`, `np.min`. New `MathFn`
(Ln/Log10/Floor/Ceil/Sinh/Cosh/Atan) and `SirExpr::Prod`/`Max`/`Min` with
pinned-order prelude helpers (reproducible ascending `prod`). The
reductions also count as array proof for the type inference of
parameters. Five oracle cases (log+log10, floor+ceil, sinh/cosh/arctan,
max−min+mean, prod). **Oracle 49/49** (200 trials each); **48 unit
tests**. Non-vacuity re-verified: mapping `np.log` onto `log10` makes the
case diverge (|Δ|≈0.9) and turns the oracle RED.

### Added — transpiler: **`np.linalg.qr`** (destructuring `Q, R = …`) proven against real NumPy (Phase 2, increment 14)
Second multi-output kernel, on the same `TupleExpr` extension point as the
SVD. `Q, R = np.linalg.qr(A)` transpiles to the verified Householder QR
`scirust_solvers::linalg::qr_decompose` (`Q` orthogonal via `.q()`, `R`
upper triangular via `.r()`). On a **square** matrix, `q()` (m×m) coincides
with numpy's reduced Q, so the shapes match. As the signs of Q/R
depend on the gauge, the proof bears on the **invariant reconstruction**
`Q @ R ≈ A`. **Oracle 44/44** (200 trials each); **45 unit tests**.
Non-vacuity re-verified: swapping `q()`/`r()` (emitting `(R, Q)`) makes
the reconstruction diverge (|Δ|≈0.48) and turns the oracle RED.

### Added — transpiler: **widened Python** (user function calls + lists) proven against real NumPy (Phase 2, increment 13)
The transpiler now composes **several functions**: a transpiled `def`
can call another defined **earlier** in the module (define-before-use),
and **literal lists** `[a, b, c]` become `Vec<f64>`. New features:
a `FuncSig`/`Sigs` signature map threaded through lowering (each
function sees the signatures of the previous ones), `SirExpr::UserCall`
(direct type-checked call) and `SirExpr::ArrayLit`, plus Python parsing of
lists. **Annotation-free type inference across calls**: a parameter
passed to a user function inherits the type of the corresponding parameter of
the callee — hence `sumdbl(x)` where `x` is inferred `&[f64]` only because `dbl`
expects an array. The parameters of called functions are restricted to
scalar/array (unambiguous argument coercion at emission). Four oracle
cases: scalar composition (`sumsq`→`sq`), annotation-free array
composition (`sumdbl`→`dbl`), a 3-level chain (`chain`→`twice`→`inc`), and a
literal list as a weight vector (`wavg` via `np.dot`). **Oracle 43/43**
(200 trials each); **43 unit tests** (7 new). Non-vacuity
re-verified: injecting a `+ 1.0` shift in the call emission makes
the three composition cases diverge (RED) while the literal list stays green.

### Added — transpiler: **tuples + `np.linalg.svd`** proven against real NumPy (Phase 2, increment 12)
First **multi-output kernel** and first **tuple destructuring**.
`U, S, Vh = np.linalg.svd(A)` transpiles to the verified thin SVD
`scirust_solvers::linalg::svd`, with `Vh = Vᵀ` to match the third return
of `numpy.linalg.svd`. New features: `TupleExpr` (enum of tuple-producing
calls, outside the `Copy` `Ty` lattice), `SirStmt::LetTuple` (destructuring
bind `let (n0, n1, …): (T0, T1, …) = …`), `SirExpr::Diag` (`np.diag(v)` → square
diagonal matrix), and Python parsing of the tuple target `a, b, c = …`. On a
**square** matrix, the thin SVD == full SVD, so the shapes match
numpy. Two oracle cases prove the route in two complementary ways:
(a) **singular values** `S` (unique, decreasing) compared directly to
numpy; (b) **reconstruction** `U @ diag(S) @ Vh ≈ A` (gauge-invariant, hence
robust to sign ambiguities of U/V — and actually exercising U and V).
**Oracle 39/39** (200 trials each); **36 unit tests** (5 new).
Non-vacuity re-verified: removing the transpose from `Vh` makes the
reconstruction diverge (|Δ|≈1.3) and turns the oracle RED, while the
"singular values" case stays green — proof that the reconstruction exercises U and V.

### Added — transpiler: **MATLAB/Octave front-end** proven against real Octave (Phase 2, increment 11)
Second source language, on the **same** SIR + emitter as Python — hence the same
determinism and the same verified `scirust-*` kernels. Dedicated front-end
(`src/front_matlab/{lexer,parser,ast,mod}.rs`) + lowering `src/lower_matlab.rs`,
and public API `transpile_matlab` / `transpile_matlab_to_sir`. MATLAB semantics
handled: **1-based** indexing (`a(i)` → `a[i-1]`), **inclusive** `for` ranges
(`1:n` → `1..n+1`), **element-wise** operators `.*`/`./`/`.^` (operands
inferred as arrays) vs scalar `* / ^`, comparisons including `~=`, `if`/`elseif`/
`else` + `while`, and **return by output variable**. New
`SirStmt::Declare` (hoisted declaration without initializer, validated by
Rust's definite-assignment analysis) for locals/outputs first
assigned in a branch. The oracle runs the MATLAB cases against **real Octave**
(9 cases × 200 trials) in addition to the Python cases against NumPy. MATLAB non-vacuity:
breaking the 1-based indexing (`i-1` → `i-2`) makes `mysum` crash and turns
the oracle RED.

### Added — transpiler: matrix-matrix `A @ B` + transpose `A.T` (Phase 1, increment 10)
Completes dense linear algebra. `A.T` (transpose) and `A @ B` (matrix-matrix
product) → `scirust_solvers::Matrix::transpose`/`matmul`. New features:
`SirExpr::Matmul`/`Transpose`; an `as_matrix` emission helper that accepts
indifferently a flat matrix-parameter or a produced `Matrix` value,
hence **chaining** (`A @ A.T`, and matrix operations accepting a
`MatrixVal`). Oracle cases: `A.T` and `A @ A.T` (Gram matrix) vs numpy.
**Oracle 28/28** (200 trials each); 24 unit tests.

### Added — transpiler: `np.linalg.inv` (2-D matrix return) (Phase 1, increment 9)
First **2-D matrix return**: `np.linalg.inv(A)` transpiles to
`scirust_solvers::Matrix::inverse` and returns a `scirust_solvers::Matrix` value
(which carries its shape). New `Ty::MatrixVal`, `SirExpr::Inv`; the oracle
serializes a matrix return by flattening row-major (via `rows()`/`row()`),
compared against `numpy.linalg.inv`. **Oracle 26/26** (200 trials each); 23 unit
tests.

### Added — six extensions of inertial tolerancing (`scirust-tolerance`)
Six new modules that extend the crate beyond the independent linear chain
and position-only dimensioning, each verified by
fuzzing cross-check against an **independent reference**:

- **`montecarlo`**: Monte-Carlo simulation of tolerances. Component
  distributions (normal, uniform, triangular, exact moments), deterministic seeded RNG
  (xorshift64\* + Box–Muller), and `simulate` which pushes `n` draws through
  an arbitrary transfer function `Y = f(X₁…Xₙ)` → mean, dispersion,
  **inertia at target**, ppm, yield, percentiles `0.135/50/99.865 %`.
  *Cross-check*: a linear combination of normals reproduces `Σαμ`, `Σα²σ²`.
- **`correlated`**: **correlated** and **non-linear** chains. Quadratic
  form `I_Y² = (α∘I)ᵀ R (α∘I)` (reduces to the `√(Σα²I²)` of `chain` for
  `R=𝕀`), linearization by finite differences (`gradient`), variance
  `gᵀΣg` (`correlated_variance`), and second-order mean correction
  `f(μ)+½ Σ Hᵢᵢσᵢ²`. *Cross-check*: gradient vs analytic derivative; second-order
  mean vs the exact moment of a quadratic; vs Monte-Carlo.
- **`geometry`**: the rest of the **ISO 1101** characteristics — straightness,
  flatness, circularity, cylindricity (form, by least squares), parallelism
  / perpendicularity / angularity (orientation, zone `L·sin Δθ`), profile and
  runout — each with its **inertial** reading (RMS of deviations).
  *Cross-check*: orthogonality of the least-squares plane residuals; perfect
  form → 0; orientation vs vector/cross products.
- **`sensitivity`**: contribution analysis — each component's share of
  the assembly inertia `cᵢ = αᵢ²Iᵢ²/I_Y²` (and correlated version), sorted. Points
  out the dimensions to tighten. *Cross-check*: the shares sum to 1 and equal the
  direct recomputation.
- **`process`**: allocation to **discrete processes** — multiple-choice
  knapsack solved **exactly** by the Pareto frontier of non-dominated
  `(weight, cost)` states: choosing one process `{inertia, cost}` per
  component minimizing the cost under an inertia budget (statistical or worst case).
  *Cross-check*: vs exhaustive enumeration.
- **`drift`**: short-term vs long-term capability — uniform drift
  variance `σ_lt = √(σ_st² + d²/3)`, Motorola `1.5σ` shift
  (`Cpk↔Ppk`), and long-term ppm. *Cross-check*: `σ_lt` vs a Monte-Carlo of a
  drifting mean plus within-lot noise.
- **`scirust-mcp`**: six new tools — `tolerance_monte_carlo`,
  `tolerance_geometry`, `tolerance_sensitivity`, `tolerance_discrete_allocate`,
  `tolerance_drift`, `tolerance_correlated`.

The `fuzz_crosscheck` harness now covers **14 modules** — **76,534
checks, 0 errors** at 1500 instances.

### Added — non-normal tolerancing + GD&T position (`scirust-tolerance`)
Two modules that extend inertial tolerancing beyond the normal
assumption and to position dimensioning:

- **`nonnormal`** (new module): **non-normal** statistical tolerancing
  from the first four moments (mean, standard deviation, skewness `S`,
  excess kurtosis `K`). Since the inertia `I = √(δ²+σ²)` is *distribution-free*
  (it is the RMS of deviation from target), it is **conformance**
  that depends on the shape. `cornish_fisher_quantile` gives the `p`-quantile via
  the Cornish–Fisher expansion `x_p = μ + σ·w(Φ⁻¹(p))`; `nonnormal_ppm`
  inverts the expansion at each limit for the non-conformance in ppm;
  `clements_capability` provides Clements' (1989) percentile `Cp`/`Cpk`
  on skewed data. All three **reduce exactly** to the classical normal
  results when `S = K = 0`. The inversion `w(z)=t` of a cubic is
  well-posed only on the **monotone branch** around `z=0`: the solver
  locates that branch (walk + bounding), snaps an outside target to its
  endpoint (a limit far in the tail ⇒ contribution ≈ 0, no spurious
  root) and bisects inside — valid for moderate non-normality
  and limits in the body of the distribution (usual capability regime).
- **`position`** (new module): **GD&T / ISO GPS position** dimensioning and
  its inertial form. `true_position = 2·√(Δx²+Δy²)` (diametral deviation),
  `mmc_bonus`/`total_position_tolerance` (maximum material condition bonus according to
  `FeatureType::Internal`/`External`), `coord_to_position`/`position_to_coord`
  (± zone ↔ diametral `Ø` zone conversion), and the **position inertia**
  `√(Iₓ²+I_y²)` — since `E[Δx²+Δy²] = Iₓ²+I_y²`, exactly the
  `vector_inertia` of the two axes, which ties position to the inertial framework.
- **`scirust-mcp`**: new tools `tolerance_nonnormal_capability`
  (non-normal ppm + Clements capability) and `tolerance_position`
  (true position + MMC bonus + position inertia).
- **Fuzzing cross-check** (`fuzz_crosscheck`) extended to these two modules:
  exact reduction to the normal case, round-trip consistency of the
  Cornish–Fisher inversion over its valid domain, tail monotonicity vs skewness,
  and radial position identities — **0 errors** over 10,000 instances. The
  fuzzing revealed and fixed a spurious root of the inversion for a target
  below the minimum of the monotone branch (inflated low tail); hence the
  robust walk-bound-bisect solver.
- **Visualization** (`scirust-tolerance/viz/inertia_cone.html`): standalone
  interactive HTML page of the **inertia cone** — the `(δ, σ)` acceptance map
  (inertial half-disc vs Cpk triangle), the 3D cone `z = √(δ²+σ²)`
  cut by the `I_max` plane, the batch distribution, and live reading of
  `I`/`Cpi`/`Cpm`/`Cp`/`Cpk`/ppm by dragging the batch point or the sliders.
  No network dependency, light/dark theme.

### Added — transpiler: exhaustive test coverage + global script
Goal "test **all** coded functions": the differential oracle
now covers **every** supported intrinsic and operator. New cases —
`np.sin`/`np.cos`/`np.abs` (scalar), `np.exp` (scalar **and** element-wise
on arrays), the `**` operator, and `np.ones` + `len` (array output) —
bringing the oracle to **19/19** (200 trials each vs real NumPy). Added the
`scripts/test_transpiler.sh` script which runs in one point the complete suite (17 unit
tests + oracle) with a clear report and a non-zero exit code if a single
transpiled function diverges from NumPy.

### Added — transpiler: `np.linalg.det` routing (Phase 1, increment 4)
Second kernel routed to `scirust-solvers`: `np.linalg.det(A)` transpiles to
`scirust_solvers::Matrix::from_row_major(...).determinant()` (proven LU
determinant). Reuses the `Ty::Matrix` infrastructure + bi-mode oracle (cargo
compilation). `SirExpr::Det` added; matrix parameter inference extended to arg 0
of `np.linalg.det`. New oracle case on 4×4 matrices compared to
`numpy.linalg.det`. **Oracle 14/14** (200 trials each); 17 unit tests.

### Added — transpiler: routing to the verified kernels (Phase 1, increment 3)
First **routing to a verified `scirust-*` kernel**: `np.linalg.solve(A, b)`
is transpiled to `scirust_solvers::linalg::solve` (proven LU solve) instead
of being re-derived in Rust std. This is the central differentiator of the
design — one does not re-implement the numerics, one routes to
oracle-validated kernels.
- SIR: `Ty::Matrix` (flat row-major 2-D matrix), `SirExpr::LinSolve`,
  `required_crates(&SirModule)` function which declares the `scirust-*` crates
  needed; matrix parameter inference (arg 0 of `np.linalg.solve`).
- **Bi-mode oracle**: std-only cases still compile with `rustc` alone;
  routed cases compile in a standalone cargo project depending (by path) on
  `scirust-solvers`, with a shared target (the deps tree compiles
  once). New case: `np.linalg.solve` on 5×5 diagonally
  dominant systems, compared to `numpy.linalg.solve`. **Oracle 13/13** (200 trials
  each). 16 unit tests.

### Added — transpiler: `while` loops (Phase 1, increment 2)
The input Python subset of the transpiler now supports **`while` loops**
(condition = scalar comparison), unlocking iterative
algorithms (Newton, fixed point, bisection). Proven by the same differential
oracle against real NumPy with two **Newton's method** cases — at
fixed iteration count and with a convergence condition (the iteration count
depends on the data but stays identical on the Rust and NumPy sides, the
floating-point operations being bit-identical). **Oracle 12/12** (200 trials each); 14
unit tests. `SirStmt::While` added; emitter, parser and parameter
inference extended.

### Added — transpiler: `if`/`elif`/`else` control flow (Phase 1, increment 1)
Extension of the Python subset with **scalar control flow**, proven
by the same differential oracle against real NumPy:
- front-end: `if`/`elif`/`else` statements (`elif` desugared into a nested `if`
  in the `else` branch); comparison operators `< <= > >= == !=` as
  boolean conditions (a comparison is only valid as a condition, never
  as a value — otherwise refused).
- SIR: `Ty::Bool`, `SirStmt::If`, `SirExpr::Cmp`; parameter inference and
  emitter extended; branches follow the same "initialize before"
  rule as loops.
- oracle: 3 new cases (relu, clamp, sign) → **10/10 conforming cases**
  (200 trials each); 13 unit tests.

### Added — minimal-cost tolerance synthesis (`scirust-tolerance`)
The "optimal computation" of inertial tolerancing: new module `optimize`
which minimizes the total manufacturing cost `Σᵢ bᵢ·Iᵢ^(−rᵢ)` (inverse-power
cost-tolerance model, Chase & Greenwood) under **several simultaneous
functional requirements** `√(Σᵢ αₖᵢ² Iᵢ²) ≤ I_max,ₖ`. In
variables `vᵢ=Iᵢ²` the cost is convex and the constraints linear, hence a
convex program with strong duality: the Lagrangian separates per component
(`Iᵢ = ((rᵢ/2)bᵢ/sᵢ)^{1/(rᵢ+2)}`, `sᵢ=Σₖ μₖ αₖᵢ²`) and the dual is
maximized by a scale-invariant multiplicative update
`μₖ ← μₖ·(reachedₖ²/I_max,ₖ²)^γ` whose fixed point is exactly the
KKT point (active constraint ⇒ reached=budget, slack constraint ⇒ μₖ→0). For a
single requirement, exactly reproduces the closed form `Allocation::CostOptimal`.
Provides `Component`, `Requirement`, `optimize`/`optimize_with`,
`OptimizeResult` (inertias, total cost, dual multipliers/prices, active
requirements), and the **cost-quality Pareto frontier** `cost_quality_frontier`.
Verified by: equality to the single-requirement closed form, satisfaction of
the KKT conditions with two requirements, cost ≤ naive per-requirement allocation, and
frontier monotonicity. **Fuzzing cross-check** (example
`fuzz_optimize`) over 1500+ random instances against an independent
purely-primal optimality certificate (feasibility + "every component
pinned": no inertia can grow without violating a constraint,
which is necessary for optimality since the cost strictly decreases in I).
The fuzzing revealed that a run that had reached `max_iters` on
nearly-parallel constraints could leave a constraint marginally
exceeded (~4 ppm); fixed by a **feasibility guard-rail** (final uniform
tightening `f = 1/maxₖ(reachedₖ/I_max,ₖ)`) which now **guarantees** that
the returned allocation always respects every budget — preferable, for
tolerancing, to a slightly infeasible solution. New MCP tool
`tolerance_optimize_cost`.

### Added — form and modal tolerancing (`scirust-tolerance`)
"Surface + modal" complement to Adragna's thesis (*Tolérancement des
Systèmes Assemblés, une approche par le Tolérancement Inertiel et Modal*,
tel-00403876; arXiv:1002.0251) which extends inertial tolerancing from a
single scalar characteristic to an entire measured surface:

- **`form`** (new module): `FormBatch` on a measurement matrix
  (parts × points, deviation from nominal). The **surface inertia**
  `I_S = √((1/m) Σⱼ Iⱼ²)` is the quadratic mean of the point inertias,
  equal to the RMS of all deviations from nominal — verified by the identity
  `I_S² = (1/(m·n)) Σᵢⱼ xᵢⱼ²`. Also provides per-point inertias, the worst
  point, and the mean form signature.
- **`modal`** (new module): modal decomposition of form defects
  "in the manner of Fourier series". `ModalBasis` (exactly orthonormal
  DCT-II basis, user basis, or Gram-Schmidt orthonormalization of
  a FEM basis), `decompose`/`reconstruct`/`residual_norm`
  (Parseval `Σ λₖ² = ‖d‖²`), and `modal_inertias` whose partition
  identity **`Σₖ Iₖ² = m·I_S²`** makes the tolerancing of modes (small
  set of physical budgets: mode 0 = size, 1 = tilt, 2 = ovality…)
  equivalent to tolerancing the whole surface.
- **`spatial`** (new module): **3D inertial tolerancing by small-displacement
  torsors** (SDT, after Bourdet & Clément;
  Adragna/Samper/Pillet, arXiv:1002.0253). The deviation of a point is
  `d(M) = T + R × OM`, and the normal deviation `e(M) = d(M)·n = T·n + R·(OM×n)
  = g(M)·θ` with the influence vector `g = [n ; OM×n]`. `Torsor`,
  `Feature` (sample of points+normals), `fit_torsor` (least-squares
  association `θ=(GᵀG)⁻¹Gᵀe` by Gaussian elimination with pivoting,
  returns `None` if the surface is under-constrained — a single plane
  only observes 3 DOF), `form_residual` (residual form defect, to be
  passed to `modal`), and the **surface inertia** `I_S² = θ̄ᵀHθ̄ + tr(HΣ_θ)`
  with `H=(1/m)Σ g gᵀ` — the exact statistical combination of the
  **location** (T) and **orientation** (R) defects, with its
  location/orientation/coupling decomposition. The analytical form is verified equal to
  the empirical one (via `FormBatch`) and the association verified by
  round-trip on a full-scale 3-2-1 datum part. This **replaces**
  the former "not delivered" limitation: 3D geometry by torsors is
  now provided and verified.
- **`scirust-mcp`**: new tools `tolerance_form_modal` (surface
  inertia + modal decomposition) and `tolerance_3d_surface_inertia`
  (3D surface inertia + location/orientation decomposition).
### Added — agentic crypto trading platform (`scirust-trader` + `scirust-mcp`)
Major extension of the MVP `scirust-trader` (market→indicators→model→
certification→risk→LLM→proof) into a pro-level trading toolkit, **fully
drivable by an agentic LLM via MCP** and **simulation/paper-trading first** (no
real order execution exposed; live Binance market data remains read-only behind
`--features live`). Everything is pure Rust, deterministic (same input ⇒ same
output and same proof fingerprints), with no new dependency.

- **Indicators (`indicators.rs`)** — +12 pro indicators beyond
  RSI/MACD/ATR/Bollinger/SMA/EMA: Stochastic (%K/%D), ADX/DMI (+DI/−DI,
  correct Wilder smoothing, ADX priming at `2·period−1`), OBV, rolling VWAP,
  Williams %R, CCI (mean absolute deviation), MFI, ROC, momentum, Z-score,
  Chaikin Money Flow, Supertrend (ATR bands + flip/reversal logic),
  Donchian and Keltner channels, rolling extrema.
- **Chart patterns (`patterns.rs`)** — deterministic detection of doji,
  hammer/hanging man, inverted hammer/shooting star, marubozu, engulfing, piercing
  line/dark cloud, morning/evening stars, three soldiers/crows.
- **Order book (`orderbook.rs`)** — microstructure: mid, size-weighted micro-price,
  spread (bps), depth, imbalance, **execution VWAP by walking the book**,
  slippage and liquidity within X bps.
- **Orders & matching engine (`orders.rs`)** — Market/Limit/
  Stop/StopLimit/TakeProfit order types, TIF (GTC/IOC/FOK), post-only/reduce-only, maker/taker
  fees, slippage model, tick/lot rounding, and deterministic *paper* fill
  logic on candlesticks (standard backtest semantics).
- **Portfolio (`portfolio.rs`)** — multi-asset accounts, netted long/short
  positions (average cost, realized/unrealized PnL, reversal through zero),
  mark-to-market equity, gross/net exposure, rebalancing toward target
  weights, isolated liquidation price (leverage).
- **Metrics (`metrics.rs`)** — Sharpe, Sortino, Calmar, CAGR, annualized
  volatility, max drawdown, Ulcer Index, historical VaR/CVaR, Kelly
  (discrete & continuous), win-rate, profit factor, expectancy, correlation, beta.
- **Strategies (`strategy.rs`)** — `Strategy` trait + archetypes: SMA/EMA
  crossover, RSI mean-reversion, MACD, Bollinger/Donchian breakout, Supertrend,
  momentum; factory by name + parameters (drivable in natural language).
- **Event-driven backtest (`backtest.rs`)** — decision at the close,
  execution at the next open (**no look-ahead**), real fees/slippage,
  round-trip trade journal, complete performance report,
  buy-and-hold comparison.
- **Opportunity discovery (`scanner.rs`)** — the core of "find me
  trades that respect these conditions, with a profit target of X":
  backtests every strategy × symbol, reads the current signal, filters on
  constraints (return, drawdown, Sharpe, win-rate, profit factor, direction),
  sizes an ATR-based entry/stop/take-profit/position plan, ranks, and
  **seals each opportunity + the report with a verifiable SHA-256 proof**.
- **Micro-order execution (`execution.rs`)** — splitting a parent order
  into fast child orders: TWAP, VWAP (volume profile), POV, Iceberg,
  micro-burst, and the **Almgren-Chriss** optimal trajectory
  (`x_j=X·sinh(κ(T−t_j))/sinh(κT)`, `η̃=η−½γτ`), plus execution-quality
  simulation (realized VWAP, slippage vs arrival price).
- **Market making (`marketmaking.rs`)** — optimal **Avellaneda-
  Stoikov** quotes: reservation price `r=s−q·γ·σ²·(T−t)`, optimal spread
  `γ·σ²·(T−t)+(2/γ)·ln(1+γ/κ)`, inventory skew, GLFT approximation.
- **Microstructure signals (`microstructure.rs`)** — Order-Flow Imbalance
  (Cont-Kukanov-Stoikov), trade-flow imbalance, VPIN (flow toxicity,
  bulk-volume classification), Kyle's lambda (price impact).
- **SVG charts (`chart.rs`)** — candlesticks + indicator overlays +
  entry/exit markers and equity curves, in standalone SVG that the LLM
  displays directly ("provide charts").
- **MCP tools (`scirust-mcp/src/tools/trader.rs`)** — 26 tools exposing the whole
  pipeline to any MCP agent: `trader_market_data`,
  `trader_indicators`, `trader_patterns`, `trader_signal`, `trader_backtest`,
  `trader_scan_opportunities`, `trader_orderbook`, `trader_size_position`,
  `trader_execution_plan`, `trader_market_making_quotes`,
  `trader_microstructure`, `trader_metrics`, `trader_chart`,
  `trader_certified_predict` (ML prediction bounded by IBP), `trader_portfolio`
  (portfolio state: realized/unrealized PnL, mark-to-market equity, gross/net
  exposure, liquidation price with leverage), `trader_rebalance`
  (orders to reach target weights) and `trader_dashboard` (standalone HTML
  report: opportunities + proofs + metrics cards + equity curve) —
  portfolio and reporting are driven from the chat.
- **Dashboard (`dashboard.rs`)** — generation of a standalone HTML page
  (inline CSS, embedded SVG, light/dark theme) bringing together the opportunity
  scan and a backtest; "show me" becomes a shareable visual report
  rather than a wall of JSON.
- **Anti-overfitting robustness (`robustness.rs` + 2 MCP tools)** — a
  scanner that keeps the best of many strategies inevitably finds
  flukes; two guardrails: `walk_forward` (backtest on sequential independent
  segments → **out-of-sample consistency** = fraction of winning windows,
  to distinguish a durable edge from curve fitting) and
  `monte_carlo` (**deterministic** bootstrap resampling of the trade
  journal → equity percentile bands, max drawdown distribution,
  probability of loss and of **ruin**). MCP tools `trader_walkforward` and
  `trader_monte_carlo`.
- **Portfolio construction (`portfolio_opt.rs` + 1 MCP tool)** — going
  from a per-asset signal to a **multi-asset allocation**: return covariance
  and correlation matrix, annualized volatilities, and
  four weighting methods — equal weights, **inverse-vol**,
  **inverse-variance** and **minimum variance** (ridge-regularized Gauss-Jordan
  inversion, falling back to inverse-variance if singular),
  long-only optional. Risk diagnostics: per-asset risk contributions,
  **diversification ratio** and portfolio variance. The
  MCP tool `trader_portfolio_construct` takes returns (or aligned OHLCV
  series), returns target weights + the correlation matrix, and
  hooks into `trader_rebalance` to issue the orders — "build me a
  portfolio" is driven from the chat.
- **Market regime detection (`regime.rs` + 1 MCP tool)** — reading *the state
  of the market* before choosing how to trade it. Three orthogonal readings
  merged into a taxonomy of six regimes (bullish/bearish × calm/volatile,
  plus range and crisis): **rolling realized volatility** classified by
  percentile (calm / high / crisis, volatility being autocorrelated —
  Mandelbrot 1963), **trend strength** = OLS slope of the log-price normalized by
  volatility (a signal/noise t-stat), and **Hurst exponent** via R/S
  analysis (Hurst 1951, Mandelbrot & Wallis 1969; `H>0.5` trending/momentum,
  `H<0.5` mean-reverting). The per-bar labels feed an empirical **Markov
  transition matrix** → expected regime durations and stationary
  (long-term) occupancy. The MCP tool `trader_regime` returns the current
  regime, a **recommended posture** (strategy family + leverage to adapt
  to conditions), and the full transition dynamics — deterministic.
- **Anti-overfitting parameter optimization (`optimize.rs` + 1 MCP
  tool)** — honestly answering "which parameters to use?". A naive
  sweep that keeps the best backtest only overfits the past; this
  module reproduces a systematic desk's validation: (1) **splits**
  the history into a *train* portion and a *holdout* never seen by the
  search; (2) **explores** the grid on the train only, ranking
  candidates not by their best full-sample fit but by their
  **out-of-sample walk-forward consistency** (via `robustness`) — a parameter
  set that only works on a lucky window is poorly ranked even
  in-sample; (3) **confirms** the finalists on the holdout, the Sharpe
  train→holdout degradation (`overfit_gap`) betraying overfitting; (4) returns
  a clear **verdict** (robust / partial / overfit). The MCP tool
  `trader_optimize` accepts an explicit `{param:[values]}` grid or
  per-strategy default grids, five ranking objectives, and bounds the
  sweep (`max_combos`, regular sampling) — deterministic.
- **Statistical arbitrage / pairs trading (`pairs.rs` + 2 MCP tools)** — trading
  the *relationship* between two assets rather than the direction of one: market-neutral
  (long one leg, short the other), profitable even in a flat or bearish market.
  Standard quant toolkit: **hedge ratio** by OLS (β such that
  `A−βB` is stationary), **Engle-Granger cointegration** test (Dickey-Fuller
  t-stat on the AR(1) mean-reversion coefficient of the spread),
  mean-reversion **half-life** (Ornstein-Uhlenbeck), Hurst exponent of the
  spread (independent confirmation `H<0.5`), and **z-score** of the spread for the
  signal (short the spread when it is rich, long when it is cheap).
  `trader_pair_analyze` analyzes a pair (cointegration + hedge + signal +
  verdict); `trader_pair_scan` tests all pairs of a basket and ranks the
  most tradable ones (most stationary spread first) — deterministic.
- **Options / derivatives (`options.rs` + 2 MCP tools)** — a new class
  of instruments: a leveraged, convex claim sensitive to
  **volatility**. The options desk toolkit: **Black-Scholes-Merton pricing**
  of European calls/puts (with continuous carry/dividend yield),
  the **Greeks** in market conventions (delta, gamma, vega per vol point,
  theta per day, rho per rate point), **implied volatility** by robust bounded
  bisection (non-arbitrage bounds checked), and analysis
  (moneyness, intrinsic/time value, breakeven, risk-neutral probability of
  finishing in the money). **Options book** aggregation: net Greeks of a
  portfolio of legs + the amount of spot that **neutralizes the delta**
  (hedging). `trader_option_price` prices an option (+ Greeks + IV);
  `trader_option_book` aggregates a book and computes the delta hedge —
  deterministic (validated: call-put parity, IV round-trip, Black-Scholes
  reference values).
- **CLI (`scirust trader …`)** — new subcommands `strategies`,
  `scan` (opportunity scan on mock data, verified proof), `chart`
  (writes an SVG equity curve) and `dashboard` (writes an HTML report).
- **Wallet connectivity (`wallet.rs` + 7 MCP tools)** — plumbing
  conforming to recognized protocols, **watch-only / dry-run by default**:
  Keccak-256 and HMAC-SHA256 in pure Rust (verified against the Ethereum
  and RFC 4231 vectors), EVM addresses with **EIP-55** checksum (verified
  against the 4 canonical examples), **WalletConnect v2** pairing URI parsing
  + `eip155`/CAIP-2 namespaces, **EIP-1559** transaction construction
  with signature hash (RLP + keccak, unsigned), **EIP-712** domain separator and
  digest, signing of exchange REST requests (Binance/Coinbase,
  HMAC), and a watch-only connector + JSON-RPC balance reading (behind
  `live`). **Security**: any action that signs or moves funds is
  locked behind a `WalletAuthorization` signed out-of-band with a
  server-side key (`SCIRUST_WALLET_KEY`) that the LLM cannot forge;
  exchange secrets come from an environment variable
  (`SCIRUST_EXCHANGE_SECRET`) and never transit through the conversation.
  MCP tools: `wallet_validate_address`, `wallet_parse_walletconnect_uri`,
  `wallet_walletconnect_namespace`, `wallet_build_evm_transaction`,
  `wallet_eip712_hash`, `wallet_sign_exchange_request`,
  `wallet_authorization_status`.
- **Wallet authorization hardening (security review, before any real
  execution)** — the authorization model is strengthened to eliminate
  bypasses of a simple native-value cap, **without enabling any real
  signature** (no ECDSA signature exists; the authorization remains a pure
  capability token). `WalletAuthorization` is now bound to the *transaction
  context* — allowlist of recipients (`allowed_to`) and of **calldata
  selectors** (`allowed_selectors`, empty ⇒ native transfers only, which
  blocks an ERC-20 `transfer` at `value=0` that dodged the native cap),
  per-transaction cap **and cumulative budget** (`cumulative_budget_wei`), and a
  **bound** mode (`bound_tx_hash`), single-use, that only authorizes a transaction
  with the exact hash. A `SpendLedger` enforces single-use and the cumulative budget
  (anti-replay). The signed canonical encoding is **length-prefixed** (no
  more delimiter ambiguity). The validity-window check uses the **server**
  clock, never a client-supplied time. On the exchange side, the REST
  signature **always** refuses withdrawal/transfer/key-management
  endpoints and honors an optional operator allowlist
  (`SCIRUST_EXCHANGE_ALLOWED_PATHS`). Everything remains in simulation; real
  execution behind `live` remains unimplemented and requires a dedicated review.

### Added — industrial verticals D2-D8 from `docs/DOMAIN_ROADMAP.md`
Each domain documented in the market roadmap now receives
an implementation (or, when a piece cannot be verified with sufficient
confidence for safety code, an explicit honest limit
rather than a guessed formula):

- **`scirust-grid`** (existing, completed — D2 network protection): new
  modules `state_estimation` (weighted-least-squares state estimation
  `x̂=(HᵀWH)⁻¹HᵀWz`, bad-data detection via the global χ² test and
  the largest normalized residual test, Abur & Expósito — verified against an
  independently computed 3-node example) and `distance_relay` (multi-zone mho
  comparator, IEEE C37.113 §5.2).
- **`scirust-biomed`** (existing, completed — D3 medical devices):
  new module `control` (`pid`, `iob`, `insulin_safety`, `barrier`) — PID
  with conditional anti-windup, active-insulin tracking via exponential
  decay, threshold-based supervision (suspension on low glucose,
  exit from automatic mode), and a **Control Barrier Function** safety
  filter (Ames et al., IEEE TAC 2017) solved in closed form. Each
  module carries an explicit non-clinical-use warning: this
  demonstrates certifiable control techniques, not an approvable
  dosing algorithm.
- **`scirust-maritime`** (new crate — D5 autonomous maritime):
  `colregs` (COLREG encounter classification by relative bearing),
  `cpa_tcpa` (collision risk assessment, verified against a worked
  two-vessel example: TCPA≈54.5min, CPA≈3.41nm), `thrust_allocation`
  (DP thrust allocation via weighted pseudo-inverse, Fossen 2011,
  verified against the numpy Moore-Penrose pseudo-inverse).
- **`scirust-fab`** (new crate — D6 semiconductors): `r2r`
  (EWMA run-to-run controller, Sachs, Hu & Ingolfsson 1995, verified against
  a worked example and a geometric-convergence proof) and `pca`
  (multivariate FDC T²/SPE, Kourti & MacGregor 1995, on the general SVD of
  `scirust-solvers`) — built on top of the already-present `scirust-spc` (`EwmaChart`,
  `HotellingT2`), without duplicating it.
- **`scirust-agtech`** (new crate — D7 precision agriculture):
  deterministic and auditable yield-map cleaning pipeline
  (`outlier_filter`: global + local filters, Sudduth & Drummond 2007;
  `idw`: inverse-distance-weighting interpolation) addressing
  the documented divergence between QGIS/Agro-Map/Farm Works (Walczykova et
  al. 2018). `agpl` exposes the three risk parameters of
  ISO 25119-2 (Severity/Exposure/Controllability, verified against the
  normative text) but **deliberately does not implement** the decision
  function `S×E×C→AgPL`: the complete risk graph (Figure 1, §6.3.7) does not appear
  in any verifiable open source found.
- **`scirust-fatigue`** (new crate — D4 structural fatigue):
  `rainflow` (cycle counting per ASTM E1049-85 §5.4.4, port of the
  stack-based algorithm verified value by value against the PyPI reference
  library `rainflow` on two independent sequences) and `miner` (the
  Palmgren-Miner damage accumulation rule, power-law generic S-N curve —
  no real material curve is claimed).
- **`scirust-sis`** (completed — D8 nuclear): new module
  `reactor_trip` (`architecture_with_bypass`, `pfd_avg_during_bypass`) —
  reconfiguration of the MooN voting when a channel is bypassed for
  maintenance (IEC 61513 §6.2.3.5, reducing `N` without changing `M`), built
  entirely on the already-verified primitives of `Architecture` and
  `pfd_moon`. The ISA-67.04 threshold methodology remains documented but
  unimplemented (an honest limit, not an omission).
- **`scirust-tolerance`** (new crate — inertial tolerancing): the
  method of M. Pillet and the SYMME laboratory (Adragna, Pillet, Formosa,
  Samper — arXiv:1002.0270), which tolerances the **inertia**
  `I = √(δ² + σ²)` (the quadratic mean deviation from target, i.e.
  `√(E[Taguchi loss]/k)`) rather than the distance to an interval. Five
  modules: `inertia` (`Inertia` type, sample estimation with the `Î²`
  unbiased estimator of `I²`, Taguchi loss, `I_max` budget, inertia
  cone), `capability` (`Cp`/`Cpk`/`Cpm`/`Cpmk`/`Pp`/`Ppk`, the inertial
  index `Cpi = I_max/I` — equal to `Cpm` at the `Cp=1` budget —, non-conformance
  in ppm with an `erfc` tail reliable up to 6σ), `chain` (analysis and
  allocation of 1D tolerance chains: worst-case / statistical / weighted /
  guarantee of a `Cpk` via the coefficient `ICC = √(Cpk²+n/9)`, **verified
  against table 2 of arXiv:1002.0270**: `0.033`/`0.075`/`0.060`),
  `chart` (inertial control chart with limit `UPL(α) = I_max·√(χ²_{n;1−α}/n)`
  and recenter/reduce-dispersion recommendation), `sampling`
  (acceptance sampling by inertia, Pillet & Maire — **non-central** χ² law
  `n·Î²/σ² ~ χ'²(n, λ=n·δ²/σ²)`, efficiency curve and
  synthesis of a `(n, k)` plan satisfying supplier risk α and
  customer risk β), and `special` (`erf`/`erfc`/normal CDF/χ² quantile and **CDF
  of the non-central χ²**, validated against reference values — including
  independent Monte-Carlo anchors for the non-central χ²). The
  `inertia` module also covers **lot mixing** (`I_c² = Σ pᵢ Iᵢ²`, a
  key advantage of inertial tolerancing), multi-DOF/3D combination
  (`vector_inertia`), correction of the observed inertia for measurement
  uncertainty, and a **minimal-cost** allocation (`CostOptimal`, closed-form
  Lagrangian minimum, verified via the KKT conditions). Pure Rust,
  single dependency `serde`. Discovered and fixed by an adversarial
  verification pass: saturation of `erf` at `|x|≥6` (overflow→NaN
  for large `x`).
- **`scirust-mcp`**: one tool per domain above
  (`grid_state_estimate`, `biomed_cbf_safe_dose`, `maritime_collision_risk`,
  `fab_r2r_update`, `agtech_clean_yield_map`, `fatigue_rainflow_damage`,
  `sis_reactor_trip_bypass`, `tolerance_inertial_capability`,
  `tolerance_chain_allocate`, `tolerance_acceptance_plan`) — each added domain
  immediately becomes drivable by an agent, in accordance with the
  single-connector doctrine of `docs/DOMAIN_ROADMAP.md`.

### Added — linear algebra and solvers
- **`scirust-solvers`**: **randomized SVD** (Halko, Martinsson & Tropp 2011 —
  projection onto a random subspace seeded by a homegrown deterministic
  `SplitMix64`, with optional power iterations and QR
  re-orthonormalization) to approximate the truncated SVD of a matrix without
  decomposing it in full; **Anderson acceleration** (Walker & Ni 2011) for
  fixed-point iterations, reduced to unconstrained least squares
  solved by the already-present QR. Same seed ⇒ bit-identical output.
- **`scirust-reliability`**: new general formula `pfd_moon(m, n, ...)`
  generalizing PFDavg to any `M`-out-of-`N` architecture beyond the five
  tabulated by IEC 61508-6 Annex B (validated against the five named cases and
  against 2oo4/3oo4 by independent derivation — see the module doc for
  the naive-generalization near-miss that motivated this thorough
  verification). `scirust-sis::voting::Architecture::pfd_avg` now falls back
  to it instead of refusing non-tabulated architectures (2oo4, etc.).
- **`scirust-sis`**: new "spurious trip" failure mode
  (`fault_injection::simulate_demand_with_spurious`) — models a channel
  stuck in the tripped position, independently of the undetected dangerous
  failures already modeled.
- **`scirust-discovery`**: three new discovery protocols —
  BACnet/IP (Who-Is/I-Am), SNMPv1 (GET sysDescr.0, minimal homegrown BER
  encoder/decoder), EtherNet/IP (CIP ListIdentity — encapsulation header at
  high confidence, internal layout of the Identity item documented as
  less verified for lack of real hardware to confirm).

### Added — process functional safety (IEC 61511/61508 — SIS)
- **`scirust-reliability`** (existing, completed): addition of the missing
  voting architectures `pfd_2oo2` (`λDU·T1`, no β term — a 2oo2 has no
  redundancy to defeat for a dangerous failure) and `pfd_1oo3`
  (`(1−β)³(λT1)³/4 + β·λT1/2`), completing the MooN family
  1oo1/1oo2/2oo2/2oo3/1oo3. `Sil` now derives `Ord` (highest band =
  strongest guarantee). New validation test against a published external
  example (Lundteigen & Rausand, NTNU, ch. 8, slide 27/43:
  2oo3, λDU=1e-6/h, τ=8760h, β=10% → PFDavg≈5.00e-4), in addition to the
  hand derivations already present.
- **`scirust-sis`** (new crate): the systems/logic layer on top of
  these primitives — `M`-out-of-`N` voting architectures (evaluation of votes
  in the trip decision), complete SIF loop (sensors → logic solver
  → final elements, total PFDavg = sum of subsystems, standard
  ISA-TR84.00.02 practice), fault injection (empirically demonstrates
  that a 2oo3 tolerates a failed channel but a 2oo2 does not), cause-and-effect
  matrices evaluated deterministically, proof-test-interval sizing by
  numerical inversion of PFDavg (reuses
  `scirust-solvers::roots::bisection`), and a SHA-256 hash-chained audit
  log of trip decisions and cause-and-effect matrix changes — motivated
  directly by the Triton/Trisis attack (2017)
  against Schneider Triconex safety controllers. Exposed as MCP
  tools (`sis_verify_sif_loop`, `sis_size_proof_test_interval`). Marks
  domain D1 of `docs/DOMAIN_ROADMAP.md` as done.

### Added — agent connector (MCP) and safe OT/IT discovery
- **`scirust-mcp`** (new crate): [Model Context Protocol](https://modelcontextprotocol.io)
  server (JSON-RPC 2.0, stdio transport) exposing SciRust's capabilities — numerical solvers,
  `scirust-sciagent` SLM development tools, OT/IT discovery — as **standard MCP tools**,
  callable by any agent (the embedded SLM, Claude, ChatGPT, a script) without integration-specific
  glue code. Reuses the existing development-tool implementation
  (`scirust_sciagent::agentic::tools::Tool::builtins()`) rather than duplicating it. Every
  `tools/call` — success or failure — is logged into a SHA-256 hash chain (`AuditLog`), modeled on
  `scirust-func-safety::audit` but with a real SHA-256 rather than a homegrown hash. Tools
  provided by default: `dev_*` (inherited from the SLM), `linalg_eigen_symmetric`, `linalg_svd`,
  `linalg_gmres`, `discovery_scan`, and the generic escape hatch `scirust_cli`.
- **`scirust-discovery`** (new crate): **safe, consented and audited** OT/IT asset discovery —
  never a generic port scanner (dangerous on industrial controllers: see
  the SQL Slammer/Davis-Besse 2003 incident and the Coffey et al. 2018 study cited in its `README.md`).
  Protocol-native probes only: OPC-UA UACP `Hello`/`Acknowledge` handshake, Modbus TCP
  `Read Device Identification` (0x2B/0x0E), mDNS/DNS-SD service enumeration. No packet is
  sent without an **HMAC-SHA256-signed** `ScopeAuthorization` validating the target against a
  whitelist of CIDR ranges, a protocol whitelist, a temporal validity window, and an
  IEC 62443 zone security level (SL3+ zones refused by default). Every attempt — in
  scope or refused — is logged into a SHA-256 hash chain. Exposed as MCP tool
  (`discovery_scan`) whose authorization key lives server-side (`SCIRUST_DISCOVERY_KEY`), never
  in the call arguments — an agent cannot self-authorize.
- **`docs/DOMAIN_ROADMAP.md`** (new): market research on regulated sectors (process
  safety IEC 61511, electric grid protection IEC 61850, medical devices IEC 62304,
  aeronautics DO-178C, autonomous maritime DNV/IMO MASS, semiconductors SEMI, precision
  agriculture ISO 25119, nuclear IEC 61513) where SciRust's determinism and auditability
  bring documented value not already covered by the existing crates.

### Added — linear algebra (`scirust-solvers`)
- **Symmetric eigenvalue decomposition** (`linalg::eigen_symmetric`): Householder tridiagonalization
  + implicit QL algorithm with Wilkinson shift (port of the `tred2`/`tql2`
  pair from EISPACK, public domain). **Public and reusable** primitive, unlike
  the private, duplicated cyclic-Jacobi implementation in `scirust-multivariate` used only for PCA.
- **General dense SVD** (`linalg::svd`): one-sided Jacobi (Hestenes 1958), for any
  `(m, n)` matrix — pseudo-inverse, rank-deficient least squares — complementary to the
  truncated-SVD-based `nalgebra` approach of `scirust-core::tn::ops` (designed for tensor networks).
- **Restarted GMRES(m) and BiCGSTAB** (`linalg::gmres`, `linalg::bicgstab`): matrix-free iterative
  solvers for **non-symmetric** systems `A·x=b` (Saad & Schultz 1986; van der Vorst 1992),
  until now covered only by conjugate gradient (SPD only). Sequential Arnoldi
  orthogonalization (modified Gram-Schmidt), deterministic.
- **Jacobi preconditioner** (`linalg::precond::JacobiPreconditioner`), usable with
  `gmres_preconditioned`/`bicgstab_preconditioned`.
- **Spectral projected gradient** (`optimize::spg`): box-constrained optimization
  (Birgin, Martínez & Raydan 2000), Barzilai-Borwein step + non-monotone Armijo line
  search — until now only an ad hoc box QP existed in `scirust-control`.

### Added — quantum simulation via tensor networks
- **MPS / Tensor-Train quantum circuit simulator** (`quantum::Mps`/`MpsNode`): represents
  an `n`-qubit state by a **chain of rank-3 tensors** instead of the `2ⁿ` amplitudes of a
  dense state-vector ⇒ as long as entanglement stays moderate, the cost goes from **exponential** to
  `O(n·χ³)` (`χ` = bond dimension bounding the entanglement at each cut). A 1-qubit gate
  contracts a `2×2` into the physical index in place; a **2-qubit** gate on adjacent
  qubits (1) contracts the two nodes into a tensor `θ`, (2) **applies** the `4×4` gate,
  (3) re-forms a matrix and performs a **truncated SVD** (the **homegrown** `tn::ops::truncated_svd`,
  **pure Rust via nalgebra — zero FFI**), keeping at most `χ` singular values to cap the
  bond dimension. Real `f32` amplitudes (real gates `H`/`X`/`Z`/`CNOT`/`CZ`/`Ry`);
  complex amplitudes (phase/`S`/`T`/`Rz`) are future work. Honest oracle (no
  mock): the MPS **reproduces the dense state-vector exactly** (reference simulator in plain)
  on a **random** 5-qubit / 40-gate circuit + Bell `(|00⟩+|11⟩)/√2` (bond 2) + 3-qubit GHZ;
  **sane truncation** (product state → bond 1; cap `χ=1` ⇒ high-fidelity approximation);
  preserved norm; bit-exact determinism. The same contraction + truncated-SVD machinery
  **is** the already-present Tensor-Train weight compression (`tn::tt_decompose`,
  `nn::tt_linear`) — directly reusable to compress local LLMs (SLHAv2).
  *Architecture note*: deliberate refusal of `openblas-src`/`cuSOLVER` (C/CUDA FFI, would break the
  zero-FFI + bit-exact-determinism thesis) and of `faer` (pure Rust but redundant with nalgebra) —
  the existing homegrown SVD suffices.

### Added — ecosystem synergy (CCOS, SLHAv2)
- **Synergy CLI commands** (`scirust kvcache | guard | attest`): expose the primitives
  below on the command line, deterministic via `--seed`. `kvcache [--budget B]` compresses a
  KV sequence and displays the **compression ratio** + the **cosine fidelity** of attention vs
  full precision (and bounded soft-paging with `--budget`); `guard [--alpha A]` calibrates the guard
  and displays the **empirical coverage** (≥ 1−α) + Accept/Abstain/Reject verdicts; `attest`
  records verifiable inferences in the **hash-chained journal**, verifies the chain, rejects
  a falsified inference and demonstrates tamper-evidence. Documented in `docs/REFERENCE.md` and in
  **8 languages** (`Documentation*.md`).
- **Statistically guaranteed guard** (`nn::guard::StatisticalGuard`): a response gate with a
  **distribution-free coverage guarantee**, to feed the **CCOS** `guard`
  (validate/abstain on a model's output) without an ad-hoc threshold. From the class
  probabilities of a decision, the guard forms the **conformal prediction set** (#21,
  `ConformalClassifier`) and derives a verdict: a single class crossing `1−q̂` ⇒ **Accept**;
  several ⇒ **Abstain** (ambiguous); none ⇒ **Reject** (out-of-distribution). Conformal
  calibration guarantees that the true class is in the set with probability **≥ 1−α** on
  exchangeable data, *whatever the distribution* — the guard thus provably lets the correct
  answer through at most a fraction `α` of the time. Honest oracle: **empirical coverage
  ≥ 1−α** on fresh data (3-class, deterministic) + verdict logic (confident→Accept,
  split→Abstain, flat/OOD→Reject). Deep ensembles (#40) provide complementary epistemic
  signal for the OOD flag.
- **SIMD-accelerated, bit-exact KV codec** (`scirust_simd::ops::dequantize_int4_into`, wired into
  `nn::elastic_kv_cache`): the INT4 dequantization (`out[i]=code[i]·scale`) goes through the
  SIMD `mul_f32` kernel; being **elementwise** (no reduction) and an IEEE-754 product identical per
  lane and in scalar, the result is **bit-identical across SIMD widths and platforms** — the KV codec
  fast path **without breaking determinism** (cosine/attention reductions
  stay on the deterministic path). Oracle: SIMD ≡ scalar **bit-exact** for any length
  (including < one lane) and a range of scales.
- **Hash-chained attestation journal** (`scirust_runtime::attest`): the bridge from scirust's
  **verifiable inference** (`vinfer` #80) to **CCOS**'s `event_log`. Each `InferenceEvent`
  freezes the model commitment, the input hash and the output hash, and chains onto the
  previous one via a **SHA-256 hash** (`inputₙ = H(inputₙ₋₁ ‖ seq ‖ commitment ‖ input ‖ output)`)
  — exactly CCOS's append-only, tamper-evident form, so a scirust runtime's inferences
  ingest into its audit journal. Recomputing the chain re-derives the **same head** (deterministic
  replay); any mutation or reordering **breaks it**. `attest_and_record` additionally verifies,
  *before* appending, that the `(input, output)` pair is an **authentic** inference of the
  committed model (Freivalds over `GF(p)`, #80) — the chain therefore attests only real inferences.
  Honest oracle: the chain verifies and replays (same head); falsification of an event /
  reordering **detected**; an authentic inference is attested and chained while a **falsified**
  output **is rejected** (journal unchanged). Completes the proof stack (#3, `proof`,
  DiFR #5, `vinfer` #80).
- **Elastic compressed KV cache** (`nn::elastic_kv_cache`): the deterministic primitive
  shared behind **SLHAv2** (compressing the KV cache to run an LLM in the CPU's cache
  rather than on a prohibitively expensive GPU) and **CCOS** (bounded-memory paging), built on
  scirust's quantization and determinism. An attention key/value pair is compressed
  into a `KvTile` via **two-level INT4** quantization (symmetric base + INT4 **residue** —
  SLHAv2's "residual tracking"), each level with **per-group adaptive scales**
  (`quantize_int4_grouped`: a finer scale per channel group ⇒ SLHAv2's cosine-aware
  "adaptive scaling", in the spirit of KVQuant #68's per-channel approach), which lifts **cosine**
  fidelity beyond 0.99 while reducing the footprint several-fold versus `f32`. The `ElasticKvCache` retains these
  tiles under an optional **budget** and evicts the oldest on overflow (soft-paging /
  elastic memory — the paging abstraction shared with CCOS), and serves attention directly
  from the compressed tiles by reusing `contiguous_attention` (#63), so the only
  gap versus a full-precision cache is the compression error (measured). Honest oracle:
  reconstruction at high **cosine fidelity** (>0.95, the residue level strictly beating the
  base alone); **compressed attention ≈ full** (cosine >0.99); **compression ratio** ≥3×
  vs `f32`; cache **bounded** under budget (oldest evicted) and **bit-exact deterministic**.
  Codec exposed (`quantize_int4`/`dequantize_int4`/`KvTile`/`cosine_similarity`) for consumption
  by SLHAv2/CCOS. Joins KVQuant (#68) and PagedAttention (#63) in the KV-cache stack.

### Fixed
- **SIMD `portable` — alignment bug (wrong results, non-deterministic)**:
  `add_f32/f64_inplace`, `dot_f32/f64` and `fma_f32` (`scirust-simd::portable`)
  split **each operand independently** via `as_simd`/`as_simd_mut`. When two
  slices had different memory alignments (common: allocation-dependent), the
  core SIMD loops paired **offset** lanes → **incorrect** results, in a
  **non-deterministic** way (hence the `test_add_f32_inplace` test that
  failed ~30–50 % of runs). Rewritten with `chunks_exact`, which pairs the k
  block of each slice identically regardless of alignment. Added a regression
  test covering all relative offsets (add/dot/fma vs scalar reference);
  12/12 runs green. Along the way, a `needless_return` in `complex.rs`
  (`portable-simd` path) fixed.

### Added — "grow scirust" campaign
- **Reluplex — *complete* SMT-style verification** (`nn::ibp::reluplex_verify`/
  `reluplex_unstable_count`, Katz et al. 2017, roadmap #4): a **satisfiability**
  search for a counterexample by **case-splitting ReLU phases** — but
  **lazy**, the signature of Reluplex: a neuron whose pre-activation interval
  stays entirely on one side of 0 over the box is **stable**, so its phase is
  **forced** (never split); only **unstable** neurons are split, i.e. `2^instable`
  leaves instead of the `2^hidden` of the MILP's *eager* enumeration (#31). On
  each leaf (a complete ReLU pattern) the network is affine and a
  counterexample is sought by minimizing each margin over the pattern region
  (the **exact 2D LP** shared with the MILP verifier); the **first**
  counterexample found (SAT) is returned, or Robust. Distinct from
  branch-and-bound (#26, splits the input domain) and MILP (#31, enumerates
  *all* patterns) by the **lazy ReLU-phase splitting**. Honest oracle:
  **agreement with MILP** over a full radius sweep (two exact methods ⇒ same
  decisions); real counterexample (margin ≤ 0, inside the box); at small
  radius, **fewer neurons split** than `hidden` (bound elimination);
  deterministic. Network (2 inputs, 1 layer). **Closes the verification
  stack** (IBP, CROWN, zonotopes, DeepPoly, randomized smoothing, Lipschitz,
  CROWN-IBP, BaB, MILP, Reluplex).
- **Verifiable inference — compact cryptographic argument**
  (`scirust_runtime::vinfer`, ZK-based Verifiable ML, roadmap #80): extends
  the `proof` certificates from bit-exact re-execution to a **succinct
  soundness guarantee**. The model (an integer linear layer quantized over
  the prime field `GF(p)`, `p = 2³¹−1`) is **committed** by hashing its
  weights. To verify a claimed batched output `Y` for inputs `X`, the
  verifier runs the **Freivalds check** over `GF(p)`: draw a random `r` and
  test `W·(X·r) = Y·r`. Computing `W·(X·r)` costs `O(out·in + in·b)` vs
  `O(out·in·b)` to recompute `Y = W·X`, so for a batch it is **succinct**
  (sublinear in the recomputation cost). A false `Y` passes with probability
  `≤ 1/p` per challenge, so a few challenges give negligible soundness
  error. The challenge `r` is derived by **Fiat-Shamir** from a hash of
  `(commitment, X, Y)`, hence non-interactive and **bound to the claimed
  output** (the prover cannot adapt `Y` to a known `r`). Honest oracle:
  accepts a correct inference (deterministic); **soundness** — over 1000
  random forgeries of one output entry, **all** rejected; the commitment
  **binds** the model (verifying against another model's commitment fails);
  Fiat-Shamir **binds** the output (the valid output of *other* inputs is
  rejected for `X`). Provides cryptographic **soundness** (the output
  provably comes from the committed model), **not** zero-knowledge — the
  verifier holds the weights; weight-hiding zk-SNARKs remain out of scope.
  Crowns the proof stack (reproducible summation #3, `proof` certificates,
  DiFR #5).
- **DiFR — inference verification despite nondeterminism**
  (`scirust_runtime::difr::difr_verify`, 2025, roadmap #5): the [`proof`]
  certificates verify an inference by **bit-exact re-execution** — which only
  works if the verifier reproduces the prover's arithmetic identically. But
  on **different hardware** (SIMD widths, FMA, thread counts) floating-point
  summation is **non-deterministic**, so a bit-exact check would reject
  honest outputs. DiFR verifies *despite* this: it recomputes a **canonical
  reference** with `reproducible_dot` (products and sum accumulated in
  `f64`, order-independent) and accepts the claimed output iff it lies
  within a **sound floating-point error envelope** of that reference. *Every*
  honest `f32` computation — in *any* summation order — is provably inside
  the envelope (hence accepted); a **forged** output beyond it is rejected.
  The envelope is the dot-product rounding bound `γ·Σ|terms|` propagated
  through the layers (ReLU is 1-Lipschitz, it transmits it without
  amplification) and stays **tiny** (a few ppm of the activation scale), so
  the check catches any significant forgery. Honest oracle: accepts an `f32`
  computation in a **different summation order**; envelope **sound** (1000
  random orders, all accepted) and **tight** (< 0.001 of scale); **rejects**
  a forgery (beyond the envelope, here enough to change the predicted
  class); deterministic. Extends reproducible summation (#3) and the
  inference-proof tooling.
- **MILP — *exact* verification** (`nn::ibp::milp_min_margin`/
  `milp_verify_robustness`, Tjeng et al. 2019, roadmap #31): exact
  verification of a ReLU network via the MILP formulation. The key
  observation: the ReLU **activation patterns** are precisely the MILP
  **binary** variables, and on the domain of a fixed pattern the network is
  **affine**. For a small network (2 inputs, 1 hidden layer) the patterns
  are **enumerated** and each LP is solved **exactly** — the margin
  `logitₜ − logitⱼ` is affine there, minimized over the box intersected
  with the pattern's activation half-spaces by **vertex enumeration** of the
  2D polygon (no fragile simplex: robust and exact). The global minimum over
  all patterns and all competing classes is therefore **exact**; `> 0` ⇒
  robust, otherwise the argmin is an **exact counterexample**. Honest
  oracle: the enumerated minimum **equals brute force** (it lower-bounds
  every value of a fine grid and the grid approaches it), the counterexample
  is **real** (margin ≤ 0, inside the box), and — being exact — it is **≥
  DeepPoly's (sound) lower bound** everywhere and **strictly tighter** at
  some radii; deterministic. Distinct from branch-and-bound (#26), complete
  **up to tolerance**: MILP is exact (it even slices the measure-zero
  boundary).
- **Branch-and-bound — *complete* verification**
  (`nn::ibp::verify_robustness`/`BabResult`, GCP-CROWN, Zhang et al. 2022,
  roadmap #26): where IBP/CROWN/DeepPoly give **one** *sound but incomplete*
  bound, branch-and-bound **decides**. It bounds the per-class **margins**
  (`logitₜ − logitⱼ`, merged into a final layer so DeepPoly follows the
  correlation) over the input box; if all lower bounds are `> 0` the box is
  **proved robust**; otherwise it probes the **center** of the box for a
  **concrete counterexample**, and failing that **splits** the box along its
  widest axis and recurses. As sub-boxes shrink, DeepPoly's ReLU relaxation
  becomes exact, so the search **decides** (up to a tolerance) — proving
  cases a single bound cannot, and returning a real counterexample when the
  class can actually change. Honest oracle: `Robust` is **sound** (5000
  sampled points well classified); the **certified ℓ∞ radius strictly
  exceeds** DeepPoly's alone (and the extra region is sampled robust);
  `Unsafe` returns a **true** counterexample (misclassified, inside the
  box); deterministic. Exposed in the `certify` CLI. (Branching is on the
  **input domain**; unstable-ReLU splitting and GCP-CROWN cutting planes are
  not implemented.) Crowns the verification stack (IBP #1, CROWN #2,
  zonotopes #29, DeepPoly #28, CROWN-IBP #30).
- **DeepPoly — relational abstract domain**
  (`nn::ibp::deeppoly_certify`/`IbpMlp::certify_deeppoly`, Singh et al. 2019,
  roadmap #28): a robustness verifier more precise than IBP. Where IBP
  treats each neuron with a plain interval (losing all correlation), DeepPoly
  keeps for each neuron a **lower and upper bound affine in the network
  inputs** and **back-substitutes** them layer by layer. The ReLU relaxation
  is **asymmetric**: for an unstable pre-activation range `[l,u]`, the upper
  bound is the **chord** `z ≤ (u/(u−l))(y−l)` and the lower bound `z ≥ λy`
  with `λ` chosen to **minimize the area** of the relaxation (`λ=1` if
  `u>−l`, else `0`). Since the bounds stay affine, correlations are
  preserved and the result is tighter than IBP — **at any depth** (where
  `crown_bounds` was limited to 2 layers). Honest oracle: **sound** (4000
  sampled points ∈ certified box, 3-layer MLP) + **strictly tighter than
  IBP** on `relu(x)+relu(−x)=|x|` over `x∈[−1,1]` (DeepPoly gives the
  **exact** box [0,1] because the `x` cancels in the upper bound, vs IBP
  [0,2]) + determinism. Exposed in the `certify` CLI (next to IBP, CROWN,
  zonotopes, smoothing). Extends IBP (#1) / CROWN (#2) / zonotopes (#29).
- **CROWN-IBP — certified (verified) training**
  (`nn::crown_ibp::CrownIbpMlp`, Zhang et al. 2020, roadmap #30): ordinary
  training minimizes the loss at *concrete* inputs — a network can fit them
  perfectly and yet **change prediction** under a minimal perturbation.
  CROWN-IBP instead trains on a **certified bound of the worst-case loss**
  over an ℓ∞ ball around each input, making the network **provably**
  robust. The key idea: **interval bound propagation (IBP) is
  differentiable**. For an affine layer `y=x·W+b`, the box transforms to
  `center'=center·W+b`, `radius'=radius·|W|` — and `|W|=relu(W)+relu(−W)`,
  so the whole bound (including the `|W|` that seemed to need a dedicated
  `abs` op) runs on the N-D tape; the ReLU on an interval `[l,u]` becomes
  `[relu(l),relu(u)]`. The **robust logits** place the true class at its
  **lower** bound and the others at their **upper** bound (`zₜ=cₜ−rₜ`,
  `z_j=c_j+r_j`): a low cross-entropy over them means the true class wins
  *even in the worst case* — the point is **certified**. Honest oracle: the
  tape IBP propagation **coincides** with the reference `IbpMlp` verifier
  (plain `f32`) and is **sound** (2000 sampled points ∈ certified box);
  after certified training, the **certified ℓ∞ radius grows** markedly
  (robust-trained vs accuracy-only network, both classifying correctly at
  100 %) + bit-exact determinism. Extends IBP (#1) / CROWN (#2) / zonotopes
  (#29) toward training.
- **Sophia — clipped second-order optimizer**
  (`nn::nd_optim::NdSophia`, Liu et al. 2023, roadmap #44): Sophia scales
  each coordinate's momentum by an estimate of the **diagonal Hessian** and
  **clips** the result: `θ ← θ − lr·clip(m/max(γ·h,eps),ρ)`. Flat
  directions (small curvature `h`) take a bounded sign-like step; curved
  directions take a **Newton-like** step `m/h` — hence robustness to bad
  conditioning. The diagonal Hessian is estimated by a **Hutchinson
  estimator** with a **finite-difference Hessian-vector product**: with a
  seeded sign vector `v∈{±1}`, `Hv ≈ (∇L(θ+εv) − ∇L(θ))/ε` and
  `ĥ = v⊙Hv` (for a quadratic this is the **exact** diagonal Hessian; my
  old blocker "we need an `abs` op on the tape" was unfounded — the
  clipping happens in the optimizer in `f32`, not on the tape). Like SAM,
  this requires **two** gradient computations per step, so the caller
  orchestrates `probe` (perturbs `θ` by `εv`) then `step` (restores `θ`,
  applies the update) — a **library optimizer outside the single-gradient
  `lm --opt` loop**. Honest oracle: **converges on an ill-conditioned
  quadratic** (curvatures 4 vs 0.25, conditioning 16) where the
  per-coordinate Newton step neutralizes the conditioning + bit-exact
  determinism (seeded probe). Joins the optimizer family (Adam, Lion, Muon,
  Shampoo, SOAP, Adafactor, LAMB, Adan, Prodigy, SAM, …).
- **QuIP# — Hadamard incoherence + E8 lattice codebook**
  (`quantization::quantize_quip`/`nearest_e8`/`random_hadamard_transform`,
  Tseng et al. 2024, roadmap #64): two ideas. (1) **Incoherence processing**:
  multiplying the weights by a **randomized Hadamard transform** (seeded ±1
  signs then FWHT, *orthogonal*) spreads outliers across coordinates and
  **narrows the dynamic range**; at an **equal** bit budget, the `2^bits`
  fixed levels then resolve the bulk of the weights far better (scalar RTN
  had to spread its few levels over the whole range to cover outliers). (2)
  The **E8 lattice codebook**: quantize the rotated weights in blocks of 8
  to the nearest point of the **E8 lattice** (`D8 ∪ (D8+½·1)`, closed
  Conway-Sloane decoder) — the densest in dimension 8, with a **lower
  quadratic moment** than the cubic grid at **equal** density (~14 %
  packing gain). Honest oracle: the RHT is orthogonal (exact round-trip)
  and **narrows the range** of an outlier weight; the E8 decoder returns a
  **valid** lattice point (coordinates all integer or all half-integer,
  even sum) and quantizes **better than the cubic grid on average** (lattice
  gain measured over 4000 vectors); end-to-end, QuIP# reconstructs **better
  than scalar RTN** at a 2-bit budget on sparse-outlier weights + bit-exact
  determinism. (QuIP#'s global large Hadamard and curated E8P codebook are
  simplified here to a per-block-of-8 Hadamard and the bare E8 lattice.)
  Complements the quantization family (AQLM, GPTQ, AWQ, NF4, SqueezeLLM,
  SpQR, KVQuant, LLM.int8, OmniQuant, BitNet).
- **AQLM — multi-codebook additive quantization**
  (`quantization::quantize_aqlm`/`AqlmResult`, Egiazarian et al. 2024,
  roadmap #70): instead of quantizing each weight **scalarly**, AQLM splits
  the weights into **groups** of dimension `g` and approximates each group
  by the **sum** of one codeword from each of `M` learned codebooks (of `K`
  words each). The codebooks are initialized by **residual k-means** then
  refined by **alternating optimization**: re-encode each group (greedy
  residual assignment across the `M` codebooks) then refit each codebook by
  least squares given the other contributions (AQLM's beam search is
  simplified here to greedy assignment — documented). Because codewords are
  **vectors**, additive quantization captures the **cross-dimension
  structure** that scalar round-to-nearest ignores, hence much better
  reconstruction at low budget. Honest oracle: error **< 0.7× scalar RTN**
  at an **equal** ~2-bit budget (`M·log₂K/g`) on structured weights
  (groups built on a few prototype directions) + exact round-trip
  (non-divisible length, zero padding) + bit-exact determinism. Joins the
  quantization family (GPTQ, AWQ, NF4, SqueezeLLM, SpQR, KVQuant, LLM.int8,
  OmniQuant, BitNet).
- **S5 — MIMO SSM + parallel associative scan**
  (`nn::nd_layers::s5_scan`/`s5_parallel_scan`/`NdS5`, Smith et al. 2023,
  roadmap #52): unlike S4D's **per-channel SISO** SSMs (each channel its own
  independent state), S5 drives a **single shared state** of dimension `n`
  with **all** inputs through a matrix `B`, and reads `m` outputs through
  `C` (hence *MIMO*): `hₜ=Ā⊙hₜ₋₁+xₜB`, `yₜ=hₜC`. Being linear, the
  recurrence can be computed by an **associative scan**: the element
  `(aₜ,uₜ)` represents the affine map `h↦aₜ⊙h+uₜ`, and these maps compose
  via the **associative** operator `(a₁,u₁)∘(a₂,u₂)=(a₂⊙a₁, a₂⊙u₁+u₂)`. An
  inclusive **Hillis-Steele** scan (fixed `log₂ seq` doubling order ⇒
  **deterministic**) produces all prefix states in parallel. Honest oracle:
  the **parallel scan ≡ the sequential recurrence** — tested with
  **time-varying** `aₜ` (a real associative scan, not the trivial constant
  case), which proves the associativity that licenses parallelization;
  `s5_scan` on the tape ≡ hand-written MIMO reference (validates the
  `B`/`C` wiring); **gradient check** (x, Ā, B, C); `NdS5` trains (MSE↓) +
  bit-exact determinism. Complements the state-space family (Mamba,
  Mamba-2/SSD, S4).
- **Mamba-2 / SSD — state-space ↔ attention duality**
  (`nn::nd_layers::ssd_dual`/`NdMamba2`, Dao & Gu 2024, roadmap #50):
  Mamba-2 restricts the SSM state matrix to a **scalar decay** `aₜ` per step
  (instead of Mamba's per-channel diagonal `A`). This restriction makes the
  linear recurrence `Hₜ=aₜHₜ₋₁+xₜBₜᵀ` (state `d×n`), `yₜ=HₜCₜ` **exactly
  equal** to a single masked attention-like quadratic form — the
  **duality**: `Y=(L⊙CBᵀ)X` with `L[i,j]=∏_{j<k≤i}aₖ` for `i≥j`. Computed
  on the tape: the cumulative log-decay `cumlogᵢ=Σ_{k≤i}a_logₖ` is a
  **prefix-sum** (matmul with a triangular matrix of ones),
  `L=exp(cumlogᵢ−cumlogⱼ)` causally masked, `Y=(L⊙CBᵀ)X`. `a_log=log a` is
  the parameter (in Mamba-2 `a_logₜ=Δₜ·A`), so **no `log` op** is needed;
  the mask is applied **before** the `exp` (`diff⊙mask`, then `exp`, then
  `⊙mask`) to keep the exponent bounded in the upper triangle (avoid
  `inf·0=NaN`) and zero it there exactly. Honest oracle: the **dual form ≡
  the hand-written sequential recurrence** (literally the paper's duality);
  **gradient check** (x, B, C, a_log); `NdMamba2` trains (MSE↓) +
  bit-exact determinism. Joins Mamba/S4/RWKV/RetNet/GLA/HGRN/DeltaNet/
  xLSTM/Hyena.
- **FNO — Fourier neural operator** (`nn::fno::FnoSpectralConv1d`/`NdFno`,
  Li et al. 2021, roadmap #75): a neural operator learns a mapping between
  **functions** (e.g. initial condition ↦ PDE solution), not between
  fixed-size vectors. FNO implements the **global** kernel integral in the
  **Fourier domain**: transform the sampled signal, keep the lower-frequency
  `modes`, multiply each mode by a **learned complex weight**
  `R_k=Ar_k+iAi_k` (a `width×width` matrix, channel mixing), then transform
  back. The real DFT and its inverse are **fixed cosine/sine matrices**:
  the whole transform is an ordinary matmul (deterministic) that the N-D
  tape differentiates directly — **no FFT, no complex type, no new op**;
  per-mode weights are applied by a **batched** matmul (`bmm`) over the
  modes. Full FNO block: `σ(SpectralConv(v)+W·v)`. Honest oracle: **exact**
  reconstruction of a band-limited signal at the kept modes (DFT⁻¹∘DFT,
  validates the matrices + the one-sided factor-2 inverse); **gradient
  check** by finite differences (signal, Ar, Ai); since differentiation is
  diagonal in Fourier (`d/dx↔×ik`), a single spectral conv **learns the
  differentiation operator** `sin(ωx+φ)↦ω cos(ωx+φ)` and **generalizes to
  an unseen phase** (test MSE <0.02, convex fit); bit-exact determinism.
  Joins the scientific-computing family (Neural ODE, PINN, DeepONet, KAN).
- **Hyena — implicit long convolutions + gating**
  (`nn::nd_layers::hyena_long_conv`/`NdHyena`, Poli et al. 2023, roadmap
  #56): an **attention-free** token mixer. The long range comes from a
  **causal convolution** whose filter is not stored tap by tap but
  **generated** by a small MLP from a fixed positional encoding, then
  windowed by a learnable exponential decay `exp(−γ·t̄)` per channel —
  that is what enables **long filters with few parameters** (the heart of
  Hyena). The attention-equivalent (data dependence) is provided by
  **multiplicative gating**: `z=x1⊙(h1*v)` then `z=x2⊙(h2*z)` (order 2).
  The per-channel causal convolution `y[t,c]=Σ_τ h[τ,c]·u[t−τ,c]` is
  expressed on the tape as `Σ_τ h[τ,:]⊙(Sτ·u)` with **constant shift
  matrices** `Sτ` (distributing the matmul over the learnable taps ⇒
  differentiable in `u` and `h` with no scatter op). Honest oracle: conv ≡
  hand-written causal reference; **gradient check** by finite differences
  (`u`, `h`); `NdHyena` training (MSE↓) + bit-exact determinism. Joins the
  sequence-model family.
- **xLSTM — scalar sLSTM + matrix mLSTM**
  (`nn::nd_layers::slstm_scan`/`mlstm_scan`/`NdXlstm`, Beck et al. 2024,
  roadmap #57): the extended LSTM replaces the sigmoid input gate with an
  **exponential gate** `iₜ=exp(ĩₜ)` accompanied by a **normalizer state**
  `nₜ=fₜnₜ₋₁+iₜ`, the output being `hₜ=oₜ⊙(cₜ/nₜ)`. Since `cₜ/nₜ` is a
  positive weighted average of `zₜ=tanh∈(−1,1)`, the output stays bounded
  in (−1,1): the recurrence is **stable without the log stabilizer**
  (omitted, a pure numerical device that cancels in the ratio). `tanh` is
  built from the single `sigmoid` op via the exact identity
  `tanh(x)=2σ(2x)−1`. The **mLSTM** variant carries a `d×d` covariance
  memory updated by outer products `vₜᵀkₜ`, read by query, with the
  stabilizing denominator `max(|nₜ·qₜ|,1)` reconstructed **exactly** via
  `|a|=relu(a)+relu(−a)` and `max(a,1)=relu(a−1)+1` (no new op, faithful
  guard). Honest oracle: mLSTM ≡ hand-written reference recurrence (active
  denominator); **gradient check** by finite differences (sLSTM: 4 gates;
  mLSTM: q,k,v,iₜ,fₜ, smooth regime); `NdXlstm` training (MSE↓) +
  bit-exact determinism. Joins the sequence-model family (Mamba, S4, RWKV,
  RetNet, GLA, HGRN, DeltaNet).
- **OmniQuant — learnable weight clipping**
  (`quantization::omniquant_quantize`, Shao et al. 2024, roadmap #65):
  round-to-nearest quantizes each channel over its **full** range
  `[−max|w|, max|w|]` — with heavy-tailed weights, most code levels are
  wasted on rare outliers. OmniQuant learns a **clipping factor** `γ∈(0,1]`
  per channel that **narrows** the range to `γ·max|w|`, trading a little
  clipping error on outliers for far finer steps on the bulk of the weights
  — found here by a deterministic grid search that **includes `γ=1`** (pure
  RTN). Honest oracle: reconstruction error **< RTN** on heavy-tailed
  weights (≥1 channel actually clips) + **never worse** than RTN (γ=1 is a
  candidate) + bit-exact determinism. Joins the quantization family (GPTQ,
  AWQ, NF4, SqueezeLLM, SpQR, KVQuant, LLM.int8).
- **S4 (S4D) — diagonal structured state space**
  (`nn::nd_layers::s4_scan`/`NdS4`, Gu et al. 2022, roadmap #51): **linear
  time-invariant** SSM (unlike Mamba's `selective_scan` whose matrices
  depend on the input) — `A` diagonal, `B`/`C`/`Δ` are **fixed
  parameters**; discretization `Ā=exp(Δ⊙A)`, `B̄=Δ⊙B`, recurrence
  `h_t=Ā⊙h_{t−1}+B̄⊙x_t` (state `(d,n)`) unrolled on the tape, readout
  `y_t=Σ_n C⊙h_t`. Diagonal **HiPPO** init (S4D-Lin) `A[:,j]=−(j+1)`,
  `A<0` contractive. The `NdS4` layer adds input/output projections + gated
  skip `D⊙x`. Oracle: **gradient check** (finite differences vs analytic on
  x, a_log, B, C, log_dt) + training (MSE↓ toward a target) + bit-exact
  determinism. Library layer.
- **AI² / zonotopes — abstract domain for verification**
  (`nn::ibp::Zonotope`/`IbpMlp::certify_zonotope`, Gehr et al. 2018, roadmap
  #29): propagation by **zonotopes** (center + generators,
  `{c+Σεᵢgᵢ : εᵢ∈[−1,1]}`) — affine layers are **exact**, the ReLU is
  relaxed DeepZ-style (`y=λx+μ±μ`, `λ=u/(u−l)`, `μ=−λl/2`, one fresh
  generator per unstable neuron). The shared `εᵢ` capture the **linear
  correlations** that intervals lose. Honest oracle: exact affine (=
  interval forward) + **soundness** (thousands of sampled points in the
  input box fall in the zonotope box of a 3-layer ReLU MLP) + **tighter
  than IBP under correlation** (network `relu(x)−relu(x)` ≡ 0: zonotope
  `[−0.5;0.5]` vs IBP `[−1;1]`, both sound). Extends `nn::ibp` (IBP #1,
  CROWN #2); displayed in the `certify` CLI next to IBP and CROWN.
- **EAGLE — feature-level speculative decoding**
  (`nn::nd_decoder::EagleHead`/`generate_eagle`, Li et al. 2024, roadmap
  #62): where Medusa predicts future *tokens*, EAGLE drafts at the
  **feature** level — a light head maps
  `(feature_t, embed(token_{t+1})) → feature_{t+1}`, and the **frozen** LM
  head turns the predicted feature into a token; chained, it gives an
  **autoregressive** draft verified in one pass (accepted prefix + greedy
  correction). `NdDecoderLM` exposes `token_embedding`/`head_logits`/
  `d_model`; `EagleHead::train` fits the head by MSE on the frozen model's
  features. Honest oracle: output **exactly = greedy** for an **arbitrary**
  head (verification) + determinism + **trained** head ⇒ ≥1 block accepts
  >1 token (forwards < 2·n) while staying exact. Library layer.
- **Medusa — multi-head decoding** (`nn::nd_decoder::MedusaHeads`/
  `generate_medusa`, Cai et al. 2024, roadmap #61): speeds up decoding by
  attaching **additional heads** to the base model (head `j` predicts the
  token at `+j+2` from the hidden state), producing a **multi-token draft
  from a single forward**; a verification pass accepts the longest prefix
  matching the model's argmax then commits a correction/bonus token.
  `NdDecoderLM` now exposes `forward_hidden`/`forward_with_hidden`
  (post-LayerNorm hidden state); `MedusaHeads::train` trains the heads on
  the **frozen** model's hidden states. Honest oracle: output **exactly =
  greedy** for **arbitrary** heads (even random — verification guarantees
  correctness) + determinism + **trained** heads ⇒ at least one block
  accepts >1 token (forwards < 2·n) while staying exact. Library layer.
- **PagedAttention — paged KV-cache** (`nn::paged_attention::PagedKvCache`,
  Kwon et al. / vLLM 2023, roadmap #63): the decoding key/value cache is
  split into **blocks** of fixed size drawn from a shared pool, addressed
  indirectly by a **block table** (like memory paging) ⇒ near-zero
  fragmentation. `append` fills blocks on demand, `gather_keys/values`
  rebuilds the contiguous cache, and `attention` does the softmax
  dot-product by indexing keys/values **through the table**. Honest oracle:
  with **decoy** blocks interleaved (non-sequential physical layout), the
  gather is **bit-identical** to the inserted vectors and the paged
  attention is **bit-identical** to attention over a contiguous cache (same
  arithmetic order) — paging is proven at zero numerical cost; + block
  accounting (`⌈len/block⌉`) and the empty case + determinism. Library
  layer (new module).
- **DoRA — weight-decomposed low-rank adaptation**
  (`nn::dora::DoraLinear`, Liu et al. 2024, roadmap #73): PEFT that
  decomposes a frozen weight `W₀` into **magnitude** (per-column vector
  `m`) × **direction** (normalized), the direction driven by a LoRA low-rank
  update `BA`: `W' = m ⊙ (W₀+BA)/‖W₀+BA‖_col`. Only `m`, `A`, `B` train.
  Backward of the column normalization in **closed form** (`u=V/‖V‖`,
  `∂L/∂V=(m/‖V‖)(gw−u·s)`, `∂L/∂m=s`). Honest oracle: init `B=0, m=‖W₀‖_col`
  ⇒ effective weight **= W₀ exactly** (adaptation starts from the
  pretrained function) + **gradient check** (central finite differences vs
  analytic, generic params) + recovers a DoRA-generated target (loss ÷100
  by gradient descent) + bit-exact determinism. Library layer (new module).
- **GaLore — low-rank gradient projection**
  (`nn::nd_optim::NdGalore`/`galore_subspace`, Zhao et al. 2024, roadmap
  #48): **memory-reduced** optimizer — for a matrix parameter, the gradient
  `G` is projected onto its own dominant rank-`r` subspace `P` (top-`r` left
  singular vectors via `jacobi_eigenvectors`, refreshed every `update_gap`
  steps), Adam runs on the small projected gradient `PᵀG` then the update is
  lifted back by `P`. The states go from `m×n` to `rank×max(m,n)`; vectors
  fall back to Adam. Honest oracle: `P` **orthonormal** (`PᵀP=I`) and
  **optimal orthogonal projection** (Pythagorean identity
  `‖G−PPᵀG‖²=‖G‖²−‖PᵀG‖²`, error decreasing in `r`, zero at full rank) +
  gradient **low-rank reconstructed exactly** (sub-rank ⇒ residual) +
  **convergence on a low-rank target** with compressed state `2×4` (≠
  `4×4`) + sub-rank does not reach it + bit-exact determinism. Joins the
  optimizer family; CLI `lm --opt galore`.
- **YaRN — RoPE context extension** (`nn::yarn`, Peng et al. 2023, roadmap
  #60): extends the usable context of a RoPE model by a factor `s` via
  **NTK-by-parts** interpolation — `yarn_frequencies` keeps the
  **high-frequency** dimensions intact (`r_p>β` ⇒ local order preserved),
  fully interpolates the **low frequencies** (`r_p<α` ⇒ `θ_p→θ_p/s`), with a
  linear ramp in between (`θ'_p=θ_p·((1−γ)/s+γ)`). `rope_apply_freqs`/
  `rope_yarn` apply the rotation (nested convention identical to the
  existing RoPE of `autodiff::nd`); `yarn_attention_scale` gives the
  temperature `0.1·ln(s)+1`. Honest oracle: **relative-position property**
  `⟨rope(q,m),rope(k,n)⟩=g(m−n)` preserved despite the modified frequencies
  + the angle of a low-frequency dimension at the **extended** length `s·L`
  returns **exactly** to its training value at `L` (where plain RoPE blows
  up) + NTK-by-parts bounds (high frequency unchanged, low = `θ/s`,
  monotone ramp) + `scale=1` ≡ plain RoPE + determinism. Library layer
  (positional primitive, no CLI).
- **Learn then Test (LtT)** (`nn::conformal::learn_then_test`/
  `hoeffding_pvalue`, Angelopoulos et al. 2021, roadmap #37): **distribution-
  free** control of **multiple arbitrary (non-nested) risks** by hypothesis
  testing. Each configuration `λ` of a grid becomes a **Hoeffding p-value**
  for `H₀: R(λ) > α` (`p = exp(−2n(α−R̂)₊²)`, super-uniform under the null),
  then a **Bonferroni family-wise** correction at level `δ`: keep the `λ`
  with `p ≤ δ/m`. Guarantees that, with probability `≥ 1−δ`, **every** kept
  configuration satisfies `R(λ) ≤ α` (FWER `≤ δ`) — **without** a
  monotonicity assumption (unlike RCPS #36). Honest oracle: FWER verified
  **by simulation** (all configs on the boundary `R=α` ⇒ measured FWER
  `≤ δ`, vs naive selection that fails ~always) + power (safe configs kept,
  unsafe ones rejected) + determinism. Library layer.
- **RDP accountant (Rényi DP)** (`dp::gaussian_rdp`/`rdp_to_dp`/
  `rdp_gaussian_epsilon`, Mironov 2017, roadmap #78): privacy-budget
  accounting by **Rényi-DP**, tighter and more principled than naive
  `(ε,δ)` composition. RDP of the Gaussian mechanism `RDP(α)=α/(2σ²)`
  (additive under composition), Mironov conversion
  `ε=RDP(α)+ln(1/δ)/(α−1)` (the `α−1` is what makes it tight), optimized
  over a grid of orders α. Strengthens the existing DP-SGD (#19). Oracle:
  exact RDP and conversion (closed forms) + `ε` **far below** the basic
  linear composition (which pays a ~√steps penalty) + monotonicity (more
  steps ⇒ larger ε; more noise ⇒ smaller ε). Library layer.
- **Watermark for LLMs** (`nn::watermark`, Kirchenbauer et al. 2023, roadmap
  #79): a statistical watermark making generated text **auditable without
  model access**. The previous token seeds a partition of the vocabulary
  into a **green** list (fraction γ) / red; `apply_green_bias` adds `δ` to
  the green logits to steer generation. The detector, knowing only the seed
  and γ, recounts the green tokens: watermarked text contains far more than
  the γ fraction expected by chance, which a **z-test**
  `(g−γn)/√(nγ(1−γ))` (`detect_z`) flags with a minuscule p-value, while
  natural text scores `z≈0`. Everything is a deterministic hash of
  `(seed, prev, token)`. Oracle: green fraction ≈ γ + bias applied to green
  tokens only + watermarked text detected (z≫8) vs natural (z≈0) + a
  **wrong seed does not detect** (no false provenance) + determinism.
  Library layer.
- **DeepONet — operator learning** (`nn::deeponet::DeepONet`, Lu et al.
  2021, roadmap #76): learns an **operator** `G : u ↦ G(u)` (function →
  function) via a **branch × trunk** factorization
  `G(u)(y) ≈ Σ_k b_k(u)·t_k(y)` — the branch encodes the input function `u`
  (sampled at fixed sensors), the trunk encodes the position `y`. Variant
  **POD-DeepONet** (fixed **cosine** trunk `cos(kπy)` + **linear** branch) ⇒
  **convex** fitting, exact for linear operators like the **antiderivative**
  `∫₀^y u`. Oracle: trained on some functions, it approximates the
  antiderivative on **unseen** functions at test MSE < 0.01 (≪ constant
  predictor) — the operator-learning property — + determinism. Library
  layer.
- **Deep Ensembles** (`nn::ensemble::DeepEnsemble`, Lakshminarayanan,
  Pritzel & Blundell 2017, roadmap #40): predictive uncertainty by
  **seeded ensemble**. N small ReLU MLPs (`1→hidden→1`) trained on the N-D
  tape with `NdAdam`, each seeded differently; `predict(x)` returns
  `(mean, std)` — the point estimate and its **epistemic uncertainty**
  (disagreement between members). Oracle: the ensemble-mean MSE is ≤ the
  members' mean MSE (Jensen) + the std is **far larger out-of-distribution**
  (far from the training range) than in-distribution + bit-exact
  determinism. Library layer.
- **LLM.int8()** (`quantization::int8_mixed_matmul`, Dettmers et al. 2022,
  roadmap #71): mixed int8/fp32 matmul. Transformer activations have a few
  **outlier feature columns** of very large magnitude; quantizing them in
  int8 with the rest inflates the scale and crushes the resolution of
  normal features. LLM.int8() keeps these columns (and the corresponding
  rows of W) in **full precision** and quantizes the rest in **int8**:
  `X·W = X_normal·W_normal (int8) + X_outlier·W_outlier (fp32)`. A column
  is an outlier if any `|X[i,j]|` exceeds the threshold (default 6.0).
  Oracle: on outlier-column activations, the error vs fp is **< 0.5×** that
  of plain int8; without outliers, reduces to pure int8; determinism.
  Library layer.
- **RCPS — Risk-Controlling Prediction Sets**
  (`nn::conformal::hoeffding_ucb` + `rcps_select`, Bates et al. 2021,
  roadmap #36): where conformal controls *coverage*, RCPS controls an
  **arbitrary bounded risk** (loss in [0,1]: false-negative rate,
  non-coverage, …) with a **high-probability (PAC)** guarantee. For a
  family of predictors `C_λ` with risk non-increasing in λ, RCPS chooses the
  smallest `λ̂` whose **Hoeffding concentration bound** on the risk is ≤ α
  (for λ̂ and every larger λ) ⇒ `R(λ̂) ≤ α` with probability ≥ 1−δ. Oracle:
  the bound exceeds the mean by the right gap + exact selection (computed
  case) + on fresh data the empirical risk stays ≤ α (conservative bound).
  Library layer.
- **Prodigy** (`nn::nd_optim::NdProdigy` + `ProdigyConfig`, Mishchenko &
  Defazio 2023, roadmap #46): a **learning-rate-free** Adam
  ("parameter-free"). It estimates online the distance `d ≈ ‖x₀ − x*‖` to
  the solution — via the accumulated global correlation `⟨g, x₀ − x⟩` —
  and uses it as the effective rate, starting from a tiny `d₀ = 1e-6` that
  grows to the problem's scale. `d`, the numerator `r` and the denominator
  norm are **global** scalars over all parameters. Oracle: `d` adapts to
  the distance scale (no lr tuning) + the quadratic loss drops sharply +
  bit-exact determinism. CLI: `scirust lm --opt prodigy` (8 languages).
- **KVQuant** (`quantization::kvquant_kv`, Hooper et al. 2024, roadmap #68):
  KV-cache quantization at the granularity that matches its outlier
  structure — **keys per-channel** (key outliers concentrate by feature
  column) and **values per-token** (per row). Far more faithful than a
  single per-tensor scale, which a handful of large key channels would
  dominate (crushing the resolution of all the others). Oracle: on keys
  with channel outliers, the attention-output error vs fp is **< 0.6×**
  that of per-tensor quantization; per-channel fixes the small columns
  (<0.1× error) where per-tensor fails; determinism. Library layer.
- **ALiBi — Attention with Linear Biases**
  (`nn::nd_layers::alibi_slopes` + `alibi_bias` +
  `NdMultiHeadAttention::with_alibi`, Press, Smith & Lewis 2022, roadmap
  #59): replaces learned/rotary positions with a **static distance-linear
  bias** added to the attention scores — for query `i` and key `j ≤ i`,
  `−slopeₕ·(i−j)`, with per-head slopes in geometric progression
  `2^(−8h/H)`. No learned position ⇒ **length extrapolation**. Wired into
  `NdMultiHeadAttention` (builder `with_alibi`, includes the causal mask).
  Oracle: geometric slopes (ratio `2^(−8/H)`) + linear/causal/Toeplitz bias
  + softmax weights decaying with distance (exactly `∝ exp(−slope·dist)`) +
  deterministic attention forward.
- **ACI — Adaptive Conformal Inference**
  (`nn::conformal::AdaptiveConformal`, Gibbs & Candès 2021, roadmap #38):
  **online** conformal robust to **distribution drift**. Static conformal
  silently loses coverage under distribution shift; ACI tracks an effective
  level `αₜ` and corrects it after each observation by feedback
  `αₜ₊₁ = αₜ + γ(α − errₜ)`, driving the long-term error rate toward `α`
  (coverage toward `1−α`) for **any** score stream. With a sliding window of
  recent scores, coverage stays ≈ 1−α through changes where static
  conformal collapses. Oracle: exact `αₜ` update rule (computed case) +
  coverage ≈ 1−α maintained under variance change (vs static conformal that
  drops) + determinism. Library layer. Complements CQR/APS/RAPS in the
  conformal pillar.
- **KAN — Kolmogorov-Arnold Networks** (`nn::kan::KanLayer`, Liu et al.
  2024; FastKAN RBF basis, Li 2024; roadmap #77): **learnable activations
  on the edges** rather than on nodes — `y_j = Σᵢ φᵢⱼ(xᵢ)` with each `φ` a
  sum of Gaussian RBFs (fixed grid) + a `SiLU` base term. The output is
  **linear in the coefficients**, so the fit is a **convex** least-squares
  problem solved by deterministic gradient descent. Oracle: a single KAN
  layer fits the non-linear additive target `sin(2x₀)+x₁²` at MSE<0.02 —
  well below the best linear model (which cannot represent sin/square);
  localized RBF basis; bit-exact determinism. Library layer (RBF/FastKAN
  variant, not the original paper's B-splines).
- **RWKV time-mixing (WKV)** (`nn::nd_layers::rwkv_wkv` + `NdRwkv`, Peng et
  al. 2023, roadmap #53): **WKV** operator — recurrent linear attention
  with **per-channel exponential time decay** `decay ∈ (0,1)` plus a
  **bonus** for the current token, normalized (numerator/denominator),
  unrolled in linear time on the tape. Required a new autograd **`div`** op
  (elementwise division, gradient `∂a=g/b`, `∂b=−g·a/b²`, gradient-checked).
  The `NdRwkv` layer adds a **receptance** `r=σ(W_r·x)` gating the output,
  with learnable per-channel decay/bonus. Oracle: the tape recurrence **≡
  the explicit weighted-sum formula** + gradient check (k, v, decay, bonus)
  + training (MSE↓) + bit-exact determinism. CLI: `scirust rwkv` (8
  languages).
- **GloRo — Lipschitz-certified robustness** (`nn::lipschitz`, Leino, Wang &
  Fredrikson 2021, roadmap #32): `spectral_norm` (spectral norm by
  deterministic power iteration), `spectral_normalize` (constrained
  **1-Lipschitz** layer) and `GloroClassifier` (linear classifier with a
  **proved L2 robustness radius** `margin/(√2·‖W‖₂)`, no search or
  sampling; the `√2` comes from the `≤ √2·L` Lipschitz of the margin
  `f_A−f_B`). Oracle: known spectral norms (diagonal, rectangular); norm ≈ 1
  after normalization; radius **sound** (the worst perturbation at that
  radius does not flip the prediction) **and conservative** (≤ exact
  distance to the nearest boundary); determinism. Library layer. Complements
  the certifiable pillar: IBP, CROWN, smoothing, GloRo.
- **Randomized Smoothing — certified L2 robustness**
  (`nn::smoothing::SmoothedClassifier` + `clopper_pearson_lower` +
  `inv_normal_cdf`, Cohen, Rosenfeld & Kolter 2019, roadmap #27): turns any
  classifier into a **smoothed** one under Gaussian noise `N(0,σ²I)`, with a
  **proved L2 robustness radius** `σ·Φ⁻¹(pₐ)`. The top-class probability
  `pₐ` is lower-bounded by **Clopper-Pearson** (regularized incomplete beta
  `betai`/`lgamma`, exact); `Φ⁻¹` by Acklam's rational approximation.
  Oracle: for a **half-space** classifier the certified radius **equals the
  exact distance to the boundary** (independent of σ) + soundness/abstention
  at the edge + determinism + reference values of `Φ⁻¹`/`betai`/
  Clopper-Pearson. CLI: `scirust certify` now displays IBP/CROWN
  (deterministic) **and** smoothing (probabilistic).
- **SpQR — Sparse-Quantized Representation**
  (`quantization::SpqrOutliers`, Dettmers et al. 2023, roadmap #67): the
  quantization error is **heavy-tailed** — a small fraction of "outlier"
  weights concentrates most of the error. SpQR keeps this fraction (the
  largest dense-quantization errors) in **full precision** (sparse channel)
  and quantizes the rest densely, so ~1 % of outliers removes a large share
  of the error for a small memory overhead. Oracle: on Gaussian weights with
  injected outliers, keeping 1 % of the weights divides the squared error by
  > 3; exact outlier reconstruction; determinism. Library layer (the
  paper's two-level grouped scales are orthogonal).
- **SqueezeLLM** (`quantization::SqueezeLlmCodebook` +
  `weighted_quant_error`, Kim et al. 2023, roadmap #66): **non-uniform**
  weight quantization by **sensitivity-weighted k-means** (proxy of the
  Hessian diagonal) — a codebook of `2^bits` centroids placed where they
  reduce the *loss* most, not where the weights are dense. Deterministic
  init (quantiles) + weighted Lloyd iterations. Oracle: weighted
  quantization error **strictly < uniform round-to-nearest** (Gaussian
  weights, 3 bits, < 0.85×) + exact round-trip on codebook values +
  determinism. Library layer (the "sparse" outlier branch is not modeled).
- **APS / RAPS — adaptive prediction sets**
  (`nn::conformal::AdaptivePredictionSets`, Romano, Sesia & Candès 2020;
  Angelopoulos et al. 2021; roadmap #34/#35): conformal **classification**
  by cumulative score `s(x,c)` = mass of all classes at least as probable
  as `c`. Set `{c : s(x,c) ≤ q̂}` ⇒ marginal coverage without distribution
  ≥ 1−α with **adaptive size** (confident input → small set, ambiguous →
  large). **RAPS** adds `λ·max(0, rank−k_reg)` to the score
  (`calibrate_raps`) to prune unlikely classes and produce **smaller** sets
  at equal coverage. Oracle: exact cumulative score (hand-computed case) +
  coverage on fresh data + adaptivity (easy vs ambiguous) + RAPS < APS in
  mean size + determinism. Library layer (like `ConformalClassifier`).
- **CQR — Conformalized Quantile Regression**
  (`nn::conformal::ConformalQuantileRegressor`, Romano, Patterson & Candès
  2019, roadmap #33): conformalizes a **quantile** regressor to produce
  **adaptive** (heteroscedastic) intervals with guaranteed coverage. Signed
  score `Eᵢ = max(q_lo(xᵢ)−yᵢ, yᵢ−q_hi(xᵢ))`, finite correction `Q`
  (conformal quantile of the `Eᵢ`, reuses `conformal_quantile`), interval
  `[q_lo(x)−Q, q_hi(x)+Q]` — **variable width depending on x** where
  split-conformal is constant-width (`Q` can be negative and tighten an
  overly wide band). Oracle: exact score semantics (hand-computed case) +
  marginal coverage ≥ 1−α on fresh data + **adaptivity** (far wider
  intervals in the high-noise region) + determinism. CLI: `scirust
  conformal` now shows both split **and** CQR.
- **SAM — Sharpness-Aware Minimization** (`nn::nd_optim::NdSam` +
  `SamConfig`, Foret et al. 2021, roadmap #47): **two-phase** optimizer that
  minimizes the *worst-case* loss in a radius-ρ ball (bias toward flat
  minima). `ascent` perturbs the weights toward `θ + ρ·g/‖g‖` (**global**
  gradient norm); `descent` restores θ and takes an SGD step with the
  gradient **at the perturbed point**. Two gradients per step ⇒ outside the
  single-gradient `lm --opt` loop (library layer). Oracle: perturbation =
  `ρ·g/‖g‖` with `‖ε‖ = ρ` + convergence on a quadratic (band ∝ lr·ρ) +
  determinism.
- **Shampoo** (`nn::nd_optim::NdShampoo` + `ShampooConfig` +
  `inverse_pth_root`, Gupta/Koren/Singer 2018, roadmap #41): structured
  **Kronecker** preconditioner — for a weight matrix, maintains the two
  factors `L = E[GGᵀ]`, `R = E[GᵀG]` and steps by the preconditioned update
  `W ← W − lr·L^(−1/4) G R^(−1/4)`. The inverse matrix roots come from a
  Jacobi decomposition (`inverse_pth_root`, reuses
  `jacobi_eigenvectors`), cached and refreshed every `precond_freq` steps.
  Non-matrix parameters: diagonal Adagrad. Oracle: `A^(−1/2)²·A ≈ I` +
  convergence on a matrix quadratic + Adagrad fallback + determinism. CLI:
  `scirust lm --opt shampoo` (11th `--opt` value).
- **Adafactor** (`nn::nd_optim::NdAdafactor` + `AdafactorConfig`, Shazeer &
  Stern 2018, roadmap #42): optimizer with **factored second-order
  moments** — for a weight matrix, stores only the **row** and **column**
  sums of the gradient square (`rows + cols` numbers instead of
  `rows·cols`) and reconstructs the rank-1
  `V[i,j] = R[i]·C[j]/ΣR` (sub-linear memory). Update `G/√V` **RMS-clipped**;
  `β2ₜ = 1 − t^(−0.8)` schedule. Non-matrix parameters: full second moment
  (RMSProp). Oracle: **exact** rank-1 reconstruction when `G²` is rank-1 +
  convergence (band) + factored matrix path reduces `½‖W−T‖²` +
  determinism. CLI: `scirust lm --opt adafactor` (10th `--opt` value).
- **NF4** (`quantization::nf4_quantize`/`nf4_dequantize` + `NF4_LEVELS`,
  QLoRA, Dettmers et al. 2023, roadmap #74): 4-bit **NormalFloat** type —
  16 levels that are the **quantiles of a normal** (per-block absmax scale).
  Optimal for Gaussian weights. Oracle: reconstruction error **< uniform
  int4** on Gaussian weights (seeded Box-Muller) + exact round-trip on the
  levels + determinism. Library layer.
- **BitNet b1.58** (`quantization::ternary_quantize` + `ternary_matmul`,
  Ma et al. 2024, roadmap #69): **ternary** weight quantization to
  `{−1,0,+1}` (absmean scale, ~1.58 bit/weight, ~20× more compact);
  **multiplication-free matmul** (add / subtract / skip by sign). Oracle:
  `ternary_matmul` = the sum-of-signs form **bit-exact** and = the
  dequantized product up to floating-point reassociation. CLI: `scirust
  bitnet` (live: max error 1.4e-6 vs dequant, 986/4096 zero weights).
  Deterministic.
- **HGRN** (`nn::nd_layers::hgrn` + `NdHgrn`, Qin et al. 2023, roadmap
  #58): linear RNN with per-channel leaky integration
  (`hₜ = fₜ⊙h_{t-1} + (1−fₜ)⊙cₜ`), **lower-bounded** forget gate
  `f = lb + (1−lb)·σ(·)` (the bound `lb` fixes the minimum memory horizon).
  No matrix state; unrolled on the tape. Tests: reference match + gradient
  check (c,f) + training + determinism. CLI: `scirust hgrn` (live: MSE
  27.37 → 4.59).
- **GLA — Gated Linear Attention** (`nn::nd_layers::gated_linear_attention`
  + `NdGla`, Yang et al. 2024, roadmap #55): **gated** linear attention —
  **input-dependent** per-channel forget gate `αₜ=σ(·)`
  (`S_t = diag(αₜ)·S_{t-1} + kₜᵀvₜ`, `o_t = q_t·S_t`), unrolled on the
  tape. Tests: match of a Vec reference + gradient check (q,k,v,α) +
  training + determinism. CLI: `scirust gla` (live: MSE 27.16 → 0.0000).
- **RetNet** (`nn::nd_layers::retention` + `NdRetention`, Sun et al. 2023,
  roadmap #54): **retention** layer — recurrent linear attention with decay
  `γ` (`S_t = γ·S_{t-1} + kₜᵀvₜ`, `o_t = q_t·S_t`), unrolled on the tape.
  **Duality oracle**: the recurrent form **equals** the parallel form
  `(QKᵀ⊙D)V` (`D_{nm}=γ^{n-m}`), tested; + gradient check (q,k,v) +
  training + determinism. CLI: `scirust retnet` (live: MSE 24.63 → 0.0002).
- **LAMB** (`nn::nd_optim::NdLamb`, You et al. 2020, roadmap #43): Adam
  with **per-layer trust** — the Adam direction `r` rescaled by `‖θ‖/‖r‖`
  per tensor. CLI `lm --opt lamb`. Tests: convergence (band ∝ lr, because
  the step norm ≈ lr·‖θ‖) + determinism.
- **Adan** (`nn::nd_optim::NdAdan`, Xie et al. 2022, roadmap #49):
  **adaptive Nesterov** momentum — 3 EMAs (gradient `m`, differences `v`,
  squared look-ahead term `n`); `θ ← (θ − η⊙(m+(1−β2)v))/(1+lr·wd)`. CLI
  `lm --opt adan`. Tests: quadratic convergence + determinism.
- **LoRA** (`nn::nd_layers::LoraLinear`, Hu et al. 2022, roadmap #72):
  **low-rank** adaptation — base weight `W` **frozen** + update
  `ΔW = (α/r)·A·B`; only `A` (`in×r`) and `B` (`r×out`) train
  (`r·(in+out)` parameters instead of `in·out`). `B=0` at init ⇒ the layer
  **equals the base exactly**. N-D tape layer. Tests: init = base,
  **gradient check** on `A` and `B`, `parameters()` exposes only `A`,`B`.
- **Temperature scaling / calibration** (`nn::calibration`, Guo et al. 2017,
  roadmap #39): `temperature_scale` (golden-section search on the NLL) +
  `expected_calibration_error` + `nll`. Post-hoc recalibration of the
  probabilities **without changing accuracy** (argmax invariant to `T>0`).
  Deterministic. CLI: `scirust calibrate` (live: ECE 0.29 → 0.004, −98.5 %,
  T=2.70). Tests: ECE decreases + accuracy unchanged + determinism.
- **Lookahead** (`nn::nd_optim::NdLookahead`, Zhang et al. 2019, roadmap
  #45): **wrapper** slow/fast-weights optimizer around Adam — `k` fast steps
  then `φ ← φ + α(θ − φ) ; θ ← φ`. Deterministic. CLI: `scirust lm --opt
  lookahead`. Tests: quadratic convergence, bit-for-bit determinism. (1st of
  the Tier 8-14 candidate pool.)
- **PINN** (`nn::pinn`: `Pinn1D`, `solve_harmonic`, Raissi et al. 2019,
  roadmap #17): **physics-informed** network — the **physics is in the
  loss** via a PDE residual at collocation points + boundary conditions.
  Solves the boundary-value problem `u'' = −u`, `u(0)=0`, `u(π/2)=1`
  (exact solution `sin x`); the second derivative `u''` is taken by
  **finite differences in the input** (the `u(x±h)` evaluations pass
  through the *same* parameters in a single forward graph), so the gradient
  w.r.t. the parameters stays exact (reverse autodiff) and deterministic.
  Verified against the analytic solution (max error ≈ 0.004). CLI:
  `scirust pinn`.
- **Mamba** (`nn::nd_layers::selective_scan` + `NdMamba`, Gu & Dao 2023,
  roadmap #18): **selective scan** S6 — state-space with diagonal matrix
  `A` and **input-dependent** (selective) parameters `Δ, B, C`;
  zero-order-hold discretization `Ā = exp(Δ·A)`, `B̄x = Δ·B·x`;
  deterministic linear-time recurrence `h_t = Ā_t ⊙ h_{t-1} + B̄x_t`,
  `y_t = h_t·C_t`, unrolled on the N-D tape. New autograd op `NdVar::exp`
  (gradient-checked). S4D-real init (`A[:,j] = −(j+1)`), skip `D⊙x`. Tests:
  `selective_scan` matches a Vec reference, gradient check (x, Δ, A, B, C),
  layer trains (MSE↓) + determinism. CLI: `scirust mamba`.
- **DeltaNet** (`nn::nd_layers::delta_rule` + `NdDeltaNet`, Yang et al.
  2024, roadmap #25): **recurrent linear attention** layer with delta rule
  (`S_t = S_{t-1} + β_t(v_t − S_{t-1}k_t)k_tᵀ`, `o_t = S_t q_t`) — fast
  weight memory, linear time, causal. The recurrence is **unrolled on the
  N-D tape** (new autograd op `NdVar::cat0`: axis-0 concatenation + slicing
  backward, **gradient-checked**), so the gradients are exact and verified
  by finite differences (q, k, v, β). Tests: matches a Vec reference,
  gradient check, training (MSE↓) + bit-for-bit determinism. CLI: `scirust
  deltanet`.
- **SOAP** (`nn::nd_optim::NdSoap` + `jacobi_eigenvectors`, Vyas et al.
  2024, roadmap #24): an optimizer that runs **Adam in Shampoo's eigenbasis**.
  For each weight matrix: factors `L = E[GGᵀ]`, `R = E[GᵀG]` (moving
  average); rotate the gradient into their eigenbasis (`Ĝ = Q_Lᵀ G Q_R`),
  Adam in that basis, then rotate the update back. Eigenbasis by a
  deterministic **cyclic Jacobi eigensolver** (`jacobi_eigenvectors`),
  refreshed every `precond_freq` steps (moments rotated into the new
  basis). Adam fallback for non-matrix parameters. Deterministic. CLI:
  `scirust lm --opt soap`. Tests: Jacobi diagonalizes (orthogonality +
  reconstruction), convergence on a matrix quadratic, bit-for-bit
  determinism.
- **AWQ** (`quantization::awq_quantize` + `awq_act_scale` + `AwqResult`,
  Lin et al. 2023, roadmap #15): **activation-aware** int8 quantization by
  scale search. Per-input-channel importance `a_j = mean|x_:,j|`; factors
  `s_j = a_j^alpha` (normalized to unit geometric mean) applied to the
  weights before per-channel int8 quantization, equivalence preserved on the
  activation side; `alpha` chosen by **grid** over `[0,1]` (`alpha=0` =
  round-to-nearest) minimizing the calibration-weighted output error. CLI:
  `scirust awq [--seed N] [--samples S] [--grid G]`. Tests: protects salient
  channels → error < round-to-nearest (`alpha>0` chosen) + bit-for-bit
  determinism. **Completes the quantization item #15** (SmoothQuant + GPTQ +
  AWQ).
- **GPTQ** (`quantization::quantize_gptq` + `gptq_hessian`, Frantar et al.
  2022, roadmap #15): int8 weight quantization by **second-order error
  feedback**. Proxy Hessian `H = XᵀX` over calibration activations; inverse
  by Cholesky (in f64, deterministic); for each output channel, sequential
  quantization of the input weights with error propagation (OBQ/GPTQ,
  natural order) and Schur complement. Symmetric per-output-channel scale.
  CLI: `scirust gptq [--seed N] [--samples S] [--damp D]`. Tests:
  **calibration-weighted reconstruction error < round-to-nearest** (≈ −85 %
  on correlated data) + soundness (never worse) + bit-for-bit determinism.
  Completes the quantization item (#15) with SmoothQuant and per-channel
  int8.
- **CROWN** (`nn::ibp::crown_bounds`, Zhang et al. 2018, roadmap #2):
  output bounds of a 1-hidden-layer ReLU MLP by **linear relaxation** +
  back-substitution over an L∞ box. Per-neuron relaxation: exact for stable
  neurons, adaptive upper chord / lower slope for unstable ones.
  **Tighter than IBP** (proved by test). CLI: `scirust certify` now
  displays IBP **and** CROWN side by side (CROWN certifies robustness where
  IBP fails). Tests: soundness (box sampling) + CROWN width ≤ IBP width per
  output.
- **AdEMAMix** (`nn::nd_optim::NdAdEMAMix`, Pagliardini et al. 2024, roadmap
  #23): Adam with **two gradient EMAs** (fast β1 + slow β3 with long memory,
  mixed by α); deterministic. CLI: `scirust lm --opt ademamix`. Tests:
  quadratic convergence (band), bit-for-bit determinism.
### Cleaned
- Deletion of `scirust-core/src/nn/.legacy/` (**2363 lines** of dead code):
  directory not wired into the module tree (dotfile, zero references),
  superseded by the real `nn::conv2d`/`batch_norm`/`layer_norm`/
  `pool`/`loss`/`transformer` implementations. Consistent with the fundamental "code under src/ wired and
  tested, otherwise archived".

### Added — "grow scirust" campaign (continued)
- **Schedule-Free** (`nn::nd_optim::NdScheduleFree`, Defazio et al. 2024, roadmap
  #22): optimizer **without a learning-rate schedule** — base sequence `z`
  (descent), Polyak average `x` (**evaluation point**), gradient taken at
  `y = (1−β)z + βx`. Deterministic. CLI: `scirust lm --opt schedule-free`
  (the eval point `x` is loaded before prediction). Tests: convergence on
  quadratic, bit-for-bit determinism.
- **Conformal prediction** (`nn::conformal`, Angelopoulos & Bates 2021, roadmap
  #21): `conformal_quantile`, `ConformalRegressor`, `ConformalClassifier` —
  prediction sets/intervals with **guaranteed coverage without any distribution
  assumption** (`≥ 1 − α`). Tests: empirical coverage reaches the target
  on fresh data (regression *and* classification). CLI: `scirust
  conformal [--seed N] [--alpha A]` (coverage measured live, e.g. 90.8%
  for a target of 90%). CLI: 41 → 42 commands.
- **Research batch 3 → functions** (tested, 8 green gates; **14 of the 20** items
  of [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)):
  - **Muon** (`nn::nd_optim`, Jordan et al. 2024): matrix optimizer —
    momentum then **Newton–Schulz orthogonalization** (quintic, no SVD) of
    the update of the 2-D matrices; `newton_schulz_orthogonalize` exposed.
    Deterministic. Tests: orthogonality (deviation ‖A·Aᵀ−I‖ collapses), matrix
    loss, determinism.
  - **Wanda** (`pruning::prune_wanda`, Sun et al. 2023): one-shot pruning by
    `|W|·‖X‖` (weights × activation norm), per output row — differs from
    magnitude pruning on channels with aberrant activations.
  - **SmoothQuant** (`quantization::smoothquant_scales`/`apply_smoothquant`,
    Xiao et al. 2022): input-channel smoothing that migrates the aberrant
    activation values into the weights; **preserves `X·W`**.
- **Research batch 2 → functions** (3 more features, tested, 8 green gates;
  **11 of the 20** items of [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)):
  - **RoPE** (`autodiff::nd`, Su et al. 2021): `rope` op (pairwise rotation,
    backward = inverse rotation); gradient-checked, norm conservation and
    **relative position property** tested; wired via
    `NdMultiHeadAttention::with_rope`.
  - **GQA / MQA** (`nn::nd_layers`, Ainslie et al. 2023):
    `NdMultiHeadAttention::new_gqa(num_kv_heads, …)` — shared K/V heads via the
    broadcast `bmm` (no new op); gradient-checked (GQA and MQA).
  - **Neural ODE** (`nn::neural_ode`, Chen et al. 2018): `rk4_integrate` +
    `NeuralOde` — backprop **through** the RK4 solver on the N-D tape (solver
    + autograd fusion). RK4 validated (`dy/dt=y → e`), gradient check through
    the solver, and the dynamics **learns** (Adam).
- **Research roadmap → functions** ([`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)):
  20 real papers translated into concrete functions, with status and effort. First
  batch **delivered this session** (tested, 8 green gates):
  - **IBP — certified output bounds** (`nn::ibp`, Gowal et al. 2018):
    interval propagation in a ReLU MLP → **proven** output box;
    `certified_robust` turns the bound into a class guarantee. Soundness
    tested by sampling (4000 points ∈ certified box). *The* "certifiable
    AI" pillar made concrete.
  - **Reproducible reductions** (`reproducible`, Demmel & Nguyen):
    `reproducible_sum`/`_mean`/`_dot` **bit-identical whatever the order /
    the number of threads** (canonical sort + exact Shewchuk expansion);
    survives catastrophic cancellation.
  - **LLaMA N-D layers** (`nn::nd_layers`): `NdRmsNorm`, `NdSwiGLU` (+
    gradient-checked `rmsnorm`/`sigmoid` ops) and `NdLlamaBlock` (Pre-RMSNorm +
    causal attention + SwiGLU) — trainable, Adam-ready.
  - **Exact speculative decoding** (`nn::nd_decoder`, Leviathan/Chen 2023):
    `generate_speculative` produces **exactly** the target's greedy output
    for any draft, with fewer forwards; + `generate_greedy`.
  - **Optimizers** (`nn::nd_optim`): **AdamW** (decoupled weight decay) and
    **Lion** (sign-momentum, deterministic).
- **`lm` CLI command**: trains a small causal decoder LM (N-D tape + Adam)
  on a token sequence and reports the loss curve + exact recall —
  `scirust lm ["t0,t1,.."] [--seed N] [--steps S] [--lr R]`. Deterministic per
  seed; exposes the whole N-D stack (embeddings, causal attention, gather,
  cross-entropy, Adam) in one command. CLI: 39 → 40 commands.
- **Reusable, deterministic N-D Adam optimizer** (`nn::nd_optim`):
  `NdAdam` (Kingma & Ba) over an ordered set of parameters. Each layer
  exposes `parameters() -> Vec<NdParam>` (`&mut` view of the values + index of the
  gradient from `backward`); the composition climbs back up the tree
  (`NdLinear`/`NdEmbedding`/`NdLayerNorm` → attention → block → `NdDecoderLM`),
  so **a single `opt.step()` updates the whole model**. f32 arithmetic in
  fixed order ⇒ **bit-for-bit deterministic**. Tests: convergence on quadratic
  (oracle), bit-for-bit determinism, and **the decoder LM trained by Adam via
  `parameters()`** (< 10% loss in 150 steps vs 300 with SGD, exact
  predictions).
- **End-to-end causal decoder language model** (`nn::nd_decoder`):
  GPT-style `NdDecoderLM` entirely on the N-D tape — token embedding
  + learned positional embedding → N **causal** Pre-LN transformer blocks →
  final LayerNorm → linear head to the vocabulary logits, trained by
  next-token cross-entropy. Flagship test: **the LM overfits a sequence
  and re-predicts it exactly** at every position (end-to-end proof that
  the whole stack learns); forward deterministic per seed. `NdEmbedding` (table
  backed by `gather`) added as a reusable layer.
- **N-D `gather` + `cross_entropy` ops** (`autodiff::nd`): `gather(indices)`
  (embedding lookup `(vocab, dim) → (n, dim)`, backward scatter-add — repeated
  indices accumulate, never-seen rows keep a zero gradient)
  and `cross_entropy(targets)` (softmax + average NLL **fused**, stable
  log-sum-exp, backward `(softmax − onehot)/n`). Gradient-checked; sanity
  `uniform logits → ln(vocab)`.
- **N-D causal attention** (`NdMultiHeadAttention { causal }`, propagated to
  `NdTransformerBlock`): additive triangular mask (`-1e9` above the
  diagonal) before the softmax — no new autograd op. **Causality** test:
  perturbing the last input token leaves **every** earlier output
  bit-for-bit unchanged, while the perturbed output moves.
- **Complete trainable N-D transformer block** (`nn::nd_layers`):
  `NdLinear`, `NdMultiHeadAttention`, `NdLayerNorm` (affine γ/β) and
  `NdTransformerBlock` (Pre-LN: `x + Attn(LN(x))`, `x₁ + FFN(LN(x₁))`) on the
  N-D tape, all **trainable** (`sgd_step`). Tests: gradient check of
  input/attention layer/LayerNorm, **an N-D MLP that learns** AND **a complete
  N-D transformer block that learns** (loss < 70% of the initial). N-D
  ops added: `bmm`, `softmax`, `transpose_last2`, `reshape`, `permute`,
  `layernorm` — all gradient-checked.
- **`MiniLLM::generate_sampled(&str)`**: public generation from a
  string, sampling seeded on the KV-cache, deterministic; greedy reproduces
  `generate`.
- **Gradient-checked N-D attention**: `autodiff::nd` expresses a **complete
  multi-head attention** `softmax(Q·Kᵀ/√d)·V` on `(heads, seq, d)` (ops
  `bmm`/`transpose_last2`/`softmax`/`mul`/`add`/`sub`/`relu`/`sum`), validated
  by gradient check. The N-D tape becomes the capable superset; the 2D
  remains the default by architectural choice (coexistence, cf. GROWTH_PLAN).
- **Seeded sampling** (`nn::sampling`): temperature / top-k / top-p driven by
  a seeded `PcgEngine` → deterministic. `MiniLLM::generate_ids_cached_sampled`
  (O(n) generation with KV-cache + sampling). Greedy reproduces the argmax path.
- **Byte-level BPE** (`ByteBpeTokenizer`, GPT-2 style): base vocab = 256
  bytes ⇒ **no OOV**, **lossless** round-trip on any UTF-8 (accents,
  emoji, unknown scripts). Deterministic. Exposed in CLI via `bpe --bytes`.
- **End-to-end LLM**: O(n) KV-cache decoding (`MiniLLM::generate_ids_cached`,
  `TransformerBlock/Encoder::infer_step`, `PositionalEncoding::encoding_at`)
  **proven equivalent** to full recomputation; generation decoupled from the
  tokenizer (`MiniLLM::generate_ids`) → a BPE can drive the generation
  (integration test in `scirust-learning`). Greedy decoding (sampling to come).
- **`bpe` CLI**: trains a deterministic BPE tokenizer on a corpus
  (documents separated by `;`), encode/decode, reports the vocab size and the
  round-trip. Backed by `scirust-learning` (38 → 39 commands; new NLP
  group).
- **N-D batched matmul** (`NdVar::bmm`): `(…,m,k)·(…,k,n)→(…,m,n)` with batch
  axes broadcast — the capability that the 2D tape cannot express
  (per-head attention scores). Forward + backward gradient-checked.
- **N-D autograd (MVP, P2.4)**: `autodiff::nd` — `NdTape`/`NdVar` on
  `TensorND` (broadcast add/mul, 2D matmul, relu, sum), alongside the production
  2D tape. Validated by a **numerical gradient check** (finite
  differences vs backward) on `sum(relu(X·W+b)·V)`.
- **Expanded GPU ops**: wgpu elementwise kernel (add/mul/relu); a whole
  layer (matmul → +bias → relu) stays **resident in VRAM**, validated against
  the CPU oracle on lavapipe.
- **ONNX import**: `import_onnx_json` + `OnnxGraph::weights` — the weights
  make an export→import round-trip **bit-exact** (checkpoint format).
- **Verified KV-cache**: test proving that incremental decoding
  (`MultiHeadAttention::infer_step`) gives the same last token as the full
  forward — O(n) decoding now tested.
- **Deterministic BPE**: pair tie-break (`(count, Reverse(pair))`) — the
  `max_by_key(count)` depended on the HashMap iteration order; +5 tests.

### Fixed
- **Code review (max-effort) — hardening**: (1) resident GPU path
  (`GpuChain`): degenerate dimensions (`m`/`n`/`k == 0`) made wgpu panic
  (zero-size buffers); guards added (4-byte placeholder,
  skipped dispatch, short-circuited `download`) + test. (2) `scirust ode`:
  `h = 0` caused an overflow (panic, code 101), `t1 ≤ t0`
  silently returned `y0` (code 0, wrong answer) and dopri5/rk4
  diverged on bad bounds; unified guard (`t1 > t0`, finite `h > 0`
  → code 2) + tests. The other review axes (GEMM/transpose math, Conv2d
  gradient routing, `matmul_gpu` av/ar, threaded reduction
  determinism, SIMD cfg restructuring) were traced by hand: correct.
- **`scirust-rustc-driver` recompiles (P2.3, infra)**: the driver (excluded from the
  workspace, `rustc_private`) no longer compiled on the current nightly
  (`get_attrs` returns an iterator, not a slice). Fixed + warning-clean;
  informative CI job `rustc-driver` (continue-on-error) to make future API
  drift visible; `scirust-rustc-driver/target/` removed from git tracking
  (build artifacts) and ignored.

### Added
- **N-D shape inference primitives (P2.4, foundation)**: `TensorND`
  gains `broadcast_shape`, `matmul_shape` (batched matmul, batch-axis
  broadcast) and `broadcast_to` (numpy materialization) — the shape-inference
  building blocks "beyond rows/cols" that the future N-D tape/IR will use, with
  the existing `from/to_tensor_2d` bridge. 3 tests. (Fusing the 2D tape
  itself remains the big workstream, to be done in tested increments.)
- **Data-parallel training with certified determinism (P2.1)**:
  `DataParallelTrainer::train_batch_threaded(n_threads, ..)` runs the
  workers on N OS threads (work stealing via atomic counter) but reduces
  the gradients in a fixed order (worker 0,1,…,n-1), independent of the
  scheduler. Since floating-point addition is not associative, the result
  is **bit-identical for 1/2/4/8 threads** and identical to the sequential one —
  guarantee tested in CI. *(Corrected 2026-07-10: the uniqueness claim noted here
  at the time is withdrawn — RepDL, arXiv:2510.09180, has provided since Oct. 2025
  the bit-for-bit cross-platform reproducibility of an f32 subset of
  PyTorch; see the 2026-07-10 entry.)* Three CI
  tests: order-sensitive contributions (±1e16), real autograd backward, and
  a **complete multi-step SGD loop** whose weight trajectory is
  bit-identical for 1/2/4 threads (the invariance composes over training).

### Added — SciPy parity of tails and Dirichlet-multinomial (4th pass of the probabilities track)
> Entry placed at the bottom of the "Unpublished" section on purpose: each parallel
> track inserts its own at the top, hence systematic conflicts on the
> same block; adding it here avoids them.
- **`scirust-stats::discrete` — log tail methods**: `logcdf`,
  `logsf` and `isf` (inverse survival function) added by default to the
  `DiscreteDistribution` trait, aligning the API with `scipy.stats`. `logsf` relies
  on the **direct** survival already overridden on each law (no
  `ln(1 − cdf)` that explodes in the tail), and `isf(p)` does its bisection on
  `sf` — more accurate than `quantile(1 − p)` for very small `p`. Validated
  against SciPy (binomial, Poisson, zeta) and by consistency
  `exp(logcdf) = cdf`, round-trip `isf∘sf`.
- **`scirust-stats::discrete::DirichletMultinomial`** — multivariate Pólya:
  a multinomial with Dirichlet(α)-distributed probabilities, vector
  generalization of the beta-binomial for **overdispersed count
  vectors** (word/topic counts, repeated categorical trials with
  drift). `ln_pmf`/`pmf` via the closed form in ln Γ, mean `n·αᵢ/A`,
  covariance with the overdispersion factor `ρ = (n+A)/(1+A)`, sequential
  drawing by conditional beta-binomials (exact stick-breaking,
  fixed order ⇒ bit-for-bit reproducible). SciPy 1.17.1 oracles
  (`dirichlet_multinomial([1,2,3], 10)` pmf/logpmf/cov) and exact fraction
  18/143; with 2 categories = beta-binomial (tested), α = [1,1] = uniform.
- 48 tests + doctest on the crate, clippy 0 warnings.

### Added — interval/expect + Yule-Simon + Boltzmann (5th pass of the probabilities track)
> Entry at the bottom of the section (see the 4th pass note) to avoid
> systematic merge conflicts on the head block.
- **`scirust-stats::discrete` — `interval` and `expect`** added by default to the
  `DiscreteDistribution` trait, completing the `scipy.stats` parity:
  `interval(c)` returns the balanced interval `(quantile((1−c)/2),
  quantile((1+c)/2))`; `expect(f)` computes `E[f(X)] = Σ f(k)·pmf(k)` by
  bounded deterministic summation (stops when the tail mass `sf(k)` is
  negligible, safety cap). Validated against SciPy (binomial/
  Poisson/Yule-Simon intervals, `E[X]`/`E[X²]` = mean / var+mean²).
- **`scirust-stats::discrete::YuleSimon`** — **heavy-tailed** law on k ≥ 1,
  `pmf(k) = α·B(k, α+1)` (preferential attachment: word frequencies,
  citations). Power-law tail `k^(−(α+1))` ⇒ finite mean iff
  α > 1, finite variance iff α > 2; closed-form survival `sf(k) = k·B(k, α+1)`.
  SciPy oracles `yulesimon(2.5)` and exact identity `α=2 ⇒ 4/(k(k+1)(k+2))`.
- **`scirust-stats::discrete::Boltzmann`** — geometric truncated to `0..=n−1`
  (truncated Planck, `scipy.stats.boltzmann`),
  `pmf(k) = (1−e^(−λ))e^(−λk)/(1−e^(−λN))`; direct pmf/cdf/survival and
  closed-form moments (normalization via `−expm1` for precision at
  small `λN`). SciPy oracles `boltzmann(1.4, 10)`.
- 51 tests + doctest on the crate, clippy 0 warnings. Coverage:
  16 discrete laws (13 univariate + 3 vector).

### Added — log-series, Planck, and Loader pmf (6th pass of the probabilities track)
> Entry at the bottom of the section (convention of the previous passes) to avoid
> merge conflicts on the head block.
- **`scirust-special` — Loader's algorithm (saddle-point, Loader 2000)**:
  `stirling_error(x)` (remainder of the Stirling series δ, asymptotic series
  in 1/x for x ≥ 16, direct form otherwise — validated against mpmath to 40 digits),
  `binom_deviance(x, np)` (D₀ by series near x ≈ np to avoid
  cancellation), and the log pmfs `ln_poisson_pmf`/`ln_binomial_pmf`. Gains
  **full relative precision at large n/λ** where `exp(Σ lnΓ)` drifted
  (~1e-10 → ~1e-15). `Binomial::ln_pmf` and `Poisson::ln_pmf` rewired on top of it
  (this is the algorithm used by R's `dbinom`/`dpois` and SciPy). Validated against
  SciPy: `binom(1e5, 0.3)`, `poisson(1e4)`, exact endpoints.
- **`scirust-stats::discrete::Logarithmic`** — log-series law on k ≥ 1,
  `pmf(k) = −pᵏ/(k·ln(1−p))` (`scipy.stats.logser`), Fisher's species-abundance
  model; closed-form mean/variance. Oracles
  `logser(0.6)`.
- **`scirust-stats::discrete::Planck`** — **untruncated** geometric on
  k ≥ 0, `pmf(k) = (1−e^(−λ))e^(−λk)` (`scipy.stats.planck`), the n → ∞ limit
  of Boltzmann; tested equal to the shifted geometric. Oracles
  `planck(0.9)`.
- scirust-special 16 tests, scirust-stats 54 tests + doctest, clippy 0
  warnings. Coverage: **18 discrete laws** (15 univariate + 3
  vector).

### Added — discrete Laplace + method-of-moments fitting (7th pass of the probabilities track)
> Entry at the bottom of the section (convention of the previous passes).
- **`scirust-stats::discrete::DiscreteLaplace`** — discrete Laplace law
  (two-sided geometric) on ℤ, `pmf(k) = tanh(a/2)·e^(−a|k|)`
  (`scipy.stats.dlaplace`): difference of two geometrics, **the law of the
  geometric mechanism of differential privacy** (integer noise
  with a pure ε-DP guarantee; for sensitivity 1 and budget ε, take
  a = ε). Support ℤ ⇒ clean `i64` API like `Skellam` (direct
  pmf/ln_pmf/cdf/sf/moments/deterministic drawing = difference of two geometrics).
  Symmetric, mean 0. SciPy oracles `dlaplace(0.8)`.
- **Method-of-moments fitting** — `Poisson::fit_mom`,
  `Geometric::fit_mom`, `NegativeBinomial::fit_mom` (associated, `-> Option`):
  an **inference capability** (the equivalent of SciPy's `.fit()`) that estimates
  the parameters from a sample. Poisson `λ̂ = mean` (= MLE),
  geometric `p̂ = 1/mean`, negative binomial `p̂ = m/v, r̂ = m²/(v−m)`
  (defined only under overdispersion `v > m`, `None` otherwise — the
  underdispersed case falls under a Poisson). Validated by mean/var round-trip.
- 57 tests + doctest on the crate, clippy 0 warnings. Coverage:
  **19 discrete laws** (16 univariate + 3 vector) + MoM inference.

### Added — χ² goodness-of-fit test for fitted discrete laws (8th pass of the probabilities track)
> Entry at the bottom of the section (convention of the previous passes).
- **`scirust-stats::htest::chi2_gof_discrete`** — Pearson's χ² test between
  a **fitted discrete law** and observed counts, the loop that was missing
  between `fit_mom`, the pmfs and `htest`. Expected counts are drawn from the
  law (`N·pmf(i)` for the exact values, `N·sf(L−2)` for the tail
  class "≥ L−1", exact sum to N); **adjacent regrouping** until
  `min_expected` (Cochran's rule ≥ 5) which also absorbs the zero-probability
  classes of supports starting at 1 (Geometric); degrees of
  freedom adjusted for the number of estimated parameters (`ddof`). Delegates
  the final computation to the existing `chi_square_gof`. Validated against SciPy
  (`chisquare`/`chi2.sf`): Poisson(1.98) over 6 classes ⇒ χ²=2.2792, df=4,
  p=0.6846; rejection of a bad fit, regrouping of a 0 class of
  Geometric, degenerate inputs → `None`.
- 59 tests + doctest on the crate, clippy 0 warnings. The probabilities
  track loops: laws → combinatorics → ζ → Loader → inference (MoM)
  → **fit validation (GOF)**.

## [0.14.0] — 2026-06-13

### Fixed
- **Honest `scirust-gpu` (P2.2, "decide" step)**: the backends
  `WgpuBackend`/`CudaBackend` returned `vec![0.0; m*n]` — **fabricated** results
  (zeros) under a "wgpu"/"cuda" label, in violation
  of the "100% wired/tested, zero over-promise" policy. Replaced by
  a real **tested** CPU reference backend (bit-deterministic GEMM oracle)
  and device paths that honestly report `BackendError::Unavailable`
  (never invented output), following the example of `scirust_core::compute_backend`.
  The crate went from 0 to 6 tests. (The real wgpu wiring followed in a
  separate step — see "Added": WGSL GEMM tested on software Vulkan.)
- **Honest `docs/GPU.md`**: the page described, in one line, a GPU API
  (`GpuContext::try_init`, `ConvGpuPipelines`, `Conv2d::on_gpu`…) that
  does not exist (archived modules; `--features wgpu` compiles nothing).
  Rewritten as a status page + honest roadmap (what exists = tested CPU
  reference backend; why the GPU is not claimed; P2.2 plan).
- Merge regression breaking the build on all architectures
  (sgemv AVX2/SSE2/NEON, slab arena field).
- CI made feasible: removal of `--all-features` (mutually exclusive BLAS
  features), `deny.toml` rewritten (invalid TOML),
  aarch64 cross-check added; 6 gates green locally.
- Lazy graph operator fusion: pointwise chains now actually fuse
  (each link used to become its own chain of length 1).
- `RandomCrop` used to write its result into the void (silent no-op).
- 22 rustdoc warnings; rustc/clippy warnings brought back to zero
  (`-D warnings` tenable on all targets).

### Changed
- **GPU status** removed from the README's delivered-features table (it
  listed unwired stuff) → replaced by an honest "Not included
  yet" note pointing to the P2.2 roadmap.
- **100% deterministic data augmentation**: RNG `PcgEngine`
  injected, per-sample streams independent of order, `with_seed`
  effective, real Gaussian noise (Box-Muller).
- README aligned with the code: GPU status requalified as "Archived — not
  wired", measured test count.
- `publish = false` on the 51 manifests (path deps, non-commercial
  license).

### Added
- **Real and tested wgpu GPU (P2.2, "rewire" step)**: real `f32` GEMM
  in WGSL (`C = A·B`) behind the `wgpu` feature, run on the
  Vulkan/Metal/DX12/GL adapter via wgpu 0.20. **Validated against the CPU
  oracle** (documented float tolerance, since GPU accumulation is not
  bit-identical) and **tested in CI** on Mesa lavapipe software Vulkan
  (`llvmpipe`) — no hardware GPU required, "no claim without a test"
  respected. `cargo deny` passes on the wgpu dep tree; optional
  dependency (the 8 default gates do not compile it). New CI job
  `GPU (wgpu / lavapipe)`.
- **wgpu GPU wired into the autograd tape (P2.2, "tape" step)**:
  `WgpuEngine` implements the `Tape`'s `GpuEngine` hook (general GEMM
  kernel `C = α·op(A)·op(B) + β·C` with transposition). `Var::matmul_gpu`
  runs **forward AND backward** (`dA = g·Bᵀ`, `dB = Aᵀ·g`) on the GPU,
  device/pipeline cached, CPU fallback if a dispatch fails. Validated
  end-to-end against the CPU tape (forward + 2 gradients, tolerance) on
  lavapipe. Opt-in (feature + `matmul_gpu`) → the bit-exact guarantee by
  default stays intact.
- **GPU Conv2d (P2.2, "Conv2d" step)**: Conv2d's im2col GEMMs
  (forward `W·col`, backward `dW = dout·colᵀ` and `dInput = Wᵀ·dout`) go
  through the engine via the new `Tape::gemm_ab` helper (native transpose
  path), when a `WgpuEngine` is attached. Validated end-to-end against CPU
  Conv2d on lavapipe (forward + dInput + dWeight, tolerance). Bit-identical
  CPU fallback without an engine (no regression). im2col/col2im stay on CPU.
- **Activations resident in VRAM (P2.2, "residency" step)**: API
  `GpuChain` — upload inputs once, chain of `matmul` over
  `GpuMatrix` handles, an intermediate stays in GPU memory and feeds the
  next GEMM without CPU round-trips; only the final result is downloaded.
  Validated against the CPU oracle on lavapipe (chain of 2 GEMMs + transpose).
  The transparent residency in the tape (DeviceTensor lazily materialized
  on GPU) remains a future workstream — no measurable benefit without hardware GPU.
- **CycloneDX SBOM + release automation**: CycloneDX 1.5
  reproducible SBOM (`docs/sbom/scirust.cdx.json`, timestamp frozen via
  `SOURCE_DATE_EPOCH`, no random serial → byte-identical for a given
  source), generated by `./scripts/generate-sbom.sh`. New CI job
  `sbom` (artifact on every build) and `release.yml` workflow (on tag `v*`:
  replays the gates, generates the SBOM, creates the release and attaches
  the SBOM to it). SBOM section in `SECURITY.md`, `docs/sbom/README.md`
  (provenance).
- **CLI: 5th wave** — `tt` (tensor-train TT-SVD compression of a matrix,
  `scirust-tn`; reports cores, bond ranks, compression ratio and
  reconstruction error, exit 1 if `--max-err` exceeded), `solve-system`
  (nonlinear system F(x)=0 via Broyden, `scirust-solvers`), `inverse`
  (LU matrix inverse), `fem-heat` (1D heat −u″=source via linear finite
  elements), and `dopri5` method (adaptive Dormand–Prince) for `ode`.
  `FemSolver1D` was untested: 2 tests added (parabolic oracle
  −u″=f exact at the nodes + symmetry). New TENSOR NETWORKS group.
  `reconstruct_matrix` re-exported from `scirust-tn` (pair of
  `tt_decompose_matrix`). `newton_system` not exposed (closure `Fn(&[Dual])`
  like `bfgs`).
- **CLI: 4th wave** — `trig` (trigonometric identities), `patterns`
  (series trend), `qr` (QR decomposition), `cg` (SPD conjugate
  gradient). `bfgs` deliberately not exposed (closure `Fn(&[Dual])`
  not constructible from a symbolic expression evaluated in f64).
- **CLI: 3rd wave** — `symreg` (symbolic regression by genetic
  programming, `scirust-symreg`), `sat` (DPLL satisfiability,
  `scirust-neuro-symbolic`), and two more methods for `root`
  (`secant`, `newton` via symbolic derivative). New LOGIC group.
- **CLI: 2nd wave of commands** (29 → all tested): `integrate
  --method simpson|gauss`, `root --method bisection`, `optimize`
  (multi-variable Nelder–Mead), `lstsq` (QR least squares), `cholesky`,
  `prove` (symbolic equivalence), `gradient` (numerical 1–2 vars). The
  expression commands reuse `scirust-symbolic::eval`.
- **Massively expanded CLI** (19 commands, all backed by tested
  code): added `cmaes`; symbolic math `to-rust`, `regress`;
  numerical solvers `integrate` (Romberg), `root`/`minimize` (Brent,
  via symbolic derivative), `linsolve`/`det` (LU), `polyroots`,
  `ode` (RK4). The expression-driven commands use
  `scirust-symbolic::eval` as a bridge to the `scirust-solvers` solvers.
  +10 CLI tests; the order bug (intercept,slope) of `regress` fixed and
  pinned by a test.
- **Expanded `scirust` CLI** (industrial level): new grouped and documented
  commands — `som train` (ownership model, accuracy vs
  baseline), `evo` (seeded genetic optimizer), `diff`/`simplify`/`eval`/
  `solve` (symbolic math), `info` (guarantees). `scirust help` lists them
  by theme. Each command is backed by already-tested code.
- **Flash Attention really tested**: 4 tests in
  `nn/transformer/flash_attention.rs` (forward vs dense attention
  oracle, causal mask, bit-exact determinism, finite gradients) — the
  status line goes from claimed to verified.
- **Unified `scirust` CLI** (`scirust-cli`): single discoverable entry
  point (`scirust help`) grouping `quickstart` (bit-deterministic MLP 2→8→2
  demo, 4/4), `analyze` (ownership, delegates to som-cli),
  `verify` (certificates, delegates to `proofcli`), `version`. Verify logic
  factored into `scirust_runtime::proofcli` (zero duplication;
  `scirust-verify` now delegates). README quickstart rewritten
  around the CLI (no more copy-pasting 40 lines of API), library
  example fixed for the real API.
- **Rust stable support**: `#![feature(portable_simd)]` made truly
  optional (`cfg_attr`), scalar fallback for tiling; all 683 tests
  pass on stable; CI job `build-test-stable`. The nightly
  `portable-simd` feature (broken by the std::simd API migration) is fixed.
- **`scirust-verify`**: `SCIRUST-PROOF-1` inference certificates
  file-to-file (emit/verify, exit codes), tamper detection of
  artifact/certificate tested, bit-identical re-emission.
- **`cargo som` + `--sarif`**: the ownership linter as a cargo
  subcommand with SARIF 2.1.0 output for CI code scanning.
- **SOM operational on real Rust**: `syn` frontend
  (`scirust-som-frontend`), **type-aware** ownership oracle
  (exact Copy/move, E0382/E0502/E0503-style), `som-analyze` CLI,
  Transformer pipeline trained/evaluated against the oracle (ownership
  87.3% vs 33.1% baseline on held-out), bit-determinism tested.
- Rewired and repaired modules: `core::lazy` (fusion),
  `core::tensor::{broadcast,device}`, `scirust_symbolic::prelude`.
- `archive/`: historical sources removed from the build with documented status
  (GPU not wired, duplicated NEON/SVE, incorrect quantization draft).
- Industrial docs: `docs/REFERENCE.md` (exhaustive commands/binaries/API),
  `CONTRIBUTING.md`, `SECURITY.md`, audit
  `scirust_complete_audit_report.md`.
