# SciRust — Radar & Optronics, Block 2 (2026-07-11)

Follow-up to block 1 (pulse-compression waveforms + matched filtering). This
block adds the **detection** stage of the radar chain — the step that both
reference projects the user pointed to (OpenRadar; AERIS/plfm_radar) place right
after pulse compression: **constant-false-alarm-rate (CFAR) detection**.

## What shipped — `scirust-signal::radar::cfar`

CFAR sets a per-cell threshold from the *local* noise/clutter level so the
false-alarm probability stays fixed as that level varies — you cannot use a
fixed threshold on a scene whose noise floor changes with range and clutter.

- **`ca_cfar`** — cell-averaging CFAR. The noise is the mean of `num_train`
  reference cells each side (skipping `num_guard` guard cells so a target does
  not bias its own estimate); a detection is `CUT > α · mean`, with the
  closed-form scaling `α = N·(P_fa^{−1/N} − 1)` (`N = 2·num_train`).
- **`ca_cfar_alpha`** — that scaling, exposed.
- **`os_cfar`** — ordered-statistic CFAR: the noise estimate is the `k`-th
  smallest reference cell instead of the mean, so a few interfering targets or a
  clutter edge in the window do not inflate the threshold.
- **`os_cfar_alpha`** — the OS threshold factor, found by bisection on the
  strictly decreasing `P_fa(α) = ∏_{i=0}^{k−1} (N−i)/(N−i+α)` (no special
  functions needed — the Gamma terms cancel to a plain product).

## The oracles

- **CFAR identity.** `ca_cfar_alpha` satisfies `(1 + α/N)^{−N} = P_fa` exactly.
- **False-alarm rate, statistically.** Over 20 000 cells of exponential noise
  (the model CFAR is designed for, from a deterministic in-test LCG), the
  empirical false-alarm rate matches the design `P_fa` — the CFAR property that
  it holds *regardless of the noise level*.
- **Detection without flooding.** A target on a flat floor is detected, and it
  does not leak into neighbours' estimates enough to trip them: exactly one
  detection.
- **OS-CFAR robustness.** A weak target next to a strong interferer (inside its
  reference window) is *masked* by CA-CFAR — the interferer inflates the mean —
  but still detected by OS-CFAR, the property that motivates it.
- **α inversion.** `os_cfar_alpha` inverts the `P_fa` formula to 1e-9.

## Verification

- `cargo test -p scirust-signal` — **123 tests + 1 doctest green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Roadmap (refined from the reference projects)

Both references center on the **range-Doppler → CFAR** detection chain, so next:

- **Block 3 — Doppler processing** (`radar::doppler`): slow-time FFT and the
  range-Doppler map that CFAR runs on; a stationary vs. moving target lands in
  the zero- vs. non-zero-Doppler bin.
- **Block 4 — ambiguity function** and MTI clutter cancellers.
- **FMCW track** (OpenRadar's focus): dechirp/beat-frequency range + the
  range-Doppler cube.
- **Beamforming / DOA** (both, AERIS's phased array): delay-and-sum + MVDR and
  MUSIC/ESPRIT (reusing `scirust-solvers` eigendecomposition).
- Then optronics (Gaussian beams, ABCD rays), optical imaging (PSF/MTF/
  deconvolution in `scirust-vision`), the optoelectronic laser rate equations
  (a `scirust-sim` `System`), and detection→track (clustering + the existing
  Kalman/IMM filters).
