# SciRust — Radar & Optronics, Block 1 (2026-07-11)

Start of a **radar / optronics** capability program — the kind of sensor
signal-processing a defense prime (Safran/Sagem: radar, EO/IR optronics,
targeting, inertial navigation) would use — built by **extending the existing
crates**, not adding new ones. A survey confirmed there was *zero* radar,
optronics or photonics code anywhere in the workspace, so every piece is a
genuine gap. This block delivers the range-processing core: **pulse
compression**.

## What shipped — `scirust-signal::radar`

A new `radar` module built directly on the crate's existing `Complex`, FFT and
window primitives (no new dependency).

### `radar::waveform`
- **`lfm_chirp(n, bandwidth, sample_rate)`** — a complex-baseband linear-FM
  (chirp) pulse: instantaneous frequency sweeps linearly across the band, unit
  amplitude, parameterizable time-bandwidth product.
- **`barker_code(length)`** — the Barker binary phase codes (lengths 2–13), the
  only codes whose autocorrelation sidelobes never exceed 1.

### `radar::matched_filter`
- **`cross_correlate(signal, replica)`** — the matched-filter response (pulse
  compression), `r[lag] = Σ signal[k]·conj(replica[k − lag])`.
- **`peak_lag`** — the echo delay at the correlation peak.
- **`peak_to_sidelobe`** — peak magnitude over the largest out-of-window
  sidelobe.

## The oracles

Radar processing has exact, checkable properties:

1. **Matched-filter energy.** The chirp autocorrelation peak equals the pulse
   energy (`n`), and its −3 dB main lobe compresses to a handful of samples
   (≈ `fs/B`) versus the 256-sample pulse — a compression ratio equal to the
   time-bandwidth product.
2. **Barker peak-to-sidelobe.** The Barker-13 autocorrelation has a peak of 13
   and every sidelobe ≤ 1, so its peak-to-sidelobe ratio is exactly 13 — the
   property that makes Barker codes useful.
3. **Delay estimation.** A chirp embedded at a known delay in a longer record
   is located exactly by the matched filter's peak.
4. **Chirp spectrum.** The instantaneous frequency (per-sample phase increment)
   runs from −B/2 to +B/2.

## Verification

- `cargo test -p scirust-signal` — **117 tests + 1 doctest green** (+8).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Program roadmap (next blocks)

Building on this, one oracle-tested PR at a time:

- **Radar (scirust-signal)** — ambiguity function; CFAR detection (CA/OS);
  Doppler / range-Doppler processing (2-D FFT); MTI clutter cancellers;
  complex analytic (I/Q) signal; low-sidelobe windows (Taylor/Chebyshev/Kaiser);
  beamforming + DOA (MUSIC/ESPRIT, reusing `scirust-solvers` eigen).
- **Optronics / precision optics (scirust-signal or scirust-vision)** — Gaussian
  beams, ABCD ray matrices.
- **Optical image processing (scirust-vision)** — PSF, MTF, deconvolution
  (Wiener / Richardson-Lucy), 2-D FFT, IR/thermal (NUC/radiometry).
- **Optoelectronics (scirust-sim)** — semiconductor-laser rate equations (a
  `System` model), photodiode response.
- **Tracking (reuse scirust-estimation / scirust-nav)** — radar motion models
  feeding the existing Kalman/EKF/UKF/IMM filters; FDOA / 3-D TDOA / data
  association.
