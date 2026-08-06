# SciRust — Radar & Optronics, Block 36 (2026-07-12)

A new radar **mode**: imaging. Every block so far has produced detections, tracks,
angles, or classifications from the return; this block turns the radar into a
camera. **Synthetic-aperture radar (SAR)** trades the platform's motion for
angular resolution a real antenna cannot reach, and it does so with the same
pulse-compression matched filter built in block 2 — now applied in the along-track
(azimuth) dimension.

## The idea

A real antenna of along-track length `D` has azimuth beamwidth `λ/D`, so at range
`R` its cross-range resolution is `λR/D` — metres-to-kilometres, coarse. As the
radar flies past a point target at closest-approach range `R₀`, the two-way range
history traces a parabola `R(x) ≈ R₀ + (x−x₀)²/(2R₀)`, imprinting a quadratic
phase — an **azimuth chirp** — on the slow-time return. Matched-filtering that
chirp focuses the target to a sharp peak. The synthesised aperture spans
`L_sa = λR/D`, and the focused resolution is the celebrated `δ_az = D/2`:
**independent of range**, and finer for a *smaller* antenna — the defining,
counter-intuitive SAR result.

## What shipped — `scirust-signal::radar::sar`

- **`synthetic_aperture_length(λ, R, D)`** = `λR/D` — the along-track span the
  coherent integration synthesises.
- **`azimuth_resolution(D)`** = `D/2` — the range-independent cross-range
  resolution.
- **`azimuth_doppler_bandwidth(v, D)`** = `2v/D` — the slow-time bandwidth the
  azimuth filter compresses.
- **`azimuth_chirp_rate(v, λ, R)`** = `2v²/(λR)` — the Doppler rate of the phase
  history.
- **`azimuth_history(R, x₀, λ, positions)`** — the slow-time phase history
  `exp(−j·2π·(x−x₀)²/(λR))` of a point target.
- **`azimuth_reference(R, λ, positions)`** — the reference chirp (target at 0).
- **`focus_azimuth(signal, reference)`** — azimuth compression by cross-correlation,
  reusing the crate's matched filter.

Built on `Complex` and `radar::matched_filter::cross_correlate`; dependency-free.

## The oracles

- **Closed-form resolution / aperture / bandwidth** — `δ_az = D/2` (no range
  term at all), `L_sa = λR/D` (grows with range), `B_d = 2v/D`.
- **Chirp-rate scaling** — `k_a ∝ v²` and `k_a ∝ 1/R`.
- **The azimuth history is a linear-FM chirp** — its phase has a constant second
  difference equal to `−4π·Δx²/(λR)`, and matches the parabolic phase exactly.
- **Matched filter focuses a point target at its position** — the headline: the
  azimuth chirp compresses to a correlation peak at exactly the target's
  along-track lag.
- **Two separated targets resolve** — two focused peaks with a valley between.
- **Guards** — empty positions, degenerate geometry (`λ ≤ 0`, `R ≤ 0`, `D ≤ 0`)
  return safe values, no NaN.

## Verification

- `cargo test -p scirust-signal` — **259 tests green** (+7).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar now spans all its principal modes: search/track pulse-Doppler (detection,
CFAR, tracking, angle estimation), airborne GMTI (STAP), and — with this block —
imaging (SAR azimuth compression). Together with the waveform/ranging chain, the
classification stack, and the complete EO/IR optronics chain, the 36-block program
remains a physically-grounded, closed-form-oracle-tested capability across both the
radar and optronics modalities.
