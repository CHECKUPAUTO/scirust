# SciRust — Radar & Optronics, Block 35 (2026-07-12)

The flagship airborne-radar capability, and the natural join of the array-processing
and Doppler-processing threads built earlier. An airborne radar sees the ground as
clutter whose Doppler is **coupled to angle** — the platform's motion makes a
clutter patch at azimuth `θ` return at Doppler `f_d = β·f_s`, so in the joint
angle-Doppler plane the clutter collapses onto a one-dimensional **ridge**. A slow
mover buried under that clutter in both range and Doppler is nonetheless *separated
from it in the 2-D plane* — it sits off the ridge. **Space-time adaptive
processing (STAP)** adapts jointly across the `N` array elements and the `M` pulses
of a coherent processing interval to null the ridge while holding unit gain on the
target, detecting movers no angle-only or Doppler-only filter can.

## What shipped — `scirust-signal::radar::stap`

- **`spatial_frequency(θ, d)`** = `(d/λ)·sin θ` — the normalised spatial frequency
  of a ULA source; a half-wavelength array maps `±90°` to `±0.5`.
- **`clutter_ridge_doppler(f_s, β)`** = `β·f_s` — the ridge Doppler of a
  side-looking airborne array (`β = 1` is the classic 45° ridge).
- **`space_time_steering(f_s, f_d, N, M)`** — the joint steering vector
  `s = b(f_d) ⊗ a(f_s)`, length `NM`, pulse-major, `|s|² = NM`.
- **`clutter_covariance(N, M, patches, β, σ_n²)`** — the interference-plus-noise
  covariance `R = σ_n²·I + Σ_c P_c·s_c s_cᴴ` built from white noise and ground
  clutter patches distributed along the ridge.
- **`adaptive_weights(R, s)`** = `R⁻¹s/(sᴴR⁻¹s)` — the MVDR/SMI adaptive weight:
  minimum output power subject to unit gain toward `s`.
- **`optimal_sinr(R, s, P)`** = `P·sᴴR⁻¹s` — the delivered output SINR: deeply
  notched on the ridge, near the full `NM` coherent gain off it.

Reuses the shared complex Gauss-Jordan inverse from `radar::doa`; dependency-free.

## The oracles

- **Steering vector is the space-time Kronecker product** — `|s|² = NM` and it
  factorises exactly as `b(f_d) ⊗ a(f_s)`.
- **White noise ⇒ the matched filter** — with `R = σ²·I` the adaptive weight
  collapses to `s/NM`, gives unit gain (`wᴴs = 1`), and delivers the full coherent
  SINR `P·NM/σ²`.
- **Clutter notch suppresses endo-clutter targets** — the headline: against a
  strong clutter ridge, a target *on* the ridge is deeply notched while a target at
  the same angle but a clear Doppler (off the ridge) keeps a large fraction of the
  clutter-free gain — the endo- vs exo-clutter separation STAP exists for.
- **The weight nulls the clutter it competes with** — the adaptive filter designed
  for an off-ridge target places a deep null (>20 dB) toward a co-Doppler clutter
  patch at a different angle while holding unit gain on the target.
- **The SINR minimum falls on the ridge** — sweeping Doppler at a fixed angle, the
  SINR notch lands exactly at the ridge Doppler for that angle.
- **Ridge / spatial-frequency relations** and **guards** (dimension mismatch,
  empty or singular covariance → safe, no NaN).

## Verification

- `cargo test -p scirust-signal` — **252 tests green** (+7).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar array-processing stack now spans conventional beamforming, MVDR/Capon,
MUSIC, ESPRIT, amplitude- and phase-comparison monopulse, and — with this block —
full space-time adaptive processing, the joint spatio-temporal filter that closes
the airborne-radar clutter-rejection problem. Together with the waveform/ranging
chain (LFM, Barker, FMCW, stepped-frequency), the detection–tracking–classification
suite, and the complete EO/IR optronics chain, the 35-block program remains a
physically-grounded, closed-form-oracle-tested detect–track–classify capability
across both the radar and optronics modalities.
