# SciRust — Radar & Optronics, Block 25 (2026-07-12)

Follow-up deepening on the optronics side. Block 24 gave the sensor's intrinsic
sensitivity (NETD) — the contrast it needs *at the aperture*. This block adds
the atmosphere in between, turning that intrinsic sensitivity into an
operational **range budget**: how much target ΔT is needed to detect through an
attenuating path at a given range.

## What shipped — `scirust-vision::atmosphere`

- **`transmittance(α, R) = e^{−αR}`** — the **Beer–Lambert** path transmittance
  for extinction coefficient `α` (per metre) over range `R`; **`optical_depth`**
  = `α·R`; **`extinction(absorption, scattering)`** — extinction is additive over
  independent loss mechanisms.
- **`extinction_from_visibility(V) = 3.912/V`** — **Koschmieder's law**, the
  extinction implied by a meteorological visibility (the range at which a black
  target's contrast falls to the 2 % threshold); **`extinction_from_transmittance(τ, R)`**
  = `−ln(τ)/R`, its inverse.
- **`apparent_contrast(C₀, α, R) = C₀·e^{−αR}`** — the contrast-transmission law
  carrying an intrinsic target contrast to its apparent value at the sensor.
- **`required_delta_t(NETD, α, R) = NETD/τ`** — the **range budget**: since the
  path attenuates the thermal signal by `τ`, the target must exceed `NETD/τ` at
  the aperture, so the required ΔT grows with range.

## The oracles

- **Transmittance** is unity at zero range and decays monotonically, matching the
  closed form `e^{−αR}`; a clear path (`α = 0`) is fully transmissive.
- **Beer–Lambert is multiplicative** — `τ(α, R₁+R₂) = τ(α, R₁)·τ(α, R₂)`, so
  transmittance composes along the path.
- **Optical depth ↔ transmittance** are inverse, and recovering the extinction
  from a measured transmittance round-trips.
- **Koschmieder visibility hits the 2 % threshold** — transmittance over the
  meteorological visibility equals `0.02` by definition.
- **Additive extinction & apparent contrast** — extinction sums; apparent
  contrast follows `C₀·e^{−αR}`.
- **Required ΔT grows with range** — equals NETD at zero range and `NETD/τ`
  beyond it.

## Verification

- `cargo test -p scirust-vision` — **60 tests green** (+6).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-vision --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The optronics chain is now complete from scene to decision: **radiometry**
(Planck/Stefan–Boltzmann/Wien) → **atmosphere** (Beer–Lambert transmission, range
budget) → **sensitivity** (NETD/MRTD) → **optics** (PSF/MTF, deconvolution) →
**detection** (image-domain CFAR) → the shared tracking stack. With the radar
front-end and the tracking deepenings (blocks 1–24), the program is a complete,
physically-grounded sensor-to-track suite across the radar and EO/IR modalities.
