# SciRust — Radar & Optronics, Block 39 (2026-07-12)

The performance-analysis complement to the estimator library. Every earlier block
*produces* a measurement — a delay, a Doppler, an angle. This block gives the
*theoretical floor* on how precise any unbiased estimate can be: the **Cramér–Rao
lower bound (CRLB)**, the number a radar link budget must close not to *detect* a
target (that is [`range_equation`], block 29) but to *measure* it to spec.

## What shipped — `scirust-signal::radar::accuracy`

All three headline bounds scale as `1/√SNR`:

- **`rms_bandwidth_lfm(B)`** = `B/√12` and **`rms_duration_rect(T)`** = `T/√12` —
  the second moments of a flat spectrum / rectangular pulse that drive the bounds.
- **`delay_crlb(SNR, β_rms)`** = `1/(2π·β_rms·√(2·SNR))` and **`range_crlb`** =
  `(c/2)·σ_τ` — sharper with wider RMS bandwidth.
- **`doppler_crlb(SNR, T_rms)`** = `1/(2π·T_rms·√(2·SNR))` and **`velocity_crlb`** =
  `(λ/2)·σ_fd` — sharper with a longer coherent dwell.
- **`angle_crlb(SNR, θ₃dB, k_m)`** = `θ₃dB/(k_m·√(2·SNR))` — the monopulse angle
  accuracy, sharper for a narrow beam and a steep difference-pattern slope.

`SNR` is linear (a power ratio). Dependency-free.

## The oracles

- **RMS bandwidth / duration** of a flat spectrum / rectangular pulse = `·/√12`.
- **Delay CRLB** matches the closed form (sub-nanosecond at 5 MHz, 20 dB).
- **Range = (c/2)·delay** and **velocity = (λ/2)·Doppler** conversions, with a
  realistic sub-metre range accuracy.
- **`1/√SNR` scaling** — quadrupling SNR halves every bound.
- **Sharpening** — wider bandwidth sharpens range, longer dwell sharpens velocity,
  a steeper monopulse slope and narrower beam sharpen angle.
- **Guards** — non-positive SNR / bandwidth / duration / wavelength / beamwidth /
  slope return `+∞` (no information), never a `0·∞ = NaN`.

## Verification

- `cargo test -p scirust-signal` — **273 tests green** (+7).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Where the program stands

The radar suite now carries its own performance-analysis layer: alongside the
detection budget (range equation, Swerling/Albersheim) sits the *measurement*
budget (CRLB), bounding the delay, Doppler, and angle estimators the rest of the
crate implements. This block was built as the seed of a wider parallel push;
subsequent blocks fan out further radar and EO/IR capability from the same base.
