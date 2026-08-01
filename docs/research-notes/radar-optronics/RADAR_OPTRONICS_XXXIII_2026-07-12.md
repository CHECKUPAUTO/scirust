# SciRust — Radar & Optronics, Block 33 (2026-07-12)

Follow-up deepening on the radar waveform side. Fine range resolution needs wide
bandwidth, but transmitting and digitising a wideband pulse is expensive. A
**stepped-frequency** waveform reaches the same resolution with cheap narrowband
hardware — the technique behind ISAR and ground-penetrating radar — and this
block synthesises its high-resolution range profile.

## What shipped — `scirust-signal::radar::stepped_frequency`

- **`synthetic_bandwidth(N, Δf)`** = `N·Δf` — the bandwidth synthesised by a burst
  of `N` pulses stepped by `Δf`.
- **`range_resolution(N, Δf)`** = `c/(2·N·Δf) = c/(2·B)` — set by the total
  synthesised bandwidth, and the spacing of the profile bins.
- **`max_unambiguous_range(Δf)`** = `c/(2·Δf)` — the periodic range window set by
  the step size (`= N·Δr`).
- **`range_profile(measurements)`** — the magnitude of the inverse DFT of the
  per-step complex reflectivity samples `H[n]` (the frequency response of the
  range profile), one value per bin. **`range_bins(N, Δf)`** maps bins to ranges.

## The oracles

- **Bandwidth / resolution formulas** — `B = N·Δf`, `Δr = c/(2B)`; resolution
  improves as the synthesised bandwidth grows, and the unambiguous window is
  exactly `N` bins wide.
- **Single scatterer localises to its bin** — the headline: a point scatterer
  placed on a range bin produces a profile that peaks exactly there, and the bin
  maps back to the true range.
- **Resolves two scatterers a few bins apart** — two peaks above the valley
  between them.
- **Finer steps widen the window without changing resolution** — resolution
  depends on the *total* bandwidth `N·Δf`, so halving `Δf` at fixed `N` coarsens
  resolution but doubles the range window.
- **Guards** — an empty or non-power-of-two burst returns an empty profile.

## Verification

- `cargo test -p scirust-signal` — **238 tests green** (+5).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar waveform/ranging repertoire now spans LFM pulse compression, Barker
phase codes, FMCW dechirp, and stepped-frequency synthetic wideband ranging —
alongside the full detection, tracking, angle-estimation, and classification
stack. Together with the complete EO/IR optronics chain, the 33-block program is
a physically-grounded, closed-form-oracle-tested detect–track–classify suite
across both modalities.
