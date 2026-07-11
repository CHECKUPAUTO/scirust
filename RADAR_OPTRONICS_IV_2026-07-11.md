# SciRust — Radar & Optronics, Block 4 (2026-07-11)

Follow-up to blocks 1–3 (the pulse-Doppler detection chain: waveform →
pulse compression → range-Doppler map → CFAR). This block adds the two pieces
that complete the classical single-channel radar signal-processing toolkit:
**waveform analysis** (the ambiguity function) and **clutter rejection** (MTI).

## What shipped — `scirust-signal::radar`

### `ambiguity::ambiguity`
The narrowband ambiguity surface `|χ(τ, ν)|` — a waveform's *joint*
delay–Doppler response, which tells you how a waveform resolves range and
velocity at once and where it is ambiguous. It is computed by cross-correlating
a Doppler-modulated copy of the waveform with the original (reusing block 1's
`cross_correlate`), one row per Doppler shift.

Oracles:
- **Origin = energy, global peak.** `|χ(0,0)|` equals the pulse energy and is
  the maximum of the whole surface.
- **Zero-Doppler cut = autocorrelation.** The `ν = 0` row is exactly the
  matched-filter output — the two views agree.
- **LFM range-Doppler coupling.** For a chirp the ridge is *sheared*: the delay
  of each Doppler row's peak moves monotonically with Doppler, so a Doppler
  shift masquerades as a range shift — the defining property of LFM waveforms.

### `mti::mti_canceller`
An `order`-pulse moving-target-indication canceller along slow-time: `order`
cascaded first differences, i.e. the binomial cancellers `[1,−1]` (2-pulse),
`[1,−2,1]` (3-pulse), … Its DC response is exactly zero.

Oracles:
- **Exact DC null.** A constant (stationary clutter) input cancels to zero at
  every order.
- **Binomial pass gain.** A moving tone at normalized Doppler `f` passes with
  the exact gain `|1 − e^{−j2πf}|^order`.
- **Clutter out, target kept.** Clutter-plus-target input returns only the
  moving target's response.

## Verification

- `cargo test -p scirust-signal` — **134 tests + 1 doctest green** (+8).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Where the radar track stands

The single-channel pulse-Doppler pipeline is now complete: **waveform design &
analysis (LFM/Barker + ambiguity) → pulse compression → range-Doppler map → MTI
clutter rejection → CFAR detection.** Remaining radar work is multi-channel and
alternative-waveform:

- **Beamforming / DOA** (angle) — delay-and-sum + MVDR and MUSIC/ESPRIT, reusing
  `scirust-solvers` eigendecomposition (AERIS's phased array; OpenRadar's AoA).
- **FMCW track** — dechirp/beat-frequency range + the range-Doppler cube
  (OpenRadar's core).
- **Detection → track** — clustering + the existing `scirust-estimation`
  Kalman/IMM filters.

Then the wider program's optronics (Gaussian beams, ABCD rays), optical imaging
(PSF/MTF/deconvolution in `scirust-vision`) and optoelectronic-device pieces
(laser rate equations as a `scirust-sim` `System`).
