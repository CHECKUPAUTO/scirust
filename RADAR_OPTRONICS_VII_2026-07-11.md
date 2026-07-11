# SciRust — Radar & Optronics, Block 7 (2026-07-11)

Follow-up to block 6 (MVDR/Capon high-resolution DOA). This block opens the
**FMCW** track — the continuous-wave / mmWave processing model of the user's
second reference project (OpenRadar, TI mmWave). Where blocks 1–6 built the
pulse-Doppler chain (coded pulse → matched filter → Doppler → CFAR → array),
FMCW replaces the matched filter with a *mixer*: the echo beats against the
still-rising transmit sweep, and range plus velocity fall out of two FFTs of
that beat signal.

## What shipped — `scirust-signal::radar::fmcw`

- **`beat_frequency_to_range(f_beat, slope, c)`** — the core FMCW range law
  `R = f_b·c / (2·slope)` (a chirp of slope `slope` and round-trip delay
  `τ = 2R/c` produces a beat `f_b = slope·τ`). Guards a non-finite / non-positive
  slope.
- **`range_resolution(bandwidth, c)`** — `ΔR = c / (2·B)`: two targets closer
  than this share a range bin no matter how finely the beat spectrum is sampled.
- **`range_profile(beat)`** — the range profile of one chirp: the fast-time FFT
  of its complex beat signal. A target at range `R` peaks at the bin of its beat
  frequency. Power-of-two guard (radix-2 FFT).
- **`range_doppler(frames)`** — the **range-Doppler cube** from a frame of raw
  beat chirps: a fast-time (range) FFT of every chirp, then a slow-time (Doppler)
  FFT of every range bin across the chirps. `N × M` magnitude map
  `[range][doppler]`, Doppler bin 0 stationary.

This deliberately does **not** overlap `radar::doppler::range_doppler_map`, which
assumes the pulses are *already* range-compressed and only does the slow-time
FFT. FMCW does both FFTs, starting from raw beat samples — the distinct
continuous-wave data model.

## The oracles

- **Range profile peaks at the beat bin** — a pure beat tone at bin `k0`
  transforms to a single spectral line at `k0` with coherent amplitude `N`.
- **Beat-frequency range round-trips** — `f_b = 2·slope·R/c` inverts back to `R`.
- **Range resolution matches the closed form** — a 4 GHz sweep gives 3.75 cm.
- **Range-Doppler localizes a moving target** — a target in range bin `r0` whose
  beat phase advances `kd` cycles over the `M` chirps peaks at `(r0, kd)` with
  coherent amplitude `N·M`, and no other range row leaks energy at Doppler `kd`.
- **Stationary target sits at zero Doppler** — an identical beat tone on every
  chirp lands in Doppler bin 0.
- **Guards** — non-power-of-two chirp count / length and ragged frames → empty.

## Verification

- `cargo test -p scirust-signal` — **147 tests + 1 doctest green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- Not in the Miri gate (scirust-signal carries the SIMD FFT path), so no
  transcendental-perturbation gating was needed.

## Where the radar track stands

- Single-channel pulse-Doppler chain: **complete** (blocks 1–4).
- Array / angle processing: conventional beamformer (block 5) + MVDR/Capon
  high-resolution DOA (block 6).
- **FMCW / mmWave** ranging + range-Doppler cube (this block).
- Remaining: **MUSIC/ESPRIT** subspace DOA (a small local complex-Hermitian
  eigensolver on the block-6 covariance), and **detection → track** (cluster
  CFAR detections, then reuse `scirust-estimation`'s Kalman/IMM). Then the wider
  program's optronics / optical-imaging / optoelectronic-device pieces.
