# SciRust — Radar & Optronics, Block 27 (2026-07-12)

Follow-up deepening on the radar side — a target-classification capability. A
target's *bulk* motion gives one Doppler line; its *moving parts* (rotor blades,
a propeller, a walking gait) add small periodic Doppler modulations on top. In a
time–frequency representation those show up as a modulated ridge whose shape and
cadence identify the target class — the basis of **non-cooperative target
recognition (NCTR)**.

## What shipped — `scirust-signal::radar::micro_doppler`

- **`spectrogram(signal, win_len, hop)`** — a Hann-windowed short-time Fourier
  transform of the complex slow-time signal, on the crate's power-of-two
  [`fft`](../scirust-signal/src/fft.rs): one magnitude spectrum per frame.
- **`bin_frequencies(win_len, sample_rate)`** — the signed Doppler frequency of
  each FFT bin (natural order, negative frequencies folded).
- Descriptors from the spectrogram:
  - **`ridge`** — the per-frame peak frequency, tracing the instantaneous Doppler.
  - **`mean_doppler`** — the bulk (body) Doppler, the ridge mean.
  - **`doppler_bandwidth`** — the ridge's peak-to-peak swing, the micro-motion
    extent.
  - **`cadence`** — the micro-motion repetition frequency, from the first
    autocorrelation peak of the ridge *beyond the main lobe* (a naive global
    maximum would land inside the broad main lobe and mis-estimate the period).

## The oracles

Driven by a synthetic rotating-scatterer signal whose instantaneous frequency is
`f_b + f_max·cos(2π f_rot t)`:

- **Mean Doppler recovers the bulk motion** — the ridge mean returns `f_b`.
- **Bandwidth reflects the micro-motion** — peak-to-peak ≈ `2·f_max`, and is
  ~zero for a pure tone (no micro-motion).
- **Cadence recovers the rotation frequency** — the ridge autocorrelation returns
  `f_rot`.
- **A pure tone's ridge sits at its frequency** — flat ridge, no cadence.
- **Spectrogram shape and guards** — frame/bin counts; non-power-of-two window,
  zero hop, and under-length signal all return empty.
- **Empty ridge degenerates gracefully.**

## Verification

- `cargo test -p scirust-signal` — **204 tests green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar side now carries the full chain from waveform through detection,
tracking (α–β, Kalman, IMM, coordinated-turn IMM, polar EKF, NIS-gated
multi-target, PDAF), DOA (beamforming, MVDR, MUSIC, ESPRIT), and now
**classification** (micro-Doppler). Together with the complete EO/IR optronics
chain (blocks 11–17, 23–25), the program is a physically-grounded
detect–track–classify suite across the radar and EO/IR modalities.
