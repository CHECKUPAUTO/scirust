# SciRust — Radar & Optronics, Block 16 (2026-07-11)

Follow-up deepening. The photodiode (block 15) is the baseline optoelectronic
detector; this block adds its high-sensitivity sibling — the **avalanche
photodiode (APD)** — the receiver behind lidar and laser rangefinders (very
relevant to a targeting/EO-IR optronics offering).

## What shipped — `scirust-sim::apd`

An APD runs its junction in avalanche breakdown so each primary photo-electron
triggers an impact-ionization cascade of average **gain** `M`. That gain lifts a
weak signal above the downstream thermal noise, but the random cascade adds
**excess noise** — the model's defining physics.

- **`excess_noise_factor(gain, k)`** — the McIntyre factor
  `F(M) = k·M + (1 − k)·(2 − 1/M)`, `k` the ionization ratio.
- **`Apd` / `ApdParams`** — a receiver at fixed gain, with the primary /
  multiplied / signal currents, the excess noise `F(M)`, the multiplied
  **shot-noise** variance `2·q·I_primary·M²·F·B`, the **thermal** variance
  `4·k_B·T·B/R_L`, and the electrical **SNR** `I_signal²/(σ²_shot + σ²_thermal)`.

The defining tension is closed-form: shot noise grows as `M²·F(M)` while the
signal grows as `M²`, so gain helps only until excess noise overtakes the
thermal floor — an **optimal gain** maximises SNR.

## The oracles

- **Excess noise limits** — `F(1) = 1` for any `k`; `F → 2 − 1/M` as `k → 0`
  (electron-only, quietest) and `F = M` for `k = 1` (noisiest); monotone in `M`,
  and larger `k` is noisier.
- **Currents & noise** — the primary/multiplied/signal currents and the
  shot/thermal variances match their closed forms.
- **Optimal gain** — for a weak-signal Si receiver, SNR at gain 50 beats both
  gain 1 (thermal-limited) and gain 1000 (excess-noise-limited) — the SNR peak.
- **Guards** — a sub-unity gain or an ionization ratio outside `[0, 1]` is
  rejected.

## Verification

- `cargo test -p scirust-sim` — **118 tests green** (+4).
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean; the crate stays Miri-clean (this
  module is closed-form only, no simulation to gate).

## Where the program stands

The full radar (1–10) + optronics (11–14) program is merged; `scirust-sim` now
models the optoelectronic set of an optical link end to end: **laser** emitter
(13), **photodiode** detector (15), and **APD** high-sensitivity detector (this
block). Remaining optional pieces: Wiener frequency-domain deconvolution (2-D
FFT) and ESPRIT DOA (reusing `music::hermitian_eig`) on the signal side, and an
IMM/Kalman tracker upgrade.
