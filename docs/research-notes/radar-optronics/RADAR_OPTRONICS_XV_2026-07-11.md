# SciRust — Radar & Optronics, Block 15 (2026-07-11)

Follow-up deepening. Block 13 added the semiconductor **laser** (the
optoelectronic emitter); this block adds its natural counterpart — the
**photodiode** (the optoelectronic receiver) — as a `scirust-sim` `System`,
completing the emitter/detector pair of an optical link.

## What shipped — `scirust-sim::photodiode`

- **`responsivity(η, λ)`** — the spectral responsivity `ℛ = η·q·λ/(h·c)` (A/W):
  each absorbed photon of energy `h·c/λ` yields `η` electrons, so responsivity
  rises linearly with wavelength. The core photon-to-electron physics.
- **`Photodiode` / `PhotodiodeParams`** — a photodiode driven by constant optical
  power into a resistive load with junction capacitance. State `y = [v]`:
  `v' = (I_ph − v/R_L)/C_j`, with the photocurrent `I_ph = ℛ·P_opt + I_dark`
  charging `C_j` through `R_L`.
- Closed-form observables: `photocurrent`, `steady_state_voltage` (`I_ph·R_L`),
  `time_constant` (`R_L·C_j`), and the `−3 dB` `bandwidth` (`1/(2π·R_L·C_j)`).

## The oracles

- **Responsivity** — matches `η·q·λ/(h·c)` (≈ 1.25 A/W at 1.55 µm, η = 1) and is
  linear in both wavelength and quantum efficiency.
- **Photocurrent / steady state / bandwidth** — the closed forms.
- **Dark floor** — with no light the output is the dark current across the load.
- **Step response** — integrating from a dark start, the load voltage charges
  with the `RC` time constant: `v(τ) = v_ss·(1 − 1/e)`, settling to `v_ss` by ten
  time constants.
- **Guards** — invalid parameters are rejected.

## Verification

- `cargo test -p scirust-sim` — **114 tests green** (+5).
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean; stays Miri-clean (the ODE
  simulation test is gated for Miri speed, the closed-form tests run under it).

## Where the program stands

The full radar (1–10) + optronics (11–14) program is merged; this is another
optional deepening. `scirust-sim` now models the complete optoelectronic pair
(laser emitter + photodiode detector). Remaining optional pieces: Wiener
frequency-domain deconvolution (2-D FFT), ESPRIT DOA (reusing
`music::hermitian_eig`), an IMM/Kalman tracker upgrade, and LED / avalanche-gain
device variants.
