# SciRust — Radar & Optronics, Block 13 (2026-07-11)

Third and final pillar of the optronics program. The user's brief named three
things — "optoélectronique optique de précision traitement de l'image":

- **traitement de l'image** (image processing) — block 11 (`scirust-vision::optics`).
- **optique de précision** (precision optics) — block 12 (`scirust-vision::beams`).
- **optoélectronique** (optoelectronics) — **this block**.

This block adds the canonical optoelectronic *device* model — the
semiconductor-laser rate equations — as a `scirust-sim` `System`, so a laser
diode joins the battery / HVAC / grid plants as a runnable, oracle-tested
dynamical system.

## What shipped — `scirust-sim::laser`

- **`SemiconductorLaser` / `LaserParams`** — the single-mode rate equations for
  the carrier density `n` and photon density `s`:
  - `n' = J − n/τ_n − g₀·(n − n_t)·s`
  - `s' = Γ·g₀·(n − n_t)·s − s/τ_p + Γ·β·n/τ_n`
- Closed-form observables (the `β → 0` limit): `threshold_density`
  (`n_th = n_t + 1/(Γ·g₀·τ_p)`), `threshold_pump` (`J_th = n_th/τ_n`),
  `steady_state_photon_density` (the linear light–current law
  `s_ss = Γ·τ_p·(J − J_th)`), `steady_state_carrier_density` (gain clamping at
  `n_th`), and `relaxation_frequency` (`f_r = √(g₀·s_ss/τ_p)/2π`).

## The oracles

- **Threshold & L–I law** — the closed forms give `n_th`, `J_th`, the linear
  `s_ss`, and gain-clamped `n_ss`; the light–current slope is exactly linear
  above threshold.
- **Turn-on** — integrating from dark (with a photon seed) converges to the
  closed-form steady state.
- **Below threshold** — the laser stays dark and carriers settle at `J·τ_n`.
- **Relaxation oscillation** — a small kick from steady state rings, and the
  oscillation period measured from the trajectory matches `1/f_r`.
- **Spontaneous emission** — with `β > 0` the laser turns on from `s = 0`
  exactly (no photon seed needed).
- **Guards** — invalid parameters are rejected.

## Verification

- `cargo test -p scirust-sim` — **109 tests green** (+7).
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean.
- The dynamical (simulation) tests carry `#[cfg_attr(miri, ignore)]` for Miri
  speed, matching the crate's other plant models; the closed-form tests run
  under the Miri gate too.

## Where the program stands

The radar/optronics program now spans all three requested pillars:

- **Radar** signal-processing pipeline: complete (blocks 1–10).
- **Optronics — image processing** (block 11) and **precision optics** (block 12)
  in `scirust-vision`.
- **Optronics — optoelectronic device dynamics** (this block) in `scirust-sim`.

Optional deepenings remain (Airy PSF + Wiener deconvolution; ESPRIT DOA;
IMM/Kalman tracker; photodetector / LED device models), but the sellable
radar-plus-optronics capability set the user asked for — the kind a Safran /
Sagem would recognise — is in place end to end.
