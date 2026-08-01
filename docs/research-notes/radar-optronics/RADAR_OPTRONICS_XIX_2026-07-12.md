# SciRust — Radar & Optronics, Block 19 (2026-07-12)

Follow-up deepening. Block 10 added the α–β multi-target tracker; this block
lifts the two limits of a fixed-gain filter — no adaptation, no uncertainty
estimate — by adding a full **Kalman filter** and an **Interacting Multiple
Model (IMM)** estimator for manoeuvring targets. The existing α–β tracker is
left intact; these are higher-fidelity filters alongside it.

## What shipped — `scirust-signal::radar::kalman`

- **`KalmanCV`** — a constant-velocity Kalman filter over the scalar state
  `(p, v)` with the `2×2` covariance carried explicitly. It uses the
  continuous-white-noise-acceleration process model
  `Q = q·[[dt³/3, dt²/2], [dt²/2, dt]]`, so the gain adapts to the
  measurement/process-noise balance instead of being frozen. `predict` /
  `update` / `step` advance it; `update` returns the Gaussian **innovation
  likelihood**, and `position_variance` exposes the state uncertainty the α–β
  filter cannot report.
- **`Imm`** — the Interacting Multiple Model estimator: a bank of `KalmanCV`
  filters blended each frame by a Markov mode-transition matrix. Each step
  **mixes** the models' states by the transition-weighted mode probabilities,
  runs every model's predict/update on the measurement, updates the mode
  probabilities from the model likelihoods, and reports the probability-weighted
  combined estimate. A quiet (low-`q`) model dominates in steady flight; the
  agile (high-`q`) model's likelihood wins the instant the target manoeuvres, so
  the estimate follows the manoeuvre with far less lag than any single fixed
  model. The 2-D state keeps every matrix a `2×2` handled in closed form —
  dependency-free.

## The oracles

- **Kalman recovers constant velocity** — on a noise-free ramp the filtered
  velocity converges to the true slope.
- **Update reduces variance, reaches steady state** — a predict grows the
  position variance and the paired update shrinks it back to the same
  steady-state value (the algebraic Riccati fixed point).
- **Likelihood peaks at the prediction** — from one predicted state, a
  measurement at the prediction is more likely than one five sigma away.
- **IMM mode probabilities are a valid distribution** — non-negative and summing
  to one.
- **IMM favours the quiet model on a steady target** — the low-process-noise
  model explains a constant-velocity track with a tighter innovation variance,
  so it wins the mode probability.
- **IMM beats a lone quiet filter through a manoeuvre** — the headline test: a
  target reverses velocity sharply mid-track; the IMM's agile-model probability
  rises and its post-manoeuvre position error is strictly smaller than a single
  quiet Kalman filter's.
- **Guard** — an empty model bank is inert.

## Verification

- `cargo test -p scirust-signal` — **175 tests green** (+7).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The full radar (1–10) + optronics (11–17) program plus the ESPRIT deepening (18)
is merged. The tracking layer now spans fixed-gain α–β, adaptive Kalman, and the
manoeuvre-adaptive IMM. This completes the two optional deepenings that were on
the list (ESPRIT DOA and the IMM/Kalman tracker upgrade).
