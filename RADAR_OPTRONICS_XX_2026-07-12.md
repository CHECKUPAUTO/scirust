# SciRust — Radar & Optronics, Block 20 (2026-07-12)

Follow-up deepening. Block 19 added a 1-D Kalman/IMM tracker whose "agile" model
follows a manoeuvre by inflating its process noise — a proxy for a turn, not a
turn. This block adds the real thing: a planar **coordinated-turn IMM** that
tracks a genuinely turning target (aircraft/missile) with a model that knows
about turns, built on a reusable general linear Kalman filter.

## What shipped — `scirust-signal::radar::imm2d`

- **`KalmanLinear`** — a general `n`-state, `m`-measurement linear Kalman filter
  with dense matrices, a **Cholesky-based** measurement update (the innovation
  covariance is factored once and reused for the gain solve, the quadratic form,
  and the determinant in the likelihood), and covariance symmetrisation for
  numerical hygiene. Reusable well beyond tracking.
- **`cv_model_2d`** / **`ct_model_2d`** — the two planar motion models over the
  Cartesian state `[x, vₓ, y, v_y]`. CV is nearly-constant-velocity; CT is the
  coordinated-turn transition
  `[[1, s/ω, 0, −(1−c)/ω], [0, c, 0, −s], [0, (1−c)/ω, 1, s/ω], [0, s, 0, c]]`
  (`s = sin ωΔt`, `c = cos ωΔt`), which rotates the velocity vector at rate ω
  and integrates position along the arc, degenerating to CV as `ω → 0`.
- **`Imm2D`** — an Interacting Multiple Model estimator over a bank of these
  models sharing the 4-state, generalising block 19's IMM to a vector state and
  matrix covariance: mixing, model-matched filtering, mode-probability update
  from the likelihoods, and the probability-weighted combined `(x, y)` estimate.

## The oracles

- **CT → CV as ω vanishes** — a coordinated turn at a negligible rate predicts
  identically to constant velocity.
- **Linear Kalman recovers 2-D constant velocity** — on a noise-free straight
  line the filtered velocity components converge.
- **CT tracks a circle far better than CV** — the headline model test: on an
  exact circular (coordinated-turn) trajectory the CT filter's position error is
  under half the CV filter's.
- **IMM picks the turn model and beats CV on a manoeuvre** — a straight run
  followed by a turn; the CT mode probability rises across the turn and the IMM
  position error beats a lone CV filter's.
- **IMM mode probabilities are a valid distribution.**
- **Cholesky solves a known system** and rejects a non-positive-definite matrix.
- **Guard** — an empty model bank is inert.

## Verification

- `cargo test -p scirust-signal` — **182 tests green** (+7).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar (1–10) + optronics (11–17) program plus the ESPRIT (18) and Kalman/IMM
(19) deepenings is merged. The tracking layer now spans fixed-gain α–β, adaptive
Kalman, the 1-D manoeuvre-adaptive IMM, and this planar coordinated-turn IMM
with its reusable general linear Kalman engine — a defense-grade
manoeuvring-target tracking stack.
