# SciRust — Radar & Optronics, Block 21 (2026-07-12)

Follow-up deepening. Every tracker so far (α–β, Kalman, IMM, coordinated-turn
IMM) assumes the measurement is already a Cartesian position `(x, y)`. A real
radar reports **range and bearing** — a *polar* measurement, a nonlinear
function of the state. This block closes that gap with an **extended Kalman
filter** that tracks a Cartesian state directly from the raw polar returns.

## What shipped — `scirust-signal::radar::ekf`

- **`RadarEkf`** — an extended Kalman filter over the Cartesian state
  `[x, vₓ, y, v_y]`. The prediction stays linear (a constant-velocity step, so
  it is exact); the measurement update linearises the nonlinear polar
  observation `h(x) = (√(x²+y²), atan2(y, x))` about the current estimate via its
  `2×4` Jacobian

  ```
  H = [[ x/r,  0,  y/r,  0 ],
       [ −y/r², 0,  x/r², 0 ]]
  ```

  and applies the standard Kalman correction. The **bearing innovation is
  wrapped** to `(−π, π]`, so a target crossing the ±π azimuth boundary is
  tracked without a `2π` discontinuity. Range and bearing carry independent
  measurement variances. A target at (or numerically at) the origin has an
  undefined bearing, so the update is skipped there.

Reuses the dense-matrix helpers (`mat_mul`, `mat_t`, `cholesky`, `chol_solve`, …)
from [`imm2d`], now `pub(super)` — no duplication, no dependency.

## The oracles

- **`wrap_pi` maps into the principal interval** — including the ±π endpoints and
  a difference straddling the boundary.
- **EKF recovers a Cartesian track from polar measurements** — the headline
  test: a straight-line target is observed only in range/bearing, and the
  filter's Cartesian position and velocity converge to the truth.
- **EKF tracks across the bearing wrap** — a target moving along the −x axis
  (bearing near ±π) stays glued to the truth, proving the innovation wrapping.
- **Update shrinks the position variance.**
- **Predicted measurement matches the state** — `h(3, 4) = (5, atan2(4, 3))`.
- **Guard** — an update at the origin is inert.

## Verification

- `cargo test -p scirust-signal` — **188 tests green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar (1–10) + optronics (11–17) program plus the ESPRIT (18), Kalman/IMM
(19), and coordinated-turn-IMM (20) deepenings is merged/queued. The tracking
layer now runs from raw **polar measurements** (EKF) through Cartesian
constant-velocity, coordinated-turn, and IMM estimators — the full
measurement→state→manoeuvre chain of a modern radar tracker.
