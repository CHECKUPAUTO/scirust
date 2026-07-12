# SciRust — Radar & Optronics, Block 26 (2026-07-12)

Follow-up deepening on the tracking side. Block 22's multi-target tracker
associates each track to at most one measurement by a hard nearest-neighbour
choice; in clutter a single false return closer than the true one hijacks the
track. This block adds the standard remedy — the **Probabilistic Data
Association Filter (PDAF)** — which keeps *every* gated measurement and weights
it by its association probability.

## What shipped — `scirust-signal::radar::pda`

- **`PdaFilter`** — a single-target PDAF over the Cartesian constant-velocity
  state `[x, vₓ, y, v_y]` with `(x, y)` position measurements. Each `step`:
  1. predicts (`x ← F·x`, `P ← F·P·Fᵀ + Q`);
  2. gates the scan's measurements by their NIS and forms each one's likelihood
     `eᵢ = e^{−½·d²ᵢ}`;
  3. computes the association probabilities `βᵢ = eᵢ/(b + Σeⱼ)` and the
     no-detection probability `β₀ = b/(b + Σeⱼ)`, with the parametric-PDA clutter
     term `b = λ·|2πS|^{1/2}·(1 − P_D·P_G)/P_D` (gate mass `P_G = 1 − e^{−gate/2}`);
  4. updates the state with the combined innovation `ν̄ = Σ βᵢ νᵢ`;
  5. inflates the covariance with the **spread-of-innovations** term
     `P = β₀·P_pred + (1−β₀)·P_c + K(Σβᵢ νᵢνᵢᵀ − ν̄ν̄ᵀ)Kᵀ` so it honestly
     reflects the association ambiguity.

  `update` returns `β₀`. Reuses the shared dense-matrix helpers from
  [`imm2d`](../scirust-signal/src/radar/imm2d.rs).

## The oracles

- **Clutter-free reduces to Kalman** — with `λ = 0` and one measurement per scan
  the PDAF collapses to a standard Kalman filter and follows a constant-velocity
  target to the truth.
- **Tracks through dense clutter** — the headline test: each scan carries the
  noisy true measurement plus five uniform clutter returns near the prediction;
  the PDAF stays locked to the truth where a hard nearest-neighbour would be
  pulled off by clutter.
- **A missed scan coasts and grows the covariance** — an empty scan gives
  `β₀ = 1`, the state coasts on its prediction (advancing by one velocity step)
  and the covariance grows.
- **`β₀` behaves** — small when a measurement sits on the prediction, exactly `1`
  on an empty scan.

## Verification

- `cargo test -p scirust-signal` — **198 tests green** (+4).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The tracking layer now spans fixed-gain α–β, adaptive Kalman, the 1-D and
coordinated-turn IMMs, the polar EKF, the NIS-gated multi-target tracker, and
this clutter-robust PDAF — the full association/estimation toolkit of a modern
radar tracker. Together with the radar front-end (blocks 1–10) and the complete
EO/IR optronics chain (blocks 11–17, 23–25), the program is a physically-grounded
sensor-to-track suite across both modalities.
