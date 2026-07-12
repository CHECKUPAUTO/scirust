# SciRust — Radar & Optronics, Block 22 (2026-07-12)

Follow-up deepening — the capstone that ties the tracking blocks together. Block
21 gave a single-target extended Kalman filter over polar measurements; block 10
gave an α–β *multi*-target tracker over Cartesian centroids. This block combines
their strengths: a **multi-target tracker of per-target EKFs**, associating
polar returns by a **statistical (NIS) validation gate**.

## What shipped — `scirust-signal::radar::mtt`

- **`RadarMultiTracker`** / **`RadarTrack`** — each track is a full
  [`RadarEkf`](../scirust-signal/src/radar/ekf.rs) fed raw `(range, bearing)`
  returns. Each `step`:
  1. predicts every track (which is also the coast for unmatched tracks);
  2. gates every (track, measurement) pair by its **normalised innovation
     squared** `yᵀ·S⁻¹·y` against a χ²-with-2-d.o.f. threshold — a Mahalanobis
     distance that tightens or widens with each track's *own* covariance instead
     of a single fixed radius;
  3. greedily associates nearest (smallest NIS) first, each track and
     measurement at most once;
  4. updates matched tracks, coasts the rest;
  5. spawns a track (converting the polar measurement to a Cartesian seed) for
     every unassociated measurement;
  6. drops tracks that have coasted past `max_misses`.
- **`RadarEkf::nis(range, bearing, …)`** — a new read-only method returning the
  gating statistic `yᵀ·S⁻¹·y` without mutating the filter.

## The oracles

- **Follows a single target** — a straight-line target tracked from polar
  returns to sub-half-unit position error, stable id.
- **Keeps two separated targets apart** — two well-separated tracks with
  distinct, stable ids.
- **NIS gate rejects clutter** — the headline test: with a track established, a
  frame carrying only a far-off clutter return fails the gate, so the real track
  coasts *untainted* (its position advances along its heading, not toward the
  clutter) and the clutter spawns its own track.
- **Spawns then drops a lost track** — birth on first detection, death after
  coasting past `max_misses`.
- **NIS is small on-target and large off-target** — a direct check of the
  gating statistic (on-target < 1, off-target > 100× larger).
- **Empty frames are inert.**

## Verification

- `cargo test -p scirust-signal` — **194 tests green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-signal --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

The radar (1–10) + optronics (11–17) program plus the tracking deepenings —
ESPRIT (18), Kalman/IMM (19), coordinated-turn IMM (20), polar EKF (21), and now
the NIS-gated multi-target EKF tracker (22) — forms a complete
measurement→association→state→manoeuvre tracking stack: raw polar detections in,
maintained multi-target Cartesian tracks out, with statistical gating,
birth/death, and adaptive/manoeuvre motion models available throughout.
