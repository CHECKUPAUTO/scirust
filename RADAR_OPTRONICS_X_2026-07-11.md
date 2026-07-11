# SciRust — Radar & Optronics, Block 10 (2026-07-11)

Follow-up to block 9 (2-D CFAR + detection clustering). This block ships the
**tracking** stage — the temporal layer that turns per-frame detections into
persistent tracks — which **completes the detection → track chain** and, with
it, the end-to-end radar signal-processing pipeline (waveform → compression →
Doppler / FMCW → CFAR → cluster → track).

## What shipped — `scirust-signal::radar::track`

- **`AlphaBeta`** — a scalar α–β track filter for a constant-velocity state
  `(x, v)`: `predict` (`x + v·dt`), `update` (predict then correct by
  `α·residual` / `(β/dt)·residual`), and `coast` (advance with no measurement).
  It is a fixed-gain steady-state form of the Kalman filter — cheap and stable,
  and unbiased with zero steady-state lag on a constant-velocity trajectory.
- **`critically_damped_gains(θ)`** — the standard critically-damped design
  `α = 1 − θ²`, `β = (1 − θ)²` for a discounting factor `θ ∈ (0, 1)`.
- **`Track`** — a 2-D target track (one `AlphaBeta` per coordinate, range and
  Doppler bin) consuming block 9's `Detection`, with a hit/miss lifecycle and the
  last associated amplitude.
- **`MultiTracker`** — a nearest-neighbour multi-target tracker: each `step`
  predicts every track, greedily associates detections to the nearest predicted
  track within a distance gate, updates matched tracks, coasts unmatched ones,
  spawns tracks for unmatched detections, and drops tracks that coast past
  `max_misses`.

Dependency-free; consumes the `Detection` centroids of `radar::detect`.

## The oracles

- **Critically-damped gains** — match the closed form (θ = 0.5 → α = 0.75,
  β = 0.25) and stay finite at the clamped extremes.
- **Zero-lag ramp tracking** — on a noise-free constant-velocity ramp the α–β
  filter converges to the truth position and the exact velocity.
- **Coasting** — a coasted frame advances position by exactly one velocity step.
- **Single-target tracking** — the multi-tracker follows a moving target as one
  track whose estimated velocity matches truth, with a stable id.
- **Two separated targets** — a tight gate keeps two far-apart targets as two
  distinct, stably-identified tracks.
- **Track birth / death** — a detection spawns a track; sustained misses past
  `max_misses` drop it.

## Verification

- `cargo test -p scirust-signal` — **163 tests + 1 doctest green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Where the radar track stands

The radar signal-processing pipeline is now **end-to-end complete**:

- Single-channel pulse-Doppler chain (blocks 1–4).
- Array / angle processing — beamformer, MVDR/Capon, MUSIC (5, 6, 8).
- FMCW / mmWave (7).
- 2-D detection — CFAR + clustering (9).
- **Tracking — α–β filter + multi-target tracker (this block).**

Optional radar extras remain available (ESPRIT DOA on the block-8 eigensolver;
an IMM/Kalman upgrade to the tracker reusing `scirust-estimation`), but the core
"radar you could sell" chain is in place. The program now pivots to the wider
optronics goal: precision optics (Gaussian beams, ABCD ray matrices), optical
image processing (PSF / MTF / deconvolution), and optoelectronic device
dynamics (semiconductor-laser rate equations as a `scirust-sim` system).
