# SciRust — Radar & Optronics, Block 9 (2026-07-11)

Follow-up to block 8 (MUSIC). With the angle track complete, this block opens
the **detection → track** stage — the part that turns a processed range-Doppler
surface into a target list. This block ships the *detection* half: **2-D CFAR**
and **detection clustering**, exactly the OpenRadar reference flow (CFAR → peak
grouping / clustering).

## What shipped — `scirust-signal::radar::detect`

- **`ca_cfar_2d(power, train, guard, pfa)`** — cell-averaging CFAR over a
  range-Doppler **power** map. For each cell under test the noise level is the
  mean of the training cells in a square window of half-width `train + guard`,
  minus the inner `(2·guard+1)²` guard region that shields the target's own
  spread; a detection is flagged when the cell exceeds `α · mean(training)` with
  `α` from the existing `ca_cfar_alpha` over
  `N = (2(train+guard)+1)² − (2·guard+1)²` training cells — the same closed form
  as the 1-D detector, so `P_fa` is held exactly in homogeneous noise. Returns a
  boolean detection mask; edge cells (no full window) are never flagged.
- **`Detection`** / **`cluster_detections(mask, map)`** — 8-connected
  connected-component labelling of the mask, each blob collapsed to one
  `Detection` with an amplitude-weighted centroid (fractional range/Doppler bin),
  the peak cell magnitude, and the cell count. Sorted strongest-peak-first. These
  centroids are what the tracker (next block) associates across frames.

Both steps are dependency-free; the CFAR reuses `cfar::ca_cfar_alpha`.

## The oracles

- **2-D CFAR point target** — a strong cell on a flat floor is detected, and
  only it (exactly one detection).
- **2-D CFAR false-alarm rate** — over a 120×120 map of exponential noise, the
  empirical `P_fa` matches the design `pfa` within a few σ.
- **Two separated targets** — two amplitude blobs cluster into two detections
  with exact amplitude-weighted centroids, strongest peak first.
- **8-connectivity** — two cells touching only at a corner merge into one
  component.
- **End-to-end** — a flat floor with two well-separated strong cells →
  `ca_cfar_2d` mask → `cluster_detections` → two centroids at those cells.
- **Guards** — empty mask, shape mismatch, too-small / ragged map.

## Verification

- `cargo test -p scirust-signal` — **157 tests + 1 doctest green** (+6).
- `cargo clippy -p scirust-signal --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-signal -- --check` — clean.

## Where the radar track stands

- Single-channel pulse-Doppler chain: **complete** (blocks 1–4).
- Array / angle processing: **complete** — beamformer, MVDR/Capon, MUSIC
  (blocks 5, 6, 8).
- FMCW / mmWave (block 7).
- Detection: 2-D CFAR + clustering (**this block**).
- Remaining in the radar track: **tracking** — associate the clustered
  detections across frames and filter them (an α–β / Kalman constant-velocity
  track filter; can stay dependency-free, or reuse `scirust-estimation`). After
  that, the wider program's optronics / optical-imaging / optoelectronic-device
  pieces.
