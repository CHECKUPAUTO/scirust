# SciRust — Radar & Optronics, Block 23 (2026-07-12)

Follow-up deepening — a bridge between the two halves of the program. The radar
side detects targets in a range-Doppler map with CFAR and tracks them; the
optronics side models the EO/IR optics. This block adds the piece that connects
them: an **image-domain small-target CFAR detector** that turns a thermal frame
into target centroids ready for the same tracking chain.

## What shipped — `scirust-vision::detect`

- **`cfar_mask(image, guard, train, k)`** — the detection mask. For each pixel it
  estimates the local background from a **ring of training cells** around a
  **guard band** (so a bright target does not corrupt its own background
  estimate) and flags the pixel when it exceeds the local mean by `k` local
  standard deviations. This is the image-domain analogue of the radar
  `ca_cfar_2d`: because the threshold rides the *local* statistics, a target is
  found on a dim sky and on a bright pedestal alike, at a false-alarm rate set by
  `k` rather than by an absolute level.
- **`detect_targets(image, guard, train, k)`** / **`TargetDetection`** — groups
  the thresholded pixels into connected components (reusing
  `connected_components`) and reduces each to an **intensity-weighted centroid**
  (sub-pixel `(x, y)`, peak amplitude, pixel count) — the same detection shape
  the radar tracker consumes.

## The oracles

- **Point target on a flat background** — one detection at the injected location,
  peak amplitude recovered.
- **No detections on a uniform background** — zero local variance ⇒ nothing
  exceeds the mean.
- **Target on a bright pedestal** — the headline test: a target sitting on a
  bright *uniform* pedestal is still found while the pedestal itself raises no
  detections, proving the threshold tracks the local level.
- **Sub-pixel weighted centroid** — a two-pixel target with 30 vs 10 intensity
  lands at `x = 10.25`, pulled toward the brighter pixel.
- **Two separated targets** — resolved as two detections.
- **Higher threshold does not grow false alarms** — on a reproducible noise
  background, raising `k` never increases the false-alarm count.

## Verification

- `cargo test -p scirust-vision` — **47 tests green** (+6).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.
- `RUSTFLAGS="-D warnings" cargo check -p scirust-vision --all-targets --target
  aarch64-unknown-linux-gnu` — clean (cross-check merge gate).

## Where the program stands

With this block the optronics and tracking halves join up: an EO/IR frame →
CFAR small-target detection → intensity-weighted centroids → (via a
range/bearing or Cartesian mapping) the NIS-gated multi-target EKF tracker. The
full radar (1–10) + optronics (11–17) program plus the tracking deepenings
(ESPRIT 18, Kalman/IMM 19, coordinated-turn IMM 20, polar EKF 21, multi-target
tracker 22) and now the image-domain detector (23) forms an end-to-end
sensor-to-track chain across both the radar and EO/IR modalities.
