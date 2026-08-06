# SciRust — Radar & Optronics, Block 12 (2026-07-11)

Second optronics block (folded into the same PR as block 11's `optics`, since
both extend `scirust-vision`'s optics offering). Where block 11 covered the
*image* side (PSF / MTF / restoration), this covers the *optical-train* side —
the ray and beam optics you use to design the collimator, beam expander, or
cavity that forms the image.

## What shipped — `scirust-vision::beams`

- **`RayMatrix`** — a 2×2 **ABCD ray-transfer matrix** `[[a,b],[c,d]]` acting on
  a paraxial ray `(height y, slope θ)`. Constructors `identity`, `free_space`,
  `thin_lens`, `curved_mirror`, `flat_interface`; composition `then` (the product
  `next · self`, "pass through self then next"); `determinant` (`= n_in/n_out`);
  and `apply`.
- **Gaussian-beam geometry** — `rayleigh_range` (`z_R = π·w0²/λ`), `beam_radius`
  (`w(z) = w0·√(1+(z/z_R)²)`), `radius_of_curvature`, `divergence` (`λ/(π·w0)`),
  `gouy_phase`.
- **Complex q-parameter** — `q_at_waist`, `propagate_q` (`q' = (A·q+B)/(C·q+D)`
  through any `RayMatrix`), and the readouts `beam_radius_from_q` /
  `radius_from_q`. Reuses `scirust_signal::Complex`.

## The oracles

- **Unit determinant** — free space, thin lens, curved mirror, identity all have
  `det = 1`; a flat interface has `det = n1/n2`.
- **Collimated ray focuses at f** — a ray parallel to the axis crosses the axis
  one focal length behind a thin lens.
- **Imaging condition** — `free(so) → lens(f) → free(si)` with `1/so+1/si=1/f`
  zeroes the `B` element and gives magnification `A = −si/so`.
- **Beam geometry closed forms** — radius `w0` at the waist, flat wavefront there,
  `√2` growth at one Rayleigh range, far-field divergence, Gouy phase `π/4` at
  `z_R`.
- **q-parameter consistency** — propagating the waist `q` through free space
  reproduces `w(z)` and `R(z)`; a lens forms a new Gaussian waist at the predicted
  plane `s' = f·z_R²/(f²+z_R²)` (flat wavefront there), which tends to `f` only as
  `z_R ≫ f` — the Gaussian correction to the geometric result.

## Verification

- `cargo test -p scirust-vision` — **37 tests green** (+8).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.

A physics bug was caught by the oracle during development: the first version
asserted a collimated Gaussian beam refocuses exactly at `f`, but for a Gaussian
beam the new waist sits at `s' = f·z_R²/(f²+z_R²)`; the oracle now uses the exact
plane.

## Where the program stands

- Radar signal-processing pipeline: complete (blocks 1–10).
- Optronics / precision optics: **image quality + restoration** (block 11) and
  **ray/beam optical-train design** (this block), both in `scirust-vision`.
- Remaining optronics: diffraction-limited **Airy PSF** (needs Bessel `J1`,
  reusing `scirust-special`); frequency-domain **Wiener deconvolution** (2-D FFT);
  and **optoelectronic device dynamics** — semiconductor-laser rate equations as
  a `scirust-sim` system (threshold current, relaxation-oscillation frequency).
