# SciRust — Radar & Optronics, Block 14 (2026-07-11)

Follow-up deepening after the core radar (1–10) and optronics (11–13) program
merged. This block adds the **diffraction-limited Airy PSF** to
`scirust-vision::optics`, completing the image-side PSF pair: the Gaussian PSF
(block 11) is the aberration stand-in, the Airy PSF is the *ideal* — the response
of a perfect circular aperture, whose central lobe sets the ultimate optical
resolution.

## What shipped — `scirust-vision::optics`

- **`airy_psf(size, first_null)`** — a normalized (`Σ = 1`) Airy point-spread
  function: intensity `[2·J₁(v)/v]²` with `v = j₁,₁·r/first_null`, so the first
  dark ring sits `first_null` pixels from the centre. Reuses
  `scirust_special::bessel_j` for `J₁` (a new, lightweight workspace dependency).
- **`rayleigh_resolution(λ, D)`** — the Rayleigh angular resolution `θ = 1.22·λ/D`
  (radians), the smallest resolvable separation through a circular aperture.
- **`airy_first_null(λ, D, f, pixel_pitch)`** — the first-null radius in detector
  pixels, `1.22·λ·f/(D·pixel_pitch)` — the physical-to-sampling bridge and the
  argument to `airy_psf`.

## The oracles

- **PSF shape** — normalized to 1, rotationally symmetric, bright central peak;
  even sizes bumped to odd.
- **First dark ring** — with the first null placed at 5 px, the pixel exactly
  5 px off-centre lands on the first zero of `J₁` and is dark (< 1e-4 of the
  peak), while a pixel just inside is brighter.
- **Closed forms** — `rayleigh_resolution` matches `1.22·λ/D`;
  `airy_first_null` matches `1.22·λ·f/(D·pixel_pitch)`.

## Verification

- `cargo test -p scirust-vision` — **38 tests green** (+3).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.

## Where the program stands

The full radar + optronics program is merged; this is an optional deepening on
top. Remaining optional pieces: **Wiener** frequency-domain deconvolution (a 2-D
FFT), **ESPRIT** rotational-invariance DOA (reusing `music::hermitian_eig`), an
**IMM/Kalman** tracker upgrade, and **photodetector/LED** optoelectronic device
models.
