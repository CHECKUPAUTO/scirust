# SciRust — Radar & Optronics, Block 11 (2026-07-11)

The radar signal-processing pipeline is complete (blocks 1–10). This block opens
the second half of the program the user asked for — **optronics / precision
optics / image processing** ("optoélectronique optique de précision traitement de
l'image") — starting with the image-quality and restoration core of an EO/IR
imager, added to the existing `scirust-vision` crate.

## What shipped — `scirust-vision::optics`

- **`gaussian_psf(size, sigma)`** — a normalized (`Σ = 1`), odd-sized Gaussian
  **point-spread function**, the standard stand-in for a well-corrected optical
  blur.
- **`apply_psf(image, psf)`** — the forward optical model: blur an image with a
  PSF (reuses the crate's spatial `convolve2d`).
- **`line_spread(psf, axis)`** / **`mtf(lsf)`** — the line-spread function, then
  the **modulation transfer function** as the normalized magnitude of its DFT
  (`MTF[0] = 1`), with frequencies in cycles per pixel. A direct DFT — no
  power-of-two constraint.
- **`mtf50(mtf, n_samples)`** — the **MTF50** resolution metric: the frequency at
  which contrast falls to 50 %, the headline number on an optics datasheet.
- **`richardson_lucy(blurred, psf, iterations)`** — **Richardson–Lucy
  deconvolution**: iterative, purely spatial (convolutions only) image
  restoration that conserves total flux and stays non-negative — the classic
  method for photon-limited optronic imagery.

Dependency-free; built on the crate's `Image` and `convolve2d`.

## The oracles

- **PSF** — normalized to 1, symmetric, peaked at the centre; even sizes bumped
  to odd.
- **MTF closed form** — a Gaussian PSF of width σ has MTF `exp(−2π²σ²f²)`; the
  computed MTF matches it at low frequency and rolls off monotonically from DC.
- **MTF50 closed form** — for a Gaussian, MTF = 0.5 at `f = √(ln2/2)/(πσ)`, which
  the interpolated `mtf50` recovers.
- **Richardson–Lucy identity** — a delta (1×1) PSF leaves the image unchanged.
- **Richardson–Lucy sharpening** — a blurred point source is re-concentrated: the
  central value rises back above the blurred value, the peak stays at the true
  location, and total flux is (approximately) conserved.
- **LSF** — integrates the PSF (sums to 1), symmetric.
- **Guards** — empty / zero-sum LSF, degenerate MTF.

## Verification

- `cargo test -p scirust-vision` — **29 tests green** (+7).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.

## Where the program stands

- **Radar signal-processing pipeline: complete** (blocks 1–10) — waveform →
  compression → Doppler/FMCW → CFAR → cluster → track, plus full array/angle
  processing.
- **Optronics / precision optics / image processing: started** (this block) —
  PSF / MTF / MTF50 / Richardson–Lucy in `scirust-vision`.
- Next optronics blocks: diffraction-limited **Airy PSF** and aberration
  transfer (would reuse `scirust-special` Bessel functions); **Wiener
  deconvolution** (frequency-domain restoration with a 2-D FFT); **Gaussian-beam
  propagation and ABCD ray-transfer matrices** for optical-train design; and
  **optoelectronic device dynamics** (semiconductor-laser rate equations as a
  `scirust-sim` system).
