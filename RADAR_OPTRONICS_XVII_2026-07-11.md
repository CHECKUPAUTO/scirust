# SciRust — Radar & Optronics, Block 17 (2026-07-11)

Follow-up deepening. Block 11 added spatial Richardson–Lucy deconvolution; this
block adds its frequency-domain counterpart — **Wiener deconvolution** — so the
optics module now offers both families of EO/IR image restoration.

## What shipped — `scirust-vision::optics`

- **`wiener_deconvolution(blurred, psf, nsr)`** — the Wiener filter
  `F̂ = 𝔉⁻¹[ conj(H)/(|H|² + nsr)·G ]`, where `G` and `H` are the DFTs of the
  blurred image and the (mean-preserving, origin-centred) PSF, and `nsr` is the
  noise-to-signal power ratio that regularises the inverse — larger `nsr`
  suppresses noise amplification, `nsr → 0` is the pure inverse filter.

Built on a small **separable 2-D FFT** (`fft2` / `ifft2`, private): a 1-D FFT of
every row then of every column, using `scirust-signal`'s power-of-two FFT/IFFT.
The PSF is embedded into an image-sized buffer with its centre at the origin and
circular wraparound, so the convolution theorem holds without introducing a
shift.

## The oracles

- **Exact inverse of a circular blur** — the headline test: a structured 16×16
  scene is blurred by a known Gaussian PSF via mean-preserving *circular*
  convolution, and Wiener deconvolution with vanishing regularisation recovers
  the original to 1e-5. This cross-checks the PSF embedding and the 2-D FFT
  against an independent spatial forward model.
- **Delta PSF is the identity** — an aberration-free optic leaves the image
  unchanged.
- **Guards** — non-power-of-two dimensions or a PSF larger than the image return
  an empty image.

## Verification

- `cargo test -p scirust-vision` — **41 tests green** (+3).
- `cargo clippy -p scirust-vision --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-vision -- --check` — clean.

## Where the program stands

The full radar (1–10) + optronics (11–16) program is merged; this is another
optional deepening. The optics module now covers PSF (Gaussian + Airy), MTF /
MTF50, and **both** spatial (Richardson–Lucy) and frequency-domain (Wiener)
restoration. Remaining optional pieces: ESPRIT DOA (reusing
`music::hermitian_eig`) and an IMM/Kalman tracker upgrade on the signal side.
