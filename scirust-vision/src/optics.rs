//! Optical image quality and restoration — PSF, MTF, and deconvolution.
//!
//! An optronic imager (EO/IR camera, seeker, telescope) is characterised by its
//! **point-spread function** (PSF) — how a point source is smeared by
//! diffraction, aberrations and defocus — and, equivalently, by its
//! **modulation transfer function** (MTF), the contrast it passes as a function
//! of spatial frequency. The MTF is the headline resolution metric of a
//! precision optical system (its `MTF50` — the frequency at which contrast falls
//! to 50 % — is the number on the datasheet). Given a known PSF, an image blurred
//! by the optics can be partly **restored** by deconvolution.
//!
//! This module works on the crate's [`Image`](crate::Image) and reuses its
//! spatial [`convolve2d`](crate::convolve2d). The MTF is a direct DFT of the
//! line-spread function (no power-of-two constraint) and Richardson–Lucy
//! deconvolution is purely spatial (convolutions only); the alternative
//! **Wiener** deconvolution is frequency-domain, built on a separable 2-D FFT
//! (via `scirust-signal`, power-of-two dimensions). The diffraction-limited Airy
//! PSF reuses `scirust-special` for its Bessel function.

use crate::{Image, Kernel, convolve2d};
use scirust_signal::{Complex, fft, ifft};
use scirust_special::bessel_j;
use std::f64::consts::PI;

/// The first zero of `J₁`, where the Airy pattern has its first dark ring — the
/// Rayleigh resolution limit.
const AIRY_FIRST_ZERO: f64 = 3.831_705_970_207_512;

/// The axis along which a line-spread function runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Profile along x (integrating over y).
    X,
    /// Profile along y (integrating over x).
    Y,
}

/// A normalized 2-D Gaussian **point-spread function** (`Σ = 1`) of odd
/// `size × size`, with standard deviation `sigma` pixels. This is the standard
/// stand-in for a well-corrected optical blur; larger `sigma` ⇒ softer optics.
/// `size` is forced odd (so the peak sits on a pixel) and at least 1.
pub fn gaussian_psf(size: usize, sigma: f64) -> Image {
    let n = size.max(1) | 1;
    let half = (n / 2) as f64;
    let mut data = vec![0.0; n * n];
    let mut sum = 0.0;
    for y in 0..n
    {
        for x in 0..n
        {
            let dx = x as f64 - half;
            let dy = y as f64 - half;
            let v = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            data[y * n + x] = v;
            sum += v;
        }
    }
    for v in &mut data
    {
        *v /= sum;
    }
    Image::from_vec(n, n, data)
}

/// Blur an image with a (square) PSF — the forward optical model. The PSF should
/// sum to 1 (as [`gaussian_psf`] does) so the mean brightness is preserved.
pub fn apply_psf(image: &Image, psf: &Image) -> Image {
    let kernel = Kernel::from_vec(psf.width, psf.data.clone());
    convolve2d(image, &kernel)
}

/// The **line-spread function** (LSF): the PSF integrated along one axis, giving
/// the 1-D response used to derive the MTF.
#[allow(clippy::needless_range_loop)] // projecting a 2-D PSF onto one axis
pub fn line_spread(psf: &Image, axis: Axis) -> Vec<f64> {
    match axis
    {
        Axis::X =>
        {
            let mut lsf = vec![0.0; psf.width];
            for y in 0..psf.height
            {
                for x in 0..psf.width
                {
                    lsf[x] += psf.get(x, y);
                }
            }
            lsf
        },
        Axis::Y =>
        {
            let mut lsf = vec![0.0; psf.height];
            for y in 0..psf.height
            {
                for x in 0..psf.width
                {
                    lsf[y] += psf.get(x, y);
                }
            }
            lsf
        },
    }
}

/// The **modulation transfer function** from a line-spread function: the
/// normalized magnitude of its DFT (`MTF[0] = 1`), returned from DC up to Nyquist
/// (`lsf.len()/2 + 1` samples). Bin `k` corresponds to spatial frequency
/// `f = k / lsf.len()` in cycles per pixel. Empty for an empty or zero-sum LSF.
pub fn mtf(lsf: &[f64]) -> Vec<f64> {
    let n = lsf.len();
    if n == 0
    {
        return Vec::new();
    }
    let dc: f64 = lsf.iter().sum();
    if dc.abs() < f64::EPSILON
    {
        return Vec::new();
    }
    (0..=n / 2)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (nn, &v) in lsf.iter().enumerate()
            {
                let phase = -2.0 * PI * k as f64 * nn as f64 / n as f64;
                re += v * phase.cos();
                im += v * phase.sin();
            }
            (re * re + im * im).sqrt() / dc.abs()
        })
        .collect()
}

/// The **MTF50** resolution metric: the spatial frequency (cycles per pixel) at
/// which the `mtf` first falls to 0.5, linearly interpolated between samples.
/// `n_samples` is the length of the line-spread function the MTF came from, so
/// bin `k` maps to its exact DFT frequency `k / n_samples`. Returns the highest
/// represented frequency if the MTF never drops to 0.5.
pub fn mtf50(mtf: &[f64], n_samples: usize) -> f64 {
    if mtf.len() < 2 || n_samples == 0
    {
        return 0.0;
    }
    let bin_to_freq = |k: f64| k / n_samples as f64;
    for k in 1..mtf.len()
    {
        if mtf[k] <= 0.5
        {
            let (a, b) = (mtf[k - 1], mtf[k]);
            let frac = if (a - b).abs() > f64::EPSILON
            {
                (a - 0.5) / (a - b)
            }
            else
            {
                0.0
            };
            return bin_to_freq(k as f64 - 1.0 + frac);
        }
    }
    bin_to_freq((mtf.len() - 1) as f64)
}

/// 180° rotation of a square PSF (its adjoint for correlation-based blurring).
fn flip_psf(psf: &Image) -> Image {
    let data: Vec<f64> = psf.data.iter().rev().copied().collect();
    Image::from_vec(psf.width, psf.height, data)
}

/// **Richardson–Lucy deconvolution**: iteratively restore `blurred`, given the
/// known `psf`, by the multiplicative update
/// `x ← x · (blurred / (x ⊛ psf)) ⊛ psfᵀ`. Purely spatial (convolutions only),
/// it conserves total intensity and stays non-negative — the classic restoration
/// for photon-limited optronic imagery. `iterations` update steps.
pub fn richardson_lucy(blurred: &Image, psf: &Image, iterations: usize) -> Image {
    let flipped = flip_psf(psf);
    let mut estimate = blurred.clone();
    for _ in 0..iterations
    {
        let reblur = apply_psf(&estimate, psf);
        let ratio: Vec<f64> = blurred
            .data
            .iter()
            .zip(reblur.data.iter())
            .map(|(&b, &r)| if r > 1e-12 { b / r } else { 0.0 })
            .collect();
        let ratio_img = Image::from_vec(blurred.width, blurred.height, ratio);
        let correction = apply_psf(&ratio_img, &flipped);
        for (e, c) in estimate.data.iter_mut().zip(correction.data.iter())
        {
            *e *= c;
        }
    }
    estimate
}

/// The separable 2-D FFT of a `w × h` complex buffer (row-major): a 1-D FFT of
/// every row followed by a 1-D FFT of every column. `w` and `h` must be powers
/// of two.
fn fft2(data: &mut [Complex], w: usize, h: usize) {
    for row in data.chunks_mut(w)
    {
        fft(row);
    }
    let mut col = vec![Complex::zero(); h];
    for c in 0..w
    {
        for (y, slot) in col.iter_mut().enumerate()
        {
            *slot = data[y * w + c];
        }
        fft(&mut col);
        for (y, &v) in col.iter().enumerate()
        {
            data[y * w + c] = v;
        }
    }
}

/// The separable 2-D inverse FFT, the exact inverse of [`fft2`].
fn ifft2(data: &mut [Complex], w: usize, h: usize) {
    for row in data.chunks_mut(w)
    {
        ifft(row);
    }
    let mut col = vec![Complex::zero(); h];
    for c in 0..w
    {
        for (y, slot) in col.iter_mut().enumerate()
        {
            *slot = data[y * w + c];
        }
        ifft(&mut col);
        for (y, &v) in col.iter().enumerate()
        {
            data[y * w + c] = v;
        }
    }
}

/// **Wiener deconvolution**: frequency-domain image restoration given a known
/// (square, odd) `psf`. It computes
/// `F̂ = 𝔉⁻¹[ conj(H)/(|H|² + nsr) · G ]`, where `G` and `H` are the DFTs of the
/// `blurred` image and the (mean-preserving, origin-centred) PSF, and `nsr` is
/// the noise-to-signal power ratio that regularises the inverse — larger `nsr`
/// suppresses noise amplification at the cost of sharpness; `nsr → 0` is the pure
/// inverse filter. Complements the spatial [`richardson_lucy`].
///
/// The image dimensions must both be powers of two and the PSF must be a square
/// no larger than the image; otherwise an empty [`Image`] is returned.
pub fn wiener_deconvolution(blurred: &Image, psf: &Image, nsr: f64) -> Image {
    let (w, h) = (blurred.width, blurred.height);
    let ps = psf.width;
    if w == 0
        || h == 0
        || w & (w - 1) != 0
        || h & (h - 1) != 0
        || psf.height != ps
        || ps == 0
        || ps > w
        || ps > h
    {
        return Image::new(0, 0);
    }
    // Embed the mean-preserving PSF into a w×h buffer with its centre at the
    // origin (circular wraparound), so the convolution theorem holds shift-free.
    let psum: f64 = psf.data.iter().sum();
    let half = ps / 2;
    let mut hbuf = vec![Complex::zero(); w * h];
    for py in 0..ps
    {
        for px in 0..ps
        {
            let x = (px + w - half) % w;
            let y = (py + h - half) % h;
            hbuf[y * w + x] += Complex::new(psf.get(px, py) / psum, 0.0);
        }
    }
    let mut gbuf: Vec<Complex> = blurred.data.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft2(&mut hbuf, w, h);
    fft2(&mut gbuf, w, h);
    for (g, hh) in gbuf.iter_mut().zip(hbuf.iter())
    {
        let denom = hh.mag_sq() + nsr;
        let numer = hh.conj() * *g; // conj(H)·G
        *g = Complex::new(numer.re / denom, numer.im / denom);
    }
    ifft2(&mut gbuf, w, h);
    Image::from_vec(w, h, gbuf.iter().map(|c| c.re).collect())
}

/// A normalized (`Σ = 1`) **Airy point-spread function** of odd `size × size` —
/// the diffraction-limited response of a circular aperture — with its first dark
/// ring `first_null` pixels from the centre. The intensity at radius `r` is
/// `[2·J₁(v)/v]²` with `v = j₁,₁·r/first_null` (and `1` at the centre). Unlike a
/// Gaussian blur this carries the characteristic ringing, and its central lobe
/// sets the optical resolution.
pub fn airy_psf(size: usize, first_null: f64) -> Image {
    let n = size.max(1) | 1;
    let half = (n / 2) as f64;
    let mut data = vec![0.0; n * n];
    let mut sum = 0.0;
    for y in 0..n
    {
        for x in 0..n
        {
            let dx = x as f64 - half;
            let dy = y as f64 - half;
            let r = (dx * dx + dy * dy).sqrt();
            let v = AIRY_FIRST_ZERO * r / first_null;
            // 2·J₁(v)/v → 1 as v → 0 (the central peak).
            let amp = if v.abs() < 1e-12
            {
                1.0
            }
            else
            {
                2.0 * bessel_j(1, v) / v
            };
            let intensity = amp * amp;
            data[y * n + x] = intensity;
            sum += intensity;
        }
    }
    for val in &mut data
    {
        *val /= sum;
    }
    Image::from_vec(n, n, data)
}

/// The **Rayleigh angular resolution** `θ = 1.22·λ/D` (radians): the smallest
/// angular separation two point sources of wavelength `wavelength` can have and
/// still be resolved through a circular aperture of diameter `aperture`.
pub fn rayleigh_resolution(wavelength: f64, aperture: f64) -> f64 {
    1.22 * wavelength / aperture
}

/// The **Airy first-null radius in pixels** on a focal-plane array:
/// `1.22·λ·f/(D·pixel_pitch)` — the radius of the diffraction central lobe, and
/// the argument to [`airy_psf`]. `focal_length` and `pixel_pitch` share a length
/// unit.
pub fn airy_first_null(wavelength: f64, aperture: f64, focal_length: f64, pixel_pitch: f64) -> f64 {
    1.22 * wavelength * focal_length / (aperture * pixel_pitch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_psf_is_normalized_symmetric_and_peaked() {
        let psf = gaussian_psf(9, 1.5);
        assert_eq!((psf.width, psf.height), (9, 9));
        // Sums to one (mean-preserving blur).
        assert!((psf.data.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // Peak at the centre, symmetric.
        let c = psf.get(4, 4);
        assert!(psf.data.iter().all(|&v| v <= c + 1e-15));
        assert!((psf.get(3, 4) - psf.get(5, 4)).abs() < 1e-15);
        assert!((psf.get(4, 3) - psf.get(4, 5)).abs() < 1e-15);
        // Even size is bumped to odd.
        assert_eq!(gaussian_psf(8, 1.0).width, 9);
    }

    #[test]
    fn mtf_of_a_gaussian_matches_the_closed_form() {
        // A Gaussian PSF of width σ has the analytic MTF exp(−2π²σ²f²).
        let sigma = 2.0;
        let psf = gaussian_psf(31, sigma);
        let lsf = line_spread(&psf, Axis::X);
        let m = mtf(&lsf);
        let n = lsf.len();
        assert!((m[0] - 1.0).abs() < 1e-12, "MTF at DC must be 1");
        for (k, &mk) in m.iter().enumerate().take(6).skip(1)
        {
            let f = k as f64 / n as f64;
            let expected = (-2.0 * PI * PI * sigma * sigma * f * f).exp();
            assert!((mk - expected).abs() < 1e-3, "MTF[{k}] {mk} vs {expected}");
        }
        // Monotonic roll-off from DC.
        for w in m.windows(2)
        {
            assert!(w[1] <= w[0] + 1e-12);
        }
    }

    #[test]
    fn mtf50_matches_the_gaussian_closed_form() {
        // exp(−2π²σ²f²) = 0.5 ⇒ f = sqrt(ln2 / 2) / (π σ).
        let sigma = 2.0;
        let psf = gaussian_psf(31, sigma);
        let lsf = line_spread(&psf, Axis::X);
        let m = mtf(&lsf);
        let f50 = mtf50(&m, lsf.len());
        let expected = (0.5_f64.ln().abs() / 2.0).sqrt() / (PI * sigma);
        assert!((f50 - expected).abs() < 3e-3, "MTF50 {f50} vs {expected}");
    }

    #[test]
    fn richardson_lucy_is_identity_for_a_delta_psf() {
        // A 1×1 PSF of weight 1 is a perfect (aberration-free) optic: RL leaves
        // the image unchanged.
        let img = Image::from_vec(3, 3, vec![0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 5.0, 0.0]);
        let delta = gaussian_psf(1, 1.0);
        assert!((delta.data[0] - 1.0).abs() < 1e-12);
        let restored = richardson_lucy(&img, &delta, 10);
        for (a, b) in restored.data.iter().zip(img.data.iter())
        {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn richardson_lucy_sharpens_a_blurred_point() {
        // A single bright pixel, blurred by the optics, is re-concentrated by RL:
        // its central value rises back toward the source and stays the peak.
        let (w, h) = (15usize, 15usize);
        let mut point = Image::new(w, h);
        point.set(7, 7, 1.0);
        let psf = gaussian_psf(7, 1.2);
        let blurred = apply_psf(&point, &psf);
        let restored = richardson_lucy(&blurred, &psf, 30);
        // The peak stays at the true location.
        let peak_idx = restored
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak_idx, 7 * w + 7);
        // And it is sharper than the blurred image at the centre.
        assert!(restored.get(7, 7) > blurred.get(7, 7) + 1e-6);
        // RL roughly conserves total flux (interior source, little edge leak).
        let (sb, sr): (f64, f64) = (blurred.data.iter().sum(), restored.data.iter().sum());
        assert!((sr - sb).abs() < 1e-2, "flux {sr} vs {sb}");
    }

    #[test]
    fn line_spread_integrates_the_psf() {
        let psf = gaussian_psf(5, 1.0);
        let lsf = line_spread(&psf, Axis::X);
        assert_eq!(lsf.len(), 5);
        // The LSF sums to the PSF total (1) and is symmetric about the centre.
        assert!((lsf.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((lsf[1] - lsf[3]).abs() < 1e-15);
        assert!(lsf[2] >= lsf[1]);
    }

    #[test]
    fn mtf_guards_degenerate_input() {
        assert!(mtf(&[]).is_empty());
        assert!(mtf(&[0.0, 0.0, 0.0]).is_empty());
        assert_eq!(mtf50(&[1.0], 1), 0.0);
    }

    #[test]
    fn airy_psf_is_normalized_symmetric_and_peaked() {
        let psf = airy_psf(21, 5.0);
        assert_eq!((psf.width, psf.height), (21, 21));
        assert!((psf.data.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // Bright central peak, rotationally symmetric.
        let c = psf.get(10, 10);
        assert!(psf.data.iter().all(|&v| v <= c + 1e-15));
        assert!((psf.get(13, 10) - psf.get(10, 13)).abs() < 1e-15);
        assert!((psf.get(7, 10) - psf.get(13, 10)).abs() < 1e-15);
        // Even size is bumped to odd.
        assert_eq!(airy_psf(20, 5.0).width, 21);
    }

    #[test]
    fn airy_first_dark_ring_falls_at_the_null_radius() {
        // With the first null placed exactly at 5 px, the pixel 5 px off-centre
        // sits on the first zero of J₁ — the dark ring — so its intensity is ~0.
        let psf = airy_psf(21, 5.0);
        let ring = psf.get(15, 10); // r = 5 from centre (10,10)
        assert!(ring < psf.get(10, 10) * 1e-4, "first ring not dark: {ring}");
        // Just inside the ring is brighter than on it (the central lobe).
        assert!(psf.get(13, 10) > ring);
    }

    #[test]
    fn rayleigh_and_airy_null_match_closed_forms() {
        // θ = 1.22·λ/D.
        assert!((rayleigh_resolution(500e-9, 0.1) - 1.22 * 500e-9 / 0.1).abs() < 1e-20);
        // First-null radius in pixels = 1.22·λ·f/(D·pitch).
        let (lambda, d, f, pitch) = (550e-9, 0.05, 0.2, 5e-6);
        let expected = 1.22 * lambda * f / (d * pitch);
        assert!((airy_first_null(lambda, d, f, pitch) - expected).abs() < 1e-12);
    }

    /// Mean-preserving **circular** convolution of an image with a (square, odd)
    /// PSF — the exact forward model Wiener deconvolution inverts. Convolution
    /// offset `px − half`, wrapped modulo the image dimensions.
    fn circular_convolve(img: &Image, psf: &Image) -> Image {
        let (w, h) = (img.width, img.height);
        let psum: f64 = psf.data.iter().sum();
        let half = (psf.width / 2) as isize;
        let mut out = Image::new(w, h);
        for y in 0..h
        {
            for x in 0..w
            {
                let mut acc = 0.0;
                for py in 0..psf.height
                {
                    for px in 0..psf.width
                    {
                        let sx =
                            (x as isize - (px as isize - half)).rem_euclid(w as isize) as usize;
                        let sy =
                            (y as isize - (py as isize - half)).rem_euclid(h as isize) as usize;
                        acc += img.get(sx, sy) * psf.get(px, py) / psum;
                    }
                }
                out.set(x, y, acc);
            }
        }
        out
    }

    #[test]
    fn wiener_inverts_a_circular_blur() {
        // A structured 16×16 scene, blurred by a known PSF, is recovered by
        // Wiener deconvolution with vanishing regularisation.
        let (w, h) = (16usize, 16usize);
        let mut scene = Image::new(w, h);
        for y in 0..h
        {
            for x in 0..w
            {
                scene.set(x, y, ((x * 7 + y * 13) % 11) as f64 + 1.0);
            }
        }
        let psf = gaussian_psf(5, 1.1);
        let blurred = circular_convolve(&scene, &psf);
        let restored = wiener_deconvolution(&blurred, &psf, 1e-12);
        for (r, s) in restored.data.iter().zip(scene.data.iter())
        {
            assert!((r - s).abs() < 1e-5, "restored {r} vs {s}");
        }
    }

    #[test]
    fn wiener_with_a_delta_psf_is_the_identity() {
        let (w, h) = (8usize, 8usize);
        let mut img = Image::new(w, h);
        for (i, v) in img.data.iter_mut().enumerate()
        {
            *v = (i % 5) as f64;
        }
        let delta = gaussian_psf(1, 1.0); // a single unit sample
        let restored = wiener_deconvolution(&img, &delta, 1e-12);
        for (r, s) in restored.data.iter().zip(img.data.iter())
        {
            assert!((r - s).abs() < 1e-9);
        }
    }

    #[test]
    fn wiener_guards_non_power_of_two_and_oversized_psf() {
        let img = Image::new(6, 8); // width not a power of two
        assert!(
            wiener_deconvolution(&img, &gaussian_psf(3, 1.0), 0.1)
                .data
                .is_empty()
        );
        // A PSF larger than the image.
        let small = Image::new(4, 4);
        assert!(
            wiener_deconvolution(&small, &gaussian_psf(5, 1.0), 0.1)
                .data
                .is_empty()
        );
    }
}
