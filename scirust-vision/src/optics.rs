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
//! This module works on the crate's [`Image`](crate::Image), reuses its spatial
//! [`convolve2d`](crate::convolve2d), and is dependency-free: the MTF is a direct
//! DFT of the line-spread function (no power-of-two constraint) and
//! Richardson–Lucy deconvolution is purely spatial (convolutions only).

use crate::{Image, Kernel, convolve2d};
use std::f64::consts::PI;

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
}
