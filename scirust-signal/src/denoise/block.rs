//! NeighBlock wavelet **block** thresholding (Cai-Silverman 2001).
//!
//! Per-coefficient thresholding rules ([`super::transform::wavelet_denoise_with`]
//! and friends) judge every wavelet coefficient in isolation, so a signal whose
//! energy is *grouped* — a sustained tone filling its resonant band, an oscillatory
//! transient — has each of its many moderate coefficients shrunk or killed
//! individually even though together they are unmistakably signal. Block
//! thresholding fixes that by making a **joint** keep/shrink decision per block of
//! neighbouring coefficients: the decision statistic is the energy of the block
//! *plus a margin of neighbours on each side* (hence "NeighBlock"), so grouped
//! energy defends itself while isolated noise blips still shrink to zero.
//!
//! Reference: T. T. Cai, B. W. Silverman, *"Incorporating Information on
//! Neighboring Coefficients into Wavelet Estimation"*, Sankhyā Ser. B 63, 127-148
//! (2001). The James-Stein-style shrinkage factor and the constant λ* below are
//! theirs.

use super::mad;
use super::transform::{Wavelet, dwt_forward, dwt_inverse};

/// The NeighBlock shrinkage constant λ* ≈ 4.50524 — the root of `λ − ln λ = 3`
/// (Cai-Silverman 2001). It is the smallest λ for which the oracle inequality
/// behind the block estimator holds, making pure-noise blocks vanish with high
/// probability without a tuning knob.
const LAMBDA: f64 = 4.50524;

/// **NeighBlock wavelet denoising** (Cai-Silverman 2001) — block thresholding
/// with overlapping decision windows on an orthogonal wavelet basis.
///
/// The pipeline mirrors the other wavelet denoisers of this module: multi-level
/// periodized DWT of the reflection-padded signal (`levels = 0` picks the depth
/// automatically), robust noise scale `σ = MAD/0.6745` from the finest detail
/// band, shrinkage of the detail coefficients, inverse transform cropped to the
/// input length. What differs is the shrinkage rule; in each detail band of
/// length `m`:
///
/// 1. block length `L0 = max(1, ⌊ln m⌋)` and extension `L1 = max(1, ⌊L0/2⌋)`;
/// 2. the band is partitioned into **disjoint** blocks of `L0` coefficients
///    (the last block may be shorter);
/// 3. each block's decision statistic `S² = Σ d_k²` is computed over the block
///    **extended** by `L1` neighbours on each side (clipped to the band), of
///    actual length `L`;
/// 4. every coefficient of the *disjoint* block is scaled by the James-Stein
///    factor `f = max(0, 1 − λ*·L·σ² / S²)` with λ* ≈ 4.50524 ([`LAMBDA`]).
///
/// Blocks whose extended energy sits at the noise floor (`S² ≤ λ*·L·σ²`) are
/// zeroed outright; blocks carrying grouped signal energy are kept almost
/// unchanged — which is exactly where per-coefficient universal thresholding
/// over-smooths dense signals.
///
/// Degrades gracefully: signals shorter than 4 samples, signals the transform
/// cannot decompose, and signals with no measurable noise in the finest band
/// (σ = 0, e.g. constants — preserved exactly) are returned unchanged.
pub fn wavelet_denoise_neighblock(signal: &[f64], levels: usize, wavelet: Wavelet) -> Vec<f64> {
    let n0 = signal.len();
    if n0 < 4
    {
        return signal.to_vec();
    }
    let h = wavelet.lowpass();
    let Some((approx, mut detail_coeffs, _)) = dwt_forward(signal, levels, &h)
    else
    {
        return signal.to_vec();
    };

    // Robust noise scale from the finest detail band (Donoho MAD estimator).
    let sigma = mad(&detail_coeffs[0]) / 0.6745;
    if !sigma.is_finite() || sigma <= 0.0
    {
        return signal.to_vec();
    }
    let sigma_sq = sigma * sigma;

    for detail in detail_coeffs.iter_mut()
    {
        let m = detail.len();
        if m == 0
        {
            continue;
        }
        let l0 = ((m as f64).ln().floor() as usize).max(1);
        let l1 = (l0 / 2).max(1);
        let mut start = 0;
        while start < m
        {
            let end = (start + l0).min(m);
            // Decision statistic over the extended block, clipped to the band.
            let ext_lo = start.saturating_sub(l1);
            let ext_hi = (end + l1).min(m);
            let l_used = (ext_hi - ext_lo) as f64;
            let s2: f64 = detail[ext_lo..ext_hi].iter().map(|&d| d * d).sum();
            let factor = if s2 > 0.0
            {
                (1.0 - LAMBDA * l_used * sigma_sq / s2).max(0.0)
            }
            else
            {
                // A zero-energy extended block is pure silence: kill it (its own
                // coefficients are zero anyway) instead of dividing by zero.
                0.0
            };
            for d in detail[start..end].iter_mut()
            {
                *d *= factor;
            }
            start = end;
        }
    }
    dwt_inverse(approx, &detail_coeffs, &h, n0)
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::super::transform::{ThresholdMode, wavelet_denoise_with};
    use super::*;
    use core::f64::consts::PI;

    fn noisy_sine(n: usize, sigma: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Lcg::new(seed);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + sigma * rng.gauss()).collect();
        (clean, obs)
    }

    #[test]
    fn beats_raw_on_noisy_sine() {
        let (clean, obs) = noisy_sine(1024, 0.4, 131);
        let out = wavelet_denoise_neighblock(&obs, 0, Wavelet::Db4);
        assert_eq!(out.len(), obs.len());
        let s_raw = snr_db(&clean, &obs);
        let s_nb = snr_db(&clean, &out);
        assert!(
            s_nb >= s_raw + 3.0,
            "NeighBlock gained only {:.2} dB ({s_raw:.2} → {s_nb:.2})",
            s_nb - s_raw
        );
    }

    #[test]
    fn beats_universal_threshold_on_dense_signal() {
        // Energy spread over many coefficients (two tones, one fast — the fixture
        // of transform.rs's sure_beats_universal_on_dense_signal): per-coefficient
        // universal thresholding kills the fast tone's many moderate coefficients
        // one by one, while the block statistic sees their grouped energy and
        // keeps them.
        let n = 1024;
        let mut rng = Lcg::new(137);
        let clean: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.7 * (2.0 * PI * 60.0 * t).sin()
            })
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let universal = wavelet_denoise_with(&obs, 0, ThresholdMode::Soft, Wavelet::Db8);
        let neighblock = wavelet_denoise_neighblock(&obs, 0, Wavelet::Db8);
        let s_univ = snr_db(&clean, &universal);
        let s_nb = snr_db(&clean, &neighblock);
        assert!(s_nb > snr_db(&clean, &obs), "NeighBlock must beat raw");
        assert!(
            s_nb > s_univ,
            "NeighBlock {s_nb:.2} dB must beat universal soft {s_univ:.2} dB"
        );
    }

    #[test]
    fn block_rule_differs_from_universal_soft_thresholding() {
        // λ*/blocking must be live: on the same basis and depth the block rule is
        // a genuinely different estimator than per-coefficient universal soft
        // thresholding, not a reimplementation of it. The fixture needs signal
        // energy *inside* the detail bands (a fast tone) — on a slow sine both
        // rules legitimately zero every pure-noise band and coincide.
        let n = 1024;
        let mut rng = Lcg::new(139);
        let obs: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.7 * (2.0 * PI * 60.0 * t).sin() + 0.3 * rng.gauss()
            })
            .collect();
        let universal = wavelet_denoise_with(&obs, 4, ThresholdMode::Soft, Wavelet::Db4);
        let neighblock = wavelet_denoise_neighblock(&obs, 4, Wavelet::Db4);
        assert_eq!(universal.len(), neighblock.len());
        let max_diff = universal
            .iter()
            .zip(neighblock.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1.0e-9,
            "block thresholding coincides with universal soft (max diff {max_diff:.3e})"
        );
    }

    #[test]
    fn degrades_gracefully_on_degenerate_inputs() {
        // Module convention: degenerate inputs come back unchanged, never panic.
        for len in 0..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            assert_eq!(
                wavelet_denoise_neighblock(&x, 0, Wavelet::Db4),
                x,
                "db4 len {len}"
            );
            assert_eq!(
                wavelet_denoise_neighblock(&x, 0, Wavelet::Haar),
                x,
                "haar len {len}"
            );
        }
        // Non-power-of-two lengths go through the reflection padding and come back
        // at the right length, finite.
        let mut rng = Lcg::new(149);
        let x: Vec<f64> = (0..300).map(|_| rng.gauss()).collect();
        let out = wavelet_denoise_neighblock(&x, 0, Wavelet::Db6);
        assert_eq!(out.len(), 300);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preserves_constant_signals_exactly() {
        // A constant has zero detail energy at every level, so σ = 0 and the
        // signal passes through bit-for-bit.
        let c = vec![3.5; 64];
        for wavelet in [Wavelet::Haar, Wavelet::Db4, Wavelet::Db6, Wavelet::Db8]
        {
            assert_eq!(
                wavelet_denoise_neighblock(&c, 0, wavelet),
                c,
                "{wavelet:?} constant not preserved"
            );
        }
    }
}
