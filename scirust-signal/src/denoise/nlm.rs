//! 1-D Non-Local Means denoising (Buades-Coll-Morel 2005).
//!
//! Where a linear smoother averages a sample with its *spatial* neighbours — blurring
//! edges and fast oscillations alike — Non-Local Means averages it with every sample
//! whose surrounding **patch looks the same**, wherever that patch sits inside the
//! search window. Self-similar signals (periodic waveforms, repeated transients,
//! piecewise-constant records) offer many near-identical patches, so the noise is
//! averaged down without mixing dissimilar structure: samples across a step edge get
//! near-zero weight and the edge survives un-blurred.
//!
//! Reference: A. Buades, B. Coll, J.-M. Morel, *"A non-local algorithm for image
//! denoising"*, CVPR 2005 (and the companion *"A review of image denoising
//! algorithms, with a new one"*, Multiscale Model. Simul. 4(2), 2005), transposed
//! here to 1-D with the noise-compensated patch distance of the same authors.

use super::{estimate_noise_std_helper, mirror_index};

/// Auto-bandwidth factor: `h = AUTO_H_FACTOR · σ̂` when [`nlm1d`] is called with
/// `h <= 0`. Filtering-strength heuristics in the NLM literature put `h` at a
/// modest multiple of σ once the patch distance is noise-compensated (Buades et
/// al. use `h ≈ 0.5–1·σ` in that regime). The exact constant was calibrated on
/// this module's test fixtures (sine + white noise for smoothing power, noisy
/// step for edge preservation): see [`nlm1d`] for the value trade-off.
const AUTO_H_FACTOR: f64 = 0.8;

/// **1-D Non-Local Means** (Buades-Coll-Morel 2005).
///
/// For each sample `i`, the patch of half-width `patch_half` centred at `i`
/// (borders mirror-reflected) is compared with every candidate patch centred at
/// `j ∈ [i − search_half, i + search_half]` (clipped to the signal). The squared
/// patch distance is the *mean* over the patch of `(x[i+k] − x[j+k])²`, and the
/// candidate's weight uses the **noise-compensated** rule of Buades et al.:
///
/// ```text
/// w(i, j) = exp(−max(0, d²(i, j) − 2σ²) / h²)
/// ```
///
/// where `σ` is the robust noise scale ([`super::estimate_noise_std_helper`]).
/// Subtracting `2σ²` — the expected distance between two noisy copies of the
/// *same* clean patch — makes the weight respond to genuine structural
/// difference instead of to the noise floor itself. The output sample is the
/// weighted mean of the candidate centres `x[j]`, with `j = i` always included
/// at weight 1.
///
/// ## Parameters
///
/// * `patch_half` — patch half-width; **3–6** is typical (patch of 7–13 samples):
///   long enough to identify structure, short enough to find many matches.
/// * `search_half` — search half-width; **10–40** is typical. Larger windows find
///   more matches on strongly self-similar signals at linear extra cost.
/// * `h` — filtering bandwidth. Larger `h` averages more aggressively; `h → 0`
///   keeps only exact matches. **`h <= 0` selects `h = 0.8·σ̂` automatically** —
///   an `h ∝ σ` rule is the standard heuristic in the 1-D noise-compensated
///   regime, and the factor 0.8 was calibrated on this module's fixtures: on a
///   noisy sine it takes most of the achievable smoothing gain (≈ +10 dB, well
///   past a 5-tap moving average) and on a noisy step it stays ≈ 4 dB ahead of a
///   `gaussian_smooth(2.0)` blur, while keeping the structural-rejection scale
///   `h² = 0.64·σ²` below one σ² so genuine low-contrast features near the noise
///   floor are not averaged away (smaller factors leave noise behind, larger
///   ones start absorbing weak structure).
///
/// Complexity is **O(n · search · patch)** — with the typical parameters above,
/// a few hundred flops per sample.
///
/// ## Degradation & robustness
///
/// * Empty or shorter-than-4 signals come back unchanged.
/// * A signal with no measurable noise (σ̂ = 0, e.g. a constant) is returned
///   unchanged when `h` is automatic — constants are preserved exactly.
/// * Non-finite patch distances (NaN-laced input) are treated as **infinite
///   distance** → weight 0, so a NaN corrupts no neighbour average and the
///   filter never panics; every output sample whose own value is finite stays
///   finite (its `j = i` term always participates).
pub fn nlm1d(signal: &[f64], patch_half: usize, search_half: usize, h: f64) -> Vec<f64> {
    let n = signal.len();
    if n < 4
    {
        return signal.to_vec();
    }
    let sigma = estimate_noise_std_helper(signal);
    let h_eff = if h > 0.0 { h } else { AUTO_H_FACTOR * sigma };
    if !h_eff.is_finite() || h_eff <= 0.0
    {
        // Auto bandwidth on a noise-free (or pathological) signal: there is nothing
        // to average away — pass through, preserving constants exactly.
        return signal.to_vec();
    }
    let noise_floor = 2.0 * sigma * sigma;
    let h_sq = h_eff * h_eff;
    let p = patch_half as isize;
    let patch_len = (2 * patch_half + 1) as f64;

    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate()
    {
        let lo = i.saturating_sub(search_half);
        let hi = (i + search_half).min(n - 1);
        let mut num = 0.0;
        let mut den = 0.0;
        for j in lo..=hi
        {
            let w = if j == i
            {
                // The reference patch always participates at full weight, so the
                // weighted mean is well-defined (den >= 1) even when every other
                // candidate is rejected.
                1.0
            }
            else
            {
                let mut d2 = 0.0;
                for k in -p..=p
                {
                    let a = signal[mirror_index(i as isize + k, n)];
                    let b = signal[mirror_index(j as isize + k, n)];
                    let diff = a - b;
                    d2 += diff * diff;
                }
                d2 /= patch_len;
                if d2.is_finite()
                {
                    (-(d2 - noise_floor).max(0.0) / h_sq).exp()
                }
                else
                {
                    // A NaN anywhere in either patch makes the distance NaN: treat
                    // it as infinitely dissimilar (weight 0) instead of letting the
                    // NaN poison the average.
                    0.0
                }
            };
            if w > 0.0
            {
                num += w * signal[j];
                den += w;
            }
        }
        *o = num / den;
    }
    out
}

/// [`nlm1d`] with the recommended defaults: `patch_half = 4`, `search_half = 24`,
/// automatic bandwidth (`h = 0.8·σ̂`). A good first choice for self-similar
/// signals — periodic waveforms, repeated transients, piecewise-constant records.
pub fn nlm1d_auto(signal: &[f64]) -> Vec<f64> {
    nlm1d(signal, 4, 24, 0.0)
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::super::{estimate_noise_std_helper, gaussian_smooth, moving_average};
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
    fn auto_beats_raw_and_moving_average_on_repetitive_signal() {
        // Self-similarity is NLM's home turf: a sine offers many near-identical
        // patches inside the search window, so NLM must beat both the raw input
        // (by a wide margin) and a plain 5-tap moving average.
        let (clean, obs) = noisy_sine(1024, 0.4, 101);
        let nlm = nlm1d_auto(&obs);
        let ma = moving_average(&obs, 5);
        let s_raw = snr_db(&clean, &obs);
        let s_nlm = snr_db(&clean, &nlm);
        let s_ma = snr_db(&clean, &ma);
        assert!(
            s_nlm >= s_raw + 3.0,
            "NLM gained only {:.2} dB ({s_raw:.2} → {s_nlm:.2})",
            s_nlm - s_raw
        );
        assert!(
            s_nlm > s_ma,
            "NLM {s_nlm:.2} dB must beat moving_average(5) {s_ma:.2} dB"
        );
    }

    #[test]
    fn preserves_step_edge_better_than_gaussian() {
        // Patches straddling the edge are structurally dissimilar (clean distance
        // far above the noise floor), so their weights vanish and each side is
        // averaged only with itself — a Gaussian blur has no such mechanism and
        // smears the step. The edge sits at an odd index, away from the borders.
        let n = 512;
        let mut rng = Lcg::new(103);
        let clean: Vec<f64> = (0..n).map(|i| if i < 201 { 0.0 } else { 2.0 }).collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let nlm = nlm1d(&obs, 3, 20, 0.0);
        let gauss = gaussian_smooth(&obs, 2.0);
        let s_nlm = snr_db(&clean, &nlm);
        let s_gauss = snr_db(&clean, &gauss);
        assert!(
            s_nlm > s_gauss,
            "NLM {s_nlm:.2} dB must beat gaussian_smooth(2.0) {s_gauss:.2} dB on a step"
        );
        assert!(s_nlm > snr_db(&clean, &obs), "NLM must also beat raw");
    }

    #[test]
    fn bandwidth_and_search_parameters_are_live() {
        let (_, obs) = noisy_sine(1024, 0.4, 107);
        let sigma = estimate_noise_std_helper(&obs);
        let dist2 = |a: &[f64], b: &[f64]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
        };
        // A tiny bandwidth keeps only exact matches: the output must stay closer
        // to the input than the auto-bandwidth output (which genuinely smooths).
        let tiny = nlm1d(&obs, 4, 24, 0.01 * sigma);
        let auto = nlm1d_auto(&obs);
        assert!(
            dist2(&tiny, &obs) < dist2(&auto, &obs),
            "h is ignored: tiny-h moved {:.4}, auto-h moved {:.4}",
            dist2(&tiny, &obs),
            dist2(&auto, &obs)
        );
        assert!(
            dist2(&auto, &obs) > 0.0,
            "auto bandwidth must actually smooth"
        );
        // search_half = 0 leaves only the j = i candidate: the output is the input.
        let frozen = nlm1d(&obs, 4, 0, 0.0);
        for (a, b) in obs.iter().zip(frozen.iter())
        {
            assert!(
                (a - b).abs() < 1.0e-12,
                "search_half = 0 must return the input: {a} vs {b}"
            );
        }
    }

    #[test]
    fn degrades_gracefully_on_degenerate_inputs() {
        // Module convention: degenerate inputs come back unchanged, never panic.
        let empty: [f64; 0] = [];
        assert!(nlm1d_auto(&empty).is_empty());
        for len in 1..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            assert_eq!(nlm1d(&x, 3, 10, 0.5), x, "len {len}");
            assert_eq!(nlm1d_auto(&x), x, "auto len {len}");
        }
        // A constant has σ̂ = 0, so the automatic bandwidth passes it through
        // bit-for-bit; an explicit bandwidth averages equals and stays put too.
        let c = vec![3.5; 64];
        assert_eq!(nlm1d_auto(&c), c, "constant not preserved exactly");
        for v in nlm1d(&c, 3, 10, 0.5)
        {
            assert!((v - 3.5).abs() < 1.0e-12, "constant drifted to {v}");
        }
    }

    #[test]
    fn nan_laced_input_does_not_panic() {
        // NaN patch distances count as infinite (weight 0), so the NaNs stay
        // quarantined: every sample that was finite on input is finite on output.
        let (_, mut obs) = noisy_sine(256, 0.3, 109);
        obs[40] = f64::NAN;
        obs[170] = f64::NAN;
        let out = nlm1d(&obs, 3, 12, 0.0);
        assert_eq!(out.len(), obs.len());
        for (i, (x, y)) in obs.iter().zip(out.iter()).enumerate()
        {
            if x.is_finite()
            {
                assert!(y.is_finite(), "sample {i} became non-finite: {y}");
            }
        }
    }
}
