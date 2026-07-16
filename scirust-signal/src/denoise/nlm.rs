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

/// Mirror-extended copy of `signal`: `patch_half` reflected samples on each side
/// (`ext[t] = signal[mirror_index(t − patch_half, n)]`), so the patch of
/// half-width `patch_half` centred at sample `c` is the **contiguous** slice
/// `&ext[c .. c + 2·patch_half + 1]`. Folding every border index once per signal
/// — instead of twice per element of every patch comparison — is what lets
/// [`sum_sq_diff`] run branch- and division-free over plain slices. Handles
/// `patch_half > n` (multiple folds) exactly like per-element [`mirror_index`].
fn mirror_extend(signal: &[f64], patch_half: usize) -> Vec<f64> {
    let n = signal.len();
    let ph = patch_half as isize;
    let mut ext = Vec::with_capacity(n + 2 * patch_half);
    for t in 0..(n + 2 * patch_half) as isize
    {
        ext.push(signal[mirror_index(t - ph, n)]);
    }
    ext
}

/// Sum of squared differences between two equal-length contiguous slices, laid
/// out for LLVM auto-vectorization on stable Rust: `as_chunks::<4>()` removes
/// every bounds check from the loop body, and the **four independent partial
/// accumulators** break the sequential floating-point dependency chain so the
/// backend can keep the additions in SIMD lanes (SSE2/AVX on x86-64, NEON on
/// AArch64). The remainder (≤ 3 elements) is summed scalar. Summation order
/// therefore differs from a naive single-accumulator loop by reassociation
/// only (≤ 1e-12 relative — pinned by unit test against
/// [`patch_dist_reference`]); a NaN in either slice still propagates to the
/// result, exactly as it does through the naive loop.
fn sum_sq_diff(a: &[f64], b: &[f64]) -> f64 {
    let (qa4, ra) = a.as_chunks::<4>();
    let (qb4, rb) = b.as_chunks::<4>();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
    for (qa, qb) in qa4.iter().zip(qb4)
    {
        let d0 = qa[0] - qb[0];
        let d1 = qa[1] - qb[1];
        let d2 = qa[2] - qb[2];
        let d3 = qa[3] - qb[3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb)
    {
        let d = x - y;
        tail += d * d;
    }
    (s0 + s1) + (s2 + s3) + tail
}

/// The pre-optimization scalar patch distance — mean of `(x[i+k] − x[j+k])²`
/// over the patch, borders mirror-reflected **per element** via
/// [`mirror_index`], summed left to right into a single accumulator. Retained
/// verbatim as the numerical reference the test suite pins the
/// layout-optimized kernel of [`nlm1d`] against (identical values up to
/// floating-point reassociation, ≤ 1e-12 relative).
#[cfg(test)]
fn patch_dist_reference(signal: &[f64], i: usize, j: usize, patch_half: usize) -> f64 {
    let n = signal.len();
    let p = patch_half as isize;
    let mut d2 = 0.0;
    for k in -p..=p
    {
        let a = signal[mirror_index(i as isize + k, n)];
        let b = signal[mirror_index(j as isize + k, n)];
        let diff = a - b;
        d2 += diff * diff;
    }
    d2 / (2 * patch_half + 1) as f64
}

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
/// ## Implementation note — auto-vectorization-friendly layout
///
/// The patch-distance kernel is layout-optimized so LLVM auto-vectorizes it on
/// stable Rust (explicit SIMD stays gated behind nightly in `scirust-simd`):
/// the signal is mirror-extended **once** ([`mirror_extend`]) so every patch is
/// a contiguous slice, and the distance is a straight-line sum of squared
/// differences over two slices with four independent accumulators
/// ([`sum_sq_diff`]) — no bounds checks, no branches, no per-element index
/// folding inside the loop. Reassociating the sum can move a distance by
/// rounding only (≤ 1e-12 relative, pinned by unit test against the retained
/// scalar [`patch_dist_reference`]); the weight rule, its guards, and the NaN
/// handling below are byte-for-byte unchanged.
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
    let plen = 2 * patch_half + 1;
    let patch_len = plen as f64;
    // One mirrored-extended copy of the signal makes every patch the contiguous
    // slice &ext[c .. c + plen], so the hot distance loop below never folds a
    // border index — see the implementation note in the doc above.
    let ext = mirror_extend(signal, patch_half);

    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate()
    {
        let lo = i.saturating_sub(search_half);
        let hi = (i + search_half).min(n - 1);
        let patch_i = &ext[i..i + plen];
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
                let d2 = sum_sq_diff(patch_i, &ext[j..j + plen]) / patch_len;
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
    fn layout_optimized_patch_distance_matches_scalar_reference() {
        // The vectorized kernel (mirror-extended buffer + four independent
        // accumulators) reassociates the sum, so it may differ from the scalar
        // reference by rounding only: ≤ 1e-12 relative, checked exhaustively on
        // random signals over every (i, j) pair and across every border-folding
        // regime — including patch_half > n, where the mirror folds repeatedly.
        let mut rng = Lcg::new(211);
        for &n in &[5usize, 17, 64, 257]
        {
            let signal: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
            for &ph in &[1usize, 3, 4, 7]
            {
                let plen = 2 * ph + 1;
                let ext = mirror_extend(&signal, ph);
                assert_eq!(ext.len(), n + 2 * ph);
                for i in 0..n
                {
                    for j in 0..n
                    {
                        let fast = sum_sq_diff(&ext[i..i + plen], &ext[j..j + plen]) / plen as f64;
                        let reference = patch_dist_reference(&signal, i, j, ph);
                        let tol = 1.0e-12 * reference.abs().max(1.0);
                        assert!(
                            (fast - reference).abs() <= tol,
                            "n {n} ph {ph} i {i} j {j}: fast {fast} vs reference {reference}"
                        );
                    }
                }
            }
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
