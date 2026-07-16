//! Rank / order-statistic filters — the robust family.
//!
//! Unlike linear filters, these reject *impulsive* noise (spikes, salt-and-pepper,
//! dropouts) without smearing it across neighbours, because a single outlier cannot
//! move a median. They are the correct tool the moment [`super::detect::classify`]
//! reports [`super::NoiseType::Impulsive`].
//!
//! All four filters run on **one shared engine**: an incrementally maintained
//! sorted window ([`super::streaming`]'s NaN-safe `total_cmp` machinery) swept over
//! the mirrored signal. Each step removes the sample leaving the window and inserts
//! the one entering it — `O(w)` per sample instead of the `O(w·log w)` of re-sorting
//! every window — and, because it is the *same* code path the streaming rank filters
//! use, the batch/streaming interior equivalence the module guarantees holds by
//! construction rather than by test alone.

use super::streaming::{median_of_sorted, sorted_insert, sorted_remove};
use super::{mad, mirror_index};

/// Sweep an incrementally sorted window of half-width `half_window` over the
/// mirrored signal, calling `emit(i, &sorted_window)` for every sample index. The
/// window multiset at step `i` is exactly `{signal[mirror(i−h)], …,
/// signal[mirror(i+h)]}` — identical to rebuilding and sorting per position, at a
/// fraction of the cost.
fn sorted_window_sweep(signal: &[f64], half_window: usize, mut emit: impl FnMut(usize, &[f64])) {
    let n = signal.len();
    let m = half_window as isize;
    let mut window: Vec<f64> = Vec::with_capacity(2 * half_window + 1);
    for k in -m..=m
    {
        sorted_insert(&mut window, signal[mirror_index(k, n)]);
    }
    emit(0, &window);
    for i in 1..n
    {
        sorted_remove(&mut window, signal[mirror_index(i as isize - 1 - m, n)]);
        sorted_insert(&mut window, signal[mirror_index(i as isize + m, n)]);
        emit(i, &window);
    }
}

/// Sliding-window median filter, half-width `half_window` (full window
/// `2*half_window+1`). The canonical impulse remover: replaces each sample by the
/// median of its neighbourhood, so isolated spikes vanish while edges survive.
pub fn median_filter(signal: &[f64], half_window: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || half_window == 0
    {
        return signal.to_vec();
    }
    let mut out = vec![0.0; n];
    sorted_window_sweep(signal, half_window, |i, window| {
        out[i] = median_of_sorted(window);
    });
    out
}

/// Hampel filter: a *decision* filter that only touches samples flagged as
/// outliers. In each window it computes the local median and a robust scale
/// `1.4826 · MAD`; if a sample deviates by more than `n_sigma` scales it is
/// replaced by the median, otherwise it is kept verbatim. This is the sharpest
/// noise/information separator for spikes — untouched samples are declared
/// "information", replaced ones "noise".
pub fn hampel_filter(signal: &[f64], half_window: usize, n_sigma: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || half_window == 0
    {
        return signal.to_vec();
    }
    let mut out = signal.to_vec();
    sorted_window_sweep(signal, half_window, |i, window| {
        let med = median_of_sorted(window);
        let scale = 1.4826 * mad(window);
        if scale > 0.0 && (signal[i] - med).abs() > n_sigma * scale
        {
            out[i] = med;
        }
    });
    out
}

/// Flag impulsive samples via the Hampel criterion without modifying the signal.
/// Returns a boolean mask where `true` marks a sample the detector considers noise
/// (an outlier) rather than information. This is the explicit "what is noise" map.
pub fn impulse_mask(signal: &[f64], half_window: usize, n_sigma: f64) -> Vec<bool> {
    let n = signal.len();
    let mut mask = vec![false; n];
    if n == 0 || half_window == 0
    {
        return mask;
    }
    sorted_window_sweep(signal, half_window, |i, window| {
        let med = median_of_sorted(window);
        let scale = 1.4826 * mad(window);
        mask[i] = scale > 0.0 && (signal[i] - med).abs() > n_sigma * scale;
    });
    mask
}

/// α-trimmed mean filter: sort each window, discard a fraction `alpha` (0..0.5)
/// from *each* tail, average the rest. Interpolates between the moving average
/// (`alpha = 0`) and the median (`alpha → 0.5`), trading impulse rejection against
/// Gaussian-noise smoothing.
pub fn alpha_trimmed_mean(signal: &[f64], half_window: usize, alpha: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || half_window == 0
    {
        return signal.to_vec();
    }
    let a = alpha.clamp(0.0, 0.499);
    let win = 2 * half_window + 1;
    let trim = (a * win as f64).floor() as usize;
    let mut out = vec![0.0; n];
    sorted_window_sweep(signal, half_window, |i, window| {
        let kept = &window[trim..win - trim];
        out[i] = kept.iter().sum::<f64>() / kept.len() as f64;
    });
    out
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn median_removes_spikes() {
        let mut sig: Vec<f64> = (0..128)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / 128.0).sin())
            .collect();
        let clean = sig.clone();
        // Inject salt-and-pepper spikes.
        for &idx in &[10usize, 33, 57, 90, 120]
        {
            sig[idx] += 8.0;
        }
        let out = median_filter(&sig, 2);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &sig) + 10.0);
    }

    #[test]
    fn hampel_keeps_clean_samples_verbatim() {
        let sig: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
        let out = hampel_filter(&sig, 3, 3.0);
        // No outliers → output essentially equals input.
        for (a, b) in sig.iter().zip(out.iter())
        {
            assert!((a - b).abs() < 1.0e-9);
        }
    }

    #[test]
    fn impulse_mask_flags_injected_spikes() {
        let mut sig: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let spikes = [15usize, 40, 77, 100];
        for &idx in &spikes
        {
            sig[idx] += 10.0;
        }
        let mask = impulse_mask(&sig, 3, 3.0);
        for &idx in &spikes
        {
            assert!(mask[idx], "spike at {idx} not flagged");
        }
        let flagged = mask.iter().filter(|&&b| b).count();
        assert!(flagged < 12, "too many false positives: {flagged}");
    }

    #[test]
    fn incremental_sweep_matches_the_naive_definition_exactly() {
        // The shared sorted-window engine must be *bit-for-bit* the per-position
        // rebuild-and-sort definition — including on mirrored borders, duplicated
        // values, and a NaN resident (total_cmp order end to end).
        let mut rng = Lcg::new(23);
        let mut sig: Vec<f64> = (0..97).map(|_| (rng.gauss() * 4.0).round() / 4.0).collect();
        sig[41] = f64::NAN;
        for half in [1usize, 2, 4]
        {
            let fast = median_filter(&sig, half);
            let n = sig.len();
            let m = half as isize;
            for i in 0..n
            {
                let mut window: Vec<f64> = (-m..=m)
                    .map(|k| sig[super::super::mirror_index(i as isize + k, n)])
                    .collect();
                window.sort_by(|a, b| a.total_cmp(b));
                let naive = super::super::streaming::median_of_sorted(&window);
                assert!(
                    fast[i].to_bits() == naive.to_bits(),
                    "half {half}, index {i}: {} vs naive {}",
                    fast[i],
                    naive
                );
            }
        }
    }

    #[test]
    fn alpha_trimmed_between_mean_and_median() {
        let (clean, obs) = {
            let mut rng = Lcg::new(7);
            let clean: Vec<f64> = (0..200)
                .map(|i| (2.0 * PI * 2.0 * i as f64 / 200.0).sin())
                .collect();
            let mut obs: Vec<f64> = clean.iter().map(|&c| c + 0.2 * rng.gauss()).collect();
            for &idx in &[20usize, 60, 130, 170]
            {
                obs[idx] += 6.0;
            }
            (clean, obs)
        };
        let out = alpha_trimmed_mean(&obs, 3, 0.25);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }
}
