//! Rank / order-statistic filters — the robust family.
//!
//! Unlike linear filters, these reject *impulsive* noise (spikes, salt-and-pepper,
//! dropouts) without smearing it across neighbours, because a single outlier cannot
//! move a median. They are the correct tool the moment [`super::detect::classify`]
//! reports [`super::NoiseType::Impulsive`].

use super::{mad, median, mirror_index};

/// Sliding-window median filter, half-width `half_window` (full window
/// `2*half_window+1`). The canonical impulse remover: replaces each sample by the
/// median of its neighbourhood, so isolated spikes vanish while edges survive.
pub fn median_filter(signal: &[f64], half_window: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || half_window == 0
    {
        return signal.to_vec();
    }
    let m = half_window as isize;
    let mut out = vec![0.0; n];
    let mut window = Vec::with_capacity(2 * half_window + 1);
    for (i, o) in out.iter_mut().enumerate()
    {
        window.clear();
        for k in -m..=m
        {
            window.push(signal[mirror_index(i as isize + k, n)]);
        }
        *o = median(&window);
    }
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
    let m = half_window as isize;
    let mut out = signal.to_vec();
    let mut window = Vec::with_capacity(2 * half_window + 1);
    for i in 0..n
    {
        window.clear();
        for k in -m..=m
        {
            window.push(signal[mirror_index(i as isize + k, n)]);
        }
        let med = median(&window);
        let scale = 1.4826 * mad(&window);
        if scale > 0.0 && (signal[i] - med).abs() > n_sigma * scale
        {
            out[i] = med;
        }
    }
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
    let m = half_window as isize;
    let mut window = Vec::with_capacity(2 * half_window + 1);
    for (i, flag) in mask.iter_mut().enumerate()
    {
        window.clear();
        for k in -m..=m
        {
            window.push(signal[mirror_index(i as isize + k, n)]);
        }
        let med = median(&window);
        let scale = 1.4826 * mad(&window);
        *flag = scale > 0.0 && (signal[i] - med).abs() > n_sigma * scale;
    }
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
    let m = half_window as isize;
    let win = 2 * half_window + 1;
    let trim = (a * win as f64).floor() as usize;
    let mut out = vec![0.0; n];
    let mut window = Vec::with_capacity(win);
    for (i, o) in out.iter_mut().enumerate()
    {
        window.clear();
        for k in -m..=m
        {
            window.push(signal[mirror_index(i as isize + k, n)]);
        }
        // Total order (NaN-safe): a partial_cmp comparator is inconsistent on NaN and
        // makes modern Rust sorts panic; total_cmp degrades gracefully instead.
        window.sort_by(|x, y| x.total_cmp(y));
        let lo = trim;
        let hi = win - trim;
        let kept = &window[lo..hi];
        *o = kept.iter().sum::<f64>() / kept.len() as f64;
    }
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
