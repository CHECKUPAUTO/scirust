//! The radar **ambiguity function** `|χ(τ, ν)|` — a waveform's joint
//! delay–Doppler response. It shows how well a waveform resolves targets in
//! range (delay) and velocity (Doppler) at once, and exposes **range-Doppler
//! coupling**: for a linear-FM chirp the response is a diagonal ridge, so a
//! Doppler shift masquerades as a range shift.

use super::matched_filter::cross_correlate;
use crate::complex::Complex;
use std::f64::consts::PI;

/// The narrowband ambiguity surface `|χ(τ, ν)|` of `waveform`, evaluated at
/// `num_doppler` normalized Doppler shifts spanning the unit frequency
/// interval. For each shift the waveform is modulated by `e^{j2πνn}` and
/// cross-correlated with the original, so each row is the delay (range)
/// response at that Doppler.
///
/// Returns a `num_doppler × (2N − 1)` magnitude grid indexed `[doppler][delay]`
/// (delay lag `−(N−1)…(N−1)`; delay index `N − 1` is zero lag). Doppler index
/// `num_doppler / 2` is zero Doppler — that row is the autocorrelation /
/// matched-filter cut. Empty when the waveform is empty or `num_doppler == 0`.
pub fn ambiguity(waveform: &[Complex], num_doppler: usize) -> Vec<Vec<f64>> {
    if waveform.is_empty() || num_doppler == 0
    {
        return Vec::new();
    }
    let half = (num_doppler / 2) as isize;
    (0..num_doppler)
        .map(|d| {
            let f = (d as isize - half) as f64 / num_doppler as f64; // normalized Doppler
            let modulated: Vec<Complex> = waveform
                .iter()
                .enumerate()
                .map(|(n, &s)| s * Complex::cis(2.0 * PI * f * n as f64))
                .collect();
            cross_correlate(&modulated, waveform)
                .iter()
                .map(Complex::mag)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::waveform::lfm_chirp;
    use super::*;

    fn peak_delay(row: &[f64]) -> isize {
        (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap() as isize
    }

    #[test]
    fn ambiguity_peak_is_at_the_origin_and_equals_the_energy() {
        let n = 64;
        let chirp = lfm_chirp(n, 4.0e6, 10.0e6);
        let num_doppler = 32;
        let amb = ambiguity(&chirp, num_doppler);
        let peak = amb[num_doppler / 2][n - 1]; // zero Doppler, zero delay
        assert!(
            (peak - n as f64).abs() < 1e-6,
            "origin {peak} != energy {n}"
        );
        // And it is the global maximum of the whole surface.
        let global = amb.iter().flatten().cloned().fold(0.0, f64::max);
        assert!(
            (global - peak).abs() < 1e-9,
            "origin is not the global peak"
        );
    }

    #[test]
    fn ambiguity_zero_doppler_cut_is_the_autocorrelation() {
        let n = 32;
        let chirp = lfm_chirp(n, 3.0e6, 10.0e6);
        let num_doppler = 16;
        let amb = ambiguity(&chirp, num_doppler);
        let auto: Vec<f64> = cross_correlate(&chirp, &chirp)
            .iter()
            .map(Complex::mag)
            .collect();
        for (a, b) in amb[num_doppler / 2].iter().zip(&auto)
        {
            assert!((a - b).abs() < 1e-9, "zero-Doppler cut ≠ autocorrelation");
        }
    }

    #[test]
    fn lfm_ambiguity_shows_range_doppler_coupling() {
        // For an LFM chirp the ridge is sheared: the delay of each Doppler
        // row's peak moves monotonically with Doppler (range-Doppler coupling),
        // while the zero-Doppler peak sits at zero lag.
        let n = 128;
        let chirp = lfm_chirp(n, 5.0e6, 10.0e6);
        let num_doppler = 32;
        let amb = ambiguity(&chirp, num_doppler);
        let mid = num_doppler / 2;
        let (lo, ctr, hi) = (
            peak_delay(&amb[mid - 8]),
            peak_delay(&amb[mid]),
            peak_delay(&amb[mid + 8]),
        );
        assert_eq!(ctr, (n - 1) as isize, "zero-Doppler peak not at zero lag");
        assert!(
            (lo < ctr && ctr < hi) || (lo > ctr && ctr > hi),
            "no coupling ridge: {lo}, {ctr}, {hi}"
        );
    }

    #[test]
    fn ambiguity_edge_cases() {
        assert!(ambiguity(&[], 8).is_empty());
        assert!(ambiguity(&[Complex::new(1.0, 0.0)], 0).is_empty());
    }
}
