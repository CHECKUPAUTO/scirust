//! Doppler processing — the range-Doppler map.
//!
//! After pulse compression each transmitted pulse yields a range profile.
//! Stacking `M` pulses and taking an FFT along **slow-time** (the pulse index)
//! resolves radial velocity: a target's Doppler shift places it in a Doppler
//! bin, separating moving targets from stationary clutter, which sits at zero
//! Doppler. This is the surface CFAR ([`super::cfar`]) then detects on.

use crate::complex::Complex;
use crate::fft::fft;

/// The Doppler spectrum of a single range bin — the FFT of its slow-time
/// sequence (one complex sample per pulse). `slow_time.len()` must be a power
/// of two (the radix-2 FFT constraint); returns an empty vector otherwise. Bin
/// `0` is zero-Doppler (a stationary target / clutter).
pub fn doppler_spectrum(slow_time: &[Complex]) -> Vec<Complex> {
    let m = slow_time.len();
    if m == 0 || m & (m - 1) != 0
    {
        return Vec::new();
    }
    let mut buf = slow_time.to_vec();
    fft(&mut buf);
    buf
}

/// The **range-Doppler map** from a stack of `M` range-compressed pulses:
/// `pulses[m]` is pulse `m`'s range profile, all of the same length `N`.
///
/// For each range bin the slow-time sequence across the `M` pulses is
/// Doppler-FFT'd; the result is an `N × M` magnitude map indexed
/// `map[range][doppler]`, with Doppler bin `0` stationary. `M` must be a power
/// of two and every pulse the same non-zero length, otherwise an empty map is
/// returned.
pub fn range_doppler_map(pulses: &[Vec<Complex>]) -> Vec<Vec<f64>> {
    let m = pulses.len();
    if m == 0 || m & (m - 1) != 0
    {
        return Vec::new();
    }
    let n = pulses[0].len();
    if n == 0 || pulses.iter().any(|p| p.len() != n)
    {
        return Vec::new();
    }
    let mut map = vec![vec![0.0; m]; n];
    let mut col = vec![Complex::zero(); m];
    for (r, row) in map.iter_mut().enumerate()
    {
        for (slow, p) in pulses.iter().enumerate()
        {
            col[slow] = p[r];
        }
        fft(&mut col);
        for (d, c) in col.iter().enumerate()
        {
            row[d] = c.mag();
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn peak_bin(row: &[f64]) -> usize {
        (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap()
    }

    #[test]
    fn stationary_target_lands_in_the_zero_doppler_bin() {
        let (m, n) = (64usize, 16usize);
        // A target at range bin 5, constant across pulses → Doppler 0.
        let pulses: Vec<Vec<Complex>> = (0..m)
            .map(|_| {
                let mut row = vec![Complex::zero(); n];
                row[5] = Complex::new(1.0, 0.0);
                row
            })
            .collect();
        let map = range_doppler_map(&pulses);
        assert_eq!(peak_bin(&map[5]), 0);
        // Coherent integration of M unit samples gives magnitude M at bin 0.
        assert!((map[5][0] - m as f64).abs() < 1e-9);
        // A range bin with no target stays empty.
        assert!(map[0].iter().all(|&v| v < 1e-9));
    }

    #[test]
    fn moving_target_lands_in_the_bin_matching_its_doppler() {
        let (m, n, k0) = (64usize, 16usize, 7usize);
        // Range bin 9 carries a phase ramp of k0 cycles over the M pulses.
        let pulses: Vec<Vec<Complex>> = (0..m)
            .map(|slow| {
                let mut row = vec![Complex::zero(); n];
                row[9] = Complex::cis(2.0 * PI * k0 as f64 * slow as f64 / m as f64);
                row
            })
            .collect();
        let map = range_doppler_map(&pulses);
        // The Doppler peak is at bin k0 (or M−k0 under the opposite FFT sign
        // convention), and it is a sharp, coherent M-magnitude line.
        let pk = peak_bin(&map[9]);
        assert!(pk == k0 || pk == m - k0, "peak at {pk}, expected {k0}");
        assert!((map[9][pk] - m as f64).abs() < 1e-9);
        assert!(pk != 0, "a moving target must not sit at zero Doppler");
    }

    #[test]
    fn doppler_processing_rejects_non_power_of_two_and_ragged_input() {
        assert!(doppler_spectrum(&[Complex::zero(); 3]).is_empty());
        let three: Vec<Vec<Complex>> = (0..3).map(|_| vec![Complex::zero(); 4]).collect();
        assert!(range_doppler_map(&three).is_empty());
        let ragged = vec![vec![Complex::zero(); 4], vec![Complex::zero(); 5]];
        assert!(range_doppler_map(&ragged).is_empty());
    }
}
