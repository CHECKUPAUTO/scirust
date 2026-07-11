//! FMCW (frequency-modulated continuous-wave) radar processing.
//!
//! A pulse-Doppler radar ([`super::waveform`]–[`super::doppler`]) transmits a
//! coded pulse and matched-filters the echo. An **FMCW** radar instead sweeps a
//! linear chirp continuously and *mixes* the echo with the transmitted sweep;
//! the delayed echo, beat against the still-rising transmit tone, produces a
//! low **beat frequency** proportional to target range. This is the whole
//! automotive / mmWave (TI, OpenRadar) processing model: one range-FFT along
//! **fast-time** (the samples inside one chirp) turns beat frequency into range,
//! and a second Doppler-FFT along **slow-time** (the chirp index) turns
//! chirp-to-chirp phase drift into radial velocity — the range-Doppler cube.
//!
//! Because the mixer already reduces the wideband echo to a baseband beat tone,
//! FMCW does the range compression with a plain FFT rather than the
//! matched-filter cross-correlation of the pulse chain — the two ranges of
//! [`super::doppler::range_doppler_map`] (which assumes pulses already
//! range-compressed) do not overlap with what [`range_doppler`] does here (raw
//! beat signals, both FFTs).

use crate::complex::Complex;
use crate::fft::fft;

/// Convert a measured **beat frequency** to target range.
///
/// A chirp of slope `slope` (Hz per second, i.e. bandwidth / sweep-time) and a
/// round-trip delay `τ = 2R/c` produce a beat `f_b = slope·τ = 2·slope·R/c`.
/// Inverting, `R = f_b·c / (2·slope)`. `c` is the propagation speed (≈ 3e8 m/s
/// in air). Returns `0` for a non-positive or non-finite slope.
pub fn beat_frequency_to_range(f_beat: f64, slope: f64, c: f64) -> f64 {
    if !slope.is_finite() || slope <= 0.0
    {
        return 0.0;
    }
    f_beat * c / (2.0 * slope)
}

/// The range resolution of an FMCW sweep of total bandwidth `bandwidth` (Hz):
/// `ΔR = c / (2·B)`. Two targets closer than this fall in the same range bin
/// regardless of how finely the beat spectrum is sampled. Returns `0` for a
/// non-positive or non-finite bandwidth.
pub fn range_resolution(bandwidth: f64, c: f64) -> f64 {
    if !bandwidth.is_finite() || bandwidth <= 0.0
    {
        return 0.0;
    }
    c / (2.0 * bandwidth)
}

/// The **range profile** of a single chirp: the fast-time FFT of its complex
/// beat signal. Each output bin is a range gate; a single target at range `R`
/// appears as a peak at the bin matching its beat frequency `f_b = 2·slope·R/c`.
/// `beat.len()` must be a power of two (the radix-2 FFT constraint); returns an
/// empty vector otherwise.
pub fn range_profile(beat: &[Complex]) -> Vec<Complex> {
    let n = beat.len();
    if n == 0 || n & (n - 1) != 0
    {
        return Vec::new();
    }
    let mut buf = beat.to_vec();
    fft(&mut buf);
    buf
}

/// The **range-Doppler cube** from a frame of `M` raw beat chirps:
/// `frames[m]` is chirp `m`'s complex beat signal, all of the same length `N`.
///
/// Two FFTs: a fast-time (range) FFT of every chirp, then a slow-time (Doppler)
/// FFT of every range bin across the `M` chirps. The result is an `N × M`
/// magnitude map indexed `map[range][doppler]`, with Doppler bin `0` stationary
/// (zero radial velocity / clutter). Both `M` and `N` must be powers of two and
/// every chirp the same non-zero length, otherwise an empty map is returned.
pub fn range_doppler(frames: &[Vec<Complex>]) -> Vec<Vec<f64>> {
    let m = frames.len();
    if m == 0 || m & (m - 1) != 0
    {
        return Vec::new();
    }
    let n = frames[0].len();
    if n == 0 || n & (n - 1) != 0 || frames.iter().any(|f| f.len() != n)
    {
        return Vec::new();
    }
    // Fast-time (range) FFT of each chirp.
    let ranges: Vec<Vec<Complex>> = frames
        .iter()
        .map(|f| {
            let mut buf = f.clone();
            fft(&mut buf);
            buf
        })
        .collect();
    // Slow-time (Doppler) FFT of each range bin, then magnitude.
    let mut map = vec![vec![0.0; m]; n];
    let mut col = vec![Complex::zero(); m];
    for (r, row) in map.iter_mut().enumerate()
    {
        for (slow, rp) in ranges.iter().enumerate()
        {
            col[slow] = rp[r];
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
    fn range_profile_peaks_at_the_beat_bin() {
        // A pure beat tone at bin k0 must transform to a single spectral line.
        let (n, k0) = (64usize, 11usize);
        let beat: Vec<Complex> = (0..n)
            .map(|i| Complex::cis(2.0 * PI * k0 as f64 * i as f64 / n as f64))
            .collect();
        let prof = range_profile(&beat);
        assert_eq!(
            peak_bin(&prof.iter().map(|c| c.mag()).collect::<Vec<_>>()),
            k0
        );
        // Coherent: N samples of unit magnitude give a bin-k0 line of height N.
        assert!((prof[k0].mag() - n as f64).abs() < 1e-9);
    }

    #[test]
    fn beat_frequency_round_trips_range() {
        // f_b = 2·slope·R/c must invert back to R.
        let (c, slope, r) = (3.0e8, 2.0e12, 42.0);
        let f_b = 2.0 * slope * r / c;
        assert!((beat_frequency_to_range(f_b, slope, c) - r).abs() < 1e-6);
        // Guards.
        assert_eq!(beat_frequency_to_range(1.0, 0.0, c), 0.0);
        assert_eq!(beat_frequency_to_range(1.0, -1.0, c), 0.0);
    }

    #[test]
    fn range_resolution_matches_the_closed_form() {
        let c = 3.0e8;
        // 4 GHz sweep ⇒ 3.75 cm resolution.
        assert!((range_resolution(4.0e9, c) - 0.0375).abs() < 1e-6);
        assert_eq!(range_resolution(0.0, c), 0.0);
        assert_eq!(range_resolution(-1.0, c), 0.0);
    }

    #[test]
    fn range_doppler_localizes_a_moving_target() {
        // A target in range bin r0 whose beat phase advances kd cycles over the
        // M chirps must land at (range r0, Doppler kd).
        let (m, n, r0, kd) = (32usize, 64usize, 11usize, 6usize);
        let frames: Vec<Vec<Complex>> = (0..m)
            .map(|chirp| {
                // Slow-time phase term places the target in Doppler bin kd.
                let slow = Complex::cis(2.0 * PI * kd as f64 * chirp as f64 / m as f64);
                (0..n)
                    .map(|i| {
                        // Fast-time beat tone places the target in range bin r0.
                        slow * Complex::cis(2.0 * PI * r0 as f64 * i as f64 / n as f64)
                    })
                    .collect()
            })
            .collect();
        let map = range_doppler(&frames);
        assert_eq!(map.len(), n);
        assert_eq!(map[0].len(), m);
        assert_eq!(peak_bin(&map[r0]), kd);
        // Coherent integration over N·M samples gives a peak of height N·M.
        assert!((map[r0][kd] - (n * m) as f64).abs() < 1e-6);
        // The target's row is the only one carrying energy at Doppler kd.
        for (r, row) in map.iter().enumerate()
        {
            if r != r0
            {
                assert!(row[kd] < 1e-6, "range bin {r} leaked at Doppler {kd}");
            }
        }
    }

    #[test]
    fn stationary_target_sits_at_zero_doppler() {
        let (m, n, r0) = (16usize, 32usize, 5usize);
        // Identical beat tone on every chirp ⇒ no chirp-to-chirp phase drift.
        let frames: Vec<Vec<Complex>> = (0..m)
            .map(|_| {
                (0..n)
                    .map(|i| Complex::cis(2.0 * PI * r0 as f64 * i as f64 / n as f64))
                    .collect()
            })
            .collect();
        let map = range_doppler(&frames);
        assert_eq!(peak_bin(&map[r0]), 0);
    }

    #[test]
    fn fmcw_rejects_non_power_of_two_and_ragged_input() {
        assert!(range_profile(&[Complex::zero(); 3]).is_empty());
        // Non-power-of-two chirp count.
        let three: Vec<Vec<Complex>> = (0..3).map(|_| vec![Complex::zero(); 4]).collect();
        assert!(range_doppler(&three).is_empty());
        // Non-power-of-two chirp length.
        let five: Vec<Vec<Complex>> = (0..4).map(|_| vec![Complex::zero(); 5]).collect();
        assert!(range_doppler(&five).is_empty());
        // Ragged chirps.
        let ragged = vec![vec![Complex::zero(); 4], vec![Complex::zero(); 8]];
        assert!(range_doppler(&ragged).is_empty());
    }
}
