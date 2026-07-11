//! Radar pulse-compression waveforms: the linear-FM (chirp) and Barker phase
//! codes. Both transmit a long pulse for energy, then recover fine range
//! resolution by matched filtering ([`super::matched_filter`]) — resolution set
//! by the bandwidth, not the pulse length.

use crate::complex::Complex;
use std::f64::consts::PI;

/// A complex-baseband **linear-FM (chirp)** pulse of `n` samples at
/// `sample_rate` Hz whose instantaneous frequency sweeps linearly across a
/// total `bandwidth` Hz, centred at baseband.
///
/// The pulse duration is `T = n / sample_rate`, the chirp rate `K = B / T`, and
/// sample `k` is `exp(j·π·K·t_k²)` with `t_k = (k − (n−1)/2) / sample_rate`, so
/// the instantaneous frequency runs from `−B/2` to `+B/2`. Unit amplitude, so
/// the pulse energy is `n`. Returns an empty vector for `n = 0` or a
/// non-positive `sample_rate`.
pub fn lfm_chirp(n: usize, bandwidth: f64, sample_rate: f64) -> Vec<Complex> {
    if n == 0 || !sample_rate.is_finite() || sample_rate <= 0.0
    {
        return Vec::new();
    }
    let t_pulse = n as f64 / sample_rate;
    let rate = bandwidth / t_pulse; // chirp rate K (Hz/s)
    let centre = (n as f64 - 1.0) / 2.0;
    (0..n)
        .map(|k| {
            let t = (k as f64 - centre) / sample_rate;
            Complex::cis(PI * rate * t * t)
        })
        .collect()
}

/// The **Barker code** of the given `length`, as `±1` samples, or `None` when
/// no Barker code of that length exists — they are known only for lengths 2, 3,
/// 4, 5, 7, 11 and 13.
///
/// Barker codes are the only binary phase codes whose aperiodic autocorrelation
/// sidelobes never exceed `1` in magnitude, giving a peak-to-sidelobe ratio
/// equal to the code length (the matched-filter property the tests check).
pub fn barker_code(length: usize) -> Option<Vec<f64>> {
    let code: &[i8] = match length
    {
        2 => &[1, -1],
        3 => &[1, 1, -1],
        4 => &[1, 1, -1, 1],
        5 => &[1, 1, 1, -1, 1],
        7 => &[1, 1, 1, -1, -1, 1, -1],
        11 => &[1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        13 => &[1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
        _ => return None,
    };
    Some(code.iter().map(|&c| f64::from(c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barker_codes_have_the_standard_sequences_and_lengths() {
        assert_eq!(barker_code(2).unwrap(), vec![1.0, -1.0]);
        assert_eq!(barker_code(3).unwrap(), vec![1.0, 1.0, -1.0]);
        assert_eq!(barker_code(13).unwrap().len(), 13);
        assert!(barker_code(6).is_none());
        assert!(barker_code(0).is_none());
        // Every entry is ±1.
        for &c in &barker_code(13).unwrap()
        {
            assert!(c == 1.0 || c == -1.0);
        }
    }

    #[test]
    fn lfm_chirp_has_unit_amplitude_and_the_right_length() {
        let chirp = lfm_chirp(128, 2.0e6, 8.0e6);
        assert_eq!(chirp.len(), 128);
        for c in &chirp
        {
            assert!((c.mag() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn lfm_instantaneous_frequency_sweeps_the_whole_band() {
        // The per-sample phase increment s[k+1]·conj(s[k]) has phase ≈ 2π·f/fs,
        // so the instantaneous frequency runs from ≈ −B/2 to ≈ +B/2.
        let (n, b, fs) = (1024usize, 2.0e6, 8.0e6);
        let chirp = lfm_chirp(n, b, fs);
        let inst_freq = |k: usize| {
            let d = chirp[k + 1] * chirp[k].conj();
            d.phase() * fs / (2.0 * PI)
        };
        assert!((inst_freq(0) + b / 2.0).abs() < b * 0.02, "start off band");
        assert!(
            (inst_freq(n - 2) - b / 2.0).abs() < b * 0.02,
            "end off band"
        );
    }

    #[test]
    fn lfm_chirp_rejects_bad_parameters() {
        assert!(lfm_chirp(0, 1.0, 1.0).is_empty());
        assert!(lfm_chirp(8, 1.0, 0.0).is_empty());
        assert!(lfm_chirp(8, 1.0, -5.0).is_empty());
    }
}
