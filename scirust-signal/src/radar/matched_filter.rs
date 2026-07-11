//! Matched filtering — pulse compression. Correlating a received signal with a
//! replica of the transmitted waveform maximises the output SNR and produces a
//! sharp peak at the echo delay, whose main lobe is far narrower than the
//! transmitted pulse: range resolution is set by the bandwidth, not the length.

use crate::complex::Complex;

/// Full **cross-correlation** of `signal` with `replica` — the matched-filter
/// response for that replica.
///
/// `r[lag] = Σ_k signal[k]·conj(replica[k − lag])`. The output has length
/// `signal.len() + replica.len() − 1`; index `j` corresponds to
/// `lag = j − (replica.len() − 1)`, so the zero-lag (full-overlap) term sits at
/// index `replica.len() − 1`. For the autocorrelation (`replica == signal`)
/// that term equals the signal energy. Returns an empty vector if either input
/// is empty.
pub fn cross_correlate(signal: &[Complex], replica: &[Complex]) -> Vec<Complex> {
    if signal.is_empty() || replica.is_empty()
    {
        return Vec::new();
    }
    let (m, n) = (signal.len(), replica.len());
    (0..m + n - 1)
        .map(|j| {
            let lag = j as isize - (n as isize - 1);
            signal
                .iter()
                .enumerate()
                .fold(Complex::zero(), |acc, (k, &s)| {
                    let idx = k as isize - lag;
                    if (0..n as isize).contains(&idx)
                    {
                        acc + s * replica[idx as usize].conj()
                    }
                    else
                    {
                        acc
                    }
                })
        })
        .collect()
}

/// The lag — the echo delay, in samples — of the correlation peak:
/// `argmax_j |r[j]| − (replica_len − 1)`. Applied to the output of
/// [`cross_correlate`] this locates an echo. `None` for an empty correlation.
pub fn peak_lag(correlation: &[Complex], replica_len: usize) -> Option<isize> {
    let (idx, _) = correlation
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.mag_sq().total_cmp(&b.1.mag_sq()))?;
    Some(idx as isize - (replica_len as isize - 1))
}

/// The **peak-to-sidelobe ratio** (linear) of a matched-filter response: the
/// peak magnitude over the largest magnitude outside a `±guard`-sample window
/// around the peak. Larger is cleaner (fewer false targets from range
/// sidelobes). `None` when no sample lies outside the guard window.
pub fn peak_to_sidelobe(correlation: &[Complex], guard: usize) -> Option<f64> {
    let (peak_idx, peak) = correlation
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.mag_sq().total_cmp(&b.1.mag_sq()))?;
    let mut max_side = 0.0_f64;
    let mut any = false;
    for (i, c) in correlation.iter().enumerate()
    {
        if (i as isize - peak_idx as isize).unsigned_abs() > guard
        {
            max_side = max_side.max(c.mag());
            any = true;
        }
    }
    if !any
    {
        return None;
    }
    Some(peak.mag() / max_side)
}

#[cfg(test)]
mod tests {
    use super::super::waveform::{barker_code, lfm_chirp};
    use super::*;

    fn to_complex(re: &[f64]) -> Vec<Complex> {
        re.iter().map(|&x| Complex::new(x, 0.0)).collect()
    }

    #[test]
    fn lfm_autocorrelation_peak_is_the_energy_and_the_main_lobe_compresses() {
        // n = 256 at fs = 10 MHz, B = 5 MHz ⇒ time-bandwidth product 128.
        let n = 256;
        let chirp = lfm_chirp(n, 5.0e6, 10.0e6);
        let r = cross_correlate(&chirp, &chirp);
        // The zero-lag term (index n−1) equals the pulse energy = n.
        assert!((r[n - 1].mag() - n as f64).abs() < 1e-6);
        assert_eq!(peak_lag(&r, n), Some(0));
        // The compressed −3 dB main lobe is a handful of samples wide (≈ fs/B),
        // far narrower than the 256-sample pulse — this is pulse compression.
        let half_power = r[n - 1].mag() / 2.0_f64.sqrt();
        let width = r.iter().filter(|c| c.mag() >= half_power).count();
        assert!(width < n / 20, "main lobe not compressed: {width} samples");
    }

    #[test]
    fn barker13_autocorrelation_peak_to_sidelobe_equals_the_code_length() {
        let code = to_complex(&barker_code(13).unwrap());
        let r = cross_correlate(&code, &code);
        assert!((r[12].mag() - 13.0).abs() < 1e-9); // peak = length
        // Every sidelobe magnitude is ≤ 1, so the peak-to-sidelobe ratio is 13.
        let psl = peak_to_sidelobe(&r, 0).unwrap();
        assert!((psl - 13.0).abs() < 1e-9, "PSL {psl} != 13");
    }

    #[test]
    fn matched_filter_locates_a_delayed_echo() {
        let n = 64;
        let chirp = lfm_chirp(n, 4.0e6, 10.0e6);
        // Embed the pulse at delay 100 in a longer, otherwise-empty record.
        let delay = 100usize;
        let mut received = vec![Complex::zero(); 400];
        for (k, &s) in chirp.iter().enumerate()
        {
            received[delay + k] = s;
        }
        let r = cross_correlate(&received, &chirp);
        assert_eq!(peak_lag(&r, n), Some(delay as isize));
    }

    #[test]
    fn cross_correlate_handles_empty_inputs() {
        assert!(cross_correlate(&[], &[Complex::zero()]).is_empty());
        assert!(cross_correlate(&[Complex::zero()], &[]).is_empty());
        assert!(peak_lag(&[], 1).is_none());
        assert!(peak_to_sidelobe(&[Complex::new(1.0, 0.0)], 5).is_none());
    }
}
