//! Matched filtering — pulse compression. Correlating a received signal with a
//! replica of the transmitted waveform maximises the output SNR and produces a
//! sharp peak at the echo delay, whose main lobe is far narrower than the
//! transmitted pulse: range resolution is set by the bandwidth, not the length.

use crate::complex::Complex;
use crate::fft::{fft, ifft};

/// Full **cross-correlation** of `signal` with `replica` — the matched-filter
/// response for that replica.
///
/// `r[lag] = Σ_k signal[k]·conj(replica[k − lag])`. The output has length
/// `signal.len() + replica.len() − 1`; index `j` corresponds to
/// `lag = j − (replica.len() − 1)`, so the zero-lag (full-overlap) term sits at
/// index `replica.len() − 1`. For the autocorrelation (`replica == signal`)
/// that term equals the signal energy. Returns an empty vector if either input
/// is empty.
///
/// Computed via zero-padded FFT (`O((m+n)·log(m+n))`) rather than the naive
/// `O(m·n)` double loop that direct summation requires — the difference
/// matters for radar records, where `signal` (the received trace) can be
/// orders of magnitude longer than `replica` (the transmitted waveform).
pub fn cross_correlate(signal: &[Complex], replica: &[Complex]) -> Vec<Complex> {
    if signal.is_empty() || replica.is_empty()
    {
        return Vec::new();
    }
    let (m, n) = (signal.len(), replica.len());
    let out_len = m + n - 1;
    let fft_len = out_len.next_power_of_two();

    let mut a = vec![Complex::zero(); fft_len];
    a[..m].copy_from_slice(signal);
    let mut b = vec![Complex::zero(); fft_len];
    b[..n].copy_from_slice(replica);

    fft(&mut a);
    fft(&mut b);
    // Cross-power spectrum: FFT(signal) · conj(FFT(replica)); its inverse FFT
    // is the circular cross-correlation (Wiener-Khinchin for correlation
    // rather than autocorrelation).
    for (ai, &bi) in a.iter_mut().zip(b.iter())
    {
        *ai *= bi.conj();
    }
    ifft(&mut a);

    // `a[k]` now holds the circular correlation at lag `k` for `k` in
    // `0..m` (non-negative lags) and at lag `k − fft_len` for `k` in
    // `fft_len − (n − 1) .. fft_len` (negative lags, wrapped). Since
    // `fft_len >= out_len = m + n − 1`, these two bands never overlap, so
    // reading them back out into linear (non-circular) order is exact.
    (0..out_len)
        .map(|j| {
            let lag = j as isize - (n as isize - 1);
            let circ_idx = if lag >= 0
            {
                lag as usize
            }
            else
            {
                (fft_len as isize + lag) as usize
            };
            a[circ_idx]
        })
        .collect()
}

/// Reference O(m·n) direct-summation implementation of [`cross_correlate`],
/// kept only to differentially test the FFT-based version above against.
#[cfg(test)]
fn cross_correlate_direct(signal: &[Complex], replica: &[Complex]) -> Vec<Complex> {
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

    #[test]
    fn single_sample_inputs_do_not_panic() {
        let a = [Complex::new(2.0, -1.0)];
        let b = [Complex::new(3.0, 0.5)];
        let r = cross_correlate(&a, &b);
        assert_eq!(r.len(), 1);
        let expected = a[0] * b[0].conj();
        assert!((r[0].re - expected.re).abs() < 1e-9);
        assert!((r[0].im - expected.im).abs() < 1e-9);
    }
}

/// Differential test: the FFT-based [`cross_correlate`] must match the naive
/// O(m·n) [`cross_correlate_direct`] on arbitrary (non-power-of-two-length,
/// unequal-length) inputs — the lag bookkeeping around the zero-padded FFT
/// is exactly the kind of off-by-one that unit tests on "nice" sizes (powers
/// of two, matched lengths) can miss.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn complex_vec(
        len: impl Into<proptest::collection::SizeRange>,
    ) -> impl Strategy<Value = Vec<Complex>> {
        proptest::collection::vec((-10.0f64..10.0, -10.0f64..10.0), len)
            .prop_map(|v| v.into_iter().map(|(re, im)| Complex::new(re, im)).collect())
    }

    proptest! {
        #[test]
        fn fft_based_matches_direct_reference(
            signal in complex_vec(1..37),
            replica in complex_vec(1..23),
        ) {
            let fast = cross_correlate(&signal, &replica);
            let direct = cross_correlate_direct(&signal, &replica);
            prop_assert_eq!(fast.len(), direct.len());
            for (f, d) in fast.iter().zip(direct.iter()) {
                prop_assert!((f.re - d.re).abs() < 1e-6, "re mismatch: {f:?} vs {d:?}");
                prop_assert!((f.im - d.im).abs() < 1e-6, "im mismatch: {f:?} vs {d:?}");
            }
        }
    }
}
