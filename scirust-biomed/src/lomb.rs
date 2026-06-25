//! Frequency-domain HRV via the Lomb–Scargle periodogram.
//!
//! The RR tachogram is sampled at the (irregular) beat times, so a plain FFT
//! would need resampling and interpolation. The **Lomb–Scargle** periodogram
//! estimates spectral power directly on unevenly-sampled data, which is the
//! correct tool for HRV. The ratio of low-frequency (0.04–0.15 Hz) to
//! high-frequency (0.15–0.4 Hz) power, **LF/HF**, indexes sympatho-vagal balance.

/// Lomb–Scargle power at angular frequency `omega` for samples `values` taken
/// at times `times` (mean-subtracted internally).
pub fn lomb_scargle_power(times: &[f64], values: &[f64], omega: f64) -> f64 {
    let n = times.len();
    if n < 2 || omega <= 0.0
    {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    // Time offset tau so the sine/cosine sums decorrelate.
    let (mut s2, mut c2) = (0.0, 0.0);
    for &t in times
    {
        s2 += (2.0 * omega * t).sin();
        c2 += (2.0 * omega * t).cos();
    }
    let tau = 0.5 * s2.atan2(c2) / omega;

    let (mut num_c, mut den_c, mut num_s, mut den_s) = (0.0, 0.0, 0.0, 0.0);
    for (&t, &v) in times.iter().zip(values)
    {
        let arg = omega * (t - tau);
        let (sa, ca) = (arg.sin(), arg.cos());
        let dv = v - mean;
        num_c += dv * ca;
        den_c += ca * ca;
        num_s += dv * sa;
        den_s += sa * sa;
    }
    let cterm = if den_c > 1e-12
    {
        num_c * num_c / den_c
    }
    else
    {
        0.0
    };
    let sterm = if den_s > 1e-12
    {
        num_s * num_s / den_s
    }
    else
    {
        0.0
    };
    0.5 * (cterm + sterm)
}

/// Integrated Lomb–Scargle power over `[f_lo, f_hi]` Hz, sampled at `df` steps.
pub fn band_power(times: &[f64], values: &[f64], f_lo: f64, f_hi: f64, df: f64) -> f64 {
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut f = f_lo;
    let mut acc = 0.0;
    while f <= f_hi
    {
        acc += lomb_scargle_power(times, values, two_pi * f) * df;
        f += df;
    }
    acc
}

/// `(LF, HF, LF/HF)` from RR intervals (seconds). LF = 0.04–0.15 Hz, HF =
/// 0.15–0.4 Hz. Returns `LF/HF = ∞` if HF power is ~0.
pub fn lf_hf(rr_seconds: &[f64]) -> (f64, f64, f64) {
    if rr_seconds.len() < 4
    {
        return (0.0, 0.0, 0.0);
    }
    // Tachogram: value = RR, time = cumulative beat time.
    let mut t = 0.0;
    let times: Vec<f64> = rr_seconds
        .iter()
        .map(|&rr| {
            t += rr;
            t
        })
        .collect();
    let lf = band_power(&times, rr_seconds, 0.04, 0.15, 0.005);
    let hf = band_power(&times, rr_seconds, 0.15, 0.40, 0.005);
    let ratio = if hf > 1e-12 { lf / hf } else { f64::INFINITY };
    (lf, hf, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn lomb_scargle_peaks_at_the_injected_frequency() {
        // Unevenly-sampled 0.25 Hz sinusoid.
        let mut times = Vec::new();
        let mut t = 0.0;
        for k in 0..200
        {
            t += 0.7 + 0.2 * ((k as f64) * 1.3).sin().abs(); // jittered spacing
            times.push(t);
        }
        let values: Vec<f64> = times.iter().map(|&t| (2.0 * PI * 0.25 * t).sin()).collect();
        let at_025 = lomb_scargle_power(&times, &values, 2.0 * PI * 0.25);
        let at_010 = lomb_scargle_power(&times, &values, 2.0 * PI * 0.10);
        assert!(
            at_025 > 5.0 * at_010,
            "power@0.25 {at_025} vs @0.10 {at_010}"
        );
    }

    #[test]
    fn hf_oscillation_gives_low_lf_hf_ratio() {
        // RR oscillating at 0.25 Hz (HF, respiratory) -> LF/HF < 1.
        let base = 0.8;
        let mut t = 0.0;
        let rr: Vec<f64> = (0..256)
            .map(|_| {
                let rr = base + 0.04 * (2.0 * PI * 0.25 * t).sin();
                t += rr;
                rr
            })
            .collect();
        let (_lf, _hf, ratio) = lf_hf(&rr);
        assert!(
            ratio < 1.0,
            "LF/HF {ratio} should be < 1 for HF-dominated HRV"
        );
    }

    #[test]
    fn lf_oscillation_gives_high_lf_hf_ratio() {
        // RR oscillating at 0.10 Hz (LF, baroreflex) -> LF/HF > 1.
        let base = 0.8;
        let mut t = 0.0;
        let rr: Vec<f64> = (0..256)
            .map(|_| {
                let rr = base + 0.04 * (2.0 * PI * 0.10 * t).sin();
                t += rr;
                rr
            })
            .collect();
        let (_lf, _hf, ratio) = lf_hf(&rr);
        assert!(
            ratio > 1.0,
            "LF/HF {ratio} should be > 1 for LF-dominated HRV"
        );
    }

    #[test]
    fn lomb_power_matches_analytic_value_for_even_sampling() {
        // Evenly-spaced pure sinusoid sampled over an integer number of periods.
        // For N samples at amplitude A on a uniform grid spanning whole periods,
        //   sum cos^2 = sum sin^2 = N/2, cross term = 0, so the Lomb power at the
        //   true angular frequency is 0.5 * (A*N/2)^2 / (N/2) = A^2 * N / 4 EXACTLY.
        // dt = 0.5 s, M = 10 samples/period -> f0 = 1/(M*dt) = 0.2 Hz; N = 400 = 40 periods.
        let dt = 0.5;
        let n = 400usize;
        let a = 2.0;
        let f0 = 0.2;
        let w0 = 2.0 * PI * f0;
        let times: Vec<f64> = (0..n).map(|j| j as f64 * dt).collect();
        let values: Vec<f64> = times.iter().map(|&t| a * (w0 * t).cos()).collect();
        let expected = a * a * n as f64 / 4.0; // = 400
        let p = lomb_scargle_power(&times, &values, w0);
        assert!(
            (p - expected).abs() < 1e-9,
            "LS power {p}, expected {expected}"
        );
    }

    #[test]
    fn lomb_power_subtracts_the_mean() {
        // A large DC offset and a phase shift must not change the power: the
        // periodogram is computed on the mean-subtracted signal. Same A, N as
        // above -> still A^2 * N / 4 = 400.
        let dt = 0.5;
        let n = 400usize;
        let a = 2.0;
        let w0 = 2.0 * PI * 0.2;
        let times: Vec<f64> = (0..n).map(|j| j as f64 * dt).collect();
        let values: Vec<f64> = times.iter().map(|&t| a * (w0 * t).sin() + 137.0).collect();
        let p = lomb_scargle_power(&times, &values, w0);
        assert!((p - 400.0).abs() < 1e-7, "LS power {p}, expected 400");
    }

    #[test]
    fn lomb_power_degenerate_inputs_are_zero() {
        // Fewer than two samples, and a non-positive angular frequency, both
        // return 0 rather than dividing by zero or producing NaN.
        assert_eq!(lomb_scargle_power(&[1.0], &[3.0], 1.0), 0.0);
        assert_eq!(lomb_scargle_power(&[], &[], 1.0), 0.0);
        assert_eq!(
            lomb_scargle_power(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], 0.0),
            0.0
        );
    }

    #[test]
    fn band_power_oracles() {
        let times: Vec<f64> = (0..400).map(|j| j as f64 * 0.5).collect();
        let w0 = 2.0 * PI * 0.2;
        let values: Vec<f64> = times.iter().map(|&t| 2.0 * (w0 * t).cos()).collect();

        // A constant signal carries no spectral power -> the integral is 0.
        let flat = vec![0.77; 400];
        let bp_flat = band_power(&times, &flat, 0.04, 0.40, 0.005);
        assert!(bp_flat.abs() < 1e-12, "band_power(flat) = {bp_flat}");

        // A single-bin band [f0, f0] evaluates the periodogram once and scales by
        // df, so it equals power(f0) * df exactly. power(0.2) = 400 -> 400*0.01 = 4.
        let bp_single = band_power(&times, &values, 0.2, 0.2, 0.01);
        assert!(
            (bp_single - 4.0).abs() < 1e-7,
            "single-bin band_power {bp_single}"
        );

        // An inverted band (f_lo > f_hi) integrates nothing -> 0.
        assert_eq!(band_power(&times, &values, 0.30, 0.10, 0.005), 0.0);
    }

    #[test]
    fn lf_hf_requires_at_least_four_intervals() {
        // Too few intervals to estimate the spectrum -> all zeros (not NaN/Inf).
        assert_eq!(lf_hf(&[0.8, 0.9, 0.85]), (0.0, 0.0, 0.0));
        // With no high-frequency power the ratio saturates to +inf rather than
        // dividing by zero: a constant tachogram has zero power in every band.
        let (lf, hf, ratio) = lf_hf(&[0.8; 64]);
        assert!(lf.abs() < 1e-12 && hf.abs() < 1e-12, "lf {lf} hf {hf}");
        assert!(ratio.is_infinite() && ratio > 0.0, "ratio {ratio}");
    }
}
