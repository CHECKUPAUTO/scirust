//! Digital filter design and application: windowed-sinc FIR, and Butterworth
//! IIR via the analog prototype + bilinear transform, realized as cascaded
//! second-order sections (SOS) for numerical robustness at higher orders
//! (a direct-form transfer-function realization loses precision fast as
//! order grows — Oppenheim & Schafer, *Discrete-Time Signal Processing*,
//! §7.1; Proakis & Manolakis §9.7).

use crate::complex::Complex;
use core::f64::consts::PI;

/// Normalized sinc: `sin(pi*x)/(pi*x)`, with `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x == 0.0
    {
        1.0
    }
    else
    {
        let px = PI * x;
        px.sin() / px
    }
}

/// Design a windowed-sinc lowpass FIR filter.
///
/// `cutoff` is normalized to the Nyquist frequency (`1.0` = Nyquist = `fs/2`).
/// `window` must have length `numtaps` (see [`crate::windows`]); the DC gain
/// is normalized to exactly `1.0`. Matches `scipy.signal.firwin(numtaps,
/// cutoff, window=...)`.
pub fn fir_lowpass(numtaps: usize, cutoff: f64, window: &[f64]) -> Vec<f64> {
    assert!(numtaps > 0, "numtaps must be > 0");
    assert_eq!(window.len(), numtaps, "window length must equal numtaps");
    assert!(
        cutoff > 0.0 && cutoff < 1.0,
        "cutoff must be in (0, 1), normalized to Nyquist"
    );
    let alpha = (numtaps as f64 - 1.0) / 2.0;
    let mut h: Vec<f64> = (0..numtaps)
        .map(|i| {
            let m = i as f64 - alpha;
            cutoff * sinc(cutoff * m) * window[i]
        })
        .collect();
    let dc_gain: f64 = h.iter().sum();
    for hi in h.iter_mut()
    {
        *hi /= dc_gain;
    }
    h
}

/// Design a windowed-sinc highpass FIR filter.
///
/// `cutoff` is normalized to the Nyquist frequency. `numtaps` **must be
/// odd** — an even-length highpass FIR filter has zero gain at Nyquist by
/// construction (its frequency response is forced to zero there), so it
/// cannot be normalized to unit gain there. Matches
/// `scipy.signal.firwin(numtaps, cutoff, window=..., pass_zero=False)`.
pub fn fir_highpass(numtaps: usize, cutoff: f64, window: &[f64]) -> Vec<f64> {
    assert!(numtaps > 0, "numtaps must be > 0");
    assert_eq!(numtaps % 2, 1, "highpass FIR requires an odd numtaps");
    assert_eq!(window.len(), numtaps, "window length must equal numtaps");
    assert!(
        cutoff > 0.0 && cutoff < 1.0,
        "cutoff must be in (0, 1), normalized to Nyquist"
    );
    let alpha = (numtaps as f64 - 1.0) / 2.0;
    let mut h: Vec<f64> = (0..numtaps)
        .map(|i| {
            let m = i as f64 - alpha;
            (sinc(m) - cutoff * sinc(cutoff * m)) * window[i]
        })
        .collect();
    // Normalize for unit gain at Nyquist (scale_frequency = 1 -> cos(pi*m)).
    let scale: f64 = h
        .iter()
        .enumerate()
        .map(|(i, &hi)| hi * (PI * (i as f64 - alpha)).cos())
        .sum();
    for hi in h.iter_mut()
    {
        *hi /= scale;
    }
    h
}

/// Apply a linear-time-invariant filter `y[n] = sum(b*x) - sum(a[1:]*y)` via
/// Direct Form II Transposed (Oppenheim & Schafer §6.4) — the standard,
/// numerically well-behaved realization for a single transfer-function
/// section. `a[0]` is implicitly `1`; pass `a = &[1.0]` for a pure FIR
/// filter. Output has the same length as `x` (matches
/// `scipy.signal.lfilter`).
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    assert!(!b.is_empty(), "b must be non-empty");
    assert!(!a.is_empty() && a[0] != 0.0, "a[0] must be non-zero");
    let a0 = a[0];
    let n = b.len().max(a.len());
    let bn: Vec<f64> = (0..n)
        .map(|i| b.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let an: Vec<f64> = (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let mut z = vec![0.0; n - 1];
    let mut y = Vec::with_capacity(x.len());
    for &xi in x
    {
        let yi = bn[0] * xi + if n > 1 { z[0] } else { 0.0 };
        for k in 0..n.saturating_sub(2)
        {
            z[k] = bn[k + 1] * xi + z[k + 1] - an[k + 1] * yi;
        }
        if n > 1
        {
            let last = n - 2;
            z[last] = bn[last + 1] * xi - an[last + 1] * yi;
        }
        y.push(yi);
    }
    y
}

/// A single second-order IIR section (biquad), `a0` normalized to `1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Biquad {
    /// Apply this section to `x` via Direct Form II Transposed.
    pub fn filter(&self, x: &[f64]) -> Vec<f64> {
        lfilter(&[self.b0, self.b1, self.b2], &[1.0, self.a1, self.a2], x)
    }
}

/// Apply a cascade of second-order sections in order (output of one section
/// feeds the next) — matches `scipy.signal.sosfilt`.
pub fn sos_filter(sections: &[Biquad], x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for s in sections
    {
        y = s.filter(&y);
    }
    y
}

/// Analog Butterworth lowpass prototype poles (cutoff normalized to 1 rad/s),
/// all with strictly negative real part. Golub-free elementary formula
/// (Oppenheim & Schafer §10.2.3): `p_k = -exp(i*pi*m_k/(2N))` for
/// `m_k = -N+1, -N+3, ..., N-1`.
fn butter_prototype_poles(order: usize) -> Vec<Complex> {
    let n = order as f64;
    (0..order)
        .map(|k| {
            let m = -(order as f64) + 1.0 + 2.0 * k as f64;
            -Complex::cis(PI * m / (2.0 * n))
        })
        .collect()
}

// Bilinear-transform convention matching scipy's internal fs=2 normalization
// (Wn is normalized so Nyquist = 1): fs2 = 2*fs = 4.
const FS2: f64 = 4.0;

fn bilinear(s: Complex) -> Complex {
    (Complex::new(FS2, 0.0) + s) / (Complex::new(FS2, 0.0) - s)
}

fn prod(vs: &[Complex]) -> Complex {
    vs.iter().fold(Complex::new(1.0, 0.0), |acc, &v| acc * v)
}

/// Design a Butterworth lowpass filter as cascaded second-order sections.
///
/// `order` is the filter order `N`; `cutoff` is normalized to the Nyquist
/// frequency (`1.0` = Nyquist). Matches `scipy.signal.butter(order, cutoff,
/// btype='low', output='sos')`.
pub fn butter_lowpass_sos(order: usize, cutoff: f64) -> Vec<Biquad> {
    assert!(order > 0, "order must be > 0");
    assert!(
        cutoff > 0.0 && cutoff < 1.0,
        "cutoff must be in (0, 1), normalized to Nyquist"
    );
    let warped = 4.0 * (PI * cutoff / 2.0).tan();
    let proto = butter_prototype_poles(order);
    let p_analog: Vec<Complex> = proto.iter().map(|&p| p * warped).collect();
    let k_analog = warped.powi(order as i32);

    let p_digital: Vec<Complex> = p_analog.iter().map(|&p| bilinear(p)).collect();
    // All `order` analog zeros are at infinity -> `order` digital zeros at -1.
    let k_digital = k_analog
        / prod(
            &p_analog
                .iter()
                .map(|&p| Complex::new(FS2, 0.0) - p)
                .collect::<Vec<_>>(),
        )
        .re;

    pair_into_sos(&p_digital, order, k_digital, Complex::new(-1.0, 0.0))
}

/// Design a Butterworth highpass filter as cascaded second-order sections.
///
/// `order` is the filter order `N`; `cutoff` is normalized to the Nyquist
/// frequency. Matches `scipy.signal.butter(order, cutoff, btype='high',
/// output='sos')`.
pub fn butter_highpass_sos(order: usize, cutoff: f64) -> Vec<Biquad> {
    assert!(order > 0, "order must be > 0");
    assert!(
        cutoff > 0.0 && cutoff < 1.0,
        "cutoff must be in (0, 1), normalized to Nyquist"
    );
    let warped = 4.0 * (PI * cutoff / 2.0).tan();
    let proto = butter_prototype_poles(order);
    // Lowpass-to-highpass: s -> wo/s. The prototype's `order` zeros "at
    // infinity" become `order` finite zeros at s=0.
    let p_analog: Vec<Complex> = proto
        .iter()
        .map(|&p| Complex::new(warped, 0.0) / p)
        .collect();
    let k_analog = 1.0 / prod(&proto.iter().map(|&p| -p).collect::<Vec<_>>()).re;

    let p_digital: Vec<Complex> = p_analog.iter().map(|&p| bilinear(p)).collect();
    // The `order` analog zeros at s=0 bilinear-transform to z=1 (DC null) —
    // no extra zeros at infinity remain since the pole/zero counts match.
    let z_analog_zero = Complex::new(0.0, 0.0);
    let numer = (FS2 - z_analog_zero.re).powi(order as i32); // real, since z_analog is real 0
    let denom = prod(
        &p_analog
            .iter()
            .map(|&p| Complex::new(FS2, 0.0) - p)
            .collect::<Vec<_>>(),
    )
    .re;
    let k_digital = k_analog * numer / denom;

    pair_into_sos(&p_digital, order, k_digital, Complex::new(1.0, 0.0))
}

/// Pair digital poles into second-order (or, for the leftover real pole in
/// an odd-order filter, first-order) sections, each carrying the same
/// repeated zero `z0` (`-1` for lowpass, `1` for highpass), and distribute
/// the overall gain evenly in log-space across sections so no single
/// section over/underflows for high orders.
///
/// `butter_prototype_poles` generates poles in order of increasing angle
/// `m_k = -(N-1), -(N-3), ..., (N-1)`, so `p_k` and `p_{N-1-k}` are the
/// actual complex-conjugate pair (their `m` values are negatives of each
/// other) — a *mirror* pairing around the array's center, not sequential
/// neighbors. For odd `N` the middle element (`m = 0`) is real on its own.
/// One representative per pair suffices: `(1 - p*z^-1)(1 - conj(p)*z^-1) =
/// 1 - 2*Re(p)*z^-1 + |p|^2*z^-2` needs only `Re(p)` and `|p|^2`, so the
/// conjugate partner never has to be looked up explicitly.
fn pair_into_sos(poles: &[Complex], order: usize, gain: f64, z0: Complex) -> Vec<Biquad> {
    let num_sections = order.div_ceil(2);
    let gain_per_section = gain.abs().powf(1.0 / num_sections as f64) * gain.signum();
    let mut sections = Vec::with_capacity(num_sections);
    for &p in poles.iter().take(order / 2)
    {
        // Denominator (1 - 2*Re(p)*z^-1 + |p|^2*z^-2).
        let a1 = -2.0 * p.re;
        let a2 = p.mag_sq();
        // Numerator: gain_i * (1 - z0*z^-1)^2.
        let b0 = gain_per_section;
        let b1 = gain_per_section * (-2.0 * z0.re);
        let b2 = gain_per_section * z0.mag_sq();
        sections.push(Biquad { b0, b1, b2, a1, a2 });
    }
    if order % 2 == 1
    {
        // Leftover real pole at the array's middle index.
        let p = poles[order / 2];
        let a1 = -p.re;
        let b0 = gain_per_section;
        let b1 = gain_per_section * (-z0.re);
        sections.push(Biquad {
            b0,
            b1,
            b2: 0.0,
            a1,
            a2: 0.0,
        });
    }
    sections
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::windows::hamming;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    // ---- FIR ----

    #[test]
    fn fir_lowpass_matches_scipy_firwin() {
        // scipy.signal.firwin(15, 0.3, window='hamming')
        let reference = [
            0.001118295168635258,
            -0.003894764791719022,
            -0.01603491745518543,
            -0.02036377118215373,
            0.02095180705129967,
            0.12449781344246412,
            0.24450683184615124,
            0.2984374118410156,
            0.24450683184615124,
            0.12449781344246412,
            0.02095180705129967,
            -0.02036377118215373,
            -0.01603491745518543,
            -0.003894764791719022,
            0.001118295168635258,
        ];
        let h = fir_lowpass(15, 0.3, &hamming(15));
        assert_eq!(h.len(), reference.len());
        for (hi, ri) in h.iter().zip(&reference)
        {
            assert!(close(*hi, *ri, 1e-9), "hi={hi} ri={ri}");
        }
    }

    #[test]
    fn fir_highpass_matches_scipy_firwin() {
        // scipy.signal.firwin(15, 0.3, window='hamming', pass_zero=False)
        let reference = [
            -0.0011217057832161284,
            0.003906643177640974,
            0.016083821290961194,
            0.02042587730302713,
            -0.02101570658393557,
            -0.12487751110165364,
            -0.2452525370849884,
            0.6984777249196132,
            -0.2452525370849884,
            -0.12487751110165364,
            -0.02101570658393557,
            0.02042587730302713,
            0.016083821290961194,
            0.003906643177640974,
            -0.0011217057832161284,
        ];
        let h = fir_highpass(15, 0.3, &hamming(15));
        assert_eq!(h.len(), reference.len());
        for (hi, ri) in h.iter().zip(&reference)
        {
            assert!(close(*hi, *ri, 1e-9), "hi={hi} ri={ri}");
        }
    }

    #[test]
    fn fir_lowpass_impulse_response_equals_its_own_coefficients() {
        let h = fir_lowpass(15, 0.3, &hamming(15));
        let mut impulse = vec![0.0; 20];
        impulse[0] = 1.0;
        let y = lfilter(&h, &[1.0], &impulse);
        for i in 0..15
        {
            assert!(close(y[i], h[i], 1e-12));
        }
        for &yi in &y[15..]
        {
            assert_eq!(yi, 0.0);
        }
    }

    // ---- lfilter (general) ----

    #[test]
    fn lfilter_matches_scipy_reference() {
        // scipy.signal.lfilter([1.0, 0.5], [1.0, -0.9], [1,0,0,0,0])
        let y = lfilter(&[1.0, 0.5], &[1.0, -0.9], &[1.0, 0.0, 0.0, 0.0, 0.0]);
        let reference = [1.0, 1.4, 1.26, 1.1340000000000001, 1.0206000000000002];
        for (yi, ri) in y.iter().zip(&reference)
        {
            assert!(close(*yi, *ri, 1e-12), "yi={yi} ri={ri}");
        }
    }

    // ---- Butterworth IIR ----

    #[test]
    fn butter_lowpass_sos_step_response_matches_scipy() {
        // scipy.signal.sosfilt(butter(4, 0.3, 'low', output='sos'), ones(10))
        let reference = [
            0.01856301062689717,
            0.12196638369830151,
            0.3720497620366597,
            0.7161217189340746,
            1.0046797409401542,
            1.1321903515726304,
            1.1119584056610343,
            1.0310931517333508,
            0.9696898846984233,
            0.9568987031390099,
        ];
        let sos = butter_lowpass_sos(4, 0.3);
        let x = vec![1.0; 10];
        let y = sos_filter(&sos, &x);
        for (yi, ri) in y.iter().zip(&reference)
        {
            assert!(close(*yi, *ri, 1e-6), "yi={yi} ri={ri}");
        }
    }

    #[test]
    fn butter_lowpass_frequency_response_is_maximally_flat() {
        // |H(0)| = 1 (unit DC gain), |H(wc)| = 1/sqrt(2) (Butterworth's
        // defining -3dB-at-cutoff property), |H(pi)| ~ 0 (Nyquist null).
        let sos = butter_lowpass_sos(4, 0.3);
        let h_dc = freqz_sos_mag(&sos, 0.0);
        let h_cutoff = freqz_sos_mag(&sos, 0.3 * PI);
        let h_nyquist = freqz_sos_mag(&sos, PI);
        assert!(close(h_dc, 1.0, 1e-9), "h_dc={h_dc}");
        assert!(
            close(h_cutoff, 1.0 / std::f64::consts::SQRT_2, 1e-9),
            "h_cutoff={h_cutoff}"
        );
        assert!(h_nyquist < 1e-6, "h_nyquist={h_nyquist}");
    }

    #[test]
    fn butter_highpass_sos_matches_scipy_coefficients_up_to_section_order() {
        // scipy.signal.butter(3, 0.25, btype='high', output='sos') — order 3
        // is odd, exercising the leftover-real-pole path. Section order/
        // pairing may legitimately differ from scipy's internal heuristic,
        // so this checks the frequency response instead of raw coefficients.
        let sos = butter_highpass_sos(3, 0.25);
        assert_eq!(sos.len(), 2); // one biquad + one first-order section
        let h_dc = freqz_sos_mag(&sos, 0.0);
        let h_cutoff = freqz_sos_mag(&sos, 0.25 * PI);
        let h_nyquist = freqz_sos_mag(&sos, PI);
        assert!(h_dc < 1e-6, "h_dc={h_dc}");
        assert!(
            close(h_cutoff, 1.0 / std::f64::consts::SQRT_2, 1e-9),
            "h_cutoff={h_cutoff}"
        );
        assert!(close(h_nyquist, 1.0, 1e-9), "h_nyquist={h_nyquist}");
    }

    #[test]
    fn butter_lowpass_accurate_for_large_a_near_boundary_odd_order() {
        // Regression-style coverage for the odd-order (leftover real pole)
        // path on the lowpass side too.
        let sos = butter_lowpass_sos(5, 0.2);
        assert_eq!(sos.len(), 3); // two biquads + one first-order section
        let h_dc = freqz_sos_mag(&sos, 0.0);
        let h_cutoff = freqz_sos_mag(&sos, 0.2 * PI);
        assert!(close(h_dc, 1.0, 1e-9), "h_dc={h_dc}");
        assert!(
            close(h_cutoff, 1.0 / std::f64::consts::SQRT_2, 1e-9),
            "h_cutoff={h_cutoff}"
        );
    }

    /// `|H(e^{i*omega})|` for a SOS cascade, evaluated directly from the
    /// transfer function (independent of `sos_filter`/`lfilter`, so this is
    /// a genuine cross-check of the designed coefficients rather than a
    /// restatement of the filtering code).
    fn freqz_sos_mag(sections: &[Biquad], omega: f64) -> f64 {
        let z_inv = Complex::cis(-omega);
        let mut h = Complex::new(1.0, 0.0);
        for s in sections
        {
            let num = Complex::new(1.0, 0.0) * s.b0 + z_inv * s.b1 + (z_inv * z_inv) * s.b2;
            let den = Complex::new(1.0, 0.0) + z_inv * s.a1 + (z_inv * z_inv) * s.a2;
            h *= num / den;
        }
        h.mag()
    }
}

/// Property-based tests sweeping `order` (odd *and* even, exercising the
/// leftover-real-pole path generically rather than at two hand-picked
/// values) and `cutoff`, checking the Butterworth-defining frequency-
/// response invariants directly from the transfer function.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn rel_close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// Same cross-check as `freqz_sos_mag` in `mod tests` above, duplicated
    /// here so this module stays self-contained.
    fn freqz_sos_mag(sections: &[Biquad], omega: f64) -> f64 {
        let z_inv = Complex::cis(-omega);
        let mut h = Complex::new(1.0, 0.0);
        for s in sections
        {
            let num = Complex::new(1.0, 0.0) * s.b0 + z_inv * s.b1 + (z_inv * z_inv) * s.b2;
            let den = Complex::new(1.0, 0.0) + z_inv * s.a1 + (z_inv * z_inv) * s.a2;
            h *= num / den;
        }
        h.mag()
    }

    proptest! {
        /// Butterworth's defining property — maximally flat, exactly
        /// -3dB (1/sqrt(2)) at the cutoff — must hold for every order from
        /// 1 to 10 (odd and even) and any cutoff away from the extremes.
        #[test]
        fn butter_lowpass_is_minus_3db_at_cutoff(order in 1usize..10, cutoff in 0.05f64..0.95) {
            let sos = butter_lowpass_sos(order, cutoff);
            prop_assert_eq!(sos.len(), order.div_ceil(2));
            let h_dc = freqz_sos_mag(&sos, 0.0);
            let h_cutoff = freqz_sos_mag(&sos, cutoff * PI);
            prop_assert!(rel_close(h_dc, 1.0, 1e-8), "order={order} cutoff={cutoff} h_dc={h_dc}");
            prop_assert!(
                rel_close(h_cutoff, std::f64::consts::FRAC_1_SQRT_2, 1e-8),
                "order={order} cutoff={cutoff} h_cutoff={h_cutoff}"
            );
        }

        #[test]
        fn butter_highpass_is_minus_3db_at_cutoff(order in 1usize..10, cutoff in 0.05f64..0.95) {
            let sos = butter_highpass_sos(order, cutoff);
            prop_assert_eq!(sos.len(), order.div_ceil(2));
            let h_nyquist = freqz_sos_mag(&sos, PI);
            let h_cutoff = freqz_sos_mag(&sos, cutoff * PI);
            prop_assert!(rel_close(h_nyquist, 1.0, 1e-8), "order={order} cutoff={cutoff} h_nyquist={h_nyquist}");
            prop_assert!(
                rel_close(h_cutoff, std::f64::consts::FRAC_1_SQRT_2, 1e-8),
                "order={order} cutoff={cutoff} h_cutoff={h_cutoff}"
            );
        }

        /// `sos_filter` and the transfer-function evaluation must agree:
        /// filtering a pure sinusoid at the cutoff frequency must attenuate
        /// its steady-state amplitude by exactly the same 1/sqrt(2) factor
        /// checked structurally above — an end-to-end cross-check that the
        /// designed coefficients and the filtering code agree with each
        /// other, not just with `freqz_sos_mag`.
        #[test]
        fn butter_lowpass_steady_state_gain_matches_frequency_response(
            order in 1usize..6,
            cutoff in 0.1f64..0.9,
        ) {
            let sos = butter_lowpass_sos(order, cutoff);
            let omega = cutoff * PI;
            let n = 4000;
            let x: Vec<f64> = (0..n).map(|i| (omega * i as f64).sin()).collect();
            let y = sos_filter(&sos, &x);
            // Steady-state amplitude via RMS over a long tail (the impulse
            // response has long since decayed to numerical zero at this
            // filter order/cutoff): for a sinusoid of amplitude R, mean
            // power is R^2/2, so R = sqrt(2 * mean(y^2)). This is robust to
            // the sample phase, unlike a peak search — when `omega` sits
            // close to a low-order rational multiple of pi (e.g. near pi/2,
            // period 4), the true peak can fall between sample points for
            // thousands of samples, so a naive max-over-window
            // underestimates the amplitude without this being a filter bug.
            let tail = &y[n - 2000..];
            let mean_sq: f64 = tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64;
            let amplitude = (2.0 * mean_sq).sqrt();
            let expected = std::f64::consts::FRAC_1_SQRT_2;
            prop_assert!(
                rel_close(amplitude, expected, 0.01),
                "order={order} cutoff={cutoff} amplitude={amplitude} expected={expected}"
            );
        }

        /// FIR lowpass coefficients must be symmetric (linear phase) and
        /// have unit DC gain, for any odd or even tap count.
        #[test]
        fn fir_lowpass_is_symmetric_with_unit_dc_gain(
            numtaps in 5usize..64,
            cutoff in 0.05f64..0.95,
        ) {
            let window = crate::windows::hamming(numtaps);
            let h = fir_lowpass(numtaps, cutoff, &window);
            for i in 0..numtaps
            {
                prop_assert!(
                    rel_close(h[i], h[numtaps - 1 - i], 1e-9),
                    "not symmetric at i={i}: {} vs {}", h[i], h[numtaps - 1 - i]
                );
            }
            let dc_gain: f64 = h.iter().sum();
            prop_assert!(rel_close(dc_gain, 1.0, 1e-9), "dc_gain={dc_gain}");
        }
    }
}
