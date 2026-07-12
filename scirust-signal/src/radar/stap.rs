//! Space-time adaptive processing (STAP) for airborne radar.
//!
//! An airborne radar sees the ground as clutter whose Doppler is *coupled to
//! angle*: because the platform moves, a clutter patch at azimuth `θ` returns at
//! a Doppler set by the along-track velocity component, `f_d = β·f_s` where
//! `f_s = (d/λ)·sin θ` is the normalised spatial frequency. In the joint
//! angle-Doppler plane the clutter therefore collapses onto a one-dimensional
//! **ridge**, and a slow-moving target buried *under* the clutter in range and
//! Doppler is nonetheless *separated from it in the 2-D plane* — it sits off the
//! ridge. A filter that adapts jointly across the `N` array elements **and** the
//! `M` pulses of a coherent processing interval can place a null along that ridge
//! while keeping unit gain on the target: this is STAP, and it detects movers no
//! 1-D (angle-only or Doppler-only) filter can.
//!
//! The space-time snapshot stacks the `N`-element spatial snapshots of `M`
//! successive pulses into one `NM`-vector; the joint steering vector is the
//! Kronecker product `s = b(f_d) ⊗ a(f_s)` of the temporal and spatial steering
//! vectors. Given the interference-plus-noise covariance `R` the optimal
//! (minimum-variance-distortionless-response) weight is
//! `w = R⁻¹ s / (sᴴ R⁻¹ s)`, and the delivered SINR is `σ_t²·sᴴ R⁻¹ s` — deep on
//! the ridge, near the full `NM` coherent gain off it. Built on the crate's
//! [`Complex`](crate::complex::Complex) and the shared complex matrix inverse
//! from [`super::doa`]; dependency-free.

use super::doa::invert;
use crate::complex::Complex;
use std::f64::consts::PI;

/// The normalised **spatial frequency** `f_s = (d/λ)·sin θ` of a ULA for a source
/// at `theta` (rad from broadside), element spacing `spacing` (wavelengths). A
/// half-wavelength array maps `±90°` to `±0.5`.
pub fn spatial_frequency(theta: f64, spacing: f64) -> f64 {
    spacing * theta.sin()
}

/// The clutter **ridge Doppler** `f_d = β·f_s` for a side-looking airborne array:
/// the normalised Doppler at which ground clutter of spatial frequency `fs`
/// returns. `beta` is the platform's along-track motion per pulse in units of the
/// element spacing (`β = 1` for the classic side-looking, half-wavelength,
/// one-element-per-pulse geometry, giving a 45° ridge).
pub fn clutter_ridge_doppler(fs: f64, beta: f64) -> f64 {
    beta * fs
}

/// The **space-time steering vector** `s = b(f_d) ⊗ a(f_s)`, length
/// `n_elements·n_pulses`, ordered pulse-major: `s[m·N + n] = exp(j2π(f_d·m +
/// f_s·n))`. Unit-magnitude entries, so `|s|² = NM`.
pub fn space_time_steering(
    spatial_freq: f64,
    doppler_freq: f64,
    n_elements: usize,
    n_pulses: usize,
) -> Vec<Complex> {
    let mut s = Vec::with_capacity(n_elements * n_pulses);
    for m in 0..n_pulses
    {
        for n in 0..n_elements
        {
            let phase = 2.0 * PI * (doppler_freq * m as f64 + spatial_freq * n as f64);
            s.push(Complex::cis(phase));
        }
    }
    s
}

/// The `NM × NM` **interference-plus-noise covariance**
/// `R = σ_n²·I + Σ_c P_c·s_c s_cᴴ`, built from white noise `noise_power` (`σ_n²`)
/// and a set of ground-clutter `patches`, each `(f_s, P_c)` a spatial frequency
/// and power. Every patch sits on the ridge, its Doppler `β·f_s` supplied by
/// `beta`. This is the covariance the adaptive weight inverts.
#[allow(clippy::needless_range_loop)]
pub fn clutter_covariance(
    n_elements: usize,
    n_pulses: usize,
    patches: &[(f64, f64)],
    beta: f64,
    noise_power: f64,
) -> Vec<Vec<Complex>> {
    let dim = n_elements * n_pulses;
    let mut r = vec![vec![Complex::zero(); dim]; dim];
    for i in 0..dim
    {
        r[i][i] = Complex::new(noise_power, 0.0);
    }
    for &(fs, power) in patches
    {
        let fd = clutter_ridge_doppler(fs, beta);
        let s = space_time_steering(fs, fd, n_elements, n_pulses);
        for i in 0..dim
        {
            for j in 0..dim
            {
                let outer = s[i] * s[j].conj();
                r[i][j] += Complex::new(outer.re * power, outer.im * power);
            }
        }
    }
    r
}

/// `R⁻¹ s` together with the real Hermitian form `sᴴ R⁻¹ s`. `None` on a
/// dimension mismatch or a singular covariance.
#[allow(clippy::needless_range_loop)]
fn rinv_apply(r: &[Vec<Complex>], s: &[Complex]) -> Option<(Vec<Complex>, f64)> {
    let n = r.len();
    if n == 0 || s.len() != n
    {
        return None;
    }
    let rinv = invert(r.to_vec())?;
    let mut y = vec![Complex::zero(); n];
    for i in 0..n
    {
        let mut acc = Complex::zero();
        for j in 0..n
        {
            acc += rinv[i][j] * s[j];
        }
        y[i] = acc;
    }
    let mut q = Complex::zero();
    for i in 0..n
    {
        q += s[i].conj() * y[i];
    }
    Some((y, q.re))
}

/// The **adaptive (MVDR/SMI) weight** `w = R⁻¹ s / (sᴴ R⁻¹ s)`: minimum output
/// power subject to unit gain toward `steering`. Empty on a dimension mismatch or
/// a singular covariance.
pub fn adaptive_weights(r: &[Vec<Complex>], steering: &[Complex]) -> Vec<Complex> {
    match rinv_apply(r, steering)
    {
        Some((y, q)) if q.abs() > 1e-300 =>
        {
            let inv_q = 1.0 / q;
            y.iter()
                .map(|&yi| Complex::new(yi.re * inv_q, yi.im * inv_q))
                .collect()
        },
        _ => Vec::new(),
    }
}

/// The **optimal output SINR** `σ_t²·(sᴴ R⁻¹ s)` a target of power `target_power`
/// achieves against the interference-plus-noise covariance `r` at look direction
/// `steering`. For white noise `R = σ_n²·I` this is `target_power·NM/σ_n²` — the
/// full coherent gain — and it notches deeply where `steering` lies on the
/// clutter ridge. `0` on a dimension mismatch or a singular covariance.
pub fn optimal_sinr(r: &[Vec<Complex>], steering: &[Complex], target_power: f64) -> f64 {
    match rinv_apply(r, steering)
    {
        Some((_, q)) => target_power * q.max(0.0),
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn steering_is_the_space_time_kronecker_product() {
        let (nn, mm) = (4usize, 6usize);
        let (fs, fd) = (0.15, 0.3);
        let s = space_time_steering(fs, fd, nn, mm);
        assert_eq!(s.len(), nn * mm);
        // |s|² = NM (unit-magnitude entries).
        let norm_sq: f64 = s.iter().map(|z| z.mag_sq()).sum();
        assert!((norm_sq - (nn * mm) as f64).abs() < 1e-9);
        // Factorises as b(f_d) ⊗ a(f_s): s[m·N+n] = b[m]·a[n].
        for m in 0..mm
        {
            for n in 0..nn
            {
                let expected =
                    Complex::cis(2.0 * PI * fd * m as f64) * Complex::cis(2.0 * PI * fs * n as f64);
                let got = s[m * nn + n];
                assert!((got.re - expected.re).abs() < 1e-9);
                assert!((got.im - expected.im).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn white_noise_weight_is_the_matched_filter() {
        let (nn, mm) = (3usize, 4usize);
        let dim = nn * mm;
        let s = space_time_steering(0.1, 0.2, nn, mm);
        // R = σ²·I (no clutter).
        let sigma2 = 2.0;
        let r = clutter_covariance(nn, mm, &[], 1.0, sigma2);
        let w = adaptive_weights(&r, &s);
        assert_eq!(w.len(), dim);
        // The adaptive weight collapses to the matched filter w = s / NM.
        for i in 0..dim
        {
            let e = Complex::new(s[i].re / dim as f64, s[i].im / dim as f64);
            assert!((w[i].re - e.re).abs() < 1e-9 && (w[i].im - e.im).abs() < 1e-9);
        }
        // Unit gain toward the look direction: wᴴ s = 1.
        let mut g = Complex::zero();
        for i in 0..dim
        {
            g += w[i].conj() * s[i];
        }
        assert!((g.re - 1.0).abs() < 1e-9 && g.im.abs() < 1e-9);
        // Optimal SINR = P·NM/σ².
        let sinr = optimal_sinr(&r, &s, 5.0);
        assert!((sinr - 5.0 * dim as f64 / sigma2).abs() < 1e-6);
    }

    /// A strong clutter ridge: patches across spatial frequency at CNR ~20 dB.
    fn ridge_covariance(nn: usize, mm: usize, beta: f64) -> Vec<Vec<Complex>> {
        let patches: Vec<(f64, f64)> = (-5..=5).map(|k| (k as f64 * 0.1, 100.0)).collect();
        clutter_covariance(nn, mm, &patches, beta, 1.0)
    }

    #[test]
    fn clutter_notch_suppresses_endo_clutter_targets() {
        let (nn, mm) = (4usize, 8usize);
        let dim = nn * mm;
        let r = ridge_covariance(nn, mm, 1.0);
        // Off the ridge (f_s=0, f_d=0.25): buried in clutter Doppler at some other
        // angle, but clear of the ridge in the joint plane.
        let s_off = space_time_steering(0.0, 0.25, nn, mm);
        // On the ridge (f_s=0, f_d=β·0=0): sits on a clutter patch.
        let s_on = space_time_steering(0.0, 0.0, nn, mm);
        let sinr_off = optimal_sinr(&r, &s_off, 1.0);
        let sinr_on = optimal_sinr(&r, &s_on, 1.0);
        // The endo-clutter target is deeply notched relative to the exo-clutter one.
        assert!(sinr_off > 10.0 * sinr_on, "off={sinr_off} on={sinr_on}");
        // The off-ridge target keeps a large fraction of the clutter-free gain NM.
        assert!(sinr_off > 0.3 * dim as f64, "off={sinr_off}");
    }

    #[test]
    fn weight_nulls_the_clutter_it_competes_with() {
        let (nn, mm) = (4usize, 8usize);
        let r = ridge_covariance(nn, mm, 1.0);
        // Design the filter for an off-ridge target.
        let s_t = space_time_steering(0.0, 0.25, nn, mm);
        let w = adaptive_weights(&r, &s_t);
        // Unit gain toward the target.
        let mut g_t = Complex::zero();
        for i in 0..w.len()
        {
            g_t += w[i].conj() * s_t[i];
        }
        assert!((g_t.re - 1.0).abs() < 1e-6 && g_t.im.abs() < 1e-6);
        // Deep suppression toward a clutter patch on the ridge (f_s=0.3 → f_d=0.3).
        let s_c = space_time_steering(0.3, 0.3, nn, mm);
        let mut g_c = Complex::zero();
        for i in 0..w.len()
        {
            g_c += w[i].conj() * s_c[i];
        }
        assert!(
            g_c.mag() < 0.1 * g_t.mag(),
            "clutter gain {} vs target {}",
            g_c.mag(),
            g_t.mag()
        );
    }

    #[test]
    fn sinr_minimum_falls_on_the_clutter_ridge() {
        let (nn, mm) = (4usize, 8usize);
        let r = ridge_covariance(nn, mm, 1.0);
        // Fix the angle at broadside (f_s=0 ⇒ ridge Doppler 0); sweep Doppler.
        let fd_grid = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4];
        let sinrs: Vec<f64> = fd_grid
            .iter()
            .map(|&fd| optimal_sinr(&r, &space_time_steering(0.0, fd, nn, mm), 1.0))
            .collect();
        // The minimum is at f_d=0 (index 4), the ridge Doppler for f_s=0.
        let min_idx = sinrs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(min_idx, 4, "SINR notch not on the ridge: {sinrs:?}");
    }

    #[test]
    fn ridge_and_spatial_frequency_relations() {
        // Ridge Doppler is β times the spatial frequency.
        assert!((clutter_ridge_doppler(0.2, 1.0) - 0.2).abs() < 1e-12);
        assert!((clutter_ridge_doppler(0.2, 2.0) - 0.4).abs() < 1e-12);
        assert!((clutter_ridge_doppler(-0.15, 1.0) + 0.15).abs() < 1e-12);
        // f_s = (d/λ)·sin θ: broadside is zero; a half-wavelength array maps ±90°
        // to ±0.5.
        assert!(spatial_frequency(0.0, 0.5).abs() < 1e-15);
        assert!((spatial_frequency(FRAC_PI_2, 0.5) - 0.5).abs() < 1e-12);
        assert!((spatial_frequency(-FRAC_PI_2, 0.5) + 0.5).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        // Dimension mismatch ⇒ empty weights, zero SINR.
        let r = clutter_covariance(2, 2, &[], 1.0, 1.0); // 4×4
        let bad = vec![Complex::zero(); 3];
        assert!(adaptive_weights(&r, &bad).is_empty());
        assert_eq!(optimal_sinr(&r, &bad, 1.0), 0.0);
        // Empty covariance.
        assert!(adaptive_weights(&[], &[]).is_empty());
        // A singular (all-zero) covariance returns safely, no NaN.
        let singular = vec![vec![Complex::zero(); 2]; 2];
        let s = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        assert!(adaptive_weights(&singular, &s).is_empty());
        assert_eq!(optimal_sinr(&singular, &s, 1.0), 0.0);
    }
}
