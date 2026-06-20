//! Output-only (operational) modal analysis.
//!
//! Real structures are excited by unmeasured ambient loads (wind, traffic), so
//! only the response is available. The response **power spectral density** still
//! peaks at the natural frequencies, and the **Modal Assurance Criterion** (MAC)
//! correlates mode shapes — a MAC drop between a baseline and a current shape is
//! a sensitive, localized damage indicator.

use scirust_signal::{Complex, fft_real};

/// Power spectral density `|X(f)|²` of a response (`signal.len()` a power of two).
pub fn power_spectral_density(signal: &[f64]) -> Vec<f64> {
    fft_real(signal).iter().map(Complex::mag_sq).collect()
}

/// Modal Assurance Criterion between two mode-shape vectors:
/// `|φa·φb|² / ((φa·φa)(φb·φb))`. `1` = identical mode (up to scale), `0` =
/// orthogonal. Returns 0 if either vector is ~0 or lengths differ.
pub fn mac(phi_a: &[f64], phi_b: &[f64]) -> f64 {
    if phi_a.len() != phi_b.len()
    {
        return 0.0;
    }
    let dot: f64 = phi_a.iter().zip(phi_b).map(|(a, b)| a * b).sum();
    let na: f64 = phi_a.iter().map(|x| x * x).sum();
    let nb: f64 = phi_b.iter().map(|x| x * x).sum();
    if na < 1e-30 || nb < 1e-30
    {
        return 0.0;
    }
    dot * dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::natural_frequencies;

    #[test]
    fn mac_is_one_for_scaled_and_zero_for_orthogonal() {
        let phi = [1.0, 0.5, -0.3, 0.8];
        assert!((mac(&phi, &phi) - 1.0).abs() < 1e-12);
        // Scale invariance.
        let scaled: Vec<f64> = phi.iter().map(|x| -2.5 * x).collect();
        assert!((mac(&phi, &scaled) - 1.0).abs() < 1e-12);
        // Orthogonal shapes.
        assert!(mac(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn psd_peaks_at_the_ambient_mode() {
        let (n, sr) = (4096usize, 4096.0);
        // Ambient response dominated by a 37 Hz mode.
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * core::f64::consts::PI * 37.0 * i as f64 / sr).sin())
            .collect();
        let psd = power_spectral_density(&sig);
        let modes = natural_frequencies(&psd, sr, n, 0.5);
        assert!(modes.iter().any(|f| (f - 37.0).abs() < 0.2), "{modes:?}");
    }

    #[test]
    fn mac_drops_when_a_mode_shape_changes() {
        // Damage perturbs one DOF of the shape -> MAC falls below 1.
        let baseline = [1.0, 0.8, 0.5, 0.2];
        let damaged = [1.0, 0.8, -0.1, 0.2];
        let m = mac(&baseline, &damaged);
        assert!(m < 0.95, "MAC {m} should signal a shape change");
    }
}
