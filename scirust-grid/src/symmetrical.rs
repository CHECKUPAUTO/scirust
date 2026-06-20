//! Symmetrical components (Fortescue) and voltage unbalance.
//!
//! Any set of three-phase phasors `(Va, Vb, Vc)` decomposes into a **positive**,
//! **negative** and **zero** sequence. The ratio of negative to positive
//! sequence magnitude is the Voltage Unbalance Factor (VUF, %), the standard
//! IEC/NEMA power-quality metric — a balanced supply has VUF 0, and motors
//! overheat as it rises.

use scirust_signal::Complex;

/// Operator `a = e^{j·120°}`.
fn op_a() -> Complex {
    Complex::cis(2.0 * core::f64::consts::PI / 3.0)
}

/// Decompose three-phase phasors into `(zero, positive, negative)` sequence
/// components.
pub fn symmetrical_components(
    va: Complex,
    vb: Complex,
    vc: Complex,
) -> (Complex, Complex, Complex) {
    let a = op_a();
    let a2 = a * a;
    let third = 1.0 / 3.0;
    let v0 = (va + vb + vc) * third;
    let v1 = (va + a * vb + a2 * vc) * third;
    let v2 = (va + a2 * vb + a * vc) * third;
    (v0, v1, v2)
}

/// Voltage Unbalance Factor `|V₂| / |V₁| · 100` (percent). Returns 0 when the
/// positive sequence is ~0.
pub fn voltage_unbalance_factor(v1: Complex, v2: Complex) -> f64 {
    let m1 = v1.mag();
    if m1 < 1e-12
    {
        0.0
    }
    else
    {
        100.0 * v2.mag() / m1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polar(mag: f64, deg: f64) -> Complex {
        Complex::cis(deg.to_radians()) * mag
    }

    #[test]
    fn balanced_set_is_pure_positive_sequence() {
        let (v0, v1, v2) =
            symmetrical_components(polar(1.0, 0.0), polar(1.0, -120.0), polar(1.0, 120.0));
        assert!(v0.mag() < 1e-12, "V0 {}", v0.mag());
        assert!((v1.mag() - 1.0).abs() < 1e-12, "V1 {}", v1.mag());
        assert!(v2.mag() < 1e-12, "V2 {}", v2.mag());
        assert!(voltage_unbalance_factor(v1, v2) < 1e-9);
    }

    #[test]
    fn single_phase_splits_evenly() {
        // Only phase A energized -> each sequence is Va/3.
        let (v0, v1, v2) = symmetrical_components(
            polar(3.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        );
        for v in [v0, v1, v2]
        {
            assert!((v.mag() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn unbalance_factor_matches_injected_negative_sequence() {
        // V_a/b/c = positive(1.0) + negative(0.1) -> VUF should be 10%.
        let a = op_a();
        let a2 = a * a;
        let (p, nseq) = (1.0, 0.1);
        let va = Complex::new(p + nseq, 0.0);
        let vb = a2 * Complex::new(p, 0.0) + a * Complex::new(nseq, 0.0);
        let vc = a * Complex::new(p, 0.0) + a2 * Complex::new(nseq, 0.0);
        let (_v0, v1, v2) = symmetrical_components(va, vb, vc);
        let vuf = voltage_unbalance_factor(v1, v2);
        assert!((vuf - 10.0).abs() < 1e-6, "VUF {vuf}");
    }
}
