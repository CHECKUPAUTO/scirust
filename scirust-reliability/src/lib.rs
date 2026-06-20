//! # scirust-reliability — IEC 61508 functional-safety reliability
//!
//! The quantitative side of SIL: average Probability of Failure on Demand
//! (`PFDavg`, low-demand mode) and Probability of dangerous Failure per Hour
//! (`PFH`, high-demand mode) for common MooN architectures with a common-cause
//! `β` factor, the SIL band a figure maps to, and a two-state Markov
//! availability. Pure deterministic `f64` — auditable safety arithmetic.

use serde::{Deserialize, Serialize};

/// Safety Integrity Level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sil {
    /// Below SIL 1 (PFDavg ≥ 0.1).
    None,
    Sil1,
    Sil2,
    Sil3,
    Sil4,
}

/// IEC 61508 low-demand SIL band for an average Probability of Failure on Demand.
pub fn sil_from_pfd(pfd: f64) -> Sil {
    // PFDavg < 1e-4 is SIL 4 (the region below 1e-5 is capped there).
    if pfd < 1e-4
    {
        Sil::Sil4
    }
    else if pfd < 1e-3
    {
        Sil::Sil3
    }
    else if pfd < 1e-2
    {
        Sil::Sil2
    }
    else if pfd < 1e-1
    {
        Sil::Sil1
    }
    else
    {
        Sil::None
    }
}

/// `PFDavg` of a single channel (1oo1): `λ_DU · T₁ / 2`, with `λ_DU` the
/// dangerous-undetected failure rate (per hour) and `T₁` the proof-test
/// interval (hours).
pub fn pfd_1oo1(lambda_du: f64, t1: f64) -> f64 {
    lambda_du * t1 / 2.0
}

/// `PFDavg` of a 1oo2 redundant pair with common-cause fraction `beta`:
/// independent term `(1−β)²(λT₁)²/3` plus common-cause term `β·λT₁/2`.
pub fn pfd_1oo2(lambda_du: f64, t1: f64, beta: f64) -> f64 {
    let lt = lambda_du * t1;
    let indep = (1.0 - beta).powi(2) * lt * lt / 3.0;
    let ccf = beta * lt / 2.0;
    indep + ccf
}

/// `PFDavg` of a 2oo3 architecture with common-cause fraction `beta`:
/// `(1−β)²(λT₁)² + β·λT₁/2`.
pub fn pfd_2oo3(lambda_du: f64, t1: f64, beta: f64) -> f64 {
    let lt = lambda_du * t1;
    (1.0 - beta).powi(2) * lt * lt + beta * lt / 2.0
}

/// `PFH` (per hour) of a 1oo1 channel in high-demand mode: simply `λ_DU`.
pub fn pfh_1oo1(lambda_du: f64) -> f64 {
    lambda_du
}

/// `PFH` of a 1oo2 pair with common-cause `beta` and repair rate `mu` (per
/// hour): `2(1−β)²λ²/μ + β·λ`.
pub fn pfh_1oo2(lambda_du: f64, mu: f64, beta: f64) -> f64 {
    let indep = if mu > 0.0
    {
        2.0 * (1.0 - beta).powi(2) * lambda_du * lambda_du / mu
    }
    else
    {
        0.0
    };
    indep + beta * lambda_du
}

/// Steady-state **unavailability** of a two-state (up/down) component with
/// failure rate `lambda` and repair rate `mu`: `λ / (λ + μ)`.
pub fn markov_unavailability(lambda: f64, mu: f64) -> f64 {
    if lambda + mu <= 0.0
    {
        0.0
    }
    else
    {
        lambda / (lambda + mu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfd_1oo1_is_half_lambda_t() {
        // λ_DU = 1e-6 /h, T1 = 8760 h (1 year) -> PFD = 4.38e-3 (SIL 2).
        let pfd = pfd_1oo1(1e-6, 8760.0);
        assert!((pfd - 4.38e-3).abs() < 1e-6, "pfd {pfd}");
        assert_eq!(sil_from_pfd(pfd), Sil::Sil2);
    }

    #[test]
    fn redundancy_lowers_pfd() {
        let (lambda, t1, beta) = (1e-6, 8760.0, 0.02);
        let single = pfd_1oo1(lambda, t1);
        let pair = pfd_1oo2(lambda, t1, beta);
        assert!(pair < single, "1oo2 {pair} should beat 1oo1 {single}");
        // With common cause, redundancy helps but is bounded below by the CCF term.
        assert!(pair >= beta * lambda * t1 / 2.0 - 1e-12);
    }

    #[test]
    fn sil_bands_match_iec_61508() {
        assert_eq!(sil_from_pfd(5e-5), Sil::Sil4);
        assert_eq!(sil_from_pfd(5e-4), Sil::Sil3);
        assert_eq!(sil_from_pfd(5e-3), Sil::Sil2);
        assert_eq!(sil_from_pfd(5e-2), Sil::Sil1);
        assert_eq!(sil_from_pfd(0.5), Sil::None);
    }

    #[test]
    fn markov_unavailability_formula() {
        // MTBF 10000 h (λ=1e-4), MTTR 10 h (μ=0.1): U = λ/(λ+μ) ≈ 9.99e-4.
        let u = markov_unavailability(1e-4, 0.1);
        assert!((u - 1e-4 / (1e-4 + 0.1)).abs() < 1e-12);
        assert!(u < 1e-3);
    }

    #[test]
    fn pfh_redundancy_helps() {
        assert!(pfh_1oo2(1e-6, 0.1, 0.02) < pfh_1oo1(1e-6));
    }
}
