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
    fn pfd_1oo2_matches_hand_derivation() {
        // Clean inputs: λ=1e-3 /h, T1=1000 h, β=0.1  ->  λT1 = 1.0.
        //   independent = (1−β)²·(λT1)²/3 = 0.81·1/3 = 0.27
        //   common-cause = β·(λT1)/2      = 0.1·1/2  = 0.05
        //   total = 0.32  (IEC 61508-6 Annex B, simplified 1oo2 PFDavg).
        let pfd = pfd_1oo2(1e-3, 1000.0, 0.1);
        assert!((pfd - 0.32).abs() < 1e-12, "pfd_1oo2 {pfd}, want 0.32");
    }

    #[test]
    fn pfd_2oo3_matches_hand_derivation() {
        // Same λT1 = 1.0, β=0.1.
        //   independent = (1−β)²·(λT1)² = 0.81·1 = 0.81   (no /3 factor for 2oo3)
        //   common-cause = β·(λT1)/2    = 0.05
        //   total = 0.86  (IEC 61508-6 Annex B, simplified 2oo3 PFDavg).
        let pfd = pfd_2oo3(1e-3, 1000.0, 0.1);
        assert!((pfd - 0.86).abs() < 1e-12, "pfd_2oo3 {pfd}, want 0.86");
    }

    #[test]
    fn pfd_2oo3_independent_term_exceeds_1oo2() {
        // For identical (λT1, β), the 2oo3 independent term ((λT1)²) is 3× the
        // 1oo2 independent term ((λT1)²/3); the shared CCF term (β·λT1/2) is
        // equal. So 2oo3 − 1oo2 must equal exactly the extra (2/3)(1−β)²(λT1)².
        let (lam, t1, beta) = (1e-3, 1000.0, 0.1);
        let lt = lam * t1;
        let diff = pfd_2oo3(lam, t1, beta) - pfd_1oo2(lam, t1, beta);
        let expected = (2.0 / 3.0) * (1.0 - beta).powi(2) * lt * lt;
        assert!(
            (diff - expected).abs() < 1e-12,
            "diff {diff}, want {expected}"
        );
    }

    #[test]
    fn redundancy_lowers_pfd() {
        // Realistic loop: λ_DU=1e-6 /h, T1=8760 h, β=2%.
        // 1oo1 = λT1/2 = 4.38e-3 (SIL 2). 1oo2 hand value = 1.1216626368e-4 (SIL 3).
        let (lambda, t1, beta) = (1e-6, 8760.0, 0.02);
        let single = pfd_1oo1(lambda, t1);
        let pair = pfd_1oo2(lambda, t1, beta);
        assert!((single - 4.38e-3).abs() < 1e-9, "1oo1 {single}");
        assert!((pair - 1.121_662_636_8e-4).abs() < 1e-15, "1oo2 {pair}");
        assert!(pair < single, "1oo2 {pair} should beat 1oo1 {single}");
        // Redundancy crosses a SIL band (SIL 2 -> SIL 3) yet is floored by CCF.
        assert_eq!(sil_from_pfd(single), Sil::Sil2);
        assert_eq!(sil_from_pfd(pair), Sil::Sil3);
        assert!(pair >= beta * lambda * t1 / 2.0 - 1e-12);
    }

    #[test]
    fn sil_bands_match_iec_61508() {
        // Mid-band representatives.
        assert_eq!(sil_from_pfd(5e-5), Sil::Sil4);
        assert_eq!(sil_from_pfd(5e-4), Sil::Sil3);
        assert_eq!(sil_from_pfd(5e-3), Sil::Sil2);
        assert_eq!(sil_from_pfd(5e-2), Sil::Sil1);
        assert_eq!(sil_from_pfd(0.5), Sil::None);
    }

    #[test]
    fn sil_band_boundaries_are_lower_inclusive() {
        // IEC 61508-1 Table 2: each band is [lower, upper). The decade powers
        // therefore land in the *lower* (higher-PFD) band, not the band below.
        assert_eq!(sil_from_pfd(1e-4), Sil::Sil3); // 1e-4 is the SIL3 floor, not SIL4
        assert_eq!(sil_from_pfd(1e-3), Sil::Sil2);
        assert_eq!(sil_from_pfd(1e-2), Sil::Sil1);
        assert_eq!(sil_from_pfd(1e-1), Sil::None); // 0.1 is too poor for any SIL
        // Just below each boundary stays in the better band.
        assert_eq!(sil_from_pfd(9.999e-5,), Sil::Sil4);
        assert_eq!(sil_from_pfd(9.999e-2), Sil::Sil1);
    }

    #[test]
    fn pfh_1oo1_is_lambda() {
        // High-demand single channel: PFH = λ_DU exactly.
        assert_eq!(pfh_1oo1(1e-6), 1e-6);
        assert_eq!(pfh_1oo1(2.5e-7), 2.5e-7);
    }

    #[test]
    fn pfh_1oo2_matches_hand_derivation() {
        // λ_DU=1e-3 /h, μ=0.5 /h (MTTR=2 h), β=0.1.
        //   independent = 2(1−β)²λ²/μ = 2·0.81·1e-6/0.5 = 3.24e-6
        //   common-cause = β·λ        = 0.1·1e-3        = 1.0e-4
        //   total = 1.0324e-4  (IEC 61508-6 Annex B, simplified 1oo2 PFH).
        let pfh = pfh_1oo2(1e-3, 0.5, 0.1);
        assert!(
            (pfh - 1.0324e-4).abs() < 1e-15,
            "pfh_1oo2 {pfh}, want 1.0324e-4"
        );
    }

    #[test]
    fn pfh_1oo2_zero_repair_keeps_ccf_only() {
        // μ=0 is a division-by-zero guard: the (finite) independent term is
        // dropped and only the common-cause floor β·λ remains.
        let pfh = pfh_1oo2(1e-6, 0.0, 0.02);
        assert_eq!(pfh, 0.02 * 1e-6);
        assert!(pfh.is_finite(), "guard must avoid an infinite PFH");
    }

    #[test]
    fn pfh_redundancy_helps() {
        // 1oo2 PFH (3.24e-6 indep + 2e-8 CCF = 3.26e-6) beats 1oo1 (1e-6)? No —
        // here the single-channel λ already *is* the 1oo1 PFH, so redundancy
        // only wins once the CCF fraction is small. Use a low β and fast repair.
        let single = pfh_1oo1(1e-6);
        let pair = pfh_1oo2(1e-6, 0.1, 0.02);
        // β·λ = 2e-8 dominates; indep = 2·0.9604·1e-12/0.1 = 1.92e-11.
        assert!(
            (pair - (2e-8 + 1.92080e-11)).abs() < 1e-15,
            "pfh_1oo2 {pair}"
        );
        assert!(pair < single, "1oo2 PFH {pair} should beat 1oo1 {single}");
    }

    #[test]
    fn markov_unavailability_hand_value() {
        // Two-state up/down chain, steady state: U = λ/(λ+μ).
        // λ=1 /h, μ=99 /h  ->  U = 1/100 = 0.01 exactly; availability = 0.99.
        let u = markov_unavailability(1.0, 99.0);
        assert!((u - 0.01).abs() < 1e-15, "U {u}, want 0.01");
        let availability = 1.0 - u;
        assert!(
            (availability - 0.99).abs() < 1e-15,
            "A {availability}, want 0.99"
        );
    }

    #[test]
    fn markov_unavailability_realistic_loop() {
        // MTBF 10000 h (λ=1e-4 /h), MTTR 10 h (μ=0.1 /h).
        //   U = 1e-4 / (1e-4 + 0.1) = 1e-4 / 0.1001 = 9.990009990...e-4.
        let u = markov_unavailability(1e-4, 0.1);
        assert!((u - 9.990_009_990_009_99e-4).abs() < 1e-15, "U {u}");
    }

    #[test]
    fn markov_no_repair_is_certain_failure() {
        // μ=0 (never repaired) but λ>0: the down state is absorbing, so the
        // steady-state unavailability is 1 (not the all-zero guard branch).
        assert_eq!(markov_unavailability(1e-3, 0.0), 1.0);
    }

    #[test]
    fn markov_degenerate_inputs_are_zero() {
        // λ+μ ≤ 0 is undefined (no transitions); the guard returns 0.
        assert_eq!(markov_unavailability(0.0, 0.0), 0.0);
    }
}
