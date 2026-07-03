//! # scirust-reliability — IEC 61508 functional-safety reliability
//!
//! The quantitative side of SIL: average Probability of Failure on Demand
//! (`PFDavg`, low-demand mode) and Probability of dangerous Failure per Hour
//! (`PFH`, high-demand mode) for the MooN family with a common-cause `β`
//! factor, the SIL band a figure maps to, and a two-state Markov
//! availability. Pure deterministic `f64` — auditable safety arithmetic.
//! `scirust-sis` builds process-safety (IEC 61511) SIS loop modeling,
//! voting simulation, and cause-and-effect matrices on top of these
//! primitives.
//!
//! ## Two tiers of formula, not one, and why
//! IEC 61508-6:2010 Annex B.3.2 tabulates closed-form `PFDavg` equations for
//! exactly five architectures — 1oo1, 1oo2, 2oo2, 2oo3, 1oo3 — and no
//! others; it does **not** give a general M-out-of-N formula. This crate
//! keeps [`pfd_1oo1`], [`pfd_1oo2`], [`pfd_2oo2`], [`pfd_2oo3`], and
//! [`pfd_1oo3`] as those exact, literally-standard closed forms — the ones
//! to cite when a reviewer asks "where does this equation come from?"
//!
//! [`pfd_moon`] additionally generalizes to arbitrary `M`-out-of-`N` via
//! Lundteigen & Rausand's minimal-cutset/RBD derivation (*Reliability of
//! Safety-Critical Systems*, Wiley 2015, ch. 8): for `M < N`, with
//! `r = N-M+1` the number of channels that must fail dangerous-undetected
//! *simultaneously* to fail the group,
//! `PFDavg = C(N,r)·(1-β)^r·(λDU·T1)^r/(r+1) + β·λDU·T1/2`; for the `M = N`
//! case (no redundancy against dangerous failure — any single channel
//! failing is already fatal to the vote), `PFDavg = N·λDU·T1/2` with no `β`
//! term, following the explicit `NooN` treatment in "The MooN Safety
//! Function Failure Probability Model" (I&E Systems / The 61508
//! Association, Rev. 3, 2023) — the same source notes that including a `β`
//! term for `M = N` would make the estimate *less* conservative, which is
//! why the convention omits it, not because a common-cause contribution is
//! physically absent. **This general formula is cross-validated to
//! reproduce all five IEC-tabulated cases exactly** (see
//! `pfd_moon_matches_all_five_named_architectures` in the test suite below)
//! but is itself a textbook/industry generalization, not a literal Annex B
//! quote — an earlier, cruder generalization attempted while building this
//! crate reproduced four of the five cases but got 2oo2 wrong by omitting
//! this M=N special case, which is exactly the kind of near-miss this
//! doc comment exists to prevent repeating. Architectures beyond what
//! either tier can cite a source for (e.g. non-identical channel failure
//! rates) remain unsupported by design; see Jin & Rausand, *Reliability
//! Engineering & System Safety* (2014) for why a *fully* general KooN
//! theory (arbitrary channel heterogeneity) is still active research, not
//! a closed textbook result.
//!
//! ## Validity range
//! These are first-order ("rare event") approximations to the underlying
//! Markov model, valid for `λDU·T1 < 0.1` per ISA-TR84.00.02 Part 2 and
//! Brissaud et al. (arXiv:1501.06487); they also assume identical channel
//! failure rates and proof-test intervals across a voted group. This crate
//! does not warn when `λDU·T1` exceeds that bound — documented here rather
//! than silently assumed away.

use serde::{Deserialize, Serialize};

/// Safety Integrity Level.
///
/// Declared worst-to-best so the derived [`Ord`] matches integrity order
/// (`Sil::None < Sil::Sil1 < ... < Sil::Sil4`) — a higher band is always a
/// stronger safety claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

/// `PFDavg` of a 2oo2 architecture: `λ_DU · T₁` (i.e. `2 × pfd_1oo1`).
///
/// Unlike 1oo2/2oo3, a 2oo2 vote requires **both** channels to agree before
/// tripping, so a single channel's dangerous failure already defeats the
/// safety function — there is no redundancy left for a common-cause fault to
/// additionally defeat. No separate `β` term applies (2oo2 architectures are
/// chosen to cut spurious trips, at the cost of the worst PFDavg of the
/// MooN family — the reverse trade-off from 1oo2).
pub fn pfd_2oo2(lambda_du: f64, t1: f64) -> f64 {
    lambda_du * t1
}

/// `PFDavg` of a 1oo3 architecture with common-cause fraction `beta`:
/// `(1−β)³(λT₁)³/4 + β·λT₁/2`. The best (lowest) PFDavg of the MooN family —
/// all three channels must fail dangerous-undetected simultaneously before
/// the group does.
pub fn pfd_1oo3(lambda_du: f64, t1: f64, beta: f64) -> f64 {
    let lt = lambda_du * t1;
    (1.0 - beta).powi(3) * lt * lt * lt / 4.0 + beta * lt / 2.0
}

/// Binomial coefficient `C(n, r)`, computed multiplicatively in `f64` to
/// avoid factorial overflow — exact for the small `n` any real voting
/// architecture uses.
fn binomial_coefficient(n: u32, r: u32) -> f64 {
    if r > n
    {
        return 0.0;
    }
    let r = r.min(n - r);
    let mut result = 1.0f64;
    for i in 0..r
    {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// `PFDavg` of an arbitrary `M`-out-of-`N` voting architecture. See the
/// module documentation ("Two tiers of formula, not one, and why") for the
/// derivation, its provenance (Lundteigen & Rausand's RBD generalization,
/// plus the industry-documented `M = N` special case), and why it is kept
/// separate from the five named IEC-tabulated functions above.
///
/// Returns `Err` if `m == 0` or `m > n` (not a valid voting architecture).
pub fn pfd_moon(m: u32, n: u32, lambda_du: f64, t1: f64, beta: f64) -> Result<f64, String> {
    if m == 0 || m > n
    {
        return Err(format!(
            "invalid MooN architecture {m}oo{n}: need 1 <= m <= n"
        ));
    }
    if m == n
    {
        // Zero redundancy against dangerous failure: any one channel
        // failing already defeats the vote, so there is no coincidence
        // left for beta to model (see module doc for the conservatism
        // caveat on deliberately omitting it here).
        return Ok(n as f64 * lambda_du * t1 / 2.0);
    }
    let r = n - m + 1;
    let lt = lambda_du * t1;
    let coeff = binomial_coefficient(n, r);
    Ok(
        coeff * (1.0 - beta).powi(r as i32) * lt.powi(r as i32) / (r as f64 + 1.0)
            + beta * lt / 2.0,
    )
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
    use approx::assert_relative_eq;

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
    fn pfd_2oo3_matches_published_worked_example() {
        // Lundteigen & Rausand, *Reliability of Safety-Critical Systems*
        // (NTNU course notes, ch. 8, slide 27/43): 2oo3, λDU=1e-6/h,
        // τ=8760h, β=10% -> PFDavg ≈ 5.00e-4 (independent ≈6.22e-5, common
        // cause ≈4.38e-4, i.e. CCF is ~87.6% of the total, matching the
        // slide's stated ~87%/~13% split).
        let pfd = pfd_2oo3(1e-6, 8760.0, 0.1);
        assert!(
            (pfd - 5.00e-4).abs() < 5e-6,
            "pfd_2oo3 {pfd}, want ~5.00e-4"
        );
        assert_eq!(sil_from_pfd(pfd), Sil::Sil3);
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
    fn pfd_2oo2_matches_hand_derivation() {
        // λ_DU=1e-6 /h, T1=8760 h -> PFD = λT1 = 8.76e-3 = exactly 2× pfd_1oo1.
        let pfd = pfd_2oo2(1e-6, 8760.0);
        assert!(
            (pfd - 8.76e-3).abs() < 1e-12,
            "pfd_2oo2 {pfd}, want 8.76e-3"
        );
        assert!((pfd - 2.0 * pfd_1oo1(1e-6, 8760.0)).abs() < 1e-15);
    }

    #[test]
    fn pfd_1oo3_matches_hand_derivation() {
        // Same λT1 = 1.0, β=0.1 as the 1oo2/2oo3 hand derivations above.
        //   independent = (1−β)³·(λT1)³/4 = 0.729/4 = 0.18225
        //   common-cause = β·(λT1)/2      = 0.05
        //   total = 0.23225 (IEC 61508-6 Annex B, simplified 1oo3 PFDavg).
        let pfd = pfd_1oo3(1e-3, 1000.0, 0.1);
        assert!(
            (pfd - 0.23225).abs() < 1e-12,
            "pfd_1oo3 {pfd}, want 0.23225"
        );
    }

    #[test]
    fn moon_family_pfd_ordering_matches_redundancy() {
        // For identical (λ, T1, β): more channels needed to *simultaneously*
        // fail dangerous before the group does ⇒ lower PFDavg. 2oo2 has zero
        // redundancy against dangerous failure (either channel failing is
        // already fatal to the vote) and is therefore the worst; 1oo3 needs
        // all three channels to fail together and is the best.
        let (lam, t1, beta) = (1e-3, 1000.0, 0.1);
        let p_1oo3 = pfd_1oo3(lam, t1, beta);
        let p_1oo2 = pfd_1oo2(lam, t1, beta);
        let p_2oo3 = pfd_2oo3(lam, t1, beta);
        let p_2oo2 = pfd_2oo2(lam, t1);
        assert!(p_1oo3 < p_1oo2, "{p_1oo3} should beat {p_1oo2}");
        assert!(p_1oo2 < p_2oo3, "{p_1oo2} should beat {p_2oo3}");
        assert!(p_2oo3 < p_2oo2, "{p_2oo3} should beat {p_2oo2}");
    }

    #[test]
    fn pfd_moon_matches_all_five_named_architectures() {
        // Cross-validates the general formula against every one of the
        // five IEC-tabulated closed forms above, at several (λ, T1, β)
        // combinations — not just the one hand-derivation each already has.
        let cases: &[(f64, f64, f64)] = &[
            (1e-3, 1000.0, 0.1),
            (1e-6, 8760.0, 0.02),
            (2e-5, 4380.0, 0.0),
        ];
        for &(lam, t1, beta) in cases
        {
            assert_relative_eq!(
                pfd_moon(1, 1, lam, t1, beta).unwrap(),
                pfd_1oo1(lam, t1),
                epsilon = 1e-15
            );
            assert_relative_eq!(
                pfd_moon(1, 2, lam, t1, beta).unwrap(),
                pfd_1oo2(lam, t1, beta),
                epsilon = 1e-15
            );
            assert_relative_eq!(
                pfd_moon(2, 2, lam, t1, beta).unwrap(),
                pfd_2oo2(lam, t1),
                epsilon = 1e-15
            );
            assert_relative_eq!(
                pfd_moon(2, 3, lam, t1, beta).unwrap(),
                pfd_2oo3(lam, t1, beta),
                epsilon = 1e-15
            );
            assert_relative_eq!(
                pfd_moon(1, 3, lam, t1, beta).unwrap(),
                pfd_1oo3(lam, t1, beta),
                epsilon = 1e-15
            );
        }
    }

    #[test]
    fn pfd_moon_2oo4_and_3oo4_match_independent_derivation() {
        // Lundteigen & Rausand's koon recursion, applied independently to
        // 2oo4/3oo4, gives (1-β)³(λT)³+βλT/2 and 2(1-β)²(λT)²+βλT/2
        // respectively — reproduced here from the same general formula.
        let (lam, t1, beta): (f64, f64, f64) = (1e-3, 1000.0, 0.1);
        let lt = lam * t1;
        let expected_2oo4 = (1.0 - beta).powi(3) * lt.powi(3) + beta * lt / 2.0;
        let expected_3oo4 = 2.0 * (1.0 - beta).powi(2) * lt.powi(2) + beta * lt / 2.0;
        assert_relative_eq!(
            pfd_moon(2, 4, lam, t1, beta).unwrap(),
            expected_2oo4,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pfd_moon(3, 4, lam, t1, beta).unwrap(),
            expected_3oo4,
            epsilon = 1e-12
        );
    }

    #[test]
    fn pfd_moon_rejects_invalid_architecture() {
        assert!(pfd_moon(0, 3, 1e-3, 1000.0, 0.1).is_err());
        assert!(pfd_moon(4, 3, 1e-3, 1000.0, 0.1).is_err());
    }

    #[test]
    fn pfd_moon_more_redundancy_is_never_worse_within_fixed_n() {
        // Within a fixed channel count N=5, requiring fewer votes to trip
        // (smaller M) means more channels must simultaneously fail to
        // defeat the group, so PFDavg should be non-increasing as M drops.
        let (lam, t1, beta) = (1e-4, 2000.0, 0.05);
        let by_m: Vec<f64> = (1..=5)
            .map(|m| pfd_moon(m, 5, lam, t1, beta).unwrap())
            .collect();
        for w in by_m.windows(2)
        {
            assert!(
                w[0] <= w[1],
                "PFDavg should not increase as M decreases: {by_m:?}"
            );
        }
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
