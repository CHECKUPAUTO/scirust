//! Backend-supplied `L2`/`L1` reproduction verdicts — the numeric half of
//! `sos-repro`'s reproduction contract.
//!
//! [`sos_repro`] decides `L3` (bit-exact) and `L0` (replay) nodes from object
//! ids alone, and deliberately stops there: an `L2` node reproduces *within
//! its certificate* and an `L1` node *in distribution*, and neither is a
//! question a kernel crate can answer. Its own docs say so — *"the numeric /
//! statistical evaluation behind an `L2`/`L1` node's verdict is the backend's
//! job (`sos-scirust`), supplied to the contract as a
//! [`Reproduced::Certified`] outcome — this crate never fabricates it."* This
//! module is that backend.
//!
//! The output is an [`Agreement`], which converts to the contract's
//! [`Reproduced`] via [`Agreement::verdict`]. It is deliberately richer than
//! the `bool` the contract consumes: a bare "did not reproduce" is close to
//! useless when a study has diverged, so an `Agreement` also carries how far
//! the two results actually differed, the bound that was allowed, and — for a
//! trajectory — which sample was worst.
//!
//! ## The `L2` rule, and why the tolerances add
//!
//! A certificate bounds *its own* result's error against the true value. If
//! the original is within `εₐ` of truth and the re-run within `ε_b`, then by
//! the triangle inequality the two are within `εₐ + ε_b` of **each other**:
//!
//! ```text
//! |a − b| ≤ |a − true| + |true − b| ≤ εₐ + ε_b
//! ```
//!
//! So the comparison threshold is the **sum** of the two certificates, each
//! evaluated in the usual mixed form `atol + rtol·|value|`. Comparing against
//! a single certificate instead — the obvious-looking choice — would report
//! divergence for two runs that both honoured their contracts exactly as
//! promised, which is a false alarm on the one signal the whole reproduction
//! machinery exists to make trustworthy. The two runs need not even share
//! tolerances: a re-run at a tighter tolerance contributes its own, smaller
//! term.
//!
//! ## Comparing adaptive trajectories
//!
//! Two adaptive runs choose *different step sequences*, so their sample grids
//! generally do not line up and a pointwise comparison is not meaningful
//! between them. Two functions, with the difference in the name rather than
//! buried in a caveat:
//!
//! * [`certify_trajectory_endpoint`] compares `y(t_end)` — well-defined for
//!   both runs whatever grids they chose, and the quantity a study usually
//!   actually claims.
//! * [`certify_trajectory_pointwise`] compares every sample, and **refuses**
//!   ([`VerdictError::NotComparable`]) when the grids differ rather than
//!   silently comparing sample *i* of one run against an unrelated sample *i*
//!   of the other. Interpolating one trajectory onto the other's grid would
//!   make them comparable, and is deliberately not done here: the
//!   interpolation error would enter the verdict unaccounted for.
//!
//! ## The `L1` rule, and what it can and cannot say
//!
//! Two functions again, because the honest test depends on something the
//! [`Estimate`] does not carry — the seed:
//!
//! * [`certify_estimate_same_seed`] requires **exact** equality. A seeded
//!   estimator re-run under its own seed is bit-reproducible; a mismatch
//!   there is a real defect, and must not be waved through by a statistical
//!   test generous enough to cover it.
//! * [`certify_estimate_in_distribution`] applies a two-sample test at
//!   `sigma` combined standard errors, for re-runs under a *different* seed.
//!
//! What the second one certifies is *"not distinguishable at `sigma`
//! sigma"* — not "identical", and not "equal in distribution". A statistical
//! test cannot establish agreement; it can only fail to detect disagreement,
//! and it fails to detect it more easily the noisier the estimates are. That
//! asymmetry is why [`Agreement`] reports the deviation and the bound instead
//! of only the verdict: a pass at a wide bound and a pass at a tight one mean
//! very different things, and the caller can see which it got.

use sos_planner::Estimate;
use sos_repro::Reproduced;
use thiserror::Error;

use crate::ode::CertifiedTrajectory;
use crate::quadrature::CertifiedIntegral;
use crate::root::CertifiedRoot;
use sos_core::DeterminismLevel;

/// The default number of combined standard errors
/// [`certify_estimate_in_distribution`] allows — about 99.7% of a normal
/// sampling distribution, so an honest re-run trips it roughly 3 times in
/// 1000.
pub const DEFAULT_SIGMA: f64 = 3.0;

/// Two results could not be compared at all — a caller error, distinct from
/// "they did not reproduce".
///
/// Returning this rather than `false` matters: a false verdict says the
/// science diverged, and reporting that because someone compared two
/// different quantities would be a lie about the result.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum VerdictError {
    /// The two results do not describe the same quantity (different
    /// integration bounds, different sample grids, different state
    /// dimensions).
    #[error("results are not comparable: {0}")]
    NotComparable(String),
    /// An estimate was not tagged `L1`, so the in-distribution test does not
    /// apply to it — an `L3` estimate must be checked exactly, not excused by
    /// a tolerance band.
    #[error("in-distribution test applied to a {0:?} estimate; only L1 qualifies")]
    NotStochastic(DeterminismLevel),
}

/// How two results compared, and by how much.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Agreement {
    /// Whether the re-run is certified to agree at the node's level.
    pub agrees: bool,
    /// The worst observed absolute difference between the two results.
    pub deviation: f64,
    /// The bound `deviation` was tested against.
    pub allowed: f64,
    /// The index of the worst-disagreeing sample, for pointwise trajectory
    /// comparisons. `None` for scalar comparisons.
    pub at: Option<usize>,
}

impl Agreement {
    /// This agreement as the reproduction contract's outcome, ready for
    /// [`sos_repro::verify_node`].
    #[must_use]
    pub fn verdict(&self) -> Reproduced {
        Reproduced::Certified(self.agrees)
    }

    /// How much of the allowed budget the observed deviation used, as a
    /// fraction. `None` when the allowed bound is zero (an exactness
    /// requirement), where a ratio is undefined.
    ///
    /// A pass at `0.02` and a pass at `0.98` are both passes, and mean very
    /// different things about how much margin the result had.
    #[must_use]
    pub fn margin_used(&self) -> Option<f64> {
        (self.allowed > 0.0).then(|| self.deviation / self.allowed)
    }

    fn decide(deviation: f64, allowed: f64, at: Option<usize>) -> Self {
        Self {
            // A non-finite deviation never agrees: a re-run that produced NaN
            // or an infinity did not reproduce, whatever the bound says.
            agrees: deviation.is_finite() && deviation <= allowed,
            deviation,
            allowed,
            at,
        }
    }
}

/// The half-width a certificate claims for one value: `atol + rtol·|value|`.
fn certified_error(value: f64, rtol: f64, atol: f64) -> f64 {
    atol + rtol * value.abs()
}

/// Certify an `L2` re-run of an adaptive quadrature against the original.
///
/// # Errors
/// [`VerdictError::NotComparable`] if the two results integrate different
/// intervals — a different integral is a different quantity, not a failed
/// reproduction of this one.
pub fn certify_integral(
    original: &CertifiedIntegral,
    rerun: &CertifiedIntegral,
) -> Result<Agreement, VerdictError> {
    if !bounds_match(original, rerun)
    {
        return Err(VerdictError::NotComparable(format!(
            "integration bounds differ: [{}, {}] vs [{}, {}]",
            original.a, original.b, rerun.a, rerun.b
        )));
    }
    // Quadrature certificates are absolute-tolerance only, so the summed
    // bound is simply the two tolerances.
    let allowed = original.tol.abs() + rerun.tol.abs();
    Ok(Agreement::decide(
        (original.value - rerun.value).abs(),
        allowed,
        None,
    ))
}

fn bounds_match(a: &CertifiedIntegral, b: &CertifiedIntegral) -> bool {
    a.a.to_bits() == b.a.to_bits() && a.b.to_bits() == b.b.to_bits()
}

/// Certify an `L2` re-run of an adaptive integration by its **endpoint**
/// `y(t_end)` — grid-independent, so two runs that chose different step
/// sequences are still comparable.
///
/// # Errors
/// [`VerdictError::NotComparable`] if either trajectory is empty, they end at
/// different times, or their state vectors have different dimensions.
pub fn certify_trajectory_endpoint(
    original: &CertifiedTrajectory,
    rerun: &CertifiedTrajectory,
) -> Result<Agreement, VerdictError> {
    let (t_a, y_a) = last_sample(original, "original")?;
    let (t_b, y_b) = last_sample(rerun, "re-run")?;
    if t_a.to_bits() != t_b.to_bits()
    {
        return Err(VerdictError::NotComparable(format!(
            "trajectories end at different times: {t_a} vs {t_b}"
        )));
    }
    if y_a.len() != y_b.len()
    {
        return Err(VerdictError::NotComparable(format!(
            "state dimensions differ: {} vs {}",
            y_a.len(),
            y_b.len()
        )));
    }
    Ok(worst_component(y_a, y_b, original, rerun, None))
}

/// Certify an `L2` re-run of an adaptive integration **sample by sample**.
///
/// # Errors
/// [`VerdictError::NotComparable`] if the two runs did not produce the same
/// sample grid — see the module docs on why this refuses rather than
/// comparing mismatched samples or silently interpolating.
pub fn certify_trajectory_pointwise(
    original: &CertifiedTrajectory,
    rerun: &CertifiedTrajectory,
) -> Result<Agreement, VerdictError> {
    if original.trajectory.len() != rerun.trajectory.len()
    {
        return Err(VerdictError::NotComparable(format!(
            "sample counts differ ({} vs {}); adaptive runs choose their own \
             grids, so compare endpoints instead",
            original.trajectory.len(),
            rerun.trajectory.len()
        )));
    }
    let mut worst = Agreement::decide(0.0, 0.0, None);
    for (i, ((t_a, y_a), (t_b, y_b))) in original
        .trajectory
        .iter()
        .zip(rerun.trajectory.iter())
        .enumerate()
    {
        if t_a.to_bits() != t_b.to_bits()
        {
            return Err(VerdictError::NotComparable(format!(
                "sample {i} is at t={t_a} in the original but t={t_b} in the \
                 re-run; the grids do not line up"
            )));
        }
        if y_a.len() != y_b.len()
        {
            return Err(VerdictError::NotComparable(format!(
                "state dimensions differ at sample {i}: {} vs {}",
                y_a.len(),
                y_b.len()
            )));
        }
        let here = worst_component(y_a, y_b, original, rerun, Some(i));
        // Keep the sample that used the most of its budget, not merely the
        // largest raw deviation: with a relative term the allowed bound
        // varies per sample, so the biggest difference is not necessarily
        // the most serious one.
        if !here.agrees && worst.agrees || (here.agrees == worst.agrees && exceeds(&here, &worst))
        {
            worst = here;
        }
    }
    Ok(worst)
}

fn exceeds(here: &Agreement, worst: &Agreement) -> bool {
    match (here.margin_used(), worst.margin_used())
    {
        (Some(a), Some(b)) => a > b,
        (Some(_), None) => true,
        _ => here.deviation > worst.deviation,
    }
}

fn last_sample<'a>(
    c: &'a CertifiedTrajectory,
    which: &str,
) -> Result<(f64, &'a [f64]), VerdictError> {
    c.trajectory
        .last()
        .map(|(t, y)| (*t, y.as_slice()))
        .ok_or_else(|| VerdictError::NotComparable(format!("the {which} trajectory is empty")))
}

/// The worst-agreeing component of two state vectors, each compared against
/// the sum of the two certificates' half-widths at that component.
fn worst_component(
    y_a: &[f64],
    y_b: &[f64],
    ca: &CertifiedTrajectory,
    cb: &CertifiedTrajectory,
    at: Option<usize>,
) -> Agreement {
    let mut worst = Agreement::decide(0.0, 0.0, at);
    for (a, b) in y_a.iter().zip(y_b)
    {
        let allowed = certified_error(*a, ca.rtol, ca.atol) + certified_error(*b, cb.rtol, cb.atol);
        let here = Agreement::decide((a - b).abs(), allowed, at);
        if !here.agrees && worst.agrees || (here.agrees == worst.agrees && exceeds(&here, &worst))
        {
            worst = here;
        }
    }
    worst
}

/// Certify an `L1` re-run performed under the **same seed**, where a seeded
/// estimator must reproduce exactly.
///
/// Exact equality is required deliberately: see the module docs.
#[must_use]
pub fn certify_estimate_same_seed(original: &Estimate, rerun: &Estimate) -> Agreement {
    let deviation = (original.bits_milli - rerun.bits_milli).abs();
    #[allow(clippy::cast_precision_loss)] // millibit magnitudes are far below 2^53
    Agreement::decide(deviation as f64, 0.0, None)
}

/// Certify an `L1` re-run performed under a **different seed**, by a
/// two-sample test at `sigma` combined standard errors.
///
/// Passing certifies only that the two are *not distinguishable* at `sigma`
/// sigma — see the module docs on what that does and does not establish. Use
/// [`DEFAULT_SIGMA`] unless you have a reason not to.
///
/// # Errors
/// [`VerdictError::NotStochastic`] if either estimate is not tagged `L1`.
pub fn certify_estimate_in_distribution(
    original: &Estimate,
    rerun: &Estimate,
    sigma: f64,
) -> Result<Agreement, VerdictError> {
    for e in [original, rerun]
    {
        if e.level != DeterminismLevel::L1
        {
            return Err(VerdictError::NotStochastic(e.level));
        }
    }
    #[allow(clippy::cast_precision_loss)] // millibit magnitudes are far below 2^53
    let (deviation, se_a, se_b) = (
        (original.bits_milli - rerun.bits_milli).abs() as f64,
        original.se_milli as f64,
        rerun.se_milli as f64,
    );
    // Two exact estimators (both standard errors zero) must agree exactly —
    // the band collapses to nothing rather than becoming a free pass.
    let allowed = sigma.max(0.0) * se_a.hypot(se_b);
    Ok(Agreement::decide(deviation, allowed, None))
}

/// Certify an `L2` re-run of a root solve against the original.
///
/// Compares the roots componentwise against the summed certificates, in the
/// same mixed `abs_tol + rel_tol·|value|` form the tolerances were requested
/// in.
///
/// # Errors
/// [`VerdictError::NotComparable`] if the two solves started from different
/// points, or reached roots of different dimension. A root is
/// basin-dependent (see [`crate::root`]): two solves from different starts
/// are answering different questions, so a disagreement between them is not
/// a failed reproduction of either — the same reason
/// [`certify_integral`] refuses differing integration bounds.
pub fn certify_root(
    original: &CertifiedRoot,
    rerun: &CertifiedRoot,
) -> Result<Agreement, VerdictError> {
    if original.started_from.len() != rerun.started_from.len()
        || original
            .started_from
            .iter()
            .zip(&rerun.started_from)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(VerdictError::NotComparable(format!(
            "solves started from different points: {:?} vs {:?}",
            original.started_from, rerun.started_from
        )));
    }
    if original.root.len() != rerun.root.len()
    {
        return Err(VerdictError::NotComparable(format!(
            "roots have different dimensions: {} vs {}",
            original.root.len(),
            rerun.root.len()
        )));
    }

    let mut worst = Agreement::decide(0.0, 0.0, None);
    for (i, (a, b)) in original.root.iter().zip(&rerun.root).enumerate()
    {
        let allowed = certified_error(*a, original.rel_tol, original.abs_tol)
            + certified_error(*b, rerun.rel_tol, rerun.abs_tol);
        let here = Agreement::decide((a - b).abs(), allowed, Some(i));
        if !here.agrees && worst.agrees || (here.agrees == worst.agrees && exceeds(&here, &worst))
        {
            worst = here;
        }
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_repro::{NodeClaim, NodeVerdict, verify_node};

    use sos_core::{HashAlgo, ObjectId};

    fn integral(value: f64, tol: f64) -> CertifiedIntegral {
        CertifiedIntegral {
            value,
            a: 0.0,
            b: 1.0,
            tol,
        }
    }

    fn traj(points: &[(f64, &[f64])], rtol: f64, atol: f64) -> CertifiedTrajectory {
        CertifiedTrajectory {
            trajectory: points.iter().map(|(t, y)| (*t, y.to_vec())).collect(),
            rtol,
            atol,
            accepted_steps: points.len(),
            rejected_steps: 0,
        }
    }

    fn l1(bits: i64, se: i64) -> Estimate {
        Estimate::new(bits, se, DeterminismLevel::L1)
    }

    // ---- L2: quadrature -----------------------------------------------------

    #[test]
    fn two_integrals_within_their_summed_tolerances_agree() {
        // Each is certified to 1e-6, so together they may differ by 2e-6.
        // A difference of 1.5e-6 is fine — and would be wrongly rejected by a
        // check that used only one certificate.
        let a = integral(1.0, 1e-6);
        let b = integral(1.0 + 1.5e-6, 1e-6);
        let agreement = certify_integral(&a, &b).unwrap();
        assert!(agreement.agrees);
        assert!((agreement.allowed - 2e-6).abs() < 1e-18);
        assert_eq!(agreement.verdict(), Reproduced::Certified(true));
    }

    #[test]
    fn an_integral_beyond_the_summed_tolerance_diverges() {
        let a = integral(1.0, 1e-6);
        let b = integral(1.0 + 3e-6, 1e-6);
        let agreement = certify_integral(&a, &b).unwrap();
        assert!(!agreement.agrees);
        assert_eq!(agreement.verdict(), Reproduced::Certified(false));
    }

    #[test]
    fn a_tighter_rerun_contributes_its_own_smaller_tolerance() {
        // The bound is not "the loosest of the two" — each side contributes.
        let a = integral(1.0, 1e-6);
        let tight = integral(1.0, 1e-9);
        assert!((certify_integral(&a, &tight).unwrap().allowed - (1e-6 + 1e-9)).abs() < 1e-18);
    }

    #[test]
    fn integrals_over_different_intervals_are_not_comparable() {
        // Not a failed reproduction — a different quantity. Reporting
        // divergence here would be a false claim about the science.
        let a = integral(1.0, 1e-6);
        let mut b = integral(1.0, 1e-6);
        b.b = 2.0;
        assert!(matches!(
            certify_integral(&a, &b),
            Err(VerdictError::NotComparable(_))
        ));
    }

    #[test]
    fn a_non_finite_rerun_never_agrees() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
        {
            let agreement = certify_integral(&integral(1.0, 1e-6), &integral(bad, 1e-6)).unwrap();
            assert!(!agreement.agrees, "{bad} must not certify");
        }
        // Even an infinite *tolerance* cannot rescue a NaN result.
        let agreement =
            certify_integral(&integral(1.0, f64::INFINITY), &integral(f64::NAN, 1e-6)).unwrap();
        assert!(!agreement.agrees);
    }

    // ---- L2: trajectories ---------------------------------------------------

    #[test]
    fn endpoints_are_comparable_across_different_grids() {
        // The whole point: two adaptive runs took different numbers of steps
        // and still reproduce, judged where both are defined.
        let a = traj(
            &[(0.0, &[1.0]), (0.5, &[0.6]), (1.0, &[0.3678794])],
            1e-6,
            1e-9,
        );
        let b = traj(&[(0.0, &[1.0]), (1.0, &[0.3678798])], 1e-6, 1e-9);
        let agreement = certify_trajectory_endpoint(&a, &b).unwrap();
        assert!(agreement.agrees);
        assert!(agreement.deviation > 0.0, "a real difference was measured");
    }

    #[test]
    fn a_diverged_endpoint_is_caught() {
        let a = traj(&[(0.0, &[1.0]), (1.0, &[0.36788])], 1e-9, 1e-12);
        let b = traj(&[(0.0, &[1.0]), (1.0, &[0.40000])], 1e-9, 1e-12);
        assert!(!certify_trajectory_endpoint(&a, &b).unwrap().agrees);
    }

    #[test]
    fn endpoints_at_different_times_are_not_comparable() {
        let a = traj(&[(0.0, &[1.0]), (1.0, &[0.5])], 1e-6, 1e-9);
        let b = traj(&[(0.0, &[1.0]), (2.0, &[0.5])], 1e-6, 1e-9);
        assert!(matches!(
            certify_trajectory_endpoint(&a, &b),
            Err(VerdictError::NotComparable(_))
        ));
    }

    #[test]
    fn an_empty_trajectory_is_not_comparable() {
        let empty = traj(&[], 1e-6, 1e-9);
        let ok = traj(&[(0.0, &[1.0])], 1e-6, 1e-9);
        assert!(matches!(
            certify_trajectory_endpoint(&empty, &ok),
            Err(VerdictError::NotComparable(_))
        ));
        assert!(matches!(
            certify_trajectory_endpoint(&ok, &empty),
            Err(VerdictError::NotComparable(_))
        ));
    }

    #[test]
    fn pointwise_comparison_refuses_mismatched_grids() {
        // Rather than comparing sample i of one run against an unrelated
        // sample i of the other.
        let a = traj(&[(0.0, &[1.0]), (0.5, &[0.6]), (1.0, &[0.36])], 1e-6, 1e-9);
        let b = traj(&[(0.0, &[1.0]), (1.0, &[0.36])], 1e-6, 1e-9);
        assert!(matches!(
            certify_trajectory_pointwise(&a, &b),
            Err(VerdictError::NotComparable(_))
        ));

        // Same count, different times — also refused.
        let c = traj(&[(0.0, &[1.0]), (0.5, &[0.6])], 1e-6, 1e-9);
        let d = traj(&[(0.0, &[1.0]), (0.6, &[0.6])], 1e-6, 1e-9);
        assert!(matches!(
            certify_trajectory_pointwise(&c, &d),
            Err(VerdictError::NotComparable(_))
        ));
    }

    #[test]
    fn pointwise_comparison_reports_the_worst_sample() {
        let a = traj(&[(0.0, &[1.0]), (0.5, &[0.60]), (1.0, &[0.36])], 0.0, 1e-3);
        let b = traj(
            &[(0.0, &[1.0]), (0.5, &[0.6005]), (1.0, &[0.3601])],
            0.0,
            1e-3,
        );
        let agreement = certify_trajectory_pointwise(&a, &b).unwrap();
        assert!(agreement.agrees);
        assert_eq!(agreement.at, Some(1), "sample 1 used the most budget");
    }

    #[test]
    fn a_single_bad_sample_diverges_the_whole_trajectory() {
        let a = traj(&[(0.0, &[1.0]), (0.5, &[0.6]), (1.0, &[0.36])], 0.0, 1e-6);
        let b = traj(&[(0.0, &[1.0]), (0.5, &[0.9]), (1.0, &[0.36])], 0.0, 1e-6);
        let agreement = certify_trajectory_pointwise(&a, &b).unwrap();
        assert!(!agreement.agrees);
        assert_eq!(agreement.at, Some(1), "the divergence is localized");
    }

    #[test]
    fn a_multi_component_state_is_judged_by_its_worst_component() {
        let a = traj(&[(1.0, &[1.0, 2.0])], 0.0, 1e-6);
        let b = traj(&[(1.0, &[1.0, 2.5])], 0.0, 1e-6);
        assert!(!certify_trajectory_endpoint(&a, &b).unwrap().agrees);
    }

    #[test]
    fn differing_state_dimensions_are_not_comparable() {
        let a = traj(&[(1.0, &[1.0, 2.0])], 1e-6, 1e-9);
        let b = traj(&[(1.0, &[1.0])], 1e-6, 1e-9);
        assert!(matches!(
            certify_trajectory_endpoint(&a, &b),
            Err(VerdictError::NotComparable(_))
        ));
    }

    #[test]
    fn the_relative_term_scales_the_bound_with_magnitude() {
        // A 1e-3 difference on a value of 1e6 is within a 1e-6 relative
        // tolerance; the same difference on a value of 1.0 is not.
        let big = certify_trajectory_endpoint(
            &traj(&[(1.0, &[1.0e6])], 1e-6, 0.0),
            &traj(&[(1.0, &[1.0e6 + 1e-3])], 1e-6, 0.0),
        )
        .unwrap();
        assert!(big.agrees);
        let small = certify_trajectory_endpoint(
            &traj(&[(1.0, &[1.0])], 1e-6, 0.0),
            &traj(&[(1.0, &[1.0 + 1e-3])], 1e-6, 0.0),
        )
        .unwrap();
        assert!(!small.agrees);
    }

    // ---- L1: estimates ------------------------------------------------------

    #[test]
    fn a_same_seed_rerun_must_match_exactly() {
        assert!(certify_estimate_same_seed(&l1(1200, 50), &l1(1200, 50)).agrees);
        // One millibit off under the same seed is a real defect, and is not
        // excused by the standard error being large enough to cover it.
        let off = certify_estimate_same_seed(&l1(1200, 500), &l1(1201, 500));
        assert!(!off.agrees);
        assert_eq!(off.allowed, 0.0);
    }

    #[test]
    fn a_different_seed_rerun_within_three_sigma_agrees() {
        // se 40 and 30 combine to 50; 3 sigma is 150.
        let agreement =
            certify_estimate_in_distribution(&l1(1200, 40), &l1(1340, 30), DEFAULT_SIGMA).unwrap();
        assert!(agreement.agrees);
        assert!((agreement.allowed - 150.0).abs() < 1e-9);
        assert!(agreement.margin_used().unwrap() < 1.0);
    }

    #[test]
    fn a_different_seed_rerun_beyond_three_sigma_diverges() {
        let agreement =
            certify_estimate_in_distribution(&l1(1200, 40), &l1(1400, 30), DEFAULT_SIGMA).unwrap();
        assert!(!agreement.agrees);
        assert_eq!(agreement.verdict(), Reproduced::Certified(false));
    }

    #[test]
    fn two_zero_error_estimates_must_agree_exactly() {
        // The band collapses to nothing rather than becoming a free pass.
        assert!(
            certify_estimate_in_distribution(&l1(1200, 0), &l1(1200, 0), DEFAULT_SIGMA)
                .unwrap()
                .agrees
        );
        assert!(
            !certify_estimate_in_distribution(&l1(1200, 0), &l1(1201, 0), DEFAULT_SIGMA)
                .unwrap()
                .agrees
        );
    }

    #[test]
    fn the_in_distribution_test_refuses_a_non_l1_estimate() {
        // An exact estimator must be checked exactly, not handed a tolerance
        // band it never needed.
        let exact = Estimate::new(1200, 0, DeterminismLevel::L3);
        assert!(matches!(
            certify_estimate_in_distribution(&exact, &l1(1200, 10), DEFAULT_SIGMA),
            Err(VerdictError::NotStochastic(DeterminismLevel::L3))
        ));
        assert!(matches!(
            certify_estimate_in_distribution(&l1(1200, 10), &exact, DEFAULT_SIGMA),
            Err(VerdictError::NotStochastic(DeterminismLevel::L3))
        ));
    }

    #[test]
    fn a_wider_sigma_admits_more() {
        let far = |sigma| {
            certify_estimate_in_distribution(&l1(1200, 40), &l1(1400, 30), sigma)
                .unwrap()
                .agrees
        };
        assert!(!far(3.0));
        assert!(far(5.0));
        // A negative sigma is clamped to zero — it cannot widen the band.
        assert!(!far(-10.0));
    }

    #[test]
    fn margin_used_distinguishes_a_comfortable_pass_from_a_marginal_one() {
        let comfortable =
            certify_estimate_in_distribution(&l1(1200, 40), &l1(1205, 30), DEFAULT_SIGMA).unwrap();
        let marginal =
            certify_estimate_in_distribution(&l1(1200, 40), &l1(1349, 30), DEFAULT_SIGMA).unwrap();
        assert!(comfortable.agrees && marginal.agrees);
        assert!(comfortable.margin_used().unwrap() < 0.05);
        assert!(marginal.margin_used().unwrap() > 0.95);
    }

    // ---- the contract seam --------------------------------------------------

    #[test]
    fn an_agreement_slots_straight_into_the_reproduction_contract() {
        // The point of the whole module: `sos-repro` asked for a
        // `Reproduced::Certified` it refused to invent, and this supplies one.
        let id = ObjectId::compute(HashAlgo::Sha256, b"t", b"node");
        let claim = NodeClaim::new(id, DeterminismLevel::L2);

        let reproduced = certify_integral(&integral(1.0, 1e-6), &integral(1.0 + 1e-7, 1e-6))
            .unwrap()
            .verdict();
        assert_eq!(verify_node(&claim, &reproduced), NodeVerdict::Reproduced);

        let diverged = certify_integral(&integral(1.0, 1e-9), &integral(2.0, 1e-9))
            .unwrap()
            .verdict();
        assert_eq!(verify_node(&claim, &diverged), NodeVerdict::Diverged);
    }

    #[test]
    fn an_l1_agreement_also_slots_in() {
        let id = ObjectId::compute(HashAlgo::Sha256, b"t", b"node");
        let claim = NodeClaim::new(id, DeterminismLevel::L1);
        let reproduced = certify_estimate_in_distribution(&l1(1200, 40), &l1(1210, 30), 3.0)
            .unwrap()
            .verdict();
        assert_eq!(verify_node(&claim, &reproduced), NodeVerdict::Reproduced);
    }

    // ---- L2: roots ----------------------------------------------------------

    fn root(value: &[f64], from: &[f64], abs_tol: f64) -> CertifiedRoot {
        CertifiedRoot {
            root: value.to_vec(),
            started_from: from.to_vec(),
            residual: 0.0,
            iterations: 3,
            abs_tol,
            rel_tol: 0.0,
        }
    }

    #[test]
    fn two_roots_within_their_summed_tolerances_agree() {
        let a = root(&[2.0], &[1.0], 1e-9);
        let b = root(&[2.0 + 1.5e-9], &[1.0], 1e-9);
        assert!(certify_root(&a, &b).unwrap().agrees);
    }

    #[test]
    fn a_root_beyond_the_summed_tolerance_diverges() {
        let a = root(&[2.0], &[1.0], 1e-9);
        let b = root(&[2.0 + 3e-9], &[1.0], 1e-9);
        assert!(!certify_root(&a, &b).unwrap().agrees);
    }

    #[test]
    fn solves_from_different_starts_are_not_comparable() {
        // A root is basin-dependent. Reaching -2 from -1 does not fail to
        // reproduce reaching +2 from +1 — they answered different questions,
        // and calling that a divergence would misreport the science.
        let a = root(&[2.0], &[1.0], 1e-9);
        let b = root(&[-2.0], &[-1.0], 1e-9);
        assert!(matches!(
            certify_root(&a, &b),
            Err(VerdictError::NotComparable(_))
        ));
    }

    #[test]
    fn a_multi_dimensional_root_is_judged_by_its_worst_component() {
        let a = root(&[1.0, 2.0], &[0.5, 2.5], 1e-9);
        let b = root(&[1.0, 2.5], &[0.5, 2.5], 1e-9);
        let agreement = certify_root(&a, &b).unwrap();
        assert!(!agreement.agrees);
        assert_eq!(agreement.at, Some(1), "the second component diverged");
    }
}
