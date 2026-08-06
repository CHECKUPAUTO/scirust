//! Bayes factors for theory comparison — the statistics `sos-theory` defers.
//!
//! `sos-theory` ranks rivals by a *supplied* weight of evidence and computes
//! none itself (Invariant VIII). This module computes it, from real
//! likelihoods: given data observed once and two rival hypotheses that each
//! assign it a probability, the Bayes factor is the ratio of those
//! probabilities, and its `log₁₀` is the weight of evidence in **bans**.
//!
//! ```text
//! log₁₀ BF(a : b) = ( Σᵢ ln pₐ(kᵢ) − Σᵢ ln p_b(kᵢ) ) / ln 10
//! ```
//!
//! ## Exact, and therefore `L3`
//!
//! Unlike [`crate::nmc`]'s nested-Monte-Carlo EIG, nothing here is sampled.
//! `scirust-stats`' [`DiscreteDistribution::ln_pmf`] is a closed form, the
//! observations are finite, and the sum is a fixed sequence of `f64`
//! operations with no adaptive branching. So this tier is honestly `L3`,
//! seedless-deterministic — the same classification the closed-form GP EIG
//! estimator carries, and for the same reason.
//!
//! ## Independence is an assumption, and it is the caller's
//!
//! Summing per-observation log-likelihoods assumes the observations are
//! conditionally independent given the hypothesis. That is a modelling
//! choice about the experiment, not a property of the arithmetic, and this
//! module cannot check it. [`log10_bayes_factor`] therefore names it in the
//! signature's documentation rather than burying it: pass data you are
//! willing to treat as i.i.d. under both hypotheses, or compute the joint
//! likelihood yourself and pass a single observation.
//!
//! ## Impossible data, handled rather than papered over
//!
//! `ln_pmf` returns `−∞` outside a distribution's support, which is the
//! honest answer — the hypothesis says the data *cannot* happen.
//!
//! * **Impossible under exactly one hypothesis.** The Bayes factor is
//!   infinite: the data refutes that rival outright. Reported as a saturated
//!   [`i64::MAX`]/[`i64::MIN`] millibans, because the value exists to *order*
//!   rivals and "refuted" is the strongest orderable position. The saturation
//!   is flagged on the result so a caller reporting a number rather than a
//!   ranking can say "decisive" instead of quoting a spurious magnitude.
//! * **Impossible under both.** The ratio is `∞/∞` — undefined, not
//!   enormous. That is [`ScirustError::UndefinedBayesFactor`], never a
//!   number. Both hypotheses failed; a comparison between them is meaningless
//!   and returning `0` ("neutral") would be the worst possible lie about it.

use scirust_stats::DiscreteDistribution;
use sos_core::DeterminismLevel;
use sos_theory::WeightOfEvidence;

use crate::error::{Result, ScirustError};

/// Millibans per ban, mirroring `sos-planner`'s millibits-per-bit scale.
pub const MILLIBANS_PER_BAN: i64 = 1_000;

/// The determinism level every result here carries: exact, seedless.
pub const BAYES_LEVEL: DeterminismLevel = DeterminismLevel::L3;

/// A computed weight of evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BayesFactor {
    /// `1000 · log₁₀ BF(a : b)`. Positive favours the first hypothesis.
    pub milli_bans: i64,
    /// Whether the value saturated because the data is impossible under one
    /// hypothesis — the evidence is *decisive*, and the magnitude printed is
    /// a bound rather than a measurement.
    pub decisive: bool,
}

impl BayesFactor {
    /// This weight, addressed to `theory`, ready for
    /// [`sos_theory::TheoryEngine::compare_by_evidence`].
    #[must_use]
    pub fn for_theory(&self, theory: sos_core::ObjectId) -> WeightOfEvidence {
        WeightOfEvidence::new(theory, self.milli_bans)
    }
}

/// The weight of evidence `data` gives to `a` over `b`, in millibans.
///
/// Assumes the observations are conditionally independent given the
/// hypothesis — see the module docs; that assumption is the caller's.
///
/// # Errors
/// [`ScirustError::NoObservations`] if `data` is empty — no data is no
/// evidence, and the neutral `0` it would otherwise produce is a claim.
/// [`ScirustError::UndefinedBayesFactor`] if the data is impossible under
/// **both** hypotheses. [`ScirustError::NonFiniteLikelihood`] if a
/// distribution returns a `NaN` log-likelihood.
pub fn log10_bayes_factor(
    data: &[u64],
    a: &dyn DiscreteDistribution,
    b: &dyn DiscreteDistribution,
) -> Result<BayesFactor> {
    if data.is_empty()
    {
        return Err(ScirustError::NoObservations);
    }
    let ln_a = total_ln_likelihood(data, a)?;
    let ln_b = total_ln_likelihood(data, b)?;

    match (ln_a.is_infinite(), ln_b.is_infinite())
    {
        // Impossible under both: ∞/∞, not "enormous".
        (true, true) => Err(ScirustError::UndefinedBayesFactor),
        // Impossible under exactly one: decisive, saturated for ordering.
        (false, true) => Ok(BayesFactor {
            milli_bans: i64::MAX,
            decisive: true,
        }),
        (true, false) => Ok(BayesFactor {
            milli_bans: i64::MIN,
            decisive: true,
        }),
        (false, false) =>
        {
            let bans = (ln_a - ln_b) / std::f64::consts::LN_10;
            let scaled = bans * MILLIBANS_PER_BAN as f64;
            // A finite log-likelihood difference can still overflow i64 for
            // absurdly separated hypotheses; clamp rather than wrap, and do
            // not call it decisive — it is measured, merely unrepresentable.
            let milli_bans = if scaled >= i64::MAX as f64
            {
                i64::MAX
            }
            else if scaled <= i64::MIN as f64
            {
                i64::MIN
            }
            else
            {
                scaled as i64
            };
            Ok(BayesFactor {
                milli_bans,
                decisive: false,
            })
        },
    }
}

/// Σ ln p(kᵢ). `−∞` when any observation is outside the support.
fn total_ln_likelihood(data: &[u64], d: &dyn DiscreteDistribution) -> Result<f64> {
    let mut total = 0.0_f64;
    for &k in data
    {
        let ln_p = d.ln_pmf(k);
        if ln_p.is_nan()
        {
            return Err(ScirustError::NonFiniteLikelihood(k));
        }
        if ln_p.is_infinite()
        {
            return Ok(f64::NEG_INFINITY);
        }
        total += ln_p;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_stats::Poisson;
    use sos_core::{HashAlgo, ObjectId};

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::Sha256, b"test", tag)
    }

    #[test]
    fn data_typical_of_a_favours_a() {
        // Counts around 2 are far likelier under Poisson(2) than Poisson(10).
        let bf =
            log10_bayes_factor(&[1, 2, 2, 3], &Poisson::new(2.0), &Poisson::new(10.0)).unwrap();
        assert!(bf.milli_bans > 0, "got {}", bf.milli_bans);
        assert!(!bf.decisive);
    }

    #[test]
    fn the_factor_is_antisymmetric() {
        let data = [1, 2, 2, 3];
        let ab = log10_bayes_factor(&data, &Poisson::new(2.0), &Poisson::new(10.0)).unwrap();
        let ba = log10_bayes_factor(&data, &Poisson::new(10.0), &Poisson::new(2.0)).unwrap();
        assert_eq!(ab.milli_bans, -ba.milli_bans);
    }

    #[test]
    fn identical_hypotheses_give_exactly_neutral_evidence() {
        // Not "approximately zero" — the same distribution twice must cancel.
        let bf = log10_bayes_factor(&[1, 5, 9], &Poisson::new(3.0), &Poisson::new(3.0)).unwrap();
        assert_eq!(bf.milli_bans, 0);
        assert!(!bf.decisive);
    }

    #[test]
    fn more_data_of_the_same_kind_strengthens_the_evidence() {
        let few = log10_bayes_factor(&[2, 2], &Poisson::new(2.0), &Poisson::new(10.0)).unwrap();
        let many = log10_bayes_factor(&[2, 2, 2, 2, 2, 2], &Poisson::new(2.0), &Poisson::new(10.0))
            .unwrap();
        assert!(many.milli_bans > few.milli_bans);
    }

    #[test]
    fn it_matches_the_closed_form_it_claims_to_compute() {
        // The arithmetic, checked against the definition rather than against
        // itself: log10 BF = (Σ ln p_a − Σ ln p_b) / ln 10.
        let (a, b) = (Poisson::new(2.0), Poisson::new(10.0));
        let data = [1_u64, 4, 7];
        let expected: f64 = (data.iter().map(|&k| a.ln_pmf(k)).sum::<f64>()
            - data.iter().map(|&k| b.ln_pmf(k)).sum::<f64>())
            / std::f64::consts::LN_10
            * 1000.0;
        let got = log10_bayes_factor(&data, &a, &b).unwrap();
        assert!(
            (f64::from(i32::try_from(got.milli_bans).unwrap()) - expected).abs() < 1.0,
            "{} vs {expected}",
            got.milli_bans
        );
    }

    #[test]
    fn no_observations_is_an_error_not_neutral_evidence() {
        // Zero millibans asserts the evidence is balanced. No data asserts
        // nothing, and must not be dressed up as the former.
        assert!(matches!(
            log10_bayes_factor(&[], &Poisson::new(2.0), &Poisson::new(3.0)),
            Err(ScirustError::NoObservations)
        ));
    }

    #[test]
    fn is_bit_reproducible() {
        let run =
            || log10_bayes_factor(&[1, 2, 3], &Poisson::new(2.0), &Poisson::new(5.0)).unwrap();
        assert_eq!(run(), run(), "exact and seedless: L3");
    }

    #[test]
    fn a_weight_addressed_to_a_theory_is_what_sos_theory_consumes() {
        let bf = log10_bayes_factor(&[2, 2], &Poisson::new(2.0), &Poisson::new(9.0)).unwrap();
        let w = bf.for_theory(oid(b"theory-a"));
        assert_eq!(w.theory, oid(b"theory-a"));
        assert_eq!(w.milli_bans, bf.milli_bans);
    }

    #[test]
    fn the_declared_level_is_l3_because_nothing_is_sampled() {
        assert_eq!(BAYES_LEVEL, DeterminismLevel::L3);
    }
}
