//! **Theory revision**: what new evidence does to assumptions already relied
//! on, and to the claims that relied on them.
//!
//! # The gap this fills
//!
//! Every certificate in this crate is a conditional: *under assumptions A,
//! property Q is identifiable, estimated by M with uncertainty U*. Phases 5C.4
//! through 5C.8 are increasingly careful about stating A. None of them does
//! anything when A later turns out to be wrong.
//!
//! That gap is not hypothetical here. `crate::effect_estimation`'s own
//! adversarial test certifies an estimate as `Identifiable` that is wrong by
//! more than a hundred standard errors, because a latent confounder violates
//! the causal sufficiency it assumed. `crate::invariance` can *detect* that
//! something is wrong. Until now nothing connected the two: the detection
//! landed in one result object and the bad certificate sat in another,
//! unchanged.
//!
//! This module connects them. Evidence revises the registry; the revision
//! re-audits certificates; a claim whose ground has moved says so.
//!
//! # Evidence bears on a conjunction, and usually cannot be attributed
//!
//! The central design decision is that [`AssumptionEvidence`] names a *list*
//! of assumptions, not one.
//!
//! A test that names **one** assumption attributes its verdict to that
//! assumption: contradicting evidence makes it
//! [`RevisionVerdict::Contradicted`], and any certificate that used it is
//! [`CertificateStanding::Retracted`].
//!
//! A test that names **several** falsified their conjunction and cannot say
//! which member broke. Every member becomes
//! [`RevisionVerdict::JointlyContradicted`], and a certificate that used one
//! of them is [`CertificateStanding::InDoubt`] — not retracted, because
//! nothing established that *this* assumption is the false one.
//!
//! That is not a technicality. Invariant Causal Prediction reporting
//! `AssumptionsViolated` is exactly this case, and `crate::invariance`'s own
//! documentation says so: it detects that *something* is wrong and "does not
//! say what". Recording its verdict against causal sufficiency alone would
//! manufacture an attribution the test never made.
//!
//! # Contradiction is not outvoted
//!
//! An assumption with both supporting and contradicting evidence is reported
//! contradicted, whatever the counts. Corroboration cannot cancel a
//! falsification; the framework is asymmetric by construction and the counts
//! are reported so a reader can judge.
//!
//! The mirror of that asymmetry: **corroboration never proves**. Supporting
//! evidence lifts a basis from "asserted" or "unverified" to "tested and not
//! contradicted", which is a statement about what was looked for and not
//! found. Nothing here can move an assumption to *true*, and no combination
//! of supporting evidence produces a certificate stronger than the one whose
//! assumptions it corroborates.

use crate::assumptions::{AssumptionBasis, AssumptionRegistry, CausalAssumption};
use crate::certificate::{CausalCertificate, IdentifiabilityStatus};
use crate::error::CausalError;
use std::collections::{BTreeMap, BTreeSet};

/// Which way a piece of evidence points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EvidenceDirection {
    /// The evidence is consistent with the assumption(s) — a check that could
    /// have failed and did not.
    Supports,
    /// The evidence points against the assumption(s).
    Contradicts,
    /// The check ran but decided nothing. Recorded rather than discarded, so
    /// "we looked and could not tell" stays distinguishable from "we did not
    /// look".
    Inconclusive,
}

/// One finding bearing on one or more assumptions.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssumptionEvidence {
    /// The assumptions this bears on. **One** entry attributes the verdict to
    /// that assumption. **Several** means the evidence bears on their
    /// conjunction and cannot attribute to any single member — see the module
    /// docs, and prefer this form whenever the test genuinely cannot say
    /// which assumption it broke.
    pub assumptions: Vec<CausalAssumption>,
    pub direction: EvidenceDirection,
    /// What produced this finding, e.g. `"invariant_causal_prediction"`.
    pub source: String,
    /// Human-readable specifics, carried into the revised registry's note.
    pub detail: String,
}

impl AssumptionEvidence {
    /// Evidence attributable to a single assumption.
    pub fn attributed(
        assumption: CausalAssumption,
        direction: EvidenceDirection,
        source: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            assumptions: vec![assumption],
            direction,
            source: source.into(),
            detail: detail.into(),
        }
    }

    /// Evidence bearing on a conjunction, attributable to no single member.
    pub fn joint(
        assumptions: Vec<CausalAssumption>,
        direction: EvidenceDirection,
        source: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            assumptions,
            direction,
            source: source.into(),
            detail: detail.into(),
        }
    }
}

/// What the evidence did to one assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RevisionVerdict {
    /// Contradicted by evidence naming this assumption alone.
    Contradicted,
    /// A member of a conjunction that was falsified. At least one member is
    /// false; which one is not determined by this evidence.
    JointlyContradicted,
    /// Supporting evidence arrived and none contradicted. This is "tested and
    /// not contradicted", never "true".
    Corroborated,
    /// Evidence arrived but decided nothing.
    Inconclusive,
    /// No evidence bore on this assumption at all.
    Untouched,
}

impl RevisionVerdict {
    /// Whether this verdict undermines a claim that relied on the assumption.
    #[must_use]
    pub fn undermines(self) -> bool {
        matches!(self, Self::Contradicted | Self::JointlyContradicted)
    }
}

/// One assumption's revision.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssumptionOutcome {
    pub assumption: CausalAssumption,
    pub prior_basis: AssumptionBasis,
    pub revised_basis: AssumptionBasis,
    pub verdict: RevisionVerdict,
    pub supporting: usize,
    pub contradicting: usize,
    pub inconclusive: usize,
}

/// The result of applying evidence to a registry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssumptionRevision {
    /// Every registered assumption, in deterministic order, with what
    /// happened to it — including the ones nothing touched, so the reader can
    /// see what was *not* examined.
    pub outcomes: Vec<AssumptionOutcome>,
    /// The registry after revision. Contradicted entries carry
    /// [`AssumptionBasis::ContradictedByEvidence`], so a later reader cannot
    /// mistake them for merely unverified.
    pub revised_registry: AssumptionRegistry,
    pub certificate: CausalCertificate,
    pub warnings: Vec<String>,
}

impl AssumptionRevision {
    /// Assumptions this revision found directly contradicted.
    #[must_use]
    pub fn contradicted(&self) -> Vec<CausalAssumption> {
        self.outcomes
            .iter()
            .filter(|o| o.verdict == RevisionVerdict::Contradicted)
            .map(|o| o.assumption.clone())
            .collect()
    }

    /// Assumptions belonging to a falsified conjunction.
    #[must_use]
    pub fn jointly_contradicted(&self) -> Vec<CausalAssumption> {
        self.outcomes
            .iter()
            .filter(|o| o.verdict == RevisionVerdict::JointlyContradicted)
            .map(|o| o.assumption.clone())
            .collect()
    }

    fn verdict_for(&self, assumption: &CausalAssumption) -> Option<RevisionVerdict> {
        self.outcomes
            .iter()
            .find(|o| &o.assumption == assumption)
            .map(|o| o.verdict)
    }
}

/// Where a certificate stands after a revision.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CertificateStanding {
    /// Nothing the certificate relied on was undermined. This is not a
    /// re-derivation: it says the ground has not moved, not that the claim
    /// was ever right.
    Stands,
    /// An assumption it relied on was contradicted outright. **The estimate
    /// must not be used.**
    Retracted { contradicted: Vec<CausalAssumption> },
    /// An assumption it relied on belongs to a conjunction that was
    /// falsified. Which member broke is unknown, so this is not a retraction
    /// — but **the estimate must not be used** either, because the claim's
    /// premises are no longer jointly tenable.
    InDoubt {
        jointly_contradicted: Vec<CausalAssumption>,
    },
    /// The certificate cites an assumption the registry has no record of, so
    /// its standing cannot be determined. A gap, reported rather than
    /// resolved in either direction.
    Unauditable { unknown: Vec<CausalAssumption> },
}

impl CertificateStanding {
    /// Whether the certificate's estimate may still be quoted.
    ///
    /// `false` for everything except [`CertificateStanding::Stands`] — an
    /// undetermined standing is not permission.
    #[must_use]
    pub fn estimate_is_usable(&self) -> bool {
        matches!(self, Self::Stands)
    }
}

/// A certificate re-audited against a revision.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CertificateAudit {
    pub query: String,
    pub original_status: IdentifiabilityStatus,
    pub original_estimate: Option<f64>,
    pub standing: CertificateStanding,
    /// A certificate describing the audit itself, so a retraction is as
    /// citable as the claim it retracts.
    pub certificate: CausalCertificate,
    pub warnings: Vec<String>,
}

const SCOPE_NOTE: &str = "This records what evidence did to stated assumptions, and which claims \
                          that undermines. Corroboration means a check that could have failed did \
                          not; nothing here establishes an assumption as true, and no revision \
                          re-derives an estimate.";

/// What one assumption accumulated across the evidence set: counts by
/// direction, whether any contradiction named it *alone* (and so is
/// attributable to it), the sources of those contradictions, and the other
/// members of any conjunction it was falsified as part of.
#[derive(Debug, Default, Clone)]
struct Tally {
    supporting: usize,
    contradicting: usize,
    inconclusive: usize,
    attributable: bool,
    contradiction_sources: Vec<String>,
    co_members: BTreeSet<CausalAssumption>,
}

fn tally(evidence: &[AssumptionEvidence]) -> BTreeMap<CausalAssumption, Tally> {
    let mut counts: BTreeMap<CausalAssumption, Tally> = BTreeMap::new();
    for item in evidence
    {
        let attributed = item.assumptions.len() == 1;
        for assumption in &item.assumptions
        {
            let slot = counts.entry(assumption.clone()).or_default();
            match item.direction
            {
                EvidenceDirection::Supports => slot.supporting += 1,
                EvidenceDirection::Contradicts =>
                {
                    slot.contradicting += 1;
                    slot.attributable |= attributed;
                    slot.contradiction_sources.push(item.source.clone());
                    for other in &item.assumptions
                    {
                        if other != assumption
                        {
                            slot.co_members.insert(other.clone());
                        }
                    }
                },
                EvidenceDirection::Inconclusive => slot.inconclusive += 1,
            }
        }
    }
    counts
}

/// Applies `evidence` to `registry`, producing a revised registry and a
/// per-assumption account of what changed.
///
/// Evidence naming an assumption the registry does not hold is recorded as a
/// warning and otherwise ignored: this function revises beliefs, it does not
/// invent them.
///
/// # Errors
///
/// [`CausalError::InvalidContract`] if any evidence item names no assumptions
/// or carries an empty source, or if the certificate cannot be built.
pub fn revise_assumptions(
    registry: &AssumptionRegistry,
    evidence: &[AssumptionEvidence],
) -> Result<AssumptionRevision, CausalError> {
    for item in evidence
    {
        if item.assumptions.is_empty()
        {
            return Err(CausalError::InvalidContract {
                detail: "evidence must name at least one assumption it bears on",
            });
        }
        if item.source.trim().is_empty()
        {
            return Err(CausalError::InvalidContract {
                detail: "evidence must name its source",
            });
        }
    }

    let counts = tally(evidence);
    let mut warnings = Vec::new();
    for assumption in counts.keys()
    {
        if registry.get(assumption).is_none()
        {
            warnings.push(format!(
                "evidence bears on {assumption:?}, which is not in the registry; it was not added \
                 — this function revises stated assumptions, it does not invent them"
            ));
        }
    }

    let mut revised_registry = AssumptionRegistry::new();
    let mut outcomes = Vec::new();
    for (assumption, record) in registry.iter()
    {
        let Tally {
            supporting,
            contradicting,
            inconclusive,
            attributable,
            contradiction_sources,
            co_members,
        } = counts.get(assumption).cloned().unwrap_or_default();

        let verdict = if contradicting > 0 && attributable
        {
            RevisionVerdict::Contradicted
        }
        else if contradicting > 0
        {
            RevisionVerdict::JointlyContradicted
        }
        else if supporting > 0
        {
            RevisionVerdict::Corroborated
        }
        else if inconclusive > 0
        {
            RevisionVerdict::Inconclusive
        }
        else
        {
            RevisionVerdict::Untouched
        };

        if contradicting > 0 && supporting > 0
        {
            warnings.push(format!(
                "{assumption:?} drew {supporting} supporting and {contradicting} contradicting \
                 finding(s); the contradiction stands, because corroboration cannot cancel a \
                 falsification — but a test at level alpha fires spuriously with probability \
                 alpha, so read the counts"
            ));
        }
        if verdict.undermines()
            && matches!(record.basis, AssumptionBasis::GuaranteedByDesign { .. })
        {
            warnings.push(format!(
                "{assumption:?} was recorded as guaranteed by design, and evidence now points \
                 against it; that indicates the design itself did not hold, not merely that a \
                 belief was optimistic"
            ));
        }

        let revised_basis = match verdict
        {
            RevisionVerdict::Contradicted | RevisionVerdict::JointlyContradicted =>
            {
                AssumptionBasis::ContradictedByEvidence {
                    source: contradiction_sources.join(", "),
                    jointly_with: co_members.into_iter().collect(),
                }
            },
            RevisionVerdict::Corroborated => match &record.basis
            {
                // Corroboration lifts an unbacked belief to a tested one. It
                // never overwrites a stronger existing basis, and never
                // claims a p-value it was not given.
                AssumptionBasis::AssertedByAnalyst | AssumptionBasis::Unverified =>
                {
                    AssumptionBasis::TestedStatistically {
                        test_name: evidence
                            .iter()
                            .filter(|e| {
                                e.direction == EvidenceDirection::Supports
                                    && e.assumptions.contains(assumption)
                            })
                            .map(|e| e.source.clone())
                            .collect::<Vec<_>>()
                            .join(", "),
                        p_value: None,
                    }
                },
                other => other.clone(),
            },
            RevisionVerdict::Inconclusive | RevisionVerdict::Untouched => record.basis.clone(),
        };

        let note = match verdict
        {
            RevisionVerdict::Untouched => record.note.clone(),
            _ => Some(
                evidence
                    .iter()
                    .filter(|e| e.assumptions.contains(assumption))
                    .map(|e| format!("[{}] {}", e.source, e.detail))
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
        };
        revised_registry.overwrite(assumption.clone(), revised_basis.clone(), note);

        outcomes.push(AssumptionOutcome {
            assumption: assumption.clone(),
            prior_basis: record.basis.clone(),
            revised_basis,
            verdict,
            supporting,
            contradicting,
            inconclusive,
        });
    }

    let directly = outcomes
        .iter()
        .filter(|o| o.verdict == RevisionVerdict::Contradicted)
        .count();
    let jointly = outcomes
        .iter()
        .filter(|o| o.verdict == RevisionVerdict::JointlyContradicted)
        .count();
    let status = if directly > 0 || jointly > 0
    {
        IdentifiabilityStatus::NotIdentifiable
    }
    else
    {
        IdentifiabilityStatus::Identifiable
    };

    let mut builder = CausalCertificate::builder(
        format!(
            "what do {} finding(s) do to {} stated assumption(s)?",
            evidence.len(),
            registry.len()
        ),
        status,
        vec![],
        format!(
            "{directly} assumption(s) directly contradicted, {jointly} member(s) of a falsified \
             conjunction, {} corroborated, {} untouched",
            outcomes
                .iter()
                .filter(|o| o.verdict == RevisionVerdict::Corroborated)
                .count(),
            outcomes
                .iter()
                .filter(|o| o.verdict == RevisionVerdict::Untouched)
                .count(),
        ),
    )
    .with_sensitivity(SCOPE_NOTE);
    for outcome in outcomes.iter().filter(|o| o.verdict.undermines())
    {
        builder = builder.with_unresolved_alternative(format!(
            "{:?} is {:?}",
            outcome.assumption, outcome.verdict
        ));
    }

    Ok(AssumptionRevision {
        outcomes,
        revised_registry,
        certificate: builder.finalize()?,
        warnings,
    })
}

/// Re-audits `certificate` against a revision, reporting whether the claim
/// still stands.
///
/// This never re-derives anything. It compares the assumptions the
/// certificate declared against what the revision found, and reports the
/// worst applicable standing — a retraction dominates a doubt, which
/// dominates an unauditable gap.
///
/// # Errors
///
/// [`CausalError::InvalidContract`] if the audit certificate cannot be built.
pub fn audit_certificate(
    certificate: &CausalCertificate,
    revision: &AssumptionRevision,
) -> Result<CertificateAudit, CausalError> {
    let mut contradicted = Vec::new();
    let mut jointly = Vec::new();
    let mut unknown = Vec::new();
    for assumption in certificate.assumptions_used()
    {
        match revision.verdict_for(assumption)
        {
            Some(RevisionVerdict::Contradicted) => contradicted.push(assumption.clone()),
            Some(RevisionVerdict::JointlyContradicted) => jointly.push(assumption.clone()),
            Some(_) =>
            {},
            None => unknown.push(assumption.clone()),
        }
    }

    let standing = if !contradicted.is_empty()
    {
        CertificateStanding::Retracted {
            contradicted: contradicted.clone(),
        }
    }
    else if !jointly.is_empty()
    {
        CertificateStanding::InDoubt {
            jointly_contradicted: jointly.clone(),
        }
    }
    else if !unknown.is_empty()
    {
        CertificateStanding::Unauditable {
            unknown: unknown.clone(),
        }
    }
    else
    {
        CertificateStanding::Stands
    };

    let mut warnings = Vec::new();
    if certificate.estimate().is_some() && !standing.estimate_is_usable()
    {
        warnings.push(format!(
            "the audited certificate carries an estimate and its standing is {standing:?}; that \
             number must not be quoted without re-establishing the assumptions beneath it"
        ));
    }

    let (status, evidence_summary) = match &standing
    {
        CertificateStanding::Stands => (
            IdentifiabilityStatus::Identifiable,
            "no assumption this claim declared was undermined".to_string(),
        ),
        CertificateStanding::Retracted { contradicted } => (
            IdentifiabilityStatus::NotIdentifiable,
            format!(
                "{} declared assumption(s) contradicted outright",
                contradicted.len()
            ),
        ),
        CertificateStanding::InDoubt {
            jointly_contradicted,
        } => (
            IdentifiabilityStatus::Inconclusive,
            format!(
                "{} declared assumption(s) belong to a falsified conjunction; which member broke \
                 is not determined",
                jointly_contradicted.len()
            ),
        ),
        CertificateStanding::Unauditable { unknown } => (
            IdentifiabilityStatus::Inconclusive,
            format!(
                "{} declared assumption(s) have no registry entry, so standing is undetermined",
                unknown.len()
            ),
        ),
    };

    let mut builder = CausalCertificate::builder(
        format!("does the claim \"{}\" still stand?", certificate.query()),
        status,
        certificate.assumptions_used().to_vec(),
        evidence_summary,
    )
    .with_sensitivity(format!(
        "{SCOPE_NOTE} A standing of Stands means the ground has not moved, not that the original \
         claim was ever correct."
    ));
    for assumption in contradicted.iter().chain(jointly.iter())
    {
        builder = builder
            .with_unresolved_alternative(format!("{assumption:?} no longer supports this claim"));
    }

    Ok(CertificateAudit {
        query: certificate.query().to_string(),
        original_status: certificate.status(),
        original_estimate: certificate.estimate(),
        standing,
        certificate: builder.finalize()?,
        warnings,
    })
}

/// Turns an [`crate::InvariantPredictionResult`] into evidence.
///
/// The mapping is deliberately conservative. Invariant Causal Prediction
/// reporting [`crate::InvariantPredictionOutcome::AssumptionsViolated`]
/// establishes that *something* among its premises is false, and its own
/// documentation records that it cannot say what. So this emits **joint**
/// evidence against the conjunction, never attributed evidence against causal
/// sufficiency alone — which would be an attribution the test never made.
///
/// A successful run emits *supporting* evidence for the same conjunction: a
/// check that could have failed and did not. An inconclusive run emits
/// inconclusive evidence rather than nothing, so "we looked and could not
/// tell" stays visible.
#[must_use]
pub fn evidence_from_invariance(
    result: &crate::invariance::InvariantPredictionResult,
) -> Vec<AssumptionEvidence> {
    use crate::invariance::InvariantPredictionOutcome;
    let conjunction = vec![
        CausalAssumption::CausalSufficiency,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::InvarianceAcrossEnvironments,
    ];
    let (direction, detail) = match &result.outcome
    {
        InvariantPredictionOutcome::AssumptionsViolated => (
            EvidenceDirection::Contradicts,
            "no predictor subset was invariant across the supplied environments; at least one of \
             causal sufficiency, correct functional form, and cross-environment invariance fails, \
             and this test cannot say which"
                .to_string(),
        ),
        InvariantPredictionOutcome::CausalPredictorsIdentified => (
            EvidenceDirection::Supports,
            format!(
                "an invariant subset survived and the intersection names {} predictor(s); the \
                 conjunction was checked and not contradicted, which is not the same as \
                 established",
                result.causal_predictors.len()
            ),
        ),
        InvariantPredictionOutcome::NoPredictorConfirmed => (
            EvidenceDirection::Inconclusive,
            "subsets survived but their intersection was empty; the test ran and decided nothing"
                .to_string(),
        ),
    };
    vec![AssumptionEvidence::joint(
        conjunction,
        direction,
        "invariant_causal_prediction",
        detail,
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> AssumptionRegistry {
        let mut registry = AssumptionRegistry::new();
        registry
            .assert(
                CausalAssumption::CausalSufficiency,
                AssumptionBasis::AssertedByAnalyst,
                None,
            )
            .unwrap();
        registry
            .assert(
                CausalAssumption::Faithfulness,
                AssumptionBasis::Unverified,
                None,
            )
            .unwrap();
        registry
            .assert(
                CausalAssumption::Acyclicity,
                AssumptionBasis::DomainKnowledge {
                    citation: "process physics".to_string(),
                },
                None,
            )
            .unwrap();
        registry
    }

    fn certificate(assumptions: Vec<CausalAssumption>, estimate: Option<f64>) -> CausalCertificate {
        let builder = CausalCertificate::builder(
            "effect of X on Y",
            IdentifiabilityStatus::Identifiable,
            assumptions,
            "backdoor adjustment",
        );
        match estimate
        {
            Some(value) => builder.with_estimate("linear adjustment", value, 0.01),
            None => builder,
        }
        .finalize()
        .unwrap()
    }

    #[test]
    fn attributed_contradiction_retracts_the_claim_that_used_it() {
        let revision = revise_assumptions(
            &registry(),
            &[AssumptionEvidence::attributed(
                CausalAssumption::CausalSufficiency,
                EvidenceDirection::Contradicts,
                "a measured confounder was found",
                "the omitted variable was located and is associated with both",
            )],
        )
        .unwrap();
        assert_eq!(
            revision.contradicted(),
            vec![CausalAssumption::CausalSufficiency]
        );

        let audit = audit_certificate(
            &certificate(vec![CausalAssumption::CausalSufficiency], Some(1.5)),
            &revision,
        )
        .unwrap();
        assert_eq!(
            audit.standing,
            CertificateStanding::Retracted {
                contradicted: vec![CausalAssumption::CausalSufficiency],
            }
        );
        assert!(!audit.standing.estimate_is_usable());
        assert_eq!(audit.original_estimate, Some(1.5));
        assert!(
            audit
                .warnings
                .iter()
                .any(|w| w.contains("must not be quoted"))
        );
    }

    #[test]
    fn joint_contradiction_puts_a_claim_in_doubt_without_attributing_blame() {
        let revision = revise_assumptions(
            &registry(),
            &[AssumptionEvidence::joint(
                vec![
                    CausalAssumption::CausalSufficiency,
                    CausalAssumption::Faithfulness,
                ],
                EvidenceDirection::Contradicts,
                "a misspecification test",
                "the conjunction failed",
            )],
        )
        .unwrap();
        assert!(
            revision.contradicted().is_empty(),
            "nothing is attributable"
        );
        assert_eq!(revision.jointly_contradicted().len(), 2);

        let audit = audit_certificate(
            &certificate(vec![CausalAssumption::CausalSufficiency], Some(1.5)),
            &revision,
        )
        .unwrap();
        assert!(matches!(
            audit.standing,
            CertificateStanding::InDoubt { .. }
        ));
        assert!(
            !audit.standing.estimate_is_usable(),
            "an undetermined standing is not permission to quote the number"
        );
    }

    #[test]
    fn the_revised_registry_distinguishes_contradicted_from_unverified() {
        let revision = revise_assumptions(
            &registry(),
            &[AssumptionEvidence::attributed(
                CausalAssumption::CausalSufficiency,
                EvidenceDirection::Contradicts,
                "test",
                "detail",
            )],
        )
        .unwrap();
        let record = revision
            .revised_registry
            .get(&CausalAssumption::CausalSufficiency)
            .unwrap();
        assert!(matches!(
            record.basis,
            AssumptionBasis::ContradictedByEvidence { .. }
        ));
        assert!(
            !revision
                .revised_registry
                .is_supported(&CausalAssumption::CausalSufficiency)
        );
        // Faithfulness was untouched and stays merely unverified — absence of
        // evidence, not evidence of absence.
        assert_eq!(
            revision
                .revised_registry
                .get(&CausalAssumption::Faithfulness)
                .unwrap()
                .basis,
            AssumptionBasis::Unverified
        );
    }

    #[test]
    fn corroboration_lifts_an_unbacked_belief_but_never_claims_proof() {
        let revision = revise_assumptions(
            &registry(),
            &[AssumptionEvidence::attributed(
                CausalAssumption::Faithfulness,
                EvidenceDirection::Supports,
                "a check that could have failed",
                "it did not",
            )],
        )
        .unwrap();
        let outcome = revision
            .outcomes
            .iter()
            .find(|o| o.assumption == CausalAssumption::Faithfulness)
            .unwrap();
        assert_eq!(outcome.verdict, RevisionVerdict::Corroborated);
        assert!(matches!(
            outcome.revised_basis,
            AssumptionBasis::TestedStatistically { p_value: None, .. }
        ));
        // A claim resting on it stands, but the audit says only that the
        // ground has not moved.
        let audit = audit_certificate(
            &certificate(vec![CausalAssumption::Faithfulness], None),
            &revision,
        )
        .unwrap();
        assert_eq!(audit.standing, CertificateStanding::Stands);
        assert!(
            audit
                .certificate
                .sensitivity_note()
                .unwrap()
                .contains("not that the original claim was ever correct")
        );
    }

    #[test]
    fn corroboration_never_overwrites_a_stronger_basis() {
        let revision = revise_assumptions(
            &registry(),
            &[AssumptionEvidence::attributed(
                CausalAssumption::Acyclicity,
                EvidenceDirection::Supports,
                "a weak check",
                "consistent",
            )],
        )
        .unwrap();
        let outcome = revision
            .outcomes
            .iter()
            .find(|o| o.assumption == CausalAssumption::Acyclicity)
            .unwrap();
        assert_eq!(outcome.prior_basis, outcome.revised_basis);
    }

    #[test]
    fn contradiction_is_not_outvoted_by_corroboration() {
        let revision = revise_assumptions(
            &registry(),
            &[
                AssumptionEvidence::attributed(
                    CausalAssumption::CausalSufficiency,
                    EvidenceDirection::Supports,
                    "check a",
                    "ok",
                ),
                AssumptionEvidence::attributed(
                    CausalAssumption::CausalSufficiency,
                    EvidenceDirection::Supports,
                    "check b",
                    "ok",
                ),
                AssumptionEvidence::attributed(
                    CausalAssumption::CausalSufficiency,
                    EvidenceDirection::Contradicts,
                    "check c",
                    "failed",
                ),
            ],
        )
        .unwrap();
        let outcome = revision
            .outcomes
            .iter()
            .find(|o| o.assumption == CausalAssumption::CausalSufficiency)
            .unwrap();
        assert_eq!(outcome.verdict, RevisionVerdict::Contradicted);
        assert_eq!((outcome.supporting, outcome.contradicting), (2, 1));
        assert!(
            revision
                .warnings
                .iter()
                .any(|w| w.contains("cannot cancel a falsification"))
        );
    }

    #[test]
    fn an_unregistered_assumption_makes_a_claim_unauditable() {
        let revision = revise_assumptions(&registry(), &[]).unwrap();
        let audit = audit_certificate(
            &certificate(vec![CausalAssumption::Positivity], Some(1.0)),
            &revision,
        )
        .unwrap();
        assert_eq!(
            audit.standing,
            CertificateStanding::Unauditable {
                unknown: vec![CausalAssumption::Positivity],
            }
        );
        assert!(!audit.standing.estimate_is_usable());
    }

    #[test]
    fn evidence_about_an_unregistered_assumption_is_warned_not_invented() {
        let revision = revise_assumptions(
            &registry(),
            &[AssumptionEvidence::attributed(
                CausalAssumption::Positivity,
                EvidenceDirection::Contradicts,
                "test",
                "detail",
            )],
        )
        .unwrap();
        assert!(
            revision
                .revised_registry
                .get(&CausalAssumption::Positivity)
                .is_none()
        );
        assert!(
            revision
                .warnings
                .iter()
                .any(|w| w.contains("not in the registry"))
        );
    }

    #[test]
    fn a_design_guarantee_that_fails_is_called_out_specifically() {
        let mut registry = AssumptionRegistry::new();
        registry
            .assert(
                CausalAssumption::Exchangeability,
                AssumptionBasis::GuaranteedByDesign {
                    mechanism: "randomization".to_string(),
                },
                None,
            )
            .unwrap();
        let revision = revise_assumptions(
            &registry,
            &[AssumptionEvidence::attributed(
                CausalAssumption::Exchangeability,
                EvidenceDirection::Contradicts,
                "balance check",
                "arms differ on a baseline covariate",
            )],
        )
        .unwrap();
        assert!(
            revision
                .warnings
                .iter()
                .any(|w| w.contains("the design itself did not hold"))
        );
    }

    #[test]
    fn untouched_assumptions_are_listed_rather_than_omitted() {
        let revision = revise_assumptions(&registry(), &[]).unwrap();
        assert_eq!(revision.outcomes.len(), 3);
        assert!(
            revision
                .outcomes
                .iter()
                .all(|o| o.verdict == RevisionVerdict::Untouched)
        );
        assert_eq!(
            revision.certificate.status(),
            IdentifiabilityStatus::Identifiable
        );
    }

    #[test]
    fn malformed_evidence_is_a_typed_error() {
        assert!(matches!(
            revise_assumptions(
                &registry(),
                &[AssumptionEvidence {
                    assumptions: vec![],
                    direction: EvidenceDirection::Contradicts,
                    source: "s".to_string(),
                    detail: "d".to_string(),
                }]
            ),
            Err(CausalError::InvalidContract { .. })
        ));
        assert!(matches!(
            revise_assumptions(
                &registry(),
                &[AssumptionEvidence::attributed(
                    CausalAssumption::Faithfulness,
                    EvidenceDirection::Contradicts,
                    "   ",
                    "d",
                )]
            ),
            Err(CausalError::InvalidContract { .. })
        ));
    }

    #[test]
    fn revision_is_deterministic() {
        let evidence = [AssumptionEvidence::joint(
            vec![
                CausalAssumption::CausalSufficiency,
                CausalAssumption::Faithfulness,
            ],
            EvidenceDirection::Contradicts,
            "test",
            "detail",
        )];
        let first = revise_assumptions(&registry(), &evidence).unwrap();
        let second = revise_assumptions(&registry(), &evidence).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.certificate.fingerprint(),
            second.certificate.fingerprint()
        );
    }
}
