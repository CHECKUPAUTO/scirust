//! Explicit trust models for identifiable majority contamination (phase 725).
//!
//! The historical robust fitter tolerates a *minority* of aberrant targets per
//! exact-source group. This module lets SRCC remain useful when aberrant
//! observations form a global numerical majority — **only under explicit,
//! machine-checkable identifiability assumptions**, never as a claim of
//! robustness to arbitrary majority corruption.
//!
//! # Principle
//!
//! If two incompatible explanations are equally consistent with the supplied
//! assumptions, the result is the typed
//! [`SrccTrustError::UnidentifiableContamination`] — never an arbitrary pick.
//!
//! # Mechanism
//!
//! Observations carry a [`SrccObservationTrust`] record (a finite non-negative
//! prior weight plus typed [`SrccTrustEvidence`]). A [`SrccTrustPolicy`] turns
//! evidence into per-observation weights and per-group acceptance requirements:
//!
//! - the target consensus of each exact-source group becomes the **weighted**
//!   medoid (`argmin_c Σᵢ wᵢ·d²(c, targetᵢ)`), evaluated over the exact same
//!   candidates in the exact same fixed order as the historical unweighted
//!   consensus — under [`SrccTrustPolicy::Unweighted`] every weight is `1.0`
//!   and `1.0 · x == x` in IEEE arithmetic, so the historical result is
//!   reproduced bit for bit;
//! - the **adversarial margin** requirement: with `support(c)` the summed
//!   weight of observations whose target is bit-identical to `c`, a policy
//!   bounding the corrupted weight by `β·W` per scope accepts only when
//!   `support(winner) − support(runner-up) > 2·β·W` — otherwise an adversary
//!   inside the declared bound could have flipped the outcome, and the typed
//!   unidentifiability error is returned instead;
//! - anchor and temporal policies gate weights to zero for observations that
//!   lack the required evidence, with typed failures when a group loses all
//!   its support or anchors disagree.
//!
//! Every accepted result carries a [`SrccTrustCertificate`] stating the
//! assumptions and the per-group supports and margins that justified it.
//!
//! # What this is not
//!
//! No policy here can identify the truth when the assumptions do not hold: a
//! 50/50 split of persistent, internally consistent regimes is unidentifiable
//! by construction and is reported as such (the benchmark demonstrates it).
//! Weights are **not** probabilities and are never multiplied across evidence
//! kinds; each policy states exactly how evidence gates weights.

use core::{cmp::Ordering, fmt};

use crate::robust::{compare_samples, squared_distance};
use crate::{
    SRCC_DIMENSION, SrccConfig, SrccFitResult, SrccProjector, SrccRobustFitError,
    SrccStabilityReport, SrccStabilityVariant, SrccTransportSample, Vector16,
    learn_transport_views,
};

/// Typed identifier of an evidence provider (no arbitrary strings).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SrccTrustProviderId(pub u16);

/// The kind of trust evidence attached to an observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrccTrustEvidenceKind {
    /// A calibration-grade trusted anchor. Score must be exactly `1.0`.
    TrustedAnchor,
    /// Consistency with an independent view. Score in `[0, 1]`.
    IndependentView,
    /// A temporal prediction check. Score is the non-negative prediction
    /// error (smaller is better) — **not** a confidence in `[0, 1]`.
    TemporalPrediction,
    /// Containment in a validated interval estimate. Score in `[0, 1]`.
    IntervalContainment,
    /// Consistency with a physical constraint. Score in `[0, 1]`.
    PhysicalConstraint,
    /// Consistency with statistical process control. Score in `[0, 1]`.
    ProcessControlConsistency,
    /// A conformal p-value. Score in `[0, 1]`.
    ConformalEvidence,
    /// External reliability information. Score in `[0, 1]`.
    ExternalReliability,
}

/// One piece of typed evidence about one observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccTrustEvidence {
    /// What kind of evidence this is (fixes the score semantics).
    pub kind: SrccTrustEvidenceKind,
    /// Which provider produced it.
    pub provider: SrccTrustProviderId,
    /// The score; semantics depend on `kind` (see the kind's documentation).
    pub score: f64,
}

/// Trust record for one observation, addressed in **caller order**
/// (`view_index`, `sample_index` before any canonical sorting).
#[derive(Clone, Debug, PartialEq)]
pub struct SrccObservationTrust {
    /// Index of the transport view in the caller's `views` slice.
    pub view_index: usize,
    /// Index of the sample inside that view, in caller order.
    pub sample_index: usize,
    /// Finite non-negative prior weight (the weight an accepted observation
    /// contributes to the consensus).
    pub prior_weight: f64,
    /// Typed evidence attached to this observation.
    pub evidence: Vec<SrccTrustEvidence>,
}

/// How evidence turns into weights and acceptance requirements.
///
/// Only mathematically meaningful policies are exposed; in particular the only
/// composite rule is the conjunction (`CompositeAll`) — a disjunction would
/// accept whenever the *weakest* assumption accepts, which certifies nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum SrccTrustPolicy {
    /// Every observation weighs `1.0`; no identifiability assumption is added.
    /// Reproduces the historical unweighted consensus bit for bit and remains
    /// silently wrong under majority corruption — exactly like the historical
    /// fitter, whose documented assumption (majority-clean groups) it keeps.
    Unweighted,
    /// Only observations carrying [`SrccTrustEvidenceKind::TrustedAnchor`]
    /// evidence (score exactly `1.0`) vote; every group must keep at least
    /// `minimum_anchor_support` anchors. Assumption: anchors are
    /// incorruptible; disagreeing anchors are a typed failure, and corrupted
    /// anchors therefore surface as failures rather than silent acceptance.
    TrustedAnchors {
        /// Minimum number of anchor observations required in every group.
        minimum_anchor_support: usize,
    },
    /// Assumption: within every view, adversarially corrupted observations
    /// carry at most a `maximum_corrupted_weight_per_view` fraction of that
    /// view's group weight. Acceptance requires the adversarial margin
    /// `support(winner) − support(runner-up) > 2·β·W_group` in every group of
    /// every view, and at least `minimum_consistent_views` views overall.
    IndependentViews {
        /// Minimum number of views that must be present.
        minimum_consistent_views: usize,
        /// Bound `β` on the corrupted weight fraction per view.
        maximum_corrupted_weight_per_view: f64,
    },
    /// Assumption: honest observations persist. An observation votes only if
    /// it carries at least `minimum_consistent_steps` temporal-prediction
    /// evidences whose error is at most `maximum_prediction_error`; short
    /// bursts lacking that history are gated out. A long coherent alternative
    /// regime *keeps* its vote and surfaces as unidentifiable rather than
    /// being silently rejected.
    TemporalPersistence {
        /// Required number of sufficiently accurate prediction steps.
        minimum_consistent_steps: usize,
        /// Maximum admissible prediction error per step.
        maximum_prediction_error: f64,
    },
    /// Assumption: within every exact-source group (across all views), the
    /// corrupted weight fraction is at most `maximum_corrupted_weight_per_group`.
    /// Acceptance requires the adversarial margin in every group.
    GroupContaminationBound {
        /// Bound `β` on the corrupted weight fraction per group.
        maximum_corrupted_weight_per_group: f64,
    },
    /// Conjunction: every sub-policy must accept; an observation's weight is
    /// the minimum of the weights the sub-policies assign it (zero if any
    /// gates it out). Conflicting sub-policies therefore fail loudly instead
    /// of averaging away.
    CompositeAll(
        /// The conjoined sub-policies (must be non-empty and non-composite).
        Vec<SrccTrustPolicy>,
    ),
}

/// A complete trust model: the policy plus per-observation records.
///
/// Observations without a record default to `prior_weight = 1.0` with no
/// evidence (explicitly documented: absence of evidence is not distrust; the
/// policy decides what un-evidenced observations may do).
#[derive(Clone, Debug, PartialEq)]
pub struct SrccTrustModel {
    /// The active policy.
    pub policy: SrccTrustPolicy,
    /// Per-observation trust records in caller addressing.
    pub observations: Vec<SrccObservationTrust>,
}

/// Provider interface for building trust evidence from observations.
///
/// This trait is the integration point for `scirust-estimation` /
/// `scirust-spc` / `scirust-pdm` style monitors (interval containment, Kalman
/// innovation consistency, CUSUM regime consistency, conformal p-values, …):
/// implement it next to the monitor and feed the collected evidence into a
/// [`SrccTrustModel`]. No heavy dependency is hard-wired here.
pub trait SrccTrustEvidenceProvider {
    /// The provider's stable identifier.
    fn provider_id(&self) -> SrccTrustProviderId;

    /// Evidence for one observation (caller addressing).
    fn evidence_for(
        &self,
        view_index: usize,
        sample_index: usize,
        sample: &SrccTransportSample,
    ) -> Result<Vec<SrccTrustEvidence>, SrccTrustError>;
}

/// Collects evidence from `providers` for every observation, with unit prior
/// weights, in deterministic (view-major, then sample, then provider) order.
pub fn collect_trust_evidence(
    views: &[&[SrccTransportSample]],
    providers: &[&dyn SrccTrustEvidenceProvider],
) -> Result<Vec<SrccObservationTrust>, SrccTrustError> {
    let mut observations = Vec::new();

    for (view_index, view) in views.iter().enumerate()
    {
        for (sample_index, sample) in view.iter().enumerate()
        {
            let mut evidence = Vec::new();

            for provider in providers
            {
                evidence.extend(provider.evidence_for(view_index, sample_index, sample)?);
            }

            observations.push(SrccObservationTrust {
                view_index,
                sample_index,
                prior_weight: 1.0,
                evidence,
            });
        }
    }

    Ok(observations)
}

/// Per-group acceptance record inside a [`SrccTrustCertificate`].
#[derive(Clone, Debug, PartialEq)]
pub struct SrccGroupTrustCertificate {
    /// The transport view.
    pub view_index: usize,
    /// The exact-source group inside the view (canonical order).
    pub source_group_index: usize,
    /// Total effective weight of the group.
    pub total_weight: f64,
    /// Summed weight of observations bit-identical to the elected target.
    pub winning_support: f64,
    /// Largest summed weight among distinct competing targets (`0.0` when the
    /// group is unanimous).
    pub runner_up_support: f64,
    /// Number of anchor observations that voted (anchor policies only).
    pub anchor_count: usize,
}

/// Machine-readable statement of why a trusted fit was accepted.
#[derive(Clone, Debug, PartialEq)]
pub struct SrccTrustCertificate {
    /// Human-readable statements of the assumptions the acceptance relies on.
    pub assumptions: Vec<String>,
    /// Per-group supports and margins.
    pub groups: Vec<SrccGroupTrustCertificate>,
    /// Total effective trusted weight across all groups.
    pub effective_trusted_weight: f64,
    /// Number of observations whose effective weight was gated to zero.
    pub gated_observation_count: usize,
}

/// Result of a trusted fit: the historical fit result plus the certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct SrccTrustedFitResult {
    /// The fitted transports and projector.
    pub fit: SrccFitResult,
    /// The acceptance certificate.
    pub certificate: SrccTrustCertificate,
}

/// Typed trust errors.
#[derive(Clone, Debug, PartialEq)]
pub enum SrccTrustError {
    /// An observation record points outside the supplied views.
    ObservationOutOfBounds {
        /// The offending view index.
        view_index: usize,
        /// The offending sample index.
        sample_index: usize,
    },
    /// Two records address the same observation.
    DuplicateObservation {
        /// The duplicated view index.
        view_index: usize,
        /// The duplicated sample index.
        sample_index: usize,
    },
    /// A prior weight is negative or non-finite.
    InvalidPriorWeight {
        /// The offending view index.
        view_index: usize,
        /// The offending sample index.
        sample_index: usize,
    },
    /// An evidence score violates its kind's documented range.
    InvalidEvidenceScore {
        /// The offending view index.
        view_index: usize,
        /// The offending sample index.
        sample_index: usize,
    },
    /// A policy parameter is invalid (bounds, emptiness, nesting).
    InvalidPolicy,
    /// Fewer views than the policy's `minimum_consistent_views`.
    InsufficientViews {
        /// Views required.
        required: usize,
        /// Views supplied.
        found: usize,
    },
    /// A group kept fewer anchors than `minimum_anchor_support`.
    InsufficientAnchorSupport {
        /// The view.
        view_index: usize,
        /// The group.
        source_group_index: usize,
        /// Anchors found.
        anchors: usize,
        /// Anchors required.
        required: usize,
    },
    /// Anchors inside one group disagree on the target: the anchor assumption
    /// itself is violated (corrupted anchors), reported explicitly.
    ConflictingAnchors {
        /// The view.
        view_index: usize,
        /// The group.
        source_group_index: usize,
    },
    /// Every observation of a group was gated to zero weight.
    AllObservationsUntrusted {
        /// The view.
        view_index: usize,
        /// The group.
        source_group_index: usize,
    },
    /// The supplied assumptions cannot distinguish competing explanations
    /// (tied or insufficiently separated weighted consensus).
    UnidentifiableContamination {
        /// Number of competing target hypotheses at the decision point.
        competing_hypotheses: usize,
    },
}

impl fmt::Display for SrccTrustError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ObservationOutOfBounds {
                view_index,
                sample_index,
            } => write!(
                formatter,
                "trust record ({view_index}, {sample_index}) is outside the supplied views"
            ),
            Self::DuplicateObservation {
                view_index,
                sample_index,
            } => write!(
                formatter,
                "trust record ({view_index}, {sample_index}) is duplicated"
            ),
            Self::InvalidPriorWeight {
                view_index,
                sample_index,
            } => write!(
                formatter,
                "trust record ({view_index}, {sample_index}) has a non-finite or negative prior weight"
            ),
            Self::InvalidEvidenceScore {
                view_index,
                sample_index,
            } => write!(
                formatter,
                "trust record ({view_index}, {sample_index}) carries an evidence score outside its kind's range"
            ),
            Self::InvalidPolicy => formatter.write_str("trust policy parameters are invalid"),
            Self::InsufficientViews { required, found } =>
            {
                write!(formatter, "policy requires {required} views, found {found}")
            },
            Self::InsufficientAnchorSupport {
                view_index,
                source_group_index,
                anchors,
                required,
            } => write!(
                formatter,
                "view {view_index}, group {source_group_index}: {anchors} anchors found, {required} required"
            ),
            Self::ConflictingAnchors {
                view_index,
                source_group_index,
            } => write!(
                formatter,
                "view {view_index}, group {source_group_index}: trusted anchors disagree on the target"
            ),
            Self::AllObservationsUntrusted {
                view_index,
                source_group_index,
            } => write!(
                formatter,
                "view {view_index}, group {source_group_index}: every observation was gated to zero weight"
            ),
            Self::UnidentifiableContamination {
                competing_hypotheses,
            } => write!(
                formatter,
                "the supplied trust assumptions cannot distinguish {competing_hypotheses} competing hypotheses"
            ),
        }
    }
}

impl std::error::Error for SrccTrustError {}

/// Errors of the trusted fit entry points: either the historical pipeline
/// failed, or the trust layer did.
#[derive(Clone, Debug, PartialEq)]
pub enum SrccTrustedFitError {
    /// The historical validation/discovery/closure pipeline failed.
    Fit(SrccRobustFitError),
    /// The trust layer failed.
    Trust(SrccTrustError),
}

impl fmt::Display for SrccTrustedFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Fit(error) => error.fmt(formatter),
            Self::Trust(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccTrustedFitError {}

impl From<SrccRobustFitError> for SrccTrustedFitError {
    fn from(error: SrccRobustFitError) -> Self {
        Self::Fit(error)
    }
}

impl From<SrccTrustError> for SrccTrustedFitError {
    fn from(error: SrccTrustError) -> Self {
        Self::Trust(error)
    }
}

// ---------------------------------------------------------------------------
// Weight resolution
// ---------------------------------------------------------------------------

/// One observation carried through canonical sorting with its trust data.
#[derive(Clone, Debug)]
struct TrustedSample {
    sample: SrccTransportSample,
    weight: f64,
    is_anchor: bool,
}

fn validate_policy(policy: &SrccTrustPolicy, nested: bool) -> Result<(), SrccTrustError> {
    match policy
    {
        SrccTrustPolicy::Unweighted => Ok(()),
        SrccTrustPolicy::TrustedAnchors {
            minimum_anchor_support,
        } =>
        {
            if *minimum_anchor_support == 0
            {
                return Err(SrccTrustError::InvalidPolicy);
            }

            Ok(())
        },
        SrccTrustPolicy::IndependentViews {
            minimum_consistent_views,
            maximum_corrupted_weight_per_view,
        } =>
        {
            if *minimum_consistent_views == 0
                || !maximum_corrupted_weight_per_view.is_finite()
                || *maximum_corrupted_weight_per_view < 0.0
                || *maximum_corrupted_weight_per_view >= 0.5
            {
                return Err(SrccTrustError::InvalidPolicy);
            }

            Ok(())
        },
        SrccTrustPolicy::TemporalPersistence {
            minimum_consistent_steps,
            maximum_prediction_error,
        } =>
        {
            if *minimum_consistent_steps == 0
                || !maximum_prediction_error.is_finite()
                || *maximum_prediction_error < 0.0
            {
                return Err(SrccTrustError::InvalidPolicy);
            }

            Ok(())
        },
        SrccTrustPolicy::GroupContaminationBound {
            maximum_corrupted_weight_per_group,
        } =>
        {
            if !maximum_corrupted_weight_per_group.is_finite()
                || *maximum_corrupted_weight_per_group < 0.0
                || *maximum_corrupted_weight_per_group >= 0.5
            {
                return Err(SrccTrustError::InvalidPolicy);
            }

            Ok(())
        },
        SrccTrustPolicy::CompositeAll(policies) =>
        {
            if nested || policies.is_empty()
            {
                return Err(SrccTrustError::InvalidPolicy);
            }

            for sub_policy in policies
            {
                validate_policy(sub_policy, true)?;
            }

            Ok(())
        },
    }
}

fn validate_model(
    model: &SrccTrustModel,
    views: &[&[SrccTransportSample]],
) -> Result<(), SrccTrustError> {
    validate_policy(&model.policy, false)?;

    let mut seen: Vec<(usize, usize)> = Vec::with_capacity(model.observations.len());

    for record in &model.observations
    {
        let in_bounds =
            record.view_index < views.len() && record.sample_index < views[record.view_index].len();

        if !in_bounds
        {
            return Err(SrccTrustError::ObservationOutOfBounds {
                view_index: record.view_index,
                sample_index: record.sample_index,
            });
        }

        let key = (record.view_index, record.sample_index);

        if seen.contains(&key)
        {
            return Err(SrccTrustError::DuplicateObservation {
                view_index: record.view_index,
                sample_index: record.sample_index,
            });
        }

        seen.push(key);

        if !record.prior_weight.is_finite() || record.prior_weight < 0.0
        {
            return Err(SrccTrustError::InvalidPriorWeight {
                view_index: record.view_index,
                sample_index: record.sample_index,
            });
        }

        for evidence in &record.evidence
        {
            let valid = match evidence.kind
            {
                SrccTrustEvidenceKind::TrustedAnchor =>
                {
                    evidence.score == 0.0 || evidence.score == 1.0
                },
                SrccTrustEvidenceKind::TemporalPrediction =>
                {
                    evidence.score.is_finite() && evidence.score >= 0.0
                },
                _ => evidence.score.is_finite() && (0.0..=1.0).contains(&evidence.score),
            };

            if !valid
            {
                return Err(SrccTrustError::InvalidEvidenceScore {
                    view_index: record.view_index,
                    sample_index: record.sample_index,
                });
            }
        }
    }

    Ok(())
}

fn is_anchor(record: &SrccObservationTrust) -> bool {
    record.evidence.iter().any(|evidence| {
        evidence.kind == SrccTrustEvidenceKind::TrustedAnchor && evidence.score == 1.0
    })
}

fn persistent_steps(record: &SrccObservationTrust, maximum_prediction_error: f64) -> usize {
    record
        .evidence
        .iter()
        .filter(|evidence| {
            evidence.kind == SrccTrustEvidenceKind::TemporalPrediction
                && evidence.score <= maximum_prediction_error
        })
        .count()
}

/// The effective weight one (non-composite) policy assigns to a record, and
/// whether the record counts as an anchor vote under that policy.
fn policy_weight(policy: &SrccTrustPolicy, record: &SrccObservationTrust) -> (f64, bool) {
    match policy
    {
        SrccTrustPolicy::Unweighted => (1.0, false),
        SrccTrustPolicy::TrustedAnchors { .. } =>
        {
            if is_anchor(record)
            {
                (record.prior_weight, true)
            }
            else
            {
                (0.0, false)
            }
        },
        SrccTrustPolicy::IndependentViews { .. }
        | SrccTrustPolicy::GroupContaminationBound { .. } => (record.prior_weight, false),
        SrccTrustPolicy::TemporalPersistence {
            minimum_consistent_steps,
            maximum_prediction_error,
        } =>
        {
            if persistent_steps(record, *maximum_prediction_error) >= *minimum_consistent_steps
            {
                (record.prior_weight, false)
            }
            else
            {
                (0.0, false)
            }
        },
        SrccTrustPolicy::CompositeAll(policies) =>
        {
            let mut weight = f64::INFINITY;
            let mut anchor = false;

            for sub_policy in policies
            {
                let (sub_weight, sub_anchor) = policy_weight(sub_policy, record);
                weight = weight.min(sub_weight);
                anchor |= sub_anchor;
            }

            (weight, anchor)
        },
    }
}

/// The margin bound `β` a (non-composite) policy demands per group, if any.
/// For `CompositeAll` the strictest (largest) bound applies.
fn margin_bound(policy: &SrccTrustPolicy) -> Option<f64> {
    match policy
    {
        SrccTrustPolicy::IndependentViews {
            maximum_corrupted_weight_per_view,
            ..
        } => Some(*maximum_corrupted_weight_per_view),
        SrccTrustPolicy::GroupContaminationBound {
            maximum_corrupted_weight_per_group,
        } => Some(*maximum_corrupted_weight_per_group),
        SrccTrustPolicy::CompositeAll(policies) => policies
            .iter()
            .filter_map(margin_bound)
            .fold(None, |acc, bound| {
                Some(acc.map_or(bound, |current: f64| current.max(bound)))
            }),
        _ => None,
    }
}

fn minimum_anchor_requirement(policy: &SrccTrustPolicy) -> Option<usize> {
    match policy
    {
        SrccTrustPolicy::TrustedAnchors {
            minimum_anchor_support,
        } => Some(*minimum_anchor_support),
        SrccTrustPolicy::CompositeAll(policies) =>
        {
            policies.iter().filter_map(minimum_anchor_requirement).max()
        },
        _ => None,
    }
}

fn policy_assumptions(policy: &SrccTrustPolicy) -> Vec<String> {
    match policy
    {
        SrccTrustPolicy::Unweighted => vec![
            "unweighted consensus: at most a minority of observations per exact-source group \
is corrupted (the historical assumption; majority corruption is silently misleading here)"
                .to_string(),
        ],
        SrccTrustPolicy::TrustedAnchors {
            minimum_anchor_support,
        } => vec![format!(
            "trusted anchors are incorruptible and every group keeps at least \
{minimum_anchor_support} of them"
        )],
        SrccTrustPolicy::IndependentViews {
            minimum_consistent_views,
            maximum_corrupted_weight_per_view,
        } => vec![format!(
            "at least {minimum_consistent_views} independent views, each with at most a \
{maximum_corrupted_weight_per_view} fraction of corrupted trusted weight per group"
        )],
        SrccTrustPolicy::TemporalPersistence {
            minimum_consistent_steps,
            maximum_prediction_error,
        } => vec![format!(
            "honest observations carry at least {minimum_consistent_steps} temporal predictions \
with error at most {maximum_prediction_error}"
        )],
        SrccTrustPolicy::GroupContaminationBound {
            maximum_corrupted_weight_per_group,
        } => vec![format!(
            "at most a {maximum_corrupted_weight_per_group} fraction of each group's trusted \
weight is corrupted"
        )],
        SrccTrustPolicy::CompositeAll(policies) =>
        {
            policies.iter().flat_map(policy_assumptions).collect()
        },
    }
}

// ---------------------------------------------------------------------------
// Trusted consensus
// ---------------------------------------------------------------------------

/// Weighted target medoid over a group, mirroring the historical fixed-order
/// scan; under unit weights `1.0 · x == x`, so the historical result is
/// reproduced bit for bit. Candidates are the targets of positive-weight
/// observations only (a fully distrusted target cannot be elected), scanned in
/// the group's canonical order.
fn weighted_target_medoid(
    group: &[TrustedSample],
    view_index: usize,
    source_group_index: usize,
) -> Result<Vector16, SrccTrustedFitError> {
    let mut best_target: Option<Vector16> = None;
    let mut best_score = f64::INFINITY;
    // Distinct targets tied at the best score (the elected one included).
    let mut tied_targets: Vec<Vector16> = Vec::new();

    for candidate in group.iter().filter(|entry| entry.weight > 0.0)
    {
        let score = group.iter().fold(0.0, |sum, entry| {
            sum + entry.weight * squared_distance(&candidate.sample.target, &entry.sample.target)
        });

        if !score.is_finite()
        {
            return Err(SrccTrustedFitError::Fit(
                SrccRobustFitError::NonFiniteTargetDistance {
                    view_index,
                    source_group_index,
                },
            ));
        }

        match best_target
        {
            None =>
            {
                best_target = Some(candidate.sample.target);
                best_score = score;
                tied_targets = vec![candidate.sample.target];
            },
            Some(_) => match score.total_cmp(&best_score)
            {
                Ordering::Less =>
                {
                    best_target = Some(candidate.sample.target);
                    best_score = score;
                    tied_targets = vec![candidate.sample.target];
                },
                Ordering::Equal =>
                {
                    if !tied_targets.contains(&candidate.sample.target)
                    {
                        tied_targets.push(candidate.sample.target);
                    }
                },
                Ordering::Greater =>
                {},
            },
        }
    }

    match best_target
    {
        Some(target) if tied_targets.len() == 1 => Ok(target),
        Some(_) => Err(SrccTrustedFitError::Trust(
            SrccTrustError::UnidentifiableContamination {
                competing_hypotheses: tied_targets.len(),
            },
        )),
        None => Err(SrccTrustedFitError::Trust(
            SrccTrustError::AllObservationsUntrusted {
                view_index,
                source_group_index,
            },
        )),
    }
}

/// Support of a target inside a group: summed weight of bit-identical targets.
fn target_support(group: &[TrustedSample], target: &Vector16) -> f64 {
    group.iter().fold(0.0, |sum, entry| {
        if entry.sample.target == *target
        {
            sum + entry.weight
        }
        else
        {
            sum
        }
    })
}

/// Largest support among targets distinct from `winner` (`0.0` if none).
fn runner_up_support(group: &[TrustedSample], winner: &Vector16) -> f64 {
    let mut best = 0.0f64;

    for entry in group
    {
        if entry.sample.target != *winner
        {
            let support = target_support(group, &entry.sample.target);
            best = best.max(support);
        }
    }

    best
}

/// Fits a robust SRCC projector under an explicit trust model.
///
/// Grouping is by exactly equal sources per view (the historical robust
/// grouping); the target consensus of every group is the trust-weighted
/// medoid, subject to the policy's anchor, persistence, support and
/// adversarial-margin requirements. Every acceptance is documented in the
/// returned [`SrccTrustCertificate`].
pub fn fit_trusted_robust_srcc_projector_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    trust: &SrccTrustModel,
    config: SrccConfig,
) -> Result<SrccTrustedFitResult, SrccTrustedFitError> {
    validate_model(trust, views)?;

    if let SrccTrustPolicy::IndependentViews {
        minimum_consistent_views,
        ..
    } = &trust.policy
    {
        if views.len() < *minimum_consistent_views
        {
            return Err(SrccTrustedFitError::Trust(
                SrccTrustError::InsufficientViews {
                    required: *minimum_consistent_views,
                    found: views.len(),
                },
            ));
        }
    }

    // Validate every original observation first (historical precedence).
    let _validated = learn_transport_views(views, config.energy_floor)
        .map_err(|error| SrccTrustedFitError::Fit(SrccRobustFitError::from(error)))?;

    // Resolve per-observation weights in caller addressing.
    let mut weights: Vec<Vec<(f64, bool)>> = views
        .iter()
        .map(|view| vec![(1.0, false); view.len()])
        .collect();

    let default_record = |view_index: usize, sample_index: usize| SrccObservationTrust {
        view_index,
        sample_index,
        prior_weight: 1.0,
        evidence: Vec::new(),
    };

    for (view_index, view) in views.iter().enumerate()
    {
        for (sample_index, weight_slot) in
            weights[view_index].iter_mut().enumerate().take(view.len())
        {
            let record = trust
                .observations
                .iter()
                .find(|candidate| {
                    candidate.view_index == view_index && candidate.sample_index == sample_index
                })
                .cloned()
                .unwrap_or_else(|| default_record(view_index, sample_index));

            *weight_slot = policy_weight(&trust.policy, &record);
        }
    }

    let gated_observation_count = weights
        .iter()
        .flatten()
        .filter(|(weight, _)| *weight == 0.0)
        .count();

    // Canonical per-view grouping, carrying weights through the stable sort.
    let bound = margin_bound(&trust.policy);
    let anchor_requirement = minimum_anchor_requirement(&trust.policy);

    let mut representative_storage: Vec<Vec<SrccTransportSample>> = Vec::with_capacity(views.len());
    let mut group_certificates: Vec<SrccGroupTrustCertificate> = Vec::new();
    let mut effective_trusted_weight = 0.0f64;

    for (view_index, view) in views.iter().enumerate()
    {
        let mut carried: Vec<TrustedSample> = view
            .iter()
            .enumerate()
            .map(|(sample_index, sample)| TrustedSample {
                sample: *sample,
                weight: weights[view_index][sample_index].0,
                is_anchor: weights[view_index][sample_index].1,
            })
            .collect();

        carried.sort_by(|left, right| compare_samples(&left.sample, &right.sample));

        let mut groups: Vec<Vec<TrustedSample>> = Vec::new();

        for entry in carried
        {
            match groups.last_mut()
            {
                Some(group) if group[0].sample.source == entry.sample.source =>
                {
                    group.push(entry);
                },
                _ => groups.push(vec![entry]),
            }
        }

        let mut representatives = Vec::with_capacity(groups.len());

        for (source_group_index, group) in groups.iter().enumerate()
        {
            let anchor_count = group.iter().filter(|entry| entry.is_anchor).count();

            if let Some(required) = anchor_requirement
            {
                if anchor_count < required
                {
                    return Err(SrccTrustedFitError::Trust(
                        SrccTrustError::InsufficientAnchorSupport {
                            view_index,
                            source_group_index,
                            anchors: anchor_count,
                            required,
                        },
                    ));
                }
            }

            let target =
                weighted_target_medoid(group, view_index, source_group_index).map_err(|error| {
                    remap_anchor_ambiguity(
                        error,
                        anchor_requirement.is_some(),
                        view_index,
                        source_group_index,
                    )
                })?;

            let total_weight: f64 = group.iter().map(|entry| entry.weight).sum();
            let winning_support = target_support(group, &target);
            let runner_up = runner_up_support(group, &target);

            if let Some(beta) = bound
            {
                let margin = winning_support - runner_up;

                if margin <= 2.0 * beta * total_weight
                {
                    return Err(SrccTrustedFitError::Trust(
                        SrccTrustError::UnidentifiableContamination {
                            competing_hypotheses: 2,
                        },
                    ));
                }
            }

            effective_trusted_weight += total_weight;

            group_certificates.push(SrccGroupTrustCertificate {
                view_index,
                source_group_index,
                total_weight,
                winning_support,
                runner_up_support: runner_up,
                anchor_count,
            });

            representatives.push(SrccTransportSample::new(group[0].sample.source, target));
        }

        representative_storage.push(representatives);
    }

    let representative_views: Vec<&[SrccTransportSample]> =
        representative_storage.iter().map(Vec::as_slice).collect();

    let transports = learn_transport_views(&representative_views, config.energy_floor)
        .map_err(|error| SrccTrustedFitError::Fit(SrccRobustFitError::from(error)))?;

    let projector = SrccProjector::build(seeds, &transports, config)
        .map_err(|error| SrccTrustedFitError::Fit(SrccRobustFitError::from(error)))?;

    Ok(SrccTrustedFitResult {
        fit: SrccFitResult {
            transports,
            projector,
        },
        certificate: SrccTrustCertificate {
            assumptions: policy_assumptions(&trust.policy),
            groups: group_certificates,
            effective_trusted_weight,
            gated_observation_count,
        },
    })
}

/// Under an anchor policy, a weighted tie between distinct targets means the
/// anchors themselves disagree — reported as `ConflictingAnchors` (the anchor
/// assumption is violated) instead of generic unidentifiability.
fn remap_anchor_ambiguity(
    error: SrccTrustedFitError,
    anchors_active: bool,
    view_index: usize,
    source_group_index: usize,
) -> SrccTrustedFitError {
    match (&error, anchors_active)
    {
        (SrccTrustedFitError::Trust(SrccTrustError::UnidentifiableContamination { .. }), true) =>
        {
            SrccTrustedFitError::Trust(SrccTrustError::ConflictingAnchors {
                view_index,
                source_group_index,
            })
        },
        _ => error,
    }
}

/// Leave-one-out stability of the trusted fit: every removal re-derives the
/// trust weights for the reduced views (records addressing the removed sample
/// are dropped; later records in the same view shift down) and refits the
/// complete trusted pipeline. Trust-dependent decisions are therefore
/// recomputed per variant — never frozen.
pub fn evaluate_trusted_robust_leave_one_out_stability(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    trust: &SrccTrustModel,
    config: SrccConfig,
) -> Result<SrccStabilityReport, SrccTrustedFitError> {
    let full = fit_trusted_robust_srcc_projector_from_views(seeds, views, trust, config)?;

    let full_dimension = full.fit.projector.rejected_dimension();
    let mut variants = Vec::new();

    for (view_index, view) in views.iter().enumerate()
    {
        if view.len() <= 1
        {
            continue;
        }

        for sample_index in 0..view.len()
        {
            let mut reduced_views: Vec<Vec<SrccTransportSample>> =
                views.iter().map(|samples| samples.to_vec()).collect();

            reduced_views[view_index].remove(sample_index);

            let reduced_references: Vec<&[SrccTransportSample]> =
                reduced_views.iter().map(Vec::as_slice).collect();

            let reduced_trust = SrccTrustModel {
                policy: trust.policy.clone(),
                observations: trust
                    .observations
                    .iter()
                    .filter(|record| {
                        !(record.view_index == view_index && record.sample_index == sample_index)
                    })
                    .map(|record| {
                        let mut shifted = record.clone();

                        if shifted.view_index == view_index && shifted.sample_index > sample_index
                        {
                            shifted.sample_index -= 1;
                        }

                        shifted
                    })
                    .collect(),
            };

            let reduced = fit_trusted_robust_srcc_projector_from_views(
                seeds,
                &reduced_references,
                &reduced_trust,
                config,
            )?;

            variants.push(SrccStabilityVariant {
                removed_view_index: view_index,
                removed_sample_index: sample_index,
                rejected_dimension: reduced.fit.projector.rejected_dimension(),
                frobenius_distance: trusted_projector_frobenius_distance(
                    &full.fit.projector,
                    &reduced.fit.projector,
                ),
            });
        }
    }

    if variants.is_empty()
    {
        return Err(SrccTrustedFitError::Trust(
            SrccTrustError::UnidentifiableContamination {
                competing_hypotheses: 0,
            },
        ));
    }

    let distance_sum = variants
        .iter()
        .map(|variant| variant.frobenius_distance)
        .sum::<f64>();

    let maximum_frobenius_distance = variants
        .iter()
        .map(|variant| variant.frobenius_distance)
        .fold(0.0, f64::max);

    let stable_dimension_count = variants
        .iter()
        .filter(|variant| variant.rejected_dimension == full_dimension)
        .count();

    Ok(SrccStabilityReport {
        full_projector: full.fit.projector,
        mean_frobenius_distance: distance_sum / variants.len() as f64,
        maximum_frobenius_distance,
        stable_dimension_count,
        variants,
    })
}

fn trusted_projector_frobenius_distance(left: &SrccProjector, right: &SrccProjector) -> f64 {
    let squared = left
        .transform()
        .iter()
        .flatten()
        .zip(right.transform().iter().flatten())
        .fold(0.0, |sum, (left_value, right_value)| {
            let difference = left_value - right_value;
            sum + difference * difference
        });

    squared.sqrt() / (SRCC_DIMENSION as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{apply_linear_map, basis_vector, fit_robust_srcc_projector_from_views};

    fn test_config() -> SrccConfig {
        SrccConfig {
            novelty_threshold: 1.0e-10,
            resonance_threshold: 0.999,
            minimum_support: 2,
            maximum_dimension: 2,
            maximum_rounds: 2,
            energy_floor: 1.0e-30,
        }
    }

    const PROVIDER: SrccTrustProviderId = SrccTrustProviderId(1);

    fn anchor_evidence() -> SrccTrustEvidence {
        SrccTrustEvidence {
            kind: SrccTrustEvidenceKind::TrustedAnchor,
            provider: PROVIDER,
            score: 1.0,
        }
    }

    fn prediction_evidence(error: f64) -> SrccTrustEvidence {
        SrccTrustEvidence {
            kind: SrccTrustEvidenceKind::TemporalPrediction,
            provider: PROVIDER,
            score: error,
        }
    }

    /// Two mirrored views over one exact source: `clean` repetitions of the
    /// clean target and `bad` repetitions of the contaminant target, clean
    /// samples FIRST in caller order (indices `0..clean` are clean).
    fn mirrored_views(
        clean: usize,
        bad: usize,
    ) -> (
        Vector16,
        Vector16,
        Vector16,
        Vec<SrccTransportSample>,
        Vec<SrccTransportSample>,
    ) {
        let source = basis_vector(1).unwrap();
        let clean_target = basis_vector(2).unwrap();
        let bad_target = basis_vector(8).unwrap();
        let negative_clean = clean_target.map(|value| -value);
        let negative_bad = bad_target.map(|value| -value);

        let mut positive = Vec::new();
        let mut negative = Vec::new();

        for _ in 0..clean
        {
            positive.push(SrccTransportSample::new(source, clean_target));
            negative.push(SrccTransportSample::new(source, negative_clean));
        }

        for _ in 0..bad
        {
            positive.push(SrccTransportSample::new(source, bad_target));
            negative.push(SrccTransportSample::new(source, negative_bad));
        }

        (source, clean_target, bad_target, positive, negative)
    }

    /// Trust records assigning `weight` to every sample of both views whose
    /// caller index is in `range`, with the given evidence.
    fn records_for(
        view_lengths: [usize; 2],
        selector: impl Fn(usize, usize) -> Option<(f64, Vec<SrccTrustEvidence>)>,
    ) -> Vec<SrccObservationTrust> {
        let mut records = Vec::new();

        for (view_index, &length) in view_lengths.iter().enumerate()
        {
            for sample_index in 0..length
            {
                if let Some((prior_weight, evidence)) = selector(view_index, sample_index)
                {
                    records.push(SrccObservationTrust {
                        view_index,
                        sample_index,
                        prior_weight,
                        evidence,
                    });
                }
            }
        }

        records
    }

    #[test]
    fn unweighted_trusted_fit_matches_historical_robust_fit() {
        // Clean 3 + contaminated 1 per view: the historical minority-robust
        // regime. The trusted Unweighted path must reproduce it bit for bit.
        let (source, _, _, positive, negative) = mirrored_views(3, 1);

        let views = [positive.as_slice(), negative.as_slice()];

        let historical =
            fit_robust_srcc_projector_from_views(&[source], &views, test_config()).unwrap();

        let trusted = fit_trusted_robust_srcc_projector_from_views(
            &[source],
            &views,
            &SrccTrustModel {
                policy: SrccTrustPolicy::Unweighted,
                observations: Vec::new(),
            },
            test_config(),
        )
        .unwrap();

        assert_eq!(historical, trusted.fit);
        assert_eq!(trusted.certificate.gated_observation_count, 0);
    }

    #[test]
    fn scenario_1_count_majority_accepted_under_per_view_weight_bound() {
        // 60% of the observations are corrupted BY COUNT, but they carry only
        // a small fraction of the trusted weight in each view; under the
        // declared per-view bound the margin holds and the clean structure is
        // accepted with a certificate.
        let (source, clean_target, _, positive, negative) = mirrored_views(2, 3);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::IndependentViews {
                minimum_consistent_views: 2,
                maximum_corrupted_weight_per_view: 0.3,
            },
            observations: records_for([5, 5], |_, sample_index| {
                let weight = if sample_index < 2 { 1.0 } else { 0.1 };
                Some((weight, Vec::new()))
            }),
        };

        let result =
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config())
                .unwrap();

        // The clean target wins: support 2.0 vs 0.3, margin 1.7 > 2*0.3*2.3.
        assert_eq!(
            apply_linear_map(&result.fit.transports[0], &source),
            clean_target
        );
        assert_eq!(result.certificate.groups.len(), 2);
        assert!(result.certificate.groups[0].winning_support == 2.0);
        assert!(result.certificate.groups[0].runner_up_support == 0.30000000000000004);
        assert!(!result.certificate.assumptions.is_empty());
    }

    #[test]
    fn scenario_2_view_concentrated_corruption_is_rejected() {
        // 80% corrupted weight concentrated in the views: the corrupt target
        // wins the raw vote, but the adversarial margin under beta = 0.3
        // cannot certify it, so the typed unidentifiability fires.
        let (source, _, _, positive, negative) = mirrored_views(1, 4);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::IndependentViews {
                minimum_consistent_views: 2,
                maximum_corrupted_weight_per_view: 0.3,
            },
            observations: Vec::new(),
        };

        assert_eq!(
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config()),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::UnidentifiableContamination {
                    competing_hypotheses: 2,
                },
            )),
        );
    }

    #[test]
    fn scenario_3_unweighted_majority_corruption_is_silently_wrong_but_bounded_policy_refuses() {
        // Without any trust assumption the historical consensus follows the
        // majority: the corrupt target is silently elected (the documented
        // historical assumption). The group-bound policy refuses instead.
        let (source, _, bad_target, positive, negative) = mirrored_views(2, 3);

        let views = [positive.as_slice(), negative.as_slice()];

        let unweighted = fit_trusted_robust_srcc_projector_from_views(
            &[source],
            &views,
            &SrccTrustModel {
                policy: SrccTrustPolicy::Unweighted,
                observations: Vec::new(),
            },
            test_config(),
        )
        .unwrap();

        assert_eq!(
            apply_linear_map(&unweighted.fit.transports[0], &source),
            bad_target
        );

        assert_eq!(
            fit_trusted_robust_srcc_projector_from_views(
                &[source],
                &views,
                &SrccTrustModel {
                    policy: SrccTrustPolicy::GroupContaminationBound {
                        maximum_corrupted_weight_per_group: 0.4,
                    },
                    observations: Vec::new(),
                },
                test_config(),
            ),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::UnidentifiableContamination {
                    competing_hypotheses: 2,
                },
            )),
        );
    }

    #[test]
    fn scenario_4_trusted_anchors_recover_under_count_majority_corruption() {
        let (source, clean_target, _, positive, negative) = mirrored_views(2, 3);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::TrustedAnchors {
                minimum_anchor_support: 2,
            },
            observations: records_for([5, 5], |_, sample_index| {
                if sample_index < 2
                {
                    Some((1.0, vec![anchor_evidence()]))
                }
                else
                {
                    None
                }
            }),
        };

        let result =
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config())
                .unwrap();

        assert_eq!(
            apply_linear_map(&result.fit.transports[0], &source),
            clean_target
        );
        assert_eq!(result.certificate.groups[0].anchor_count, 2);
        assert_eq!(result.certificate.gated_observation_count, 6);
    }

    #[test]
    fn scenario_5_conflicting_anchors_fail_explicitly() {
        // One anchor on the clean target, one anchor on the corrupt target:
        // the anchor assumption itself is violated and the failure is typed.
        let (source, _, _, positive, negative) = mirrored_views(1, 1);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::TrustedAnchors {
                minimum_anchor_support: 1,
            },
            observations: records_for([2, 2], |_, _| Some((1.0, vec![anchor_evidence()]))),
        };

        assert_eq!(
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config()),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::ConflictingAnchors {
                    view_index: 0,
                    source_group_index: 0,
                }
            )),
        );
    }

    #[test]
    fn scenario_6_temporal_persistence_defeats_burst_attack() {
        // The corrupt count-majority is a short burst with no persistent
        // prediction history; persistent clean observations keep their vote.
        let (source, clean_target, _, positive, negative) = mirrored_views(2, 3);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::TemporalPersistence {
                minimum_consistent_steps: 3,
                maximum_prediction_error: 0.1,
            },
            observations: records_for([5, 5], |_, sample_index| {
                if sample_index < 2
                {
                    Some((
                        1.0,
                        vec![
                            prediction_evidence(0.01),
                            prediction_evidence(0.02),
                            prediction_evidence(0.05),
                        ],
                    ))
                }
                else
                {
                    Some((1.0, vec![prediction_evidence(0.01)]))
                }
            }),
        };

        let result =
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config())
                .unwrap();

        assert_eq!(
            apply_linear_map(&result.fit.transports[0], &source),
            clean_target
        );
        assert_eq!(result.certificate.gated_observation_count, 6);
    }

    #[test]
    fn scenario_7_long_coherent_regime_is_unidentifiable_not_blindly_rejected() {
        // Two equally persistent, equally weighted regimes: no assumption can
        // pick one, and the typed unidentifiability names both hypotheses.
        let (source, _, _, positive, negative) = mirrored_views(3, 3);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::TemporalPersistence {
                minimum_consistent_steps: 2,
                maximum_prediction_error: 0.1,
            },
            observations: records_for([6, 6], |_, _| {
                Some((
                    1.0,
                    vec![prediction_evidence(0.01), prediction_evidence(0.02)],
                ))
            }),
        };

        assert_eq!(
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config()),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::UnidentifiableContamination {
                    competing_hypotheses: 2,
                },
            )),
        );
    }

    #[test]
    fn scenario_8_conflicting_policies_fail_loudly_under_composite() {
        // The anchor qualifies under TrustedAnchors but lacks the persistence
        // the second policy demands: the conjunction gates everything to zero
        // and fails loudly instead of averaging the conflict away.
        let (source, _, _, positive, negative) = mirrored_views(2, 1);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::CompositeAll(vec![
                SrccTrustPolicy::TrustedAnchors {
                    minimum_anchor_support: 1,
                },
                SrccTrustPolicy::TemporalPersistence {
                    minimum_consistent_steps: 2,
                    maximum_prediction_error: 0.1,
                },
            ]),
            observations: records_for([3, 3], |_, sample_index| {
                if sample_index == 0
                {
                    Some((1.0, vec![anchor_evidence()]))
                }
                else
                {
                    None
                }
            }),
        };

        assert_eq!(
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config()),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::AllObservationsUntrusted {
                    view_index: 0,
                    source_group_index: 0,
                },
            )),
        );
    }

    #[test]
    fn scenario_9_weighted_ties_are_deterministically_unidentifiable() {
        let (source, _, _, positive, negative) = mirrored_views(2, 2);

        let views = [positive.as_slice(), negative.as_slice()];

        let model = SrccTrustModel {
            policy: SrccTrustPolicy::Unweighted,
            observations: Vec::new(),
        };

        let first =
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config());

        let second =
            fit_trusted_robust_srcc_projector_from_views(&[source], &views, &model, test_config());

        assert_eq!(
            first,
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::UnidentifiableContamination {
                    competing_hypotheses: 2,
                },
            )),
        );
        assert_eq!(first, second);
    }

    #[test]
    fn scenario_10_trusted_loo_recomputes_trust_dependent_decisions() {
        // With exactly the minimum anchor support, removing an anchor must
        // surface as a typed leave-one-out failure (the trust decision is
        // recomputed, not frozen); with one spare anchor the report certifies.
        let (source, _, _, positive, negative) = mirrored_views(2, 1);

        let views = [positive.as_slice(), negative.as_slice()];

        let tight = SrccTrustModel {
            policy: SrccTrustPolicy::TrustedAnchors {
                minimum_anchor_support: 2,
            },
            observations: records_for([3, 3], |_, sample_index| {
                if sample_index < 2
                {
                    Some((1.0, vec![anchor_evidence()]))
                }
                else
                {
                    None
                }
            }),
        };

        assert!(matches!(
            evaluate_trusted_robust_leave_one_out_stability(
                &[source],
                &views,
                &tight,
                test_config(),
            ),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::InsufficientAnchorSupport { .. },
            )),
        ));

        let (source, _, _, positive, negative) = mirrored_views(3, 1);

        let views = [positive.as_slice(), negative.as_slice()];

        let spare = SrccTrustModel {
            policy: SrccTrustPolicy::TrustedAnchors {
                minimum_anchor_support: 2,
            },
            observations: records_for([4, 4], |_, sample_index| {
                if sample_index < 3
                {
                    Some((1.0, vec![anchor_evidence()]))
                }
                else
                {
                    None
                }
            }),
        };

        let report = evaluate_trusted_robust_leave_one_out_stability(
            &[source],
            &views,
            &spare,
            test_config(),
        )
        .unwrap();

        assert_eq!(report.removal_count(), 8);
        assert_eq!(report.stable_dimension_count, 8);
        assert_eq!(report.maximum_frobenius_distance, 0.0);
    }

    #[test]
    fn invalid_models_are_typed_errors() {
        let (source, _, _, positive, negative) = mirrored_views(2, 1);

        let views = [positive.as_slice(), negative.as_slice()];

        let out_of_bounds = SrccTrustModel {
            policy: SrccTrustPolicy::Unweighted,
            observations: vec![SrccObservationTrust {
                view_index: 5,
                sample_index: 0,
                prior_weight: 1.0,
                evidence: Vec::new(),
            }],
        };

        assert_eq!(
            fit_trusted_robust_srcc_projector_from_views(
                &[source],
                &views,
                &out_of_bounds,
                test_config(),
            ),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::ObservationOutOfBounds {
                    view_index: 5,
                    sample_index: 0,
                },
            )),
        );

        let duplicate = SrccTrustModel {
            policy: SrccTrustPolicy::Unweighted,
            observations: vec![
                SrccObservationTrust {
                    view_index: 0,
                    sample_index: 0,
                    prior_weight: 1.0,
                    evidence: Vec::new(),
                },
                SrccObservationTrust {
                    view_index: 0,
                    sample_index: 0,
                    prior_weight: 2.0,
                    evidence: Vec::new(),
                },
            ],
        };

        assert!(matches!(
            fit_trusted_robust_srcc_projector_from_views(
                &[source],
                &views,
                &duplicate,
                test_config(),
            ),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::DuplicateObservation { .. }
            )),
        ));

        let bad_weight = SrccTrustModel {
            policy: SrccTrustPolicy::Unweighted,
            observations: vec![SrccObservationTrust {
                view_index: 0,
                sample_index: 0,
                prior_weight: -1.0,
                evidence: Vec::new(),
            }],
        };

        assert!(matches!(
            fit_trusted_robust_srcc_projector_from_views(
                &[source],
                &views,
                &bad_weight,
                test_config(),
            ),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::InvalidPriorWeight { .. }
            )),
        ));

        let bad_score = SrccTrustModel {
            policy: SrccTrustPolicy::Unweighted,
            observations: vec![SrccObservationTrust {
                view_index: 0,
                sample_index: 0,
                prior_weight: 1.0,
                evidence: vec![SrccTrustEvidence {
                    kind: SrccTrustEvidenceKind::TrustedAnchor,
                    provider: PROVIDER,
                    score: 0.5,
                }],
            }],
        };

        assert!(matches!(
            fit_trusted_robust_srcc_projector_from_views(
                &[source],
                &views,
                &bad_score,
                test_config(),
            ),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::InvalidEvidenceScore { .. }
            )),
        ));

        for policy in [
            SrccTrustPolicy::TrustedAnchors {
                minimum_anchor_support: 0,
            },
            SrccTrustPolicy::IndependentViews {
                minimum_consistent_views: 0,
                maximum_corrupted_weight_per_view: 0.3,
            },
            SrccTrustPolicy::IndependentViews {
                minimum_consistent_views: 1,
                maximum_corrupted_weight_per_view: 0.5,
            },
            SrccTrustPolicy::GroupContaminationBound {
                maximum_corrupted_weight_per_group: -0.1,
            },
            SrccTrustPolicy::CompositeAll(Vec::new()),
            SrccTrustPolicy::CompositeAll(vec![SrccTrustPolicy::CompositeAll(vec![
                SrccTrustPolicy::Unweighted,
            ])]),
        ]
        {
            assert_eq!(
                fit_trusted_robust_srcc_projector_from_views(
                    &[source],
                    &views,
                    &SrccTrustModel {
                        policy,
                        observations: Vec::new(),
                    },
                    test_config(),
                ),
                Err(SrccTrustedFitError::Trust(SrccTrustError::InvalidPolicy)),
            );
        }
    }

    #[test]
    fn insufficient_views_are_rejected() {
        let (source, _, _, positive, _) = mirrored_views(2, 0);

        let views = [positive.as_slice()];

        assert_eq!(
            fit_trusted_robust_srcc_projector_from_views(
                &[source],
                &views,
                &SrccTrustModel {
                    policy: SrccTrustPolicy::IndependentViews {
                        minimum_consistent_views: 2,
                        maximum_corrupted_weight_per_view: 0.2,
                    },
                    observations: Vec::new(),
                },
                test_config(),
            ),
            Err(SrccTrustedFitError::Trust(
                SrccTrustError::InsufficientViews {
                    required: 2,
                    found: 1,
                }
            )),
        );
    }

    #[test]
    fn collect_trust_evidence_is_deterministic_and_ordered() {
        struct ConstantProvider;

        impl SrccTrustEvidenceProvider for ConstantProvider {
            fn provider_id(&self) -> SrccTrustProviderId {
                PROVIDER
            }

            fn evidence_for(
                &self,
                _view_index: usize,
                _sample_index: usize,
                _sample: &SrccTransportSample,
            ) -> Result<Vec<SrccTrustEvidence>, SrccTrustError> {
                Ok(vec![prediction_evidence(0.01)])
            }
        }

        let (_, _, _, positive, negative) = mirrored_views(1, 1);

        let views = [positive.as_slice(), negative.as_slice()];

        let provider = ConstantProvider;

        let first = collect_trust_evidence(&views, &[&provider]).unwrap();
        let second = collect_trust_evidence(&views, &[&provider]).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        assert_eq!(first[0].view_index, 0);
        assert_eq!(first[0].sample_index, 0);
        assert_eq!(first[3].view_index, 1);
        assert_eq!(first[3].sample_index, 1);
    }
}
