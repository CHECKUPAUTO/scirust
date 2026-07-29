//! **Claim audit**: checking a body of certificates against the crate's own
//! discipline.
//!
//! # What this checks, and what it cannot
//!
//! Everything here is about **contract compliance and internal coherence**. A
//! clean audit says a set of claims obeys the rules this crate states about
//! itself: their fingerprints match their content, none rests on an assumption
//! evidence has contradicted, none quotes a method without declaring what that
//! method requires, and no two of them answer the same question differently.
//!
//! It says nothing whatever about whether a claim is *true*. No procedure here
//! touches data. A perfectly compliant claim set can be uniformly wrong, and
//! the audit will pass it.
//!
//! # The check worth having
//!
//! [`ClaimFinding::EstimateOnUnbackedProvenance`] fires when a certificate
//! quotes a number while every assumption it declares is merely asserted or
//! unverified. That is not a rule violation — analysts assert things, and
//! sometimes they are right — but it is the shape of a claim that has nothing
//! behind it but confidence.
//!
//! The crate's own adversarial fixture is exactly that shape. The estimate it
//! certifies is wrong by tens of standard errors, and phases 5C.5, 5C.6 and
//! 5C.9 each caught it in turn by *measuring* something. This check catches the
//! same claim from its provenance metadata alone, before any test is run. It
//! is the cheapest signal in the program and the weakest: it flags a claim's
//! epistemic posture, never its correctness.
//!
//! # Silence is not compliance
//!
//! A method this audit has no requirement table for is reported
//! ([`ClaimFinding::UnrecognizedMethod`]) rather than passed over. An unknown
//! method that happens to declare nothing would otherwise be indistinguishable
//! from a compliant one, and "we had no rule to apply" must not read as "the
//! rule was satisfied".
//!
//! # What is enforced rather than audited
//!
//! There is no check here for an estimate attached to a non-`Identifiable`
//! status. That rule is enforced at *both* boundaries — by
//! [`crate::CausalCertificateBuilder::finalize`] on construction and by
//! `CausalCertificate`'s `Deserialize` on parsing — so a certificate violating
//! it cannot exist to be audited. Auditing for an impossible condition would
//! be dead weight dressed as rigour; tests assert the enforcement instead.

use crate::assumptions::{AssumptionBasis, AssumptionRegistry, CausalAssumption};
use crate::certificate::{CausalCertificate, IdentifiabilityStatus};
use crate::error::CausalError;
use std::collections::BTreeMap;

/// How seriously to take a finding.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum FindingSeverity {
    /// A rule of the crate's contract is broken.
    Violation,
    /// Not a rule violation, but a claim whose posture warrants a look.
    Warning,
}

/// One thing the audit found.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClaimFinding {
    /// The stored fingerprint does not match the content it commits to: the
    /// certificate was altered after issue, or produced by a build with a
    /// different canonical encoding.
    FingerprintMismatch { claim: String },
    /// The claim declares an assumption the registry records as contradicted.
    ContradictedAssumptionCited {
        claim: String,
        assumption: CausalAssumption,
    },
    /// Two claims answer the same query with different statuses or estimates.
    ConflictingClaims {
        query: String,
        first: String,
        second: String,
    },
    /// The claim's method is known to require an assumption it does not
    /// declare.
    UndeclaredAssumption {
        claim: String,
        method: String,
        missing: CausalAssumption,
    },
    /// The claim's method has no requirement table here, so nothing could be
    /// checked about what it should declare.
    UnrecognizedMethod { claim: String, method: String },
    /// The claim declares an assumption the registry has never heard of.
    UnregisteredAssumption {
        claim: String,
        assumption: CausalAssumption,
    },
    /// The claim quotes a number while every assumption it declares is merely
    /// asserted or unverified.
    EstimateOnUnbackedProvenance {
        claim: String,
        unbacked: Vec<CausalAssumption>,
    },
}

impl ClaimFinding {
    #[must_use]
    pub fn severity(&self) -> FindingSeverity {
        match self
        {
            Self::FingerprintMismatch { .. }
            | Self::ContradictedAssumptionCited { .. }
            | Self::ConflictingClaims { .. }
            | Self::UndeclaredAssumption { .. } => FindingSeverity::Violation,
            Self::UnrecognizedMethod { .. }
            | Self::UnregisteredAssumption { .. }
            | Self::EstimateOnUnbackedProvenance { .. } => FindingSeverity::Warning,
        }
    }

    /// The claim or query this finding is about, for stable ordering.
    fn subject(&self) -> &str {
        match self
        {
            Self::FingerprintMismatch { claim }
            | Self::ContradictedAssumptionCited { claim, .. }
            | Self::UndeclaredAssumption { claim, .. }
            | Self::UnrecognizedMethod { claim, .. }
            | Self::UnregisteredAssumption { claim, .. }
            | Self::EstimateOnUnbackedProvenance { claim, .. } => claim,
            Self::ConflictingClaims { query, .. } => query,
        }
    }
}

/// What a method is expected to declare.
///
/// Matched by **substring** against a certificate's method string, so a method
/// can be renamed in its details without silently escaping its requirements.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MethodRequirement {
    /// Substring identifying the method, e.g. `"backdoor"`.
    pub method_key: String,
    pub required: Vec<CausalAssumption>,
}

/// How to audit.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClaimAuditConfig {
    /// Method requirement table. [`ClaimAuditConfig::default`] carries the
    /// methods this crate itself issues; a caller with their own methods must
    /// add them, or every such claim is reported as unrecognized.
    pub method_requirements: Vec<MethodRequirement>,
}

impl Default for ClaimAuditConfig {
    fn default() -> Self {
        Self {
            method_requirements: vec![
                MethodRequirement {
                    method_key: "backdoor".to_string(),
                    required: vec![
                        CausalAssumption::Acyclicity,
                        CausalAssumption::CausalSufficiency,
                        CausalAssumption::CorrectFunctionalForm,
                        CausalAssumption::Positivity,
                    ],
                },
                MethodRequirement {
                    method_key: "structural counterfactual".to_string(),
                    required: vec![
                        CausalAssumption::Acyclicity,
                        CausalAssumption::CausalSufficiency,
                        CausalAssumption::CorrectFunctionalForm,
                    ],
                },
            ],
        }
    }
}

/// The audit's overall verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ClaimAuditVerdict {
    /// Nothing was found. The claims obey the crate's rules — which is not the
    /// same as being right about anything.
    Compliant,
    /// At least one rule of the contract is broken.
    Violations { count: usize },
    /// No violations, but at least one claim's posture warrants a look.
    WarningsOnly { count: usize },
}

/// The result of auditing a claim set.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClaimAudit {
    pub claims_examined: usize,
    /// Findings, violations first, then by subject — a total order, so the
    /// report does not depend on input order.
    pub findings: Vec<ClaimFinding>,
    pub verdict: ClaimAuditVerdict,
    pub certificate: CausalCertificate,
}

const SCOPE_NOTE: &str = "This audits contract compliance and internal coherence across a set of \
                          claims. It reads no data and evaluates no method: a fully compliant set \
                          can be uniformly wrong, and a Compliant verdict is a statement about \
                          the claims' form and provenance, never about their correctness.";

/// Whether a basis is anything more than the analyst's word.
fn is_backed(basis: &AssumptionBasis) -> bool {
    match basis
    {
        AssumptionBasis::GuaranteedByDesign { .. }
        | AssumptionBasis::TestedStatistically { .. }
        | AssumptionBasis::DomainKnowledge { .. } => true,
        AssumptionBasis::AssertedByAnalyst
        | AssumptionBasis::Unverified
        | AssumptionBasis::ContradictedByEvidence { .. } => false,
    }
}

/// Audits `claims` against `registry`.
///
/// See the module docs for the scope of what "audited" means here, and in
/// particular for what this deliberately does *not* check.
///
/// # Errors
///
/// [`CausalError::InvalidContract`] if a method requirement has an empty key,
/// or if the audit's own certificate cannot be built.
pub fn audit_claim_set(
    claims: &[CausalCertificate],
    registry: &AssumptionRegistry,
    config: &ClaimAuditConfig,
) -> Result<ClaimAudit, CausalError> {
    for requirement in &config.method_requirements
    {
        if requirement.method_key.trim().is_empty()
        {
            return Err(CausalError::InvalidContract {
                detail: "a method requirement must carry a non-empty method key",
            });
        }
    }

    let mut findings = Vec::new();

    for claim in claims
    {
        if !claim.verify_fingerprint()
        {
            findings.push(ClaimFinding::FingerprintMismatch {
                claim: claim.query().to_string(),
            });
        }

        for assumption in claim.assumptions_used()
        {
            match registry.get(assumption)
            {
                None => findings.push(ClaimFinding::UnregisteredAssumption {
                    claim: claim.query().to_string(),
                    assumption: assumption.clone(),
                }),
                Some(record) =>
                {
                    if matches!(record.basis, AssumptionBasis::ContradictedByEvidence { .. })
                    {
                        findings.push(ClaimFinding::ContradictedAssumptionCited {
                            claim: claim.query().to_string(),
                            assumption: assumption.clone(),
                        });
                    }
                },
            }
        }

        if let Some(method) = claim.method()
        {
            let matched: Vec<&MethodRequirement> = config
                .method_requirements
                .iter()
                .filter(|r| method.contains(&r.method_key))
                .collect();
            if matched.is_empty()
            {
                findings.push(ClaimFinding::UnrecognizedMethod {
                    claim: claim.query().to_string(),
                    method: method.to_string(),
                });
            }
            for requirement in matched
            {
                for required in &requirement.required
                {
                    if !claim.assumptions_used().contains(required)
                    {
                        findings.push(ClaimFinding::UndeclaredAssumption {
                            claim: claim.query().to_string(),
                            method: method.to_string(),
                            missing: required.clone(),
                        });
                    }
                }
            }
        }

        if claim.estimate().is_some() && !claim.assumptions_used().is_empty()
        {
            let unbacked: Vec<CausalAssumption> = claim
                .assumptions_used()
                .iter()
                .filter(|a| registry.get(a).is_none_or(|r| !is_backed(&r.basis)))
                .cloned()
                .collect();
            if unbacked.len() == claim.assumptions_used().len()
            {
                findings.push(ClaimFinding::EstimateOnUnbackedProvenance {
                    claim: claim.query().to_string(),
                    unbacked,
                });
            }
        }
    }

    // Two claims about the same question must agree.
    let mut by_query: BTreeMap<&str, Vec<&CausalCertificate>> = BTreeMap::new();
    for claim in claims
    {
        by_query.entry(claim.query()).or_default().push(claim);
    }
    for (query, group) in by_query
    {
        for (index, first) in group.iter().enumerate()
        {
            for second in group.iter().skip(index + 1)
            {
                let disagrees = first.status() != second.status()
                    || match (first.estimate(), second.estimate())
                    {
                        (Some(a), Some(b)) => a != b,
                        (None, None) => false,
                        _ => true,
                    };
                if disagrees
                {
                    findings.push(ClaimFinding::ConflictingClaims {
                        query: query.to_string(),
                        first: describe(first),
                        second: describe(second),
                    });
                }
            }
        }
    }

    findings.sort_by(|a, b| {
        a.severity()
            .cmp(&b.severity())
            .then(a.subject().cmp(b.subject()))
            .then_with(|| format!("{a:?}").cmp(&format!("{b:?}")))
    });

    let violations = findings
        .iter()
        .filter(|f| f.severity() == FindingSeverity::Violation)
        .count();
    let warnings = findings.len() - violations;
    let verdict = if violations > 0
    {
        ClaimAuditVerdict::Violations { count: violations }
    }
    else if warnings > 0
    {
        ClaimAuditVerdict::WarningsOnly { count: warnings }
    }
    else
    {
        ClaimAuditVerdict::Compliant
    };

    let status = match verdict
    {
        ClaimAuditVerdict::Violations { .. } => IdentifiabilityStatus::NotIdentifiable,
        ClaimAuditVerdict::WarningsOnly { .. } | ClaimAuditVerdict::Compliant =>
        {
            IdentifiabilityStatus::Identifiable
        },
    };

    let mut builder = CausalCertificate::builder(
        format!(
            "do {} claim(s) obey this crate's own contract?",
            claims.len()
        ),
        status,
        vec![],
        format!(
            "{violations} violation(s) and {warnings} warning(s) across {} claim(s), {} \
                 registered assumption(s)",
            claims.len(),
            registry.len()
        ),
    )
    .with_sensitivity(SCOPE_NOTE);
    for finding in findings.iter().take(8)
    {
        builder = builder.with_unresolved_alternative(format!("{finding:?}"));
    }

    Ok(ClaimAudit {
        claims_examined: claims.len(),
        findings,
        verdict,
        certificate: builder.finalize()?,
    })
}

fn describe(claim: &CausalCertificate) -> String {
    format!(
        "{:?}/{}",
        claim.status(),
        claim
            .estimate()
            .map_or("none".to_string(), |v| format!("{v:.6e}"))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backed_registry() -> AssumptionRegistry {
        let mut registry = AssumptionRegistry::new();
        for assumption in [
            CausalAssumption::Acyclicity,
            CausalAssumption::CausalSufficiency,
            CausalAssumption::CorrectFunctionalForm,
            CausalAssumption::Positivity,
            CausalAssumption::AdequateSampleSize,
        ]
        {
            registry
                .assert(
                    assumption,
                    AssumptionBasis::GuaranteedByDesign {
                        mechanism: "randomized assignment".to_string(),
                    },
                    None,
                )
                .unwrap();
        }
        registry
    }

    fn compliant_claim() -> CausalCertificate {
        CausalCertificate::builder(
            "effect of X on Y",
            IdentifiabilityStatus::Identifiable,
            vec![
                CausalAssumption::Acyclicity,
                CausalAssumption::CausalSufficiency,
                CausalAssumption::CorrectFunctionalForm,
                CausalAssumption::Positivity,
            ],
            "evidence",
        )
        .with_estimate("linear backdoor adjustment", 0.7, 0.01)
        .finalize()
        .unwrap()
    }

    fn audit(claims: &[CausalCertificate], registry: &AssumptionRegistry) -> ClaimAudit {
        audit_claim_set(claims, registry, &ClaimAuditConfig::default()).unwrap()
    }

    #[test]
    fn a_well_formed_claim_set_is_compliant() {
        let result = audit(&[compliant_claim()], &backed_registry());
        assert_eq!(result.verdict, ClaimAuditVerdict::Compliant);
        assert!(result.findings.is_empty());
        assert_eq!(result.claims_examined, 1);
    }

    #[test]
    fn compliant_says_nothing_about_correctness() {
        let result = audit(&[compliant_claim()], &backed_registry());
        assert!(
            result
                .certificate
                .sensitivity_note()
                .unwrap()
                .contains("can be uniformly wrong")
        );
    }

    #[test]
    fn a_tampered_certificate_is_caught_by_its_own_fingerprint() {
        // Round-trip through JSON with the estimate edited. Deserialization
        // permits it -- the coherence rule is untouched -- so only the
        // fingerprint check can see it.
        let json = serde_json::to_string(&compliant_claim()).unwrap();
        let tampered = json.replace(r#""estimate":0.7"#, r#""estimate":99.0"#);
        let claim: CausalCertificate = serde_json::from_str(&tampered).unwrap();
        assert_eq!(claim.estimate(), Some(99.0));
        assert!(!claim.verify_fingerprint());

        let result = audit(&[claim], &backed_registry());
        assert!(matches!(
            result.verdict,
            ClaimAuditVerdict::Violations { .. }
        ));
        assert!(
            result
                .findings
                .iter()
                .any(|f| matches!(f, ClaimFinding::FingerprintMismatch { .. }))
        );
    }

    #[test]
    fn an_undeclared_requirement_of_a_known_method_is_a_violation() {
        let claim = CausalCertificate::builder(
            "effect of X on Y",
            IdentifiabilityStatus::Identifiable,
            vec![CausalAssumption::Acyclicity], // positivity, sufficiency, form all missing
            "evidence",
        )
        .with_estimate("linear backdoor adjustment", 0.7, 0.01)
        .finalize()
        .unwrap();
        let result = audit(&[claim], &backed_registry());
        let missing: Vec<_> = result
            .findings
            .iter()
            .filter_map(|f| match f
            {
                ClaimFinding::UndeclaredAssumption { missing, .. } => Some(missing.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(missing.len(), 3);
        assert!(missing.contains(&CausalAssumption::CausalSufficiency));
    }

    #[test]
    fn an_unknown_method_is_reported_rather_than_passed_over() {
        let claim = CausalCertificate::builder(
            "q",
            IdentifiabilityStatus::Identifiable,
            vec![CausalAssumption::Acyclicity],
            "e",
        )
        .with_estimate("some bespoke procedure", 1.0, 0.1)
        .finalize()
        .unwrap();
        let result = audit(&[claim], &backed_registry());
        assert!(
            result
                .findings
                .iter()
                .any(|f| matches!(f, ClaimFinding::UnrecognizedMethod { .. })),
            "silence must not read as compliance: {:?}",
            result.findings
        );
    }

    #[test]
    fn an_estimate_resting_only_on_assertion_is_flagged() {
        let mut registry = AssumptionRegistry::new();
        for assumption in [
            CausalAssumption::Acyclicity,
            CausalAssumption::CausalSufficiency,
            CausalAssumption::CorrectFunctionalForm,
            CausalAssumption::Positivity,
        ]
        {
            registry
                .assert(assumption, AssumptionBasis::AssertedByAnalyst, None)
                .unwrap();
        }
        let result = audit(&[compliant_claim()], &registry);
        assert!(
            result
                .findings
                .iter()
                .any(|f| matches!(f, ClaimFinding::EstimateOnUnbackedProvenance { .. }))
        );
        // A warning, not a violation: asserting is allowed, it just leaves
        // nothing behind the number.
        assert!(matches!(
            result.verdict,
            ClaimAuditVerdict::WarningsOnly { .. }
        ));
    }

    #[test]
    fn one_backed_assumption_is_enough_to_clear_the_provenance_check() {
        let mut registry = AssumptionRegistry::new();
        registry
            .assert(
                CausalAssumption::Acyclicity,
                AssumptionBasis::DomainKnowledge {
                    citation: "process physics".to_string(),
                },
                None,
            )
            .unwrap();
        for assumption in [
            CausalAssumption::CausalSufficiency,
            CausalAssumption::CorrectFunctionalForm,
            CausalAssumption::Positivity,
        ]
        {
            registry
                .assert(assumption, AssumptionBasis::AssertedByAnalyst, None)
                .unwrap();
        }
        let result = audit(&[compliant_claim()], &registry);
        assert!(
            !result
                .findings
                .iter()
                .any(|f| matches!(f, ClaimFinding::EstimateOnUnbackedProvenance { .. })),
            "the check is for claims resting on nothing at all, not for any weak link"
        );
    }

    #[test]
    fn a_contradicted_assumption_still_cited_is_a_violation() {
        let mut registry = backed_registry();
        registry.overwrite(
            CausalAssumption::CausalSufficiency,
            AssumptionBasis::ContradictedByEvidence {
                source: "a follow-up measurement".to_string(),
                jointly_with: vec![],
            },
            None,
        );
        let result = audit(&[compliant_claim()], &registry);
        assert!(
            result
                .findings
                .iter()
                .any(|f| matches!(f, ClaimFinding::ContradictedAssumptionCited { .. }))
        );
        assert!(matches!(
            result.verdict,
            ClaimAuditVerdict::Violations { .. }
        ));
    }

    #[test]
    fn two_claims_disagreeing_about_one_question_is_a_violation() {
        let other = CausalCertificate::builder(
            "effect of X on Y", // same query
            IdentifiabilityStatus::Identifiable,
            vec![
                CausalAssumption::Acyclicity,
                CausalAssumption::CausalSufficiency,
                CausalAssumption::CorrectFunctionalForm,
                CausalAssumption::Positivity,
            ],
            "evidence",
        )
        .with_estimate("linear backdoor adjustment", 1.4, 0.01) // different number
        .finalize()
        .unwrap();
        let result = audit(&[compliant_claim(), other], &backed_registry());
        assert!(
            result
                .findings
                .iter()
                .any(|f| matches!(f, ClaimFinding::ConflictingClaims { .. }))
        );
    }

    #[test]
    fn an_unregistered_assumption_is_a_warning_not_silence() {
        let result = audit(&[compliant_claim()], &AssumptionRegistry::new());
        assert_eq!(
            result
                .findings
                .iter()
                .filter(|f| matches!(f, ClaimFinding::UnregisteredAssumption { .. }))
                .count(),
            4
        );
    }

    #[test]
    fn findings_are_ordered_violations_first_and_input_order_does_not_matter() {
        let mut registry = backed_registry();
        registry.overwrite(
            CausalAssumption::CausalSufficiency,
            AssumptionBasis::ContradictedByEvidence {
                source: "s".to_string(),
                jointly_with: vec![],
            },
            None,
        );
        let bespoke = CausalCertificate::builder(
            "another question",
            IdentifiabilityStatus::Identifiable,
            vec![CausalAssumption::Acyclicity],
            "e",
        )
        .with_estimate("bespoke", 1.0, 0.1)
        .finalize()
        .unwrap();

        let forward = audit(&[compliant_claim(), bespoke.clone()], &registry);
        let backward = audit(&[bespoke, compliant_claim()], &registry);
        assert_eq!(forward.findings, backward.findings);

        let severities: Vec<_> = forward.findings.iter().map(|f| f.severity()).collect();
        let mut sorted = severities.clone();
        sorted.sort();
        assert_eq!(severities, sorted, "violations must come first");
    }

    #[test]
    fn an_empty_claim_set_is_compliant_and_says_how_many_it_saw() {
        let result = audit(&[], &backed_registry());
        assert_eq!(result.verdict, ClaimAuditVerdict::Compliant);
        assert_eq!(result.claims_examined, 0);
    }

    #[test]
    fn a_malformed_requirement_table_is_a_typed_error() {
        assert!(matches!(
            audit_claim_set(
                &[],
                &backed_registry(),
                &ClaimAuditConfig {
                    method_requirements: vec![MethodRequirement {
                        method_key: "  ".to_string(),
                        required: vec![],
                    }],
                }
            ),
            Err(CausalError::InvalidContract { .. })
        ));
    }

    #[test]
    fn results_are_deterministic_and_json_round_trip() {
        let registry = backed_registry();
        let claims = [compliant_claim()];
        let first = audit(&claims, &registry);
        let second = audit(&claims, &registry);
        assert_eq!(first, second);
        assert_eq!(
            first.certificate.fingerprint(),
            second.certificate.fingerprint()
        );
        let encoded = serde_json::to_string(&first).unwrap();
        assert_eq!(serde_json::from_str::<ClaimAudit>(&encoded).unwrap(), first);
    }
}
