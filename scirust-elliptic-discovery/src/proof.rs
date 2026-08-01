//! Exact proof attempts and machine-verifiable certificates.

use scirust_modalg::bigint::BigInt;
use scirust_modalg::poly::Poly;

use crate::{
    CandidateEvaluation, CatalogEntry, ClassificationStatus, PointExpression, Relation, ToyPrime,
};

/// Exact certificate for one polynomial identity over a toy prime field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialIdentityCertificate
{
    prime: ToyPrime,
    left_coefficients: Vec<u64>,
    right_coefficients: Vec<u64>,
}

impl PolynomialIdentityCertificate
{
    /// Creates a certificate only if the normalized polynomial difference is zero.
    pub fn new(
        prime: ToyPrime,
        left_coefficients: Vec<u64>,
        right_coefficients: Vec<u64>,
    ) -> Option<Self>
    {
        let certificate = Self {
            prime,
            left_coefficients,
            right_coefficients,
        };
        certificate.verify().then_some(certificate)
    }

    /// Replays the exact identity with the existing polynomial abstraction.
    pub fn verify(&self) -> bool
    {
        let modulus = self.prime.value();
        let left = Poly::from_coeffs(modulus, &self.left_coefficients);
        let right = Poly::from_coeffs(modulus, &self.right_coefficients);
        left.sub(&right).is_zero()
    }

    pub const fn prime(&self) -> ToyPrime
    {
        self.prime
    }

    pub fn left_coefficients(&self) -> &[u64]
    {
        &self.left_coefficients
    }

    pub fn right_coefficients(&self) -> &[u64]
    {
        &self.right_coefficients
    }
}

/// Exact evidence which can be replayed independently.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProofCertificate
{
    /// Both point expressions normalize to the same integer multiple of the input.
    GroupModuleIdentity { coefficient: String },
    /// The expression always normalizes to the group identity.
    GroupIdentity { coefficient: String },
    /// A finite-field polynomial identity.
    PolynomialIdentity(PolynomialIdentityCertificate),
}

/// Outcome of an exact proof attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Justification
{
    Catalog(CatalogEntry),
    Proved(ProofCertificate),
    NoCertificate { reason: &'static str },
    NotEligible { reason: &'static str },
}

/// Attempts a proof only after falsification and coverage gates have passed.
pub fn attempt_justification(candidate: &CandidateEvaluation) -> Justification
{
    if candidate.counterexample().is_some()
    {
        return Justification::NotEligible {
            reason: "candidate has a counterexample",
        };
    }
    if candidate
        .gates()
        .iter()
        .any(|gate| gate.state() != crate::GateState::Passed)
    {
        return Justification::NotEligible {
            reason: "candidate has not passed every required dataset gate",
        };
    }
    if matches!(
        candidate.classification().status(),
        ClassificationStatus::Known | ClassificationStatus::RepresentationArtifact
    )
    {
        return candidate
            .classification()
            .catalog()
            .map(Justification::Catalog)
            .unwrap_or(Justification::NoCertificate {
                reason: "known classification has no catalog metadata",
            });
    }
    match candidate.relation()
    {
        Relation::PointEqual(left, right) =>
        {
            let left_coefficient = point_coefficient(left);
            let right_coefficient = point_coefficient(right);
            if left_coefficient == right_coefficient
            {
                Justification::Proved(ProofCertificate::GroupModuleIdentity {
                    coefficient: left_coefficient.to_decimal(),
                })
            }
            else
            {
                Justification::NoCertificate {
                    reason: "point expressions normalize to different integer coefficients",
                }
            }
        },
        Relation::IsInfinity(point) =>
        {
            let coefficient = point_coefficient(point);
            if coefficient == BigInt::from_i128(0)
            {
                Justification::Proved(ProofCertificate::GroupIdentity {
                    coefficient: coefficient.to_decimal(),
                })
            }
            else
            {
                Justification::NoCertificate {
                    reason: "point expression does not normalize to zero",
                }
            }
        },
        Relation::CurveAEquals(_) | Relation::CurveJEquals(_) => Justification::NoCertificate {
            reason: "no universal one-variable polynomial certificate applies",
        },
    }
}

fn point_coefficient(expression: &PointExpression) -> BigInt
{
    match expression
    {
        PointExpression::Input => BigInt::from_i128(1),
        PointExpression::Identity => BigInt::from_i128(0),
        PointExpression::Negate(point) => point_coefficient(point).neg(),
        PointExpression::Double(point) =>
        {
            point_coefficient(point).mul(&BigInt::from_i128(2))
        },
        PointExpression::ScalarMultiply { scalar, point } =>
        {
            point_coefficient(point).mul(&BigInt::from_i128(i128::from(*scalar)))
        },
        PointExpression::Add(left, right) =>
        {
            point_coefficient(left).add(&point_coefficient(right))
        },
    }
}

/// Proves (zeta*x)^3 + b = x^3 + b exactly when zeta^3 = 1.
pub fn prove_j_zero_identity(
    prime: ToyPrime,
    zeta: u64,
    b: u64,
) -> Option<PolynomialIdentityCertificate>
{
    let modulus = prime.value();
    let zeta_cubed = crate::Fp::new(prime, zeta).pow(3).value();
    PolynomialIdentityCertificate::new(
        prime,
        vec![b % modulus, 0, 0, zeta_cubed],
        vec![b % modulus, 0, 0, 1],
    )
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::{
        CorpusKind, ExperimentManifest, LocalResearchCase, ResearchCorpora, SearchPlan,
        evaluate_candidate,
    };

    fn corpora() -> (SearchPlan, ResearchCorpora)
    {
        let plan = SearchPlan::new(9, 2, 3, 1, 1_000_000, 1).expect("bounded plan");
        let corpora = ResearchCorpora::generate(plan);
        (plan, corpora)
    }

    #[test]
    fn syntactic_group_identity_receives_exact_certificate()
    {
        let (plan, corpora) = corpora();
        let input = PointExpression::Input;
        let relation = Relation::PointEqual(
            PointExpression::Add(
                Box::new(input.clone()),
                Box::new(PointExpression::Identity),
            ),
            input,
        );
        let candidate = evaluate_candidate(
            relation,
            &corpora,
            plan.tuple_budget_per_candidate(),
        );
        assert!(matches!(
            attempt_justification(&candidate),
            Justification::Proved(ProofCertificate::GroupModuleIdentity { .. })
        ));
    }

    #[test]
    fn j_zero_polynomial_certificate_replays()
    {
        let prime = ToyPrime::new(13).expect("prime");
        let certificate = prove_j_zero_identity(prime, 3, 2).expect("3 cubed is one modulo 13");
        assert!(certificate.verify());
        assert!(prove_j_zero_identity(prime, 2, 2).is_none());
    }

    #[test]
    fn local_scope_type_remains_required()
    {
        let case = LocalResearchCase::new(1, CorpusKind::IndependentHoldout, 1, 1)
            .expect("local case");
        let manifest = ExperimentManifest::new(case);
        assert_eq!(manifest.research_case(), case);
    }
}
