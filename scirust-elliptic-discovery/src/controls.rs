//! Mandatory positive and negative controls for the catalog and falsifier.

use crate::{
    Classification, ClassificationStatus, Corpus, Counterexample, Fp, RelationSignature, classify,
    first_point_counterexample,
};

/// Built-in controls required by the research contract.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ControlId {
    TrueNegation,
    FalseNegationKeepsY,
    FalseDoublingSign,
    JZeroClaimedUniversal,
    EncodingSignClaimedNovel,
    OverfitAZero,
}

/// Exact result of one mandatory control.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlResult {
    id: ControlId,
    classification: Classification,
    counterexample: Option<Counterexample>,
}

impl ControlResult {
    pub const fn id(&self) -> ControlId {
        self.id
    }

    pub const fn status(&self) -> ClassificationStatus {
        self.classification.status()
    }

    pub const fn classification(&self) -> Classification {
        self.classification
    }

    pub const fn counterexample(&self) -> Option<&Counterexample> {
        self.counterexample.as_ref()
    }
}

/// Executes one control in canonical corpus order.
pub fn run_control(corpus: &Corpus, id: ControlId) -> ControlResult {
    let (signature, counterexample) = match id
    {
        ControlId::TrueNegation =>
        {
            let counterexample =
                first_point_counterexample(corpus, "control.true-negation", |curve, point| {
                    let Ok(negative) = curve.negate(point)
                    else
                    {
                        return false;
                    };
                    let Ok(double_negative) = curve.negate(negative)
                    else
                    {
                        return false;
                    };
                    let Ok(sum) = curve.add(point, negative)
                    else
                    {
                        return false;
                    };
                    double_negative == point && sum == curve.identity()
                });
            (RelationSignature::NegationInvolution, counterexample)
        },
        ControlId::FalseNegationKeepsY =>
        {
            let counterexample = first_point_counterexample(
                corpus,
                "control.false-negation-keeps-y",
                |curve, point| {
                    if point.is_infinity()
                    {
                        return true;
                    }
                    let Ok(negative) = curve.negate(point)
                    else
                    {
                        return false;
                    };
                    negative.affine_coordinates() == point.affine_coordinates()
                },
            );
            (RelationSignature::NegationInvolution, counterexample)
        },
        ControlId::FalseDoublingSign =>
        {
            let counterexample = first_point_counterexample(
                corpus,
                "control.false-doubling-sign",
                |curve, point| {
                    let Ok(double) = curve.scalar_mul(point, 2)
                    else
                    {
                        return false;
                    };
                    let Ok(negative_double) = curve.negate(double)
                    else
                    {
                        return false;
                    };
                    double == negative_double
                },
            );
            (RelationSignature::ScalarComposition, counterexample)
        },
        ControlId::JZeroClaimedUniversal =>
        {
            let counterexample = first_point_counterexample(
                corpus,
                "control.j-zero-claimed-universal",
                |curve, point| {
                    // 2^3 = 1 mod 7, while 2 is not one. Restricting this
                    // control to F_7 distinguishes the j=0 condition from an
                    // invalid scale constant in another field.
                    if curve.prime().value() != 7
                    {
                        return true;
                    }
                    let Some((x, y)) = point.affine_coordinates()
                    else
                    {
                        return true;
                    };
                    let scaled_x = Fp::new(curve.prime(), x)
                        .checked_mul(Fp::new(curve.prime(), 2))
                        .expect("values use the same prime")
                        .value();
                    curve.point_from_local_residues(scaled_x, y).is_ok()
                },
            );
            (RelationSignature::JZeroXScale { zeta: 2 }, counterexample)
        },
        ControlId::EncodingSignClaimedNovel =>
        {
            let counterexample = first_point_counterexample(
                corpus,
                "control.encoding-sign-claimed-novel",
                |curve, point| {
                    let Ok(negative) = curve.negate(point)
                    else
                    {
                        return false;
                    };
                    match (point.affine_coordinates(), negative.affine_coordinates())
                    {
                        (None, None) => true,
                        (Some((x, y)), Some((negative_x, negative_y))) =>
                        {
                            x == negative_x && Fp::new(curve.prime(), y).neg().value() == negative_y
                        },
                        _ => false,
                    }
                },
            );
            (RelationSignature::EncodingYSign, counterexample)
        },
        ControlId::OverfitAZero =>
        {
            let counterexample =
                first_point_counterexample(corpus, "control.overfit-a-zero", |curve, _| {
                    curve.a() == 0
                });
            (RelationSignature::Unrecognized, counterexample)
        },
    };
    ControlResult {
        id,
        classification: classify(signature, counterexample.is_some()),
        counterexample,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CorpusKind, ExperimentManifest, LocalResearchCase};

    fn exhaustive() -> Corpus {
        Corpus::generate(ExperimentManifest::new(
            LocalResearchCase::new(0, CorpusKind::ExhaustiveSmall, 1, u64::MAX)
                .expect("valid exhaustive case"),
        ))
    }

    #[test]
    fn all_mandatory_controls_are_classified_conservatively() {
        let corpus = exhaustive();
        assert_eq!(
            run_control(&corpus, ControlId::TrueNegation).status(),
            ClassificationStatus::Known
        );
        assert_eq!(
            run_control(&corpus, ControlId::FalseNegationKeepsY).status(),
            ClassificationStatus::Refuted
        );
        assert_eq!(
            run_control(&corpus, ControlId::FalseDoublingSign).status(),
            ClassificationStatus::Refuted
        );
        assert_eq!(
            run_control(&corpus, ControlId::JZeroClaimedUniversal).status(),
            ClassificationStatus::Refuted
        );
        assert_eq!(
            run_control(&corpus, ControlId::EncodingSignClaimedNovel).status(),
            ClassificationStatus::RepresentationArtifact
        );
        assert_eq!(
            run_control(&corpus, ControlId::OverfitAZero).status(),
            ClassificationStatus::Refuted
        );
    }

    #[test]
    fn j_zero_control_uses_a_valid_cube_root_before_refuting_non_j_zero_curves() {
        let corpus = exhaustive();
        let result = run_control(&corpus, ControlId::JZeroClaimedUniversal);
        let counterexample = result
            .counterexample()
            .expect("a universal j=0 claim must be refuted");
        assert_eq!(counterexample.curve_key().0, 7);
        assert_ne!(counterexample.curve_key().1, 0);
        let prime = crate::ToyPrime::new(7).expect("7 is a toy prime");
        assert_eq!(Fp::new(prime, 2).pow(3).value(), 1);
        assert_ne!(Fp::new(prime, 2).value(), 1);
    }
}
