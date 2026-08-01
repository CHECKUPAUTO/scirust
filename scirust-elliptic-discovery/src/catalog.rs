//! Executable catalog signatures for known properties and representation artifacts.

/// Families which must never be presented as new discoveries.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CatalogFamily
{
    NegationAndIdentity,
    GroupLinearity,
    JZeroAutomorphism,
    CubeRootsOfUnity,
    J1728Automorphism,
    GlvEndomorphism,
    CoordinateChange,
    EncodingSymmetry,
    TwistAndJClass,
}

/// Typed structural signature produced before classification.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RelationSignature
{
    NegationInvolution,
    AdditiveInverse,
    ScalarComposition,
    JZeroXScale { zeta: u64 },
    CubeRootOfUnity { zeta: u64 },
    J1728Automorphism,
    GlvEigenRelation { lambda: u64 },
    CoordinateScale { factor: u64 },
    EncodingYSign,
    EqualJInvariant,
    Unrecognized,
}

/// Immutable catalog metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogEntry
{
    pub id: &'static str,
    pub family: CatalogFamily,
    pub conditional: bool,
    pub representation_artifact: bool,
    pub reference: &'static str,
}

const SILVERMAN: &str = "Silverman, The Arithmetic of Elliptic Curves";
const GLV: &str = "Gallant-Lambert-Vanstone, CRYPTO 2001";
const SEC1: &str = "SEC 1 v2, point representation";

/// Looks up an exact structural signature in the built-in known-property catalog.
pub const fn catalog_entry(signature: RelationSignature) -> Option<CatalogEntry>
{
    let entry = match signature
    {
        RelationSignature::NegationInvolution | RelationSignature::AdditiveInverse => CatalogEntry {
            id: "group.negation",
            family: CatalogFamily::NegationAndIdentity,
            conditional: false,
            representation_artifact: false,
            reference: SILVERMAN,
        },
        RelationSignature::ScalarComposition => CatalogEntry {
            id: "group.scalar-composition",
            family: CatalogFamily::GroupLinearity,
            conditional: false,
            representation_artifact: false,
            reference: SILVERMAN,
        },
        RelationSignature::JZeroXScale { .. } => CatalogEntry {
            id: "automorphism.j-zero",
            family: CatalogFamily::JZeroAutomorphism,
            conditional: true,
            representation_artifact: false,
            reference: SILVERMAN,
        },
        RelationSignature::CubeRootOfUnity { .. } => CatalogEntry {
            id: "field.cube-root-unity",
            family: CatalogFamily::CubeRootsOfUnity,
            conditional: true,
            representation_artifact: false,
            reference: SILVERMAN,
        },
        RelationSignature::J1728Automorphism => CatalogEntry {
            id: "automorphism.j-1728",
            family: CatalogFamily::J1728Automorphism,
            conditional: true,
            representation_artifact: false,
            reference: SILVERMAN,
        },
        RelationSignature::GlvEigenRelation { .. } => CatalogEntry {
            id: "endomorphism.glv",
            family: CatalogFamily::GlvEndomorphism,
            conditional: true,
            representation_artifact: false,
            reference: GLV,
        },
        RelationSignature::CoordinateScale { .. } => CatalogEntry {
            id: "representation.coordinate-scale",
            family: CatalogFamily::CoordinateChange,
            conditional: true,
            representation_artifact: true,
            reference: SILVERMAN,
        },
        RelationSignature::EncodingYSign => CatalogEntry {
            id: "representation.y-sign",
            family: CatalogFamily::EncodingSymmetry,
            conditional: false,
            representation_artifact: true,
            reference: SEC1,
        },
        RelationSignature::EqualJInvariant => CatalogEntry {
            id: "curve.equal-j-class",
            family: CatalogFamily::TwistAndJClass,
            conditional: true,
            representation_artifact: false,
            reference: SILVERMAN,
        },
        RelationSignature::Unrecognized => return None,
    };
    Some(entry)
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::{ClassificationStatus, classify};

    #[test]
    fn every_required_known_family_has_an_executable_entry()
    {
        let signatures = [
            RelationSignature::NegationInvolution,
            RelationSignature::AdditiveInverse,
            RelationSignature::ScalarComposition,
            RelationSignature::JZeroXScale { zeta: 3 },
            RelationSignature::CubeRootOfUnity { zeta: 3 },
            RelationSignature::J1728Automorphism,
            RelationSignature::GlvEigenRelation { lambda: 7 },
            RelationSignature::CoordinateScale { factor: 2 },
            RelationSignature::EncodingYSign,
            RelationSignature::EqualJInvariant,
        ];
        for signature in signatures
        {
            assert!(catalog_entry(signature).is_some());
        }
    }

    #[test]
    fn representation_rules_cannot_be_classified_as_candidates()
    {
        for signature in [
            RelationSignature::CoordinateScale { factor: 2 },
            RelationSignature::EncodingYSign,
        ]
        {
            assert_eq!(
                classify(signature, false).status(),
                ClassificationStatus::RepresentationArtifact
            );
        }
    }

    #[test]
    fn unknown_signature_requires_literature_review()
    {
        assert_eq!(
            classify(RelationSignature::Unrecognized, false).status(),
            ClassificationStatus::NeedsLiteratureReview
        );
    }
}
