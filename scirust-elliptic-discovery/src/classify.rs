//! Conservative classification with no "new" or "discovered" status.

use crate::{CatalogEntry, RelationSignature, catalog_entry};

/// Allowed automated outcomes.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ClassificationStatus
{
    Refuted,
    Known,
    RepresentationArtifact,
    NeedsLiteratureReview,
    Inconclusive,
    CandidateUnclassified,
}

/// Catalog-backed classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Classification
{
    status: ClassificationStatus,
    catalog: Option<CatalogEntry>,
}

impl Classification
{
    /// Explicit insufficient-coverage outcome.
    pub const fn inconclusive() -> Self
    {
        Self {
            status: ClassificationStatus::Inconclusive,
            catalog: None,
        }
    }

    pub const fn status(self) -> ClassificationStatus
    {
        self.status
    }

    pub const fn catalog(self) -> Option<CatalogEntry>
    {
        self.catalog
    }
}

/// Classifies a structural signature after falsification.
pub const fn classify(signature: RelationSignature, has_counterexample: bool) -> Classification
{
    if has_counterexample
    {
        return Classification {
            status: ClassificationStatus::Refuted,
            catalog: catalog_entry(signature),
        };
    }
    match catalog_entry(signature)
    {
        Some(entry) if entry.representation_artifact => Classification {
            status: ClassificationStatus::RepresentationArtifact,
            catalog: Some(entry),
        },
        Some(entry) => Classification {
            status: ClassificationStatus::Known,
            catalog: Some(entry),
        },
        None => Classification {
            status: ClassificationStatus::NeedsLiteratureReview,
            catalog: None,
        },
    }
}
