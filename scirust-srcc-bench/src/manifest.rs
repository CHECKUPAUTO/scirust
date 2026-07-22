//! Machine-readable dataset manifests.
//!
//! Every dataset the harness touches must carry a [`DatasetManifest`]: the
//! provenance (source, version, license), the canonical content checksum and
//! the declared shape. Manifests are validated *against the dataset itself*
//! — a manifest whose checksum or shape disagrees with the data is a typed
//! error, never a warning.
//!
//! Network policy, stated once: nothing in this crate downloads anything.
//! Large real datasets (phase 728) are fetched by explicit scripts outside
//! `cargo test`, verified against `sha256` recorded here, and integration
//! tests **skip** (not fail, not download) when the data is absent.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::dataset::TabularDataset;

/// One feature column's declared identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FeatureDescriptor {
    /// Stable column name.
    pub name: String,
    /// Physical unit as written in the source documentation (`"°C"`,
    /// `"rpm"`, `"dimensionless"`, …). Free text, but mandatory: unit
    /// ambiguity is exactly what the scale-aware phases are about.
    pub unit: String,
    /// One-line human description.
    pub description: String,
}

/// Machine-readable dataset identity and provenance.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DatasetManifest {
    /// Stable dataset name (the `dataset` key of emitted records).
    pub name: String,
    /// Dataset version as published by its source.
    pub version: String,
    /// Where the data comes from (URL or citation; documentation, not a
    /// fetch instruction — the harness never downloads).
    pub source: String,
    /// License or redistribution terms, verbatim identifier.
    pub license: String,
    /// Canonical content checksum ([`TabularDataset::content_sha256`]).
    pub sha256: String,
    /// Declared row count.
    pub sample_count: usize,
    /// Declared feature-column count.
    pub feature_count: usize,
    /// What the target column means, including units.
    pub target_description: String,
    /// One descriptor per feature column, in column order.
    pub feature_descriptors: Vec<FeatureDescriptor>,
}

/// Typed manifest errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestError {
    /// A mandatory text field is empty.
    EmptyField {
        /// The offending field name.
        field: &'static str,
    },
    /// `sha256` is not 64 lowercase hex characters.
    MalformedChecksum,
    /// `feature_descriptors.len() != feature_count`.
    DescriptorCountMismatch {
        /// Declared `feature_count`.
        declared: usize,
        /// `feature_descriptors.len()`.
        found: usize,
    },
    /// The manifest's declared shape disagrees with the dataset.
    ShapeMismatch {
        /// Declared (rows, columns).
        declared: (usize, usize),
        /// Actual (rows, columns).
        found: (usize, usize),
    },
    /// The manifest's checksum disagrees with the dataset content.
    ChecksumMismatch {
        /// Declared checksum.
        declared: String,
        /// Actual content checksum.
        found: String,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyField { field } => write!(formatter, "manifest field `{field}` is empty"),
            Self::MalformedChecksum =>
            {
                formatter.write_str("manifest sha256 must be 64 lowercase hex characters")
            },
            Self::DescriptorCountMismatch { declared, found } => write!(
                formatter,
                "manifest declares {declared} features but carries {found} descriptors"
            ),
            Self::ShapeMismatch { declared, found } => write!(
                formatter,
                "manifest declares shape {declared:?}, dataset has {found:?}"
            ),
            Self::ChecksumMismatch { declared, found } => write!(
                formatter,
                "manifest checksum {declared} does not match dataset content {found}"
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

impl DatasetManifest {
    /// Builds a manifest for an in-memory dataset, computing shape and
    /// checksum from the data itself (the descriptors still come from the
    /// caller — provenance cannot be derived from bytes).
    pub fn for_dataset(
        dataset: &TabularDataset,
        name: impl Into<String>,
        version: impl Into<String>,
        source: impl Into<String>,
        license: impl Into<String>,
        target_description: impl Into<String>,
        feature_descriptors: Vec<FeatureDescriptor>,
    ) -> Result<Self, ManifestError> {
        let manifest = Self {
            name: name.into(),
            version: version.into(),
            source: source.into(),
            license: license.into(),
            sha256: dataset.content_sha256(),
            sample_count: dataset.sample_count(),
            feature_count: dataset.feature_count(),
            target_description: target_description.into(),
            feature_descriptors,
        };

        manifest.validate_against(dataset)?;

        Ok(manifest)
    }

    /// Validates internal consistency only (fields, checksum shape,
    /// descriptor count).
    pub fn validate(&self) -> Result<(), ManifestError> {
        for (field, value) in [
            ("name", &self.name),
            ("version", &self.version),
            ("source", &self.source),
            ("license", &self.license),
            ("target_description", &self.target_description),
        ]
        {
            if value.trim().is_empty()
            {
                return Err(ManifestError::EmptyField { field });
            }
        }

        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        {
            return Err(ManifestError::MalformedChecksum);
        }

        if self.feature_descriptors.len() != self.feature_count
        {
            return Err(ManifestError::DescriptorCountMismatch {
                declared: self.feature_count,
                found: self.feature_descriptors.len(),
            });
        }

        Ok(())
    }

    /// Validates internal consistency **and** agreement with a dataset:
    /// declared shape and checksum must match the actual content.
    pub fn validate_against(&self, dataset: &TabularDataset) -> Result<(), ManifestError> {
        self.validate()?;

        let found = (dataset.sample_count(), dataset.feature_count());
        let declared = (self.sample_count, self.feature_count);

        if declared != found
        {
            return Err(ManifestError::ShapeMismatch { declared, found });
        }

        let content = dataset.content_sha256();

        if self.sha256 != content
        {
            return Err(ManifestError::ChecksumMismatch {
                declared: self.sha256.clone(),
                found: content,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> TabularDataset {
        TabularDataset {
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            targets: vec![0.0, 1.0],
            groups: None,
            time_index: None,
        }
    }

    fn descriptors() -> Vec<FeatureDescriptor> {
        vec![
            FeatureDescriptor {
                name: "temperature".into(),
                unit: "°C".into(),
                description: "inlet temperature".into(),
            },
            FeatureDescriptor {
                name: "speed".into(),
                unit: "rpm".into(),
                description: "shaft speed".into(),
            },
        ]
    }

    #[test]
    fn for_dataset_builds_a_manifest_that_validates() {
        let data = dataset();

        let manifest = DatasetManifest::for_dataset(
            &data,
            "synthetic_demo",
            "1",
            "generated in-repo",
            "MIT",
            "held-out quality score (dimensionless)",
            descriptors(),
        )
        .unwrap();

        assert_eq!(manifest.sample_count, 2);
        assert_eq!(manifest.feature_count, 2);
        assert_eq!(manifest.validate_against(&data), Ok(()));
    }

    #[test]
    fn tampered_content_is_a_checksum_mismatch() {
        let data = dataset();

        let manifest = DatasetManifest::for_dataset(
            &data,
            "synthetic_demo",
            "1",
            "generated in-repo",
            "MIT",
            "score",
            descriptors(),
        )
        .unwrap();

        let mut tampered = data.clone();
        tampered.features[0][0] = 1.0000001;

        assert!(matches!(
            manifest.validate_against(&tampered),
            Err(ManifestError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn declared_shape_must_match() {
        let data = dataset();

        let mut manifest = DatasetManifest::for_dataset(
            &data,
            "synthetic_demo",
            "1",
            "generated in-repo",
            "MIT",
            "score",
            descriptors(),
        )
        .unwrap();

        manifest.sample_count = 3;

        assert_eq!(
            manifest.validate_against(&data),
            Err(ManifestError::ShapeMismatch {
                declared: (3, 2),
                found: (2, 2),
            }),
        );
    }

    #[test]
    fn field_and_checksum_validation_is_typed() {
        let data = dataset();

        let manifest = DatasetManifest::for_dataset(
            &data,
            "synthetic_demo",
            "1",
            "generated in-repo",
            "MIT",
            "score",
            descriptors(),
        )
        .unwrap();

        let mut empty_name = manifest.clone();
        empty_name.name = "  ".into();
        assert_eq!(
            empty_name.validate(),
            Err(ManifestError::EmptyField { field: "name" }),
        );

        let mut bad_hash = manifest.clone();
        bad_hash.sha256 = "ABC".into();
        assert_eq!(bad_hash.validate(), Err(ManifestError::MalformedChecksum));

        let mut uppercase_hash = manifest.clone();
        uppercase_hash.sha256 = uppercase_hash.sha256.to_uppercase();
        assert_eq!(
            uppercase_hash.validate(),
            Err(ManifestError::MalformedChecksum),
        );

        let mut missing_descriptor = manifest.clone();
        missing_descriptor.feature_descriptors.pop();
        assert_eq!(
            missing_descriptor.validate(),
            Err(ManifestError::DescriptorCountMismatch {
                declared: 2,
                found: 1,
            }),
        );
    }

    #[test]
    fn manifest_serializes_to_stable_json() {
        let manifest = DatasetManifest::for_dataset(
            &dataset(),
            "synthetic_demo",
            "1",
            "generated in-repo",
            "MIT",
            "score",
            descriptors(),
        )
        .unwrap();

        let json = serde_json::to_string(&manifest).unwrap();
        let back: DatasetManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(back, manifest);
    }
}
