//! Environments: labeled data-generating regimes.
//!
//! An [`Environment`] names *which regime* produced a block of rows — plain
//! observational data (`interventions` empty) or one or more simultaneous
//! [`Intervention`]s. Tagging data by environment is the precondition later
//! invariance testing (comparing a relationship across environments) needs to
//! operate on; this phase only defines the type.

use crate::error::CausalError;
use crate::intervention::Intervention;
use std::collections::BTreeSet;

/// A labeled data-generating regime.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Environment {
    /// Stable, human-readable identifier (e.g. `"observational"`, `"site_b"`,
    /// `"post_policy_change"`). Must be non-empty.
    pub id: String,
    /// Simultaneous interventions active in this environment. Empty means
    /// purely observational.
    pub interventions: Vec<Intervention>,
}

impl Environment {
    /// A purely observational environment (no interventions).
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidContract`] if `id` is empty.
    pub fn observational(id: impl Into<String>) -> Result<Self, CausalError> {
        Self::new(id, Vec::new())
    }

    /// An environment with the given simultaneous interventions.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidContract`] if `id` is empty, or two interventions
    /// target the same variable (an incoherent simultaneous regime).
    pub fn new(
        id: impl Into<String>,
        interventions: Vec<Intervention>,
    ) -> Result<Self, CausalError> {
        let id = id.into();
        if id.trim().is_empty()
        {
            return Err(CausalError::InvalidContract {
                detail: "environment id must not be empty",
            });
        }
        let mut seen_targets = BTreeSet::new();
        for iv in &interventions
        {
            if !seen_targets.insert(iv.target)
            {
                return Err(CausalError::InvalidContract {
                    detail: "environment cannot intervene on the same variable twice",
                });
            }
        }
        Ok(Self { id, interventions })
    }

    /// `true` if this environment carries no interventions.
    #[must_use]
    pub fn is_observational(&self) -> bool {
        self.interventions.is_empty()
    }
}
