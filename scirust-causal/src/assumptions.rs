//! A typed, provenance-tracked registry of causal assumptions.
//!
//! Every nontrivial causal claim rests on assumptions that cannot be read off
//! the data (acyclicity, causal sufficiency, faithfulness, …) — see the crate
//! root's "Causal interpretation" section. [`AssumptionRegistry`] makes
//! recording *which* assumptions a conclusion relies on, and *why* they are
//! believed to hold, a first-class, inspectable object rather than a comment
//! in a docstring.

use crate::error::CausalError;
use std::collections::BTreeMap;

/// A named causal assumption. The closed variants are the ones this crate's
/// own honesty documentation names explicitly; [`CausalAssumption::Other`] is
/// an escape hatch for anything not yet named.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum CausalAssumption {
    /// The true causal structure is acyclic.
    Acyclicity,
    /// No unmeasured common cause of any two measured variables (no latent
    /// confounding).
    CausalSufficiency,
    /// Every conditional independence in the data reflects a d-separation in
    /// the true graph (no coincidental cancellation).
    Faithfulness,
    /// The assumed functional form / noise model matches the true
    /// data-generating process.
    CorrectFunctionalForm,
    /// The sample size is adequate for the statistical claims being made.
    AdequateSampleSize,
    /// The Stable Unit Treatment Value Assumption: one unit's outcome does not
    /// depend on another unit's treatment, and each treatment level
    /// corresponds to one well-defined intervention.
    Sutva,
    /// Treatment assignment is as-good-as-random given the adjustment set
    /// (ignorability / exchangeability).
    Exchangeability,
    /// Every adjustment-set stratum has a nonzero probability of every
    /// treatment level.
    Positivity,
    /// The causal mechanism of interest is the same across the environments
    /// being compared (the precondition invariance-based methods rely on).
    InvarianceAcrossEnvironments,
    /// An assumption not covered by the named variants above.
    Other(String),
}

/// The reason an assumption is believed to hold — its **provenance**.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AssumptionBasis {
    /// Asserted by the analyst as a judgment call, with no supporting check.
    AssertedByAnalyst,
    /// Guaranteed by the data-collection design itself (e.g. randomization).
    GuaranteedByDesign { mechanism: String },
    /// Checked by a named statistical test or procedure.
    TestedStatistically {
        test_name: String,
        p_value: Option<f64>,
    },
    /// Backed by external domain knowledge or literature, not by this dataset.
    DomainKnowledge { citation: String },
    /// Explicitly flagged as not checked at all. The safe default.
    Unverified,
}

/// One registry entry: an assumption's basis plus an optional free-text note.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssumptionRecord {
    pub basis: AssumptionBasis,
    pub note: Option<String>,
}

/// A provenance-tracked set of assumptions, keyed by [`CausalAssumption`] so
/// each is asserted at most once — re-asserting requires an explicit
/// [`AssumptionRegistry::overwrite`], never a silent replace. Iteration order
/// is the assumption's `Ord` order, deterministic regardless of insertion
/// order, which is what lets a [`crate::CausalCertificate`] built from this
/// registry have a stable fingerprint.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AssumptionRegistry {
    entries: BTreeMap<CausalAssumption, AssumptionRecord>,
}

impl AssumptionRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Asserts an assumption. Errors if it is already registered.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidContract`] if `assumption` is already registered
    /// — use [`AssumptionRegistry::overwrite`] to knowingly replace it.
    pub fn assert(
        &mut self,
        assumption: CausalAssumption,
        basis: AssumptionBasis,
        note: Option<String>,
    ) -> Result<(), CausalError> {
        if self.entries.contains_key(&assumption)
        {
            return Err(CausalError::InvalidContract {
                detail: "assumption is already registered; use `overwrite` to replace it",
            });
        }
        self.entries
            .insert(assumption, AssumptionRecord { basis, note });
        Ok(())
    }

    /// Replaces an existing entry (or inserts if absent), on purpose.
    pub fn overwrite(
        &mut self,
        assumption: CausalAssumption,
        basis: AssumptionBasis,
        note: Option<String>,
    ) {
        self.entries
            .insert(assumption, AssumptionRecord { basis, note });
    }

    #[must_use]
    pub fn get(&self, assumption: &CausalAssumption) -> Option<&AssumptionRecord> {
        self.entries.get(assumption)
    }

    /// `true` iff `assumption` is registered with anything other than
    /// [`AssumptionBasis::Unverified`].
    #[must_use]
    pub fn is_supported(&self, assumption: &CausalAssumption) -> bool {
        matches!(
            self.entries.get(assumption),
            Some(AssumptionRecord {
                basis: b,
                ..
            }) if !matches!(b, AssumptionBasis::Unverified)
        )
    }

    /// Iterates registered assumptions in deterministic (`Ord`) order.
    pub fn iter(&self) -> impl Iterator<Item = (&CausalAssumption, &AssumptionRecord)> {
        self.entries.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
