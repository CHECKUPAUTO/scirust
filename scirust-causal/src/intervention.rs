//! Interventions: what changed, on which variable, and how.
//!
//! An [`Intervention`] is a **typed record of what the experimenter (or
//! nature) did** to a variable's data-generating mechanism. It is not a claim
//! about causal consequences — inferring those is later, separate work (see
//! [`crate::CausalCertificate`]).

use crate::error::CausalError;

/// How a variable's data-generating mechanism was altered.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InterventionKind {
    /// Pearl's `do(X = value)`: the variable is forced to a fixed value,
    /// severing it from its usual causes.
    Atomic { value: f64 },
    /// An additive, mechanism-preserving shift `X ← X + delta` (as in
    /// shift-intervention / anchor-regression settings): the variable still
    /// responds to its usual causes, offset by `delta`.
    Shift { delta: f64 },
    /// The variable's mechanism is known to have changed, but not
    /// parametrically how (e.g. "a different sensor batch", "a policy
    /// change") — an honest placeholder for real-world regime changes without
    /// a modeled functional form.
    MechanismChange { description: String },
    /// An intervention is known to have occurred but its kind was not
    /// recorded. Never treated as equivalent to no intervention.
    Unspecified,
}

/// One intervention: `kind` applied to the variable at `target` (a
/// [`crate::CausalVariable::index`]).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Intervention {
    pub target: usize,
    pub kind: InterventionKind,
}

impl Intervention {
    /// Validates and constructs an intervention. Does **not** check `target`
    /// against a variable set — callers that have one (e.g.
    /// [`crate::CausalDataset::new`]) validate that separately, since
    /// `Intervention` alone has no such context.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidContract`] if an `Atomic`/`Shift` parameter is
    /// non-finite, or a `MechanismChange` description is empty.
    pub fn new(target: usize, kind: InterventionKind) -> Result<Self, CausalError> {
        let ok = match &kind
        {
            InterventionKind::Atomic { value } => value.is_finite(),
            InterventionKind::Shift { delta } => delta.is_finite(),
            InterventionKind::MechanismChange { description } => !description.trim().is_empty(),
            InterventionKind::Unspecified => true,
        };
        if !ok
        {
            return Err(CausalError::InvalidContract {
                detail: "intervention parameter must be finite (or description non-empty)",
            });
        }
        Ok(Self { target, kind })
    }
}
