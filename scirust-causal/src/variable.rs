//! Typed causal variables — identity, semantic role, and value kind.
//!
//! A [`CausalVariable::index`] **must** match its column position in any
//! [`crate::CausalDataset`] built over the same variable set, and its node id
//! in any [`scirust_graph::dag::CausalDag`] derived from that dataset — the
//! typed-contracts layer indexes variables positionally, the same convention
//! [`crate::extract_causal_dag`] already uses.

use crate::error::CausalError;

/// The semantic role a variable plays in a specific causal query. Roles are
/// relative to a question ("the effect of X on Y"), not an intrinsic property
/// of the variable — the same variable can be a `Covariate` in one query and
/// the `Treatment` in another.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum VariableRole {
    /// The purported cause / exposure whose effect is of interest.
    Treatment,
    /// The effect / response variable.
    Outcome,
    /// Measured but not the direct subject of the query (a candidate for
    /// adjustment, not yet assigned a more specific role).
    Covariate,
    /// Believed to influence both treatment and outcome (a confounder).
    Confounder,
    /// Believed to lie on a causal path between treatment and outcome.
    Mediator,
    /// Believed to affect treatment assignment but not the outcome except
    /// through treatment (an instrumental-variable candidate).
    Instrument,
    /// A common effect of two or more other variables; conditioning on it can
    /// induce spurious association (a collider).
    Collider,
    /// No role asserted yet.
    Unspecified,
}

/// The measurement scale of a variable's values.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum VariableKind {
    /// Real-valued.
    Continuous,
    /// A finite set of unordered categories.
    Discrete,
    /// Exactly two states — called out separately from `Discrete` because
    /// several downstream checks (positivity, propensity) are binary-specific.
    Binary,
}

/// One named, typed variable in a causal contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CausalVariable {
    /// Positional index — must match the variable's column in any
    /// [`crate::CausalDataset`] and node id in any `CausalDag` over the same set.
    pub index: usize,
    /// Human-readable name (must be non-empty; unique within a variable set,
    /// checked by [`validate_variable_set`]).
    pub name: String,
    /// Semantic role for the query this variable set was built for.
    pub role: VariableRole,
    /// Measurement scale.
    pub kind: VariableKind,
}

impl CausalVariable {
    /// Validates and constructs a variable.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidContract`] if `name` is empty (or all whitespace).
    pub fn new(
        index: usize,
        name: impl Into<String>,
        role: VariableRole,
        kind: VariableKind,
    ) -> Result<Self, CausalError> {
        let name = name.into();
        if name.trim().is_empty()
        {
            return Err(CausalError::InvalidContract {
                detail: "variable name must not be empty",
            });
        }
        Ok(Self {
            index,
            name,
            role,
            kind,
        })
    }
}

/// Validates a set of variables: indices are exactly `0..variables.len()`
/// with no gaps or duplicates (so they can serve directly as
/// [`crate::CausalDataset`] columns and `CausalDag` node ids), and names are
/// unique.
///
/// # Errors
///
/// [`CausalError::InvalidContract`] for an out-of-range or duplicate index, or
/// a duplicate name.
pub fn validate_variable_set(variables: &[CausalVariable]) -> Result<(), CausalError> {
    let n = variables.len();
    let mut seen_index = vec![false; n];
    let mut seen_names: Vec<&str> = Vec::with_capacity(n);

    for v in variables
    {
        if v.index >= n
        {
            return Err(CausalError::InvalidContract {
                detail: "variable index is out of range for this variable set",
            });
        }
        if seen_index[v.index]
        {
            return Err(CausalError::InvalidContract {
                detail: "duplicate variable index in variable set",
            });
        }
        seen_index[v.index] = true;

        if seen_names.contains(&v.name.as_str())
        {
            return Err(CausalError::InvalidContract {
                detail: "duplicate variable name in variable set",
            });
        }
        seen_names.push(&v.name);
    }
    Ok(())
}
