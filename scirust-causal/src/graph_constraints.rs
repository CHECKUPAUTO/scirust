//! Background knowledge about a causal graph, independent of any fitted
//! model: edges known to exist or be absent, and a partial temporal/tier
//! ordering.
//!
//! [`GraphConstraints::check`] validates a candidate [`CausalDag`] — e.g. one
//! produced by [`crate::extract_causal_dag`] — against this background
//! knowledge. This phase defines the type and the check, not any discovery
//! procedure that would consume it.

use crate::error::CausalError;
use scirust_graph::dag::CausalDag;
use std::collections::BTreeSet;

/// One way a candidate DAG can violate a [`GraphConstraints`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConstraintViolation {
    /// A required edge `from -> to` is absent from the candidate DAG.
    MissingRequiredEdge { from: usize, to: usize },
    /// A forbidden edge `from -> to` is present in the candidate DAG.
    PresentForbiddenEdge { from: usize, to: usize },
    /// An edge `from -> to` runs from a later tier to an earlier one.
    TierViolation {
        from: usize,
        to: usize,
        from_tier: usize,
        to_tier: usize,
    },
}

/// Background knowledge over `n_variables` variables: edges known to exist or
/// be absent, and an optional tier (partial temporal order) per variable. An
/// edge may only run from an equal-or-earlier tier to an equal-or-later one.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphConstraints {
    n_variables: usize,
    required_edges: BTreeSet<(usize, usize)>,
    forbidden_edges: BTreeSet<(usize, usize)>,
    tier_of: Vec<Option<usize>>,
}

impl GraphConstraints {
    #[must_use]
    pub fn new(n_variables: usize) -> Self {
        Self {
            n_variables,
            required_edges: BTreeSet::new(),
            forbidden_edges: BTreeSet::new(),
            tier_of: vec![None; n_variables],
        }
    }

    fn check_index(&self, i: usize) -> Result<(), CausalError> {
        if i >= self.n_variables
        {
            return Err(CausalError::UnknownVariableIndex { index: i });
        }
        Ok(())
    }

    /// Declares that `from -> to` must be present in any acceptable DAG.
    ///
    /// # Errors
    ///
    /// [`CausalError::UnknownVariableIndex`] for an out-of-range index;
    /// [`CausalError::InvalidContract`] if `from == to`, the edge is already
    /// forbidden, or it violates an existing tier ordering.
    pub fn require_edge(&mut self, from: usize, to: usize) -> Result<(), CausalError> {
        self.check_index(from)?;
        self.check_index(to)?;
        if from == to
        {
            return Err(CausalError::InvalidContract {
                detail: "a required edge cannot be a self-loop",
            });
        }
        if self.forbidden_edges.contains(&(from, to))
        {
            return Err(CausalError::InvalidContract {
                detail: "edge is already forbidden; cannot also require it",
            });
        }
        if let (Some(ft), Some(tt)) = (self.tier_of[from], self.tier_of[to])
        {
            if ft > tt
            {
                return Err(CausalError::InvalidContract {
                    detail: "required edge runs from a later tier to an earlier tier",
                });
            }
        }
        self.required_edges.insert((from, to));
        Ok(())
    }

    /// Declares that `from -> to` must be absent from any acceptable DAG.
    ///
    /// # Errors
    ///
    /// [`CausalError::UnknownVariableIndex`] for an out-of-range index;
    /// [`CausalError::InvalidContract`] if the edge is already required.
    pub fn forbid_edge(&mut self, from: usize, to: usize) -> Result<(), CausalError> {
        self.check_index(from)?;
        self.check_index(to)?;
        if self.required_edges.contains(&(from, to))
        {
            return Err(CausalError::InvalidContract {
                detail: "edge is already required; cannot also forbid it",
            });
        }
        self.forbidden_edges.insert((from, to));
        Ok(())
    }

    /// Assigns a tier (partial temporal order) to a variable.
    ///
    /// # Errors
    ///
    /// [`CausalError::UnknownVariableIndex`] for an out-of-range index;
    /// [`CausalError::InvalidContract`] if this tier would retroactively
    /// violate an already-required edge (the assignment is rolled back).
    pub fn set_tier(&mut self, variable: usize, tier: usize) -> Result<(), CausalError> {
        self.check_index(variable)?;
        let previous = self.tier_of[variable];
        self.tier_of[variable] = Some(tier);

        for &(from, to) in &self.required_edges
        {
            if let (Some(ft), Some(tt)) = (self.tier_of[from], self.tier_of[to])
            {
                if ft > tt
                {
                    self.tier_of[variable] = previous;
                    return Err(CausalError::InvalidContract {
                        detail: "tier assignment would violate an existing required edge",
                    });
                }
            }
        }
        Ok(())
    }

    /// Checks a candidate DAG against every constraint, returning every
    /// violation found (empty ⇒ the DAG is consistent with this background
    /// knowledge). Safe against any `dag.n_nodes()` — a `dag` smaller than
    /// `n_variables` simply cannot satisfy edges or tiers that reference the
    /// missing indices, so those surface as ordinary violations rather than a
    /// panic.
    #[must_use]
    pub fn check(&self, dag: &CausalDag) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        for &(from, to) in &self.required_edges
        {
            let present = from < dag.n_nodes() && dag.children(from).contains(&to);
            if !present
            {
                violations.push(ConstraintViolation::MissingRequiredEdge { from, to });
            }
        }
        for &(from, to) in &self.forbidden_edges
        {
            let present = from < dag.n_nodes() && dag.children(from).contains(&to);
            if present
            {
                violations.push(ConstraintViolation::PresentForbiddenEdge { from, to });
            }
        }
        for from in 0..dag.n_nodes().min(self.n_variables)
        {
            for &to in dag.children(from)
            {
                if to >= self.n_variables
                {
                    continue;
                }
                if let (Some(ft), Some(tt)) = (self.tier_of[from], self.tier_of[to])
                {
                    if ft > tt
                    {
                        violations.push(ConstraintViolation::TierViolation {
                            from,
                            to,
                            from_tier: ft,
                            to_tier: tt,
                        });
                    }
                }
            }
        }
        violations
    }
}
