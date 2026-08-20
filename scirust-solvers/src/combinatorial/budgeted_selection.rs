//! Exact deterministic budgeted subset selection.
//!
//! This module solves the additive 0/1 problem
//!
//! ```text
//! maximize   Σ utility_i x_i
//! subject to Σ cost_i x_i ≤ budget
//!            x_i ∈ {0, 1}
//! ```
//!
//! using a sparse Pareto-frontier dynamic program. Unlike the classical
//! capacity-indexed knapsack table, memory does not scale directly with the
//! numeric value of `budget`; after every item, states dominated in
//! `(cost, utility)` are removed.
//!
//! The solver is **exact** for the supplied integer costs/utilities. This is
//! useful when floating-point scores have first been converted to a documented
//! fixed-point scale (basis points, micro-units, etc.). Identical inputs produce
//! identical selections: ties maximize utility, then minimize total cost, then
//! choose the lexicographically smallest vector of selected item indices.

use thiserror::Error;

/// One independently selectable item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetedItem {
    /// Non-negative additive resource cost.
    pub cost: u64,
    /// Non-negative additive utility, usually a fixed-point score.
    pub utility: u64,
}

/// Exact-solver statistics and optimality statement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetedSelectionCertificate {
    /// The dynamic program exhausted all non-dominated possibilities.
    pub proven_optimal: bool,
    /// Number of candidate states considered before Pareto pruning.
    pub explored_states: usize,
    /// Number of non-dominated states in the final frontier.
    pub final_frontier_size: usize,
}

/// Globally optimal selection under the supplied budget.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BudgetedSelection {
    /// Selected item indices in ascending input order.
    pub selected_indices: Vec<usize>,
    /// Sum of selected costs.
    pub total_cost: u64,
    /// Sum of selected utilities.
    pub total_utility: u64,
    /// Exactness / work metadata.
    pub certificate: BudgetedSelectionCertificate,
}

/// Typed failures of [`exact_budgeted_selection`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BudgetedSelectionError {
    /// The exact additive utility cannot be represented in `u64`.
    #[error("utility sum overflowed u64")]
    UtilityOverflow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct State {
    cost: u64,
    utility: u64,
    selected: Vec<usize>,
}

/// Solves additive 0/1 budgeted selection exactly and deterministically.
///
/// The empty set is always feasible, including for `budget == 0`. Zero-cost
/// positive-utility items are handled normally and will be selected when they
/// improve the objective.
///
/// # Exactness
///
/// After considering each item, a state `(c₂, u₂)` is discarded when another
/// retained state `(c₁, u₁)` satisfies `c₁ ≤ c₂` and `u₁ ≥ u₂` (with the
/// deterministic tie-break selecting one representative when both are equal).
/// Such a dominated prefix can never become better after adding the same set of
/// remaining non-negative item costs/utilities, so Pareto pruning preserves the
/// global optimum.
pub fn exact_budgeted_selection(
    items: &[BudgetedItem],
    budget: u64,
) -> Result<BudgetedSelection, BudgetedSelectionError> {
    let mut frontier = vec![State {
        cost: 0,
        utility: 0,
        selected: Vec::new(),
    }];
    let mut explored_states = 1usize;

    for (index, item) in items.iter().enumerate()
    {
        let previous = frontier;
        let mut candidates = Vec::with_capacity(previous.len().saturating_mul(2));

        // Excluding the current item preserves every previous state.
        candidates.extend(previous.iter().cloned());

        // Including the current item creates one candidate from each feasible
        // previous state. A cost overflow is necessarily above any representable
        // `budget`, so it is safely infeasible; utility overflow cannot be
        // represented exactly and is therefore reported.
        for state in &previous
        {
            let Some(cost) = state.cost.checked_add(item.cost)
            else
            {
                continue;
            };
            if cost > budget
            {
                continue;
            }
            let utility = state
                .utility
                .checked_add(item.utility)
                .ok_or(BudgetedSelectionError::UtilityOverflow)?;
            let mut selected = state.selected.clone();
            selected.push(index);
            candidates.push(State {
                cost,
                utility,
                selected,
            });
        }

        explored_states = explored_states.saturating_add(candidates.len());
        frontier = pareto_prune(candidates);
    }

    let best = frontier
        .iter()
        .max_by(|a, b| compare_objective(a, b))
        .expect("the empty selection always keeps the frontier non-empty")
        .clone();

    Ok(BudgetedSelection {
        selected_indices: best.selected,
        total_cost: best.cost,
        total_utility: best.utility,
        certificate: BudgetedSelectionCertificate {
            proven_optimal: true,
            explored_states,
            final_frontier_size: frontier.len(),
        },
    })
}

/// Ordering where `Greater` means "better final objective".
fn compare_objective(a: &State, b: &State) -> core::cmp::Ordering {
    a.utility
        .cmp(&b.utility)
        .then_with(|| b.cost.cmp(&a.cost))
        .then_with(|| b.selected.cmp(&a.selected))
}

fn pareto_prune(mut candidates: Vec<State>) -> Vec<State> {
    // For each exact cost, put the best utility first; for an exact objective
    // tie, put the lexicographically smallest selected-index vector first.
    candidates.sort_by(|a, b| {
        a.cost
            .cmp(&b.cost)
            .then_with(|| b.utility.cmp(&a.utility))
            .then_with(|| a.selected.cmp(&b.selected))
    });

    let mut unique_cost = Vec::with_capacity(candidates.len());
    let mut last_cost = None;
    for state in candidates
    {
        if last_cost == Some(state.cost)
        {
            continue;
        }
        last_cost = Some(state.cost);
        unique_cost.push(state);
    }

    // Costs are increasing. Retain only states whose utility is strictly above
    // every cheaper state's utility. Equal utility at larger cost is dominated.
    let mut frontier = Vec::with_capacity(unique_cost.len());
    let mut best_utility: Option<u64> = None;
    for state in unique_cost
    {
        if best_utility.is_none_or(|utility| state.utility > utility)
        {
            best_utility = Some(state.utility);
            frontier.push(state);
        }
    }
    frontier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_classic_knapsack_exactly() {
        let items = [
            BudgetedItem {
                cost: 10,
                utility: 60,
            },
            BudgetedItem {
                cost: 20,
                utility: 100,
            },
            BudgetedItem {
                cost: 30,
                utility: 120,
            },
        ];
        let result = exact_budgeted_selection(&items, 50).unwrap();
        assert_eq!(result.selected_indices, vec![1, 2]);
        assert_eq!(result.total_cost, 50);
        assert_eq!(result.total_utility, 220);
        assert!(result.certificate.proven_optimal);
    }

    #[test]
    fn objective_ties_prefer_lower_cost_then_lexicographic_indices() {
        let lower_cost = [
            BudgetedItem {
                cost: 4,
                utility: 10,
            },
            BudgetedItem {
                cost: 3,
                utility: 10,
            },
        ];
        let result = exact_budgeted_selection(&lower_cost, 4).unwrap();
        assert_eq!(result.selected_indices, vec![1]);
        assert_eq!(result.total_cost, 3);

        let lexicographic = [
            BudgetedItem {
                cost: 3,
                utility: 10,
            },
            BudgetedItem {
                cost: 3,
                utility: 10,
            },
        ];
        let result = exact_budgeted_selection(&lexicographic, 3).unwrap();
        assert_eq!(result.selected_indices, vec![0]);
    }

    #[test]
    fn zero_budget_keeps_positive_utility_zero_cost_items() {
        let items = [
            BudgetedItem {
                cost: 0,
                utility: 3,
            },
            BudgetedItem {
                cost: 1,
                utility: 100,
            },
            BudgetedItem {
                cost: 0,
                utility: 4,
            },
        ];
        let result = exact_budgeted_selection(&items, 0).unwrap();
        assert_eq!(result.selected_indices, vec![0, 2]);
        assert_eq!(result.total_cost, 0);
        assert_eq!(result.total_utility, 7);
    }

    #[test]
    fn empty_input_returns_empty_optimum() {
        let result = exact_budgeted_selection(&[], 123).unwrap();
        assert!(result.selected_indices.is_empty());
        assert_eq!(result.total_cost, 0);
        assert_eq!(result.total_utility, 0);
        assert_eq!(result.certificate.final_frontier_size, 1);
    }

    #[test]
    fn reports_utility_overflow_instead_of_wrapping() {
        let items = [
            BudgetedItem {
                cost: 0,
                utility: u64::MAX,
            },
            BudgetedItem {
                cost: 0,
                utility: 1,
            },
        ];
        assert_eq!(
            exact_budgeted_selection(&items, 0),
            Err(BudgetedSelectionError::UtilityOverflow)
        );
    }

    #[test]
    fn pareto_pruning_preserves_non_dominated_tradeoffs() {
        let items = [
            BudgetedItem {
                cost: 2,
                utility: 3,
            },
            BudgetedItem {
                cost: 3,
                utility: 5,
            },
            BudgetedItem {
                cost: 5,
                utility: 8,
            },
        ];
        let result = exact_budgeted_selection(&items, 5).unwrap();
        assert_eq!(result.total_utility, 8);
        // Equal utility is available as item 2 alone (cost 5) or items 0+1
        // (also cost 5); deterministic lexicographic tie-break chooses [0, 1].
        assert_eq!(result.selected_indices, vec![0, 1]);
    }
}
