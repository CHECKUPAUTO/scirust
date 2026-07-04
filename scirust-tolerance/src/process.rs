//! Discrete-process tolerance allocation.
//!
//! [`optimize`](mod@crate::optimize) allocates inertia on a *continuous*
//! cost–tolerance curve.
//! A shop floor instead offers a **discrete menu**: each component can be made
//! by one of several processes, each with its own achievable inertia `Iᵢⱼ` and
//! unit cost `cᵢⱼ` (a finer process costs more but holds a tighter inertia). The
//! task is to pick **one process per component** that minimises total cost while
//! the assembled inertia still meets a budget:
//!
//! ```text
//! minimise Σᵢ cᵢ,sel(i)   subject to   I_Y(sel) ≤ budget ,
//! ```
//!
//! with `I_Y` the statistical `√(Σ αᵢ² Iᵢ²)` or the worst-case `Σ |αᵢ| Iᵢ`
//! ([`Combination`]). This is a **multiple-choice knapsack** problem. Because the
//! per-component "weights" (`αᵢ² Iᵢⱼ²` or `|αᵢ| Iᵢⱼ`) are real-valued, it is
//! solved **exactly** by carrying the Pareto frontier of non-dominated
//! `(weight, cost)` partial selections across the components and reading off the
//! cheapest whose total weight is within budget — validated in `fuzz_crosscheck`
//! against exhaustive enumeration.

use serde::{Deserialize, Serialize};

/// A candidate manufacturing process for a component: the inertia it holds and
/// what it costs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProcessOption {
    /// Inertia `Iᵢⱼ` achievable by this process.
    pub inertia: f64,
    /// Cost `cᵢⱼ` of this process (per part).
    pub cost: f64,
}

impl ProcessOption {
    /// A process option with the given inertia and cost.
    pub fn new(inertia: f64, cost: f64) -> Self {
        Self { inertia, cost }
    }
}

/// How the component inertias combine into the assembly inertia constrained by
/// the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Combination {
    /// Statistical `I_Y = √(Σ αᵢ² Iᵢ²)`.
    Statistical,
    /// Worst-case `I_Y = Σ |αᵢ| Iᵢ`.
    WorstCase,
}

/// The chosen process for each component and the resulting cost / inertia.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessAllocation {
    /// `selection[i]` is the index into `options[i]` of the chosen process.
    pub selection: Vec<usize>,
    /// Total cost `Σᵢ cᵢ,sel(i)` — minimal among all budget-feasible selections.
    pub total_cost: f64,
    /// Achieved assembly inertia `I_Y(selection)` (`≤ budget`).
    pub assembly_inertia: f64,
}

#[derive(Clone)]
struct State {
    weight: f64,
    cost: f64,
    sel: Vec<usize>,
}

/// Drop Pareto-dominated states: keep only selections that are not beaten on
/// **both** accumulated weight and cost. Since every remaining component adds
/// non-negative weight and cost, a dominated partial selection can never lead to
/// a cheaper feasible whole, so pruning is exact.
fn prune(mut states: Vec<State>) -> Vec<State> {
    states.sort_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.cost
                    .partial_cmp(&b.cost)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    let mut out: Vec<State> = Vec::new();
    let mut best_cost = f64::INFINITY;
    for s in states
    {
        // At a weight ≥ every kept state's, keep only if strictly cheaper.
        if s.cost < best_cost - 1e-15
        {
            best_cost = s.cost;
            out.push(s);
        }
    }
    out
}

/// Minimum-cost discrete-process allocation. `coeffs[i]` is the influence
/// coefficient `αᵢ`; `options[i]` is component `i`'s process menu. Returns the
/// cheapest [`ProcessAllocation`] whose assembly inertia (per `method`) is within
/// `budget`, or `None` if no selection can meet the budget (or the inputs are
/// empty / mismatched).
pub fn allocate_discrete(
    coeffs: &[f64],
    options: &[Vec<ProcessOption>],
    budget: f64,
    method: Combination,
) -> Option<ProcessAllocation> {
    let n = coeffs.len();
    if n == 0 || options.len() != n || options.iter().any(|o| o.is_empty())
    {
        return None;
    }
    // Weight of choosing option j for component i, additive in the constraint.
    let weight = |i: usize, opt: &ProcessOption| match method
    {
        Combination::Statistical => coeffs[i] * coeffs[i] * opt.inertia * opt.inertia,
        Combination::WorstCase => coeffs[i].abs() * opt.inertia,
    };
    let limit = match method
    {
        Combination::Statistical => budget * budget,
        Combination::WorstCase => budget,
    };

    let mut states = vec![State {
        weight: 0.0,
        cost: 0.0,
        sel: Vec::with_capacity(n),
    }];
    for (i, menu) in options.iter().enumerate()
    {
        let mut next = Vec::with_capacity(states.len() * menu.len());
        for s in &states
        {
            for (j, opt) in menu.iter().enumerate()
            {
                let mut sel = s.sel.clone();
                sel.push(j);
                next.push(State {
                    weight: s.weight + weight(i, opt),
                    cost: s.cost + opt.cost,
                    sel,
                });
            }
        }
        states = prune(next);
    }

    // Cheapest state within the budget.
    let best = states
        .into_iter()
        .filter(|s| s.weight <= limit * (1.0 + 1e-12))
        .min_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let assembly_inertia = match method
    {
        Combination::Statistical => best.weight.max(0.0).sqrt(),
        Combination::WorstCase => best.weight,
    };
    Some(ProcessAllocation {
        selection: best.sel,
        total_cost: best.cost,
        assembly_inertia,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn menu(pairs: &[(f64, f64)]) -> Vec<ProcessOption> {
        pairs
            .iter()
            .map(|&(i, c)| ProcessOption::new(i, c))
            .collect()
    }

    #[test]
    fn picks_cheapest_feasible_processes() {
        // Two components, ±1 stack. Each: coarse (I=0.10, cost 1) or fine
        // (I=0.05, cost 3). Budget on worst-case Σ I.
        let coeffs = [1.0, -1.0];
        let opts = vec![
            menu(&[(0.10, 1.0), (0.05, 3.0)]),
            menu(&[(0.10, 1.0), (0.05, 3.0)]),
        ];
        // Budget 0.20 ⇒ both coarse (0.10+0.10=0.20), cost 2.
        let a = allocate_discrete(&coeffs, &opts, 0.20, Combination::WorstCase).unwrap();
        assert_eq!(a.selection, vec![0, 0]);
        assert_relative_eq!(a.total_cost, 2.0, epsilon = 1e-12);
        assert_relative_eq!(a.assembly_inertia, 0.20, epsilon = 1e-12);
        // Budget 0.15 ⇒ one must go fine (0.10+0.05=0.15), cost 1+3=4.
        let b = allocate_discrete(&coeffs, &opts, 0.15, Combination::WorstCase).unwrap();
        assert_relative_eq!(b.total_cost, 4.0, epsilon = 1e-12);
        assert_relative_eq!(b.assembly_inertia, 0.15, epsilon = 1e-12);
    }

    #[test]
    fn statistical_budget_is_looser_than_worst_case() {
        let coeffs = [1.0, -1.0];
        let opts = vec![
            menu(&[(0.10, 1.0), (0.05, 3.0)]),
            menu(&[(0.10, 1.0), (0.05, 3.0)]),
        ];
        // Statistical of two coarse: √(0.01+0.01)=0.1414 ≤ 0.15 ⇒ both coarse ok.
        let a = allocate_discrete(&coeffs, &opts, 0.15, Combination::Statistical).unwrap();
        assert_eq!(a.selection, vec![0, 0]);
        assert_relative_eq!(a.total_cost, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn infeasible_when_even_finest_exceeds_budget() {
        let coeffs = [1.0, -1.0];
        let opts = vec![
            menu(&[(0.10, 1.0), (0.05, 3.0)]),
            menu(&[(0.10, 1.0), (0.05, 3.0)]),
        ];
        // Finest worst-case = 0.05+0.05 = 0.10 > 0.09 ⇒ infeasible.
        assert!(allocate_discrete(&coeffs, &opts, 0.09, Combination::WorstCase).is_none());
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(allocate_discrete(&[], &[], 1.0, Combination::WorstCase).is_none());
        assert!(allocate_discrete(&[1.0], &[vec![]], 1.0, Combination::WorstCase).is_none());
        assert!(
            allocate_discrete(
                &[1.0, 1.0],
                &[menu(&[(0.1, 1.0)])],
                1.0,
                Combination::WorstCase
            )
            .is_none()
        );
    }

    #[test]
    fn matches_brute_force_on_a_small_instance() {
        let coeffs = [1.0, -1.5, 0.5];
        let opts = vec![
            menu(&[(0.12, 1.0), (0.08, 2.5), (0.05, 5.0)]),
            menu(&[(0.10, 1.2), (0.06, 3.0)]),
            menu(&[(0.20, 0.5), (0.10, 2.0), (0.05, 4.0)]),
        ];
        let budget = 0.18;
        let got = allocate_discrete(&coeffs, &opts, budget, Combination::Statistical);
        // Brute force.
        let mut best: Option<(f64, Vec<usize>)> = None;
        for a in 0..opts[0].len()
        {
            for b in 0..opts[1].len()
            {
                for c in 0..opts[2].len()
                {
                    let sel = [a, b, c];
                    let iy2: f64 = sel
                        .iter()
                        .enumerate()
                        .map(|(i, &j)| coeffs[i].powi(2) * opts[i][j].inertia.powi(2))
                        .sum();
                    if iy2.sqrt() <= budget
                    {
                        let cost: f64 = sel.iter().enumerate().map(|(i, &j)| opts[i][j].cost).sum();
                        if best.as_ref().map(|(bc, _)| cost < *bc).unwrap_or(true)
                        {
                            best = Some((cost, sel.to_vec()));
                        }
                    }
                }
            }
        }
        let (bc, _) = best.unwrap();
        assert_relative_eq!(got.unwrap().total_cost, bc, epsilon = 1e-12);
    }
}
