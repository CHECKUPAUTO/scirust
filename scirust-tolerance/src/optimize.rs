//! Minimum-cost inertial tolerance synthesis under **several** functional
//! requirements at once (*tolérancement inertiel, calcul optimal*).
//!
//! [`crate::chain::allocate`] distributes a *single* assembly inertia budget in
//! closed form. Real products carry many functional requirements — several
//! dimension chains sharing the same components — and a manufacturing cost that
//! rises as a characteristic is tightened. This module solves the full problem:
//!
//! ```text
//! minimise   Σᵢ bᵢ · Iᵢ^(−rᵢ)                          (total tightening cost)
//! subject to Σᵢ αₖᵢ² Iᵢ² ≤ I_max,ₖ²   for each requirement k,
//!            Iᵢ > 0,
//! ```
//!
//! the standard reciprocal-power cost–tolerance model (Chase & Greenwood): a
//! per-component cost coefficient `bᵢ` (difficulty) and exponent `rᵢ`
//! (process sensitivity). In the squared variables `vᵢ = Iᵢ²` the cost is
//! convex and every constraint is **linear**, so this is a convex program with
//! strong duality.
//!
//! **Method.** The Lagrangian separates per component: given multipliers
//! `μₖ ≥ 0` and the shadow price `sᵢ = Σₖ μₖ αₖᵢ²`, stationarity gives the
//! closed form
//!
//! ```text
//! Iᵢ = ( (rᵢ/2)·bᵢ / sᵢ )^( 1/(rᵢ+2) ) .
//! ```
//!
//! The dual is maximised by a scale-free multiplicative update
//! `μₖ ← μₖ·(achievedₖ² / I_max,ₖ²)^γ`, whose fixed point is exactly the KKT
//! point: a binding constraint has `achievedₖ = I_max,ₖ`, a slack one has
//! `μₖ → 0` (complementary slackness). For a single requirement it reproduces
//! [`crate::chain::Allocation::CostOptimal`] in closed form.

use serde::{Deserialize, Serialize};

/// A component to be toleranced, with its reciprocal-power cost model
/// `C(I) = cost_coefficient · I^(−cost_exponent)` (larger coefficient ⇒ harder
/// to make; larger exponent ⇒ cost climbs faster as it is tightened).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Component {
    /// Human-readable name.
    pub name: String,
    /// Cost coefficient `bᵢ > 0`.
    pub cost_coefficient: f64,
    /// Cost exponent `rᵢ > 0`.
    pub cost_exponent: f64,
}

impl Component {
    /// A component with the given name and cost model.
    pub fn new(name: impl Into<String>, cost_coefficient: f64, cost_exponent: f64) -> Self {
        Self {
            name: name.into(),
            cost_coefficient,
            cost_exponent,
        }
    }

    /// The tightening cost `bᵢ · I^(−rᵢ)` at inertia `i`.
    pub fn cost(&self, i: f64) -> f64 {
        if i <= 0.0
        {
            return f64::INFINITY;
        }
        self.cost_coefficient * i.powf(-self.cost_exponent)
    }
}

/// A functional requirement: a linear dimension chain `Σᵢ αₖᵢ Xᵢ` whose
/// resultant inertia must stay within `i_max`. `coeffs[i]` is the influence
/// coefficient `αₖᵢ` of component `i` on this requirement (0 if it does not
/// participate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Requirement {
    /// Human-readable name.
    pub name: String,
    /// Influence coefficients `αₖᵢ`, one per component.
    pub coeffs: Vec<f64>,
    /// Maximum admissible resultant inertia `I_max,ₖ`.
    pub i_max: f64,
}

impl Requirement {
    /// A requirement with the given name, coefficients and inertia budget.
    pub fn new(name: impl Into<String>, coeffs: Vec<f64>, i_max: f64) -> Self {
        Self {
            name: name.into(),
            coeffs,
            i_max,
        }
    }

    /// The statistical resultant inertia `√(Σᵢ αₖᵢ² Iᵢ²)` for component
    /// inertias `inertias`.
    pub fn achieved(&self, inertias: &[f64]) -> f64 {
        self.coeffs
            .iter()
            .zip(inertias)
            .map(|(a, i)| a * a * i * i)
            .sum::<f64>()
            .sqrt()
    }
}

/// Result of [`optimize`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeResult {
    /// Optimal per-component inertias `Iᵢ`. Guaranteed feasible: every
    /// requirement's resultant inertia is within its budget (a final uniform
    /// tightening seats the worst constraint exactly on its budget even when
    /// the dual iteration did not fully converge).
    pub inertias: Vec<f64>,
    /// Total tightening cost `Σᵢ bᵢ Iᵢ^(−rᵢ)`.
    pub total_cost: f64,
    /// Optimal dual multipliers `μₖ` (shadow prices of the requirements).
    pub multipliers: Vec<f64>,
    /// Achieved resultant inertia per requirement.
    pub achieved: Vec<f64>,
    /// Whether each requirement is binding (active at the optimum).
    pub binding: Vec<bool>,
    /// Primal-infeasibility residual `maxₖ max(achievedₖ²/I_max,ₖ² − 1, 0)` at
    /// convergence — a quality measure, near 0 when solved (the closed-form
    /// primal keeps stationarity and dual feasibility exact by construction, so
    /// this is the binding KKT residual). `+∞` if the result was non-finite.
    pub kkt_residual: f64,
    /// Dual iterations performed.
    pub iterations: usize,
    /// Whether the dual iteration met the tolerance.
    pub converged: bool,
}

/// Error returned by [`optimize`].
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizeError {
    /// No components or no requirements were supplied.
    Empty,
    /// A requirement's coefficient count differs from the component count.
    CoeffCountMismatch {
        /// The requirement's index.
        requirement: usize,
        /// Coefficients supplied.
        got: usize,
        /// Components expected.
        expected: usize,
    },
    /// A component participates in no requirement (all coefficients zero), so
    /// its inertia is unbounded.
    UnconstrainedComponent(usize),
    /// A cost coefficient/exponent or an inertia budget was non-positive.
    NonPositiveParameter,
}

impl core::fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            OptimizeError::Empty => write!(f, "need at least one component and one requirement"),
            OptimizeError::CoeffCountMismatch {
                requirement,
                got,
                expected,
            } => write!(
                f,
                "requirement {requirement} has {got} coefficients but there are {expected} components"
            ),
            OptimizeError::UnconstrainedComponent(i) =>
            {
                write!(
                    f,
                    "component {i} participates in no requirement (unbounded inertia)"
                )
            },
            OptimizeError::NonPositiveParameter =>
            {
                write!(
                    f,
                    "cost coefficients, exponents and inertia budgets must be > 0"
                )
            },
        }
    }
}

impl std::error::Error for OptimizeError {}

/// Tuning for the dual ascent.
#[derive(Debug, Clone, Copy)]
pub struct OptimizeOptions {
    /// Maximum dual iterations.
    pub max_iters: usize,
    /// Convergence tolerance on the KKT residual.
    pub tol: f64,
    /// Multiplicative-update damping `γ ∈ (0, 1]`.
    pub damping: f64,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            max_iters: 20_000,
            tol: 1e-10,
            damping: 0.5,
        }
    }
}

/// Solve the minimum-cost multi-requirement inertia allocation with default
/// options. See the [module docs](mod@crate::optimize).
pub fn optimize(
    components: &[Component],
    requirements: &[Requirement],
) -> Result<OptimizeResult, OptimizeError> {
    optimize_with(components, requirements, OptimizeOptions::default())
}

/// Solve with explicit [`OptimizeOptions`].
pub fn optimize_with(
    components: &[Component],
    requirements: &[Requirement],
    opts: OptimizeOptions,
) -> Result<OptimizeResult, OptimizeError> {
    let n = components.len();
    let k = requirements.len();
    if n == 0 || k == 0
    {
        return Err(OptimizeError::Empty);
    }
    for (r, req) in requirements.iter().enumerate()
    {
        if req.coeffs.len() != n
        {
            return Err(OptimizeError::CoeffCountMismatch {
                requirement: r,
                got: req.coeffs.len(),
                expected: n,
            });
        }
        if req.i_max <= 0.0
        {
            return Err(OptimizeError::NonPositiveParameter);
        }
    }
    for c in components
    {
        if c.cost_coefficient <= 0.0 || c.cost_exponent <= 0.0
        {
            return Err(OptimizeError::NonPositiveParameter);
        }
    }
    // Every component must be reached by at least one requirement.
    for i in 0..n
    {
        let touched = requirements.iter().any(|req| req.coeffs[i] != 0.0);
        if !touched
        {
            return Err(OptimizeError::UnconstrainedComponent(i));
        }
    }

    // c_k = I_max,k², a_{k,i}² precomputed.
    let ck: Vec<f64> = requirements.iter().map(|r| r.i_max * r.i_max).collect();
    let a2: Vec<Vec<f64>> = requirements
        .iter()
        .map(|r| r.coeffs.iter().map(|a| a * a).collect())
        .collect();

    // Inertia of component i given the shadow price s_i = Σ_k μ_k a_{k,i}².
    let inertia_of = |i: usize, s_i: f64| -> f64 {
        let c = &components[i];
        let r = c.cost_exponent;
        ((r / 2.0) * c.cost_coefficient / s_i).powf(1.0 / (r + 2.0))
    };

    let mut mu = vec![1.0f64; k];
    let mut inertias = vec![0.0f64; n];
    let mut prev = vec![0.0f64; n];
    let mut iterations = 0;
    let mut kkt = f64::INFINITY;
    let mut converged = false;

    for it in 0..opts.max_iters
    {
        iterations = it + 1;
        // Primal from the current multipliers (closed-form stationarity).
        for i in 0..n
        {
            let s_i: f64 = (0..k).map(|kk| mu[kk] * a2[kk][i]).sum();
            // s_i > 0 is guaranteed: every component is touched and μ ≥ tiny.
            inertias[i] = inertia_of(i, s_i.max(f64::MIN_POSITIVE));
        }
        // Largest relative change in the primal since the previous iterate —
        // a scale-free fixed-point convergence measure.
        let mut max_change = 0.0f64;
        for i in 0..n
        {
            max_change =
                max_change.max((inertias[i] - prev[i]).abs() / (inertias[i].abs() + 1e-300));
        }
        // Multiplicative dual update; track primal feasibility (overshoot).
        let mut feas = 0.0f64;
        for kk in 0..k
        {
            let achieved2: f64 = (0..n).map(|i| a2[kk][i] * inertias[i] * inertias[i]).sum();
            let ratio = achieved2 / ck[kk];
            feas = feas.max((ratio - 1.0).max(0.0));
            mu[kk] = (mu[kk] * ratio.powf(opts.damping)).max(0.0);
        }
        kkt = feas;
        // Converged once the primal has stabilised and is feasible: a binding
        // constraint sits on ratio = 1, a slack one has driven its μ to ~0 so
        // it no longer moves the primal (complementary slackness). The
        // finiteness guard is essential: `f64::max` and `NaN.max(0.0)` silently
        // swallow NaN, so a non-finite iterate would otherwise read
        // `max_change = feas = 0` and falsely report convergence.
        if it > 0
            && max_change <= opts.tol
            && feas <= opts.tol
            && inertias.iter().all(|v| v.is_finite())
        {
            converged = true;
            break;
        }
        prev.copy_from_slice(&inertias);
    }

    // Feasibility safeguard. Rounding — or a run that hit `max_iters` on
    // ill-conditioned (near-parallel) constraints — can leave a requirement
    // marginally over budget. For tolerance allocation a feasible, slightly
    // costlier answer is strictly preferable to an infeasible one, so uniformly
    // tighten: scaling every Iᵢ by `f` scales every achievedₖ by `f`, and
    // `f = 1/maxₖ(achievedₖ/I_max,ₖ)` seats the worst constraint exactly on its
    // budget while leaving the (scale-invariant) relative allocation untouched.
    let max_ratio = requirements
        .iter()
        .map(|r| r.achieved(&inertias) / r.i_max)
        .fold(0.0_f64, f64::max);
    if max_ratio.is_finite() && max_ratio > 1.0
    {
        let f = 1.0 / max_ratio;
        for v in &mut inertias
        {
            *v *= f;
        }
    }

    let achieved: Vec<f64> = requirements.iter().map(|r| r.achieved(&inertias)).collect();
    let binding: Vec<bool> = requirements
        .iter()
        .zip(&achieved)
        .map(|(r, a)| *a >= r.i_max * (1.0 - 1e-6))
        .collect();
    let total_cost: f64 = components
        .iter()
        .zip(&inertias)
        .map(|(c, i)| c.cost(*i))
        .sum();

    // Never claim success on a non-finite result. This can only arise on
    // pathologically-scaled inputs (a shadow price × coefficient² overflowing
    // f64 — e.g. an `i_max` of ~1e-70), but if it does the `converged` flag
    // must not lie: report it as unconverged with an infinite residual so the
    // caller can react.
    if !total_cost.is_finite() || inertias.iter().any(|v| !v.is_finite())
    {
        converged = false;
        kkt = f64::INFINITY;
    }

    Ok(OptimizeResult {
        inertias,
        total_cost,
        multipliers: mu,
        achieved,
        binding,
        kkt_residual: kkt,
        iterations,
        converged,
    })
}

/// A point on the cost–quality trade-off frontier.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrontierPoint {
    /// Scale applied to every requirement's `i_max` (smaller ⇒ tighter).
    pub scale: f64,
    /// Minimum total cost achievable at this quality level.
    pub total_cost: f64,
    /// Whether the allocation converged at this scale.
    pub converged: bool,
}

/// Cost–quality Pareto frontier: for each `scale` in `scales`, re-solve with
/// every requirement's `i_max` multiplied by `scale`, returning the minimum
/// total cost. Cost is non-increasing in `scale` (looser ⇒ cheaper). Points
/// that fail to solve are skipped.
pub fn cost_quality_frontier(
    components: &[Component],
    requirements: &[Requirement],
    scales: &[f64],
) -> Vec<FrontierPoint> {
    scales
        .iter()
        .filter_map(|&scale| {
            if scale <= 0.0
            {
                return None;
            }
            let scaled: Vec<Requirement> = requirements
                .iter()
                .map(|r| Requirement::new(r.name.clone(), r.coeffs.clone(), r.i_max * scale))
                .collect();
            optimize(components, &scaled).ok().map(|res| FrontierPoint {
                scale,
                total_cost: res.total_cost,
                converged: res.converged,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{Allocation, allocate};
    use approx::assert_relative_eq;

    #[test]
    fn single_requirement_matches_closed_form_cost_optimal() {
        // One requirement ⇒ the optimizer must reproduce chain::CostOptimal.
        let r = 2.0;
        let costs = [1.0, 4.0, 2.0, 0.5];
        let coeffs = vec![1.0, -1.0, 2.0, -1.0];
        let i_max = 0.05;
        let comps: Vec<Component> = costs
            .iter()
            .enumerate()
            .map(|(i, &b)| Component::new(format!("X{i}"), b, r))
            .collect();
        let reqs = vec![Requirement::new("Y", coeffs.clone(), i_max)];

        let got = optimize(&comps, &reqs).unwrap();
        let closed = allocate(
            i_max,
            &coeffs,
            &Allocation::CostOptimal {
                costs: costs.to_vec(),
                exponent: r,
            },
        )
        .unwrap();
        for (a, b) in got.inertias.iter().zip(&closed)
        {
            assert_relative_eq!(a, b, epsilon = 1e-7, max_relative = 1e-7);
        }
        assert!(got.converged);
        assert!(got.binding[0]);
        assert_relative_eq!(got.achieved[0], i_max, epsilon = 1e-7);
    }

    #[test]
    fn uniform_single_requirement_is_the_statistical_allocation() {
        // Equal costs + exponents, α = ±1 ⇒ equal inertias = i_max/√n.
        let n = 5;
        let comps: Vec<Component> = (0..n)
            .map(|i| Component::new(format!("X{i}"), 1.0, 2.0))
            .collect();
        let coeffs = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let i_max = 0.1;
        let res = optimize(&comps, &[Requirement::new("Y", coeffs, i_max)]).unwrap();
        for i in &res.inertias
        {
            assert_relative_eq!(*i, i_max / (n as f64).sqrt(), epsilon = 1e-7);
        }
    }

    #[test]
    fn two_requirements_satisfy_kkt() {
        // Components 0,1,2 shared by two chains with different budgets.
        let comps = vec![
            Component::new("A", 1.0, 2.0),
            Component::new("B", 3.0, 2.0),
            Component::new("C", 2.0, 3.0),
        ];
        let reqs = vec![
            Requirement::new("Y1", vec![1.0, -1.0, 1.0], 0.06),
            Requirement::new("Y2", vec![1.0, 1.0, 0.0], 0.05),
        ];
        let res = optimize(&comps, &reqs).unwrap();
        assert!(res.converged, "kkt residual {}", res.kkt_residual);
        // Primal feasibility.
        for (a, r) in res.achieved.iter().zip(&reqs)
        {
            assert!(
                *a <= r.i_max * (1.0 + 1e-6),
                "req {} infeasible: {a}",
                r.name
            );
        }
        // At least one requirement binds (otherwise cost could still drop).
        assert!(res.binding.iter().any(|&b| b));
        // Complementary slackness: a non-binding requirement has ~zero multiplier.
        for (kk, &b) in res.binding.iter().enumerate()
        {
            if !b
            {
                assert!(
                    res.multipliers[kk] < 1e-6,
                    "slack req has μ={}",
                    res.multipliers[kk]
                );
            }
        }
    }

    #[test]
    fn optimizer_beats_a_naive_per_requirement_allocation() {
        // Baseline: satisfy each requirement independently (statistical),
        // then take the tightest inertia per component. Feasible but not optimal.
        let comps = vec![
            Component::new("A", 1.0, 2.0),
            Component::new("B", 5.0, 2.0),
            Component::new("C", 2.0, 2.0),
        ];
        let reqs = vec![
            Requirement::new("Y1", vec![1.0, -1.0, 1.0], 0.06),
            Requirement::new("Y2", vec![1.0, 1.0, 1.0], 0.05),
        ];
        let res = optimize(&comps, &reqs).unwrap();

        let mut naive = vec![f64::INFINITY; 3];
        for req in &reqs
        {
            let a = allocate(
                req.i_max,
                &req.coeffs,
                &Allocation::CostOptimal {
                    costs: comps.iter().map(|c| c.cost_coefficient).collect(),
                    exponent: 2.0,
                },
            )
            .unwrap();
            for (n, v) in naive.iter_mut().zip(a)
            {
                *n = n.min(v);
            }
        }
        let naive_cost: f64 = comps.iter().zip(&naive).map(|(c, i)| c.cost(*i)).sum();
        assert!(
            res.total_cost <= naive_cost + 1e-9,
            "optimizer {} should not exceed naive {}",
            res.total_cost,
            naive_cost
        );
    }

    #[test]
    fn never_reports_convergence_on_a_non_finite_result() {
        // Pathologically-scaled input (a shadow price × coefficient² can
        // overflow f64): the answer may be non-finite, but `converged` must not
        // lie — NaN/Inf must never masquerade as a solved optimum.
        let comps = vec![Component::new("A", 1.0, 2.0), Component::new("B", 1.0, 2.0)];
        let reqs = vec![Requirement::new("Y", vec![1.0, 1e7], 1e-74)];
        let res = optimize(&comps, &reqs).unwrap();
        if !res.total_cost.is_finite() || res.inertias.iter().any(|v| !v.is_finite())
        {
            assert!(
                !res.converged,
                "non-finite result must not report converged"
            );
            assert!(res.kkt_residual.is_infinite());
        }
        // A finite result must be genuinely feasible.
        if res.converged
        {
            assert!(res.inertias.iter().all(|v| v.is_finite() && *v > 0.0));
            assert!(res.total_cost.is_finite());
        }
    }

    #[test]
    fn result_is_always_feasible_even_when_ill_conditioned() {
        // Two nearly-parallel constraints with slightly different budgets — an
        // ill-conditioned instance where the dual can stall. The feasibility
        // safeguard must still return an allocation inside every budget.
        let comps = vec![
            Component::new("A", 1.0, 2.0),
            Component::new("B", 1.0, 2.0),
            Component::new("C", 2.0, 3.0),
        ];
        let reqs = vec![
            Requirement::new("Y1", vec![1.0, 1.000_000_1, 0.5], 0.050),
            Requirement::new("Y2", vec![1.0, 1.0, 0.5], 0.050_01),
        ];
        let res = optimize(&comps, &reqs).unwrap();
        for (a, r) in res.achieved.iter().zip(&reqs)
        {
            assert!(
                *a <= r.i_max * (1.0 + 1e-9),
                "{} infeasible: achieved {a} > budget {}",
                r.name,
                r.i_max
            );
        }
        assert!(res.inertias.iter().all(|i| i.is_finite() && *i > 0.0));
    }

    #[test]
    fn cost_quality_frontier_is_monotone() {
        let comps = vec![Component::new("A", 1.0, 2.0), Component::new("B", 2.0, 2.0)];
        let reqs = vec![Requirement::new("Y", vec![1.0, -1.0], 0.05)];
        let scales = [0.5, 0.75, 1.0, 1.5, 2.0];
        let front = cost_quality_frontier(&comps, &reqs, &scales);
        assert_eq!(front.len(), scales.len());
        // Looser tolerance (larger scale) is never more expensive.
        for w in front.windows(2)
        {
            assert!(w[1].total_cost <= w[0].total_cost + 1e-9);
        }
    }

    #[test]
    fn rejects_invalid_input() {
        let comps = vec![Component::new("A", 1.0, 2.0)];
        assert_eq!(optimize(&comps, &[]).unwrap_err(), OptimizeError::Empty);
        assert_eq!(
            optimize(&comps, &[Requirement::new("Y", vec![1.0, 2.0], 0.1)]).unwrap_err(),
            OptimizeError::CoeffCountMismatch {
                requirement: 0,
                got: 2,
                expected: 1
            }
        );
        assert_eq!(
            optimize(&comps, &[Requirement::new("Y", vec![0.0], 0.1)]).unwrap_err(),
            OptimizeError::UnconstrainedComponent(0)
        );
    }
}
