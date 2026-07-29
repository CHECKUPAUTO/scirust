//! **Structural causal models**: intervention simulation (Pearl's rung 2) and
//! counterfactual queries (rung 3).
//!
//! # The three rungs, and where this crate sits on them
//!
//! - **Rung 1, association** — `P(Y | X)`. `crate::conditional_independence`.
//! - **Rung 2, intervention** — `P(Y | do(X))`. `crate::effect_estimation`
//!   *identifies* this from a graph plus observational data;
//!   [`LinearScm::simulate`] *computes* it directly when the model is known.
//! - **Rung 3, counterfactual** — "given that we observed this world, what
//!   would `Y` have been had `X` been different?" This is
//!   [`LinearScm::counterfactual`], and it is **strictly stronger** than rung
//!   2: no amount of interventional data determines it.
//!
//! # Abduction–action–prediction
//!
//! A counterfactual is computed in Pearl's three steps:
//!
//! 1. **Abduction** — recover the exogenous noise from the observed world.
//!    For an additive-noise linear model this is *exact* and needs no
//!    inversion: `ε_i = x_i − c_i − Σ_j B[i,j]·x_j`, read straight off the
//!    structural equations. It does require the factual world to be **fully
//!    observed**; a partially observed factual leaves the noise underdetermined
//!    and [`LinearScm::counterfactual`] rejects it rather than imputing.
//! 2. **Action** — sever each intervened variable from its parents and fix its
//!    value, giving the mutilated model.
//! 3. **Prediction** — recompute every variable in topological order, holding
//!    the abducted noise fixed. Holding the *same* noise is what makes this a
//!    counterfactual about the observed unit rather than a fresh draw.
//!
//! # The honesty problem specific to rung 3
//!
//! Given the SCM, a counterfactual is **deterministic** — abduction is exact,
//! so [`CounterfactualOutcome`]'s certificate reports zero *computational*
//! uncertainty. That number is not a claim of confidence. All the epistemic
//! risk sits in the SCM itself, and **the SCM is not identified by data**.
//!
//! The crate's tests make that concrete: two SCMs, `X → Y` and `Y → X`, with
//! coefficients chosen so their joint distributions are provably *identical* —
//! indistinguishable by any observational test, and exactly the ambiguity
//! `crate::PcStable` reports as an undirected edge — return **different**
//! counterfactual answers to the same query (`3` versus `1`). Choosing the
//! wrong one is not a statistical error that more data would fix; it is a
//! modelling commitment, and this module requires the caller to make it
//! explicitly rather than inferring one.

use crate::assumptions::CausalAssumption;
use crate::certificate::{CausalCertificate, IdentifiabilityStatus};
use crate::error::CausalError;
use scirust_graph::dag::CausalDag;
use scirust_solvers::Matrix;
use std::collections::BTreeMap;

/// A linear additive-noise structural causal model:
/// `X_i = c_i + Σ_j B[i,j]·X_j + ε_i`.
///
/// `B[i,j] != 0` means `j` is a direct cause of `i`. The induced graph must be
/// acyclic; the variables need **not** be given in topological order (the
/// order is computed and stored at construction).
#[derive(Debug, Clone, PartialEq)]
pub struct LinearScm {
    n_variables: usize,
    coefficients: Matrix,
    intercepts: Vec<f64>,
    topological_order: Vec<usize>,
}

impl LinearScm {
    /// Validates and constructs a model.
    ///
    /// A coefficient is treated as an edge iff it is exactly non-zero — no
    /// thresholding, so the induced graph is a deterministic function of the
    /// matrix a caller supplied rather than of a tolerance they did not.
    ///
    /// # Errors
    ///
    /// [`CausalError::NotSquare`] for a non-square coefficient matrix;
    /// [`CausalError::DimensionMismatch`] if `intercepts` has the wrong
    /// length; [`CausalError::NonFiniteWeight`] / [`CausalError::NonFiniteInput`]
    /// for a non-finite coefficient or intercept; [`CausalError::CyclicGraph`]
    /// if the induced graph has a directed cycle (including a self-loop, which
    /// a non-zero diagonal entry would be).
    pub fn new(coefficients: Matrix, intercepts: Vec<f64>) -> Result<Self, CausalError> {
        let (rows, cols) = coefficients.shape();
        if rows != cols
        {
            return Err(CausalError::NotSquare { rows, cols });
        }
        if intercepts.len() != rows
        {
            return Err(CausalError::DimensionMismatch {
                expected: rows,
                got: intercepts.len(),
            });
        }
        for (index, &value) in intercepts.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(CausalError::NonFiniteInput { index, value });
            }
        }
        for row in 0..rows
        {
            for col in 0..cols
            {
                let value = coefficients[(row, col)];
                if !value.is_finite()
                {
                    return Err(CausalError::NonFiniteWeight { row, col, value });
                }
            }
        }

        // The induced graph, which also decides acyclicity: `add_directed_edge`
        // rejects both cycles and self-loops.
        let mut dag = CausalDag::new(rows);
        for row in 0..rows
        {
            for col in 0..cols
            {
                if coefficients[(row, col)] != 0.0
                {
                    dag.add_directed_edge(col, row)
                        .map_err(|_| CausalError::CyclicGraph)?;
                }
            }
        }
        let topological_order = dag.topo_order().map_err(|_| CausalError::CyclicGraph)?;

        Ok(Self {
            n_variables: rows,
            coefficients,
            intercepts,
            topological_order,
        })
    }

    #[must_use]
    pub fn n_variables(&self) -> usize {
        self.n_variables
    }

    /// The graph this model induces: an edge `j -> i` for every non-zero
    /// `B[i,j]`.
    #[must_use]
    pub fn to_dag(&self) -> CausalDag {
        let mut dag = CausalDag::new(self.n_variables);
        for row in 0..self.n_variables
        {
            for col in 0..self.n_variables
            {
                if self.coefficients[(row, col)] != 0.0
                {
                    dag.add_directed_edge(col, row)
                        .expect("acyclicity was validated at construction");
                }
            }
        }
        dag
    }

    fn validate_interventions(
        &self,
        interventions: &[InterventionAssignment],
    ) -> Result<BTreeMap<usize, f64>, CausalError> {
        let mut fixed = BTreeMap::new();
        for assignment in interventions
        {
            if assignment.target >= self.n_variables
            {
                return Err(CausalError::UnknownVariableIndex {
                    index: assignment.target,
                });
            }
            if !assignment.value.is_finite()
            {
                return Err(CausalError::NonFiniteInput {
                    index: assignment.target,
                    value: assignment.value,
                });
            }
            if fixed.insert(assignment.target, assignment.value).is_some()
            {
                return Err(CausalError::InvalidContract {
                    detail: "the same variable cannot be intervened on twice",
                });
            }
        }
        Ok(fixed)
    }

    /// **Rung 2.** Computes every variable's value from exogenous `noise`,
    /// under `interventions` (each `do(X_k = v)` severs `X_k` from its parents
    /// and from its own noise term).
    ///
    /// # Errors
    ///
    /// [`CausalError::DimensionMismatch`] if `noise` has the wrong length;
    /// [`CausalError::NonFiniteInput`] for a non-finite noise value or
    /// intervention value; [`CausalError::UnknownVariableIndex`] for an
    /// out-of-range target; [`CausalError::InvalidContract`] for a repeated
    /// target; [`CausalError::NonFiniteComputation`] if the propagation
    /// overflows.
    pub fn simulate(
        &self,
        noise: &[f64],
        interventions: &[InterventionAssignment],
    ) -> Result<Vec<f64>, CausalError> {
        if noise.len() != self.n_variables
        {
            return Err(CausalError::DimensionMismatch {
                expected: self.n_variables,
                got: noise.len(),
            });
        }
        for (index, &value) in noise.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(CausalError::NonFiniteInput { index, value });
            }
        }
        let fixed = self.validate_interventions(interventions)?;

        let mut values = vec![0.0; self.n_variables];
        for &variable in &self.topological_order
        {
            if let Some(&forced) = fixed.get(&variable)
            {
                values[variable] = forced;
                continue;
            }
            let mut value = self.intercepts[variable] + noise[variable];
            for (parent, &parent_value) in values.iter().enumerate()
            {
                let coefficient = self.coefficients[(variable, parent)];
                if coefficient != 0.0
                {
                    value += coefficient * parent_value;
                }
            }
            if !value.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "scm_simulate",
                    index: variable,
                    value,
                });
            }
            values[variable] = value;
        }
        Ok(values)
    }

    /// **Abduction.** Recovers the exogenous noise that must have produced
    /// `observation`. Exact for an additive-noise linear model, and requires
    /// the world to be **fully** observed — every variable's noise is read off
    /// its own structural equation.
    ///
    /// # Errors
    ///
    /// [`CausalError::DimensionMismatch`] if `observation` has the wrong
    /// length; [`CausalError::NonFiniteInput`] for a non-finite observation;
    /// [`CausalError::NonFiniteComputation`] if the arithmetic overflows.
    pub fn abduct(&self, observation: &[f64]) -> Result<Vec<f64>, CausalError> {
        if observation.len() != self.n_variables
        {
            return Err(CausalError::DimensionMismatch {
                expected: self.n_variables,
                got: observation.len(),
            });
        }
        for (index, &value) in observation.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(CausalError::NonFiniteInput { index, value });
            }
        }

        let mut noise = vec![0.0; self.n_variables];
        for variable in 0..self.n_variables
        {
            let mut residual = observation[variable] - self.intercepts[variable];
            for (parent, &parent_value) in observation.iter().enumerate()
            {
                let coefficient = self.coefficients[(variable, parent)];
                if coefficient != 0.0
                {
                    residual -= coefficient * parent_value;
                }
            }
            if !residual.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "scm_abduct",
                    index: variable,
                    value: residual,
                });
            }
            noise[variable] = residual;
        }
        Ok(noise)
    }

    /// **Rung 3.** Answers "given that we observed `query.factual`, what would
    /// the world have been had `query.interventions` held instead?" via
    /// abduction, action, and prediction — see the module docs.
    ///
    /// # Errors
    ///
    /// As [`LinearScm::abduct`] and [`LinearScm::simulate`], plus
    /// [`CausalError::UnknownVariableIndex`] if `query.outcome` is out of
    /// range, and [`CausalError::InvalidContract`] if the certificate cannot
    /// be built.
    pub fn counterfactual(
        &self,
        query: &CounterfactualQuery,
    ) -> Result<CounterfactualOutcome, CausalError> {
        if query.outcome >= self.n_variables
        {
            return Err(CausalError::UnknownVariableIndex {
                index: query.outcome,
            });
        }

        // 1. Abduction: what noise must have produced the world we saw?
        let recovered_noise = self.abduct(&query.factual)?;
        // 2 & 3. Action and prediction: same noise, mutilated model.
        let counterfactual = self.simulate(&recovered_noise, &query.interventions)?;

        let mut warnings = Vec::new();
        if query.interventions.is_empty()
        {
            warnings.push(
                "no interventions were supplied, so the counterfactual world is by construction \
                 the factual one"
                    .to_string(),
            );
        }
        for assignment in &query.interventions
        {
            if assignment.target == query.outcome
            {
                warnings.push(format!(
                    "the outcome variable {} is itself intervened on, so its counterfactual \
                     value is the value that was set, not a consequence of the model",
                    assignment.target
                ));
            }
        }

        let certificate = CausalCertificate::builder(
            format!(
                "counterfactual value of variable {} under {:?}, given the observed world",
                query.outcome, query.interventions
            ),
            IdentifiabilityStatus::Identifiable,
            vec![
                CausalAssumption::Acyclicity,
                CausalAssumption::CausalSufficiency,
                CausalAssumption::CorrectFunctionalForm,
                CausalAssumption::Other(
                    "the supplied structural causal model is the true one".to_string(),
                ),
            ],
            format!(
                "abduction-action-prediction on a fully observed {}-variable factual world",
                self.n_variables
            ),
        )
        .with_estimate(
            "structural counterfactual (abduction, action, prediction)",
            counterfactual[query.outcome],
            // Zero *computational* uncertainty: abduction is exact and the
            // prediction is deterministic. See the sensitivity note -- all the
            // epistemic risk is in the model, not in this arithmetic.
            0.0,
        )
        .with_sensitivity(
            "identified only RELATIVE to the supplied structural causal model, which is assumed \
             and is not identified by data: a Markov-equivalent model fitting the same \
             distribution can give a different counterfactual answer, as the crate's own tests \
             demonstrate. The reported uncertainty of zero is computational, not epistemic",
        )
        .with_unresolved_alternative(
            "any other structural causal model consistent with the same observational \
             distribution would answer this query differently",
        )
        .finalize()?;

        Ok(CounterfactualOutcome {
            outcome: query.outcome,
            factual: query.factual.clone(),
            counterfactual_world: counterfactual,
            recovered_noise,
            interventions: query.interventions.clone(),
            certificate,
            warnings,
        })
    }
}

/// A single `do(X_target = value)`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InterventionAssignment {
    pub target: usize,
    pub value: f64,
}

impl InterventionAssignment {
    #[must_use]
    pub fn new(target: usize, value: f64) -> Self {
        Self { target, value }
    }
}

/// A rung-3 query: a fully observed factual world, the interventions to
/// impose instead, and which variable's counterfactual value is being asked
/// about.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CounterfactualQuery {
    pub factual: Vec<f64>,
    pub interventions: Vec<InterventionAssignment>,
    pub outcome: usize,
}

/// One counterfactual query's full, reproducible outcome.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CounterfactualOutcome {
    pub outcome: usize,
    pub factual: Vec<f64>,
    /// Every variable's value in the counterfactual world.
    pub counterfactual_world: Vec<f64>,
    /// The exogenous noise abduction recovered from the factual world.
    pub recovered_noise: Vec<f64>,
    pub interventions: Vec<InterventionAssignment>,
    pub certificate: CausalCertificate,
    pub warnings: Vec<String>,
}

impl CounterfactualOutcome {
    /// The counterfactual value of the queried outcome variable.
    #[must_use]
    pub fn counterfactual_value(&self) -> f64 {
        self.counterfactual_world[self.outcome]
    }

    /// The factual value of the queried outcome variable.
    #[must_use]
    pub fn factual_value(&self) -> f64 {
        self.factual[self.outcome]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `X0 -> X1 -> X2` with unit coefficients and zero intercepts.
    fn chain() -> LinearScm {
        let coefficients = Matrix::from_row_major(
            3,
            3,
            vec![
                0.0, 0.0, 0.0, //
                2.0, 0.0, 0.0, //
                0.0, 3.0, 0.0,
            ],
        );
        LinearScm::new(coefficients, vec![0.0; 3]).unwrap()
    }

    #[test]
    fn simulation_propagates_through_the_chain() {
        let scm = chain();
        // noise (1, 0, 0): X0 = 1, X1 = 2*1 = 2, X2 = 3*2 = 6.
        let values = scm.simulate(&[1.0, 0.0, 0.0], &[]).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 6.0]);
    }

    #[test]
    fn an_intervention_severs_a_variable_from_its_parents_and_its_noise() {
        let scm = chain();
        // do(X1 = 10) with the same noise: X0 still 1, X1 forced to 10 (its
        // own noise term is discarded too), X2 = 3*10 = 30.
        let values = scm
            .simulate(&[1.0, 99.0, 0.0], &[InterventionAssignment::new(1, 10.0)])
            .unwrap();
        assert_eq!(values, vec![1.0, 10.0, 30.0]);
    }

    #[test]
    fn intervening_downstream_leaves_upstream_untouched() {
        let scm = chain();
        let values = scm
            .simulate(&[1.0, 0.0, 0.0], &[InterventionAssignment::new(2, -5.0)])
            .unwrap();
        assert_eq!(values, vec![1.0, 2.0, -5.0]);
    }

    #[test]
    fn abduction_inverts_simulation_exactly() {
        let scm = chain();
        let noise = vec![0.25, -1.5, 3.75];
        let world = scm.simulate(&noise, &[]).unwrap();
        let recovered = scm.abduct(&world).unwrap();
        for (original, recovered) in noise.iter().zip(&recovered)
        {
            assert!(
                (original - recovered).abs() < 1e-12,
                "abduction must be exact: {original} vs {recovered}"
            );
        }
    }

    #[test]
    fn a_counterfactual_with_no_intervention_reproduces_the_factual_world() {
        let scm = chain();
        let factual = scm.simulate(&[0.5, 0.25, -1.0], &[]).unwrap();
        let outcome = scm
            .counterfactual(&CounterfactualQuery {
                factual: factual.clone(),
                interventions: vec![],
                outcome: 2,
            })
            .unwrap();
        for (a, b) in factual.iter().zip(&outcome.counterfactual_world)
        {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(!outcome.warnings.is_empty(), "the no-op should be flagged");
    }

    #[test]
    fn the_variables_are_not_required_in_topological_order() {
        // X1 -> X0: the coefficient matrix is *upper* triangular, which a
        // triangularity requirement would reject. Acyclicity is what matters.
        let coefficients = Matrix::from_row_major(2, 2, vec![0.0, 4.0, 0.0, 0.0]);
        let scm = LinearScm::new(coefficients, vec![0.0, 0.0]).unwrap();
        let values = scm.simulate(&[1.0, 2.0], &[]).unwrap();
        // X1 = 2 (root), X0 = 4*2 + 1 = 9.
        assert_eq!(values, vec![9.0, 2.0]);
    }

    #[test]
    fn a_cycle_is_rejected_at_construction() {
        let coefficients = Matrix::from_row_major(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        assert!(matches!(
            LinearScm::new(coefficients, vec![0.0, 0.0]),
            Err(CausalError::CyclicGraph)
        ));
    }

    #[test]
    fn a_self_loop_is_rejected_at_construction() {
        let coefficients = Matrix::from_row_major(2, 2, vec![0.5, 0.0, 0.0, 0.0]);
        assert!(matches!(
            LinearScm::new(coefficients, vec![0.0, 0.0]),
            Err(CausalError::CyclicGraph)
        ));
    }

    #[test]
    fn malformed_models_and_queries_are_typed_errors() {
        assert!(matches!(
            LinearScm::new(Matrix::from_row_major(2, 3, vec![0.0; 6]), vec![0.0; 2]),
            Err(CausalError::NotSquare { rows: 2, cols: 3 })
        ));
        assert!(matches!(
            LinearScm::new(Matrix::from_row_major(2, 2, vec![0.0; 4]), vec![0.0; 3]),
            Err(CausalError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
        assert!(matches!(
            LinearScm::new(
                Matrix::from_row_major(2, 2, vec![0.0, 0.0, f64::NAN, 0.0]),
                vec![0.0; 2]
            ),
            Err(CausalError::NonFiniteWeight { .. })
        ));

        let scm = chain();
        assert!(matches!(
            scm.simulate(&[1.0, 2.0], &[]),
            Err(CausalError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
        assert!(matches!(
            scm.simulate(&[1.0; 3], &[InterventionAssignment::new(9, 0.0)]),
            Err(CausalError::UnknownVariableIndex { index: 9 })
        ));
        assert!(matches!(
            scm.simulate(
                &[1.0; 3],
                &[
                    InterventionAssignment::new(1, 0.0),
                    InterventionAssignment::new(1, 1.0)
                ]
            ),
            Err(CausalError::InvalidContract { .. })
        ));
        assert!(matches!(
            scm.counterfactual(&CounterfactualQuery {
                factual: vec![0.0; 3],
                interventions: vec![],
                outcome: 9,
            }),
            Err(CausalError::UnknownVariableIndex { index: 9 })
        ));
    }

    #[test]
    fn the_induced_graph_matches_the_coefficient_pattern() {
        let scm = chain();
        let dag = scm.to_dag();
        assert_eq!(dag.n_edges(), 2);
        assert!(dag.can_reach(0, 2));
        assert!(!dag.can_reach(2, 0));
    }
}
