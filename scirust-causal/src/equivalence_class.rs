//! Constraint-based causal-structure discovery: **PC-Stable**, returning a
//! [`Cpdag`] (a Markov-equivalence-class representative — directed edges are
//! *compelled*, undirected edges are *reversible*) instead of a single
//! hypothesis DAG.
//!
//! # What this is, in one paragraph
//!
//! [`PcStable`] repeatedly calls a [`ConditionalIndependenceTest`] oracle
//! (`crate::conditional_independence`, phase 5C.2) to discover which pairs of
//! variables can be separated by some conditioning set (the *skeleton*),
//! then reads off the unshielded colliders the separating sets imply (the
//! *v-structures*), then propagates those orientations as far as the
//! evidence logically forces via Meek's rules — see `crate::skeleton_discovery`
//! and `crate::orientation` for the two stages' own detailed docs. This is a
//! **different, additive discovery paradigm** from the crate's existing
//! continuous-optimization structure learner ([`crate::optimize_causal`]):
//! that one is score-based and returns a single fully-directed hypothesis
//! DAG; this one is constraint-based and returns a CPDAG that honestly marks
//! which edges the evidence cannot orient. Neither calls the other; neither
//! supersedes the other.
//!
//! # What a [`Cpdag`] from this procedure claims, and what it does not
//!
//! Under three assumptions this crate already has named variants for
//! ([`CausalAssumption::Acyclicity`], [`CausalAssumption::CausalSufficiency`],
//! [`CausalAssumption::Faithfulness`] — see [`EquivalenceClassResult::assumptions`],
//! which always includes exactly these three, unioned with whatever the
//! underlying CI-testing method itself assumed) — a correct implementation of
//! this procedure, given a perfect independence oracle, recovers the *exact*
//! Markov equivalence class of the true causal DAG: every directed edge is
//! compelled in every DAG consistent with the observed (in)dependencies,
//! every undirected edge is genuinely ambiguous from this evidence alone.
//!
//! It must **not** be read as: proof that causal sufficiency holds (a latent
//! confounder between two observed variables makes them look exactly like a
//! direct causal edge — see the crate's latent-confounding adversarial test
//! for an undisguised demonstration); proof that faithfulness holds (a
//! coincidental cancellation can hide a real dependency, causing an edge to
//! be dropped that should not have been); a claim that an undirected edge
//! reflects "no causal relationship" (it means exactly the opposite — a
//! causal relationship whose *direction* the data cannot determine); or
//! immunity from a bounded [`EquivalenceClassConfig::max_conditioning_set_size`]
//! silently retaining an edge whose true separating set was larger than the
//! bound (a known, standard limitation of *any* bounded-order constraint-based
//! search — see the crate's own adversarial test demonstrating it).
//!
//! [`EquivalenceClassResult::warnings`] also surfaces anything the search
//! itself flagged as scientifically noteworthy: a candidate conditioning set
//! that could not be tested (rank-deficient, too few samples, …) and a
//! conflicting v-structure demand that was left undirected rather than
//! silently resolved (see `crate::orientation`'s docs — under a perfect
//! oracle and the stated assumptions this cannot occur, so seeing it is
//! itself diagnostic).

use crate::assumptions::CausalAssumption;
use crate::conditional_independence::ConditionalIndependenceTest;
use crate::cpdag::Cpdag;
use crate::dataset::CausalDataset;
use crate::error::CausalError;
use crate::orientation::{apply_meek_rules, orient_v_structures};
use crate::skeleton_discovery::discover_skeleton;

/// Configuration for [`PcStable`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EquivalenceClassConfig {
    /// Largest conditioning-set size the skeleton search will try. `None`
    /// (the default) is unbounded — every pair is tested against every
    /// subset of its (shrinking, as edges are removed) neighbor set,
    /// regardless of size. Bounding this trades completeness (a true
    /// separating set larger than the bound is missed, so that edge is
    /// **incorrectly retained**) for tractability on a densely connected or
    /// high-dimensional variable set — see the module docs.
    pub max_conditioning_set_size: Option<usize>,
}

impl EquivalenceClassConfig {
    /// Unbounded conditioning-set search (see
    /// [`EquivalenceClassConfig::max_conditioning_set_size`]'s docs for the
    /// tradeoff a finite bound makes).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_max_conditioning_set_size(mut self, max_conditioning_set_size: usize) -> Self {
        self.max_conditioning_set_size = Some(max_conditioning_set_size);
        self
    }
}

/// One PC-Stable run's full, reproducible outcome.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EquivalenceClassResult {
    pub cpdag: Cpdag,
    /// `((min, max), conditioning_set)` for every pair the search found
    /// non-adjacent in `cpdag`, sorted by key; absent for every adjacent
    /// pair. A `Vec` of pairs rather than a map keyed by `(usize, usize)`
    /// because `serde_json` rejects a non-string map key at serialize time
    /// (tested — see this module's own JSON round-trip test).
    pub separating_sets: Vec<((usize, usize), Vec<usize>)>,
    /// Total conditional-independence tests actually invoked (including any
    /// that returned a benign, skipped-and-warned error) — an honest measure
    /// of how much statistical evidence the result rests on.
    pub tests_performed: usize,
    pub warnings: Vec<String>,
    /// This procedure's own three assumptions, unioned with every underlying
    /// [`ConditionalIndependenceTest`] call's own reported assumptions. See
    /// the module docs for what each named assumption means here.
    pub assumptions: Vec<CausalAssumption>,
}

/// Discovers a Markov equivalence class from data via repeated conditional-
/// independence testing. See the module docs for the exact scientific scope.
pub trait EquivalenceClassDiscovery {
    /// # Errors
    ///
    /// Any [`CausalError`] the underlying `test` reports that is not one of
    /// the benign, expected-at-large-conditioning-set-size variants (see
    /// `crate::skeleton_discovery`'s docs) — those are recorded as a warning
    /// on the result instead.
    fn discover(
        &self,
        dataset: &CausalDataset,
        test: &dyn ConditionalIndependenceTest,
    ) -> Result<EquivalenceClassResult, CausalError>;
}

/// PC-Stable (Colombo & Maathuis, JMLR 2014): the order-independent variant
/// of Spirtes, Glymour & Scheines's PC algorithm. See the module docs.
pub struct PcStable {
    config: EquivalenceClassConfig,
}

impl PcStable {
    #[must_use]
    pub fn new(config: EquivalenceClassConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn config(&self) -> &EquivalenceClassConfig {
        &self.config
    }
}

impl EquivalenceClassDiscovery for PcStable {
    fn discover(
        &self,
        dataset: &CausalDataset,
        test: &dyn ConditionalIndependenceTest,
    ) -> Result<EquivalenceClassResult, CausalError> {
        let n_vars = dataset.variables.len();

        let skeleton_result =
            discover_skeleton(dataset, test, n_vars, self.config.max_conditioning_set_size)?;

        let mut cpdag = skeleton_result.skeleton;
        let mut warnings = skeleton_result.warnings;

        orient_v_structures(&mut cpdag, &skeleton_result.separating_sets, &mut warnings);
        apply_meek_rules(&mut cpdag);

        let mut assumptions = skeleton_result.test_assumptions;
        assumptions.insert(CausalAssumption::Acyclicity);
        assumptions.insert(CausalAssumption::CausalSufficiency);
        assumptions.insert(CausalAssumption::Faithfulness);

        Ok(EquivalenceClassResult {
            cpdag,
            separating_sets: skeleton_result.separating_sets.into_iter().collect(),
            tests_performed: skeleton_result.tests_performed,
            warnings,
            assumptions: assumptions.into_iter().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditional_independence::{
        ConditionalIndependenceConfig, ConditionalIndependenceMethod, PartialCorrelationTest,
    };
    use crate::environment::Environment;
    use crate::variable::{CausalVariable, VariableKind, VariableRole};
    use scirust_solvers::Matrix;
    use scirust_stats::SplitMix64;

    fn dataset_from_columns(columns: &[Vec<f64>]) -> CausalDataset {
        let n = columns[0].len();
        let d = columns.len();
        let mut data = vec![0.0; n * d];
        for row in 0..n
        {
            for col in 0..d
            {
                data[row * d + col] = columns[col][row];
            }
        }
        let variables: Vec<CausalVariable> = (0..d)
            .map(|i| {
                CausalVariable::new(
                    i,
                    format!("v{i}"),
                    VariableRole::Unspecified,
                    VariableKind::Continuous,
                )
                .unwrap()
            })
            .collect();
        let matrix = Matrix::from_row_major(n, d, data);
        let env = Environment::observational("obs").unwrap();
        CausalDataset::single_environment(variables, env, &matrix, "test fixture").unwrap()
    }

    fn noise(rng: &mut SplitMix64) -> f64 {
        rng.next_f64() - 0.5
    }

    fn fisher_z_test() -> PartialCorrelationTest {
        PartialCorrelationTest::new(
            ConditionalIndependenceConfig::new(
                0.05,
                ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z: true },
            )
            .unwrap(),
        )
    }

    /// `0 -> 2 <- 1`.
    fn collider_dataset(seed: u64, n: usize) -> CausalDataset {
        let mut rng = SplitMix64::new(seed);
        let (mut v0, mut v1, mut v2) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for _ in 0..n
        {
            let a = noise(&mut rng);
            let b = noise(&mut rng);
            let c = 0.8 * a + 0.8 * b + noise(&mut rng);
            v0.push(a);
            v1.push(b);
            v2.push(c);
        }
        dataset_from_columns(&[v0, v1, v2])
    }

    #[test]
    fn end_to_end_collider_is_fully_oriented_and_carries_all_three_assumptions() {
        let dataset = collider_dataset(12, 400);
        let test = fisher_z_test();
        let pc = PcStable::new(EquivalenceClassConfig::new());
        let result = pc.discover(&dataset, &test).unwrap();

        assert!(result.cpdag.is_directed(0, 2));
        assert!(result.cpdag.is_directed(1, 2));
        assert_eq!(result.cpdag.undirected_edges().len(), 0);
        assert_eq!(result.separating_sets, vec![((0, 1), Vec::new())]);
        assert!(result.tests_performed > 0);
        assert!(result.assumptions.contains(&CausalAssumption::Acyclicity));
        assert!(
            result
                .assumptions
                .contains(&CausalAssumption::CausalSufficiency)
        );
        assert!(result.assumptions.contains(&CausalAssumption::Faithfulness));
        assert!(
            result
                .assumptions
                .contains(&CausalAssumption::CorrectFunctionalForm)
        );
        assert!(
            result
                .assumptions
                .contains(&CausalAssumption::AdequateSampleSize)
        );
    }

    #[test]
    fn result_json_round_trips() {
        let dataset = collider_dataset(12, 400);
        let test = fisher_z_test();
        let pc = PcStable::new(EquivalenceClassConfig::new());
        let result = pc.discover(&dataset, &test).unwrap();

        let json = serde_json::to_string(&result).unwrap();
        let round_tripped: EquivalenceClassResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, round_tripped);
    }

    #[test]
    fn discovery_is_deterministic_across_repeated_runs() {
        let dataset = collider_dataset(12, 400);
        let test = fisher_z_test();
        let pc = PcStable::new(EquivalenceClassConfig::new());
        let first = pc.discover(&dataset, &test).unwrap();
        let second = pc.discover(&dataset, &test).unwrap();
        assert_eq!(first, second);
    }
}
