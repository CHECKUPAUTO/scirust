//! Serialisable specification of a reproducible tensor discovery problem.
//!
//! A [`TensorProblem`] fully determines an experiment: the evaluation dataset,
//! the operator and resource budget, the generation and population parameters,
//! the deterministic seed, the generation budget and explicit success criteria.
//! Everything is serialisable so a problem can be committed as a stable fixture
//! and replayed exactly. Tensors are stored as [`TensorFixture`] values (shape +
//! flat data) so the whole problem round-trips through serde without depending
//! on tensor-core serialisation.

use serde::{Deserialize, Serialize};

use super::dataset::{Dataset, DatasetError, TensorCase};
use super::fitness::FitnessReport;
use super::population::EvolutionConfig;
use super::verify::VerificationLimits;
use scirust_tensor_core::TensorND;

/// A serialisable tensor value: shape plus row-major flat data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorFixture {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorFixture {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }

    /// Build a [`TensorND`], rejecting a shape/data-length mismatch.
    pub fn to_tensor(&self) -> Result<TensorND, String> {
        TensorND::try_new(self.data.clone(), self.shape.clone())
    }

    pub fn from_tensor(tensor: &TensorND) -> Self {
        Self {
            shape: tensor.shape.clone(),
            data: tensor.data.clone(),
        }
    }
}

/// A serialisable evaluation case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseFixture {
    pub inputs: Vec<TensorFixture>,
    pub expected: TensorFixture,
}

impl CaseFixture {
    pub fn new(inputs: Vec<TensorFixture>, expected: TensorFixture) -> Self {
        Self { inputs, expected }
    }

    fn to_case(&self) -> Result<TensorCase, String> {
        let inputs = self
            .inputs
            .iter()
            .map(TensorFixture::to_tensor)
            .collect::<Result<Vec<_>, _>>()?;
        let expected = self.expected.to_tensor()?;
        Ok(TensorCase::new(inputs, expected))
    }
}

/// Serialisable mirror of [`VerificationLimits`] (which is not itself serde).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProblemLimits {
    pub max_instructions: usize,
    pub max_rank: usize,
    pub max_elements_per_tensor: usize,
    pub max_total_register_elements: usize,
}

impl Default for ProblemLimits {
    fn default() -> Self {
        Self::from(VerificationLimits::default())
    }
}

impl From<VerificationLimits> for ProblemLimits {
    fn from(limits: VerificationLimits) -> Self {
        Self {
            max_instructions: limits.max_instructions,
            max_rank: limits.max_rank,
            max_elements_per_tensor: limits.max_elements_per_tensor,
            max_total_register_elements: limits.max_total_register_elements,
        }
    }
}

impl From<ProblemLimits> for VerificationLimits {
    fn from(limits: ProblemLimits) -> Self {
        Self {
            max_instructions: limits.max_instructions,
            max_rank: limits.max_rank,
            max_elements_per_tensor: limits.max_elements_per_tensor,
            max_total_register_elements: limits.max_total_register_elements,
        }
    }
}

/// Explicit, deterministic success criteria. A criterion of `None` is ignored.
///
/// Success requires the program to have been evaluated and to satisfy every
/// specified bound. At least one bound must be present.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuccessCriteria {
    pub max_loss: Option<f64>,
    pub max_active_instructions: Option<usize>,
    pub max_estimated_flops: Option<u64>,
    pub max_peak_live_elements: Option<u64>,
}

impl SuccessCriteria {
    /// A criterion on correctness loss alone.
    pub fn max_loss(max_loss: f64) -> Self {
        Self {
            max_loss: Some(max_loss),
            max_active_instructions: None,
            max_estimated_flops: None,
            max_peak_live_elements: None,
        }
    }

    fn has_any(&self) -> bool {
        self.max_loss.is_some()
            || self.max_active_instructions.is_some()
            || self.max_estimated_flops.is_some()
            || self.max_peak_live_elements.is_some()
    }

    /// Whether `report` satisfies every specified bound.
    pub fn is_met(&self, report: &FitnessReport) -> bool {
        if !report.evaluated
        {
            return false;
        }
        self.max_loss.is_none_or(|bound| report.loss <= bound)
            && self
                .max_active_instructions
                .is_none_or(|bound| report.cost.active_instructions <= bound)
            && self
                .max_estimated_flops
                .is_none_or(|bound| report.cost.estimated_flops <= bound)
            && self
                .max_peak_live_elements
                .is_none_or(|bound| report.cost.peak_live_elements <= bound)
    }
}

/// A validation failure for a [`TensorProblem`].
#[derive(Debug, Clone, PartialEq)]
pub enum ProblemError {
    EmptyId,
    NoCases,
    InvalidDataset(DatasetError),
    MalformedFixture(String),
    InputShapeMismatch,
    InvalidPopulation(String),
    InvalidNumericConfiguration(String),
    NoSuccessCriteria,
}

/// A fully specified, reproducible discovery problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorProblem {
    /// Stable identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Evaluation cases (each with exact expected output).
    pub cases: Vec<CaseFixture>,
    /// Resource limits.
    pub limits: ProblemLimits,
    /// Generation, population and budget configuration.
    pub evolution: EvolutionConfig,
    /// Deterministic seed.
    pub seed: u64,
    /// Explicit success criteria.
    pub success: SuccessCriteria,
}

impl TensorProblem {
    /// The resource limits as a [`VerificationLimits`].
    pub fn verification_limits(&self) -> VerificationLimits {
        self.limits.into()
    }

    /// The generation budget (number of offspring generations).
    pub fn generation_budget(&self) -> usize {
        self.evolution.generations
    }

    /// Build the runtime [`Dataset`], rejecting malformed fixtures.
    pub fn dataset(&self) -> Result<Dataset, ProblemError> {
        let cases = self
            .cases
            .iter()
            .map(CaseFixture::to_case)
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProblemError::MalformedFixture)?;
        Dataset::new(cases).map_err(ProblemError::InvalidDataset)
    }

    /// Validate the problem, rejecting inconsistent or impossible configuration
    /// before any experiment begins.
    pub fn validate(&self) -> Result<(), ProblemError> {
        if self.id.is_empty()
        {
            return Err(ProblemError::EmptyId);
        }
        if self.cases.is_empty()
        {
            return Err(ProblemError::NoCases);
        }

        let dataset = self.dataset()?;

        if self.evolution.generation.input_shapes != dataset.input_shapes()
        {
            return Err(ProblemError::InputShapeMismatch);
        }
        if self.evolution.population_size == 0
        {
            return Err(ProblemError::InvalidPopulation(
                "population_size must be at least 1".to_string(),
            ));
        }
        if self.evolution.tournament.size == 0
        {
            return Err(ProblemError::InvalidPopulation(
                "tournament size must be at least 1".to_string(),
            ));
        }
        if self.evolution.elitism > self.evolution.population_size
        {
            return Err(ProblemError::InvalidPopulation(
                "elitism must not exceed population_size".to_string(),
            ));
        }
        let generation_scale = self.evolution.generation.scale_magnitude;
        let mutation_scale = self.evolution.scale_magnitude;
        if !generation_scale.is_finite() || generation_scale < 0.0
        {
            return Err(ProblemError::InvalidNumericConfiguration(
                "generation scale_magnitude must be finite and non-negative".to_string(),
            ));
        }
        if !mutation_scale.is_finite() || mutation_scale < 0.0
        {
            return Err(ProblemError::InvalidNumericConfiguration(
                "mutation scale_magnitude must be finite and non-negative".to_string(),
            ));
        }
        for (name, probability) in [
            (
                "crossover_probability",
                self.evolution.crossover_probability,
            ),
            ("mutation_probability", self.evolution.mutation_probability),
        ]
        {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability)
            {
                return Err(ProblemError::InvalidNumericConfiguration(format!(
                    "{name} must be finite and in [0, 1]"
                )));
            }
        }
        if self
            .success
            .max_loss
            .is_some_and(|value| !value.is_finite())
        {
            return Err(ProblemError::InvalidNumericConfiguration(
                "success max_loss must be finite".to_string(),
            ));
        }
        if !self.success.has_any()
        {
            return Err(ProblemError::NoSuccessCriteria);
        }

        Ok(())
    }
}

/// Built-in, mathematically exact benchmark problems.
///
/// Every expected output is produced by an explicit oracle implemented directly
/// in this module — never by executing the candidate IR under test — and every
/// problem uses multiple cases with distinct expected outputs so a
/// constant-output program cannot satisfy them all.
pub mod benchmarks {
    use super::*;
    use crate::tensor::{GenerationConfig, OperatorSet, TournamentConfig};

    fn evolution(
        input_shapes: Vec<Vec<usize>>,
        operators: OperatorSet,
        max_instructions: usize,
    ) -> EvolutionConfig {
        EvolutionConfig {
            generation: GenerationConfig {
                input_shapes,
                min_instructions: 1,
                max_instructions,
                operators,
                scale_magnitude: 3.0,
            },
            population_size: 24,
            generations: 20,
            elitism: 2,
            tournament: TournamentConfig { size: 3 },
            scale_magnitude: 3.0,
            crossover_probability: 0.7,
            mutation_probability: 0.6,
        }
    }

    fn problem(
        id: &str,
        description: &str,
        cases: Vec<CaseFixture>,
        operators: OperatorSet,
        max_instructions: usize,
        success: SuccessCriteria,
    ) -> TensorProblem {
        let input_shapes = cases[0]
            .inputs
            .iter()
            .map(|fixture| fixture.shape.clone())
            .collect();
        TensorProblem {
            id: id.to_string(),
            description: description.to_string(),
            cases,
            limits: ProblemLimits::default(),
            evolution: evolution(input_shapes, operators, max_instructions),
            seed: 0x5EED_u64,
            success,
        }
    }

    // ---- independent oracles -------------------------------------------------

    /// Element-wise ReLU.
    pub fn relu_oracle(data: &[f32]) -> Vec<f32> {
        data.iter().map(|&value| value.max(0.0)).collect()
    }

    /// Element-wise scaling by a constant.
    pub fn scale_oracle(data: &[f32], factor: f32) -> Vec<f32> {
        data.iter().map(|&value| value * factor).collect()
    }

    /// Element-wise addition.
    pub fn add_oracle(left: &[f32], right: &[f32]) -> Vec<f32> {
        left.iter().zip(right).map(|(&a, &b)| a + b).collect()
    }

    /// Row-major matrix multiplication of `[m, k]` by `[k, n]`.
    pub fn matmul_oracle(left: &[f32], right: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m
        {
            for column in 0..n
            {
                let mut sum = 0.0f32;
                for inner in 0..k
                {
                    sum += left[row * k + inner] * right[inner * n + column];
                }
                out[row * n + column] = sum;
            }
        }
        out
    }

    /// Row-major transpose of `[m, n]` into `[n, m]`.
    pub fn transpose_oracle(data: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m
        {
            for column in 0..n
            {
                out[column * m + row] = data[row * n + column];
            }
        }
        out
    }

    // ---- problems ------------------------------------------------------------

    /// Reproduce the input exactly.
    pub fn identity() -> TensorProblem {
        let inputs = [
            vec![1.0, -2.0, 3.0, -4.0],
            vec![0.5, 0.25, -0.75, 2.0],
            vec![-1.0, -1.0, 1.0, 1.0],
        ];
        let cases = inputs
            .iter()
            .map(|data| {
                CaseFixture::new(
                    vec![TensorFixture::new(vec![2, 2], data.clone())],
                    TensorFixture::new(vec![2, 2], data.clone()),
                )
            })
            .collect();
        problem(
            "identity",
            "Return the input tensor unchanged.",
            cases,
            OperatorSet::all(),
            4,
            SuccessCriteria::max_loss(0.0),
        )
    }

    /// Scale the input by a fixed constant.
    pub fn scale_by_three() -> TensorProblem {
        let factor = 3.0;
        let inputs = [
            vec![1.0, 2.0, -1.0, 0.0],
            vec![-2.0, 0.5, 4.0, -3.0],
            vec![10.0, -10.0, 0.25, 1.5],
        ];
        let cases = inputs
            .iter()
            .map(|data| {
                CaseFixture::new(
                    vec![TensorFixture::new(vec![2, 2], data.clone())],
                    TensorFixture::new(vec![2, 2], scale_oracle(data, factor)),
                )
            })
            .collect();
        problem(
            "scale_by_three",
            "Multiply every element by 3.",
            cases,
            OperatorSet::all(),
            4,
            SuccessCriteria::max_loss(1e-6),
        )
    }

    /// Element-wise ReLU.
    pub fn relu() -> TensorProblem {
        let inputs = [
            vec![-1.0, 2.0, -3.0, 4.0],
            vec![5.0, -6.0, 7.0, -8.0],
            vec![0.0, -0.5, 0.5, -1.5],
        ];
        let cases = inputs
            .iter()
            .map(|data| {
                CaseFixture::new(
                    vec![TensorFixture::new(vec![2, 2], data.clone())],
                    TensorFixture::new(vec![2, 2], relu_oracle(data)),
                )
            })
            .collect();
        problem(
            "relu",
            "Apply element-wise ReLU.",
            cases,
            OperatorSet::all(),
            4,
            SuccessCriteria::max_loss(0.0),
        )
    }

    /// Transpose a rank-two tensor.
    pub fn transpose() -> TensorProblem {
        let inputs = [
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![-1.0, 0.0, 2.0, 7.0, -3.0, 4.0],
            vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        ];
        let cases = inputs
            .iter()
            .map(|data| {
                CaseFixture::new(
                    vec![TensorFixture::new(vec![2, 3], data.clone())],
                    TensorFixture::new(vec![3, 2], transpose_oracle(data, 2, 3)),
                )
            })
            .collect();
        problem(
            "transpose",
            "Transpose a 2x3 matrix into 3x2.",
            cases,
            OperatorSet::all(),
            4,
            SuccessCriteria::max_loss(0.0),
        )
    }

    /// Element-wise matrix addition.
    pub fn matrix_add() -> TensorProblem {
        let pairs = [
            (vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]),
            (vec![-1.0, -2.0, -3.0, -4.0], vec![1.0, 2.0, 3.0, 4.0]),
            (vec![0.5, 0.5, 0.5, 0.5], vec![-0.25, 0.75, 1.25, -1.75]),
        ];
        let cases = pairs
            .iter()
            .map(|(a, b)| {
                CaseFixture::new(
                    vec![
                        TensorFixture::new(vec![2, 2], a.clone()),
                        TensorFixture::new(vec![2, 2], b.clone()),
                    ],
                    TensorFixture::new(vec![2, 2], add_oracle(a, b)),
                )
            })
            .collect();
        problem(
            "matrix_add",
            "Add two 2x2 matrices element-wise.",
            cases,
            OperatorSet::all(),
            6,
            SuccessCriteria::max_loss(0.0),
        )
    }

    /// Matrix multiplication of `[2, 3]` by `[3, 2]`.
    pub fn matrix_multiply() -> TensorProblem {
        let pairs = [
            (
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ),
            (
                vec![-1.0, 0.0, 1.0, 2.0, -2.0, 3.0],
                vec![1.0, 1.0, 0.0, 2.0, 3.0, -1.0],
            ),
            (
                vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                vec![1.0, 0.0, 0.0, 1.0, 2.0, 2.0],
            ),
        ];
        let cases = pairs
            .iter()
            .map(|(a, b)| {
                CaseFixture::new(
                    vec![
                        TensorFixture::new(vec![2, 3], a.clone()),
                        TensorFixture::new(vec![3, 2], b.clone()),
                    ],
                    TensorFixture::new(vec![2, 2], matmul_oracle(a, b, 2, 3, 2)),
                )
            })
            .collect();
        problem(
            "matrix_multiply",
            "Multiply a 2x3 matrix by a 3x2 matrix.",
            cases,
            OperatorSet::all(),
            6,
            SuccessCriteria::max_loss(1e-4),
        )
    }

    /// The two-stage composition `ReLU(A x B + C)`.
    pub fn relu_affine() -> TensorProblem {
        let triples = [
            (
                vec![1.0, -2.0, 3.0, 0.0, 1.0, -1.0], // A [2,3]
                vec![1.0, 0.0, 0.0, 1.0, 1.0, -1.0],  // B [3,2]
                vec![-5.0, 0.5, 0.0, -0.5],           // C [2,2]
            ),
            (
                vec![2.0, 1.0, 0.0, -1.0, 2.0, 1.0],
                vec![0.0, 1.0, 1.0, 0.0, -1.0, 2.0],
                vec![0.0, -3.0, 1.0, -1.0],
            ),
            (
                vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![-2.0, -2.0, -2.0, -2.0],
            ),
        ];
        let cases = triples
            .iter()
            .map(|(a, b, c)| {
                let product = matmul_oracle(a, b, 2, 3, 2);
                let affine = add_oracle(&product, c);
                let expected = relu_oracle(&affine);
                CaseFixture::new(
                    vec![
                        TensorFixture::new(vec![2, 3], a.clone()),
                        TensorFixture::new(vec![3, 2], b.clone()),
                        TensorFixture::new(vec![2, 2], c.clone()),
                    ],
                    TensorFixture::new(vec![2, 2], expected),
                )
            })
            .collect();
        problem(
            "relu_affine",
            "Compute ReLU(A x B + C) for A[2,3], B[3,2], C[2,2].",
            cases,
            OperatorSet::all(),
            8,
            SuccessCriteria::max_loss(1e-4),
        )
    }

    /// Every built-in problem.
    pub fn all() -> Vec<TensorProblem> {
        vec![
            identity(),
            scale_by_three(),
            relu(),
            transpose(),
            matrix_add(),
            matrix_multiply(),
            relu_affine(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_built_in_problems_validate() {
        for problem in benchmarks::all()
        {
            problem
                .validate()
                .unwrap_or_else(|error| panic!("problem {} invalid: {error:?}", problem.id));
        }
    }

    #[test]
    fn each_problem_has_distinct_expected_outputs() {
        // At least two cases must differ, so a constant-output program cannot
        // satisfy every case.
        for problem in benchmarks::all()
        {
            let first = &problem.cases[0].expected;
            let differs = problem.cases[1..]
                .iter()
                .any(|case| &case.expected != first);
            assert!(
                differs,
                "problem {} has constant expected output",
                problem.id
            );
        }
    }

    #[test]
    fn oracles_are_exact() {
        assert_eq!(benchmarks::relu_oracle(&[-1.0, 2.0]), vec![0.0, 2.0]);
        assert_eq!(benchmarks::scale_oracle(&[1.0, -2.0], 3.0), vec![3.0, -6.0]);
        assert_eq!(
            benchmarks::add_oracle(&[1.0, 2.0], &[3.0, 4.0]),
            vec![4.0, 6.0]
        );
        // [1 2 3; 4 5 6] x [7 8; 9 10; 11 12] = [58 64; 139 154]
        assert_eq!(
            benchmarks::matmul_oracle(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                2,
                3,
                2
            ),
            vec![58.0, 64.0, 139.0, 154.0]
        );
        // transpose [1 2 3; 4 5 6] = [1 4; 2 5; 3 6]
        assert_eq!(
            benchmarks::transpose_oracle(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn rejects_invalid_configurations() {
        let mut empty_id = benchmarks::identity();
        empty_id.id = String::new();
        assert_eq!(empty_id.validate(), Err(ProblemError::EmptyId));

        let mut no_success = benchmarks::identity();
        no_success.success = SuccessCriteria {
            max_loss: None,
            max_active_instructions: None,
            max_estimated_flops: None,
            max_peak_live_elements: None,
        };
        assert_eq!(no_success.validate(), Err(ProblemError::NoSuccessCriteria));

        let mut bad_shapes = benchmarks::identity();
        bad_shapes.evolution.generation.input_shapes = vec![vec![9, 9]];
        assert_eq!(bad_shapes.validate(), Err(ProblemError::InputShapeMismatch));

        let mut zero_pop = benchmarks::identity();
        zero_pop.evolution.population_size = 0;
        assert!(matches!(
            zero_pop.validate(),
            Err(ProblemError::InvalidPopulation(_))
        ));

        for probability in [f64::NAN, f64::INFINITY, -0.1, 1.1]
        {
            let mut invalid = benchmarks::identity();
            invalid.evolution.mutation_probability = probability;
            assert!(matches!(
                invalid.validate(),
                Err(ProblemError::InvalidNumericConfiguration(_))
            ));
        }

        let mut invalid_loss = benchmarks::identity();
        invalid_loss.success.max_loss = Some(f64::from_bits(0x7ff8_0000_0000_0001));
        assert!(matches!(
            invalid_loss.validate(),
            Err(ProblemError::InvalidNumericConfiguration(_))
        ));
    }

    #[test]
    fn success_criteria_semantics() {
        let dataset = benchmarks::identity().dataset().unwrap();
        let program = crate::tensor::TensorProgram::new(
            vec![crate::tensor::TensorInstruction::Input { input: 0 }],
            0,
        );
        let report = crate::tensor::evaluate_program(
            &program,
            &dataset,
            benchmarks::identity().verification_limits(),
        );
        // Identity solves the identity problem exactly.
        assert_eq!(report.loss, 0.0);
        assert!(SuccessCriteria::max_loss(0.0).is_met(&report));
        assert!(!SuccessCriteria::max_loss(-1.0).is_met(&report));
    }

    #[test]
    fn problem_round_trips_through_serde() {
        let problem = benchmarks::relu_affine();
        let json = serde_json::to_string(&problem).unwrap();
        let decoded: TensorProblem = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, problem);
    }
}
