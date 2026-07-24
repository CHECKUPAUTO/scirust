//! [`CausalDataset`]: a typed, provenance-carrying bundle of variables and the
//! (possibly multi-environment) samples measured over them.

use crate::environment::Environment;
use crate::error::CausalError;
use crate::variable::{CausalVariable, validate_variable_set};
use scirust_solvers::Matrix;

/// One block of samples, all drawn under the same [`Environment`].
///
/// Stores raw row-major data (rather than embedding [`Matrix`] directly, which
/// has no `serde` support) so the block stays plain-data serializable;
/// [`SampleBlock::to_matrix`] / [`SampleBlock::from_matrix`] convert to/from
/// the numerical [`Matrix`] type the rest of the crate (e.g. `optimize_causal`)
/// operates on.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SampleBlock {
    /// Which regime produced these rows.
    pub environment: Environment,
    n_samples: usize,
    n_variables: usize,
    data: Vec<f64>,
}

impl SampleBlock {
    /// Builds a block from row-major sample data (`data[row * n_variables + col]`).
    ///
    /// # Errors
    ///
    /// [`CausalError::ZeroSamples`] if `n_samples == 0`;
    /// [`CausalError::DimensionMismatch`] if `data.len() != n_samples * n_variables`;
    /// [`CausalError::NonFiniteComputation`] at the first non-finite entry.
    pub fn new(
        environment: Environment,
        n_samples: usize,
        n_variables: usize,
        data: Vec<f64>,
    ) -> Result<Self, CausalError> {
        if n_samples == 0
        {
            return Err(CausalError::ZeroSamples);
        }
        if data.len() != n_samples * n_variables
        {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n_variables,
                got: data.len(),
            });
        }
        for (index, &v) in data.iter().enumerate()
        {
            if !v.is_finite()
            {
                return Err(CausalError::NonFiniteComputation {
                    operation: "sample_block",
                    index,
                    value: v,
                });
            }
        }
        Ok(Self {
            environment,
            n_samples,
            n_variables,
            data,
        })
    }

    /// Builds a block from an existing [`Matrix`] (rows = samples, columns =
    /// variables).
    ///
    /// # Errors
    ///
    /// Same as [`SampleBlock::new`].
    pub fn from_matrix(environment: Environment, matrix: &Matrix) -> Result<Self, CausalError> {
        Self::new(
            environment,
            matrix.rows(),
            matrix.cols(),
            matrix.data().to_vec(),
        )
    }

    /// Materializes this block's samples as a [`Matrix`] (rows = samples,
    /// columns = variables).
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        Matrix::from_row_major(self.n_samples, self.n_variables, self.data.clone())
    }

    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[must_use]
    pub fn n_variables(&self) -> usize {
        self.n_variables
    }

    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

/// A typed, provenance-carrying causal dataset: a fixed set of
/// [`CausalVariable`]s and one or more [`SampleBlock`]s, each tagged with the
/// [`Environment`] it was drawn under.
///
/// # What this is not
///
/// A `CausalDataset` records what was measured and under what regime. It
/// makes **no identifiability claim** — see [`crate::CausalCertificate`] for
/// that, which is deliberately a separate, explicit type.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CausalDataset {
    pub variables: Vec<CausalVariable>,
    pub blocks: Vec<SampleBlock>,
    /// Free-text provenance note (e.g. `"synthetic: TriangularCubicFlow seed=42"`,
    /// `"real: OBD2 telemetry log 2026-07-01"`). Never parsed, only recorded.
    pub source: String,
}

impl CausalDataset {
    /// Validates and constructs a dataset.
    ///
    /// # Errors
    ///
    /// [`CausalError::InvalidContract`] for an invalid/inconsistent variable
    /// set (see [`validate_variable_set`]) or an empty block list;
    /// [`CausalError::DimensionMismatch`] if a block's column count does not
    /// match `variables.len()`;
    /// [`CausalError::UnknownVariableIndex`] if an intervention in some
    /// block's environment targets an out-of-range variable.
    pub fn new(
        variables: Vec<CausalVariable>,
        blocks: Vec<SampleBlock>,
        source: impl Into<String>,
    ) -> Result<Self, CausalError> {
        validate_variable_set(&variables)?;
        if blocks.is_empty()
        {
            return Err(CausalError::InvalidContract {
                detail: "a causal dataset must have at least one sample block",
            });
        }
        let n = variables.len();
        for block in &blocks
        {
            if block.n_variables() != n
            {
                return Err(CausalError::DimensionMismatch {
                    expected: n,
                    got: block.n_variables(),
                });
            }
            for iv in &block.environment.interventions
            {
                if iv.target >= n
                {
                    return Err(CausalError::UnknownVariableIndex { index: iv.target });
                }
            }
        }
        Ok(Self {
            variables,
            blocks,
            source: source.into(),
        })
    }

    /// Convenience constructor for the common single-block case.
    ///
    /// # Errors
    ///
    /// Same as [`CausalDataset::new`].
    pub fn single_environment(
        variables: Vec<CausalVariable>,
        environment: Environment,
        matrix: &Matrix,
        source: impl Into<String>,
    ) -> Result<Self, CausalError> {
        let block = SampleBlock::from_matrix(environment, matrix)?;
        Self::new(variables, vec![block], source)
    }

    /// `true` iff every block's environment is purely observational.
    #[must_use]
    pub fn is_purely_observational(&self) -> bool {
        self.blocks.iter().all(|b| b.environment.is_observational())
    }

    /// Total sample count across all blocks.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.blocks.iter().map(SampleBlock::n_samples).sum()
    }
}
