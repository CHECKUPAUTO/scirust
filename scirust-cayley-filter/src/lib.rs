//! Experimental Cayley–Dickson filtering for SciRust.
//!
//! The scalar `f64` implementation is the mathematical reference.
//! Optimized kernels must demonstrate numerical parity with this oracle.

#![forbid(unsafe_code)]

pub mod autotune;

pub mod analysis;
pub mod baseline;
pub mod clifford;
pub mod filter;
pub mod operator;
pub mod optimizer;
pub mod projector;
pub mod scalar;
pub mod search;
pub mod selection;
pub mod soft;
pub mod spectral;
pub mod subspace;
pub mod temporal;

pub use analysis::{AnalysisError, MatrixAnalysis, analyze_matrix, kernel_residual_norm};
pub use autotune::{CayleyAutotuneResult, CayleyCase, autotune_threshold};
pub use baseline::{IdentityFilter, NoiseDirectionProjector, ProjectionError};
pub use clifford::{CliffordProjectorError, SplitCliffordProjector, score_clifford_projector};
pub use filter::{CayleyFilter, FilterEvaluation, FilterMetrics};
pub use operator::{
    LeftMultiplicationOperator, Matrix16, left_multiplication_matrix, matrix_vector_mul,
};
pub use optimizer::{
    MultiplierCase, MultiplierOptimizationResult, MultiplierScore, optimize_multiplier,
    score_multiplier,
};
pub use projector::{CayleyProjector, ProjectorError};
pub use scalar::{
    SEDENION_DIMENSION, Sedenion, basis_vector, conjugate, sedenion_mul, squared_norm,
};
pub use search::{
    SparseMultiplierCandidate, SparseMultiplierDirection, rank_imaginary_two_term_multipliers,
    rank_two_term_multipliers, rank_zero_divisor_two_term_multipliers,
    zero_divisor_two_term_directions,
};
pub use soft::{SoftCayleyFilter, SoftFilterError};
pub use subspace::{NoiseSubspaceProjector, SubspaceProjectionError};

pub use selection::{
    DevelopmentGateDecision, IDENTITY_DEVELOPMENT_LOSS, MultiplierSelectionResult,
    SelectedMultiplierCandidate, development_gate, select_multiplier_train_dev,
    select_zero_divisor_train_dev,
};
pub use spectral::{SPECTRAL_COMPLEX_BINS, SpectralBlockFilter};
pub use temporal::{TEMPORAL_BLOCK_SIZE, TemporalBlockFilter};
