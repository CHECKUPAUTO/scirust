//! Deterministic discovery of cache refresh policies for diffusion LLMs.
//!
//! The competing Elastic-Cache implementation refreshes when a single mean
//! cosine similarity falls below a fixed `gamma`. This experiment learns an
//! interpretable multi-signal policy with SciRust's seeded CMA-ES, calibrates it
//! to an explicit stale-cache quality budget, and compares it with a complete
//! fixed-gamma sweep at the same measured quality.
//!
//! Synthetic traces validate the machinery only. A claim about LLaDA or Dream
//! requires counterfactual dual-run traces from those models.

mod discovery;
mod model;
mod synthetic;
mod trace_io;

pub use discovery::{
    best_fixed_gamma, best_fixed_gamma_robust, calibrate_threshold,
    calibrate_threshold_robust, compare_on_holdout, compare_on_holdout_robust,
    discover_linear_policy, discover_symbolic_surrogate,
};
pub use model::{
    DiscoveryConfig, DiscoveryResult, FEATURE_NAMES, GammaBaseline, HoldoutComparison,
    LinearPolicy, PolicyMetrics, RobustHoldoutComparison, SymbolicCandidate,
    TrajectoryPolicyMetrics, TraceRow, evaluate_policy, evaluate_policy_by_trajectory,
    split_by_trajectory, split_by_trajectory_fold,
};
pub use synthetic::synthetic_trace;
pub use trace_io::{TRACE_HEADER, read_trace_csv, write_trace_csv};
