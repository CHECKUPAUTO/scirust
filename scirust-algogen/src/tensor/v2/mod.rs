//! V2 scientific algorithm discovery IR.
//!
//! A typed, sectioned, multi-output program representation with statically
//! bounded recurrences — see `docs/SCIRUST_ALGOGEN_IR_V2_ARCHITECTURE.md`.
//! V1 (`super`) remains frozen and byte-stable; this module is additive.

pub mod canonical;
pub mod compat;
pub mod cost;
pub mod evolve;
pub mod generate;
pub mod interpret;
pub mod ir;
pub mod range;
pub mod reference;
pub mod search;
pub mod semantics;
pub mod serialization;
pub mod simplify;
pub mod types;
pub mod verify;

pub use canonical::{
    CANONICAL_FORMAT_VERSION, CANONICAL_MAGIC, CANONICALIZATION_VERSION, canonical_bytes,
    canonical_equal, program_digest, program_fingerprint,
};
pub use cost::{CostReport, estimate_cost, estimate_cost_verified};
pub use evolve::{
    AppliedMutation, CrossoverError, CrossoverResult, CrossoverUnit, MutationConfig, MutationError,
    MutationKind, MutationResult, crossover_programs, mutate_program,
};
pub use generate::{
    GeneratedProgram, GenerationError, GenerationRequest, GenerationStats, Grammar, GrammarProfile,
    OperatorClass, StateInitializer, StateSpec, generate_program,
};
pub use interpret::{
    ExecutionError, ExecutionPolicy, ExecutionResult, FloatPolicy, TensorDataError, ValueTensor,
    execute_program,
};
pub use ir::{Bin, IR_VERSION, Op, Ref, ResearchProgram, Section, Un, ValueId};
pub use range::{
    Finiteness, Interval, MAX_RANGE_RECURRENCE_PASSES, RangeAnalysis, Sign, ValueFacts,
    analyze_ranges, analyze_ranges_verified,
};
pub use reference::{
    AdaReadinessPrograms, ada_readiness_programs, affine_scalar_program, attention_recurrence,
    bounded_root_recurrence, compensated_sum_recurrence, error_budget_program,
    indexed_masked_accumulation_program, masked_update_program, matrix_multiplication_program,
    online_softmax_recurrence, reduction_max_program, reduction_statistics_program,
    reduction_sum_program, shape_broadcast_program, threshold_support_program,
    two_pass_softmax_building_blocks, welford_recurrence,
};
pub use search::{
    ArchiveDecision, CASE_FAILURE_PENALTY, CandidateRecord, CaseTensorRole, CorrectnessObjectives,
    CounterexampleCase, CounterexampleError, CounterexampleSet, EXPERIMENT_SCHEMA_VERSION,
    ExperimentConfig, ParetoArchive, ParetoEntry, RejectionCategory, ScientificExperimentArchive,
    ScientificExperimentError, ScientificFitness, SearchDiagnostics, dominates,
    evaluate_on_counterexamples, replay_scientific_experiment, run_scientific_experiment,
};
pub use semantics::NumericalSemantics;
pub use serialization::{
    ProgramEnvelope, SERIALIZATION_VERSION, SerializationError, deserialize_program,
    serialize_program,
};
pub use simplify::{
    Canonicalized, MAX_SIMPLIFY_PASSES, RewriteApplication, RewriteRuleId, SimplifyStats,
    canonicalize,
};
pub use types::{DType, ScalarValue, ShapeError, ValueType};
pub use verify::{
    ProgramError, SectionKind, SignatureKind, VerificationLimits, VerifiedProgram, verify_program,
};

#[cfg(test)]
mod tests;
