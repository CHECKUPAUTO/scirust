//! Bounded deterministic V2 discovery smoke.
//!
//! This specifies a recurrence signature and counterexamples, not a target AST.

use scirust_algogen::tensor::v2::{
    CounterexampleCase, CounterexampleSet, DType, ExecutionPolicy, ExperimentConfig, FloatPolicy,
    GenerationRequest, Grammar, GrammarProfile, Op, OperatorClass, ScalarValue, StateInitializer,
    StateSpec, ValueTensor, ValueType, VerificationLimits, run_scientific_experiment,
};

fn scalar(value: f64) -> ValueTensor {
    ValueTensor::scalar_f64(value)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let value_type = ValueType::scalar(DType::F64);
    let mut grammar = Grammar::profile(GrammarProfile::StreamingRecurrence);
    grammar.allowed_classes = vec![
        OperatorClass::Constant,
        OperatorClass::Arithmetic,
        OperatorClass::Shape,
    ];
    grammar.allowed_dtypes = vec![DType::F64];
    grammar.constants = vec![
        ScalarValue::F64(-1.0),
        ScalarValue::F64(0.0),
        ScalarValue::F64(1.0),
    ];
    grammar.max_operations = 8;
    grammar.max_values = 8;
    grammar.max_depth = 3;
    grammar.max_shape_ops = 4;

    let config = ExperimentConfig {
        source_revision: std::env::var("SCIRUST_SOURCE_REVISION")
            .unwrap_or_else(|_| "explicitly-unset".to_string()),
        seed: 0xA11C_E5E5,
        max_candidates: 256,
        archive_capacity: 16,
        stop_on_exact: true,
        grammar,
        request: GenerationRequest {
            inputs: vec![],
            items: vec![value_type.clone()],
            state: vec![StateSpec {
                value_type: value_type.clone(),
                initializer: StateInitializer::Constant(ScalarValue::F64(0.0)),
            }],
            steps: 3,
            output_types: vec![value_type],
            min_random_step_ops: 1,
            max_random_step_ops: 1,
            min_random_finalize_ops: 0,
            max_random_finalize_ops: 0,
            require_state_update: true,
        },
        verification_limits: VerificationLimits::default(),
        execution_policy: ExecutionPolicy {
            floats: FloatPolicy::FiniteOutputs,
        },
    };
    let dataset = CounterexampleSet::new(
        "sum-recurrence-adversarial-v1",
        vec![
            CounterexampleCase {
                inputs: vec![],
                items: vec![scalar(1.0), scalar(2.0), scalar(3.0)],
                expected_outputs: vec![scalar(6.0)],
            },
            CounterexampleCase {
                inputs: vec![],
                items: vec![scalar(0.0), scalar(-0.0), scalar(0.0)],
                expected_outputs: vec![scalar(0.0)],
            },
            CounterexampleCase {
                inputs: vec![],
                items: vec![scalar(-2.0), scalar(5.0), scalar(-1.0)],
                expected_outputs: vec![scalar(2.0)],
            },
            CounterexampleCase {
                inputs: vec![],
                items: vec![scalar(1.0e-12), scalar(-1.0e-12), scalar(4.0)],
                expected_outputs: vec![scalar(4.0)],
            },
        ],
    )?;

    let archive = run_scientific_experiment(&config, &dataset)?;
    let exact = archive
        .pareto
        .iter()
        .find(|entry| entry.fitness.correctness.exact);
    println!("success={}", archive.success);
    println!(
        "attempted={} generated={} duplicates={}",
        archive.diagnostics.candidates_attempted,
        archive.diagnostics.candidates_generated,
        archive.diagnostics.canonical_duplicates
    );
    println!("archive_digest={}", archive.digest);
    if let Some(entry) = exact
    {
        println!("exact_candidate_index={}", entry.candidate_index);
        println!("exact_candidate_seed={}", entry.candidate_seed);
        println!("exact_program_digest={}", entry.digest);
        println!(
            "step_contains_add={}",
            entry
                .program
                .step
                .ops
                .iter()
                .any(|op| matches!(op, Op::Add(_)))
        );
    }
    if !archive.success
    {
        return Err("bounded discovery did not find an exact candidate".into());
    }
    Ok(())
}
