//! Behavioral signatures and recognition of rediscovered algorithms.
//!
//! A behavioral signature summarizes what a program *computes* on a fixed,
//! deterministic probe dataset derived only from its declared signature —
//! never from its structure. When the search rediscovers a known algorithm
//! (Kahan summation, Welford moments, online softmax…), `recognize`
//! names it, separating genuine novelty from rediscovery.
//!
//! Signature equality is finite evidence of behavioral identity on the probe
//! set, not proof over the input continuum; structural identity remains the
//! domain of canonical bytes.

use sha2::{Digest, Sha256};

use super::interpret::{ExecutionPolicy, ExecutionResult, ValueTensor, execute_program};
use super::ir::ResearchProgram;
use super::types::ValueType;
use super::verify::VerificationLimits;

/// Maximum scan steps probed for recurrent programs (bounded evidence).
pub const MAX_PROBED_STEPS: u32 = 4;

/// Deterministic probe values cycled into generated tensors. They exercise
/// sign, magnitude and fraction behaviour while staying finite so the default
/// execution policy accepts them.
pub const PROBE_VALUES: &[f64] = &[1.0, -2.0, 0.5, 3.0, -0.25, 4.0];

/// Build one deterministic probe tensor matching `value_type`.
#[must_use]
pub fn probe_tensor(value_type: &ValueType, salt: usize) -> ValueTensor {
    let elements = value_type.checked_elements().unwrap_or(0);
    let data = (0..elements)
        .map(|index| PROBE_VALUES[(index + salt) % PROBE_VALUES.len()])
        .collect();
    ValueTensor::new(value_type.dtype, value_type.shape.clone(), data)
        .expect("probe tensors are well-formed by construction")
}

/// Execute a program on the deterministic probe dataset.
///
/// Returns `None` when the program cannot be executed under the default
/// policy (e.g. it requires non-finite identities to reach outputs); such
/// programs simply have no behavioral signature here.
#[must_use]
pub fn probe_execution(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Option<ExecutionResult> {
    let inputs: Vec<ValueTensor> = program
        .inputs
        .iter()
        .enumerate()
        .map(|(index, value_type)| probe_tensor(value_type, index))
        .collect();
    let items_per_step = program.items.len();
    let steps = (program.steps).min(MAX_PROBED_STEPS);
    let mut items = Vec::new();
    if items_per_step > 0 && steps > 0
    {
        items.reserve(steps as usize * items_per_step);
        for step in 0..steps
        {
            for (slot, value_type) in program.items.iter().enumerate()
            {
                items.push(probe_tensor(value_type, step as usize + slot));
            }
        }
    }
    // A program whose declared steps exceed the probe budget still needs a
    // consistent stream: verify against the *probed* shape first.
    let mut probed = program.clone();
    probed.steps = steps;
    super::verify_program(&probed, limits).ok().and_then(|_| {
        execute_program(&probed, &inputs, &items, ExecutionPolicy::default(), limits).ok()
    })
}

/// Content digest of one executed result: output shapes, dtypes and exact
/// element bits in program order.
#[must_use]
fn result_digest(result: &ExecutionResult) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SCIRUST-RIR2-BEHAVIOR\0");
    for output in &result.outputs
    {
        bytes.push(output.dtype.tag());
        bytes.extend_from_slice(&(output.shape.len() as u64).to_le_bytes());
        for &dimension in &output.shape
        {
            bytes.extend_from_slice(&(dimension as u64).to_le_bytes());
        }
        for &value in &output.data
        {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    let hash = Sha256::digest(&bytes);
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Behavioral signature of a program on the standardized probes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehavioralSignature {
    pub digest: String,
}

/// Compute the behavioral signature, or `None` when the program cannot run
/// under the default policy on the probes.
#[must_use]
pub fn behavioral_signature(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Option<BehavioralSignature> {
    probe_execution(program, limits).map(|result| BehavioralSignature {
        digest: result_digest(&result),
    })
}

/// Registry mapping behavioral digests to human-readable algorithm names.
#[derive(Debug, Clone, Default)]
pub struct AlgorithmRegistry {
    entries: std::collections::BTreeMap<String, String>,
}

impl AlgorithmRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one program under `name` using its computed signature.
    pub fn register(&mut self, name: &str, program: &ResearchProgram, limits: VerificationLimits) {
        if let Some(signature) = behavioral_signature(program, limits)
        {
            self.entries.insert(signature.digest, name.to_string());
        }
    }

    /// Look up a program's behavior among registered algorithms.
    #[must_use]
    pub fn recognize(&self, program: &ResearchProgram, limits: VerificationLimits) -> Option<&str> {
        let signature = behavioral_signature(program, limits)?;
        self.entries.get(&signature.digest).map(String::as_str)
    }

    /// Number of registered behaviors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The built-in registry of reference fixtures (representation examples,
/// not privileged search knowledge).
#[must_use]
pub fn known_algorithm_registry(limits: VerificationLimits) -> AlgorithmRegistry {
    let mut registry = AlgorithmRegistry::new();
    registry.register(
        "online_softmax_recurrence",
        &super::reference::online_softmax_recurrence(3),
        limits,
    );
    registry.register(
        "welford_recurrence",
        &super::reference::welford_recurrence(3),
        limits,
    );
    registry.register(
        "compensated_sum_recurrence",
        &super::reference::compensated_sum_recurrence(3),
        limits,
    );
    registry.register(
        "two_pass_softmax_building_blocks",
        &super::reference::two_pass_softmax_building_blocks(4),
        limits,
    );
    registry.register(
        "reduction_sum",
        &super::reference::reduction_sum_program(4),
        limits,
    );
    registry.register(
        "reduction_max",
        &super::reference::reduction_max_program(4),
        limits,
    );
    registry.register(
        "matrix_multiplication_2_2_2",
        &super::reference::matrix_multiplication_program(2, 2, 2),
        limits,
    );
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::ir::{Op, Ref, Section};
    use crate::tensor::v2::types::DType;
    use crate::tensor::v2::types::ScalarValue;

    #[test]
    fn known_fixtures_are_recognized_by_behavior() {
        let limits = VerificationLimits::default();
        let registry = known_algorithm_registry(limits);
        assert!(registry.len() >= 7);

        // The exact fixture...
        let welford = super::super::reference::welford_recurrence(3);
        assert_eq!(
            registry.recognize(&welford, limits),
            Some("welford_recurrence")
        );

        // ...and a structurally different but behaviorally identical variant:
        // same fold with an extra dead-free passthrough reshape on outputs.
        let mut renamed = welford.clone();
        let finalize_len = renamed.finalize.ops.len();
        for slot in 0..3
        {
            renamed
                .finalize
                .ops
                .push(Op::Reshape(super::super::ir::ShapeTo {
                    src: Ref::StateFinal(slot),
                    shape: vec![],
                }));
            renamed.outputs[slot] = finalize_len + slot;
        }
        assert_ne!(renamed, welford, "variants must differ structurally");
        assert_eq!(
            registry.recognize(&renamed, limits),
            Some("welford_recurrence"),
            "behavioral identity survives structural variation"
        );
    }

    #[test]
    fn different_behaviors_are_not_confused() {
        let limits = VerificationLimits::default();
        let registry = known_algorithm_registry(limits);
        let sum4 = super::super::reference::reduction_sum_program(5);
        assert_eq!(
            registry.recognize(&sum4, limits),
            None,
            "different length => different probe shape => unrecognized"
        );
    }

    #[test]
    fn constant_programs_have_signatures_too() {
        let limits = VerificationLimits::default();
        let program = ResearchProgram::expression(
            vec![],
            Section::new(vec![Op::Const(ScalarValue::F64(42.0))]),
            vec![0],
        );
        let signature = behavioral_signature(&program, limits).unwrap();
        assert_eq!(signature.digest.len(), 64);
        let mut registry = AlgorithmRegistry::new();
        registry.register("the_answer", &program, limits);
        assert_eq!(registry.recognize(&program, limits), Some("the_answer"));
    }

    #[test]
    fn probe_tensors_are_deterministic_and_finite() {
        let value_type = ValueType::new(DType::F64, vec![4]);
        let left = probe_tensor(&value_type, 0);
        let right = probe_tensor(&value_type, 0);
        assert_eq!(left, right);
        assert!(left.data.iter().all(|value| value.is_finite()));
        let shifted = probe_tensor(&value_type, 1);
        assert_ne!(left.data, shifted.data);
    }
}
