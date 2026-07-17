//! Deterministic structural cost model for tensor programs.
//!
//! Every metric is a pure function of the program structure and its statically
//! inferred shapes — never of wall-clock time. Costs are therefore bit-exactly
//! reproducible and safe to use as optimisation objectives. Integer accumulation
//! saturates instead of overflowing, so a pathological program yields a large
//! finite cost rather than wrapping or panicking.

use serde::{Deserialize, Serialize};

use super::ir::{TensorInstruction, TensorProgram};
use super::verify::VerifiedProgram;

/// Bytes occupied by one `f32` element.
const BYTES_PER_ELEMENT: u64 = 4;

/// Deterministic structural cost of a tensor program.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostReport {
    /// Number of instructions that contribute to the output.
    pub active_instructions: usize,

    /// Estimated floating-point operations across active instructions.
    pub estimated_flops: u64,

    /// Sum of element counts of all active registers.
    pub total_active_elements: u64,

    /// Maximum number of live tensor elements at any point during execution.
    pub peak_live_elements: u64,

    /// Bytes of intermediate (non-input) active tensors that must be
    /// materialised.
    pub generated_intermediate_bytes: u64,

    /// Number of instructions that do not contribute to the output.
    pub dead_instructions: usize,

    /// Fraction of instructions that are dead, in `[0, 1]`.
    pub bloat_ratio: f64,
}

impl CostReport {
    /// The worst-case cost, used for programs that cannot be statically
    /// evaluated. It is dominated by every genuine cost under the ordering in
    /// [`super::population`].
    pub fn unevaluable(total_instructions: usize) -> Self {
        Self {
            active_instructions: total_instructions,
            estimated_flops: u64::MAX,
            total_active_elements: u64::MAX,
            peak_live_elements: u64::MAX,
            generated_intermediate_bytes: u64::MAX,
            dead_instructions: total_instructions,
            bloat_ratio: 1.0,
        }
    }
}

/// Compute the structural cost of `program` from its verification result.
///
/// The FLOP model counts floating-point arithmetic only:
/// `Add`, `Scale` and `Relu` cost one operation per element; `MatMul` of
/// `[m, k] x [k, n]` costs `2 * m * k * n`; `Transpose2d` and `Input` perform no
/// arithmetic and cost zero.
pub fn estimate_cost(program: &TensorProgram, verified: &VerifiedProgram) -> CostReport {
    let shapes = &verified.register_shapes;
    let active = &verified.active;

    let total_instructions = program.instructions.len();
    let active_instructions = verified.active_count();
    let dead_instructions = total_instructions - active_instructions;
    let bloat_ratio = if total_instructions == 0
    {
        0.0
    }
    else
    {
        dead_instructions as f64 / total_instructions as f64
    };

    let mut estimated_flops = 0u64;
    let mut total_active_elements = 0u64;
    let mut generated_intermediate_bytes = 0u64;

    for (node, instruction) in program.instructions.iter().enumerate()
    {
        if !active[node]
        {
            continue;
        }

        let elements = element_count(&shapes[node]);
        total_active_elements = total_active_elements.saturating_add(elements);

        let op_flops = match *instruction
        {
            TensorInstruction::Input { .. } | TensorInstruction::Transpose2d { .. } => 0,
            TensorInstruction::Add { .. }
            | TensorInstruction::Relu { .. }
            | TensorInstruction::Scale { .. } => elements,
            TensorInstruction::MatMul { lhs, .. } =>
            {
                // shapes[node] = [m, n]; k is the shared inner dimension.
                let m = element_axis(&shapes[node], 0);
                let n = element_axis(&shapes[node], 1);
                let k = element_axis(&shapes[lhs], 1);
                2u64.saturating_mul(m).saturating_mul(k).saturating_mul(n)
            },
        };
        estimated_flops = estimated_flops.saturating_add(op_flops);

        if !matches!(instruction, TensorInstruction::Input { .. })
        {
            generated_intermediate_bytes = generated_intermediate_bytes
                .saturating_add(elements.saturating_mul(BYTES_PER_ELEMENT));
        }
    }

    let peak_live_elements = peak_live_elements(program, verified);

    CostReport {
        active_instructions,
        estimated_flops,
        total_active_elements,
        peak_live_elements,
        generated_intermediate_bytes,
        dead_instructions,
        bloat_ratio,
    }
}

/// Peak number of simultaneously live tensor elements during execution.
///
/// A register becomes live when its instruction executes and stays live until
/// its last active consumer; the output register stays live until the end. Only
/// active instructions execute, and the sources of an active instruction are
/// themselves active, so the sweep never touches dead registers.
fn peak_live_elements(program: &TensorProgram, verified: &VerifiedProgram) -> u64 {
    let shapes = &verified.register_shapes;
    let active = &verified.active;
    let length = program.instructions.len();

    let mut last_use = vec![0usize; length];
    for (node, &is_active) in active.iter().enumerate()
    {
        if is_active
        {
            last_use[node] = node;
        }
    }

    for (node, instruction) in program.instructions.iter().enumerate()
    {
        if !active[node]
        {
            continue;
        }
        instruction.for_each_source(|source| {
            if active[source]
            {
                last_use[source] = last_use[source].max(node);
            }
        });
    }

    // The output must remain live until the final active instruction retires.
    if let Some(last_active) = (0..length).rev().find(|&node| active[node])
    {
        last_use[program.output] = last_use[program.output].max(last_active);
    }

    let mut current = 0u64;
    let mut peak = 0u64;
    for node in 0..length
    {
        if !active[node]
        {
            continue;
        }

        current = current.saturating_add(element_count(&shapes[node]));
        peak = peak.max(current);

        for (register, &use_end) in last_use.iter().enumerate().take(node + 1)
        {
            if active[register] && use_end == node
            {
                current = current.saturating_sub(element_count(&shapes[register]));
            }
        }
    }

    peak
}

/// Number of elements described by `shape`, saturating on overflow.
fn element_count(shape: &[usize]) -> u64 {
    shape
        .iter()
        .try_fold(1u64, |product, &dimension| {
            product.checked_mul(dimension as u64)
        })
        .unwrap_or(u64::MAX)
}

/// Length of a specific axis as `u64`, or `0` if the axis is absent.
fn element_axis(shape: &[usize], axis: usize) -> u64 {
    shape.get(axis).map(|&value| value as u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::verify_program;
    use crate::tensor::{TensorInstruction, VerificationLimits};

    fn verified(program: &TensorProgram, input_shapes: &[Vec<usize>]) -> VerifiedProgram {
        verify_program(program, input_shapes, VerificationLimits::default()).unwrap()
    }

    #[test]
    fn exact_flop_oracle_for_each_operator() {
        // Add / Scale / Relu each cost one op per element (shape [2, 3] -> 6).
        let unary = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Relu { src: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.0,
                },
                TensorInstruction::Add { lhs: 1, rhs: 2 },
            ],
            3,
        );
        let v = verified(&unary, &[vec![2, 3]]);
        // Relu 6 + Scale 6 + Add 6 = 18 (Input contributes 0).
        assert_eq!(estimate_cost(&unary, &v).estimated_flops, 18);

        // MatMul [2, 3] x [3, 4] -> 2 * 2 * 3 * 4 = 48.
        let matmul = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
            ],
            2,
        );
        let v = verified(&matmul, &[vec![2, 3], vec![3, 4]]);
        assert_eq!(estimate_cost(&matmul, &v).estimated_flops, 48);

        // Transpose2d performs no arithmetic.
        let transpose = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Transpose2d { src: 0 },
            ],
            1,
        );
        let v = verified(&transpose, &[vec![2, 3]]);
        assert_eq!(estimate_cost(&transpose, &v).estimated_flops, 0);
    }

    #[test]
    fn dead_instructions_are_excluded_from_active_cost() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 9.0,
                }, // dead
                TensorInstruction::Relu { src: 0 },
            ],
            2,
        );
        let v = verified(&program, &[vec![2, 2]]);
        let cost = estimate_cost(&program, &v);

        assert_eq!(cost.active_instructions, 2);
        assert_eq!(cost.dead_instructions, 1);
        assert!((cost.bloat_ratio - 1.0 / 3.0).abs() < 1e-12);
        // Only Input (0 flops) and Relu (4 flops) are active.
        assert_eq!(cost.estimated_flops, 4);
        // Active elements: Input 4 + Relu 4 = 8 (the dead Scale is excluded).
        assert_eq!(cost.total_active_elements, 8);
        // Generated bytes: only the Relu tensor (4 elements * 4 bytes).
        assert_eq!(cost.generated_intermediate_bytes, 16);
    }

    #[test]
    fn exact_peak_live_oracle() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },        // [2, 3] = 6
                TensorInstruction::Input { input: 1 },        // [3, 2] = 6
                TensorInstruction::MatMul { lhs: 0, rhs: 1 }, // [2, 2] = 4
                TensorInstruction::Relu { src: 2 },           // [2, 2] = 4
            ],
            3,
        );
        let v = verified(&program, &[vec![2, 3], vec![3, 2]]);
        let cost = estimate_cost(&program, &v);

        // Peak occurs at the MatMul: inputs 6 + 6 plus the 4-element result = 16.
        assert_eq!(cost.peak_live_elements, 16);
        assert_eq!(cost.total_active_elements, 20);
        assert_eq!(cost.generated_intermediate_bytes, 32);
        assert_eq!(cost.dead_instructions, 0);
        assert_eq!(cost.bloat_ratio, 0.0);
    }

    #[test]
    fn cost_report_survives_serde() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);
        let v = verified(&program, &[vec![2, 2]]);
        let cost = estimate_cost(&program, &v);

        let json = serde_json::to_string(&cost).unwrap();
        let decoded: CostReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cost);
    }
}
