//! Invocation-time validation, performed in full before any output element is
//! written.
//!
//! A [`PreparedReferenceKernel`] already knows the element type, the opcode, the
//! attribute payload and the exact length every operand and the result must
//! have — all established once, at preparation. What it cannot know is whether
//! the caller will actually hand it slices of those lengths. That is what this
//! module checks.
//!
//! # Ordering guarantee
//!
//! [`validate`] runs every check and returns on the first failure, so a rejected
//! invocation leaves the output buffer **bit-for-bit unchanged**: the caller's
//! `&mut [f32]` is never written before validation completes. `execute` calls
//! `validate` before dispatching to any kernel body, and the kernel bodies
//! themselves write only after the dispatch.

use scirust_compute::DType;

use crate::error::ReferenceExecutionError;
use crate::format::ReferenceOpcode;
use crate::interpreter::{PreparedAttributes, PreparedReferenceKernel};

/// Validates one invocation against its prepared kernel.
///
/// Checks, in this order:
///
/// 1. the opcode is executable by this interpreter;
/// 2. the prepared attribute payload is coherent with that opcode;
/// 3. the element type is `F32`;
/// 4. the operand count matches exactly;
/// 5. every operand length matches exactly;
/// 6. the output length matches exactly.
///
/// Steps 1 to 3 are re-checked here even though preparation already established
/// them: re-verifying an invariant costs one `match` and keeps the guarantee
/// local to the function that is about to write, rather than inherited from a
/// distant constructor.
pub(crate) fn validate(
    prepared: &PreparedReferenceKernel,
    operands: &[&[f32]],
    output: &[f32],
) -> Result<(), ReferenceExecutionError> {
    let opcode = prepared.opcode();

    // 1. Executable opcode. `Exp` and `Log` are rejected at preparation, so a
    //    prepared kernel should never carry them; re-checked rather than
    //    assumed.
    match opcode
    {
        ReferenceOpcode::Relu
        | ReferenceOpcode::Scale
        | ReferenceOpcode::Add
        | ReferenceOpcode::Sub
        | ReferenceOpcode::Mul
        | ReferenceOpcode::Div
        | ReferenceOpcode::ShapeCopy
        | ReferenceOpcode::Permute =>
        {},
        ReferenceOpcode::Exp | ReferenceOpcode::Log =>
        {
            return Err(ReferenceExecutionError::DeterministicMathUnavailable { opcode });
        },
    }

    // 2. Attribute payload coherent with the opcode.
    let attributes_match = matches!(
        (opcode, prepared.attributes()),
        (ReferenceOpcode::Scale, PreparedAttributes::Scale { .. })
            | (ReferenceOpcode::Permute, PreparedAttributes::Permute { .. })
            | (
                ReferenceOpcode::Relu
                    | ReferenceOpcode::Add
                    | ReferenceOpcode::Sub
                    | ReferenceOpcode::Mul
                    | ReferenceOpcode::Div
                    | ReferenceOpcode::ShapeCopy,
                PreparedAttributes::None,
            )
    );
    if !attributes_match
    {
        return Err(ReferenceExecutionError::AttributeMismatch { opcode });
    }

    // 3. Element type.
    let dtype = prepared.dtype();
    if dtype != DType::F32
    {
        return Err(ReferenceExecutionError::UnsupportedDType { dtype });
    }

    // 4. Operand count.
    let expected_operands = prepared.operand_lengths().len();
    if operands.len() != expected_operands
    {
        return Err(ReferenceExecutionError::OperandCountMismatch {
            expected: expected_operands,
            actual: operands.len(),
        });
    }

    // 5. Operand lengths, in operand order.
    for (operand_index, (operand, &expected)) in
        operands.iter().zip(prepared.operand_lengths()).enumerate()
    {
        if operand.len() != expected
        {
            return Err(ReferenceExecutionError::OperandLengthMismatch {
                operand_index,
                expected,
                actual: operand.len(),
            });
        }
    }

    // 6. Output length.
    if output.len() != prepared.result_length()
    {
        return Err(ReferenceExecutionError::OutputLengthMismatch {
            expected: prepared.result_length(),
            actual: output.len(),
        });
    }

    Ok(())
}
