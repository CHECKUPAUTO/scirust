//! Safe execution of verified tensor programs.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use scirust_tensor_core::TensorND;
use scirust_tensor_einsum::einsum;

use super::ir::{TensorInstruction, TensorProgram};
use super::verify::{ProgramError, VerificationLimits, VerifiedProgram, verify_program};

/// Successful execution of a tensor program.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionResult {
    /// Owned copy of the selected program output.
    pub output: TensorND,

    /// Number of active instructions that were actually evaluated.
    pub executed_instructions: usize,

    /// Static verification information used by the interpreter.
    pub verified: VerifiedProgram,
}

/// Recoverable execution failure.
///
/// Generated programs and externally supplied tensors never need to cause a
/// panic: malformed inputs, invalid programs and non-finite results are
/// represented explicitly.
///
/// `Eq` is intentionally not derived because [`Self::NonFiniteScaleFactor`]
/// carries an `f32`.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionError {
    Verification(ProgramError),

    InvalidInputTensor {
        input: usize,
        reason: String,
    },

    NonFiniteInput {
        input: usize,
        element: usize,
    },

    MissingRegister {
        node: usize,
        source: usize,
    },

    /// An active `Scale` instruction specifies a non-finite constant factor.
    ///
    /// This is a defect in the program itself and is reported independently of
    /// the register contents, so an empty tensor scaled by a non-finite factor
    /// is still rejected.
    NonFiniteScaleFactor {
        node: usize,
        factor: f32,
    },

    Backend {
        node: usize,
        message: String,
    },

    NonFiniteResult {
        node: usize,
        element: usize,
    },

    MissingOutput {
        output: usize,
    },
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Verification(error) => write!(formatter, "program verification failed: {error}"),
            Self::InvalidInputTensor { input, reason } =>
            {
                write!(formatter, "input tensor {input} is inconsistent: {reason}")
            },
            Self::NonFiniteInput { input, element } => write!(
                formatter,
                "input tensor {input} contains a non-finite value at element {element}"
            ),
            Self::MissingRegister { node, source } => write!(
                formatter,
                "node {node} requires unavailable register {source}"
            ),
            Self::NonFiniteScaleFactor { node, factor } => write!(
                formatter,
                "node {node} scales by a non-finite factor {factor}"
            ),
            Self::Backend { node, message } =>
            {
                write!(formatter, "tensor backend failed at node {node}: {message}")
            },
            Self::NonFiniteResult { node, element } => write!(
                formatter,
                "node {node} produced a non-finite value at element {element}"
            ),
            Self::MissingOutput { output } =>
            {
                write!(formatter, "output register {output} was not materialised")
            },
        }
    }
}

impl Error for ExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self
        {
            Self::Verification(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ProgramError> for ExecutionError {
    fn from(error: ProgramError) -> Self {
        Self::Verification(error)
    }
}

/// Verify and execute a generated tensor program.
///
/// Inputs are borrowed rather than cloned. Only active computed registers own
/// newly allocated tensors. Inactive instructions are represented by `None`
/// and perform no numerical work.
pub fn execute_program<'a>(
    program: &TensorProgram,
    inputs: &'a [TensorND],
    limits: VerificationLimits,
) -> Result<ExecutionResult, ExecutionError> {
    let input_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();

    let verified = verify_program(program, &input_shapes, limits)?;

    validate_active_inputs(program, &verified.active, inputs)?;

    let mut registers: Vec<Option<Cow<'a, TensorND>>> =
        Vec::with_capacity(program.instructions.len());
    let mut executed_instructions = 0usize;

    for (node, instruction) in program.instructions.iter().enumerate()
    {
        if !verified.active[node]
        {
            registers.push(None);
            continue;
        }

        let value: Cow<'a, TensorND> = match instruction
        {
            TensorInstruction::Input { input } =>
            {
                let tensor = inputs.get(*input).ok_or_else(|| ExecutionError::Backend {
                    node,
                    message: format!("verified input index {input} is no longer available"),
                })?;

                Cow::Borrowed(tensor)
            },

            TensorInstruction::Add { lhs, rhs } =>
            {
                let left = register_ref(&registers, node, *lhs)?;
                let right = register_ref(&registers, node, *rhs)?;

                let data = left
                    .data
                    .iter()
                    .zip(&right.data)
                    .map(|(left_value, right_value)| left_value + right_value)
                    .collect();

                Cow::Owned(create_tensor(node, data, left.shape.clone())?)
            },

            TensorInstruction::MatMul { lhs, rhs } =>
            {
                let left = register_ref(&registers, node, *lhs)?;
                let right = register_ref(&registers, node, *rhs)?;

                // The verifier guarantees both operands are rank two with a
                // matching inner dimension.
                let rows = left.shape[0];
                let columns = right.shape[1];

                // The einsum backend indexes operand data unconditionally and
                // panics on any zero-length dimension. Compute the result
                // directly instead: a matmul with an empty contraction axis or
                // an empty outer axis is the all-zeros tensor of shape
                // `[rows, columns]` (an empty sum is zero), which is empty
                // whenever `rows` or `columns` is zero.
                let result = if left.data.is_empty() || right.data.is_empty()
                {
                    let elements =
                        rows.checked_mul(columns)
                            .ok_or_else(|| ExecutionError::Backend {
                                node,
                                message: format!(
                                    "matmul output shape [{rows}, {columns}] overflows usize"
                                ),
                            })?;
                    create_tensor(node, vec![0.0; elements], vec![rows, columns])?
                }
                else
                {
                    einsum("ij,jk->ik", &[left, right])
                        .map_err(|message| ExecutionError::Backend { node, message })?
                };

                Cow::Owned(result)
            },

            TensorInstruction::Transpose2d { src } =>
            {
                let source = register_ref(&registers, node, *src)?;

                // The verifier guarantees a rank-two source.
                let rows = source.shape[0];
                let columns = source.shape[1];

                // As with matmul, guard the einsum backend against zero-length
                // dimensions. A transpose of an empty matrix is the empty
                // matrix of the transposed shape.
                let result = if source.data.is_empty()
                {
                    create_tensor(node, Vec::new(), vec![columns, rows])?
                }
                else
                {
                    einsum("ij->ji", &[source])
                        .map_err(|message| ExecutionError::Backend { node, message })?
                };

                Cow::Owned(result)
            },

            TensorInstruction::Relu { src } =>
            {
                let source = register_ref(&registers, node, *src)?;

                let data = source.data.iter().map(|value| value.max(0.0)).collect();

                Cow::Owned(create_tensor(node, data, source.shape.clone())?)
            },

            TensorInstruction::Scale { src, factor } =>
            {
                if !factor.is_finite()
                {
                    return Err(ExecutionError::NonFiniteScaleFactor {
                        node,
                        factor: *factor,
                    });
                }

                let source = register_ref(&registers, node, *src)?;

                let data = source.data.iter().map(|value| value * factor).collect();

                Cow::Owned(create_tensor(node, data, source.shape.clone())?)
            },
        };

        if let Some(element) = first_non_finite(&value.data)
        {
            return Err(ExecutionError::NonFiniteResult { node, element });
        }

        registers.push(Some(value));
        executed_instructions += 1;
    }

    let output = registers
        .get(program.output)
        .and_then(Option::as_ref)
        .ok_or(ExecutionError::MissingOutput {
            output: program.output,
        })?
        .clone()
        .into_owned();

    Ok(ExecutionResult {
        output,
        executed_instructions,
        verified,
    })
}

/// Validate the layout and finiteness of every input tensor that is actually
/// read by the program.
///
/// Contract: only inputs referenced by an **active** `Input` instruction are
/// validated. A dead branch performs no numerical validation merely because an
/// `Input` instruction exists, so a malformed or non-finite tensor that only
/// feeds dead code is ignored. This mirrors execution, which never reads such a
/// tensor. Static shape reasoning in `verify_program` is unaffected: it may
/// still consult the declared shape of every input register.
fn validate_active_inputs(
    program: &TensorProgram,
    active: &[bool],
    inputs: &[TensorND],
) -> Result<(), ExecutionError> {
    let mut referenced = vec![false; inputs.len()];

    for (node, instruction) in program.instructions.iter().enumerate()
    {
        if !active.get(node).copied().unwrap_or(false)
        {
            continue;
        }

        if let TensorInstruction::Input { input } = *instruction
        {
            if let Some(is_referenced) = referenced.get_mut(input)
            {
                *is_referenced = true;
            }
        }
    }

    for (input, tensor) in inputs.iter().enumerate()
    {
        if !referenced[input]
        {
            continue;
        }

        validate_tensor_layout(tensor)
            .map_err(|reason| ExecutionError::InvalidInputTensor { input, reason })?;

        if let Some(element) = first_non_finite(&tensor.data)
        {
            return Err(ExecutionError::NonFiniteInput { input, element });
        }
    }

    Ok(())
}

fn validate_tensor_layout(tensor: &TensorND) -> Result<(), String> {
    let expected_elements = tensor
        .shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| format!("shape product overflows usize: {:?}", tensor.shape))?;

    if tensor.data.len() != expected_elements
    {
        return Err(format!(
            "data length {} does not match shape {:?}, which requires {} elements",
            tensor.data.len(),
            tensor.shape,
            expected_elements
        ));
    }

    let expected_strides = checked_row_major_strides(&tensor.shape).ok_or_else(|| {
        format!(
            "row-major stride computation overflows for shape {:?}",
            tensor.shape
        )
    })?;

    if tensor.strides != expected_strides
    {
        return Err(format!(
            "strides {:?} do not match expected row-major strides {:?}",
            tensor.strides, expected_strides
        ));
    }

    Ok(())
}

fn checked_row_major_strides(shape: &[usize]) -> Option<Vec<usize>> {
    let mut strides = vec![1usize; shape.len()];

    if shape.len() <= 1
    {
        return Some(strides);
    }

    for axis in (0..shape.len() - 1).rev()
    {
        strides[axis] = strides[axis + 1].checked_mul(shape[axis + 1])?;
    }

    Some(strides)
}

fn register_ref<'registers, 'tensor>(
    registers: &'registers [Option<Cow<'tensor, TensorND>>],
    node: usize,
    source: usize,
) -> Result<&'registers TensorND, ExecutionError> {
    registers
        .get(source)
        .and_then(Option::as_ref)
        .map(Cow::as_ref)
        .ok_or(ExecutionError::MissingRegister { node, source })
}

fn create_tensor(
    node: usize,
    data: Vec<f32>,
    shape: Vec<usize>,
) -> Result<TensorND, ExecutionError> {
    TensorND::try_new(data, shape).map_err(|message| ExecutionError::Backend { node, message })
}

fn first_non_finite(data: &[f32]) -> Option<usize> {
    data.iter().position(|value| !value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: &[f32], shape: &[usize]) -> TensorND {
        TensorND::new(data.to_vec(), shape.to_vec())
    }

    #[test]
    fn executes_matmul_and_add_with_exact_oracle() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Input { input: 2 },
                TensorInstruction::Add { lhs: 2, rhs: 3 },
            ],
            4,
        );

        let inputs = vec![
            tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
            tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]),
        ];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.shape, vec![2, 2]);
        assert_eq!(result.output.data, vec![59.0, 65.0, 140.0, 155.0]);
        assert_eq!(result.executed_instructions, 5);
    }

    #[test]
    fn executes_transpose_with_exact_oracle() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Transpose2d { src: 0 },
            ],
            1,
        );

        let inputs = vec![tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.shape, vec![3, 2]);
        assert_eq!(result.output.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn skips_dead_instructions_during_execution() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::Scale {
                    src: 1,
                    factor: 1000.0,
                },
                TensorInstruction::Relu { src: 0 },
            ],
            3,
        );

        let inputs = vec![tensor(&[-2.0, 3.0], &[2]), tensor(&[10.0, 20.0], &[2])];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.data, vec![0.0, 3.0]);
        assert_eq!(result.verified.active, vec![true, false, false, true]);
        assert_eq!(result.executed_instructions, 2);
    }

    #[test]
    fn rejects_non_finite_referenced_input() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);

        let inputs = vec![tensor(&[1.0, f32::NAN], &[2])];

        assert_eq!(
            execute_program(&program, &inputs, VerificationLimits::default()),
            Err(ExecutionError::NonFiniteInput {
                input: 0,
                element: 1,
            })
        );
    }

    #[test]
    fn ignores_unreferenced_extra_input() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);

        let inputs = vec![tensor(&[2.0], &[1]), tensor(&[f32::NAN], &[1])];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.data, vec![2.0]);
    }

    #[test]
    fn rejects_active_non_finite_scale_factor() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: f32::INFINITY,
                },
            ],
            1,
        );

        let inputs = vec![tensor(&[2.0], &[1])];

        assert_eq!(
            execute_program(&program, &inputs, VerificationLimits::default()),
            Err(ExecutionError::NonFiniteScaleFactor {
                node: 1,
                factor: f32::INFINITY,
            })
        );
    }

    #[test]
    fn rejects_generated_overflow_to_infinity() {
        // Finite inputs and a finite factor whose product overflows f32.
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 1.0e30,
                },
            ],
            1,
        );

        let inputs = vec![tensor(&[1.0e30], &[1])];

        assert_eq!(
            execute_program(&program, &inputs, VerificationLimits::default()),
            Err(ExecutionError::NonFiniteResult {
                node: 1,
                element: 0,
            })
        );
    }

    #[test]
    fn ignores_dead_input_containing_nan() {
        // Instruction 1 exists but is dead (output is register 0). Its tensor
        // contains NaN, yet a dead branch must not trigger numerical
        // validation.
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
            ],
            0,
        );

        let inputs = vec![tensor(&[7.0, 8.0], &[2]), tensor(&[1.0, f32::NAN], &[2])];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.data, vec![7.0, 8.0]);
        assert_eq!(result.verified.active, vec![true, false]);
        assert_eq!(result.executed_instructions, 1);
    }

    #[test]
    fn ignores_dead_branch_with_infinite_scale_factor() {
        // The Scale on register 1 is dead; its infinite factor never applies.
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::Scale {
                    src: 1,
                    factor: f32::INFINITY,
                },
            ],
            0,
        );

        let inputs = vec![tensor(&[3.0], &[1]), tensor(&[4.0], &[1])];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.data, vec![3.0]);
        assert_eq!(result.verified.active, vec![true, false, false]);
    }

    #[test]
    fn executes_scalar_input() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.5,
                },
            ],
            1,
        );

        let inputs = vec![TensorND::scalar(4.0)];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.shape, Vec::<usize>::new());
        assert_eq!(result.output.data, vec![10.0]);
    }

    #[test]
    fn executes_empty_dimension_tensors_without_panic() {
        // Relu on an empty column vector, plus a matmul whose contraction axis
        // has length zero (result is the all-zeros [2, 2] tensor), and a
        // transpose of an empty matrix.
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },        // [2, 0]
                TensorInstruction::Input { input: 1 },        // [0, 2]
                TensorInstruction::Relu { src: 0 },           // [2, 0], empty
                TensorInstruction::MatMul { lhs: 0, rhs: 1 }, // [2, 2] zeros
                TensorInstruction::Transpose2d { src: 2 },    // [0, 2], empty
                TensorInstruction::Add { lhs: 3, rhs: 3 },    // [2, 2] zeros
            ],
            3,
        );

        let empty_rows = TensorND::new(Vec::new(), vec![2, 0]);
        let empty_cols = TensorND::new(Vec::new(), vec![0, 2]);
        let inputs = vec![empty_rows, empty_cols];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.shape, vec![2, 2]);
        assert_eq!(result.output.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn output_equal_to_borrowed_input() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);

        let inputs = vec![tensor(&[5.0, 6.0, 7.0], &[3])];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.data, vec![5.0, 6.0, 7.0]);
        assert_eq!(result.output.shape, vec![3]);
        assert_eq!(result.executed_instructions, 1);
    }

    #[test]
    fn rejects_missing_required_input() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);

        let inputs: Vec<TensorND> = Vec::new();

        assert_eq!(
            execute_program(&program, &inputs, VerificationLimits::default()),
            Err(ExecutionError::Verification(
                ProgramError::InputOutOfBounds {
                    node: 0,
                    input: 0,
                    available_inputs: 0,
                }
            ))
        );
    }

    #[test]
    fn repeated_execution_is_bit_identical() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Relu { src: 2 },
                TensorInstruction::Scale {
                    src: 3,
                    factor: 0.25,
                },
            ],
            4,
        );

        let inputs = vec![
            tensor(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]),
            tensor(&[7.0, -8.0, 9.0, -10.0, 11.0, -12.0], &[3, 2]),
        ];

        let first = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();
        let second = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(first.output.data, second.output.data);
        assert_eq!(first.output.shape, second.output.shape);
        for (left, right) in first.output.data.iter().zip(&second.output.data)
        {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    #[test]
    fn runtime_output_shape_matches_verified_output_shape() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Input { input: 1 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Transpose2d { src: 2 },
            ],
            3,
        );

        let inputs = vec![
            tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]),
        ];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.shape, result.verified.output_shape);
    }

    #[test]
    fn rejects_inconsistent_public_tensor_fields() {
        let program = TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0);

        let mut malformed = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        malformed.strides[0] = 1;

        let error =
            execute_program(&program, &[malformed], VerificationLimits::default()).unwrap_err();

        assert!(matches!(
            error,
            ExecutionError::InvalidInputTensor { input: 0, .. }
        ));
    }

    #[test]
    fn output_can_precede_dead_trailing_instructions() {
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Relu { src: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 50.0,
                },
            ],
            1,
        );

        let inputs = vec![tensor(&[-1.0, 4.0], &[2])];

        let result = execute_program(&program, &inputs, VerificationLimits::default()).unwrap();

        assert_eq!(result.output.data, vec![0.0, 4.0]);
        assert_eq!(result.executed_instructions, 2);
        assert_eq!(result.verified.active, vec![true, true, false]);
    }
}
