//! Static validation and shape inference for generated tensor programs.

use std::error::Error;
use std::fmt;

use super::active::analyze_active;
use super::ir::{TensorInstruction, TensorProgram};

/// Resource limits applied before a generated program may be executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationLimits {
    pub max_instructions: usize,
    pub max_rank: usize,
    pub max_elements_per_tensor: usize,
    pub max_total_register_elements: usize,
}

impl Default for VerificationLimits {
    fn default() -> Self {
        Self {
            max_instructions: 1_024,
            max_rank: 8,
            max_elements_per_tensor: 16 * 1024 * 1024,
            max_total_register_elements: 64 * 1024 * 1024,
        }
    }
}

/// Successful static analysis of a tensor program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedProgram {
    /// Inferred shape of every register.
    pub register_shapes: Vec<Vec<usize>>,

    /// Backward liveness map relative to `TensorProgram::output`.
    pub active: Vec<bool>,

    /// Inferred shape of the selected output.
    pub output_shape: Vec<usize>,

    /// Sum of the element counts represented by all registers.
    pub total_register_elements: usize,
}

impl VerifiedProgram {
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&value| value).count()
    }
}

/// A deterministic validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramError {
    EmptyProgram,

    OutputOutOfBounds {
        output: usize,
        instructions: usize,
    },

    TooManyInstructions {
        instructions: usize,
        maximum: usize,
    },

    InputOutOfBounds {
        node: usize,
        input: usize,
        available_inputs: usize,
    },

    NonCausalDependency {
        node: usize,
        source: usize,
    },

    RankLimitExceeded {
        node: usize,
        rank: usize,
        maximum: usize,
    },

    ShapeProductOverflow {
        node: usize,
        shape: Vec<usize>,
    },

    TensorTooLarge {
        node: usize,
        elements: usize,
        maximum: usize,
    },

    TotalRegisterElementsOverflow {
        node: usize,
    },

    TotalRegisterElementsLimitExceeded {
        node: usize,
        elements: usize,
        maximum: usize,
    },

    AddShapeMismatch {
        node: usize,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },

    MatMulRankMismatch {
        node: usize,
        lhs_rank: usize,
        rhs_rank: usize,
    },

    MatMulShapeMismatch {
        node: usize,
        lhs_columns: usize,
        rhs_rows: usize,
    },

    TransposeRankMismatch {
        node: usize,
        rank: usize,
    },
}

impl fmt::Display for ProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyProgram => write!(formatter, "tensor program is empty"),
            Self::OutputOutOfBounds {
                output,
                instructions,
            } => write!(
                formatter,
                "output register {output} is out of bounds for {instructions} instructions"
            ),
            Self::TooManyInstructions {
                instructions,
                maximum,
            } => write!(
                formatter,
                "program contains {instructions} instructions, exceeding limit {maximum}"
            ),
            Self::InputOutOfBounds {
                node,
                input,
                available_inputs,
            } => write!(
                formatter,
                "node {node} requests input {input}, but only {available_inputs} inputs exist"
            ),
            Self::NonCausalDependency { node, source } => write!(
                formatter,
                "node {node} reads register {source}, which is not strictly earlier"
            ),
            Self::RankLimitExceeded {
                node,
                rank,
                maximum,
            } => write!(
                formatter,
                "node {node} has rank {rank}, exceeding rank limit {maximum}"
            ),
            Self::ShapeProductOverflow { node, shape } =>
            {
                write!(formatter, "node {node} shape product overflows: {shape:?}")
            },
            Self::TensorTooLarge {
                node,
                elements,
                maximum,
            } => write!(
                formatter,
                "node {node} represents {elements} elements, exceeding limit {maximum}"
            ),
            Self::TotalRegisterElementsOverflow { node } => write!(
                formatter,
                "total register element count overflows while processing node {node}"
            ),
            Self::TotalRegisterElementsLimitExceeded {
                node,
                elements,
                maximum,
            } => write!(
                formatter,
                "node {node} raises total register elements to {elements}, exceeding limit {maximum}"
            ),
            Self::AddShapeMismatch {
                node,
                lhs_shape,
                rhs_shape,
            } => write!(
                formatter,
                "node {node} cannot add shapes {lhs_shape:?} and {rhs_shape:?}"
            ),
            Self::MatMulRankMismatch {
                node,
                lhs_rank,
                rhs_rank,
            } => write!(
                formatter,
                "node {node} matrix multiplication requires rank 2, got ranks {lhs_rank} and {rhs_rank}"
            ),
            Self::MatMulShapeMismatch {
                node,
                lhs_columns,
                rhs_rows,
            } => write!(
                formatter,
                "node {node} matrix multiplication has incompatible inner dimensions {lhs_columns} and {rhs_rows}"
            ),
            Self::TransposeRankMismatch { node, rank } => write!(
                formatter,
                "node {node} transpose requires rank 2, got rank {rank}"
            ),
        }
    }
}

impl Error for ProgramError {}

/// Validate register causality, infer shapes, enforce resource limits and
/// compute output liveness.
pub fn verify_program(
    program: &TensorProgram,
    input_shapes: &[Vec<usize>],
    limits: VerificationLimits,
) -> Result<VerifiedProgram, ProgramError> {
    if program.instructions.is_empty()
    {
        return Err(ProgramError::EmptyProgram);
    }

    if program.instructions.len() > limits.max_instructions
    {
        return Err(ProgramError::TooManyInstructions {
            instructions: program.instructions.len(),
            maximum: limits.max_instructions,
        });
    }

    if program.output >= program.instructions.len()
    {
        return Err(ProgramError::OutputOutOfBounds {
            output: program.output,
            instructions: program.instructions.len(),
        });
    }

    let mut register_shapes: Vec<Vec<usize>> = Vec::with_capacity(program.instructions.len());
    let mut total_register_elements = 0usize;

    for (node, instruction) in program.instructions.iter().enumerate()
    {
        let shape = match *instruction
        {
            TensorInstruction::Input { input } =>
            {
                let shape = input_shapes
                    .get(input)
                    .ok_or(ProgramError::InputOutOfBounds {
                        node,
                        input,
                        available_inputs: input_shapes.len(),
                    })?;
                shape.clone()
            },

            TensorInstruction::Add { lhs, rhs } =>
            {
                validate_source(node, lhs)?;
                validate_source(node, rhs)?;

                let lhs_shape = &register_shapes[lhs];
                let rhs_shape = &register_shapes[rhs];

                if lhs_shape != rhs_shape
                {
                    return Err(ProgramError::AddShapeMismatch {
                        node,
                        lhs_shape: lhs_shape.clone(),
                        rhs_shape: rhs_shape.clone(),
                    });
                }

                lhs_shape.clone()
            },

            TensorInstruction::MatMul { lhs, rhs } =>
            {
                validate_source(node, lhs)?;
                validate_source(node, rhs)?;

                let lhs_shape = &register_shapes[lhs];
                let rhs_shape = &register_shapes[rhs];

                if lhs_shape.len() != 2 || rhs_shape.len() != 2
                {
                    return Err(ProgramError::MatMulRankMismatch {
                        node,
                        lhs_rank: lhs_shape.len(),
                        rhs_rank: rhs_shape.len(),
                    });
                }

                if lhs_shape[1] != rhs_shape[0]
                {
                    return Err(ProgramError::MatMulShapeMismatch {
                        node,
                        lhs_columns: lhs_shape[1],
                        rhs_rows: rhs_shape[0],
                    });
                }

                vec![lhs_shape[0], rhs_shape[1]]
            },

            TensorInstruction::Transpose2d { src } =>
            {
                validate_source(node, src)?;

                let source_shape = &register_shapes[src];

                if source_shape.len() != 2
                {
                    return Err(ProgramError::TransposeRankMismatch {
                        node,
                        rank: source_shape.len(),
                    });
                }

                vec![source_shape[1], source_shape[0]]
            },

            TensorInstruction::Relu { src } | TensorInstruction::Scale { src, .. } =>
            {
                validate_source(node, src)?;
                register_shapes[src].clone()
            },
        };

        let elements = validate_shape(node, &shape, limits)?;

        total_register_elements = total_register_elements
            .checked_add(elements)
            .ok_or(ProgramError::TotalRegisterElementsOverflow { node })?;

        if total_register_elements > limits.max_total_register_elements
        {
            return Err(ProgramError::TotalRegisterElementsLimitExceeded {
                node,
                elements: total_register_elements,
                maximum: limits.max_total_register_elements,
            });
        }

        register_shapes.push(shape);
    }

    let active = analyze_active(program);
    let output_shape = register_shapes[program.output].clone();

    Ok(VerifiedProgram {
        register_shapes,
        active,
        output_shape,
        total_register_elements,
    })
}

fn validate_source(node: usize, source: usize) -> Result<(), ProgramError> {
    if source >= node
    {
        return Err(ProgramError::NonCausalDependency { node, source });
    }

    Ok(())
}

fn validate_shape(
    node: usize,
    shape: &[usize],
    limits: VerificationLimits,
) -> Result<usize, ProgramError> {
    if shape.len() > limits.max_rank
    {
        return Err(ProgramError::RankLimitExceeded {
            node,
            rank: shape.len(),
            maximum: limits.max_rank,
        });
    }

    let elements = shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension));

    let elements = elements.ok_or_else(|| ProgramError::ShapeProductOverflow {
        node,
        shape: shape.to_vec(),
    })?;

    if elements > limits.max_elements_per_tensor
    {
        return Err(ProgramError::TensorTooLarge {
            node,
            elements,
            maximum: limits.max_elements_per_tensor,
        });
    }

    Ok(elements)
}
