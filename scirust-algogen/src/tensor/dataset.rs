//! Deterministic evaluation datasets with exact expected tensor outputs.
//!
//! A [`Dataset`] is a fixed list of cases, each pairing concrete input tensors
//! with the exact tensor the program is expected to produce. All cases must
//! share the same input arity and shapes, and every expected tensor must be a
//! well-formed, finite tensor so that the correctness loss can never become
//! `NaN` through the oracle.

use std::error::Error;
use std::fmt;

use scirust_tensor_core::TensorND;

/// One evaluation case: inputs and the exact expected output.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorCase {
    pub inputs: Vec<TensorND>,
    pub expected: TensorND,
}

impl TensorCase {
    pub fn new(inputs: Vec<TensorND>, expected: TensorND) -> Self {
        Self { inputs, expected }
    }
}

/// A validated collection of evaluation cases sharing one input signature.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    cases: Vec<TensorCase>,
    input_shapes: Vec<Vec<usize>>,
}

/// A deterministic dataset construction failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetError {
    /// No cases were supplied.
    Empty,

    /// Case `case` supplies a different number of inputs than the first case.
    InconsistentInputArity {
        case: usize,
        expected: usize,
        found: usize,
    },

    /// Case `case` input `input` has a different shape than the first case.
    InconsistentInputShape { case: usize, input: usize },

    /// A tensor's declared shape and data length are inconsistent.
    MalformedTensor { case: usize, reason: String },

    /// The expected tensor of `case` contains a non-finite value.
    NonFiniteExpected { case: usize, element: usize },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Empty => write!(formatter, "dataset contains no cases"),
            Self::InconsistentInputArity {
                case,
                expected,
                found,
            } => write!(
                formatter,
                "case {case} supplies {found} inputs but {expected} were expected"
            ),
            Self::InconsistentInputShape { case, input } => write!(
                formatter,
                "case {case} input {input} has an inconsistent shape"
            ),
            Self::MalformedTensor { case, reason } =>
            {
                write!(formatter, "case {case} has a malformed tensor: {reason}")
            },
            Self::NonFiniteExpected { case, element } => write!(
                formatter,
                "case {case} expected tensor has a non-finite value at element {element}"
            ),
        }
    }
}

impl Error for DatasetError {}

impl Dataset {
    /// Validate `cases` and build a dataset.
    ///
    /// Every case must share the input arity and per-input shape of the first
    /// case, every tensor must have a data length matching its shape, and every
    /// expected tensor must be finite.
    pub fn new(cases: Vec<TensorCase>) -> Result<Self, DatasetError> {
        let first = cases.first().ok_or(DatasetError::Empty)?;
        let input_shapes: Vec<Vec<usize>> = first
            .inputs
            .iter()
            .map(|tensor| tensor.shape.clone())
            .collect();

        for (case_index, case) in cases.iter().enumerate()
        {
            if case.inputs.len() != input_shapes.len()
            {
                return Err(DatasetError::InconsistentInputArity {
                    case: case_index,
                    expected: input_shapes.len(),
                    found: case.inputs.len(),
                });
            }

            for (input_index, tensor) in case.inputs.iter().enumerate()
            {
                check_tensor(case_index, tensor)?;
                if tensor.shape != input_shapes[input_index]
                {
                    return Err(DatasetError::InconsistentInputShape {
                        case: case_index,
                        input: input_index,
                    });
                }
            }

            check_tensor(case_index, &case.expected)?;
            if let Some(element) = case
                .expected
                .data
                .iter()
                .position(|value| !value.is_finite())
            {
                return Err(DatasetError::NonFiniteExpected {
                    case: case_index,
                    element,
                });
            }
        }

        Ok(Self {
            cases,
            input_shapes,
        })
    }

    /// Shared input shapes of every case.
    pub fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }

    /// The evaluation cases.
    pub fn cases(&self) -> &[TensorCase] {
        &self.cases
    }

    /// Number of cases.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// A dataset is never empty once constructed, so this is always `false`;
    /// provided for API completeness alongside [`Self::len`].
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }
}

/// Verify that a tensor's data length matches the product of its shape.
fn check_tensor(case: usize, tensor: &TensorND) -> Result<(), DatasetError> {
    let expected = tensor
        .shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension));

    match expected
    {
        Some(expected) if expected == tensor.data.len() => Ok(()),
        Some(expected) => Err(DatasetError::MalformedTensor {
            case,
            reason: format!(
                "shape {:?} implies {expected} elements but data length is {}",
                tensor.shape,
                tensor.data.len()
            ),
        }),
        None => Err(DatasetError::MalformedTensor {
            case,
            reason: format!("shape {:?} overflows usize", tensor.shape),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: &[f32], shape: &[usize]) -> TensorND {
        TensorND::new(data.to_vec(), shape.to_vec())
    }

    #[test]
    fn accepts_consistent_cases() {
        let dataset = Dataset::new(vec![
            TensorCase::new(vec![tensor(&[1.0, 2.0], &[2])], tensor(&[1.0, 2.0], &[2])),
            TensorCase::new(vec![tensor(&[3.0, 4.0], &[2])], tensor(&[3.0, 4.0], &[2])),
        ])
        .unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.input_shapes(), &[vec![2]]);
    }

    #[test]
    fn rejects_empty_dataset() {
        assert_eq!(Dataset::new(Vec::new()), Err(DatasetError::Empty));
    }

    #[test]
    fn rejects_inconsistent_input_shape() {
        let error = Dataset::new(vec![
            TensorCase::new(vec![tensor(&[1.0, 2.0], &[2])], tensor(&[1.0], &[1])),
            TensorCase::new(vec![tensor(&[1.0, 2.0, 3.0], &[3])], tensor(&[1.0], &[1])),
        ])
        .unwrap_err();
        assert_eq!(
            error,
            DatasetError::InconsistentInputShape { case: 1, input: 0 }
        );
    }

    #[test]
    fn rejects_non_finite_expected() {
        let error = Dataset::new(vec![TensorCase::new(
            vec![tensor(&[1.0], &[1])],
            tensor(&[f32::NAN], &[1]),
        )])
        .unwrap_err();
        assert_eq!(
            error,
            DatasetError::NonFiniteExpected {
                case: 0,
                element: 0,
            }
        );
    }
}
