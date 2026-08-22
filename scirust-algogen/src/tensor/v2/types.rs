//! Value model for the V2 scientific-discovery IR.
//!
//! The IR is deliberately typed: every value carries a [`DType`] and a concrete
//! row-major [`ValueType::shape`]. Shapes are static: symbolic dimensions are
//! deferred (see `docs/SCIRUST_ALGOGEN_IR_V2_ARCHITECTURE.md` §12) so shape
//! inference stays total and deterministic.
//!
//! Dtype extension points (`F16`, `Bf16`, integer/index types) are documented
//! but intentionally unimplemented; labelling `f32` data as a low-precision
//! dtype without implementing its arithmetic is forbidden by the research
//! contract.

use serde::{Deserialize, Serialize};

/// Scalar element type of an IR value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary64.
    F64,
    /// Boolean mask element.
    Bool,
}

impl DType {
    /// Every dtype currently admitted by the IR.
    pub const ALL: [DType; 3] = [DType::F32, DType::F64, DType::Bool];

    /// Whether the dtype denotes a floating-point value.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Whether the dtype denotes a Boolean mask.
    pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Stable byte tag used by canonical encodings.
    pub const fn tag(self) -> u8 {
        match self
        {
            Self::F32 => 0,
            Self::F64 => 1,
            Self::Bool => 2,
        }
    }

    /// Decode a canonical byte tag.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag
        {
            0 => Some(Self::F32),
            1 => Some(Self::F64),
            2 => Some(Self::Bool),
            _ => None,
        }
    }

    /// Human-readable name.
    pub const fn name(self) -> &'static str {
        match self
        {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Bool => "bool",
        }
    }
}

/// A compile-time tensor type: element dtype plus concrete shape.
///
/// An empty shape is a rank-0 scalar holding exactly one element.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueType {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl ValueType {
    /// A scalar (rank-0) value.
    pub fn scalar(dtype: DType) -> Self {
        Self {
            dtype,
            shape: Vec::new(),
        }
    }

    /// An arbitrary typed shape.
    pub fn new(dtype: DType, shape: Vec<usize>) -> Self {
        Self { dtype, shape }
    }

    /// Whether this type is a rank-0 scalar.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Element count, saturating instead of overflowing.
    ///
    /// Verification enforces the true (checked) budget separately; this
    /// saturating view is for cost accounting.
    pub fn elements(&self) -> u64 {
        self.shape.iter().fold(1u64, |product, &dimension| {
            product.saturating_mul(dimension as u64)
        })
    }

    /// Exact element count, or `None` on `usize` overflow.
    pub fn checked_elements(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
    }
}

/// A compile-time scalar constant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl ScalarValue {
    /// The dtype of the constant.
    pub fn dtype(&self) -> DType {
        match self
        {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::Bool(_) => DType::Bool,
        }
    }

    /// Whether the constant is admissible.
    ///
    /// Contract: constants may be finite or exactly `±Infinity` (stable
    /// identities such as a running-max initialiser require `−∞`); NaN
    /// constants are rejected because they carry no definable role in a
    /// straight-line numerical program. Computed non-finite intermediates are
    /// governed separately by [`super::interpret::FloatPolicy`].
    pub fn is_admissible(&self) -> bool {
        match self
        {
            Self::F32(value) => !value.is_nan(),
            Self::F64(value) => !value.is_nan(),
            Self::Bool(_) => true,
        }
    }
}

/// Failure modes of static shape algebra.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Two shapes cannot be broadcast together.
    BroadcastIncompatible { left: Vec<usize>, right: Vec<usize> },
    /// `left` cannot be broadcast to `target`.
    BroadcastToIncompatible {
        source: Vec<usize>,
        target: Vec<usize>,
    },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::BroadcastIncompatible { left, right } => write!(
                formatter,
                "shapes {left:?} and {right:?} cannot be broadcast together"
            ),
            Self::BroadcastToIncompatible { source, target } => write!(
                formatter,
                "shape {source:?} cannot be broadcast to {target:?}"
            ),
        }
    }
}

/// NumPy-style right-aligned broadcast of two shapes.
///
/// Dimensions pair up from the right; each pair must be equal, or one side
/// must be `1`, or one side must be absent. A zero-sized dimension broadcasts
/// only against `0` or `1`.
pub fn broadcast_shapes(left: &[usize], right: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let rank = left.len().max(right.len());
    let mut shape = vec![0usize; rank];
    for (axis, slot) in shape.iter_mut().enumerate()
    {
        // Right-aligned pairing: an operand has a dimension at output axis
        // `axis` only when `axis >= rank - operand_rank`.
        let left_dimension = (axis + left.len())
            .checked_sub(rank)
            .and_then(|index| left.get(index))
            .copied();
        let right_dimension = (axis + right.len())
            .checked_sub(rank)
            .and_then(|index| right.get(index))
            .copied();
        *slot = match (left_dimension, right_dimension)
        {
            (Some(a), Some(b)) if a == b => a,
            (Some(1), Some(b)) => b,
            (Some(a), Some(1)) => a,
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => unreachable!("axis {axis} lies within the broadcast rank"),
            (Some(_), Some(_)) =>
            {
                return Err(ShapeError::BroadcastIncompatible {
                    left: left.to_vec(),
                    right: right.to_vec(),
                });
            },
        };
    }
    Ok(shape)
}

/// Whether `source` can be broadcast **to** an explicit `target` shape.
///
/// `target` must be at least the rank of `source`; right-aligned dimensions of
/// `source` must equal the target dimension or be `1`.
pub fn can_broadcast_to(source: &[usize], target: &[usize]) -> Result<(), ShapeError> {
    if target.len() < source.len()
    {
        return Err(ShapeError::BroadcastToIncompatible {
            source: source.to_vec(),
            target: target.to_vec(),
        });
    }
    let offset = target.len() - source.len();
    for (index, (&dimension, &target_dimension)) in source.iter().zip(&target[offset..]).enumerate()
    {
        let _ = index;
        if dimension != target_dimension && dimension != 1
        {
            return Err(ShapeError::BroadcastToIncompatible {
                source: source.to_vec(),
                target: target.to_vec(),
            });
        }
    }
    Ok(())
}

/// Checked element count of a shape.
pub fn shape_elements(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
}

/// Checked row-major strides of a shape.
pub fn row_major_strides(shape: &[usize]) -> Option<Vec<usize>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_tags_round_trip() {
        for dtype in DType::ALL
        {
            assert_eq!(DType::from_tag(dtype.tag()), Some(dtype));
        }
        assert_eq!(DType::from_tag(3), None);
    }

    #[test]
    fn value_type_element_counts() {
        assert_eq!(ValueType::scalar(DType::F32).elements(), 1);
        assert_eq!(
            ValueType::new(DType::F64, vec![2, 3]).checked_elements(),
            Some(6)
        );
        assert_eq!(
            ValueType::new(DType::Bool, vec![2, 0]).checked_elements(),
            Some(0)
        );
    }

    #[test]
    fn broadcast_rules() {
        assert_eq!(broadcast_shapes(&[], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[2, 1], &[1, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[5], &[2, 5]).unwrap(), vec![2, 5]);
        assert_eq!(broadcast_shapes(&[2, 0], &[2, 1]).unwrap(), vec![2, 0]);
        assert!(broadcast_shapes(&[2, 3], &[3, 2]).is_err());
        assert!(broadcast_shapes(&[0], &[4]).is_err());
    }

    #[test]
    fn broadcast_to_rules() {
        assert!(can_broadcast_to(&[], &[2, 2]).is_ok());
        assert!(can_broadcast_to(&[1, 3], &[4, 3]).is_ok());
        assert!(can_broadcast_to(&[2], &[3]).is_err());
        assert!(can_broadcast_to(&[2, 3], &[3]).is_err());
    }

    #[test]
    fn scalar_constants_report_admissibility() {
        assert!(ScalarValue::F32(0.0).is_admissible());
        // NaN is never admissible.
        assert!(!ScalarValue::F32(f32::NAN).is_admissible());
        assert!(!ScalarValue::F64(f64::NAN).is_admissible());
        // ±Infinity is admissible (stable identities such as running max).
        assert!(ScalarValue::F64(f64::NEG_INFINITY).is_admissible());
        assert!(ScalarValue::F32(f32::INFINITY).is_admissible());
        assert!(ScalarValue::Bool(false).is_admissible());
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(row_major_strides(&[2, 3, 4]), Some(vec![12, 4, 1]));
        assert_eq!(row_major_strides(&[]), Some(vec![]));
        assert_eq!(row_major_strides(&[7]), Some(vec![1]));
    }
}
