use std::fmt;
use std::sync::Arc;

use scirust_compute::{DType, Shape, Strides};

/// Logical placement carried by a canonical tensor value.
///
/// `scirust-tensor-core` deliberately owns no backend buffer. A non-host
/// placement therefore describes where a runtime should materialise the value;
/// the canonical byte payload remains host-visible and deterministic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorDevice {
    Host,
    Backend { name: String, ordinal: u32 },
}

impl Default for TensorDevice {
    fn default() -> Self {
        Self::Host
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ShapeOverflow,
    ByteLengthMismatch { expected: usize, actual: usize },
    RankMismatch { shape_rank: usize, stride_rank: usize },
    ViewOutOfBounds,
    InvalidPermutation,
    InvalidAxis { axis: usize, rank: usize },
    InvalidSlice,
    BroadcastMismatch { source: Vec<usize>, target: Vec<usize> },
    ReshapeElementCount { current: usize, requested: usize },
    ReshapeRequiresContiguous,
    DTypeMismatch { expected: DType, actual: DType },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => f.write_str("tensor shape or layout overflows usize"),
            Self::ByteLengthMismatch { expected, actual } => {
                write!(f, "tensor byte length mismatch: expected {expected}, got {actual}")
            }
            Self::RankMismatch {
                shape_rank,
                stride_rank,
            } => write!(
                f,
                "tensor rank mismatch: shape rank {shape_rank}, stride rank {stride_rank}"
            ),
            Self::ViewOutOfBounds => f.write_str("tensor view reaches outside its storage"),
            Self::InvalidPermutation => f.write_str("tensor permutation is not a rank permutation"),
            Self::InvalidAxis { axis, rank } => {
                write!(f, "tensor axis {axis} is invalid for rank {rank}")
            }
            Self::InvalidSlice => f.write_str("invalid tensor slice"),
            Self::BroadcastMismatch { source, target } => {
                write!(f, "cannot broadcast shape {source:?} to {target:?}")
            }
            Self::ReshapeElementCount { current, requested } => write!(
                f,
                "reshape changes element count from {current} to {requested}"
            ),
            Self::ReshapeRequiresContiguous => {
                f.write_str("zero-copy reshape requires a contiguous tensor view")
            }
            Self::DTypeMismatch { expected, actual } => {
                write!(f, "dtype mismatch: expected {expected:?}, got {actual:?}")
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Backend-neutral, dtype-aware N-dimensional tensor value.
///
/// Unlike the historical [`crate::TensorND`], metadata is private and cannot be
/// made inconsistent by callers. Views share immutable storage through `Arc`;
/// transpose, slicing, broadcasting and contiguous reshape therefore allocate
/// no tensor payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    storage: Arc<[u8]>,
    dtype: DType,
    shape: Shape,
    strides: Strides,
    offset_elements: usize,
    device: TensorDevice,
}

impl Tensor {
    /// Construct a contiguous tensor from canonical little-endian scalar bytes.
    pub fn from_bytes(
        bytes: Vec<u8>,
        dtype: DType,
        shape: Shape,
        device: TensorDevice,
    ) -> Result<Self, TensorError> {
        let numel = checked_numel(&shape)?;
        let expected = numel
            .checked_mul(dtype.size_bytes())
            .ok_or(TensorError::ShapeOverflow)?;
        if bytes.len() != expected {
            return Err(TensorError::ByteLengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        let strides = Strides::contiguous(&shape).map_err(|_| TensorError::ShapeOverflow)?;
        Self::from_shared_parts(Arc::from(bytes), dtype, shape, strides, 0, device)
    }

    pub fn from_f32(data: Vec<f32>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let mut bytes = Vec::with_capacity(data.len().saturating_mul(4));
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_bytes(
            bytes,
            DType::F32,
            Shape::new(shape.into()),
            TensorDevice::Host,
        )
    }

    pub fn from_f64(data: Vec<f64>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let mut bytes = Vec::with_capacity(data.len().saturating_mul(8));
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_bytes(
            bytes,
            DType::F64,
            Shape::new(shape.into()),
            TensorDevice::Host,
        )
    }

    pub fn from_i64(data: Vec<i64>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let mut bytes = Vec::with_capacity(data.len().saturating_mul(8));
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_bytes(
            bytes,
            DType::I64,
            Shape::new(shape.into()),
            TensorDevice::Host,
        )
    }

    pub fn zeros(dtype: DType, shape: Shape, device: TensorDevice) -> Result<Self, TensorError> {
        let bytes = checked_numel(&shape)?
            .checked_mul(dtype.size_bytes())
            .ok_or(TensorError::ShapeOverflow)?;
        Self::from_bytes(vec![0; bytes], dtype, shape, device)
    }

    pub fn scalar_f32(value: f32) -> Self {
        Self::from_f32(vec![value], Vec::<usize>::new())
            .expect("a scalar f32 tensor is always representable")
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn device(&self) -> &TensorDevice {
        &self.device
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn numel(&self) -> usize {
        self.shape
            .checked_num_elements()
            .expect("validated tensor shape")
    }

    pub fn storage_offset(&self) -> usize {
        self.offset_elements
    }

    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    pub fn is_contiguous(&self) -> bool {
        Strides::contiguous(&self.shape)
            .map(|expected| expected == self.strides)
            .unwrap_or(false)
    }

    /// Return the logical tensor payload in row-major order.
    pub fn to_contiguous_bytes(&self) -> Vec<u8> {
        let width = self.dtype.size_bytes();
        let mut out = Vec::with_capacity(self.numel().saturating_mul(width));
        for storage_index in self.logical_storage_offsets() {
            let byte_start = storage_index * width;
            out.extend_from_slice(&self.storage[byte_start..byte_start + width]);
        }
        out
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F32,
                actual: self.dtype,
            });
        }
        let bytes = self.to_contiguous_bytes();
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    /// Zero-copy reshape. Non-contiguous views must be made contiguous first.
    pub fn reshape(&self, new_shape: Shape) -> Result<Self, TensorError> {
        let requested = checked_numel(&new_shape)?;
        if requested != self.numel() {
            return Err(TensorError::ReshapeElementCount {
                current: self.numel(),
                requested,
            });
        }
        if !self.is_contiguous() {
            return Err(TensorError::ReshapeRequiresContiguous);
        }
        let strides = Strides::contiguous(&new_shape).map_err(|_| TensorError::ShapeOverflow)?;
        Self::from_shared_parts(
            self.storage.clone(),
            self.dtype,
            new_shape,
            strides,
            self.offset_elements,
            self.device.clone(),
        )
    }

    pub fn transpose(&self, first: usize, second: usize) -> Result<Self, TensorError> {
        if first >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis: first,
                rank: self.rank(),
            });
        }
        if second >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis: second,
                rank: self.rank(),
            });
        }
        let mut permutation: Vec<usize> = (0..self.rank()).collect();
        permutation.swap(first, second);
        self.permute(&permutation)
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Self, TensorError> {
        if permutation.len() != self.rank() {
            return Err(TensorError::InvalidPermutation);
        }
        let mut seen = vec![false; self.rank()];
        for &axis in permutation {
            if axis >= self.rank() || seen[axis] {
                return Err(TensorError::InvalidPermutation);
            }
            seen[axis] = true;
        }

        let dims = permutation
            .iter()
            .map(|&axis| self.shape.dims()[axis])
            .collect::<Vec<_>>();
        let strides = permutation
            .iter()
            .map(|&axis| self.strides.values()[axis])
            .collect::<Vec<_>>();
        Self::from_shared_parts(
            self.storage.clone(),
            self.dtype,
            Shape::new(dims),
            Strides::new(strides),
            self.offset_elements,
            self.device.clone(),
        )
    }

    /// Positive-step, half-open slice along one axis, implemented as a view.
    pub fn slice(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let dim = self.shape.dims()[axis];
        if step == 0 || start > end || end > dim {
            return Err(TensorError::InvalidSlice);
        }
        let span = end - start;
        let new_dim = if span == 0 { 0 } else { 1 + (span - 1) / step };
        let axis_stride = self.strides.values()[axis];
        let new_offset = self
            .offset_elements
            .checked_add(
                start
                    .checked_mul(axis_stride)
                    .ok_or(TensorError::ShapeOverflow)?,
            )
            .ok_or(TensorError::ShapeOverflow)?;
        let new_axis_stride = axis_stride
            .checked_mul(step)
            .ok_or(TensorError::ShapeOverflow)?;

        let mut dims = self.shape.dims().to_vec();
        dims[axis] = new_dim;
        let mut strides = self.strides.values().to_vec();
        strides[axis] = new_axis_stride;
        Self::from_shared_parts(
            self.storage.clone(),
            self.dtype,
            Shape::new(dims),
            Strides::new(strides),
            new_offset,
            self.device.clone(),
        )
    }

    /// NumPy-style right-aligned broadcasting as a zero-stride view.
    pub fn broadcast_to(&self, target: Shape) -> Result<Self, TensorError> {
        if target.rank() < self.rank() {
            return Err(TensorError::BroadcastMismatch {
                source: self.shape.dims().to_vec(),
                target: target.dims().to_vec(),
            });
        }

        let mut strides = vec![0usize; target.rank()];
        let rank_delta = target.rank() - self.rank();
        for source_axis in 0..self.rank() {
            let target_axis = source_axis + rank_delta;
            let source_dim = self.shape.dims()[source_axis];
            let target_dim = target.dims()[target_axis];
            if source_dim == target_dim {
                strides[target_axis] = self.strides.values()[source_axis];
            } else if source_dim == 1 {
                strides[target_axis] = 0;
            } else {
                return Err(TensorError::BroadcastMismatch {
                    source: self.shape.dims().to_vec(),
                    target: target.dims().to_vec(),
                });
            }
        }

        Self::from_shared_parts(
            self.storage.clone(),
            self.dtype,
            target,
            Strides::new(strides),
            self.offset_elements,
            self.device.clone(),
        )
    }

    /// Materialise this value as a compact row-major tensor while preserving
    /// dtype and logical placement metadata.
    pub fn contiguous(&self) -> Result<Self, TensorError> {
        Self::from_bytes(
            self.to_contiguous_bytes(),
            self.dtype,
            self.shape.clone(),
            self.device.clone(),
        )
    }

    fn from_shared_parts(
        storage: Arc<[u8]>,
        dtype: DType,
        shape: Shape,
        strides: Strides,
        offset_elements: usize,
        device: TensorDevice,
    ) -> Result<Self, TensorError> {
        if shape.rank() != strides.values().len() {
            return Err(TensorError::RankMismatch {
                shape_rank: shape.rank(),
                stride_rank: strides.values().len(),
            });
        }
        let width = dtype.size_bytes();
        if storage.len() % width != 0 {
            return Err(TensorError::ViewOutOfBounds);
        }
        let storage_elements = storage.len() / width;
        validate_view_bounds(&shape, &strides, offset_elements, storage_elements)?;
        Ok(Self {
            storage,
            dtype,
            shape,
            strides,
            offset_elements,
            device,
        })
    }

    fn logical_storage_offsets(&self) -> LogicalOffsets<'_> {
        LogicalOffsets {
            shape: self.shape.dims(),
            strides: self.strides.values(),
            base: self.offset_elements,
            linear: 0,
            total: self.numel(),
        }
    }
}

struct LogicalOffsets<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    base: usize,
    linear: usize,
    total: usize,
}

impl Iterator for LogicalOffsets<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.linear >= self.total {
            return None;
        }
        let mut remainder = self.linear;
        let mut offset = self.base;
        for axis in (0..self.shape.len()).rev() {
            let dim = self.shape[axis];
            let coordinate = remainder % dim;
            remainder /= dim;
            offset += coordinate * self.strides[axis];
        }
        self.linear += 1;
        Some(offset)
    }
}

fn checked_numel(shape: &Shape) -> Result<usize, TensorError> {
    shape
        .checked_num_elements()
        .map_err(|_| TensorError::ShapeOverflow)
}

fn validate_view_bounds(
    shape: &Shape,
    strides: &Strides,
    offset: usize,
    storage_elements: usize,
) -> Result<(), TensorError> {
    let numel = checked_numel(shape)?;
    if numel == 0 {
        return if offset <= storage_elements {
            Ok(())
        } else {
            Err(TensorError::ViewOutOfBounds)
        };
    }

    let mut max_index = offset;
    for (&dim, &stride) in shape.dims().iter().zip(strides.values()) {
        debug_assert!(dim > 0);
        max_index = max_index
            .checked_add(
                (dim - 1)
                    .checked_mul(stride)
                    .ok_or(TensorError::ShapeOverflow)?,
            )
            .ok_or(TensorError::ShapeOverflow)?;
    }
    if max_index >= storage_elements {
        Err(TensorError::ViewOutOfBounds)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_and_slice_share_storage() {
        let tensor = Tensor::from_f32((0..12).map(|v| v as f32).collect(), vec![3, 4]).unwrap();
        let transposed = tensor.transpose(0, 1).unwrap();
        let sliced = transposed.slice(0, 1, 4, 2).unwrap();

        assert!(tensor.shares_storage_with(&transposed));
        assert!(tensor.shares_storage_with(&sliced));
        assert_eq!(transposed.shape().dims(), &[4, 3]);
        assert_eq!(sliced.shape().dims(), &[2, 3]);
        assert_eq!(
            sliced.to_f32_vec().unwrap(),
            vec![1., 5., 9., 3., 7., 11.]
        );
    }

    #[test]
    fn reshape_is_zero_copy_for_contiguous_values() {
        let tensor = Tensor::from_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
        let reshaped = tensor.reshape(Shape::new(vec![3, 2])).unwrap();
        assert!(tensor.shares_storage_with(&reshaped));
        assert_eq!(
            reshaped.to_f32_vec().unwrap(),
            vec![1., 2., 3., 4., 5., 6.]
        );
    }

    #[test]
    fn broadcast_uses_zero_strides_and_materialises_correctly() {
        let row = Tensor::from_f32(vec![10., 20., 30.], vec![1, 3]).unwrap();
        let broadcast = row.broadcast_to(Shape::new(vec![2, 3])).unwrap();
        assert_eq!(broadcast.strides().values(), &[0, 1]);
        assert!(row.shares_storage_with(&broadcast));
        assert_eq!(
            broadcast.to_f32_vec().unwrap(),
            vec![10., 20., 30., 10., 20., 30.]
        );
    }

    #[test]
    fn non_contiguous_reshape_requires_materialisation() {
        let tensor = Tensor::from_f32(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let transpose = tensor.transpose(0, 1).unwrap();
        assert_eq!(
            transpose.reshape(Shape::new(vec![4])),
            Err(TensorError::ReshapeRequiresContiguous)
        );
        let compact = transpose.contiguous().unwrap();
        assert_eq!(
            compact
                .reshape(Shape::new(vec![4]))
                .unwrap()
                .to_f32_vec()
                .unwrap(),
            vec![1., 3., 2., 4.]
        );
    }

    #[test]
    fn dtype_is_not_implicitly_reinterpreted() {
        let tensor = Tensor::from_i64(vec![1, 2], vec![2]).unwrap();
        assert_eq!(tensor.dtype(), DType::I64);
        assert!(matches!(
            tensor.to_f32_vec(),
            Err(TensorError::DTypeMismatch { .. })
        ));
    }
}
