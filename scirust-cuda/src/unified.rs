//! CUDA unified-memory buffers built on cudarc's safe ownership model.
//!
//! SciRust does not bind `cudaMallocManaged` manually here. `cudarc` owns the
//! driver allocation through `CudaContext::alloc_unified`, and `UnifiedSlice`
//! keeps the allocation/context lifetime tied to Rust ownership. This module
//! narrows the unsafe construction boundary to `f32`, initializes every byte
//! before exposing host access, and presents primal/tangent SoA storage.

use cudarc::driver::{CudaContext, DriverError, UnifiedSlice};
use std::sync::Arc;

/// Error returned by SciRust's CUDA unified-memory wrappers.
#[derive(Debug)]
pub enum CudaUnifiedError {
    Driver(DriverError),
    RuntimeUnavailable,
    ShapeOverflow,
    TangentLaneOutOfBounds { lane: usize, width: usize },
}

impl core::fmt::Display for CudaUnifiedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::Driver(error) => write!(f, "CUDA unified-memory error: {error}"),
            Self::RuntimeUnavailable => write!(f, "CUDA driver runtime is unavailable"),
            Self::ShapeOverflow => write!(f, "CUDA unified tensor shape overflows usize"),
            Self::TangentLaneOutOfBounds { lane, width } =>
            {
                write!(f, "tangent lane {lane} is outside width {width}")
            },
        }
    }
}

impl std::error::Error for CudaUnifiedError {}

impl From<DriverError> for CudaUnifiedError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

/// Host+device accessible `f32` allocation managed by the CUDA driver.
///
/// Logical zero-length buffers allocate one hidden element because CUDA memory
/// allocators are not required to accept zero-byte requests. Public host views
/// are still exactly `len()` elements long.
#[derive(Debug)]
pub struct CudaUnifiedF32Buffer {
    context: Arc<CudaContext>,
    data: UnifiedSlice<f32>,
    logical_len: usize,
}

impl CudaUnifiedF32Buffer {
    /// Allocate managed memory on `device_ordinal` and initialize it to zero.
    ///
    /// cudarc's dynamic CUDA loader panics when the driver shared library is not
    /// installed. CUDA is optional for SciRust's generic builds, so this public
    /// wrapper translates that loader panic into a typed error instead of letting
    /// a no-runtime process unwind out of the API.
    pub fn new(device_ordinal: usize, len: usize) -> Result<Self, CudaUnifiedError> {
        let context = std::panic::catch_unwind(|| CudaContext::new(device_ordinal))
            .map_err(|_| CudaUnifiedError::RuntimeUnavailable)??;
        let allocation_len = len.max(1);

        // SAFETY: cudarc marks alloc_unified<T> unsafe because arbitrary T may
        // have invalid bit patterns. `f32` accepts every initialized 32-bit
        // pattern, and we immediately initialize the complete allocation to 0.0
        // before returning this safe wrapper or exposing any host view.
        let mut data = unsafe { context.alloc_unified::<f32>(allocation_len, true)? };
        data.as_mut_slice()?.fill(0.0);

        Ok(Self {
            context,
            data,
            logical_len: len,
        })
    }

    pub const fn len(&self) -> usize {
        self.logical_len
    }

    pub const fn is_empty(&self) -> bool {
        self.logical_len == 0
    }

    /// Host read view. cudarc synchronizes tracked device work before returning.
    pub fn as_slice(&self) -> Result<&[f32], CudaUnifiedError> {
        Ok(&self.data.as_slice()?[..self.logical_len])
    }

    /// Host mutable view. cudarc synchronizes tracked device work before returning.
    pub fn as_mut_slice(&mut self) -> Result<&mut [f32], CudaUnifiedError> {
        Ok(&mut self.data.as_mut_slice()?[..self.logical_len])
    }

    /// Underlying unified allocation for typed cudarc kernel arguments/views.
    pub fn unified_slice(&self) -> &UnifiedSlice<f32> {
        &self.data
    }

    /// Mutable underlying unified allocation for typed cudarc kernel arguments/views.
    pub fn unified_slice_mut(&mut self) -> &mut UnifiedSlice<f32> {
        &mut self.data
    }

    /// CUDA context that owns the allocation.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

/// Structure-of-arrays differentiable matrix in CUDA unified memory.
///
/// `values` stores `rows*cols` primals. `tangents` stores `W` complete tangent
/// planes consecutively: lane `g` occupies `g*len .. (g+1)*len`. Kernels that
/// need only primals can therefore avoid streaming tangent payloads.
#[derive(Debug)]
pub struct CudaUnifiedDualMatrixSoA<const W: usize> {
    rows: usize,
    cols: usize,
    len: usize,
    values: CudaUnifiedF32Buffer,
    tangents: CudaUnifiedF32Buffer,
}

impl<const W: usize> CudaUnifiedDualMatrixSoA<W> {
    pub fn new(device_ordinal: usize, rows: usize, cols: usize) -> Result<Self, CudaUnifiedError> {
        let len = rows
            .checked_mul(cols)
            .ok_or(CudaUnifiedError::ShapeOverflow)?;
        let tangent_len = len.checked_mul(W).ok_or(CudaUnifiedError::ShapeOverflow)?;
        Ok(Self {
            rows,
            cols,
            len,
            values: CudaUnifiedF32Buffer::new(device_ordinal, len)?,
            tangents: CudaUnifiedF32Buffer::new(device_ordinal, tangent_len)?,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn width(&self) -> usize {
        W
    }

    pub fn values_host(&self) -> Result<&[f32], CudaUnifiedError> {
        self.values.as_slice()
    }

    pub fn values_host_mut(&mut self) -> Result<&mut [f32], CudaUnifiedError> {
        self.values.as_mut_slice()
    }

    pub fn tangent_host(&self, lane: usize) -> Result<&[f32], CudaUnifiedError> {
        let range = self.tangent_range(lane)?;
        Ok(&self.tangents.as_slice()?[range])
    }

    pub fn tangent_host_mut(&mut self, lane: usize) -> Result<&mut [f32], CudaUnifiedError> {
        let range = self.tangent_range(lane)?;
        Ok(&mut self.tangents.as_mut_slice()?[range])
    }

    pub fn values_unified(&self) -> &UnifiedSlice<f32> {
        self.values.unified_slice()
    }

    pub fn values_unified_mut(&mut self) -> &mut UnifiedSlice<f32> {
        self.values.unified_slice_mut()
    }

    /// Flattened `W * len` tangent planes for device kernels.
    pub fn tangents_unified(&self) -> &UnifiedSlice<f32> {
        self.tangents.unified_slice()
    }

    pub fn tangents_unified_mut(&mut self) -> &mut UnifiedSlice<f32> {
        self.tangents.unified_slice_mut()
    }

    fn tangent_range(&self, lane: usize) -> Result<std::ops::Range<usize>, CudaUnifiedError> {
        if lane >= W
        {
            return Err(CudaUnifiedError::TangentLaneOutOfBounds { lane, width: W });
        }
        let start = lane * self.len;
        Ok(start..start + self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_overflow_is_rejected_before_cuda_initialization() {
        let result = CudaUnifiedDualMatrixSoA::<2>::new(usize::MAX, usize::MAX, 2);
        assert!(matches!(result, Err(CudaUnifiedError::ShapeOverflow)));
    }

    #[test]
    fn managed_memory_round_trip_when_cuda_is_available() {
        let Ok(mut matrix) = CudaUnifiedDualMatrixSoA::<2>::new(0, 2, 3)
        else
        {
            // CUDA is optional in generic CI. Compilation of this module is the
            // required gate; device semantics are exercised when hardware exists.
            return;
        };

        matrix.values_host_mut().unwrap()[2] = 7.5;
        matrix.tangent_host_mut(1).unwrap()[2] = 1.0;
        assert_eq!(matrix.values_host().unwrap()[2], 7.5);
        assert_eq!(matrix.tangent_host(1).unwrap()[2], 1.0);
        assert_eq!(matrix.tangent_host(0).unwrap()[2], 0.0);
    }

    #[test]
    fn tangent_lane_bounds_are_explicit_when_cuda_is_available() {
        let Ok(matrix) = CudaUnifiedDualMatrixSoA::<2>::new(0, 1, 1)
        else
        {
            return;
        };
        assert!(matches!(
            matrix.tangent_host(2),
            Err(CudaUnifiedError::TangentLaneOutOfBounds { lane: 2, width: 2 })
        ));
    }
}
