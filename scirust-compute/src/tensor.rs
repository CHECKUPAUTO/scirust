use crate::{ComputeError, ComputeResult, DType, DeviceId, Layout, Shape, Strides};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub dtype: DType,
    pub shape: Shape,
    pub strides: Strides,
    pub layout: Layout,
    pub device: DeviceId,
}

impl TensorSpec {
    pub fn contiguous(dtype: DType, shape: Shape, device: DeviceId) -> ComputeResult<Self> {
        let strides = Strides::contiguous(&shape)?;

        Ok(Self {
            dtype,
            shape,
            strides,
            layout: Layout::ContiguousRowMajor,
            device,
        })
    }

    pub fn checked_byte_len(&self) -> ComputeResult<usize> {
        self.shape
            .checked_num_elements()?
            .checked_mul(self.dtype.size_bytes())
            .ok_or(ComputeError::ByteSizeOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_tensor_has_checked_size() {
        let spec =
            TensorSpec::contiguous(DType::F32, Shape::new(vec![2, 3, 4]), DeviceId::cpu()).unwrap();

        assert_eq!(spec.checked_byte_len(), Ok(96));
        assert_eq!(spec.strides.values(), &[12, 4, 1]);
    }
}
