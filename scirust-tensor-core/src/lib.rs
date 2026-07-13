//! Core N-dimensional tensor type shared by the `scirust-tensor-*` crates.

/// A dense, row-major N-dimensional tensor of `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorND {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl TensorND {
    /// Create a tensor from data and shape. Returns an error if `data.len()`
    /// does not match the product of `shape`.
    pub fn try_new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        let expected = shape_product(&shape).ok_or_else(|| {
            format!("TensorND::try_new: product of shape {shape:?} overflows usize")
        })?;
        if data.len() != expected
        {
            return Err(format!(
                "TensorND::try_new: data length {} != product of shape {:?}",
                data.len(),
                shape
            ));
        }
        let strides = compute_strides(&shape).ok_or_else(|| {
            format!("TensorND::try_new: strides for shape {shape:?} overflow usize")
        })?;
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Create a tensor from data and shape. Panics if `data.len()` does not match
    /// the product of `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::try_new(data, shape).expect("TensorND::new")
    }

    /// Zero-filled tensor of the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape_product(&shape).unwrap_or_else(|| {
            panic!("TensorND::zeros: product of shape {shape:?} overflows usize")
        });
        Self::new(vec![0.0; n], shape)
    }

    /// A 0-dimensional (scalar) tensor.
    pub fn scalar(v: f32) -> Self {
        Self::new(vec![v], vec![])
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Checked row-major flat offset of a multi-index.
    pub fn try_offset(&self, index: &[usize]) -> Result<usize, String> {
        if index.len() != self.shape.len()
        {
            return Err(format!(
                "TensorND index rank {} does not match tensor rank {}",
                index.len(),
                self.shape.len()
            ));
        }
        let expected_len = shape_product(&self.shape)
            .ok_or_else(|| "TensorND shape product overflows usize".to_string())?;
        let expected_strides = compute_strides(&self.shape)
            .ok_or_else(|| "TensorND stride computation overflows usize".to_string())?;
        if expected_len != self.data.len() || self.strides != expected_strides
        {
            return Err("TensorND has inconsistent public shape/data/strides fields".to_string());
        }
        let mut offset = 0usize;
        for (axis, ((&coordinate, &dimension), &stride)) in
            index.iter().zip(&self.shape).zip(&self.strides).enumerate()
        {
            if coordinate >= dimension
            {
                return Err(format!(
                    "TensorND index {coordinate} is out of bounds for axis {axis} of size {dimension}"
                ));
            }
            offset = offset
                .checked_add(coordinate.checked_mul(stride).ok_or_else(|| {
                    format!("TensorND offset multiplication overflows at axis {axis}")
                })?)
                .ok_or_else(|| format!("TensorND offset addition overflows at axis {axis}"))?;
        }
        Ok(offset)
    }

    /// Row-major flat offset of a multi-index.
    ///
    /// Panics with a descriptive message for an invalid index. Use
    /// [`TensorND::try_offset`] when invalid user input should be recoverable.
    pub fn offset(&self, index: &[usize]) -> usize {
        self.try_offset(index).expect("TensorND::offset")
    }

    /// Checked element lookup at a multi-index.
    pub fn try_get(&self, index: &[usize]) -> Result<f32, String> {
        self.try_offset(index).map(|offset| self.data[offset])
    }

    /// Element at a multi-index. Panics for an invalid index; use
    /// [`TensorND::try_get`] for a fallible lookup.
    pub fn get(&self, index: &[usize]) -> f32 {
        self.try_get(index).expect("TensorND::get")
    }

    /// Reshape without copying data; errors if the element count changes.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<TensorND, String> {
        let n = shape_product(&new_shape)
            .ok_or_else(|| format!("reshape: product of shape {new_shape:?} overflows usize"))?;
        if n != self.data.len()
        {
            return Err(format!(
                "reshape: {} elements cannot fit shape {:?}",
                self.data.len(),
                new_shape
            ));
        }
        Ok(TensorND::new(self.data.clone(), new_shape))
    }
}

/// Number of elements described by `shape`, i.e. the product of its dimensions.
///
/// Returns `Some(1)` for an empty (scalar) shape and `None` if the product
/// overflows `usize` instead of silently wrapping (release) or panicking (debug).
fn shape_product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

/// Row-major strides for `shape`. Handles 0- and 1-D shapes without underflow.
fn compute_strides(shape: &[usize]) -> Option<Vec<usize>> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    if ndim <= 1
    {
        return Some(strides);
    }
    for i in (0..ndim - 1).rev()
    {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1])?;
    }
    Some(strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        let t = TensorND::zeros(vec![2, 3, 4]);
        assert_eq!(t.strides, vec![12, 4, 1]);
        assert_eq!(t.size(), 24);
    }

    #[test]
    fn scalar_has_no_dims() {
        let s = TensorND::scalar(3.5);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.data, vec![3.5]);
        assert_eq!(s.strides, Vec::<usize>::new());
    }

    #[test]
    fn get_and_reshape() {
        let t = TensorND::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.get(&[1, 2]), 6.0);
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.get(&[2, 1]), 6.0);
        assert!(t.reshape(vec![5]).is_err());
    }

    #[test]
    fn shape_product_semantics() {
        // Empty (scalar) shape has exactly one element.
        assert_eq!(shape_product(&[]), Some(1));
        // A zero dimension yields an empty tensor.
        assert_eq!(shape_product(&[2, 0, 3]), Some(0));
        assert_eq!(shape_product(&[2, 3, 4]), Some(24));
        // An overflowing product is reported rather than wrapped/panicked.
        assert_eq!(shape_product(&[usize::MAX, 2]), None);
    }

    #[test]
    fn try_new_rejects_overflowing_shape() {
        // Previously this wrapped the product (release) or panicked on overflow
        // (debug); now it returns a clean error instead of touching `data.len()`.
        let err = TensorND::try_new(vec![0.0], vec![usize::MAX, 2]);
        assert!(err.is_err());
        // The total product is zero, but a row-major suffix stride still
        // overflows and must not wrap.
        assert!(TensorND::try_new(vec![], vec![0, usize::MAX, 2]).is_err());
    }

    #[test]
    fn reshape_rejects_overflowing_shape() {
        let t = TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(t.reshape(vec![usize::MAX, 2]).is_err());
    }

    #[test]
    fn out_of_bounds_coordinate_cannot_alias_another_element() {
        let t = TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(t.try_get(&[0, 2]).is_err());
        assert!(t.try_get(&[2, 0]).is_err());
        assert!(t.try_get(&[0]).is_err());
        assert!(t.try_get(&[0, 0, 0]).is_err());
    }

    #[test]
    fn lookup_rejects_inconsistent_public_fields() {
        let mut t = TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        t.strides[0] = 1;
        assert!(t.try_get(&[1, 0]).is_err());
    }
}
