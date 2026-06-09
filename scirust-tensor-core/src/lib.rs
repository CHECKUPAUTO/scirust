//! Core N-dimensional tensor type shared by the `scirust-tensor-*` crates.

/// A dense, row-major N-dimensional tensor of `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorND {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl TensorND {
    /// Create a tensor from data and shape. Panics if `data.len()` does not match
    /// the product of `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product::<usize>().max(if shape.is_empty() { 1 } else { 0 });
        assert_eq!(
            data.len(),
            expected,
            "TensorND::new: data length {} != product of shape {:?}",
            data.len(),
            shape
        );
        let strides = compute_strides(&shape);
        Self { data, shape, strides }
    }

    /// Zero-filled tensor of the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product::<usize>().max(if shape.is_empty() { 1 } else { 0 });
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

    /// Row-major flat offset of a multi-index.
    pub fn offset(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.shape.len());
        index
            .iter()
            .zip(&self.strides)
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Element at a multi-index.
    pub fn get(&self, index: &[usize]) -> f32 {
        self.data[self.offset(index)]
    }

    /// Reshape without copying data; errors if the element count changes.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<TensorND, String> {
        let n: usize = new_shape.iter().product::<usize>().max(if new_shape.is_empty() { 1 } else { 0 });
        if n != self.data.len() {
            return Err(format!(
                "reshape: {} elements cannot fit shape {:?}",
                self.data.len(),
                new_shape
            ));
        }
        Ok(TensorND::new(self.data.clone(), new_shape))
    }
}

/// Row-major strides for `shape`. Handles 0- and 1-D shapes without underflow.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    if ndim <= 1 {
        return strides;
    }
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
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
}
