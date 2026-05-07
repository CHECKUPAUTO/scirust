pub use scirust_autodiff::*;
pub use scirust_macros::autodiff;
pub use scirust_simd::*;
pub use scirust_gpu::dispatch;

/// A multi-dimensional array for scientific computing.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with the given data and shape.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(data.len(), size, "Data length does not match shape");
        Tensor { data, shape }
    }

    /// Create a tensor of zeros with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Element-wise addition.
    pub fn add(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Shapes must match for addition");
        let mut out_data = vec![0.0; self.data.len()];
        scirust_simd::ops::add_f64(&self.data, &rhs.data, &mut out_data);
        Tensor {
            data: out_data,
            shape: self.shape.clone(),
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape, rhs.shape, "Shapes must match for multiplication");
        let mut out_data = vec![0.0; self.data.len()];
        scirust_simd::ops::mul_f64(&self.data, &rhs.data, &mut out_data);
        Tensor {
            data: out_data,
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_add() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let t3 = t1.add(&t2);
        assert_eq!(t3.data, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(t3.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_mul() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let t3 = t1.mul(&t2);
        assert_eq!(t3.data, vec![5.0, 12.0, 21.0, 32.0]);
    }
}
