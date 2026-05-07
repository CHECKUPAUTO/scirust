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

    /// Matrix multiplication for 2D tensors.
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "LHS must be 2D for matmul");
        assert_eq!(rhs.shape.len(), 2, "RHS must be 2D for matmul");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        assert_eq!(k, rhs.shape[0], "Inner dimensions must match for matmul");

        let mut out_data = vec![0.0; m * n];
        unsafe {
            matrixmultiply::dgemm(
                m, k, n,
                1.0,
                self.data.as_ptr(), k as isize, 1,
                rhs.data.as_ptr(), n as isize, 1,
                0.0,
                out_data.as_mut_ptr(), n as isize, 1,
            );
        }

        Tensor {
            data: out_data,
            shape: vec![m, n],
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

    #[test]
    fn test_tensor_matmul() {
        // [1 2]   [5 6]   [1*5+2*7 1*6+2*8]   [19 22]
        // [3 4] * [7 8] = [3*5+4*7 3*6+4*8] = [43 50]
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let t3 = t1.matmul(&t2);
        assert_eq!(t3.data, vec![19.0, 22.0, 43.0, 50.0]);
        assert_eq!(t3.shape, vec![2, 2]);
    }
}
