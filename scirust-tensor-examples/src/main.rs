//! Demonstrates the scirust-tensor einsum engine on a few classic patterns.

use scirust_tensor_core::TensorND;
use scirust_tensor_einsum::einsum;

fn main() {
    // Matrix multiplication A(2x3) · B(3x2).
    let a = TensorND::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = TensorND::new(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]);
    let c = einsum("ij,jk->ik", &[&a, &b]).expect("matmul");
    println!("A·B = {:?} shape {:?}", c.data, c.shape);

    // Trace of a square matrix.
    let m = TensorND::new(vec![1., 2., 3., 4.], vec![2, 2]);
    let tr = einsum("ii->", &[&m]).expect("trace");
    println!("trace(M) = {}", tr.data[0]);

    // Batched multi-head attention scores: (b,h,i,d),(b,h,j,d) -> (b,h,i,j).
    let q = TensorND::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let k = TensorND::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
    let scores = einsum("bhid,bhjd->bhij", &[&q, &k]).expect("attention");
    println!("attention scores = {:?} shape {:?}", scores.data, scores.shape);
}
