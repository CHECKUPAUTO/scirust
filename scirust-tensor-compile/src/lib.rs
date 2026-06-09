//! A minimal graph compiler demonstrating **operator fusion**.
//!
//! A chain of element-wise operations (scale, bias, ReLU, …) is compiled into a
//! single [`FusedKernel`] that is evaluated in **one pass** over the data,
//! instead of materialising an intermediate tensor per operation. This is the
//! same memory-bandwidth win described for the tensor stack: fewer passes, fewer
//! temporaries.

use scirust_tensor_core::TensorND;

// Re-export the contraction planner so multi-operand contractions and
// element-wise fusion share one entry point.
pub use scirust_tensor_contraction::ContractionPlan;

/// A single element-wise operation `f(x)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementwiseOp {
    AddScalar(f32),
    MulScalar(f32),
    Relu,
}

impl ElementwiseOp {
    #[inline]
    fn eval(&self, x: f32) -> f32 {
        match self {
            ElementwiseOp::AddScalar(c) => x + c,
            ElementwiseOp::MulScalar(c) => x * c,
            ElementwiseOp::Relu => x.max(0.0),
        }
    }
}

/// Builds a fused element-wise pipeline.
#[derive(Default)]
pub struct GraphCompiler {
    ops: Vec<ElementwiseOp>,
}

impl GraphCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an element-wise op (builder style).
    pub fn op(mut self, op: ElementwiseOp) -> Self {
        self.ops.push(op);
        self
    }

    /// Number of ops that will be fused into one pass.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Compile the chain into a single fused kernel.
    pub fn compile(self) -> FusedKernel {
        FusedKernel { ops: self.ops }
    }
}

/// A compiled, fused element-wise kernel.
pub struct FusedKernel {
    ops: Vec<ElementwiseOp>,
}

impl FusedKernel {
    /// Apply all fused ops in a single pass over the input, allocating exactly
    /// one output buffer regardless of how many ops were fused.
    pub fn apply(&self, t: &TensorND) -> TensorND {
        let data = t
            .data
            .iter()
            .map(|&x| self.ops.iter().fold(x, |acc, op| op.eval(acc)))
            .collect();
        TensorND::new(data, t.shape.clone())
    }

    pub fn num_fused(&self) -> usize {
        self.ops.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_matches_sequential() {
        let t = TensorND::new(vec![-2.0, -0.5, 1.0, 3.0], vec![4]);
        // (x * 2 + 1) then ReLU
        let kernel = GraphCompiler::new()
            .op(ElementwiseOp::MulScalar(2.0))
            .op(ElementwiseOp::AddScalar(1.0))
            .op(ElementwiseOp::Relu)
            .compile();
        assert_eq!(kernel.num_fused(), 3);
        let fused = kernel.apply(&t);

        // Sequential reference: three separate passes.
        let seq: Vec<f32> = t
            .data
            .iter()
            .map(|&x| (x * 2.0 + 1.0).max(0.0))
            .collect();
        assert_eq!(fused.data, seq);
        assert_eq!(fused.data, vec![0.0, 0.0, 3.0, 7.0]);
    }
}
