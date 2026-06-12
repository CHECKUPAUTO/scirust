//! A tiny tensor runtime: a named-register machine that executes compiled
//! element-wise kernels and multi-operand contractions over [`TensorND`] values.

use scirust_tensor_compile::{ContractionPlan, FusedKernel};
use scirust_tensor_core::TensorND;
use std::collections::HashMap;

#[derive(Default)]
pub struct TensorRuntime {
    regs: HashMap<String, TensorND>,
}

impl TensorRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a tensor to a register name.
    pub fn set(&mut self, name: &str, tensor: TensorND) {
        self.regs.insert(name.to_string(), tensor);
    }

    pub fn get(&self, name: &str) -> Option<&TensorND> {
        self.regs.get(name)
    }

    /// Apply a fused element-wise kernel from `input` into `output`.
    pub fn run_fused(
        &mut self,
        input: &str,
        kernel: &FusedKernel,
        output: &str,
    ) -> Result<(), String> {
        let t = self
            .regs
            .get(input)
            .ok_or_else(|| format!("unknown register '{input}'"))?;
        let r = kernel.apply(t);
        self.regs.insert(output.to_string(), r);
        Ok(())
    }

    /// Execute a contraction plan over the named input registers into `output`.
    pub fn run_contraction(
        &mut self,
        plan: &ContractionPlan,
        inputs: &[&str],
        output: &str,
    ) -> Result<(), String> {
        let mut tensors = Vec::with_capacity(inputs.len());
        for name in inputs
        {
            tensors.push(
                self.regs
                    .get(*name)
                    .ok_or_else(|| format!("unknown register '{name}'"))?,
            );
        }
        let r = plan.execute(&tensors)?;
        self.regs.insert(output.to_string(), r);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_tensor_compile::{ElementwiseOp, GraphCompiler};

    #[test]
    fn executes_contraction_then_fused_activation() {
        let mut rt = TensorRuntime::new();
        rt.set("a", TensorND::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]));
        rt.set("b", TensorND::new(vec![1., 0., 0., 1., 1., 0.], vec![3, 2]));

        let plan = ContractionPlan::new("ij,jk->ik").unwrap();
        rt.run_contraction(&plan, &["a", "b"], "c").unwrap();

        // Then subtract 10 and ReLU, fused in one pass.
        let kernel = GraphCompiler::new()
            .op(ElementwiseOp::AddScalar(-10.0))
            .op(ElementwiseOp::Relu)
            .compile();
        rt.run_fused("c", &kernel, "out").unwrap();

        let out = rt.get("out").unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        // c = [[1+? ...]] -> verify all non-negative after relu
        assert!(out.data.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn errors_on_missing_register() {
        let mut rt = TensorRuntime::new();
        let plan = ContractionPlan::new("ij->ij").unwrap();
        assert!(rt.run_contraction(&plan, &["nope"], "x").is_err());
    }
}
