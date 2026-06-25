use crate::core::{DifferentiableReasoner, Reasoner, Result};
use scirust_core::autodiff::reverse::Tensor;

/// Differentiable reasoning layer.
pub struct DifferentiableLogicLayer {
    pub name: String,
}

impl DifferentiableLogicLayer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    /// Implement fuzzy logic operations on tensors.
    pub fn fuzzy_and(&self, a: &Tensor, b: &Tensor) -> Tensor {
        // Use hadamard product for fuzzy AND (Product T-norm)
        a.hadamard(b)
    }

    pub fn fuzzy_or(&self, a: &Tensor, b: &Tensor) -> Tensor {
        // Probabilistic sum: a + b - a*b
        let sum = a.add(b);
        let prod = a.hadamard(b);
        sum.sub(&prod)
    }
}

impl Reasoner for DifferentiableLogicLayer {
    fn name(&self) -> &str {
        &self.name
    }
}

impl DifferentiableReasoner for DifferentiableLogicLayer {
    /// Combine all input tensors with the fuzzy-AND (product T-norm) semantics,
    /// folding left-to-right: `forward([a, b, c]) = fuzzy_and(fuzzy_and(a, b), c)`.
    /// The result therefore depends on every input, unlike a passthrough.
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        if inputs.is_empty()
        {
            return Err(crate::core::ReasoningError::Neural("No inputs".to_string()));
        }
        let mut acc = inputs[0].clone();
        for t in &inputs[1..]
        {
            acc = self.fuzzy_and(&acc, t);
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differentiable_logic_name() {
        let layer = DifferentiableLogicLayer::new("LogicLayer1");
        assert_eq!(layer.name(), "LogicLayer1");
    }

    #[test]
    fn test_fuzzy_ops() {
        let layer = DifferentiableLogicLayer::new("LogicLayer1");
        let a = Tensor::from_vec(vec![0.5], 1, 1);
        let b = Tensor::from_vec(vec![0.5], 1, 1);
        let and_res = layer.fuzzy_and(&a, &b);
        assert_eq!(and_res.data[0], 0.25);

        let or_res = layer.fuzzy_or(&a, &b);
        assert_eq!(or_res.data[0], 0.75);
    }

    #[test]
    fn forward_folds_inputs_with_fuzzy_and() {
        let layer = DifferentiableLogicLayer::new("AndLayer");
        let a = Tensor::from_vec(vec![0.5], 1, 1);
        let b = Tensor::from_vec(vec![0.5], 1, 1);
        // forward([a, b]) = fuzzy_and(a, b) = 0.5 * 0.5 = 0.25 (depends on b).
        let out = layer.forward(&[a.clone(), b]).unwrap();
        assert!((out.data[0] - 0.25).abs() < 1e-6, "got {}", out.data[0]);

        // Three inputs: 0.5 * 0.5 * 0.5 = 0.125.
        let c = Tensor::from_vec(vec![0.5], 1, 1);
        let d = Tensor::from_vec(vec![0.5], 1, 1);
        let e = Tensor::from_vec(vec![0.5], 1, 1);
        let out3 = layer.forward(&[c, d, e]).unwrap();
        assert!((out3.data[0] - 0.125).abs() < 1e-6, "got {}", out3.data[0]);
    }

    #[test]
    fn forward_rejects_empty_inputs() {
        let layer = DifferentiableLogicLayer::new("AndLayer");
        assert!(layer.forward(&[]).is_err());
    }
}
