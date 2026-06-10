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
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        if inputs.is_empty()
        {
            return Err(crate::core::ReasoningError::Neural("No inputs".to_string()));
        }
        Ok(inputs[0].clone())
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
}
