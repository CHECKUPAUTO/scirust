#[derive(Debug, Clone)]
pub struct LossValue {
    pub total: f32,
    pub components: Vec<(String, f32)>,
}

impl LossValue {
    pub fn new(total: f32) -> Self {
        Self {
            total,
            components: Vec::new(),
        }
    }

    pub fn with_component(mut self, name: &str, value: f32) -> Self {
        self.components.push((name.to_string(), value));
        self
    }
}

pub fn mse_loss(predicted: &[f32], target: &[f32]) -> f32 {
    let n = predicted.len().min(target.len());
    let sum: f32 = predicted[..n]
        .iter()
        .zip(target[..n].iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    sum / n as f32
}

pub fn acceleration_loss(predicted_ddq: &[f32], target_ddq: &[f32]) -> LossValue {
    let loss = mse_loss(predicted_ddq, target_ddq);
    LossValue::new(loss).with_component("acceleration", loss)
}

pub fn euler_lagrange_residual_loss(el_residual: &[f32]) -> LossValue {
    let loss = el_residual.iter().map(|r| r.powi(2)).sum::<f32>() / el_residual.len() as f32;
    LossValue::new(loss).with_component("el_residual", loss)
}

pub fn parameter_norm(params: &[Vec<f32>]) -> f32 {
    let sum_sq: f32 = params.iter().flat_map(|p| p.iter()).map(|&v| v * v).sum();
    sum_sq.sqrt()
}

pub fn hessian_conditioning_penalty(condition_number: f32, threshold: f32) -> f32 {
    if condition_number > threshold
    {
        (condition_number / threshold - 1.0).powi(2)
    }
    else
    {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss_identical() {
        let loss = mse_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((loss).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_different() {
        let loss = mse_loss(&[1.0, 2.0], &[1.0, 3.0]);
        assert!((loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_parameter_norm() {
        let params = vec![vec![3.0, 4.0]];
        let norm = parameter_norm(&params);
        assert!((norm - 5.0).abs() < 1e-6);
    }
}
