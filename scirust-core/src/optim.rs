/// Advanced optimizers for neural network training
/// Includes: RMSprop, AdamW, LAMB, and other variants
use std::collections::HashMap;

/// RMSprop optimizer - uses exponential moving average of squared gradients
#[derive(Debug, Clone)]
pub struct RMSprop {
    learning_rate: f32,
    decay_rate: f32,              // Default: 0.99
    epsilon: f32,                 // Default: 1e-8
    momentum: f32,                // Default: 0.0
    v: HashMap<String, Vec<f32>>, // Accumulated squared gradients
    m: HashMap<String, Vec<f32>>, // Momentum buffers
}

impl RMSprop {
    pub fn new(learning_rate: f32) -> Self {
        RMSprop {
            learning_rate,
            decay_rate: 0.99,
            epsilon: 1e-8,
            momentum: 0.0,
            v: HashMap::new(),
            m: HashMap::new(),
        }
    }

    pub fn with_decay(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn step(&mut self, param_id: &str, grad: &[f32]) {
        let v = self
            .v
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);
        let m = self
            .m
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);

        for i in 0..grad.len()
        {
            // Update biased second moment estimate
            v[i] = self.decay_rate * v[i] + (1.0 - self.decay_rate) * grad[i] * grad[i];

            // Update momentum (if enabled)
            m[i] =
                self.momentum * m[i] + self.learning_rate * grad[i] / (v[i].sqrt() + self.epsilon);
        }
    }

    pub fn get_update(&self, param_id: &str) -> Option<Vec<f32>> {
        self.m.get(param_id).cloned()
    }
}

/// AdamW optimizer - Adam with decoupled weight decay
#[derive(Debug, Clone)]
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,   // Default: 0.9 (exponential decay for mean)
    beta2: f32,   // Default: 0.999 (exponential decay for variance)
    epsilon: f32, // Default: 1e-8
    #[allow(dead_code)]
    pub weight_decay: f32, // Default: 0.01 (decoupled weight decay)
    m: HashMap<String, Vec<f32>>, // First moment (mean)
    v: HashMap<String, Vec<f32>>, // Second moment (variance)
    t: u32,       // Timestep for bias correction
}

impl AdamW {
    pub fn new(learning_rate: f32) -> Self {
        AdamW {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn step(&mut self, param_id: &str, grad: &[f32], param: &mut [f32]) {
        self.t += 1;

        let m = self
            .m
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);
        let v = self
            .v
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..grad.len()
        {
            // Update biased first moment estimate
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];

            // Update biased second raw moment estimate
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            // Compute bias-corrected moment estimates
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // Update parameters with decoupled weight decay
            param[i] = param[i] * (1.0 - self.weight_decay * self.learning_rate)
                - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

/// LAMB optimizer - Layer-wise Adaptive Moments optimizer for Batch training
#[derive(Debug, Clone)]
pub struct LAMB {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    #[allow(dead_code)]
    pub weight_decay: f32,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
    t: u32,
}

impl LAMB {
    pub fn new(learning_rate: f32) -> Self {
        LAMB {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    pub fn step(&mut self, param_id: &str, grad: &[f32], param: &mut [f32]) {
        self.t += 1;

        let m = self
            .m
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);
        let v = self
            .v
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..grad.len()
        {
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            let update = m_hat / (v_hat.sqrt() + self.epsilon);
            let param_norm = param.iter().map(|p| p * p).sum::<f32>().sqrt();
            let update_norm = update;

            let adaptive_lr = if param_norm > 0.0
            {
                self.learning_rate * (param_norm / (update_norm + self.epsilon))
            }
            else
            {
                self.learning_rate
            };

            param[i] -= adaptive_lr * update;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsprop() {
        let mut optimizer = RMSprop::new(0.01);
        let grads = vec![0.1, 0.2, 0.3];

        optimizer.step("param_0", &grads);
        let update = optimizer.get_update("param_0").unwrap();
        assert!(update.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_adamw() {
        let mut optimizer = AdamW::new(0.001);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        optimizer.step("param_0", &grads, &mut params);
        assert_ne!(params[0], 1.0);
    }

    #[test]
    fn test_lamb() {
        let mut optimizer = LAMB::new(0.001);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];

        optimizer.step("param_0", &grads, &mut params);
        assert_ne!(params[0], 1.0);
    }
}
