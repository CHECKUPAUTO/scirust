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
    // Bias-correction timestep, tracked **per parameter**. `step` is called once
    // per parameter tensor per optimization step, so a single global counter
    // would advance by (number of parameters) each step and make the bias
    // correction `1 - beta^t` wrong for any model with more than one tensor. A
    // per-parameter counter makes each tensor's correction match the number of
    // updates it has actually received (the Adam paper's `t`).
    t: HashMap<String, u32>,
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
            t: HashMap::new(),
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn step(&mut self, param_id: &str, grad: &[f32], param: &mut [f32]) {
        let t = {
            let e = self.t.entry(param_id.to_string()).or_insert(0);
            *e += 1;
            *e
        };

        let m = self
            .m
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);
        let v = self
            .v
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);

        let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

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
    pub weight_decay: f32,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
    // Per-parameter bias-correction timestep — see the note on `AdamW::t`. A
    // single global counter would be wrong for any multi-tensor model.
    t: HashMap<String, u32>,
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
            t: HashMap::new(),
        }
    }

    pub fn step(&mut self, param_id: &str, grad: &[f32], param: &mut [f32]) {
        let t = {
            let e = self.t.entry(param_id.to_string()).or_insert(0);
            *e += 1;
            *e
        };

        let m = self
            .m
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);
        let v = self
            .v
            .entry(param_id.to_string())
            .or_insert_with(|| vec![0.0; grad.len()]);

        let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

        // First pass: build the per-element Adam direction (with decoupled weight
        // decay folded in) and accumulate the per-tensor norms of the parameters
        // and of the update. LAMB uses a single per-tensor ("layer-wise") trust
        // ratio, so the norms must be computed over the whole tensor before any
        // element is written back.
        let mut update = vec![0.0f32; grad.len()];
        let mut param_norm_sq = 0.0f32;
        let mut update_norm_sq = 0.0f32;
        for i in 0..grad.len()
        {
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            let r = m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * param[i];
            update[i] = r;
            param_norm_sq += param[i] * param[i];
            update_norm_sq += r * r;
        }

        let param_norm = param_norm_sq.sqrt();
        let update_norm = update_norm_sq.sqrt();

        // Trust ratio = ||param|| / ||update||, defaulting to 1.0 when either norm
        // vanishes (matching the reference implementation and NdLamb).
        let trust_ratio = if param_norm > 0.0 && update_norm > 0.0
        {
            param_norm / update_norm
        }
        else
        {
            1.0
        };

        // Second pass: apply the scaled update.
        for i in 0..grad.len()
        {
            param[i] -= self.learning_rate * trust_ratio * update[i];
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

    // Regression: the bias-correction timestep must be per-parameter. Two
    // parameters stepped once each must both behave as t==1. A shared global
    // counter would give the second parameter t==2 (inflated), producing a
    // different update than an identical single-parameter optimizer at t==1.
    #[test]
    fn adamw_timestep_is_per_parameter_not_global() {
        let grads = vec![0.1_f32, 0.2, 0.3];

        // Reference: a fresh optimizer taking one step on "a" (t == 1).
        let mut reference = AdamW::new(0.001);
        let mut ref_param = vec![1.0_f32, 2.0, 3.0];
        reference.step("a", &grads, &mut ref_param);

        // Two-parameter optimizer: step "a" then "b". With a per-parameter
        // timestep, "b"'s update must equal the reference (both at t==1).
        let mut multi = AdamW::new(0.001);
        let mut pa = vec![1.0_f32, 2.0, 3.0];
        let mut pb = vec![1.0_f32, 2.0, 3.0];
        multi.step("a", &grads, &mut pa);
        multi.step("b", &grads, &mut pb);

        for (got, want) in pb.iter().zip(ref_param.iter())
        {
            assert!(
                (got - want).abs() < 1e-9,
                "second parameter must see t==1, got {got} vs {want}"
            );
        }
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

    /// LAMB must apply a single per-tensor ("layer-wise") trust ratio
    /// `r = ||param|| / ||update||`, honour decoupled `weight_decay`, and be
    /// invariant to the internal element traversal order.
    ///
    /// Before the fix this test fails on all three counts: the trust ratio was
    /// computed per element using the scalar update as its own norm and using a
    /// param norm recomputed from `param` *while it was being mutated*, and
    /// `weight_decay` was completely ignored.
    #[test]
    fn test_lamb_layerwise_trust_ratio_and_weight_decay() {
        let lr = 0.1_f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let epsilon = 1e-8_f32;
        let weight_decay = 0.5_f32;

        let param = vec![1.0_f32, -2.0, 3.0, 0.5];
        let grad = vec![0.3_f32, -0.1, 0.2, 0.4];

        // Reference: exact single-step layer-wise LAMB (t == 1).
        let bc1 = 1.0 - beta1;
        let bc2 = 1.0 - beta2;
        let mut update = vec![0.0_f32; param.len()];
        let mut param_norm_sq = 0.0_f32;
        let mut update_norm_sq = 0.0_f32;
        for i in 0..param.len()
        {
            let m = (1.0 - beta1) * grad[i];
            let v = (1.0 - beta2) * grad[i] * grad[i];
            let m_hat = m / bc1;
            let v_hat = v / bc2;
            let r = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * param[i];
            update[i] = r;
            param_norm_sq += param[i] * param[i];
            update_norm_sq += r * r;
        }
        let trust = param_norm_sq.sqrt() / update_norm_sq.sqrt();
        let expected: Vec<f32> = param
            .iter()
            .zip(&update)
            .map(|(&p, &u)| p - lr * trust * u)
            .collect();

        let mut optimizer = LAMB::new(lr);
        optimizer.weight_decay = weight_decay;
        let mut params = param.clone();
        optimizer.step("param_0", &grad, &mut params);

        for (got, want) in params.iter().zip(&expected)
        {
            assert!(
                (got - want).abs() < 1e-5,
                "LAMB update mismatch: got {got}, want {want}"
            );
        }

        // The effective per-element step is `-lr * trust * update[i]`. Since a
        // single trust ratio scales every element, the ratio of two elements'
        // steps must equal the ratio of their raw update directions. The old
        // per-element code cannot reproduce this.
        let step0 = params[0] - param[0];
        let step2 = params[2] - param[2];
        assert!(
            (step0 / step2 - update[0] / update[2]).abs() < 1e-4,
            "per-element steps are not scaled by a single trust ratio"
        );

        // weight_decay must actually influence the result: a decay of 0 gives a
        // different update.
        let mut opt_no_wd = LAMB::new(lr);
        let mut params_no_wd = param.clone();
        opt_no_wd.step("param_0", &grad, &mut params_no_wd);
        assert!(
            params_no_wd
                .iter()
                .zip(&params)
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "weight_decay had no effect on the LAMB update"
        );
    }
}
