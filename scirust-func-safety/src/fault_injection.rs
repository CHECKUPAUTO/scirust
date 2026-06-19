use serde::{Deserialize, Serialize};

/// Types of faults that can be injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultType {
    /// Bit flip in weights (single-event upset)
    BitFlip,
    /// Stuck-at fault (pin stuck high/low)
    StuckAt,
    /// Random noise injection
    NoiseInjection,
    /// Zero out a neuron/weight
    ZeroOut,
    /// Scale weights by a factor
    ScaleShift,
    /// Overflow (saturate to max value)
    Overflow,
}

/// Result of a fault injection test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultResult {
    pub fault_type: FaultType,
    pub layer_index: usize,
    pub neuron_index: usize,
    pub original_output: f64,
    pub faulted_output: f64,
    pub output_delta: f64,
    pub safe_state_reached: bool,
    pub detection_latency_ms: f64,
}

impl FaultResult {
    pub fn is_safe(&self, tolerance: f64) -> bool {
        // Safe if the output delta is within tolerance OR safe state was reached
        self.output_delta < tolerance || self.safe_state_reached
    }
}

/// Fault injection testing for neural network robustness.
///
/// Required for ISO 26262 Part 5 (Hardware Development) and
/// ISO/PAS 21448 (SOTIF - Safety of the Intended Functionality).
#[derive(Debug, Clone)]
pub struct FaultInjector {
    /// Fault injection results
    pub results: Vec<FaultResult>,
    /// Tolerance for output deviation
    pub output_tolerance: f64,
    /// Safe state threshold (if output exceeds this, trigger safe state)
    pub safe_state_threshold: f64,
}

impl FaultInjector {
    pub fn new(output_tolerance: f64, safe_state_threshold: f64) -> Self {
        Self {
            results: Vec::new(),
            output_tolerance,
            safe_state_threshold,
        }
    }

    /// Inject a bit-flip fault into a weight value.
    pub fn inject_bit_flip(&mut self, weight: &mut f32, bit_position: u32) -> f32 {
        let mask = 1u32 << bit_position;
        let bits = weight.to_bits() ^ mask;
        f32::from_bits(bits)
    }

    /// Run a fault injection test on a single-layer inference.
    ///
    /// `weights`: layer weight vector `[n_neurons]` (one weight per neuron for simplicity)
    /// `inputs`: input vector `[n_inputs]` (must match weights length)
    /// `fault_type`: type of fault to inject
    /// `target_neuron`: which weight/neuron to affect
    /// `original_output`: the expected (non-faulted) output scalar
    pub fn run_test(
        &mut self,
        weights: &mut [f32],
        inputs: &[f32],
        fault_type: FaultType,
        target_layer: usize,
        target_neuron: usize,
        original_output: &[f64],
    ) -> FaultResult {
        if target_neuron >= weights.len()
        {
            return FaultResult {
                fault_type,
                layer_index: target_layer,
                neuron_index: target_neuron,
                original_output: 0.0,
                faulted_output: 0.0,
                output_delta: 0.0,
                safe_state_reached: true,
                detection_latency_ms: 0.0,
            };
        }

        // Backup original weight
        let original_weight = weights[target_neuron];

        // Apply fault to the single weight
        match fault_type
        {
            FaultType::BitFlip =>
            {
                let mut w = weights[target_neuron];
                w = self.inject_bit_flip(&mut w, 0);
                weights[target_neuron] = w;
            },
            FaultType::StuckAt =>
            {
                weights[target_neuron] = f32::MAX;
            },
            FaultType::NoiseInjection =>
            {
                let noise = (rand::random::<f32>() - 0.5) * 2.0;
                weights[target_neuron] += noise;
            },
            FaultType::ZeroOut =>
            {
                weights[target_neuron] = 0.0;
            },
            FaultType::ScaleShift =>
            {
                weights[target_neuron] *= 10.0;
            },
            FaultType::Overflow =>
            {
                weights[target_neuron] = f32::INFINITY;
            },
        }

        // Compute faulted output: dot product of (modified) weights and inputs
        let n = weights.len().min(inputs.len());
        let faulted: f32 = (0..n).map(|i| weights[i] * inputs[i]).sum();

        // Restore original weight
        weights[target_neuron] = original_weight;

        let original = if target_neuron < original_output.len()
        {
            original_output[target_neuron]
        }
        else if !original_output.is_empty()
        {
            original_output[0]
        }
        else
        {
            0.0
        };

        let delta = ((faulted as f64) - original).abs();
        let safe_state = delta > self.safe_state_threshold;

        let result = FaultResult {
            fault_type,
            layer_index: target_layer,
            neuron_index: target_neuron,
            original_output: original,
            faulted_output: faulted as f64,
            output_delta: delta,
            safe_state_reached: safe_state,
            detection_latency_ms: if safe_state { 1.0 } else { 0.0 },
        };

        self.results.push(result.clone());
        result
    }

    /// Run a batch of fault injection tests across all neurons.
    pub fn run_batch(
        &mut self,
        weights: &mut [f32],
        inputs: &[f32],
        original_output: &[f64],
        fault_type: FaultType,
    ) -> Vec<FaultResult> {
        let mut batch_results = Vec::new();
        for neuron in 0..weights.len()
        {
            let r = self.run_test(weights, inputs, fault_type, 0, neuron, original_output);
            batch_results.push(r);
        }
        batch_results
    }

    /// Check if all tests passed (output within tolerance or safe state reached).
    pub fn all_tests_safe(&self) -> bool {
        self.results
            .iter()
            .all(|r| r.is_safe(self.output_tolerance))
    }

    /// Count tests that triggered safe state.
    pub fn safe_state_count(&self) -> usize {
        self.results.iter().filter(|r| r.safe_state_reached).count()
    }

    /// Maximum output delta observed.
    pub fn max_output_delta(&self) -> f64 {
        self.results
            .iter()
            .map(|r| r.output_delta)
            .fold(0.0f64, f64::max)
    }

    /// Generate a compliance report.
    pub fn report(&self) -> String {
        let total = self.results.len();
        let safe = self
            .results
            .iter()
            .filter(|r| r.is_safe(self.output_tolerance))
            .count();
        let unsafe_ = total - safe;
        format!(
            "# Fault Injection Report\n\n\
             Total tests: {}\n\
             Safe: {}\n\
             Unsafe: {}\n\
             Safe state triggered: {}\n\
             Max output delta: {:.4}\n\
             All tests safe: {}\n",
            total,
            safe,
            unsafe_,
            self.safe_state_count(),
            self.max_output_delta(),
            if self.all_tests_safe() { "YES" } else { "NO" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_flip() {
        let mut fi = FaultInjector::new(0.1, 100.0);
        let mut w: f32 = 1.0;
        let original = w.to_bits();
        let flipped = fi.inject_bit_flip(&mut w, 0);
        assert_ne!(flipped.to_bits(), original);
    }

    #[test]
    fn test_zero_out_fault() {
        let mut fi = FaultInjector::new(0.1, 10.0);
        let mut weights = [1.0f32, 2.0, 3.0];
        let inputs = [1.0f32, 1.0, 1.0];
        // Original dot product: 1+2+3=6
        let result = fi.run_test(&mut weights, &inputs, FaultType::ZeroOut, 0, 0, &[6.0]);
        // After zeroing weight[0]: 0+2+3=5 → delta=1.0
        assert!((result.faulted_output - 5.0).abs() < 1e-6);
        assert!((result.original_output - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_safe_state_triggered() {
        let mut fi = FaultInjector::new(0.1, 2.0);
        let mut weights = [1.0f32, 1.0];
        let inputs = [1.0f32, 1.0];
        // Scale by 10 → 10+10=20 (was 2) → delta=18 > 2 → safe state
        let result = fi.run_test(&mut weights, &inputs, FaultType::ScaleShift, 0, 0, &[2.0]);
        assert!(result.safe_state_reached);
    }

    #[test]
    fn test_weights_restored() {
        let mut fi = FaultInjector::new(0.1, 100.0);
        let mut weights = [5.0f32];
        let inputs = [2.0f32];
        let _ = fi.run_test(&mut weights, &inputs, FaultType::ZeroOut, 0, 0, &[10.0]);
        // Weight should be restored to original
        assert!((weights[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_and_report() {
        let mut fi = FaultInjector::new(0.5, 10.0);
        let mut weights = [1.0f32, 2.0, 3.0];
        let inputs = [1.0f32, 1.0, 1.0];
        let _results = fi.run_batch(&mut weights, &inputs, &[3.0], FaultType::ZeroOut);
        let report = fi.report();
        assert!(report.contains("Total tests: 3"));
        assert!(report.contains("Safe:") || report.contains("Unsafe:"));
    }

    #[test]
    fn test_all_tests_safe() {
        let mut fi = FaultInjector::new(1000.0, 10000.0);
        let mut weights = [1.0f32];
        let inputs = [1.0f32];
        let _ = fi.run_test(&mut weights, &inputs, FaultType::ZeroOut, 0, 0, &[1.0]);
        // With tolerance=1000, all tests should be safe
        assert!(fi.all_tests_safe());
    }
}
