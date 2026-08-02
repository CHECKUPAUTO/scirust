#![forbid(unsafe_code)]
//! Quantum isogeny and Shor's algorithm curve vulnerability evaluator.
//! Uses the dense/MPS quantum simulator from `scirust-core::quantum`.

use scirust_core::quantum::{Mps, gates};
use crate::ToyCurve;

/// Assessment results of Shor's algorithm simulation.
#[derive(Debug, Clone, Copy)]
pub struct ShorAssessment {
    pub simulated_qubits: usize,
    pub success_probability: f32,
    pub resonance_score: f32,
}

/// Assessment results of isogeny quantum walk simulation.
#[derive(Debug, Clone, Copy)]
pub struct IsogenyAssessment {
    pub graph_depth: usize,
    pub walk_amplitude_entropy: f32,
    pub security_bits: f32,
}

/// Dynamic evaluator for quantum-classical hybrid resistance pipelines.
pub struct QuantumIsogenyEvaluator {
    curve: ToyCurve,
}

impl QuantumIsogenyEvaluator {
    pub fn new(curve: ToyCurve) -> Self {
        Self { curve }
    }

    /// Evaluates Shor's algorithm susceptibility on the point order of the curve.
    pub fn evaluate_shor_security(&self, point_order: u64) -> ShorAssessment {
        // We simulate a phase estimation circuit with 4 qubits for the local toy prime domain.
        let num_qubits = 4;
        let mut mps = Mps::zero(num_qubits);

        // 1. Initialize qubits in superposition
        for q in 0..num_qubits {
            mps.apply_1qubit_gate(q, &gates::H);
        }

        // 2. Apply parameterized rotations modeling modular exponentiation (resonance of point_order)
        let angle = (point_order as f32) * std::f32::consts::FRAC_PI_4;
        let ry_gate = gates::ry(angle);
        for q in 0..num_qubits {
            mps.apply_1qubit_gate(q, &ry_gate);
        }

        // 3. Apply entangling gates to simulate the quantum arithmetic coupling
        for q in 0..(num_qubits - 1) {
            mps.apply_2qubit_gate(q, &gates::CNOT, 4);
        }

        // 4. Compute the amplitude vector and state collapse
        let statevector = mps.to_statevector();
        let norm_sq = mps.norm_sqr();

        // Calculate the maximum amplitude peak (the resonant peak)
        let max_amp_sq = statevector
            .iter()
            .map(|&amp| amp * amp)
            .fold(0.0f32, |max, val| if val > max { val } else { max });

        let success_probability = if norm_sq > 0.0 { max_amp_sq / norm_sq } else { 0.0 };
        // Resonance score based on standard deviation of amplitude distribution
        let resonance_score = success_probability * 100.0;

        ShorAssessment {
            simulated_qubits: num_qubits,
            success_probability,
            resonance_score,
        }
    }

    /// Evaluates curve resistance against isogeny-based quantum path searches.
    pub fn evaluate_isogeny_resistance(&self) -> IsogenyAssessment {
        // We model a 3-qubit quantum walk on a local isogeny graph
        let num_qubits = 3;
        let mut mps = Mps::zero(num_qubits);

        // Superposition representing the start of the quantum random walk
        for q in 0..num_qubits {
            mps.apply_1qubit_gate(q, &gates::H);
        }

        // Simulate step transitions using CNOT and CZ gates on the isogeny vertices
        mps.apply_2qubit_gate(0, &gates::CZ, 4);
        mps.apply_2qubit_gate(1, &gates::CNOT, 4);

        let statevector = mps.to_statevector();
        let total_prob: f32 = statevector.iter().map(|&a| a * a).sum();

        // Compute walk Shannon entropy representing amplitude dispersion
        let mut entropy = 0.0f32;
        for &amp in &statevector {
            let p = if total_prob > 0.0 { (amp * amp) / total_prob } else { 0.0 };
            if p > 1e-6 {
                entropy -= p * p.ln();
            }
        }

        // Security level is high if the path distribution remains highly dispersed (high entropy)
        let p_max = self.curve.prime().value() as f32;
        let security_bits = (p_max.log2() * (1.0 + entropy)).min(128.0);

        IsogenyAssessment {
            graph_depth: num_qubits,
            walk_amplitude_entropy: entropy,
            security_bits,
        }
    }

    /// Models resonance and collapse within a unified quantum-classical pipeline.
    pub fn model_hybrid_pipeline(&self, key_parameter: u32) -> f32 {
        let shor = self.evaluate_shor_security(key_parameter as u64);
        let isogeny = self.evaluate_isogeny_resistance();

        // Pipeline combination score of quantum vulnerability resonance and path dispersion collapse
        let collapse_factor = shor.success_probability * (1.0 - (isogeny.walk_amplitude_entropy / 3.0));
        collapse_factor.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::ToyPrime;

    #[test]
    fn test_quantum_evaluation_pipeline() {
        let prime = ToyPrime::new(17).unwrap();
        let curve = ToyCurve::new(prime, 1, 16).unwrap();
        let evaluator = QuantumIsogenyEvaluator::new(curve);

        let shor = evaluator.evaluate_shor_security(8);
        assert_eq!(shor.simulated_qubits, 4);
        assert!(shor.success_probability >= 0.0 && shor.success_probability <= 1.0);

        let isogeny = evaluator.evaluate_isogeny_resistance();
        assert_eq!(isogeny.graph_depth, 3);
        assert!(isogeny.security_bits > 0.0);

        let hybrid_score = evaluator.model_hybrid_pipeline(8);
        assert!(hybrid_score >= 0.0 && hybrid_score <= 1.0);
    }
}
