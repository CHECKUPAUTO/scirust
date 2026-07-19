//! Minimal truthful backend abstraction and dense CPU implementation.

use super::error::QuantumResult;
use super::ir::BoundCircuit;
use super::observable::Observable;
use std::collections::BTreeMap;

/// Capabilities currently exposed by a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub maximum_qubits: usize,
    pub exact_expectation: bool,
    pub shot_sampling: bool,
    pub parameter_shift_gradients: bool,
    pub noise: bool,
    pub batching: bool,
    pub dynamic_circuits: bool,
    pub local: bool,
}

/// One exact expectation and/or one explicitly seeded sampling request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRequest {
    pub observable: Option<Observable>,
    pub sampling: Option<(usize, u64)>,
}

/// Values returned by backend execution.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumExecutionResult {
    pub expectation: Option<f32>,
    pub counts: Option<BTreeMap<String, usize>>,
}

/// Backend boundary between circuit representation and execution.
pub trait QuantumBackend {
    type Error;

    fn capabilities(&self) -> BackendCapabilities;

    fn execute(
        &self,
        circuit: &BoundCircuit,
        request: &ExecutionRequest,
    ) -> Result<QuantumExecutionResult, Self::Error>;
}

/// Exact dense complex CPU backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct DenseBackend;

impl QuantumBackend for DenseBackend {
    type Error = super::error::QuantumError;

    fn capabilities(&self) -> BackendCapabilities {
        // 2^27 Complex32 values consume exactly the explicit one-GiB limit.
        BackendCapabilities {
            maximum_qubits: 27,
            exact_expectation: true,
            shot_sampling: true,
            parameter_shift_gradients: false,
            noise: false,
            batching: false,
            dynamic_circuits: false,
            local: true,
        }
    }

    fn execute(
        &self,
        circuit: &BoundCircuit,
        request: &ExecutionRequest,
    ) -> QuantumResult<QuantumExecutionResult> {
        let state = circuit.execute_dense()?;
        let expectation = request
            .observable
            .as_ref()
            .map(|observable| state.expectation(observable))
            .transpose()?;
        let counts = request
            .sampling
            .map(|(shots, seed)| state.sample(shots, seed))
            .transpose()?;
        Ok(QuantumExecutionResult {
            expectation,
            counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::ir::{Circuit, Operation, ParameterValues};

    #[test]
    fn dense_backend_advertises_only_implemented_capabilities() {
        let capabilities = DenseBackend.capabilities();
        assert!(capabilities.exact_expectation);
        assert!(capabilities.shot_sampling);
        assert!(!capabilities.parameter_shift_gradients);
        assert!(!capabilities.noise);
        assert!(!capabilities.batching);
        assert!(!capabilities.dynamic_circuits);
        assert!(capabilities.local);
    }

    #[test]
    fn backend_executes_exact_and_sampled_requests_separately() {
        let mut circuit = Circuit::new(1).unwrap();
        circuit.push(Operation::H { target: 0 }).unwrap();
        let bound = circuit.bind(&ParameterValues::new()).unwrap();
        let exact = DenseBackend
            .execute(
                &bound,
                &ExecutionRequest {
                    observable: Some(Observable::x(0)),
                    sampling: None,
                },
            )
            .unwrap();
        assert!((exact.expectation.unwrap() - 1.0).abs() < 3.0e-5);
        assert!(exact.counts.is_none());
    }
}
