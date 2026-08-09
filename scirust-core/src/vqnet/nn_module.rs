//! Integration of [`QuantumModule`] with SciRust's classical
//! [`Module`](crate::nn::module::Module) stack.

use super::QuantumModule;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::error::{Result, SciRustError};
use crate::nn::module::Module;
use std::collections::HashMap;

const PARAMETER_KEY: &str = "parameters";

impl Module for QuantumModule {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        self.try_forward(tape, input)
            .expect("QuantumModule::forward received an invalid quantum input")
    }

    fn try_forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Result<Var<'t>> {
        let forward = QuantumModule::forward_batch(self, tape, input).map_err(|error| {
            SciRustError::InvalidConfig(format!("VQNet quantum module forward: {error}"))
        })?;
        self.record_parameter_index(forward.parameter_index());
        Ok(forward.output())
    }

    fn parameter_indices(&self) -> Vec<usize> {
        self.last_parameter_index().into_iter().collect()
    }

    fn sync(&mut self, tape: &Tape) {
        self.sync_last_from_tape(tape)
            .expect("QuantumModule::sync encountered invalid trainable state");
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::from([(PARAMETER_KEY.to_string(), self.parameter_tensor())])
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> Result<()> {
        let tensor = state.get(PARAMETER_KEY).ok_or_else(|| {
            SciRustError::InvalidConfig(format!(
                "VQNet quantum module state is missing key: {PARAMETER_KEY}"
            ))
        })?;

        self.replace_parameter_tensor(tensor.clone()).map_err(|error| {
            SciRustError::InvalidConfig(format!("invalid VQNet quantum module state: {error}"))
        })
    }
}
