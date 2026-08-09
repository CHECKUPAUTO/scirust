//! Integration of [`HamiltonianReadout`] with SciRust's classical `nn::Module`
//! composition stack.

use super::HamiltonianReadout;
use crate::autodiff::reverse::{Tape, Var};
use crate::error::{Result, SciRustError};
use crate::nn::module::Module;

impl Module for HamiltonianReadout {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        self.try_forward(tape, input)
            .expect("HamiltonianReadout::forward received an invalid expectation tensor")
    }

    fn try_forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Result<Var<'t>> {
        if !std::ptr::eq(input.tape, tape)
        {
            return Err(SciRustError::InvalidConfig(
                "VQNet Hamiltonian readout input belongs to a different autodiff tape".to_string(),
            ));
        }

        self.apply(input).map_err(|error| {
            SciRustError::InvalidConfig(format!("VQNet Hamiltonian readout: {error}"))
        })
    }

    fn parameter_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    fn sync(&mut self, _tape: &Tape) {}
}
