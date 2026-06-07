use crate::autodiff::reverse::{Tape, Var, Tensor};
use crate::nn::Module;
use std::marker::PhantomData;

/// Trait defining a mathematical contract for a module.
pub trait Contract {
    /// Checks the tensor for contract violations and returns a safe fallback if necessary.
    fn validate(t: &Tensor) -> Option<Tensor>;
}

/// A contract that ensures values stay within [MIN, MAX] range.
pub struct ValueBoundedContract<const MIN_BITS: i32, const MAX_BITS: i32>;

impl<const MIN_BITS: i32, const MAX_BITS: i32> Contract for ValueBoundedContract<MIN_BITS, MAX_BITS> {
    fn validate(t: &Tensor) -> Option<Tensor> {
        let min = MIN_BITS as f32;
        let max = MAX_BITS as f32;
        let mut violated = false;
        let mut clean_data = t.data.clone();

        for x in clean_data.iter_mut() {
            if *x < min || *x > max || x.is_nan() || x.is_infinite() {
                *x = x.clamp(min, max);
                if x.is_nan() || x.is_infinite() {
                    *x = 0.0; // Predictable fallback
                }
                violated = true;
            }
        }

        if violated {
            Some(Tensor::from_vec(clean_data, t.rows, t.cols))
        } else {
            None
        }
    }
}

/// A wrapper that enforces formal invariants on a module's execution.
pub struct CertifiedModule<M: Module, C: Contract> {
    pub inner: M,
    _contract: PhantomData<C>,
}

impl<M: Module, C: Contract> CertifiedModule<M, C> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            _contract: PhantomData,
        }
    }
}

impl<M: Module, C: Contract> Module for CertifiedModule<M, C> {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        // 1. Enforce contract on input
        let input_val = tape.value(input.idx());
        let validated_input = if let Some(safe_input) = C::validate(&input_val) {
            tape.input(safe_input)
        } else {
            input
        };

        // 2. Execute inner module
        let output = self.inner.forward(tape, validated_input);

        // 3. Enforce contract on output
        let output_val = tape.value(output.idx());
        if let Some(safe_output) = C::validate(&output_val) {
            tape.input(safe_output)
        } else {
            output
        }
    }

    fn parameter_indices(&self) -> Vec<usize> {
        self.inner.parameter_indices()
    }

    fn sync(&mut self, tape: &Tape) {
        self.inner.sync(tape);
    }
}
