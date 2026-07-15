//! **Runtime value-bounds enforcement** (not a formal certificate).
//!
//! ⚠️ Naming honesty: despite the "certified/contract/invariant" vocabulary,
//! this module performs a **runtime clamp** — it inspects a tensor's values and,
//! if any fall outside `[MIN, MAX]` or are NaN/Inf, returns a scrubbed copy
//! (clamped, with NaN/Inf replaced by 0). There is **no proof, no static
//! guarantee, and no certificate**: it is a defensive output sanitizer, useful
//! for keeping activations finite/bounded, nothing more. For *provable* bounds
//! see `crown_ibp`/`ibp` (interval bounds), `lipschitz` (Lipschitz radius), or
//! `smoothing` (randomized smoothing).

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::Module;
use std::marker::PhantomData;

/// A runtime value-bounds check for a module's output.
pub trait Contract {
    /// Returns a sanitized copy of `t` if it violated the bounds, else `None`.
    fn validate(t: &Tensor) -> Option<Tensor>;
}

/// Clamps values into `[MIN, MAX]` (and scrubs NaN/Inf to 0) at runtime.
///
/// Note: the bounds are `i32` const generics cast to `f32`, so only **integer**
/// bounds are expressible (e.g. `[-1, 1]` works, `[-0.5, 0.5]` cannot).
pub struct ValueBoundedContract<const MIN_BITS: i32, const MAX_BITS: i32>;

impl<const MIN_BITS: i32, const MAX_BITS: i32> Contract
    for ValueBoundedContract<MIN_BITS, MAX_BITS>
{
    fn validate(t: &Tensor) -> Option<Tensor> {
        let min = MIN_BITS as f32;
        let max = MAX_BITS as f32;
        let mut violated = false;
        let mut clean_data = t.data.clone();

        for x in clean_data.iter_mut()
        {
            if *x < min || *x > max || x.is_nan() || x.is_infinite()
            {
                *x = x.clamp(min, max);
                if x.is_nan() || x.is_infinite()
                {
                    *x = 0.0; // Predictable fallback
                }
                violated = true;
            }
        }

        if violated
        {
            Some(Tensor::from_vec(clean_data, t.rows, t.cols))
        }
        else
        {
            None
        }
    }
}

/// A wrapper that applies a runtime [`Contract`] (value-bounds sanitizer) to a
/// module's output. Not a formal certificate — see the module note.
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
        let validated_input = if let Some(safe_input) = C::validate(&input_val)
        {
            tape.input(safe_input)
        }
        else
        {
            input
        };

        // 2. Execute inner module
        let output = self.inner.forward(tape, validated_input);

        // 3. Enforce contract on output
        let output_val = tape.value(output.idx());
        if let Some(safe_output) = C::validate(&output_val)
        {
            tape.input(safe_output)
        }
        else
        {
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
