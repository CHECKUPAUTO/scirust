//! **Runtime value-bounds enforcement** (not a formal certificate).
//!
//! ⚠️ Naming honesty: despite the "certified/contract/invariant" vocabulary,
//! this module performs a **runtime clamp** — it inspects a tensor's values and
//! returns a scrubbed copy where out-of-range and ±Inf values are clamped to the
//! nearest bound (`MIN`/`MAX`) and NaN is replaced by a deterministic value
//! inside the declared interval. There is **no proof, no static guarantee, and
//! no certificate**: it is a defensive output sanitizer, useful for keeping
//! activations finite/bounded, nothing more. For *provable* bounds see
//! `crown_ibp`/`ibp` (interval bounds), `lipschitz` (Lipschitz radius), or
//! `smoothing` (randomized smoothing).

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::Module;
use std::marker::PhantomData;

/// A runtime value-bounds check for a module's output.
pub trait Contract {
    /// Returns a sanitized copy of `t` if it violated the bounds, else `None`.
    fn validate(t: &Tensor) -> Option<Tensor>;
}

/// Clamps values into `[MIN, MAX]` at runtime and replaces NaN with a
/// deterministic value that is guaranteed to lie inside that interval.
///
/// Note: the bounds are `i32` const generics cast to `f32`, so only **integer**
/// bounds are expressible (e.g. `[-1, 1]` works, `[-0.5, 0.5]` cannot).
pub struct ValueBoundedContract<const MIN_BITS: i32, const MAX_BITS: i32>;

impl<const MIN_BITS: i32, const MAX_BITS: i32> Contract
    for ValueBoundedContract<MIN_BITS, MAX_BITS>
{
    /// # Panics
    ///
    /// Panics if `MIN_BITS > MAX_BITS` because such a type does not describe a
    /// valid interval.
    fn validate(t: &Tensor) -> Option<Tensor> {
        let min = MIN_BITS as f32;
        let max = MAX_BITS as f32;
        assert!(
            min <= max,
            "ValueBoundedContract: contract bounds must satisfy MIN <= MAX"
        );

        let mut violated = false;
        let mut clean_data = t.data.clone();

        for x in clean_data.iter_mut()
        {
            if x.is_nan()
            {
                // Prefer zero when it satisfies the contract. For intervals
                // wholly above/below zero, the nearest endpoint is the least
                // surprising deterministic replacement and remains in-bounds.
                *x = if min <= 0.0 && 0.0 <= max
                {
                    0.0
                }
                else if min > 0.0
                {
                    min
                }
                else
                {
                    max
                };
                violated = true;
            }
            else if *x < min || *x > max || x.is_infinite()
            {
                *x = x.clamp(min, max);
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

    fn train(&mut self, on: bool) {
        self.inner.train(on);
    }

    fn parameter_indices(&self) -> Vec<usize> {
        self.inner.parameter_indices()
    }

    fn sync(&mut self, tape: &Tape) {
        self.inner.sync(tape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: Vec<f32>) -> Tensor {
        Tensor::from_vec(values, 1, 1)
    }

    #[test]
    fn nan_fallback_stays_inside_positive_interval() {
        let clean = ValueBoundedContract::<1, 2>::validate(&tensor(vec![f32::NAN])).unwrap();
        assert_eq!(clean.data, vec![1.0]);
    }

    #[test]
    fn nan_fallback_stays_inside_negative_interval() {
        let clean = ValueBoundedContract::<-2, -1>::validate(&tensor(vec![f32::NAN])).unwrap();
        assert_eq!(clean.data, vec![-1.0]);
    }

    #[test]
    fn nan_fallback_prefers_zero_when_zero_is_in_bounds() {
        let clean = ValueBoundedContract::<-1, 1>::validate(&tensor(vec![f32::NAN])).unwrap();
        assert_eq!(clean.data, vec![0.0]);
    }

    #[test]
    fn infinities_clamp_to_nearest_bound() {
        let t = Tensor::from_vec(vec![f32::NEG_INFINITY, f32::INFINITY], 1, 2);
        let clean = ValueBoundedContract::<-2, 3>::validate(&t).unwrap();
        assert_eq!(clean.data, vec![-2.0, 3.0]);
    }

    #[test]
    fn in_range_tensor_is_not_copied() {
        let t = Tensor::from_vec(vec![1.0, 2.0], 1, 2);
        assert!(ValueBoundedContract::<1, 2>::validate(&t).is_none());
    }

    #[test]
    #[should_panic(expected = "contract bounds must satisfy MIN <= MAX")]
    fn reversed_bounds_are_rejected_explicitly() {
        let _ = ValueBoundedContract::<2, 1>::validate(&tensor(vec![1.5]));
    }
}
