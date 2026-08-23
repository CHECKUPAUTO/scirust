//! Declarative adversarial numerical coverage for V2 programs.
//!
//! A single metadata-driven source of adversarial values and contract
//! properties. New operators (V2.1 index/gather family) inherit signed-zero,
//! NaN and infinity coverage by declaring their class here instead of
//! hand-writing matrices.
//!
//! The checks are bounded finite evidence, never proofs over the continuum.

use super::interpret::{ExecutionPolicy, ValueTensor, execute_program};
use super::ir::{Bin, Op, Ref, ResearchProgram, Section};
use super::types::{DType, ScalarValue};
use super::verify::VerificationLimits;

/// The canonical adversarial scalar set: signed zeros, quiet NaN, both
/// infinities, unit/fractional magnitudes and the binary64 range extremes.
#[must_use]
pub fn adversarial_scalars() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0,
        -1.0,
        0.5,
        -2.5,
        f64::MIN_POSITIVE,
        -(f64::MIN_POSITIVE),
        f64::MAX,
        f64::MIN,
    ]
}

/// Deterministic ordered pairs over [`adversarial_scalars`] including both
/// zero orders; capped to keep exhaustive sweeps cheap.
#[must_use]
pub fn adversarial_pairs() -> Vec<(f64, f64)> {
    let scalars = adversarial_scalars();
    let mut pairs = Vec::with_capacity(scalars.len() * scalars.len());
    for &left in &scalars
    {
        for &right in &scalars
        {
            pairs.push((left, right));
        }
    }
    pairs
}

/// The finite subset of the adversarial scalars, for operators whose
/// documented validity domain excludes non-finite operands.
#[must_use]
pub fn finite_adversarial_scalars() -> Vec<f64> {
    adversarial_scalars()
        .into_iter()
        .filter(|value| value.is_finite())
        .collect()
}

/// Filter helper retaining only pairs whose elements are all finite.
#[must_use]
pub fn only_finite_pairs(pairs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    pairs
        .iter()
        .copied()
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .collect()
}

/// Binary float operators sweepable by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

impl BinaryOpKind {
    /// The operator as an IR node reading two inputs.
    #[must_use]
    pub fn program(self) -> ResearchProgram {
        let op = match self
        {
            Self::Add => Op::Add(Bin::new(Ref::Input(0), Ref::Input(1))),
            Self::Sub => Op::Sub(Bin::new(Ref::Input(0), Ref::Input(1))),
            Self::Mul => Op::Mul(Bin::new(Ref::Input(0), Ref::Input(1))),
            Self::Div => Op::Div(Bin::new(Ref::Input(0), Ref::Input(1))),
            Self::Min => Op::Min(Bin::new(Ref::Input(0), Ref::Input(1))),
            Self::Max => Op::Max(Bin::new(Ref::Input(0), Ref::Input(1))),
        };
        ResearchProgram::expression(
            vec![scalar_type(), scalar_type()],
            Section::new(vec![op]),
            vec![0],
        )
    }
}

fn scalar_type() -> super::types::ValueType {
    super::types::ValueType::scalar(DType::F64)
}

fn scalar_tensor(value: f64) -> ValueTensor {
    ValueTensor::scalar_f64(value)
}

fn run_pair(
    program: &ResearchProgram,
    left: f64,
    right: f64,
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> Option<u64> {
    execute_program(
        program,
        &[scalar_tensor(left), scalar_tensor(right)],
        &[],
        policy,
        limits,
    )
    .ok()
    .map(|result| result.outputs[0].data[0].to_bits())
}

/// One operand-order observation for a binary operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwapObservation {
    pub left: f64,
    pub right: f64,
    pub forward_bits: u64,
    pub swapped_bits: u64,
    pub invariant: bool,
}

/// Sweep `op` over `pairs` and report whether swapping operands changes any
/// observable output bit. For `Min`/`Max` every case must be invariant by
/// contract; for `Add`/`Mul`/`Dot` invariance is only promised on the finite
/// domain, so pass [`only_finite_pairs`] there.
pub fn check_swap_invariance(
    op: BinaryOpKind,
    pairs: &[(f64, f64)],
    policy: ExecutionPolicy,
    limits: VerificationLimits,
) -> Vec<SwapObservation> {
    let forward = op.program();
    let swapped = swap_operands_program(op);
    let mut observations = Vec::with_capacity(pairs.len());
    for &(left, right) in pairs
    {
        // Both sides consume the SAME input order; only the operand
        // references inside the program are exchanged.
        let (Some(forward_bits), Some(swapped_bits)) = (
            run_pair(&forward, left, right, policy, limits),
            run_pair(&swapped, left, right, policy, limits),
        )
        else
        {
            // A policy rejection on one side is itself order information.
            observations.push(SwapObservation {
                left,
                right,
                forward_bits: u64::MAX,
                swapped_bits: u64::MAX,
                invariant: true,
            });
            continue;
        };
        observations.push(SwapObservation {
            left,
            right,
            forward_bits,
            swapped_bits,
            invariant: forward_bits == swapped_bits,
        });
    }
    observations
}

/// The same operator with its operand references exchanged.
fn swap_operands_program(op: BinaryOpKind) -> ResearchProgram {
    let node = match op
    {
        BinaryOpKind::Add => Op::Add(Bin::new(Ref::Input(1), Ref::Input(0))),
        BinaryOpKind::Sub => Op::Sub(Bin::new(Ref::Input(1), Ref::Input(0))),
        BinaryOpKind::Mul => Op::Mul(Bin::new(Ref::Input(1), Ref::Input(0))),
        BinaryOpKind::Div => Op::Div(Bin::new(Ref::Input(1), Ref::Input(0))),
        BinaryOpKind::Min => Op::Min(Bin::new(Ref::Input(1), Ref::Input(0))),
        BinaryOpKind::Max => Op::Max(Bin::new(Ref::Input(1), Ref::Input(0))),
    };
    ResearchProgram::expression(
        vec![scalar_type(), scalar_type()],
        Section::new(vec![node]),
        vec![0],
    )
}

/// Whether a constant is usable inside generated programs (NaN never is).
#[must_use]
pub fn admissible_constant(value: f64) -> bool {
    ScalarValue::F64(value).is_admissible()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::interpret::FloatPolicy;

    #[test]
    fn extrema_are_swap_invariant_over_the_full_adversarial_matrix() {
        for op in [BinaryOpKind::Min, BinaryOpKind::Max]
        {
            let observations = check_swap_invariance(
                op,
                &adversarial_pairs(),
                ExecutionPolicy {
                    floats: FloatPolicy::AllowNonFinite,
                },
                VerificationLimits::default(),
            );
            assert!(!observations.is_empty());
            for observation in &observations
            {
                assert!(
                    observation.invariant,
                    "{op:?} not swap-invariant at ({:?},{:?})",
                    observation.left, observation.right
                );
            }
        }
    }

    #[test]
    fn subtraction_is_not_swap_invariant_and_reports_it() {
        let observations = check_swap_invariance(
            BinaryOpKind::Sub,
            &[(3.0, 1.0)],
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        );
        assert_eq!(observations.len(), 1);
        assert!(!observations[0].invariant);
    }

    #[test]
    fn adversarial_set_contains_the_documented_extremes() {
        let scalars = adversarial_scalars();
        assert!(scalars.contains(&0.0));
        assert!(scalars.contains(&-0.0));
        assert!(scalars.iter().any(|v| v.is_nan()));
        assert_eq!(finite_adversarial_scalars().len(), scalars.len() - 3);
        assert_eq!(only_finite_pairs(&[(0.0, 1.0), (f64::NAN, 1.0)]).len(), 1);
        assert!(!admissible_constant(f64::NAN));
    }
}
