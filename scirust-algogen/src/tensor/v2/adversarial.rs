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
///
/// Every element is exact `binary64`. For `f32` probes use
/// [`adversarial_scalars_f32`] instead: several `binary64` extremes are not
/// representable in `binary32`, and feeding them to an `F32` tensor would
/// silently turn an `f32` experiment into an `f64` one (or reject it).
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

/// The `binary32` counterpart of [`adversarial_scalars`]: same coverage
/// classes (signed zeros, NaN, ±∞, unit/fractional magnitudes, smallest
/// positive normal, range extremes) but every element is exactly
/// representable as `f32`, so it survives the lossless-carrier check of
/// `ValueTensor::new(DType::F32, …)` unchanged.
#[must_use]
pub fn adversarial_scalars_f32() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        f32::NAN as f64,
        f32::INFINITY as f64,
        f32::NEG_INFINITY as f64,
        1.0,
        -1.0,
        0.5,
        -2.5,
        f32::MIN_POSITIVE as f64,
        -(f32::MIN_POSITIVE as f64),
        f32::MAX as f64,
        f32::MIN as f64,
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

/// Observable outcome of one operand-order execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SwapOutcome {
    /// The execution succeeded; the first output element's exact bits.
    Value(u64),
    /// The execution was rejected under the policy (non-finite input,
    /// non-finite intermediate, budget…). Never encoded as a magic bit value.
    Rejected,
}

/// What the forward/swapped pair of observations establishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapRelation {
    /// Both executions succeeded and produced identical bits: positive
    /// evidence of order invariance *on this case*.
    BitIdentical,
    /// Both executed but the output bits differ: order dependence.
    ValueDivergence,
    /// Exactly one side executed: outcome asymmetry is itself order
    /// dependence, never invariance.
    OutcomeAsymmetric,
    /// Both sides rejected. This is *symmetric rejection* — recorded as its
    /// own class because it is NOT positive value-level evidence of
    /// invariance.
    SymmetricRejection,
}

/// One operand-order observation for a binary operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwapObservation {
    pub left: f64,
    pub right: f64,
    pub forward: SwapOutcome,
    pub swapped: SwapOutcome,
}

impl SwapObservation {
    /// Classify this observation honestly: only two successful executions
    /// with identical bits count as (bounded) invariance evidence.
    #[must_use]
    pub fn relation(&self) -> SwapRelation {
        match (self.forward, self.swapped)
        {
            (SwapOutcome::Value(forward_bits), SwapOutcome::Value(swapped_bits)) =>
            {
                if forward_bits == swapped_bits
                {
                    SwapRelation::BitIdentical
                }
                else
                {
                    SwapRelation::ValueDivergence
                }
            },
            (SwapOutcome::Rejected, SwapOutcome::Rejected) => SwapRelation::SymmetricRejection,
            (SwapOutcome::Value(_), SwapOutcome::Rejected)
            | (SwapOutcome::Rejected, SwapOutcome::Value(_)) => SwapRelation::OutcomeAsymmetric,
        }
    }
}

/// Sweep `op` over `pairs` and report, per pair, what swapping operands
/// changes about the observable outcome. For `Min`/`Max` every pair must be
/// at least [`SwapRelation::BitIdentical`] by contract; for
/// `Add`/`Mul`/`Dot` invariance is only promised on the finite domain, so
/// pass [`only_finite_pairs`] there.
///
/// Rejections are preserved as first-class outcomes ([`SwapOutcome::Rejected`]);
/// they are never folded into `invariant = true`.
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
        let forward_outcome = run_pair(&forward, left, right, policy, limits);
        let swapped_outcome = run_pair(&swapped, left, right, policy, limits);
        observations.push(SwapObservation {
            left,
            right,
            forward: forward_outcome.map_or(SwapOutcome::Rejected, SwapOutcome::Value),
            swapped: swapped_outcome.map_or(SwapOutcome::Rejected, SwapOutcome::Value),
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
                assert_eq!(
                    observation.relation(),
                    SwapRelation::BitIdentical,
                    "{op:?} not swap-invariant at ({:?},{:?}): {observation:?}",
                    observation.left,
                    observation.right
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
        assert_eq!(observations[0].relation(), SwapRelation::ValueDivergence);
    }

    /// Regression: a policy rejection on both sides used to be reported as
    /// `invariant = true` via the `u64::MAX` sentinel. It must be recorded as
    /// its own class — symmetric rejection is NOT invariance evidence.
    #[test]
    fn jointly_rejected_cases_are_symmetric_rejections_never_invariant() {
        // NaN inputs are rejected outright by the default finite policy.
        let observations = check_swap_invariance(
            BinaryOpKind::Min,
            &[(f64::NAN, f64::NAN)],
            ExecutionPolicy::default(),
            VerificationLimits::default(),
        );
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].relation(), SwapRelation::SymmetricRejection);
    }

    /// Outcome asymmetry must classify as order dependence, never as
    /// invariance, whatever the underlying cause.
    #[test]
    fn asymmetric_outcomes_are_order_dependence() {
        let forward_only = SwapObservation {
            left: 1.0,
            right: 2.0,
            forward: SwapOutcome::Value(7),
            swapped: SwapOutcome::Rejected,
        };
        let swapped_only = SwapObservation {
            left: 1.0,
            right: 2.0,
            forward: SwapOutcome::Rejected,
            swapped: SwapOutcome::Value(7),
        };
        assert_eq!(forward_only.relation(), SwapRelation::OutcomeAsymmetric);
        assert_eq!(swapped_only.relation(), SwapRelation::OutcomeAsymmetric);
        assert_ne!(forward_only.relation(), SwapRelation::BitIdentical);
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

    /// Every `f32`-class adversarial value must survive the exact-binary32
    /// round trip, unlike several `binary64` extremes.
    #[test]
    fn f32_adversarial_set_is_exactly_representable() {
        for &value in &adversarial_scalars_f32()
        {
            let round_trip = (value as f32) as f64;
            let exact = if value.is_nan()
            {
                round_trip.is_nan()
            }
            else
            {
                round_trip.to_bits() == value.to_bits()
            };
            assert!(exact, "{value} is not exactly representable as f32");
        }
        // The binary64 set deliberately contains values that would fail that
        // check — which is why the f32 set exists.
        assert!(
            !adversarial_scalars()
                .iter()
                .filter(|value| !value.is_nan())
                .all(|&value| ((value as f32) as f64).to_bits() == value.to_bits())
        );
        assert_eq!(adversarial_scalars().len(), adversarial_scalars_f32().len());
    }
}
