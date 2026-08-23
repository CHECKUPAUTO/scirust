//! Conservative lightweight constant/sign/finiteness analysis.
//!
//! This pass is intentionally incomplete. It summarizes all elements of a
//! tensor with one abstract value and uses a bounded recurrence iteration. It
//! may answer "unknown" even when a stronger theorem is true; it must never be
//! used as a proof of arbitrary real-algebraic equivalence.

use serde::{Deserialize, Serialize};

use super::ir::{Op, Ref, ResearchProgram, Section};
use super::semantics::NumericalSemantics;
use super::types::{DType, ScalarValue};
use super::verify::{ProgramError, VerificationLimits, VerifiedProgram, verify_program};

/// Coarse sign class shared by every element in a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sign {
    Negative,
    NonPositive,
    Zero,
    NonNegative,
    Positive,
    Unknown,
}

/// Possible floating-point exceptional values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Finiteness {
    /// Every element is known finite.
    Finite,
    /// Infinity may occur, but NaN is not currently implied.
    MayBeInfinite,
    /// NaN may occur (infinity and finite values may also occur).
    MayBeNan,
    /// No useful fact is known.
    Unknown,
}

/// Closed finite interval. Bounds are never NaN or infinite.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

/// Abstract facts for one scalar or tensor value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValueFacts {
    pub constant: Option<ScalarValue>,
    pub sign: Sign,
    pub finiteness: Finiteness,
    pub interval: Option<Interval>,
}

impl ValueFacts {
    pub const fn unknown() -> Self {
        Self {
            constant: None,
            sign: Sign::Unknown,
            finiteness: Finiteness::Unknown,
            interval: None,
        }
    }

    pub const fn finite_unknown() -> Self {
        Self {
            finiteness: Finiteness::Finite,
            ..Self::unknown()
        }
    }

    fn bool_unknown() -> Self {
        Self {
            finiteness: Finiteness::Finite,
            ..Self::unknown()
        }
    }

    fn from_constant(value: ScalarValue) -> Self {
        match value
        {
            ScalarValue::Bool(_) => Self {
                constant: Some(value),
                sign: Sign::Unknown,
                finiteness: Finiteness::Finite,
                interval: None,
            },
            ScalarValue::F32(inner) => float_constant(value, f64::from(inner)),
            ScalarValue::F64(inner) => float_constant(value, inner),
        }
    }
}

fn float_constant(value: ScalarValue, inner: f64) -> ValueFacts {
    ValueFacts {
        constant: Some(value),
        sign: sign_of(inner),
        finiteness: if inner.is_finite()
        {
            Finiteness::Finite
        }
        else if inner.is_infinite()
        {
            Finiteness::MayBeInfinite
        }
        else
        {
            Finiteness::MayBeNan
        },
        interval: inner.is_finite().then_some(Interval {
            lower: inner,
            upper: inner,
        }),
    }
}

fn sign_of(value: f64) -> Sign {
    if value == 0.0
    {
        Sign::Zero
    }
    else if value > 0.0
    {
        Sign::Positive
    }
    else if value < 0.0
    {
        Sign::Negative
    }
    else
    {
        Sign::Unknown
    }
}

/// Full-program analysis result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RangeAnalysis {
    pub init: Vec<ValueFacts>,
    pub step: Vec<ValueFacts>,
    pub finalize: Vec<ValueFacts>,
    pub final_state: Vec<ValueFacts>,
    /// Number of bounded abstract scan iterations actually performed.
    pub recurrence_iterations: u32,
    /// Whether the abstract recurrence reached a fixed point before the cap.
    pub recurrence_converged: bool,
}

/// Hard analysis cap. It bounds analysis cost independently of program steps.
pub const MAX_RANGE_RECURRENCE_PASSES: u32 = 8;

/// Verify and analyze a program.
pub fn analyze_ranges(
    program: &ResearchProgram,
    limits: VerificationLimits,
) -> Result<RangeAnalysis, ProgramError> {
    let verified = verify_program(program, limits)?;
    Ok(analyze_ranges_verified(program, &verified))
}

/// Analyze an already verified program.
pub fn analyze_ranges_verified(
    program: &ResearchProgram,
    verified: &VerifiedProgram,
) -> RangeAnalysis {
    let assumed_finite = program.semantics == NumericalSemantics::FiniteOnly;
    let input_facts = signature_facts(&program.inputs, assumed_finite);
    let item_facts = signature_facts(&program.items, assumed_finite);

    let init = analyze_section(
        &program.init,
        &input_facts,
        &[],
        &[],
        &verified
            .init_types
            .iter()
            .map(|value| value.dtype)
            .collect::<Vec<_>>(),
    );
    let mut state = program
        .init_state
        .iter()
        .map(|&id| init.get(id).copied().unwrap_or_else(ValueFacts::unknown))
        .collect::<Vec<_>>();

    let mut step = Vec::new();
    let mut iterations = 0;
    let mut converged = program.steps == 0;
    let cap = program.steps.min(MAX_RANGE_RECURRENCE_PASSES);
    for _ in 0..cap
    {
        step = analyze_section(
            &program.step,
            &[],
            &item_facts,
            &state,
            &verified
                .step_types
                .iter()
                .map(|value| value.dtype)
                .collect::<Vec<_>>(),
        );
        let next = program
            .next_state
            .iter()
            .map(|&id| step.get(id).copied().unwrap_or_else(ValueFacts::unknown))
            .collect::<Vec<_>>();
        iterations += 1;
        if next == state
        {
            state = next;
            converged = true;
            break;
        }
        state = state
            .iter()
            .copied()
            .zip(next)
            .map(|(left, right)| join(left, right))
            .collect();
    }

    // A truncated recurrence is widened to facts that remain conservative.
    if iterations < program.steps && !converged
    {
        state.iter_mut().for_each(|fact| {
            fact.constant = None;
            fact.interval = None;
            fact.sign = widen_sign(fact.sign);
        });
    }

    let finalize = analyze_section(
        &program.finalize,
        &input_facts,
        &[],
        &state,
        &verified
            .finalize_types
            .iter()
            .map(|value| value.dtype)
            .collect::<Vec<_>>(),
    );

    RangeAnalysis {
        init,
        step,
        finalize,
        final_state: state,
        recurrence_iterations: iterations,
        recurrence_converged: converged,
    }
}

fn signature_facts(types: &[super::types::ValueType], finite: bool) -> Vec<ValueFacts> {
    types
        .iter()
        .map(|value_type| {
            if value_type.dtype == DType::Bool || finite
            {
                ValueFacts::finite_unknown()
            }
            else
            {
                ValueFacts::unknown()
            }
        })
        .collect()
}

fn analyze_section(
    section: &Section,
    inputs: &[ValueFacts],
    items: &[ValueFacts],
    state: &[ValueFacts],
    dtypes: &[DType],
) -> Vec<ValueFacts> {
    let mut facts = Vec::with_capacity(section.ops.len());
    for (node, op) in section.ops.iter().enumerate()
    {
        let mut operands = Vec::with_capacity(3);
        op.for_each_ref(|reference| {
            operands.push(resolve_fact(reference, &facts, inputs, items, state));
        });
        facts.push(transfer(
            op,
            &operands,
            dtypes.get(node).copied().unwrap_or(DType::F64),
        ));
    }
    facts
}

fn resolve_fact(
    reference: Ref,
    locals: &[ValueFacts],
    inputs: &[ValueFacts],
    items: &[ValueFacts],
    state: &[ValueFacts],
) -> ValueFacts {
    match reference
    {
        Ref::Input(index) => inputs.get(index),
        Ref::Local(index) => locals.get(index),
        Ref::Item(index) => items.get(index),
        Ref::StatePrev(index) | Ref::StateFinal(index) => state.get(index),
    }
    .copied()
    .unwrap_or_else(ValueFacts::unknown)
}

#[allow(clippy::too_many_lines)]
fn transfer(op: &Op, operands: &[ValueFacts], dtype: DType) -> ValueFacts {
    let at = |index: usize| {
        operands
            .get(index)
            .copied()
            .unwrap_or_else(ValueFacts::unknown)
    };
    match op
    {
        Op::Const(value) => ValueFacts::from_constant(*value),
        Op::Neg(_) => negate(at(0)),
        Op::Abs(_) => ValueFacts {
            constant: None,
            sign: Sign::NonNegative,
            finiteness: at(0).finiteness,
            interval: at(0).interval.map(|interval| {
                let upper = interval.lower.abs().max(interval.upper.abs());
                Interval {
                    lower: if interval.lower <= 0.0 && interval.upper >= 0.0
                    {
                        0.0
                    }
                    else
                    {
                        interval.lower.abs().min(interval.upper.abs())
                    },
                    upper,
                }
            }),
        },
        Op::Add(_) => arithmetic_binary(at(0), at(1), Arithmetic::Add),
        Op::Sub(_) => arithmetic_binary(at(0), at(1), Arithmetic::Sub),
        Op::Mul(_) | Op::MulAdd(_) | Op::Pow(_) => ValueFacts {
            finiteness: exceptional_binary(at(0), at(1)),
            ..ValueFacts::unknown()
        },
        Op::Div(_) | Op::Rsqrt(_) | Op::Log(_) | Op::Log2(_) | Op::Log1p(_) | Op::Sqrt(_) =>
        {
            ValueFacts {
                finiteness: Finiteness::MayBeNan,
                ..ValueFacts::unknown()
            }
        },
        Op::Exp(_) | Op::Exp2(_) | Op::Expm1(_) => ValueFacts {
            sign: if matches!(op, Op::Expm1(_))
            {
                Sign::Unknown
            }
            else
            {
                Sign::NonNegative
            },
            finiteness: Finiteness::MayBeInfinite,
            ..ValueFacts::unknown()
        },
        Op::Sin(_) | Op::Cos(_) | Op::Tanh(_) => ValueFacts {
            finiteness: if at(0).finiteness == Finiteness::Finite
            {
                Finiteness::Finite
            }
            else
            {
                Finiteness::MayBeNan
            },
            interval: Some(Interval {
                lower: -1.0,
                upper: 1.0,
            }),
            ..ValueFacts::unknown()
        },
        Op::Min(_) | Op::Max(_) | Op::Clamp(_) =>
        {
            let mut result = join(at(0), at(1));
            if matches!(op, Op::Clamp(_))
            {
                result = join(result, at(2));
            }
            result.constant = None;
            result
        },
        Op::Select(_) => join(at(1), at(2)),
        Op::Eq(_)
        | Op::Ne(_)
        | Op::Lt(_)
        | Op::Le(_)
        | Op::Gt(_)
        | Op::Ge(_)
        | Op::And(_)
        | Op::Or(_)
        | Op::Not(_) => ValueFacts::bool_unknown(),
        Op::ReduceSum(_) | Op::ReduceMean(_) | Op::ReduceMax(_) | Op::ReduceMin(_) =>
        {
            let source = at(0);
            ValueFacts {
                constant: None,
                sign: preserve_reduction_sign(source.sign),
                finiteness: if source.finiteness == Finiteness::Finite
                {
                    Finiteness::MayBeInfinite
                }
                else
                {
                    source.finiteness
                },
                interval: None,
            }
        },
        Op::ReduceProd(_)
        | Op::Dot(_)
        | Op::MatVec(_)
        | Op::VecMat(_)
        | Op::MatMul(_)
        | Op::BatchedMatMul(_)
        | Op::Outer(_) => ValueFacts {
            finiteness: exceptional_binary(at(0), at(1)),
            ..ValueFacts::unknown()
        },
        Op::Reshape(_)
        | Op::Squeeze(_)
        | Op::Unsqueeze(_)
        | Op::Transpose(_)
        | Op::BroadcastTo(_)
        | Op::Narrow(_) => at(0),
        Op::Concat { .. } => join(at(0), at(1)),
    }
    .with_dtype(dtype)
}

trait FactsDType {
    fn with_dtype(self, dtype: DType) -> Self;
}

impl FactsDType for ValueFacts {
    fn with_dtype(mut self, dtype: DType) -> Self {
        if dtype == DType::Bool
        {
            self.sign = Sign::Unknown;
            self.interval = None;
            self.finiteness = Finiteness::Finite;
        }
        self
    }
}

#[derive(Clone, Copy)]
enum Arithmetic {
    Add,
    Sub,
}

fn arithmetic_binary(left: ValueFacts, right: ValueFacts, operation: Arithmetic) -> ValueFacts {
    let interval = match (left.interval, right.interval, operation)
    {
        (Some(a), Some(b), Arithmetic::Add) =>
        {
            checked_interval(a.lower + b.lower, a.upper + b.upper)
        },
        (Some(a), Some(b), Arithmetic::Sub) =>
        {
            checked_interval(a.lower - b.upper, a.upper - b.lower)
        },
        _ => None,
    };
    ValueFacts {
        constant: None,
        sign: interval.map_or(Sign::Unknown, interval_sign),
        finiteness: exceptional_binary(left, right),
        interval,
    }
}

fn checked_interval(lower: f64, upper: f64) -> Option<Interval> {
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Interval { lower, upper })
}

fn interval_sign(interval: Interval) -> Sign {
    if interval.lower > 0.0
    {
        Sign::Positive
    }
    else if interval.lower >= 0.0
    {
        Sign::NonNegative
    }
    else if interval.upper < 0.0
    {
        Sign::Negative
    }
    else if interval.upper <= 0.0
    {
        Sign::NonPositive
    }
    else
    {
        Sign::Unknown
    }
}

fn exceptional_binary(left: ValueFacts, right: ValueFacts) -> Finiteness {
    match (left.finiteness, right.finiteness)
    {
        (Finiteness::MayBeNan | Finiteness::Unknown, _)
        | (_, Finiteness::MayBeNan | Finiteness::Unknown) => Finiteness::MayBeNan,
        (Finiteness::MayBeInfinite, _) | (_, Finiteness::MayBeInfinite) => Finiteness::MayBeNan,
        (Finiteness::Finite, Finiteness::Finite) => Finiteness::MayBeInfinite,
    }
}

fn negate(mut value: ValueFacts) -> ValueFacts {
    value.constant = None;
    value.sign = match value.sign
    {
        Sign::Negative => Sign::Positive,
        Sign::NonPositive => Sign::NonNegative,
        Sign::Zero => Sign::Zero,
        Sign::NonNegative => Sign::NonPositive,
        Sign::Positive => Sign::Negative,
        Sign::Unknown => Sign::Unknown,
    };
    value.interval = value.interval.map(|interval| Interval {
        lower: -interval.upper,
        upper: -interval.lower,
    });
    value
}

fn preserve_reduction_sign(sign: Sign) -> Sign {
    match sign
    {
        Sign::Positive | Sign::NonNegative | Sign::Zero => Sign::NonNegative,
        Sign::Negative | Sign::NonPositive => Sign::NonPositive,
        Sign::Unknown => Sign::Unknown,
    }
}

fn widen_sign(sign: Sign) -> Sign {
    match sign
    {
        Sign::Zero => Sign::Unknown,
        other => other,
    }
}

fn join(left: ValueFacts, right: ValueFacts) -> ValueFacts {
    ValueFacts {
        constant: (left.constant == right.constant)
            .then_some(left.constant)
            .flatten(),
        sign: join_sign(left.sign, right.sign),
        finiteness: join_finiteness(left.finiteness, right.finiteness),
        interval: match (left.interval, right.interval)
        {
            (Some(a), Some(b)) => Some(Interval {
                lower: a.lower.min(b.lower),
                upper: a.upper.max(b.upper),
            }),
            _ => None,
        },
    }
}

fn join_sign(left: Sign, right: Sign) -> Sign {
    if left == right
    {
        return left;
    }
    match (left, right)
    {
        (
            Sign::Positive | Sign::NonNegative | Sign::Zero,
            Sign::Positive | Sign::NonNegative | Sign::Zero,
        ) => Sign::NonNegative,
        (
            Sign::Negative | Sign::NonPositive | Sign::Zero,
            Sign::Negative | Sign::NonPositive | Sign::Zero,
        ) => Sign::NonPositive,
        _ => Sign::Unknown,
    }
}

fn join_finiteness(left: Finiteness, right: Finiteness) -> Finiteness {
    use Finiteness::{Finite, MayBeInfinite, MayBeNan, Unknown};
    match (left, right)
    {
        (Finite, Finite) => Finite,
        (Unknown, _) | (_, Unknown) => Unknown,
        (MayBeNan, _) | (_, MayBeNan) => MayBeNan,
        _ => MayBeInfinite,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::{Bin, Op, Ref, Section, Un, ValueType};

    #[test]
    fn constants_abs_and_exp_have_conservative_facts() {
        let program = ResearchProgram::expression(
            vec![],
            Section::new(vec![
                Op::Const(ScalarValue::F64(-2.0)),
                Op::Abs(Un::new(Ref::Local(0))),
                Op::Exp(Un::new(Ref::Local(1))),
            ]),
            vec![2],
        );
        let analysis = analyze_ranges(&program, VerificationLimits::default()).unwrap();
        assert_eq!(analysis.finalize[0].sign, Sign::Negative);
        assert_eq!(analysis.finalize[1].sign, Sign::NonNegative);
        assert_eq!(analysis.finalize[2].sign, Sign::NonNegative);
        assert_eq!(analysis.finalize[2].finiteness, Finiteness::MayBeInfinite);
    }

    #[test]
    fn bounded_recurrence_analysis_never_tracks_all_runtime_steps() {
        let scalar = ValueType::scalar(DType::F64);
        let program = ResearchProgram {
            semantics: NumericalSemantics::FiniteOnly,
            inputs: vec![],
            items: vec![scalar.clone()],
            state: vec![scalar],
            steps: 100,
            init: Section::new(vec![Op::Const(ScalarValue::F64(0.0))]),
            init_state: vec![0],
            step: Section::new(vec![Op::Add(Bin::new(Ref::StatePrev(0), Ref::Item(0)))]),
            next_state: vec![0],
            finalize: Section::new(vec![Op::Abs(Un::new(Ref::StateFinal(0)))]),
            outputs: vec![0],
        };
        let analysis = analyze_ranges(&program, VerificationLimits::default()).unwrap();
        assert!(analysis.recurrence_iterations <= MAX_RANGE_RECURRENCE_PASSES);
        assert_eq!(analysis.final_state.len(), 1);
    }

    #[test]
    fn boolean_results_are_known_finite() {
        let value = ValueType::scalar(DType::F64);
        let program = ResearchProgram::expression(
            vec![value],
            Section::new(vec![Op::Lt(Bin::new(Ref::Input(0), Ref::Input(0)))]),
            vec![0],
        );
        let analysis = analyze_ranges(&program, VerificationLimits::default()).unwrap();
        assert_eq!(analysis.finalize[0].finiteness, Finiteness::Finite);
    }
}
