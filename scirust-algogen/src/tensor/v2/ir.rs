//! Semantic IR for scientific algorithm discovery (V2).
//!
//! A [`ResearchProgram`] is a typed, straight-line, SSA-flavoured program in
//! three sections — `init`, `step`, `finalize` — where the `step` section is a
//! statically counted scan over declared state components and incoming items.
//! There is no other control flow: no loops, no recursion, no dynamic
//! dispatch. This keeps interpretation, verification, cost accounting,
//! mutation and canonicalization fully deterministic.
//!
//! # Sections and references
//!
//! | Reference | Legal in | Meaning |
//! |---|---|---|
//! | [`Ref::Input`] | init, finalize | program input |
//! | [`Ref::Local`] | all | an earlier node of the same section |
//! | [`Ref::Item`] | step only | one incoming value of the current step |
//! | [`Ref::StatePrev`] | step only | previous value of a state slot |
//! | [`Ref::StateFinal`] | finalize only | final value of a state slot |
//!
//! Constants are ordinary nodes (`Op::Const`) so common-subexpression
//! elimination and folding treat them uniformly.

use serde::{Deserialize, Serialize};

use super::semantics::NumericalSemantics;
use super::types::{ScalarValue, ValueType};

/// Version of the V2 IR. Bump on any semantic change to the operator set or
/// program structure; archives record it so identity can never silently drift.
pub const IR_VERSION: u32 = 2;

/// Index of a value inside its section (`0..section.ops.len()`).
pub type ValueId = usize;

/// A reference to an already-defined value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ref {
    /// Program input (init/finalize sections).
    Input(usize),
    /// Earlier node of the same section.
    Local(ValueId),
    /// Per-step incoming value (step section only).
    Item(usize),
    /// Previous value of state slot (step section only).
    StatePrev(usize),
    /// Final value of state slot (finalize section only).
    StateFinal(usize),
}

/// One source operand of a binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bin {
    pub lhs: Ref,
    pub rhs: Ref,
}

impl Bin {
    pub fn new(lhs: Ref, rhs: Ref) -> Self {
        Self { lhs, rhs }
    }
}

/// The single source operand of a unary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Un {
    pub src: Ref,
}

impl Un {
    pub fn new(src: Ref) -> Self {
        Self { src }
    }
}

/// Three source operands (fused multiply-add, clamp, select).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ter {
    pub a: Ref,
    pub b: Ref,
    pub c: Ref,
}

impl Ter {
    pub fn new(a: Ref, b: Ref, c: Ref) -> Self {
        Self { a, b, c }
    }
}

/// Reduction descriptor: source plus explicit axis.
///
/// `axis == None` reduces every axis to a rank-0 scalar. `axis == Some(axis)`
/// removes exactly that axis (keep-dim = false).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Reduce {
    pub src: Ref,
    pub axis: Option<usize>,
}

/// Reshape to an explicit target shape (element count must be preserved).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeTo {
    pub src: Ref,
    pub shape: Vec<usize>,
}

/// Axis-indexed unary shape operation (squeeze / unsqueeze / narrow axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AxisOp {
    pub src: Ref,
    pub axis: usize,
}

/// Transposition by an explicit permutation of the source axes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permute {
    pub src: Ref,
    pub perm: Vec<usize>,
}

/// Static slice: keep `len` elements starting at `start` along `axis`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Narrow {
    pub src: Ref,
    pub axis: usize,
    pub start: usize,
    pub len: usize,
}

/// One IR operation. Every variant's exact semantics are specified in
/// `docs/SCIRUST_ALGOGEN_IR_V2_NUMERICAL_SEMANTICS.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// Typed compile-time constant. Non-finite float constants are rejected by
    /// the verifier.
    Const(ScalarValue),

    // ---- broadcast arithmetic (float dtypes) --------------------------------
    Add(Bin),
    Sub(Bin),
    Mul(Bin),
    Div(Bin),
    /// Fused multiply-add: `a * b + c` as a single rounding at the operand
    /// dtype (the interpreter uses the native `mul_add`).
    MulAdd(Ter),
    /// Elementwise power: `lhs ^ rhs`. Defined for finite results only under
    /// the default policy; `powi`-style integer exponents are expressed by
    /// repeated multiplication when exactness matters.
    Pow(Bin),

    // ---- unary float --------------------------------------------------------
    Neg(Un),
    Abs(Un),
    Exp(Un),
    Exp2(Un),
    Expm1(Un),
    Log(Un),
    Log2(Un),
    Log1p(Un),
    Sqrt(Un),
    /// Reciprocal square root: `1 / sqrt(x)`.
    Rsqrt(Un),
    Sin(Un),
    Cos(Un),
    Tanh(Un),

    // ---- extrema / selection ------------------------------------------------
    Min(Bin),
    Max(Bin),
    Clamp(Ter),
    /// Masked selection: `b` where Boolean mask `a` is true, otherwise `c`.
    Select(Ter),

    // ---- comparisons (float operands, Boolean result) -----------------------
    Eq(Bin),
    Ne(Bin),
    Lt(Bin),
    Le(Bin),
    Gt(Bin),
    Ge(Bin),

    // ---- Boolean logic -------------------------------------------------------
    And(Bin),
    Or(Bin),
    Not(Un),

    // ---- reductions -----------------------------------------------------------
    ReduceSum(Reduce),
    ReduceProd(Reduce),
    ReduceMax(Reduce),
    ReduceMin(Reduce),
    ReduceMean(Reduce),

    // ---- linear algebra ---------------------------------------------------------
    /// Inner product of two equal-length rank-1 vectors into a scalar.
    Dot(Bin),
    /// `[m, n] x [n] -> [m]`.
    MatVec(Bin),
    /// `[n] x [n, k] -> [k]`.
    VecMat(Bin),
    /// `[m, k] x [k, n] -> [m, n]`.
    MatMul(Bin),
    /// `[b, m, k] x [b, k, n] -> [b, m, n]` over a shared leading batch axis.
    BatchedMatMul(Bin),
    /// Outer product `[m] x [n] -> [m, n]`.
    Outer(Bin),

    // ---- shape algebra ------------------------------------------------------------
    Reshape(ShapeTo),
    Squeeze(AxisOp),
    Unsqueeze(AxisOp),
    Transpose(Permute),
    BroadcastTo(ShapeTo),
    Concat {
        lhs: Ref,
        rhs: Ref,
        axis: usize,
    },
    Narrow(Narrow),
}

impl Op {
    /// Apply `visit` to every reference read by this operation.
    ///
    /// References are passed by value ([`Ref`] is `Copy`) so visitors cannot
    /// retain borrows into the operation.
    pub fn for_each_ref(&self, mut visit: impl FnMut(Ref)) {
        match self
        {
            Self::Const(_) =>
            {},
            Self::Add(b)
            | Self::Sub(b)
            | Self::Mul(b)
            | Self::Div(b)
            | Self::Pow(b)
            | Self::Min(b)
            | Self::Max(b)
            | Self::Eq(b)
            | Self::Ne(b)
            | Self::Lt(b)
            | Self::Le(b)
            | Self::Gt(b)
            | Self::Ge(b)
            | Self::And(b)
            | Self::Or(b)
            | Self::Dot(b)
            | Self::MatVec(b)
            | Self::VecMat(b)
            | Self::MatMul(b)
            | Self::BatchedMatMul(b)
            | Self::Outer(b) =>
            {
                visit(b.lhs);
                visit(b.rhs);
            },
            Self::MulAdd(t) | Self::Clamp(t) | Self::Select(t) =>
            {
                visit(t.a);
                visit(t.b);
                visit(t.c);
            },
            Self::Neg(u)
            | Self::Abs(u)
            | Self::Exp(u)
            | Self::Exp2(u)
            | Self::Expm1(u)
            | Self::Log(u)
            | Self::Log2(u)
            | Self::Log1p(u)
            | Self::Sqrt(u)
            | Self::Rsqrt(u)
            | Self::Sin(u)
            | Self::Cos(u)
            | Self::Tanh(u)
            | Self::Not(u) => visit(u.src),
            Self::ReduceSum(r)
            | Self::ReduceProd(r)
            | Self::ReduceMax(r)
            | Self::ReduceMin(r)
            | Self::ReduceMean(r) => visit(r.src),
            Self::Reshape(s) | Self::BroadcastTo(s) => visit(s.src),
            Self::Squeeze(a) | Self::Unsqueeze(a) => visit(a.src),
            Self::Transpose(p) => visit(p.src),
            Self::Concat { lhs, rhs, .. } =>
            {
                visit(*lhs);
                visit(*rhs);
            },
            Self::Narrow(n) => visit(n.src),
        }
    }

    /// Rewrite every reference through `map`.
    pub fn map_refs(&mut self, mut map: impl FnMut(Ref) -> Ref) {
        match self
        {
            Self::Const(_) =>
            {},
            Self::Add(b)
            | Self::Sub(b)
            | Self::Mul(b)
            | Self::Div(b)
            | Self::Pow(b)
            | Self::Min(b)
            | Self::Max(b)
            | Self::Eq(b)
            | Self::Ne(b)
            | Self::Lt(b)
            | Self::Le(b)
            | Self::Gt(b)
            | Self::Ge(b)
            | Self::And(b)
            | Self::Or(b)
            | Self::Dot(b)
            | Self::MatVec(b)
            | Self::VecMat(b)
            | Self::MatMul(b)
            | Self::BatchedMatMul(b)
            | Self::Outer(b) =>
            {
                b.lhs = map(b.lhs);
                b.rhs = map(b.rhs);
            },
            Self::MulAdd(t) | Self::Clamp(t) | Self::Select(t) =>
            {
                t.a = map(t.a);
                t.b = map(t.b);
                t.c = map(t.c);
            },
            Self::Neg(u)
            | Self::Abs(u)
            | Self::Exp(u)
            | Self::Exp2(u)
            | Self::Expm1(u)
            | Self::Log(u)
            | Self::Log2(u)
            | Self::Log1p(u)
            | Self::Sqrt(u)
            | Self::Rsqrt(u)
            | Self::Sin(u)
            | Self::Cos(u)
            | Self::Tanh(u)
            | Self::Not(u) => u.src = map(u.src),
            Self::ReduceSum(r)
            | Self::ReduceProd(r)
            | Self::ReduceMax(r)
            | Self::ReduceMin(r)
            | Self::ReduceMean(r) => r.src = map(r.src),
            Self::Reshape(s) | Self::BroadcastTo(s) => s.src = map(s.src),
            Self::Squeeze(a) | Self::Unsqueeze(a) => a.src = map(a.src),
            Self::Transpose(p) => p.src = map(p.src),
            Self::Concat { lhs, rhs, .. } =>
            {
                *lhs = map(*lhs);
                *rhs = map(*rhs);
            },
            Self::Narrow(n) => n.src = map(n.src),
        }
    }

    /// Stable opcode tag for canonical encodings. Tags are append-only:
    /// existing tags never change meaning within an IR version.
    pub fn tag(&self) -> u16 {
        match self
        {
            Self::Const(_) => 0,
            Self::Add(_) => 1,
            Self::Sub(_) => 2,
            Self::Mul(_) => 3,
            Self::Div(_) => 4,
            Self::MulAdd(_) => 5,
            Self::Pow(_) => 6,
            Self::Neg(_) => 7,
            Self::Abs(_) => 8,
            Self::Exp(_) => 9,
            Self::Exp2(_) => 10,
            Self::Expm1(_) => 11,
            Self::Log(_) => 12,
            Self::Log2(_) => 13,
            Self::Log1p(_) => 14,
            Self::Sqrt(_) => 15,
            Self::Rsqrt(_) => 16,
            Self::Sin(_) => 17,
            Self::Cos(_) => 18,
            Self::Tanh(_) => 19,
            Self::Min(_) => 20,
            Self::Max(_) => 21,
            Self::Clamp(_) => 22,
            Self::Select(_) => 23,
            Self::Eq(_) => 24,
            Self::Ne(_) => 25,
            Self::Lt(_) => 26,
            Self::Le(_) => 27,
            Self::Gt(_) => 28,
            Self::Ge(_) => 29,
            Self::And(_) => 30,
            Self::Or(_) => 31,
            Self::Not(_) => 32,
            Self::ReduceSum(_) => 33,
            Self::ReduceProd(_) => 34,
            Self::ReduceMax(_) => 35,
            Self::ReduceMin(_) => 36,
            Self::ReduceMean(_) => 37,
            Self::Dot(_) => 38,
            Self::MatVec(_) => 39,
            Self::VecMat(_) => 40,
            Self::MatMul(_) => 41,
            Self::BatchedMatMul(_) => 42,
            Self::Outer(_) => 43,
            Self::Reshape(_) => 44,
            Self::Squeeze(_) => 45,
            Self::Unsqueeze(_) => 46,
            Self::Transpose(_) => 47,
            Self::BroadcastTo(_) => 48,
            Self::Concat { .. } => 49,
            Self::Narrow(_) => 50,
        }
    }

    /// Stable operator name (diagnostics and reports only; never identity).
    pub fn name(&self) -> &'static str {
        match self
        {
            Self::Const(_) => "const",
            Self::Add(_) => "add",
            Self::Sub(_) => "sub",
            Self::Mul(_) => "mul",
            Self::Div(_) => "div",
            Self::MulAdd(_) => "mul_add",
            Self::Pow(_) => "pow",
            Self::Neg(_) => "neg",
            Self::Abs(_) => "abs",
            Self::Exp(_) => "exp",
            Self::Exp2(_) => "exp2",
            Self::Expm1(_) => "expm1",
            Self::Log(_) => "log",
            Self::Log2(_) => "log2",
            Self::Log1p(_) => "log1p",
            Self::Sqrt(_) => "sqrt",
            Self::Rsqrt(_) => "rsqrt",
            Self::Sin(_) => "sin",
            Self::Cos(_) => "cos",
            Self::Tanh(_) => "tanh",
            Self::Min(_) => "min",
            Self::Max(_) => "max",
            Self::Clamp(_) => "clamp",
            Self::Select(_) => "select",
            Self::Eq(_) => "eq",
            Self::Ne(_) => "ne",
            Self::Lt(_) => "lt",
            Self::Le(_) => "le",
            Self::Gt(_) => "gt",
            Self::Ge(_) => "ge",
            Self::And(_) => "and",
            Self::Or(_) => "or",
            Self::Not(_) => "not",
            Self::ReduceSum(_) => "reduce_sum",
            Self::ReduceProd(_) => "reduce_prod",
            Self::ReduceMax(_) => "reduce_max",
            Self::ReduceMin(_) => "reduce_min",
            Self::ReduceMean(_) => "reduce_mean",
            Self::Dot(_) => "dot",
            Self::MatVec(_) => "mat_vec",
            Self::VecMat(_) => "vec_mat",
            Self::MatMul(_) => "mat_mul",
            Self::BatchedMatMul(_) => "batched_mat_mul",
            Self::Outer(_) => "outer",
            Self::Reshape(_) => "reshape",
            Self::Squeeze(_) => "squeeze",
            Self::Unsqueeze(_) => "unsqueeze",
            Self::Transpose(_) => "transpose",
            Self::BroadcastTo(_) => "broadcast_to",
            Self::Concat { .. } => "concat",
            Self::Narrow(_) => "narrow",
        }
    }
}

/// An ordered list of operations defining consecutive values.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Section {
    pub ops: Vec<Op>,
}

impl Section {
    pub fn new(ops: Vec<Op>) -> Self {
        Self { ops }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// A complete scientific algorithm candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchProgram {
    /// Equivalence regime governing canonicalization and rewrite validity.
    /// It is part of canonical identity.
    pub semantics: NumericalSemantics,
    /// Types of outer program inputs.
    pub inputs: Vec<ValueType>,
    /// Types of per-step incoming values (stream signature). May be empty.
    pub items: Vec<ValueType>,
    /// Declared recurrence state component types. May be empty.
    pub state: Vec<ValueType>,
    /// Static trip count of the scan. Must be `0` iff `state` is empty.
    pub steps: u32,

    /// Computes the initial state values from the inputs.
    pub init: Section,
    /// Value produced by `init` for each state slot (length = `state.len()`).
    pub init_state: Vec<ValueId>,

    /// Executed exactly `steps` times; computes next state values from the
    /// previous state and the current step's items.
    pub step: Section,
    /// Next-state value for each slot (length = `state.len()`).
    pub next_state: Vec<ValueId>,

    /// Computes observable outputs from the final state (and the inputs).
    pub finalize: Section,
    /// Observable outputs (non-empty, unique).
    pub outputs: Vec<ValueId>,
}

impl ResearchProgram {
    /// A straight-line program: no recurrence, computation lives in the
    /// finalize section reading directly from the inputs.
    pub fn expression(inputs: Vec<ValueType>, finalize: Section, outputs: Vec<ValueId>) -> Self {
        Self {
            semantics: NumericalSemantics::StrictIeee,
            inputs,
            items: Vec::new(),
            state: Vec::new(),
            steps: 0,
            init: Section::default(),
            init_state: Vec::new(),
            step: Section::default(),
            next_state: Vec::new(),
            finalize,
            outputs,
        }
    }

    /// Change the declared equivalence regime explicitly.
    #[must_use]
    pub fn with_semantics(mut self, semantics: NumericalSemantics) -> Self {
        self.semantics = semantics;
        self
    }

    /// Total number of defined values across all sections.
    pub fn node_count(&self) -> usize {
        self.init.len() + self.step.len() + self.finalize.len()
    }

    /// Number of elements held by one instance of the recurrence state.
    pub fn state_elements(&self) -> u64 {
        self.state.iter().fold(0u64, |total, value_type| {
            total.saturating_add(value_type.elements())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::v2::types::{DType, ScalarValue};

    #[test]
    fn op_tags_are_unique_and_stable() {
        let ops = vec![
            Op::Const(ScalarValue::F32(1.0)),
            Op::Add(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Sub(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Mul(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Div(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::MulAdd(Ter::new(Ref::Local(0), Ref::Local(0), Ref::Local(0))),
            Op::Pow(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Neg(Un::new(Ref::Local(0))),
            Op::Abs(Un::new(Ref::Local(0))),
            Op::Exp(Un::new(Ref::Local(0))),
            Op::Exp2(Un::new(Ref::Local(0))),
            Op::Expm1(Un::new(Ref::Local(0))),
            Op::Log(Un::new(Ref::Local(0))),
            Op::Log2(Un::new(Ref::Local(0))),
            Op::Log1p(Un::new(Ref::Local(0))),
            Op::Sqrt(Un::new(Ref::Local(0))),
            Op::Rsqrt(Un::new(Ref::Local(0))),
            Op::Sin(Un::new(Ref::Local(0))),
            Op::Cos(Un::new(Ref::Local(0))),
            Op::Tanh(Un::new(Ref::Local(0))),
            Op::Min(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Max(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Clamp(Ter::new(Ref::Local(0), Ref::Local(0), Ref::Local(0))),
            Op::Select(Ter::new(Ref::Local(0), Ref::Local(0), Ref::Local(0))),
            Op::Eq(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Ne(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Lt(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Le(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Gt(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Ge(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::And(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Or(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Not(Un::new(Ref::Local(0))),
            Op::ReduceSum(Reduce {
                src: Ref::Local(0),
                axis: None,
            }),
            Op::ReduceProd(Reduce {
                src: Ref::Local(0),
                axis: None,
            }),
            Op::ReduceMax(Reduce {
                src: Ref::Local(0),
                axis: None,
            }),
            Op::ReduceMin(Reduce {
                src: Ref::Local(0),
                axis: None,
            }),
            Op::ReduceMean(Reduce {
                src: Ref::Local(0),
                axis: None,
            }),
            Op::Dot(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::MatVec(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::VecMat(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::MatMul(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::BatchedMatMul(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Outer(Bin::new(Ref::Local(0), Ref::Local(0))),
            Op::Reshape(ShapeTo {
                src: Ref::Local(0),
                shape: vec![3],
            }),
            Op::Squeeze(AxisOp {
                src: Ref::Local(0),
                axis: 0,
            }),
            Op::Unsqueeze(AxisOp {
                src: Ref::Local(0),
                axis: 0,
            }),
            Op::Transpose(Permute {
                src: Ref::Local(0),
                perm: vec![1, 0],
            }),
            Op::BroadcastTo(ShapeTo {
                src: Ref::Local(0),
                shape: vec![2, 3],
            }),
            Op::Concat {
                lhs: Ref::Local(0),
                rhs: Ref::Local(0),
                axis: 0,
            },
            Op::Narrow(Narrow {
                src: Ref::Local(0),
                axis: 0,
                start: 0,
                len: 1,
            }),
        ];
        let mut tags = ops.iter().map(Op::tag).collect::<Vec<_>>();
        let unique_count = tags.len();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), unique_count, "op tags must be unique");
    }

    #[test]
    fn for_each_ref_covers_every_operand_exactly() {
        let add = Op::Add(Bin::new(Ref::Input(1), Ref::Local(7)));
        let mut seen = Vec::new();
        add.for_each_ref(|reference| seen.push(reference));
        assert_eq!(seen, vec![Ref::Input(1), Ref::Local(7)]);

        let select = Op::Select(Ter::new(Ref::Local(0), Ref::Item(1), Ref::StatePrev(2)));
        let mut seen = Vec::new();
        select.for_each_ref(|reference| seen.push(reference));
        assert_eq!(seen, vec![Ref::Local(0), Ref::Item(1), Ref::StatePrev(2)]);
    }

    #[test]
    fn map_refs_rewrites_every_operand() {
        let mut concat = Op::Concat {
            lhs: Ref::Local(0),
            rhs: Ref::Local(1),
            axis: 0,
        };
        concat.map_refs(|reference| match reference
        {
            Ref::Local(id) => Ref::Local(id + 10),
            other => other,
        });
        assert_eq!(
            concat,
            Op::Concat {
                lhs: Ref::Local(10),
                rhs: Ref::Local(11),
                axis: 0
            }
        );
    }

    #[test]
    fn expression_programs_have_no_recurrence() {
        let program = ResearchProgram::expression(
            vec![ValueType::scalar(DType::F64)],
            Section::new(vec![Op::Abs(Un::new(Ref::Input(0)))]),
            vec![0],
        );
        assert_eq!(program.steps, 0);
        assert!(program.state.is_empty());
        assert!(program.init.is_empty());
        assert!(program.step.is_empty());
        assert_eq!(program.node_count(), 1);
        assert_eq!(program.state_elements(), 0);
    }

    #[test]
    fn programs_round_trip_through_serde() {
        let program = ResearchProgram::expression(
            vec![ValueType::new(DType::F32, vec![2, 2])],
            Section::new(vec![
                Op::Const(ScalarValue::F32(-1.5)),
                Op::Add(Bin::new(Ref::Input(0), Ref::Local(0))),
            ]),
            vec![1],
        );
        let json = serde_json::to_string(&program).unwrap();
        let decoded: ResearchProgram = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, program);
    }
}
