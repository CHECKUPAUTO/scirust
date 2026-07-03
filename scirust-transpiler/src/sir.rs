//! Scientific IR (SIR): a small, typed intermediate representation.
//!
//! Every value carries a [`Ty`] (scalar `f64` or 1-D `f64` array). The SIR is
//! the only place numeric semantics is reasoned about; front-ends lower into
//! it and the Rust emitter lowers out of it. Keeping it typed is what lets the
//! emitter produce *compiling* Rust (typed function signatures, `&[f64]` vs
//! `f64`) — something a purely untyped AST cannot do.

/// The MVP type lattice: a scalar `f64` or a 1-D `Vec<f64>` / `&[f64]` array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Scalar,
    Array,
    /// Integer index / length (emitted as `usize`). Internal to loops/indexing.
    Int,
    /// Boolean (emitted as `bool`). Internal to conditions only.
    Bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SirModule {
    pub funcs: Vec<SirFunc>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SirFunc {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
    pub body: Vec<SirStmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SirStmt {
    /// First binding of a name: `let mut name: ty = value;`
    Let {
        name: String,
        ty: Ty,
        value: SirExpr,
    },
    /// Re-assignment of an already-bound name: `name = value;`
    Reassign {
        name: String,
        value: SirExpr,
    },
    /// `name[index] = value;`
    SetIndex {
        name: String,
        index: SirExpr,
        value: SirExpr,
    },
    /// `for var in start..end { body }` — deterministic ascending range.
    For {
        var: String,
        start: SirExpr,
        end: SirExpr,
        body: Vec<SirStmt>,
    },
    /// `if cond { then } else { els }` (else omitted when empty).
    If {
        cond: SirExpr,
        then: Vec<SirStmt>,
        els: Vec<SirStmt>,
    },
    Return(SirExpr),
}

/// A typed SIR expression. `ty()` reports the static type used by the emitter.
#[derive(Debug, Clone, PartialEq)]
pub enum SirExpr {
    ScalarLit(f64),
    IntLit(i64),
    Var {
        name: String,
        ty: Ty,
    },
    /// Scalar arithmetic (both operands scalar).
    ScalarBin {
        op: Op,
        l: Box<SirExpr>,
        r: Box<SirExpr>,
    },
    /// Integer arithmetic on `usize` (indices, lengths, ranges).
    IntBin {
        op: Op,
        l: Box<SirExpr>,
        r: Box<SirExpr>,
    },
    ScalarNeg(Box<SirExpr>),
    /// `f64::powf` / integer power folded to `powi` when exponent is an int lit.
    ScalarPow {
        base: Box<SirExpr>,
        exp: Box<SirExpr>,
    },
    /// `base[idx]` : Array indexed by Int -> Scalar.
    Index {
        base: Box<SirExpr>,
        idx: Box<SirExpr>,
    },
    /// Scalar math intrinsic: sqrt/exp/sin/cos/... on a scalar.
    ScalarUnaryFn {
        func: MathFn,
        arg: Box<SirExpr>,
    },

    // ---- array-producing / array-consuming intrinsics (routed to the prelude)
    /// Elementwise binary op between two arrays of equal length -> Array.
    EwBin {
        op: Op,
        l: Box<SirExpr>,
        r: Box<SirExpr>,
    },
    /// Broadcast a scalar against an array (scalar on the left) -> Array.
    ScalarBroadcast {
        op: Op,
        scalar: Box<SirExpr>,
        arr: Box<SirExpr>,
        /// true if the array is the left operand (`arr op scalar`), false for
        /// `scalar op arr`. Matters for non-commutative sub/div.
        arr_is_left: bool,
    },
    /// Elementwise math intrinsic over an array -> Array.
    ArrayUnaryFn {
        func: MathFn,
        arg: Box<SirExpr>,
    },
    /// `np.sum(a)` : Array -> Scalar, fixed ascending reduction order.
    Sum(Box<SirExpr>),
    /// `np.dot(a, b)` : (Array, Array) -> Scalar, fixed reduction order.
    Dot(Box<SirExpr>, Box<SirExpr>),
    /// `len(a)` / `a.shape[0]` : Array -> Int.
    Len(Box<SirExpr>),
    /// `np.zeros(n)` : Int -> Array.
    Zeros(Box<SirExpr>),
    /// `np.ones(n)` : Int -> Array.
    Ones(Box<SirExpr>),
    /// Scalar comparison `l <op> r` -> Bool (conditions only).
    Cmp {
        op: CmpOp,
        l: Box<SirExpr>,
        r: Box<SirExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpOp {
    pub fn rust_sym(self) -> &'static str {
        match self
        {
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFn {
    Sqrt,
    Exp,
    Sin,
    Cos,
    Abs,
    Tanh,
}

impl MathFn {
    pub fn rust_method(self) -> &'static str {
        match self
        {
            MathFn::Sqrt => "sqrt",
            MathFn::Exp => "exp",
            MathFn::Sin => "sin",
            MathFn::Cos => "cos",
            MathFn::Abs => "abs",
            MathFn::Tanh => "tanh",
        }
    }
}

impl SirExpr {
    /// Static type of the expression (used by the emitter and the checker).
    pub fn ty(&self) -> Ty {
        match self
        {
            SirExpr::ScalarLit(_) => Ty::Scalar,
            SirExpr::IntLit(_) => Ty::Int,
            SirExpr::Var { ty, .. } => *ty,
            SirExpr::ScalarBin { .. }
            | SirExpr::ScalarNeg(_)
            | SirExpr::ScalarPow { .. }
            | SirExpr::Index { .. }
            | SirExpr::ScalarUnaryFn { .. }
            | SirExpr::Sum(_)
            | SirExpr::Dot(_, _) => Ty::Scalar,
            SirExpr::IntBin { .. } | SirExpr::Len(_) => Ty::Int,
            SirExpr::EwBin { .. }
            | SirExpr::ScalarBroadcast { .. }
            | SirExpr::ArrayUnaryFn { .. }
            | SirExpr::Zeros(_)
            | SirExpr::Ones(_) => Ty::Array,
            SirExpr::Cmp { .. } => Ty::Bool,
        }
    }
}
