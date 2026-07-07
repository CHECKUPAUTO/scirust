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
    /// A 2-D matrix, stored flat row-major (param emitted as `&[f64]`).
    /// Used to route `np.linalg.solve` to `scirust-solvers`.
    Matrix,
    /// A 1-D complex array (`Vec<scirust_signal::complex::Complex>`), produced
    /// by `np.fft.fft`. Consumed by `np.abs` (→ magnitude, real) or returned.
    ComplexArray,
    /// A produced 2-D matrix value (`scirust_solvers::Matrix`, carrying its own
    /// shape), e.g. the result of `np.linalg.inv`. Distinct from `Matrix`,
    /// which is a flat `&[f64]` *parameter*.
    MatrixVal,
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
    /// Hoisted declaration without initialiser: `let mut name: ty;` — Rust's
    /// definite-assignment analysis validates it is assigned before use. Used
    /// by the MATLAB front-end (output/locals assigned inside branches).
    Declare {
        name: String,
        ty: Ty,
    },
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
    /// `while cond { body }`.
    While {
        cond: SirExpr,
        body: Vec<SirStmt>,
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
    /// `np.linalg.solve(A, b)` : (Matrix, Array) -> Array, routed to the
    /// verified LU solver in `scirust-solvers` (not re-derived in std Rust).
    LinSolve {
        a: Box<SirExpr>,
        b: Box<SirExpr>,
    },
    /// `np.linalg.det(A)` : Matrix -> Scalar, routed to `scirust-solvers`
    /// (LU-based determinant).
    Det(Box<SirExpr>),
    /// `np.linalg.eigvalsh(A)` : symmetric Matrix -> Array (eigenvalues, sorted
    /// ascending), routed to `scirust-solvers::eigen_symmetric`.
    Eigvalsh(Box<SirExpr>),
    /// `A @ b` : (Matrix, Array) -> Array, matrix-vector product routed to
    /// `scirust-solvers::Matrix::matvec`.
    Matvec {
        a: Box<SirExpr>,
        b: Box<SirExpr>,
    },
    /// `np.linalg.inv(A)` : Matrix -> MatrixVal, routed to
    /// `scirust-solvers::Matrix::inverse`.
    Inv(Box<SirExpr>),
    /// `A @ B` : (matrix, matrix) -> MatrixVal, routed to
    /// `scirust-solvers::Matrix::matmul`.
    Matmul {
        a: Box<SirExpr>,
        b: Box<SirExpr>,
    },
    /// `A.T` : matrix -> MatrixVal (transpose).
    Transpose(Box<SirExpr>),
    /// `np.fft.fft(x)` : real Array -> ComplexArray (full spectrum), routed to
    /// the verified in-place FFT in `scirust-signal`.
    Fft(Box<SirExpr>),
    /// `np.fft.rfft(x)` : real Array -> ComplexArray (positive-frequency half
    /// spectrum, `N/2+1` bins), routed to `scirust-signal::fft::fft_real`.
    Rfft(Box<SirExpr>),
    /// `np.fft.ifft(c)` : ComplexArray -> ComplexArray (inverse DFT, `1/N`
    /// normalised), routed to `scirust-signal::fft::ifft`.
    Ifft(Box<SirExpr>),
    /// `np.abs(c)` where `c` is a ComplexArray -> real Array of magnitudes.
    ComplexAbs(Box<SirExpr>),
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
            | SirExpr::Det(_)
            | SirExpr::Dot(_, _) => Ty::Scalar,
            SirExpr::IntBin { .. } | SirExpr::Len(_) => Ty::Int,
            SirExpr::EwBin { .. }
            | SirExpr::ScalarBroadcast { .. }
            | SirExpr::ArrayUnaryFn { .. }
            | SirExpr::Zeros(_)
            | SirExpr::Ones(_)
            | SirExpr::LinSolve { .. }
            | SirExpr::Eigvalsh(_)
            | SirExpr::Matvec { .. }
            | SirExpr::ComplexAbs(_) => Ty::Array,
            SirExpr::Fft(_) | SirExpr::Rfft(_) | SirExpr::Ifft(_) => Ty::ComplexArray,
            SirExpr::Inv(_) | SirExpr::Matmul { .. } | SirExpr::Transpose(_) => Ty::MatrixVal,
            SirExpr::Cmp { .. } => Ty::Bool,
        }
    }
}

/// Which external `scirust-*` crates the emitted code for `m` depends on
/// (empty for std-only modules). Drives the oracle's compile mode.
pub fn required_crates(m: &SirModule) -> Vec<&'static str> {
    let mut solvers = false;
    let mut signal = false;
    for f in &m.funcs
    {
        for s in &f.body
        {
            scan_stmt(s, &mut solvers, &mut signal);
        }
    }
    let mut out = Vec::new();
    if signal
    {
        out.push("scirust-signal");
    }
    if solvers
    {
        out.push("scirust-solvers");
    }
    out
}

fn scan_stmt(s: &SirStmt, solvers: &mut bool, signal: &mut bool) {
    match s
    {
        SirStmt::Declare { .. } =>
        {},
        SirStmt::Let { value, .. } | SirStmt::Reassign { value, .. } | SirStmt::Return(value) =>
        {
            scan_expr(value, solvers, signal)
        },
        SirStmt::SetIndex { index, value, .. } =>
        {
            scan_expr(index, solvers, signal);
            scan_expr(value, solvers, signal);
        },
        SirStmt::For {
            start, end, body, ..
        } =>
        {
            scan_expr(start, solvers, signal);
            scan_expr(end, solvers, signal);
            body.iter().for_each(|s| scan_stmt(s, solvers, signal));
        },
        SirStmt::If { cond, then, els } =>
        {
            scan_expr(cond, solvers, signal);
            then.iter().for_each(|s| scan_stmt(s, solvers, signal));
            els.iter().for_each(|s| scan_stmt(s, solvers, signal));
        },
        SirStmt::While { cond, body } =>
        {
            scan_expr(cond, solvers, signal);
            body.iter().for_each(|s| scan_stmt(s, solvers, signal));
        },
    }
}

fn scan_expr(e: &SirExpr, solvers: &mut bool, signal: &mut bool) {
    match e
    {
        SirExpr::LinSolve { a, b } | SirExpr::Matvec { a, b } | SirExpr::Matmul { a, b } =>
        {
            *solvers = true;
            scan_expr(a, solvers, signal);
            scan_expr(b, solvers, signal);
        },
        SirExpr::Det(x) | SirExpr::Eigvalsh(x) | SirExpr::Inv(x) | SirExpr::Transpose(x) =>
        {
            *solvers = true;
            scan_expr(x, solvers, signal);
        },
        SirExpr::Fft(x) | SirExpr::Rfft(x) | SirExpr::Ifft(x) | SirExpr::ComplexAbs(x) =>
        {
            *signal = true;
            scan_expr(x, solvers, signal);
        },
        SirExpr::ScalarBin { l, r, .. }
        | SirExpr::IntBin { l, r, .. }
        | SirExpr::EwBin { l, r, .. }
        | SirExpr::Cmp { l, r, .. }
        | SirExpr::Dot(l, r) =>
        {
            scan_expr(l, solvers, signal);
            scan_expr(r, solvers, signal);
        },
        SirExpr::ScalarNeg(x)
        | SirExpr::ScalarUnaryFn { arg: x, .. }
        | SirExpr::ArrayUnaryFn { arg: x, .. }
        | SirExpr::Sum(x)
        | SirExpr::Len(x)
        | SirExpr::Zeros(x)
        | SirExpr::Ones(x) => scan_expr(x, solvers, signal),
        SirExpr::ScalarPow { base, exp } =>
        {
            scan_expr(base, solvers, signal);
            scan_expr(exp, solvers, signal);
        },
        SirExpr::Index { base, idx } =>
        {
            scan_expr(base, solvers, signal);
            scan_expr(idx, solvers, signal);
        },
        SirExpr::ScalarBroadcast { scalar, arr, .. } =>
        {
            scan_expr(scalar, solvers, signal);
            scan_expr(arr, solvers, signal);
        },
        SirExpr::ScalarLit(_) | SirExpr::IntLit(_) | SirExpr::Var { .. } =>
        {},
    }
}
