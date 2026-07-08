//! MATLAB (subset) AST produced by the front-end parser.
//!
//! A contractual subset of MATLAB/Octave: `function out = f(args) … end` and
//! `function [o1, o2] = f(args) … end`, scalar/vector arithmetic (with
//! element-wise `.*` `./` `.^`), 1-based indexing, `if`/`for`/`while`, and a
//! small intrinsic set. Anything else is refused with a diagnostic.

#[derive(Debug, Clone, PartialEq)]
pub struct MModule {
    pub funcs: Vec<MFunc>,
}

/// `function out = name(params) body end` or
/// `function [o1, o2, …] = name(params) body end`. `outs` holds the output
/// variable(s) in declaration order (length 1 for a single-output function).
#[derive(Debug, Clone, PartialEq)]
pub struct MFunc {
    pub name: String,
    pub outs: Vec<String>,
    pub params: Vec<String>,
    pub body: Vec<MStmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MStmt {
    /// `x = expr`
    Assign { target: String, value: MExpr },
    /// `a(i) = expr`  (1-based index)
    AssignIndex {
        target: String,
        index: MExpr,
        value: MExpr,
    },
    /// `for var = lo:hi body end`  (inclusive range)
    For {
        var: String,
        lo: MExpr,
        hi: MExpr,
        body: Vec<MStmt>,
    },
    /// `if cond body [else els] end`  (`elseif` desugars to a nested `If`)
    If {
        cond: MExpr,
        then: Vec<MStmt>,
        els: Vec<MStmt>,
    },
    /// `while cond body end`
    While { cond: MExpr, body: Vec<MStmt> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MExpr {
    Num(f64),
    Ident(String),
    Bin {
        op: MBinOp,
        l: Box<MExpr>,
        r: Box<MExpr>,
    },
    Neg(Box<MExpr>),
    /// A call to an intrinsic (`sqrt`, `sum`, `zeros`, `length`, …).
    Call {
        func: String,
        args: Vec<MExpr>,
    },
    /// `base(index)` — 1-based indexing of a variable.
    Index {
        base: String,
        index: Box<MExpr>,
    },
    /// A comparison (conditions only).
    Cmp {
        op: MCmpOp,
        l: Box<MExpr>,
        r: Box<MExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// Element-wise `.*`
    EMul,
    /// Element-wise `./`
    EDiv,
    /// Element-wise `.^`
    EPow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MCmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}
