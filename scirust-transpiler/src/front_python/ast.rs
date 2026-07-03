//! Python (subset) AST produced by the front-end parser.
//!
//! This is deliberately small: the transpiler accepts a *contractual subset*
//! of Python/NumPy (statically analysable numeric code) and refuses — with a
//! diagnostic — anything outside it, rather than guessing.

/// A parsed module: a flat list of top-level `def`s.
#[derive(Debug, Clone, PartialEq)]
pub struct PyModule {
    pub funcs: Vec<PyFunc>,
}

/// A `def name(params) -> ret: body` function.
#[derive(Debug, Clone, PartialEq)]
pub struct PyFunc {
    pub name: String,
    pub params: Vec<PyParam>,
    pub ret_hint: Option<TypeHint>,
    pub body: Vec<PyStmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PyParam {
    pub name: String,
    pub hint: Option<TypeHint>,
}

/// A supported type annotation. `Array` covers `np.ndarray`, `ndarray`,
/// `"np.ndarray"` and list-of-float.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeHint {
    Float,
    Int,
    Array,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PyStmt {
    /// `x = expr`
    Assign { target: String, value: PyExpr },
    /// `a[i] = expr`
    AssignIndex {
        target: String,
        index: PyExpr,
        value: PyExpr,
    },
    /// `for var in range(start, end): body`  (start defaults to 0)
    For {
        var: String,
        start: PyExpr,
        end: PyExpr,
        body: Vec<PyStmt>,
    },
    /// `if cond: then [else: els]`  (`elif` desugars to a nested `If` in `els`).
    If {
        cond: PyExpr,
        then: Vec<PyStmt>,
        els: Vec<PyStmt>,
    },
    /// `while cond: body`
    While { cond: PyExpr, body: Vec<PyStmt> },
    /// `return expr`
    Return(Option<PyExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PyExpr {
    Int(i64),
    Float(f64),
    Name(String),
    /// Binary arithmetic: `+ - * / **`.
    Bin {
        op: BinOp,
        l: Box<PyExpr>,
        r: Box<PyExpr>,
    },
    /// Unary minus.
    Neg(Box<PyExpr>),
    /// A call to a (possibly dotted) function name, e.g. `np.sum`, `len`.
    Call {
        func: String,
        args: Vec<PyExpr>,
    },
    /// `base[index]`.
    Index {
        base: Box<PyExpr>,
        index: Box<PyExpr>,
    },
    /// A comparison `l <op> r` (used only in conditions; yields a boolean).
    Cmp {
        op: CmpOp,
        l: Box<PyExpr>,
        r: Box<PyExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
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
