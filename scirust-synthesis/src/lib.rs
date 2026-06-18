//! scirust-synthesis — Enhanced Program Synthesis
//!
//! Multi-variable sketch-based synthesis with rich expression grammar,
//! bottom-up enumeration, top-down synthesis, genetic programming,
//! stochastic beam search, inductive bias, and expression rewriting.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Comparison operators for conditional expressions
// ============================================================================

/// Comparison operator used in [`SExpr::Cmp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    #[serde(rename = "eq")]
    Equal,
}

impl CmpOp {
    fn apply(&self, a: f64, b: f64) -> f64 {
        let ok = match self
        {
            CmpOp::Lt => a < b,
            CmpOp::Gt => a > b,
            CmpOp::Le => a <= b,
            CmpOp::Ge => a >= b,
            CmpOp::Equal => (a - b).abs() < 1e-12,
        };
        if ok { 1.0 } else { 0.0 }
    }

    fn symbol(&self) -> &str {
        match self
        {
            CmpOp::Lt => "<",
            CmpOp::Gt => ">",
            CmpOp::Le => "<=",
            CmpOp::Ge => ">=",
            CmpOp::Equal => "==",
        }
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

// ============================================================================
// SExpr — Rich expression grammar
// ============================================================================

/// A rich expression tree supporting 30+ operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SExpr {
    // --- Terminals ---
    Var(String),
    Const(f64),
    /// Numbered hole for sketch templates.
    Hole(usize),

    // --- Unary ---
    Neg(Box<SExpr>),
    Sin(Box<SExpr>),
    Cos(Box<SExpr>),
    Tan(Box<SExpr>),
    Exp(Box<SExpr>),
    Ln(Box<SExpr>),
    Sqrt(Box<SExpr>),
    Abs(Box<SExpr>),
    Sign(Box<SExpr>),
    Square(Box<SExpr>),
    Cube(Box<SExpr>),
    SinPi(Box<SExpr>),
    CosPi(Box<SExpr>),

    // --- Binary ---
    Add(Box<SExpr>, Box<SExpr>),
    Sub(Box<SExpr>, Box<SExpr>),
    Mul(Box<SExpr>, Box<SExpr>),
    Div(Box<SExpr>, Box<SExpr>),
    Pow(Box<SExpr>, Box<SExpr>),
    Max(Box<SExpr>, Box<SExpr>),
    Min(Box<SExpr>, Box<SExpr>),
    Hypot(Box<SExpr>, Box<SExpr>),
    Atan2(Box<SExpr>, Box<SExpr>),

    // --- N-ary ---
    Sum(Vec<SExpr>),
    Prod(Vec<SExpr>),

    // --- Conditional ---
    Cmp {
        op: CmpOp,
        lhs: Box<SExpr>,
        rhs: Box<SExpr>,
    },
    /// IfElse(condition, then, else) -- if cond != 0 then then else else
    IfElse(Box<SExpr>, Box<SExpr>, Box<SExpr>),
}

// --- Display ---

impl fmt::Display for SExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            SExpr::Var(v) => write!(f, "{v}"),
            SExpr::Const(c) =>
            {
                if *c == std::f64::consts::PI
                {
                    write!(f, "pi")
                }
                else if *c == std::f64::consts::E
                {
                    write!(f, "e")
                }
                else if c.fract() == 0.0
                {
                    write!(f, "{}", *c as i64)
                }
                else
                {
                    write!(f, "{c}")
                }
            },
            SExpr::Hole(i) => write!(f, "???{i}"),
            SExpr::Neg(a) => write!(f, "(-{a})"),
            SExpr::Sin(a) => write!(f, "sin({a})"),
            SExpr::Cos(a) => write!(f, "cos({a})"),
            SExpr::Tan(a) => write!(f, "tan({a})"),
            SExpr::Exp(a) => write!(f, "exp({a})"),
            SExpr::Ln(a) => write!(f, "ln({a})"),
            SExpr::Sqrt(a) => write!(f, "sqrt({a})"),
            SExpr::Abs(a) => write!(f, "abs({a})"),
            SExpr::Sign(a) => write!(f, "sign({a})"),
            SExpr::Square(a) => write!(f, "({a}^2)"),
            SExpr::Cube(a) => write!(f, "({a}^3)"),
            SExpr::SinPi(a) => write!(f, "sinPI({a})"),
            SExpr::CosPi(a) => write!(f, "cosPI({a})"),
            SExpr::Add(a, b) => write!(f, "({a} + {b})"),
            SExpr::Sub(a, b) => write!(f, "({a} - {b})"),
            SExpr::Mul(a, b) => write!(f, "({a} * {b})"),
            SExpr::Div(a, b) => write!(f, "({a} / {b})"),
            SExpr::Pow(a, b) => write!(f, "({a}^{b})"),
            SExpr::Max(a, b) => write!(f, "max({a}, {b})"),
            SExpr::Min(a, b) => write!(f, "min({a}, {b})"),
            SExpr::Hypot(a, b) => write!(f, "hypot({a}, {b})"),
            SExpr::Atan2(a, b) => write!(f, "atan2({a}, {b})"),
            SExpr::Sum(vs) =>
            {
                write!(f, "sum[")?;
                for (i, v) in vs.iter().enumerate()
                {
                    if i > 0
                    {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            },
            SExpr::Prod(vs) =>
            {
                write!(f, "prod[")?;
                for (i, v) in vs.iter().enumerate()
                {
                    if i > 0
                    {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            },
            SExpr::Cmp { op, lhs, rhs } => write!(f, "({lhs} {op} {rhs})"),
            SExpr::IfElse(c, t, e) => write!(f, "if({c}) {{{t}}} else {{{e}}}"),
        }
    }
}

// --- PartialEq / Eq / Hash ---

impl PartialEq for SExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other)
        {
            (SExpr::Var(a), SExpr::Var(b)) => a == b,
            (SExpr::Const(a), SExpr::Const(b)) => a.to_bits() == b.to_bits(),
            (SExpr::Hole(a), SExpr::Hole(b)) => a == b,
            (SExpr::Neg(a), SExpr::Neg(b)) => a == b,
            (SExpr::Sin(a), SExpr::Sin(b)) => a == b,
            (SExpr::Cos(a), SExpr::Cos(b)) => a == b,
            (SExpr::Tan(a), SExpr::Tan(b)) => a == b,
            (SExpr::Exp(a), SExpr::Exp(b)) => a == b,
            (SExpr::Ln(a), SExpr::Ln(b)) => a == b,
            (SExpr::Sqrt(a), SExpr::Sqrt(b)) => a == b,
            (SExpr::Abs(a), SExpr::Abs(b)) => a == b,
            (SExpr::Sign(a), SExpr::Sign(b)) => a == b,
            (SExpr::Square(a), SExpr::Square(b)) => a == b,
            (SExpr::Cube(a), SExpr::Cube(b)) => a == b,
            (SExpr::SinPi(a), SExpr::SinPi(b)) => a == b,
            (SExpr::CosPi(a), SExpr::CosPi(b)) => a == b,
            (SExpr::Add(a, b), SExpr::Add(c, d)) => a == c && b == d,
            (SExpr::Sub(a, b), SExpr::Sub(c, d)) => a == c && b == d,
            (SExpr::Mul(a, b), SExpr::Mul(c, d)) => a == c && b == d,
            (SExpr::Div(a, b), SExpr::Div(c, d)) => a == c && b == d,
            (SExpr::Pow(a, b), SExpr::Pow(c, d)) => a == c && b == d,
            (SExpr::Max(a, b), SExpr::Max(c, d)) => a == c && b == d,
            (SExpr::Min(a, b), SExpr::Min(c, d)) => a == c && b == d,
            (SExpr::Hypot(a, b), SExpr::Hypot(c, d)) => a == c && b == d,
            (SExpr::Atan2(a, b), SExpr::Atan2(c, d)) => a == c && b == d,
            (SExpr::Sum(a), SExpr::Sum(b)) => a == b,
            (SExpr::Prod(a), SExpr::Prod(b)) => a == b,
            (
                SExpr::Cmp {
                    op: oa,
                    lhs: la,
                    rhs: ra,
                },
                SExpr::Cmp {
                    op: ob,
                    lhs: lb,
                    rhs: rb,
                },
            ) => oa == ob && la == lb && ra == rb,
            (SExpr::IfElse(ca, ta, ea), SExpr::IfElse(cb, tb, eb)) =>
            {
                ca == cb && ta == tb && ea == eb
            },
            _ => false,
        }
    }
}

impl Eq for SExpr {}

impl Hash for SExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tag().hash(state);
        match self
        {
            SExpr::Var(v) => v.hash(state),
            SExpr::Const(c) =>
            {
                let bits = if *c == 0.0
                {
                    0.0f64.to_bits()
                }
                else
                {
                    c.to_bits()
                };
                bits.hash(state);
            },
            SExpr::Hole(i) => i.hash(state),
            SExpr::Neg(a)
            | SExpr::Sin(a)
            | SExpr::Cos(a)
            | SExpr::Tan(a)
            | SExpr::Exp(a)
            | SExpr::Ln(a)
            | SExpr::Sqrt(a)
            | SExpr::Abs(a)
            | SExpr::Sign(a)
            | SExpr::Square(a)
            | SExpr::Cube(a)
            | SExpr::SinPi(a)
            | SExpr::CosPi(a) => a.hash(state),
            SExpr::Add(a, b)
            | SExpr::Sub(a, b)
            | SExpr::Mul(a, b)
            | SExpr::Div(a, b)
            | SExpr::Pow(a, b)
            | SExpr::Max(a, b)
            | SExpr::Min(a, b)
            | SExpr::Hypot(a, b)
            | SExpr::Atan2(a, b) =>
            {
                a.hash(state);
                b.hash(state);
            },
            SExpr::Sum(vs) | SExpr::Prod(vs) =>
            {
                for v in vs
                {
                    v.hash(state);
                }
            },
            SExpr::Cmp { op, lhs, rhs } =>
            {
                op.hash(state);
                lhs.hash(state);
                rhs.hash(state);
            },
            SExpr::IfElse(c, t, e) =>
            {
                c.hash(state);
                t.hash(state);
                e.hash(state);
            },
        }
    }
}

impl SExpr {
    fn tag(&self) -> u8 {
        match self
        {
            SExpr::Var(_) => 0,
            SExpr::Const(_) => 1,
            SExpr::Hole(_) => 2,
            SExpr::Neg(_) => 10,
            SExpr::Sin(_) => 11,
            SExpr::Cos(_) => 12,
            SExpr::Tan(_) => 13,
            SExpr::Exp(_) => 14,
            SExpr::Ln(_) => 15,
            SExpr::Sqrt(_) => 16,
            SExpr::Abs(_) => 17,
            SExpr::Sign(_) => 18,
            SExpr::Square(_) => 19,
            SExpr::Cube(_) => 20,
            SExpr::SinPi(_) => 21,
            SExpr::CosPi(_) => 22,
            SExpr::Add(_, _) => 30,
            SExpr::Sub(_, _) => 31,
            SExpr::Mul(_, _) => 32,
            SExpr::Div(_, _) => 33,
            SExpr::Pow(_, _) => 34,
            SExpr::Max(_, _) => 35,
            SExpr::Min(_, _) => 36,
            SExpr::Hypot(_, _) => 37,
            SExpr::Atan2(_, _) => 38,
            SExpr::Sum(_) => 40,
            SExpr::Prod(_) => 41,
            SExpr::Cmp { .. } => 50,
            SExpr::IfElse(_, _, _) => 51,
        }
    }

    /// Merkle-style 64-bit hash for fast deduplication.
    pub fn merkle_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Number of nodes in the tree.
    pub fn size(&self) -> usize {
        match self
        {
            SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) => 1,
            SExpr::Neg(a)
            | SExpr::Sin(a)
            | SExpr::Cos(a)
            | SExpr::Tan(a)
            | SExpr::Exp(a)
            | SExpr::Ln(a)
            | SExpr::Sqrt(a)
            | SExpr::Abs(a)
            | SExpr::Sign(a)
            | SExpr::Square(a)
            | SExpr::Cube(a)
            | SExpr::SinPi(a)
            | SExpr::CosPi(a) => 1 + a.size(),
            SExpr::Add(a, b)
            | SExpr::Sub(a, b)
            | SExpr::Mul(a, b)
            | SExpr::Div(a, b)
            | SExpr::Pow(a, b)
            | SExpr::Max(a, b)
            | SExpr::Min(a, b)
            | SExpr::Hypot(a, b)
            | SExpr::Atan2(a, b) => 1 + a.size() + b.size(),
            SExpr::Sum(vs) | SExpr::Prod(vs) => 1 + vs.iter().map(|v| v.size()).sum::<usize>(),
            SExpr::Cmp { lhs, rhs, .. } => 1 + lhs.size() + rhs.size(),
            SExpr::IfElse(c, t, e) => 1 + c.size() + t.size() + e.size(),
        }
    }

    /// Depth of the expression tree.
    pub fn depth(&self) -> usize {
        match self
        {
            SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) => 0,
            SExpr::Neg(a)
            | SExpr::Sin(a)
            | SExpr::Cos(a)
            | SExpr::Tan(a)
            | SExpr::Exp(a)
            | SExpr::Ln(a)
            | SExpr::Sqrt(a)
            | SExpr::Abs(a)
            | SExpr::Sign(a)
            | SExpr::Square(a)
            | SExpr::Cube(a)
            | SExpr::SinPi(a)
            | SExpr::CosPi(a) => 1 + a.depth(),
            SExpr::Add(a, b)
            | SExpr::Sub(a, b)
            | SExpr::Mul(a, b)
            | SExpr::Div(a, b)
            | SExpr::Pow(a, b)
            | SExpr::Max(a, b)
            | SExpr::Min(a, b)
            | SExpr::Hypot(a, b)
            | SExpr::Atan2(a, b) => 1 + a.depth().max(b.depth()),
            SExpr::Sum(vs) | SExpr::Prod(vs) => 1 + vs.iter().map(|v| v.depth()).max().unwrap_or(0),
            SExpr::Cmp { lhs, rhs, .. } => 1 + lhs.depth().max(rhs.depth()),
            SExpr::IfElse(c, t, e) => 1 + c.depth().max(t.depth()).max(e.depth()),
        }
    }

    /// Collect all free variables used in the expression.
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, set: &mut HashSet<String>) {
        match self
        {
            SExpr::Var(v) =>
            {
                set.insert(v.clone());
            },
            SExpr::Const(_) | SExpr::Hole(_) =>
            {},
            SExpr::Neg(a)
            | SExpr::Sin(a)
            | SExpr::Cos(a)
            | SExpr::Tan(a)
            | SExpr::Exp(a)
            | SExpr::Ln(a)
            | SExpr::Sqrt(a)
            | SExpr::Abs(a)
            | SExpr::Sign(a)
            | SExpr::Square(a)
            | SExpr::Cube(a)
            | SExpr::SinPi(a)
            | SExpr::CosPi(a) => a.collect_vars(set),
            SExpr::Add(a, b)
            | SExpr::Sub(a, b)
            | SExpr::Mul(a, b)
            | SExpr::Div(a, b)
            | SExpr::Pow(a, b)
            | SExpr::Max(a, b)
            | SExpr::Min(a, b)
            | SExpr::Hypot(a, b)
            | SExpr::Atan2(a, b) =>
            {
                a.collect_vars(set);
                b.collect_vars(set);
            },
            SExpr::Sum(vs) | SExpr::Prod(vs) =>
            {
                for v in vs
                {
                    v.collect_vars(set);
                }
            },
            SExpr::Cmp { lhs, rhs, .. } =>
            {
                lhs.collect_vars(set);
                rhs.collect_vars(set);
            },
            SExpr::IfElse(c, t, e) =>
            {
                c.collect_vars(set);
                t.collect_vars(set);
                e.collect_vars(set);
            },
        }
    }

    /// Check whether any sub-tree is a Hole.
    pub fn has_holes(&self) -> bool {
        match self
        {
            SExpr::Hole(_) => true,
            SExpr::Var(_) | SExpr::Const(_) => false,
            SExpr::Neg(a)
            | SExpr::Sin(a)
            | SExpr::Cos(a)
            | SExpr::Tan(a)
            | SExpr::Exp(a)
            | SExpr::Ln(a)
            | SExpr::Sqrt(a)
            | SExpr::Abs(a)
            | SExpr::Sign(a)
            | SExpr::Square(a)
            | SExpr::Cube(a)
            | SExpr::SinPi(a)
            | SExpr::CosPi(a) => a.has_holes(),
            SExpr::Add(a, b)
            | SExpr::Sub(a, b)
            | SExpr::Mul(a, b)
            | SExpr::Div(a, b)
            | SExpr::Pow(a, b)
            | SExpr::Max(a, b)
            | SExpr::Min(a, b)
            | SExpr::Hypot(a, b)
            | SExpr::Atan2(a, b) => a.has_holes() || b.has_holes(),
            SExpr::Sum(vs) | SExpr::Prod(vs) => vs.iter().any(|v| v.has_holes()),
            SExpr::Cmp { lhs, rhs, .. } => lhs.has_holes() || rhs.has_holes(),
            SExpr::IfElse(c, t, e) => c.has_holes() || t.has_holes() || e.has_holes(),
        }
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// Numerically evaluate an SExpr with variable bindings.
/// Returns `Err` on domain violations.
pub fn eval(expr: &SExpr, vars: &HashMap<String, f64>) -> Result<f64, String> {
    match expr
    {
        SExpr::Const(c) => Ok(*c),
        SExpr::Var(v) => vars
            .get(v)
            .copied()
            .ok_or_else(|| format!("Undefined variable: {v}")),
        SExpr::Hole(_) => Err("Cannot evaluate hole".into()),
        SExpr::Neg(a) => Ok(-eval(a, vars)?),
        SExpr::Sin(a) => Ok(eval(a, vars)?.sin()),
        SExpr::Cos(a) => Ok(eval(a, vars)?.cos()),
        SExpr::Tan(a) => Ok(eval(a, vars)?.tan()),
        SExpr::Exp(a) => Ok(eval(a, vars)?.exp()),
        SExpr::Ln(a) =>
        {
            let v = eval(a, vars)?;
            if v <= 0.0
            {
                Err("ln of non-positive number".into())
            }
            else
            {
                Ok(v.ln())
            }
        },
        SExpr::Sqrt(a) =>
        {
            let v = eval(a, vars)?;
            if v < 0.0
            {
                Err("sqrt of negative number".into())
            }
            else
            {
                Ok(v.sqrt())
            }
        },
        SExpr::Abs(a) => Ok(eval(a, vars)?.abs()),
        SExpr::Sign(a) => Ok(eval(a, vars)?.signum()),
        SExpr::Square(a) =>
        {
            let v = eval(a, vars)?;
            Ok(v * v)
        },
        SExpr::Cube(a) =>
        {
            let v = eval(a, vars)?;
            Ok(v * v * v)
        },
        SExpr::SinPi(a) => Ok((std::f64::consts::PI * eval(a, vars)?).sin()),
        SExpr::CosPi(a) => Ok((std::f64::consts::PI * eval(a, vars)?).cos()),
        SExpr::Add(a, b) => Ok(eval(a, vars)? + eval(b, vars)?),
        SExpr::Sub(a, b) => Ok(eval(a, vars)? - eval(b, vars)?),
        SExpr::Mul(a, b) => Ok(eval(a, vars)? * eval(b, vars)?),
        SExpr::Div(a, b) =>
        {
            let den = eval(b, vars)?;
            if den == 0.0
            {
                Err("Division by zero".into())
            }
            else
            {
                Ok(eval(a, vars)? / den)
            }
        },
        SExpr::Pow(a, b) => Ok(eval(a, vars)?.powf(eval(b, vars)?)),
        SExpr::Max(a, b) => Ok(eval(a, vars)?.max(eval(b, vars)?)),
        SExpr::Min(a, b) => Ok(eval(a, vars)?.min(eval(b, vars)?)),
        SExpr::Hypot(a, b) => Ok(eval(a, vars)?.hypot(eval(b, vars)?)),
        SExpr::Atan2(a, b) => Ok(eval(a, vars)?.atan2(eval(b, vars)?)),
        SExpr::Sum(vs) =>
        {
            let mut s = 0.0;
            for v in vs
            {
                s += eval(v, vars)?;
            }
            Ok(s)
        },
        SExpr::Prod(vs) =>
        {
            let mut p = 1.0;
            for v in vs
            {
                p *= eval(v, vars)?;
            }
            Ok(p)
        },
        SExpr::Cmp { op, lhs, rhs } => Ok(op.apply(eval(lhs, vars)?, eval(rhs, vars)?)),
        SExpr::IfElse(cond, then, else_) =>
        {
            let c = eval(cond, vars)?;
            if c != 0.0
            {
                eval(then, vars)
            }
            else
            {
                eval(else_, vars)
            }
        },
    }
}

/// Create variable bindings from a specification entry.
pub fn bind_vars(variables: &[String], inputs: &[f64]) -> HashMap<String, f64> {
    variables
        .iter()
        .zip(inputs.iter())
        .map(|(v, &x)| (v.clone(), x))
        .collect()
}

/// Evaluate an expression on a single specification datum.
pub fn eval_on_entry(expr: &SExpr, variables: &[String], input: &[f64]) -> Result<f64, String> {
    eval(expr, &bind_vars(variables, input))
}

// ============================================================================
// Simplification / expression rewriting
// ============================================================================

/// Recursively simplify an expression via algebraic identities and constant
/// folding. Idempotent.
pub fn simplify(expr: &SExpr) -> SExpr {
    let s = match expr
    {
        SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) => expr.clone(),

        SExpr::Neg(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(-c),
            SExpr::Neg(inner) => *inner,
            sa => SExpr::Neg(Box::new(sa)),
        },
        SExpr::Sin(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c.sin()),
            sa => SExpr::Sin(Box::new(sa)),
        },
        SExpr::Cos(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c.cos()),
            sa => SExpr::Cos(Box::new(sa)),
        },
        SExpr::Tan(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c.tan()),
            sa => SExpr::Tan(Box::new(sa)),
        },
        SExpr::Exp(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c.exp()),
            sa => SExpr::Exp(Box::new(sa)),
        },
        SExpr::Ln(a) => match simplify(a)
        {
            SExpr::Const(c) if c > 0.0 => SExpr::Const(c.ln()),
            SExpr::Const(1.0) => SExpr::Const(0.0),
            sa => SExpr::Ln(Box::new(sa)),
        },
        SExpr::Sqrt(a) => match simplify(a)
        {
            SExpr::Const(c) if c >= 0.0 => SExpr::Const(c.sqrt()),
            SExpr::Const(0.0) => SExpr::Const(0.0),
            SExpr::Const(1.0) => SExpr::Const(1.0),
            sa => SExpr::Sqrt(Box::new(sa)),
        },
        SExpr::Abs(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c.abs()),
            SExpr::Square(inner) => SExpr::Square(inner),
            sa => SExpr::Abs(Box::new(sa)),
        },
        SExpr::Sign(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c.signum()),
            SExpr::Exp(_) | SExpr::Abs(_) | SExpr::Square(_) => SExpr::Const(1.0),
            sa => SExpr::Sign(Box::new(sa)),
        },
        SExpr::Square(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c * c),
            SExpr::Sqrt(inner) => *inner,
            SExpr::Neg(inner) => SExpr::Square(inner),
            sa => SExpr::Square(Box::new(sa)),
        },
        SExpr::Cube(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const(c * c * c),
            sa => SExpr::Cube(Box::new(sa)),
        },
        SExpr::SinPi(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const((std::f64::consts::PI * c).sin()),
            sa => SExpr::SinPi(Box::new(sa)),
        },
        SExpr::CosPi(a) => match simplify(a)
        {
            SExpr::Const(c) => SExpr::Const((std::f64::consts::PI * c).cos()),
            sa => SExpr::CosPi(Box::new(sa)),
        },

        SExpr::Add(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca + cb),
                (SExpr::Const(0.0), _) => sb,
                (_, SExpr::Const(0.0)) => sa,
                _ => SExpr::Add(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Sub(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca - cb),
                (_, SExpr::Const(0.0)) => sa,
                _ if sa == sb => SExpr::Const(0.0),
                _ => SExpr::Sub(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Mul(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca * cb),
                (SExpr::Const(0.0), _) | (_, SExpr::Const(0.0)) => SExpr::Const(0.0),
                (SExpr::Const(1.0), _) => sb,
                (_, SExpr::Const(1.0)) => sa,
                (SExpr::Const(-1.0), other) => SExpr::Neg(Box::new(other.clone())),
                (other, SExpr::Const(-1.0)) => SExpr::Neg(Box::new(other.clone())),
                _ => SExpr::Mul(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Div(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(0.0), _) => SExpr::Const(0.0),
                (SExpr::Const(ca), SExpr::Const(cb)) if *cb != 0.0 => SExpr::Const(ca / cb),
                (_, SExpr::Const(1.0)) => sa,
                _ if sa == sb => SExpr::Const(1.0),
                _ => SExpr::Div(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Pow(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca.powf(*cb)),
                (_, SExpr::Const(0.0)) => SExpr::Const(1.0),
                (_, SExpr::Const(1.0)) => sa,
                (SExpr::Const(0.0), _) => SExpr::Const(0.0),
                (SExpr::Const(1.0), _) => SExpr::Const(1.0),
                _ => SExpr::Pow(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Max(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca.max(*cb)),
                _ if sa == sb => sa,
                _ => SExpr::Max(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Min(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca.min(*cb)),
                _ if sa == sb => sa,
                _ => SExpr::Min(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Hypot(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca.hypot(*cb)),
                _ => SExpr::Hypot(Box::new(sa), Box::new(sb)),
            }
        },
        SExpr::Atan2(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (SExpr::Const(ca), SExpr::Const(cb)) => SExpr::Const(ca.atan2(*cb)),
                _ => SExpr::Atan2(Box::new(sa), Box::new(sb)),
            }
        },

        SExpr::Sum(vs) =>
        {
            let sv: Vec<SExpr> = vs.iter().map(simplify).collect();
            let mut const_sum = 0.0;
            let mut rest = vec![];
            for v in sv
            {
                match v
                {
                    SExpr::Const(0.0) =>
                    {},
                    SExpr::Const(c) => const_sum += c,
                    other => rest.push(other),
                }
            }
            if const_sum != 0.0
            {
                rest.push(SExpr::Const(const_sum));
            }
            match rest.len()
            {
                0 => SExpr::Const(0.0),
                1 => rest.into_iter().next().unwrap(),
                _ => SExpr::Sum(rest),
            }
        },
        SExpr::Prod(vs) =>
        {
            let sv: Vec<SExpr> = vs.iter().map(simplify).collect();
            let mut has_zero = false;
            let mut const_prod = 1.0;
            let mut rest = vec![];
            for v in sv
            {
                match v
                {
                    SExpr::Const(0.0) =>
                    {
                        has_zero = true;
                        break;
                    },
                    SExpr::Const(1.0) =>
                    {},
                    SExpr::Const(c) => const_prod *= c,
                    other => rest.push(other),
                }
            }
            if has_zero
            {
                return SExpr::Const(0.0);
            }
            if const_prod != 1.0
            {
                rest.push(SExpr::Const(const_prod));
            }
            match rest.len()
            {
                0 => SExpr::Const(1.0),
                1 => rest.into_iter().next().unwrap(),
                _ => SExpr::Prod(rest),
            }
        },

        SExpr::Cmp { op, lhs, rhs } =>
        {
            let sl = simplify(lhs);
            let sr = simplify(rhs);
            match (&sl, &sr)
            {
                (SExpr::Const(cl), SExpr::Const(cr)) => SExpr::Const(op.apply(*cl, *cr)),
                _ => SExpr::Cmp {
                    op: *op,
                    lhs: Box::new(sl),
                    rhs: Box::new(sr),
                },
            }
        },
        SExpr::IfElse(cond, then, else_) =>
        {
            let sc = simplify(cond);
            match &sc
            {
                SExpr::Const(c) if *c != 0.0 => simplify(then),
                SExpr::Const(_) => simplify(else_),
                _ => SExpr::IfElse(
                    Box::new(sc),
                    Box::new(simplify(then)),
                    Box::new(simplify(else_)),
                ),
            }
        },
    };
    s
}

/// Pattern-based rewrite: apply simplification until fix-point (max 100 iters).
pub fn rewrite(expr: &SExpr) -> SExpr {
    let mut current = expr.clone();
    for _ in 0..100
    {
        let next = simplify(&current);
        if next == current
        {
            return next;
        }
        current = next;
    }
    current
}

/// Constant folding: recursively evaluate all sub-expressions that contain
/// only constant terms.
pub fn constant_fold(expr: &SExpr) -> SExpr {
    simplify(expr)
}

/// Common subexpression elimination: identify structurally identical subtrees
/// and report duplicate counts.
pub fn cse_find_duplicates(expr: &SExpr) -> Vec<(SExpr, usize)> {
    let mut counts: HashMap<u64, (SExpr, usize)> = HashMap::new();
    cse_count_inner(expr, &mut counts);
    counts.into_values().filter(|(_, c)| *c > 1).collect()
}

fn cse_count_inner(expr: &SExpr, counts: &mut HashMap<u64, (SExpr, usize)>) {
    let h = expr.merkle_hash();
    let entry = counts.entry(h).or_insert_with(|| (expr.clone(), 0));
    entry.1 += 1;
    match expr
    {
        SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) =>
        {},
        SExpr::Neg(a)
        | SExpr::Sin(a)
        | SExpr::Cos(a)
        | SExpr::Tan(a)
        | SExpr::Exp(a)
        | SExpr::Ln(a)
        | SExpr::Sqrt(a)
        | SExpr::Abs(a)
        | SExpr::Sign(a)
        | SExpr::Square(a)
        | SExpr::Cube(a)
        | SExpr::SinPi(a)
        | SExpr::CosPi(a) => cse_count_inner(a, counts),
        SExpr::Add(a, b)
        | SExpr::Sub(a, b)
        | SExpr::Mul(a, b)
        | SExpr::Div(a, b)
        | SExpr::Pow(a, b)
        | SExpr::Max(a, b)
        | SExpr::Min(a, b)
        | SExpr::Hypot(a, b)
        | SExpr::Atan2(a, b) =>
        {
            cse_count_inner(a, counts);
            cse_count_inner(b, counts);
        },
        SExpr::Sum(vs) | SExpr::Prod(vs) =>
        {
            for v in vs
            {
                cse_count_inner(v, counts);
            }
        },
        SExpr::Cmp { lhs, rhs, .. } =>
        {
            cse_count_inner(lhs, counts);
            cse_count_inner(rhs, counts);
        },
        SExpr::IfElse(c, t, e) =>
        {
            cse_count_inner(c, counts);
            cse_count_inner(t, counts);
            cse_count_inner(e, counts);
        },
    }
}

// ============================================================================
// Cost model
// ============================================================================

/// Base cost weight for each operation kind. Higher => more expensive.
pub fn op_weight(expr: &SExpr) -> f64 {
    match expr
    {
        SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) => 0.0,
        SExpr::Neg(_) | SExpr::Abs(_) | SExpr::Sign(_) => 0.5,
        SExpr::Square(_) | SExpr::Cube(_) => 0.5,
        SExpr::Add(_, _) | SExpr::Sub(_, _) => 1.0,
        SExpr::Mul(_, _) | SExpr::Div(_, _) => 1.2,
        SExpr::Pow(_, _) | SExpr::Sqrt(_) => 2.0,
        SExpr::Exp(_) | SExpr::Ln(_) => 2.5,
        SExpr::Sin(_) | SExpr::Cos(_) | SExpr::Tan(_) => 2.0,
        SExpr::SinPi(_) | SExpr::CosPi(_) => 2.0,
        SExpr::Max(_, _) | SExpr::Min(_, _) => 1.0,
        SExpr::Hypot(_, _) | SExpr::Atan2(_, _) => 2.5,
        SExpr::Sum(_) | SExpr::Prod(_) => 1.5,
        SExpr::Cmp { .. } => 1.0,
        SExpr::IfElse(_, _, _) => 2.0,
    }
}

/// Total model cost = sum of node-op weights + size penalty.
pub fn cost(expr: &SExpr, size_penalty: f64) -> f64 {
    let mut c = op_weight(expr) + size_penalty;
    match expr
    {
        SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) =>
        {},
        SExpr::Neg(a)
        | SExpr::Sin(a)
        | SExpr::Cos(a)
        | SExpr::Tan(a)
        | SExpr::Exp(a)
        | SExpr::Ln(a)
        | SExpr::Sqrt(a)
        | SExpr::Abs(a)
        | SExpr::Sign(a)
        | SExpr::Square(a)
        | SExpr::Cube(a)
        | SExpr::SinPi(a)
        | SExpr::CosPi(a) => c += cost(a, size_penalty),
        SExpr::Add(a, b)
        | SExpr::Sub(a, b)
        | SExpr::Mul(a, b)
        | SExpr::Div(a, b)
        | SExpr::Pow(a, b)
        | SExpr::Max(a, b)
        | SExpr::Min(a, b)
        | SExpr::Hypot(a, b)
        | SExpr::Atan2(a, b) =>
        {
            c += cost(a, size_penalty) + cost(b, size_penalty);
        },
        SExpr::Sum(vs) | SExpr::Prod(vs) =>
        {
            for v in vs
            {
                c += cost(v, size_penalty);
            }
        },
        SExpr::Cmp { lhs, rhs, .. } =>
        {
            c += cost(lhs, size_penalty) + cost(rhs, size_penalty);
        },
        SExpr::IfElse(cond, then, else_) =>
        {
            c += cost(cond, size_penalty) + cost(then, size_penalty) + cost(else_, size_penalty);
        },
    }
    c
}

// ============================================================================
// Fitness helpers
// ============================================================================

/// Mean squared error of an expression on a specification.
pub fn mse(expr: &SExpr, variables: &[String], data: &[(Vec<f64>, f64)]) -> f64 {
    let n = data.len() as f64;
    if n == 0.0
    {
        return 0.0;
    }
    let mut sse = 0.0;
    for (inputs, target) in data
    {
        match eval_on_entry(expr, variables, inputs)
        {
            Ok(pred) if pred.is_finite() =>
            {
                sse += (pred - target).powi(2);
            },
            _ => return f64::INFINITY,
        }
    }
    sse / n
}

/// Regularised fitness = MSE + size_penalty * size.
pub fn regularised_fitness(
    expr: &SExpr,
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    size_penalty: f64,
) -> f64 {
    let m = mse(expr, variables, data);
    if !m.is_finite()
    {
        return f64::INFINITY;
    }
    m + size_penalty * expr.size() as f64
}

// ============================================================================
// Simple pseudo-random generator
// ============================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, n: usize) -> usize {
        if n == 0
        {
            0
        }
        else
        {
            (self.next() % n as u64) as usize
        }
    }
    fn bool(&mut self, p: f64) -> bool {
        self.f64() < p
    }
}

// ============================================================================
// Tree mutation operations
// ============================================================================

fn map_children(e: &SExpr, f: &mut dyn FnMut(&SExpr) -> SExpr) -> SExpr {
    match e
    {
        SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) => e.clone(),
        SExpr::Neg(a) => SExpr::Neg(Box::new(f(a))),
        SExpr::Sin(a) => SExpr::Sin(Box::new(f(a))),
        SExpr::Cos(a) => SExpr::Cos(Box::new(f(a))),
        SExpr::Tan(a) => SExpr::Tan(Box::new(f(a))),
        SExpr::Exp(a) => SExpr::Exp(Box::new(f(a))),
        SExpr::Ln(a) => SExpr::Ln(Box::new(f(a))),
        SExpr::Sqrt(a) => SExpr::Sqrt(Box::new(f(a))),
        SExpr::Abs(a) => SExpr::Abs(Box::new(f(a))),
        SExpr::Sign(a) => SExpr::Sign(Box::new(f(a))),
        SExpr::Square(a) => SExpr::Square(Box::new(f(a))),
        SExpr::Cube(a) => SExpr::Cube(Box::new(f(a))),
        SExpr::SinPi(a) => SExpr::SinPi(Box::new(f(a))),
        SExpr::CosPi(a) => SExpr::CosPi(Box::new(f(a))),
        SExpr::Add(a, b) => SExpr::Add(Box::new(f(a)), Box::new(f(b))),
        SExpr::Sub(a, b) => SExpr::Sub(Box::new(f(a)), Box::new(f(b))),
        SExpr::Mul(a, b) => SExpr::Mul(Box::new(f(a)), Box::new(f(b))),
        SExpr::Div(a, b) => SExpr::Div(Box::new(f(a)), Box::new(f(b))),
        SExpr::Pow(a, b) => SExpr::Pow(Box::new(f(a)), Box::new(f(b))),
        SExpr::Max(a, b) => SExpr::Max(Box::new(f(a)), Box::new(f(b))),
        SExpr::Min(a, b) => SExpr::Min(Box::new(f(a)), Box::new(f(b))),
        SExpr::Hypot(a, b) => SExpr::Hypot(Box::new(f(a)), Box::new(f(b))),
        SExpr::Atan2(a, b) => SExpr::Atan2(Box::new(f(a)), Box::new(f(b))),
        SExpr::Sum(vs) => SExpr::Sum(vs.iter().map(|v| f(v)).collect()),
        SExpr::Prod(vs) => SExpr::Prod(vs.iter().map(|v| f(v)).collect()),
        SExpr::Cmp { op, lhs, rhs } => SExpr::Cmp {
            op: *op,
            lhs: Box::new(f(lhs)),
            rhs: Box::new(f(rhs)),
        },
        SExpr::IfElse(c, t, e) => SExpr::IfElse(Box::new(f(c)), Box::new(f(t)), Box::new(f(e))),
    }
}

/// Returns the subtree at a pre-order index.
fn subtree_at(e: &SExpr, target: usize, counter: &mut usize) -> Option<SExpr> {
    let here = *counter;
    *counter += 1;
    if here == target
    {
        return Some(e.clone());
    }
    match e
    {
        SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) => None,
        SExpr::Neg(a)
        | SExpr::Sin(a)
        | SExpr::Cos(a)
        | SExpr::Tan(a)
        | SExpr::Exp(a)
        | SExpr::Ln(a)
        | SExpr::Sqrt(a)
        | SExpr::Abs(a)
        | SExpr::Sign(a)
        | SExpr::Square(a)
        | SExpr::Cube(a)
        | SExpr::SinPi(a)
        | SExpr::CosPi(a) => subtree_at(a, target, counter),
        SExpr::Add(a, b)
        | SExpr::Sub(a, b)
        | SExpr::Mul(a, b)
        | SExpr::Div(a, b)
        | SExpr::Pow(a, b)
        | SExpr::Max(a, b)
        | SExpr::Min(a, b)
        | SExpr::Hypot(a, b)
        | SExpr::Atan2(a, b) =>
        {
            let l = subtree_at(a, target, counter);
            if l.is_some()
            {
                l
            }
            else
            {
                subtree_at(b, target, counter)
            }
        },
        SExpr::Sum(vs) | SExpr::Prod(vs) =>
        {
            for v in vs
            {
                let r = subtree_at(v, target, counter);
                if r.is_some()
                {
                    return r;
                }
            }
            None
        },
        SExpr::Cmp { lhs, rhs, .. } =>
        {
            let l = subtree_at(lhs, target, counter);
            if l.is_some()
            {
                l
            }
            else
            {
                subtree_at(rhs, target, counter)
            }
        },
        SExpr::IfElse(c, t, e) =>
        {
            let r = subtree_at(c, target, counter);
            if r.is_some()
            {
                r
            }
            else
            {
                let r = subtree_at(t, target, counter);
                if r.is_some()
                {
                    r
                }
                else
                {
                    subtree_at(e, target, counter)
                }
            }
        },
    }
}

/// Replace the subtree at pre-order index `target` with `replacement`.
fn replace_at(e: &SExpr, target: usize, counter: &mut usize, replacement: &SExpr) -> SExpr {
    let here = *counter;
    *counter += 1;
    if here == target
    {
        return replacement.clone();
    }
    map_children(e, &mut |ch| replace_at(ch, target, counter, replacement))
}

// ============================================================================
// Random tree generation
// ============================================================================

/// Terminal constants available during synthesis.
const TERMINAL_CONSTANTS: [f64; 12] = [
    0.0,
    0.5,
    1.0,
    2.0,
    -1.0,
    3.0,
    4.0,
    5.0,
    std::f64::consts::PI,
    std::f64::consts::E,
    0.25,
    1.5,
];

const INT_CONSTANTS: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

/// Generate a random expression tree.
fn gen_tree(rng: &mut Rng, depth: usize, max_depth: usize, variables: &[String]) -> SExpr {
    let p_term = if depth >= max_depth
    {
        1.0
    }
    else if depth == 0
    {
        0.15
    }
    else
    {
        0.4
    };

    if rng.bool(p_term)
    {
        let r = rng.f64();
        if r < 0.35
        {
            SExpr::Var(variables[rng.range(variables.len())].clone())
        }
        else if r < 0.65
        {
            SExpr::Const(INT_CONSTANTS[rng.range(INT_CONSTANTS.len())])
        }
        else
        {
            SExpr::Const(TERMINAL_CONSTANTS[rng.range(TERMINAL_CONSTANTS.len())])
        }
    }
    else
    {
        match rng.range(17)
        {
            0 => SExpr::Add(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            1 => SExpr::Sub(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            2 => SExpr::Mul(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            3 => SExpr::Div(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            4 => SExpr::Pow(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(SExpr::Const(INT_CONSTANTS[rng.range(INT_CONSTANTS.len())])),
            ),
            5 => SExpr::Neg(Box::new(gen_tree(rng, depth + 1, max_depth, variables))),
            6 => SExpr::Sin(Box::new(gen_tree(rng, depth + 1, max_depth, variables))),
            7 => SExpr::Cos(Box::new(gen_tree(rng, depth + 1, max_depth, variables))),
            8 => SExpr::Exp(Box::new(gen_tree(rng, depth + 1, max_depth, variables))),
            9 => SExpr::Abs(Box::new(gen_tree(rng, depth + 1, max_depth, variables))),
            10 => SExpr::Sqrt(Box::new(gen_tree(rng, depth + 1, max_depth, variables))),
            11 => SExpr::Square(Box::new(gen_tree(rng, depth + 1, max_depth, variables))),
            12 => SExpr::Max(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            13 => SExpr::Min(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            14 => SExpr::Hypot(
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            15 => SExpr::IfElse(
                Box::new(gen_cmp_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
                Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
            ),
            _ => SExpr::Sum(
                (0..(2 + rng.range(2)))
                    .map(|_| gen_tree(rng, depth + 1, max_depth, variables))
                    .collect(),
            ),
        }
    }
}

fn gen_cmp_tree(rng: &mut Rng, depth: usize, max_depth: usize, variables: &[String]) -> SExpr {
    let ops = [CmpOp::Lt, CmpOp::Gt, CmpOp::Le, CmpOp::Ge, CmpOp::Equal];
    SExpr::Cmp {
        op: ops[rng.range(ops.len())],
        lhs: Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
        rhs: Box::new(gen_tree(rng, depth + 1, max_depth, variables)),
    }
}

// ============================================================================
// Sketch -- template with holes
// ============================================================================

/// A program sketch: a template expression where some nodes are `SExpr::Hole`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sketch {
    pub template: SExpr,
    pub variables: Vec<String>,
}

impl Sketch {
    pub fn new(template: SExpr, variables: Vec<String>) -> Self {
        Sketch {
            template,
            variables,
        }
    }

    pub fn hole_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        collect_holes(&self.template, &mut indices);
        indices.sort();
        indices.dedup();
        indices
    }

    pub fn num_holes(&self) -> usize {
        self.hole_indices().len()
    }
}

fn collect_holes(e: &SExpr, indices: &mut Vec<usize>) {
    match e
    {
        SExpr::Hole(i) => indices.push(*i),
        SExpr::Var(_) | SExpr::Const(_) =>
        {},
        SExpr::Neg(a)
        | SExpr::Sin(a)
        | SExpr::Cos(a)
        | SExpr::Tan(a)
        | SExpr::Exp(a)
        | SExpr::Ln(a)
        | SExpr::Sqrt(a)
        | SExpr::Abs(a)
        | SExpr::Sign(a)
        | SExpr::Square(a)
        | SExpr::Cube(a)
        | SExpr::SinPi(a)
        | SExpr::CosPi(a) => collect_holes(a, indices),
        SExpr::Add(a, b)
        | SExpr::Sub(a, b)
        | SExpr::Mul(a, b)
        | SExpr::Div(a, b)
        | SExpr::Pow(a, b)
        | SExpr::Max(a, b)
        | SExpr::Min(a, b)
        | SExpr::Hypot(a, b)
        | SExpr::Atan2(a, b) =>
        {
            collect_holes(a, indices);
            collect_holes(b, indices);
        },
        SExpr::Sum(vs) | SExpr::Prod(vs) =>
        {
            for v in vs
            {
                collect_holes(v, indices);
            }
        },
        SExpr::Cmp { lhs, rhs, .. } =>
        {
            collect_holes(lhs, indices);
            collect_holes(rhs, indices);
        },
        SExpr::IfElse(c, t, e) =>
        {
            collect_holes(c, indices);
            collect_holes(t, indices);
            collect_holes(e, indices);
        },
    }
}

/// Replace every `Hole(i)` in `e` with `fills[i]`.
fn fill_holes(e: &SExpr, fills: &[SExpr]) -> SExpr {
    match e
    {
        SExpr::Hole(i) =>
        {
            if *i < fills.len()
            {
                fills[*i].clone()
            }
            else
            {
                SExpr::Hole(*i)
            }
        },
        SExpr::Var(_) | SExpr::Const(_) => e.clone(),
        _ => map_children(e, &mut |ch| fill_holes(ch, fills)),
    }
}

// ============================================================================
// Bottom-up enumeration (width-bounded)
// ============================================================================

/// Bottom-up synthesis: enumerates expressions from terminals up, bounded by
/// `max_size`. At each size bucket the best `width` expressions are kept.
pub fn bottom_up(
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    max_size: usize,
    width: usize,
) -> Vec<SExpr> {
    let mut bank: Vec<Vec<SExpr>> = vec![vec![]; max_size + 1];

    // Seed terminals at size 1
    for v in variables
    {
        bank[1].push(SExpr::Var(v.clone()));
    }
    for &c in &TERMINAL_CONSTANTS
    {
        bank[1].push(SExpr::Const(c));
    }
    for &c in &INT_CONSTANTS
    {
        if !TERMINAL_CONSTANTS.contains(&c)
        {
            bank[1].push(SExpr::Const(c));
        }
    }

    let mut seen: HashSet<u64> = bank[1].iter().map(|e| e.merkle_hash()).collect();

    for sz in 2..=max_size
    {
        let mut candidates = Vec::new();

        // Unary ops
        for a_ix in 1..sz
        {
            for a in &bank[a_ix]
            {
                for una in unary_ops_iter(a)
                {
                    let sz_una = una.size();
                    if sz_una == sz
                    {
                        let h = una.merkle_hash();
                        if !seen.contains(&h) && looks_ok(&una, variables, data)
                        {
                            seen.insert(h);
                            candidates.push(una);
                        }
                    }
                }
            }
        }

        // Binary ops
        for a_ix in 1..sz
        {
            for a in &bank[a_ix]
            {
                let b_ix = sz - a.size();
                if b_ix < 1 || b_ix > bank.len()
                {
                    continue;
                }
                for b in &bank[b_ix]
                {
                    for bn in binary_ops_iter(a, b)
                    {
                        if bn.size() == sz
                        {
                            let h = bn.merkle_hash();
                            if !seen.contains(&h) && looks_ok(&bn, variables, data)
                            {
                                seen.insert(h);
                                candidates.push(bn);
                            }
                        }
                    }
                }
            }
        }

        // N-ary Sum/Prod (2-3 children, each child from bank)
        for num_children in 2..=3_usize
        {
            let child_budget = sz - 1;
            for a_ix in 1..=child_budget.saturating_sub(1)
            {
                if a_ix >= bank.len()
                {
                    continue;
                }
                for b_ix in 1..=(child_budget - a_ix)
                {
                    if b_ix >= bank.len()
                    {
                        continue;
                    }
                    let c_ix = child_budget - a_ix - b_ix;
                    for a in &bank[a_ix]
                    {
                        for b in &bank[b_ix]
                        {
                            let mut children = vec![a.clone(), b.clone()];
                            if num_children >= 3 && c_ix >= 1 && c_ix < bank.len()
                            {
                                for c in &bank[c_ix]
                                {
                                    children.push(c.clone());
                                    try_nary(
                                        &mut candidates,
                                        &mut seen,
                                        children.clone(),
                                        sz,
                                        variables,
                                        data,
                                    );
                                    children.pop();
                                }
                            }
                            else if num_children == 2
                            {
                                try_nary(
                                    &mut candidates,
                                    &mut seen,
                                    children.clone(),
                                    sz,
                                    variables,
                                    data,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Keep top `width` by MSE
        candidates.sort_by(|a, b| {
            let ma = mse(a, variables, data);
            let mb = mse(b, variables, data);
            ma.partial_cmp(&mb).unwrap_or(Ordering::Equal)
        });
        candidates.truncate(width);
        bank[sz] = candidates;
    }

    let mut all: Vec<SExpr> = bank.into_iter().flatten().collect();
    all.sort_by(|a, b| {
        let ma = mse(a, variables, data);
        let mb = mse(b, variables, data);
        ma.partial_cmp(&mb).unwrap_or(Ordering::Equal)
    });
    all
}

fn unary_ops_iter(a: &SExpr) -> Vec<SExpr> {
    let b = || Box::new(a.clone());
    vec![
        SExpr::Neg(b()),
        SExpr::Sin(b()),
        SExpr::Cos(b()),
        SExpr::Tan(b()),
        SExpr::Exp(b()),
        SExpr::Ln(b()),
        SExpr::Sqrt(b()),
        SExpr::Abs(b()),
        SExpr::Sign(b()),
        SExpr::Square(b()),
        SExpr::Cube(b()),
        SExpr::SinPi(b()),
        SExpr::CosPi(b()),
    ]
}

fn binary_ops_iter(a: &SExpr, b: &SExpr) -> Vec<SExpr> {
    let ab = || (Box::new(a.clone()), Box::new(b.clone()));
    vec![
        SExpr::Add(ab().0, ab().1),
        SExpr::Sub(ab().0, ab().1),
        SExpr::Mul(ab().0, ab().1),
        SExpr::Div(ab().0, ab().1),
        SExpr::Pow(ab().0, ab().1),
        SExpr::Max(ab().0, ab().1),
        SExpr::Min(ab().0, ab().1),
        SExpr::Hypot(ab().0, ab().1),
        SExpr::Atan2(ab().0, ab().1),
    ]
}

fn try_nary(
    candidates: &mut Vec<SExpr>,
    seen: &mut HashSet<u64>,
    children: Vec<SExpr>,
    max_sz: usize,
    variables: &[String],
    data: &[(Vec<f64>, f64)],
) {
    for e in [SExpr::Sum(children.clone()), SExpr::Prod(children)]
    {
        if e.size() <= max_sz
        {
            let h = e.merkle_hash();
            if !seen.contains(&h) && looks_ok(&e, variables, data)
            {
                seen.insert(h);
                candidates.push(e);
            }
        }
    }
}

fn looks_ok(e: &SExpr, variables: &[String], data: &[(Vec<f64>, f64)]) -> bool {
    for (inputs, _) in data.iter().take(5)
    {
        match eval_on_entry(e, variables, inputs)
        {
            Ok(v) if v.is_finite() =>
            {},
            _ => return false,
        }
    }
    true
}

// ============================================================================
// Top-down synthesis with type-directed search
// ============================================================================

/// Top-down (hole-driven) synthesis.
pub fn top_down(
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    max_depth: usize,
    beam_width: usize,
) -> Vec<SExpr> {
    let hole = SExpr::Hole(0);
    let mut beam = vec![hole];
    let mut seen: HashSet<u64> = HashSet::new();
    seen.insert(hole_hash());

    for _ in 0..(max_depth * 3)
    {
        let mut candidates: BinaryHeap<Reverse<OrderedExpr>> = BinaryHeap::new();

        for expr in &beam
        {
            if !expr.has_holes()
            {
                let f = regularised_fitness(expr, variables, data, 0.001);
                if f.is_finite()
                {
                    candidates.push(Reverse(OrderedExpr(f, expr.clone())));
                }
                continue;
            }
            for refn in hole_refinements(expr, variables)
            {
                let h = refn.merkle_hash();
                if seen.contains(&h)
                {
                    continue;
                }
                seen.insert(h);
                let f = regularised_fitness(&refn, variables, data, 0.001);
                if f.is_finite()
                {
                    candidates.push(Reverse(OrderedExpr(f, refn)));
                }
            }
        }

        beam.clear();
        while let Some(Reverse(OrderedExpr(_, e))) = candidates.pop()
        {
            if beam.len() >= beam_width
            {
                break;
            }
            beam.push(e);
        }
        if beam.is_empty()
        {
            break;
        }
    }

    let mut result: Vec<SExpr> = beam.into_iter().filter(|e| !e.has_holes()).collect();
    result.sort_by(|a, b| {
        let fa = regularised_fitness(a, variables, data, 0.001);
        let fb = regularised_fitness(b, variables, data, 0.001);
        fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
    });
    result
}

use std::sync::OnceLock;
fn hole_hash() -> u64 {
    static H: OnceLock<u64> = OnceLock::new();
    *H.get_or_init(|| SExpr::Hole(0).merkle_hash())
}

fn hole_refinements(e: &SExpr, variables: &[String]) -> Vec<SExpr> {
    let hole_pos = find_first_hole(e, &mut 0);
    if hole_pos.is_none()
    {
        return vec![e.clone()];
    }
    let pos = hole_pos.unwrap();

    let mut fillers: Vec<SExpr> = Vec::new();
    for v in variables
    {
        fillers.push(SExpr::Var(v.clone()));
    }
    for &c in &TERMINAL_CONSTANTS
    {
        fillers.push(SExpr::Const(c));
    }
    for &c in &INT_CONSTANTS
    {
        if !TERMINAL_CONSTANTS.contains(&c)
        {
            fillers.push(SExpr::Const(c));
        }
    }
    // Unary ops with new holes
    for &tag in &[10u8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    {
        let child = SExpr::Hole(1);
        let u = match tag
        {
            10 => SExpr::Neg(Box::new(child)),
            11 => SExpr::Sin(Box::new(child)),
            12 => SExpr::Cos(Box::new(child)),
            13 => SExpr::Tan(Box::new(child)),
            14 => SExpr::Exp(Box::new(child)),
            15 => SExpr::Ln(Box::new(child)),
            16 => SExpr::Sqrt(Box::new(child)),
            17 => SExpr::Abs(Box::new(child)),
            18 => SExpr::Sign(Box::new(child)),
            19 => SExpr::Square(Box::new(child)),
            20 => SExpr::Cube(Box::new(child)),
            21 => SExpr::SinPi(Box::new(child)),
            _ => SExpr::CosPi(Box::new(child)),
        };
        fillers.push(u);
    }
    // Binary ops with new holes
    for &tag in &[30u8, 31, 32, 33, 34, 35, 36, 37, 38]
    {
        let l = Box::new(SExpr::Hole(1));
        let r = Box::new(SExpr::Hole(2));
        let b = match tag
        {
            30 => SExpr::Add(l, r),
            31 => SExpr::Sub(l, r),
            32 => SExpr::Mul(l, r),
            33 => SExpr::Div(l, r),
            34 => SExpr::Pow(l, r),
            35 => SExpr::Max(l, r),
            36 => SExpr::Min(l, r),
            37 => SExpr::Hypot(l, r),
            _ => SExpr::Atan2(l, r),
        };
        fillers.push(b);
    }

    fillers
        .into_iter()
        .map(|filler| replace_at(e, pos, &mut 0, &filler))
        .collect()
}

fn find_first_hole(e: &SExpr, counter: &mut usize) -> Option<usize> {
    let here = *counter;
    *counter += 1;
    if let SExpr::Hole(_) = e
    {
        return Some(here);
    }
    match e
    {
        SExpr::Var(_) | SExpr::Const(_) | SExpr::Hole(_) => None,
        SExpr::Neg(a)
        | SExpr::Sin(a)
        | SExpr::Cos(a)
        | SExpr::Tan(a)
        | SExpr::Exp(a)
        | SExpr::Ln(a)
        | SExpr::Sqrt(a)
        | SExpr::Abs(a)
        | SExpr::Sign(a)
        | SExpr::Square(a)
        | SExpr::Cube(a)
        | SExpr::SinPi(a)
        | SExpr::CosPi(a) => find_first_hole(a, counter),
        SExpr::Add(a, b)
        | SExpr::Sub(a, b)
        | SExpr::Mul(a, b)
        | SExpr::Div(a, b)
        | SExpr::Pow(a, b)
        | SExpr::Max(a, b)
        | SExpr::Min(a, b)
        | SExpr::Hypot(a, b)
        | SExpr::Atan2(a, b) =>
        {
            let r = find_first_hole(a, counter);
            if r.is_some()
            {
                r
            }
            else
            {
                find_first_hole(b, counter)
            }
        },
        SExpr::Sum(vs) | SExpr::Prod(vs) =>
        {
            for v in vs
            {
                let r = find_first_hole(v, counter);
                if r.is_some()
                {
                    return r;
                }
            }
            None
        },
        SExpr::Cmp { lhs, rhs, .. } =>
        {
            let r = find_first_hole(lhs, counter);
            if r.is_some()
            {
                r
            }
            else
            {
                find_first_hole(rhs, counter)
            }
        },
        SExpr::IfElse(c, t, e) =>
        {
            let r = find_first_hole(c, counter);
            if r.is_some()
            {
                r
            }
            else
            {
                let r = find_first_hole(t, counter);
                if r.is_some()
                {
                    r
                }
                else
                {
                    find_first_hole(e, counter)
                }
            }
        },
    }
}

#[derive(Debug)]
struct OrderedExpr(f64, SExpr);

impl PartialEq for OrderedExpr {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for OrderedExpr {}
impl PartialOrd for OrderedExpr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedExpr {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// ============================================================================
// Genetic programming
// ============================================================================

/// Run tree-based genetic programming.
pub fn genetic_programming(
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    pop_size: usize,
    generations: usize,
    max_size: usize,
    size_penalty: f64,
    seeds: &[u64],
) -> Vec<(f64, SExpr)> {
    let mut rng = Rng::new(seeds[0]);
    let mut pop: Vec<SExpr> = (0..pop_size)
        .map(|_| gen_tree(&mut rng, 0, 5, variables))
        .collect();
    let mut best_overall: Vec<(f64, SExpr)> = Vec::new();
    let mut best_seen = f64::INFINITY;

    for _gen in 0..generations
    {
        let scores: Vec<f64> = pop
            .iter()
            .map(|e| regularised_fitness(e, variables, data, size_penalty))
            .collect();

        for (i, &s) in scores.iter().enumerate()
        {
            if s.is_finite() && s < best_seen
            {
                best_seen = s;
                best_overall.push((s, pop[i].clone()));
            }
        }

        let elite_n = (pop_size / 20).max(1);
        let mut indices: Vec<usize> = (0..pop.len()).collect();
        indices.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(Ordering::Equal));
        let mut next: Vec<SExpr> = indices[..elite_n].iter().map(|&i| pop[i].clone()).collect();

        while next.len() < pop_size
        {
            let pa = tournament_select(&pop, &scores, 3, &mut rng);
            let pb = tournament_select(&pop, &scores, 3, &mut rng);

            let mut child = if rng.bool(0.75)
            {
                let sx = pop[pa].size();
                let sy = pop[pb].size();
                if sx > 0 && sy > 0
                {
                    let ix = rng.range(sx);
                    let iy = rng.range(sy);
                    let sub = subtree_at(&pop[pb], iy, &mut 0).unwrap_or(pop[pb].clone());
                    replace_at(&pop[pa], ix, &mut 0, &sub)
                }
                else
                {
                    pop[pa].clone()
                }
            }
            else
            {
                pop[pa].clone()
            };

            if rng.bool(0.25)
            {
                if child.size() > 0
                {
                    let ix = rng.range(child.size());
                    let fresh = gen_tree(&mut rng, 0, 3, variables);
                    child = replace_at(&child, ix, &mut 0, &fresh);
                }
            }

            if child.size() > max_size
            {
                child = pop[pa].clone();
            }

            next.push(child);
        }

        pop = next;
        if best_seen < 1e-12
        {
            break;
        }
    }

    best_overall.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    best_overall.dedup_by(|a, b| a.1 == b.1);
    best_overall
}

fn tournament_select(pop: &[SExpr], scores: &[f64], k: usize, rng: &mut Rng) -> usize {
    let mut best_idx = rng.range(pop.len());
    let mut best_score = scores[best_idx];
    for _ in 1..k
    {
        let idx = rng.range(pop.len());
        let sc = scores[idx];
        if sc.is_finite() && (!best_score.is_finite() || sc < best_score)
        {
            best_idx = idx;
            best_score = sc;
        }
    }
    best_idx
}

// ============================================================================
// Stochastic beam search
// ============================================================================

/// Stochastic beam search: maintains a beam of `beam_width` candidates.
pub fn beam_search(
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    beam_width: usize,
    max_iter: usize,
    size_penalty: f64,
    seed: u64,
) -> Vec<SExpr> {
    let mut rng = Rng::new(seed);
    let mut beam: Vec<(f64, SExpr)> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();

    for v in variables
    {
        let e = SExpr::Var(v.clone());
        let h = e.merkle_hash();
        if seen.insert(h)
        {
            let f = regularised_fitness(&e, variables, data, size_penalty);
            if f.is_finite()
            {
                beam.push((f, e));
            }
        }
    }
    for &c in &TERMINAL_CONSTANTS
    {
        let e = SExpr::Const(c);
        let h = e.merkle_hash();
        if seen.insert(h)
        {
            let f = regularised_fitness(&e, variables, data, size_penalty);
            if f.is_finite()
            {
                beam.push((f, e));
            }
        }
    }
    beam.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    beam.truncate(beam_width);

    for _ in 0..max_iter
    {
        let mut candidates = beam.clone();
        for (_, expr) in &beam
        {
            let sib = rng.range(beam.len().max(1));
            let sib_expr = &beam[sib].1;
            for new_e in stochastic_expansions(expr, sib_expr, variables, &mut rng)
            {
                let h = new_e.merkle_hash();
                if seen.contains(&h)
                {
                    continue;
                }
                seen.insert(h);
                let f = regularised_fitness(&new_e, variables, data, size_penalty);
                if f.is_finite()
                {
                    candidates.push((f, new_e));
                }
            }
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        candidates.truncate(beam_width);
        beam = candidates;
        if beam.first().map_or(false, |(f, _)| *f < 1e-12)
        {
            break;
        }
    }

    beam.into_iter().map(|(_, e)| e).collect()
}

fn stochastic_expansions(a: &SExpr, b: &SExpr, _variables: &[String], rng: &mut Rng) -> Vec<SExpr> {
    let mut res = Vec::new();
    let ab = || (Box::new(a.clone()), Box::new(b.clone()));
    if rng.bool(0.5)
    {
        res.push(SExpr::Neg(Box::new(a.clone())));
        res.push(SExpr::Sin(Box::new(a.clone())));
        res.push(SExpr::Cos(Box::new(a.clone())));
        res.push(SExpr::Exp(Box::new(a.clone())));
        res.push(SExpr::Square(Box::new(a.clone())));
    }
    if rng.bool(0.5)
    {
        res.push(SExpr::Add(ab().0, ab().1));
        res.push(SExpr::Sub(ab().0, ab().1));
        res.push(SExpr::Mul(ab().0, ab().1));
        res.push(SExpr::Div(ab().0, ab().1));
        res.push(SExpr::Max(ab().0, ab().1));
    }
    if rng.bool(0.25)
    {
        res.push(SExpr::Sum(vec![a.clone(), b.clone()]));
        res.push(SExpr::Prod(vec![a.clone(), b.clone()]));
    }
    res
}

// ============================================================================
// Sketch completion
// ============================================================================

/// Complete a sketch by enumerating hole fillers.
pub fn sketch_complete(
    sketch: &Sketch,
    data: &[(Vec<f64>, f64)],
    max_filler_size: usize,
    max_results: usize,
    size_penalty: f64,
) -> Vec<(f64, SExpr)> {
    let holes = sketch.hole_indices();
    if holes.is_empty()
    {
        let f = regularised_fitness(&sketch.template, &sketch.variables, data, size_penalty);
        return if f.is_finite()
        {
            vec![(f, sketch.template.clone())]
        }
        else
        {
            vec![]
        };
    }

    let filler_bank: Vec<Vec<SExpr>> = holes
        .iter()
        .map(|_| bottom_up(&sketch.variables, data, max_filler_size.min(4), 20))
        .collect();

    let mut results = Vec::new();
    let mut counter = 0;
    enumerate_cartesian(
        &filler_bank,
        &sketch.template,
        &sketch.variables,
        data,
        size_penalty,
        &mut vec![],
        &mut results,
        &mut counter,
        max_results * 5,
    );

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    results.truncate(max_results);
    results
}

fn enumerate_cartesian(
    bank: &[Vec<SExpr>],
    template: &SExpr,
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    size_penalty: f64,
    current: &mut Vec<SExpr>,
    results: &mut Vec<(f64, SExpr)>,
    counter: &mut usize,
    max_evals: usize,
) {
    if current.len() == bank.len()
    {
        let filled = fill_holes(template, current);
        let f = regularised_fitness(&filled, variables, data, size_penalty);
        if f.is_finite()
        {
            results.push((f, filled));
        }
        *counter += 1;
        return;
    }
    for filler in &bank[current.len()]
    {
        if *counter >= max_evals
        {
            return;
        }
        current.push(filler.clone());
        enumerate_cartesian(
            bank,
            template,
            variables,
            data,
            size_penalty,
            current,
            results,
            counter,
            max_evals,
        );
        current.pop();
    }
}

// ============================================================================
// Verification utilities
// ============================================================================

/// Split specification into train / test (80/20).
pub fn train_test_split(data: &[(Vec<f64>, f64)]) -> (Vec<(Vec<f64>, f64)>, Vec<(Vec<f64>, f64)>) {
    let n = data.len();
    if n <= 1
    {
        return (data.to_vec(), vec![]);
    }
    let split = (n as f64 * 0.8) as usize;
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev()
    {
        let j = (i
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
            % (i + 1);
        idx.swap(i, j);
    }
    let train: Vec<_> = idx[..split].iter().map(|&i| data[i].clone()).collect();
    let test: Vec<_> = idx[split..].iter().map(|&i| data[i].clone()).collect();
    (train, test)
}

/// MSE on held-out (test) data.
pub fn verify_on_held_out(expr: &SExpr, variables: &[String], test: &[(Vec<f64>, f64)]) -> f64 {
    mse(expr, variables, test)
}

/// Extrapolation check: test expression on inputs beyond training range.
pub fn extrapolation_check(
    expr: &SExpr,
    variables: &[String],
    train_data: &[(Vec<f64>, f64)],
    extrap_factor: f64,
    num_points: usize,
) -> f64 {
    if train_data.is_empty()
    {
        return 0.0;
    }
    let dim = variables.len();
    let mut test_data = Vec::new();

    for i in 0..num_points
    {
        let mut inp = Vec::with_capacity(dim);
        for v in 0..dim
        {
            let col: Vec<f64> = train_data.iter().map(|(iv, _)| iv[v]).collect();
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max - min;
            let t = (i as f64 * 1.7 + 0.3) % 1.0;
            let val = if t < 0.5
            {
                min - range * extrap_factor * (0.5 - t)
            }
            else
            {
                max + range * extrap_factor * (t - 0.5)
            };
            inp.push(val);
        }
        test_data.push((inp, 0.0));
    }

    let mut sse = 0.0;
    let mut count = 0;
    for (inputs, _) in &test_data
    {
        match eval_on_entry(expr, variables, inputs)
        {
            Ok(v) if v.is_finite() =>
            {
                count += 1;
                if v.abs() > 1e6
                {
                    sse += v.abs();
                }
            },
            _ =>
            {
                sse += 1e3;
            },
        }
    }
    if count == 0
    {
        f64::INFINITY
    }
    else
    {
        sse / test_data.len() as f64
    }
}

// ============================================================================
// Inductive synthesis with bias
// ============================================================================

/// Occam's razor: simplicity penalty proportional to expression size.
pub fn occam_penalty(expr: &SExpr, lambda: f64) -> f64 {
    lambda * expr.size() as f64
}

/// Domain-specific bias: penalise non-polynomial ops.
pub fn domain_bias(expr: &SExpr, _data: &[(Vec<f64>, f64)]) -> f64 {
    let mut penalty = 0.0;
    count_ops_bias(expr, &mut penalty);
    penalty
}

fn count_ops_bias(e: &SExpr, p: &mut f64) {
    match e
    {
        SExpr::Sin(_) | SExpr::Cos(_) | SExpr::Tan(_) => *p += 0.5,
        SExpr::Exp(_) | SExpr::Ln(_) => *p += 0.5,
        SExpr::IfElse(_, _, _) => *p += 0.3,
        SExpr::Hole(_) | SExpr::Var(_) | SExpr::Const(_) =>
        {},
        SExpr::Neg(a)
        | SExpr::Sqrt(a)
        | SExpr::Abs(a)
        | SExpr::Sign(a)
        | SExpr::Square(a)
        | SExpr::Cube(a)
        | SExpr::SinPi(a)
        | SExpr::CosPi(a) => count_ops_bias(a, p),
        SExpr::Add(a, b)
        | SExpr::Sub(a, b)
        | SExpr::Mul(a, b)
        | SExpr::Div(a, b)
        | SExpr::Pow(a, b)
        | SExpr::Max(a, b)
        | SExpr::Min(a, b)
        | SExpr::Hypot(a, b)
        | SExpr::Atan2(a, b) =>
        {
            count_ops_bias(a, p);
            count_ops_bias(b, p);
        },
        SExpr::Sum(vs) | SExpr::Prod(vs) =>
        {
            for v in vs
            {
                count_ops_bias(v, p);
            }
        },
        SExpr::Cmp { lhs, rhs, .. } =>
        {
            count_ops_bias(lhs, p);
            count_ops_bias(rhs, p);
        },
    }
}

/// Incremental synthesis: increase complexity gradually.
pub fn incremental_synthesis(
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    initial_complexity: usize,
    step: usize,
    max_complexity: usize,
    size_penalty: f64,
) -> Vec<(f64, SExpr)> {
    let mut all_results = Vec::new();

    for budget in (initial_complexity..=max_complexity).step_by(step)
    {
        let results = bottom_up(variables, data, budget, 50);
        for e in results
        {
            let f = regularised_fitness(&e, variables, data, size_penalty);
            if f.is_finite()
            {
                all_results.push((f, e));
            }
        }
        if all_results.iter().any(|(f, _)| *f < 1e-10)
        {
            break;
        }
    }

    all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    all_results.dedup_by(|a, b| a.1 == b.1);
    all_results
}

/// Quick polynomial-fit.
pub fn try_polynomial_fit(
    variables: &[String],
    data: &[(Vec<f64>, f64)],
    max_degree: usize,
) -> Option<SExpr> {
    if variables.len() != 1
    {
        return None;
    }
    let xs: Vec<f64> = data.iter().map(|(iv, _)| iv[0]).collect();
    let ys: Vec<f64> = data.iter().map(|(_, t)| *t).collect();

    let mut best: Option<(SExpr, f64)> = None;
    for deg in 1..=max_degree
    {
        if let Some(coeffs) = polyfit(&xs, &ys, deg)
        {
            let expr = poly_to_expr(variables, &coeffs);
            let m = mse(&expr, variables, data);
            if m.is_finite() && m < 1e-3
            {
                if best.as_ref().map_or(true, |(_, bm)| m < *bm)
                {
                    best = Some((expr, m));
                }
            }
        }
    }
    best.map(|(e, _)| e)
}

fn polyfit(xs: &[f64], ys: &[f64], degree: usize) -> Option<Vec<f64>> {
    let n = xs.len();
    if n < degree + 1
    {
        return None;
    }
    let k = degree + 1;

    let mut v = vec![0.0; n * k];
    for (row, &x) in xs.iter().enumerate()
    {
        let mut pow = 1.0;
        for col in 0..k
        {
            v[row * k + col] = pow;
            pow *= x;
        }
    }

    let mut vtv = vec![0.0; k * k];
    let mut vty = vec![0.0; k];
    for i in 0..k
    {
        for j in 0..k
        {
            let mut s = 0.0;
            for row in 0..n
            {
                s += v[row * k + i] * v[row * k + j];
            }
            vtv[i * k + j] = s;
        }
        let mut s = 0.0;
        for row in 0..n
        {
            s += v[row * k + i] * ys[row];
        }
        vty[i] = s;
    }

    for col in 0..k
    {
        let pivot = vtv[col * k + col];
        if pivot.abs() < 1e-15
        {
            return None;
        }
        for j in 0..k
        {
            vtv[col * k + j] /= pivot;
        }
        vty[col] /= pivot;
        for row in 0..k
        {
            if row == col
            {
                continue;
            }
            let factor = vtv[row * k + col];
            for j in 0..k
            {
                vtv[row * k + j] -= factor * vtv[col * k + j];
            }
            vty[row] -= factor * vty[col];
        }
    }
    Some(vty)
}

fn poly_to_expr(variables: &[String], coeffs: &[f64]) -> SExpr {
    let var = &variables[0];
    let mut terms = Vec::new();
    for (deg, &coef) in coeffs.iter().enumerate()
    {
        if coef.abs() < 1e-15
        {
            continue;
        }
        let term = if deg == 0
        {
            SExpr::Const(coef)
        }
        else if deg == 1
        {
            SExpr::Mul(
                Box::new(SExpr::Const(coef)),
                Box::new(SExpr::Var(var.clone())),
            )
        }
        else
        {
            let pow = SExpr::Pow(
                Box::new(SExpr::Var(var.clone())),
                Box::new(SExpr::Const(deg as f64)),
            );
            SExpr::Mul(Box::new(SExpr::Const(coef)), Box::new(pow))
        };
        terms.push(term);
    }
    if terms.is_empty()
    {
        SExpr::Const(0.0)
    }
    else if terms.len() == 1
    {
        terms.into_iter().next().unwrap()
    }
    else
    {
        SExpr::Sum(terms)
    }
}

// ============================================================================
// Conversion to/from scirust-symbolic Expr
// ============================================================================

/// Convert SExpr to scirust-symbolic Expr (best-effort).
pub fn to_symbolic(expr: &SExpr) -> Option<scirust_symbolic::Expr> {
    use scirust_symbolic::Expr as S;
    match expr
    {
        SExpr::Var(v) => Some(S::Var(v.clone())),
        SExpr::Const(c) => Some(S::Const(*c)),
        SExpr::Neg(a) => Some(S::Neg(Box::new(to_symbolic(a)?))),
        SExpr::Sin(a) => Some(S::Sin(Box::new(to_symbolic(a)?))),
        SExpr::Cos(a) => Some(S::Cos(Box::new(to_symbolic(a)?))),
        SExpr::Exp(a) => Some(S::Exp(Box::new(to_symbolic(a)?))),
        SExpr::Ln(a) => Some(S::Ln(Box::new(to_symbolic(a)?))),
        SExpr::Sqrt(a) => Some(S::Sqrt(Box::new(to_symbolic(a)?))),
        SExpr::Abs(a) => Some(S::Abs(Box::new(to_symbolic(a)?))),
        SExpr::Add(a, b) => Some(S::Add(Box::new(to_symbolic(a)?), Box::new(to_symbolic(b)?))),
        SExpr::Sub(a, b) => Some(S::Sub(Box::new(to_symbolic(a)?), Box::new(to_symbolic(b)?))),
        SExpr::Mul(a, b) => Some(S::Mul(Box::new(to_symbolic(a)?), Box::new(to_symbolic(b)?))),
        SExpr::Div(a, b) => Some(S::Div(Box::new(to_symbolic(a)?), Box::new(to_symbolic(b)?))),
        SExpr::Pow(a, b) => Some(S::Pow(Box::new(to_symbolic(a)?), Box::new(to_symbolic(b)?))),
        SExpr::Square(a) => Some(S::Pow(Box::new(to_symbolic(a)?), Box::new(S::Const(2.0)))),
        SExpr::Tan(a) =>
        {
            let sa = to_symbolic(a)?;
            Some(S::Div(
                Box::new(S::Sin(Box::new(sa.clone()))),
                Box::new(S::Cos(Box::new(sa))),
            ))
        },
        _ => None,
    }
}

/// Convert scirust-symbolic Expr to SExpr.
pub fn from_symbolic(expr: &scirust_symbolic::Expr) -> SExpr {
    use scirust_symbolic::Expr as E;
    match expr
    {
        E::Const(c) => SExpr::Const(*c),
        E::Var(v) => SExpr::Var(v.clone()),
        E::Neg(a) => SExpr::Neg(Box::new(from_symbolic(a))),
        E::Sin(a) => SExpr::Sin(Box::new(from_symbolic(a))),
        E::Cos(a) => SExpr::Cos(Box::new(from_symbolic(a))),
        E::Exp(a) => SExpr::Exp(Box::new(from_symbolic(a))),
        E::Ln(a) => SExpr::Ln(Box::new(from_symbolic(a))),
        E::Sqrt(a) => SExpr::Sqrt(Box::new(from_symbolic(a))),
        E::Abs(a) => SExpr::Abs(Box::new(from_symbolic(a))),
        E::Add(a, b) => SExpr::Add(Box::new(from_symbolic(a)), Box::new(from_symbolic(b))),
        E::Sub(a, b) => SExpr::Sub(Box::new(from_symbolic(a)), Box::new(from_symbolic(b))),
        E::Mul(a, b) => SExpr::Mul(Box::new(from_symbolic(a)), Box::new(from_symbolic(b))),
        E::Div(a, b) => SExpr::Div(Box::new(from_symbolic(a)), Box::new(from_symbolic(b))),
        E::Pow(a, b) => SExpr::Pow(Box::new(from_symbolic(a)), Box::new(from_symbolic(b))),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_simple() {
        let x = SExpr::Var("x".into());
        let e = SExpr::Add(Box::new(x), Box::new(SExpr::Const(1.0)));
        assert_eq!(format!("{e}"), "(x + 1)");
    }

    #[test]
    fn test_display_pi() {
        assert_eq!(format!("{}", SExpr::Const(std::f64::consts::PI)), "pi");
    }

    #[test]
    fn test_display_if_else() {
        let x = SExpr::Var("x".into());
        let cmp = SExpr::Cmp {
            op: CmpOp::Gt,
            lhs: Box::new(x),
            rhs: Box::new(SExpr::Const(0.0)),
        };
        let e = SExpr::IfElse(
            Box::new(cmp),
            Box::new(SExpr::Const(1.0)),
            Box::new(SExpr::Neg(Box::new(SExpr::Const(1.0)))),
        );
        assert_eq!(format!("{e}"), "if((x > 0)) {1} else {(-1)}");
    }

    #[test]
    fn test_size() {
        let x = SExpr::Var("x".into());
        assert_eq!(x.size(), 1);
        let e = SExpr::Sin(Box::new(SExpr::Add(Box::new(x.clone()), Box::new(x))));
        assert_eq!(e.size(), 4);
    }

    #[test]
    fn test_depth() {
        let x = SExpr::Var("x".into());
        assert_eq!(x.depth(), 0);
        let e = SExpr::Sin(Box::new(SExpr::Add(Box::new(x.clone()), Box::new(x))));
        assert_eq!(e.depth(), 2);
    }

    #[test]
    fn test_free_vars() {
        let e = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Mul(
                Box::new(SExpr::Var("y".into())),
                Box::new(SExpr::Const(3.0)),
            )),
        );
        let vars = e.free_vars();
        assert!(vars.contains("x") && vars.contains("y") && vars.len() == 2);
    }

    #[test]
    fn test_eval_simple() {
        let e = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        let mut vars = HashMap::new();
        vars.insert("x".into(), 3.0);
        assert!((eval(&e, &vars).unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_trig() {
        let e = SExpr::Sin(Box::new(SExpr::Var("x".into())));
        let mut vars = HashMap::new();
        vars.insert("x".into(), std::f64::consts::PI / 2.0);
        assert!((eval(&e, &vars).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_hypot() {
        let e = SExpr::Hypot(Box::new(SExpr::Const(3.0)), Box::new(SExpr::Const(4.0)));
        assert!((eval(&e, &HashMap::new()).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_max_min() {
        let e = SExpr::Max(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(5.0)),
        );
        let mut vars = HashMap::new();
        vars.insert("x".into(), 3.0);
        assert!((eval(&e, &vars).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_if_else() {
        let x = SExpr::Var("x".into());
        let cmp = SExpr::Cmp {
            op: CmpOp::Gt,
            lhs: Box::new(x),
            rhs: Box::new(SExpr::Const(0.0)),
        };
        let e = SExpr::IfElse(
            Box::new(cmp),
            Box::new(SExpr::Const(1.0)),
            Box::new(SExpr::Const(-1.0)),
        );
        let mut vars = HashMap::new();
        vars.insert("x".into(), 5.0);
        assert!((eval(&e, &vars).unwrap() - 1.0).abs() < 1e-10);
        vars.insert("x".into(), -3.0);
        assert!((eval(&e, &vars).unwrap() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eval_sinpi_cospi() {
        let e = SExpr::SinPi(Box::new(SExpr::Const(0.5)));
        assert!((eval(&e, &HashMap::new()).unwrap() - 1.0).abs() < 1e-10);
        let e2 = SExpr::CosPi(Box::new(SExpr::Const(1.0)));
        assert!((eval(&e2, &HashMap::new()).unwrap() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eval_sign_square_cube() {
        let v = HashMap::new();
        assert!(
            (eval(&SExpr::Sign(Box::new(SExpr::Const(-7.0))), &v).unwrap() - (-1.0)).abs() < 1e-10
        );
        assert!(
            (eval(&SExpr::Square(Box::new(SExpr::Const(3.0))), &v).unwrap() - 9.0).abs() < 1e-10
        );
        assert!((eval(&SExpr::Cube(Box::new(SExpr::Const(2.0))), &v).unwrap() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_sum_prod() {
        let v = HashMap::new();
        let s = SExpr::Sum(vec![
            SExpr::Const(1.0),
            SExpr::Const(2.0),
            SExpr::Const(3.0),
        ]);
        assert!((eval(&s, &v).unwrap() - 6.0).abs() < 1e-10);
        let p = SExpr::Prod(vec![
            SExpr::Const(2.0),
            SExpr::Const(3.0),
            SExpr::Const(4.0),
        ]);
        assert!((eval(&p, &v).unwrap() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_multivariate() {
        let e = SExpr::Mul(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Var("y".into())),
        );
        let vars = bind_vars(&["x".into(), "y".into()], &[3.0, 4.0]);
        assert!((eval(&e, &vars).unwrap() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplify_x_plus_0() {
        let e = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(0.0)),
        );
        assert_eq!(simplify(&e), SExpr::Var("x".into()));
    }

    #[test]
    fn test_simplify_x_mul_1() {
        let e = SExpr::Mul(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        assert_eq!(simplify(&e), SExpr::Var("x".into()));
    }

    #[test]
    fn test_simplify_x_mul_0() {
        let e = SExpr::Mul(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(0.0)),
        );
        assert_eq!(simplify(&e), SExpr::Const(0.0));
    }

    #[test]
    fn test_simplify_x_sub_x() {
        let x = SExpr::Var("x".into());
        let e = SExpr::Sub(Box::new(x.clone()), Box::new(x));
        assert_eq!(simplify(&e), SExpr::Const(0.0));
    }

    #[test]
    fn test_simplify_pow_0() {
        let e = SExpr::Pow(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(0.0)),
        );
        assert_eq!(simplify(&e), SExpr::Const(1.0));
    }

    #[test]
    fn test_constant_folding() {
        let e = SExpr::Add(
            Box::new(SExpr::Const(2.0)),
            Box::new(SExpr::Mul(
                Box::new(SExpr::Const(3.0)),
                Box::new(SExpr::Const(4.0)),
            )),
        );
        assert_eq!(constant_fold(&e), SExpr::Const(14.0));
    }

    #[test]
    fn test_rewrite_double_neg() {
        let e = SExpr::Neg(Box::new(SExpr::Neg(Box::new(SExpr::Var("x".into())))));
        assert_eq!(rewrite(&e), SExpr::Var("x".into()));
    }

    #[test]
    fn test_expr_eq() {
        let a = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        let b = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn test_merkle_hash_different() {
        let a = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        let b = SExpr::Sub(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        assert_ne!(a.merkle_hash(), b.merkle_hash());
    }

    #[test]
    fn test_deduplication() {
        let mut set: HashSet<SExpr> = HashSet::new();
        let e = SExpr::Mul(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(2.0)),
        );
        set.insert(e.clone());
        set.insert(e.clone());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_sketch_hole_indices() {
        let t = SExpr::Add(Box::new(SExpr::Hole(0)), Box::new(SExpr::Hole(1)));
        let s = Sketch::new(t, vec!["x".into()]);
        assert_eq!(s.num_holes(), 2);
    }

    #[test]
    fn test_fill_holes() {
        let t = SExpr::Mul(Box::new(SExpr::Hole(0)), Box::new(SExpr::Hole(1)));
        let f = fill_holes(&t, &[SExpr::Const(2.0), SExpr::Var("x".into())]);
        assert_eq!(
            f,
            SExpr::Mul(
                Box::new(SExpr::Const(2.0)),
                Box::new(SExpr::Var("x".into()))
            )
        );
    }

    #[test]
    fn test_bottom_up_basic() {
        let data = vec![(vec![1.0], 2.0), (vec![2.0], 4.0), (vec![3.0], 6.0)];
        let r = bottom_up(&["x".into()], &data, 5, 100);
        assert!(!r.is_empty());
        let bm = r
            .iter()
            .map(|e| mse(e, &["x".into()], &data))
            .fold(f64::INFINITY, f64::min);
        assert!(bm < 1e-8);
    }

    #[test]
    fn test_mse_perfect() {
        let e = SExpr::Mul(
            Box::new(SExpr::Const(2.0)),
            Box::new(SExpr::Var("x".into())),
        );
        let data = vec![(vec![1.0], 2.0), (vec![2.0], 4.0), (vec![3.0], 6.0)];
        assert!(mse(&e, &["x".into()], &data) < 1e-15);
    }

    #[test]
    fn test_mse_imperfect() {
        let e = SExpr::Var("x".into());
        let data = vec![(vec![1.0], 2.0), (vec![2.0], 4.0)];
        assert!(mse(&e, &["x".into()], &data) > 1.0);
    }

    #[test]
    fn test_regularised_fitness() {
        let small = SExpr::Var("x".into());
        let big = SExpr::Add(
            Box::new(SExpr::Sin(Box::new(SExpr::Var("x".into())))),
            Box::new(SExpr::Cos(Box::new(SExpr::Var("x".into())))),
        );
        let d = vec![(vec![1.0], 2.0)];
        assert!(
            regularised_fitness(&big, &["x".into()], &d, 10.0)
                > regularised_fitness(&small, &["x".into()], &d, 10.0)
        );
    }

    #[test]
    fn test_cost_increases_with_size() {
        assert!(
            cost(&SExpr::Sin(Box::new(SExpr::Var("x".into()))), 0.1)
                > cost(&SExpr::Var("x".into()), 0.1)
        );
    }

    #[test]
    fn test_op_weights_trig_costlier_than_add() {
        let add = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        let sin = SExpr::Sin(Box::new(SExpr::Var("x".into())));
        assert!(op_weight(&sin) > op_weight(&add));
    }

    #[test]
    fn test_train_test_split() {
        let data: Vec<(Vec<f64>, f64)> = (0..10).map(|i| (vec![i as f64], i as f64)).collect();
        let (train, test) = train_test_split(&data);
        assert!(!train.is_empty() && !test.is_empty() && train.len() + test.len() == data.len());
    }

    #[test]
    fn test_extrapolation_check_finite() {
        let data = vec![(vec![1.0], 1.0), (vec![2.0], 2.0)];
        let pen = extrapolation_check(&SExpr::Var("x".into()), &["x".into()], &data, 1.0, 5);
        assert!(pen.is_finite());
    }

    #[test]
    fn test_occam_penalty() {
        let large = SExpr::Add(
            Box::new(SExpr::Add(
                Box::new(SExpr::Var("x".into())),
                Box::new(SExpr::Var("x".into())),
            )),
            Box::new(SExpr::Var("x".into())),
        );
        assert!(occam_penalty(&large, 1.0) > occam_penalty(&SExpr::Var("x".into()), 1.0));
    }

    #[test]
    fn test_domain_bias_prefers_polynomial() {
        let poly = SExpr::Add(
            Box::new(SExpr::Mul(
                Box::new(SExpr::Const(2.0)),
                Box::new(SExpr::Var("x".into())),
            )),
            Box::new(SExpr::Const(1.0)),
        );
        let trig = SExpr::Sin(Box::new(SExpr::Var("x".into())));
        let d = vec![(vec![0.0], 1.0)];
        assert!(domain_bias(&trig, &d) > domain_bias(&poly, &d));
    }

    #[test]
    fn test_incremental_synthesis() {
        let data = vec![(vec![1.0], 2.0), (vec![2.0], 4.0), (vec![3.0], 6.0)];
        let r = incremental_synthesis(&["x".into()], &data, 1, 1, 6, 0.001);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_try_polynomial_fit() {
        let data = vec![
            (vec![0.0], 1.0),
            (vec![1.0], 3.0),
            (vec![2.0], 5.0),
            (vec![3.0], 7.0),
        ];
        let e = try_polynomial_fit(&["x".into()], &data, 3);
        assert!(e.is_some() && mse(&e.unwrap(), &["x".into()], &data) < 1e-6);
    }

    #[test]
    fn test_roundtrip_symbolic() {
        let e = SExpr::Add(
            Box::new(SExpr::Var("x".into())),
            Box::new(SExpr::Const(1.0)),
        );
        let sym = to_symbolic(&e).unwrap();
        assert_eq!(e, from_symbolic(&sym));
    }

    #[test]
    fn test_cse_finds_duplicates() {
        let x = SExpr::Var("x".into());
        let xx = SExpr::Add(Box::new(x.clone()), Box::new(x));
        let expr = SExpr::Add(Box::new(xx.clone()), Box::new(xx));
        let dups = cse_find_duplicates(&expr);
        assert!(!dups.is_empty() && dups.iter().any(|(_, c)| *c > 1));
    }

    #[test]
    fn test_serde_roundtrip() {
        let e = SExpr::IfElse(
            Box::new(SExpr::Cmp {
                op: CmpOp::Gt,
                lhs: Box::new(SExpr::Var("x".into())),
                rhs: Box::new(SExpr::Const(0.0)),
            }),
            Box::new(SExpr::Sin(Box::new(SExpr::Var("x".into())))),
            Box::new(SExpr::Cos(Box::new(SExpr::Var("x".into())))),
        );
        let json = serde_json::to_string(&e).unwrap();
        let back: SExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn test_sketch_serde_roundtrip() {
        let t = SExpr::Add(
            Box::new(SExpr::Hole(0)),
            Box::new(SExpr::Mul(
                Box::new(SExpr::Hole(1)),
                Box::new(SExpr::Var("x".into())),
            )),
        );
        let s = Sketch::new(t, vec!["x".into(), "y".into()]);
        let json = serde_json::to_string(&s).unwrap();
        let back: Sketch = serde_json::from_str(&json).unwrap();
        assert_eq!(s.template, back.template);
        assert_eq!(s.variables, back.variables);
    }

    #[test]
    fn test_atan2_eval() {
        let e = SExpr::Atan2(Box::new(SExpr::Const(1.0)), Box::new(SExpr::Const(0.0)));
        assert!((eval(&e, &HashMap::new()).unwrap() - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplify_square_of_sqrt() {
        let x = SExpr::Var("x".into());
        let e = SExpr::Square(Box::new(SExpr::Sqrt(Box::new(x.clone()))));
        assert_eq!(simplify(&e), x);
    }

    #[test]
    fn test_x_div_x_is_1() {
        let x = SExpr::Var("x".into());
        let e = SExpr::Div(Box::new(x.clone()), Box::new(x));
        assert_eq!(simplify(&e), SExpr::Const(1.0));
    }
}
