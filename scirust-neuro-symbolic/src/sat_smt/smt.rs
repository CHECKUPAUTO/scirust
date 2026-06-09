use crate::core::{Reasoner, ReasoningError, Result};
use std::collections::HashMap;

pub enum SmtExpr {
    Const(f64),
    Var(String),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Add(Box<SmtExpr>, Box<SmtExpr>),
}

/// A decision procedure for the **linear-arithmetic equality fragment**
/// (`Const`, `Var`, `Add`, `Eq`). A single top-level `Eq` constraint
/// `lhs = rhs` is decided exactly by linearising both sides: it is satisfiable
/// unless it reduces to `c1 = c2` with `c1 != c2` (i.e. all variable
/// coefficients cancel and the constants disagree).
///
/// This is intentionally narrow — it is *not* a general SMT solver — but it is a
/// real, sound decision for the fragment it supports rather than a hardcoded
/// answer. Unsupported shapes return an explicit error.
pub struct SmtInterface {
    pub solver_path: Option<String>,
}

/// Linear form: variable coefficients plus a constant term.
struct Linear {
    coeffs: HashMap<String, f64>,
    constant: f64,
}

fn linearize(expr: &SmtExpr) -> Result<Linear> {
    match expr {
        SmtExpr::Const(c) => Ok(Linear {
            coeffs: HashMap::new(),
            constant: *c,
        }),
        SmtExpr::Var(v) => {
            let mut coeffs = HashMap::new();
            coeffs.insert(v.clone(), 1.0);
            Ok(Linear {
                coeffs,
                constant: 0.0,
            })
        }
        SmtExpr::Add(a, b) => {
            let mut la = linearize(a)?;
            let lb = linearize(b)?;
            for (k, v) in lb.coeffs {
                *la.coeffs.entry(k).or_insert(0.0) += v;
            }
            la.constant += lb.constant;
            Ok(la)
        }
        SmtExpr::Eq(_, _) => Err(ReasoningError::Solver(
            "nested equalities are not supported".into(),
        )),
    }
}

impl SmtInterface {
    pub fn new(path: Option<String>) -> Self {
        Self { solver_path: path }
    }

    /// Decides satisfiability of a single linear equality constraint.
    pub fn check_sat(&self, expr: &SmtExpr) -> Result<bool> {
        let (lhs, rhs) = match expr {
            SmtExpr::Eq(a, b) => (a, b),
            _ => {
                return Err(ReasoningError::Solver(
                    "check_sat expects a top-level equality (Eq)".into(),
                ))
            }
        };
        let la = linearize(lhs)?;
        let lb = linearize(rhs)?;

        // Move everything to one side: (la - lb) = 0.
        let mut coeffs = la.coeffs;
        for (k, v) in lb.coeffs {
            *coeffs.entry(k).or_insert(0.0) -= v;
        }
        let rhs_const = lb.constant - la.constant;

        let all_zero = coeffs.values().all(|c| c.abs() < 1e-12);
        if all_zero {
            // 0 = rhs_const ⇒ satisfiable iff rhs_const == 0
            Ok(rhs_const.abs() < 1e-12)
        } else {
            // At least one free variable ⇒ always satisfiable.
            Ok(true)
        }
    }
}

impl Reasoner for SmtInterface {
    fn name(&self) -> &str {
        "SmtInterface"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: &str) -> Box<SmtExpr> {
        Box::new(SmtExpr::Var(n.to_string()))
    }
    fn cst(c: f64) -> Box<SmtExpr> {
        Box::new(SmtExpr::Const(c))
    }

    #[test]
    fn satisfiable_equation_with_variable() {
        // x + 1 = 1  ⇒  x = 0  (SAT)
        let smt = SmtInterface::new(None);
        let e = SmtExpr::Eq(Box::new(SmtExpr::Add(var("x"), cst(1.0))), cst(1.0));
        assert!(smt.check_sat(&e).unwrap());
    }

    #[test]
    fn contradictory_constants_are_unsat() {
        // 1 = 2 ⇒ UNSAT
        let smt = SmtInterface::new(None);
        let e = SmtExpr::Eq(cst(1.0), cst(2.0));
        assert!(!smt.check_sat(&e).unwrap());
    }

    #[test]
    fn cancelling_variables_reduce_to_constants() {
        // x = x + 3  ⇒ 0 = 3 ⇒ UNSAT
        let smt = SmtInterface::new(None);
        let e = SmtExpr::Eq(var("x"), Box::new(SmtExpr::Add(var("x"), cst(3.0))));
        assert!(!smt.check_sat(&e).unwrap());
    }

    #[test]
    fn non_equality_is_rejected_honestly() {
        let smt = SmtInterface::new(None);
        assert!(smt.check_sat(&SmtExpr::Const(1.0)).is_err());
    }
}
