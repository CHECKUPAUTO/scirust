use crate::core::{Reasoner, Result};

pub enum SmtExpr {
    Const(f64),
    Var(String),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Add(Box<SmtExpr>, Box<SmtExpr>),
}

pub struct SmtInterface {
    pub solver_path: Option<String>,
}

impl SmtInterface {
    pub fn new(path: Option<String>) -> Self {
        Self { solver_path: path }
    }

    pub fn check_sat(&self, _expr: &SmtExpr) -> Result<bool> {
        // FFI or process call to Z3/cvc5
        Ok(true)
    }
}

impl Reasoner for SmtInterface {
    fn name(&self) -> &str {
        "SmtInterface"
    }
}
