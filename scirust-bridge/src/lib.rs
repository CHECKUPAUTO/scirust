//! IA Bridge (NLP → math pipeline) stub.
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use scirust_symbolic::Expr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub parsed: String,
    pub simplified: String,
    pub derivative: String,
    pub value: f64,
    pub rust_code: String,
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    vars: HashMap<String, f64>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { vars: HashMap::new() }
    }

    pub fn with_vars(mut self, vars: HashMap<String, f64>) -> Self {
        self.vars = vars;
        self
    }

    pub fn run(&self, expr_str: &str) -> Result<PipelineOutput, String> {
        let expr = scirust_symbolic::parse(expr_str)?;
        let value = scirust_symbolic::eval(&expr, &self.vars)?;
        Ok(PipelineOutput {
            parsed: format!("{}", expr),
            simplified: format!("{}", scirust_symbolic::simplify(&expr)),
            derivative: format!("{}", scirust_symbolic::diff(&expr, "x")),
            value,
            rust_code: scirust_symbolic::to_rust_code(&expr),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalCommand {
    pub action: String,
    pub expression: Option<String>,
    pub variables: Option<HashMap<String, f64>>,
}

pub fn parse_natural(text: &str) -> NaturalCommand {
    NaturalCommand {
        action: text.to_string(),
        expression: None,
        variables: None,
    }
}
