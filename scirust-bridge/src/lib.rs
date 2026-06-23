//! IA Bridge: a natural-language → symbolic-math pipeline.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
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
    let lower = text.to_lowercase();
    let (action, expr) = if lower.contains("solve") || lower.contains("résoudre")
    {
        ("solve".to_string(), Some(text.to_string()))
    }
    else if lower.contains("derive") || lower.contains("dérivée")
    {
        ("derive".to_string(), Some(text.to_string()))
    }
    else if lower.contains("simplify") || lower.contains("simplifie")
    {
        ("simplify".to_string(), Some(text.to_string()))
    }
    else if lower.contains("eval") || lower.contains("évalue") || lower.contains("calcule")
    {
        ("evaluate".to_string(), Some(text.to_string()))
    }
    else
    {
        ("unknown".to_string(), Some(text.to_string()))
    };
    NaturalCommand {
        action,
        expression: expr,
        variables: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_runs_a_full_nlp_to_math_pass() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let out = Pipeline::new().with_vars(vars).run("x^2 + 1").unwrap();
        assert_eq!(out.value, 10.0);
        assert!(!out.parsed.is_empty());
        assert!(!out.derivative.is_empty());
        assert!(!out.rust_code.is_empty());
    }

    #[test]
    fn parse_natural_classifies_intent() {
        assert_eq!(parse_natural("please simplify x + x").action, "simplify");
        assert_eq!(parse_natural("derive x^2").action, "derive");
        assert_eq!(parse_natural("solve x - 1").action, "solve");
        assert_eq!(parse_natural("evaluate x").action, "evaluate");
        assert_eq!(parse_natural("hello there").action, "unknown");
    }
}
