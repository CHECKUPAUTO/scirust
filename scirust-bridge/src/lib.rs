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

    /// Every field of `PipelineOutput` is checked against values derived by
    /// hand from the source expression `x^2 + 1` with `x = 3`:
    ///   parsed     : Display of `Add(Pow(Var x, 2), 1)`      = "((x^2) + 1)"
    ///   simplified : already in lowest terms                 = "((x^2) + 1)"
    ///   derivative : d/dx of the UNSIMPLIFIED expr,
    ///                2·x^(2-1) + 0                            = "(2 * (x^1))"
    ///   value      : 3^2 + 1                                 = 10
    ///   rust_code  : evaluable Rust source                   = "((x).powf(2) + 1)"
    #[test]
    fn pipeline_emits_exact_fields_for_quadratic() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let out = Pipeline::new().with_vars(vars).run("x^2 + 1").unwrap();
        assert_eq!(out.parsed, "((x^2) + 1)");
        assert_eq!(out.simplified, "((x^2) + 1)");
        assert_eq!(out.derivative, "(2 * (x^1))");
        assert_eq!(out.value, 10.0);
        assert_eq!(out.rust_code, "((x).powf(2) + 1)");
    }

    /// d/dx (2x + 1) = 2 (a folded constant), and 2·5 + 1 = 11.
    #[test]
    fn pipeline_emits_exact_fields_for_linear() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        let out = Pipeline::new().with_vars(vars).run("2*x + 1").unwrap();
        assert_eq!(out.parsed, "((2 * x) + 1)");
        assert_eq!(out.simplified, "((2 * x) + 1)");
        assert_eq!(out.derivative, "2");
        assert_eq!(out.value, 11.0);
        assert_eq!(out.rust_code, "((2 * x) + 1)");
    }

    /// A non-polynomial expression: d/dx sin(x) = cos(x)·1, sin(0) = 0.
    #[test]
    fn pipeline_handles_transcendental() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let out = Pipeline::new().with_vars(vars).run("sin(x)").unwrap();
        assert_eq!(out.parsed, "sin(x)");
        assert_eq!(out.derivative, "(cos(x) * 1)");
        assert_eq!(out.value, 0.0);
        assert_eq!(out.rust_code, "(x).sin()");
    }

    /// The bridge must propagate evaluation errors, not silently yield a
    /// placeholder value: an undefined variable is an `Err`.
    #[test]
    fn pipeline_propagates_eval_errors() {
        let err = Pipeline::new().run("y + 1").unwrap_err();
        assert_eq!(err, "Undefined variable: y");
    }

    /// A bad parse is surfaced as an error rather than swallowed.
    #[test]
    fn pipeline_propagates_parse_errors() {
        assert!(Pipeline::new().run("x +").is_err());
        assert!(Pipeline::new().run("x @ 1").is_err());
    }

    /// `with_vars` actually feeds the bindings into evaluation: the same
    /// expression yields different values for different `x`.
    #[test]
    fn with_vars_binds_the_evaluation_point() {
        let run_at = |x: f64| {
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), x);
            Pipeline::new().with_vars(vars).run("x^2").unwrap().value
        };
        assert_eq!(run_at(2.0), 4.0);
        assert_eq!(run_at(4.0), 16.0);
        assert_eq!(run_at(-3.0), 9.0);
    }

    /// Round-trip across the JSON "external representation": serialising a
    /// `PipelineOutput` and parsing it back reproduces every field exactly,
    /// and the on-the-wire JSON matches the expected byte string.
    #[test]
    fn pipeline_output_json_round_trips_exactly() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let out = Pipeline::new().with_vars(vars).run("x^2 + 1").unwrap();

        let json = serde_json::to_string(&out).unwrap();
        assert_eq!(
            json,
            r#"{"parsed":"((x^2) + 1)","simplified":"((x^2) + 1)","derivative":"(2 * (x^1))","value":10.0,"rust_code":"((x).powf(2) + 1)"}"#
        );

        let back: PipelineOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.parsed, out.parsed);
        assert_eq!(back.simplified, out.simplified);
        assert_eq!(back.derivative, out.derivative);
        assert_eq!(back.value, out.value);
        assert_eq!(back.rust_code, out.rust_code);
    }

    #[test]
    fn parse_natural_classifies_intent() {
        assert_eq!(parse_natural("please simplify x + x").action, "simplify");
        assert_eq!(parse_natural("derive x^2").action, "derive");
        assert_eq!(parse_natural("solve x - 1").action, "solve");
        assert_eq!(parse_natural("evaluate x").action, "evaluate");
        assert_eq!(parse_natural("hello there").action, "unknown");
    }

    /// French keywords map to the same intents as their English counterparts.
    #[test]
    fn parse_natural_recognises_french_keywords() {
        assert_eq!(parse_natural("résoudre x - 1").action, "solve");
        assert_eq!(parse_natural("dérivée de x^2").action, "derive");
        assert_eq!(parse_natural("simplifie x + x").action, "simplify");
        assert_eq!(parse_natural("calcule x + 1").action, "evaluate");
        assert_eq!(parse_natural("évalue x").action, "evaluate");
    }

    /// The classifier preserves the original text verbatim in `expression`
    /// (it is a thin tagger; it does not extract a sub-expression) and never
    /// fabricates variable bindings.
    #[test]
    fn parse_natural_preserves_text_and_leaves_vars_unset() {
        let cmd = parse_natural("please simplify x + x");
        assert_eq!(cmd.action, "simplify");
        assert_eq!(cmd.expression.as_deref(), Some("please simplify x + x"));
        assert!(cmd.variables.is_none());
    }

    /// Round-trip a `NaturalCommand` (including populated variable bindings)
    /// through JSON: structure and contents survive intact.
    #[test]
    fn natural_command_json_round_trips_exactly() {
        let mut vars = HashMap::new();
        vars.insert("a".to_string(), 2.5);
        let cmd = NaturalCommand {
            action: "evaluate".to_string(),
            expression: Some("a + 1".to_string()),
            variables: Some(vars),
        };

        let json = serde_json::to_string(&cmd).unwrap();
        let back: NaturalCommand = serde_json::from_str(&json).unwrap();

        assert_eq!(back.action, "evaluate");
        assert_eq!(back.expression.as_deref(), Some("a + 1"));
        let bound = back.variables.expect("variables must survive round-trip");
        assert_eq!(bound.len(), 1);
        assert_eq!(bound.get("a"), Some(&2.5));
    }
}
