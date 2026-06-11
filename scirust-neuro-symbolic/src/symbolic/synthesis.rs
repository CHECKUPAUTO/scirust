use crate::core::{Reasoner, ReasoningError, Result};
use scirust_symbolic::Expr;
use std::collections::HashMap;

/// Program (expression) synthesis.
///
/// * [`synthesize_from_examples`](Self::synthesize_from_examples) performs a real
///   bottom-up **enumerative search** over the grammar
///   `{x, small constants} × {+, -, *}` and returns an expression reproducing the
///   given input/output pairs.
/// * [`synthesize`](Self::synthesize) parses a textual specification into an
///   [`Expr`] (honest pass-through; returns a parse error rather than a dummy).
pub struct ProgramSynthesis {
    timeout_ms: u64,
}

impl ProgramSynthesis {
    pub fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }

    /// Parse a specification string (e.g. `"x*x + 1"`) into an expression.
    pub fn synthesize(&self, spec: &str) -> Result<Expr> {
        scirust_symbolic::parse(spec)
            .map(|e| scirust_symbolic::simplify(&e))
            .map_err(ReasoningError::Symbolic)
    }

    /// Enumerative synthesis: find an expression in `x` matching the examples.
    pub fn synthesize_from_examples(&self, xs: &[f64], ys: &[f64]) -> Result<Expr> {
        if xs.is_empty() || xs.len() != ys.len()
        {
            return Err(ReasoningError::Symbolic(
                "empty or mismatched examples".into(),
            ));
        }
        let tol = 1e-6;
        let matches = |e: &Expr| -> bool {
            xs.iter().zip(ys).all(|(&xi, &yi)| {
                let mut m = HashMap::new();
                m.insert("x".to_string(), xi);
                scirust_symbolic::eval(e, &m)
                    .map(|v| (v - yi).abs() < tol)
                    .unwrap_or(false)
            })
        };

        let mut pool: Vec<Expr> = vec![
            Expr::Var("x".into()),
            Expr::Const(0.0),
            Expr::Const(1.0),
            Expr::Const(2.0),
            Expr::Const(-1.0),
        ];
        for e in &pool
        {
            if matches(e)
            {
                return Ok(e.clone());
            }
        }

        let max_pool = (self.timeout_ms as usize).clamp(64, 2000);
        for _ in 0..3
        {
            let current = pool.clone();
            for i in 0..current.len()
            {
                for j in 0..current.len()
                {
                    for cand in [
                        Expr::Add(Box::new(current[i].clone()), Box::new(current[j].clone())),
                        Expr::Sub(Box::new(current[i].clone()), Box::new(current[j].clone())),
                        Expr::Mul(Box::new(current[i].clone()), Box::new(current[j].clone())),
                    ]
                    {
                        if matches(&cand)
                        {
                            return Ok(scirust_symbolic::simplify(&cand));
                        }
                        if pool.len() < max_pool
                        {
                            pool.push(cand);
                        }
                    }
                }
                if pool.len() >= max_pool
                {
                    break;
                }
            }
        }
        Err(ReasoningError::Symbolic(
            "no expression found within search budget".into(),
        ))
    }
}

impl Reasoner for ProgramSynthesis {
    fn name(&self) -> &str {
        "ProgramSynthesis"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesizes_quadratic_from_examples() {
        // y = x*x + 1
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 2.0, 5.0, 10.0];
        let ps = ProgramSynthesis::new(1000);
        let expr = ps.synthesize_from_examples(&xs, &ys).unwrap();
        // verify it reproduces the data
        for (&x, &y) in xs.iter().zip(&ys)
        {
            let mut m = std::collections::HashMap::new();
            m.insert("x".to_string(), x);
            assert!((scirust_symbolic::eval(&expr, &m).unwrap() - y).abs() < 1e-6);
        }
    }

    #[test]
    fn synthesize_parses_spec() {
        let ps = ProgramSynthesis::new(1000);
        assert!(ps.synthesize("x + 1").is_ok());
        assert!(ps.synthesize("@@@not valid@@@").is_err());
    }
}
