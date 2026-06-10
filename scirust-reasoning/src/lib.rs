//! Reasoning engine — equality proving, trig identities, numeric optimization.

use std::collections::HashMap;
use std::collections::HashSet;

/// Numerical optimizer using gradient descent.
#[derive(Debug, Clone)]
pub struct Optimizer {
    learning_rate: f64,
    max_iterations: usize,
    tolerance: f64,
}

impl Optimizer {
    /// Create a new optimizer.
    ///
    /// * `learning_rate` — step size for each gradient descent iteration.
    /// * `max_iterations` — maximum number of iterations before stopping.
    /// * `tolerance` — convergence threshold (stop when |x_new - x| < tolerance).
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance,
        }
    }

    /// Minimize a 1-D function using gradient descent with finite-difference gradients.
    ///
    /// Returns the approximate minimizer `x*`.
    pub fn minimize<F: Fn(f64) -> f64>(&self, f: F, x0: f64) -> f64 {
        let h = 1e-8;
        let mut x = x0;

        for _ in 0..self.max_iterations
        {
            // Central finite-difference gradient
            let grad = (f(x + h) - f(x - h)) / (2.0 * h);
            let x_new = x - self.learning_rate * grad;
            if (x_new - x).abs() < self.tolerance
            {
                return x_new;
            }
            x = x_new;
        }

        x
    }
}

/// Solve a linear equation `a*x + b = 0`.
pub fn solve_linear(a: f64, b: f64) -> Result<f64, String> {
    if a == 0.0
    {
        Err("no solution: a == 0".into())
    }
    else
    {
        Ok(-b / a)
    }
}

/// Solve a quadratic equation `a*x^2 + b*x + c = 0`.
///
/// Returns the two real roots.  Errors if `a == 0` or the discriminant is negative.
pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Result<(f64, f64), String> {
    if a == 0.0
    {
        return Err("a must be non-zero".into());
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0
    {
        return Err("no real solutions".into());
    }
    let sqrt_disc = disc.sqrt();
    Ok(((-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)))
}

/// Prove that two expression strings are approximately equal by parsing and
/// evaluating them at several random points.
///
/// Returns `true` if all sampled evaluations agree within `1e-8`.
pub fn prove_equal(a: &str, b: &str) -> bool {
    let expr_a = match scirust_symbolic::parse(a)
    {
        Ok(e) => e,
        Err(_) => return false,
    };
    let expr_b = match scirust_symbolic::parse(b)
    {
        Ok(e) => e,
        Err(_) => return false,
    };

    // Collect variable names from both expressions.
    let mut vars_set = HashSet::new();
    collect_vars(&expr_a, &mut vars_set);
    collect_vars(&expr_b, &mut vars_set);
    let vars: Vec<&String> = vars_set.iter().collect();

    if vars.is_empty()
    {
        // Both are constant expressions — evaluate once and compare.
        let empty_map = HashMap::new();
        match (
            scirust_symbolic::eval(&expr_a, &empty_map),
            scirust_symbolic::eval(&expr_b, &empty_map),
        )
        {
            (Ok(va), Ok(vb)) => return (va - vb).abs() < 1e-8,
            _ => return false,
        }
    }

    for i in 0..20
    {
        let mut bindings = HashMap::new();
        for (j, v) in vars.iter().enumerate()
        {
            let val = ((i * 7919 + j * 6271 + 127) as f64 / 1000.0) % 20.0 - 10.0;
            bindings.insert((*v).clone(), val);
        }

        match (
            scirust_symbolic::eval(&expr_a, &bindings),
            scirust_symbolic::eval(&expr_b, &bindings),
        )
        {
            (Ok(va), Ok(vb)) =>
            {
                if (va - vb).abs() > 1e-8
                {
                    return false;
                }
            },
            _ => return false,
        }
    }

    true
}

/// Recursively collect all variable names from an expression tree.
fn collect_vars(expr: &scirust_symbolic::Expr, vars: &mut HashSet<String>) {
    use scirust_symbolic::Expr;
    match expr
    {
        Expr::Var(v) =>
        {
            vars.insert(v.clone());
        },
        Expr::Const(_) =>
        {},
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) =>
        {
            collect_vars(a, vars);
            collect_vars(b, vars);
        },
        Expr::Neg(a)
        | Expr::Sin(a)
        | Expr::Cos(a)
        | Expr::Exp(a)
        | Expr::Ln(a)
        | Expr::Sqrt(a)
        | Expr::Abs(a) =>
        {
            collect_vars(a, vars);
        },
    }
}

/// Apply trigonometric identity transformations to an expression string.
///
/// Implemented identities:
/// - `sin(...)^2`  →  `(1 - cos(2 * ...)) / 2`
/// - `cos(...)^2`  →  `(1 + cos(2 * ...)) / 2`
///
/// Matching is performed via a single pass over the input string with
/// parenthesis-depth tracking, so nested parentheses inside the function
/// argument are handled correctly.
pub fn apply_trig_identity(expr: &str) -> String {
    let bytes = expr.as_bytes();
    let n = bytes.len();
    let mut result = String::with_capacity(n);

    let mut i = 0;
    while i < n
    {
        // Check for "sin("
        if i + 4 <= n && &bytes[i..i + 4] == b"sin("
        {
            if let Some(close) = matching_paren(expr, i + 3)
            {
                if close + 2 < n && &bytes[close + 1..close + 3] == b"^2"
                {
                    let inner = &expr[i + 4..close];
                    result.push_str("(1-cos(2*");
                    result.push_str(inner);
                    result.push_str("))/2");
                    i = close + 3;
                    continue;
                }
            }
        }
        // Check for "cos("
        if i + 4 <= n && &bytes[i..i + 4] == b"cos("
        {
            if let Some(close) = matching_paren(expr, i + 3)
            {
                if close + 2 < n && &bytes[close + 1..close + 3] == b"^2"
                {
                    let inner = &expr[i + 4..close];
                    result.push_str("(1+cos(2*");
                    result.push_str(inner);
                    result.push_str("))/2");
                    i = close + 3;
                    continue;
                }
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

/// Find the matching closing parenthesis for the opening paren at `open_idx`.
fn matching_paren(s: &str, open_idx: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    if open_idx >= bytes.len() || bytes[open_idx] != b'('
    {
        return None;
    }
    let mut depth: u32 = 1;
    let mut i = open_idx + 1;
    while i < bytes.len() && depth > 0
    {
        match bytes[i]
        {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ =>
            {},
        }
        if depth > 0
        {
            i += 1;
        }
    }
    if depth == 0 { Some(i) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Optimizer ──

    #[test]
    fn test_optimizer_minimize_quadratic() {
        // f(x) = x^2, minimum at x = 0
        let opt = Optimizer::new(0.1, 1000, 1e-6);
        let x_min = opt.minimize(|x| x * x, 3.0);
        assert!(x_min.abs() < 1e-3);
    }

    #[test]
    fn test_optimizer_minimize_linear_combined() {
        // f(x) = (x + 3)^2, minimum at x = -3
        let opt = Optimizer::new(0.1, 1000, 1e-6);
        let x_min = opt.minimize(|x| (x + 3.0).powi(2), 1.0);
        assert!((x_min + 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_optimizer_converges_to_tolerance() {
        // f(x) = x^2, stop early due to tolerance
        let opt = Optimizer::new(0.5, 100_000, 1e-4);
        let x_min = opt.minimize(|x| x * x, 10.0);
        assert!(x_min.abs() < 1e-3);
    }

    // ── prove_equal ──

    #[test]
    fn test_prove_equal_same() {
        assert!(prove_equal("x + 1", "x + 1"));
    }

    #[test]
    fn test_prove_equal_equivalent() {
        // (x+1)^2 == x^2 + 2*x + 1
        assert!(prove_equal("(x+1)^2", "x^2 + 2*x + 1"));
    }

    #[test]
    fn test_prove_equal_not_equivalent() {
        assert!(!prove_equal("x + 1", "x + 2"));
    }

    #[test]
    fn test_prove_equal_constants() {
        assert!(prove_equal("2 + 3", "5"));
        assert!(!prove_equal("2 + 3", "6"));
    }

    #[test]
    fn test_prove_equal_unparseable() {
        assert!(!prove_equal("sin((", "x"));
    }

    // ── apply_trig_identity ──

    #[test]
    fn test_trig_sin_squared() {
        let result = apply_trig_identity("sin(x)^2");
        assert_eq!(result, "(1-cos(2*x))/2");
    }

    #[test]
    fn test_trig_cos_squared() {
        let result = apply_trig_identity("cos(x)^2");
        assert_eq!(result, "(1+cos(2*x))/2");
    }

    #[test]
    fn test_trig_no_change() {
        let result = apply_trig_identity("sin(x) + 1");
        assert_eq!(result, "sin(x) + 1");
    }

    #[test]
    fn test_trig_with_coeff() {
        let result = apply_trig_identity("2*sin(y)^2");
        assert_eq!(result, "2*(1-cos(2*y))/2");
    }

    // ── solve_linear / solve_quadratic (kept from original) ──

    #[test]
    fn test_solve_linear_basic() {
        let x = solve_linear(2.0, 4.0).unwrap();
        assert!((x - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_no_solution() {
        assert!(solve_linear(0.0, 5.0).is_err());
    }

    #[test]
    fn test_solve_quadratic_basic() {
        let (r1, r2) = solve_quadratic(1.0, 0.0, -4.0).unwrap();
        assert!((r1 - 2.0).abs() < 1e-10);
        assert!((r2 + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_quadratic_no_real() {
        assert!(solve_quadratic(1.0, 0.0, 1.0).is_err());
    }
}
