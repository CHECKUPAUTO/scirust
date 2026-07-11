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
/// Returns the two real roots (first the "+√disc" root, then the "−√disc"
/// root — the same order the naive formula would give). Errors if `a == 0`
/// or the discriminant is negative.
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
    // Stable form (Numerical Recipes §5.6; Goldberg 1991, "What Every
    // Computer Scientist Should Know About Floating-Point Arithmetic"):
    // computing q = -½(b + sign(b)·√disc) — always a sum of same-signed
    // terms — and then the roots q/a and c/q (Vieta) avoids the
    // catastrophic cancellation of the naive (-b ± √disc)/(2a) when
    // b² ≫ 4ac (well-separated roots).
    if sqrt_disc.abs() < 1e-300 && b.abs() < 1e-300
    {
        // q would be 0 (disc == 0 and b == 0, i.e. a repeated root at 0).
        let x = -b / (2.0 * a);
        return Ok((x, x));
    }
    let sign_b = if b < 0.0 { -1.0 } else { 1.0 };
    let q = -0.5 * (b + sign_b * sqrt_disc);
    // q = a·root₊ when b < 0, or a·root₋ when b ≥ 0 (Vieta: root₊·root₋ = c/a
    // pins down the other root as c/q either way).
    Ok(
        if b < 0.0
        {
            (q / a, c / q)
        }
        else
        {
            (c / q, q / a)
        },
    )
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

    // Sample on a larger deterministic grid than before. A denser, more varied
    // set of points makes accidental agreement (false positives) far less likely
    // than the original 20-point grid, while remaining fully deterministic.
    //
    // A sample point where *either* side fails to evaluate (division by zero,
    // ln/sqrt of a non-positive value, …) is outside the common domain, so it is
    // not evidence of inequality — such points are skipped rather than treated as
    // a disproof. To avoid vacuously "proving" equality when the two expressions
    // never share a defined point, we require a minimum number of agreeing
    // evaluations before returning `true`.
    const SAMPLES: usize = 200;
    const MIN_AGREEING: usize = 8;
    let mut agreeing = 0usize;

    for i in 0..SAMPLES
    {
        let mut bindings = HashMap::new();
        for (j, v) in vars.iter().enumerate()
        {
            // Spread the samples across a wide range that includes the positive
            // half-line (so domain-restricted expressions like ln/sqrt still get
            // evaluated) as well as negative and near-zero values.
            let raw = (i * 7919 + j * 6271 + 127) as f64;
            let val = (raw / 733.0) % 40.0 - 15.0;
            bindings.insert((*v).clone(), val);
        }

        // When at least one side is undefined — outside the common domain — skip
        // the point instead of concluding inequality (hence `if let`, not `match`).
        if let (Ok(va), Ok(vb)) = (
            scirust_symbolic::eval(&expr_a, &bindings),
            scirust_symbolic::eval(&expr_b, &bindings),
        )
        {
            if (va - vb).abs() > 1e-8
            {
                return false;
            }
            agreeing += 1;
        }
    }

    agreeing >= MIN_AGREEING
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
        // No identity matched at `i`. Copy the whole UTF-8 character that starts
        // here (not a single raw byte reinterpreted as a `char`, which would
        // corrupt multi-byte characters) and advance past it. `sin(`/`cos(`,
        // `^2` and the parentheses are all ASCII, so the matched branches above
        // only ever start on a character boundary; `i` is therefore always on a
        // boundary here too.
        let ch = expr[i..].chars().next().expect("i is on a char boundary");
        result.push(ch);
        i += ch.len_utf8();
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

    #[test]
    fn test_prove_equal_domain_restricted_still_proves() {
        // Regression: these are genuinely equal on their common domain, but the
        // old fixed grid sampled negative/zero points where `sqrt`/`ln` are
        // undefined and treated the resulting eval error as a *disproof*,
        // returning false. Out-of-domain points must be skipped, not counted
        // against equality.
        //
        // sqrt(x)^2 == x  (both require x >= 0)
        assert!(prove_equal("sqrt(x)^2", "x"));
        // ln(exp(x)) == x  (ln's argument exp(x) is always positive)
        assert!(prove_equal("ln(exp(x))", "x"));
        // A restricted expression compared with itself must still hold.
        assert!(prove_equal("ln(x)", "ln(x)"));
    }

    #[test]
    fn test_prove_equal_domain_restricted_rejects_unequal() {
        // Guard against the fix vacuously returning true: distinct expressions
        // that share a domain must still be rejected.
        assert!(!prove_equal("sqrt(x)", "sqrt(x) + 1"));
    }

    #[test]
    fn test_prove_equal_no_common_domain_is_not_true() {
        // If the two sides never share a defined sample point, we must not
        // vacuously "prove" them equal. `ln(x)` needs x > 0 while `ln(-x)`
        // needs x < 0, so they never both evaluate at the same sample.
        assert!(!prove_equal("ln(x)", "ln(-x)"));
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

    #[test]
    fn test_trig_preserves_non_ascii_passthrough() {
        // Regression: the copy-through path used `bytes[i] as char`, which
        // reinterprets each raw UTF-8 byte as a `char` and mangles multi-byte
        // characters. Text with no matching identity must be returned verbatim.
        let input = "α + β·sin(θ) + 漢字";
        assert_eq!(apply_trig_identity(input), input);
    }

    #[test]
    fn test_trig_rewrites_with_non_ascii_argument() {
        // A genuine sin(...)^2 whose argument contains multi-byte characters:
        // the identity must fire and the non-ASCII argument survive intact.
        let result = apply_trig_identity("sin(θ)^2");
        assert_eq!(result, "(1-cos(2*θ))/2");
    }

    #[test]
    fn test_trig_no_panic_on_multibyte_before_match() {
        // A multi-byte character sits before a real match so the byte cursor
        // advances through it; the old byte-at-a-time slicing risked landing on
        // a non-char boundary. This must neither panic nor corrupt output.
        let result = apply_trig_identity("漢cos(x)^2");
        assert_eq!(result, "漢(1+cos(2*x))/2");
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
    fn test_solve_quadratic_no_cancellation_with_well_separated_roots() {
        // Regression test for a P1 audit finding: the naive
        // (-b ± √disc)/(2a) formula subtracts two ~1e8-magnitude quantities
        // that agree to ~8 digits when b² ≫ 4ac, losing essentially all
        // precision on the small root. True roots of x² + 1e8·x + 1 = 0 are
        // ≈ -1e-8 and ≈ -1e8 (product = c/a = 1, by Vieta).
        let (r_plus, r_minus) = solve_quadratic(1.0, 1e8, 1.0).unwrap();
        assert!(
            ((r_plus - (-1e-8)) / 1e-8).abs() < 1e-6,
            "small root should be accurate to ~1e-6 relative, got {r_plus:e}"
        );
        assert!(
            ((r_minus - (-1e8)) / 1e8).abs() < 1e-12,
            "large root: {r_minus:e}"
        );
    }

    #[test]
    fn test_solve_quadratic_no_real() {
        assert!(solve_quadratic(1.0, 0.0, 1.0).is_err());
    }
}
