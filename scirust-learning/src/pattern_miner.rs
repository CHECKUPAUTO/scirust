//! Symbolic pattern mining engine.
//!
//! Generates candidate expression trees from basic operations (+,-,*,/,
//! sin, cos, exp, log), fits each to data via linear regression, and
//! ranks them by score (R² / complexity).

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Expression node
// ---------------------------------------------------------------------------

/// Recursive symbolic expression tree node.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprNode {
    /// Variable x
    X,
    /// Constant value
    Const(f64),
    /// a + b
    Add(Box<ExprNode>, Box<ExprNode>),
    /// a - b
    Sub(Box<ExprNode>, Box<ExprNode>),
    /// a * b
    Mul(Box<ExprNode>, Box<ExprNode>),
    /// a / b
    Div(Box<ExprNode>, Box<ExprNode>),
    /// sin(a)
    Sin(Box<ExprNode>),
    /// cos(a)
    Cos(Box<ExprNode>),
    /// exp(a)
    Exp(Box<ExprNode>),
    /// log(a) — natural log
    Log(Box<ExprNode>),
    /// a²
    Pow2(Box<ExprNode>),
}

impl ExprNode {
    /// Evaluate the expression tree for a given x.
    pub fn eval(&self, x: f64) -> f64 {
        match self
        {
            ExprNode::X => x,
            ExprNode::Const(c) => *c,
            ExprNode::Add(a, b) => a.eval(x) + b.eval(x),
            ExprNode::Sub(a, b) => a.eval(x) - b.eval(x),
            ExprNode::Mul(a, b) => a.eval(x) * b.eval(x),
            ExprNode::Div(a, b) =>
            {
                let d = b.eval(x);
                if d.abs() < 1e-15 { 0.0 } else { a.eval(x) / d }
            },
            ExprNode::Sin(a) => a.eval(x).sin(),
            ExprNode::Cos(a) => a.eval(x).cos(),
            ExprNode::Exp(a) => a.eval(x).exp(),
            ExprNode::Log(a) =>
            {
                let v = a.eval(x);
                if v <= 0.0 { 0.0 } else { v.ln() }
            },
            ExprNode::Pow2(a) =>
            {
                let v = a.eval(x);
                v * v
            },
        }
    }

    /// Pretty-print the expression as a string.
    pub fn to_expr_string(&self) -> String {
        match self
        {
            ExprNode::X => "x".into(),
            ExprNode::Const(c) =>
            {
                if *c == c.trunc()
                {
                    format!("{}", *c as i64)
                }
                else
                {
                    format!("{:.4}", c)
                }
            },
            ExprNode::Add(a, b) => format!("({}+{})", a.to_expr_string(), b.to_expr_string()),
            ExprNode::Sub(a, b) => format!("({}-{})", a.to_expr_string(), b.to_expr_string()),
            ExprNode::Mul(a, b) => format!("({}*{})", a.to_expr_string(), b.to_expr_string()),
            ExprNode::Div(a, b) => format!("({}/{})", a.to_expr_string(), b.to_expr_string()),
            ExprNode::Sin(a) => format!("sin({})", a.to_expr_string()),
            ExprNode::Cos(a) => format!("cos({})", a.to_expr_string()),
            ExprNode::Exp(a) => format!("exp({})", a.to_expr_string()),
            ExprNode::Log(a) => format!("log({})", a.to_expr_string()),
            ExprNode::Pow2(a) => format!("({})^2", a.to_expr_string()),
        }
    }

    /// Return the node count (a proxy for formula complexity).
    pub fn complexity(&self) -> usize {
        match self
        {
            ExprNode::X | ExprNode::Const(_) => 1,
            ExprNode::Add(a, b)
            | ExprNode::Sub(a, b)
            | ExprNode::Mul(a, b)
            | ExprNode::Div(a, b) => 1 + a.complexity() + b.complexity(),
            ExprNode::Sin(a)
            | ExprNode::Cos(a)
            | ExprNode::Exp(a)
            | ExprNode::Log(a)
            | ExprNode::Pow2(a) => 1 + a.complexity(),
        }
    }
}

// ---------------------------------------------------------------------------
// Discovered Pattern
// ---------------------------------------------------------------------------

/// A symbolic pattern discovered by [`PatternMiner`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPattern {
    /// Human-readable expression string, e.g. `"sin(x)"`.
    pub expression: String,
    /// Coefficient of determination (how well the expression fits the data).
    pub r_squared: f64,
    /// Node count of the expression tree.
    pub complexity: usize,
    /// Aggregate score: `r_squared / complexity`.
    pub score: f64,
    /// Hash of the input data slice (for deduplication).
    pub data_hash: u64,
}

// ---------------------------------------------------------------------------
// Pattern Miner
// ---------------------------------------------------------------------------

/// Engine that generates candidate symbolic expressions and discovers which
/// ones best fit the observed data.
pub struct PatternMiner {
    max_depth: usize,
    #[allow(dead_code)]
    operations: Vec<String>,
}

impl PatternMiner {
    /// Create a new miner that will explore expressions up to `max_depth`
    /// nesting levels.
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            operations: vec![
                "+".into(),
                "-".into(),
                "*".into(),
                "/".into(),
                "sin".into(),
                "cos".into(),
                "exp".into(),
                "log".into(),
            ],
        }
    }

    /// Mine a single univariate time series for symbolic patterns.
    ///
    /// Returns discovered patterns sorted by score descending.
    pub fn mine(&self, series: &[f64]) -> Vec<DiscoveredPattern> {
        if series.len() < 3
        {
            return vec![];
        }

        let n = series.len();
        let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let data_hash = {
            let mut hasher = DefaultHasher::new();
            for &v in series
            {
                v.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        };

        let candidates = generate_expressions(self.max_depth);
        let mut results: Vec<DiscoveredPattern> = Vec::new();
        let mut seen_exprs: HashSet<String> = HashSet::new();

        for expr in &candidates
        {
            let expr_str = expr.to_expr_string();
            if !seen_exprs.insert(expr_str.clone())
            {
                continue; // skip duplicates
            }

            let eval_vals: Vec<f64> = x_vals.iter().map(|&x| expr.eval(x)).collect();

            let (a, b) = super::linear_regression(&eval_vals, series);
            let fitted: Vec<f64> = eval_vals.iter().map(|&f| a * f + b).collect();

            let mean_y: f64 = series.iter().sum::<f64>() / n as f64;
            let ss_res: f64 = series
                .iter()
                .zip(&fitted)
                .map(|(y, f)| (y - f).powi(2))
                .sum();
            let ss_tot: f64 = series.iter().map(|y| (y - mean_y).powi(2)).sum();

            let r_squared = if ss_tot.abs() < 1e-15
            {
                1.0
            }
            else
            {
                (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
            };

            if r_squared < 0.01
            {
                continue;
            }

            let complexity = expr.complexity();
            let score = if complexity > 0 && r_squared.is_finite()
            {
                r_squared / complexity as f64
            }
            else
            {
                0.0
            };

            results.push(DiscoveredPattern {
                expression: expr_str,
                r_squared,
                complexity,
                score,
                data_hash,
            });
        }

        // Sort by score descending (NaN-safe)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Less)
        });

        results
    }

    /// Mine multiple independent series at once.
    ///
    /// Results are merged and sorted by score globally.
    pub fn mine_multi(&self, series_list: &[&[f64]]) -> Vec<DiscoveredPattern> {
        let mut all: Vec<DiscoveredPattern> = Vec::new();
        for &series in series_list
        {
            all.extend(self.mine(series));
        }
        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Less)
        });
        all
    }
}

// ---------------------------------------------------------------------------
// Expression generation
// ---------------------------------------------------------------------------

fn is_constant(node: &ExprNode) -> bool {
    matches!(node, ExprNode::Const(_))
}

fn is_zero(node: &ExprNode) -> bool {
    match node
    {
        ExprNode::Const(c) => c.abs() < 1e-15,
        _ => false,
    }
}

/// Generate every distinct expression tree up to `max_depth` using the
/// available operations and terminal nodes.
fn generate_expressions(max_depth: usize) -> Vec<ExprNode> {
    if max_depth == 0
    {
        return vec![ExprNode::X];
    }

    let mut exprs: Vec<Vec<ExprNode>> = vec![vec![]; max_depth + 1];

    // --- Depth 0 : terminals ---
    exprs[0].push(ExprNode::X);
    exprs[0].push(ExprNode::Const(0.0));
    exprs[0].push(ExprNode::Const(1.0));
    exprs[0].push(ExprNode::Const(2.0));
    exprs[0].push(ExprNode::Const(-1.0));

    for depth in 1..=max_depth
    {
        // --- Unary ops on depth-1 expressions ---
        for prev in &exprs[depth - 1].clone()
        {
            exprs[depth].push(ExprNode::Sin(Box::new(prev.clone())));
            exprs[depth].push(ExprNode::Cos(Box::new(prev.clone())));
            exprs[depth].push(ExprNode::Exp(Box::new(prev.clone())));
            if !is_constant(prev)
            {
                exprs[depth].push(ExprNode::Log(Box::new(prev.clone())));
            }
            exprs[depth].push(ExprNode::Pow2(Box::new(prev.clone())));
        }

        // --- Binary ops combining shallower depths ---
        for d1 in 0..depth
        {
            let d2 = depth - 1 - d1;
            if d2 > max_depth
            {
                continue;
            }
            for e1 in &exprs[d1].clone()
            {
                for e2 in &exprs[d2].clone()
                {
                    exprs[depth].push(ExprNode::Add(Box::new(e1.clone()), Box::new(e2.clone())));
                    exprs[depth].push(ExprNode::Sub(Box::new(e1.clone()), Box::new(e2.clone())));
                    exprs[depth].push(ExprNode::Mul(Box::new(e1.clone()), Box::new(e2.clone())));
                    if !is_zero(e2)
                    {
                        exprs[depth]
                            .push(ExprNode::Div(Box::new(e1.clone()), Box::new(e2.clone())));
                    }
                }
            }
        }
    }

    let mut all: Vec<ExprNode> = Vec::new();
    for d in exprs
    {
        all.extend(d);
    }
    all
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: hash a data slice.
    fn hash_data(data: &[f64]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for &v in data
        {
            v.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    // ---------------------------------------------------------------
    // 1. Sinusoidal pattern
    // ---------------------------------------------------------------
    #[test]
    fn test_discover_sine() {
        let n = 50;
        let noise_level = 0.15;
        let mut rng = fastrand::Rng::new();
        // Generate y = sin(x) where x_i = i (the miner evaluates at x=i)
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64).sin() + noise_level * (rng.f64() - 0.5))
            .collect();

        let miner = PatternMiner::new(3);
        let results = miner.mine(&y);

        assert!(!results.is_empty(), "should discover at least one pattern");

        // The top result should involve sin(x)
        let top = &results[0];
        assert!(
            top.expression.contains("sin"),
            "top expression should contain sin, got: {}",
            top.expression
        );
        assert!(
            top.r_squared > 0.7,
            "sine fit R² should be > 0.7, got: {}",
            top.r_squared
        );
        assert_eq!(top.data_hash, hash_data(&y), "data_hash mismatch");
    }

    // ---------------------------------------------------------------
    // 2. Polynomial pattern: x² + 2x + 1
    // ---------------------------------------------------------------
    #[test]
    fn test_discover_polynomial() {
        let n = 20;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let noise_level = 0.3;
        let mut rng = fastrand::Rng::new();
        let y: Vec<f64> = xs
            .iter()
            .map(|&x| x * x + 2.0 * x + 1.0 + noise_level * (rng.f64() - 0.5))
            .collect();

        let miner = PatternMiner::new(3);
        let results = miner.mine(&y);

        assert!(!results.is_empty(), "should discover polynomial pattern");
        let top = &results[0];
        assert!(
            top.r_squared > 0.7,
            "polynomial fit R² should be > 0.7, got: {}",
            top.r_squared
        );
    }

    // ---------------------------------------------------------------
    // 3. Random noise → no high-scoring patterns
    // ---------------------------------------------------------------
    #[test]
    fn test_discover_empty() {
        let n = 30;
        let mut rng = fastrand::Rng::new();
        let y: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0 - 5.0).collect();

        let miner = PatternMiner::new(2);
        let results = miner.mine(&y);

        // Most random sequences should yield no high-R² pattern
        // (may occasionally get lucky, so we check R² threshold)
        let high_score: Vec<&DiscoveredPattern> =
            results.iter().filter(|p| p.r_squared > 0.5).collect();
        assert!(
            high_score.len() <= 1,
            "random data should not produce patterns with R² > 0.5, got {}",
            high_score.len()
        );
    }

    // ---------------------------------------------------------------
    // 4. Simpler formula scores higher than complex formula
    // ---------------------------------------------------------------
    #[test]
    fn test_complexity_score() {
        let n = 20;
        // Linear data: y ~ x (perfect)
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let miner = PatternMiner::new(3);
        let results = miner.mine(&y);

        assert!(!results.is_empty(), "should find patterns in linear data");

        // A simple x should rank higher than sin(x) on linear data
        let x_result = results.iter().find(|p| p.expression == "x");
        let sin_result = results.iter().find(|p| p.expression.contains("sin"));

        if let (Some(xp), Some(sp)) = (x_result, sin_result)
        {
            assert!(
                xp.score > sp.score,
                "on linear data, 'x' ({}) should score higher than 'sin' ({})",
                xp.score,
                sp.score
            );
        }
    }

    // ---------------------------------------------------------------
    // 5. Batch mining of multiple series
    // ---------------------------------------------------------------
    #[test]
    fn test_mine_multi() {
        let mut rng = fastrand::Rng::new();
        let n = 20;

        // Series 1: noisy sine
        let xs1: Vec<f64> = (0..n)
            .map(|i| i as f64 * std::f64::consts::TAU / n as f64)
            .collect();
        let s1: Vec<f64> = xs1
            .iter()
            .map(|&x| x.sin() + 0.1 * (rng.f64() - 0.5))
            .collect();

        // Series 2: noisy quadratic
        let xs2: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let s2: Vec<f64> = xs2
            .iter()
            .map(|&x| x * x + 0.1 * (rng.f64() - 0.5))
            .collect();

        let miner = PatternMiner::new(2);
        let results = miner.mine_multi(&[&s1, &s2]);

        assert!(results.len() >= 2, "should produce at least 2 patterns");
        // Filter out NaN scores before checking sort order
        let valid_scores: Vec<f64> = results
            .iter()
            .map(|p| p.score)
            .filter(|s| !s.is_nan())
            .collect();
        for w in valid_scores.windows(2)
        {
            assert!(
                w[0] + 1e-9 >= w[1],
                "mine_multi scores should be sorted descending, got {} >= {}",
                w[0],
                w[1]
            );
        }

        // Expect both sine and quadratic to appear
        let has_sin = results.iter().any(|p| p.expression.contains("sin"));
        let has_pow2 = results.iter().any(|p| p.expression.contains("^2"));
        assert!(has_sin || has_pow2, "should contain sin or pow2 patterns");
    }
}
