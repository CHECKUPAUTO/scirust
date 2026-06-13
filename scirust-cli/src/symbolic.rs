//! Symbolic-math subcommands: differentiate, simplify, evaluate, solve.
//!
//! Thin wrappers over `scirust_symbolic` (parse / diff / simplify / eval /
//! solve_*). Each returns a process exit code: 0 on success, 2 on a parse
//! or usage error.

use std::collections::HashMap;

use scirust_symbolic::{
    diff, eval, linear_regression, parse, polynomial_fit, simplify, solve_linear, solve_quadratic,
    to_rust_code,
};

fn parse_or_report(expr: &str) -> Result<scirust_symbolic::Expr, u8> {
    parse(expr).map_err(|e| {
        eprintln!("error: cannot parse `{expr}`: {e}");
        2
    })
}

/// `diff <expr> [var]` — symbolic derivative (default variable `x`).
pub fn run_diff(args: &[String]) -> u8 {
    let Some(expr) = args.first()
    else
    {
        eprintln!("usage: scirust diff <expr> [var]   e.g. scirust diff \"x^2 + 3*x\"");
        return 2;
    };
    let var = args.get(1).map(String::as_str).unwrap_or("x");
    let parsed = match parse_or_report(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    let d = simplify(&diff(&parsed, var));
    println!("d/d{var} [ {expr} ] = {d}");
    0
}

/// `simplify <expr>` — algebraic simplification.
pub fn run_simplify(args: &[String]) -> u8 {
    let Some(expr) = args.first()
    else
    {
        eprintln!("usage: scirust simplify <expr>");
        return 2;
    };
    match parse_or_report(expr)
    {
        Ok(e) =>
        {
            println!("{}", simplify(&e));
            0
        },
        Err(c) => c,
    }
}

/// `eval <expr> [x=.. y=..]` — evaluate at given variable values.
pub fn run_eval(args: &[String]) -> u8 {
    let Some(expr) = args.first()
    else
    {
        eprintln!("usage: scirust eval <expr> [x=1.5 y=2 ...]");
        return 2;
    };
    let mut vars: HashMap<String, f64> = HashMap::new();
    for a in &args[1..]
    {
        let Some((name, val)) = a.split_once('=')
        else
        {
            eprintln!("error: bindings must look like `x=1.5`, got `{a}`");
            return 2;
        };
        match val.parse::<f64>()
        {
            Ok(v) =>
            {
                vars.insert(name.to_string(), v);
            },
            Err(_) =>
            {
                eprintln!("error: `{val}` is not a number (in `{a}`)");
                return 2;
            },
        }
    }
    let parsed = match parse_or_report(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    match eval(&parsed, &vars)
    {
        Ok(v) =>
        {
            println!("{v}");
            0
        },
        Err(e) =>
        {
            eprintln!("error: {e}");
            2
        },
    }
}

/// `solve <expr> [var]` — real roots of `expr = 0` (linear or quadratic).
pub fn run_solve(args: &[String]) -> u8 {
    let Some(expr) = args.first()
    else
    {
        eprintln!("usage: scirust solve <expr> [var]   e.g. scirust solve \"x^2 - 4\"");
        return 2;
    };
    let var = args.get(1).map(String::as_str).unwrap_or("x");
    let parsed = match parse_or_report(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    let quad = solve_quadratic(&parsed, var);
    if !quad.is_empty()
    {
        let roots: Vec<String> = quad.iter().map(|r| format!("{r:.6}")).collect();
        println!("{var} ∈ {{ {} }}", roots.join(", "));
        return 0;
    }
    match solve_linear(&parsed, var)
    {
        Some(r) =>
        {
            println!("{var} = {r:.6}");
            0
        },
        None =>
        {
            println!("no real root found for `{expr}` in {var} (linear/quadratic only)");
            0
        },
    }
}

/// `to-rust <expr>` — transpile an expression to Rust source.
pub fn run_to_rust(args: &[String]) -> u8 {
    let Some(expr) = args.first()
    else
    {
        eprintln!("usage: scirust to-rust <expr>");
        return 2;
    };
    match parse_or_report(expr)
    {
        Ok(e) =>
        {
            println!("{}", to_rust_code(&simplify(&e)));
            0
        },
        Err(c) => c,
    }
}

/// `regress <xs> <ys> [degree]` — least-squares fit. With no degree (or
/// degree 1) reports slope/intercept; with degree ≥ 2 reports polynomial
/// coefficients (ascending). Both are comma-separated number lists.
pub fn run_regress(args: &[String]) -> u8 {
    let (Some(xs_s), Some(ys_s)) = (args.first(), args.get(1))
    else
    {
        eprintln!("usage: scirust regress <x1,x2,..> <y1,y2,..> [degree]");
        return 2;
    };
    let parse_list = |s: &str| -> Result<Vec<f64>, u8> {
        s.split(',')
            .map(|t| {
                t.trim().parse::<f64>().map_err(|_| {
                    eprintln!("error: `{}` is not a number", t.trim());
                    2u8
                })
            })
            .collect()
    };
    let xs = match parse_list(xs_s)
    {
        Ok(v) => v,
        Err(c) => return c,
    };
    let ys = match parse_list(ys_s)
    {
        Ok(v) => v,
        Err(c) => return c,
    };
    if xs.len() != ys.len() || xs.is_empty()
    {
        eprintln!("error: xs and ys must be non-empty and the same length");
        return 2;
    }
    let degree = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    if degree <= 1
    {
        match linear_regression(&xs, &ys)
        {
            // scirust_symbolic returns (intercept, slope).
            Ok((intercept, slope)) =>
            {
                println!("y = {slope:.6} * x + {intercept:.6}");
                0
            },
            Err(e) =>
            {
                eprintln!("error: {e}");
                2
            },
        }
    }
    else
    {
        match polynomial_fit(&xs, &ys, degree)
        {
            Ok(coeffs) =>
            {
                let terms: Vec<String> = coeffs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("{c:.6}·x^{i}"))
                    .collect();
                println!("y = {}", terms.join(" + "));
                0
            },
            Err(e) =>
            {
                eprintln!("error: {e}");
                2
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn to_rust_and_regress() {
        assert_eq!(run_to_rust(&s(&["x^2 + 1"])), 0);
        assert_eq!(run_to_rust(&[]), 2);
        // y = 2x + 1 through (0,1),(1,3),(2,5)
        assert_eq!(run_regress(&s(&["0,1,2", "1,3,5"])), 0);
        assert_eq!(run_regress(&s(&["0,1,2", "0,1,4", "2"])), 0); // quadratic
        assert_eq!(run_regress(&s(&["0,1", "1,2,3"])), 2); // length mismatch
    }

    #[test]
    fn regress_recovers_correct_slope_intercept() {
        // Pin the (intercept, slope) convention so the CLI never mislabels it:
        // points lie exactly on y = 2x + 1.
        let (intercept, slope) =
            linear_regression(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0]).expect("fit ok");
        assert!((slope - 2.0).abs() < 1e-9, "slope should be 2, got {slope}");
        assert!(
            (intercept - 1.0).abs() < 1e-9,
            "intercept should be 1, got {intercept}"
        );
    }

    #[test]
    fn diff_ok_and_parse_error() {
        assert_eq!(run_diff(&s(&["x*x"])), 0);
        assert_eq!(run_diff(&s(&["x^3", "x"])), 0);
        assert_eq!(run_diff(&[]), 2);
        assert_eq!(run_diff(&s(&["@@@"])), 2);
    }

    #[test]
    fn eval_computes_value() {
        // 2*x + 1 at x=3 → 7
        assert_eq!(run_eval(&s(&["2*x + 1", "x=3"])), 0);
        assert_eq!(run_eval(&s(&["x", "x=notanumber"])), 2);
        assert_eq!(run_eval(&s(&["x", "bad"])), 2);
    }

    #[test]
    fn solve_and_simplify() {
        assert_eq!(run_solve(&s(&["x^2 - 4"])), 0);
        assert_eq!(run_solve(&s(&["2*x - 4"])), 0);
        assert_eq!(run_simplify(&s(&["x + x"])), 0);
        assert_eq!(run_simplify(&[]), 2);
    }
}
