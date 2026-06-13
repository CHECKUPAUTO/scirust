//! Symbolic-math subcommands: differentiate, simplify, evaluate, solve.
//!
//! Thin wrappers over `scirust_symbolic` (parse / diff / simplify / eval /
//! solve_*). Each returns a process exit code: 0 on success, 2 on a parse
//! or usage error.

use std::collections::HashMap;

use scirust_symbolic::{diff, eval, parse, simplify, solve_linear, solve_quadratic};

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

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
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
