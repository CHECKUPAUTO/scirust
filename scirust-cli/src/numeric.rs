//! Numerical subcommands over `scirust-solvers`, driven by `scirust-symbolic`
//! for the expression-based ones (the parsed expression becomes the closure
//! the solver integrates / root-finds). All are deterministic.
//!
//! Commands: integrate, root, minimize, linsolve, det, polyroots, ode.

use std::collections::HashMap;

use scirust_solvers::Matrix;
use scirust_solvers::linalg::solve as lin_solve;
use scirust_solvers::polynomial::Polynomial;
use scirust_solvers::polynomial::roots::real_roots;
use scirust_solvers::quadrature::romberg::romberg;
use scirust_solvers::roots::brent::brent;
use scirust_solvers::{Solution, Tolerance};
use scirust_symbolic::{Expr, diff, eval, parse, simplify};

// ---- shared helpers ----------------------------------------------------

fn parse_expr(expr: &str) -> Result<Expr, u8> {
    parse(expr).map_err(|e| {
        eprintln!("error: cannot parse `{expr}`: {e}");
        2
    })
}

fn parse_f64(s: &str, what: &str) -> Result<f64, u8> {
    s.parse::<f64>().map_err(|_| {
        eprintln!("error: `{s}` is not a number ({what})");
        2
    })
}

/// Build a one-variable closure `f(x) = eval(expr)[var := x]`.
fn fn1(expr: Expr, var: String) -> impl Fn(f64) -> f64 {
    move |x| {
        let mut m = HashMap::new();
        m.insert(var.clone(), x);
        eval(&expr, &m).unwrap_or(f64::NAN)
    }
}

fn parse_list(s: &str, what: &str) -> Result<Vec<f64>, u8> {
    s.split(',')
        .map(|t| parse_f64(t.trim(), what))
        .collect::<Result<Vec<_>, _>>()
}

// ---- commands ----------------------------------------------------------

/// `integrate <expr> <a> <b> [var]` — definite integral via Romberg.
pub fn run_integrate(args: &[String]) -> u8 {
    let (Some(expr), Some(a), Some(b)) = (args.first(), args.get(1), args.get(2))
    else
    {
        eprintln!("usage: scirust integrate <expr> <a> <b> [var]");
        return 2;
    };
    let var = args.get(3).map(String::as_str).unwrap_or("x").to_string();
    let parsed = match parse_expr(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    let (a, b) = match (parse_f64(a, "a"), parse_f64(b, "b"))
    {
        (Ok(a), Ok(b)) => (a, b),
        _ => return 2,
    };
    let value = romberg(fn1(parsed, var), a, b, 1e-10, 20);
    println!(
        "∫[{a}, {b}] {expr} d{} = {value:.10}",
        args.get(3).map(String::as_str).unwrap_or("x")
    );
    0
}

/// `root <expr> <a> <b> [var]` — a root in `[a,b]` via Brent (needs a sign change).
pub fn run_root(args: &[String]) -> u8 {
    let (Some(expr), Some(a), Some(b)) = (args.first(), args.get(1), args.get(2))
    else
    {
        eprintln!("usage: scirust root <expr> <a> <b> [var]   (f(a) and f(b) must differ in sign)");
        return 2;
    };
    let var = args.get(3).map(String::as_str).unwrap_or("x").to_string();
    let parsed = match parse_expr(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    let (a, b) = match (parse_f64(a, "a"), parse_f64(b, "b"))
    {
        (Ok(a), Ok(b)) => (a, b),
        _ => return 2,
    };
    match brent(fn1(parsed, var), a, b, Tolerance::new(1e-12, 1e-12, 200))
    {
        Ok(Solution { value, info }) =>
        {
            println!("root ≈ {value:.10}  ({} iterations)", info.iterations);
            0
        },
        Err(e) =>
        {
            println!("no root found in [{a}, {b}]: {e:?}");
            1
        },
    }
}

/// `minimize <expr> <a> <b> [var]` — local minimum in `[a,b]` by finding a root
/// of the symbolic derivative (Brent). Uses scirust-symbolic for f'.
pub fn run_minimize(args: &[String]) -> u8 {
    let (Some(expr), Some(a), Some(b)) = (args.first(), args.get(1), args.get(2))
    else
    {
        eprintln!("usage: scirust minimize <expr> <a> <b> [var]");
        return 2;
    };
    let var = args.get(3).map(String::as_str).unwrap_or("x").to_string();
    let parsed = match parse_expr(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    let (a, b) = match (parse_f64(a, "a"), parse_f64(b, "b"))
    {
        (Ok(a), Ok(b)) => (a, b),
        _ => return 2,
    };
    let deriv = simplify(&diff(&parsed, &var));
    let f = fn1(parsed, var.clone());
    match brent(fn1(deriv, var), a, b, Tolerance::new(1e-12, 1e-12, 200))
    {
        Ok(sol) =>
        {
            let x = sol.into_inner();
            println!("minimum near x ≈ {x:.8}, f(x) ≈ {:.8}", f(x));
            0
        },
        Err(_) =>
        {
            // No stationary point bracketed: report the better endpoint.
            let (fa, fb) = (f(a), f(b));
            let (x, fx) = if fa <= fb { (a, fa) } else { (b, fb) };
            println!(
                "no interior stationary point in [{a}, {b}]; best endpoint x = {x}, f = {fx:.8}"
            );
            0
        },
    }
}

/// `linsolve "<rows>" "<b>"` — solve A·x = b (LU). Matrix rows are
/// semicolon-separated, entries comma-separated: `"2,1;1,3"` and `"3,5"`.
pub fn run_linsolve(args: &[String]) -> u8 {
    let (Some(mat), Some(rhs)) = (args.first(), args.get(1))
    else
    {
        eprintln!("usage: scirust linsolve \"2,1;1,3\" \"3,5\"");
        return 2;
    };
    let rows: Vec<&str> = mat.split(';').collect();
    let n = rows.len();
    let mut data = Vec::with_capacity(n * n);
    for r in &rows
    {
        let vals = match parse_list(r, "matrix entry")
        {
            Ok(v) => v,
            Err(c) => return c,
        };
        if vals.len() != n
        {
            eprintln!(
                "error: matrix must be square ({n} rows but a row has {} cols)",
                vals.len()
            );
            return 2;
        }
        data.extend(vals);
    }
    let b = match parse_list(rhs, "rhs entry")
    {
        Ok(v) => v,
        Err(c) => return c,
    };
    if b.len() != n
    {
        eprintln!("error: rhs length {} must equal matrix size {n}", b.len());
        return 2;
    }
    match lin_solve(Matrix::from_row_major(n, n, data), &b)
    {
        Ok(x) =>
        {
            let xs: Vec<String> = x.iter().map(|v| format!("{v:.6}")).collect();
            println!("x = [ {} ]", xs.join(", "));
            0
        },
        Err(e) =>
        {
            println!("no unique solution: {e:?}");
            1
        },
    }
}

/// `det "<rows>"` — determinant of a square matrix.
pub fn run_det(args: &[String]) -> u8 {
    let Some(mat) = args.first()
    else
    {
        eprintln!("usage: scirust det \"2,1;1,3\"");
        return 2;
    };
    let rows: Vec<&str> = mat.split(';').collect();
    let n = rows.len();
    let mut data = Vec::with_capacity(n * n);
    for r in &rows
    {
        let vals = match parse_list(r, "matrix entry")
        {
            Ok(v) => v,
            Err(c) => return c,
        };
        if vals.len() != n
        {
            eprintln!("error: matrix must be square");
            return 2;
        }
        data.extend(vals);
    }
    match Matrix::from_row_major(n, n, data).determinant()
    {
        Ok(d) =>
        {
            println!("det = {d:.8}");
            0
        },
        Err(e) =>
        {
            println!("error: {e:?}");
            2
        },
    }
}

/// `polyroots "<c0,c1,...>"` — real roots of c0 + c1·x + c2·x² + … (ascending).
pub fn run_polyroots(args: &[String]) -> u8 {
    let Some(coeffs) = args.first()
    else
    {
        eprintln!("usage: scirust polyroots \"c0,c1,c2,...\"   (ascending powers)");
        return 2;
    };
    let c = match parse_list(coeffs, "coefficient")
    {
        Ok(v) => v,
        Err(code) => return code,
    };
    let p = Polynomial::new(c);
    match real_roots(&p, 1e-9)
    {
        Ok(r) if !r.is_empty() =>
        {
            let rs: Vec<String> = r.iter().map(|x| format!("{x:.6}")).collect();
            println!("real roots: {{ {} }}", rs.join(", "));
            0
        },
        Ok(_) =>
        {
            println!("no real roots");
            0
        },
        Err(e) =>
        {
            println!("error: {e:?}");
            2
        },
    }
}

/// `ode <expr> <y0> <t0> <t1> [h]` — integrate dy/dt = f(t, y) with RK4.
/// The expression may use variables `t` and `y`.
pub fn run_ode(args: &[String]) -> u8 {
    let (Some(expr), Some(y0), Some(t0), Some(t1)) =
        (args.first(), args.get(1), args.get(2), args.get(3))
    else
    {
        eprintln!("usage: scirust ode <expr in t,y> <y0> <t0> <t1> [h]   e.g. ode \"y\" 1 0 1");
        return 2;
    };
    let parsed = match parse_expr(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    let (y0, t0, t1) = match (
        parse_f64(y0, "y0"),
        parse_f64(t0, "t0"),
        parse_f64(t1, "t1"),
    )
    {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => return 2,
    };
    let h = match args.get(4)
    {
        Some(s) => match parse_f64(s, "h")
        {
            Ok(v) => v,
            Err(c) => return c,
        },
        None => ((t1 - t0) / 100.0).abs().max(1e-6),
    };
    let f = move |t: f64, y: &[f64], dy: &mut [f64]| {
        let mut m = HashMap::new();
        m.insert("t".to_string(), t);
        m.insert("y".to_string(), y[0]);
        dy[0] = eval(&parsed, &m).unwrap_or(f64::NAN);
    };
    let traj = scirust_solvers::ode::rk4::rk4_fixed(f, t0, t1, vec![y0], h);
    let (tf, yf) = traj.last().expect("at least the initial point");
    println!("dy/dt = {expr},  y({t0}) = {y0}");
    println!("y({tf:.4}) ≈ {:.8}   ({} steps)", yf[0], traj.len() - 1);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn integrate_polynomial() {
        // ∫₀¹ x dx = 0.5
        assert_eq!(run_integrate(&s(&["x", "0", "1"])), 0);
        assert_eq!(run_integrate(&[]), 2);
        assert_eq!(run_integrate(&s(&["x", "bad", "1"])), 2);
    }

    #[test]
    fn root_and_minimize() {
        assert_eq!(run_root(&s(&["x^2 - 2", "0", "2"])), 0); // √2
        assert_eq!(run_minimize(&s(&["x^2 - 4*x + 7", "0", "5"])), 0); // min at x=2
        assert_eq!(run_root(&[]), 2);
    }

    #[test]
    fn linsolve_det_polyroots() {
        // [[2,1],[1,3]] x = [3,5] → x=[0.8,1.4]
        assert_eq!(run_linsolve(&s(&["2,1;1,3", "3,5"])), 0);
        assert_eq!(run_det(&s(&["2,1;1,3"])), 0); // det=5
        assert_eq!(run_polyroots(&s(&["-2,0,1"])), 0); // x²-2 → ±√2
        assert_eq!(run_linsolve(&s(&["1,2,3", "1,2"])), 2); // non-square
    }

    #[test]
    fn ode_exponential_growth() {
        // dy/dt = y, y(0)=1 → y(1) ≈ e
        assert_eq!(run_ode(&s(&["y", "1", "0", "1"])), 0);
        assert_eq!(run_ode(&[]), 2);
    }
}
