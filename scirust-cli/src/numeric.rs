//! Numerical subcommands over `scirust-solvers`, driven by `scirust-symbolic`
//! for the expression-based ones (the parsed expression becomes the closure
//! the solver integrates / root-finds). All are deterministic.
//!
//! Commands: integrate, root, minimize, linsolve, det, polyroots, ode.

use std::collections::HashMap;

use scirust_solvers::Matrix;
use scirust_solvers::linalg::cholesky::cholesky_decompose;
use scirust_solvers::linalg::qr::{qr_decompose, solve_qr_least_squares};
use scirust_solvers::linalg::solve as lin_solve;
use scirust_solvers::optimize::nelder_mead::nelder_mead;
use scirust_solvers::polynomial::Polynomial;
use scirust_solvers::polynomial::roots::real_roots;
use scirust_solvers::quadrature::gauss::{GaussOrder, gauss_legendre};
use scirust_solvers::quadrature::romberg::romberg;
use scirust_solvers::quadrature::simpson::simpson_adaptive;
use scirust_solvers::roots::bisection::bisection;
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

/// Remove a `--flag value` pair from `args`, returning the value (if any)
/// and the remaining (positional) arguments.
fn take_flag(args: &[String], name: &str) -> (Option<String>, Vec<String>) {
    let mut value = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len()
    {
        if args[i] == name && i + 1 < args.len()
        {
            value = Some(args[i + 1].clone());
            i += 2;
        }
        else
        {
            rest.push(args[i].clone());
            i += 1;
        }
    }
    (value, rest)
}

/// Build a one-variable closure `f(x) = eval(expr)[var := x]`.
fn fn1(expr: Expr, var: String) -> impl Fn(f64) -> f64 {
    move |x| {
        let mut m = HashMap::new();
        m.insert(var.clone(), x);
        eval(&expr, &m).unwrap_or(f64::NAN)
    }
}

/// Build an N-variable closure `f(xs) = eval(expr)[vars := xs]`.
fn fn_n(expr: Expr, vars: Vec<String>) -> impl Fn(&[f64]) -> f64 {
    move |xs: &[f64]| {
        let mut m = HashMap::new();
        for (v, x) in vars.iter().zip(xs)
        {
            m.insert(v.clone(), *x);
        }
        eval(&expr, &m).unwrap_or(f64::NAN)
    }
}

fn parse_list(s: &str, what: &str) -> Result<Vec<f64>, u8> {
    s.split(',')
        .map(|t| parse_f64(t.trim(), what))
        .collect::<Result<Vec<_>, _>>()
}

// ---- commands ----------------------------------------------------------

/// `integrate <expr> <a> <b> [var] [--method romberg|simpson|gauss]` —
/// definite integral. Default method: Romberg.
pub fn run_integrate(args: &[String]) -> u8 {
    let (method, pos) = take_flag(args, "--method");
    let method = method.unwrap_or_else(|| "romberg".to_string());
    let (Some(expr), Some(a), Some(b)) = (pos.first(), pos.get(1), pos.get(2))
    else
    {
        eprintln!("usage: scirust integrate <expr> <a> <b> [var] [--method romberg|simpson|gauss]");
        return 2;
    };
    let var = pos.get(3).cloned().unwrap_or_else(|| "x".to_string());
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
    let f = fn1(parsed, var.clone());
    let value = match method.as_str()
    {
        "romberg" => romberg(f, a, b, 1e-10, 20),
        "simpson" => match simpson_adaptive(f, a, b, 1e-10, 50)
        {
            Ok(v) => v,
            Err(e) =>
            {
                println!("error: {e:?}");
                return 2;
            },
        },
        "gauss" => gauss_legendre(f, a, b, GaussOrder::Twenty),
        other =>
        {
            eprintln!("error: unknown method `{other}` (romberg|simpson|gauss)");
            return 2;
        },
    };
    println!("∫[{a}, {b}] {expr} d{var} = {value:.10}  ({method})");
    0
}

/// `root <expr> <a> <b> [var] [--method brent|bisection]` — a root in
/// `[a,b]` (needs a sign change). Default method: Brent.
pub fn run_root(args: &[String]) -> u8 {
    let (method, pos) = take_flag(args, "--method");
    let method = method.unwrap_or_else(|| "brent".to_string());
    let (Some(expr), Some(a), Some(b)) = (pos.first(), pos.get(1), pos.get(2))
    else
    {
        eprintln!(
            "usage: scirust root <expr> <a> <b> [var] [--method brent|bisection]   (sign change needed)"
        );
        return 2;
    };
    let var = pos.get(3).cloned().unwrap_or_else(|| "x".to_string());
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
    let tol = Tolerance::new(1e-12, 1e-12, 200);
    let f = fn1(parsed, var);
    let result = match method.as_str()
    {
        "brent" => brent(f, a, b, tol),
        "bisection" => bisection(f, a, b, tol),
        other =>
        {
            eprintln!("error: unknown method `{other}` (brent|bisection)");
            return 2;
        },
    };
    match result
    {
        Ok(Solution { value, info }) =>
        {
            println!(
                "root ≈ {value:.10}  ({} iterations, {method})",
                info.iterations
            );
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

/// Parse an `"r;r;r"` matrix string into (rows, cols, row-major data).
fn parse_matrix(s: &str) -> Result<(usize, usize, Vec<f64>), u8> {
    let rows: Vec<&str> = s.split(';').collect();
    let m = rows.len();
    let mut data = Vec::new();
    let mut n = 0;
    for (i, r) in rows.iter().enumerate()
    {
        let vals = parse_list(r, "matrix entry")?;
        if i == 0
        {
            n = vals.len();
        }
        else if vals.len() != n
        {
            eprintln!("error: every row must have {n} columns");
            return Err(2);
        }
        data.extend(vals);
    }
    Ok((m, n, data))
}

/// `lstsq "<rows>" "<b>"` — least-squares solution of an over-determined
/// A·x ≈ b via QR (m ≥ n).
pub fn run_lstsq(args: &[String]) -> u8 {
    let (Some(mat), Some(rhs)) = (args.first(), args.get(1))
    else
    {
        eprintln!("usage: scirust lstsq \"1,0;1,1;1,2\" \"1,2,3\"   (rows ≥ cols)");
        return 2;
    };
    let (m, n, data) = match parse_matrix(mat)
    {
        Ok(t) => t,
        Err(c) => return c,
    };
    let b = match parse_list(rhs, "rhs entry")
    {
        Ok(v) => v,
        Err(c) => return c,
    };
    if b.len() != m
    {
        eprintln!(
            "error: rhs length {} must equal the number of rows {m}",
            b.len()
        );
        return 2;
    }
    if m < n
    {
        eprintln!("error: need rows ≥ cols for least squares (got {m}x{n})");
        return 2;
    }
    let qr = match qr_decompose(Matrix::from_row_major(m, n, data))
    {
        Ok(q) => q,
        Err(e) =>
        {
            println!("error: {e:?}");
            return 2;
        },
    };
    match solve_qr_least_squares(&qr, &b)
    {
        Ok(x) =>
        {
            let xs: Vec<String> = x.iter().map(|v| format!("{v:.6}")).collect();
            println!("least-squares x = [ {} ]", xs.join(", "));
            0
        },
        Err(e) =>
        {
            println!("error: {e:?}");
            1
        },
    }
}

/// `cholesky "<rows>"` — Cholesky factor L (A = L·Lᵀ) of an SPD matrix.
pub fn run_cholesky(args: &[String]) -> u8 {
    let Some(mat) = args.first()
    else
    {
        eprintln!("usage: scirust cholesky \"4,2;2,3\"   (symmetric positive-definite)");
        return 2;
    };
    let (m, n, data) = match parse_matrix(mat)
    {
        Ok(t) => t,
        Err(c) => return c,
    };
    if m != n
    {
        eprintln!("error: matrix must be square");
        return 2;
    }
    match cholesky_decompose(Matrix::from_row_major(m, n, data))
    {
        Ok(l) =>
        {
            println!("L (lower-triangular, A = L·Lᵀ):");
            for i in 0..n
            {
                let row: Vec<String> = (0..n)
                    .map(|j| format!("{:.6}", l.data()[i * n + j]))
                    .collect();
                println!("  [ {} ]", row.join(", "));
            }
            0
        },
        Err(e) =>
        {
            println!("not SPD / error: {e:?}");
            1
        },
    }
}

/// `optimize "<expr>" --vars x,y --start 1,1` — local minimum of a
/// multi-variable expression via Nelder–Mead (derivative-free).
pub fn run_optimize(args: &[String]) -> u8 {
    let (vars_s, rest) = take_flag(args, "--vars");
    let (start_s, rest) = take_flag(&rest, "--start");
    let Some(expr) = rest.first()
    else
    {
        eprintln!("usage: scirust optimize \"<expr>\" --start a,b,.. [--vars x,y,..]");
        return 2;
    };
    let start = match start_s
    {
        Some(s) => match parse_list(&s, "start coordinate")
        {
            Ok(v) => v,
            Err(c) => return c,
        },
        None =>
        {
            eprintln!("error: --start a,b,.. is required");
            return 2;
        },
    };
    let vars: Vec<String> = match vars_s
    {
        Some(v) => v.split(',').map(|t| t.trim().to_string()).collect(),
        None if start.len() == 1 => vec!["x".to_string()],
        None =>
        {
            eprintln!("error: --vars x,y,.. is required when --start has more than one value");
            return 2;
        },
    };
    if vars.len() != start.len()
    {
        eprintln!("error: --vars and --start must have the same length");
        return 2;
    }
    let parsed = match parse_expr(expr)
    {
        Ok(e) => e,
        Err(c) => return c,
    };
    let f = fn_n(parsed, vars.clone());
    match nelder_mead(f, start, 0.5, Tolerance::new(1e-10, 1e-10, 1000))
    {
        Ok(sol) =>
        {
            let x = sol.into_inner();
            let pt: Vec<String> = vars
                .iter()
                .zip(&x)
                .map(|(v, xi)| format!("{v}={xi:.6}"))
                .collect();
            println!("minimum at  {}", pt.join(", "));
            0
        },
        Err(e) =>
        {
            println!("error: {e:?}");
            1
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
    fn integrate_polynomial() {
        // ∫₀¹ x dx = 0.5
        assert_eq!(run_integrate(&s(&["x", "0", "1"])), 0);
        assert_eq!(run_integrate(&[]), 2);
        assert_eq!(run_integrate(&s(&["x", "bad", "1"])), 2);
    }

    #[test]
    fn integrate_methods_and_root_methods() {
        for m in ["romberg", "simpson", "gauss"]
        {
            assert_eq!(
                run_integrate(&s(&["sin(x)", "0", "3.14159", "--method", m])),
                0
            );
        }
        assert_eq!(run_integrate(&s(&["x", "0", "1", "--method", "nope"])), 2);
        assert_eq!(
            run_root(&s(&["x^2 - 2", "0", "2", "--method", "bisection"])),
            0
        );
        assert_eq!(run_root(&s(&["x^2 - 2", "0", "2", "--method", "brent"])), 0);
    }

    #[test]
    fn lstsq_cholesky_optimize() {
        // Over-determined line fit y≈x+? through (0,1),(1,2),(2,3): exact.
        assert_eq!(run_lstsq(&s(&["1,0;1,1;1,2", "1,2,3"])), 0);
        assert_eq!(run_lstsq(&s(&["1,0;1,1", "1,2,3"])), 2); // rhs len mismatch
        assert_eq!(run_cholesky(&s(&["4,2;2,3"])), 0); // SPD
        assert_eq!(run_cholesky(&s(&["1,2,3"])), 2); // non-square
        // min of (x-1)²+(y-2)² at (1,2)
        assert_eq!(
            run_optimize(&s(&[
                "(x - 1)^2 + (y - 2)^2",
                "--vars",
                "x,y",
                "--start",
                "0,0"
            ])),
            0
        );
        assert_eq!(run_optimize(&s(&["x^2"])), 2); // missing --start
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
