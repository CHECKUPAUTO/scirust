//! Numerical subcommands over `scirust-solvers` and `scirust-tn`, driven by
//! `scirust-symbolic` for the expression-based ones (the parsed expression
//! becomes the closure the solver integrates / root-finds). All are
//! deterministic.
//!
//! Commands: integrate, root, minimize, optimize, linsolve, lstsq, det,
//! cholesky, qr, cg, inverse, solve-system, polyroots, ode, fem-heat, tt.

use std::collections::HashMap;

use scirust_solvers::Matrix;
use scirust_solvers::linalg::cholesky::cholesky_decompose;
use scirust_solvers::linalg::conjugate_gradient;
use scirust_solvers::linalg::qr::{qr_decompose, solve_qr_least_squares};
use scirust_solvers::linalg::solve as lin_solve;
use scirust_solvers::nonlinear::broyden;
use scirust_solvers::ode::dopri5::dopri5;
use scirust_solvers::optimize::nelder_mead::nelder_mead;
use scirust_solvers::polynomial::Polynomial;
use scirust_solvers::polynomial::roots::real_roots;
use scirust_solvers::quadrature::gauss::{GaussOrder, gauss_legendre};
use scirust_solvers::quadrature::romberg::romberg;
use scirust_solvers::quadrature::simpson::simpson_adaptive;
use scirust_solvers::roots::bisection::bisection;
use scirust_solvers::roots::brent::brent;
use scirust_solvers::roots::newton::newton_with_derivative;
use scirust_solvers::roots::secant::secant;
use scirust_solvers::scientific::FemSolver1D;
use scirust_solvers::{Solution, Tolerance};
use scirust_symbolic::{Expr, diff, eval, parse, simplify};
use scirust_tn::{auto_factorize, reconstruct_matrix, tt_decompose_matrix};

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

/// `root <expr> <a> <b> [var] [--method brent|bisection|secant|newton]` — a
/// root using `[a,b]` as a bracket (brent/bisection, sign change needed),
/// as two initial guesses (secant), or as a midpoint start with the
/// symbolic derivative (newton). Default: Brent.
pub fn run_root(args: &[String]) -> u8 {
    let (method, pos) = take_flag(args, "--method");
    let method = method.unwrap_or_else(|| "brent".to_string());
    let (Some(expr), Some(a), Some(b)) = (pos.first(), pos.get(1), pos.get(2))
    else
    {
        eprintln!(
            "usage: scirust root <expr> <a> <b> [var] [--method brent|bisection|secant|newton]"
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
    let result = match method.as_str()
    {
        "brent" => brent(fn1(parsed, var), a, b, tol),
        "bisection" => bisection(fn1(parsed, var), a, b, tol),
        "secant" => secant(fn1(parsed, var), a, b, tol),
        "newton" =>
        {
            let deriv = simplify(&diff(&parsed, &var));
            let f = fn1(parsed, var.clone());
            let df = fn1(deriv, var);
            newton_with_derivative(f, df, 0.5 * (a + b), tol)
        },
        other =>
        {
            eprintln!("error: unknown method `{other}` (brent|bisection|secant|newton)");
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

/// `ode <expr> <y0> <t0> <t1> [h] [--method rk4|dopri5]` — integrate
/// dy/dt = f(t, y). The expression may use variables `t` and `y`. `rk4` is a
/// fixed-step Runge–Kutta (default); `dopri5` is adaptive Dormand–Prince (the
/// `h` argument seeds its initial step). Returns 1 if dopri5 fails to step.
pub fn run_ode(args: &[String]) -> u8 {
    let (method, pos) = take_flag(args, "--method");
    let method = method.unwrap_or_else(|| "rk4".to_string());
    let (Some(expr), Some(y0), Some(t0), Some(t1)) =
        (pos.first(), pos.get(1), pos.get(2), pos.get(3))
    else
    {
        eprintln!(
            "usage: scirust ode <expr in t,y> <y0> <t0> <t1> [h] [--method rk4|dopri5]   e.g. ode \"y\" 1 0 1"
        );
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
    let h = match pos.get(4)
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
    println!("dy/dt = {expr},  y({t0}) = {y0}");
    match method.as_str()
    {
        "rk4" =>
        {
            let traj = scirust_solvers::ode::rk4::rk4_fixed(f, t0, t1, vec![y0], h);
            let (tf, yf) = traj.last().expect("at least the initial point");
            println!(
                "y({tf:.4}) ≈ {:.8}   ({} steps, rk4)",
                yf[0],
                traj.len() - 1
            );
            0
        },
        "dopri5" => match dopri5(f, t0, t1, vec![y0], 1e-8, 1e-10, h)
        {
            Ok(out) =>
            {
                let tf = out.t.last().copied().unwrap_or(t0);
                let yf = out.y.last().map(|v| v[0]).unwrap_or(y0);
                println!(
                    "y({tf:.4}) ≈ {yf:.8}   ({} accepted / {} rejected steps, dopri5)",
                    out.accepted, out.rejected
                );
                0
            },
            Err(e) =>
            {
                println!("integration failed: {e:?}");
                1
            },
        },
        other =>
        {
            eprintln!("error: unknown method `{other}` (rk4|dopri5)");
            2
        },
    }
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

/// `qr "<rows>"` — QR decomposition A = Q·R. Prints Q and R.
pub fn run_qr(args: &[String]) -> u8 {
    let Some(mat) = args.first()
    else
    {
        eprintln!("usage: scirust qr \"1,1;0,1;1,0\"");
        return 2;
    };
    let (m, n, data) = match parse_matrix(mat)
    {
        Ok(t) => t,
        Err(c) => return c,
    };
    let qr = match qr_decompose(Matrix::from_row_major(m, n, data))
    {
        Ok(q) => q,
        Err(e) =>
        {
            println!("error: {e:?}");
            return 2;
        },
    };
    let print_mat = |name: &str, mtx: &Matrix| {
        println!("{name} ({}x{}):", mtx.rows(), mtx.cols());
        for i in 0..mtx.rows()
        {
            let row: Vec<String> = (0..mtx.cols())
                .map(|j| format!("{:.6}", mtx.data()[i * mtx.cols() + j]))
                .collect();
            println!("  [ {} ]", row.join(", "));
        }
    };
    print_mat("Q", &qr.q());
    print_mat("R", &qr.r());
    0
}

/// `cg "<rows>" "<b>"` — solve a symmetric-positive-definite A·x = b with
/// the conjugate-gradient iterative method (matrix-free).
pub fn run_cg(args: &[String]) -> u8 {
    let (Some(mat), Some(rhs)) = (args.first(), args.get(1))
    else
    {
        eprintln!("usage: scirust cg \"4,1;1,3\" \"1,2\"   (A symmetric positive-definite)");
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
    let matvec = move |x: &[f64], out: &mut [f64]| {
        for (i, oi) in out.iter_mut().enumerate()
        {
            *oi = (0..n).map(|j| data[i * n + j] * x[j]).sum();
        }
    };
    match conjugate_gradient(matvec, &b, vec![0.0; n], Tolerance::new(1e-12, 1e-12, 1000))
    {
        Ok(sol) =>
        {
            let iters = sol.info.iterations;
            let xs: Vec<String> = sol.value.iter().map(|v| format!("{v:.6}")).collect();
            println!("x = [ {} ]  ({iters} iterations)", xs.join(", "));
            0
        },
        Err(e) =>
        {
            println!("did not converge (is A SPD?): {e:?}");
            1
        },
    }
}

/// `inverse "<rows>"` — inverse of a square matrix via LU. Prints the inverse,
/// or exits 1 if the matrix is singular.
pub fn run_inverse(args: &[String]) -> u8 {
    let Some(mat) = args.first()
    else
    {
        eprintln!("usage: scirust inverse \"4,7;2,6\"   (square, non-singular)");
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
    match Matrix::from_row_major(m, n, data).inverse()
    {
        Ok(inv) =>
        {
            println!("A⁻¹ ({n}x{n}):");
            for i in 0..n
            {
                let row: Vec<String> = (0..n)
                    .map(|j| format!("{:.6}", inv.data()[i * n + j]))
                    .collect();
                println!("  [ {} ]", row.join(", "));
            }
            0
        },
        Err(e) =>
        {
            println!("singular / error: {e:?}");
            1
        },
    }
}

/// `solve-system "f1; f2; ..." --vars x,y,.. --start a,b,..` — solve the
/// nonlinear system F(x) = 0 with Broyden's quasi-Newton method. Each
/// semicolon-separated expression is one equation `fᵢ(vars) = 0`; there must
/// be exactly as many equations as unknowns. Exits 1 if it fails to converge.
pub fn run_solve_system(args: &[String]) -> u8 {
    let (vars_s, rest) = take_flag(args, "--vars");
    let (start_s, rest) = take_flag(&rest, "--start");
    let Some(sys) = rest.first()
    else
    {
        eprintln!("usage: scirust solve-system \"f1; f2\" --vars x,y --start a,b");
        return 2;
    };
    let exprs_src: Vec<&str> = sys
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
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
    if exprs_src.len() != vars.len()
    {
        eprintln!(
            "error: need exactly as many equations as unknowns ({} equations, {} unknowns)",
            exprs_src.len(),
            vars.len()
        );
        return 2;
    }
    let mut parsed = Vec::with_capacity(exprs_src.len());
    for e in &exprs_src
    {
        match parse_expr(e)
        {
            Ok(p) => parsed.push(p),
            Err(c) => return c,
        }
    }
    let vars2 = vars.clone();
    let f = move |x: &[f64], out: &mut [f64]| {
        let mut m = HashMap::new();
        for (v, xi) in vars2.iter().zip(x)
        {
            m.insert(v.clone(), *xi);
        }
        for (i, e) in parsed.iter().enumerate()
        {
            out[i] = eval(e, &m).unwrap_or(f64::NAN);
        }
    };
    match broyden(f, start, Tolerance::new(1e-12, 1e-12, 500))
    {
        Ok(sol) =>
        {
            let x = sol.into_inner();
            let pt: Vec<String> = vars
                .iter()
                .zip(&x)
                .map(|(v, xi)| format!("{v} = {xi:.6}"))
                .collect();
            println!("solution: {}", pt.join(", "));
            0
        },
        Err(e) =>
        {
            println!("did not converge: {e:?}");
            1
        },
    }
}

/// `fem-heat <nodes> <length> <source>` — 1D steady-state heat / Poisson
/// equation `-u'' = source` on `[0, length]` with `u(0) = u(L) = 0`, solved by
/// linear finite elements. Prints the nodal displacements.
pub fn run_fem_heat(args: &[String]) -> u8 {
    let (Some(nodes), Some(length), Some(source)) = (args.first(), args.get(1), args.get(2))
    else
    {
        eprintln!("usage: scirust fem-heat <nodes> <length> <source>   e.g. fem-heat 9 1 1");
        return 2;
    };
    let nodes = match nodes.parse::<usize>()
    {
        Ok(v) if v >= 2 => v,
        _ =>
        {
            eprintln!("error: nodes must be an integer ≥ 2");
            return 2;
        },
    };
    let (length, source) = match (parse_f64(length, "length"), parse_f64(source, "source"))
    {
        (Ok(a), Ok(b)) if a > 0.0 => (a, b),
        (Ok(_), Ok(_)) =>
        {
            eprintln!("error: length must be positive");
            return 2;
        },
        _ => return 2,
    };
    let u = FemSolver1D::new(nodes, length).solve_steady_heat(source);
    let h = length / (nodes as f64 - 1.0);
    println!("-u'' = {source}  on [0, {length}],  u(0) = u(L) = 0  ({nodes} nodes, linear FEM)");
    for (i, ui) in u.iter().enumerate()
    {
        println!("  x = {:.4}   u = {ui:.6}", i as f64 * h);
    }
    0
}

/// `tt "<rows>" [--factors d] [--max-rank r] [--tol t] [--max-err e]` —
/// tensor-train (TT) compression of a matrix via TT-SVD (Oseledets/Novikov).
/// Reports the number of cores, bond ranks, parameter counts, compression
/// ratio and reconstruction error. With `--max-err e`, exits 1 if the relative
/// Frobenius error exceeds `e` (use it as an accuracy gate); otherwise exits 0.
pub fn run_tt(args: &[String]) -> u8 {
    let (factors_s, rest) = take_flag(args, "--factors");
    let (rank_s, rest) = take_flag(&rest, "--max-rank");
    let (tol_s, rest) = take_flag(&rest, "--tol");
    let (err_s, rest) = take_flag(&rest, "--max-err");
    let Some(mat) = rest.first()
    else
    {
        eprintln!(
            "usage: scirust tt \"<rows>\" [--factors d] [--max-rank r] [--tol t] [--max-err e]"
        );
        return 2;
    };
    let (m, n, data) = match parse_matrix(mat)
    {
        Ok(t) => t,
        Err(c) => return c,
    };
    let d = match factors_s
    {
        Some(s) => match s.parse::<usize>()
        {
            Ok(v) if v >= 1 => v,
            _ =>
            {
                eprintln!("error: --factors must be a positive integer");
                return 2;
            },
        },
        None => 2,
    };
    let max_rank = match rank_s
    {
        Some(s) => match s.parse::<usize>()
        {
            Ok(v) if v >= 1 => v,
            _ =>
            {
                eprintln!("error: --max-rank must be a positive integer");
                return 2;
            },
        },
        None => m.max(n), // no effective cap: truncate by tolerance only
    };
    let tol = match tol_s
    {
        Some(s) => match parse_f64(&s, "tol")
        {
            Ok(v) => v as f32,
            Err(c) => return c,
        },
        None => 0.0f32,
    };
    let in_dims = auto_factorize(m, d);
    let out_dims = auto_factorize(n, d);
    let w: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let tt = tt_decompose_matrix(&w, &in_dims, &out_dims, max_rank, tol);
    let recon = reconstruct_matrix(&tt, &in_dims, &out_dims);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in w.iter().zip(&recon)
    {
        let diff = (*a - *b) as f64;
        num += diff * diff;
        den += (*a as f64) * (*a as f64);
    }
    let rel = num.sqrt() / den.sqrt().max(1e-30);
    let orig = m * n;
    let comp = tt.num_params();
    let ratio = orig as f64 / comp as f64;
    println!(
        "TT decomposition of {m}x{n} matrix  (in_dims = {in_dims:?}, out_dims = {out_dims:?})"
    );
    println!("  cores: {}   bond ranks: {:?}", tt.ndim(), tt.ranks);
    println!("  params: {orig} → {comp}   (compression {ratio:.2}x)");
    println!("  reconstruction rel. error: {rel:.3e}");
    if let Some(e) = err_s
    {
        let budget = match parse_f64(&e, "max-err")
        {
            Ok(v) => v,
            Err(c) => return c,
        };
        if rel > budget
        {
            println!("error budget exceeded: {rel:.3e} > {budget:.3e}");
            return 1;
        }
    }
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
    fn qr_and_cg() {
        // QR of a tall matrix.
        assert_eq!(run_qr(&s(&["1,1;0,1;1,0"])), 0);
        assert_eq!(run_qr(&[]), 2);
        // SPD system [[4,1],[1,3]] x = [1,2].
        assert_eq!(run_cg(&s(&["4,1;1,3", "1,2"])), 0);
        assert_eq!(run_cg(&s(&["4,1;1,3", "1,2,3"])), 2); // rhs mismatch
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
        assert_eq!(run_ode(&s(&["y", "1", "0", "1", "--method", "dopri5"])), 0);
        assert_eq!(run_ode(&s(&["y", "1", "0", "1", "--method", "nope"])), 2);
        assert_eq!(run_ode(&[]), 2);
    }

    #[test]
    fn inverse_and_solve_system() {
        // [[4,7],[2,6]] is non-singular.
        assert_eq!(run_inverse(&s(&["4,7;2,6"])), 0);
        assert_eq!(run_inverse(&s(&["1,2;2,4"])), 1); // singular
        assert_eq!(run_inverse(&s(&["1,2,3"])), 2); // non-square
        // Nonlinear system: x²+y²=4, x−y=0 → x=y=√2.
        assert_eq!(
            run_solve_system(&s(&[
                "x^2 + y^2 - 4; x - y",
                "--vars",
                "x,y",
                "--start",
                "1,1"
            ])),
            0
        );
        // Wrong equation count vs unknowns.
        assert_eq!(
            run_solve_system(&s(&["x - 1", "--vars", "x,y", "--start", "1,1"])),
            2
        );
        assert_eq!(run_solve_system(&s(&["x - 1"])), 2); // missing --start
    }

    #[test]
    fn fem_heat_runs() {
        // -u'' = 1 on [0,1], u(0)=u(L)=0.
        assert_eq!(run_fem_heat(&s(&["9", "1", "1"])), 0);
        assert_eq!(run_fem_heat(&s(&["1", "1", "1"])), 2); // nodes < 2
        assert_eq!(run_fem_heat(&s(&["5", "0", "1"])), 2); // length not positive
        assert_eq!(run_fem_heat(&[]), 2);
    }

    #[test]
    fn tt_compression() {
        // Rank-1 outer-product 4x4 matrix compresses with tiny error.
        assert_eq!(run_tt(&s(&["1,2,3,4;2,4,6,8;3,6,9,12;4,8,12,16"])), 0);
        // Accuracy gate: exact (tol 0) rank-1 stays well under 1e-3.
        assert_eq!(
            run_tt(&s(&[
                "1,2,3,4;2,4,6,8;3,6,9,12;4,8,12,16",
                "--max-err",
                "1e-3"
            ])),
            0
        );
        // Forcing max-rank 1 on a full-rank matrix breaks a tight budget.
        assert_eq!(
            run_tt(&s(&[
                "1,2,3,4;5,6,7,8;9,10,11,12;13,14,15,16",
                "--max-rank",
                "1",
                "--tol",
                "0.5",
                "--max-err",
                "1e-9"
            ])),
            1
        );
        assert_eq!(run_tt(&[]), 2);
    }
}
