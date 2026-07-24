use scirust_symbolic::{Expr, diff, eval};
use std::collections::HashMap;

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct ConsistencyReport {
    pub function_name: String,
    pub symbolic_gradient: Vec<f64>,
    pub autodiff_gradient: Vec<f64>,
    pub fd_gradient: Vec<f64>,
    pub symbolic_autodiff_max_error: f64,
    pub symbolic_fd_max_error: f64,
    pub passed: bool,
}

pub fn compare_derivatives<F>(
    f_expr: &Expr,
    f_autodiff: F,
    vars: &[&str],
    point: &[f64],
    fd_eps: f64,
    tolerance: f64,
) -> Result<ConsistencyReport>
where
    F: Fn(&[f64]) -> Vec<f64> + std::panic::UnwindSafe,
{
    let n = vars.len();
    let mut symbolic_grad = Vec::with_capacity(n);
    let mut fd_grad = Vec::with_capacity(n);

    for var in vars.iter()
    {
        let d_expr = diff(f_expr, var);

        let mut binding = HashMap::new();
        for (j, v) in vars.iter().enumerate()
        {
            binding.insert(v.to_string(), point[j]);
        }
        let sym_val = eval(&d_expr, &binding).unwrap_or(f64::NAN);
        symbolic_grad.push(sym_val);
    }

    let autodiff_grad =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f_autodiff(point)))
            .unwrap_or_else(|_| vec![f64::NAN; n]);

    let f_num = |x: &[f64]| -> f64 {
        let mut binding = HashMap::new();
        for (j, v) in vars.iter().enumerate()
        {
            binding.insert(v.to_string(), x[j]);
        }
        eval(f_expr, &binding).unwrap_or(f64::NAN)
    };

    for i in 0..n
    {
        let mut xp = point.to_vec();
        xp[i] += fd_eps;
        let fp = f_num(&xp);

        let mut xm = point.to_vec();
        xm[i] -= fd_eps;
        let fm = f_num(&xm);

        fd_grad.push((fp - fm) / (2.0 * fd_eps));
    }

    let sym_ad_err = symbolic_grad
        .iter()
        .zip(autodiff_grad.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    let sym_fd_err = symbolic_grad
        .iter()
        .zip(fd_grad.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    let passed = sym_ad_err < tolerance && sym_fd_err < tolerance;

    Ok(ConsistencyReport {
        function_name: format!("{:?}", f_expr),
        symbolic_gradient: symbolic_grad,
        autodiff_gradient: autodiff_grad,
        fd_gradient: fd_grad,
        symbolic_autodiff_max_error: sym_ad_err,
        symbolic_fd_max_error: sym_fd_err,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_consistency() {
        let x = Expr::Var("x".to_string());
        let f = x.clone() * x.clone();
        let report =
            compare_derivatives(&f, |pt| vec![2.0 * pt[0]], &["x"], &[2.0], 1e-6, 1e-4).unwrap();
        assert!(report.passed, "consistency check failed");
    }
}
