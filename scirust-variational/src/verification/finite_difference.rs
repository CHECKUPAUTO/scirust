use crate::error::Result;

#[derive(Debug, Clone)]
pub struct FDCheckReport {
    pub max_absolute_error: f32,
    pub rms_error: f32,
    pub max_relative_error: f32,
    pub n_checks: usize,
}

pub fn check_gradient<F, G>(
    analytical_grad: G,
    f: F,
    x: &[f32],
    eps: f32,
    _tolerance: f32,
) -> Result<FDCheckReport>
where
    F: Fn(&[f32]) -> f32,
    G: Fn(&[f32]) -> Vec<f32>,
{
    let n = x.len();
    let analytic = analytical_grad(x);
    let mut abs_errors = Vec::with_capacity(n);

    for i in 0..n
    {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let fp = f(&xp);

        let mut xm = x.to_vec();
        xm[i] -= eps;
        let fm = f(&xm);

        let fd = (fp - fm) / (2.0 * eps);
        abs_errors.push((analytic[i] - fd).abs());
    }

    let max_abs = abs_errors.iter().copied().fold(0.0, f32::max);
    let rms = (abs_errors.iter().map(|e| e * e).sum::<f32>() / n as f32).sqrt();
    let max_rel = abs_errors
        .iter()
        .zip(analytic.iter())
        .map(|(e, a)| if a.abs() > 1e-10 { e / a.abs() } else { *e })
        .fold(0.0, f32::max);

    Ok(FDCheckReport {
        max_absolute_error: max_abs,
        rms_error: rms,
        max_relative_error: max_rel,
        n_checks: n,
    })
}

pub fn check_hessian<F, G>(
    analytical_hess: G,
    grad_f: F,
    x: &[f32],
    eps: f32,
    _tolerance: f32,
) -> Result<FDCheckReport>
where
    F: Fn(&[f32]) -> Vec<f32>,
    G: Fn(&[f32]) -> Vec<Vec<f32>>,
{
    let n = x.len();
    let analytic = analytical_hess(x);
    let mut abs_errors = Vec::new();

    for i in 0..n
    {
        for j in 0..n
        {
            let mut xp = x.to_vec();
            xp[j] += eps;
            let fp_i = grad_f(&xp)[i];

            let mut xm = x.to_vec();
            xm[j] -= eps;
            let fm_i = grad_f(&xm)[i];

            let fd = (fp_i - fm_i) / (2.0 * eps);
            abs_errors.push((analytic[i][j] - fd).abs());
        }
    }

    let n_checks = abs_errors.len();
    let max_abs = abs_errors.iter().copied().fold(0.0, f32::max);
    let rms = (abs_errors.iter().map(|e| e * e).sum::<f32>() / n_checks as f32).sqrt();

    Ok(FDCheckReport {
        max_absolute_error: max_abs,
        rms_error: rms,
        max_relative_error: max_abs,
        n_checks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_f(x: &[f32]) -> f32 {
        3.0 * x[0] * x[0] + 2.0 * x[0] * x[1] + x[1] * x[1]
    }

    fn quad_grad(x: &[f32]) -> Vec<f32> {
        vec![6.0 * x[0] + 2.0 * x[1], 2.0 * x[0] + 2.0 * x[1]]
    }

    #[test]
    fn test_gradient_check() {
        let report = check_gradient(quad_grad, quad_f, &[1.0, 2.0], 1e-3, 1e-3).unwrap();
        assert!(
            report.max_absolute_error < 1e-3,
            "gradient error too large: {}",
            report.max_absolute_error
        );
    }
}
