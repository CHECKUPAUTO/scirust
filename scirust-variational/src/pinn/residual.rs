#[derive(Debug, Clone)]
pub enum DifferentialOperator {
    /// ∂u/∂x (first derivative w.r.t. coordinate index)
    FirstDerivative(usize),
    /// ∂²u/∂x² (second derivative w.r.t. coordinate index)
    SecondDerivative(usize),
    /// ∇u (gradient magnitude squared)
    Gradient,
    /// Δu (Laplacian, sum of second derivatives)
    Laplacian,
    /// ∂u/∂t (time derivative)
    TimeDerivative,
}

impl std::fmt::Display for DifferentialOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::FirstDerivative(i) => write!(f, "d/dx{i}"),
            Self::SecondDerivative(i) => write!(f, "d²/dx{i}²"),
            Self::Gradient => write!(f, "∇"),
            Self::Laplacian => write!(f, "Δ"),
            Self::TimeDerivative => write!(f, "∂/∂t"),
        }
    }
}

pub fn central_difference_1d<F: Fn(f32) -> f32>(f: F, x: f32, h: f32) -> (f32, f32) {
    let fp = f(x + h);
    let fm = f(x - h);
    let f0 = f(x);
    let df = (fp - fm) / (2.0 * h);
    let d2f = (fp - 2.0 * f0 + fm) / (h * h);
    (df, d2f)
}

pub fn central_difference<F: Fn(&[f32]) -> f32>(
    f: F,
    x: &[f32],
    axis: usize,
    h: f32,
) -> (f32, f32) {
    let mut xp = x.to_vec();
    xp[axis] += h;
    let fp = f(&xp);

    let mut xm = x.to_vec();
    xm[axis] -= h;
    let fm = f(&xm);

    let f0 = f(x);

    let df = (fp - fm) / (2.0 * h);
    let d2f = (fp - 2.0 * f0 + fm) / (h * h);
    (df, d2f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_central_difference_sin() {
        let (df, d2f) = central_difference_1d(|x| x.sin(), 0.5, 1e-2);
        assert!((df - 0.5_f32.cos()).abs() < 1e-3);
        assert!((d2f + 0.5_f32.sin()).abs() < 1e-2);
    }

    #[test]
    fn test_central_difference_quadratic() {
        let (df, d2f) = central_difference_1d(|x| x * x, 2.0, 1e-2);
        assert!((df - 4.0).abs() < 1e-3);
        assert!((d2f - 2.0).abs() < 1e-2);
    }
}
