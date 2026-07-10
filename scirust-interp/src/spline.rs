//! Natural and clamped cubic-spline interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, thomas, validate_nodes};

/// A C² cubic spline through the given nodes.
///
/// The spline is built by solving a tridiagonal system (via the Thomas
/// algorithm) for the second derivatives — the *moments* — at each node, which
/// makes the interpolant twice continuously differentiable across interior
/// nodes. Two boundary conditions are offered:
///
/// * [`CubicSpline::natural`] sets the second derivative to zero at both ends.
/// * [`CubicSpline::clamped`] fixes the first derivative at both ends.
///
/// **Extrapolation** continues the boundary cubic piece: queries outside the
/// node range are evaluated with the polynomial of the nearest end segment.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives (moments) at each node.
    m: Vec<f64>,
}

impl CubicSpline {
    /// Build a natural cubic spline (`s''(x_0) = s''(x_{n-1}) = 0`).
    ///
    /// Requires at least two nodes with strictly increasing, finite `xs` and
    /// finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn natural(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        let m = solve_moments(xs, ys, BoundaryCondition::Natural);
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            m,
        })
    }

    /// Build a clamped cubic spline with prescribed end slopes.
    ///
    /// `dy0` is the required first derivative `s'(x_0)` and `dy_n` the required
    /// `s'(x_{n-1})`. Requires at least two nodes with strictly increasing,
    /// finite `xs` and finite `ys` of matching length; `dy0` and `dy_n` must be
    /// finite. Returns [`InterpError`] on any violation.
    pub fn clamped(xs: &[f64], ys: &[f64], dy0: f64, dy_n: f64) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        if !dy0.is_finite()
        {
            return Err(InterpError::NonFinite { index: 0 });
        }
        if !dy_n.is_finite()
        {
            return Err(InterpError::NonFinite {
                index: xs.len() - 1,
            });
        }
        let m = solve_moments(xs, ys, BoundaryCondition::Clamped { dy0, dy_n });
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            m,
        })
    }

    /// The computed second derivatives (moments) at each node.
    ///
    /// Exposed mainly so callers can verify boundary behaviour (a natural
    /// spline has `moments()[0] == moments()[n - 1] == 0`).
    pub fn moments(&self) -> &[f64] {
        &self.m
    }
}

/// Which boundary condition to impose when solving for the moments.
enum BoundaryCondition {
    /// Zero curvature at both ends.
    Natural,
    /// Prescribed first derivatives at both ends.
    Clamped { dy0: f64, dy_n: f64 },
}

/// Assemble and solve the tridiagonal moment system for the chosen boundary.
fn solve_moments(xs: &[f64], ys: &[f64], bc: BoundaryCondition) -> Vec<f64> {
    let n = xs.len();
    if n == 2
    {
        // A single segment: handle boundaries directly to keep the solver
        // uniform. Both boundary conditions reduce to a 2x2 system.
        return solve_moments_two(xs, ys, bc);
    }
    let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
    let slope: Vec<f64> = (0..n - 1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    for i in 1..n - 1
    {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 * (slope[i] - slope[i - 1]);
    }

    match bc
    {
        BoundaryCondition::Natural =>
        {
            b[0] = 1.0;
            c[0] = 0.0;
            d[0] = 0.0;
            a[n - 1] = 0.0;
            b[n - 1] = 1.0;
            d[n - 1] = 0.0;
        },
        BoundaryCondition::Clamped { dy0, dy_n } =>
        {
            b[0] = 2.0 * h[0];
            c[0] = h[0];
            d[0] = 6.0 * (slope[0] - dy0);
            a[n - 1] = h[n - 2];
            b[n - 1] = 2.0 * h[n - 2];
            d[n - 1] = 6.0 * (dy_n - slope[n - 2]);
        },
    }

    thomas(&a, &b, &c, &d)
}

/// Closed-form moments for the two-node special case.
fn solve_moments_two(xs: &[f64], ys: &[f64], bc: BoundaryCondition) -> Vec<f64> {
    match bc
    {
        BoundaryCondition::Natural => vec![0.0, 0.0],
        BoundaryCondition::Clamped { dy0, dy_n } =>
        {
            let h = xs[1] - xs[0];
            let slope = (ys[1] - ys[0]) / h;
            // 2x2 system:
            //   2h*M0 +  h*M1 = 6(slope - dy0)
            //    h*M0 + 2h*M1 = 6(dy_n - slope)
            let r0 = 6.0 * (slope - dy0);
            let r1 = 6.0 * (dy_n - slope);
            let det = (2.0 * h) * (2.0 * h) - h * h; // 3h^2
            let m0 = (r0 * (2.0 * h) - h * r1) / det;
            let m1 = ((2.0 * h) * r1 - h * r0) / det;
            vec![m0, m1]
        },
    }
}

impl Interpolator for CubicSpline {
    fn eval(&self, x: f64) -> f64 {
        let i = find_segment(&self.xs, x);
        let (x0, x1) = (self.xs[i], self.xs[i + 1]);
        let (y0, y1) = (self.ys[i], self.ys[i + 1]);
        let (m0, m1) = (self.m[i], self.m[i + 1]);
        let h = x1 - x0;
        let a = x1 - x;
        let b = x - x0;
        // Standard moment form of the cubic on [x0, x1].
        m0 * a * a * a / (6.0 * h)
            + m1 * b * b * b / (6.0 * h)
            + (y0 / h - m0 * h / 6.0) * a
            + (y1 / h - m1 * h / 6.0) * b
    }
}
