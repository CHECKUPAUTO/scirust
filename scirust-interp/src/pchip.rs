//! Monotone piecewise cubic Hermite interpolation (PCHIP).

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, hermite, validate_nodes};

/// Piecewise cubic Hermite interpolant with Fritsch–Carlson slopes (PCHIP).
///
/// The node slopes are chosen by the Fritsch–Carlson rule so that the result
/// is shape-preserving: **strictly monotone input data yields a monotone
/// interpolant with no overshoot**, and local extrema occur only at nodes.
///
/// **Extrapolation** continues the boundary cubic piece: queries outside the
/// node range are evaluated with the Hermite polynomial of the nearest end
/// segment.
#[derive(Debug, Clone)]
pub struct PchipInterp {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Hermite slope (first derivative) at each node.
    d: Vec<f64>,
}

impl PchipInterp {
    /// Build a PCHIP interpolant.
    ///
    /// Requires at least two nodes with strictly increasing, finite `xs` and
    /// finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 2)?;
        let d = compute_slopes(xs, ys);
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            d,
        })
    }
}

/// Fritsch–Carlson node slopes for the given data.
fn compute_slopes(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
    let delta: Vec<f64> = (0..n - 1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

    if n == 2
    {
        // Two points: the monotone Hermite reduces to the straight line.
        return vec![delta[0], delta[0]];
    }

    let mut d = vec![0.0; n];

    // Interior nodes: weighted harmonic mean of the two adjacent secant slopes.
    for k in 1..n - 1
    {
        let dl = delta[k - 1];
        let dr = delta[k];
        if dl * dr <= 0.0
        {
            // Sign change or a zero secant → local extremum → flat.
            d[k] = 0.0;
        }
        else
        {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / dl + w2 / dr);
        }
    }

    d[0] = edge_slope(h[0], h[1], delta[0], delta[1]);
    d[n - 1] = edge_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    d
}

/// Three-valued sign with `sign(0) == 0` (unlike `f64::signum`).
fn sign(x: f64) -> f64 {
    if x > 0.0
    {
        1.0
    }
    else if x < 0.0
    {
        -1.0
    }
    else
    {
        0.0
    }
}

/// One-sided endpoint slope, clipped to preserve monotonicity.
///
/// `h0`/`m0` are the width and secant slope of the boundary interval and
/// `h1`/`m1` those of the adjacent interval (mirrored for the right end).
fn edge_slope(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let mut d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if sign(d) != sign(m0)
    {
        // Overshoot into the wrong direction → clamp flat.
        d = 0.0;
    }
    else if sign(m0) != sign(m1) && d.abs() > 3.0 * m0.abs()
    {
        // Bound the slope so the boundary cubic stays monotone.
        d = 3.0 * m0;
    }
    d
}

impl Interpolator for PchipInterp {
    fn eval(&self, x: f64) -> f64 {
        let i = find_segment(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        hermite(
            self.ys[i],
            self.ys[i + 1],
            self.d[i],
            self.d[i + 1],
            h,
            x - self.xs[i],
        )
    }
}
