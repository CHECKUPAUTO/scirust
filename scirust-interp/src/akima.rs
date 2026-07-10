//! Akima piecewise-cubic interpolation.

use crate::error::InterpError;
use crate::traits::Interpolator;
use crate::util::{find_segment, hermite, validate_nodes};

/// Akima cubic-spline interpolant.
///
/// Akima's method chooses each node slope from a weighted combination of the
/// four surrounding secant slopes, which suppresses the oscillations a global
/// spline can show near sharp changes. The two missing secants at each end are
/// filled in by Akima's quadratic extrapolation rule, so **at least five nodes
/// are required**.
///
/// **Extrapolation** continues the boundary cubic piece: queries outside the
/// node range are evaluated with the Hermite polynomial of the nearest end
/// segment.
#[derive(Debug, Clone)]
pub struct AkimaSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Hermite slope at each node.
    d: Vec<f64>,
}

impl AkimaSpline {
    /// Build an Akima spline.
    ///
    /// Requires at least **five** nodes with strictly increasing, finite `xs`
    /// and finite `ys` of matching length; otherwise returns [`InterpError`].
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, InterpError> {
        validate_nodes(xs, ys, 5)?;
        let d = compute_slopes(xs, ys);
        Ok(Self {
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            d,
        })
    }
}

/// Akima node slopes for the given data.
fn compute_slopes(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    // Interval secant slopes m[0..n-1].
    let m: Vec<f64> = (0..n - 1)
        .map(|i| (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
        .collect();

    // Extend with two phantom slopes at each end (Akima's rule):
    //   s[k] corresponds to secant m_{k-2}, for k in 0..=n+1.
    let mut s = vec![0.0; n + 3];
    for (slot, &mi) in s[2..2 + m.len()].iter_mut().zip(m.iter()) {
        *slot = mi;
    }
    s[1] = 2.0 * s[2] - s[3]; // m_{-1}
    s[0] = 2.0 * s[1] - s[2]; // m_{-2}
    s[n + 1] = 2.0 * s[n] - s[n - 1]; // m_{n-1}
    s[n + 2] = 2.0 * s[n + 1] - s[n]; // m_{n}

    // Node i uses phantom-array entries s[i]..=s[i+3]
    // (= m_{i-2}, m_{i-1}, m_i, m_{i+1}).
    (0..n)
        .map(|i| {
            let w_right = (s[i + 3] - s[i + 2]).abs(); // |m_{i+1} - m_i|
            let w_left = (s[i + 1] - s[i]).abs(); // |m_{i-1} - m_{i-2}|
            let denom = w_right + w_left;
            if denom == 0.0 {
                // Four equal surrounding slopes → average of the two central.
                0.5 * (s[i + 1] + s[i + 2])
            } else {
                (w_right * s[i + 1] + w_left * s[i + 2]) / denom
            }
        })
        .collect()
}

impl Interpolator for AkimaSpline {
    fn eval(&self, x: f64) -> f64 {
        let i = find_segment(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        hermite(self.ys[i], self.ys[i + 1], self.d[i], self.d[i + 1], h, x - self.xs[i])
    }
}
