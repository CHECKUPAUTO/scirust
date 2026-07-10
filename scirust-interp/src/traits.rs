//! The common [`Interpolator`] trait implemented by every method.

/// A deterministic 1-D interpolant.
///
/// Implementors are built from a fixed set of nodes and then evaluated at
/// arbitrary query points. Evaluation never panics: queries outside the node
/// range extrapolate according to each method's documented natural convention.
pub trait Interpolator {
    /// Evaluate the interpolant at a single point `x`.
    ///
    /// For `x` inside the node range this returns the interpolated value; for
    /// `x` outside it, the value follows the method's extrapolation rule (see
    /// each type's documentation).
    fn eval(&self, x: f64) -> f64;

    /// Evaluate the interpolant at many points, returning one value per query.
    ///
    /// The default implementation simply maps [`Interpolator::eval`] over the
    /// slice; methods with cheaper batch strategies may override it.
    fn eval_all(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }
}
