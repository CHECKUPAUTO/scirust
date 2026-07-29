//! Uniform periodic grids in `D` dimensions, and centred finite differences.
//!
//! This module is deliberately free of any relativity: it is the discretisation
//! substrate that [`crate::bssn_grid`] builds on, and nothing here knows what a
//! metric is.
//!
//! It began as a one-dimensional grid. Generalising it to `D` axes was what a
//! second spatial dimension required — and it was *all* that a second spatial
//! dimension required, because [`crate::bssn`] takes complete 3x3 derivative
//! tensors and never assumes how many of the axes actually vary. The third
//! dimension then needed nothing at all from this module: [`UniformGrid3d`] is
//! an alias, and the stencils were already written for arbitrary axis pairs.
//! [`UniformGrid1d`] remains as an alias with its original scalar API intact.
//!
//! # Convention
//!
//! Every axis is **half-open**. Along axis `a` with `N_a` points on
//! `[lower_a, upper_a)`:
//!
//! ```text
//! x_n = lower_a + n * dx_a,   n = 0, ..., N_a - 1,   dx_a = (upper_a - lower_a) / N_a
//! ```
//!
//! The upper endpoint is **not** stored — it is the periodic image of the lower
//! one. Storing both would duplicate a degree of freedom.
//!
//! # Index space
//!
//! Points carry a single **linear index** in `[0, total_points)`, with axis `0`
//! varying fastest:
//!
//! ```text
//! linear = n_0 + N_0 * ( n_1 + N_1 * ( n_2 + ... ) )
//! ```
//!
//! Fixing axis 0 as the fastest keeps a one-dimensional grid's linear index
//! identical to its point index, so the `D = 1` behaviour is unchanged
//! bit-for-bit. Iterating `0..total_points` visits points in a deterministic
//! order in any dimension.
//!
//! # Periodic indexing
//!
//! Neighbour lookup wraps with Euclidean remainder, never with unsigned integer
//! wraparound, so the neighbour below index `0` along any axis is `N_a - 1` and
//! no offset of any sign can underflow.
//!
//! # Finite differences
//!
//! Second-order centred, per axis:
//!
//! ```text
//! D1_a f = ( f(+1_a) - f(-1_a) ) / ( 2 dx_a )
//! D2_a f = ( f(+1_a) - 2 f + f(-1_a) ) / ( dx_a^2 )
//! ```
//!
//! and, for `a != b`, the four-corner **mixed** difference:
//!
//! ```text
//! D_ab f = ( f(+1_a,+1_b) - f(+1_a,-1_b) - f(-1_a,+1_b) + f(-1_a,-1_b) )
//!          / ( 4 dx_a dx_b )
//! ```
//!
//! The mixed stencil has no one-dimensional analogue, and it is the term BSSN's
//! principal-part restructuring exists to control — so a one-dimensional grid
//! could never exercise the very thing that makes BSSN worth using.
//!
//! All three are validated against analytic trigonometric oracles; the observed
//! order is measured, not assumed from the formula.
//!
//! ```
//! use scirust_relativity::grid::{UniformGrid1d, periodic_first_derivative};
//!
//! let grid = UniformGrid1d::new(64, 0.0, 1.0).unwrap();
//! let k = 2.0 * std::f64::consts::PI;
//! let samples: Vec<f64> = (0..grid.points()).map(|i| (k * grid.coordinate(i)).sin()).collect();
//! let numerical = periodic_first_derivative(&samples, &grid, 8).unwrap();
//! let exact = k * (k * grid.coordinate(8)).cos();
//! assert!((numerical - exact).abs() < 1.0e-2);
//! ```

use core::fmt;

/// The minimum number of points per axis a second-order centred stencil needs.
///
/// Three points is the smallest count on which `f(-1)`, `f`, and `f(+1)` are
/// not all the same sample under periodic wrapping.
pub const MINIMUM_STENCIL_POINTS: usize = 3;

/// A failure constructing or using a grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridError {
    /// Fewer points along some axis than the centred stencil requires.
    InsufficientPoints {
        /// Which axis.
        axis: usize,
        /// The requested point count.
        requested: usize,
        /// The minimum the stencil needs.
        minimum: usize,
    },
    /// A domain bound is not finite.
    NonFiniteBound {
        /// Which axis.
        axis: usize,
        /// Which bound: `"lower"` or `"upper"`.
        bound: &'static str,
    },
    /// A domain length is not strictly positive.
    NonPositiveLength {
        /// Which axis.
        axis: usize,
    },
    /// A derived spacing is not finite and strictly positive.
    InvalidSpacing {
        /// Which axis.
        axis: usize,
    },
    /// A sample array's length does not match the grid.
    LengthMismatch {
        /// The grid's total point count.
        expected: usize,
        /// The supplied array's length.
        actual: usize,
    },
    /// An axis index is out of range for this grid's dimensionality.
    AxisOutOfRange {
        /// The requested axis.
        axis: usize,
        /// The grid's dimensionality.
        dimensions: usize,
    },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InsufficientPoints {
                axis,
                requested,
                minimum,
            } => write!(
                f,
                "axis {axis} needs at least {minimum} points for a centred stencil, got {requested}"
            ),
            Self::NonFiniteBound { axis, bound } =>
            {
                write!(f, "{bound} bound of axis {axis} is not finite")
            },
            Self::NonPositiveLength { axis } =>
            {
                write!(f, "length of axis {axis} must be strictly positive")
            },
            Self::InvalidSpacing { axis } =>
            {
                write!(
                    f,
                    "spacing of axis {axis} must be finite and strictly positive"
                )
            },
            Self::LengthMismatch { expected, actual } =>
            {
                write!(f, "expected {expected} samples for this grid, got {actual}")
            },
            Self::AxisOutOfRange { axis, dimensions } =>
            {
                write!(
                    f,
                    "axis {axis} is out of range for a {dimensions}-dimensional grid"
                )
            },
        }
    }
}

impl std::error::Error for GridError {}

/// The pre-generalisation name for [`GridError`].
pub type Grid1dError = GridError;

/// A uniform, half-open, periodic grid in `D` dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformGrid<const D: usize> {
    points: [usize; D],
    lower: [f64; D],
    spacing: [f64; D],
}

/// A one-dimensional uniform periodic grid.
pub type UniformGrid1d = UniformGrid<1>;

/// A two-dimensional uniform periodic grid.
pub type UniformGrid2d = UniformGrid<2>;

/// A three-dimensional uniform periodic grid.
pub type UniformGrid3d = UniformGrid<3>;

impl<const D: usize> UniformGrid<D> {
    /// Build a grid spanning the half-open box `[lower, upper)` with `points`
    /// samples along each axis.
    ///
    /// The upper bounds are *not* stored points: they are periodic images of the
    /// lower ones, so each spacing is `(upper - lower) / points`, not
    /// `(upper - lower) / (points - 1)`.
    pub fn from_axes(
        points: [usize; D],
        lower: [f64; D],
        upper: [f64; D],
    ) -> Result<Self, GridError> {
        let mut spacing = [0.0_f64; D];
        for axis in 0..D
        {
            if points[axis] < MINIMUM_STENCIL_POINTS
            {
                return Err(GridError::InsufficientPoints {
                    axis,
                    requested: points[axis],
                    minimum: MINIMUM_STENCIL_POINTS,
                });
            }
            if !lower[axis].is_finite()
            {
                return Err(GridError::NonFiniteBound {
                    axis,
                    bound: "lower",
                });
            }
            if !upper[axis].is_finite()
            {
                return Err(GridError::NonFiniteBound {
                    axis,
                    bound: "upper",
                });
            }
            // Both bounds are finite, so the difference is finite and a plain
            // comparison is exact here.
            let length = upper[axis] - lower[axis];
            if length <= 0.0
            {
                return Err(GridError::NonPositiveLength { axis });
            }
            let step = length / points[axis] as f64;
            if !step.is_finite() || step <= 0.0
            {
                return Err(GridError::InvalidSpacing { axis });
            }
            spacing[axis] = step;
        }
        Ok(Self {
            points,
            lower,
            spacing,
        })
    }

    /// The number of spatial dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> usize {
        D
    }

    /// The total number of stored points, `prod_a N_a`.
    #[must_use]
    pub fn total_points(&self) -> usize {
        let mut total = 1_usize;
        let mut axis = 0;
        while axis < D
        {
            total *= self.points[axis];
            axis += 1;
        }
        total
    }

    /// The number of points along `axis`.
    #[must_use]
    pub fn points_along(&self, axis: usize) -> usize {
        self.points[axis]
    }

    /// The common spacing, when every axis shares one.
    ///
    /// Returns `None` for an anisotropic grid. Callers that difference along
    /// several axes with a single step size need this: a step that lands on grid
    /// points along one axis lands between them along an axis with a different
    /// spacing.
    #[must_use]
    pub fn uniform_spacing(&self) -> Option<f64> {
        let first = self.spacing[0];
        for axis in 1..D
        {
            if self.spacing[axis] != first
            {
                return None;
            }
        }
        Some(first)
    }

    /// The spacing along `axis`.
    #[must_use]
    pub fn spacing_along(&self, axis: usize) -> f64 {
        self.spacing[axis]
    }

    /// The lower bound of `axis` (a stored coordinate).
    #[must_use]
    pub fn lower_along(&self, axis: usize) -> f64 {
        self.lower[axis]
    }

    /// The periodic length of `axis`.
    #[must_use]
    pub fn length_along(&self, axis: usize) -> f64 {
        self.spacing[axis] * self.points[axis] as f64
    }

    /// The upper bound of `axis` (**not** a stored coordinate).
    #[must_use]
    pub fn upper_along(&self, axis: usize) -> f64 {
        self.lower[axis] + self.length_along(axis)
    }

    /// The stride between consecutive indices along `axis` in the linear index.
    #[must_use]
    pub fn stride(&self, axis: usize) -> usize {
        let mut stride = 1_usize;
        for lower_axis in 0..axis
        {
            stride *= self.points[lower_axis];
        }
        stride
    }

    /// Decompose a linear index into per-axis indices.
    #[must_use]
    pub fn multi_index(&self, linear: usize) -> [usize; D] {
        let mut remaining = linear % self.total_points();
        let mut multi = [0_usize; D];
        // Indexed rather than iterated: the loop walks axes and consumes the
        // remaining linear index in the same order, which the index expresses.
        #[allow(clippy::needless_range_loop)]
        for axis in 0..D
        {
            multi[axis] = remaining % self.points[axis];
            remaining /= self.points[axis];
        }
        multi
    }

    /// Compose per-axis indices into a linear index, wrapping each axis.
    #[must_use]
    pub fn linear_index(&self, multi: &[usize; D]) -> usize {
        let mut linear = 0_usize;
        for axis in (0..D).rev()
        {
            linear = linear * self.points[axis] + (multi[axis] % self.points[axis]);
        }
        linear
    }

    /// The linear index `delta` steps from `linear` along `axis`, wrapped
    /// periodically.
    ///
    /// `delta` may be negative and of any magnitude; the Euclidean remainder
    /// makes underflow impossible.
    #[must_use]
    pub fn neighbour(&self, linear: usize, axis: usize, delta: isize) -> usize {
        let mut multi = self.multi_index(linear);
        let count = self.points[axis] as isize;
        let shifted = multi[axis] as isize + delta;
        multi[axis] = shifted.rem_euclid(count) as usize;
        self.linear_index(&multi)
    }

    /// The physical position of a stored point, padded with zeros on the axes
    /// this grid does not span.
    ///
    /// Padding rather than erroring is deliberate: the relativity layer always
    /// works with three-component positions, and an axis the grid does not span
    /// is one along which nothing varies, so any fixed value is as good as
    /// another. Zero is the conventional choice.
    #[must_use]
    pub fn position(&self, linear: usize) -> [f64; 3] {
        let multi = self.multi_index(linear);
        let mut position = [0.0_f64; 3];
        for axis in 0..D.min(3)
        {
            position[axis] = self.lower[axis] + multi[axis] as f64 * self.spacing[axis];
        }
        position
    }

    /// The stored point nearest `position`, wrapped periodically on every axis.
    ///
    /// Components beyond this grid's dimensionality are ignored: nothing varies
    /// along them, so they cannot select a different point.
    ///
    /// Rounding — rather than truncation — is what makes a sample taken at
    /// exactly one spacing away land on the neighbouring index despite
    /// floating-point representation error.
    #[must_use]
    pub fn nearest_index(&self, position: &[f64; 3]) -> usize {
        let mut multi = [0_usize; D];
        for axis in 0..D.min(3)
        {
            let value = position[axis];
            if !value.is_finite()
            {
                multi[axis] = 0;
                continue;
            }
            let raw = ((value - self.lower[axis]) / self.spacing[axis]).round();
            let count = self.points[axis] as f64;
            let wrapped = raw - (raw / count).floor() * count;
            multi[axis] = (wrapped as usize) % self.points[axis];
        }
        self.linear_index(&multi)
    }

    fn check_length(&self, samples: &[f64]) -> Result<(), GridError> {
        if samples.len() != self.total_points()
        {
            return Err(GridError::LengthMismatch {
                expected: self.total_points(),
                actual: samples.len(),
            });
        }
        Ok(())
    }

    fn check_axis(&self, axis: usize) -> Result<(), GridError> {
        if axis >= D
        {
            return Err(GridError::AxisOutOfRange {
                axis,
                dimensions: D,
            });
        }
        Ok(())
    }
}

impl UniformGrid<1> {
    /// Build a one-dimensional grid of `points` samples on `[lower, upper)`.
    pub fn new(points: usize, lower: f64, upper: f64) -> Result<Self, GridError> {
        Self::from_axes([points], [lower], [upper])
    }

    /// The number of stored points.
    #[must_use]
    pub fn points(&self) -> usize {
        self.points_along(0)
    }

    /// The spacing `dx`.
    #[must_use]
    pub fn spacing(&self) -> f64 {
        self.spacing_along(0)
    }

    /// The lower domain bound (a stored point).
    #[must_use]
    pub fn lower(&self) -> f64 {
        self.lower_along(0)
    }

    /// The upper domain bound (**not** a stored point).
    #[must_use]
    pub fn upper(&self) -> f64 {
        self.upper_along(0)
    }

    /// The periodic domain length.
    #[must_use]
    pub fn length(&self) -> f64 {
        self.length_along(0)
    }

    /// The coordinate of a stored point, wrapped periodically.
    #[must_use]
    pub fn coordinate(&self, index: usize) -> f64 {
        self.position(index % self.points())[0]
    }

    /// Wrap an index into `[0, N)`.
    #[must_use]
    pub fn wrap_usize(&self, index: usize) -> usize {
        index % self.points()
    }

    /// The index `delta` steps away, wrapped periodically.
    #[must_use]
    pub fn offset(&self, index: usize, delta: isize) -> usize {
        self.neighbour(index % self.points(), 0, delta)
    }
}

/// The second-order centred periodic first derivative along `axis`.
pub fn periodic_first_derivative_along<const D: usize>(
    samples: &[f64],
    grid: &UniformGrid<D>,
    linear: usize,
    axis: usize,
) -> Result<f64, GridError> {
    grid.check_length(samples)?;
    grid.check_axis(axis)?;
    let back = samples[grid.neighbour(linear, axis, -1)];
    let forward = samples[grid.neighbour(linear, axis, 1)];
    Ok((forward - back) / (2.0 * grid.spacing_along(axis)))
}

/// The second-order centred periodic second derivative along `axis`.
pub fn periodic_second_derivative_along<const D: usize>(
    samples: &[f64],
    grid: &UniformGrid<D>,
    linear: usize,
    axis: usize,
) -> Result<f64, GridError> {
    grid.check_length(samples)?;
    grid.check_axis(axis)?;
    let centre = samples[linear % grid.total_points()];
    let back = samples[grid.neighbour(linear, axis, -1)];
    let forward = samples[grid.neighbour(linear, axis, 1)];
    let step = grid.spacing_along(axis);
    Ok((forward - 2.0 * centre + back) / (step * step))
}

/// The second-order centred periodic **mixed** second derivative
/// `d_first d_second`, from the four-corner stencil.
///
/// For `first == second` this delegates to
/// [`periodic_second_derivative_along`], so a caller can form a full Hessian
/// without special-casing the diagonal.
///
/// The mixed stencil has no one-dimensional analogue. It is the derivative BSSN
/// removes from its principal part, so it is the term a second spatial dimension
/// exists to exercise.
pub fn periodic_mixed_derivative<const D: usize>(
    samples: &[f64],
    grid: &UniformGrid<D>,
    linear: usize,
    first: usize,
    second: usize,
) -> Result<f64, GridError> {
    if first == second
    {
        return periodic_second_derivative_along(samples, grid, linear, first);
    }
    grid.check_length(samples)?;
    grid.check_axis(first)?;
    grid.check_axis(second)?;

    // Canonicalise the axis order. The four corners are the same samples either
    // way, but floating-point addition is not associative, so summing them in
    // the caller's order would make `d_a d_b` and `d_b d_a` differ in the last
    // bit. A Hessian built from those would not be exactly symmetric, and the
    // asymmetry would propagate into `Rtilde_ij`, which must be. Sorting makes
    // the result bit-for-bit symmetric by construction.
    let (first, second) = if first <= second
    {
        (first, second)
    }
    else
    {
        (second, first)
    };

    let forward_first = grid.neighbour(linear, first, 1);
    let back_first = grid.neighbour(linear, first, -1);
    let plus_plus = samples[grid.neighbour(forward_first, second, 1)];
    let plus_minus = samples[grid.neighbour(forward_first, second, -1)];
    let minus_plus = samples[grid.neighbour(back_first, second, 1)];
    let minus_minus = samples[grid.neighbour(back_first, second, -1)];

    let denominator = 4.0 * grid.spacing_along(first) * grid.spacing_along(second);
    Ok((plus_plus - plus_minus - minus_plus + minus_minus) / denominator)
}

/// The centred periodic first derivative of a one-dimensional sample array.
pub fn periodic_first_derivative(
    samples: &[f64],
    grid: &UniformGrid1d,
    index: usize,
) -> Result<f64, GridError> {
    periodic_first_derivative_along(samples, grid, index, 0)
}

/// The centred periodic second derivative of a one-dimensional sample array.
pub fn periodic_second_derivative(
    samples: &[f64],
    grid: &UniformGrid1d,
    index: usize,
) -> Result<f64, GridError> {
    periodic_second_derivative_along(samples, grid, index, 0)
}

/// Fill `out` with the centred periodic first derivative along `axis`.
///
/// Deterministic ascending linear-index order.
pub fn periodic_first_derivative_all_along<const D: usize>(
    samples: &[f64],
    grid: &UniformGrid<D>,
    axis: usize,
    out: &mut [f64],
) -> Result<(), GridError> {
    grid.check_length(samples)?;
    grid.check_length(out)?;
    grid.check_axis(axis)?;
    let scale = 1.0 / (2.0 * grid.spacing_along(axis));
    for linear in 0..grid.total_points()
    {
        out[linear] = (samples[grid.neighbour(linear, axis, 1)]
            - samples[grid.neighbour(linear, axis, -1)])
            * scale;
    }
    Ok(())
}

/// Fill `out` with the centred periodic second derivative along `axis`.
///
/// Deterministic ascending linear-index order.
pub fn periodic_second_derivative_all_along<const D: usize>(
    samples: &[f64],
    grid: &UniformGrid<D>,
    axis: usize,
    out: &mut [f64],
) -> Result<(), GridError> {
    grid.check_length(samples)?;
    grid.check_length(out)?;
    grid.check_axis(axis)?;
    let step = grid.spacing_along(axis);
    let scale = 1.0 / (step * step);
    for linear in 0..grid.total_points()
    {
        out[linear] = (samples[grid.neighbour(linear, axis, 1)] - 2.0 * samples[linear]
            + samples[grid.neighbour(linear, axis, -1)])
            * scale;
    }
    Ok(())
}

/// Fill `out` with the centred periodic first derivative of a one-dimensional
/// sample array.
pub fn periodic_first_derivative_all(
    samples: &[f64],
    grid: &UniformGrid1d,
    out: &mut [f64],
) -> Result<(), GridError> {
    periodic_first_derivative_all_along(samples, grid, 0, out)
}

/// Fill `out` with the centred periodic second derivative of a one-dimensional
/// sample array.
pub fn periodic_second_derivative_all(
    samples: &[f64],
    grid: &UniformGrid1d,
    out: &mut [f64],
) -> Result<(), GridError> {
    periodic_second_derivative_all_along(samples, grid, 0, out)
}

/// A deterministic reduction of a residual field over the grid.
///
/// Every component is reported; nothing is blended into a single score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridReduction {
    /// The signed arithmetic mean.
    pub signed_mean: f64,
    /// The discrete `L1` norm `(1/N) sum |f_i|`.
    pub l1: f64,
    /// The discrete `L2` norm `sqrt( (1/N) sum f_i^2 )`.
    pub l2: f64,
    /// The maximum absolute value.
    pub max_abs: f64,
    /// The lowest linear index attaining `max_abs` (ties resolved
    /// deterministically).
    pub max_index: usize,
    /// How many samples were not finite.
    pub non_finite: usize,
}

impl GridReduction {
    /// Reduce `samples` in ascending linear-index order.
    ///
    /// Non-finite samples are counted rather than propagated into the norms, so
    /// a single `NaN` reports itself instead of erasing every other diagnostic.
    #[must_use]
    pub fn of(samples: &[f64]) -> Self {
        let mut sum = 0.0_f64;
        let mut absolute = 0.0_f64;
        let mut squared = 0.0_f64;
        let mut max_abs = 0.0_f64;
        let mut max_index = 0_usize;
        let mut non_finite = 0_usize;
        let mut counted = 0_usize;

        for (index, &value) in samples.iter().enumerate()
        {
            if !value.is_finite()
            {
                non_finite += 1;
                continue;
            }
            counted += 1;
            sum += value;
            absolute += value.abs();
            squared += value * value;
            if value.abs() > max_abs
            {
                max_abs = value.abs();
                max_index = index;
            }
        }

        let denominator = if counted == 0 { 1.0 } else { counted as f64 };
        Self {
            signed_mean: sum / denominator,
            l1: absolute / denominator,
            l2: (squared / denominator).sqrt(),
            max_abs,
            max_index,
            non_finite,
        }
    }
}
