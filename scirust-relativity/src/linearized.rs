//! Linearized gravity: the field equations to first order in a metric
//! perturbation about Minkowski.
//!
//! Writing `g_(mu nu) = eta_(mu nu) + h_(mu nu)` with `|h| << 1` and
//! `eta = diag(-1, +1, ..., +1)`, the curvature is linear in `h` to leading
//! order. This module computes, at a chart point and by central finite
//! differences of a perturbation sampler `h(x)`:
//!
//! - the linearized Riemann tensor (all indices down)
//!   `R^(1)_(mu nu rho sigma) = 1/2 (d_nu d_rho h_(mu sigma) + d_mu d_sigma h_(nu rho)
//!    - d_nu d_sigma h_(mu rho) - d_mu d_rho h_(nu sigma))`;
//! - the linearized Ricci tensor `R^(1)_(nu sigma) = eta^(mu rho) R^(1)_(mu nu rho sigma)`
//!   and scalar `R^(1) = eta^(nu sigma) R^(1)_(nu sigma)`;
//! - the linearized Einstein tensor `G^(1)_(mu nu) = R^(1)_(mu nu) - 1/2 eta_(mu nu) R^(1)`;
//! - the trace-reversed perturbation `hbar_(mu nu) = h_(mu nu) - 1/2 eta_(mu nu) h`,
//!   `h = eta^(mu nu) h_(mu nu)`, whose vacuum equation in the Lorenz gauge
//!   `d^mu hbar_(mu nu) = 0` is the wave equation `box hbar_(mu nu) = 0`
//!   (`G^(1)_(mu nu) = -1/2 box hbar_(mu nu)` there).
//!
//! This is established general relativity, the standard weak-field theory. The
//! second derivatives are numerical (central differences), so results carry the
//! same disclosed `O(step^2)` truncation as the geometry core's curvature
//! engine; for a polynomial perturbation the differences are exact. The
//! perturbation is supplied in coordinates where the background is
//! `eta = diag(-1, +1, ..., +1)` (Cartesian-like), so `h = g - eta` there.
//!
//! See `docs/LAYER_2_COVARIANT_GRAVITY.md` for the oracles that pin this down
//! (weak-field Schwarzschild is linearized-vacuum, the Newtonian Poisson limit,
//! gauge invariance of `R^(1)`, and an `O(h^2)` cross-check against the
//! nonlinear [`crate::CurvatureTensors`]).

use crate::RelativityError;

/// Minkowski signature entry `eta_(aa) = eta^(aa)`: `-1` for the timelike index
/// `0`, `+1` for the spacelike indices.
const fn signature(index: usize) -> f64 {
    if index == 0 { -1.0 } else { 1.0 }
}

/// The linearized curvature and field tensors at a chart point.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearizedField<const D: usize> {
    /// Linearized Riemann tensor `R^(1)_(mu nu rho sigma)`, all indices down.
    pub riemann: [[[[f64; D]; D]; D]; D],
    /// Linearized Ricci tensor `R^(1)_(mu nu)`.
    pub ricci: [[f64; D]; D],
    /// Linearized Ricci scalar `R^(1)`.
    pub ricci_scalar: f64,
    /// Linearized Einstein tensor `G^(1)_(mu nu)`.
    pub einstein: [[f64; D]; D],
    /// Trace-reversed perturbation `hbar_(mu nu)` at the point.
    pub trace_reversed: [[f64; D]; D],
}

impl<const D: usize> LinearizedField<D> {
    /// Compute the linearized field tensors for the perturbation `perturbation`
    /// (a sampler `h(x)` returning the covariant components `h_(mu nu)(x)`) at
    /// `coordinates`, using central-difference step `difference_step`.
    ///
    /// Returns [`RelativityError::NonFiniteCoordinate`] for a non-finite
    /// coordinate, [`RelativityError::InvalidDifferenceStep`] for a non-finite or
    /// non-positive step, and [`RelativityError::NonFiniteCurvatureComponent`] if
    /// any output is non-finite. It never panics.
    // Tensor-index arithmetic (symmetric fills, index contractions, diagonal
    // traces) reads most clearly with explicit indices than with iterator
    // adapters, matching the curvature engine's convention.
    #[allow(clippy::needless_range_loop)]
    pub fn compute(
        perturbation: impl Fn(&[f64; D]) -> [[f64; D]; D],
        coordinates: &[f64; D],
        difference_step: f64,
    ) -> Result<Self, RelativityError> {
        if let Some((index, _)) = coordinates
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(RelativityError::NonFiniteCoordinate(index));
        }
        if !difference_step.is_finite() || difference_step <= 0.0
        {
            return Err(RelativityError::InvalidDifferenceStep(difference_step));
        }

        let base = perturbation(coordinates);

        // second[a][b] = d_a d_b h  (a symmetric D x D tensor for each a, b).
        let mut second = [[[[0.0_f64; D]; D]; D]; D];
        for a in 0..D
        {
            for b in a..D
            {
                let tensor =
                    second_derivative(&perturbation, coordinates, &base, a, b, difference_step);
                second[a][b] = tensor;
                second[b][a] = tensor;
            }
        }

        // Linearized Riemann, all indices down.
        let mut riemann = [[[[0.0_f64; D]; D]; D]; D];
        for mu in 0..D
        {
            for nu in 0..D
            {
                for rho in 0..D
                {
                    for sigma in 0..D
                    {
                        let value = 0.5
                            * (second[nu][rho][mu][sigma] + second[mu][sigma][nu][rho]
                                - second[nu][sigma][mu][rho]
                                - second[mu][rho][nu][sigma]);
                        if !value.is_finite()
                        {
                            return Err(RelativityError::NonFiniteCurvatureComponent {
                                quantity: "linearized_riemann",
                            });
                        }
                        riemann[mu][nu][rho][sigma] = value;
                    }
                }
            }
        }

        // Ricci R_(nu sigma) = eta^(mu rho) R_(mu nu rho sigma); eta diagonal.
        let mut ricci = [[0.0_f64; D]; D];
        for nu in 0..D
        {
            for sigma in 0..D
            {
                let mut sum = 0.0;
                for m in 0..D
                {
                    sum += signature(m) * riemann[m][nu][m][sigma];
                }
                if !sum.is_finite()
                {
                    return Err(RelativityError::NonFiniteCurvatureComponent {
                        quantity: "linearized_ricci",
                    });
                }
                ricci[nu][sigma] = sum;
            }
        }

        // Scalar R = eta^(nu sigma) R_(nu sigma).
        let mut ricci_scalar = 0.0;
        for n in 0..D
        {
            ricci_scalar += signature(n) * ricci[n][n];
        }
        if !ricci_scalar.is_finite()
        {
            return Err(RelativityError::NonFiniteCurvatureComponent {
                quantity: "linearized_ricci_scalar",
            });
        }

        // Einstein G_(mu nu) = R_(mu nu) - 1/2 eta_(mu nu) R.
        let mut einstein = [[0.0_f64; D]; D];
        for mu in 0..D
        {
            for nu in 0..D
            {
                let eta = if mu == nu { signature(mu) } else { 0.0 };
                einstein[mu][nu] = ricci[mu][nu] - 0.5 * eta * ricci_scalar;
            }
        }

        // Trace-reversed hbar_(mu nu) = h_(mu nu) - 1/2 eta_(mu nu) h.
        let mut trace = 0.0;
        for a in 0..D
        {
            trace += signature(a) * base[a][a];
        }
        let mut trace_reversed = [[0.0_f64; D]; D];
        for mu in 0..D
        {
            for nu in 0..D
            {
                let eta = if mu == nu { signature(mu) } else { 0.0 };
                trace_reversed[mu][nu] = base[mu][nu] - 0.5 * eta * trace;
            }
        }

        Ok(Self {
            riemann,
            ricci,
            ricci_scalar,
            einstein,
            trace_reversed,
        })
    }
}

/// Second partial derivative `d_a d_b h` of the perturbation by central
/// differences (`base = h(coordinates)` supplied to avoid recomputing it).
fn second_derivative<const D: usize>(
    perturbation: &impl Fn(&[f64; D]) -> [[f64; D]; D],
    coordinates: &[f64; D],
    base: &[[f64; D]; D],
    a: usize,
    b: usize,
    step: f64,
) -> [[f64; D]; D] {
    let mut result = [[0.0_f64; D]; D];
    if a == b
    {
        let forward = perturbation(&shifted(coordinates, a, step));
        let backward = perturbation(&shifted(coordinates, a, -step));
        for (mu, row) in result.iter_mut().enumerate()
        {
            for (nu, value) in row.iter_mut().enumerate()
            {
                *value = (forward[mu][nu] - 2.0 * base[mu][nu] + backward[mu][nu]) / (step * step);
            }
        }
    }
    else
    {
        let plus_plus = perturbation(&shifted2(coordinates, a, step, b, step));
        let plus_minus = perturbation(&shifted2(coordinates, a, step, b, -step));
        let minus_plus = perturbation(&shifted2(coordinates, a, -step, b, step));
        let minus_minus = perturbation(&shifted2(coordinates, a, -step, b, -step));
        for (mu, row) in result.iter_mut().enumerate()
        {
            for (nu, value) in row.iter_mut().enumerate()
            {
                *value = (plus_plus[mu][nu] - plus_minus[mu][nu] - minus_plus[mu][nu]
                    + minus_minus[mu][nu])
                    / (4.0 * step * step);
            }
        }
    }
    result
}

/// `coordinates` with `coordinates[index]` shifted by `delta`.
fn shifted<const D: usize>(coordinates: &[f64; D], index: usize, delta: f64) -> [f64; D] {
    let mut shifted = *coordinates;
    shifted[index] += delta;
    shifted
}

/// `coordinates` with two independent coordinate shifts applied.
fn shifted2<const D: usize>(
    coordinates: &[f64; D],
    first: usize,
    first_delta: f64,
    second: usize,
    second_delta: f64,
) -> [f64; D] {
    let mut shifted = *coordinates;
    shifted[first] += first_delta;
    shifted[second] += second_delta;
    shifted
}
