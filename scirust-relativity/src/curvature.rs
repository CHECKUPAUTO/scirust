//! Curvature tensors of a fixed background.
//!
//! From any background that provides both a [`Metric`] and a [`Connection`],
//! this module evaluates the Riemann tensor, the Ricci tensor, the Ricci
//! scalar, the Einstein tensor, and the Kretschmann invariant, using central
//! finite differences of the Christoffel symbols.
//!
//! ## Conventions (signature `(-,+,+,+)`)
//!
//! ```text
//! R^rho_(sigma mu nu) = d_mu Gamma^rho_(nu sigma) - d_nu Gamma^rho_(mu sigma)
//!                     + Gamma^rho_(mu lambda) Gamma^lambda_(nu sigma)
//!                     - Gamma^rho_(nu lambda) Gamma^lambda_(mu sigma)
//! R_(sigma nu) = R^rho_(sigma rho nu)          (Ricci)
//! R            = g^(sigma nu) R_(sigma nu)      (Ricci scalar)
//! G_(mu nu)    = R_(mu nu) - 1/2 R g_(mu nu)    (Einstein)
//! K            = R_(alpha beta gamma delta) R^(alpha beta gamma delta)
//!                                              (Kretschmann)
//! ```
//!
//! The Christoffel derivatives `d_mu Gamma` are central finite differences of
//! [`Connection::christoffel`]. For a background with an *analytic* connection
//! (Schwarzschild, Reissner-Nordström, de Sitter, anti-de Sitter) this is a
//! single finite-difference layer and is second-order accurate; for a
//! background whose connection is itself a finite difference (Kerr) the result
//! is a nested difference and correspondingly noisier. Results are therefore
//! numerical approximations validated against exact analytic oracles to a
//! stated tolerance (see this crate's curvature tests), except where a value
//! is exactly zero by construction (flat spacetime).

use crate::{Connection, Metric, RelativityError, invert_metric, numerical_christoffel};

/// The curvature tensors of a background at one point.
///
/// All tensors use the conventions in the [module documentation](self).
/// `riemann` stores `R^rho_(sigma mu nu)` indexed `[rho][sigma][mu][nu]`;
/// `ricci` and `einstein` store the covariant `R_(mu nu)` and `G_(mu nu)`
/// indexed `[mu][nu]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureTensors<const D: usize> {
    riemann: [[[[f64; D]; D]; D]; D],
    ricci: [[f64; D]; D],
    ricci_scalar: f64,
    einstein: [[f64; D]; D],
    kretschmann: f64,
}

impl<const D: usize> CurvatureTensors<D> {
    /// Evaluate all curvature tensors of `background` at `coordinates`, using
    /// central finite differences of the Christoffel symbols with step
    /// `difference_step`.
    ///
    /// Returns a typed [`RelativityError`] for non-finite coordinates, an
    /// invalid step, a singular metric, or any non-finite intermediate or
    /// output component; it never panics and never silently returns a
    /// non-finite result.
    ///
    /// # Example
    ///
    /// De Sitter spacetime is maximally symmetric, so its Ricci scalar is
    /// exactly `4 * Lambda`; the finite-difference engine recovers it to the
    /// stated tolerance.
    ///
    /// ```
    /// use scirust_relativity::{CurvatureTensors, DeSitter};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let lambda = 0.03;
    /// let spacetime = DeSitter::try_new(lambda).expect("valid cosmological constant");
    /// let curvature =
    ///     CurvatureTensors::compute(&spacetime, &[0.0, 3.0, FRAC_PI_2, 0.0], 1.0e-4)
    ///         .expect("finite curvature in the static patch");
    ///
    /// assert!((curvature.ricci_scalar() - 4.0 * lambda).abs() < 1.0e-5);
    /// ```
    pub fn compute<B>(
        background: &B,
        coordinates: &[f64; D],
        difference_step: f64,
    ) -> Result<Self, RelativityError>
    where
        B: Metric<D> + Connection<D>,
    {
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

        let metric = background.components(coordinates);
        let inverse = invert_metric(&metric)?;
        let christoffel = background.christoffel(coordinates);

        let christoffel_derivatives =
            christoffel_derivatives(background, coordinates, difference_step)?;

        let riemann = riemann_from_connection(&christoffel, &christoffel_derivatives);
        validate_tensor4("riemann", &riemann)?;

        let ricci = ricci_from_riemann(&riemann);
        validate_tensor2("ricci", &ricci)?;

        let ricci_scalar = contract_scalar(&inverse, &ricci);
        if !ricci_scalar.is_finite()
        {
            return Err(RelativityError::NonFiniteCurvatureComponent {
                quantity: "ricci_scalar",
            });
        }

        let einstein = einstein_from_ricci(&ricci, ricci_scalar, &metric);
        validate_tensor2("einstein", &einstein)?;

        let kretschmann = kretschmann_from_riemann(&riemann, &metric, &inverse);
        if !kretschmann.is_finite()
        {
            return Err(RelativityError::NonFiniteCurvatureComponent {
                quantity: "kretschmann",
            });
        }

        Ok(Self {
            riemann,
            ricci,
            ricci_scalar,
            einstein,
            kretschmann,
        })
    }

    /// The Riemann tensor `R^rho_(sigma mu nu)`, indexed `[rho][sigma][mu][nu]`.
    #[must_use]
    pub const fn riemann(&self) -> &[[[[f64; D]; D]; D]; D] {
        &self.riemann
    }

    /// The covariant Ricci tensor `R_(mu nu)`, indexed `[mu][nu]`.
    #[must_use]
    pub const fn ricci(&self) -> &[[f64; D]; D] {
        &self.ricci
    }

    /// The Ricci scalar `R = g^(mu nu) R_(mu nu)`.
    #[must_use]
    pub const fn ricci_scalar(&self) -> f64 {
        self.ricci_scalar
    }

    /// The covariant Einstein tensor `G_(mu nu) = R_(mu nu) - 1/2 R g_(mu nu)`.
    #[must_use]
    pub const fn einstein(&self) -> &[[f64; D]; D] {
        &self.einstein
    }

    /// The Kretschmann invariant `K = R_(abcd) R^(abcd)`.
    #[must_use]
    pub const fn kretschmann(&self) -> f64 {
        self.kretschmann
    }
}

/// The Ricci tensor `R_(mu nu)` of a metric from its **components alone**, by a
/// nested central difference: the Christoffel symbols are built from
/// differences of the metric (step `metric_step`, via [`numerical_christoffel`])
/// and *those* are differenced again (step `connection_step`) for `d Gamma`,
/// reusing the same Riemann -> Ricci assembly as [`CurvatureTensors::compute`].
///
/// Unlike [`CurvatureTensors::compute`], which differentiates an *analytic*
/// [`Connection`], this needs no connection — it is the curvature of a bare
/// metric field, one finite-difference layer deeper and correspondingly noisier.
/// [`ricci_scalar_from_metric`] is a thin wrapper contracting this tensor with
/// the inverse metric; the Layer 3 ADM evolution equations need the full tensor
/// (see `docs/LAYER_3_ADM_EVOLUTION.md`), not only its trace.
///
/// Returns a typed [`RelativityError`] for non-finite coordinates, an invalid
/// step, a singular metric, or any non-finite intermediate; it never panics and
/// never silently returns a non-finite result.
///
/// # Example
///
/// De Sitter is maximally symmetric, so its Ricci tensor is exactly
/// `Lambda * g_(mu nu)`.
///
/// ```
/// use scirust_relativity::{DeSitter, Metric, ricci_tensor_from_metric};
/// use std::f64::consts::FRAC_PI_2;
///
/// let lambda = 0.03;
/// let spacetime = DeSitter::try_new(lambda).expect("valid cosmological constant");
/// let point = [0.0, 3.0, FRAC_PI_2, 0.0];
/// let ricci = ricci_tensor_from_metric(&spacetime, &point, 1.0e-3, 1.0e-3).expect("finite");
/// let metric = spacetime.components(&point);
/// assert!((ricci[1][1] - lambda * metric[1][1]).abs() < 1.0e-4);
/// ```
pub fn ricci_tensor_from_metric<M, const D: usize>(
    metric: &M,
    coordinates: &[f64; D],
    connection_step: f64,
    metric_step: f64,
) -> Result<[[f64; D]; D], RelativityError>
where
    M: Metric<D>,
{
    if let Some((index, _)) = coordinates
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RelativityError::NonFiniteCoordinate(index));
    }
    if !connection_step.is_finite() || connection_step <= 0.0
    {
        return Err(RelativityError::InvalidDifferenceStep(connection_step));
    }

    // `metric_step` is validated inside `numerical_christoffel`.
    let christoffel = numerical_christoffel(metric, coordinates, metric_step)?;

    // Nested layer: d_dir Gamma by central differences of the metric-built
    // Christoffel symbols.
    let mut derivatives = [[[[0.0_f64; D]; D]; D]; D];
    for direction in 0..D
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[direction] += connection_step;
        minus[direction] -= connection_step;

        let christoffel_plus = numerical_christoffel(metric, &plus, metric_step)?;
        let christoffel_minus = numerical_christoffel(metric, &minus, metric_step)?;

        for rho in 0..D
        {
            for mu in 0..D
            {
                for nu in 0..D
                {
                    let value = (christoffel_plus[rho][mu][nu] - christoffel_minus[rho][mu][nu])
                        / (2.0 * connection_step);
                    if !value.is_finite()
                    {
                        return Err(RelativityError::NonFiniteCurvatureComponent {
                            quantity: "christoffel_derivative",
                        });
                    }
                    derivatives[direction][rho][mu][nu] = value;
                }
            }
        }
    }

    let riemann = riemann_from_connection(&christoffel, &derivatives);
    validate_tensor4("riemann", &riemann)?;
    let ricci = ricci_from_riemann(&riemann);
    validate_tensor2("ricci", &ricci)?;
    Ok(ricci)
}

/// The Ricci scalar `R = g^(mu nu) R_(mu nu)` of a metric from its **components
/// alone**. A thin wrapper contracting [`ricci_tensor_from_metric`] with the
/// inverse metric; see that function for the method and conventions.
///
/// Returns a typed [`RelativityError`] for non-finite coordinates, an invalid
/// step, a singular metric, or any non-finite intermediate; it never panics and
/// never silently returns a non-finite result.
///
/// # Example
///
/// De Sitter's Ricci scalar is exactly `4 * Lambda`; the nested difference
/// recovers it from the metric components alone.
///
/// ```
/// use scirust_relativity::{DeSitter, ricci_scalar_from_metric};
/// use std::f64::consts::FRAC_PI_2;
///
/// let lambda = 0.03;
/// let spacetime = DeSitter::try_new(lambda).expect("valid cosmological constant");
/// let scalar = ricci_scalar_from_metric(&spacetime, &[0.0, 3.0, FRAC_PI_2, 0.0], 1.0e-3, 1.0e-3)
///     .expect("finite curvature");
/// assert!((scalar - 4.0 * lambda).abs() < 1.0e-5);
/// ```
pub fn ricci_scalar_from_metric<M, const D: usize>(
    metric: &M,
    coordinates: &[f64; D],
    connection_step: f64,
    metric_step: f64,
) -> Result<f64, RelativityError>
where
    M: Metric<D>,
{
    let ricci = ricci_tensor_from_metric(metric, coordinates, connection_step, metric_step)?;
    let covariant = metric.components(coordinates);
    let inverse = invert_metric(&covariant)?;
    let ricci_scalar = contract_scalar(&inverse, &ricci);
    if !ricci_scalar.is_finite()
    {
        return Err(RelativityError::NonFiniteCurvatureComponent {
            quantity: "ricci_scalar",
        });
    }
    Ok(ricci_scalar)
}

/// Central finite differences of the Christoffel symbols:
/// `derivatives[dir][rho][mu][nu] = d_dir Gamma^rho_(mu nu)`.
fn christoffel_derivatives<B, const D: usize>(
    background: &B,
    coordinates: &[f64; D],
    difference_step: f64,
) -> Result<[[[[f64; D]; D]; D]; D], RelativityError>
where
    B: Connection<D>,
{
    let mut derivatives = [[[[0.0_f64; D]; D]; D]; D];

    for direction in 0..D
    {
        let mut plus = *coordinates;
        let mut minus = *coordinates;
        plus[direction] += difference_step;
        minus[direction] -= difference_step;

        let christoffel_plus = background.christoffel(&plus);
        let christoffel_minus = background.christoffel(&minus);

        for rho in 0..D
        {
            for mu in 0..D
            {
                for nu in 0..D
                {
                    let value = (christoffel_plus[rho][mu][nu] - christoffel_minus[rho][mu][nu])
                        / (2.0 * difference_step);
                    if !value.is_finite()
                    {
                        return Err(RelativityError::NonFiniteCurvatureComponent {
                            quantity: "christoffel_derivative",
                        });
                    }
                    derivatives[direction][rho][mu][nu] = value;
                }
            }
        }
    }

    Ok(derivatives)
}

/// Assemble `R^rho_(sigma mu nu)` from the Christoffel symbols and their
/// derivatives.
fn riemann_from_connection<const D: usize>(
    christoffel: &[[[f64; D]; D]; D],
    derivatives: &[[[[f64; D]; D]; D]; D],
) -> [[[[f64; D]; D]; D]; D] {
    let mut riemann = [[[[0.0_f64; D]; D]; D]; D];

    for rho in 0..D
    {
        for sigma in 0..D
        {
            for mu in 0..D
            {
                for nu in 0..D
                {
                    // d_mu Gamma^rho_(nu sigma) - d_nu Gamma^rho_(mu sigma)
                    let mut value =
                        derivatives[mu][rho][nu][sigma] - derivatives[nu][rho][mu][sigma];

                    // + Gamma^rho_(mu lambda) Gamma^lambda_(nu sigma)
                    // - Gamma^rho_(nu lambda) Gamma^lambda_(mu sigma)
                    for (lambda, christoffel_lambda) in christoffel.iter().enumerate()
                    {
                        value += christoffel[rho][mu][lambda] * christoffel_lambda[nu][sigma]
                            - christoffel[rho][nu][lambda] * christoffel_lambda[mu][sigma];
                    }

                    riemann[rho][sigma][mu][nu] = value;
                }
            }
        }
    }

    riemann
}

/// Ricci tensor `R_(sigma nu) = R^rho_(sigma rho nu)`.
fn ricci_from_riemann<const D: usize>(riemann: &[[[[f64; D]; D]; D]; D]) -> [[f64; D]; D] {
    let mut ricci = [[0.0_f64; D]; D];

    for sigma in 0..D
    {
        for nu in 0..D
        {
            let mut value = 0.0;
            for (rho, riemann_rho) in riemann.iter().enumerate()
            {
                value += riemann_rho[sigma][rho][nu];
            }
            ricci[sigma][nu] = value;
        }
    }

    ricci
}

/// Contract a symmetric covariant 2-tensor with the inverse metric:
/// `g^(mu nu) T_(mu nu)`.
fn contract_scalar<const D: usize>(inverse: &[[f64; D]; D], tensor: &[[f64; D]; D]) -> f64 {
    let mut value = 0.0;
    for mu in 0..D
    {
        for nu in 0..D
        {
            value += inverse[mu][nu] * tensor[mu][nu];
        }
    }
    value
}

/// Einstein tensor `G_(mu nu) = R_(mu nu) - 1/2 R g_(mu nu)`.
fn einstein_from_ricci<const D: usize>(
    ricci: &[[f64; D]; D],
    ricci_scalar: f64,
    metric: &[[f64; D]; D],
) -> [[f64; D]; D] {
    let mut einstein = [[0.0_f64; D]; D];
    for mu in 0..D
    {
        for nu in 0..D
        {
            einstein[mu][nu] = ricci[mu][nu] - 0.5 * ricci_scalar * metric[mu][nu];
        }
    }
    einstein
}

/// Kretschmann invariant `K = R_(abcd) R^(abcd)`.
fn kretschmann_from_riemann<const D: usize>(
    riemann: &[[[[f64; D]; D]; D]; D],
    metric: &[[f64; D]; D],
    inverse: &[[f64; D]; D],
) -> f64 {
    // Fully covariant Riemann: R_(rho sigma mu nu) = g_(rho lambda) R^lambda_(sigma mu nu).
    let mut lower = [[[[0.0_f64; D]; D]; D]; D];
    for rho in 0..D
    {
        for sigma in 0..D
        {
            for mu in 0..D
            {
                for nu in 0..D
                {
                    let mut value = 0.0;
                    for lambda in 0..D
                    {
                        value += metric[rho][lambda] * riemann[lambda][sigma][mu][nu];
                    }
                    lower[rho][sigma][mu][nu] = value;
                }
            }
        }
    }

    // Fully contravariant Riemann: raise all four indices with the inverse.
    let mut upper = [[[[0.0_f64; D]; D]; D]; D];
    for a in 0..D
    {
        for b in 0..D
        {
            for c in 0..D
            {
                for d in 0..D
                {
                    let mut value = 0.0;
                    for a2 in 0..D
                    {
                        for b2 in 0..D
                        {
                            for c2 in 0..D
                            {
                                for d2 in 0..D
                                {
                                    value += inverse[a][a2]
                                        * inverse[b][b2]
                                        * inverse[c][c2]
                                        * inverse[d][d2]
                                        * lower[a2][b2][c2][d2];
                                }
                            }
                        }
                    }
                    upper[a][b][c][d] = value;
                }
            }
        }
    }

    let mut kretschmann = 0.0;
    for a in 0..D
    {
        for b in 0..D
        {
            for c in 0..D
            {
                for d in 0..D
                {
                    kretschmann += lower[a][b][c][d] * upper[a][b][c][d];
                }
            }
        }
    }

    kretschmann
}

fn validate_tensor2<const D: usize>(
    quantity: &'static str,
    tensor: &[[f64; D]; D],
) -> Result<(), RelativityError> {
    for row in tensor
    {
        for value in row
        {
            if !value.is_finite()
            {
                return Err(RelativityError::NonFiniteCurvatureComponent { quantity });
            }
        }
    }
    Ok(())
}

fn validate_tensor4<const D: usize>(
    quantity: &'static str,
    tensor: &[[[[f64; D]; D]; D]; D],
) -> Result<(), RelativityError> {
    for block in tensor
    {
        for plane in block
        {
            for row in plane
            {
                for value in row
                {
                    if !value.is_finite()
                    {
                        return Err(RelativityError::NonFiniteCurvatureComponent { quantity });
                    }
                }
            }
        }
    }
    Ok(())
}
