//! Oracle-backed extraction of the Eddington–Robertson PPN parameters `gamma`
//! and `beta` from static, spherically symmetric, asymptotically flat metrics in
//! a PPN-compatible isotropic radial coordinate.
//!
//! See `docs/LAYER_2_PPN.md` for the full design, conventions, and oracle
//! hierarchy. In brief, with signature `(-,+,+,+)` and compactness
//! `U = G M / rho`,
//!
//! ```text
//! g_00 = -1 + 2U - 2 beta U^2 + O(U^3),
//! g_ij =  A delta_ij ,   A = 1 + 2 gamma U + O(U^2),
//! ```
//!
//! so the per-radius **effective** estimators are
//!
//! ```text
//! gamma_eff(rho) = (A - 1) / (2U),     beta_eff(rho) = -(g_00 + 1 - 2U) / (2U^2),
//! ```
//!
//! each contaminated at `O(U)`. The asymptotic values are the `U -> 0` intercepts
//! of a deterministic low-degree polynomial fit ([`fit_polynomial_intercept`]).
//!
//! This is a **numerical approximation** (an asymptotic extrapolation), not an
//! exact result; the reported uncertainty is an *estimated* numerical
//! sensitivity, not a rigorous bound. Recovering `gamma = beta = 1` for the
//! exact isotropic-Schwarzschild oracle validates the *implementation*, not any
//! alternative theory. Only `gamma` and `beta` are implemented; the exclusions
//! (preferred-frame / nonconservative parameters, the full ten-parameter
//! formalism, rotating / time-dependent / non-spherical / cosmological metrics,
//! automated coordinate conversion, observational likelihoods) are listed in the
//! design note.

mod coordinate;
mod error;
mod extrapolation;
mod oracle;

pub use coordinate::{IsotropicChartAdapter, StaticIsotropicMetric};
pub use error::PpnError;
pub use extrapolation::{
    CONDITIONING_MARGINAL_THRESHOLD, ConditioningClass, MAX_DEGREE, PolynomialFit,
    classify_conditioning, fit_polynomial_intercept,
};
pub use oracle::SyntheticPpnMetric;

/// Hard weak-field gate: the strongest-field sample's compactness must not exceed
/// this. Beyond it the truncated PPN expansion is not a controlled description.
pub const WEAK_FIELD_COMPACTNESS_MAX: f64 = 0.25;

/// A sample whose `|g_00 + 1|` or `|A - 1|` exceeds this is not treated as a
/// weak-field perturbation of Minkowski (catches non-asymptotically-flat input).
const WEAK_FIELD_PERTURBATION_MAX: f64 = 0.9;

/// How the radial samples are laid out. All variants are deterministic.
#[derive(Debug, Clone, PartialEq)]
pub enum PpnSampling {
    /// Uniform in compactness `U` between the two bounds (`0 < min < max`).
    UniformCompactness {
        /// Smallest compactness (largest radius).
        compactness_min: f64,
        /// Largest compactness (smallest radius).
        compactness_max: f64,
    },
    /// Logarithmic in radius between the two bounds (`0 < min < max`).
    LogarithmicRadius {
        /// Smallest radius.
        radius_min: f64,
        /// Largest radius.
        radius_max: f64,
    },
    /// Caller-provided radii (validated, then sorted ascending).
    ExplicitRadii(Vec<f64>),
}

/// The radial sampling domain for an extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct PpnDomain {
    /// The sampling policy.
    pub sampling: PpnSampling,
    /// Number of samples for the ranged policies (ignored by `ExplicitRadii`).
    pub sample_count: usize,
}

impl PpnDomain {
    /// Uniform-in-compactness domain.
    #[must_use]
    pub fn uniform_compactness(
        compactness_min: f64,
        compactness_max: f64,
        sample_count: usize,
    ) -> Self {
        Self {
            sampling: PpnSampling::UniformCompactness {
                compactness_min,
                compactness_max,
            },
            sample_count,
        }
    }

    /// Logarithmic-in-radius domain.
    #[must_use]
    pub fn logarithmic_radius(radius_min: f64, radius_max: f64, sample_count: usize) -> Self {
        Self {
            sampling: PpnSampling::LogarithmicRadius {
                radius_min,
                radius_max,
            },
            sample_count,
        }
    }

    /// Explicit-radii domain.
    #[must_use]
    pub fn explicit_radii(radii: Vec<f64>) -> Self {
        let sample_count = radii.len();
        Self {
            sampling: PpnSampling::ExplicitRadii(radii),
            sample_count,
        }
    }

    /// Deterministic ascending radii for mass scale `mass_scale`. Validates the
    /// domain and the weak-field compactness gate.
    fn radii(&self, mass_scale: f64) -> Result<Vec<f64>, PpnError> {
        let mut radii = match &self.sampling
        {
            PpnSampling::UniformCompactness {
                compactness_min,
                compactness_max,
            } =>
            {
                if !compactness_min.is_finite()
                    || !compactness_max.is_finite()
                    || *compactness_min <= 0.0
                    || compactness_min >= compactness_max
                {
                    return Err(PpnError::InvalidRadialDomain {
                        radius_min: mass_scale / compactness_max,
                        radius_max: mass_scale / compactness_min,
                    });
                }
                if self.sample_count < 2
                {
                    return Err(PpnError::InsufficientSamples {
                        available: self.sample_count,
                        required: 2,
                    });
                }
                let last = (self.sample_count - 1) as f64;
                (0..self.sample_count)
                    .map(|index| {
                        let fraction = index as f64 / last;
                        let compactness =
                            compactness_min + fraction * (compactness_max - compactness_min);
                        mass_scale / compactness
                    })
                    .collect()
            },
            PpnSampling::LogarithmicRadius {
                radius_min,
                radius_max,
            } =>
            {
                if !radius_min.is_finite()
                    || !radius_max.is_finite()
                    || *radius_min <= 0.0
                    || radius_min >= radius_max
                {
                    return Err(PpnError::InvalidRadialDomain {
                        radius_min: *radius_min,
                        radius_max: *radius_max,
                    });
                }
                if self.sample_count < 2
                {
                    return Err(PpnError::InsufficientSamples {
                        available: self.sample_count,
                        required: 2,
                    });
                }
                let ratio = radius_max / radius_min;
                let last = (self.sample_count - 1) as f64;
                (0..self.sample_count)
                    .map(|index| {
                        let fraction = index as f64 / last;
                        radius_min * ratio.powf(fraction)
                    })
                    .collect()
            },
            PpnSampling::ExplicitRadii(radii) =>
            {
                if radii
                    .iter()
                    .any(|radius| !radius.is_finite() || *radius <= 0.0)
                {
                    return Err(PpnError::InvalidRadialDomain {
                        radius_min: radii.iter().cloned().fold(f64::INFINITY, f64::min),
                        radius_max: radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    });
                }
                radii.clone()
            },
        };

        radii.sort_by(f64::total_cmp);
        if radii.len() < 2
        {
            return Err(PpnError::InsufficientSamples {
                available: radii.len(),
                required: 2,
            });
        }

        // Weak-field gate on the strongest-field (smallest-radius) sample.
        let strongest = mass_scale / radii[0];
        if strongest > WEAK_FIELD_COMPACTNESS_MAX
        {
            return Err(PpnError::CompactnessOutOfRange {
                compactness: strongest,
                maximum: WEAK_FIELD_COMPACTNESS_MAX,
            });
        }
        Ok(radii)
    }
}

/// A single numerical-sensitivity probe: how much the fitted intercept moves
/// when one aspect of the extraction (radial window, polynomial order, or
/// sample resolution) is deliberately perturbed. Not a bound — see
/// [`ParameterEstimate::estimated_uncertainty`] and the module documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterSensitivity {
    /// The absolute deviation of the perturbed fit's intercept from the primary
    /// fit's. Only meaningful when [`available`](Self::available) is `true`.
    pub deviation: f64,
    /// Whether a perturbed fit was actually computed. `false` means no valid
    /// perturbed fit could be attempted (for example, order sensitivity at the
    /// minimum degree with too few samples for the adjacent degree) — `deviation`
    /// carries no information in that case and must never be read as "zero
    /// sensitivity".
    pub available: bool,
}

impl ParameterSensitivity {
    /// No perturbed fit was attempted or none succeeded.
    const UNAVAILABLE: Self = Self {
        deviation: 0.0,
        available: false,
    };

    /// Record a successful perturbed fit, keeping the larger deviation if one is
    /// already recorded (used where more than one probe feeds a single axis, for
    /// example both adjacent polynomial orders).
    fn record(self, deviation: f64) -> Self {
        let deviation = if self.available
        {
            self.deviation.max(deviation)
        }
        else
        {
            deviation
        };
        Self {
            deviation,
            available: true,
        }
    }
}

/// Structured numerical diagnostics for one extracted parameter, breaking
/// [`ParameterEstimate::estimated_uncertainty`] down into its individual
/// sensitivity axes plus the fit's conditioning classification. See
/// `docs/LAYER_2_PPN.md` for how each is computed and why none is a bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PpnParameterDiagnostics {
    /// Sensitivity to restricting the fit to the weaker-field half of the domain.
    pub radial_window_sensitivity: ParameterSensitivity,
    /// Sensitivity to the adjacent supported polynomial degree(s).
    pub fit_order_sensitivity: ParameterSensitivity,
    /// Sensitivity to halving the sample resolution (every other sample).
    pub resolution_sensitivity: ParameterSensitivity,
    /// The fit's conditioning classification (see [`classify_conditioning`]).
    pub conditioning_class: ConditioningClass,
}

/// A per-radius effective estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteRadiusEstimate {
    /// The isotropic radius.
    pub radius: f64,
    /// The compactness `U = G M / radius`.
    pub compactness: f64,
    /// The effective estimator value at this radius.
    pub value: f64,
}

/// The extraction result for one PPN parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterEstimate {
    /// The asymptotic (`U -> 0`) value.
    pub asymptotic_value: f64,
    /// A conservative *estimated* numerical uncertainty (the largest available
    /// sensitivity in [`diagnostics`](Self::diagnostics), or the fit residual if
    /// none was available), not a rigorous bound.
    pub estimated_uncertainty: f64,
    /// The least-squares residual norm of the accepted fit.
    pub fit_residual: f64,
    /// The fit's conditioning indicator (see [`classify_conditioning`] for the
    /// classification carried in [`diagnostics`](Self::diagnostics)).
    pub conditioning: f64,
    /// The per-radius effective estimates.
    pub finite_radius_values: Vec<FiniteRadiusEstimate>,
    /// The individual sensitivity axes and conditioning classification that
    /// [`estimated_uncertainty`](Self::estimated_uncertainty) conservatively
    /// summarizes.
    pub diagnostics: PpnParameterDiagnostics,
}

/// The full PPN extraction result.
#[derive(Debug, Clone, PartialEq)]
pub struct PpnEstimate {
    /// The `gamma` estimate.
    pub gamma: ParameterEstimate,
    /// The `beta` estimate.
    pub beta: ParameterEstimate,
    /// The mass scale `G M` used, completing the weak-field domain summary
    /// together with [`sample_count`](Self::sample_count),
    /// [`compactness_min`](Self::compactness_min), and
    /// [`compactness_max`](Self::compactness_max).
    pub mass_scale: f64,
    /// The polynomial degree used.
    pub fit_order: usize,
    /// The number of samples used.
    pub sample_count: usize,
    /// Smallest compactness sampled.
    pub compactness_min: f64,
    /// Largest compactness sampled.
    pub compactness_max: f64,
}

/// Extract `gamma` and `beta` from `metric` over `domain`, fitting the effective
/// estimators with a degree-`degree` polynomial and reading the `U -> 0`
/// intercepts.
///
/// Returns a typed [`PpnError`] for an invalid mass scale, a malformed or
/// too-strong-field domain, a non-isotropic chart, a non-asymptotically-flat
/// metric, insufficient samples, non-finite metric values, or a singular /
/// ill-conditioned fit. It never panics.
pub fn extract_ppn<M: StaticIsotropicMetric>(
    metric: &M,
    domain: &PpnDomain,
    degree: usize,
) -> Result<PpnEstimate, PpnError> {
    let mass = metric.mass_scale();
    if !mass.is_finite() || mass <= 0.0
    {
        return Err(PpnError::InvalidMassScale(mass));
    }
    if degree == 0 || degree > MAX_DEGREE
    {
        return Err(PpnError::UnsupportedExtrapolationOrder {
            order: degree,
            maximum: MAX_DEGREE,
        });
    }

    let radii = domain.radii(mass)?;
    if radii.len() < degree + 1
    {
        return Err(PpnError::InsufficientSamples {
            available: radii.len(),
            required: degree + 1,
        });
    }

    let mut compactness = Vec::with_capacity(radii.len());
    let mut gamma_effective = Vec::with_capacity(radii.len());
    let mut beta_effective = Vec::with_capacity(radii.len());
    for &radius in &radii
    {
        let u = mass / radius;
        let g_tt = metric.g_tt(radius)?;
        let conformal = metric.spatial_conformal_factor(radius)?;
        if (g_tt + 1.0).abs() > WEAK_FIELD_PERTURBATION_MAX
            || (conformal - 1.0).abs() > WEAK_FIELD_PERTURBATION_MAX
        {
            return Err(PpnError::NonAsymptoticallyFlat { radius });
        }
        let gamma_value = (conformal - 1.0) / (2.0 * u);
        let beta_value = -(g_tt + 1.0 - 2.0 * u) / (2.0 * u * u);
        if !gamma_value.is_finite() || !beta_value.is_finite()
        {
            return Err(PpnError::NonFiniteEstimate);
        }
        compactness.push(u);
        gamma_effective.push(gamma_value);
        beta_effective.push(beta_value);
    }

    let gamma = extrapolate_parameter(&radii, &compactness, &gamma_effective, degree)?;
    let beta = extrapolate_parameter(&radii, &compactness, &beta_effective, degree)?;

    let compactness_min = compactness.iter().cloned().fold(f64::INFINITY, f64::min);
    let compactness_max = compactness
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(PpnEstimate {
        gamma,
        beta,
        mass_scale: mass,
        fit_order: degree,
        sample_count: radii.len(),
        compactness_min,
        compactness_max,
    })
}

/// Fit one effective estimator and assemble its [`ParameterEstimate`], including
/// the sensitivity diagnostics.
fn extrapolate_parameter(
    radii: &[f64],
    compactness: &[f64],
    effective: &[f64],
    degree: usize,
) -> Result<ParameterEstimate, PpnError> {
    let fit = fit_polynomial_intercept(compactness, effective, degree)?;
    let asymptotic_value = fit.intercept;

    let probes = probe_sensitivities(
        compactness,
        effective,
        degree,
        asymptotic_value,
        fit.residual_norm,
    );

    let finite_radius_values = radii
        .iter()
        .zip(compactness.iter())
        .zip(effective.iter())
        .map(|((&radius, &compactness), &value)| FiniteRadiusEstimate {
            radius,
            compactness,
            value,
        })
        .collect();

    Ok(ParameterEstimate {
        asymptotic_value,
        estimated_uncertainty: probes.conservative_uncertainty,
        fit_residual: fit.residual_norm,
        conditioning: fit.conditioning,
        finite_radius_values,
        diagnostics: PpnParameterDiagnostics {
            radial_window_sensitivity: probes.window,
            fit_order_sensitivity: probes.order,
            resolution_sensitivity: probes.resolution,
            conditioning_class: classify_conditioning(fit.conditioning),
        },
    })
}

/// The three individual sensitivity probes plus the conservative blended
/// uncertainty derived from them.
struct SensitivityProbes {
    window: ParameterSensitivity,
    order: ParameterSensitivity,
    resolution: ParameterSensitivity,
    /// The largest available deviation among `window`, `order`, and
    /// `resolution`; the fit residual if none was available. Not a bound.
    conservative_uncertainty: f64,
}

/// Compute the window / order / resolution sensitivity probes of the fitted
/// intercept. Sub-fits that fail (too few points after windowing/thinning, or
/// ill-conditioned) are recorded as [`ParameterSensitivity::UNAVAILABLE`] on
/// that axis, never silently treated as zero sensitivity.
fn probe_sensitivities(
    compactness: &[f64],
    effective: &[f64],
    degree: usize,
    primary: f64,
    fit_residual: f64,
) -> SensitivityProbes {
    // Window sensitivity: fit only the weaker-field (smaller-U) half.
    let mut indexed: Vec<usize> = (0..compactness.len()).collect();
    indexed.sort_by(|&a, &b| compactness[a].total_cmp(&compactness[b]));
    let half = indexed.len().div_ceil(2);
    let window_u: Vec<f64> = indexed[..half].iter().map(|&i| compactness[i]).collect();
    let window_y: Vec<f64> = indexed[..half].iter().map(|&i| effective[i]).collect();
    let window = match fit_polynomial_intercept(&window_u, &window_y, degree)
    {
        Ok(fit) => ParameterSensitivity::UNAVAILABLE.record((fit.intercept - primary).abs()),
        Err(_) => ParameterSensitivity::UNAVAILABLE,
    };

    // Order sensitivity: the adjacent degrees. The higher adjacent degree is the
    // important one — for a biased low-degree primary it exposes the leading
    // systematic error, keeping the estimate conservative. Both are folded into
    // one axis (the worse of the two) since either alone answers "is the chosen
    // degree adequate?".
    let mut order = ParameterSensitivity::UNAVAILABLE;
    if degree > 1
    {
        if let Ok(fit) = fit_polynomial_intercept(compactness, effective, degree - 1)
        {
            order = order.record((fit.intercept - primary).abs());
        }
    }
    if degree < MAX_DEGREE
    {
        if let Ok(fit) = fit_polynomial_intercept(compactness, effective, degree + 1)
        {
            order = order.record((fit.intercept - primary).abs());
        }
    }

    // Resolution sensitivity: every other sample.
    let coarse_u: Vec<f64> = compactness.iter().step_by(2).copied().collect();
    let coarse_y: Vec<f64> = effective.iter().step_by(2).copied().collect();
    let resolution = match fit_polynomial_intercept(&coarse_u, &coarse_y, degree)
    {
        Ok(fit) => ParameterSensitivity::UNAVAILABLE.record((fit.intercept - primary).abs()),
        Err(_) => ParameterSensitivity::UNAVAILABLE,
    };

    let mut conservative = 0.0_f64;
    let mut any = false;
    for probe in [window, order, resolution]
    {
        if probe.available
        {
            conservative = conservative.max(probe.deviation);
            any = true;
        }
    }
    let conservative_uncertainty = if any { conservative } else { fit_residual };

    SensitivityProbes {
        window,
        order,
        resolution,
        conservative_uncertainty,
    }
}
