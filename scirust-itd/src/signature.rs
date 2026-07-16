//! The five-component structural signature of a vorticity field.
//!
//! Given a vorticity field `ω` (and, optionally, its predecessor for the
//! temporal term), [`structural_metrics`] returns five deterministic
//! descriptors and a normalized-weighted scalar score:
//!
//! * **heterogeneity** — coefficient of variation of `|ω|`;
//! * **localization** — flatness/kurtosis excess `⟨ω⁴⟩/⟨ω²⟩² − 1`;
//! * **roughness** — `L · ⟨‖∇ω‖⟩ / rms(ω)`;
//! * **sign-mixing** — `1 − |⟨ω⟩| / ⟨|ω|⟩`, clamped to `[0, 1]`;
//! * **temporal deformation** — `rms(ω − ω_prev) / (Δt · rms_ref)`.

use crate::error::{ItdError, Result};
use crate::field::Field2;
use crate::geometry::{BoundaryMode, Geometry};
use crate::operators::{bounded, gradient, spatial_mean};

/// Values with magnitude below this threshold are treated as zero (matching
/// the reference `ZERO_THRESHOLD`).
pub const ZERO_THRESHOLD: f64 = 1.0e-12;

/// The default structural length scale used by the reference simulator.
pub const STRUCTURAL_LENGTH: f64 = 0.5;

/// The five non-negative weights applied to the bounded structural components,
/// in order: heterogeneity, localization, roughness, sign-mixing, temporal
/// deformation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralWeights(pub [f64; 5]);

impl Default for StructuralWeights {
    fn default() -> Self {
        StructuralWeights([1.0; 5])
    }
}

impl StructuralWeights {
    /// Validates the weights: each must be finite and non-negative, with a
    /// strictly positive sum.
    pub fn new(weights: [f64; 5]) -> Result<Self> {
        if !weights.iter().all(|w| w.is_finite())
        {
            return Err(ItdError::InvalidWeights("weights must be finite".into()));
        }
        if weights.iter().any(|&w| w < 0.0)
        {
            return Err(ItdError::InvalidWeights(
                "weights must be non-negative".into(),
            ));
        }
        if weights.iter().sum::<f64>() <= 0.0
        {
            return Err(ItdError::InvalidWeights(
                "at least one weight must be strictly positive".into(),
            ));
        }
        Ok(StructuralWeights(weights))
    }

    /// The weights normalized to sum to one.
    pub fn normalized(&self) -> [f64; 5] {
        let total: f64 = self.0.iter().sum();
        let mut out = self.0;
        for w in &mut out
        {
            *w /= total;
        }
        out
    }
}

/// The five structural descriptors plus the weighted structure score. The raw
/// component values are reported unbounded (as the reference does for its
/// per-step series); the score applies the saturating [`bounded`] map to
/// heterogeneity, localization, roughness and temporal deformation (sign-mixing
/// is already in `[0, 1]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralMetrics {
    /// Coefficient of variation of `|ω|`.
    pub heterogeneity: f64,
    /// Flatness/kurtosis excess `⟨ω⁴⟩/⟨ω²⟩² − 1`.
    pub localization: f64,
    /// `L · ⟨‖∇ω‖⟩ / rms(ω)`.
    pub roughness: f64,
    /// `1 − |⟨ω⟩| / ⟨|ω|⟩`, clamped to `[0, 1]`.
    pub sign_mixing: f64,
    /// `rms(ω − ω_prev) / (Δt · rms_ref)`, or `0` when no predecessor is given.
    pub temporal_deformation: f64,
    /// The normalized-weighted score of the bounded components.
    pub structure_score: f64,
}

impl StructuralMetrics {
    const ZERO: StructuralMetrics = StructuralMetrics {
        heterogeneity: 0.0,
        localization: 0.0,
        roughness: 0.0,
        sign_mixing: 0.0,
        temporal_deformation: 0.0,
        structure_score: 0.0,
    };
}

/// Computes the structural signature of `omega`.
///
/// `previous_omega` and `delta_time` are used only for the temporal-deformation
/// term; pass `None` (or a non-positive `delta_time`) at the first step. The
/// weights are normalized internally.
pub fn structural_metrics(
    omega: &Field2,
    geometry: &Geometry,
    previous_omega: Option<&Field2>,
    delta_time: Option<f64>,
    structural_length: f64,
    weights: StructuralWeights,
    boundary: BoundaryMode,
) -> Result<StructuralMetrics> {
    if !structural_length.is_finite() || structural_length < 0.0
    {
        return Err(ItdError::InvalidGeometry(
            "structural length must be finite and non-negative".into(),
        ));
    }
    geometry.validate_field(omega)?;
    if !omega.all_finite()
    {
        return Err(ItdError::NonFinite("vorticity field".into()));
    }
    let weights = weights.normalized();
    let mean = |field: &Field2| spatial_mean(field, geometry, boundary);

    let omega_sq = omega.map(|w| w * w);
    let mean_square = mean(&omega_sq)?;
    let rms = mean_square.max(0.0).sqrt();
    if rms < ZERO_THRESHOLD
    {
        return Ok(StructuralMetrics::ZERO);
    }

    let abs_omega = omega.map(f64::abs);
    let mean_absolute = mean(&abs_omega)?;

    let absolute_deviation_sq = abs_omega.map(|a| {
        let d = a - mean_absolute;
        d * d
    });
    let weighted_variance = mean(&absolute_deviation_sq)?;
    let heterogeneity = weighted_variance.max(0.0).sqrt() / mean_absolute.max(ZERO_THRESHOLD);

    let omega_fourth = omega.map(|w| {
        let s = w * w;
        s * s
    });
    let localization = mean(&omega_fourth)? / (mean_square * mean_square).max(ZERO_THRESHOLD) - 1.0;

    let (gradient_y, gradient_x) = gradient(omega, geometry, boundary)?;
    let gradient_norm = gradient_x.zip_map(&gradient_y, |gx, gy| (gx * gx + gy * gy).sqrt())?;
    let roughness = structural_length * mean(&gradient_norm)? / rms.max(ZERO_THRESHOLD);

    let mean_omega = mean(omega)?;
    let sign_mixing = (1.0 - mean_omega.abs() / mean_absolute.max(ZERO_THRESHOLD)).clamp(0.0, 1.0);

    let mut temporal_deformation = 0.0;
    if let (Some(prev), Some(dt)) = (previous_omega, delta_time)
    {
        if dt > 0.0
        {
            if prev.shape() != omega.shape()
            {
                return Err(ItdError::ShapeMismatch(
                    "successive vorticity fields must share a shape".into(),
                ));
            }
            let prev_sq = prev.map(|w| w * w);
            let previous_rms = mean(&prev_sq)?.max(0.0).sqrt();
            let reference_rms = 0.5 * (rms + previous_rms);
            if reference_rms >= ZERO_THRESHOLD
            {
                let diff_sq = omega.zip_map(prev, |a, b| {
                    let d = a - b;
                    d * d
                })?;
                temporal_deformation = mean(&diff_sq)?.max(0.0).sqrt() / (dt * reference_rms);
            }
        }
    }

    let components = [
        bounded(heterogeneity),
        bounded(localization),
        bounded(roughness),
        sign_mixing,
        bounded(temporal_deformation),
    ];
    let structure_score = weights
        .iter()
        .zip(components.iter())
        .map(|(w, c)| w * c)
        .sum();

    Ok(StructuralMetrics {
        heterogeneity,
        localization,
        roughness,
        sign_mixing,
        temporal_deformation,
        structure_score,
    })
}
