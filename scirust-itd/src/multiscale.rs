//! Multi-scale structural profile derived from a single reference run.
//!
//! The structural signature's **roughness** component is the only one that
//! depends on the structural length scale `ℓ`, and it does so *linearly* in the
//! raw (unbounded) roughness:
//!
//! ```text
//! roughness_raw(ℓ) = ℓ · roughness_raw(1)
//! ```
//!
//! (the reference simulator computes roughness as `ℓ · ⟨‖∇ω‖⟩ / rms(ω)`, so the
//! `ℓ`-dependence factors out exactly). The other four components —
//! heterogeneity, localization, sign-mixing and temporal deformation — do not
//! depend on `ℓ` at all.
//!
//! This means an entire family of structural profiles, one per structural
//! length, can be derived from **one** reference run performed at `ℓ = 1`
//! without re-simulating: rescale the raw roughness by each `ℓ`, re-apply the
//! saturating bound, and re-integrate the signature and the derived indices.
//! [`derive_multiscale_profile`] does exactly that, reproducing the reference
//! `derive_multiscale_profile` to machine precision.

use crate::error::{ItdError, Result};
use crate::operators::bounded;

/// The per-step reference series a multi-scale profile is derived from — the
/// output of a single reference run performed at structural length `ℓ = 1`.
///
/// `intensity_rate`, `heterogeneity`, `localization`, `unit_roughness` and
/// `sign_mixing` are **nodal** series of the same length `N ≥ 2` (one value per
/// time step); `temporal_deformation_interval` and `interval_dt` are **interval**
/// series of length `N − 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiscaleReference {
    /// Nodal curvature-weighted rotational-intensity rate.
    pub intensity_rate: Vec<f64>,
    /// Nodal heterogeneity component (raw, unbounded).
    pub heterogeneity: Vec<f64>,
    /// Nodal localization component (raw, unbounded).
    pub localization: Vec<f64>,
    /// Nodal roughness component evaluated at structural length `ℓ = 1`.
    pub unit_roughness: Vec<f64>,
    /// Nodal sign-mixing component (already in `[0, 1]`).
    pub sign_mixing: Vec<f64>,
    /// Interval temporal-deformation component (raw, unbounded), length `N − 1`.
    pub temporal_deformation_interval: Vec<f64>,
    /// Interval durations `Δt`, length `N − 1`, each strictly positive.
    pub interval_dt: Vec<f64>,
    /// The five structural weights (heterogeneity, localization, roughness,
    /// sign-mixing, temporal deformation), applied as given (not renormalised).
    pub weights: [f64; 5],
    /// The scale-independent intensity index carried through unchanged.
    pub intensity_index: f64,
    /// The scale-independent temporal-deformation index carried through.
    pub temporal_deformation_index: f64,
}

/// A multi-scale profile: one structural signature and one set of derived
/// indices per structural length, as returned by [`derive_multiscale_profile`].
#[derive(Debug, Clone, PartialEq)]
pub struct MultiscaleProfile {
    /// The structural lengths, in the order supplied.
    pub structural_lengths: Vec<f64>,
    /// The five-component interval-integrated signature at each length
    /// (heterogeneity, localization, roughness, sign-mixing, temporal
    /// deformation).
    pub signatures: Vec<[f64; 5]>,
    /// The weighted structure index at each length.
    pub structure_indices: Vec<f64>,
    /// The intensity-coupled index `⟨intensity · (1 + structure)⟩` at each
    /// length.
    pub coupled_indices: Vec<f64>,
    /// The raw (unbounded) roughness index at each length.
    pub raw_roughness_indices: Vec<f64>,
    /// The scale-independent intensity index, echoed from the reference.
    pub intensity_index: f64,
    /// The scale-independent temporal-deformation index, echoed from the
    /// reference.
    pub temporal_deformation_index: f64,
}

/// Validates a grid of structural lengths: at least two finite, non-negative and
/// strictly increasing values (matching the reference
/// `validate_structural_length_grid`).
fn validate_structural_lengths(lengths: &[f64]) -> Result<()> {
    if lengths.len() < 2
    {
        return Err(ItdError::TooFewPoints(
            "a multi-scale profile needs at least two structural lengths".into(),
        ));
    }
    if !lengths.iter().all(|v| v.is_finite())
    {
        return Err(ItdError::NonFinite("structural lengths".into()));
    }
    if lengths.iter().any(|&v| v < 0.0)
    {
        return Err(ItdError::InvalidGeometry(
            "structural lengths must be non-negative".into(),
        ));
    }
    if !lengths.windows(2).all(|w| w[1] > w[0])
    {
        return Err(ItdError::InvalidGeometry(
            "structural lengths must be strictly increasing".into(),
        ));
    }
    Ok(())
}

/// Left-to-right sum of the element-wise product `a · b` (matching NumPy's
/// sequential `np.sum` for the small interval series used here).
fn weighted_sum(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for k in 0..a.len()
    {
        acc += a[k] * b[k];
    }
    acc
}

/// Nodal-to-interval midpoints: `0.5 · (v[m] + v[m+1])`.
fn midpoints(v: &[f64]) -> Vec<f64> {
    v.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
}

/// Derives a multi-scale structural profile from a single `ℓ = 1` reference run.
///
/// For each structural length `ℓ` the raw roughness is rescaled by `ℓ`, bounded,
/// and re-integrated together with the (scale-independent) other components into
/// the signature, the weighted structure index and the intensity-coupled index.
pub fn derive_multiscale_profile(
    reference: &MultiscaleReference,
    structural_lengths: &[f64],
) -> Result<MultiscaleProfile> {
    validate_structural_lengths(structural_lengths)?;

    let nodal_size = reference.intensity_rate.len();
    if nodal_size < 2
    {
        return Err(ItdError::TooFewPoints(
            "the reference run must contain at least two time steps".into(),
        ));
    }
    let interval_size = nodal_size - 1;

    for (series, name) in [
        (&reference.heterogeneity, "heterogeneity"),
        (&reference.localization, "localization"),
        (&reference.unit_roughness, "roughness"),
        (&reference.sign_mixing, "sign_mixing"),
    ]
    {
        if series.len() != nodal_size
        {
            return Err(ItdError::ShapeMismatch(format!(
                "reference {name} series has length {}, expected {nodal_size}",
                series.len()
            )));
        }
    }
    for (series, name) in [
        (
            &reference.temporal_deformation_interval,
            "temporal_deformation_interval",
        ),
        (&reference.interval_dt, "interval_dt"),
    ]
    {
        if series.len() != interval_size
        {
            return Err(ItdError::ShapeMismatch(format!(
                "reference {name} series has length {}, expected {interval_size}",
                series.len()
            )));
        }
    }

    let finite_series = [
        &reference.intensity_rate,
        &reference.heterogeneity,
        &reference.localization,
        &reference.unit_roughness,
        &reference.sign_mixing,
        &reference.temporal_deformation_interval,
        &reference.interval_dt,
    ];
    if !finite_series
        .iter()
        .all(|s| s.iter().all(|v| v.is_finite()))
        || !reference.weights.iter().all(|w| w.is_finite())
    {
        return Err(ItdError::NonFinite("reference series".into()));
    }
    if reference.interval_dt.iter().any(|&dt| dt <= 0.0)
    {
        return Err(ItdError::InvalidGeometry(
            "reference interval durations must be strictly positive".into(),
        ));
    }

    let duration: f64 = reference.interval_dt.iter().sum();
    if !duration.is_finite() || duration <= 0.0
    {
        return Err(ItdError::InvalidGeometry(
            "the reference profile duration must be finite and strictly positive".into(),
        ));
    }

    // Scale-independent bounded components (interval midpoints).
    let heterogeneity_interval = midpoints(
        &reference
            .heterogeneity
            .iter()
            .map(|&v| bounded(v))
            .collect::<Vec<_>>(),
    );
    let localization_interval = midpoints(
        &reference
            .localization
            .iter()
            .map(|&v| bounded(v))
            .collect::<Vec<_>>(),
    );
    let sign_mixing_interval = midpoints(
        &reference
            .sign_mixing
            .iter()
            .map(|&v| v.clamp(0.0, 1.0))
            .collect::<Vec<_>>(),
    );
    let interval_deformation: Vec<f64> = reference
        .temporal_deformation_interval
        .iter()
        .map(|&v| bounded(v))
        .collect();
    let intensity_interval = midpoints(&reference.intensity_rate);

    let dt = &reference.interval_dt;
    let weights = &reference.weights;

    let mut signatures = Vec::with_capacity(structural_lengths.len());
    let mut structure_indices = Vec::with_capacity(structural_lengths.len());
    let mut coupled_indices = Vec::with_capacity(structural_lengths.len());
    let mut raw_roughness_indices = Vec::with_capacity(structural_lengths.len());

    for &length in structural_lengths
    {
        let raw_roughness: Vec<f64> = reference
            .unit_roughness
            .iter()
            .map(|&r| length * r)
            .collect();
        let roughness_interval = midpoints(
            &raw_roughness
                .iter()
                .map(|&r| bounded(r))
                .collect::<Vec<_>>(),
        );
        let raw_roughness_interval = midpoints(&raw_roughness);

        let components: [&[f64]; 5] = [
            &heterogeneity_interval,
            &localization_interval,
            &roughness_interval,
            &sign_mixing_interval,
            &interval_deformation,
        ];

        let mut signature = [0.0f64; 5];
        for (c, component) in components.iter().enumerate()
        {
            signature[c] = weighted_sum(component, dt) / duration;
        }

        let mut structure_interval = vec![0.0f64; interval_size];
        for (c, component) in components.iter().enumerate()
        {
            for m in 0..interval_size
            {
                structure_interval[m] += weights[c] * component[m];
            }
        }
        let coupled_interval: Vec<f64> = (0..interval_size)
            .map(|m| intensity_interval[m] * (1.0 + structure_interval[m]))
            .collect();

        signatures.push(signature);
        structure_indices.push(weighted_sum(&structure_interval, dt) / duration);
        coupled_indices.push(weighted_sum(&coupled_interval, dt) / duration);
        raw_roughness_indices.push(weighted_sum(&raw_roughness_interval, dt) / duration);
    }

    Ok(MultiscaleProfile {
        structural_lengths: structural_lengths.to_vec(),
        signatures,
        structure_indices,
        coupled_indices,
        raw_roughness_indices,
        intensity_index: reference.intensity_index,
        temporal_deformation_index: reference.temporal_deformation_index,
    })
}
