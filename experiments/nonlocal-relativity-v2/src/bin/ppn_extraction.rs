//! PPN (gamma, beta) extraction checks: synthetic oracles (GR and non-GR),
//! controlled contamination, isotropic Schwarzschild convergence, and
//! areal-coordinate rejection (a Layer 2 slice; see `docs/LAYER_2_PPN.md`).
//!
//! For each metric and each (window, sample count, sampling spacing, fit order)
//! the experiment runs the deterministic asymptotic extractor and reports the
//! estimate, its absolute and relative error against the known oracle, the
//! *estimated* numerical uncertainty (a conservative sensitivity, not a bound),
//! the fit residual, the three individual sensitivity axes (radial window / fit
//! order / resolution -- each independently, per `docs/LAYER_2_PPN.md`'s
//! hardening addendum), and the conditioning classification. The areal-
//! Schwarzschild rows show the coordinate-validity rejection.
//!
//! Established general relativity only. The extraction is a numerical
//! approximation; the isotropic-Schwarzschild oracle (`gamma = beta = 1`)
//! validates the implementation, not an alternative theory. The exact and
//! contaminated synthetic metrics are validation oracles / controlled numerical
//! validation models, not physical predictions.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{print_experiment_header, require_finite};
use scirust_relativity::ppn::{
    ConditioningClass, IsotropicChartAdapter, ParameterSensitivity, PpnDomain, PpnError,
    PpnSampling, StaticIsotropicMetric, SyntheticPpnMetric, extract_ppn,
};
use scirust_relativity::{IsotropicSchwarzschild, Schwarzschild};

const MASS: f64 = 1.0;

/// Short, comma-free status token for a rejected extraction.
fn rejection_token(error: &PpnError) -> &'static str {
    match error
    {
        PpnError::NonIsotropicCoordinates { .. } => "rejected_non_isotropic",
        PpnError::CompactnessOutOfRange { .. } => "rejected_compactness",
        PpnError::NonAsymptoticallyFlat { .. } => "rejected_not_flat",
        _ => "rejected_other",
    }
}

/// Short, comma-free token naming the domain's sampling spacing, plus its
/// radial bounds -- both derived from the same `PpnDomain` that is actually
/// sampled, so the reported bounds can never drift from what was extracted.
fn domain_summary(domain: &PpnDomain, mass: f64) -> (&'static str, f64, f64) {
    match &domain.sampling
    {
        PpnSampling::UniformCompactness {
            compactness_min,
            compactness_max,
        } => (
            "linear_compactness",
            mass / compactness_max,
            mass / compactness_min,
        ),
        PpnSampling::LogarithmicRadius {
            radius_min,
            radius_max,
        } => ("logarithmic_radius", *radius_min, *radius_max),
        PpnSampling::ExplicitRadii(radii) =>
        {
            let min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            ("explicit_radii", min, max)
        },
    }
}

/// Comma-free token for a conditioning classification.
fn conditioning_token(class: ConditioningClass) -> &'static str {
    match class
    {
        ConditioningClass::WellConditioned => "well_conditioned",
        ConditioningClass::Marginal => "marginal",
        ConditioningClass::IllConditioned => "ill_conditioned",
    }
}

/// A sensitivity's deviation, or `na` if no perturbed fit was available --
/// never a misleading zero.
fn sensitivity_token(sensitivity: ParameterSensitivity) -> String {
    if sensitivity.available
    {
        format!("{:.3e}", sensitivity.deviation)
    }
    else
    {
        "na".to_string()
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_row<M: StaticIsotropicMetric>(
    name: &str,
    oracle_gamma: f64,
    oracle_beta: f64,
    mass: f64,
    metric: &M,
    domain: &PpnDomain,
    fit_order: usize,
) -> Result<(), String> {
    let (sampling_spacing, radius_min, radius_max) = domain_summary(domain, mass);
    let sample_count = domain.sample_count;

    match extract_ppn(metric, domain, fit_order)
    {
        Ok(estimate) =>
        {
            let gamma_error = (estimate.gamma.asymptotic_value - oracle_gamma).abs();
            let beta_error = (estimate.beta.asymptotic_value - oracle_beta).abs();
            let gamma_relative = gamma_error / oracle_gamma.abs();
            let beta_relative = beta_error / oracle_beta.abs();
            require_finite(&[
                ("gamma", estimate.gamma.asymptotic_value),
                ("beta", estimate.beta.asymptotic_value),
                ("gamma_uncertainty", estimate.gamma.estimated_uncertainty),
                ("beta_uncertainty", estimate.beta.estimated_uncertainty),
                ("gamma_residual", estimate.gamma.fit_residual),
                ("beta_residual", estimate.beta.fit_residual),
                ("conditioning", estimate.gamma.conditioning),
            ])?;
            println!(
                "{name},{mass},{oracle_gamma},{oracle_beta},{radius_min:.3},{radius_max:.3},\
                 {:.4e},{:.4e},{sample_count},{sampling_spacing},{fit_order},\
                 {:.10},{gamma_error:.3e},{gamma_relative:.3e},{:.3e},{:.3e},{},{},{},{},\
                 {:.10},{beta_error:.3e},{beta_relative:.3e},{:.3e},{:.3e},{},{},{},{},\
                 {:.3e},ok",
                estimate.compactness_min,
                estimate.compactness_max,
                estimate.gamma.asymptotic_value,
                estimate.gamma.estimated_uncertainty,
                estimate.gamma.fit_residual,
                sensitivity_token(estimate.gamma.diagnostics.radial_window_sensitivity),
                sensitivity_token(estimate.gamma.diagnostics.fit_order_sensitivity),
                sensitivity_token(estimate.gamma.diagnostics.resolution_sensitivity),
                conditioning_token(estimate.gamma.diagnostics.conditioning_class),
                estimate.beta.asymptotic_value,
                estimate.beta.estimated_uncertainty,
                estimate.beta.fit_residual,
                sensitivity_token(estimate.beta.diagnostics.radial_window_sensitivity),
                sensitivity_token(estimate.beta.diagnostics.fit_order_sensitivity),
                sensitivity_token(estimate.beta.diagnostics.resolution_sensitivity),
                conditioning_token(estimate.beta.diagnostics.conditioning_class),
                estimate.gamma.conditioning,
            );
        },
        Err(error) =>
        {
            let (compactness_min, compactness_max) = (mass / radius_max, mass / radius_min);
            println!(
                "{name},{mass},{oracle_gamma},{oracle_beta},{radius_min:.3},{radius_max:.3},\
                 {compactness_min:.4e},{compactness_max:.4e},{sample_count},{sampling_spacing},\
                 {fit_order},nan,nan,nan,nan,nan,na,na,na,na,nan,nan,nan,nan,nan,na,na,na,na,nan,{}",
                rejection_token(&error)
            );
        },
    }
    Ok(())
}

fn main() -> Result<(), String> {
    print_experiment_header(
        "PPN parameter (gamma, beta) extraction",
        "scirust-relativity geometry core (established general relativity)",
        "asymptotic extraction of gamma, beta from isotropic weak-field metrics; numerical, not a bound.",
    );

    println!(
        "metric,mass_scale,oracle_gamma,oracle_beta,radius_min,radius_max,compactness_min,\
         compactness_max,sample_count,sampling_spacing,fit_order,gamma_estimate,gamma_abs_error,\
         gamma_relative_error,gamma_estimated_uncertainty,gamma_fit_residual,\
         gamma_window_sensitivity,gamma_order_sensitivity,gamma_resolution_sensitivity,\
         gamma_conditioning_class,beta_estimate,beta_abs_error,beta_relative_error,\
         beta_estimated_uncertainty,beta_fit_residual,beta_window_sensitivity,\
         beta_order_sensitivity,beta_resolution_sensitivity,beta_conditioning_class,\
         conditioning_indicator,status"
    );

    let exact_gr = SyntheticPpnMetric::exact(MASS, 1.0, 1.0);
    // Non-GR exact oracle: exercises extraction correctness away from (1, 1),
    // avoiding the accidental symmetry a GR-only sweep could hide.
    let exact_non_gr = SyntheticPpnMetric::exact(MASS, 0.8, 1.2);
    let contaminated = SyntheticPpnMetric::contaminated(MASS, 1.0, 1.0, 3.0, -5.0, 2.0, 4.0);
    let isotropic =
        IsotropicSchwarzschild::try_new(MASS).ok_or("invalid isotropic Schwarzschild")?;
    let isotropic_adapter =
        IsotropicChartAdapter::new(&isotropic, MASS).map_err(|e| e.to_string())?;
    let areal = Schwarzschild::try_new(MASS).ok_or("invalid Schwarzschild")?;
    let areal_adapter = IsotropicChartAdapter::new(&areal, MASS).map_err(|e| e.to_string())?;

    // Deterministic sweep: compactness window x sample count x fit order, in the
    // default linear-in-compactness spacing.
    for &compactness_max in &[0.1, 0.05]
    {
        let compactness_min = compactness_max * 0.1;
        for &sample_count in &[12, 24]
        {
            for &fit_order in &[2, 3]
            {
                let domain =
                    PpnDomain::uniform_compactness(compactness_min, compactness_max, sample_count);
                emit_row(
                    "synthetic_exact_gr",
                    1.0,
                    1.0,
                    MASS,
                    &exact_gr,
                    &domain,
                    fit_order,
                )?;
                emit_row(
                    "synthetic_exact_non_gr",
                    0.8,
                    1.2,
                    MASS,
                    &exact_non_gr,
                    &domain,
                    fit_order,
                )?;
                emit_row(
                    "synthetic_contaminated",
                    1.0,
                    1.0,
                    MASS,
                    &contaminated,
                    &domain,
                    fit_order,
                )?;
                emit_row(
                    "isotropic_schwarzschild",
                    1.0,
                    1.0,
                    MASS,
                    &isotropic_adapter,
                    &domain,
                    fit_order,
                )?;
            }
        }
    }

    // Sampling-spacing sweep: the same physical window resampled logarithmically
    // in radius instead of linearly in compactness, at one representative
    // (sample count, fit order) -- demonstrating the CSV's sampling_spacing
    // column without duplicating the full cross-product above.
    let log_domain = PpnDomain::logarithmic_radius(MASS / 0.1, MASS / 0.01, 24);
    emit_row(
        "synthetic_exact_gr",
        1.0,
        1.0,
        MASS,
        &exact_gr,
        &log_domain,
        3,
    )?;
    emit_row(
        "synthetic_contaminated",
        1.0,
        1.0,
        MASS,
        &contaminated,
        &log_domain,
        3,
    )?;
    emit_row(
        "isotropic_schwarzschild",
        1.0,
        1.0,
        MASS,
        &isotropic_adapter,
        &log_domain,
        3,
    )?;

    // Coordinate-validity: areal Schwarzschild is rejected, not silently trusted.
    let areal_domain = PpnDomain::uniform_compactness(0.005, 0.05, 16);
    emit_row(
        "areal_schwarzschild",
        1.0,
        1.0,
        MASS,
        &areal_adapter,
        &areal_domain,
        3,
    )?;

    println!("# interpretation: the exact synthetic metrics (GR and non-GR) recover the injected");
    println!("# pair to near machine precision; contamination biases the finite-radius estimators");
    println!("# while the extrapolation recovers the injected values (a weaker-field window, more");
    println!("# samples, or a better-matched order improves it -- each reported as its own");
    println!("# sensitivity axis, not blended away); isotropic Schwarzschild converges to");
    println!(
        "# gamma = beta = 1 under both sampling spacings; and areal Schwarzschild is rejected"
    );
    println!("# as non-isotropic. Numerical extraction, not a bound.");
    Ok(())
}
