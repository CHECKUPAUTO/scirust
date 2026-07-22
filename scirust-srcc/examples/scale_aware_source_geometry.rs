//! Deterministic mechanism benchmark for scale-aware SRCC source geometry.
//!
//! Three sections. Every scientific claim printed as `true`, `ok`, `certified`
//! or `typed_breakdown` is asserted at runtime and the process exits non-zero
//! if it fails; purely descriptive fields (error names, numeric magnitudes)
//! are printed verbatim from the computation:
//!
//! 1. **Compatibility** — the scale-aware pipeline under
//!    `SrccSourceGeometrySpec::RawEuclidean` must equal the historical
//!    source-clustered pipeline exactly (fit, search, stable search) across the
//!    PR #720 jitter grid, and must equal the exact-source robust pipeline at
//!    zero radius.
//! 2. **Global and anisotropic scaling** — a two-operating-state fixture whose
//!    state signal lives on a coordinate three orders of magnitude below a
//!    pure-noise coordinate. No raw radius groups the intra-state jitter while
//!    keeping the states apart: below the signal separation raw clustering
//!    fragments into singletons, in the bridging band (exercised here, asserted
//!    at every scale) it fails with a typed consensus ambiguity, and above the
//!    noise spread it silently merges the states by majority vote. Refit
//!    robust-diagonal geometry groups and separates simultaneously, with an
//!    identical structural outcome under common source rescaling by 1, 1e3,
//!    1e6, 1e9.
//! 3. **Majority breakdown (honest negative)** — the geometry is fitted on the
//!    sources pooled across every view, so with two six-to-three views the
//!    pooled majority (12 vs 6) survives any single removal and the
//!    leave-one-out certificate is perfect. An unbalanced view pair
//!    (four-to-three plus three-to-three) starts at a pooled 7 vs 6: one
//!    removal balances the pooled states 6–6, the MAD of the signal coordinate
//!    inflates to the separation itself, and the reduced geometry can no longer
//!    separate the states — the leave-one-out evaluation fails with a typed
//!    ambiguity. This is the documented breakdown limit of the robust scale,
//!    reported, not hidden.
//!
//! Output is deterministic CSV on stdout (`{:.17e}` for floats); run twice and
//! compare byte-for-byte (`cmp` / SHA-256). No timestamps, no timings.

use scirust_srcc::{
    SRCC_DIMENSION, SrccCase, SrccConfig, SrccRobustFitError, SrccRobustSourceClusteringConfig,
    SrccScaleAwareSourceClusteredSearchConfig, SrccScaleAwareSourceClusteringConfig,
    SrccSourceClusteredSearchConfig, SrccSourceGeometrySpec, SrccStableSearchConfig,
    SrccTransportSample, Vector16, basis_vector,
    evaluate_scale_aware_source_clustered_robust_leave_one_out_stability,
    fit_robust_srcc_projector_from_views,
    fit_scale_aware_source_clustered_robust_srcc_projector_from_views,
    fit_source_clustered_robust_srcc_projector_from_views,
    search_scale_aware_source_clustered_robust_srcc_structures_from_views,
    search_source_clustered_robust_srcc_structures_from_views,
    search_stable_scale_aware_source_clustered_robust_srcc_structures_from_views,
    search_stable_source_clustered_robust_srcc_structures_from_views, squared_norm,
};

use scirust_multivariate::{RobustScaleMethod, RobustScalerConfig, ZeroScalePolicy};

const JITTER_RADIUS: f64 = 1.0001e-2;
const SCALED_RADIUS: f64 = 10.0;
const GLOBAL_SCALES: [f64; 4] = [1.0, 1.0e3, 1.0e6, 1.0e9];

fn base_config() -> SrccConfig {
    SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: 2,
        maximum_rounds: 2,
        energy_floor: 1.0e-30,
    }
}

fn stable_config() -> SrccStableSearchConfig {
    SrccStableSearchConfig {
        base_config: base_config(),
        distortion_weight: 10.0,
        maximum_frobenius_distance: 0.0,
        minimum_dimension_stability_ratio: 1.0,
        minimum_rejected_dimension: 2,
    }
}

fn diagonal_geometry() -> SrccSourceGeometrySpec {
    SrccSourceGeometrySpec::RobustDiagonal {
        scaler_config: RobustScalerConfig {
            center: true,
            scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
            zero_scale_policy: ZeroScalePolicy::DropDimension,
            minimum_scale: 0.0,
        },
    }
}

fn scale_aware(
    geometry: SrccSourceGeometrySpec,
    maximum_source_distance: f64,
) -> SrccScaleAwareSourceClusteringConfig {
    SrccScaleAwareSourceClusteringConfig {
        geometry,
        clustering: SrccRobustSourceClusteringConfig {
            maximum_source_distance,
        },
    }
}

fn normalize(vector: Vector16) -> Vector16 {
    let norm = squared_norm(&vector).sqrt();

    vector.map(|value| value / norm)
}

fn cases() -> [SrccCase; 1] {
    [SrccCase::new(
        basis_vector(8).expect("valid basis index"),
        basis_vector(2).expect("valid basis index"),
    )]
}

/// The PR #720 jittered fixture: `clean_repetitions` exact repetitions plus one
/// jittered contaminated observation per view.
fn jittered_views(
    clean_repetitions: usize,
    jitter: f64,
) -> (Vector16, Vec<SrccTransportSample>, Vec<SrccTransportSample>) {
    let source = basis_vector(1).expect("valid basis index");
    let perturbation = basis_vector(8).expect("valid basis index");
    let target = basis_vector(2).expect("valid basis index");
    let target_outlier = basis_vector(9).expect("valid basis index");

    let nearby_source = normalize(core::array::from_fn(|index| {
        source[index] + jitter * perturbation[index]
    }));

    let negative_target = target.map(|value| -value);

    let mut positive = Vec::with_capacity(clean_repetitions + 1);
    let mut negative = Vec::with_capacity(clean_repetitions + 1);

    for _ in 0..clean_repetitions
    {
        positive.push(SrccTransportSample::new(source, target));

        negative.push(SrccTransportSample::new(source, negative_target));
    }

    positive.push(SrccTransportSample::new(nearby_source, target_outlier));

    negative.push(SrccTransportSample::new(nearby_source, target_outlier));

    (source, positive, negative)
}

/// Two operating states: coordinate 0 is pure jitter around `1.0`, coordinate 8
/// carries the state signal (`0.001` vs `0.002`, jitter `1e-6`). The raw
/// inter-state distance can be as small as `0.001` while the raw intra-state
/// spread reaches `0.04`, so no raw radius separates the states; in fitted MAD
/// units (majority state pins the coordinate-8 scale at jitter level) the
/// states are hundreds of scale units apart.
fn anisotropic_state_views(majority: usize, scale: f64) -> (Vector16, Vec<SrccTransportSample>) {
    let target_a = basis_vector(2).expect("valid basis index");
    let target_b = basis_vector(3).expect("valid basis index");

    let make_source = |base8: f64, noise: f64, epsilon: f64| -> Vector16 {
        let mut source = [0.0; SRCC_DIMENSION];
        source[0] = (1.0 + noise) * scale;
        source[8] = (base8 + epsilon) * scale;
        source
    };

    let a_noise = [-0.02, -0.01, 0.0, 0.005, 0.01, 0.02];
    let a_epsilon = [0.0, 1.0e-6, -1.0e-6, 5.0e-7, -5.0e-7, 2.0e-7];
    let b_noise = [-0.02, 0.0, 0.02];
    let b_epsilon = [0.0, -1.0e-6, 1.0e-6];

    let mut view = Vec::with_capacity(majority + b_noise.len());

    for (noise, epsilon) in a_noise.iter().zip(a_epsilon.iter()).take(majority)
    {
        view.push(SrccTransportSample::new(
            make_source(0.001, *noise, *epsilon),
            target_a,
        ));
    }

    for (noise, epsilon) in b_noise.iter().zip(b_epsilon.iter())
    {
        view.push(SrccTransportSample::new(
            make_source(0.002, *noise, *epsilon),
            target_b,
        ));
    }

    (make_source(0.001, 0.0, 0.0), view)
}

fn fit_error_name(error: &SrccRobustFitError) -> &'static str {
    match error
    {
        SrccRobustFitError::Discovery(_) => "discovery",
        SrccRobustFitError::Closure(_) => "closure",
        SrccRobustFitError::InvalidMaximumSourceDistance => "invalid_maximum_source_distance",
        SrccRobustFitError::NonFiniteSourceDistance { .. } => "non_finite_source_distance",
        SrccRobustFitError::AmbiguousSourceClusterAssignment { .. } =>
        {
            "ambiguous_source_cluster_assignment"
        },
        SrccRobustFitError::AmbiguousTargetConsensus { .. } => "ambiguous_target_consensus",
        SrccRobustFitError::NonFiniteTargetDistance { .. } => "non_finite_target_distance",
        SrccRobustFitError::InvalidSourceGeometry => "invalid_source_geometry",
        SrccRobustFitError::DegenerateSourceScale { .. } => "degenerate_source_scale",
        SrccRobustFitError::NonFiniteSourceScale { .. } => "non_finite_source_scale",
        SrccRobustFitError::NoActiveSourceDimensions => "no_active_source_dimensions",
        SrccRobustFitError::CertifiedClusteringFailed { .. } => "certified_clustering_failed",
    }
}

/// Section 1: the RawEuclidean scale-aware pipeline must equal the historical
/// clustered pipeline on the PR #720 grid, and the exact-source pipeline at
/// zero radius.
fn compatibility_section() -> Result<(), String> {
    println!("# section 1: raw-euclidean compatibility (runtime-verified equalities)");
    println!("# columns: fixture,radius,pipeline,identical");

    let grid = [
        (2usize, 0.0f64, 0.0f64),
        (3, 0.0, 0.0),
        (2, 1.0e-3, JITTER_RADIUS),
        (3, 1.0e-3, JITTER_RADIUS),
        (3, 1.0e-3, 0.0),
    ];

    for (clean_repetitions, jitter, radius) in grid
    {
        let (source, positive, negative) = jittered_views(clean_repetitions, jitter);

        let views = [positive.as_slice(), negative.as_slice()];

        let historical = fit_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            SrccRobustSourceClusteringConfig {
                maximum_source_distance: radius,
            },
            base_config(),
        );

        let scale_aware_fit = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            scale_aware(SrccSourceGeometrySpec::RawEuclidean, radius),
            base_config(),
        );

        if historical != scale_aware_fit
        {
            return Err(format!(
                "fit drift: clean={clean_repetitions} jitter={jitter} radius={radius}"
            ));
        }

        println!("clean{clean_repetitions}_jitter{jitter:.0e},{radius:.17e},fit,true");

        let cases = cases();

        let historical_search = search_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            SrccSourceClusteredSearchConfig {
                source_clustering: SrccRobustSourceClusteringConfig {
                    maximum_source_distance: radius,
                },
                base_config: base_config(),
                distortion_weight: 10.0,
            },
            &cases,
            &cases,
        );

        let scale_aware_search =
            search_scale_aware_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[1.0, 0.999],
                SrccScaleAwareSourceClusteredSearchConfig {
                    source: scale_aware(SrccSourceGeometrySpec::RawEuclidean, radius),
                    base_config: base_config(),
                    distortion_weight: 10.0,
                },
                &cases,
                &cases,
            );

        if historical_search != scale_aware_search
        {
            return Err(format!(
                "search drift: clean={clean_repetitions} jitter={jitter} radius={radius}"
            ));
        }

        println!("clean{clean_repetitions}_jitter{jitter:.0e},{radius:.17e},search,true");

        let historical_stable = search_stable_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            SrccRobustSourceClusteringConfig {
                maximum_source_distance: radius,
            },
            &[1.0, 0.999],
            stable_config(),
            &cases,
            &cases,
        );

        let scale_aware_stable =
            search_stable_scale_aware_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                scale_aware(SrccSourceGeometrySpec::RawEuclidean, radius),
                &[1.0, 0.999],
                stable_config(),
                &cases,
                &cases,
            );

        if historical_stable != scale_aware_stable
        {
            return Err(format!(
                "stable-search drift: clean={clean_repetitions} jitter={jitter} radius={radius}"
            ));
        }

        println!("clean{clean_repetitions}_jitter{jitter:.0e},{radius:.17e},stable_search,true");
    }

    // Zero radius must also reproduce the exact-source robust pipeline.
    let (source, positive, negative) = jittered_views(3, 0.0);

    let views = [positive.as_slice(), negative.as_slice()];

    let exact = fit_robust_srcc_projector_from_views(&[source], &views, base_config())
        .map_err(|error| format!("exact fit failed: {error}"))?;

    let zero_radius = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
        &[source],
        &views,
        scale_aware(SrccSourceGeometrySpec::RawEuclidean, 0.0),
        base_config(),
    )
    .map_err(|error| format!("zero-radius fit failed: {error}"))?;

    if exact != zero_radius
    {
        return Err("zero-radius drift against the exact-source robust pipeline".to_string());
    }

    println!("clean3_jitter0e0,0.00000000000000000e0,exact_source_equivalence,true");

    Ok(())
}

/// Section 2: raw vs robust-diagonal geometry on the anisotropic two-state
/// fixture, replayed under common source rescaling.
fn scaling_section() -> Result<(), String> {
    println!("# section 2: anisotropic two-state fixture under common rescaling");
    println!(
        "# columns: scale,geometry,radius,outcome,rejected_dimension,loo_ratio,loo_max_frobenius"
    );

    for &scale in &GLOBAL_SCALES
    {
        let (seed, view) = anisotropic_state_views(6, scale);

        let views = [view.as_slice(), view.as_slice()];

        // Raw geometry at a radius that must bridge the states: the typed
        // ambiguity is the honest outcome (asserted at unit scale where the
        // fixture guarantees it; reported verbatim at every scale).
        let raw = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(SrccSourceGeometrySpec::RawEuclidean, 2.0e-2 * scale),
            base_config(),
        );

        let raw_outcome = match &raw
        {
            Ok(result) => format!("ok,{},nan,nan", result.projector.rejected_dimension()),
            Err(error) => format!("err:{},nan,nan,nan", fit_error_name(error)),
        };

        // The bridging radius is rescaled with the data, so the typed
        // ambiguity is the asserted outcome at every scale.
        if !matches!(
            raw,
            Err(SrccRobustFitError::AmbiguousTargetConsensus { .. })
        )
        {
            return Err(format!(
                "raw geometry failed to produce the expected typed ambiguity at scale {scale}"
            ));
        }

        println!(
            "{scale:.17e},raw_euclidean,{:.17e},{raw_outcome}",
            2.0e-2 * scale
        );

        // Robust-diagonal geometry: refit per call, so the SAME scaled radius
        // works at every common rescaling, and the leave-one-out certificate
        // must be structurally identical across scales.
        let diagonal = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), SCALED_RADIUS),
            base_config(),
        )
        .map_err(|error| format!("diagonal fit failed at scale {scale}: {error}"))?;

        if diagonal.projector.rejected_dimension() != 2
        {
            return Err(format!(
                "diagonal geometry lost the two-state structure at scale {scale}"
            ));
        }

        let stability = evaluate_scale_aware_source_clustered_robust_leave_one_out_stability(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), SCALED_RADIUS),
            base_config(),
        )
        .map_err(|error| format!("diagonal LOO failed at scale {scale}: {error}"))?;

        if stability.dimension_stability_ratio() != 1.0
        {
            return Err(format!(
                "diagonal LOO dimension stability broke at scale {scale}"
            ));
        }

        println!(
            "{scale:.17e},robust_diagonal,{SCALED_RADIUS:.17e},ok,{},{:.17e},{:.17e}",
            diagonal.projector.rejected_dimension(),
            stability.dimension_stability_ratio(),
            stability.maximum_frobenius_distance,
        );
    }

    Ok(())
}

/// Section 3: the six-to-three majority certifies; the four-to-three majority
/// breaks down under a single removal (3–3 MAD inflation) with a typed error.
fn breakdown_section() -> Result<(), String> {
    println!("# section 3: majority breakdown of the robust scale (honest negative)");
    println!("# columns: majority,outcome,detail");

    let cases = cases();

    // Six-to-three: full stable-search certification (bounded medoid movement,
    // perfect dimension stability).
    let (seed, view) = anisotropic_state_views(6, 1.0);

    let views = [view.as_slice(), view.as_slice()];

    let certified = search_stable_scale_aware_source_clustered_robust_srcc_structures_from_views(
        &[seed],
        &views,
        scale_aware(diagonal_geometry(), SCALED_RADIUS),
        &[0.999],
        SrccStableSearchConfig {
            maximum_frobenius_distance: 1.0,
            ..stable_config()
        },
        &cases,
        &cases,
    )
    .map_err(|error| format!("six-to-three stable search failed: {error}"))?;

    let selected = certified
        .selected
        .as_ref()
        .ok_or_else(|| "six-to-three stable search selected no candidate".to_string())?;

    if !selected.passes_stability_gate || selected.stability.dimension_stability_ratio() != 1.0
    {
        return Err("six-to-three certification failed".to_string());
    }

    println!(
        "6v3_pair,certified,ratio={:.17e};max_frobenius={:.17e}",
        selected.stability.dimension_stability_ratio(),
        selected.stability.maximum_frobenius_distance,
    );

    // Unbalanced view pair: the pooled sources start at 7 A vs 6 B, so a
    // single A removal balances the pooled states 6-6; the reduced MAD of the
    // signal coordinate inflates to the separation and the reduced geometry
    // can no longer separate the states. The leave-one-out evaluation must
    // fail with a typed error rather than certify silently.
    let (seed, view_four) = anisotropic_state_views(4, 1.0);
    let (_, view_three) = anisotropic_state_views(3, 1.0);

    let views = [view_four.as_slice(), view_three.as_slice()];

    // The FULL fit must succeed: the breakdown is a leave-one-out property
    // (one removal balances the pooled states), not an outright failure.
    fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
        &[seed],
        &views,
        scale_aware(diagonal_geometry(), SCALED_RADIUS),
        base_config(),
    )
    .map_err(|error| format!("unbalanced-pair full fit unexpectedly failed: {error}"))?;

    let breakdown = evaluate_scale_aware_source_clustered_robust_leave_one_out_stability(
        &[seed],
        &views,
        scale_aware(diagonal_geometry(), SCALED_RADIUS),
        base_config(),
    );

    match breakdown
    {
        Err(
            error @ scirust_srcc::SrccRobustStabilityError::Fit(
                SrccRobustFitError::AmbiguousTargetConsensus { .. },
            ),
        ) =>
        {
            println!("4v3_plus_3v3,typed_breakdown,{}", error);
        },
        Err(error) =>
        {
            return Err(format!(
                "unbalanced-view leave-one-out failed with an unexpected error kind: {error}"
            ));
        },
        Ok(_) =>
        {
            return Err(
                "unbalanced-view leave-one-out unexpectedly certified despite the MAD breakdown"
                    .to_string(),
            );
        },
    }

    Ok(())
}

fn main() {
    println!("# scale_aware_source_geometry deterministic mechanism benchmark");
    println!(
        "# jitter_radius={JITTER_RADIUS} scaled_radius={SCALED_RADIUS} \
global_scales=1,1e3,1e6,1e9"
    );

    let result = compatibility_section()
        .and_then(|()| scaling_section())
        .and_then(|()| breakdown_section());

    if let Err(message) = result
    {
        eprintln!("scale_aware_source_geometry benchmark failed: {message}");
        std::process::exit(1);
    }
}
