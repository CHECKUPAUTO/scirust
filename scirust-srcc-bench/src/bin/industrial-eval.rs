//! Phase-728 industrial evaluation binary.
//!
//! Runs the preregistered evaluation
//! (`docs/research/SRCC_INDUSTRIAL_BENCHMARK_PREREGISTRATION.md`) on the
//! three workloads (C-MAPSS FD001, C-MAPSS FD003, SECOM) under the
//! frozen configuration embedded from `configs/phase728.json` (its SHA-256
//! is recorded in the run metadata). No network access; every input file is
//! verified against its pinned checksum before use; a missing file aborts
//! with instructions to run `scripts/fetch_industrial_datasets.sh`.
//!
//! Outputs:
//! - `<out>/industrial_728.jsonl` — every measurement as a `BenchRecord`;
//! - `<out>/manifests/*.json` — dataset/split/contamination manifests,
//!   imputer record, SRCC typed outcomes;
//! - `<out>/manifests/run_metadata.json` — environment identity (git
//!   commit, toolchain, configuration hash); **excluded** from the
//!   determinism comparison by design (identity may vary, science may not).
//!
//! Determinism: run twice, `cmp` the JSONL and scientific manifests.
//! Stdout carries only deterministic section summaries.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_multivariate::{
    FittedDistanceMetric, Matrix as MultivariateMatrix, RobustScaleMethod, RobustScalerConfig,
    ZeroScalePolicy,
};
use scirust_solvers::combinatorial::{
    CertifiedClusteringMode, CertifiedMedoidClusteringConfig, DistanceMatrix,
    certified_medoid_clustering,
};
use scirust_srcc::{
    CertifiedClusteringMode as SrccCertifiedMode, SRCC_DIMENSION, SrccConfig, SrccProjector,
    SrccRobustSourceClusteringConfig, SrccScaleAwareSourceClusteringConfig,
    SrccSourceClusteringSolver, SrccSourceGeometrySpec, SrccTransportSample, SrccTrustModel,
    SrccTrustPolicy, Vector16, basis_vector, evaluate_robust_leave_one_out_stability,
    fit_certified_source_clustered_robust_srcc_projector_from_views,
    fit_robust_srcc_projector_from_views,
    fit_scale_aware_source_clustered_robust_srcc_projector_from_views,
    fit_source_clustered_robust_srcc_projector_from_views,
    fit_trusted_robust_srcc_projector_from_views,
};
use scirust_srcc_bench::{
    AdapterOutput, BaselineAdapter, ContaminationConfig, ContaminationKind, CusumAdapter,
    DbscanAdapter, EwmaAdapter, FittedImputer, HotellingT2Adapter, IsolationForestAdapter,
    LofAdapter, MahalanobisAdapter, MissingValuePolicy, RecordKey, RobustRegressionAdapter,
    RunMetadata, SplitStrategy, TabularDataset, TransportViewSpec, alarm_records,
    anomaly_label_records, anomaly_score_records, apply_contamination, build_transport_views,
    metrics::auroc, paired_bootstrap, paired_differences, parse_cmapss_training, parse_secom,
    regression_records, sha256_hex, split_dataset,
};
use scirust_unsupervised::{DbscanConfig, IForestConfig, LofConfig};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const TRAIN_FD001_SHA: &str = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8";
const TRAIN_FD003_SHA: &str = "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2";
const SECOM_DATA_SHA: &str = "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c";
const SECOM_LABELS_SHA: &str = "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03";

#[derive(Deserialize)]
struct Config {
    bootstrap: BootstrapConfig,
    cmapss: CmapssConfig,
    cmapss_fd003: CmapssFd003Config,
    secom: SecomConfig,
}

#[derive(Deserialize)]
struct BootstrapConfig {
    resamples: usize,
    level: f64,
    seed: u64,
}

#[derive(Deserialize)]
struct CmapssConfig {
    regression: RegressionConfig,
    srcc: SrccSectionConfig,
}

#[derive(Deserialize)]
struct RegressionConfig {
    split_seed: u64,
    #[serde(default)]
    decimation_stride: usize,
    missing_maximum_fraction: f64,
    train_fraction: f64,
    validation_fraction: f64,
    contamination_seed: u64,
    fractions: Vec<f64>,
    coherent_feature_offset: f64,
    coherent_target_offset: f64,
    huber_delta: f64,
    trimmed_fraction: f64,
    median_of_means_blocks: usize,
    median_of_means_seed: u64,
    paired_fraction: f64,
}

#[derive(Deserialize)]
struct SrccSectionConfig {
    #[serde(default)]
    engines: Vec<u64>,
    channel_columns: Vec<usize>,
    horizon: usize,
    #[serde(default)]
    decimation_stride: usize,
    raw_radii: Vec<f64>,
    scaled_radii: Vec<f64>,
    rescale_column: usize,
    rescale_factors: Vec<f64>,
    hybrid_maximum_nodes: usize,
    hybrid_maximum_iterations: usize,
    #[serde(default)]
    stability_engines: Vec<u64>,
    #[serde(default)]
    trust_engines: Vec<u64>,
    #[serde(default)]
    trust_attacked_view: usize,
    #[serde(default)]
    trust_target_shift: f64,
    #[serde(default)]
    trust_group_bound: f64,
}

#[derive(Deserialize)]
struct SecomConfig {
    train_fraction: f64,
    validation_fraction: f64,
    missing_maximum_fraction: f64,
    isolation_forest: IForestJson,
    lof_k_grid: Vec<usize>,
    mahalanobis_ridge_grid: Vec<f64>,
    dbscan_eps_grid: Vec<f64>,
    dbscan_min_pts: usize,
}

#[derive(Deserialize)]
struct IForestJson {
    n_trees: usize,
    subsample_size: usize,
    max_depth: usize,
    seed: u64,
}

#[derive(Deserialize)]
struct CmapssFd003Config {
    regression: RegressionConfig,
    srcc: SrccSectionConfig,
    stream: StreamConfig,
}

#[derive(Deserialize)]
struct StreamConfig {
    monitored_column: usize,
    train_engine_rank: usize,
    evaluation_engine_rank: usize,
    burst_fraction: f64,
    burst_seed: u64,
    burst_shift: f64,
    cusum_k: f64,
    cusum_h: f64,
    ewma_lambda: f64,
    ewma_l: f64,
}

fn read_verified(path: &Path, expected_sha: Option<&str>) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}. Run scripts/fetch_industrial_datasets.sh first.",
            path.display()
        )
    });

    if let Some(expected) = expected_sha
    {
        let actual = sha256_hex(&bytes);

        assert!(
            actual == expected,
            "checksum mismatch for {}: expected {expected}, found {actual}",
            path.display()
        );
    }

    String::from_utf8(bytes).expect("dataset files are valid UTF-8")
}

fn frobenius_between(left: &SrccProjector, right: &SrccProjector) -> f64 {
    let mut sum = 0.0;

    for index in 0..SRCC_DIMENSION
    {
        let basis = basis_vector(index).expect("index < 16");
        let a = left.apply(&basis);
        let b = right.apply(&basis);

        for slot in 0..SRCC_DIMENSION
        {
            sum += (a[slot] - b[slot]).powi(2);
        }
    }

    sum.sqrt()
}

/// The harness's own greedy reference for H4: deterministic first-fit in
/// index order with observed-medoid cost (documented: this is the
/// benchmark's greedy baseline, not `scirust-srcc`'s complete-link pass).
fn greedy_first_fit_reference(matrix: &DistanceMatrix, diameter: f64) -> (usize, f64) {
    let n = matrix.size;
    let mut assignments: Vec<usize> = vec![0; n];
    let mut cluster_count = 0usize;

    for point in 0..n
    {
        let mut chosen = None;

        for cluster in 0..cluster_count
        {
            let compatible = (0..point)
                .filter(|&other| assignments[other] == cluster)
                .all(|other| matrix.distance(point, other) <= diameter);

            if compatible
            {
                chosen = Some(cluster);
                break;
            }
        }

        assignments[point] = chosen.unwrap_or_else(|| {
            cluster_count += 1;
            cluster_count - 1
        });
    }

    let mut total_cost = 0.0;

    for cluster in 0..cluster_count
    {
        let members: Vec<usize> = (0..n).filter(|&p| assignments[p] == cluster).collect();

        let mut best = f64::INFINITY;

        for &candidate in &members
        {
            let cost: f64 = members
                .iter()
                .map(|&member| matrix.distance(member, candidate))
                .sum();

            if cost.total_cmp(&best).is_lt()
            {
                best = cost;
            }
        }

        total_cost += best;
    }

    (cluster_count, total_cost)
}

fn sources_of(view: &[SrccTransportSample]) -> Vec<Vector16> {
    view.iter().map(|sample| sample.source).collect()
}

/// Deterministic per-group temporal decimation: within each group, keep
/// every `stride`-th row in `(time_index, row)` order. A stride of 0 or 1 is
/// a no-op. Real trajectories give one near-duplicate source per cycle, so
/// this keeps per-view source sets small enough for certified clustering and
/// leave-one-out refitting without tuning on outcomes.
fn decimate_by_group(dataset: &TabularDataset, stride: usize) -> TabularDataset {
    if stride <= 1
    {
        return dataset.clone();
    }

    let groups = dataset.groups.as_ref().expect("decimation needs groups");
    let time = dataset.time_index.as_ref().expect("decimation needs time");

    let mut distinct: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !distinct.contains(&group)
        {
            distinct.push(group);
        }
    }

    distinct.sort_unstable();

    let mut keep: Vec<usize> = Vec::new();

    for &group in &distinct
    {
        let mut members: Vec<usize> = (0..dataset.sample_count())
            .filter(|&row| groups[row] == group)
            .collect();

        members.sort_by_key(|&row| (time[row], row));

        keep.extend(members.iter().copied().step_by(stride));
    }

    keep.sort_unstable();
    dataset.select_rows(&keep)
}

fn euclidean_distance(a: &Vector16, b: &Vector16) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0, |sum, (x, y)| sum + (x - y).powi(2))
        .sqrt()
}

enum SourceGeometry {
    Raw,
    Fitted(Box<FittedDistanceMetric>),
}

fn distance_matrix_for(sources: &[Vector16], geometry: &SourceGeometry) -> DistanceMatrix {
    let n = sources.len();
    let mut values = vec![0.0; n * n];

    for i in 0..n
    {
        for j in (i + 1)..n
        {
            let distance = match geometry
            {
                SourceGeometry::Raw => euclidean_distance(&sources[i], &sources[j]),
                SourceGeometry::Fitted(metric) => metric
                    .distance(&sources[i], &sources[j])
                    .expect("fitted metric distances are finite on finite sources"),
            };

            values[i * n + j] = distance;
            values[j * n + i] = distance;
        }
    }

    DistanceMatrix::new(n, values).expect("constructed matrices are valid")
}

fn fit_diagonal_metric(pooled_sources: &[Vector16]) -> FittedDistanceMetric {
    let data = MultivariateMatrix {
        rows: pooled_sources.len(),
        cols: SRCC_DIMENSION,
        data: pooled_sources
            .iter()
            .map(|source| source.to_vec())
            .collect(),
    };

    FittedDistanceMetric::fit_robust_diagonal(
        &data,
        RobustScalerConfig {
            center: true,
            scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
            zero_scale_policy: ZeroScalePolicy::DropDimension,
            minimum_scale: 0.0,
        },
    )
    .expect("pooled industrial sources have at least one varying dimension")
}

fn srcc_fit_config() -> SrccConfig {
    SrccConfig::default()
}

fn regression_adapters(config: &RegressionConfig) -> Vec<RobustRegressionAdapter> {
    vec![
        RobustRegressionAdapter::ordinary_least_squares(),
        RobustRegressionAdapter::huber(config.huber_delta),
        RobustRegressionAdapter::trimmed(config.trimmed_fraction),
        RobustRegressionAdapter::median_of_means(
            config.median_of_means_blocks,
            config.median_of_means_seed,
        ),
    ]
}

fn coherent_kind(config: &RegressionConfig) -> ContaminationKind {
    ContaminationKind::CoherentAlternativeCluster {
        feature_offset: config.coherent_feature_offset,
        target_offset: config.coherent_target_offset,
    }
}

/// Section: regression under training contamination + paired leave-one-group-out.
#[allow(clippy::too_many_arguments)]
fn regression_section(
    label: &str,
    dataset: &TabularDataset,
    config: &RegressionConfig,
    bootstrap: &BootstrapConfig,
    records: &mut Vec<BenchRecord>,
    manifests: &mut BTreeMap<String, String>,
) {
    let decimated_dataset = decimate_by_group(dataset, config.decimation_stride);
    let dataset = &decimated_dataset;

    let split = split_dataset(
        dataset,
        &SplitStrategy::GroupedHoldout {
            train_fraction: config.train_fraction,
            validation_fraction: config.validation_fraction,
        },
        config.split_seed,
        Some("unit"),
    )
    .expect("grouped split is valid");

    manifests.insert(
        format!("{label}_split.json"),
        serde_json::to_string_pretty(&split.manifest).expect("split manifests serialize"),
    );

    let train_raw = dataset.select_rows(&split.train);
    let test_raw = dataset.select_rows(&split.test);

    // Preregistered schema stage: the train-fitted imputer also drops
    // constant training columns (rank-deficiency guards for least squares).
    let imputer = FittedImputer::fit(
        &train_raw.features,
        MissingValuePolicy {
            maximum_missing_fraction: config.missing_maximum_fraction,
        },
    )
    .expect("training features keep at least one varying column");

    manifests.insert(
        format!("{label}_regression_imputer.json"),
        serde_json::to_string_pretty(&imputer).expect("imputer serializes"),
    );

    let train = imputer.transform(&train_raw).expect("shapes match");
    let test = imputer.transform(&test_raw).expect("shapes match");

    let mut contamination_manifests = Vec::new();

    for &fraction in &config.fractions
    {
        for (kind_name, kind) in [
            ("coherent_cluster", coherent_kind(config)),
            ("target_flip", ContaminationKind::TargetFlip),
        ]
        {
            let (corrupted, manifest) = apply_contamination(
                &train,
                &ContaminationConfig {
                    kind,
                    fraction,
                    seed: config.contamination_seed,
                },
            )
            .expect("contamination request is valid");

            contamination_manifests.push(manifest);

            for adapter in regression_adapters(config)
            {
                let key = RecordKey {
                    kernel: format!("industrial_728/{label}_regression"),
                    dataset: format!("{label}/{kind_name}_{fraction:.2}"),
                    method: adapter.name().into(),
                    seed: config.contamination_seed,
                };

                match adapter.run(&corrupted, &test)
                {
                    Ok(AdapterOutput::Predictions(predictions)) =>
                    {
                        records.extend(
                            regression_records(&key, &predictions, &test.targets)
                                .expect("test targets are finite"),
                        );
                    },
                    Ok(_) => unreachable!("regression adapters produce predictions"),
                    Err(error) =>
                    {
                        // A typed failure is a result: recorded as fit_ok = 0
                        // with the error text in the manifest.
                        records.push(BenchRecord::new(
                            key.kernel.clone(),
                            key.dataset.clone(),
                            key.method.clone(),
                            key.seed,
                            "fit_ok",
                            0.0,
                        ));

                        contamination_manifests.push(manifest_error_stub(&key, &error.to_string()));
                    },
                }
            }
        }
    }

    manifests.insert(
        format!("{label}_contamination.json"),
        serde_json::to_string_pretty(&contamination_manifests)
            .expect("contamination manifests serialize"),
    );

    // Paired leave-one-group-out at the preregistered fraction.
    let groups = dataset
        .groups
        .as_ref()
        .expect("grouped datasets have groups");

    let mut distinct: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !distinct.contains(&group)
        {
            distinct.push(group);
        }
    }

    distinct.sort_unstable();

    let mut per_method: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut paired_dropped_units = 0usize;

    for &held_out in &distinct
    {
        let split = split_dataset(
            dataset,
            &SplitStrategy::LeaveOneGroupOut {
                held_out_group: held_out,
            },
            0,
            Some("unit"),
        )
        .expect("every group occurs");

        let train_raw = dataset.select_rows(&split.train);
        let test_raw = dataset.select_rows(&split.test);

        let unit_imputer = FittedImputer::fit(
            &train_raw.features,
            MissingValuePolicy {
                maximum_missing_fraction: config.missing_maximum_fraction,
            },
        )
        .expect("training features keep at least one varying column");

        let train = unit_imputer.transform(&train_raw).expect("shapes match");
        let test = unit_imputer.transform(&test_raw).expect("shapes match");

        let (corrupted, _) = apply_contamination(
            &train,
            &ContaminationConfig {
                kind: coherent_kind(config),
                fraction: config.paired_fraction,
                seed: config.contamination_seed,
            },
        )
        .expect("contamination request is valid");

        let mut unit_rmse: Vec<(&'static str, f64)> = Vec::new();
        let mut unit_ok = true;

        for adapter in [
            RobustRegressionAdapter::ordinary_least_squares(),
            RobustRegressionAdapter::huber(config.huber_delta),
            RobustRegressionAdapter::trimmed(config.trimmed_fraction),
        ]
        {
            match adapter.run(&corrupted, &test)
            {
                Ok(AdapterOutput::Predictions(predictions)) =>
                {
                    let rmse = regression_records(
                        &RecordKey {
                            kernel: String::new(),
                            dataset: String::new(),
                            method: String::new(),
                            seed: 0,
                        },
                        &predictions,
                        &test.targets,
                    )
                    .expect("test targets are finite")[0]
                        .value;

                    unit_rmse.push((adapter.name(), rmse));
                },
                Ok(_) => unreachable!("regression adapters produce predictions"),
                Err(_) => unit_ok = false,
            }
        }

        // Pairing integrity: a unit enters the paired vectors only when
        // every method succeeded on it; dropped units are counted.
        if unit_ok
        {
            for (name, rmse) in unit_rmse
            {
                per_method.entry(name).or_default().push(rmse);
            }
        }
        else
        {
            paired_dropped_units += 1;
        }
    }

    records.push(BenchRecord::new(
        format!("industrial_728/{label}_paired"),
        format!("{label}/leave_one_group_out_{:.2}", config.paired_fraction),
        "all_methods",
        0,
        "paired_dropped_units",
        paired_dropped_units as f64,
    ));

    for (baseline, robust) in [("ols", "huber_irls"), ("ols", "trimmed_ls")]
    {
        let differences = paired_differences(&per_method[baseline], &per_method[robust])
            .expect("per-unit vectors are aligned and finite");

        let report = paired_bootstrap(
            &differences,
            bootstrap.resamples,
            bootstrap.level,
            bootstrap.seed,
        )
        .expect("enough units for the bootstrap");

        records.push(
            BenchRecord::new(
                format!("industrial_728/{label}_paired"),
                format!("{label}/leave_one_group_out_{:.2}", config.paired_fraction),
                format!("{baseline}_minus_{robust}"),
                bootstrap.seed,
                "rmse_paired_mean_difference",
                report.mean_difference,
            )
            .with_ci(report.confidence_interval),
        );
    }
}

fn manifest_error_stub(key: &RecordKey, detail: &str) -> scirust_srcc_bench::ContaminationManifest {
    // Reuse the manifest JSON stream for typed failures: a zero-effect
    // manifest whose kind field is irrelevant; the detail is what matters.
    // (Kept deliberately simple: failures also appear as fit_ok = 0 rows.)
    scirust_srcc_bench::ContaminationManifest {
        kind: ContaminationKind::TargetFlip,
        seed: key.seed,
        requested_fraction: 0.0,
        affected_rows: Vec::new(),
        appended_rows: 0,
        input_sha256: format!("typed_failure:{}:{}", key.method, detail),
        output_sha256: String::new(),
    }
}

/// Section: SRCC clustering questions (H1 invariance, H4 certified-vs-greedy)
/// plus the srcc-level pipeline outcomes.
fn srcc_section(
    label: &str,
    dataset: &TabularDataset,
    section: &SrccSectionConfig,
    records: &mut Vec<BenchRecord>,
    manifests: &mut BTreeMap<String, String>,
) {
    let restricted = if section.engines.is_empty()
    {
        dataset.clone()
    }
    else
    {
        let groups = dataset.groups.as_ref().expect("srcc datasets have groups");
        let rows: Vec<usize> = (0..dataset.sample_count())
            .filter(|&row| section.engines.contains(&groups[row]))
            .collect();

        dataset.select_rows(&rows)
    };

    let decimated = decimate_by_group(&restricted, section.decimation_stride);

    let spec = TransportViewSpec {
        channel_columns: section.channel_columns.clone(),
        horizon: section.horizon,
        center_per_trajectory: true,
    };

    let views = build_transport_views(&decimated, &spec).expect("view construction is valid");

    let mut outcome_log: Vec<String> = Vec::new();

    // Solvers-level H1/H4: per (geometry, radius, view) certified clustering
    // + the harness greedy reference, plus assignment invariance under the
    // preregistered unit rescalings.
    let baseline_metric = fit_diagonal_metric(&pooled_sources(&views.views));

    let geometries: Vec<(&str, SourceGeometry, &Vec<f64>)> = vec![
        ("raw_euclidean", SourceGeometry::Raw, &section.raw_radii),
        (
            "robust_diagonal",
            SourceGeometry::Fitted(Box::new(baseline_metric)),
            &section.scaled_radii,
        ),
    ];

    // Baseline assignments per (geometry, radius, view).
    let mut baseline_assignments: BTreeMap<(String, String, usize), Vec<usize>> = BTreeMap::new();

    for (geometry_name, geometry, radii) in &geometries
    {
        for &radius in radii.iter()
        {
            for (view_index, view) in views.views.iter().enumerate()
            {
                let sources = sources_of(view);
                let matrix = distance_matrix_for(&sources, geometry);

                let certified = certified_medoid_clustering(
                    &matrix,
                    CertifiedMedoidClusteringConfig {
                        maximum_cluster_diameter: radius,
                        mode: CertifiedClusteringMode::Hybrid {
                            maximum_nodes: section.hybrid_maximum_nodes,
                            maximum_iterations: section.hybrid_maximum_iterations,
                        },
                    },
                )
                .expect("distance matrices are valid");

                let (greedy_count, greedy_cost) = greedy_first_fit_reference(&matrix, radius);

                let dataset_key =
                    format!("{label}/{geometry_name}/radius_{radius}/view_{view_index}");

                let key = |metric: &str, value: f64| {
                    BenchRecord::new(
                        format!("industrial_728/{label}_srcc_clustering"),
                        dataset_key.clone(),
                        "certified_hybrid",
                        0,
                        metric,
                        value,
                    )
                };

                records.push(key(
                    "cluster_count",
                    certified.certificate.objective_cluster_count as f64,
                ));
                records.push(key(
                    "medoid_cost",
                    certified.certificate.objective_medoid_cost,
                ));
                records.push(key(
                    "proven_optimal",
                    f64::from(u8::from(certified.certificate.proven_optimal)),
                ));
                records.push(key("optimality_gap", certified.certificate.optimality_gap));
                records.push(key(
                    "explored_nodes",
                    certified.certificate.explored_nodes as f64,
                ));

                records.push(BenchRecord::new(
                    format!("industrial_728/{label}_srcc_clustering"),
                    dataset_key.clone(),
                    "greedy_first_fit",
                    0,
                    "cluster_count",
                    greedy_count as f64,
                ));
                records.push(BenchRecord::new(
                    format!("industrial_728/{label}_srcc_clustering"),
                    dataset_key.clone(),
                    "greedy_first_fit",
                    0,
                    "medoid_cost",
                    greedy_cost,
                ));

                baseline_assignments.insert(
                    (
                        (*geometry_name).to_string(),
                        format!("{radius}"),
                        view_index,
                    ),
                    certified.assignments,
                );
            }
        }
    }

    // H1 invariance: rescale one channel, rebuild everything, compare
    // assignments. The rescale is itself a manifested contamination.
    for &factor in &section.rescale_factors
    {
        let (rescaled, rescale_manifest) = apply_contamination(
            &decimated,
            &ContaminationConfig {
                kind: ContaminationKind::CoordinateScaleShift {
                    column: section.rescale_column,
                    factor,
                },
                fraction: 1.0,
                seed: 0,
            },
        )
        .expect("rescale request is valid");

        outcome_log.push(format!(
            "rescale factor {factor} column {} affected {} rows",
            section.rescale_column,
            rescale_manifest.affected_rows.len(),
        ));

        let rescaled_views =
            build_transport_views(&rescaled, &spec).expect("view construction is valid");

        let rescaled_metric = fit_diagonal_metric(&pooled_sources(&rescaled_views.views));

        let rescaled_geometries: Vec<(&str, SourceGeometry, &Vec<f64>)> = vec![
            ("raw_euclidean", SourceGeometry::Raw, &section.raw_radii),
            (
                "robust_diagonal",
                SourceGeometry::Fitted(Box::new(rescaled_metric)),
                &section.scaled_radii,
            ),
        ];

        for (geometry_name, geometry, radii) in &rescaled_geometries
        {
            for &radius in radii.iter()
            {
                for (view_index, view) in rescaled_views.views.iter().enumerate()
                {
                    let sources = sources_of(view);
                    let matrix = distance_matrix_for(&sources, geometry);

                    let certified = certified_medoid_clustering(
                        &matrix,
                        CertifiedMedoidClusteringConfig {
                            maximum_cluster_diameter: radius,
                            mode: CertifiedClusteringMode::Hybrid {
                                maximum_nodes: section.hybrid_maximum_nodes,
                                maximum_iterations: section.hybrid_maximum_iterations,
                            },
                        },
                    )
                    .expect("distance matrices are valid");

                    let baseline = &baseline_assignments[&(
                        (*geometry_name).to_string(),
                        format!("{radius}"),
                        view_index,
                    )];

                    let invariant = f64::from(u8::from(&certified.assignments == baseline));

                    records.push(BenchRecord::new(
                        format!("industrial_728/{label}_srcc_invariance"),
                        format!(
                            "{label}/{geometry_name}/radius_{radius}/view_{view_index}/factor_{factor}"
                        ),
                        "certified_hybrid",
                        0,
                        "assignment_invariant",
                        invariant,
                    ));
                }
            }
        }
    }

    // srcc-level pipeline outcomes (typed errors are results).
    let view_slices: Vec<&[SrccTransportSample]> = views.views.iter().map(Vec::as_slice).collect();

    let seeds: Vec<Vector16> = (0..4).map(|i| basis_vector(i).expect("i < 16")).collect();

    let pipeline_outcome = |records: &mut Vec<BenchRecord>,
                            outcome_log: &mut Vec<String>,
                            pipeline: &str,
                            radius: f64,
                            outcome: Result<(), String>| {
        let ok = f64::from(u8::from(outcome.is_ok()));

        if let Err(detail) = outcome
        {
            outcome_log.push(format!("{pipeline} radius {radius}: {detail}"));
        }
        else
        {
            outcome_log.push(format!("{pipeline} radius {radius}: ok"));
        }

        records.push(BenchRecord::new(
            format!("industrial_728/{label}_srcc_pipeline"),
            format!("{label}/radius_{radius}"),
            pipeline.to_string(),
            0,
            "fit_ok",
            ok,
        ));
    };

    pipeline_outcome(
        records,
        &mut outcome_log,
        "historical_exact",
        0.0,
        fit_robust_srcc_projector_from_views(&seeds, &view_slices, srcc_fit_config())
            .map(|_| ())
            .map_err(|error| error.to_string()),
    );

    for &radius in &section.raw_radii
    {
        pipeline_outcome(
            records,
            &mut outcome_log,
            "source_clustered_raw",
            radius,
            fit_source_clustered_robust_srcc_projector_from_views(
                &seeds,
                &view_slices,
                SrccRobustSourceClusteringConfig {
                    maximum_source_distance: radius,
                },
                srcc_fit_config(),
            )
            .map(|_| ())
            .map_err(|error| error.to_string()),
        );
    }

    for &radius in &section.scaled_radii
    {
        let scale_aware_config = SrccScaleAwareSourceClusteringConfig {
            geometry: SrccSourceGeometrySpec::RobustDiagonal {
                scaler_config: RobustScalerConfig {
                    center: true,
                    scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
                    zero_scale_policy: ZeroScalePolicy::DropDimension,
                    minimum_scale: 0.0,
                },
            },
            clustering: SrccRobustSourceClusteringConfig {
                maximum_source_distance: radius,
            },
        };

        pipeline_outcome(
            records,
            &mut outcome_log,
            "scale_aware_greedy",
            radius,
            fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                &seeds,
                &view_slices,
                scale_aware_config,
                srcc_fit_config(),
            )
            .map(|_| ())
            .map_err(|error| error.to_string()),
        );

        match fit_certified_source_clustered_robust_srcc_projector_from_views(
            &seeds,
            &view_slices,
            scale_aware_config,
            SrccSourceClusteringSolver::CertifiedMedoid {
                mode: SrccCertifiedMode::Hybrid {
                    maximum_nodes: section.hybrid_maximum_nodes,
                    maximum_iterations: section.hybrid_maximum_iterations,
                },
            },
            srcc_fit_config(),
        )
        {
            Ok(result) =>
            {
                pipeline_outcome(
                    records,
                    &mut outcome_log,
                    "scale_aware_certified",
                    radius,
                    Ok(()),
                );

                let proven = result
                    .view_certificates
                    .iter()
                    .filter(|certificate| certificate.proven_optimal)
                    .count();

                records.push(BenchRecord::new(
                    format!("industrial_728/{label}_srcc_pipeline"),
                    format!("{label}/radius_{radius}"),
                    "scale_aware_certified",
                    0,
                    "proven_view_fraction",
                    proven as f64 / result.view_certificates.len().max(1) as f64,
                ));
            },
            Err(error) =>
            {
                pipeline_outcome(
                    records,
                    &mut outcome_log,
                    "scale_aware_certified",
                    radius,
                    Err(error.to_string()),
                );
            },
        }
    }

    manifests.insert(
        format!("{label}_srcc_outcomes.json"),
        serde_json::to_string_pretty(&outcome_log).expect("outcome log serializes"),
    );
}

fn pooled_sources(views: &[Vec<SrccTransportSample>]) -> Vec<Vector16> {
    views.iter().flat_map(|view| sources_of(view)).collect()
}

/// Section: SRCC leave-one-out stability and the trust vacuity measurement.
fn cmapss_stability_and_trust(
    dataset: &TabularDataset,
    section: &SrccSectionConfig,
    records: &mut Vec<BenchRecord>,
    manifests: &mut BTreeMap<String, String>,
) {
    let mut outcome_log: Vec<String> = Vec::new();

    let spec = TransportViewSpec {
        channel_columns: section.channel_columns.clone(),
        horizon: section.horizon,
        center_per_trajectory: true,
    };

    let subset = |engines: &[u64]| -> TabularDataset {
        let groups = dataset.groups.as_ref().expect("grouped");
        let rows: Vec<usize> = (0..dataset.sample_count())
            .filter(|&row| engines.contains(&groups[row]))
            .collect();

        // Same decimation as the clustering section: leave-one-out refitting
        // over hundreds of per-cycle sources is intractable and dominated by
        // near-duplicates.
        decimate_by_group(&dataset.select_rows(&rows), section.decimation_stride)
    };

    let seeds: Vec<Vector16> = (0..4).map(|i| basis_vector(i).expect("i < 16")).collect();

    // Stability (preregistered engines).
    let stability_views =
        build_transport_views(&subset(&section.stability_engines), &spec).expect("views build");

    let stability_slices: Vec<&[SrccTransportSample]> =
        stability_views.views.iter().map(Vec::as_slice).collect();

    match evaluate_robust_leave_one_out_stability(&seeds, &stability_slices, srcc_fit_config())
    {
        Ok(report) =>
        {
            for (metric, value) in [
                ("mean_frobenius_distance", report.mean_frobenius_distance),
                (
                    "maximum_frobenius_distance",
                    report.maximum_frobenius_distance,
                ),
                (
                    "stable_dimension_count",
                    report.stable_dimension_count as f64,
                ),
                ("removal_count", report.removal_count() as f64),
            ]
            {
                records.push(BenchRecord::new(
                    "industrial_728/cmapss_stability",
                    "cmapss/stability_engines",
                    "historical_exact",
                    0,
                    metric,
                    value,
                ));
            }
        },
        Err(error) =>
        {
            outcome_log.push(format!("stability: {error}"));

            records.push(BenchRecord::new(
                "industrial_728/cmapss_stability",
                "cmapss/stability_engines",
                "historical_exact",
                0,
                "fit_ok",
                0.0,
            ));
        },
    }

    // Trust vacuity: clean vs one fully attacked view, Unweighted vs
    // GroupContaminationBound. On continuous sources every exact-source
    // group is a singleton, so the margin criterion is expected to gate
    // nothing — this section MEASURES that instead of assuming it.
    let trust_dataset = subset(&section.trust_engines);
    let trust_views = build_transport_views(&trust_dataset, &spec).expect("views build");

    let attacked_views: Vec<Vec<SrccTransportSample>> = trust_views
        .views
        .iter()
        .enumerate()
        .map(|(view_index, view)| {
            if view_index == section.trust_attacked_view
            {
                view.iter()
                    .map(|sample| {
                        let mut target = sample.target;
                        target[0] += section.trust_target_shift;

                        SrccTransportSample::new(sample.source, target)
                    })
                    .collect()
            }
            else
            {
                view.clone()
            }
        })
        .collect();

    let clean_slices: Vec<&[SrccTransportSample]> =
        trust_views.views.iter().map(Vec::as_slice).collect();
    let attacked_slices: Vec<&[SrccTransportSample]> =
        attacked_views.iter().map(Vec::as_slice).collect();

    let policies: Vec<(&str, SrccTrustModel)> = vec![
        (
            "unweighted",
            SrccTrustModel {
                policy: SrccTrustPolicy::Unweighted,
                observations: Vec::new(),
            },
        ),
        (
            "group_contamination_bound",
            SrccTrustModel {
                policy: SrccTrustPolicy::GroupContaminationBound {
                    maximum_corrupted_weight_per_group: section.trust_group_bound,
                },
                observations: Vec::new(),
            },
        ),
    ];

    for (policy_name, model) in &policies
    {
        let clean = fit_trusted_robust_srcc_projector_from_views(
            &seeds,
            &clean_slices,
            model,
            srcc_fit_config(),
        );

        let attacked = fit_trusted_robust_srcc_projector_from_views(
            &seeds,
            &attacked_slices,
            model,
            srcc_fit_config(),
        );

        match (clean, attacked)
        {
            (Ok(clean), Ok(attacked)) =>
            {
                let displacement = frobenius_between(&clean.fit.projector, &attacked.fit.projector);

                records.push(BenchRecord::new(
                    "industrial_728/cmapss_trust",
                    "cmapss/one_view_target_shift",
                    policy_name.to_string(),
                    0,
                    "projector_frobenius_displacement",
                    displacement,
                ));

                outcome_log.push(format!(
                    "trust {policy_name}: clean ok, attacked ok, displacement {displacement:.6e}"
                ));
            },
            (clean, attacked) =>
            {
                let describe =
                    |outcome: &Result<_, scirust_srcc::SrccTrustedFitError>| match outcome
                    {
                        Ok(_) => "ok".to_string(),
                        Err(error) => error.to_string(),
                    };

                outcome_log.push(format!(
                    "trust {policy_name}: clean {}, attacked {}",
                    describe(&clean),
                    describe(&attacked),
                ));

                records.push(BenchRecord::new(
                    "industrial_728/cmapss_trust",
                    "cmapss/one_view_target_shift",
                    policy_name.to_string(),
                    0,
                    "fit_ok",
                    f64::from(u8::from(clean.is_ok() && attacked.is_ok())),
                ));
            },
        }
    }

    manifests.insert(
        "cmapss_trust_outcomes.json".to_string(),
        serde_json::to_string_pretty(&outcome_log).expect("outcome log serializes"),
    );
}

/// Section: SECOM anomaly detection with validation-selected grids.
fn secom_section(
    dataset_raw: &TabularDataset,
    config: &SecomConfig,
    records: &mut Vec<BenchRecord>,
    manifests: &mut BTreeMap<String, String>,
) {
    let split = split_dataset(
        dataset_raw,
        &SplitStrategy::Temporal {
            train_fraction: config.train_fraction,
            validation_fraction: config.validation_fraction,
        },
        0,
        None,
    )
    .expect("temporal split is valid");

    manifests.insert(
        "secom_split.json".to_string(),
        serde_json::to_string_pretty(&split.manifest).expect("split manifests serialize"),
    );

    let train_raw = dataset_raw.select_rows(&split.train);
    let validation_raw = dataset_raw.select_rows(&split.validation);
    let test_raw = dataset_raw.select_rows(&split.test);

    let imputer = FittedImputer::fit(
        &train_raw.features,
        MissingValuePolicy {
            maximum_missing_fraction: config.missing_maximum_fraction,
        },
    )
    .expect("secom training columns are not all degenerate");

    manifests.insert(
        "secom_imputer.json".to_string(),
        serde_json::to_string_pretty(&imputer).expect("imputer serializes"),
    );

    let train = imputer.transform(&train_raw).expect("shapes match");
    let validation = imputer.transform(&validation_raw).expect("shapes match");
    let test = imputer.transform(&test_raw).expect("shapes match");

    let mut selection_log: Vec<String> = Vec::new();

    let validation_score = |adapter: &dyn BaselineAdapter| -> Option<f64> {
        match adapter.run(&train, &validation)
        {
            Ok(AdapterOutput::AnomalyScores(scores)) => auroc(&scores, &validation.targets).ok(),
            Ok(AdapterOutput::AnomalyLabels(flags)) =>
            {
                let as_scores: Vec<f64> = flags
                    .iter()
                    .map(|&flag| f64::from(u8::from(flag)))
                    .collect();

                anomaly_label_records(
                    &RecordKey {
                        kernel: String::new(),
                        dataset: String::new(),
                        method: String::new(),
                        seed: 0,
                    },
                    &flags,
                    &validation.targets,
                )
                .ok()
                .and_then(|rows| {
                    let _ = as_scores;
                    rows.into_iter()
                        .find(|row| row.metric == "balanced_accuracy")
                        .map(|row| row.value)
                })
            },
            Ok(_) => None,
            Err(_) => None,
        }
    };

    // Grid selections on VALIDATION only; every candidate value published.
    let mut best_lof = None;

    for &k in &config.lof_k_grid
    {
        let adapter = LofAdapter {
            configuration: LofConfig { k },
        };

        let score = validation_score(&adapter);

        selection_log.push(format!("lof k={k}: validation auroc {score:?}"));

        records.push(BenchRecord::new(
            "industrial_728/secom_validation",
            format!("secom/lof_k_{k}"),
            "local_outlier_factor",
            0,
            "validation_auroc",
            score.unwrap_or(f64::NAN),
        ));

        if let Some(score) = score
            && best_lof.is_none_or(|(_, best)| score > best)
        {
            best_lof = Some((k, score));
        }
    }

    let mut best_ridge = None;

    for &ridge in &config.mahalanobis_ridge_grid
    {
        let adapter = MahalanobisAdapter { ridge };
        let score = validation_score(&adapter);

        selection_log.push(format!(
            "mahalanobis ridge={ridge}: validation auroc {score:?}"
        ));

        records.push(BenchRecord::new(
            "industrial_728/secom_validation",
            format!("secom/mahalanobis_ridge_{ridge}"),
            "regularized_mahalanobis",
            0,
            "validation_auroc",
            score.unwrap_or(f64::NAN),
        ));

        if let Some(score) = score
            && best_ridge.is_none_or(|(_, best)| score > best)
        {
            best_ridge = Some((ridge, score));
        }
    }

    let mut best_eps = None;

    for &eps in &config.dbscan_eps_grid
    {
        let adapter = DbscanAdapter {
            configuration: DbscanConfig {
                eps,
                min_pts: config.dbscan_min_pts,
            },
        };

        let score = validation_score(&adapter);

        selection_log.push(format!(
            "dbscan eps={eps}: validation balanced accuracy {score:?}"
        ));

        records.push(BenchRecord::new(
            "industrial_728/secom_validation",
            format!("secom/dbscan_eps_{eps}"),
            "dbscan_noise",
            0,
            "validation_balanced_accuracy",
            score.unwrap_or(f64::NAN),
        ));

        if let Some(score) = score
            && best_eps.is_none_or(|(_, best)| score > best)
        {
            best_eps = Some((eps, score));
        }
    }

    selection_log.push(format!(
        "selected: lof {best_lof:?}, mahalanobis {best_ridge:?}, dbscan {best_eps:?}"
    ));

    manifests.insert(
        "secom_selection.json".to_string(),
        serde_json::to_string_pretty(&selection_log).expect("selection log serializes"),
    );

    // Frozen test evaluation.
    let mut detectors: Vec<Box<dyn BaselineAdapter>> = vec![
        Box::new(IsolationForestAdapter {
            configuration: IForestConfig {
                n_trees: config.isolation_forest.n_trees,
                subsample_size: config.isolation_forest.subsample_size,
                max_depth: config.isolation_forest.max_depth,
                seed: config.isolation_forest.seed,
            },
        }),
        Box::new(HotellingT2Adapter),
    ];

    if let Some((k, _)) = best_lof
    {
        detectors.push(Box::new(LofAdapter {
            configuration: LofConfig { k },
        }));
    }

    if let Some((ridge, _)) = best_ridge
    {
        detectors.push(Box::new(MahalanobisAdapter { ridge }));
    }

    if let Some((eps, _)) = best_eps
    {
        detectors.push(Box::new(DbscanAdapter {
            configuration: DbscanConfig {
                eps,
                min_pts: config.dbscan_min_pts,
            },
        }));
    }

    for detector in detectors
    {
        let key = RecordKey {
            kernel: "industrial_728/secom_anomaly".into(),
            dataset: "secom/temporal_test".into(),
            method: detector.name().into(),
            seed: config.isolation_forest.seed,
        };

        match detector.run(&train, &test)
        {
            Ok(AdapterOutput::AnomalyScores(scores)) =>
            {
                let mut sorted = scores.clone();
                sorted.sort_by(f64::total_cmp);
                let threshold = sorted[sorted.len() / 2];

                match anomaly_score_records(&key, &scores, &test.targets, threshold)
                {
                    Ok(rows) => records.extend(rows),
                    Err(error) =>
                    {
                        records.push(BenchRecord::new(
                            key.kernel.clone(),
                            key.dataset.clone(),
                            key.method.clone(),
                            key.seed,
                            "metric_ok",
                            0.0,
                        ));

                        selection_log_push(manifests, &format!("{}: {error}", key.method));
                    },
                }
            },
            Ok(AdapterOutput::AnomalyLabels(flags)) =>
            {
                match anomaly_label_records(&key, &flags, &test.targets)
                {
                    Ok(rows) => records.extend(rows),
                    Err(error) =>
                    {
                        selection_log_push(manifests, &format!("{}: {error}", key.method));
                    },
                }
            },
            Ok(_) => unreachable!("anomaly detectors produce scores or labels"),
            Err(error) =>
            {
                records.push(BenchRecord::new(
                    key.kernel.clone(),
                    key.dataset.clone(),
                    key.method.clone(),
                    key.seed,
                    "fit_ok",
                    0.0,
                ));

                selection_log_push(manifests, &format!("{}: {error}", key.method));
            },
        }
    }
}

fn selection_log_push(manifests: &mut BTreeMap<String, String>, line: &str) {
    let entry = manifests
        .entry("secom_typed_failures.json".to_string())
        .or_insert_with(|| "[]".to_string());

    let mut log: Vec<String> = serde_json::from_str(entry).expect("log is valid JSON");
    log.push(line.to_string());
    *entry = serde_json::to_string_pretty(&log).expect("log serializes");
}

/// Section: C-MAPSS single-sensor stream alarms against a manifested burst.
///
/// The stream is one engine's trajectory of the monitored sensor channel in
/// cycle order; the in-control reference is a different engine's trajectory
/// of the same channel. A temporally contiguous burst is injected into the
/// evaluation engine's channel (via `BurstAttack` on a stream whose target
/// carries the sensor values), and the onset is the earliest cycle it
/// touches — explicit in the manifest, not discovered.
fn cmapss_stream_section(
    dataset: &TabularDataset,
    config: &StreamConfig,
    records: &mut Vec<BenchRecord>,
    manifests: &mut BTreeMap<String, String>,
) {
    let groups = dataset.groups.as_ref().expect("cmapss has engines");

    let mut distinct: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !distinct.contains(&group)
        {
            distinct.push(group);
        }
    }

    distinct.sort_unstable();

    let train_engine = distinct[config.train_engine_rank];
    let evaluation_engine = distinct[config.evaluation_engine_rank];

    let engine_rows = |engine: u64| -> Vec<usize> {
        (0..dataset.sample_count())
            .filter(|&row| groups[row] == engine)
            .collect()
    };

    // A single-sensor stream dataset: the target carries the monitored
    // channel's values (so BurstAttack shifts them), time_index = cycles.
    let sensor_stream = |engine: u64| -> TabularDataset {
        let rows = engine_rows(engine);
        let time = dataset.time_index.as_ref().expect("cmapss has cycles");

        let mut ordered = rows;
        ordered.sort_by_key(|&row| (time[row], row));

        TabularDataset {
            features: vec![vec![0.0]; ordered.len()],
            targets: ordered
                .iter()
                .map(|&row| dataset.features[row][config.monitored_column])
                .collect(),
            groups: None,
            time_index: Some((0..ordered.len() as u64).collect()),
        }
    };

    let train_full = sensor_stream(train_engine);
    let evaluation_full = sensor_stream(evaluation_engine);

    let (burst_data, burst_manifest) = apply_contamination(
        &evaluation_full,
        &ContaminationConfig {
            kind: ContaminationKind::BurstAttack {
                target_shift: config.burst_shift,
            },
            fraction: config.burst_fraction,
            seed: config.burst_seed,
        },
    )
    .expect("burst request is valid");

    manifests.insert(
        "cmapss_fd003_stream_burst.json".to_string(),
        serde_json::to_string_pretty(&burst_manifest).expect("manifests serialize"),
    );

    // The stream datasets are already in temporal order (row order == cycle
    // order by construction); the onset is the smallest affected row.
    let onset = *burst_manifest
        .affected_rows
        .iter()
        .min()
        .expect("the burst affected at least one row");

    let as_feature_stream = |values: &[f64]| -> TabularDataset {
        TabularDataset {
            features: values.iter().map(|&value| vec![value]).collect(),
            targets: vec![0.0; values.len()],
            groups: None,
            time_index: None,
        }
    };

    let train_stream = as_feature_stream(&train_full.targets);
    let evaluation_stream = as_feature_stream(&burst_data.targets);

    let adapters: Vec<Box<dyn BaselineAdapter>> = vec![
        Box::new(CusumAdapter {
            column: 0,
            k: config.cusum_k,
            h: config.cusum_h,
        }),
        Box::new(EwmaAdapter {
            column: 0,
            lambda: config.ewma_lambda,
            l: config.ewma_l,
        }),
    ];

    for adapter in adapters
    {
        let key = RecordKey {
            kernel: "industrial_728/cmapss_fd003_stream".into(),
            dataset: format!("cmapss_fd003/engine_{evaluation_engine}_sensor_burst"),
            method: adapter.name().into(),
            seed: config.burst_seed,
        };

        match adapter.run(&train_stream, &evaluation_stream)
        {
            Ok(AdapterOutput::AlarmSteps(alarms)) =>
            {
                records.extend(
                    alarm_records(&key, &alarms, onset, evaluation_stream.sample_count())
                        .expect("the onset is inside the stream"),
                );
            },
            Ok(_) => unreachable!("stream adapters produce alarm steps"),
            Err(error) =>
            {
                records.push(BenchRecord::new(
                    key.kernel.clone(),
                    key.dataset.clone(),
                    key.method.clone(),
                    key.seed,
                    "fit_ok",
                    0.0,
                ));

                manifests.insert(
                    format!("cmapss_fd003_stream_failure_{}.json", key.method),
                    serde_json::to_string_pretty(&error.to_string()).expect("errors serialize"),
                );
            },
        }
    }
}

fn main() {
    let mut data_dir = PathBuf::from("data/industrial");
    let mut out_dir = PathBuf::from("results");
    let mut git_commit = String::from("unknown");

    let mut arguments = std::env::args().skip(1);

    while let Some(argument) = arguments.next()
    {
        let mut value = |name: &str| {
            arguments
                .next()
                .unwrap_or_else(|| panic!("{name} requires a value"))
        };

        match argument.as_str()
        {
            "--data-dir" => data_dir = PathBuf::from(value("--data-dir")),
            "--out" => out_dir = PathBuf::from(value("--out")),
            "--git-commit" => git_commit = value("--git-commit"),
            other => panic!("unknown argument: {other}"),
        }
    }

    let config: Config = serde_json::from_str(CONFIG_TEXT).expect("embedded config is valid");
    let configuration_sha256 = sha256_hex(CONFIG_TEXT.as_bytes());

    let mut records: Vec<BenchRecord> = Vec::new();
    let mut manifests: BTreeMap<String, String> = BTreeMap::new();

    // --- C-MAPSS ---
    let cmapss_text = read_verified(
        &data_dir.join("cmapss/train_FD001.txt"),
        Some(TRAIN_FD001_SHA),
    );

    let cmapss = parse_cmapss_training(&cmapss_text).expect("train_FD001 parses");
    cmapss
        .validate()
        .expect("train_FD001 is finite and rectangular");

    println!(
        "# cmapss rows={} features={} sha256={}",
        cmapss.sample_count(),
        cmapss.feature_count(),
        cmapss.content_sha256(),
    );

    regression_section(
        "cmapss",
        &cmapss,
        &config.cmapss.regression,
        &config.bootstrap,
        &mut records,
        &mut manifests,
    );

    srcc_section(
        "cmapss",
        &cmapss,
        &config.cmapss.srcc,
        &mut records,
        &mut manifests,
    );

    cmapss_stability_and_trust(&cmapss, &config.cmapss.srcc, &mut records, &mut manifests);

    // --- C-MAPSS FD003 (the PdM replication partner) ---
    let fd003_text = read_verified(
        &data_dir.join("cmapss/train_FD003.txt"),
        Some(TRAIN_FD003_SHA),
    );

    let fd003 = parse_cmapss_training(&fd003_text).expect("train_FD003 parses");
    fd003
        .validate()
        .expect("train_FD003 is finite and rectangular");

    println!(
        "# cmapss_fd003 rows={} features={} sha256={}",
        fd003.sample_count(),
        fd003.feature_count(),
        fd003.content_sha256(),
    );

    regression_section(
        "cmapss_fd003",
        &fd003,
        &config.cmapss_fd003.regression,
        &config.bootstrap,
        &mut records,
        &mut manifests,
    );

    srcc_section(
        "cmapss_fd003",
        &fd003,
        &config.cmapss_fd003.srcc,
        &mut records,
        &mut manifests,
    );

    cmapss_stream_section(
        &fd003,
        &config.cmapss_fd003.stream,
        &mut records,
        &mut manifests,
    );

    // --- SECOM ---
    let secom_data = read_verified(&data_dir.join("secom/secom.data"), Some(SECOM_DATA_SHA));
    let secom_labels = read_verified(
        &data_dir.join("secom/secom_labels.data"),
        Some(SECOM_LABELS_SHA),
    );

    let secom = parse_secom(&secom_data, &secom_labels).expect("secom parses");

    println!(
        "# secom rows={} features={} (raw, pre-imputation)",
        secom.sample_count(),
        secom.feature_count(),
    );

    secom_section(&secom, &config.secom, &mut records, &mut manifests);

    // --- outputs ---
    fs::create_dir_all(out_dir.join("manifests")).expect("results directory is writable");

    fs::write(out_dir.join("industrial_728.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    for (name, content) in &manifests
    {
        fs::write(out_dir.join("manifests").join(name), content)
            .expect("manifest files are writable");
    }

    let metadata = RunMetadata {
        git_commit,
        dataset_sha256: cmapss.content_sha256(),
        configuration_sha256,
        toolchain: "nightly-2026-07-02".to_string(),
        feature_flags: vec!["default".to_string()],
    };

    fs::write(
        out_dir.join("manifests/run_metadata.json"),
        serde_json::to_string_pretty(&metadata).expect("metadata serializes"),
    )
    .expect("metadata file is writable");

    println!(
        "# records={} manifests={}",
        records.len(),
        manifests.len() + 1
    );
}
