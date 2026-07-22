//! End-to-end demonstration of the industrial benchmark **protocol** on a
//! synthetic plant.
//!
//! This example exercises every stage the harness provides — manifest,
//! anti-leakage split, contamination with exact manifests, the three adapter
//! families, capability-honest metric emission, and the deterministic paired
//! bootstrap — on deterministic synthetic data. It demonstrates that the
//! *protocol* works; it proves nothing about real industrial superiority,
//! and its numbers must never be quoted as evidence for any method. Real
//! evaluation happens in phase 728 under the committed preregistration
//! (`docs/research/SRCC_INDUSTRIAL_BENCHMARK_PREREGISTRATION.md`).
//!
//! Structure:
//!
//! 1. a 6-machine synthetic plant (180 rows × 4 features, affine target with
//!    seeded noise), manifested and checksummed;
//! 2. **regression under training contamination**: grouped split by machine;
//!    the training targets are corrupted by a coherent alternative cluster
//!    at increasing fractions; four estimators are fitted on the corrupted
//!    train and scored on the untouched test. Unfavourable regimes are part
//!    of the output;
//! 3. **paired per-machine comparison**: leave-one-machine-out at a fixed
//!    contamination level, OLS vs Huber RMSE per held-out machine, seeded
//!    paired bootstrap over the six differences (the interval says what six
//!    machines can say — no more);
//! 4. **anomaly detection**: five detectors score or flag an evaluation set
//!    contaminated by a coherent cluster; score producers get AUROC,
//!    the label-only detector structurally cannot;
//! 5. **stream alarms**: CUSUM and EWMA on a constructed level shift with a
//!    known onset (the onset is explicit in the output, not discovered).
//!
//! Output is deterministic (`{:.17e}` for floats, JSONL for records); run
//! twice and compare byte-for-byte (`cmp` / SHA-256). No timestamps, no
//! timings, no environment identity in the scientific stdout.

use scirust_bench_schema::to_jsonl;
use scirust_srcc_bench::{
    AdapterOutput, BaselineAdapter, ContaminationConfig, ContaminationKind, CusumAdapter,
    DbscanAdapter, EwmaAdapter, FittingProtocol, HotellingT2Adapter, IsolationForestAdapter,
    LofAdapter, MahalanobisAdapter, RecordKey, RobustRegressionAdapter, SplitStrategy,
    TabularDataset, alarm_records, anomaly_label_records, anomaly_score_records,
    apply_contamination, paired_bootstrap, paired_differences, regression_records, split_dataset,
};
use scirust_srcc_bench::{DatasetManifest, FeatureDescriptor};
use scirust_stats::{Distribution, Normal, SplitMix64};
use scirust_unsupervised::{DbscanConfig, IForestConfig, LofConfig};

const MACHINES: u64 = 6;
const ROWS_PER_MACHINE: usize = 30;
const NOISE_SEED: u64 = 0x5EED_0727;
const CONTAMINATION_SEED: u64 = 0xC0AA_0727;
const BOOTSTRAP_SEED: u64 = 0xB007_0727;
const TRUE_COEFFICIENTS: [f64; 4] = [2.0, -1.5, 0.75, 3.0];
const TRUE_INTERCEPT: f64 = 12.0;

/// The synthetic plant: 6 machines × 30 rows, 4 features, affine target.
fn synthetic_plant() -> TabularDataset {
    let standard = Normal::standard();
    let mut rng = SplitMix64::new(NOISE_SEED);

    let total = MACHINES as usize * ROWS_PER_MACHINE;
    let mut features = Vec::with_capacity(total);
    let mut targets = Vec::with_capacity(total);
    let mut groups = Vec::with_capacity(total);
    let mut time_index = Vec::with_capacity(total);

    for machine in 0..MACHINES
    {
        for step in 0..ROWS_PER_MACHINE
        {
            let temperature = 60.0 + machine as f64 * 1.5 + ((step % 8) as f64) * 0.5;
            let speed = 1400.0 + ((step % 5) as f64) * 20.0 - machine as f64 * 10.0;
            let pressure = 5.0 + ((step % 7) as f64) * 0.25;
            let flow = 0.8 + ((step % 4) as f64) * 0.05 + machine as f64 * 0.02;

            let uniform = 1.0e-6 + rng.next_f64() * (1.0 - 2.0e-6);
            let noise = 0.5 * standard.quantile(uniform);

            let row = vec![temperature, speed / 100.0, pressure, flow];

            let target = TRUE_INTERCEPT
                + TRUE_COEFFICIENTS[0] * row[0]
                + TRUE_COEFFICIENTS[1] * row[1]
                + TRUE_COEFFICIENTS[2] * row[2]
                + TRUE_COEFFICIENTS[3] * row[3]
                + noise;

            features.push(row);
            targets.push(target);
            groups.push(machine);
            time_index.push((step as u64) * MACHINES + machine);
        }
    }

    TabularDataset {
        features,
        targets,
        groups: Some(groups),
        time_index: Some(time_index),
    }
}

fn plant_manifest(plant: &TabularDataset) -> DatasetManifest {
    DatasetManifest::for_dataset(
        plant,
        "synthetic_plant",
        "1",
        "generated in-repo by examples/industrial_protocol_demo.rs",
        "MIT OR Apache-2.0 (repository license)",
        "quality index (dimensionless affine law of the four sensors)",
        vec![
            FeatureDescriptor {
                name: "temperature".into(),
                unit: "°C".into(),
                description: "inlet temperature".into(),
            },
            FeatureDescriptor {
                name: "speed".into(),
                unit: "hectorpm".into(),
                description: "shaft speed / 100".into(),
            },
            FeatureDescriptor {
                name: "pressure".into(),
                unit: "bar".into(),
                description: "line pressure".into(),
            },
            FeatureDescriptor {
                name: "flow".into(),
                unit: "m³/h".into(),
                description: "coolant flow".into(),
            },
        ],
    )
    .expect("the synthetic plant manifest is consistent by construction")
}

fn regression_adapters() -> Vec<RobustRegressionAdapter> {
    vec![
        RobustRegressionAdapter::ordinary_least_squares(),
        RobustRegressionAdapter::huber(1.345),
        RobustRegressionAdapter::trimmed(0.7),
        RobustRegressionAdapter::median_of_means(5, 0x00B1_0C55),
    ]
}

fn protocol_name(protocol: FittingProtocol) -> &'static str {
    match protocol
    {
        FittingProtocol::Inductive => "inductive",
        FittingProtocol::Transductive => "transductive",
    }
}

fn main() {
    let plant = synthetic_plant();
    let manifest = plant_manifest(&plant);

    println!("# industrial_protocol_demo — protocol demonstration, not evidence");
    println!(
        "# dataset {} v{} rows={} features={} sha256={}",
        manifest.name,
        manifest.version,
        manifest.sample_count,
        manifest.feature_count,
        manifest.sha256,
    );

    let mut all_records = Vec::new();

    // ------------------------------------------------------------------
    // Section 1: regression under training contamination (grouped split).
    // ------------------------------------------------------------------
    println!("# section 1: regression under coherent training contamination");
    println!("# columns: fraction,affected_train_rows,method,rmse,median_abs_error");

    let split = split_dataset(
        &plant,
        &SplitStrategy::GroupedHoldout {
            train_fraction: 0.5,
            validation_fraction: 0.2,
        },
        0x5717_0001,
        Some("machine_id"),
    )
    .expect("the grouped split is valid by construction");

    let train = plant.select_rows(&split.train);
    let test = plant.select_rows(&split.test);

    for fraction in [0.0, 0.1, 0.2, 0.3]
    {
        let (corrupted_train, contamination) = apply_contamination(
            &train,
            &ContaminationConfig {
                kind: ContaminationKind::CoherentAlternativeCluster {
                    feature_offset: 3.0,
                    target_offset: 40.0,
                },
                fraction,
                seed: CONTAMINATION_SEED,
            },
        )
        .expect("the contamination request is valid");

        for adapter in regression_adapters()
        {
            let output = adapter
                .run(&corrupted_train, &test)
                .expect("regression adapters fit this fixture");

            let AdapterOutput::Predictions(predictions) = output
            else
            {
                unreachable!("regression adapters produce predictions");
            };

            let key = RecordKey {
                kernel: "industrial_demo/regression".into(),
                dataset: format!("synthetic_plant/coherent_cluster_{fraction:.2}"),
                method: adapter.name().into(),
                seed: CONTAMINATION_SEED,
            };

            let records = regression_records(&key, &predictions, &test.targets)
                .expect("test targets are finite");

            let rmse = records[0].value;
            let median = records[2].value;

            println!(
                "{fraction:.17e},{},{},{rmse:.17e},{median:.17e}",
                contamination.affected_rows.len(),
                adapter.name(),
            );

            all_records.extend(records);
        }
    }

    // ------------------------------------------------------------------
    // Section 2: paired per-machine comparison (leave-one-machine-out).
    // ------------------------------------------------------------------
    println!("# section 2: paired ols-vs-huber rmse, leave-one-machine-out at fraction 0.2");
    println!("# columns: machine,ols_rmse,huber_rmse");

    let mut ols_per_machine = Vec::new();
    let mut huber_per_machine = Vec::new();

    for machine in 0..MACHINES
    {
        let split = split_dataset(
            &plant,
            &SplitStrategy::LeaveOneGroupOut {
                held_out_group: machine,
            },
            0,
            Some("machine_id"),
        )
        .expect("every machine occurs in the plant");

        let train = plant.select_rows(&split.train);
        let test = plant.select_rows(&split.test);

        let (corrupted_train, _) = apply_contamination(
            &train,
            &ContaminationConfig {
                kind: ContaminationKind::CoherentAlternativeCluster {
                    feature_offset: 3.0,
                    target_offset: 40.0,
                },
                fraction: 0.2,
                seed: CONTAMINATION_SEED,
            },
        )
        .expect("the contamination request is valid");

        let rmse_of = |adapter: &RobustRegressionAdapter| -> f64 {
            let AdapterOutput::Predictions(predictions) = adapter
                .run(&corrupted_train, &test)
                .expect("regression adapters fit this fixture")
            else
            {
                unreachable!("regression adapters produce predictions");
            };

            regression_records(
                &RecordKey {
                    kernel: "industrial_demo/paired".into(),
                    dataset: format!("synthetic_plant/machine_{machine}"),
                    method: "internal".into(),
                    seed: CONTAMINATION_SEED,
                },
                &predictions,
                &test.targets,
            )
            .expect("test targets are finite")[0]
                .value
        };

        let ols = rmse_of(&RobustRegressionAdapter::ordinary_least_squares());
        let huber = rmse_of(&RobustRegressionAdapter::huber(1.345));

        println!("{machine},{ols:.17e},{huber:.17e}");

        ols_per_machine.push(ols);
        huber_per_machine.push(huber);
    }

    let differences = paired_differences(&ols_per_machine, &huber_per_machine)
        .expect("per-machine vectors are aligned and finite");

    let bootstrap = paired_bootstrap(&differences, 2000, 0.95, BOOTSTRAP_SEED)
        .expect("six machines are enough units for the bootstrap");

    println!(
        "# paired ols_minus_huber_rmse: mean={:.17e} ci=[{:.17e},{:.17e}] level={} \
effect_size={} units={} resamples={} seed={}",
        bootstrap.mean_difference,
        bootstrap.confidence_interval.lo,
        bootstrap.confidence_interval.hi,
        bootstrap.confidence_interval.level,
        bootstrap
            .effect_size
            .map_or_else(|| "undefined".to_string(), |d| format!("{d:.17e}")),
        bootstrap.unit_count,
        bootstrap.resamples,
        bootstrap.seed,
    );

    all_records.push(
        scirust_bench_schema::BenchRecord::new(
            "industrial_demo/paired",
            "synthetic_plant/leave_one_machine_out_0.2",
            "ols_minus_huber",
            BOOTSTRAP_SEED,
            "rmse_paired_mean_difference",
            bootstrap.mean_difference,
        )
        .with_ci(bootstrap.confidence_interval),
    );

    // ------------------------------------------------------------------
    // Section 3: anomaly detection on a contaminated evaluation set.
    // ------------------------------------------------------------------
    println!("# section 3: anomaly detection, coherent cluster in the evaluation set");
    println!("# columns: method,protocol,output,auroc_or_none,balanced_accuracy_or_none");

    // Train on machines 0..4 (clean); evaluate machine 5 with contamination.
    let anomaly_split = split_dataset(
        &plant,
        &SplitStrategy::LeaveOneGroupOut { held_out_group: 5 },
        0,
        Some("machine_id"),
    )
    .expect("machine 5 exists");

    let anomaly_train = plant.select_rows(&anomaly_split.train);
    let clean_evaluation = plant.select_rows(&anomaly_split.test);

    let (contaminated_evaluation, evaluation_manifest) = apply_contamination(
        &clean_evaluation,
        &ContaminationConfig {
            kind: ContaminationKind::CoherentAlternativeCluster {
                feature_offset: 8.0,
                target_offset: 0.0,
            },
            fraction: 0.25,
            seed: CONTAMINATION_SEED,
        },
    )
    .expect("the contamination request is valid");

    let labels: Vec<f64> = (0..contaminated_evaluation.sample_count())
        .map(|row| f64::from(u8::from(evaluation_manifest.affected_rows.contains(&row))))
        .collect();

    let detectors: Vec<Box<dyn BaselineAdapter>> = vec![
        Box::new(IsolationForestAdapter {
            configuration: IForestConfig {
                n_trees: 100,
                subsample_size: 64,
                max_depth: 8,
                seed: 0x1F0_0727,
            },
        }),
        Box::new(LofAdapter {
            configuration: LofConfig { k: 5 },
        }),
        Box::new(MahalanobisAdapter { ridge: 1.0e-6 }),
        Box::new(HotellingT2Adapter),
        Box::new(DbscanAdapter {
            configuration: DbscanConfig {
                eps: 2.5,
                min_pts: 4,
            },
        }),
    ];

    for detector in detectors
    {
        let key = RecordKey {
            kernel: "industrial_demo/anomaly".into(),
            dataset: "synthetic_plant/machine5_coherent_cluster_0.25".into(),
            method: detector.name().into(),
            seed: CONTAMINATION_SEED,
        };

        let output = detector
            .run(&anomaly_train, &contaminated_evaluation)
            .expect("detectors fit this fixture");

        let (records, output_kind, auroc_text) = match &output
        {
            AdapterOutput::AnomalyScores(scores) =>
            {
                // Threshold convention for count metrics: the midpoint
                // between the per-detector score medians is arbitrary, so we
                // use each detector's median score — documented, not tuned.
                let mut sorted = scores.clone();
                sorted.sort_by(f64::total_cmp);
                let threshold = sorted[sorted.len() / 2];

                let records = anomaly_score_records(&key, scores, &labels, threshold)
                    .expect("labels are binary with both classes");

                let auroc = records[0].value;

                (records, "scores", format!("{auroc:.17e}"))
            },
            AdapterOutput::AnomalyLabels(flags) =>
            {
                let records = anomaly_label_records(&key, flags, &labels)
                    .expect("labels are binary with both classes");

                (records, "labels", "none".to_string())
            },
            _ => unreachable!("anomaly detectors produce scores or labels"),
        };

        let balanced = records
            .iter()
            .find(|record| record.metric == "balanced_accuracy")
            .map_or_else(
                || "none".to_string(),
                |record| format!("{:.17e}", record.value),
            );

        println!(
            "{},{},{output_kind},{auroc_text},{balanced}",
            detector.name(),
            protocol_name(detector.protocol()),
        );

        all_records.extend(records);
    }

    // ------------------------------------------------------------------
    // Section 4: stream alarms on a constructed level shift.
    // ------------------------------------------------------------------
    println!("# section 4: stream alarms, constructed +2.0 shift on temperature at onset 20");
    println!("# columns: method,detected,delay_or_none,false_alarms");

    let stream_train = plant.select_rows(
        &(0..ROWS_PER_MACHINE)
            .map(|step| step * MACHINES as usize)
            .collect::<Vec<_>>(),
    );

    let onset = 20usize;

    let stream_evaluation = TabularDataset {
        features: stream_train
            .features
            .iter()
            .cycle()
            .take(40)
            .enumerate()
            .map(|(step, row)| {
                let mut shifted = row.clone();

                if step >= onset
                {
                    shifted[0] += 2.0;
                }

                shifted
            })
            .collect(),
        targets: vec![0.0; 40],
        groups: None,
        time_index: None,
    };

    let stream_adapters: Vec<Box<dyn BaselineAdapter>> = vec![
        Box::new(CusumAdapter {
            column: 0,
            k: 0.5,
            h: 5.0,
        }),
        Box::new(EwmaAdapter {
            column: 0,
            lambda: 0.2,
            l: 2.7,
        }),
    ];

    for adapter in stream_adapters
    {
        let AdapterOutput::AlarmSteps(alarms) = adapter
            .run(&stream_train, &stream_evaluation)
            .expect("stream adapters fit this fixture")
        else
        {
            unreachable!("stream adapters produce alarm steps");
        };

        let key = RecordKey {
            kernel: "industrial_demo/stream".into(),
            dataset: "synthetic_plant/machine0_shift2_onset20".into(),
            method: adapter.name().into(),
            seed: NOISE_SEED,
        };

        let records =
            alarm_records(&key, &alarms, onset, 40).expect("the onset is inside the stream");

        let detected = records
            .iter()
            .find(|record| record.metric == "detected")
            .expect("alarm records always carry `detected`")
            .value;

        let delay = records
            .iter()
            .find(|record| record.metric == "detection_delay_steps")
            .map_or_else(|| "none".to_string(), |record| format!("{}", record.value));

        let false_alarms = records
            .iter()
            .find(|record| record.metric == "false_alarm_count")
            .expect("alarm records always carry `false_alarm_count`")
            .value;

        println!("{},{detected},{delay},{false_alarms}", adapter.name());

        all_records.extend(records);
    }

    // ------------------------------------------------------------------
    // Section 5: every record as JSON Lines (the machine-readable output).
    // ------------------------------------------------------------------
    println!("# section 5: bench records (JSONL)");
    print!("{}", to_jsonl(&all_records));
}
