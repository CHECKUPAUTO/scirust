use std::collections::BTreeMap;

use scirust_tdi::{
    Action, State, TableSystem, TdiSignature, analyze_recovery, explore,
    uniform_future_block_entropy_bits,
};

const WIDTH: u8 = 3;
const STATE_COUNT: usize = 1 << WIDTH;

const TRAIN_SYSTEMS: u64 = 12_000;
const TEST_SYSTEMS: u64 = 4_000;
const TEST_SEED_OFFSET: u64 = 1_000_000;

const ENTROPY_HORIZON: usize = 8;
const TDI_HORIZON: usize = 4;
const RECOVERY_LIMIT: usize = 32;

const BOOTSTRAP_REPLICATES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0x5444_4931_2026_0712;

#[derive(Clone, Debug)]
struct Record {
    entropy_key: u64,
    return_profile: Vec<(u128, u128)>,
    reference_transient_len: usize,
    reference_period: usize,
    perturbed_transient_len: usize,
    perturbed_period: usize,
    recovered: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CombinedKey {
    entropy_key: u64,
    return_profile: Vec<(u128, u128)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OrbitKey {
    reference_transient_len: usize,
    reference_period: usize,
    perturbed_transient_len: usize,
    perturbed_period: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EntropyOrbitKey {
    entropy_key: u64,
    orbit: OrbitKey,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OrbitTdiKey {
    orbit: OrbitKey,
    return_profile: Vec<(u128, u128)>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FullKey {
    entropy_key: u64,
    orbit: OrbitKey,
    return_profile: Vec<(u128, u128)>,
}

#[derive(Clone, Debug)]
struct BucketModel<K> {
    buckets: BTreeMap<K, (usize, usize)>,
    global_probability: f64,
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    accuracy: f64,
    balanced_accuracy: f64,
    brier: f64,
    average_precision: f64,
}

#[derive(Clone, Copy, Debug)]
struct ConfidenceInterval {
    lower: f64,
    median: f64,
    upper: f64,
}

#[derive(Clone, Copy, Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);

        splitmix64(self.state)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);

    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    mixed ^ (mixed >> 31)
}

fn generate_transitions(seed: u64) -> [u8; STATE_COUNT] {
    let mut transitions = [0_u8; STATE_COUNT];
    let mut generator = seed;

    for target in &mut transitions
    {
        generator = splitmix64(generator);
        *target = (generator % STATE_COUNT as u64) as u8;
    }

    transitions
}

fn build_system(transitions: &[u8; STATE_COUNT]) -> Result<TableSystem, String> {
    let mut system =
        TableSystem::new(WIDTH).map_err(|error| format!("cannot create system: {error:?}"))?;

    for (source, &target) in transitions.iter().enumerate()
    {
        let source_state = State::new(source as u64, WIDTH).map_err(|error| error.to_string())?;

        let target_state =
            State::new(u64::from(target), WIDTH).map_err(|error| error.to_string())?;

        system
            .insert(source_state, Action::Noop, vec![target_state])
            .map_err(|error| format!("cannot insert transition {source}->{target}: {error:?}"))?;
    }

    Ok(system)
}

fn analyze_seed(seed: u64) -> Result<Record, String> {
    let transitions = generate_transitions(seed);
    let system = build_system(&transitions)?;

    let initial = State::new(0, WIDTH).map_err(|error| error.to_string())?;

    let entropy = uniform_future_block_entropy_bits(&system, Action::Noop, ENTROPY_HORIZON)
        .map_err(|error| format!("entropy failed for seed {seed}: {error:?}"))?;

    let recovery = analyze_recovery(
        &system,
        initial,
        Action::Flip { node: WIDTH - 1 },
        RECOVERY_LIMIT,
    )
    .map_err(|error| format!("recovery failed for seed {seed}: {error:?}"))?;

    let actions = [Action::Noop; TDI_HORIZON];

    let report = explore(&system, recovery.perturbed_state(), &actions)
        .map_err(|error| format!("exploration failed for seed {seed}: {error:?}"))?;

    let signature = TdiSignature::from_report(&report)
        .map_err(|error| format!("signature failed for seed {seed}: {error:?}"))?;

    let return_profile = signature
        .return_profile()
        .iter()
        .map(|ratio| {
            ratio
                .components_u128()
                .ok_or_else(|| "return-profile ratio exceeds u128".to_owned())
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(Record {
        entropy_key: entropy.to_bits(),
        return_profile,
        reference_transient_len: recovery.reference_orbit().transient_len(),
        reference_period: recovery.reference_orbit().period(),
        perturbed_transient_len: recovery.perturbed_orbit().transient_len(),
        perturbed_period: recovery.perturbed_orbit().period(),
        recovered: recovery.recovered(),
    })
}

fn orbit_key(record: &Record) -> OrbitKey {
    OrbitKey {
        reference_transient_len: record.reference_transient_len,
        reference_period: record.reference_period,
        perturbed_transient_len: record.perturbed_transient_len,
        perturbed_period: record.perturbed_period,
    }
}

fn entropy_orbit_key(record: &Record) -> EntropyOrbitKey {
    EntropyOrbitKey {
        entropy_key: record.entropy_key,
        orbit: orbit_key(record),
    }
}

fn orbit_tdi_key(record: &Record) -> OrbitTdiKey {
    OrbitTdiKey {
        orbit: orbit_key(record),
        return_profile: record.return_profile.clone(),
    }
}

fn full_key(record: &Record) -> FullKey {
    FullKey {
        entropy_key: record.entropy_key,
        orbit: orbit_key(record),
        return_profile: record.return_profile.clone(),
    }
}

fn generate_records(start_seed: u64, count: u64) -> Result<Vec<Record>, String> {
    (start_seed..start_seed + count).map(analyze_seed).collect()
}

fn fit_model<K, F>(records: &[Record], key_fn: F) -> BucketModel<K>
where
    K: Ord + Clone,
    F: Fn(&Record) -> K,
{
    let mut buckets = BTreeMap::<K, (usize, usize)>::new();

    let positives = records.iter().filter(|record| record.recovered).count();

    for record in records
    {
        let bucket = buckets.entry(key_fn(record)).or_default();
        bucket.0 += 1;

        if record.recovered
        {
            bucket.1 += 1;
        }
    }

    BucketModel {
        buckets,
        global_probability: positives as f64 / records.len() as f64,
    }
}

fn predict<K>(model: &BucketModel<K>, key: &K) -> f64
where
    K: Ord,
{
    model
        .buckets
        .get(key)
        .map(|(total, positives)| *positives as f64 / *total as f64)
        .unwrap_or(model.global_probability)
}

fn calculate_metrics(records: &[Record], probabilities: &[f64]) -> Metrics {
    assert_eq!(records.len(), probabilities.len());

    let mut correct = 0_usize;
    let mut positives = 0_usize;
    let mut negatives = 0_usize;
    let mut true_positive = 0_usize;
    let mut true_negative = 0_usize;
    let mut brier_sum = 0.0_f64;

    let mut score_groups = BTreeMap::<u64, (usize, usize)>::new();

    for (record, &probability) in records.iter().zip(probabilities)
    {
        let predicted = probability >= 0.5;

        if predicted == record.recovered
        {
            correct += 1;
        }

        if record.recovered
        {
            positives += 1;

            if predicted
            {
                true_positive += 1;
            }
        }
        else
        {
            negatives += 1;

            if !predicted
            {
                true_negative += 1;
            }
        }

        let target = if record.recovered { 1.0 } else { 0.0 };
        brier_sum += (target - probability).powi(2);

        let group = score_groups.entry(probability.to_bits()).or_default();

        group.0 += 1;

        if record.recovered
        {
            group.1 += 1;
        }
    }

    let accuracy = correct as f64 / records.len() as f64;

    let sensitivity = true_positive as f64 / positives as f64;

    let specificity = true_negative as f64 / negatives as f64;

    let balanced_accuracy = (sensitivity + specificity) / 2.0;

    let brier = brier_sum / records.len() as f64;

    let mut ordered_groups: Vec<(f64, usize, usize)> = score_groups
        .into_iter()
        .map(|(bits, (total, positive))| (f64::from_bits(bits), total, positive))
        .collect();

    ordered_groups.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut cumulative_true_positive = 0_usize;
    let mut cumulative_false_positive = 0_usize;
    let mut previous_recall = 0.0_f64;
    let mut average_precision = 0.0_f64;

    for (_, total, positive) in ordered_groups
    {
        cumulative_true_positive += positive;
        cumulative_false_positive += total - positive;

        let recall = cumulative_true_positive as f64 / positives as f64;

        let precision = cumulative_true_positive as f64
            / (cumulative_true_positive + cumulative_false_positive) as f64;

        average_precision += (recall - previous_recall) * precision;

        previous_recall = recall;
    }

    Metrics {
        accuracy,
        balanced_accuracy,
        brier,
        average_precision,
    }
}

fn average_precision_for_indices(
    records: &[Record],
    probabilities: &[f64],
    indices: &[usize],
) -> f64 {
    let positives = indices
        .iter()
        .filter(|&&index| records[index].recovered)
        .count();

    if positives == 0
    {
        return 0.0;
    }

    let mut score_groups = BTreeMap::<u64, (usize, usize)>::new();

    for &index in indices
    {
        let group = score_groups
            .entry(probabilities[index].to_bits())
            .or_default();

        group.0 += 1;

        if records[index].recovered
        {
            group.1 += 1;
        }
    }

    let mut ordered_groups: Vec<(f64, usize, usize)> = score_groups
        .into_iter()
        .map(|(bits, (total, positive))| (f64::from_bits(bits), total, positive))
        .collect();

    ordered_groups.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut cumulative_true_positive = 0_usize;
    let mut cumulative_false_positive = 0_usize;
    let mut previous_recall = 0.0_f64;
    let mut average_precision = 0.0_f64;

    for (_, total, positive) in ordered_groups
    {
        cumulative_true_positive += positive;
        cumulative_false_positive += total - positive;

        let recall = cumulative_true_positive as f64 / positives as f64;

        let precision = cumulative_true_positive as f64
            / (cumulative_true_positive + cumulative_false_positive) as f64;

        average_precision += (recall - previous_recall) * precision;

        previous_recall = recall;
    }

    average_precision
}

fn brier_for_indices(records: &[Record], probabilities: &[f64], indices: &[usize]) -> f64 {
    indices
        .iter()
        .map(|&index| {
            let target = if records[index].recovered { 1.0 } else { 0.0 };

            (target - probabilities[index]).powi(2)
        })
        .sum::<f64>()
        / indices.len() as f64
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;

    if lower == upper
    {
        sorted[lower]
    }
    else
    {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn confidence_interval(mut values: Vec<f64>) -> ConfidenceInterval {
    values.sort_by(f64::total_cmp);

    ConfidenceInterval {
        lower: percentile(&values, 0.025),
        median: percentile(&values, 0.500),
        upper: percentile(&values, 0.975),
    }
}

fn paired_bootstrap(
    records: &[Record],
    baseline: &[f64],
    challenger: &[f64],
) -> (ConfidenceInterval, ConfidenceInterval) {
    let mut rng = DeterministicRng::new(BOOTSTRAP_SEED);
    let mut indices = vec![0_usize; records.len()];
    let mut auprc_gains = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut brier_improvements = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    for _ in 0..BOOTSTRAP_REPLICATES
    {
        for index in &mut indices
        {
            *index = rng.index(records.len());
        }

        let baseline_auprc = average_precision_for_indices(records, baseline, &indices);

        let challenger_auprc = average_precision_for_indices(records, challenger, &indices);

        let baseline_brier = brier_for_indices(records, baseline, &indices);

        let challenger_brier = brier_for_indices(records, challenger, &indices);

        auprc_gains.push(challenger_auprc - baseline_auprc);
        brier_improvements.push(baseline_brier - challenger_brier);
    }

    (
        confidence_interval(auprc_gains),
        confidence_interval(brier_improvements),
    )
}

fn print_interval(label: &str, interval: ConfidenceInterval) {
    println!(
        "{label}: [{:.6}, {:.6}] (médiane {:.6})",
        interval.lower, interval.upper, interval.median
    );
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!("{label}");
    println!("  accuracy          : {:.6}", metrics.accuracy);
    println!("  balanced accuracy : {:.6}", metrics.balanced_accuracy);
    println!("  Brier score       : {:.6}", metrics.brier);
    println!("  average precision : {:.6}", metrics.average_precision);
}

fn main() -> Result<(), String> {
    println!("Generating training systems...");
    let training = generate_records(0, TRAIN_SYSTEMS)?;

    println!("Generating untouched holdout systems...");
    let test = generate_records(TEST_SEED_OFFSET, TEST_SYSTEMS)?;

    let entropy_model = fit_model(&training, |record| record.entropy_key);

    let tdi_model = fit_model(&training, |record| record.return_profile.clone());

    let combined_model = fit_model(&training, |record| CombinedKey {
        entropy_key: record.entropy_key,
        return_profile: record.return_profile.clone(),
    });

    let orbit_model = fit_model(&training, orbit_key);
    let entropy_orbit_model = fit_model(&training, entropy_orbit_key);
    let orbit_tdi_model = fit_model(&training, orbit_tdi_key);
    let full_model = fit_model(&training, full_key);

    let entropy_probabilities: Vec<f64> = test
        .iter()
        .map(|record| predict(&entropy_model, &record.entropy_key))
        .collect();

    let tdi_probabilities: Vec<f64> = test
        .iter()
        .map(|record| predict(&tdi_model, &record.return_profile))
        .collect();

    let combined_probabilities: Vec<f64> = test
        .iter()
        .map(|record| {
            predict(
                &combined_model,
                &CombinedKey {
                    entropy_key: record.entropy_key,
                    return_profile: record.return_profile.clone(),
                },
            )
        })
        .collect();

    let orbit_probabilities: Vec<f64> = test
        .iter()
        .map(|record| predict(&orbit_model, &orbit_key(record)))
        .collect();

    let entropy_orbit_probabilities: Vec<f64> = test
        .iter()
        .map(|record| predict(&entropy_orbit_model, &entropy_orbit_key(record)))
        .collect();

    let orbit_tdi_probabilities: Vec<f64> = test
        .iter()
        .map(|record| predict(&orbit_tdi_model, &orbit_tdi_key(record)))
        .collect();

    let full_probabilities: Vec<f64> = test
        .iter()
        .map(|record| predict(&full_model, &full_key(record)))
        .collect();

    let entropy = calculate_metrics(&test, &entropy_probabilities);

    let tdi = calculate_metrics(&test, &tdi_probabilities);

    let combined = calculate_metrics(&test, &combined_probabilities);
    let orbit = calculate_metrics(&test, &orbit_probabilities);
    let entropy_orbit = calculate_metrics(&test, &entropy_orbit_probabilities);
    let orbit_tdi = calculate_metrics(&test, &orbit_tdi_probabilities);
    let full = calculate_metrics(&test, &full_probabilities);

    let training_positive = training.iter().filter(|record| record.recovered).count();

    let test_positive = test.iter().filter(|record| record.recovered).count();

    println!();
    println!("TDI-1 untouched holdout evaluation");
    println!("width             : {WIDTH}");
    println!("training systems  : {}", training.len());
    println!("holdout systems   : {}", test.len());
    println!("training recovered: {training_positive}");
    println!("holdout recovered : {test_positive}");
    println!();

    print_metrics("ENTROPY ONLY", entropy);
    println!();

    print_metrics("ORBIT BASELINE", orbit);
    println!();

    print_metrics("ENTROPY + ORBIT", entropy_orbit);
    println!();

    print_metrics("TDI RETURN PROFILE", tdi);
    println!();

    print_metrics("ENTROPY + TDI", combined);
    println!();

    print_metrics("ORBIT + TDI", orbit_tdi);
    println!();

    print_metrics("ENTROPY + ORBIT + TDI", full);
    println!();

    println!(
        "TDI AUPRC gain over entropy      : {:.6}",
        tdi.average_precision - entropy.average_precision
    );

    println!(
        "combined AUPRC gain over entropy : {:.6}",
        combined.average_precision - entropy.average_precision
    );

    println!(
        "TDI Brier improvement            : {:.6}",
        entropy.brier - tdi.brier
    );

    println!(
        "combined Brier improvement       : {:.6}",
        entropy.brier - combined.brier
    );

    println!(
        "TDI AUPRC gain over orbit        : {:.6}",
        orbit_tdi.average_precision - orbit.average_precision
    );

    println!(
        "TDI Brier improvement over orbit : {:.6}",
        orbit.brier - orbit_tdi.brier
    );

    println!(
        "incremental AUPRC over entropy + orbit : {:.6}",
        full.average_precision - entropy_orbit.average_precision
    );

    println!(
        "incremental Brier over entropy + orbit : {:.6}",
        entropy_orbit.brier - full.brier
    );

    println!();
    println!(
        "Bootstrap apparié déterministe : \
         {BOOTSTRAP_REPLICATES} réplications"
    );

    let (tdi_auprc_ci, tdi_brier_ci) =
        paired_bootstrap(&test, &entropy_probabilities, &tdi_probabilities);

    let (combined_auprc_ci, combined_brier_ci) =
        paired_bootstrap(&test, &entropy_probabilities, &combined_probabilities);

    let (orbit_tdi_auprc_ci, orbit_tdi_brier_ci) =
        paired_bootstrap(&test, &orbit_probabilities, &orbit_tdi_probabilities);

    let (full_auprc_ci, full_brier_ci) =
        paired_bootstrap(&test, &entropy_orbit_probabilities, &full_probabilities);

    print_interval("IC 95 % gain AUPRC TDI", tdi_auprc_ci);

    print_interval("IC 95 % amélioration Brier TDI", tdi_brier_ci);

    print_interval("IC 95 % gain AUPRC combiné", combined_auprc_ci);

    print_interval("IC 95 % amélioration Brier combiné", combined_brier_ci);

    print_interval(
        "IC 95 % gain AUPRC TDI sur baseline orbitale",
        orbit_tdi_auprc_ci,
    );

    print_interval(
        "IC 95 % amélioration Brier TDI sur baseline orbitale",
        orbit_tdi_brier_ci,
    );

    print_interval(
        "IC 95 % gain AUPRC incrémental sur entropie + orbite",
        full_auprc_ci,
    );

    print_interval(
        "IC 95 % amélioration Brier incrémentale sur entropie + orbite",
        full_brier_ci,
    );

    // CANR §9 records (`scirust-bench-schema`): the same holdout metrics and
    // paired-bootstrap gains, machine-readable. Seeds are the real generator
    // inputs — holdout rows carry the test-stream seed base, CI rows carry
    // the bootstrap seed that produced the interval.
    let mut bench_records: Vec<scirust_bench_schema::BenchRecord> = Vec::new();
    for (model, metrics) in [
        ("entropy_only", entropy),
        ("orbit_baseline", orbit),
        ("entropy+orbit", entropy_orbit),
        ("tdi_return_profile", tdi),
        ("entropy+tdi", combined),
        ("orbit+tdi", orbit_tdi),
        ("entropy+orbit+tdi", full),
    ]
    {
        for (metric, value) in [
            ("auprc", metrics.average_precision),
            ("brier", metrics.brier),
            ("accuracy", metrics.accuracy),
            ("balanced_accuracy", metrics.balanced_accuracy),
        ]
        {
            bench_records.push(scirust_bench_schema::BenchRecord::new(
                "tdi_holdout/width3",
                format!(
                    "holdout/seeds={TEST_SEED_OFFSET}..{}",
                    TEST_SEED_OFFSET + TEST_SYSTEMS
                ),
                model,
                TEST_SEED_OFFSET,
                metric,
                value,
            ));
        }
    }
    let ci95 = |interval: ConfidenceInterval| scirust_bench_schema::ConfidenceInterval {
        lo: interval.lower,
        hi: interval.upper,
        level: 0.95,
    };
    for (metric, value, interval) in [
        (
            "auprc_gain_tdi_vs_entropy",
            tdi.average_precision - entropy.average_precision,
            tdi_auprc_ci,
        ),
        (
            "brier_improvement_tdi_vs_entropy",
            entropy.brier - tdi.brier,
            tdi_brier_ci,
        ),
        (
            "auprc_gain_combined_vs_entropy",
            combined.average_precision - entropy.average_precision,
            combined_auprc_ci,
        ),
        (
            "brier_improvement_combined_vs_entropy",
            entropy.brier - combined.brier,
            combined_brier_ci,
        ),
        (
            "auprc_gain_orbit_tdi_vs_orbit",
            orbit_tdi.average_precision - orbit.average_precision,
            orbit_tdi_auprc_ci,
        ),
        (
            "brier_improvement_orbit_tdi_vs_orbit",
            orbit.brier - orbit_tdi.brier,
            orbit_tdi_brier_ci,
        ),
        (
            "auprc_gain_full_vs_entropy_orbit",
            full.average_precision - entropy_orbit.average_precision,
            full_auprc_ci,
        ),
        (
            "brier_improvement_full_vs_entropy_orbit",
            entropy_orbit.brier - full.brier,
            full_brier_ci,
        ),
    ]
    {
        bench_records.push(
            scirust_bench_schema::BenchRecord::new(
                "tdi_holdout/width3",
                format!("paired_bootstrap/replicates={BOOTSTRAP_REPLICATES}"),
                "paired_bootstrap",
                BOOTSTRAP_SEED,
                metric,
                value,
            )
            .with_ci(ci95(interval)),
        );
    }
    println!();
    println!(
        "=== bench-schema JSONL ({} records, scirust-bench-schema) ===",
        bench_records.len()
    );
    print!("{}", scirust_bench_schema::to_jsonl(&bench_records));

    let observed_tdi_gain = tdi.average_precision - entropy.average_precision;

    let tdi_success =
        observed_tdi_gain >= 0.05 && tdi_auprc_ci.lower > 0.0 && tdi_brier_ci.lower > 0.0;

    let observed_orbit_gain = orbit_tdi.average_precision - orbit.average_precision;

    let orbit_incremental_success = observed_orbit_gain > 0.0
        && orbit_tdi_auprc_ci.lower > 0.0
        && orbit_tdi_brier_ci.lower > 0.0;

    println!();
    println!(
        "CRITÈRE ORIGINAL TDI-1 VS ENTROPIE : {}",
        if tdi_success { "RÉUSSI" } else { "ÉCHOUÉ" }
    );

    println!(
        "CONTRÔLE DE NOUVEAUTÉ VS ORBITE     : {}",
        if orbit_incremental_success
        {
            "RÉUSSI"
        }
        else
        {
            "ÉCHOUÉ"
        }
    );

    if !tdi_success
    {
        return Err("TDI-1 failed its original entropy-baseline criterion".to_owned());
    }

    Ok(())
}
