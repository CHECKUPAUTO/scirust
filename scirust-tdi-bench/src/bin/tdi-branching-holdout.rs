use std::collections::BTreeMap;

use scirust_tdi::{
    Action, ExactRatio, State, TableSystem, analyze_branching_recovery, explore,
    uniform_branching_path_entropy_bits,
};

const WIDTH: u8 = 3;
const STATE_COUNT: usize = 1 << WIDTH;
const NONEMPTY_SUCCESSOR_SETS: u64 = (1_u64 << STATE_COUNT) - 1;

const TRAIN_SYSTEMS: u64 = 12_000;
const TEST_SYSTEMS: u64 = 4_000;
const TEST_SEED_OFFSET: u64 = 1_000_000;

const OBSERVATION_HORIZON: usize = 2;
const OUTCOME_HORIZON: usize = 6;

const BOOTSTRAP_REPLICATES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0x5444_4932_2026_0712;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EntropyKey {
    reference: [u64; OBSERVATION_HORIZON],
    perturbed: [u64; OBSERVATION_HORIZON],
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TopologyKey {
    reference_reachable: [usize; OBSERVATION_HORIZON],
    reference_paths: [u128; OBSERVATION_HORIZON],
    perturbed_reachable: [usize; OBSERVATION_HORIZON],
    perturbed_paths: [u128; OBSERVATION_HORIZON],
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OverlapKey {
    profile: Vec<(u128, u128)>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EntropyTopologyKey {
    entropy: EntropyKey,
    topology: TopologyKey,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FullKey {
    entropy: EntropyKey,
    topology: TopologyKey,
    overlap: OverlapKey,
}

#[derive(Clone, Debug)]
struct Record {
    entropy: EntropyKey,
    topology: TopologyKey,
    overlap: OverlapKey,
    recovered: bool,
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

fn generate_successor_masks(seed: u64) -> [u16; STATE_COUNT] {
    let mut masks = [0_u16; STATE_COUNT];
    let mut generator = seed;

    for mask in &mut masks
    {
        generator = splitmix64(generator);

        *mask = (generator % NONEMPTY_SUCCESSOR_SETS + 1) as u16;
    }

    masks
}

fn build_system(masks: &[u16; STATE_COUNT]) -> Result<TableSystem, String> {
    let mut system = TableSystem::new(WIDTH)
        .map_err(|error| format!("cannot create branching system: {error:?}"))?;

    for (source_bits, &mask) in masks.iter().enumerate()
    {
        let source = State::new(source_bits as u64, WIDTH).map_err(|error| error.to_string())?;

        let successors = (0..STATE_COUNT)
            .filter(|target| mask & (1_u16 << target) != 0)
            .map(|target| State::new(target as u64, WIDTH).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        system
            .insert(source, Action::Noop, successors)
            .map_err(|error| {
                format!(
                    "cannot insert branching transition \
                     for state {source_bits}: {error:?}"
                )
            })?;
    }

    Ok(system)
}

fn entropy_profile(
    system: &TableSystem,
    initial: State,
) -> Result<[u64; OBSERVATION_HORIZON], String> {
    let mut profile = [0_u64; OBSERVATION_HORIZON];

    for depth in 1..=OBSERVATION_HORIZON
    {
        let entropy = uniform_branching_path_entropy_bits(system, initial, Action::Noop, depth)
            .map_err(|error| {
                format!(
                    "branching entropy failed at depth \
                         {depth}: {error:?}"
                )
            })?;

        profile[depth - 1] = entropy.to_bits();
    }

    Ok(profile)
}

fn topology_profile(
    system: &TableSystem,
    initial: State,
) -> Result<([usize; OBSERVATION_HORIZON], [u128; OBSERVATION_HORIZON]), String> {
    let actions = [Action::Noop; OBSERVATION_HORIZON];

    let report = explore(system, initial, &actions)
        .map_err(|error| format!("branching exploration failed: {error:?}"))?;

    let mut reachable = [0_usize; OBSERVATION_HORIZON];
    let mut paths = [0_u128; OBSERVATION_HORIZON];

    for depth in 1..=OBSERVATION_HORIZON
    {
        reachable[depth - 1] = report
            .reachable_count(depth)
            .ok_or_else(|| format!("missing reachable layer {depth}"))?;

        paths[depth - 1] = report
            .path_count(depth)
            .ok_or_else(|| format!("missing path-count layer {depth}"))?;
    }

    Ok((reachable, paths))
}

fn ratio_pair(ratio: &ExactRatio) -> Result<(u128, u128), String> {
    ratio
        .components_u128()
        .ok_or_else(|| "overlap ratio exceeds u128".to_owned())
}

fn analyze_seed(seed: u64) -> Result<Option<Record>, String> {
    let masks = generate_successor_masks(seed);
    let system = build_system(&masks)?;

    let reference = State::new(0, WIDTH).map_err(|error| error.to_string())?;

    let perturbed = Action::Flip { node: WIDTH - 1 }
        .apply(reference)
        .map_err(|error| error.to_string())?;

    let reference_entropy = entropy_profile(&system, reference)?;
    let perturbed_entropy = entropy_profile(&system, perturbed)?;

    let (reference_reachable, reference_paths) = topology_profile(&system, reference)?;

    let (perturbed_reachable, perturbed_paths) = topology_profile(&system, perturbed)?;

    let observation = analyze_branching_recovery(
        &system,
        reference,
        Action::Flip { node: WIDTH - 1 },
        Action::Noop,
        OBSERVATION_HORIZON,
    )
    .map_err(|error| {
        format!(
            "observation recovery analysis failed \
             for seed {seed}: {error:?}"
        )
    })?;

    let outcome = analyze_branching_recovery(
        &system,
        reference,
        Action::Flip { node: WIDTH - 1 },
        Action::Noop,
        OUTCOME_HORIZON,
    )
    .map_err(|error| {
        format!(
            "outcome recovery analysis failed \
             for seed {seed}: {error:?}"
        )
    })?;

    // Une égalité distributionnelle déjà atteinte à l’horizon
    // d’observation implique automatiquement l’égalité à tout horizon
    // ultérieur sous le même noyau de transition. Ces cas constitueraient
    // donc une fuite logique de la cible et sont exclus.
    if observation.fully_recovered()
    {
        return Ok(None);
    }

    Ok(Some(Record {
        entropy: EntropyKey {
            reference: reference_entropy,
            perturbed: perturbed_entropy,
        },
        topology: TopologyKey {
            reference_reachable,
            reference_paths,
            perturbed_reachable,
            perturbed_paths,
        },
        overlap: OverlapKey {
            profile: observation
                .overlap_profile()
                .iter()
                .map(ratio_pair)
                .collect::<Result<Vec<_>, String>>()?,
        },
        recovered: outcome.fully_recovered(),
    }))
}

fn generate_records(start_seed: u64, count: u64) -> Result<Vec<Record>, String> {
    let mut records = Vec::with_capacity(count as usize);
    let mut seed = start_seed;

    while records.len() < count as usize
    {
        if let Some(record) = analyze_seed(seed)?
        {
            records.push(record);
        }

        seed = seed
            .checked_add(1)
            .ok_or_else(|| "seed range overflow".to_owned())?;
    }

    Ok(records)
}

fn entropy_topology_key(record: &Record) -> EntropyTopologyKey {
    EntropyTopologyKey {
        entropy: record.entropy.clone(),
        topology: record.topology.clone(),
    }
}

fn full_key(record: &Record) -> FullKey {
    FullKey {
        entropy: record.entropy.clone(),
        topology: record.topology.clone(),
        overlap: record.overlap.clone(),
    }
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
        .map(|(total, positives)| (*positives as f64 + 1.0) / (*total as f64 + 2.0))
        .unwrap_or(model.global_probability)
}

fn average_precision(records: &[Record], probabilities: &[f64]) -> f64 {
    let indices = (0..records.len()).collect::<Vec<_>>();
    average_precision_for_indices(records, probabilities, &indices)
}

fn calculate_metrics(records: &[Record], probabilities: &[f64]) -> Metrics {
    assert_eq!(records.len(), probabilities.len());

    let mut correct = 0_usize;
    let mut positives = 0_usize;
    let mut negatives = 0_usize;
    let mut true_positive = 0_usize;
    let mut true_negative = 0_usize;
    let mut brier_sum = 0.0_f64;

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
    }

    let sensitivity = if positives == 0
    {
        0.0
    }
    else
    {
        true_positive as f64 / positives as f64
    };

    let specificity = if negatives == 0
    {
        0.0
    }
    else
    {
        true_negative as f64 / negatives as f64
    };

    Metrics {
        accuracy: correct as f64 / records.len() as f64,
        balanced_accuracy: (sensitivity + specificity) / 2.0,
        brier: brier_sum / records.len() as f64,
        average_precision: average_precision(records, probabilities),
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

    let mut ordered_groups = score_groups
        .into_iter()
        .map(|(bits, (total, positive))| (f64::from_bits(bits), total, positive))
        .collect::<Vec<_>>();

    ordered_groups.sort_by(|left, right| right.0.total_cmp(&left.0));

    let mut cumulative_true_positive = 0_usize;
    let mut cumulative_false_positive = 0_usize;
    let mut previous_recall = 0.0_f64;
    let mut result = 0.0_f64;

    for (_, total, positive) in ordered_groups
    {
        cumulative_true_positive += positive;
        cumulative_false_positive += total - positive;

        let recall = cumulative_true_positive as f64 / positives as f64;

        let precision = cumulative_true_positive as f64
            / (cumulative_true_positive + cumulative_false_positive) as f64;

        result += (recall - previous_recall) * precision;
        previous_recall = recall;
    }

    result
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

fn print_metrics(label: &str, metrics: Metrics) {
    println!("{label}");
    println!("  accuracy          : {:.6}", metrics.accuracy);
    println!("  balanced accuracy : {:.6}", metrics.balanced_accuracy);
    println!("  Brier score       : {:.6}", metrics.brier);
    println!("  average precision : {:.6}", metrics.average_precision);
}

fn print_interval(label: &str, interval: ConfidenceInterval) {
    println!(
        "{label}: [{:.6}, {:.6}] (médiane {:.6})",
        interval.lower, interval.upper, interval.median
    );
}

fn main() -> Result<(), String> {
    println!("Generating TDI-2 branching training systems...");
    let training = generate_records(0, TRAIN_SYSTEMS)?;

    println!("Generating untouched branching holdout systems...");
    let test = generate_records(TEST_SEED_OFFSET, TEST_SYSTEMS)?;

    let entropy_model = fit_model(&training, |record| record.entropy.clone());

    let topology_model = fit_model(&training, |record| record.topology.clone());

    let matched_model = fit_model(&training, entropy_topology_key);

    let overlap_model = fit_model(&training, |record| record.overlap.clone());

    let full_model = fit_model(&training, full_key);

    let entropy_probabilities = test
        .iter()
        .map(|record| predict(&entropy_model, &record.entropy))
        .collect::<Vec<_>>();

    let topology_probabilities = test
        .iter()
        .map(|record| predict(&topology_model, &record.topology))
        .collect::<Vec<_>>();

    let matched_probabilities = test
        .iter()
        .map(|record| predict(&matched_model, &entropy_topology_key(record)))
        .collect::<Vec<_>>();

    let overlap_probabilities = test
        .iter()
        .map(|record| predict(&overlap_model, &record.overlap))
        .collect::<Vec<_>>();

    let full_probabilities = test
        .iter()
        .map(|record| predict(&full_model, &full_key(record)))
        .collect::<Vec<_>>();

    let entropy = calculate_metrics(&test, &entropy_probabilities);

    let topology = calculate_metrics(&test, &topology_probabilities);

    let matched = calculate_metrics(&test, &matched_probabilities);

    let overlap = calculate_metrics(&test, &overlap_probabilities);

    let full = calculate_metrics(&test, &full_probabilities);

    let training_positive = training.iter().filter(|record| record.recovered).count();

    let test_positive = test.iter().filter(|record| record.recovered).count();

    println!();
    println!("TDI-2 untouched branching holdout evaluation");
    println!("width               : {WIDTH}");
    println!("training systems    : {}", training.len());
    println!("holdout systems     : {}", test.len());
    println!("observation horizon : {OBSERVATION_HORIZON}");
    println!("outcome horizon     : {OUTCOME_HORIZON}");
    println!("target              : delayed full recovery only");
    println!("excluded            : recovered by observation horizon");
    println!("training recovered  : {training_positive}");
    println!("holdout recovered   : {test_positive}");

    println!();
    print_metrics("ENTROPY PROFILE", entropy);

    println!();
    print_metrics("MATCHED-HORIZON TOPOLOGY", topology);

    println!();
    print_metrics("ENTROPY + TOPOLOGY", matched);

    println!();
    print_metrics("EARLY OVERLAP PROFILE", overlap);

    println!();
    print_metrics("ENTROPY + TOPOLOGY + OVERLAP", full);

    let overlap_gain = overlap.average_precision - matched.average_precision;

    let overlap_brier = matched.brier - overlap.brier;

    let full_gain = full.average_precision - matched.average_precision;

    let full_brier = matched.brier - full.brier;

    println!();
    println!(
        "overlap AUPRC gain over matched baseline : \
         {overlap_gain:.6}"
    );

    println!(
        "overlap Brier improvement                : \
         {overlap_brier:.6}"
    );

    println!(
        "full incremental AUPRC gain              : \
         {full_gain:.6}"
    );

    println!(
        "full incremental Brier improvement       : \
         {full_brier:.6}"
    );

    println!();
    println!(
        "Bootstrap apparié déterministe : \
         {BOOTSTRAP_REPLICATES} réplications"
    );

    let (overlap_auprc_ci, overlap_brier_ci) =
        paired_bootstrap(&test, &matched_probabilities, &overlap_probabilities);

    let (full_auprc_ci, full_brier_ci) =
        paired_bootstrap(&test, &matched_probabilities, &full_probabilities);

    print_interval(
        "IC 95 % gain AUPRC overlap sur baseline appariée",
        overlap_auprc_ci,
    );

    print_interval("IC 95 % amélioration Brier overlap", overlap_brier_ci);

    print_interval("IC 95 % gain AUPRC incrémental complet", full_auprc_ci);

    print_interval(
        "IC 95 % amélioration Brier incrémentale complète",
        full_brier_ci,
    );

    let success = full_gain >= 0.01 && full_auprc_ci.lower > 0.0 && full_brier_ci.lower > 0.0;

    println!();
    println!(
        "CRITÈRE TDI-2 VS BASELINE APPARIÉE : {}",
        if success { "RÉUSSI" } else { "ÉCHOUÉ" }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{NONEMPTY_SUCCESSOR_SETS, STATE_COUNT, generate_successor_masks};

    #[test]
    fn branching_generation_is_deterministic_and_nonempty() {
        let first = generate_successor_masks(42);
        let second = generate_successor_masks(42);

        assert_eq!(first, second);
        assert_eq!(first.len(), STATE_COUNT);

        for mask in first
        {
            assert!(mask != 0);
            assert!(u64::from(mask) <= NONEMPTY_SUCCESSOR_SETS);
        }
    }
}
