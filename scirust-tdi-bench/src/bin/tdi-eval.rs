use std::collections::BTreeMap;

use scirust_tdi::{
    Action, State, TableSystem, TdiSignature, analyze_recovery, explore,
    uniform_future_block_entropy_bits,
};

const WIDTH: u8 = 2;
const STATE_COUNT: usize = 1 << WIDTH;
const SYSTEM_COUNT: u16 = 256;
const ENTROPY_HORIZON: usize = 8;
const TDI_HORIZON: usize = 2;
const RECOVERY_LIMIT: usize = 16;

#[derive(Clone, Debug)]
struct Record {
    entropy_key: u64,
    return_profile: Vec<(u128, u128)>,
    recovered: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CombinedKey {
    entropy_key: u64,
    return_profile: Vec<(u128, u128)>,
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    accuracy: f64,
    balanced_accuracy: f64,
    brier: f64,
    average_precision: f64,
}

fn decode_transitions(mut system_id: u16) -> [u8; STATE_COUNT] {
    let mut transitions = [0_u8; STATE_COUNT];

    for target in &mut transitions
    {
        *target = (system_id % STATE_COUNT as u16) as u8;
        system_id /= STATE_COUNT as u16;
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

fn analyze_system(system_id: u16) -> Result<Record, String> {
    let transitions = decode_transitions(system_id);
    let system = build_system(&transitions)?;
    let initial = State::new(0, WIDTH).map_err(|error| error.to_string())?;

    let entropy = uniform_future_block_entropy_bits(&system, Action::Noop, ENTROPY_HORIZON)
        .map_err(|error| format!("entropy failed for system {system_id}: {error:?}"))?;

    let recovery = analyze_recovery(&system, initial, Action::Flip { node: 1 }, RECOVERY_LIMIT)
        .map_err(|error| format!("recovery analysis failed for system {system_id}: {error:?}"))?;

    let actions = [Action::Noop; TDI_HORIZON];

    let report = explore(&system, recovery.perturbed_state(), &actions).map_err(|error| {
        format!(
            "prospective exploration failed for system {system_id}: \
             {error:?}"
        )
    })?;

    let signature = TdiSignature::from_report(&report).map_err(|error| {
        format!(
            "signature extraction failed for system {system_id}: \
                 {error:?}"
        )
    })?;

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
        recovered: recovery.recovered(),
    })
}

fn generate_records() -> Result<Vec<Record>, String> {
    (0..SYSTEM_COUNT).map(analyze_system).collect()
}

fn leave_one_out_probabilities<K, F>(records: &[Record], key_fn: F) -> Vec<f64>
where
    K: Ord + Clone,
    F: Fn(&Record) -> K,
{
    let mut buckets = BTreeMap::<K, (usize, usize)>::new();

    let total_positives = records.iter().filter(|record| record.recovered).count();

    for record in records
    {
        let key = key_fn(record);
        let bucket = buckets.entry(key).or_default();
        bucket.0 += 1;

        if record.recovered
        {
            bucket.1 += 1;
        }
    }

    records
        .iter()
        .map(|record| {
            let key = key_fn(record);
            let (bucket_total, bucket_positive) = buckets
                .get(&key)
                .copied()
                .expect("every record must belong to a bucket");

            let remaining_total = bucket_total - 1;
            let remaining_positive = bucket_positive - usize::from(record.recovered);

            if remaining_total != 0
            {
                remaining_positive as f64 / remaining_total as f64
            }
            else
            {
                let global_total = records.len() - 1;
                let global_positive = total_positives - usize::from(record.recovered);

                global_positive as f64 / global_total as f64
            }
        })
        .collect()
}

fn calculate_metrics(records: &[Record], probabilities: &[f64]) -> Metrics {
    assert_eq!(records.len(), probabilities.len());

    let mut correct = 0_usize;
    let mut true_positive = 0_usize;
    let mut true_negative = 0_usize;
    let mut positives = 0_usize;
    let mut negatives = 0_usize;
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
        .map(|(score_bits, (total, positive))| (f64::from_bits(score_bits), total, positive))
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

fn evaluate(records: &[Record]) -> (Metrics, Metrics, Metrics) {
    let entropy_probabilities = leave_one_out_probabilities(records, |record| record.entropy_key);

    let tdi_probabilities =
        leave_one_out_probabilities(records, |record| record.return_profile.clone());

    let combined_probabilities = leave_one_out_probabilities(records, |record| CombinedKey {
        entropy_key: record.entropy_key,
        return_profile: record.return_profile.clone(),
    });

    (
        calculate_metrics(records, &entropy_probabilities),
        calculate_metrics(records, &tdi_probabilities),
        calculate_metrics(records, &combined_probabilities),
    )
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!("{label}");
    println!("  accuracy          : {:.6}", metrics.accuracy);
    println!("  balanced accuracy : {:.6}", metrics.balanced_accuracy);
    println!("  Brier score       : {:.6}", metrics.brier);
    println!("  average precision : {:.6}", metrics.average_precision);
}

fn main() -> Result<(), String> {
    let records = generate_records()?;
    let positives = records.iter().filter(|record| record.recovered).count();

    let (entropy, tdi, combined) = evaluate(&records);

    println!("TDI-1 leave-one-out predictive evaluation");
    println!("systems  : {}", records.len());
    println!("recovered: {positives}");
    println!("failed   : {}", records.len() - positives);
    println!();

    print_metrics("ENTROPY ONLY", entropy);
    println!();

    print_metrics("TDI RETURN PROFILE", tdi);
    println!();

    print_metrics("ENTROPY + TDI", combined);
    println!();

    println!(
        "AUPRC gain over entropy : {:.6}",
        combined.average_precision - entropy.average_precision
    );

    println!(
        "Brier improvement       : {:.6}",
        entropy.brier - combined.brier
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{evaluate, generate_records};

    #[test]
    fn tdi_improves_leave_one_out_prediction() {
        let records = generate_records().expect("record generation succeeds");

        let (entropy, tdi, combined) = evaluate(&records);

        assert!(tdi.accuracy > entropy.accuracy);
        assert!(tdi.balanced_accuracy > entropy.balanced_accuracy);
        assert!(tdi.brier < entropy.brier);
        assert!(tdi.average_precision > entropy.average_precision);

        assert!(combined.accuracy > entropy.accuracy);
        assert!(combined.balanced_accuracy > entropy.balanced_accuracy);
        assert!(combined.brier < entropy.brier);
        assert!(combined.average_precision > entropy.average_precision);
    }
}
