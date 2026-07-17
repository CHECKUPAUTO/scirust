use std::collections::BTreeMap;

use scirust_tdi::{
    Action, ExactRatio, State, TableSystem, analyze_branching_recovery, explore,
    uniform_branching_path_entropy_bits,
};

const WIDTH: u8 = 2;
const STATE_COUNT: usize = 1 << WIDTH;
const NONEMPTY_SUCCESSOR_SETS: u64 = (1_u64 << STATE_COUNT) - 1;
const SYSTEM_COUNT: u64 = 50_625;

const OBSERVATION_HORIZON: usize = 2;
const OUTCOME_HORIZON: usize = 6;

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
    final_overlap: (u128, u128),
    recovered: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct BucketSummary {
    buckets: usize,
    mixed_buckets: usize,
    records_in_mixed_buckets: usize,
}

fn decode_successor_masks(mut system_id: u64) -> [u8; STATE_COUNT] {
    let mut masks = [0_u8; STATE_COUNT];

    for mask in &mut masks
    {
        *mask = (system_id % NONEMPTY_SUCCESSOR_SETS + 1) as u8;

        system_id /= NONEMPTY_SUCCESSOR_SETS;
    }

    masks
}

fn build_system(masks: &[u8; STATE_COUNT]) -> Result<TableSystem, String> {
    let mut system = TableSystem::new(WIDTH)
        .map_err(|error| format!("cannot create branching system: {error:?}"))?;

    for (source_bits, &mask) in masks.iter().enumerate()
    {
        let source = State::new(source_bits as u64, WIDTH).map_err(|error| error.to_string())?;

        let successors = (0..STATE_COUNT)
            .filter(|target| mask & (1_u8 << target) != 0)
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

fn analyze_system(system_id: u64) -> Result<Record, String> {
    let masks = decode_successor_masks(system_id);
    let system = build_system(&masks)?;

    let reference = State::new(0, WIDTH).map_err(|error| error.to_string())?;

    let perturbed = Action::Flip { node: 0 }
        .apply(reference)
        .map_err(|error| error.to_string())?;

    let reference_entropy = entropy_profile(&system, reference)?;

    let perturbed_entropy = entropy_profile(&system, perturbed)?;

    let (reference_reachable, reference_paths) = topology_profile(&system, reference)?;

    let (perturbed_reachable, perturbed_paths) = topology_profile(&system, perturbed)?;

    let observation = analyze_branching_recovery(
        &system,
        reference,
        Action::Flip { node: 0 },
        Action::Noop,
        OBSERVATION_HORIZON,
    )
    .map_err(|error| {
        format!(
            "observation recovery analysis failed \
             for system {system_id}: {error:?}"
        )
    })?;

    let outcome = analyze_branching_recovery(
        &system,
        reference,
        Action::Flip { node: 0 },
        Action::Noop,
        OUTCOME_HORIZON,
    )
    .map_err(|error| {
        format!(
            "outcome recovery analysis failed \
             for system {system_id}: {error:?}"
        )
    })?;

    let overlap = OverlapKey {
        profile: observation
            .overlap_profile()
            .iter()
            .map(ratio_pair)
            .collect::<Result<Vec<_>, String>>()?,
    };

    let final_overlap = outcome
        .final_overlap()
        .ok_or_else(|| "outcome horizon unexpectedly produced no overlap".to_owned())?;

    Ok(Record {
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
        overlap,
        final_overlap: ratio_pair(&final_overlap)?,
        recovered: outcome.fully_recovered(),
    })
}

fn summarize_buckets<K, F>(records: &[Record], key_fn: F) -> BucketSummary
where
    K: Ord,
    F: Fn(&Record) -> K,
{
    let mut buckets = BTreeMap::<K, (usize, usize)>::new();

    for record in records
    {
        let bucket = buckets.entry(key_fn(record)).or_default();

        if record.recovered
        {
            bucket.0 += 1;
        }
        else
        {
            bucket.1 += 1;
        }
    }

    let mut summary = BucketSummary {
        buckets: buckets.len(),
        ..BucketSummary::default()
    };

    for (recovered, failed) in buckets.into_values()
    {
        if recovered != 0 && failed != 0
        {
            summary.mixed_buckets += 1;
            summary.records_in_mixed_buckets += recovered + failed;
        }
    }

    summary
}

fn print_summary(label: &str, summary: BucketSummary) {
    println!("{label}");
    println!("  buckets                  : {}", summary.buckets);
    println!("  mixed-outcome buckets    : {}", summary.mixed_buckets);
    println!(
        "  records in mixed buckets : {}",
        summary.records_in_mixed_buckets
    );
}

fn main() -> Result<(), String> {
    let mut records = Vec::with_capacity(SYSTEM_COUNT as usize);
    let mut final_overlap_counts = BTreeMap::<(u128, u128), usize>::new();

    println!("Scanning {SYSTEM_COUNT} exhaustive branching systems...");

    for system_id in 0..SYSTEM_COUNT
    {
        let record = analyze_system(system_id)?;

        *final_overlap_counts
            .entry(record.final_overlap)
            .or_default() += 1;

        records.push(record);
    }

    let recovered = records.iter().filter(|record| record.recovered).count();

    let entropy_summary = summarize_buckets(&records, |record| record.entropy.clone());

    let topology_summary = summarize_buckets(&records, |record| record.topology.clone());

    let entropy_topology_summary = summarize_buckets(&records, |record| {
        (record.entropy.clone(), record.topology.clone())
    });

    let overlap_summary = summarize_buckets(&records, |record| record.overlap.clone());

    let full_summary = summarize_buckets(&records, |record| FullKey {
        entropy: record.entropy.clone(),
        topology: record.topology.clone(),
        overlap: record.overlap.clone(),
    });

    println!();
    println!("TDI-2 exhaustive branching scan");
    println!("width                    : {WIDTH}");
    println!("systems                  : {}", records.len());
    println!("observation horizon      : {OBSERVATION_HORIZON}");
    println!("outcome horizon          : {OUTCOME_HORIZON}");
    println!("fully recovered          : {recovered}");
    println!("not fully recovered      : {}", records.len() - recovered);
    println!("distinct final overlaps  : {}", final_overlap_counts.len());

    let zero_overlap = final_overlap_counts.get(&(0, 1)).copied().unwrap_or(0);

    let full_overlap = final_overlap_counts.get(&(1, 1)).copied().unwrap_or(0);

    let partial_overlap = records.len() - zero_overlap - full_overlap;

    let mut most_frequent = final_overlap_counts
        .iter()
        .map(|(&(numerator, denominator), &count)| (numerator, denominator, count))
        .collect::<Vec<_>>();

    most_frequent.sort_by(|left, right| {
        right
            .2
            .cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
    });

    println!();
    println!("FINAL OVERLAP SUMMARY");
    println!("  zero overlap    : {zero_overlap}");
    println!("  partial overlap : {partial_overlap}");
    println!("  full overlap    : {full_overlap}");

    println!();
    println!("20 MOST FREQUENT FINAL OVERLAPS");

    for (numerator, denominator, count) in most_frequent.into_iter().take(20)
    {
        println!("  {numerator}/{denominator:<7} : {count}");
    }

    println!();
    print_summary("ENTROPY PROFILE", entropy_summary);

    println!();
    print_summary("MATCHED-HORIZON TOPOLOGY", topology_summary);

    println!();
    print_summary("ENTROPY + TOPOLOGY", entropy_topology_summary);

    println!();
    print_summary("EARLY OVERLAP PROFILE", overlap_summary);

    println!();
    print_summary("ENTROPY + TOPOLOGY + OVERLAP", full_summary);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        EntropyKey, OverlapKey, Record, TopologyKey, decode_successor_masks, summarize_buckets,
    };

    fn record(entropy: u64, overlap: u128, recovered: bool) -> Record {
        Record {
            entropy: EntropyKey {
                reference: [entropy; 2],
                perturbed: [entropy; 2],
            },
            topology: TopologyKey {
                reference_reachable: [1; 2],
                reference_paths: [1; 2],
                perturbed_reachable: [1; 2],
                perturbed_paths: [1; 2],
            },
            overlap: OverlapKey {
                profile: vec![(overlap, 1)],
            },
            final_overlap: if recovered { (1, 1) } else { (0, 1) },
            recovered,
        }
    }

    #[test]
    fn decodes_nonempty_successor_masks() {
        assert_eq!(decode_successor_masks(0), [1, 1, 1, 1]);

        assert_eq!(decode_successor_masks(14), [15, 1, 1, 1]);
    }

    #[test]
    fn identifies_mixed_prediction_buckets() {
        let records = vec![record(1, 0, true), record(1, 1, false), record(2, 0, true)];

        let entropy = summarize_buckets(&records, |record| record.entropy.clone());

        assert_eq!(entropy.buckets, 2);
        assert_eq!(entropy.mixed_buckets, 1);
        assert_eq!(entropy.records_in_mixed_buckets, 2);

        let overlap = summarize_buckets(&records, |record| record.overlap.clone());

        assert_eq!(overlap.mixed_buckets, 0);
        assert_eq!(overlap.records_in_mixed_buckets, 0);
    }
}
