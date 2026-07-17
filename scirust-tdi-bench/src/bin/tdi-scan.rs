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
    system_id: u16,
    transitions: [u8; STATE_COUNT],
    entropy: f64,
    recovered: bool,
    return_profile: Vec<(u128, u128)>,
}

#[derive(Clone, Debug)]
struct Witness {
    recovered: Record,
    failed: Record,
}

#[derive(Clone, Debug)]
struct ScanSummary {
    systems: usize,
    entropy_buckets: usize,
    mixed_recovery_pairs: usize,
    tdi_separated_pairs: usize,
    witness: Witness,
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
    let mut system = TableSystem::new(WIDTH)
        .map_err(|error| format!("cannot create table system: {error:?}"))?;

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
        .map_err(|error| format!("entropy failed for {system_id}: {error:?}"))?;

    let recovery = analyze_recovery(&system, initial, Action::Flip { node: 1 }, RECOVERY_LIMIT)
        .map_err(|error| format!("recovery analysis failed for {system_id}: {error:?}"))?;

    let actions = [Action::Noop; TDI_HORIZON];

    let report = explore(&system, recovery.perturbed_state(), &actions)
        .map_err(|error| format!("prospective exploration failed for {system_id}: {error:?}"))?;

    let signature = TdiSignature::from_report(&report)
        .map_err(|error| format!("signature extraction failed for {system_id}: {error:?}"))?;

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
        system_id,
        transitions,
        entropy,
        recovered: recovery.recovered(),
        return_profile,
    })
}

fn exhaustive_scan() -> Result<ScanSummary, String> {
    let mut groups = BTreeMap::<u64, Vec<Record>>::new();

    for system_id in 0..SYSTEM_COUNT
    {
        let record = analyze_system(system_id)?;
        groups
            .entry(record.entropy.to_bits())
            .or_default()
            .push(record);
    }

    let mut mixed_recovery_pairs = 0_usize;
    let mut tdi_separated_pairs = 0_usize;
    let mut witness = None;

    for records in groups.values()
    {
        for left_index in 0..records.len()
        {
            for right_index in (left_index + 1)..records.len()
            {
                let left = &records[left_index];
                let right = &records[right_index];

                if left.recovered == right.recovered
                {
                    continue;
                }

                mixed_recovery_pairs += 1;

                if left.return_profile == right.return_profile
                {
                    continue;
                }

                tdi_separated_pairs += 1;

                if witness.is_none()
                {
                    let (recovered, failed) = if left.recovered
                    {
                        (left.clone(), right.clone())
                    }
                    else
                    {
                        (right.clone(), left.clone())
                    };

                    witness = Some(Witness { recovered, failed });
                }
            }
        }
    }

    let witness = witness.ok_or_else(|| {
        "no entropy-equivalent pair with different recovery and TDI profile found".to_owned()
    })?;

    Ok(ScanSummary {
        systems: usize::from(SYSTEM_COUNT),
        entropy_buckets: groups.len(),
        mixed_recovery_pairs,
        tdi_separated_pairs,
        witness,
    })
}

fn print_record(label: &str, record: &Record) {
    println!("{label}");
    println!("  system id      : {}", record.system_id);
    println!("  transitions    : {:?}", record.transitions);
    println!("  entropy        : {:.12} bits", record.entropy);
    println!("  recovered      : {}", record.recovered);
    println!("  return profile : {:?}", record.return_profile);
}

fn main() -> Result<(), String> {
    let summary = exhaustive_scan()?;

    println!("TDI-1 exhaustive deterministic scan");
    println!("systems analyzed          : {}", summary.systems);
    println!("entropy buckets            : {}", summary.entropy_buckets);
    println!(
        "same-entropy opposite outcomes : {}",
        summary.mixed_recovery_pairs
    );
    println!(
        "pairs separated by TDI profile : {}",
        summary.tdi_separated_pairs
    );
    println!();

    print_record("RECOVERED WITNESS", &summary.witness.recovered);
    println!();
    print_record("FAILED WITNESS", &summary.witness.failed);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SYSTEM_COUNT, exhaustive_scan};

    #[test]
    fn exhaustive_scan_finds_a_predictive_tdi_separation() {
        let summary = exhaustive_scan().expect("exhaustive scan succeeds");

        assert_eq!(summary.systems, usize::from(SYSTEM_COUNT));
        assert!(summary.mixed_recovery_pairs > 0);
        assert!(summary.tdi_separated_pairs > 0);
        assert!(summary.witness.recovered.recovered);
        assert!(!summary.witness.failed.recovered);
        assert_eq!(
            summary.witness.recovered.entropy,
            summary.witness.failed.entropy
        );
        assert_ne!(
            summary.witness.recovered.return_profile,
            summary.witness.failed.return_profile
        );
    }
}
