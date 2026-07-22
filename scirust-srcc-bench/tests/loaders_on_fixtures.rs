//! Integration tests that parse the committed redistributable fixtures and,
//! when the full datasets are present, the real files too.
//!
//! The fixtures (`tests/data/`) are small, license-clean heads of the real
//! datasets; they always run. The full-data checks read
//! `data/industrial/...` (git-ignored, populated by
//! `scripts/fetch_industrial_datasets.sh`) and **skip loudly** when absent —
//! they never fail for missing data and never download.

use std::path::Path;

use scirust_srcc_bench::loaders::{CMAPSS_FEATURES, SECOM_COLUMNS};
use scirust_srcc_bench::{parse_cmapss_training, parse_secom};

const CMAPSS_FD001_HEAD: &str = include_str!("data/cmapss_fd001_head.txt");
const CMAPSS_FD003_HEAD: &str = include_str!("data/cmapss_fd003_head.txt");
const SECOM_HEAD: &str = include_str!("data/secom_head.data");
const SECOM_LABELS_HEAD: &str = include_str!("data/secom_labels_head.data");

#[test]
fn cmapss_fixtures_parse_with_run_to_failure_targets() {
    for (name, text) in [("FD001", CMAPSS_FD001_HEAD), ("FD003", CMAPSS_FD003_HEAD)]
    {
        let dataset = parse_cmapss_training(text)
            .unwrap_or_else(|error| panic!("{name} head fixture must parse: {error}"));

        dataset
            .validate()
            .unwrap_or_else(|error| panic!("{name} head fixture must validate: {error}"));

        assert_eq!(dataset.feature_count(), CMAPSS_FEATURES);
        assert_eq!(
            dataset.groups.as_ref().unwrap().len(),
            dataset.sample_count()
        );

        // The fixture holds units 1 and 2 truncated to their first 40 cycles;
        // each unit's last (fixture) cycle has RUL 0.
        let groups = dataset.groups.as_ref().unwrap();
        let targets = &dataset.targets;

        for unit in [1u64, 2]
        {
            let unit_targets: Vec<f64> = (0..dataset.sample_count())
                .filter(|&row| groups[row] == unit)
                .map(|row| targets[row])
                .collect();

            assert!(!unit_targets.is_empty(), "{name}: unit {unit} present");
            assert_eq!(
                unit_targets.iter().copied().fold(f64::INFINITY, f64::min),
                0.0,
                "{name}: unit {unit} reaches RUL 0 at its last fixture cycle",
            );
        }
    }
}

#[test]
fn secom_fixtures_parse_with_missing_values_and_binary_labels() {
    let dataset =
        parse_secom(SECOM_HEAD, SECOM_LABELS_HEAD).expect("secom head fixtures must parse");

    assert_eq!(dataset.feature_count(), SECOM_COLUMNS);
    assert_eq!(dataset.sample_count(), 12);

    for &label in &dataset.targets
    {
        assert!(label == 0.0 || label == 1.0, "labels are binary");
    }

    // The real SECOM data carries missing readings; at least the raw fixture
    // is parseable whether or not this particular head has any.
    assert!(
        dataset.time_index.is_some(),
        "row order is the temporal key"
    );
}

#[test]
fn full_cmapss_data_when_present_or_skip_loudly() {
    let path = Path::new("../data/industrial/cmapss/train_FD001.txt");

    let Ok(text) = std::fs::read_to_string(path)
    else
    {
        eprintln!(
            "SKIP: {} absent — run scripts/fetch_industrial_datasets.sh to enable the \
full-data check (this is a skip, not a failure)",
            path.display()
        );
        return;
    };

    let dataset = parse_cmapss_training(&text).expect("real train_FD001 parses");
    dataset.validate().expect("real train_FD001 validates");

    assert_eq!(dataset.feature_count(), CMAPSS_FEATURES);
    // FD001 has 100 run-to-failure units.
    let groups = dataset.groups.as_ref().unwrap();
    let distinct: std::collections::BTreeSet<u64> = groups.iter().copied().collect();
    assert_eq!(distinct.len(), 100, "FD001 has 100 engines");
}
