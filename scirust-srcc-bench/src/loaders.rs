//! Text-format loaders for the preregistered industrial datasets.
//!
//! Pure parsing: `&str` in, typed structures out — no file I/O, no network
//! (the evaluation binary reads files and verifies checksums before calling
//! these). Every malformed token is a typed error naming its line and
//! column; nothing is skipped silently.
//!
//! # C-MAPSS (NASA turbofan degradation, run-to-failure)
//!
//! Space-separated rows of 26 columns: unit id, cycle, 3 operational
//! settings, 21 sensor channels. [`parse_cmapss_training`] returns a
//! [`TabularDataset`] with the 24 settings+sensor columns as features,
//! `groups` = unit id, `time_index` = cycle, and the standard run-to-failure
//! target `RUL(row) = max_cycle(unit) − cycle(row)` (each training unit runs
//! to failure, so its last cycle has RUL 0).
//!
//! # SECOM (real semiconductor process data)
//!
//! `secom.data`: space-separated rows of 590 sensor readings with literal
//! `NaN` for missing values. `secom_labels.data`: one `−1` (pass) / `1`
//! (fail) label per row plus a quoted timestamp. [`parse_secom`] returns the
//! raw feature matrix (missing values kept as `f64::NAN` — the dataset is
//! **not** yet valid by [`TabularDataset::validate`] and must go through the
//! train-fitted missing-value policy first), anomaly labels remapped to
//! `{0.0, 1.0}` (fail = 1), and `time_index` = row order (the file is
//! chronologically ordered; the timestamp text is not otherwise consumed).
//!
//! # OBD2 telemetry (real automotive condition monitoring, in-repo)
//!
//! `examples/obd2_diagnostic/data/opel_corsa_telemetry.csv`: a header line
//! then comma-separated rows of 12 numeric sensor channels plus a trailing
//! integer `segment_id`. [`parse_obd2`] takes the target column by name,
//! uses the remaining 11 channels as features, `groups` = segment id, and
//! `time_index` = row order within the file (the capture is
//! chronologically ordered).

use core::fmt;

use crate::dataset::TabularDataset;

/// Typed parse errors.
#[derive(Clone, Debug, PartialEq)]
pub enum LoaderError {
    /// The input has no data rows.
    EmptyInput,
    /// A row has the wrong number of columns.
    ColumnCountMismatch {
        /// 1-based line number.
        line: usize,
        /// Expected column count.
        expected: usize,
        /// Found column count.
        found: usize,
    },
    /// A token failed to parse as a number.
    MalformedNumber {
        /// 1-based line number.
        line: usize,
        /// 1-based token position.
        column: usize,
    },
    /// A C-MAPSS unit id or cycle is not a positive integer.
    MalformedIdentifier {
        /// 1-based line number.
        line: usize,
    },
    /// A SECOM label is neither `-1` nor `1`.
    MalformedLabel {
        /// 1-based line number.
        line: usize,
    },
    /// The SECOM data and label files disagree on the row count.
    LabelCountMismatch {
        /// Data rows.
        data_rows: usize,
        /// Label rows.
        label_rows: usize,
    },
    /// A required header column is absent (or coincides with the group
    /// column).
    MissingHeaderColumn {
        /// The requested column name.
        name: String,
    },
}

impl fmt::Display for LoaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyInput => formatter.write_str("input has no data rows"),
            Self::ColumnCountMismatch {
                line,
                expected,
                found,
            } => write!(
                formatter,
                "line {line}: expected {expected} columns, found {found}"
            ),
            Self::MalformedNumber { line, column } =>
            {
                write!(formatter, "line {line}, token {column}: malformed number")
            },
            Self::MalformedIdentifier { line } =>
            {
                write!(formatter, "line {line}: malformed unit id or cycle")
            },
            Self::MalformedLabel { line } =>
            {
                write!(formatter, "line {line}: label must be -1 or 1")
            },
            Self::LabelCountMismatch {
                data_rows,
                label_rows,
            } => write!(
                formatter,
                "secom data has {data_rows} rows but labels file has {label_rows}"
            ),
            Self::MissingHeaderColumn { name } =>
            {
                write!(formatter, "header column `{name}` is absent or ambiguous")
            },
        }
    }
}

impl std::error::Error for LoaderError {}

/// Number of whitespace-separated columns in a C-MAPSS row.
pub const CMAPSS_COLUMNS: usize = 26;

/// Number of feature columns extracted from a C-MAPSS row (3 settings + 21
/// sensors).
pub const CMAPSS_FEATURES: usize = 24;

/// Number of sensor columns in a SECOM row.
pub const SECOM_COLUMNS: usize = 590;

/// Parses a C-MAPSS training file (run-to-failure) into a tabular dataset
/// with per-row remaining-useful-life targets.
pub fn parse_cmapss_training(text: &str) -> Result<TabularDataset, LoaderError> {
    let mut units: Vec<u64> = Vec::new();
    let mut cycles: Vec<u64> = Vec::new();
    let mut features: Vec<Vec<f64>> = Vec::new();

    for (index, line) in text.lines().enumerate()
    {
        let line_number = index + 1;
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens.is_empty()
        {
            continue;
        }

        if tokens.len() != CMAPSS_COLUMNS
        {
            return Err(LoaderError::ColumnCountMismatch {
                line: line_number,
                expected: CMAPSS_COLUMNS,
                found: tokens.len(),
            });
        }

        let unit: u64 = tokens[0]
            .parse()
            .map_err(|_| LoaderError::MalformedIdentifier { line: line_number })?;

        let cycle: u64 = tokens[1]
            .parse()
            .map_err(|_| LoaderError::MalformedIdentifier { line: line_number })?;

        if unit == 0 || cycle == 0
        {
            return Err(LoaderError::MalformedIdentifier { line: line_number });
        }

        let mut row = Vec::with_capacity(CMAPSS_FEATURES);

        for (position, token) in tokens[2..].iter().enumerate()
        {
            let value: f64 = token.parse().map_err(|_| LoaderError::MalformedNumber {
                line: line_number,
                column: position + 3,
            })?;

            row.push(value);
        }

        units.push(unit);
        cycles.push(cycle);
        features.push(row);
    }

    if features.is_empty()
    {
        return Err(LoaderError::EmptyInput);
    }

    // Run-to-failure target: RUL = last observed cycle of the unit − cycle.
    let mut last_cycle: Vec<(u64, u64)> = Vec::new();

    for (&unit, &cycle) in units.iter().zip(&cycles)
    {
        match last_cycle
            .iter_mut()
            .find(|(candidate, _)| *candidate == unit)
        {
            Some((_, maximum)) => *maximum = (*maximum).max(cycle),
            None => last_cycle.push((unit, cycle)),
        }
    }

    let targets: Vec<f64> = units
        .iter()
        .zip(&cycles)
        .map(|(&unit, &cycle)| {
            let (_, maximum) = last_cycle
                .iter()
                .find(|(candidate, _)| *candidate == unit)
                .expect("every unit was recorded");

            (maximum - cycle) as f64
        })
        .collect();

    Ok(TabularDataset {
        features,
        targets,
        groups: Some(units),
        time_index: Some(cycles),
    })
}

/// Parses the SECOM data and label files. Missing sensor readings stay as
/// `f64::NAN`; the caller must apply the train-fitted missing-value policy
/// before validation or hashing-dependent stages that require finiteness.
pub fn parse_secom(data_text: &str, labels_text: &str) -> Result<TabularDataset, LoaderError> {
    let mut features: Vec<Vec<f64>> = Vec::new();

    for (index, line) in data_text.lines().enumerate()
    {
        let line_number = index + 1;
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens.is_empty()
        {
            continue;
        }

        if tokens.len() != SECOM_COLUMNS
        {
            return Err(LoaderError::ColumnCountMismatch {
                line: line_number,
                expected: SECOM_COLUMNS,
                found: tokens.len(),
            });
        }

        let mut row = Vec::with_capacity(SECOM_COLUMNS);

        for (position, token) in tokens.iter().enumerate()
        {
            let value: f64 = if *token == "NaN"
            {
                f64::NAN
            }
            else
            {
                token.parse().map_err(|_| LoaderError::MalformedNumber {
                    line: line_number,
                    column: position + 1,
                })?
            };

            row.push(value);
        }

        features.push(row);
    }

    if features.is_empty()
    {
        return Err(LoaderError::EmptyInput);
    }

    let mut targets: Vec<f64> = Vec::new();

    for (index, line) in labels_text.lines().enumerate()
    {
        let line_number = index + 1;
        let mut tokens = line.split_whitespace();

        let Some(label) = tokens.next()
        else
        {
            continue;
        };

        match label
        {
            "-1" => targets.push(0.0),
            "1" => targets.push(1.0),
            _ => return Err(LoaderError::MalformedLabel { line: line_number }),
        }
    }

    if targets.len() != features.len()
    {
        return Err(LoaderError::LabelCountMismatch {
            data_rows: features.len(),
            label_rows: targets.len(),
        });
    }

    let rows = features.len();

    Ok(TabularDataset {
        features,
        targets,
        groups: None,
        time_index: Some((0..rows as u64).collect()),
    })
}

/// Expected trailing group column of the OBD2 telemetry CSV.
pub const OBD2_GROUP_COLUMN: &str = "segment_id";

/// Parses the in-repo OBD2 telemetry CSV. `target_name` selects the target
/// column by header name; the remaining numeric channels (in header order)
/// become features; the trailing `segment_id` becomes `groups`;
/// `time_index` is the row order of the chronologically ordered capture.
pub fn parse_obd2(text: &str, target_name: &str) -> Result<TabularDataset, LoaderError> {
    let mut lines = text.lines().enumerate();

    let Some((_, header)) = lines.next()
    else
    {
        return Err(LoaderError::EmptyInput);
    };

    let names: Vec<&str> = header.split(',').map(str::trim).collect();
    let column_count = names.len();

    let Some(group_position) = names.iter().position(|name| *name == OBD2_GROUP_COLUMN)
    else
    {
        return Err(LoaderError::MissingHeaderColumn {
            name: OBD2_GROUP_COLUMN.to_string(),
        });
    };

    let Some(target_position) = names.iter().position(|name| *name == target_name)
    else
    {
        return Err(LoaderError::MissingHeaderColumn {
            name: target_name.to_string(),
        });
    };

    if target_position == group_position
    {
        return Err(LoaderError::MissingHeaderColumn {
            name: target_name.to_string(),
        });
    }

    let feature_positions: Vec<usize> = (0..column_count)
        .filter(|&position| position != group_position && position != target_position)
        .collect();

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();
    let mut groups: Vec<u64> = Vec::new();

    for (index, line) in lines
    {
        let line_number = index + 1;

        if line.trim().is_empty()
        {
            continue;
        }

        let tokens: Vec<&str> = line.split(',').map(str::trim).collect();

        if tokens.len() != column_count
        {
            return Err(LoaderError::ColumnCountMismatch {
                line: line_number,
                expected: column_count,
                found: tokens.len(),
            });
        }

        let parse_value = |position: usize| -> Result<f64, LoaderError> {
            tokens[position]
                .parse()
                .map_err(|_| LoaderError::MalformedNumber {
                    line: line_number,
                    column: position + 1,
                })
        };

        let mut row = Vec::with_capacity(feature_positions.len());

        for &position in &feature_positions
        {
            row.push(parse_value(position)?);
        }

        let group: u64 = tokens[group_position]
            .parse()
            .map_err(|_| LoaderError::MalformedIdentifier { line: line_number })?;

        features.push(row);
        targets.push(parse_value(target_position)?);
        groups.push(group);
    }

    if features.is_empty()
    {
        return Err(LoaderError::EmptyInput);
    }

    let rows = features.len();

    Ok(TabularDataset {
        features,
        targets,
        groups: Some(groups),
        time_index: Some((0..rows as u64).collect()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmapss_line(unit: u64, cycle: u64, seed: f64) -> String {
        let mut tokens = vec![unit.to_string(), cycle.to_string()];

        for position in 0..CMAPSS_FEATURES
        {
            tokens.push(format!("{:.4}", seed + position as f64));
        }

        tokens.join(" ")
    }

    #[test]
    fn cmapss_training_computes_run_to_failure_targets() {
        let text = [
            cmapss_line(1, 1, 0.5),
            cmapss_line(1, 2, 0.6),
            cmapss_line(1, 3, 0.7),
            cmapss_line(2, 1, 1.5),
            cmapss_line(2, 2, 1.6),
        ]
        .join("\n");

        let dataset = parse_cmapss_training(&text).unwrap();

        assert_eq!(dataset.sample_count(), 5);
        assert_eq!(dataset.feature_count(), CMAPSS_FEATURES);
        assert_eq!(dataset.groups, Some(vec![1, 1, 1, 2, 2]));
        assert_eq!(dataset.time_index, Some(vec![1, 2, 3, 1, 2]));
        // Unit 1 fails at cycle 3, unit 2 at cycle 2.
        assert_eq!(dataset.targets, vec![2.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(dataset.validate(), Ok(()));
    }

    #[test]
    fn cmapss_rejects_malformed_rows() {
        assert_eq!(parse_cmapss_training(""), Err(LoaderError::EmptyInput));

        assert_eq!(
            parse_cmapss_training("1 1 0.5"),
            Err(LoaderError::ColumnCountMismatch {
                line: 1,
                expected: CMAPSS_COLUMNS,
                found: 3,
            }),
        );

        let bad_number = cmapss_line(1, 1, 0.5).replace("23.5000", "not_a_number");

        assert!(matches!(
            parse_cmapss_training(&bad_number),
            Err(LoaderError::MalformedNumber { line: 1, .. })
        ));

        let zero_unit = cmapss_line(0, 1, 0.5);

        assert_eq!(
            parse_cmapss_training(&zero_unit),
            Err(LoaderError::MalformedIdentifier { line: 1 }),
        );
    }

    fn secom_row(seed: f64, missing_at: Option<usize>) -> String {
        (0..SECOM_COLUMNS)
            .map(|position| {
                if Some(position) == missing_at
                {
                    "NaN".to_string()
                }
                else
                {
                    format!("{:.2}", seed + position as f64)
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn secom_parses_missing_values_and_labels() {
        let data = [secom_row(0.5, Some(3)), secom_row(1.5, None)].join("\n");
        let labels = "-1 \"19/07/2008 11:55:00\"\n1 \"19/07/2008 12:32:00\"\n";

        let dataset = parse_secom(&data, labels).unwrap();

        assert_eq!(dataset.sample_count(), 2);
        assert_eq!(dataset.feature_count(), SECOM_COLUMNS);
        assert!(dataset.features[0][3].is_nan());
        assert!(!dataset.features[1][3].is_nan());
        assert_eq!(dataset.targets, vec![0.0, 1.0]);
        assert_eq!(dataset.time_index, Some(vec![0, 1]));
    }

    #[test]
    fn obd2_extracts_target_groups_and_time() {
        let text = "\
RPM,SPEED,LONG_FUEL_TRIM_1,segment_id
1898,39,17.97,6
1900,40,18.02,6
1500,20,-1.50,2
";

        let dataset = parse_obd2(text, "LONG_FUEL_TRIM_1").unwrap();

        assert_eq!(dataset.sample_count(), 3);
        assert_eq!(dataset.feature_count(), 2);
        assert_eq!(dataset.features[0], vec![1898.0, 39.0]);
        assert_eq!(dataset.targets, vec![17.97, 18.02, -1.5]);
        assert_eq!(dataset.groups, Some(vec![6, 6, 2]));
        assert_eq!(dataset.time_index, Some(vec![0, 1, 2]));
        assert_eq!(dataset.validate(), Ok(()));
    }

    #[test]
    fn obd2_rejects_missing_columns_and_malformed_rows() {
        let text = "RPM,SPEED,segment_id\n1,2,3\n";

        assert_eq!(
            parse_obd2(text, "LONG_FUEL_TRIM_1"),
            Err(LoaderError::MissingHeaderColumn {
                name: "LONG_FUEL_TRIM_1".into(),
            }),
        );

        let no_group = "RPM,SPEED\n1,2\n";

        assert_eq!(
            parse_obd2(no_group, "RPM"),
            Err(LoaderError::MissingHeaderColumn {
                name: OBD2_GROUP_COLUMN.into(),
            }),
        );

        let ragged = "RPM,SPEED,segment_id\n1,2\n";

        assert_eq!(
            parse_obd2(ragged, "RPM"),
            Err(LoaderError::ColumnCountMismatch {
                line: 2,
                expected: 3,
                found: 2,
            }),
        );

        let bad_group = "RPM,SPEED,segment_id\n1,2,x\n";

        assert_eq!(
            parse_obd2(bad_group, "RPM"),
            Err(LoaderError::MalformedIdentifier { line: 2 }),
        );
    }

    #[test]
    fn secom_rejects_malformed_input() {
        let data = secom_row(0.5, None);

        assert_eq!(
            parse_secom(&data, "0 \"x\"\n"),
            Err(LoaderError::MalformedLabel { line: 1 }),
        );

        assert_eq!(
            parse_secom(&data, "-1 \"a\"\n-1 \"b\"\n"),
            Err(LoaderError::LabelCountMismatch {
                data_rows: 1,
                label_rows: 2,
            }),
        );

        assert_eq!(
            parse_secom("1.0 2.0", "-1 \"a\"\n"),
            Err(LoaderError::ColumnCountMismatch {
                line: 1,
                expected: SECOM_COLUMNS,
                found: 2,
            }),
        );
    }
}
