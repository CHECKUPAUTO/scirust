//! Scenario parsing and validation errors, mapped onto the shared
//! `SRST-VAL-*` error catalogue from `scirust-studio-command`.

use std::fmt;

use scirust_studio_command::{CatalogedError, ErrorCode, ErrorFamily};

/// Everything that can go wrong parsing or validating a `.scirust.toml`
/// scenario. Every variant is user-fixable (a malformed or out-of-range
/// scenario), never an internal bug — see [`SchemaError::to_cataloged`].
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaError {
    /// The TOML text itself did not parse. The message is `toml`'s own
    /// diagnostic, which already includes a line and column.
    Parse(String),
    /// `schema_version` is not one this crate understands.
    UnsupportedSchemaVersion {
        /// The version the scenario declared.
        found: u32,
        /// The version this crate actually supports.
        supported: u32,
    },
    /// A `unit` string is not in the supported symbol table.
    UnknownUnit {
        /// The scenario field the unit was attached to, e.g. `"model.mass"`.
        field: String,
        /// The unrecognised unit symbol.
        unit: String,
    },
    /// A numeric field is `NaN` or infinite.
    NonFinite {
        /// The scenario field, e.g. `"solver.step"`.
        field: String,
        /// The non-finite value.
        value: f64,
    },
    /// The solver's end time is not strictly after its start time.
    EndNotAfterStart {
        /// `solver.start`, in SI units.
        start: f64,
        /// `solver.end`, in SI units.
        end: f64,
    },
    /// `solver.step` was given but is zero or negative.
    NonPositiveStep {
        /// The offending step value, in SI units.
        step: f64,
    },
    /// `backend.precision` is not one this Studio build supports.
    UnsupportedPrecision {
        /// The precision string the scenario declared.
        precision: String,
    },
    /// `backend.kind` is not one this Studio build supports.
    UnsupportedBackendKind {
        /// The backend kind string the scenario declared.
        kind: String,
    },
    /// `capability.id` is not in the caller-supplied list of known
    /// capabilities (only checked when the caller supplies one).
    UnknownCapability {
        /// The unrecognised capability id.
        id: String,
    },
    /// A free-text field exceeded the maximum accepted length.
    StringTooLong {
        /// The scenario field, e.g. `"experiment.name"`.
        field: String,
        /// The field's actual length.
        length: usize,
        /// The maximum accepted length.
        max: usize,
    },
    /// `outputs.record` listed more series than the accepted maximum.
    TooManyOutputs {
        /// The number of series listed.
        count: usize,
        /// The maximum accepted count.
        max: usize,
    },
    /// `outputs.sample_every` was explicitly set to zero (would mean "record
    /// nothing" or divide-by-zero downstream, depending on the reader).
    ZeroSampleEvery,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            SchemaError::Parse(msg) => write!(f, "{msg}"),
            SchemaError::UnsupportedSchemaVersion { found, supported } =>
            {
                write!(
                    f,
                    "schema_version {found} is not supported (this build reads version {supported})"
                )
            },
            SchemaError::UnknownUnit { field, unit } =>
            {
                write!(f, "{field}: unit `{unit}` is not a recognised unit symbol")
            },
            SchemaError::NonFinite { field, value } =>
            {
                write!(f, "{field} = {value} must be finite")
            },
            SchemaError::EndNotAfterStart { start, end } =>
            {
                write!(f, "solver.end ({end}) must be after solver.start ({start})")
            },
            SchemaError::NonPositiveStep { step } =>
            {
                write!(f, "solver.step ({step}) must be positive")
            },
            SchemaError::UnsupportedPrecision { precision } =>
            {
                write!(
                    f,
                    "backend.precision `{precision}` is not supported (use `f32` or `f64`)"
                )
            },
            SchemaError::UnsupportedBackendKind { kind } =>
            {
                write!(f, "backend.kind `{kind}` is not supported (use `cpu`)")
            },
            SchemaError::UnknownCapability { id } =>
            {
                write!(f, "capability.id `{id}` is not a known capability")
            },
            SchemaError::StringTooLong { field, length, max } =>
            {
                write!(
                    f,
                    "{field} is {length} characters, above the {max}-character limit"
                )
            },
            SchemaError::TooManyOutputs { count, max } =>
            {
                write!(
                    f,
                    "outputs.record lists {count} series, above the {max} limit"
                )
            },
            SchemaError::ZeroSampleEvery =>
            {
                write!(f, "outputs.sample_every must not be zero")
            },
        }
    }
}

impl std::error::Error for SchemaError {}

impl SchemaError {
    /// The stable `SRST-VAL-*` code for this error. Once assigned, a number
    /// must not be reused for a different variant.
    pub fn code(&self) -> ErrorCode {
        let number = match self
        {
            SchemaError::Parse(_) => 1,
            SchemaError::UnsupportedSchemaVersion { .. } => 2,
            SchemaError::UnknownUnit { .. } => 3,
            SchemaError::NonFinite { .. } => 4,
            SchemaError::EndNotAfterStart { .. } => 5,
            SchemaError::NonPositiveStep { .. } => 6,
            SchemaError::UnsupportedPrecision { .. } => 7,
            SchemaError::UnsupportedBackendKind { .. } => 8,
            SchemaError::UnknownCapability { .. } => 9,
            SchemaError::StringTooLong { .. } => 10,
            SchemaError::TooManyOutputs { .. } => 11,
            SchemaError::ZeroSampleEvery => 12,
        };
        ErrorCode::new(ErrorFamily::Validation, number)
    }

    /// A short, human title distinct from the full explanation.
    pub fn title(&self) -> &'static str {
        match self
        {
            SchemaError::Parse(_) => "Scenario file did not parse",
            SchemaError::UnsupportedSchemaVersion { .. } => "Unsupported schema version",
            SchemaError::UnknownUnit { .. } => "Unrecognised unit",
            SchemaError::NonFinite { .. } => "Non-finite value",
            SchemaError::EndNotAfterStart { .. } => "End time not after start time",
            SchemaError::NonPositiveStep { .. } => "Non-positive step",
            SchemaError::UnsupportedPrecision { .. } => "Unsupported precision",
            SchemaError::UnsupportedBackendKind { .. } => "Unsupported backend",
            SchemaError::UnknownCapability { .. } => "Unknown capability",
            SchemaError::StringTooLong { .. } => "Text field too long",
            SchemaError::TooManyOutputs { .. } => "Too many output series",
            SchemaError::ZeroSampleEvery => "Invalid sample interval",
        }
    }

    /// What the user should try next.
    pub fn suggested_action(&self) -> Option<String> {
        match self
        {
            SchemaError::Parse(_) => None,
            SchemaError::UnsupportedSchemaVersion { supported, .. } =>
            {
                Some(format!("set schema_version = {supported}"))
            },
            SchemaError::UnknownUnit { unit, .. } => Some(format!(
                "use one of the supported unit symbols instead of `{unit}`"
            )),
            SchemaError::NonFinite { field, .. } => Some(format!("give {field} a finite value")),
            SchemaError::EndNotAfterStart { .. } =>
            {
                Some("set solver.end greater than solver.start".to_string())
            },
            SchemaError::NonPositiveStep { .. } =>
            {
                Some("set solver.step to a positive value".to_string())
            },
            SchemaError::UnsupportedPrecision { .. } =>
            {
                Some("set backend.precision to `f32` or `f64`".to_string())
            },
            SchemaError::UnsupportedBackendKind { .. } =>
            {
                Some("set backend.kind to `cpu`".to_string())
            },
            SchemaError::UnknownCapability { .. } =>
            {
                Some("run `scirust catalog` to see the available capability ids".to_string())
            },
            SchemaError::StringTooLong { field, max, .. } =>
            {
                Some(format!("shorten {field} to at most {max} characters"))
            },
            SchemaError::TooManyOutputs { max, .. } =>
            {
                Some(format!("list at most {max} series in outputs.record"))
            },
            SchemaError::ZeroSampleEvery =>
            {
                Some("remove outputs.sample_every or set it to a positive integer".to_string())
            },
        }
    }

    /// Render as a [`CatalogedError`] for display in a UI or a bug report.
    /// Every scenario error is recoverable: it always means "fix the
    /// scenario," never "this build has a bug."
    pub fn to_cataloged(&self) -> CatalogedError {
        CatalogedError {
            code: self.code(),
            title: self.title().to_string(),
            explanation: self.to_string(),
            recoverable: true,
            suggested_action: self.suggested_action(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<SchemaError> {
        vec![
            SchemaError::Parse("bad toml".to_string()),
            SchemaError::UnsupportedSchemaVersion {
                found: 2,
                supported: 1,
            },
            SchemaError::UnknownUnit {
                field: "model.mass".to_string(),
                unit: "lb".to_string(),
            },
            SchemaError::NonFinite {
                field: "solver.step".to_string(),
                value: f64::NAN,
            },
            SchemaError::EndNotAfterStart {
                start: 1.0,
                end: 0.0,
            },
            SchemaError::NonPositiveStep { step: -1.0 },
            SchemaError::UnsupportedPrecision {
                precision: "f16".to_string(),
            },
            SchemaError::UnsupportedBackendKind {
                kind: "gpu".to_string(),
            },
            SchemaError::UnknownCapability {
                id: "no.such.capability".to_string(),
            },
            SchemaError::StringTooLong {
                field: "experiment.name".to_string(),
                length: 5000,
                max: 4096,
            },
            SchemaError::TooManyOutputs {
                count: 100,
                max: 64,
            },
            SchemaError::ZeroSampleEvery,
        ]
    }

    #[test]
    fn every_variant_has_a_distinct_stable_code() {
        let codes: Vec<String> = all_variants()
            .iter()
            .map(|e| e.code().to_string())
            .collect();
        let mut unique = codes.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            codes.len(),
            unique.len(),
            "duplicate SRST-VAL code among {codes:?}"
        );
        assert!(codes.iter().all(|c| c.starts_with("SRST-VAL-")));
    }

    #[test]
    fn to_cataloged_is_always_recoverable_and_carries_the_code() {
        for err in all_variants()
        {
            let cataloged = err.to_cataloged();
            assert!(cataloged.recoverable);
            assert_eq!(cataloged.code, err.code());
            assert!(!cataloged.explanation.is_empty());
        }
    }

    #[test]
    fn display_messages_mention_the_offending_field_or_value() {
        assert!(
            SchemaError::EndNotAfterStart {
                start: 5.0,
                end: 5.0
            }
            .to_string()
            .contains("5")
        );
        assert!(
            SchemaError::UnknownUnit {
                field: "model.mass".to_string(),
                unit: "lb".to_string()
            }
            .to_string()
            .contains("model.mass")
        );
    }
}
