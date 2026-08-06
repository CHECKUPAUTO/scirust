use core::{cmp::Ordering, fmt};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const FORMATS: [Format; 3] = [Format::F32, Format::I8, Format::I4];
const F32_ONLY: [Format; 1] = [Format::F32];

/// Errors returned by the deterministic Phase 6 experiment.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase6Error {
    /// A required count or dimension was zero.
    Zero {
        /// Human-readable field name.
        field: &'static str,
    },
    /// A flat buffer had an unexpected length.
    Length {
        /// Human-readable buffer name.
        name: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// A rank, slot count, or residual coordinate exceeded its bound.
    Shape {
        /// Human-readable field name.
        name: &'static str,
        /// Supplied value.
        value: usize,
        /// Maximum accepted value.
        maximum: usize,
    },
    /// A scalar was non-finite or outside its accepted range.
    Scalar {
        /// Human-readable scalar name.
        name: &'static str,
        /// Invalid scalar value.
        value: f64,
    },
    /// No candidate fitted the strict storage budget.
    NoBudgetCandidate {
        /// Strict budget in bytes.
        budget_bytes: u64,
    },
    /// Integer arithmetic overflowed.
    Overflow,
}

impl fmt::Display for Phase6Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero { field } => write!(output, "{field} must be non-zero"),
            Self::Length {
                name,
                expected,
                actual,
            } => write!(
                output,
                "{name} length mismatch: expected {expected}, received {actual}"
            ),
            Self::Shape {
                name,
                value,
                maximum,
            } => write!(output, "{name}={value} exceeds maximum {maximum}"),
            Self::Scalar { name, value } => {
                write!(output, "invalid {name} value: {value}")
            }
            Self::NoBudgetCandidate { budget_bytes } => {
                write!(output, "no candidate fits {budget_bytes} bytes")
            }
            Self::Overflow => write!(output, "Phase 6 arithmetic overflow"),
        }
    }
}

impl std::error::Error for Phase6Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Format {
    F32,
    I8,
    I4,
}

impl Format {
    const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I8 => "int8",
            Self::I4 => "int4",
        }
    }

    const fn code(self) -> u8 {
        match self {
            Self::I4 => 0,
            Self::I8 => 1,
            Self::F32 => 2,
        }
    }

    const fn limit(self) -> Option<i8> {
        match self {
            Self::F32 => None,
            Self::I8 => Some(127),
            Self::I4 => Some(7),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct QuantizedRows {
    format: Format,
    rows: usize,
    columns: usize,
    scales: Vec<f32>,
    payload: Vec<u8>,
}

impl QuantizedRows {
    fn encode(
        source: &[f32],
        rows: usize,
        columns: usize,
        requested: Format,
    ) -> Result<Self, Phase6Error> {
        non_zero("rows", rows)?;
        let elements = product(rows, columns)?;
        length("quantized source", source.len(), elements)?;
        finite("quantized source", source)?;

        if columns == 0 {
            return Ok(Self {
                format: Format::F32,
                rows,
                columns,
                scales: Vec::new(),
                payload: Vec::new(),
            });
        }

        if requested == Format::F32 {
            let capacity = product(elements, 4)?;
            let mut payload = Vec::with_capacity(capacity);
            for value in source {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            return Ok(Self {
                format: requested,
                rows,
                columns,
                scales: Vec::new(),
                payload,
            });
        }

        let limit = requested.limit().ok_or(Phase6Error::Overflow)?;
        let row_bytes = payload_bytes(requested, columns)?;
        let mut scales = Vec::with_capacity(rows);
        let mut payload = vec![0_u8; product(rows, row_bytes)?];

        for (row_index, row) in source.chunks_exact(columns).enumerate() {
            let maximum = row
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0_f32, f32::max);
            let scale = if maximum == 0.0 {
                1.0
            } else {
                maximum / f32::from(limit)
            };
            scales.push(scale);
            let offset = row_index * row_bytes;

            for (column, value) in row.iter().copied().enumerate() {
                let code = quantize(value, scale, limit);
                match requested {
                    Format::I8 => payload[offset + column] = code.to_ne_bytes()[0],
                    Format::I4 => {
                        let nibble = code.to_ne_bytes()[0] & 0x0f;
                        let position = offset + column / 2;
                        if column % 2 == 0 {
                            payload[position] = nibble;
                        } else {
                            payload[position] |= nibble << 4;
                        }
                    }
                    Format::F32 => return Err(Phase6Error::Overflow),
                }
            }
        }

        Ok(Self {
            format: requested,
            rows,
            columns,
            scales,
            payload,
        })
    }

    fn decode(&self) -> Result<Vec<f32>, Phase6Error> {
        if self.columns == 0 {
            return Ok(Vec::new());
        }

        let elements = product(self.rows, self.columns)?;
        let mut output = Vec::with_capacity(elements);
        match self.format {
            Format::F32 => {
                length("FP32 payload", self.payload.len(), product(elements, 4)?)?;
                for bytes in self.payload.chunks_exact(4) {
                    output.push(f32::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                    ]));
                }
            }
            Format::I8 => {
                length("INT8 scales", self.scales.len(), self.rows)?;
                length("INT8 payload", self.payload.len(), elements)?;
                for row in 0..self.rows {
                    let offset = row * self.columns;
                    for byte in &self.payload[offset..offset + self.columns] {
                        output.push(f32::from(i8::from_ne_bytes([*byte])) * self.scales[row]);
                    }
                }
            }
            Format::I4 => {
                length("INT4 scales", self.scales.len(), self.rows)?;
                let row_bytes = payload_bytes(self.format, self.columns)?;
                length("INT4 payload", self.payload.len(), product(self.rows, row_bytes)?)?;
                for row in 0..self.rows {
                    let offset = row * row_bytes;
                    for column in 0..self.columns {
                        let packed = self.payload[offset + column / 2];
                        let nibble = if column % 2 == 0 {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let signed = if nibble < 8 {
                            nibble as i8
                        } else {
                            (i16::from(nibble) - 16) as i8
                        };
                        output.push(f32::from(signed) * self.scales[row]);
                    }
                }
            }
        }
        Ok(output)
    }

    fn bytes(&self) -> Result<u64, Phase6Error> {
        let payload = u64::try_from(self.payload.len()).map_err(|_| Phase6Error::Overflow)?;
        let scales = u64::try_from(self.scales.len()).map_err(|_| Phase6Error::Overflow)?;
        payload
            .checked_add(scales.checked_mul(4).ok_or(Phase6Error::Overflow)?)
            .ok_or(Phase6Error::Overflow)
    }
}

#[derive(Debug, Clone)]
struct Scenario {
    seed: u64,
    tokens: usize,
    dimension: usize,
    queries: usize,
    key_rank: usize,
    value_rank: usize,
    key_slots: usize,
    value_slots: usize,
    residual_amplitude: f32,
    budget_percent: u64,
    key_target: f64,
    value_target: f64,
    attention_target: f64,
}

impl Scenario {
    fn validate(&self) -> Result<(), Phase6Error> {
        for (name, value) in [
            ("tokens", self.tokens),
            ("dimension", self.dimension),
            ("queries", self.queries),
            ("key_rank", self.key_rank),
            ("value_rank", self.value_rank),
        ] {
            non_zero(name, value)?;
        }
        for (name, value) in [
            ("key_rank", self.key_rank),
            ("value_rank", self.value_rank),
            ("key_slots", self.key_slots),
            ("value_slots", self.value_slots),
        ] {
            if value > self.dimension {
                return Err(Phase6Error::Shape {
                    name,
                    value,
                    maximum: self.dimension,
                });
            }
        }
        if self.dimension > usize::from(u16::MAX) {
            return Err(Phase6Error::Shape {
                name: "dimension",
                value: self.dimension,
                maximum: usize::from(u16::MAX),
            });
        }
        if self.budget_percent == 0 {
            return Err(Phase6Error::Zero {
                field: "budget_percent",
            });
        }
        for (name, value) in [
            ("residual_amplitude", f64::from(self.residual_amplitude)),
            ("key_target", self.key_target),
            ("value_target", self.value_target),
            ("attention_target", self.attention_target),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Phase6Error::Scalar { name, value });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Data {
    key_coefficients: Vec<f32>,
    value_coefficients: Vec<f32>,
    key_indices: Vec<u16>,
    key_residuals: Vec<f32>,
    value_indices: Vec<u16>,
    value_residuals: Vec<f32>,
    dense_keys: Vec<f32>,
    dense_values: Vec<f32>,
    queries: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Formats {
    key_coefficients: Format,
    value_coefficients: Format,
    key_residuals: Format,
    value_residuals: Format,
}

impl Formats {
    const fn fp32() -> Self {
        Self {
            key_coefficients: Format::F32,
            value_coefficients: Format::F32,
            key_residuals: Format::F32,
            value_residuals: Format::F32,
        }
    }

    const fn key(self) -> [u8; 4] {
        [
            self.key_coefficients.code(),
            self.value_coefficients.code(),
            self.key_residuals.code(),
            self.value_residuals.code(),
        ]
    }

    fn counts(self) -> (usize, usize, usize) {
        let mut int4 = 0;
        let mut int8 = 0;
        let mut f32_count = 0;
        for format in [
            self.key_coefficients,
            self.value_coefficients,
            self.key_residuals,
            self.value_residuals,
        ] {
            match format {
                Format::I4 => int4 += 1,
                Format::I8 => int8 += 1,
                Format::F32 => f32_count += 1,
            }
        }
        (int4, int8, f32_count)
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    formats: Formats,
    bytes: u64,
    key_error: f64,
    value_error: f64,
    attention_error: f64,
    ratio: f64,
    guard: bool,
    fingerprint: u64,
}

/// One deterministic Phase 6 scenario report.
#[derive(Debug, Clone)]
pub struct ScenarioReport {
    scenario: Scenario,
    dense_bytes: u64,
    budget_bytes: u64,
    baseline: Candidate,
    selected: Candidate,
    evaluated: usize,
    budget_feasible: usize,
    quality_feasible: usize,
}

impl ScenarioReport {
    fn csv_row(&self) -> String {
        let (int4_count, int8_count, f32_count) = self.selected.formats.counts();
        let fields = vec![
            self.scenario.seed.to_string(),
            self.scenario.tokens.to_string(),
            self.scenario.dimension.to_string(),
            self.scenario.queries.to_string(),
            self.scenario.key_rank.to_string(),
            self.scenario.value_rank.to_string(),
            self.scenario.key_slots.to_string(),
            self.scenario.value_slots.to_string(),
            self.scenario.budget_percent.to_string(),
            "100".to_owned(),
            self.budget_bytes.to_string(),
            scientific(self.scenario.key_target),
            scientific(self.scenario.value_target),
            scientific(self.scenario.attention_target),
            self.baseline.bytes.to_string(),
            self.selected.bytes.to_string(),
            self.dense_bytes.to_string(),
            scientific(self.dense_bytes as f64 / self.selected.bytes as f64),
            self.selected.formats.key_coefficients.label().to_owned(),
            self.selected.formats.value_coefficients.label().to_owned(),
            self.selected.formats.key_residuals.label().to_owned(),
            self.selected.formats.value_residuals.label().to_owned(),
            u8::from(self.baseline.guard).to_string(),
            u8::from(self.selected.guard).to_string(),
            scientific(self.baseline.ratio),
            scientific(self.selected.ratio),
            scientific(self.selected.key_error),
            scientific(self.selected.value_error),
            scientific(self.selected.attention_error),
            self.evaluated.to_string(),
            self.budget_feasible.to_string(),
            self.quality_feasible.to_string(),
            int4_count.to_string(),
            int8_count.to_string(),
            f32_count.to_string(),
            format!("{:016x}", self.selected.fingerprint),
        ];
        debug_assert_eq!(fields.len(), 36);
        fields.join(",")
    }
}

const CSV_HEADER: &str = concat!(
    "seed,token_count,head_dimension,query_count,key_rank,value_rank,",
    "key_slots_per_token,value_slots_per_token,budget_numerator,budget_denominator,",
    "budget_bytes,key_target_relative_rms,value_target_relative_rms,",
    "attention_target_max_absolute,baseline_total_bytes,selected_total_bytes,",
    "dense_bytes,compression_ratio,selected_key_coeff_format,",
    "selected_value_coeff_format,selected_key_residual_format,",
    "selected_value_residual_format,baseline_quality_guard_met,",
    "selected_quality_guard_met,baseline_worst_target_ratio,",
    "selected_worst_target_ratio,key_reconstruction_relative_rms,",
    "value_reconstruction_relative_rms,attention_max_absolute,",
    "evaluated_candidates,budget_feasible_candidates,quality_feasible_candidates,",
    "int4_component_count,int8_component_count,f32_component_count,output_fingerprint"
);
