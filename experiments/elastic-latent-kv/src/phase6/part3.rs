fn relative_rms(reference: &[f32], candidate: &[f32]) -> Result<f64, Phase6Error> {
    length("relative RMS", candidate.len(), reference.len())?;
    non_zero("relative RMS elements", reference.len())?;
    let mut signal = 0.0_f64;
    let mut error = 0.0_f64;
    for (&left, &right) in reference.iter().zip(candidate) {
        let left = f64::from(left);
        let delta = f64::from(right) - left;
        signal += left * left;
        error += delta * delta;
    }
    if signal == 0.0 {
        return Ok(if error == 0.0 { 0.0 } else { f64::INFINITY });
    }
    Ok((error / signal).sqrt())
}

fn max_error(reference: &[f32], candidate: &[f32]) -> Result<f64, Phase6Error> {
    length("attention output", candidate.len(), reference.len())?;
    Ok(reference
        .iter()
        .zip(candidate)
        .map(|(&left, &right)| (f64::from(left) - f64::from(right)).abs())
        .fold(0.0_f64, f64::max))
}

fn softmax(values: &mut [f32]) -> Result<(), Phase6Error> {
    non_zero("softmax values", values.len())?;
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Phase6Error::Scalar {
            name: "softmax_sum",
            value: f64::from(sum),
        });
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(left, right)| left * right).sum()
}

fn target_ratio(value: f64, target: f64) -> f64 {
    if target == 0.0 {
        if value == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        value / target
    }
}

fn ordered(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn options(columns: usize) -> &'static [Format] {
    if columns == 0 { &F32_ONLY } else { &FORMATS }
}

fn payload_bytes(format: Format, columns: usize) -> Result<usize, Phase6Error> {
    match format {
        Format::F32 => product(columns, 4),
        Format::I8 => Ok(columns),
        Format::I4 => columns.checked_add(1).map(|value| value / 2).ok_or(Phase6Error::Overflow),
    }
}

fn quantize(value: f32, scale: f32, limit: i8) -> i8 {
    if value == 0.0 {
        0
    } else {
        let bound = f32::from(limit);
        (value / scale).round().clamp(-bound, bound) as i8
    }
}

fn scientific(value: f64) -> String {
    format!("{value:.9e}")
}

fn product(left: usize, right: usize) -> Result<usize, Phase6Error> {
    left.checked_mul(right).ok_or(Phase6Error::Overflow)
}

fn to_u64(value: usize) -> Result<u64, Phase6Error> {
    u64::try_from(value).map_err(|_| Phase6Error::Overflow)
}

fn non_zero(name: &'static str, value: usize) -> Result<(), Phase6Error> {
    if value == 0 {
        Err(Phase6Error::Zero { field: name })
    } else {
        Ok(())
    }
}

fn length(name: &'static str, actual: usize, expected: usize) -> Result<(), Phase6Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Phase6Error::Length {
            name,
            expected,
            actual,
        })
    }
}

fn finite(name: &'static str, values: &[f32]) -> Result<(), Phase6Error> {
    if let Some(value) = values.iter().copied().find(|value| !value.is_finite()) {
        Err(Phase6Error::Scalar {
            name,
            value: f64::from(value),
        })
    } else {
        Ok(())
    }
}

fn random(rng: &mut Rng, length: usize) -> Vec<f32> {
    (0..length).map(|_| rng.symmetric()).collect()
}

#[derive(Debug, Clone, Copy)]
struct Rng {
    state: u64,
}

impl Rng {
    const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn symmetric(&mut self) -> f32 {
        let mantissa = (self.next() >> 40) as u32;
        mantissa as f32 / 16_777_215.0 * 2.0 - 1.0
    }
}

fn fnv_byte(mut hash: u64, byte: u8) -> u64 {
    hash ^= u64::from(byte);
    hash.wrapping_mul(FNV_PRIME)
}

fn fnv_u32(mut hash: u64, value: u32) -> u64 {
    for byte in value.to_le_bytes() {
        hash = fnv_byte(hash, byte);
    }
    hash
}

fn fnv_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash = fnv_byte(hash, byte);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{
        CSV_HEADER, Format, Phase6Error, QuantizedRows, Scenario, run_scenario,
        run_standard_suite, suite_to_csv,
    };

    #[test]
    fn zero_rows_round_trip_exactly() {
        let source = vec![0.0_f32; 15];
        for format in [Format::I8, Format::I4] {
            let encoded = QuantizedRows::encode(&source, 3, 5, format).unwrap();
            assert_eq!(encoded.decode().unwrap(), source);
        }
    }

    #[test]
    fn int4_uses_packed_rows() {
        let encoded = QuantizedRows::encode(
            &[-1.0, -0.5, 0.0, 0.5, 1.0],
            1,
            5,
            Format::I4,
        )
        .unwrap();
        assert_eq!(encoded.payload.len(), 3);
        assert_eq!(encoded.scales.len(), 1);
    }

    #[test]
    fn int8_error_is_bounded_by_half_scale() {
        let source = [-0.91_f32, -0.33, 0.12, 0.78, 1.0];
        let encoded = QuantizedRows::encode(&source, 1, 5, Format::I8).unwrap();
        let decoded = encoded.decode().unwrap();
        let bound = encoded.scales[0] * 0.500_001;
        for (&left, &right) in source.iter().zip(&decoded) {
            assert!((left - right).abs() <= bound);
        }
    }

    #[test]
    fn non_finite_input_is_rejected() {
        assert!(matches!(
            QuantizedRows::encode(&[0.0, f32::NAN], 1, 2, Format::I8),
            Err(Phase6Error::Scalar { .. })
        ));
    }

    #[test]
    fn storage_accounting_matches_closed_form() {
        let source = vec![0.25_f32; 15];
        let int8 = QuantizedRows::encode(&source, 3, 5, Format::I8).unwrap();
        let int4 = QuantizedRows::encode(&source, 3, 5, Format::I4).unwrap();
        assert_eq!(int8.bytes().unwrap(), 27);
        assert_eq!(int4.bytes().unwrap(), 21);
    }

    #[test]
    fn standard_suite_is_deterministic() {
        let first = suite_to_csv(&run_standard_suite().unwrap());
        let second = suite_to_csv(&run_standard_suite().unwrap());
        assert_eq!(first, second);
    }

    #[test]
    fn standard_suite_is_budget_and_quality_safe() {
        let reports = run_standard_suite().unwrap();
        assert_eq!(reports.len(), 12);
        for report in reports {
            assert!(report.selected.bytes <= report.budget_bytes);
            assert!(report.selected.guard);
            assert!(report.quality_feasible > 0);
        }
    }

    #[test]
    fn exact_scenarios_select_fp32_coefficients() {
        for report in run_standard_suite().unwrap() {
            if report.scenario.residual_amplitude == 0.0 {
                assert_eq!(report.selected.formats.key_coefficients, Format::F32);
                assert_eq!(report.selected.formats.value_coefficients, Format::F32);
            }
        }
    }

    #[test]
    fn structured_scenarios_compress_the_baseline() {
        for report in run_standard_suite().unwrap() {
            if report.scenario.residual_amplitude > 0.0 {
                assert!(report.selected.bytes < report.baseline.bytes);
                let (int4, int8, _) = report.selected.formats.counts();
                assert!(int4 + int8 > 0);
            }
        }
    }

    #[test]
    fn csv_has_36_columns_and_12_rows() {
        let csv = suite_to_csv(&run_standard_suite().unwrap());
        let lines = csv.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 13);
        assert_eq!(lines[0], CSV_HEADER);
        assert!(lines.iter().all(|line| line.split(',').count() == 36));
    }

    #[test]
    fn impossible_budget_is_reported() {
        let scenario = Scenario {
            seed: 1,
            tokens: 8,
            dimension: 16,
            queries: 2,
            key_rank: 2,
            value_rank: 2,
            key_slots: 1,
            value_slots: 1,
            residual_amplitude: 0.1,
            budget_percent: 1,
            key_target: 1.0,
            value_target: 1.0,
            attention_target: 1.0,
        };
        assert!(matches!(
            run_scenario(&scenario),
            Err(Phase6Error::NoBudgetCandidate { .. })
        ));
    }
}
