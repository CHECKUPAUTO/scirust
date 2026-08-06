/// Runs all 12 deterministic Phase 6 quantization scenarios.
pub fn run_standard_suite() -> Result<Vec<ScenarioReport>, Phase6Error> {
    scenarios().iter().map(run_scenario).collect()
}

/// Serializes Phase 6 reports into a stable 36-column CSV document.
#[must_use]
pub fn suite_to_csv(reports: &[ScenarioReport]) -> String {
    let mut output = String::from(CSV_HEADER);
    output.push('\n');
    for report in reports {
        output.push_str(&report.csv_row());
        output.push('\n');
    }
    output
}

fn run_scenario(scenario: &Scenario) -> Result<ScenarioReport, Phase6Error> {
    scenario.validate()?;
    let data = generate(scenario)?;
    let dense_bytes = to_u64(product(product(scenario.tokens, scenario.dimension)?, 8)?)?;
    let budget_bytes = dense_bytes
        .checked_mul(scenario.budget_percent)
        .ok_or(Phase6Error::Overflow)?
        / 100;
    let baseline = evaluate(scenario, &data, Formats::fp32())?;
    let key_residual_formats = options(scenario.key_slots);
    let value_residual_formats = options(scenario.value_slots);
    let mut candidates = Vec::new();
    let mut evaluated = 0_usize;

    for key_coefficients in FORMATS {
        for value_coefficients in FORMATS {
            for &key_residuals in key_residual_formats {
                for &value_residuals in value_residual_formats {
                    evaluated = evaluated.checked_add(1).ok_or(Phase6Error::Overflow)?;
                    let candidate = evaluate(
                        scenario,
                        &data,
                        Formats {
                            key_coefficients,
                            value_coefficients,
                            key_residuals,
                            value_residuals,
                        },
                    )?;
                    if candidate.bytes <= budget_bytes {
                        candidates.push(candidate);
                    }
                }
            }
        }
    }

    if candidates.is_empty() {
        return Err(Phase6Error::NoBudgetCandidate { budget_bytes });
    }
    let budget_feasible = candidates.len();
    let quality_feasible = candidates.iter().filter(|candidate| candidate.guard).count();
    candidates.sort_by(compare_candidates);
    let selected = candidates.remove(0);

    Ok(ScenarioReport {
        scenario: scenario.clone(),
        dense_bytes,
        budget_bytes,
        baseline,
        selected,
        evaluated,
        budget_feasible,
        quality_feasible,
    })
}

fn evaluate(
    scenario: &Scenario,
    data: &Data,
    formats: Formats,
) -> Result<Candidate, Phase6Error> {
    let key_coefficients = QuantizedRows::encode(
        &data.key_coefficients,
        scenario.tokens,
        scenario.key_rank,
        formats.key_coefficients,
    )?;
    let value_coefficients = QuantizedRows::encode(
        &data.value_coefficients,
        scenario.tokens,
        scenario.value_rank,
        formats.value_coefficients,
    )?;
    let key_residuals = QuantizedRows::encode(
        &data.key_residuals,
        scenario.tokens,
        scenario.key_slots,
        formats.key_residuals,
    )?;
    let value_residuals = QuantizedRows::encode(
        &data.value_residuals,
        scenario.tokens,
        scenario.value_slots,
        formats.value_residuals,
    )?;

    let keys = reconstruct(
        scenario.tokens,
        scenario.dimension,
        scenario.key_rank,
        &key_coefficients.decode()?,
        scenario.key_slots,
        &data.key_indices,
        &key_residuals.decode()?,
    )?;
    let values = reconstruct(
        scenario.tokens,
        scenario.dimension,
        scenario.value_rank,
        &value_coefficients.decode()?,
        scenario.value_slots,
        &data.value_indices,
        &value_residuals.decode()?,
    )?;
    let reference_attention = attention(
        &data.dense_keys,
        &data.dense_values,
        &data.queries,
        scenario,
    )?;
    let candidate_attention = attention(&keys, &values, &data.queries, scenario)?;

    let key_error = relative_rms(&data.dense_keys, &keys)?;
    let value_error = relative_rms(&data.dense_values, &values)?;
    let attention_error = max_error(&reference_attention, &candidate_attention)?;
    let ratio = target_ratio(key_error, scenario.key_target)
        .max(target_ratio(value_error, scenario.value_target))
        .max(target_ratio(attention_error, scenario.attention_target));
    let guard = ratio <= 1.000_001;
    let bytes = storage(
        scenario,
        &key_coefficients,
        &value_coefficients,
        &key_residuals,
        &value_residuals,
    )?;

    let mut fingerprint = fnv_u64(FNV_OFFSET, scenario.seed);
    for code in formats.key() {
        fingerprint = fnv_byte(fingerprint, code);
    }
    for tensor in [
        &key_coefficients,
        &value_coefficients,
        &key_residuals,
        &value_residuals,
    ] {
        for scale in &tensor.scales {
            fingerprint = fnv_u32(fingerprint, scale.to_bits());
        }
        for byte in &tensor.payload {
            fingerprint = fnv_byte(fingerprint, *byte);
        }
    }
    for value in candidate_attention {
        fingerprint = fnv_u32(fingerprint, value.to_bits());
    }

    Ok(Candidate {
        formats,
        bytes,
        key_error,
        value_error,
        attention_error,
        ratio,
        guard,
        fingerprint,
    })
}

fn compare_candidates(left: &Candidate, right: &Candidate) -> Ordering {
    match (left.guard, right.guard) {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }
    if left.guard {
        left.bytes
            .cmp(&right.bytes)
            .then_with(|| ordered(left.ratio, right.ratio))
            .then_with(|| left.formats.key().cmp(&right.formats.key()))
    } else {
        ordered(left.ratio, right.ratio)
            .then_with(|| left.bytes.cmp(&right.bytes))
            .then_with(|| left.formats.key().cmp(&right.formats.key()))
    }
}

fn scenarios() -> Vec<Scenario> {
    let dimensions = [(16, 24, 3, 4), (32, 32, 4, 5), (64, 40, 5, 6)];
    let variants = [
        (0, 0, 0.0, 50, 0.0, 0.0, 0.0),
        (1, 1, 0.06, 42, 0.008, 0.008, 0.003),
        (2, 1, 0.10, 38, 0.060, 0.012, 0.006),
        (2, 2, 0.18, 42, 0.070, 0.070, 0.030),
    ];
    let mut output = Vec::with_capacity(12);
    for (dimension_index, (dimension, tokens, key_rank, value_rank)) in
        dimensions.into_iter().enumerate()
    {
        for (variant_index, (key_slots, value_slots, amplitude, budget, kt, vt, at)) in
            variants.into_iter().enumerate()
        {
            output.push(Scenario {
                seed: 0x6a09_0000_0000_0000 + (dimension_index * 16 + variant_index) as u64,
                tokens,
                dimension,
                queries: 4,
                key_rank,
                value_rank,
                key_slots,
                value_slots,
                residual_amplitude: amplitude,
                budget_percent: budget,
                key_target: kt,
                value_target: vt,
                attention_target: at,
            });
        }
    }
    output
}

fn generate(scenario: &Scenario) -> Result<Data, Phase6Error> {
    let mut rng = Rng::new(scenario.seed);
    let key_coefficients = random(&mut rng, product(scenario.tokens, scenario.key_rank)?);
    let value_coefficients = random(&mut rng, product(scenario.tokens, scenario.value_rank)?);
    let (key_indices, key_residuals) = residuals(
        scenario.tokens,
        scenario.dimension,
        scenario.key_rank,
        scenario.key_slots,
        scenario.residual_amplitude,
        3,
    )?;
    let (value_indices, value_residuals) = residuals(
        scenario.tokens,
        scenario.dimension,
        scenario.value_rank,
        scenario.value_slots,
        scenario.residual_amplitude,
        5,
    )?;
    let dense_keys = reconstruct(
        scenario.tokens,
        scenario.dimension,
        scenario.key_rank,
        &key_coefficients,
        scenario.key_slots,
        &key_indices,
        &key_residuals,
    )?;
    let dense_values = reconstruct(
        scenario.tokens,
        scenario.dimension,
        scenario.value_rank,
        &value_coefficients,
        scenario.value_slots,
        &value_indices,
        &value_residuals,
    )?;
    let queries = random(&mut rng, product(scenario.queries, scenario.dimension)?);
    Ok(Data {
        key_coefficients,
        value_coefficients,
        key_indices,
        key_residuals,
        value_indices,
        value_residuals,
        dense_keys,
        dense_values,
        queries,
    })
}

fn residuals(
    tokens: usize,
    dimension: usize,
    rank: usize,
    slots: usize,
    amplitude: f32,
    stride: usize,
) -> Result<(Vec<u16>, Vec<f32>), Phase6Error> {
    if slots == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let tail = dimension.checked_sub(rank).ok_or(Phase6Error::Overflow)?;
    if slots > tail {
        return Err(Phase6Error::Shape {
            name: "residual_slots",
            value: slots,
            maximum: tail,
        });
    }
    let count = product(tokens, slots)?;
    let mut indices = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    for token in 0..tokens {
        for slot in 0..slots {
            let coordinate = rank + (token * (slots + 1) + slot * stride) % tail;
            indices.push(u16::try_from(coordinate).map_err(|_| Phase6Error::Overflow)?);
            let sign = if (token + slot) % 2 == 0 { 1.0 } else { -1.0 };
            values.push(sign * amplitude * (1.0 + slot as f32 * 0.25));
        }
    }
    Ok((indices, values))
}

fn reconstruct(
    tokens: usize,
    dimension: usize,
    rank: usize,
    coefficients: &[f32],
    slots: usize,
    indices: &[u16],
    residual_values: &[f32],
) -> Result<Vec<f32>, Phase6Error> {
    length("coefficients", coefficients.len(), product(tokens, rank)?)?;
    length("residual indices", indices.len(), product(tokens, slots)?)?;
    length(
        "residual values",
        residual_values.len(),
        product(tokens, slots)?,
    )?;
    let mut output = vec![0.0_f32; product(tokens, dimension)?];
    for token in 0..tokens {
        let dense_offset = token * dimension;
        let coefficient_offset = token * rank;
        output[dense_offset..dense_offset + rank]
            .copy_from_slice(&coefficients[coefficient_offset..coefficient_offset + rank]);
        let residual_offset = token * slots;
        for slot in 0..slots {
            let coordinate = usize::from(indices[residual_offset + slot]);
            if coordinate >= dimension {
                return Err(Phase6Error::Shape {
                    name: "residual_coordinate",
                    value: coordinate,
                    maximum: dimension - 1,
                });
            }
            output[dense_offset + coordinate] += residual_values[residual_offset + slot];
        }
    }
    Ok(output)
}

fn attention(
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    scenario: &Scenario,
) -> Result<Vec<f32>, Phase6Error> {
    length(
        "keys",
        keys.len(),
        product(scenario.tokens, scenario.dimension)?,
    )?;
    length(
        "values",
        values.len(),
        product(scenario.tokens, scenario.dimension)?,
    )?;
    length(
        "queries",
        queries.len(),
        product(scenario.queries, scenario.dimension)?,
    )?;
    let mut output = vec![0.0_f32; product(scenario.queries, scenario.dimension)?];
    let mut scores = vec![0.0_f32; scenario.tokens];
    let scale = 1.0 / (scenario.dimension as f32).sqrt();

    for query_index in 0..scenario.queries {
        let query_offset = query_index * scenario.dimension;
        let query = &queries[query_offset..query_offset + scenario.dimension];
        for (token, score) in scores.iter_mut().enumerate() {
            let offset = token * scenario.dimension;
            *score = dot(query, &keys[offset..offset + scenario.dimension]) * scale;
        }
        softmax(&mut scores)?;
        let destination = &mut output[query_offset..query_offset + scenario.dimension];
        for (token, probability) in scores.iter().copied().enumerate() {
            let offset = token * scenario.dimension;
            for (target, source) in destination
                .iter_mut()
                .zip(&values[offset..offset + scenario.dimension])
            {
                *target += probability * source;
            }
        }
    }
    Ok(output)
}

fn storage(
    scenario: &Scenario,
    key_coefficients: &QuantizedRows,
    value_coefficients: &QuantizedRows,
    key_residuals: &QuantizedRows,
    value_residuals: &QuantizedRows,
) -> Result<u64, Phase6Error> {
    let basis_elements = product(
        scenario.dimension,
        scenario
            .key_rank
            .checked_add(scenario.value_rank)
            .ok_or(Phase6Error::Overflow)?,
    )?;
    let index_elements = product(
        scenario.tokens,
        scenario
            .key_slots
            .checked_add(scenario.value_slots)
            .ok_or(Phase6Error::Overflow)?,
    )?;
    let basis_bytes = to_u64(product(basis_elements, 4)?)?;
    let index_bytes = to_u64(product(index_elements, 2)?)?;
    [
        basis_bytes,
        index_bytes,
        key_coefficients.bytes()?,
        value_coefficients.bytes()?,
        key_residuals.bytes()?,
        value_residuals.bytes()?,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or(Phase6Error::Overflow)
    })
}
