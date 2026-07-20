use scirust_cayley_filter::{
    CayleyProjector, MultiplierCase, Sedenion, SoftCayleyFilter, TemporalBlockFilter,
    select_zero_divisor_train_dev,
};
use scirust_signal::denoise::{classify, denoise_auto, stft_wiener_auto};

const SAMPLE_RATE: f64 = 16_000.0;
const BLOCK_SIZE: usize = 16;
const TRAIN_SAMPLES: usize = 4_096;
const DEV_SAMPLES: usize = 4_096;
const TOP_K: usize = 16;
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;

#[derive(Clone, Copy, Debug)]
struct Metrics {
    mse: f64,
    snr_db: f64,
    improvement_db: f64,
}

fn load_fixture() -> Result<(Vec<f64>, Vec<f64>), String> {
    let raw = include_str!("../../scirust-signal/tests/data/voicebank_demand.csv");

    let mut clean = Vec::new();
    let mut noisy = Vec::new();

    for (line_index, line) in raw.lines().enumerate()
    {
        if line.starts_with('#') || line.trim().is_empty()
        {
            continue;
        }

        let (clean_text, noisy_text) = line
            .split_once(',')
            .ok_or_else(|| format!("invalid CSV line {}", line_index + 1))?;

        clean.push(clean_text.trim().parse::<f64>().map_err(|error| {
            format!("invalid clean sample at line {}: {error}", line_index + 1,)
        })?);

        noisy.push(noisy_text.trim().parse::<f64>().map_err(|error| {
            format!("invalid noisy sample at line {}: {error}", line_index + 1,)
        })?);
    }

    if clean.len() != noisy.len() || clean.is_empty()
    {
        return Err("invalid clean/noisy fixture lengths".into());
    }

    if clean.len() % BLOCK_SIZE != 0
    {
        return Err("fixture length is not divisible by 16".into());
    }

    Ok((clean, noisy))
}

fn make_cases(clean: &[f64], noisy: &[f64]) -> Result<Vec<MultiplierCase>, String> {
    if clean.len() != noisy.len() || !clean.len().is_multiple_of(BLOCK_SIZE)
    {
        return Err("invalid block-aligned slices".into());
    }

    let (clean_blocks, clean_remainder) = clean.as_chunks::<BLOCK_SIZE>();
    let (noisy_blocks, noisy_remainder) = noisy.as_chunks::<BLOCK_SIZE>();

    if !clean_remainder.is_empty() || !noisy_remainder.is_empty()
    {
        return Err("unexpected incomplete block".into());
    }

    Ok(clean_blocks
        .iter()
        .zip(noisy_blocks)
        .map(|(clean_block, noisy_block)| {
            let signal = *clean_block;

            let noise: Sedenion =
                core::array::from_fn(|index| noisy_block[index] - clean_block[index]);

            MultiplierCase::new(signal, noise)
        })
        .collect())
}

fn metrics(clean: &[f64], noisy: &[f64], estimate: &[f64]) -> Metrics {
    let signal_energy = clean.iter().map(|value| value * value).sum::<f64>();

    let input_error = clean
        .iter()
        .zip(noisy)
        .map(|(clean_value, noisy_value)| {
            let error = clean_value - noisy_value;
            error * error
        })
        .sum::<f64>();

    let output_error = clean
        .iter()
        .zip(estimate)
        .map(|(clean_value, estimate_value)| {
            let error = clean_value - estimate_value;
            error * error
        })
        .sum::<f64>();

    let snr_db = 10.0 * (signal_energy / output_error.max(ENERGY_FLOOR)).log10();

    let improvement_db = 10.0 * (input_error / output_error.max(ENERGY_FLOOR)).log10();

    Metrics {
        mse: output_error / clean.len() as f64,
        snr_db,
        improvement_db,
    }
}

fn print_metrics(method: &str, metrics: Metrics) {
    println!(
        "{method},{},{},{}",
        metrics.mse, metrics.snr_db, metrics.improvement_db,
    );
}

fn main() -> Result<(), String> {
    let (clean, noisy) = load_fixture()?;

    let dev_end = TRAIN_SAMPLES + DEV_SAMPLES;

    if dev_end >= clean.len()
    {
        return Err("fixture is too short for train/dev/test split".into());
    }

    let train_cases = make_cases(&clean[..TRAIN_SAMPLES], &noisy[..TRAIN_SAMPLES])?;

    let dev_cases = make_cases(
        &clean[TRAIN_SAMPLES..dev_end],
        &noisy[TRAIN_SAMPLES..dev_end],
    )?;

    let selection = select_zero_divisor_train_dev(
        &train_cases,
        &dev_cases,
        TOP_K,
        RELATIVE_SCALE,
        DISTORTION_WEIGHT,
        1.0e-12,
    )?;

    let selected = &selection.selected;
    let decision = selection.decision;

    let soft = SoftCayleyFilter::new(selected.multiplier, RELATIVE_SCALE)
        .map_err(|error| error.to_string())?;

    let hard = CayleyProjector::new(selected.multiplier, RELATIVE_SCALE)
        .map_err(|error| error.to_string())?;

    let test_clean = &clean[dev_end..];
    let test_noisy = &noisy[dev_end..];

    let soft_output = if decision.uses_cayley()
    {
        TemporalBlockFilter::from_soft(&soft).apply(test_noisy)
    }
    else
    {
        test_noisy.to_vec()
    };

    let hard_output = if decision.uses_cayley()
    {
        TemporalBlockFilter::from_hard(&hard).apply(test_noisy)
    }
    else
    {
        test_noisy.to_vec()
    };

    let stft_output = stft_wiener_auto(test_noisy);
    let auto_result = denoise_auto(test_noisy, SAMPLE_RATE);
    let profile = classify(test_noisy, SAMPLE_RATE);

    println!(
        "fixture_samples={},train_samples={},dev_samples={},test_samples={}",
        clean.len(),
        TRAIN_SAMPLES,
        DEV_SAMPLES,
        test_clean.len(),
    );

    println!(
        "noise_type={:?},estimated_noise_std={},estimated_snr_db={}",
        profile.dominant, profile.noise_std, profile.snr_db,
    );

    println!(
        "selected_seed=e{}{}e{},train_rank={},train_loss={},dev_loss={},soft_effective_rejected={},hard_rejected={}",
        selected.seed_first_index,
        if selected.seed_second_sign > 0
        {
            "+"
        }
        else
        {
            "-"
        },
        selected.seed_second_index,
        selected.seed_rank,
        selected.train_score.loss,
        selected.dev_score.loss,
        soft.gains().iter().filter(|&&gain| gain <= 0.5).count(),
        hard.rejected_dimension(),
    );

    println!("development_gate={decision:?}");
    println!("auto_method={}", auto_result.method);
    println!("method,mse,snr_db,snr_improvement_db");

    print_metrics("noisy_input", metrics(test_clean, test_noisy, test_noisy));

    print_metrics(
        "gated_cayley_soft_temporal",
        metrics(test_clean, test_noisy, &soft_output),
    );

    print_metrics(
        "gated_cayley_hard_temporal",
        metrics(test_clean, test_noisy, &hard_output),
    );

    print_metrics(
        "stft_wiener_auto",
        metrics(test_clean, test_noisy, &stft_output),
    );

    print_metrics(
        "denoise_auto",
        metrics(test_clean, test_noisy, &auto_result.output),
    );

    Ok(())
}
