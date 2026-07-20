use scirust_cayley_filter::{
    CayleyProjector, MultiplierCase, Sedenion, SoftCayleyFilter, TemporalBlockFilter,
    select_zero_divisor_train_dev,
};
use scirust_signal::denoise::{ThresholdMode, classify, denoise_auto, wavelet_denoise};

const SAMPLE_RATE: f64 = 360.0;
const BLOCK_SIZE: usize = 16;

const TRAIN_SAMPLES: usize = 1_536;
const DEV_SAMPLES: usize = 1_024;

const TOP_K: usize = 16;
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;
const ANALYSIS_TOLERANCE: f64 = 1.0e-12;
const ENERGY_FLOOR: f64 = 1.0e-30;

#[derive(Clone, Copy, Debug)]
struct Metrics {
    mse: f64,
    snr_db: f64,
    improvement_db: f64,
}

type EcgFixture = (Vec<f64>, Vec<f64>, Vec<f64>);

fn load_fixture() -> Result<EcgFixture, String> {
    let raw = include_str!("../../scirust-signal/tests/data/ecg_mitbih.csv");

    let mut ecg = Vec::new();
    let mut muscle_artifact = Vec::new();
    let mut baseline_wander = Vec::new();

    for (line_index, line) in raw.lines().enumerate()
    {
        if line.starts_with('#') || line.trim().is_empty()
        {
            continue;
        }

        let mut fields = line.split(',');

        let parse = |value: Option<&str>, field: &str| -> Result<f64, String> {
            value
                .ok_or_else(|| format!("missing {field} at line {}", line_index + 1,))?
                .trim()
                .parse::<f64>()
                .map_err(|error| format!("invalid {field} at line {}: {error}", line_index + 1,))
        };

        ecg.push(parse(fields.next(), "ECG")?);
        muscle_artifact.push(parse(fields.next(), "MA")?);
        baseline_wander.push(parse(fields.next(), "BW")?);

        if fields.next().is_some()
        {
            return Err(format!("too many fields at line {}", line_index + 1,));
        }
    }

    if ecg.len() != 4_096
        || muscle_artifact.len() != ecg.len()
        || baseline_wander.len() != ecg.len()
    {
        return Err(format!(
            "unexpected fixture dimensions: ecg={}, ma={}, bw={}",
            ecg.len(),
            muscle_artifact.len(),
            baseline_wander.len(),
        ));
    }

    Ok((ecg, muscle_artifact, baseline_wander))
}

fn squared_energy(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
}

fn corrupt(clean: &[f64], noise: &[f64], target_snr_db: f64) -> Result<Vec<f64>, String> {
    if clean.len() != noise.len() || clean.is_empty()
    {
        return Err("invalid clean/noise dimensions".into());
    }

    let clean_energy = squared_energy(clean);
    let noise_energy = squared_energy(noise);

    if clean_energy <= 0.0 || noise_energy <= 0.0
    {
        return Err("clean and noise must have positive energy".into());
    }

    let amplitude = (clean_energy / (noise_energy * 10.0_f64.powf(target_snr_db / 10.0))).sqrt();

    Ok(clean
        .iter()
        .zip(noise)
        .map(|(signal, disturbance)| signal + amplitude * disturbance)
        .collect())
}

fn make_cases(clean: &[f64], noisy: &[f64]) -> Result<Vec<MultiplierCase>, String> {
    if clean.len() != noisy.len() || !clean.len().is_multiple_of(BLOCK_SIZE)
    {
        return Err("invalid block-aligned split".into());
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
    let signal_energy = squared_energy(clean);

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

    Metrics {
        mse: output_error / clean.len() as f64,
        snr_db: 10.0 * (signal_energy / output_error.max(ENERGY_FLOOR)).log10(),
        improvement_db: 10.0 * (input_error / output_error.max(ENERGY_FLOOR)).log10(),
    }
}

fn print_metrics(noise_name: &str, target_snr_db: f64, method: &str, metrics: Metrics) {
    println!(
        "{noise_name},{target_snr_db},{method},{},{},{}",
        metrics.mse, metrics.snr_db, metrics.improvement_db,
    );
}

fn run_scenario(
    clean: &[f64],
    noise: &[f64],
    noise_name: &str,
    target_snr_db: f64,
) -> Result<(), String> {
    let noisy = corrupt(clean, noise, target_snr_db)?;

    let dev_end = TRAIN_SAMPLES + DEV_SAMPLES;

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
        ANALYSIS_TOLERANCE,
    )?;

    let selected = &selection.selected;

    let soft = SoftCayleyFilter::new(selected.multiplier, RELATIVE_SCALE)
        .map_err(|error| error.to_string())?;

    let hard = CayleyProjector::new(selected.multiplier, RELATIVE_SCALE)
        .map_err(|error| error.to_string())?;

    let test_clean = &clean[dev_end..];
    let test_noisy = &noisy[dev_end..];

    let soft_output = if selection.decision.uses_cayley()
    {
        TemporalBlockFilter::from_soft(&soft).apply(test_noisy)
    }
    else
    {
        test_noisy.to_vec()
    };

    let hard_output = if selection.decision.uses_cayley()
    {
        TemporalBlockFilter::from_hard(&hard).apply(test_noisy)
    }
    else
    {
        test_noisy.to_vec()
    };

    let wavelet_output = wavelet_denoise(test_noisy, 0, ThresholdMode::Soft);

    let auto_result = denoise_auto(test_noisy, SAMPLE_RATE);

    let profile = classify(test_noisy, SAMPLE_RATE);

    println!(
        "# noise={noise_name},target_snr_db={target_snr_db},noise_type={:?},selected_seed=e{}{}e{},train_rank={},train_loss={},dev_loss={},decision={:?},rejected_dimension={}",
        profile.dominant,
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
        selection.decision,
        selected.dev_score.rejected_dimension,
    );

    println!("# auto_method={}", auto_result.method,);

    print_metrics(
        noise_name,
        target_snr_db,
        "noisy_input",
        metrics(test_clean, test_noisy, test_noisy),
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "gated_cayley_soft_temporal",
        metrics(test_clean, test_noisy, &soft_output),
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "gated_cayley_hard_temporal",
        metrics(test_clean, test_noisy, &hard_output),
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "wavelet_soft",
        metrics(test_clean, test_noisy, &wavelet_output),
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "denoise_auto",
        metrics(test_clean, test_noisy, &auto_result.output),
    );

    Ok(())
}

fn main() -> Result<(), String> {
    let (ecg, muscle_artifact, baseline_wander) = load_fixture()?;

    println!(
        "fixture_samples={},sample_rate={},train_samples={},dev_samples={},test_samples={}",
        ecg.len(),
        SAMPLE_RATE,
        TRAIN_SAMPLES,
        DEV_SAMPLES,
        ecg.len() - TRAIN_SAMPLES - DEV_SAMPLES,
    );

    println!("noise,target_snr_db,method,mse,snr_db,snr_improvement_db");

    for (noise_name, noise) in [
        ("muscle_artifact", &muscle_artifact),
        ("baseline_wander", &baseline_wander),
    ]
    {
        for target_snr_db in [6.0, 0.0]
        {
            run_scenario(&ecg, noise, noise_name, target_snr_db)?;
        }
    }

    Ok(())
}
