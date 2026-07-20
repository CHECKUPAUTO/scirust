use scirust_cayley_filter::{
    CayleyProjector, MultiplierCase, MultiplierScore, SPECTRAL_COMPLEX_BINS, Sedenion,
    SoftCayleyFilter, SpectralBlockFilter, development_gate,
    rank_zero_divisor_two_term_multipliers, score_multiplier, squared_norm,
};
use scirust_signal::Complex;
use scirust_signal::denoise::{classify, denoise_auto, stft_wiener_auto};
use scirust_signal::fft::fft;
use scirust_signal::windows::hanning;

const SAMPLE_RATE: f64 = 16_000.0;
const TRAIN_SAMPLES: usize = 4_096;
const DEV_SAMPLES: usize = 4_096;

const FRAME_LEN: usize = 256;
const HOP: usize = 64;

const TOP_K: usize = 16;
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;

const CASE_ENERGY_FLOOR: f64 = 1.0e-24;
const METRIC_ENERGY_FLOOR: f64 = 1.0e-30;

#[derive(Clone, Debug)]
struct SelectedSeed {
    train_rank: usize,
    first_index: usize,
    second_index: usize,
    second_sign: i8,
    multiplier: Sedenion,
    train_score: MultiplierScore,
    dev_score: MultiplierScore,
}

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
        return Err("invalid fixture dimensions".into());
    }

    Ok((clean, noisy))
}

fn mirror_index(index: isize, length: usize) -> usize {
    if length <= 1
    {
        return 0;
    }

    let period = 2 * (length - 1);
    let mut wrapped = index % period as isize;

    if wrapped < 0
    {
        wrapped += period as isize;
    }

    let wrapped = wrapped as usize;

    if wrapped < length
    {
        wrapped
    }
    else
    {
        period - wrapped
    }
}

fn spectral_cases(clean: &[f64], noisy: &[f64]) -> Result<Vec<MultiplierCase>, String> {
    if clean.len() != noisy.len() || clean.len() < 2
    {
        return Err("invalid clean/noisy split".into());
    }

    let frame_len = FRAME_LEN.max(4).next_power_of_two();
    let hop = HOP.clamp(1, frame_len / 2);
    let window = hanning(frame_len);

    let padded_len = clean.len() + 2 * frame_len;

    let clean_padded: Vec<f64> = (0..padded_len)
        .map(|index| {
            let source = mirror_index(index as isize - frame_len as isize, clean.len());

            clean[source]
        })
        .collect();

    let noise_padded: Vec<f64> = (0..padded_len)
        .map(|index| {
            let source = mirror_index(index as isize - frame_len as isize, clean.len());

            noisy[source] - clean[source]
        })
        .collect();

    let mut cases = Vec::new();
    let mut offset = 0;

    while offset + frame_len <= padded_len
    {
        let mut signal_spectrum: Vec<Complex> = (0..frame_len)
            .map(|index| Complex::new(window[index] * clean_padded[offset + index], 0.0))
            .collect();

        let mut noise_spectrum: Vec<Complex> = (0..frame_len)
            .map(|index| Complex::new(window[index] * noise_padded[offset + index], 0.0))
            .collect();

        fft(&mut signal_spectrum);
        fft(&mut noise_spectrum);

        let nyquist = frame_len / 2;
        let mut first_bin = 1;

        while first_bin + SPECTRAL_COMPLEX_BINS <= nyquist
        {
            let signal: Sedenion = core::array::from_fn(|coordinate| {
                let bin = first_bin + coordinate / 2;

                if coordinate.is_multiple_of(2)
                {
                    signal_spectrum[bin].re
                }
                else
                {
                    signal_spectrum[bin].im
                }
            });

            let noise: Sedenion = core::array::from_fn(|coordinate| {
                let bin = first_bin + coordinate / 2;

                if coordinate.is_multiple_of(2)
                {
                    noise_spectrum[bin].re
                }
                else
                {
                    noise_spectrum[bin].im
                }
            });

            if squared_norm(&signal) > CASE_ENERGY_FLOOR && squared_norm(&noise) > CASE_ENERGY_FLOOR
            {
                cases.push(MultiplierCase::new(signal, noise));
            }

            first_bin += SPECTRAL_COMPLEX_BINS;
        }

        offset += hop;
    }

    if cases.is_empty()
    {
        return Err("no usable spectral cases generated".into());
    }

    Ok(cases)
}

fn select_sparse_seed(
    train: &[MultiplierCase],
    dev: &[MultiplierCase],
) -> Result<SelectedSeed, String> {
    let ranked =
        rank_zero_divisor_two_term_multipliers(train, RELATIVE_SCALE, DISTORTION_WEIGHT, 1.0e-12)?;

    let mut best: Option<SelectedSeed> = None;

    for (train_rank, candidate) in ranked.into_iter().take(TOP_K).enumerate()
    {
        let dev_score = score_multiplier(
            dev,
            &candidate.multiplier,
            RELATIVE_SCALE,
            DISTORTION_WEIGHT,
        )?;

        let selected = SelectedSeed {
            train_rank,
            first_index: candidate.first_index,
            second_index: candidate.second_index,
            second_sign: candidate.second_sign,
            multiplier: candidate.multiplier,
            train_score: candidate.score,
            dev_score,
        };

        let replace = best.as_ref().is_none_or(|current| {
            selected.dev_score.loss < current.dev_score.loss
                || (selected.dev_score.loss == current.dev_score.loss
                    && selected.train_rank < current.train_rank)
        });

        if replace
        {
            best = Some(selected);
        }
    }

    best.ok_or_else(|| "no multiplier selected".into())
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

    Metrics {
        mse: output_error / clean.len() as f64,
        snr_db: 10.0 * (signal_energy / output_error.max(METRIC_ENERGY_FLOOR)).log10(),
        improvement_db: 10.0 * (input_error / output_error.max(METRIC_ENERGY_FLOOR)).log10(),
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
        return Err("fixture too short for split".into());
    }

    let train_cases = spectral_cases(&clean[..TRAIN_SAMPLES], &noisy[..TRAIN_SAMPLES])?;

    let dev_cases = spectral_cases(
        &clean[TRAIN_SAMPLES..dev_end],
        &noisy[TRAIN_SAMPLES..dev_end],
    )?;

    let selected = select_sparse_seed(&train_cases, &dev_cases)?;

    let decision = development_gate(&selected.dev_score)?;

    let soft = SoftCayleyFilter::new(selected.multiplier, RELATIVE_SCALE)
        .map_err(|error| error.to_string())?;

    let hard = CayleyProjector::new(selected.multiplier, RELATIVE_SCALE)
        .map_err(|error| error.to_string())?;

    let soft_filter = SpectralBlockFilter::from_soft(&soft, FRAME_LEN, HOP);

    let hard_filter = SpectralBlockFilter::from_hard(&hard, FRAME_LEN, HOP);

    let test_clean = &clean[dev_end..];
    let test_noisy = &noisy[dev_end..];

    let soft_output = if decision.uses_cayley()
    {
        soft_filter.apply(test_noisy)
    }
    else
    {
        test_noisy.to_vec()
    };

    let hard_output = if decision.uses_cayley()
    {
        hard_filter.apply(test_noisy)
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
        "frame_len={},hop={},groups_per_frame={},train_cases={},dev_cases={}",
        soft_filter.frame_len(),
        soft_filter.hop(),
        soft_filter.groups_per_frame(),
        train_cases.len(),
        dev_cases.len(),
    );

    println!(
        "noise_type={:?},estimated_noise_std={},estimated_snr_db={}",
        profile.dominant, profile.noise_std, profile.snr_db,
    );

    println!(
        "selected_seed=e{}{}e{},train_rank={},train_loss={},dev_loss={},soft_effective_rejected={},hard_rejected={}",
        selected.first_index,
        if selected.second_sign > 0 { "+" } else { "-" },
        selected.second_index,
        selected.train_rank,
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
        "gated_cayley_soft_spectral",
        metrics(test_clean, test_noisy, &soft_output),
    );

    print_metrics(
        "gated_cayley_hard_spectral",
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
