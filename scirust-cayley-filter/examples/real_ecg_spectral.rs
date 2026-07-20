use scirust_cayley_filter::{
    CayleyProjector, MultiplierCase, SPECTRAL_COMPLEX_BINS, Sedenion, SoftCayleyFilter,
    SpectralBlockFilter, select_zero_divisor_train_dev, squared_norm,
};
use scirust_signal::Complex;
use scirust_signal::denoise::{
    ThresholdMode, classify, denoise_auto, stft_wiener_auto, wavelet_denoise,
};
use scirust_signal::fft::fft;
use scirust_signal::windows::hanning;

const SAMPLE_RATE: f64 = 360.0;

const TRAIN_SAMPLES: usize = 1_536;
const DEV_SAMPLES: usize = 1_024;

const FRAME_LEN: usize = 256;
const HOP: usize = 64;

const TOP_K: usize = 16;
const RELATIVE_SCALE: f64 = 5.0e-2;
const DISTORTION_WEIGHT: f64 = 10.0;
const ANALYSIS_TOLERANCE: f64 = 1.0e-12;

const CASE_ENERGY_FLOOR: f64 = 1.0e-24;
const METRIC_ENERGY_FLOOR: f64 = 1.0e-30;

type EcgFixture = (Vec<f64>, Vec<f64>, Vec<f64>);

#[derive(Clone, Copy, Debug)]
struct Metrics {
    mse: f64,
    snr_db: f64,
    improvement_db: f64,
}

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
        snr_db: 10.0 * (signal_energy / output_error.max(METRIC_ENERGY_FLOOR)).log10(),
        improvement_db: 10.0 * (input_error / output_error.max(METRIC_ENERGY_FLOOR)).log10(),
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

    let train_cases = spectral_cases(&clean[..TRAIN_SAMPLES], &noisy[..TRAIN_SAMPLES])?;

    let dev_cases = spectral_cases(
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

    let soft_filter = SpectralBlockFilter::from_soft(&soft, FRAME_LEN, HOP);

    let hard_filter = SpectralBlockFilter::from_hard(&hard, FRAME_LEN, HOP);

    let test_clean = &clean[dev_end..];
    let test_noisy = &noisy[dev_end..];

    let soft_output = if selection.decision.uses_cayley()
    {
        soft_filter.apply(test_noisy)
    }
    else
    {
        test_noisy.to_vec()
    };

    let hard_output = if selection.decision.uses_cayley()
    {
        hard_filter.apply(test_noisy)
    }
    else
    {
        test_noisy.to_vec()
    };

    let stft_output = stft_wiener_auto(test_noisy);

    let wavelet_output = wavelet_denoise(test_noisy, 0, ThresholdMode::Soft);

    let auto_result = denoise_auto(test_noisy, SAMPLE_RATE);

    let profile = classify(test_noisy, SAMPLE_RATE);

    println!(
        "# noise={noise_name},target_snr_db={target_snr_db},noise_type={:?},selected_seed=e{}{}e{},train_rank={},train_loss={},dev_loss={},decision={:?},soft_effective_rejected={},hard_rejected={}",
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
        soft.gains().iter().filter(|&&gain| gain <= 0.5).count(),
        hard.rejected_dimension(),
    );

    println!(
        "# frame_len={},hop={},groups_per_frame={},train_cases={},dev_cases={},auto_method={}",
        soft_filter.frame_len(),
        soft_filter.hop(),
        soft_filter.groups_per_frame(),
        train_cases.len(),
        dev_cases.len(),
        auto_result.method,
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "noisy_input",
        metrics(test_clean, test_noisy, test_noisy),
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "gated_cayley_soft_spectral",
        metrics(test_clean, test_noisy, &soft_output),
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "gated_cayley_hard_spectral",
        metrics(test_clean, test_noisy, &hard_output),
    );

    print_metrics(
        noise_name,
        target_snr_db,
        "stft_wiener_auto",
        metrics(test_clean, test_noisy, &stft_output),
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
