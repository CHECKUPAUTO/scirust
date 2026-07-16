//! Honest boundary-mapping of the denoise toolkit on **real noisy speech**
//! (VoiceBank+DEMAND: real speech + real recorded DEMAND environmental noise).
//!
//! Run with `cargo run --release -p scirust-signal --example denoise_real_speech`.
//! Deterministic: real recorded audio from the committed fixture, no RNG.
//!
//! ## Data provenance (CC-BY-4.0)
//!
//! `tests/data/voicebank_demand.csv` — one VoiceBank test utterance (`p232_022`,
//! 16 kHz) and its DEMAND-noised version (global SNR ≈ 7 dB), a 16384-sample voiced
//! excerpt. Source: <https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k>;
//! original corpus Valentini-Botinhao et al. (2016/2017), University of Edinburgh.

use scirust_signal::denoise::{ThresholdMode, denoise_auto, stft_wiener_auto, wavelet_denoise};

fn load_fixture() -> (Vec<f64>, Vec<f64>) {
    let raw = include_str!("../tests/data/voicebank_demand.csv");
    let (mut clean, mut noisy) = (Vec::new(), Vec::new());
    for line in raw.lines()
    {
        if line.starts_with('#') || line.is_empty()
        {
            continue;
        }
        let mut it = line.split(',');
        clean.push(it.next().unwrap().trim().parse().unwrap());
        noisy.push(it.next().unwrap().trim().parse().unwrap());
    }
    (clean, noisy)
}

fn snr_improvement(clean: &[f64], noisy: &[f64], est: &[f64]) -> f64 {
    let en: f64 = clean
        .iter()
        .zip(noisy)
        .map(|(&c, &n)| (c - n) * (c - n))
        .sum();
    let ee: f64 = clean
        .iter()
        .zip(est)
        .map(|(&c, &e)| (c - e) * (c - e))
        .sum();
    10.0 * (en / ee.max(1.0e-30)).log10()
}

fn main() {
    let (clean, noisy) = load_fixture();
    let fs = 16_000.0;
    let sig: f64 = clean.iter().map(|&c| c * c).sum();
    let err: f64 = clean
        .iter()
        .zip(&noisy)
        .map(|(&c, &n)| (c - n) * (c - n))
        .sum();
    let raw = 10.0 * (sig / err.max(1.0e-30)).log10();
    let auto = denoise_auto(&noisy, fs);
    println!(
        "# VoiceBank+DEMAND real noisy speech, 16 kHz, 16384-sample excerpt (raw {raw:.2} dB)\n"
    );
    println!(
        "classifier verdict: {:?};  denoise_auto method: {}\n",
        auto.profile.dominant, auto.method
    );
    println!("{:<28} {:>8}", "method", "Δ dB");
    let rows: [(&str, Vec<f64>); 3] = [
        ("denoise_auto", auto.output.clone()),
        ("stft_wiener_auto", stft_wiener_auto(&noisy)),
        (
            "wavelet(soft)",
            wavelet_denoise(&noisy, 0, ThresholdMode::Soft),
        ),
    ];
    for (name, out) in &rows
    {
        println!("{name:<28} {:>+8.2}", snr_improvement(&clean, &noisy, out));
    }
    println!(
        "\n# Finding (honest): the general-purpose toolkit is NOT a speech enhancer.\n\
         # * denoise_auto reads broadband-noisy speech as `Colored` and routes to aggressive\n\
         #   level-dependent wavelet shrinkage, which over-smooths non-stationary speech and\n\
         #   *degrades* the waveform (large negative Δ). For speech, call stft_wiener_auto\n\
         #   directly — it tracks a noise floor rather than assuming stationary colored noise,\n\
         #   and is far less destructive (pinned by tests/real_data_audio.rs).\n\
         # * Waveform SNR is itself a harsh proxy for speech quality; segmental SNR / PESQ / STOI\n\
         #   are the domain metrics. '≈ neutral on waveform SNR' is a floor, not the full story."
    );
}
