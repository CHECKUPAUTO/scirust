//! Real-data validation on a third domain: **noisy speech** (VoiceBank+DEMAND) —
//! real speech corrupted by *real recorded* environmental noise (DEMAND) at a known
//! SNR. This maps a **boundary** of the general-purpose toolkit honestly: it is not a
//! speech-specialized enhancer, and the finding is a usage guide, not a triumph.
//!
//! Data (CC-BY-4.0), committed as `tests/data/voicebank_demand.csv`:
//! * `clean` — a VoiceBank test utterance (`p232_022`).
//! * `noisy` — the same utterance with real DEMAND noise added (global SNR ≈ 7 dB).
//! * Source: <https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k>.
//!   Original corpus: Valentini-Botinhao et al. (2016/2017), University of Edinburgh.
//!   16 kHz, a 16384-sample voiced excerpt.
//!
//! Deterministic: real recorded audio from the fixture, no RNG.
//!
//! ## The honest finding (see also `examples/denoise_real_speech.rs`)
//!
//! On the waveform-SNR metric, the general toolkit does **not** enhance real speech.
//! `denoise_auto` classifies broadband-noisy speech as `Colored` and routes to
//! aggressive level-dependent wavelet shrinkage, which over-smooths the non-stationary
//! speech and *degrades* the waveform (a large negative ΔSNR). The lesson is concrete
//! and useful: for speech, call the short-time Wiener ([`stft_wiener_auto`]) directly
//! — it assumes a tracked noise floor rather than stationary colored noise, and it is
//! far less destructive than the auto-router's speech-inappropriate verdict. (Waveform
//! SNR is itself a harsh proxy for speech quality — segmental SNR / PESQ / STOI are the
//! domain metrics — so "≈ neutral on waveform SNR" is a *floor*, not the full picture.)

use scirust_signal::denoise::{ThresholdMode, denoise_auto, stft_wiener_auto, wavelet_denoise};

fn load_fixture() -> (Vec<f64>, Vec<f64>) {
    let raw = include_str!("data/voicebank_demand.csv");
    let (mut clean, mut noisy) = (Vec::new(), Vec::new());
    for line in raw.lines()
    {
        if line.starts_with('#') || line.is_empty()
        {
            continue;
        }
        let mut it = line.split(',');
        clean.push(it.next().unwrap().trim().parse::<f64>().unwrap());
        noisy.push(it.next().unwrap().trim().parse::<f64>().unwrap());
    }
    (clean, noisy)
}

/// Waveform SNR-improvement of `est` over `noisy`, both against `clean`.
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

#[test]
fn stft_wiener_is_the_right_tool_for_speech_not_the_auto_router() {
    // The concrete, useful finding: on real noisy speech the short-time Wiener is far
    // less destructive than denoise_auto's speech-inappropriate `Colored` routing
    // (aggressive wavelet shrinkage that over-smooths non-stationary speech). This
    // guards the usage guidance "for speech, use stft_wiener_auto directly".
    let (clean, noisy) = load_fixture();
    let stft = snr_improvement(&clean, &noisy, &stft_wiener_auto(&noisy));
    let auto = snr_improvement(&clean, &noisy, &denoise_auto(&noisy, 16_000.0).output);
    assert!(
        stft > auto + 2.0,
        "stft_wiener ({stft:.2} dB) should be far less destructive than denoise_auto ({auto:.2} dB) on speech"
    );
    // And the specialized method does not catastrophically degrade the speech.
    assert!(
        stft > -2.0,
        "stft_wiener should not wreck real speech (Δ {stft:.2} dB)"
    );
}

#[test]
fn denoisers_run_on_real_speech_without_crashing() {
    // Length-preserving, finite output on real 16 kHz speech for the toolkit's methods.
    let (_, noisy) = load_fixture();
    for out in [
        denoise_auto(&noisy, 16_000.0).output,
        stft_wiener_auto(&noisy),
        wavelet_denoise(&noisy, 0, ThresholdMode::Soft),
    ]
    {
        assert_eq!(out.len(), noisy.len());
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn fixture_is_well_formed_noisy_speech() {
    let (clean, noisy) = load_fixture();
    assert_eq!(clean.len(), 16_384);
    assert_eq!(noisy.len(), 16_384);
    assert!(clean.iter().chain(&noisy).all(|v| v.is_finite()));
    // The committed excerpt really is noisy (the noise is audible energy, not a
    // rounding artifact): the raw SNR is finite and modest, not near-infinite.
    let sig: f64 = clean.iter().map(|&c| c * c).sum();
    let err: f64 = clean
        .iter()
        .zip(&noisy)
        .map(|(&c, &n)| (c - n) * (c - n))
        .sum();
    let raw = 10.0 * (sig / err.max(1.0e-30)).log10();
    assert!(
        (0.0..20.0).contains(&raw),
        "excerpt should carry real, modest noise (raw SNR {raw:.1} dB)"
    );
}
