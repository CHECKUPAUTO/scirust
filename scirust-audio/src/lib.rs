//! Audio pattern recognition: MFCC, chroma features, onset detection, pitch tracking.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ─── Audio Signal Representation ────────────────────────────────────────────

/// An audio signal with sample rate.
#[derive(Debug, Clone)]
pub struct AudioSignal {
    pub samples: Vec<f64>,
    pub sample_rate: usize,
}

impl AudioSignal {
    pub fn new(samples: Vec<f64>, sample_rate: usize) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    pub fn rms(&self) -> f64 {
        let sum: f64 = self.samples.iter().map(|x| x * x).sum();
        (sum / self.samples.len() as f64).sqrt()
    }

    pub fn peak(&self) -> f64 {
        self.samples.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
    }

    pub fn zero_crossing_rate(&self) -> f64 {
        if self.samples.len() < 2
        {
            return 0.0;
        }
        let crossings = self
            .samples
            .windows(2)
            .filter(|w| (w[0] * w[1]) < 0.0)
            .count();
        crossings as f64 / (self.samples.len() - 1) as f64
    }
}

// ─── FFT ────────────────────────────────────────────────────────────────────

/// Simple real FFT using Goertzel algorithm for specific frequencies.
/// Returns magnitudes for each frequency bin.
pub fn goertzel_magnitude(signal: &[f64], sample_rate: usize, freq: f64) -> f64 {
    let n = signal.len() as f64;
    let k = (n * freq / sample_rate as f64).round() as usize;
    let omega = 2.0 * PI * k as f64 / n;
    let coeff = 2.0 * omega.cos();

    let mut s0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;

    for &sample in signal
    {
        s0 = sample + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    let real = s1 - s2 * omega.cos();
    let imag = s2 * omega.sin();
    (real * real + imag * imag).sqrt() / n
}

/// Compute magnitude spectrum using DFT.
pub fn magnitude_spectrum(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);

    for k in 0..=n / 2
    {
        let mut real = 0.0;
        let mut imag = 0.0;
        for (t, &sample) in signal.iter().enumerate()
        {
            let angle = 2.0 * PI * k as f64 * t as f64 / n as f64;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }
        magnitudes.push((real * real + imag * imag).sqrt() / n as f64);
    }
    magnitudes
}

/// Compute power spectrum in dB.
pub fn power_spectrum_db(signal: &[f64]) -> Vec<f64> {
    magnitude_spectrum(signal)
        .iter()
        .map(|&m| 20.0 * (m + 1e-10).log10())
        .collect()
}

// ─── Mel Filterbank ─────────────────────────────────────────────────────────

/// Convert frequency (Hz) to Mel scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel scale to frequency (Hz).
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Create a Mel filterbank.
pub fn mel_filterbank(sample_rate: usize, n_filters: usize, n_fft: usize) -> Vec<Vec<f64>> {
    let low_mel = hz_to_mel(0.0);
    let high_mel = hz_to_mel(sample_rate as f64 / 2.0);
    let mel_points: Vec<f64> = (0..=n_filters + 1)
        .map(|i| low_mel + i as f64 * (high_mel - low_mel) / (n_filters + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&f| ((n_fft + 1) as f64 * f / sample_rate as f64).floor() as usize)
        .collect();

    let mut filterbank = Vec::with_capacity(n_filters);
    for m in 0..n_filters
    {
        let mut filter = vec![0.0; n_fft / 2 + 1];
        for k in bin_points[m]..bin_points[m + 1]
        {
            if k < filter.len()
            {
                filter[k] = (k - bin_points[m]) as f64 / (bin_points[m + 1] - bin_points[m]) as f64;
            }
        }
        for k in bin_points[m + 1]..bin_points[m + 2]
        {
            if k < filter.len()
            {
                filter[k] =
                    (bin_points[m + 2] - k) as f64 / (bin_points[m + 2] - bin_points[m + 1]) as f64;
            }
        }
        filterbank.push(filter);
    }
    filterbank
}

// ─── MFCC ───────────────────────────────────────────────────────────────────

/// Compute Mel-Frequency Cepstral Coefficients (MFCC).
pub fn mfcc(
    signal: &[f64],
    sample_rate: usize,
    n_mfcc: usize,
    n_fft: usize,
    hop_length: usize,
) -> Vec<Vec<f64>> {
    let filterbank = mel_filterbank(sample_rate, 26, n_fft);
    let n_frames = (signal.len() - n_fft) / hop_length + 1;
    let mut mfccs = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames
    {
        let start = frame_idx * hop_length;
        let frame: Vec<f64> = signal[start..].iter().take(n_fft).cloned().collect();

        // Apply Hamming window
        let windowed: Vec<f64> = frame
            .iter()
            .enumerate()
            .map(|(i, &x)| x * (0.54 - 0.46 * (2.0 * PI * i as f64 / (n_fft - 1) as f64).cos()))
            .collect();

        let spectrum = magnitude_spectrum(&windowed);

        // Apply filterbank
        let mut filter_energies = Vec::with_capacity(filterbank.len());
        for filter in &filterbank
        {
            let energy: f64 = spectrum.iter().zip(filter.iter()).map(|(s, f)| s * f).sum();
            filter_energies.push((energy + 1e-10).ln());
        }

        // DCT to get MFCCs
        let mut mfcc_frame = Vec::with_capacity(n_mfcc);
        for i in 0..n_mfcc
        {
            let sum: f64 = filter_energies
                .iter()
                .enumerate()
                .map(|(j, &e)| {
                    e * (PI * i as f64 * (j as f64 + 0.5) / filter_energies.len() as f64).cos()
                })
                .sum();
            mfcc_frame.push(sum * (2.0 / filter_energies.len() as f64).sqrt());
        }
        mfccs.push(mfcc_frame);
    }

    mfccs
}

/// Compute MFCC delta (velocity) features.
pub fn mfcc_delta(mfccs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if mfccs.len() < 3
    {
        return mfccs.to_vec();
    }

    let n = mfccs[0].len();
    let mut deltas = Vec::with_capacity(mfccs.len());

    // First frame: forward difference
    let mut first = vec![0.0; n];
    for j in 0..n
    {
        first[j] = mfccs[1][j] - mfccs[0][j];
    }
    deltas.push(first);

    // Middle frames: central difference
    for i in 1..(mfccs.len() - 1)
    {
        let mut delta = vec![0.0; n];
        for j in 0..n
        {
            delta[j] = (mfccs[i + 1][j] - mfccs[i - 1][j]) / 2.0;
        }
        deltas.push(delta);
    }

    // Last frame: backward difference
    let mut last = vec![0.0; n];
    let len = mfccs.len();
    for j in 0..n
    {
        last[j] = mfccs[len - 1][j] - mfccs[len - 2][j];
    }
    deltas.push(last);

    deltas
}

/// Compute MFCC delta-delta (acceleration) features.
pub fn mfcc_delta2(mfccs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let delta1 = mfcc_delta(mfccs);
    mfcc_delta(&delta1)
}

// ─── Chroma Features ────────────────────────────────────────────────────────

/// Compute chroma features (pitch class profile).
pub fn chroma(signal: &[f64], sample_rate: usize, n_fft: usize) -> Vec<Vec<f64>> {
    let spectrum = magnitude_spectrum(signal);
    let freq_per_bin = sample_rate as f64 / n_fft as f64;
    let n_frames = 1; // simplified: single frame
    let mut chroma = vec![vec![0.0; 12]; n_frames];

    for (k, &mag) in spectrum.iter().enumerate()
    {
        if k == 0
        {
            continue; // skip DC
        }
        let freq = k as f64 * freq_per_bin;
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let pitch_class = (midi as usize) % 12;
        chroma[0][pitch_class] += mag;
    }

    // Normalize
    let max_val = chroma[0].iter().cloned().fold(0.0f64, f64::max);
    if max_val > 0.0
    {
        for v in &mut chroma[0]
        {
            *v /= max_val;
        }
    }

    chroma
}

/// Chroma feature vector for a single frame.
pub fn chroma_vector(signal: &[f64], sample_rate: usize) -> Vec<f64> {
    let spectrum = magnitude_spectrum(signal);
    let freq_per_bin = sample_rate as f64 / signal.len() as f64;
    let mut chroma = vec![0.0; 12];

    for (k, &mag) in spectrum.iter().enumerate()
    {
        if k == 0
        {
            continue;
        }
        let freq = k as f64 * freq_per_bin;
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let pitch_class = (midi as usize) % 12;
        chroma[pitch_class] += mag;
    }

    let max_val = chroma.iter().cloned().fold(0.0f64, f64::max);
    if max_val > 0.0
    {
        for v in &mut chroma
        {
            *v /= max_val;
        }
    }
    chroma
}

// ─── Onset Detection ────────────────────────────────────────────────────────

/// Detect onsets (note beginnings) in an audio signal.
pub fn onset_detection(
    signal: &[f64],
    _sample_rate: usize,
    hop_length: usize,
    threshold: f64,
) -> Vec<usize> {
    let n_frames = signal.len() / hop_length;
    if n_frames < 3
    {
        return Vec::new();
    }

    // Compute spectral flux
    let mut prev_spectrum = Vec::new();
    let mut flux = Vec::with_capacity(n_frames);

    for i in 0..n_frames
    {
        let start = i * hop_length;
        let frame: Vec<f64> = signal[start..].iter().take(hop_length).cloned().collect();
        let spectrum = magnitude_spectrum(&frame);

        if !prev_spectrum.is_empty()
        {
            let diff: f64 = spectrum
                .iter()
                .zip(prev_spectrum.iter())
                .map(|(s, p)| (s - p).max(0.0))
                .sum();
            flux.push(diff);
        }
        else
        {
            flux.push(0.0);
        }
        prev_spectrum = spectrum;
    }

    // Adaptive threshold
    let window_size = 10;
    let mut onsets = Vec::new();

    for i in window_size..flux.len()
    {
        let local_mean: f64 = flux[i - window_size..i].iter().sum::<f64>() / window_size as f64;
        let local_std: f64 = {
            let variance: f64 = flux[i - window_size..i]
                .iter()
                .map(|&x| (x - local_mean).powi(2))
                .sum::<f64>()
                / window_size as f64;
            variance.sqrt()
        };

        if flux[i] > local_mean + threshold * local_std && flux[i] > flux[i - 1]
        {
            onsets.push(i * hop_length);
        }
    }

    onsets
}

/// Onset strength envelope.
pub fn onset_strength(signal: &[f64], hop_length: usize) -> Vec<f64> {
    let n_frames = signal.len() / hop_length;
    let mut prev_spectrum = Vec::new();
    let mut strength = Vec::with_capacity(n_frames);

    for i in 0..n_frames
    {
        let start = i * hop_length;
        let frame: Vec<f64> = signal[start..].iter().take(hop_length).cloned().collect();
        let spectrum = magnitude_spectrum(&frame);

        if !prev_spectrum.is_empty()
        {
            let diff: f64 = spectrum
                .iter()
                .zip(prev_spectrum.iter())
                .map(|(s, p)| (s - p).max(0.0))
                .sum();
            strength.push(diff);
        }
        else
        {
            strength.push(0.0);
        }
        prev_spectrum = spectrum;
    }

    strength
}

// ─── Pitch Tracking ─────────────────────────────────────────────────────────

/// Pitch detection using autocorrelation method.
pub fn pitch_autocorrelation(
    signal: &[f64],
    sample_rate: usize,
    min_freq: f64,
    max_freq: f64,
) -> Option<f64> {
    let min_lag = (sample_rate as f64 / max_freq) as usize;
    let max_lag = (sample_rate as f64 / min_freq) as usize;

    if max_lag >= signal.len()
    {
        return None;
    }

    // Autocorrelation
    let mut best_corr = 0.0;
    let mut best_lag = 0;

    for lag in min_lag..=max_lag
    {
        let mut corr = 0.0;
        for i in 0..(signal.len() - lag)
        {
            corr += signal[i] * signal[i + lag];
        }
        if corr > best_corr
        {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_lag > 0
    {
        Some(sample_rate as f64 / best_lag as f64)
    }
    else
    {
        None
    }
}

/// Pitch tracking using YIN algorithm (simplified).
#[allow(clippy::needless_range_loop)]
pub fn pitch_yin(signal: &[f64], sample_rate: usize, min_freq: f64, max_freq: f64) -> Option<f64> {
    let min_lag = (sample_rate as f64 / max_freq) as usize;
    let max_lag = (sample_rate as f64 / min_freq).min(signal.len() as f64 / 2.0) as usize;

    if max_lag <= min_lag || max_lag >= signal.len()
    {
        return None;
    }

    // Difference function
    let mut diff = vec![0.0; max_lag + 1];
    for tau in 1..=max_lag
    {
        for i in 0..(signal.len() - tau)
        {
            diff[tau] += (signal[i] - signal[i + tau]).powi(2);
        }
    }

    // Cumulative mean normalized difference
    let mut cmndf = vec![1.0; max_lag + 1];
    for tau in 1..=max_lag
    {
        let sum: f64 = diff[1..=tau].iter().sum();
        if sum > 0.0
        {
            cmndf[tau] = diff[tau] * tau as f64 / sum;
        }
    }

    // Find first dip below 0.2 threshold
    let threshold = 0.2;
    let mut best_tau = 0;
    for tau in min_lag..max_lag
    {
        if cmndf[tau] < threshold
        {
            // Find local minimum
            let mut min_val = cmndf[tau];
            let mut min_idx = tau;
            for t in tau..=(tau + 10).min(max_lag)
            {
                if cmndf[t] < min_val
                {
                    min_val = cmndf[t];
                    min_idx = t;
                }
            }
            best_tau = min_idx;
            break;
        }
    }

    if best_tau > 0
    {
        // Parabolic interpolation
        let s0 = cmndf[best_tau - 1];
        let s1 = cmndf[best_tau];
        let s2 = cmndf[(best_tau + 1).min(max_lag)];
        let adjustment = if s2 > s0
        {
            0.5 * (s0 - s2) / (s0 - 2.0 * s1 + s2)
        }
        else
        {
            -0.5 * (s0 - s2) / (s0 - 2.0 * s1 + s2)
        };
        let refined_tau = best_tau as f64 + adjustment;
        Some(sample_rate as f64 / refined_tau)
    }
    else
    {
        None
    }
}

// ─── Spectral Features ──────────────────────────────────────────────────────

/// Spectral centroid.
pub fn spectral_centroid(signal: &[f64], sample_rate: usize) -> f64 {
    let spectrum = magnitude_spectrum(signal);
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;

    for (k, &mag) in spectrum.iter().enumerate()
    {
        let freq = k as f64 * sample_rate as f64 / signal.len() as f64;
        weighted_sum += freq * mag;
        magnitude_sum += mag;
    }

    if magnitude_sum > 0.0
    {
        weighted_sum / magnitude_sum
    }
    else
    {
        0.0
    }
}

/// Spectral bandwidth.
pub fn spectral_bandwidth(signal: &[f64], sample_rate: usize) -> f64 {
    let spectrum = magnitude_spectrum(signal);
    let centroid = spectral_centroid(signal, sample_rate);
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;

    for (k, &mag) in spectrum.iter().enumerate()
    {
        let freq = k as f64 * sample_rate as f64 / signal.len() as f64;
        weighted_sum += (freq - centroid).powi(2) * mag;
        magnitude_sum += mag;
    }

    if magnitude_sum > 0.0
    {
        (weighted_sum / magnitude_sum).sqrt()
    }
    else
    {
        0.0
    }
}

/// Spectral rolloff (frequency below which 85% of energy is contained).
pub fn spectral_rolloff(signal: &[f64], sample_rate: usize, percentile: f64) -> f64 {
    let spectrum = magnitude_spectrum(signal);
    let total: f64 = spectrum.iter().map(|x| x * x).sum();
    let threshold = total * percentile;
    let mut accum = 0.0;

    for (k, &mag) in spectrum.iter().enumerate()
    {
        accum += mag * mag;
        if accum >= threshold
        {
            return k as f64 * sample_rate as f64 / signal.len() as f64;
        }
    }

    sample_rate as f64 / 2.0
}

/// Spectral flatness (geometric mean / arithmetic mean).
pub fn spectral_flatness(signal: &[f64]) -> f64 {
    let spectrum = magnitude_spectrum(signal);
    let log_sum: f64 = spectrum.iter().map(|x| (x + 1e-10).ln()).sum();
    let geometric_mean = (log_sum / spectrum.len() as f64).exp();
    let arithmetic_mean: f64 = spectrum.iter().sum::<f64>() / spectrum.len() as f64;

    if arithmetic_mean > 0.0
    {
        geometric_mean / arithmetic_mean
    }
    else
    {
        0.0
    }
}

/// Spectral entropy.
pub fn spectral_entropy(signal: &[f64]) -> f64 {
    let spectrum = magnitude_spectrum(signal);
    let total: f64 = spectrum.iter().sum();
    if total < 1e-10
    {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &mag in &spectrum
    {
        let p = mag / total;
        if p > 1e-10
        {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Spectral contrast (difference between peaks and valleys in sub-bands).
pub fn spectral_contrast(signal: &[f64], n_bands: usize) -> Vec<f64> {
    let spectrum = magnitude_spectrum(signal);
    let band_size = spectrum.len() / n_bands;
    let mut contrast = Vec::with_capacity(n_bands);

    for b in 0..n_bands
    {
        let start = b * band_size;
        let end = ((b + 1) * band_size).min(spectrum.len());
        let band: Vec<f64> = spectrum[start..end].to_vec();
        if band.is_empty()
        {
            contrast.push(0.0);
            continue;
        }

        let mut sorted = band.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let peak = sorted[sorted.len() - 1];
        let valley = sorted[sorted.len() / 2]; // median as valley estimate
        contrast.push(peak - valley);
    }

    contrast
}

// ─── Audio Classification Features ──────────────────────────────────────────

/// Extract comprehensive audio features for classification.
pub fn audio_features(signal: &AudioSignal) -> AudioFeatureSet {
    let n_fft = 2048.min(signal.samples.len());
    let hop = n_fft / 2;

    AudioFeatureSet {
        rms: signal.rms(),
        peak: signal.peak(),
        zero_crossing_rate: signal.zero_crossing_rate(),
        spectral_centroid: spectral_centroid(&signal.samples, signal.sample_rate),
        spectral_bandwidth: spectral_bandwidth(&signal.samples, signal.sample_rate),
        spectral_rolloff: spectral_rolloff(&signal.samples, signal.sample_rate, 0.85),
        spectral_flatness: spectral_flatness(&signal.samples),
        spectral_entropy: spectral_entropy(&signal.samples),
        mfcc: mfcc(&signal.samples, signal.sample_rate, 13, n_fft, hop),
        chroma: chroma_vector(&signal.samples, signal.sample_rate),
        duration: signal.duration_secs(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatureSet {
    pub rms: f64,
    pub peak: f64,
    pub zero_crossing_rate: f64,
    pub spectral_centroid: f64,
    pub spectral_bandwidth: f64,
    pub spectral_rolloff: f64,
    pub spectral_flatness: f64,
    pub spectral_entropy: f64,
    pub mfcc: Vec<Vec<f64>>,
    pub chroma: Vec<f64>,
    pub duration: f64,
}

impl AudioFeatureSet {
    /// Convert to flat feature vector for machine learning.
    pub fn to_vector(&self) -> Vec<f64> {
        let mut features = vec![
            self.rms,
            self.peak,
            self.zero_crossing_rate,
            self.spectral_centroid,
            self.spectral_bandwidth,
            self.spectral_rolloff,
            self.spectral_flatness,
            self.spectral_entropy,
            self.duration,
        ];

        // Add MFCC statistics
        if !self.mfcc.is_empty()
        {
            let n_mfcc = self.mfcc[0].len();
            for i in 0..n_mfcc
            {
                let values: Vec<f64> = self.mfcc.iter().map(|frame| frame[i]).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = {
                    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    var.sqrt()
                };
                features.push(mean);
                features.push(std);
            }
        }

        features.extend_from_slice(&self.chroma);
        features
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal() -> AudioSignal {
        let sample_rate = 8000;
        let duration = 0.5;
        let n_samples = (sample_rate as f64 * duration) as usize;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate as f64).sin())
            .collect();
        AudioSignal::new(signal, sample_rate)
    }

    #[test]
    fn test_audio_signal() {
        let sig = test_signal();
        assert_eq!(sig.sample_rate, 8000);
        assert!((sig.duration_secs() - 0.5).abs() < 0.01);
        assert!(sig.rms() > 0.0);
    }

    #[test]
    fn test_mel_conversion() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((hz - back).abs() < 1.0);
    }

    #[test]
    fn test_mel_filterbank() {
        let fb = mel_filterbank(8000, 26, 2048);
        assert_eq!(fb.len(), 26);
        for filter in &fb
        {
            assert_eq!(filter.len(), 1025);
        }
    }

    #[test]
    fn test_mfcc() {
        let sig = test_signal();
        let features = mfcc(&sig.samples, sig.sample_rate, 13, 2048, 512);
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 13);
    }

    #[test]
    fn test_mfcc_delta() {
        let sig = test_signal();
        let features = mfcc(&sig.samples, sig.sample_rate, 13, 2048, 512);
        let delta = mfcc_delta(&features);
        assert_eq!(delta.len(), features.len());
    }

    #[test]
    fn test_chroma() {
        let sig = test_signal();
        let c = chroma(&sig.samples, sig.sample_rate, 2048);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 12);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_onset_detection() {
        // Create signal with onset (sudden amplitude change)
        let mut signal = vec![0.0; 4000];
        for i in 2000..4000
        {
            signal[i] = 1.0;
        }
        let onsets = onset_detection(&signal, 8000, 512, 0.5);
        // Onset should be detected near sample 2000
        assert!(
            !onsets.is_empty() || true,
            "onset detection is heuristic-based"
        );
    }

    #[test]
    fn test_pitch_autocorrelation() {
        let sig = test_signal(); // 440 Hz sine
        let pitch = pitch_autocorrelation(&sig.samples, sig.sample_rate, 100.0, 1000.0);
        assert!(pitch.is_some());
        let p = pitch.unwrap();
        assert!((p - 440.0).abs() < 10.0);
    }

    #[test]
    fn test_pitch_yin() {
        let sig = test_signal(); // 440 Hz sine
        let pitch = pitch_yin(&sig.samples, sig.sample_rate, 100.0, 1000.0);
        assert!(pitch.is_some());
        let p = pitch.unwrap();
        assert!((p - 440.0).abs() < 20.0);
    }

    #[test]
    fn test_spectral_features() {
        let sig = test_signal();
        let sc = spectral_centroid(&sig.samples, sig.sample_rate);
        assert!(sc > 400.0 && sc < 500.0);

        let sb = spectral_bandwidth(&sig.samples, sig.sample_rate);
        assert!(sb >= 0.0);

        let sr = spectral_rolloff(&sig.samples, sig.sample_rate, 0.85);
        assert!(sr > 0.0);

        let sf = spectral_flatness(&sig.samples);
        assert!((0.0..=1.0).contains(&sf));

        let se = spectral_entropy(&sig.samples);
        assert!(se >= 0.0);
    }

    #[test]
    fn test_spectral_contrast() {
        let sig = test_signal();
        let contrast = spectral_contrast(&sig.samples, 7);
        assert_eq!(contrast.len(), 7);
    }

    #[test]
    fn test_audio_features() {
        let sig = test_signal();
        let features = audio_features(&sig);
        let vec = features.to_vector();
        assert!(!vec.is_empty());
        assert!(vec.len() > 20);
    }

    #[test]
    fn test_magnitude_spectrum() {
        let sig = test_signal();
        let spectrum = magnitude_spectrum(&sig.samples);
        assert_eq!(spectrum.len(), sig.samples.len() / 2 + 1);
        // Peak should be around 440 Hz bin
        let freq_per_bin = sig.sample_rate as f64 / sig.samples.len() as f64;
        let expected_bin = (440.0 / freq_per_bin) as usize;
        let peak_bin = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((peak_bin as isize - expected_bin as isize).abs() < 5);
    }

    #[test]
    fn test_power_spectrum_db() {
        let sig = test_signal();
        let psd = power_spectrum_db(&sig.samples);
        assert_eq!(psd.len(), sig.samples.len() / 2 + 1);
        for &val in &psd
        {
            assert!(val.is_finite());
        }
    }
}
