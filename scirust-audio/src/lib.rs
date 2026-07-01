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
    // No full frame fits (or a degenerate hop) -> no coefficients. Guard the
    // subtraction so it cannot underflow `usize` into an enormous frame count.
    if hop_length == 0 || signal.len() < n_fft
    {
        return Vec::new();
    }

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
pub fn chroma(signal: &[f64], sample_rate: usize, _n_fft: usize) -> Vec<Vec<f64>> {
    let spectrum = magnitude_spectrum(signal);
    // `magnitude_spectrum` runs a length-`signal.len()` DFT, so bin `k` sits at
    // `k * sample_rate / signal.len()` Hz. Deriving the bin spacing from the
    // (unused) `n_fft` argument would misplace every bin — and thus every pitch
    // class — whenever `signal.len() != n_fft`. Use the actual signal length,
    // matching `chroma_vector`.
    let freq_per_bin = sample_rate as f64 / signal.len() as f64;
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
        // Quantize to the NEAREST semitone (round, not truncate), and use
        // Euclidean remainder so negative MIDI values (sub-audible bins) wrap
        // correctly into [0, 12) instead of saturating to 0.
        let pitch_class = (midi.round() as i64).rem_euclid(12) as usize;
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
/// Compute Short-Time Fourier Transform (STFT) magnitude spectrogram.
pub fn stft(signal: &[f64], n_fft: usize, hop_length: usize) -> Vec<Vec<f64>> {
    if hop_length == 0 || signal.len() < n_fft
    {
        return Vec::new();
    }

    let n_frames = (signal.len() - n_fft) / hop_length + 1;
    let mut spectrogram = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames
    {
        let start = frame_idx * hop_length;
        let frame: Vec<f64> = signal[start..start + n_fft].to_vec();

        // Apply Hamming window
        let windowed: Vec<f64> = frame
            .iter()
            .enumerate()
            .map(|(i, &x)| x * (0.54 - 0.46 * (2.0 * PI * i as f64 / (n_fft - 1) as f64).cos()))
            .collect();

        spectrogram.push(magnitude_spectrum(&windowed));
    }

    spectrogram
}

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
        // Quantize to the NEAREST semitone (round, not truncate); rem_euclid
        // wraps negative MIDI (sub-audible bins) correctly into [0, 12).
        let pitch_class = (midi.round() as i64).rem_euclid(12) as usize;
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
        // Parabolic interpolation around the local minimum at `best_tau`.
        // Vertex offset of the parabola through (s0, s1, s2) is
        //     delta = 0.5 * (s0 - s2) / (s0 - 2*s1 + s2),
        // which already carries the correct sign for either side of the minimum;
        // no conditional sign flip is needed (a flip biases the estimate).
        let s0 = cmndf[best_tau - 1];
        let s1 = cmndf[best_tau];
        let s2 = cmndf[(best_tau + 1).min(max_lag)];
        let denom = s0 - 2.0 * s1 + s2;
        let adjustment = if denom.abs() > 1e-12
        {
            0.5 * (s0 - s2) / denom
        }
        else
        {
            0.0
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

    /// A full-scale 440 Hz sine, 0.5 s at 8 kHz (4000 samples).
    fn test_signal() -> AudioSignal {
        let sample_rate = 8000;
        let duration = 0.5;
        let n_samples = (sample_rate as f64 * duration) as usize;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate as f64).sin())
            .collect();
        AudioSignal::new(signal, sample_rate)
    }

    /// A real cosine at exactly DFT bin `k0` (period divides `n`).
    fn cosine_at_bin(n: usize, k0: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * k0 as f64 * i as f64 / n as f64).cos())
            .collect()
    }

    // ── AudioSignal scalar statistics ──────────────────────────────────────

    #[test]
    fn test_audio_signal_stats() {
        let sig = test_signal();
        assert_eq!(sig.sample_rate, 8000);
        assert!((sig.duration_secs() - 0.5).abs() < 1e-9);
        // RMS of a unit-amplitude sine is 1/sqrt(2).
        assert!((sig.rms() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        // Peak of a 440 Hz sine over 4000 samples reaches essentially 1.0.
        assert!((sig.peak() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_rms_and_peak_constants() {
        // Constant 3.0 signal: RMS == |value|, peak == |value|.
        let sig = AudioSignal::new(vec![3.0; 100], 1000);
        assert!((sig.rms() - 3.0).abs() < 1e-12);
        assert!((sig.peak() - 3.0).abs() < 1e-12);
        // Mixed-sign peak picks the largest magnitude.
        let sig2 = AudioSignal::new(vec![0.5, -2.0, 1.0], 1000);
        assert!((sig2.peak() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_crossing_rate() {
        // Alternating +1/-1: every adjacent pair crosses zero.
        // 6 samples -> 5 pairs -> 5 crossings -> rate 5/5 = 1.0.
        let sig = AudioSignal::new(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0], 1000);
        assert!((sig.zero_crossing_rate() - 1.0).abs() < 1e-12);

        // All-positive: no sign changes -> rate 0.
        let flat = AudioSignal::new(vec![0.2, 0.4, 0.6, 0.8], 1000);
        assert_eq!(flat.zero_crossing_rate(), 0.0);

        // Too short to have a pair.
        let tiny = AudioSignal::new(vec![1.0], 1000);
        assert_eq!(tiny.zero_crossing_rate(), 0.0);

        // 440 Hz sine over 0.5 s makes ~2*440*0.5 = 440 crossings out of 3999 pairs.
        let sine = test_signal();
        let zcr = sine.zero_crossing_rate();
        assert!((zcr - 440.0 / 3999.0).abs() < 5e-3);
    }

    // ── DFT / spectra ──────────────────────────────────────────────────────

    #[test]
    fn test_magnitude_spectrum_cosine_oracle() {
        // Cosine at exact bin k0 -> magnitude/N == 0.5 there, ~0 elsewhere.
        let n = 64;
        let k0 = 8;
        let spectrum = magnitude_spectrum(&cosine_at_bin(n, k0));
        assert_eq!(spectrum.len(), n / 2 + 1);
        assert!((spectrum[k0] - 0.5).abs() < 1e-9);
        for (k, &m) in spectrum.iter().enumerate()
        {
            if k != k0
            {
                assert!(m < 1e-9, "bin {k} should be ~0, got {m}");
            }
        }
    }

    #[test]
    fn test_magnitude_spectrum_impulse_is_flat() {
        // A unit impulse has a perfectly flat magnitude spectrum at 1/N.
        let n = 16;
        let mut sig = vec![0.0; n];
        sig[0] = 1.0;
        let spectrum = magnitude_spectrum(&sig);
        for &m in &spectrum
        {
            assert!((m - 1.0 / n as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn test_magnitude_spectrum_dc() {
        // All-ones (DC): only bin 0 is non-zero and equals 1.0; rest ~0.
        let n = 32;
        let spectrum = magnitude_spectrum(&vec![1.0; n]);
        assert!((spectrum[0] - 1.0).abs() < 1e-12);
        for &m in &spectrum[1..]
        {
            assert!(m < 1e-12);
        }
    }

    #[test]
    fn test_power_spectrum_db_impulse() {
        // Impulse spectrum is 1/16 in every bin -> 20*log10(1/16) ≈ -24.08 dB.
        let n = 16;
        let mut sig = vec![0.0; n];
        sig[0] = 1.0;
        let psd = power_spectrum_db(&sig);
        assert_eq!(psd.len(), n / 2 + 1);
        let expected = 20.0 * (1.0 / 16.0_f64).log10();
        for &v in &psd
        {
            assert!(v.is_finite());
            assert!((v - expected).abs() < 1e-4, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_magnitude_spectrum_sine_peak_bin() {
        let sig = test_signal();
        let spectrum = magnitude_spectrum(&sig.samples);
        assert_eq!(spectrum.len(), sig.samples.len() / 2 + 1);
        // 440 Hz at 8 kHz over 4000 samples -> bin 440/(8000/4000) = 220.
        let freq_per_bin = sig.sample_rate as f64 / sig.samples.len() as f64;
        let expected_bin = (440.0 / freq_per_bin).round() as usize;
        assert_eq!(expected_bin, 220);
        let peak_bin = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!((peak_bin as isize - expected_bin as isize).abs() <= 1);
    }

    // ── Goertzel ───────────────────────────────────────────────────────────

    #[test]
    fn test_goertzel_on_and_off_bin() {
        // Sine at exact bin -> magnitude ~0.5 (matching DFT normalization);
        // a far-off frequency -> ~0.
        let n = 800;
        let sr = 8000;
        let k = 44;
        let bin_freq = k as f64 * sr as f64 / n as f64; // 440 Hz
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * k as f64 * i as f64 / n as f64).sin())
            .collect();
        let on = goertzel_magnitude(&sig, sr, bin_freq);
        assert!((on - 0.5).abs() < 1e-6, "on-bin magnitude {on}");

        let off_freq = 100.0 * sr as f64 / n as f64; // bin 100, far from 44
        let off = goertzel_magnitude(&sig, sr, off_freq);
        assert!(off < 1e-6, "off-bin magnitude {off}");
    }

    #[test]
    fn test_goertzel_matches_dft_bin() {
        // Goertzel magnitude must equal the DFT magnitude at the same bin.
        let n = 256;
        let sr = 4000;
        let sig = cosine_at_bin(n, 17);
        let bin_freq = 17.0 * sr as f64 / n as f64;
        let g = goertzel_magnitude(&sig, sr, bin_freq);
        let dft = magnitude_spectrum(&sig)[17];
        assert!((g - dft).abs() < 1e-9, "goertzel {g} vs dft {dft}");
    }

    // ── Mel / MFCC ─────────────────────────────────────────────────────────

    #[test]
    fn test_mel_conversion_roundtrip() {
        for &hz in &[100.0, 440.0, 1000.0, 4000.0]
        {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((hz - back).abs() < 1e-6, "{hz} -> {back}");
        }
        // Anchor value: mel(700 Hz) = 2595*log10(2).
        assert!((hz_to_mel(700.0) - 2595.0 * 2.0_f64.log10()).abs() < 1e-9);
        // 0 Hz maps to 0 mel.
        assert_eq!(hz_to_mel(0.0), 0.0);
    }

    #[test]
    fn test_mel_filterbank_shape_and_peaks() {
        let fb = mel_filterbank(8000, 26, 2048);
        assert_eq!(fb.len(), 26);
        for filter in &fb
        {
            assert_eq!(filter.len(), 1025);
            // Triangular filters are normalized to peak at exactly 1.0,
            // and all weights are non-negative.
            let peak = filter.iter().cloned().fold(0.0_f64, f64::max);
            assert!((peak - 1.0).abs() < 1e-12, "filter peak {peak}");
            assert!(filter.iter().all(|&w| w >= 0.0));
        }
    }

    #[test]
    fn test_mfcc_shape() {
        let sig = test_signal();
        let features = mfcc(&sig.samples, sig.sample_rate, 13, 2048, 512);
        // (4000 - 2048)/512 + 1 = 4 frames, 13 coefficients each.
        assert_eq!(features.len(), 4);
        assert!(features.iter().all(|f| f.len() == 13));
    }

    #[test]
    fn test_mfcc_short_signal_is_empty() {
        // Signal shorter than n_fft must not panic; it yields no frames.
        let sig = vec![0.1; 100];
        assert!(mfcc(&sig, 8000, 13, 2048, 512).is_empty());
        // A degenerate hop of 0 is also handled gracefully.
        assert!(mfcc(&vec![0.1; 4096], 8000, 13, 2048, 0).is_empty());
    }

    #[test]
    fn test_mfcc_of_silence() {
        // Silence -> every Mel energy is ln(1e-10). The DCT then gives
        // c0 = sqrt(2/26) * 26 * ln(1e-10) and c1..c12 == 0.
        let silence = vec![0.0; 4096];
        let mfccs = mfcc(&silence, 8000, 13, 2048, 1024);
        assert!(!mfccs.is_empty());
        let n_filters = 26.0_f64;
        let c0_expected = (2.0 / n_filters).sqrt() * n_filters * 1e-10_f64.ln();
        for frame in &mfccs
        {
            assert!((frame[0] - c0_expected).abs() < 1e-6, "c0 {}", frame[0]);
            for (i, &c) in frame.iter().enumerate().skip(1)
            {
                assert!(c.abs() < 1e-6, "c{i} should be 0, got {c}");
            }
        }
    }

    #[test]
    fn test_mfcc_delta_central_difference() {
        // Linear ramp per coefficient: frame i = [i, 2*i].
        // Central difference of the middle frames is exactly [1, 2].
        let ramp: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, 2.0 * i as f64]).collect();
        let delta = mfcc_delta(&ramp);
        assert_eq!(delta.len(), 5);
        for d in &delta[1..4]
        {
            assert!((d[0] - 1.0).abs() < 1e-12);
            assert!((d[1] - 2.0).abs() < 1e-12);
        }
        // Boundaries use one-sided differences with the same slope here.
        assert!((delta[0][0] - 1.0).abs() < 1e-12);
        assert!((delta[4][1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_mfcc_delta_of_constant_is_zero() {
        // Identical frames -> zero velocity everywhere.
        let frames = vec![vec![1.0, 2.0, 3.0]; 6];
        let delta = mfcc_delta(&frames);
        assert_eq!(delta.len(), 6);
        for d in &delta
        {
            assert!(d.iter().all(|&x| x.abs() < 1e-12));
        }
        // Acceleration of a constant signal is also zero.
        let delta2 = mfcc_delta2(&frames);
        for d in &delta2
        {
            assert!(d.iter().all(|&x| x.abs() < 1e-12));
        }
    }

    // ── Chroma ─────────────────────────────────────────────────────────────

    #[test]
    fn test_chroma_shape() {
        let sig = test_signal();
        let c = chroma(&sig.samples, sig.sample_rate, 2048);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 12);
    }

    #[test]
    fn test_chroma_identifies_a440_when_len_ne_nfft() {
        // Regression: `chroma` must derive bin frequencies from the signal
        // length (what `magnitude_spectrum` actually uses), not from `n_fft`.
        // 3000 samples at 8 kHz -> 8000/3000 Hz/bin, and 440 Hz lands exactly on
        // bin 165, so the correct spacing reads 440 Hz -> pitch class 9 (A).
        // The old `sample_rate / n_fft` spacing (8000/2048 ≈ 3.9 Hz/bin) read
        // bin 165 as ~644.5 Hz -> pitch class 4 (E). 3000/2048 is not near a
        // power of two, so the error is not masked by octave equivalence.
        let sample_rate = 8000;
        let n = 3000;
        let n_fft = 2048; // deliberately != n
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate as f64).sin())
            .collect();
        let c = chroma(&sig, sample_rate, n_fft);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 12);
        let (argmax, _) = c[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(
            argmax, 9,
            "A4 should map to pitch class 9 (A), got {argmax}"
        );
    }

    #[test]
    fn test_chroma_vector_identifies_a440() {
        // Pure 440 Hz tone -> dominant pitch class is A (index 9),
        // and the profile is peak-normalized so that maximum == 1.0.
        let sig = test_signal();
        let c = chroma_vector(&sig.samples, sig.sample_rate);
        assert_eq!(c.len(), 12);
        let (argmax, &maxv) = c
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(argmax, 9, "A4 should map to pitch class 9, got {argmax}");
        assert!((maxv - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_chroma_pitch_class_rounding() {
        // Build a tone at C5 = 523.25 Hz so its bin lands on MIDI 72 -> pitch
        // class 0 (C). Truncation (the old bug) would push the lower half of
        // the semitone band to class 11 (B); rounding keeps it at C.
        let sr = 8000;
        let n = 8000; // 1 Hz/bin so the tone sits near an exact bin
        let freq = 523.25_f64;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr as f64).sin())
            .collect();
        let c = chroma_vector(&sig, sr);
        let argmax = c
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            argmax, 0,
            "C5 should map to pitch class 0 (C), got {argmax}"
        );
    }

    // ── Onset detection ────────────────────────────────────────────────────

    #[test]
    fn test_onset_detection_finds_burst() {
        // 8192 samples, hop 128 -> 64 frames (enough to pass the 10-frame
        // adaptive-threshold warm-up). A tone burst that starts at sample 4096
        // produces a spectral-flux spike, so an onset is reported exactly there.
        let n = 8192;
        let hop = 128;
        let mut sig = vec![0.0; n];
        for (i, s) in sig.iter_mut().enumerate().skip(4096)
        {
            *s = (2.0 * PI * 20.0 * i as f64 / hop as f64).sin();
        }
        let onsets = onset_detection(&sig, 8000, hop, 1.0);
        assert!(
            onsets.contains(&4096),
            "onsets {onsets:?} must include 4096"
        );
    }

    #[test]
    fn test_onset_detection_silence_has_none() {
        // Pure silence has zero spectral flux -> no onsets.
        let onsets = onset_detection(&vec![0.0; 8192], 8000, 128, 1.0);
        assert!(onsets.is_empty());
    }

    #[test]
    fn test_onset_strength_zero_for_silence() {
        // Spectral-flux envelope of silence is all zeros, one value per frame.
        let hop = 256;
        let strength = onset_strength(&vec![0.0; 4096], hop);
        assert_eq!(strength.len(), 4096 / hop);
        assert!(strength.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_onset_strength_nonneg_and_spikes() {
        // Half silence, half tone: the flux envelope is non-negative and has
        // its maximum at the frame where the tone switches on.
        let n = 4096;
        let hop = 256;
        let mut sig = vec![0.0; n];
        for (i, s) in sig.iter_mut().enumerate().skip(n / 2)
        {
            *s = (2.0 * PI * 30.0 * i as f64 / hop as f64).sin();
        }
        let strength = onset_strength(&sig, hop);
        assert!(strength.iter().all(|&x| x >= 0.0));
        let argmax = strength
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, (n / 2) / hop, "spike should be at the onset frame");
    }

    // ── Pitch ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pitch_autocorrelation_440() {
        let sig = test_signal();
        let pitch = pitch_autocorrelation(&sig.samples, sig.sample_rate, 100.0, 1000.0)
            .expect("pitch should be found");
        // True period is 8000/440 = 18.18 samples; the integer-lag autocorrelation
        // picks lag 18 -> 8000/18 ≈ 444.4 Hz, within ~5 Hz of 440.
        assert!((pitch - 444.444).abs() < 1.0, "autocorr pitch {pitch}");
    }

    #[test]
    fn test_pitch_autocorrelation_none_when_too_short() {
        // max_lag = sr/min_freq must fit inside the signal.
        let sig = vec![0.0; 10];
        assert!(pitch_autocorrelation(&sig, 8000, 100.0, 1000.0).is_none());
    }

    #[test]
    fn test_pitch_yin_440() {
        let sig = test_signal();
        let pitch =
            pitch_yin(&sig.samples, sig.sample_rate, 100.0, 1000.0).expect("pitch should be found");
        // With the corrected parabolic interpolation YIN lands within ~1 Hz of
        // 440 (the buggy sign-flipped version produced ~448 Hz).
        assert!((pitch - 440.0).abs() < 1.5, "yin pitch {pitch}");
    }

    #[test]
    fn test_pitch_yin_220() {
        // Independent frequency to guard against a coincidental fit at 440.
        let sr = 8000;
        let n = 4000;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 220.0 * i as f64 / sr as f64).sin())
            .collect();
        let pitch = pitch_yin(&sig, sr, 80.0, 1000.0).expect("pitch should be found");
        assert!((pitch - 220.0).abs() < 1.5, "yin pitch {pitch}");
    }

    // ── Spectral features (single-tone oracles) ────────────────────────────

    #[test]
    fn test_spectral_centroid_single_bin() {
        // All energy concentrated in one bin -> centroid equals that bin's freq.
        let n = 64;
        let k0 = 8;
        let sr = 64; // so bin frequency = k0 Hz exactly
        let centroid = spectral_centroid(&cosine_at_bin(n, k0), sr);
        assert!((centroid - k0 as f64).abs() < 1e-6, "centroid {centroid}");
    }

    #[test]
    fn test_spectral_centroid_dc_is_zero() {
        // All energy at bin 0 (DC) -> centroid is 0 Hz.
        let centroid = spectral_centroid(&vec![1.0; 64], 8000);
        assert!(centroid.abs() < 1e-6, "centroid {centroid}");
    }

    #[test]
    fn test_spectral_centroid_440() {
        let sig = test_signal();
        let sc = spectral_centroid(&sig.samples, sig.sample_rate);
        assert!((sc - 440.0).abs() < 5.0, "centroid {sc}");
    }

    #[test]
    fn test_spectral_bandwidth_single_bin_is_zero() {
        // A single spectral line has zero spread about its own centroid. The
        // residual here (~1e-6 Hz) is pure FP roundoff from the ~1e-9 leakage
        // in off-bin magnitudes weighted by (freq - centroid)^2; it is seven
        // orders of magnitude below the 8 Hz bin frequency, i.e. effectively 0.
        let n = 64;
        let bw = spectral_bandwidth(&cosine_at_bin(n, 8), 64);
        assert!(bw < 1e-3, "bandwidth {bw}");
    }

    #[test]
    fn test_spectral_rolloff_single_tone() {
        // 100% of the energy sits in bin k0, so any percentile <= 1 rolls off
        // exactly at that bin's frequency.
        let n = 64;
        let k0 = 8;
        let sr = 64;
        let roll = spectral_rolloff(&cosine_at_bin(n, k0), sr, 0.85);
        assert!((roll - k0 as f64).abs() < 1e-6, "rolloff {roll}");
    }

    #[test]
    fn test_spectral_flatness_bounds() {
        // Impulse -> flat spectrum -> flatness ≈ 1.
        let n = 16;
        let mut imp = vec![0.0; n];
        imp[0] = 1.0;
        let flat = spectral_flatness(&imp);
        assert!((flat - 1.0).abs() < 1e-6, "flatness {flat}");

        // Pure tone -> highly peaky -> flatness near 0.
        let tone = spectral_flatness(&cosine_at_bin(64, 8));
        assert!(tone < 1e-3, "tone flatness {tone}");
    }

    #[test]
    fn test_spectral_entropy_oracles() {
        // Flat (impulse) spectrum over N/2+1 = 9 bins -> entropy = log2(9).
        let n = 16;
        let mut imp = vec![0.0; n];
        imp[0] = 1.0;
        let ent = spectral_entropy(&imp);
        assert!((ent - 9.0_f64.log2()).abs() < 1e-6, "entropy {ent}");

        // Single tone -> all probability in one bin -> entropy ≈ 0.
        let tone = spectral_entropy(&cosine_at_bin(64, 8));
        assert!(tone < 1e-6, "tone entropy {tone}");

        // Silence is defined to give 0.
        assert_eq!(spectral_entropy(&vec![0.0; 32]), 0.0);
    }

    #[test]
    fn test_spectral_contrast_shape_and_nonneg() {
        let sig = test_signal();
        let contrast = spectral_contrast(&sig.samples, 7);
        assert_eq!(contrast.len(), 7);
        // peak - median is always >= 0 by construction.
        assert!(contrast.iter().all(|&c| c >= 0.0));
    }

    // ── Feature aggregation ────────────────────────────────────────────────

    #[test]
    fn test_audio_features_vector_layout() {
        let sig = test_signal();
        let features = audio_features(&sig);
        // n_fft = min(2048, 4000) = 2048, hop = 1024 -> (4000-2048)/1024 + 1 = 2
        // frames of 13 MFCCs each.
        assert_eq!(features.mfcc.len(), 2);
        assert_eq!(features.mfcc[0].len(), 13);
        assert_eq!(features.chroma.len(), 12);

        let v = features.to_vector();
        // 9 scalars + 2*13 MFCC mean/std + 12 chroma = 47.
        assert_eq!(v.len(), 9 + 2 * 13 + 12);
        // First three entries mirror the scalar statistics exactly.
        assert!((v[0] - features.rms).abs() < 1e-12);
        assert!((v[1] - features.peak).abs() < 1e-12);
        assert!((v[2] - features.zero_crossing_rate).abs() < 1e-12);
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_audio_features_roundtrip_serde() {
        let sig = test_signal();
        let features = audio_features(&sig);
        let json = serde_json::to_string(&features).expect("serialize");
        let back: AudioFeatureSet = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.to_vector().len(), features.to_vector().len());
        assert!((back.spectral_centroid - features.spectral_centroid).abs() < 1e-12);
    }

    #[test]
    fn test_stft_spectrogram_shape() {
        let sig = test_signal();
        let spec = stft(&sig.samples, 512, 256);
        // (4000 - 512) / 256 + 1 = 14 frames. Each magnitude bin 512/2 + 1 = 257.
        assert_eq!(spec.len(), 14);
        assert_eq!(spec[0].len(), 257);
    }
}
