//! SciRust Signal Processing
//!
//! Pure-Rust DSP primitives for industrial monitoring and automotive applications.
//! Zero external dependencies beyond `scirust-core`.
//!
//! ## Modules
//! - **Complex numbers** — basic complex arithmetic (`Complex`)
//! - **FFT** — radix-2 Cooley-Tukey forward/inverse FFT
//! - **Windows** — Hanning, Hamming, Blackman, Blackman-Harris, Flat-top
//! - **Feature extraction** — time-domain (RMS, crest factor, kurtosis, skewness,
//!   zero-crossing rate, autocorrelation), frequency-domain (PSD, spectral centroid,
//!   spectral entropy, band power)
//! - **Bearing diagnostics** — BPFO, BPFI, BSF, FTF calculation, fault frequency
//!   detection for rolling-element bearings
//! - **Order analysis** — order tracking, resampling for variable-speed rotating machinery
//! - **Denoising & noise detection** — extensible noise-removal families (linear,
//!   rank, transform, variational, adaptive) plus a noise characterizer/classifier
//!   and a detect-then-denoise auto pipeline with a residual whiteness self-check

pub mod bearing;
pub mod cepstrum;
pub mod complex;
pub mod denoise;
pub mod envelope;
pub mod features;
pub mod fft;
pub mod mcsa;
pub mod order;
pub mod windows;

pub use bearing::{BearingFault, BearingGeometry, bpfi, bpfo, bsf, detect_bearing_faults, ftf};
pub use cepstrum::{dominant_quefrency, real_cepstrum};
/// Re-export commonly used types.
pub use complex::Complex;
pub use denoise::{
    AutoResult, Denoiser, DenoiserFamily, NoiseProfile, NoiseType, Separation, Wavelet, catalog,
    classify, denoise_auto, estimate_noise_std, kalman_smooth, kalman_smooth_auto,
    moving_average as denoise_moving_average, savitzky_golay, separate, total_variation,
    wavelet_denoise, wavelet_denoise_with,
};
pub use envelope::{dominant_envelope_freq, envelope_spectrum, hilbert_envelope};
pub use features::spectral::{
    band_power, psd, spectral_centroid, spectral_entropy, spectral_flatness, spectral_rolloff,
    spectral_spread,
};
pub use features::{
    autocorrelation, crest_factor, energy, entropy, kurtosis, peak_to_peak, rms, skewness,
    zero_crossing_rate,
};
pub use fft::{fft, fft_real, ifft};
pub use mcsa::{
    BarSeverity, BrokenBarResult, EccentricityResult, MotorDiagnosis, MotorFault,
    analyze_broken_bar, analyze_eccentricity, diagnose_motor, slip,
};
pub use order::{order_spectrum, order_track, resample_constant_angle, rpm_profile, tacho_to_rpm};
pub use windows::{apply_window, blackman, blackman_harris, flattop, hamming, hanning};
