//! SciRust Signal Processing
//!
//! Pure-Rust DSP primitives for industrial monitoring and automotive applications.
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
//!   rank, transform, variational, adaptive) plus a noise characterizer/classifier,
//!   a detect-then-denoise auto pipeline with a residual whiteness self-check, and
//!   the composite wavelet–RLS–RTS estimation pipeline
//! - **Digital filters** — windowed-sinc FIR (lowpass/highpass), Butterworth IIR
//!   via cascaded second-order sections, and a general direct-form-II-transposed
//!   `lfilter`
//! - **Radar** — pulse-compression waveforms (linear-FM chirp, Barker phase
//!   codes), matched filtering (cross-correlation, peak/echo-delay estimation,
//!   peak-to-sidelobe ratio), the ambiguity function (joint delay-Doppler
//!   response, range-Doppler coupling), Doppler processing (range-Doppler map),
//!   MTI clutter cancellers and CFAR detection (cell-averaging and
//!   ordered-statistic, with the closed-form threshold scaling)

pub mod bearing;
pub mod cepstrum;
pub mod complex;
pub mod denoise;
pub mod envelope;
pub mod features;
pub mod fft;
pub mod filter;
pub mod mcsa;
pub mod order;
pub mod radar;
pub mod windows;

pub use bearing::{BearingFault, BearingGeometry, bpfi, bpfo, bsf, detect_bearing_faults, ftf};
pub use cepstrum::{dominant_quefrency, real_cepstrum};
/// Re-export commonly used types.
pub use complex::Complex;
pub use denoise::{
    AutoResult, Denoiser, DenoiserFamily, NoiseProfile, NoiseType, Separation, Wavelet,
    WaveletRlsRtsParams, catalog, classify, denoise_auto, estimate_noise_std, kalman_smooth,
    kalman_smooth_auto, kalman_trend_smooth, moving_average as denoise_moving_average,
    savitzky_golay, separate, total_variation, total_variation_exact, wavelet_denoise,
    wavelet_denoise_sure, wavelet_denoise_with, wavelet_rls_rts_smooth, wavelet_rls_rts_smooth_1d,
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
pub use filter::{
    Biquad, butter_highpass_sos, butter_lowpass_sos, fir_highpass, fir_lowpass, lfilter, sos_filter,
};
pub use mcsa::{
    BarSeverity, BrokenBarResult, EccentricityResult, MotorDiagnosis, MotorFault,
    analyze_broken_bar, analyze_eccentricity, diagnose_motor, slip,
};
pub use order::{order_spectrum, order_track, resample_constant_angle, rpm_profile, tacho_to_rpm};
pub use radar::{
    ambiguity, barker_code, ca_cfar, ca_cfar_alpha, cross_correlate, doppler_spectrum, lfm_chirp,
    mti_canceller, os_cfar, os_cfar_alpha, peak_lag, peak_to_sidelobe, range_doppler_map,
};
pub use windows::{apply_window, blackman, blackman_harris, flattop, hamming, hanning};
