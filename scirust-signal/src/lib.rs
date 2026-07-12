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
//!   MTI clutter cancellers, CFAR detection (cell-averaging and
//!   ordered-statistic in 1-D, plus 2-D cell-averaging CFAR over a
//!   range-Doppler map with connected-component clustering of detections into
//!   target centroids), array processing (ULA steering vectors,
//!   delay-and-sum beamforming,
//!   high-resolution MVDR/Capon DOA that resolves sub-beamwidth sources,
//!   MUSIC subspace direction finding via a from-scratch complex-Hermitian
//!   eigensolver, and gridless ESPRIT that reads the angles straight off the
//!   eigenvalues of the subspace rotation), FMCW processing (beat-frequency
//!   ranging, range
//!   resolution, and the range-Doppler cube from raw beat chirps) and target
//!   tracking (α–β constant-velocity track filters and a nearest-neighbour
//!   multi-target tracker over the clustered detections, plus a full
//!   constant-velocity Kalman filter and an Interacting-Multiple-Model
//!   estimator that switches between quiet and agile models to follow
//!   manoeuvring targets, a planar coordinated-turn IMM — a general linear
//!   Kalman filter blending constant-velocity and constant-turn-rate models to
//!   track turning targets in the (x, y) plane, an extended Kalman filter
//!   that tracks a Cartesian state directly from raw polar range/bearing
//!   measurements, a multi-target tracker of per-target EKFs associated by
//!   a statistical normalised-innovation-squared validation gate, and a
//!   probabilistic data association filter that tracks through clutter by
//!   soft-combining every gated measurement rather than a hard
//!   nearest-neighbour pick), micro-Doppler analysis (a Hann-windowed
//!   spectrogram of the slow-time return with ridge / bulk-Doppler / bandwidth /
//!   cadence descriptors for target classification), detection statistics
//!   (Swerling I and Albersheim probability-of-detection versus SNR, the
//!   complement to the CFAR threshold), the radar range equation (a
//!   monostatic link budget giving delivered SNR versus RCS and range, and the
//!   maximum detection range that closes with the required SNR), and clutter
//!   amplitude statistics (Rayleigh, Weibull, and log-normal distributions — the
//!   spiky-clutter models CFAR thresholds are designed against, with a
//!   self-contained error function)

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
    AlphaBeta, Detection, Imm, Imm2D, KalmanCV, KalmanLinear, MultiTracker, PdaFilter, RadarEkf,
    RadarLink, RadarMultiTracker, RadarTrack, Track, albersheim_pd, albersheim_snr, ambiguity,
    barker_code, beamform_spectrum, beat_frequency_to_range, bin_frequencies, ca_cfar, ca_cfar_2d,
    ca_cfar_alpha, cadence, cluster_detections, covariance, critically_damped_gains,
    cross_correlate, ct_model_2d, cv_model_2d, doppler_bandwidth, doppler_spectrum, erf as erf_fn,
    esprit_doa, estimate_doa, lfm_chirp, lognormal_cdf, lognormal_pdf, mean_doppler, mti_canceller,
    music_spectrum, mvdr_spectrum, os_cfar, os_cfar_alpha, peak_lag, peak_to_sidelobe,
    range_doppler, range_doppler_map, range_profile, range_resolution,
    rayleigh_cdf as clutter_rayleigh_cdf, rayleigh_pdf as clutter_rayleigh_pdf, rayleigh_quantile,
    ridge as micro_doppler_ridge, single_pulse_threshold, spectrogram, steering_vector,
    swerling1_pd, swerling1_required_snr, weibull_cdf, weibull_pdf, weibull_quantile,
};
pub use windows::{apply_window, blackman, blackman_harris, flattop, hamming, hanning};
