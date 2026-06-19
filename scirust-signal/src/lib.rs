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

pub mod bearing;
pub mod complex;
pub mod features;
pub mod fft;
pub mod mcsa;
pub mod order;
pub mod windows;

pub use bearing::{BearingFault, BearingGeometry, bpfi, bpfo, bsf, detect_bearing_faults, ftf};
/// Re-export commonly used types.
pub use complex::Complex;
pub use features::spectral::{
    band_power, psd, spectral_centroid, spectral_entropy, spectral_flatness, spectral_rolloff,
    spectral_spread,
};
pub use features::{
    autocorrelation, crest_factor, energy, entropy, kurtosis, peak_to_peak, rms, skewness,
    zero_crossing_rate,
};
pub use fft::{fft, fft_real, ifft};
pub use mcsa::{BarSeverity, BrokenBarResult, analyze_broken_bar, slip};
pub use order::{order_spectrum, order_track, resample_constant_angle, rpm_profile, tacho_to_rpm};
pub use windows::{apply_window, blackman, blackman_harris, flattop, hamming, hanning};
