//! # scirust-biomed — biomedical signal analysis
//!
//! Pure-Rust, deterministic ECG analytics for diagnostic support (IEC 62304):
//!
//! - [`ecg::detect_r_peaks`] / [`ecg::heart_rate_bpm`] — Pan–Tompkins-style QRS
//!   detection and heart rate.
//! - [`ecg::classify_rhythm`] — coarse rhythm class (normal / brady / tachy /
//!   irregular) from RR intervals.
//! - [`ConformalBeats`] — guaranteed-coverage prediction *sets* for beat
//!   classification (coverage `≥ 1 − α`), the safe object to surface clinically.

pub mod conformal_beats;
pub mod ecg;

pub use conformal_beats::ConformalBeats;
pub use ecg::{RhythmClass, classify_rhythm, detect_r_peaks, heart_rate_bpm, rr_intervals};
