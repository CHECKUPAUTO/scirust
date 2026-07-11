//! # scirust-biomed — biomedical signal analysis & closed-loop control
//!
//! Pure-Rust, deterministic ECG analytics for diagnostic support (IEC 62304):
//!
//! - [`ecg::detect_r_peaks`] / [`ecg::heart_rate_bpm`] — Pan–Tompkins-style QRS
//!   detection and heart rate.
//! - [`ecg::classify_rhythm`] — coarse rhythm class (normal / brady / tachy /
//!   irregular) from RR intervals.
//! - [`ConformalBeats`] — guaranteed-coverage prediction *sets* for beat
//!   classification (coverage `≥ 1 − α`), the safe object to surface clinically.
//!
//! [`control`] adds the *control* side of a closed-loop device (dosing,
//! not just signal analysis): a generic PID controller, insulin-on-board
//! tracking, threshold-based supervisory safety (low-glucose suspend,
//! auto-mode exit), and a Control-Barrier-Function QP safety filter. See
//! [`control`]'s module doc for the non-clinical-use caveat that applies
//! to all of it.

pub mod conformal_beats;
pub mod control;
pub mod ecg;
pub mod hrv;
pub mod lomb;

pub use conformal_beats::ConformalBeats;
#[cfg(feature = "sim")]
pub use control::GlucoseSystem;
pub use control::{
    AutoModeMonitor, GlucoseModel, InsulinOnBoard, PidController, PidGains, SafeDose,
    cbf_safe_dose, max_safe_bolus, predictive_suspend, suspend_on_low,
};
pub use ecg::{RhythmClass, classify_rhythm, detect_r_peaks, heart_rate_bpm, rr_intervals};
pub use hrv::{HrvMetrics, compute_hrv};
pub use lomb::{band_power, lf_hf, lomb_scargle_power};
