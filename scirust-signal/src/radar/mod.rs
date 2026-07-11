//! Radar signal processing.
//!
//! Pulse-compression waveforms ([`waveform`]) and matched filtering
//! ([`matched_filter`]) — the range-processing core of a pulse-Doppler radar,
//! built directly on this crate's [`Complex`](crate::complex::Complex)
//! primitive. A long coded pulse is transmitted for energy, then compressed on
//! receive into a sharp peak at the echo delay whose resolution is set by the
//! bandwidth, not the pulse length.

pub mod matched_filter;
pub mod waveform;

pub use matched_filter::{cross_correlate, peak_lag, peak_to_sidelobe};
pub use waveform::{barker_code, lfm_chirp};
