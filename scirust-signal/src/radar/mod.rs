//! Radar signal processing.
//!
//! Pulse-compression waveforms ([`waveform`]), matched filtering
//! ([`matched_filter`]) and constant-false-alarm-rate detection ([`cfar`]) —
//! the range-processing and detection core of a pulse-Doppler radar, built
//! directly on this crate's [`Complex`](crate::complex::Complex) primitive. A
//! long coded pulse is transmitted for energy, compressed on receive into a
//! sharp peak at the echo delay (resolution set by the bandwidth, not the pulse
//! length), then thresholded adaptively so the false-alarm rate stays fixed as
//! the noise/clutter level varies.

pub mod cfar;
pub mod matched_filter;
pub mod waveform;

pub use cfar::{ca_cfar, ca_cfar_alpha, os_cfar, os_cfar_alpha};
pub use matched_filter::{cross_correlate, peak_lag, peak_to_sidelobe};
pub use waveform::{barker_code, lfm_chirp};
