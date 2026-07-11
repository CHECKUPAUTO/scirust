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
//!
//! Alongside the pulse chain sit array processing ([`beamform`], [`doa`],
//! [`music`]) for angle estimation — from the conventional beamformer through
//! MVDR/Capon to the MUSIC subspace method — [`fmcw`] for the continuous-wave /
//! mmWave model, where range and velocity fall out of two FFTs of the mixer's
//! beat signal rather than a matched filter, [`detect`] for the 2-D detection
//! stage — CFAR over the range-Doppler map followed by clustering of the
//! detections into target centroids — and [`track`] for the temporal layer that
//! associates those centroids across frames and smooths them with α–β track
//! filters.

pub mod ambiguity;
pub mod beamform;
pub mod cfar;
pub mod detect;
pub mod doa;
pub mod doppler;
pub mod fmcw;
pub mod matched_filter;
pub mod mti;
pub mod music;
pub mod track;
pub mod waveform;

pub use ambiguity::ambiguity;
pub use beamform::{beamform_spectrum, estimate_doa, steering_vector};
pub use cfar::{ca_cfar, ca_cfar_alpha, os_cfar, os_cfar_alpha};
pub use detect::{Detection, ca_cfar_2d, cluster_detections};
pub use doa::{covariance, mvdr_spectrum};
pub use doppler::{doppler_spectrum, range_doppler_map};
pub use fmcw::{beat_frequency_to_range, range_doppler, range_profile, range_resolution};
pub use matched_filter::{cross_correlate, peak_lag, peak_to_sidelobe};
pub use mti::mti_canceller;
pub use music::music_spectrum;
pub use track::{AlphaBeta, MultiTracker, Track, critically_damped_gains};
pub use waveform::{barker_code, lfm_chirp};
