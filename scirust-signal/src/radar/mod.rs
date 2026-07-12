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
//! [`music`], [`esprit`]) for angle estimation — from the conventional
//! beamformer through MVDR/Capon to the MUSIC and ESPRIT subspace methods —
//! [`fmcw`] for the continuous-wave /
//! mmWave model, where range and velocity fall out of two FFTs of the mixer's
//! beat signal rather than a matched filter, [`detect`] for the 2-D detection
//! stage — CFAR over the range-Doppler map followed by clustering of the
//! detections into target centroids — and [`track`] for the temporal layer that
//! associates those centroids across frames and smooths them with α–β track
//! filters, with [`kalman`] adding a full constant-velocity Kalman filter and
//! an Interacting-Multiple-Model estimator for manoeuvring targets, and
//! [`imm2d`] extending that to a planar coordinated-turn IMM (a general linear
//! Kalman filter blending constant-velocity and constant-turn-rate models) for
//! tracking turning targets in the (x, y) plane, [`ekf`] adding an extended
//! Kalman filter that tracks a Cartesian state directly from raw polar
//! (range/bearing) radar measurements, [`mtt`] closing the loop with a
//! multi-target tracker of per-target EKFs associated by a statistical
//! (normalised-innovation-squared) validation gate, and [`pda`] adding a
//! probabilistic data association filter that tracks through clutter by
//! soft-combining every gated measurement instead of a hard nearest-neighbour
//! pick. [`micro_doppler`] adds the time–frequency signature of target
//! micro-motion (a Hann-windowed spectrogram plus ridge / bulk-Doppler /
//! bandwidth / cadence descriptors) for target classification.

pub mod ambiguity;
pub mod beamform;
pub mod cfar;
pub mod detect;
pub mod doa;
pub mod doppler;
pub mod ekf;
pub mod esprit;
pub mod fmcw;
pub mod imm2d;
pub mod kalman;
pub mod matched_filter;
pub mod micro_doppler;
pub mod mti;
pub mod mtt;
pub mod music;
pub mod pda;
pub mod track;
pub mod waveform;

pub use ambiguity::ambiguity;
pub use beamform::{beamform_spectrum, estimate_doa, steering_vector};
pub use cfar::{ca_cfar, ca_cfar_alpha, os_cfar, os_cfar_alpha};
pub use detect::{Detection, ca_cfar_2d, cluster_detections};
pub use doa::{covariance, mvdr_spectrum};
pub use doppler::{doppler_spectrum, range_doppler_map};
pub use ekf::RadarEkf;
pub use esprit::esprit_doa;
pub use fmcw::{beat_frequency_to_range, range_doppler, range_profile, range_resolution};
pub use imm2d::{Imm2D, KalmanLinear, ct_model_2d, cv_model_2d};
pub use kalman::{Imm, KalmanCV};
pub use matched_filter::{cross_correlate, peak_lag, peak_to_sidelobe};
pub use micro_doppler::{
    bin_frequencies, cadence, doppler_bandwidth, mean_doppler, ridge, spectrogram,
};
pub use mti::mti_canceller;
pub use mtt::{RadarMultiTracker, RadarTrack};
pub use music::music_spectrum;
pub use pda::PdaFilter;
pub use track::{AlphaBeta, MultiTracker, Track, critically_damped_gains};
pub use waveform::{barker_code, lfm_chirp};
