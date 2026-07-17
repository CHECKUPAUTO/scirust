//! Radar signal processing.
//!
//! Pulse-compression waveforms ([`waveform`]), matched filtering
//! ([`matched_filter`]) and constant-false-alarm-rate detection ([`cfar`]) —
//! the range-processing and detection core of a pulse-Doppler radar, built
//! directly on this crate's [`Complex`](crate::complex::Complex) primitive. A
//! long coded pulse is transmitted for energy, compressed on receive into a
//! sharp peak at the echo delay (resolution set by the bandwidth, not the pulse
//! length), then thresholded adaptively so the false-alarm rate stays fixed as
//! the noise/clutter level varies. [`polyphase`] widens the waveform library
//! beyond the length-13 Barker limit with Frank, P3/P4 and Zadoff-Chu (CAZAC)
//! codes — the perfect-periodic-autocorrelation and LPI waveforms of modern
//! radar.
//!
//! Alongside the pulse chain sit array processing ([`beamform`], [`doa`],
//! [`music`], [`esprit`]) for angle estimation — from the conventional
//! beamformer through MVDR/Capon to the MUSIC and ESPRIT subspace methods, with
//! [`monopulse`] for single-dwell sum/difference angle estimation and
//! [`interferometer`] for phase-comparison (interferometric) angle estimation —
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
//! bandwidth / cadence descriptors) for target classification, and
//! [`swerling`] for detection statistics — the probability of detecting a
//! fluctuating or steady target versus SNR (Swerling I closed form and
//! Albersheim's equation), the complement to the CFAR threshold — and
//! [`range_equation`] for the link budget: the SNR a radar delivers on a target
//! of a given RCS at range, and the maximum detection range that closes with the
//! [`swerling`] required SNR. [`clutter`] supplies the amplitude distributions
//! (Rayleigh, Weibull, log-normal) that CFAR thresholds are designed against, and
//! [`prf`] gives the pulse-repetition-frequency ambiguities — unambiguous range
//! and velocity, blind speeds, and range/velocity folding.
//! [`stepped_frequency`] synthesises wideband range resolution from a burst of
//! narrowband pulses stepped in frequency, resolved by an inverse DFT.
//! [`stap`] closes the airborne-radar loop with space-time adaptive processing —
//! a joint angle-Doppler adaptive filter that nulls the ground-clutter ridge
//! (`f_d = β·f_s`) while holding unit gain on the target, pulling slow movers out
//! of clutter that no angle-only or Doppler-only filter can separate. [`sar`]
//! adds the imaging mode: synthetic-aperture azimuth compression that focuses a
//! target's quadratic slow-time phase history (an azimuth chirp) into a sharp
//! peak, synthesising a `λR/D` aperture for the range-independent `D/2`
//! cross-range resolution. [`accuracy`] closes the performance side with the
//! Cramér–Rao lower bounds on measurement precision — the `1/√SNR` floors on
//! delay/range, Doppler/velocity, and monopulse angle that bound the estimators
//! above and that a link budget must close to meet an accuracy spec.
//! [`vi_cfar`] complements the fixed-strategy detectors in [`cfar`]/
//! [`cfar_variants`] with the Variability-Index composite: a per-CUT switch
//! among cell-averaging, greatest-of and smallest-of driven by each
//! half-window's own variability and their mean ratio, plus a pooled
//! trimmed-mean fallback (this crate's own extension, not classical VI-CFAR)
//! for the case classical VI-CFAR handles worst — interferers in *both*
//! reference half-windows at once. Built on [`crate::sliding_stats`]'s O(1)
//! sliding moments for its streaming path; see [`vi_cfar`]'s module docs for
//! the full mathematical contract, provenance, and exactly which parts of the
//! switching logic are verified classical behavior versus this crate's own
//! documented extension.

pub mod accuracy;
pub mod ambiguity;
pub mod beamform;
pub mod binary_integration;
pub mod cfar;
pub mod cfar_variants;
pub mod clutter;
pub mod costas;
pub mod crt_prf;
pub mod dbs;
pub mod detect;
pub mod doa;
pub mod doppler;
pub mod ekf;
pub mod esprit;
pub mod fmcw;
pub mod imm2d;
pub mod interferometer;
pub mod kalman;
pub mod matched_filter;
pub mod micro_doppler;
pub mod monopulse;
pub mod mti;
pub mod mtt;
pub mod music;
pub mod pda;
pub mod polyphase;
pub mod prf;
pub mod propagation;
pub mod range_equation;
pub mod sar;
pub mod stap;
pub mod stepped_frequency;
pub mod swerling;
pub mod track;
pub mod vi_cfar;
pub mod waveform;

pub use accuracy::{
    angle_crlb, delay_crlb, doppler_crlb, range_crlb, rms_bandwidth_lfm, rms_duration_rect,
    velocity_crlb,
};
pub use ambiguity::ambiguity;
pub use beamform::{beamform_spectrum, estimate_doa, steering_vector};
pub use binary_integration::{
    binomial_pmf, binomial_sf_ge, integrated_pd, integrated_pfa, optimal_m,
};
pub use cfar::{ca_cfar, ca_cfar_alpha, os_cfar, os_cfar_alpha};
pub use cfar_variants::{go_cfar, so_cfar, tm_cfar};
pub use clutter::{
    erf, lognormal_cdf, lognormal_pdf, rayleigh_cdf, rayleigh_pdf, rayleigh_quantile, weibull_cdf,
    weibull_pdf, weibull_quantile,
};
pub use costas::{is_costas, max_coincidence, primitive_root, welch_costas};
pub use crt_prf::{combined_ambiguity, crt_pair, egcd, mod_inverse, resolve_range};
pub use dbs::{azimuth_doppler, dbs_azimuth_resolution, doppler_gradient, sharpening_ratio};
pub use detect::{Detection, ca_cfar_2d, cluster_detections};
pub use doa::{covariance, mvdr_spectrum};
pub use doppler::{doppler_spectrum, range_doppler_map};
pub use ekf::RadarEkf;
pub use esprit::esprit_doa;
pub use fmcw::{beat_frequency_to_range, range_doppler, range_profile, range_resolution};
pub use imm2d::{Imm2D, KalmanLinear, ct_model_2d, cv_model_2d};
pub use interferometer::{
    angle_from_phase, phase_difference, phase_from_signals, unambiguous_angle, wrap_phase,
};
pub use kalman::{Imm, KalmanCV};
pub use matched_filter::{cross_correlate, peak_lag, peak_to_sidelobe};
pub use micro_doppler::{
    bin_frequencies, cadence, doppler_bandwidth, mean_doppler, ridge, spectrogram,
};
pub use monopulse::{
    beam_voltage, estimate_angle as monopulse_estimate_angle, monopulse_ratio, monopulse_slope,
};
pub use mti::mti_canceller;
pub use mtt::{RadarMultiTracker, RadarTrack};
pub use music::music_spectrum;
pub use pda::PdaFilter;
pub use polyphase::{frank_code, p3_code, p4_code, periodic_autocorrelation, zadoff_chu};
pub use prf::{
    blind_speed, fold_range, fold_velocity, max_doppler, unambiguous_range, unambiguous_velocity,
    velocity_from_doppler,
};
pub use propagation::{
    first_null_range, path_length_difference, phase_difference as multipath_phase_difference,
    power_factor, propagation_factor,
};
pub use range_equation::RadarLink;
pub use sar::{
    azimuth_chirp_rate, azimuth_doppler_bandwidth, azimuth_history, azimuth_reference,
    azimuth_resolution, focus_azimuth, synthetic_aperture_length,
};
pub use stap::{
    adaptive_weights, clutter_covariance, clutter_ridge_doppler, optimal_sinr, space_time_steering,
    spatial_frequency,
};
pub use stepped_frequency::{
    max_unambiguous_range, range_bins, range_profile as stepped_range_profile,
    range_resolution as stepped_range_resolution, synthetic_bandwidth,
};
pub use swerling::{
    albersheim_pd, albersheim_snr, single_pulse_threshold, swerling1_pd, swerling1_required_snr,
};
pub use track::{AlphaBeta, MultiTracker, Track, critically_damped_gains};
pub use vi_cfar::{
    CensoredMeanResult, CfarConfig, CfarDecision, CfarDetector, CfarError, CfarMode,
    CfarStreamDetector, DetectorPolicy, EdgePolicy, InputValidationPolicy, RobustNoiseEstimator,
    SwitchingThresholds, TrimmedMeanResult, censored_mean,
    evaluate_slice as vi_cfar_evaluate_slice, mean_ratio, trimmed_mean, variability_index,
};
pub use waveform::{barker_code, lfm_chirp};
