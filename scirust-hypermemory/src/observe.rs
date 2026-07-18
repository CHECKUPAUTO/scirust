//! Observation fusion — SciRust's noise-processing toolkit as the denoising
//! front-end (feature `signal-denoise`).
//!
//! The cleanup memory ([`crate::S16ExactIndex::denoise`]) recognizes a *single*
//! noisy code. The realistic agent-memory scenario is stronger: the same
//! underlying code is observed **repeatedly** through a noisy channel. Per
//! lane, those observations form a time series — exactly what
//! [`scirust_signal::denoise`] is built for. This module runs that toolkit
//! per lane across the observation sequence, producing one fused code to hand
//! to the cleanup memory:
//!
//! ```text
//! observations ──(per-lane scirust-signal denoising)──▶ fused code ──▶ denoise()
//! ```
//!
//! Three [`FusionStrategy`] variants keep the comparison honest:
//!
//! * [`FusionStrategy::Mean`] — the naive per-lane average. The in-crate
//!   baseline every toolkit strategy must beat to justify itself.
//! * [`FusionStrategy::Kalman`] — [`scirust_signal::denoise::adaptive::kalman_smooth`]
//!   (local-level Kalman + RTS smoother) per lane, then the smoothed series'
//!   mean. Right for broadband observation noise.
//! * [`FusionStrategy::HampelKalman`] — [`scirust_signal::denoise::rank::hampel_filter`]
//!   first (order-statistics impulse rejection), then the Kalman-RTS pass.
//!   Right when observations suffer occasional gross corruption (spikes), the
//!   regime where a plain mean fails — measured in `tests/observe_fusion.rs`.
//!
//! Everything stays deterministic: `scirust-signal`'s routines are fixed-order
//! scalar `f64` pipelines with no RNG, and the per-lane loop runs in fixed lane
//! order.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::error::{HypermemoryError, Result};
use crate::index::{Denoised, S16ExactIndex};
use crate::representation::validate_finite;

/// How repeated observations of one code are fused into a single estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FusionStrategy {
    /// Naive per-lane mean — the baseline the toolkit strategies must beat.
    Mean,
    /// Per-lane local-level Kalman filter + RTS smoother
    /// (`scirust_signal::denoise::adaptive::kalman_smooth`), then the smoothed
    /// series' mean. `process_var` sets agility (small = heavy smoothing for a
    /// static code), `meas_var` the observation-noise variance; both must be
    /// finite and strictly positive.
    Kalman {
        /// Process variance of the local-level model.
        process_var: f64,
        /// Observation-noise variance.
        meas_var: f64,
    },
    /// Per-lane Hampel impulse rejection
    /// (`scirust_signal::denoise::rank::hampel_filter`) followed by the same
    /// Kalman-RTS pass — for observation streams with occasional gross
    /// corruption. `half_window ≥ 1`; `n_sigma` finite and strictly positive.
    HampelKalman {
        /// Hampel half-window (in observations).
        half_window: usize,
        /// Hampel outlier threshold in robust sigmas.
        n_sigma: f64,
        /// Process variance of the Kalman stage.
        process_var: f64,
        /// Observation-noise variance of the Kalman stage.
        meas_var: f64,
    },
}

impl FusionStrategy {
    fn validate(&self) -> Result<()> {
        let ok = match *self
        {
            FusionStrategy::Mean => true,
            FusionStrategy::Kalman {
                process_var,
                meas_var,
            } =>
            {
                process_var.is_finite()
                    && meas_var.is_finite()
                    && process_var > 0.0
                    && meas_var > 0.0
            },
            FusionStrategy::HampelKalman {
                half_window,
                n_sigma,
                process_var,
                meas_var,
            } =>
            {
                half_window >= 1
                    && n_sigma.is_finite()
                    && n_sigma > 0.0
                    && process_var.is_finite()
                    && meas_var.is_finite()
                    && process_var > 0.0
                    && meas_var > 0.0
            },
        };
        if ok
        {
            Ok(())
        }
        else
        {
            Err(HypermemoryError::InvalidRepresentation {
                reason: "fusion strategy parameters must be finite and positive",
            })
        }
    }

    /// Denoise one lane's observation series and reduce it to one estimate.
    fn fuse_lane(&self, series: &[f64]) -> f64 {
        let denoised: Vec<f64> = match *self
        {
            FusionStrategy::Mean => series.to_vec(),
            FusionStrategy::Kalman {
                process_var,
                meas_var,
            } => scirust_signal::denoise::adaptive::kalman_smooth(series, process_var, meas_var),
            FusionStrategy::HampelKalman {
                half_window,
                n_sigma,
                process_var,
                meas_var,
            } =>
            {
                let despiked =
                    scirust_signal::denoise::rank::hampel_filter(series, half_window, n_sigma);
                scirust_signal::denoise::adaptive::kalman_smooth(&despiked, process_var, meas_var)
            },
        };
        // Fixed left-to-right mean of the denoised series (for a static code the
        // RTS-smoothed series is nearly constant; the mean is its natural
        // single-value reduction and is what `Mean` degenerates to).
        let mut acc = 0.0f64;
        for &x in &denoised
        {
            acc += x;
        }
        acc / denoised.len() as f64
    }
}

/// Fuse repeated noisy observations of one code into a single estimate using
/// SciRust's noise-processing toolkit, per lane across the observation
/// sequence.
///
/// Errors: an empty observation slice, a non-finite observation, or invalid
/// strategy parameters are typed errors. The result is validated finite.
pub fn fuse_observations(
    observations: &[SedenionSimd],
    strategy: FusionStrategy,
) -> Result<SedenionSimd> {
    strategy.validate()?;
    if observations.is_empty()
    {
        return Err(HypermemoryError::InvalidRepresentation {
            reason: "observation fusion needs at least one observation",
        });
    }
    for obs in observations
    {
        validate_finite(obs)?;
    }

    let mut fused = [0.0f32; 16];
    let mut series = vec![0.0f64; observations.len()];
    for (lane, out) in fused.iter_mut().enumerate()
    {
        for (i, obs) in observations.iter().enumerate()
        {
            series[i] = f64::from(obs.to_array()[lane]);
        }
        *out = strategy.fuse_lane(&series) as f32;
    }
    let result = SedenionSimd::from_array(fused);
    validate_finite(&result)?;
    Ok(result)
}

impl S16ExactIndex {
    /// Fuse repeated noisy observations with SciRust's noise toolkit, then
    /// snap the fused estimate to the nearest stored prototype (cleanup).
    ///
    /// Composition of [`fuse_observations`] and
    /// [`S16ExactIndex::denoise`], with the same error and threshold
    /// semantics.
    ///
    /// [`S16ExactIndex::denoise`]: crate::S16ExactIndex::denoise
    #[must_use = "the outcome says whether the observations were recognized"]
    pub fn denoise_observations(
        &self,
        observations: &[SedenionSimd],
        strategy: FusionStrategy,
        threshold: f32,
    ) -> Result<Option<Denoised>> {
        let fused = fuse_observations(observations, strategy)?;
        self.denoise(&fused, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_observations_are_rejected() {
        assert!(fuse_observations(&[], FusionStrategy::Mean).is_err());
    }

    #[test]
    fn non_finite_observation_is_rejected() {
        let mut bad = [0.0f32; 16];
        bad[2] = f32::NAN;
        let obs = [SedenionSimd::unit(0), SedenionSimd::from_array(bad)];
        assert!(fuse_observations(&obs, FusionStrategy::Mean).is_err());
    }

    #[test]
    fn invalid_strategy_parameters_are_rejected() {
        let obs = [SedenionSimd::unit(0)];
        for strategy in [
            FusionStrategy::Kalman {
                process_var: 0.0,
                meas_var: 1.0,
            },
            FusionStrategy::Kalman {
                process_var: f64::NAN,
                meas_var: 1.0,
            },
            FusionStrategy::HampelKalman {
                half_window: 0,
                n_sigma: 3.0,
                process_var: 1e-6,
                meas_var: 1.0,
            },
            FusionStrategy::HampelKalman {
                half_window: 2,
                n_sigma: -1.0,
                process_var: 1e-6,
                meas_var: 1.0,
            },
        ]
        {
            assert!(fuse_observations(&obs, strategy).is_err(), "{strategy:?}");
        }
    }

    #[test]
    fn mean_of_identical_observations_is_the_observation() {
        let code = SedenionSimd::unit(3).scale(0.5) + SedenionSimd::unit(7).scale(0.25);
        let obs = [code, code, code];
        let fused = fuse_observations(&obs, FusionStrategy::Mean).unwrap();
        assert_eq!(fused.to_array(), code.to_array());
    }
}
