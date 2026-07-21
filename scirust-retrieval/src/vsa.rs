//! VSA / HRR primitives — the stable port of what *won* in the
//! `scirust-hypermemory` falsification program.
//!
//! That program (see `docs/research/SCIRUST_HYPERMEMORY_CONCLUSIONS.md`)
//! measured 16-dimensional sedenion binding against Holographic Reduced
//! Representations and found HRR the stronger structure encoding
//! (0.946 vs 0.869 retrieval agreement at noise 0.5), with the cleanup
//! memory and observation fusion as the components that actually carried
//! the noise robustness. None of those winners needs nightly `portable_simd`
//! — they are ordinary fixed-order `f32` arithmetic — so this module makes
//! them available to the whole workspace at any dimension:
//!
//! * [`circular_convolution`] / [`circular_correlation`] — HRR binding and
//!   approximate unbinding (`x ⊛ y`, then `x† ⊛ (x ⊛ y) ≈ y`);
//! * [`superpose`] — fixed-order bundling of several bound pairs into one
//!   trace;
//! * [`CleanupMemory`] — nearest-prototype recognition with an explicit
//!   acceptance threshold: a noisy code is replaced by the **exact stored**
//!   prototype (never an interpolation, so cleanup is idempotent), and
//!   uncorrelated input is rejected rather than silently snapped;
//! * with the `fusion` feature, [`fuse_observations`] — repeated noisy
//!   observations of one code fused per lane with `scirust-signal`'s
//!   denoisers (Kalman-RTS smoothing, Hampel impulse rejection) before the
//!   cleanup snap.
//!
//! Everything is deterministic: fixed-order scalar reductions, no RNG, and
//! `f32::total_cmp` with an ascending-id tie-break wherever an order is
//! chosen.

use crate::RetrievalError;
use crate::vector;

/// The HRR involution `a†`: `a†[j] = a[(n − j) mod n]`.
///
/// Used to turn convolution into (approximate) unbinding:
/// `circular_correlation(a, b) = a† ⊛ b`.
pub fn involution(a: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    for (j, o) in out.iter_mut().enumerate()
    {
        *o = a[(n - j) % n];
    }
    out
}

/// Circular convolution `(a ⊛ b)[k] = Σⱼ a[j] · b[(k − j) mod n]`, the HRR
/// binding operator, summed in ascending `j` for bit-reproducibility.
///
/// Returns [`RetrievalError::DimMismatch`] if the lengths differ.
pub fn circular_convolution(a: &[f32], b: &[f32]) -> Result<Vec<f32>, RetrievalError> {
    if a.len() != b.len()
    {
        return Err(RetrievalError::DimMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let n = a.len();
    let mut out = vec![0.0f32; n];
    for (k, o) in out.iter_mut().enumerate()
    {
        let mut acc = 0.0f32;
        for (j, &aj) in a.iter().enumerate()
        {
            acc += aj * b[(k + n - j) % n];
        }
        *o = acc;
    }
    Ok(out)
}

/// Circular correlation `a† ⊛ b` — the HRR **approximate unbinding**: for a
/// trace `t = a ⊛ y` (possibly superposed with other bound pairs),
/// `circular_correlation(a, t)` is a noisy estimate of `y`, to be snapped by a
/// [`CleanupMemory`].
///
/// Returns [`RetrievalError::DimMismatch`] if the lengths differ.
pub fn circular_correlation(a: &[f32], b: &[f32]) -> Result<Vec<f32>, RetrievalError> {
    circular_convolution(&involution(a), b)
}

/// Fixed-order superposition (bundling): the lane-wise sum of `vs` in slice
/// order.
///
/// Returns [`RetrievalError::EmptyInput`] for an empty slice and
/// [`RetrievalError::DimMismatch`] if the lengths disagree.
pub fn superpose(vs: &[Vec<f32>]) -> Result<Vec<f32>, RetrievalError> {
    let Some(first) = vs.first()
    else
    {
        return Err(RetrievalError::EmptyInput);
    };
    let n = first.len();
    let mut out = vec![0.0f32; n];
    for v in vs
    {
        if v.len() != n
        {
            return Err(RetrievalError::DimMismatch {
                expected: n,
                got: v.len(),
            });
        }
        for (o, &x) in out.iter_mut().zip(v)
        {
            *o += x;
        }
    }
    Ok(out)
}

fn validate_finite(v: &[f32]) -> Result<(), RetrievalError> {
    if v.iter().all(|x| x.is_finite())
    {
        Ok(())
    }
    else
    {
        Err(RetrievalError::NonFiniteInput)
    }
}

/// A recognized code: the id of the nearest stored prototype, the **exact
/// stored** prototype itself, and the cosine score that won.
#[derive(Debug, Clone, PartialEq)]
pub struct Cleaned {
    /// Id of the recognized prototype.
    pub id: u64,
    /// The exact stored prototype — never an interpolation, so feeding it back
    /// through [`CleanupMemory::clean`] is a fixed point.
    pub prototype: Vec<f32>,
    /// Cosine similarity between the noisy input and the prototype.
    pub score: f32,
}

/// The VSA cleanup memory: nearest-prototype recognition with an explicit
/// acceptance threshold.
///
/// `clean` scans the stored prototypes in insertion order, picks the best
/// cosine score (`f32::total_cmp`, ties broken by ascending id), and accepts
/// only if that score is at least `threshold` — uncorrelated input is
/// *rejected* (`Ok(None)`), never silently snapped to an arbitrary prototype.
#[derive(Debug, Clone, Default)]
pub struct CleanupMemory {
    dim: usize,
    ids: Vec<u64>,
    prototypes: Vec<Vec<f32>>,
}

impl CleanupMemory {
    /// New empty cleanup memory over `dim`-dimensional codes.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ids: Vec::new(),
            prototypes: Vec::new(),
        }
    }

    /// The code dimension this memory expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored prototypes.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether no prototype is stored.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Store `prototype` under `id`. The vector is kept **exactly** as given
    /// (recognition is cosine, which is scale-invariant). Errors:
    /// [`RetrievalError::DimMismatch`], [`RetrievalError::NonFiniteInput`],
    /// and [`RetrievalError::ZeroVector`] (a zero prototype can never be
    /// recognized and would silently poison the codebook).
    pub fn insert(&mut self, id: u64, prototype: &[f32]) -> Result<(), RetrievalError> {
        if prototype.len() != self.dim
        {
            return Err(RetrievalError::DimMismatch {
                expected: self.dim,
                got: prototype.len(),
            });
        }
        validate_finite(prototype)?;
        // norm is non-negative, so `<= 0.0` means exactly zero (no float-eq lint).
        if vector::norm(prototype) <= 0.0
        {
            return Err(RetrievalError::ZeroVector);
        }
        self.ids.push(id);
        self.prototypes.push(prototype.to_vec());
        Ok(())
    }

    /// Recognize `noisy`: nearest stored prototype by cosine, accepted iff the
    /// score is at least `threshold`.
    ///
    /// Returns `Ok(None)` when the memory is empty or the best score falls
    /// below the threshold (rejection). Errors: non-finite `noisy` or
    /// `threshold` ([`RetrievalError::NonFiniteInput`]), wrong dimension
    /// ([`RetrievalError::DimMismatch`]).
    #[must_use = "the outcome says whether the input was recognized"]
    pub fn clean(&self, noisy: &[f32], threshold: f32) -> Result<Option<Cleaned>, RetrievalError> {
        if noisy.len() != self.dim
        {
            return Err(RetrievalError::DimMismatch {
                expected: self.dim,
                got: noisy.len(),
            });
        }
        validate_finite(noisy)?;
        if !threshold.is_finite()
        {
            return Err(RetrievalError::NonFiniteInput);
        }
        let mut best: Option<(f32, usize)> = None;
        for (i, proto) in self.prototypes.iter().enumerate()
        {
            let score = vector::cosine(noisy, proto);
            let better = match best
            {
                None => true,
                Some((bs, bi)) => match score.total_cmp(&bs)
                {
                    core::cmp::Ordering::Greater => true,
                    core::cmp::Ordering::Equal => self.ids[i] < self.ids[bi],
                    core::cmp::Ordering::Less => false,
                },
            };
            if better
            {
                best = Some((score, i));
            }
        }
        Ok(best.and_then(|(score, i)| {
            if score.total_cmp(&threshold).is_lt()
            {
                None
            }
            else
            {
                Some(Cleaned {
                    id: self.ids[i],
                    prototype: self.prototypes[i].clone(),
                    score,
                })
            }
        }))
    }
}

/// How repeated noisy observations of one code are fused into a single
/// estimate before the cleanup snap (`scirust-signal` backends).
#[cfg(feature = "fusion")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FusionStrategy {
    /// Naive per-lane mean — the baseline the toolkit strategies must beat to
    /// justify themselves.
    Mean,
    /// Per-lane local-level Kalman filter + RTS smoother
    /// (`scirust_signal::denoise::adaptive::kalman_smooth`), then the smoothed
    /// series' mean. Right for broadband observation noise. Both variances
    /// must be finite and strictly positive.
    Kalman {
        /// Process variance of the local-level model.
        process_var: f64,
        /// Observation-noise variance.
        meas_var: f64,
    },
    /// Per-lane Hampel impulse rejection
    /// (`scirust_signal::denoise::rank::hampel_filter`) followed by the same
    /// Kalman-RTS pass — for observation streams with occasional gross
    /// corruption, the regime where a plain mean fails. `half_window ≥ 1`;
    /// `n_sigma` finite and strictly positive.
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

#[cfg(feature = "fusion")]
impl FusionStrategy {
    fn validate(&self) -> Result<(), RetrievalError> {
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
            Err(RetrievalError::InvalidParameter {
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
        // Fixed left-to-right mean of the denoised series (for a static code
        // the RTS-smoothed series is nearly constant; the mean is its natural
        // single-value reduction and is what `Mean` degenerates to).
        let mut acc = 0.0f64;
        for &x in &denoised
        {
            acc += x;
        }
        acc / denoised.len() as f64
    }
}

/// Fuse repeated noisy observations of one code into a single estimate: per
/// lane, the observation sequence is a time series handed to
/// `scirust-signal`'s denoisers.
///
/// Errors: an empty observation slice ([`RetrievalError::EmptyInput`]),
/// observations of differing lengths ([`RetrievalError::DimMismatch`]), a
/// non-finite observation ([`RetrievalError::NonFiniteInput`]), or invalid
/// strategy parameters ([`RetrievalError::InvalidParameter`]).
#[cfg(feature = "fusion")]
pub fn fuse_observations(
    observations: &[Vec<f32>],
    strategy: FusionStrategy,
) -> Result<Vec<f32>, RetrievalError> {
    strategy.validate()?;
    let Some(first) = observations.first()
    else
    {
        return Err(RetrievalError::EmptyInput);
    };
    let dim = first.len();
    for obs in observations
    {
        if obs.len() != dim
        {
            return Err(RetrievalError::DimMismatch {
                expected: dim,
                got: obs.len(),
            });
        }
        validate_finite(obs)?;
    }

    let mut fused = vec![0.0f32; dim];
    let mut series = vec![0.0f64; observations.len()];
    for (lane, out) in fused.iter_mut().enumerate()
    {
        for (i, obs) in observations.iter().enumerate()
        {
            series[i] = f64::from(obs[lane]);
        }
        *out = strategy.fuse_lane(&series) as f32;
    }
    Ok(fused)
}

#[cfg(feature = "fusion")]
impl CleanupMemory {
    /// Fuse repeated noisy observations with `scirust-signal`'s toolkit, then
    /// snap the fused estimate to the nearest stored prototype — the
    /// composition of [`fuse_observations`] and [`CleanupMemory::clean`], with
    /// the same error and threshold semantics.
    #[must_use = "the outcome says whether the observations were recognized"]
    pub fn clean_observations(
        &self,
        observations: &[Vec<f32>],
        strategy: FusionStrategy,
        threshold: f32,
    ) -> Result<Option<Cleaned>, RetrievalError> {
        let fused = fuse_observations(observations, strategy)?;
        self.clean(&fused, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn involution_reverses_all_but_the_first_lane() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(involution(&a), vec![1.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn convolution_with_the_identity_is_the_identity() {
        // e0 = [1, 0, 0, 0] is the unit of circular convolution.
        let e0 = [1.0, 0.0, 0.0, 0.0];
        let x = [0.5, -1.0, 2.0, 0.25];
        assert_eq!(circular_convolution(&e0, &x).unwrap(), x.to_vec());
        assert_eq!(circular_convolution(&x, &e0).unwrap(), x.to_vec());
    }

    #[test]
    fn convolution_matches_a_hand_computed_case() {
        // n = 3: (a ⊛ b)[k] = Σⱼ a[j]·b[(k−j) mod 3].
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // k=0: 1·4 + 2·6 + 3·5 = 31; k=1: 1·5 + 2·4 + 3·6 = 31;
        // k=2: 1·6 + 2·5 + 3·4 = 28.
        assert_eq!(
            circular_convolution(&a, &b).unwrap(),
            vec![31.0, 31.0, 28.0]
        );
    }

    #[test]
    fn correlation_approximately_unbinds() {
        // With unit-norm quasi-orthogonal vectors, a† ⊛ (a ⊛ y) ≈ y. Use an
        // exact small case: for a = e1 (a rotation), unbinding is exact.
        let e1 = [0.0, 1.0, 0.0, 0.0];
        let y = [0.5, -1.0, 2.0, 0.25];
        let bound = circular_convolution(&e1, &y).unwrap();
        let recovered = circular_correlation(&e1, &bound).unwrap();
        assert_eq!(recovered, y.to_vec());
    }

    #[test]
    fn dimension_mismatch_is_typed() {
        assert_eq!(
            circular_convolution(&[1.0, 2.0], &[1.0]),
            Err(RetrievalError::DimMismatch {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(superpose(&[]), Err(RetrievalError::EmptyInput));
        assert_eq!(
            superpose(&[vec![1.0], vec![1.0, 2.0]]),
            Err(RetrievalError::DimMismatch {
                expected: 1,
                got: 2
            })
        );
    }

    #[test]
    fn cleanup_recognizes_rejects_and_is_idempotent() {
        let mut mem = CleanupMemory::new(3);
        mem.insert(1, &[1.0, 0.0, 0.0]).unwrap();
        mem.insert(2, &[0.0, 1.0, 0.0]).unwrap();

        let hit = mem.clean(&[0.9, 0.1, 0.0], 0.5).unwrap().unwrap();
        assert_eq!(hit.id, 1);
        assert_eq!(hit.prototype, vec![1.0, 0.0, 0.0]);

        // Idempotence: the exact stored prototype is a fixed point.
        let again = mem.clean(&hit.prototype, 0.5).unwrap().unwrap();
        assert_eq!(again.id, 1);
        assert_eq!(again.prototype, hit.prototype);
        assert!((again.score - 1.0).abs() < 1e-6);

        // Rejection: orthogonal input falls below the threshold.
        assert_eq!(mem.clean(&[0.0, 0.0, 1.0], 0.5).unwrap(), None);
        // Empty memory rejects without error.
        assert_eq!(
            CleanupMemory::new(3).clean(&[1.0, 0.0, 0.0], 0.0).unwrap(),
            None
        );
    }

    #[test]
    fn cleanup_ties_break_by_ascending_id() {
        let mut mem = CleanupMemory::new(2);
        // Same direction (cosine ties exactly), inserted high id first.
        mem.insert(42, &[1.0, 0.0]).unwrap();
        mem.insert(7, &[2.0, 0.0]).unwrap();
        let hit = mem.clean(&[3.0, 0.0], 0.5).unwrap().unwrap();
        assert_eq!(hit.id, 7, "tie must go to the smaller id");
    }

    #[test]
    fn cleanup_insert_rejects_bad_prototypes() {
        let mut mem = CleanupMemory::new(2);
        assert_eq!(
            mem.insert(1, &[1.0]),
            Err(RetrievalError::DimMismatch {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            mem.insert(1, &[f32::NAN, 0.0]),
            Err(RetrievalError::NonFiniteInput)
        );
        assert_eq!(mem.insert(1, &[0.0, 0.0]), Err(RetrievalError::ZeroVector));
        assert!(mem.is_empty());
    }

    #[test]
    fn clean_rejects_bad_inputs() {
        let mut mem = CleanupMemory::new(2);
        mem.insert(1, &[1.0, 0.0]).unwrap();
        assert_eq!(
            mem.clean(&[1.0], 0.5),
            Err(RetrievalError::DimMismatch {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            mem.clean(&[f32::NAN, 0.0], 0.5),
            Err(RetrievalError::NonFiniteInput)
        );
        assert_eq!(
            mem.clean(&[1.0, 0.0], f32::NAN),
            Err(RetrievalError::NonFiniteInput)
        );
    }

    #[cfg(feature = "fusion")]
    mod fusion {
        use super::*;

        #[test]
        fn empty_and_ragged_observations_are_rejected() {
            assert_eq!(
                fuse_observations(&[], FusionStrategy::Mean),
                Err(RetrievalError::EmptyInput)
            );
            assert_eq!(
                fuse_observations(&[vec![1.0, 2.0], vec![1.0]], FusionStrategy::Mean),
                Err(RetrievalError::DimMismatch {
                    expected: 2,
                    got: 1
                })
            );
            assert_eq!(
                fuse_observations(&[vec![f32::INFINITY]], FusionStrategy::Mean),
                Err(RetrievalError::NonFiniteInput)
            );
        }

        #[test]
        fn invalid_strategy_parameters_are_rejected() {
            let obs = [vec![1.0, 2.0]];
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
                assert_eq!(
                    fuse_observations(&obs, strategy),
                    Err(RetrievalError::InvalidParameter {
                        reason: "fusion strategy parameters must be finite and positive",
                    }),
                    "{strategy:?}"
                );
            }
        }

        #[test]
        fn mean_of_identical_observations_is_the_observation() {
            let code = vec![0.5, -0.25, 2.0];
            let obs = [code.clone(), code.clone(), code.clone()];
            assert_eq!(fuse_observations(&obs, FusionStrategy::Mean).unwrap(), code);
        }
    }
}
