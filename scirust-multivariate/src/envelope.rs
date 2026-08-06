//! Training-envelope gating: per-axis shift severity, attribution and a
//! three-state accept / reduced-confidence / abstain policy.
//!
//! Ported from the ITD research simulator's shift-aware OOD detector and
//! calibrated abstention policy (`itd_research.ood_shift`), validated against
//! the reference implementation as an oracle (`tests/envelope.rs`).
//!
//! ## Why per-axis, and why three states
//!
//! A single global distance (Mahalanobis radius, reconstruction error) says
//! *how far* an input sits from the calibration data but not *along which
//! feature axis* — and gating on it alone over-abstains on inputs that are
//! still perfectly usable (the reference measured ~85 % unnecessary abstention
//! on subtle-but-predictable shifts before this design). This module keeps the
//! decision transparent:
//!
//! * [`TrainingEnvelope::per_axis_deviation`] — the absolute standardized
//!   deviation `|x_d − mean_d| / scale_d` per feature axis;
//! * [`TrainingEnvelope::severity`] — the mean of the top-`k` deviations:
//!   robust to a single noisy axis, still monotone in a genuine multi-axis
//!   shift;
//! * [`TrainingEnvelope::attribution`] — the index of the most-shifted axis
//!   (which a global radius cannot provide);
//! * [`three_state_gate`] — accept below `s_low`, abstain at or above
//!   `s_high`, and in between predict with a monotone
//!   [`confidence_discount`] instead of the binary keep/drop that causes the
//!   over-abstention.
//!
//! For the complementary *global* view (a robust Mahalanobis radius over all
//! axes jointly), see [`crate::robust_geometry::RobustScatterModel`]; the two
//! are designed to be reported side by side.
//!
//! ## Determinism
//!
//! Fitting and scoring are single-pass, fixed-order `f64` reductions — no RNG,
//! no hashing — and reproduce the reference values to the oracle tolerance.

use core::fmt;

use serde::{Deserialize, Serialize};

/// Scale floor mirrored from the reference: a per-axis population standard
/// deviation below this is replaced by `1.0`, so a constant calibration axis
/// yields a zero deviation instead of a division by zero.
const SCALE_FLOOR: f64 = 1.0e-9;

/// Errors raised by envelope fitting, scoring and the separation diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    /// The calibration matrix has no rows or a zero-length first row.
    EmptyCalibration,
    /// A calibration row's length differs from the first row's.
    RaggedRow {
        /// Index of the offending row.
        row: usize,
        /// The expected (first-row) length.
        expected: usize,
        /// The offending row's actual length.
        found: usize,
    },
    /// A probe's length differs from the fitted dimension.
    DimensionMismatch {
        /// The fitted feature dimension.
        expected: usize,
        /// The probe's actual length.
        found: usize,
    },
    /// A calibration value, probe value, severity or band bound is `NaN` or
    /// `±∞`.
    NonFinite {
        /// Where the non-finite value was encountered.
        context: &'static str,
    },
    /// Two paired slices (scores and levels) differ in length.
    LengthMismatch {
        /// Length of the score slice.
        scores: usize,
        /// Length of the level slice.
        levels: usize,
    },
}

impl fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            EnvelopeError::EmptyCalibration =>
            {
                write!(f, "calibration matrix must have at least one non-empty row")
            },
            EnvelopeError::RaggedRow {
                row,
                expected,
                found,
            } => write!(
                f,
                "calibration row {row} has length {found}, expected {expected}"
            ),
            EnvelopeError::DimensionMismatch { expected, found } =>
            {
                write!(
                    f,
                    "probe has length {found}, fitted dimension is {expected}"
                )
            },
            EnvelopeError::NonFinite { context } =>
            {
                write!(f, "non-finite value in {context}")
            },
            EnvelopeError::LengthMismatch { scores, levels } => write!(
                f,
                "scores ({scores}) and levels ({levels}) must have the same length"
            ),
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Per-axis calibration statistics for envelope gating.
///
/// `scale` is the per-axis *population* standard deviation (denominator `n`,
/// matching the reference) with the [`SCALE_FLOOR`] substitution for
/// effectively constant axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingEnvelope {
    mean: Vec<f64>,
    scale: Vec<f64>,
    severity_k: usize,
}

impl TrainingEnvelope {
    /// Fits the envelope on calibration rows (one row per in-distribution
    /// sample).
    ///
    /// `severity_k` is the number of largest per-axis deviations averaged into
    /// the severity; it is clamped to `1..=dimension` exactly as in the
    /// reference, so `0` behaves as `1`.
    ///
    /// # Errors
    ///
    /// [`EnvelopeError::EmptyCalibration`] when there are no rows or the first
    /// row is empty; [`EnvelopeError::RaggedRow`] when row lengths differ;
    /// [`EnvelopeError::NonFinite`] when any calibration value is not finite.
    pub fn fit(rows: &[Vec<f64>], severity_k: usize) -> Result<Self, EnvelopeError> {
        let n_rows = rows.len();
        if n_rows == 0 || rows[0].is_empty()
        {
            return Err(EnvelopeError::EmptyCalibration);
        }
        let dimension = rows[0].len();
        for (row_index, row) in rows.iter().enumerate()
        {
            if row.len() != dimension
            {
                return Err(EnvelopeError::RaggedRow {
                    row: row_index,
                    expected: dimension,
                    found: row.len(),
                });
            }
            if row.iter().any(|v| !v.is_finite())
            {
                return Err(EnvelopeError::NonFinite {
                    context: "calibration matrix",
                });
            }
        }
        let n = n_rows as f64;
        let mut mean = vec![0.0_f64; dimension];
        for row in rows
        {
            for (axis, &value) in row.iter().enumerate()
            {
                mean[axis] += value;
            }
        }
        for value in &mut mean
        {
            *value /= n;
        }
        let mut scale = vec![0.0_f64; dimension];
        for row in rows
        {
            for (axis, &value) in row.iter().enumerate()
            {
                let centered = value - mean[axis];
                scale[axis] += centered * centered;
            }
        }
        for value in &mut scale
        {
            *value = (*value / n).sqrt();
            if *value < SCALE_FLOOR
            {
                *value = 1.0;
            }
        }
        Ok(Self {
            mean,
            scale,
            severity_k,
        })
    }

    /// The fitted feature dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// The fitted per-axis means.
    #[inline]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The fitted per-axis scales (population std with the constant-axis
    /// floor).
    #[inline]
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    fn validate_probe(&self, probe: &[f64]) -> Result<(), EnvelopeError> {
        if probe.len() != self.dimension()
        {
            return Err(EnvelopeError::DimensionMismatch {
                expected: self.dimension(),
                found: probe.len(),
            });
        }
        if probe.iter().any(|v| !v.is_finite())
        {
            return Err(EnvelopeError::NonFinite { context: "probe" });
        }
        Ok(())
    }

    /// Absolute standardized deviation per feature axis.
    ///
    /// # Errors
    ///
    /// [`EnvelopeError::DimensionMismatch`] or [`EnvelopeError::NonFinite`]
    /// on an invalid probe.
    pub fn per_axis_deviation(&self, probe: &[f64]) -> Result<Vec<f64>, EnvelopeError> {
        self.validate_probe(probe)?;
        Ok(probe
            .iter()
            .zip(self.mean.iter().zip(&self.scale))
            .map(|(&value, (&mean, &scale))| ((value - mean) / scale).abs())
            .collect())
    }

    /// Robust shift severity: the mean of the `k` largest per-axis deviations
    /// (`k` clamped to `1..=dimension`). Monotone in shift magnitude by
    /// construction.
    ///
    /// # Errors
    ///
    /// Propagates [`TrainingEnvelope::per_axis_deviation`]'s validation.
    pub fn severity(&self, probe: &[f64]) -> Result<f64, EnvelopeError> {
        let mut deviations = self.per_axis_deviation(probe)?;
        let k = self.severity_k.clamp(1, deviations.len());
        deviations.sort_by(f64::total_cmp);
        let top = &deviations[deviations.len() - k..];
        Ok(top.iter().sum::<f64>() / k as f64)
    }

    /// Index of the most-shifted feature axis (first axis on an exact tie,
    /// matching the reference's arg-max convention).
    ///
    /// # Errors
    ///
    /// Propagates [`TrainingEnvelope::per_axis_deviation`]'s validation.
    pub fn attribution(&self, probe: &[f64]) -> Result<usize, EnvelopeError> {
        let deviations = self.per_axis_deviation(probe)?;
        let mut best_axis = 0_usize;
        let mut best_value = deviations[0];
        for (axis, &value) in deviations.iter().enumerate().skip(1)
        {
            if value > best_value
            {
                best_axis = axis;
                best_value = value;
            }
        }
        Ok(best_axis)
    }
}

fn validate_band(severity: f64, s_low: f64, s_high: f64) -> Result<(), EnvelopeError> {
    if !severity.is_finite()
    {
        return Err(EnvelopeError::NonFinite {
            context: "severity",
        });
    }
    if !s_low.is_finite() || !s_high.is_finite()
    {
        return Err(EnvelopeError::NonFinite {
            context: "band bounds",
        });
    }
    Ok(())
}

/// Monotone confidence multiplier: `1` at or below `s_low`, linear down to `0`
/// at `s_high`, clipped to `[0, 1]`. A degenerate band (`s_high <= s_low`)
/// collapses to a hard step at `s_low` — higher severity never yields more
/// confidence.
///
/// # Errors
///
/// [`EnvelopeError::NonFinite`] when the severity or a band bound is not
/// finite.
pub fn confidence_discount(severity: f64, s_low: f64, s_high: f64) -> Result<f64, EnvelopeError> {
    validate_band(severity, s_low, s_high)?;
    if s_high <= s_low
    {
        return Ok(if severity <= s_low { 1.0 } else { 0.0 });
    }
    let fraction = (severity - s_low) / (s_high - s_low);
    Ok((1.0 - fraction).clamp(0.0, 1.0))
}

/// The three gate states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateState {
    /// Severity at or below `s_low`: predict at full confidence.
    Accept,
    /// Severity inside the band: predict, flagged, at discounted confidence.
    ReducedConfidence,
    /// Severity at or above `s_high`: no prediction.
    Abstain,
}

/// A gate decision: the state plus its confidence multiplier (`1` for accept,
/// `0` for abstain, the [`confidence_discount`] in between).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GateDecision {
    /// The three-state verdict.
    pub state: GateState,
    /// Confidence multiplier in `[0, 1]`.
    pub confidence: f64,
}

/// The three-state policy: accept below `s_low`, abstain at or above `s_high`,
/// reduced-confidence prediction in between.
///
/// # Errors
///
/// [`EnvelopeError::NonFinite`] when the severity or a band bound is not
/// finite.
pub fn three_state_gate(
    severity: f64,
    s_low: f64,
    s_high: f64,
) -> Result<GateDecision, EnvelopeError> {
    validate_band(severity, s_low, s_high)?;
    if severity <= s_low
    {
        return Ok(GateDecision {
            state: GateState::Accept,
            confidence: 1.0,
        });
    }
    if severity >= s_high
    {
        return Ok(GateDecision {
            state: GateState::Abstain,
            confidence: 0.0,
        });
    }
    Ok(GateDecision {
        state: GateState::ReducedConfidence,
        confidence: confidence_discount(severity, s_low, s_high)?,
    })
}

/// Rank agreement between a severity score and a *known* ordinal shift level:
/// the fraction of correctly ordered cross-level pairs (`1.0` perfect, `0.5`
/// chance; score ties count half). `Ok(None)` when no cross-level pair exists
/// (the reference returns `NaN` there).
///
/// This is the diagnostic the reference uses to accept a severity definition:
/// a usable severity must order a progressive degradation sweep correctly.
///
/// # Errors
///
/// [`EnvelopeError::LengthMismatch`] when the slices differ in length;
/// [`EnvelopeError::NonFinite`] when a score is not finite.
pub fn monotone_separation(scores: &[f64], levels: &[usize]) -> Result<Option<f64>, EnvelopeError> {
    if scores.len() != levels.len()
    {
        return Err(EnvelopeError::LengthMismatch {
            scores: scores.len(),
            levels: levels.len(),
        });
    }
    if scores.iter().any(|v| !v.is_finite())
    {
        return Err(EnvelopeError::NonFinite { context: "scores" });
    }
    let mut concordant = 0.0_f64;
    let mut total = 0_usize;
    for i in 0..scores.len()
    {
        for j in 0..scores.len()
        {
            if levels[i] < levels[j]
            {
                total += 1;
                if scores[i] < scores[j]
                {
                    concordant += 1.0;
                }
                else if scores[i] == scores[j]
                {
                    concordant += 0.5;
                }
            }
        }
    }
    if total == 0
    {
        return Ok(None);
    }
    Ok(Some(concordant / total as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 10.0, 5.0],
            vec![2.0, 12.0, 5.0],
            vec![3.0, 14.0, 5.0],
            vec![4.0, 16.0, 5.0],
        ]
    }

    #[test]
    fn fit_rejects_empty_ragged_and_non_finite_calibration() {
        assert_eq!(
            TrainingEnvelope::fit(&[], 3),
            Err(EnvelopeError::EmptyCalibration)
        );
        assert_eq!(
            TrainingEnvelope::fit(&[vec![]], 3),
            Err(EnvelopeError::EmptyCalibration)
        );
        let ragged = vec![vec![1.0, 2.0], vec![1.0]];
        assert_eq!(
            TrainingEnvelope::fit(&ragged, 3),
            Err(EnvelopeError::RaggedRow {
                row: 1,
                expected: 2,
                found: 1
            })
        );
        let poisoned = vec![vec![1.0, f64::NAN]];
        assert_eq!(
            TrainingEnvelope::fit(&poisoned, 3),
            Err(EnvelopeError::NonFinite {
                context: "calibration matrix"
            })
        );
    }

    #[test]
    fn constant_axis_gets_the_scale_floor_substitution() {
        let envelope = TrainingEnvelope::fit(&calibration(), 3).expect("valid calibration");
        assert_eq!(envelope.scale()[2], 1.0);
        // A probe on the constant axis deviates by |value - 5.0| exactly.
        let deviations = envelope
            .per_axis_deviation(&[2.5, 13.0, 7.0])
            .expect("valid probe");
        assert!((deviations[2] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn severity_is_monotone_in_a_growing_single_axis_shift() {
        let envelope = TrainingEnvelope::fit(&calibration(), 2).expect("valid calibration");
        let base = envelope.mean().to_vec();
        let mut previous = envelope.severity(&base).expect("valid probe");
        for step in 1..=5
        {
            let mut probe = base.clone();
            probe[1] += step as f64;
            let severity = envelope.severity(&probe).expect("valid probe");
            assert!(
                severity >= previous,
                "severity must not decrease as the shift grows"
            );
            previous = severity;
        }
    }

    #[test]
    fn attribution_returns_the_first_axis_on_an_exact_tie() {
        let envelope = TrainingEnvelope::fit(&[vec![0.0, 0.0], vec![2.0, 2.0], vec![4.0, 4.0]], 1)
            .expect("valid calibration");
        // Both axes have identical mean/scale; an equal shift ties exactly.
        assert_eq!(envelope.attribution(&[6.0, 6.0]).expect("valid probe"), 0);
    }

    #[test]
    fn probe_validation_is_typed() {
        let envelope = TrainingEnvelope::fit(&calibration(), 3).expect("valid calibration");
        assert_eq!(
            envelope.severity(&[1.0, 2.0]),
            Err(EnvelopeError::DimensionMismatch {
                expected: 3,
                found: 2
            })
        );
        assert_eq!(
            envelope.severity(&[1.0, 2.0, f64::INFINITY]),
            Err(EnvelopeError::NonFinite { context: "probe" })
        );
    }

    #[test]
    fn discount_is_monotone_and_band_edges_are_inclusive_as_documented() {
        let low = three_state_gate(1.5, 1.5, 3.0).expect("finite inputs");
        assert_eq!(low.state, GateState::Accept);
        assert_eq!(low.confidence, 1.0);
        let high = three_state_gate(3.0, 1.5, 3.0).expect("finite inputs");
        assert_eq!(high.state, GateState::Abstain);
        assert_eq!(high.confidence, 0.0);
        let mut previous = 1.0_f64;
        for step in 0..=10
        {
            let severity = 1.5 + 1.5 * step as f64 / 10.0;
            let discount = confidence_discount(severity, 1.5, 3.0).expect("finite inputs");
            assert!(discount <= previous, "discount must be non-increasing");
            previous = discount;
        }
    }

    #[test]
    fn degenerate_band_collapses_to_a_step() {
        assert_eq!(confidence_discount(1.9, 2.0, 2.0), Ok(1.0));
        assert_eq!(confidence_discount(2.1, 2.0, 2.0), Ok(0.0));
    }

    #[test]
    fn gate_rejects_non_finite_inputs() {
        assert_eq!(
            three_state_gate(f64::NAN, 1.0, 2.0),
            Err(EnvelopeError::NonFinite {
                context: "severity"
            })
        );
        assert_eq!(
            three_state_gate(1.0, f64::NEG_INFINITY, 2.0),
            Err(EnvelopeError::NonFinite {
                context: "band bounds"
            })
        );
    }

    #[test]
    fn separation_handles_perfect_reversed_and_undefined_orderings() {
        let perfect = monotone_separation(&[0.1, 0.2, 0.3], &[0, 1, 2]).expect("valid input");
        assert_eq!(perfect, Some(1.0));
        let reversed = monotone_separation(&[0.3, 0.2, 0.1], &[0, 1, 2]).expect("valid input");
        assert_eq!(reversed, Some(0.0));
        let undefined = monotone_separation(&[0.5, 0.6], &[1, 1]).expect("valid input");
        assert_eq!(undefined, None);
        assert_eq!(
            monotone_separation(&[0.5], &[1, 2]),
            Err(EnvelopeError::LengthMismatch {
                scores: 1,
                levels: 2
            })
        );
    }

    #[test]
    fn fit_and_scoring_are_deterministic() {
        let rows = calibration();
        let first = TrainingEnvelope::fit(&rows, 2).expect("valid calibration");
        let second = TrainingEnvelope::fit(&rows, 2).expect("valid calibration");
        assert_eq!(first, second);
        let probe = [2.5, 11.0, 6.0];
        assert_eq!(
            first.severity(&probe).expect("valid probe").to_bits(),
            second.severity(&probe).expect("valid probe").to_bits()
        );
    }
}
