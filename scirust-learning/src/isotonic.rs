//! Deterministic isotonic (monotone) regression via Pool-Adjacent-Violators.
//!
//! A companion to the linear/robust regressors in [`crate::robust_regression`]:
//! where those fit an affine map, [`IsotonicRegression`] fits the best
//! *monotone* step function to one-dimensional data, then predicts by
//! continuous linear interpolation between its knots. It is the natural model
//! when the response is known to move monotonically with a scalar predictor but
//! the shape is nonlinear — for example recalibrating a degradation score into
//! a remaining-useful-life target whose profile is flat then declining.
//!
//! # Algorithm
//!
//! The classic Pool-Adjacent-Violators Algorithm (PAVA) computes, in `O(n)`
//! after an `O(n log n)` sort, the weighted-least-squares monotone fit — the
//! unique minimiser of `Σ wᵢ (ŷᵢ − yᵢ)²` subject to `ŷ` non-decreasing (or
//! non-increasing) in the predictor. Points sharing a predictor value are
//! pooled into one weighted block first (a tie in `x` forces one fitted value),
//! then a single left-to-right stack pass merges any adjacent blocks that
//! violate monotonicity.
//!
//! # Determinism contract
//!
//! No RNG, no hidden state. Points are ordered by `f64::total_cmp` on the
//! predictor with the original index as a total tie-break, so the fit is
//! invariant to input row order (verified by test) and bit-identical across
//! runs. A non-increasing fit is obtained by negating the response, fitting
//! non-decreasing, and negating the levels back — the same code path, so both
//! directions are equally deterministic.
//!
//! # Scope honesty
//!
//! Isotonic regression is a shape constraint, not a robust estimator: a single
//! grossly corrupted response still enters its block's weighted mean. It buys
//! nonlinearity under a monotonicity assumption; it does not buy contamination
//! resistance. Predictions are clamped to the fitted response range outside the
//! training predictor span (no extrapolation of the trend).

use core::fmt;

/// Direction of the monotonicity constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonotoneDirection {
    /// The fitted response never decreases as the predictor increases.
    NonDecreasing,
    /// The fitted response never increases as the predictor increases.
    NonIncreasing,
}

/// A fitted monotone regressor: a set of `(predictor, level)` knots with a
/// monotone level sequence, evaluated by clamped linear interpolation.
#[derive(Clone, Debug, PartialEq)]
pub struct IsotonicRegression {
    /// Distinct predictor values in ascending order (strictly increasing).
    knots_x: Vec<f64>,
    /// Fitted response level at each knot, monotone per [`Self::direction`].
    knots_y: Vec<f64>,
    direction: MonotoneDirection,
}

/// Errors returned by [`IsotonicRegression::fit`].
#[derive(Clone, Debug, PartialEq)]
pub enum IsotonicRegressionError {
    /// No samples were supplied.
    EmptyDataset,
    /// The predictor and response slices have different lengths.
    LengthMismatch {
        /// Number of predictor values.
        predictors: usize,
        /// Number of response values.
        responses: usize,
    },
    /// The optional weight slice length differs from the sample count.
    WeightLengthMismatch {
        /// Number of samples.
        samples: usize,
        /// Number of weights.
        weights: usize,
    },
    /// A predictor value is not finite.
    NonFinitePredictor {
        /// Index of the offending predictor.
        index: usize,
    },
    /// A response value is not finite.
    NonFiniteResponse {
        /// Index of the offending response.
        index: usize,
    },
    /// A weight is not finite.
    NonFiniteWeight {
        /// Index of the offending weight.
        index: usize,
    },
    /// A weight is negative.
    NegativeWeight {
        /// Index of the offending weight.
        index: usize,
    },
    /// The supplied weights sum to zero (or less), leaving nothing to fit.
    NonPositiveWeightTotal,
}

impl fmt::Display for IsotonicRegressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyDataset =>
            {
                write!(formatter, "isotonic regression needs at least one sample")
            },
            Self::LengthMismatch {
                predictors,
                responses,
            } => write!(
                formatter,
                "predictor/response length mismatch: {predictors} predictors, {responses} responses"
            ),
            Self::WeightLengthMismatch { samples, weights } => write!(
                formatter,
                "weight length mismatch: {samples} samples, {weights} weights"
            ),
            Self::NonFinitePredictor { index } =>
            {
                write!(formatter, "predictor at index {index} is not finite")
            },
            Self::NonFiniteResponse { index } =>
            {
                write!(formatter, "response at index {index} is not finite")
            },
            Self::NonFiniteWeight { index } =>
            {
                write!(formatter, "weight at index {index} is not finite")
            },
            Self::NegativeWeight { index } =>
            {
                write!(formatter, "weight at index {index} is negative")
            },
            Self::NonPositiveWeightTotal =>
            {
                write!(formatter, "weights sum to zero; nothing to fit")
            },
        }
    }
}

impl std::error::Error for IsotonicRegressionError {}

/// A weighted, x-sorted block used during pool-adjacent-violators merging.
///
/// `sum_wy` accumulates `Σ wᵢ · sign · yᵢ` (the response is pre-signed so the
/// fit is always non-decreasing internally); `value` is `sum_wy / sum_w`.
#[derive(Clone, Copy)]
struct Block {
    /// First distinct-predictor index this block covers.
    start: usize,
    sum_wy: f64,
    sum_w: f64,
}

impl Block {
    fn value(&self) -> f64 {
        self.sum_wy / self.sum_w
    }
}

impl IsotonicRegression {
    /// Fits a monotone regressor to `predictors → responses`.
    ///
    /// `weights` are optional per-sample non-negative weights (each sample
    /// counts once when `None`). Samples sharing a predictor value are pooled
    /// before the monotone fit.
    ///
    /// # Errors
    ///
    /// Returns an [`IsotonicRegressionError`] when the dataset is empty, the
    /// slice lengths disagree, any value is non-finite, a weight is negative,
    /// or the weights sum to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use scirust_learning::{IsotonicRegression, MonotoneDirection};
    /// // A monotone-with-noise relationship.
    /// let x = [1.0, 2.0, 3.0, 4.0];
    /// let y = [1.0, 3.0, 2.0, 4.0];
    /// let fit = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();
    /// // The 3.0 / 2.0 inversion is pooled to their mean 2.5, restoring monotonicity.
    /// assert!(fit.predict(2.0) <= fit.predict(3.0));
    /// ```
    pub fn fit(
        predictors: &[f64],
        responses: &[f64],
        direction: MonotoneDirection,
        weights: Option<&[f64]>,
    ) -> Result<Self, IsotonicRegressionError> {
        if predictors.is_empty()
        {
            return Err(IsotonicRegressionError::EmptyDataset);
        }

        if predictors.len() != responses.len()
        {
            return Err(IsotonicRegressionError::LengthMismatch {
                predictors: predictors.len(),
                responses: responses.len(),
            });
        }

        if let Some(weights) = weights
        {
            if weights.len() != predictors.len()
            {
                return Err(IsotonicRegressionError::WeightLengthMismatch {
                    samples: predictors.len(),
                    weights: weights.len(),
                });
            }
        }

        for (index, &value) in predictors.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(IsotonicRegressionError::NonFinitePredictor { index });
            }
        }

        for (index, &value) in responses.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(IsotonicRegressionError::NonFiniteResponse { index });
            }
        }

        if let Some(weights) = weights
        {
            for (index, &weight) in weights.iter().enumerate()
            {
                if !weight.is_finite()
                {
                    return Err(IsotonicRegressionError::NonFiniteWeight { index });
                }

                if weight < 0.0
                {
                    return Err(IsotonicRegressionError::NegativeWeight { index });
                }
            }
        }

        let weight_at = |index: usize| weights.map_or(1.0, |weights| weights[index]);

        let total_weight: f64 = (0..predictors.len()).map(weight_at).sum();

        // Validated weights are finite and non-negative, so the total is
        // finite and non-negative — `<= 0.0` isolates the all-zero case.
        if total_weight <= 0.0
        {
            return Err(IsotonicRegressionError::NonPositiveWeightTotal);
        }

        // Fit non-decreasing internally; a non-increasing request is the same
        // problem on the negated response, undone at the end.
        let sign = match direction
        {
            MonotoneDirection::NonDecreasing => 1.0,
            MonotoneDirection::NonIncreasing => -1.0,
        };

        // Deterministic order: ascending predictor, original index as tie-break.
        let mut order: Vec<usize> = (0..predictors.len()).collect();
        order.sort_by(|&a, &b| predictors[a].total_cmp(&predictors[b]).then(a.cmp(&b)));

        // Aggregate equal-predictor points into one weighted block each, so a
        // tie in `x` cannot receive two different fitted values.
        let mut distinct_x: Vec<f64> = Vec::new();
        let mut distinct_wy: Vec<f64> = Vec::new();
        let mut distinct_w: Vec<f64> = Vec::new();

        for &index in &order
        {
            let x = predictors[index];
            let w = weight_at(index);
            let wy = w * sign * responses[index];

            match distinct_x.last()
            {
                Some(&last) if last == x =>
                {
                    let position = distinct_x.len() - 1;
                    distinct_wy[position] += wy;
                    distinct_w[position] += w;
                },
                _ =>
                {
                    distinct_x.push(x);
                    distinct_wy.push(wy);
                    distinct_w.push(w);
                },
            }
        }

        // Pool-Adjacent-Violators: a left-to-right stack pass merging any top
        // pair whose levels violate the non-decreasing constraint.
        let mut stack: Vec<Block> = Vec::with_capacity(distinct_x.len());

        for (position, (&sum_wy, &sum_w)) in distinct_wy.iter().zip(&distinct_w).enumerate()
        {
            let mut block = Block {
                start: position,
                sum_wy,
                sum_w,
            };

            while let Some(&top) = stack.last()
            {
                if top.value() > block.value()
                {
                    block.start = top.start;
                    block.sum_wy += top.sum_wy;
                    block.sum_w += top.sum_w;
                    stack.pop();
                }
                else
                {
                    break;
                }
            }

            stack.push(block);
        }

        // Expand each block back to a level for every distinct predictor it
        // covers, undoing the internal sign so knots are in response units.
        let mut knots_x: Vec<f64> = Vec::with_capacity(distinct_x.len());
        let mut knots_y: Vec<f64> = Vec::with_capacity(distinct_x.len());

        for (block_index, block) in stack.iter().enumerate()
        {
            let end = stack
                .get(block_index + 1)
                .map_or(distinct_x.len(), |next| next.start);
            let level = sign * block.value();

            for &x in &distinct_x[block.start..end]
            {
                knots_x.push(x);
                knots_y.push(level);
            }
        }

        Ok(Self {
            knots_x,
            knots_y,
            direction,
        })
    }

    /// Predicts the fitted response at `query` by clamped linear interpolation.
    ///
    /// A query at or beyond the training predictor span returns the nearest
    /// endpoint level (no trend extrapolation); a non-finite query returns the
    /// lower-end level. Between knots the fit is linear.
    pub fn predict(&self, query: f64) -> f64 {
        if self.knots_x.len() == 1
        {
            return self.knots_y[0];
        }

        // First knot strictly greater than `query`; a non-finite query makes
        // every comparison false, yielding position 0 (the lower endpoint).
        let position = self.knots_x.partition_point(|&knot| knot <= query);

        if position == 0
        {
            return self.knots_y[0];
        }

        if position == self.knots_x.len()
        {
            return self.knots_y[self.knots_y.len() - 1];
        }

        let (x0, x1) = (self.knots_x[position - 1], self.knots_x[position]);
        let (y0, y1) = (self.knots_y[position - 1], self.knots_y[position]);
        let t = (query - x0) / (x1 - x0);

        y0 + t * (y1 - y0)
    }

    /// Predicts the fitted response for every value in `queries`.
    pub fn predict_slice(&self, queries: &[f64]) -> Vec<f64> {
        queries.iter().map(|&query| self.predict(query)).collect()
    }

    /// The monotonicity direction this model was fitted with.
    pub fn direction(&self) -> MonotoneDirection {
        self.direction
    }

    /// The fitted knots as `(predictors, levels)`; predictors are strictly
    /// ascending and levels are monotone per [`Self::direction`].
    pub fn knots(&self) -> (&[f64], &[f64]) {
        (&self.knots_x, &self.knots_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn assert_monotone(fit: &IsotonicRegression) {
        let (_, levels) = fit.knots();
        for window in levels.windows(2)
        {
            match fit.direction()
            {
                MonotoneDirection::NonDecreasing =>
                {
                    assert!(
                        window[0] <= window[1] + 1e-12,
                        "not non-decreasing: {window:?}"
                    )
                },
                MonotoneDirection::NonIncreasing =>
                {
                    assert!(
                        window[0] >= window[1] - 1e-12,
                        "not non-increasing: {window:?}"
                    )
                },
            }
        }
    }

    #[test]
    fn already_monotone_data_is_interpolated_through() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let fit = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();

        for (&xi, &yi) in x.iter().zip(&y)
        {
            assert!(approx_eq(fit.predict(xi), yi, 1e-12));
        }
        // Linear interpolation halfway between two knots.
        assert!(approx_eq(fit.predict(2.5), 2.5, 1e-12));
    }

    #[test]
    fn adjacent_violation_is_pooled_to_the_weighted_mean() {
        // The 3.0 / 2.0 inversion at x = 2, 3 must pool to 2.5.
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 2.0, 4.0];
        let fit = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();

        assert_monotone(&fit);
        assert!(approx_eq(fit.predict(2.0), 2.5, 1e-12));
        assert!(approx_eq(fit.predict(3.0), 2.5, 1e-12));
        assert!(approx_eq(fit.predict(1.0), 1.0, 1e-12));
        assert!(approx_eq(fit.predict(4.0), 4.0, 1e-12));
    }

    #[test]
    fn fully_decreasing_input_pools_to_a_single_mean_when_non_decreasing() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [4.0, 3.0, 2.0, 1.0];
        let fit = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();

        // The only non-decreasing fit is the flat global mean, 2.5.
        for &q in &[1.0, 2.0, 3.0, 4.0, 2.5]
        {
            assert!(approx_eq(fit.predict(q), 2.5, 1e-12));
        }
    }

    #[test]
    fn non_increasing_mirrors_non_decreasing_on_negated_response() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 2.0, 3.0, 1.0, 0.5];

        let up = {
            let negated: Vec<f64> = y.iter().map(|v| -v).collect();
            IsotonicRegression::fit(&x, &negated, MonotoneDirection::NonDecreasing, None).unwrap()
        };
        let down = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonIncreasing, None).unwrap();

        assert_monotone(&down);
        for &q in &[1.0, 1.5, 2.7, 3.3, 4.9, 5.0]
        {
            assert!(approx_eq(down.predict(q), -up.predict(q), 1e-12), "q = {q}");
        }
    }

    #[test]
    fn tied_predictors_receive_one_pooled_level() {
        // Two responses at x = 2 must share their mean (1.0), then stay monotone.
        let x = [1.0, 2.0, 2.0, 3.0];
        let y = [0.0, 0.0, 2.0, 3.0];
        let fit = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();

        assert_monotone(&fit);
        assert!(approx_eq(fit.predict(2.0), 1.0, 1e-12));
    }

    #[test]
    fn weights_pull_the_pooled_level() {
        // Same inversion as the pooling test, but the low response at x = 3 is
        // heavily weighted, so the pooled block drops below the unweighted 2.5.
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 2.0, 4.0];
        let w = [1.0, 1.0, 9.0, 1.0];
        let fit =
            IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, Some(&w)).unwrap();

        // Pool of x=2 (w1,y3) and x=3 (w9,y2): (1*3 + 9*2)/(1+9) = 2.1.
        assert!(approx_eq(fit.predict(2.0), 2.1, 1e-12));
        assert!(approx_eq(fit.predict(3.0), 2.1, 1e-12));
    }

    #[test]
    fn predictions_are_monotone_in_the_query() {
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        // A noisy increasing signal.
        let y: Vec<f64> = x.iter().map(|&xi| xi + 5.0 * ((xi * 1.7).sin())).collect();
        let fit = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();

        let mut previous = f64::NEG_INFINITY;
        let mut q = -2.0;
        while q < 42.0
        {
            let p = fit.predict(q);
            assert!(p >= previous - 1e-12, "prediction dropped at q = {q}");
            previous = p;
            q += 0.3;
        }
    }

    #[test]
    fn clamps_outside_the_training_span() {
        let x = [10.0, 20.0, 30.0];
        let y = [1.0, 2.0, 3.0];
        let fit = IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();

        assert!(approx_eq(fit.predict(-100.0), 1.0, 1e-12));
        assert!(approx_eq(fit.predict(1000.0), 3.0, 1e-12));
        assert!(approx_eq(fit.predict(f64::NAN), 1.0, 1e-12));
    }

    #[test]
    fn fit_is_invariant_to_input_order() {
        let x = [3.0, 1.0, 4.0, 2.0, 5.0];
        let y = [2.0, 1.0, 3.5, 2.0, 3.0];
        let forward =
            IsotonicRegression::fit(&x, &y, MonotoneDirection::NonDecreasing, None).unwrap();

        let mut xr = x;
        let mut yr = y;
        xr.reverse();
        yr.reverse();
        let reversed =
            IsotonicRegression::fit(&xr, &yr, MonotoneDirection::NonDecreasing, None).unwrap();

        assert_eq!(forward, reversed);
    }

    #[test]
    fn single_sample_predicts_that_value_everywhere() {
        let fit = IsotonicRegression::fit(&[7.0], &[3.0], MonotoneDirection::NonDecreasing, None)
            .unwrap();
        assert!(approx_eq(fit.predict(-1.0), 3.0, 1e-12));
        assert!(approx_eq(fit.predict(7.0), 3.0, 1e-12));
        assert!(approx_eq(fit.predict(100.0), 3.0, 1e-12));
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(
            IsotonicRegression::fit(&[], &[], MonotoneDirection::NonDecreasing, None),
            Err(IsotonicRegressionError::EmptyDataset)
        );
        assert_eq!(
            IsotonicRegression::fit(&[1.0, 2.0], &[1.0], MonotoneDirection::NonDecreasing, None),
            Err(IsotonicRegressionError::LengthMismatch {
                predictors: 2,
                responses: 1,
            })
        );
        assert!(matches!(
            IsotonicRegression::fit(
                &[1.0, f64::NAN],
                &[1.0, 2.0],
                MonotoneDirection::NonDecreasing,
                None
            ),
            Err(IsotonicRegressionError::NonFinitePredictor { index: 1 })
        ));
        assert!(matches!(
            IsotonicRegression::fit(
                &[1.0, 2.0],
                &[1.0, 2.0],
                MonotoneDirection::NonDecreasing,
                Some(&[1.0, -1.0])
            ),
            Err(IsotonicRegressionError::NegativeWeight { index: 1 })
        ));
        assert!(matches!(
            IsotonicRegression::fit(
                &[1.0, 2.0],
                &[1.0, 2.0],
                MonotoneDirection::NonDecreasing,
                Some(&[0.0, 0.0])
            ),
            Err(IsotonicRegressionError::NonPositiveWeightTotal)
        ));
    }
}
