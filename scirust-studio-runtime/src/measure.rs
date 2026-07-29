//! Measurements taken *from* a finished trajectory, as opposed to
//! [`crate::execute_support`], which is about producing one.
//!
//! There is one thing in here so far, and it earns a module because two
//! capabilities need it and it is easy to get subtly wrong: recovering an
//! oscillation's period from samples. A laser diode ringing back to its
//! operating point and a generator rotor swinging about its equilibrium angle
//! are the same measurement problem, and neither should be reimplemented
//! beside the other.

/// Where a series crosses a level, located by linear interpolation between
/// the two samples that straddle the crossing.
///
/// Interpolating rather than returning the sample index matters: snapping a
/// crossing to the nearest sample quantises every period measurement to the
/// step size, which for a run with a few thousand steps per period is a
/// larger error than everything the measurement is trying to see.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn crossing_times(t: &[f64], values: &[f64], level: f64) -> Vec<f64> {
    let mut crossings = Vec::new();
    for i in 1..values.len().min(t.len())
    {
        let (a, b) = (values[i - 1] - level, values[i] - level);
        if (a < 0.0 && b >= 0.0) || (a > 0.0 && b <= 0.0)
        {
            let span = b - a;
            let fraction = if span != 0.0 { -a / span } else { 0.0 };
            crossings.push(t[i - 1] + fraction * (t[i] - t[i - 1]));
        }
    }
    crossings
}

/// The oscillation period of `values` about `level`, averaged over the
/// *upward* crossings in the run's second half.
///
/// # Why upward crossings only
///
/// Successive crossings in *either* direction look like half-periods, and
/// treating them as such is wrong for any oscillation that is not symmetric
/// about `level`. A finite-amplitude oscillation in an asymmetric potential —
/// a rotor swinging about a loaded equilibrium, a laser ringing about its
/// operating point — acquires a second-harmonic term and a DC offset, so its
/// crossings alternate short-long-short-long. Averaging over an even number
/// of them cancels that; averaging over an odd number does not, and leaves a
/// bias that can exceed the physical effect being measured. (It did: this
/// function measured a rotor swing as *faster* than its small-signal limit,
/// which is the wrong side of a result that only ever goes one way.)
///
/// Crossings in one direction are one full period apart whatever the shape of
/// the waveform, so counting only those removes the failure mode rather than
/// making it rarer.
///
/// # Why the second half
///
/// Two reasons that happen to point the same way. For a decaying oscillation
/// the closed forms being compared against are usually *linearisations*, and
/// by the second half the amplitude is small enough for that to be an
/// excellent description; measuring across the whole run would fold in the
/// large-amplitude early cycles and then need a tolerance loose enough to
/// hide them. For an undamped oscillation the amplitude never changes, so
/// restricting the window costs nothing.
///
/// Returns `None` when the window holds fewer than three upward crossings:
/// two is a single period with no averaging at all, and one is not a
/// measurement.
pub(crate) fn period_from_second_half(t: &[f64], values: &[f64], level: f64) -> Option<f64> {
    let (first, last) = (*t.first()?, *t.last()?);
    let midpoint = first + (last - first) / 2.0;
    let late: Vec<f64> = upward_crossing_times(t, values, level)
        .into_iter()
        .filter(|c| *c >= midpoint)
        .collect();
    if late.len() < 3
    {
        return None;
    }
    let periods = (late.len() - 1) as f64;
    Some((late[late.len() - 1] - late[0]) / periods)
}

/// [`crossing_times`], restricted to crossings where the series is rising.
pub(crate) fn upward_crossing_times(t: &[f64], values: &[f64], level: f64) -> Vec<f64> {
    let mut crossings = Vec::new();
    for i in 1..values.len().min(t.len())
    {
        let (a, b) = (values[i - 1] - level, values[i] - level);
        if a < 0.0 && b >= 0.0
        {
            let span = b - a;
            let fraction = if span != 0.0 { -a / span } else { 0.0 };
            crossings.push(t[i - 1] + fraction * (t[i] - t[i - 1]));
        }
    }
    crossings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossings_are_interpolated_not_snapped_to_samples() {
        // A straight line from -1 to 1 across [0, 2] crosses zero at t = 1,
        // which is not a sample.
        let crossings = crossing_times(&[0.0, 2.0], &[-1.0, 1.0], 0.0);
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0] - 1.0).abs() < 1e-15, "{}", crossings[0]);
    }

    #[test]
    fn a_level_that_is_not_zero_is_honoured() {
        let crossings = crossing_times(&[0.0, 1.0], &[4.0, 6.0], 5.0);
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn a_series_that_never_crosses_has_no_crossings() {
        assert!(crossing_times(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], 0.0).is_empty());
    }

    /// The measurement must recover a known period to far better than the
    /// step size, or the interpolation is not doing its job.
    #[test]
    fn a_sampled_sine_gives_back_its_own_period() {
        let period = 0.7;
        let omega = 2.0 * std::f64::consts::PI / period;
        let h = period / 37.0; // deliberately not a round divisor
        let n = 400;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
        let v: Vec<f64> = t.iter().map(|t| (omega * t).sin()).collect();
        let measured = period_from_second_half(&t, &v, 0.0).expect("many crossings");
        let error = (measured - period).abs() / period;
        assert!(error < 1e-4, "measured {measured}, error {error:e}");
        // And far better than simply snapping to the nearest sample would
        // have given, which is the whole reason for interpolating.
        assert!(error < h / period / 10.0);
    }

    /// The case that motivated counting one direction only: an oscillation
    /// offset from the level it is measured about. Its crossings alternate
    /// short-long, so averaging over half-periods is biased by the offset
    /// while averaging over full periods is exact.
    #[test]
    fn an_offset_oscillation_still_gives_its_exact_period() {
        let period = 0.7;
        let omega = 2.0 * std::f64::consts::PI / period;
        let h = period / 37.0;
        let n = 400;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
        // Sits 0.3 above the level it is measured about: the two crossings
        // per cycle are 0.403 and 0.297 of a period apart, not 0.35 each.
        let v: Vec<f64> = t.iter().map(|t| 0.3 + (omega * t).sin()).collect();

        let measured = period_from_second_half(&t, &v, 0.0).expect("many crossings");
        assert!(
            (measured - period).abs() / period < 1e-4,
            "measured {measured}"
        );

        // And the biased alternative really would have been wrong, so this
        // test is not asserting a distinction without a difference. The
        // crossings here alternate 0.597 and 0.403 of a period apart, so an
        // ODD number of them — three, below — carries the full bias, while an
        // even number cancels it. That an even count happens to come out
        // right is exactly what makes the bug intermittent rather than
        // obvious.
        let all = crossing_times(&t, &v, 0.0);
        let late: Vec<f64> = all.into_iter().filter(|c| *c >= t[n / 2]).collect();
        let over_three_halves = 2.0 * (late[3] - late[0]) / 3.0;
        assert!(
            (over_three_halves - period).abs() / period > 1e-2,
            "the half-period average over an odd count came out at {over_three_halves}, so this \
             test proves nothing"
        );
        let over_four_halves = 2.0 * (late[4] - late[0]) / 4.0;
        assert!(
            (over_four_halves - period).abs() / period < 1e-4,
            "an even count should have cancelled the bias, but gave {over_four_halves}"
        );
    }

    #[test]
    fn upward_crossings_are_half_of_all_crossings_for_a_sine() {
        let t: Vec<f64> = (0..400).map(|i| i as f64 * 0.01).collect();
        let v: Vec<f64> = t.iter().map(|t| (10.0 * t).sin()).collect();
        let all = crossing_times(&t, &v, 0.0).len();
        let up = upward_crossing_times(&t, &v, 0.0).len();
        assert!(all == 2 * up || all == 2 * up - 1, "all {all}, up {up}");
    }

    #[test]
    fn too_few_crossings_in_the_window_is_none() {
        let t: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let v = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        assert_eq!(period_from_second_half(&t, &v, 0.0), None);
    }

    #[test]
    fn an_empty_run_is_none_rather_than_a_panic() {
        assert_eq!(period_from_second_half(&[], &[], 0.0), None);
    }
}
