//! Order analysis and order tracking for rotating machinery.
//!
//! Order analysis is used when a machine operates at variable speed.
//! Instead of analyzing vs. time or frequency, we analyze vs. **order**
//! (multiples of the shaft rotation frequency), making spectra invariant
//! to speed changes.

#[cfg(test)]
use std::f64::consts::PI;

/// Convert a tacho pulse train (rising-edge timestamps in seconds) to an
/// instantaneous RPM profile.
///
/// `tacho_times`: timestamps of each tacho pulse (e.g., once per revolution).
/// `n_points`: number of output RPM samples (evenly spaced in time).
///
/// Returns (time_values, rpm_values) where `rpm_values[i] = 60 / (tacho_times[i+1] - tacho_times[i])`.
pub fn rpm_profile(tacho_times: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>) {
    if tacho_times.len() < 2 || n_points < 2
    {
        return (vec![], vec![]);
    }
    let t_start = tacho_times[0];
    let t_end = tacho_times[tacho_times.len() - 1];
    let dt = (t_end - t_start) / (n_points - 1) as f64;

    let mut times = Vec::with_capacity(n_points);
    let mut rpms = Vec::with_capacity(n_points);

    let mut pulse_idx = 0usize;
    for i in 0..n_points
    {
        let t = t_start + i as f64 * dt;
        times.push(t);
        // Find the tacho pulse interval containing time t
        while pulse_idx + 1 < tacho_times.len() && tacho_times[pulse_idx + 1] < t
        {
            pulse_idx += 1;
        }
        let rpm = if pulse_idx + 1 < tacho_times.len()
        {
            let period = tacho_times[pulse_idx + 1] - tacho_times[pulse_idx];
            if period > f64::EPSILON
            {
                60.0 / period
            }
            else
            {
                0.0
            }
        }
        else
        {
            60.0 / (tacho_times[pulse_idx] - tacho_times[pulse_idx.saturating_sub(1)])
                .max(f64::EPSILON)
        };
        rpms.push(rpm);
    }
    (times, rpms)
}

/// Convert tacho timestamps to an array of accumulated shaft angle (in revolutions)
/// and corresponding timestamps.
///
/// Returns (timestamps, cumulative_revolutions).
pub fn tacho_to_rpm(tacho_times: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if tacho_times.is_empty()
    {
        return (vec![], vec![]);
    }
    let revs: Vec<f64> = (0..tacho_times.len()).map(|i| i as f64).collect();
    (tacho_times.to_vec(), revs)
}

/// Resample a signal from constant-time spacing to constant-angle spacing
/// using linear interpolation.
///
/// `signal`: the time-domain vibration/acceleration signal.
/// `time_values`: timestamps corresponding to each signal sample.
/// `tacho_times`: timestamps of each shaft revolution (one pulse per rev).
/// `samples_per_rev`: desired number of samples per revolution in the output.
///
/// Returns the constant-angle resampled signal.
pub fn resample_constant_angle(
    signal: &[f64],
    time_values: &[f64],
    tacho_times: &[f64],
    samples_per_rev: usize,
) -> Vec<f64> {
    if signal.is_empty() || tacho_times.len() < 2 || samples_per_rev == 0
    {
        return vec![];
    }

    let total_revs = (tacho_times.len() - 1) as f64;
    let n_out = (total_revs * samples_per_rev as f64) as usize + 1;
    let mut result = Vec::with_capacity(n_out);

    let mut pulse_idx = 0usize;
    for i in 0..n_out
    {
        let rev = i as f64 / samples_per_rev as f64;
        // Find which two tacho pulses bracket this revolution
        while pulse_idx + 2 < tacho_times.len() && ((pulse_idx + 1) as f64) < rev
        {
            pulse_idx += 1;
        }
        let rev_start = pulse_idx as f64;
        let rev_end = (pulse_idx + 1).min(tacho_times.len() - 1) as f64;
        let frac = if rev_end > rev_start
        {
            (rev - rev_start) / (rev_end - rev_start)
        }
        else
        {
            0.0
        };
        let t_target =
            tacho_times[pulse_idx] + frac * (tacho_times[pulse_idx + 1] - tacho_times[pulse_idx]);

        // Linear interpolation in time_values to get signal at t_target
        let val = interpolate_linear(signal, time_values, t_target);
        result.push(val);
    }
    result
}

/// Linear interpolation helper.
fn interpolate_linear(signal: &[f64], times: &[f64], t_target: f64) -> f64 {
    if signal.is_empty()
    {
        return 0.0;
    }
    if t_target <= times[0]
    {
        return signal[0];
    }
    if t_target >= times[times.len() - 1]
    {
        return signal[signal.len() - 1];
    }

    // Binary search for the right interval
    let mut lo = 0usize;
    let mut hi = times.len() - 1;
    while hi - lo > 1
    {
        let mid = (lo + hi) / 2;
        if times[mid] <= t_target
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    // Now t_target is between times[lo] and times[hi]
    let dt = times[hi] - times[lo];
    if dt < f64::EPSILON
    {
        return signal[lo];
    }
    let frac = (t_target - times[lo]) / dt;
    signal[lo] + frac * (signal[hi] - signal[lo])
}

/// Compute order spectrum: FFT-based magnitude spectrum indexed by order.
///
/// `angle_domain_signal`: the signal resampled to constant-angle increments.
/// `samples_per_rev`: number of samples per revolution.
///
/// Returns (orders, magnitudes) where `orders[k] = k / samples_per_rev`.
pub fn order_spectrum(angle_domain_signal: &[f64], samples_per_rev: usize) -> (Vec<f64>, Vec<f64>) {
    use crate::fft_real;

    let n = angle_domain_signal.len().next_power_of_two();
    let mut padded = angle_domain_signal.to_vec();
    padded.resize(n, 0.0);

    let spectrum = fft_real(&padded);
    let n_half = spectrum.len();

    let total_revs = n as f64 / samples_per_rev as f64;

    let orders: Vec<f64> = (0..n_half).map(|k| k as f64 / total_revs).collect();
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.mag()).collect();

    (orders, magnitudes)
}

/// Perform order tracking: given a vibration signal and tacho pulses, produce
/// an order spectrum that is invariant to speed changes.
///
/// `signal`: time-domain vibration/acceleration signal.
/// `sample_rate`: sample rate in Hz.
/// `tacho_times`: timestamps of each tacho pulse (once per rev).
/// `samples_per_rev`: output resolution in the angle domain.
/// `max_order`: maximum order to include in output.
///
/// Returns (orders, magnitudes) for orders 0..max_order.
pub fn order_track(
    signal: &[f64],
    sample_rate: f64,
    tacho_times: &[f64],
    samples_per_rev: usize,
    max_order: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Time vector
    let dt = 1.0 / sample_rate;
    let time_values: Vec<f64> = (0..signal.len()).map(|i| i as f64 * dt).collect();

    // Resample to angle domain
    let angle_signal = resample_constant_angle(signal, &time_values, tacho_times, samples_per_rev);

    // Compute order spectrum
    let (orders, mags) = order_spectrum(&angle_signal, samples_per_rev);

    // Truncate to max_order
    let max_idx = (max_order * samples_per_rev).min(orders.len() - 1);
    (orders[..=max_idx].to_vec(), mags[..=max_idx].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpm_profile_constant_speed() {
        // Tacho pulses at 0, 0.1, 0.2, 0.3... → 600 RPM (period = 0.1 s → 60/0.1 = 600)
        let tach: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
        let (_times, rpms) = rpm_profile(&tach, 10);
        for &rpm in &rpms
        {
            assert!((rpm - 600.0).abs() < 1e-6, "expected 600 RPM, got {}", rpm);
        }
    }

    #[test]
    fn test_resample_constant_angle() {
        // Simple linear signal, evenly spaced in time
        let signal: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let tach: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect(); // 10 revs
        let result = resample_constant_angle(&signal, &times, &tach, 10);
        // Should have ~10*10+1 = 101 samples
        assert!(result.len() >= 90);
        // Values should be monotonically increasing
        for w in result.windows(2)
        {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn test_order_track_constant_speed() {
        // A sine wave at exactly 3x shaft speed should show a peak at order 3
        let sample_rate = 1000.0;
        let shaft_hz = 10.0;
        let n_samples = 1000;
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / sample_rate).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * 3.0 * shaft_hz * ti).sin())
            .collect();
        // Tacho: once per revolution
        let tach: Vec<f64> = (0..=10).map(|i| i as f64 / shaft_hz).collect();

        let (orders, mags) = order_track(&signal, sample_rate, &tach, 100, 10);
        // Find the maximum magnitude order
        let max_idx = (1..mags.len())
            .max_by(|&a, &b| mags[a].partial_cmp(&mags[b]).unwrap())
            .unwrap();
        let detected_order = orders[max_idx];
        // Should be close to 3
        assert!(
            (detected_order - 3.0).abs() < 0.5,
            "detected order: {}",
            detected_order
        );
    }
}
