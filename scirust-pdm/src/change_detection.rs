use serde::{Deserialize, Serialize};

/// A detected change point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Index in the data stream where the change was detected
    pub index: usize,
    /// Timestamp (if available)
    pub timestamp: f64,
    /// Direction of change: +1 = increase, -1 = decrease
    pub direction: i8,
    /// Magnitude of the change
    pub magnitude: f64,
    /// Detection method
    pub method: String,
}

/// CUSUM (Cumulative Sum) change detector.
///
/// Detects shifts in the mean of a process by accumulating deviations
/// from a target value. Triggers when the cumulative sum exceeds a threshold.
///
/// Common in industrial process control (ISO 7870).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CUSUM {
    /// Target (expected) value
    pub target: f64,
    /// Allowed slack (tolerance band)
    pub slack: f64,
    /// Detection threshold
    pub threshold: f64,
    /// Positive cumulative sum
    pos_sum: f64,
    /// Negative cumulative sum
    neg_sum: f64,
    /// Sample index counter
    index: usize,
    /// Timestamp accumulator
    timestamp: f64,
}

impl CUSUM {
    pub fn new(target: f64, slack: f64, threshold: f64) -> Self {
        Self {
            target,
            slack,
            threshold,
            pos_sum: 0.0,
            neg_sum: 0.0,
            index: 0,
            timestamp: 0.0,
        }
    }

    /// Process one sample. Returns Some(ChangePoint) if a change is detected.
    ///
    /// `value`: new observation
    /// `dt`: time since previous sample (for timestamp tracking)
    pub fn update(&mut self, value: f64, dt: f64) -> Option<ChangePoint> {
        self.index += 1;
        self.timestamp += dt;
        let dev = value - self.target;

        // One-sided CUSUM with slack
        self.pos_sum = (self.pos_sum + dev - self.slack).max(0.0);
        self.neg_sum = (self.neg_sum - dev - self.slack).max(0.0);

        if self.pos_sum > self.threshold
        {
            let mag = self.pos_sum;
            self.pos_sum = 0.0; // reset after detection
            return Some(ChangePoint {
                index: self.index,
                timestamp: self.timestamp,
                direction: 1,
                magnitude: mag,
                method: "CUSUM+".to_string(),
            });
        }
        if self.neg_sum > self.threshold
        {
            let mag = self.neg_sum;
            self.neg_sum = 0.0;
            return Some(ChangePoint {
                index: self.index,
                timestamp: self.timestamp,
                direction: -1,
                magnitude: mag,
                method: "CUSUM-".to_string(),
            });
        }
        None
    }

    pub fn reset(&mut self) {
        self.pos_sum = 0.0;
        self.neg_sum = 0.0;
        self.index = 0;
        self.timestamp = 0.0;
    }

    pub fn positive_sum(&self) -> f64 {
        self.pos_sum
    }

    pub fn negative_sum(&self) -> f64 {
        self.neg_sum
    }
}

/// Page-Hinkley test for detecting abrupt changes in the mean of a sequence.
///
/// More robust than CUSUM for slowly drifting processes. Common in
/// predictive maintenance for detecting onset of degradation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHinkley {
    /// Expected value under null hypothesis (no change)
    pub target: f64,
    /// Forgetting factor (0..1, typically 0.95-0.99)
    pub gamma: f64,
    /// Detection threshold
    pub threshold: f64,
    /// Minimum number of observations before detection
    pub min_observations: usize,
    /// Cumulative sum
    cumulative_sum: f64,
    /// Running estimate of mean
    running_mean: f64,
    /// Minimum cumulative sum seen so far
    min_sum: f64,
    /// Observation count
    count: usize,
    /// Timestamp
    timestamp: f64,
}

impl PageHinkley {
    pub fn new(target: f64, gamma: f64, threshold: f64, min_observations: usize) -> Self {
        Self {
            target,
            gamma: gamma.clamp(0.0, 1.0),
            threshold,
            min_observations,
            cumulative_sum: 0.0,
            running_mean: target,
            min_sum: 0.0,
            count: 0,
            timestamp: 0.0,
        }
    }

    /// Process one sample.
    pub fn update(&mut self, value: f64, dt: f64) -> Option<ChangePoint> {
        self.count += 1;
        self.timestamp += dt;

        // Update running mean with forgetting
        self.running_mean = self.gamma * self.running_mean + (1.0 - self.gamma) * value;

        // Update cumulative sum
        self.cumulative_sum += value - self.target;
        self.min_sum = self.min_sum.min(self.cumulative_sum);

        // PH test: cumulative_sum - min_sum > threshold
        if self.count >= self.min_observations
        {
            let test_stat = self.cumulative_sum - self.min_sum;
            if test_stat > self.threshold
            {
                let direction = if self.running_mean > self.target
                {
                    1
                }
                else
                {
                    -1
                };
                let mag = test_stat;
                // Reset for continued detection
                self.min_sum = self.cumulative_sum;
                return Some(ChangePoint {
                    index: self.count,
                    timestamp: self.timestamp,
                    direction,
                    magnitude: mag,
                    method: "PageHinkley".to_string(),
                });
            }
        }
        None
    }

    pub fn reset(&mut self) {
        self.cumulative_sum = 0.0;
        self.running_mean = self.target;
        self.min_sum = 0.0;
        self.count = 0;
        self.timestamp = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusum_detect_increase() {
        let mut cusum = CUSUM::new(10.0, 0.5, 5.0);
        // Normal: values around 10 → no detection
        for _ in 0..10
        {
            assert!(cusum.update(10.0, 1.0).is_none());
        }
        // Shift to 12 → should detect after a few samples
        let mut detected = false;
        for _ in 0..10
        {
            if cusum.update(12.0, 1.0).is_some()
            {
                detected = true;
                break;
            }
        }
        assert!(detected, "CUSUM should detect upward shift");
    }

    #[test]
    fn test_cusum_detect_decrease() {
        let mut cusum = CUSUM::new(10.0, 0.5, 5.0);
        for _ in 0..10
        {
            cusum.update(10.0, 1.0);
        }
        let mut detected = false;
        for _ in 0..10
        {
            if let Some(cp) = cusum.update(8.0, 1.0)
            {
                detected = true;
                assert_eq!(cp.direction, -1);
                break;
            }
        }
        assert!(detected, "CUSUM should detect downward shift");
    }

    #[test]
    fn test_cusum_no_false_alarm() {
        let mut cusum = CUSUM::new(10.0, 0.5, 5.0);
        for _ in 0..100
        {
            let noisy = 10.0 + (rand::random::<f64>() - 0.5) * 0.5;
            assert!(
                cusum.update(noisy, 1.0).is_none(),
                "CUSUM false alarm on noisy data around target"
            );
        }
    }

    #[test]
    fn test_page_hinkley_detect_shift() {
        let mut ph = PageHinkley::new(0.0, 0.99, 50.0, 10);
        // Start at 0, then shift to 2
        for _ in 0..20
        {
            ph.update(0.0, 1.0);
        }
        let mut detected = false;
        for _ in 0..50
        {
            if ph.update(2.0, 1.0).is_some()
            {
                detected = true;
                break;
            }
        }
        assert!(detected, "Page-Hinkley should detect shift");
    }

    #[test]
    fn test_page_hinkley_no_false_alarm() {
        let mut ph = PageHinkley::new(0.0, 0.99, 50.0, 10);
        for _ in 0..200
        {
            let noise = (rand::random::<f64>() - 0.5) * 0.2;
            assert!(
                ph.update(noise, 1.0).is_none(),
                "Page-Hinkley false alarm on noise"
            );
        }
    }

    #[test]
    fn test_cusum_reset() {
        let mut cusum = CUSUM::new(10.0, 0.5, 5.0);
        cusum.update(15.0, 1.0);
        assert!(cusum.positive_sum() > 0.0);
        cusum.reset();
        assert_eq!(cusum.positive_sum(), 0.0);
        assert_eq!(cusum.index, 0);
    }
}
