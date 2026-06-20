//! Shewhart control chart with the Western Electric run rules.

use serde::{Deserialize, Serialize};

/// A control chart calibrated on in-control data (center line and σ).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ControlChart {
    pub center: f64,
    pub sigma: f64,
}

impl ControlChart {
    /// Estimate the center and σ from an in-control reference sample.
    pub fn from_samples(data: &[f64]) -> Self {
        let n = data.len().max(1) as f64;
        let center = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|&x| (x - center).powi(2)).sum::<f64>() / n;
        Self {
            center,
            sigma: var.sqrt(),
        }
    }

    /// Signed number of σ a value sits from the center.
    pub fn z(&self, x: f64) -> f64 {
        if self.sigma <= 0.0
        {
            0.0
        }
        else
        {
            (x - self.center) / self.sigma
        }
    }
}

/// Which Western Electric rule fired (evaluated at the latest point).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WesternElectric {
    /// Rule 1: one point beyond 3σ.
    Beyond3Sigma,
    /// Rule 2: two of three consecutive points beyond 2σ on the same side.
    TwoOfThreeBeyond2Sigma,
    /// Rule 3: four of five consecutive points beyond 1σ on the same side.
    FourOfFiveBeyond1Sigma,
    /// Rule 4: eight consecutive points on the same side of the center.
    EightOnOneSide,
}

/// Evaluate the Western Electric rules at the last point of `points` (most
/// recent last). Returns the highest-priority rule that fires, if any.
pub fn western_electric(chart: &ControlChart, points: &[f64]) -> Option<WesternElectric> {
    if points.is_empty()
    {
        return None;
    }
    let z: Vec<f64> = points.iter().map(|&x| chart.z(x)).collect();
    let last = *z.last().unwrap();

    // Rule 1.
    if last.abs() > 3.0
    {
        return Some(WesternElectric::Beyond3Sigma);
    }
    // Rule 2: 2 of last 3 beyond 2σ same side.
    if z.len() >= 3
    {
        let w = &z[z.len() - 3..];
        for side in [1.0, -1.0]
        {
            if w.iter().filter(|&&v| v * side > 2.0).count() >= 2 && last * side > 2.0
            {
                return Some(WesternElectric::TwoOfThreeBeyond2Sigma);
            }
        }
    }
    // Rule 3: 4 of last 5 beyond 1σ same side.
    if z.len() >= 5
    {
        let w = &z[z.len() - 5..];
        for side in [1.0, -1.0]
        {
            if w.iter().filter(|&&v| v * side > 1.0).count() >= 4 && last * side > 1.0
            {
                return Some(WesternElectric::FourOfFiveBeyond1Sigma);
            }
        }
    }
    // Rule 4: 8 consecutive on one side.
    if z.len() >= 8
    {
        let w = &z[z.len() - 8..];
        if w.iter().all(|&v| v > 0.0) || w.iter().all(|&v| v < 0.0)
        {
            return Some(WesternElectric::EightOnOneSide);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrates_center_and_sigma() {
        let chart = ControlChart::from_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((chart.center - 3.0).abs() < 1e-9);
        assert!((chart.sigma - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn rule1_flags_a_gross_outlier() {
        let chart = ControlChart {
            center: 0.0,
            sigma: 1.0,
        };
        let pts = [0.1, -0.2, 0.0, 3.5];
        assert_eq!(
            western_electric(&chart, &pts),
            Some(WesternElectric::Beyond3Sigma)
        );
    }

    #[test]
    fn rule4_flags_a_sustained_shift() {
        let chart = ControlChart {
            center: 0.0,
            sigma: 1.0,
        };
        // Eight small but all-positive points — a mean shift Shewhart-rule-1 misses.
        let pts = [0.3; 8];
        assert_eq!(
            western_electric(&chart, &pts),
            Some(WesternElectric::EightOnOneSide)
        );
    }

    #[test]
    fn in_control_data_is_quiet() {
        let chart = ControlChart {
            center: 0.0,
            sigma: 1.0,
        };
        // Alternating around the mean, none beyond limits.
        let pts = [0.5, -0.4, 0.3, -0.6, 0.2, -0.5, 0.4, -0.3];
        assert_eq!(western_electric(&chart, &pts), None);
    }
}
