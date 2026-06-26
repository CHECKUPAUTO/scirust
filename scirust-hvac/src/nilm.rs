//! NILM — Non-Intrusive Load Monitoring (energy disaggregation).
//!
//! From a single whole-building power meter, step changes in the aggregate
//! demand mark appliances switching on/off; matching each step's magnitude to a
//! library of appliance signatures disaggregates the load without per-appliance
//! sensors. Deterministic edge detection + nearest-signature matching.

use serde::{Deserialize, Serialize};

/// A detected load switching event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadEvent {
    /// Sample index of the step.
    pub index: usize,
    /// Power change (W); positive = turn-on, negative = turn-off.
    pub delta_w: f64,
    /// Matched appliance name, if a signature is within tolerance.
    pub appliance: Option<String>,
}

/// Disaggregate an aggregate `power` trace (W). A step of magnitude ≥ `min_step`
/// is an event, matched to the nearest appliance in `library` within `tol` W.
pub fn disaggregate(
    power: &[f64],
    library: &[(&str, f64)],
    min_step: f64,
    tol: f64,
) -> Vec<LoadEvent> {
    let mut events = Vec::new();
    for i in 1..power.len()
    {
        let d = power[i] - power[i - 1];
        if d.abs() < min_step
        {
            continue;
        }
        let appliance = library
            .iter()
            .map(|(name, w)| (name, (w - d.abs()).abs()))
            .filter(|(_, err)| *err <= tol)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(name, _)| name.to_string());
        events.push(LoadEvent {
            index: i,
            delta_w: d,
            appliance,
        });
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_and_identifies_appliance_cycles() {
        let library = [("fridge", 150.0), ("kettle", 2000.0), ("heater", 1500.0)];
        // Baseline 200 W; heater on (+1500), kettle on (+2000), kettle off, heater off.
        let mut p = vec![200.0; 6];
        p.extend(vec![1700.0; 6]); // heater on
        p.extend(vec![3700.0; 4]); // + kettle on
        p.extend(vec![1700.0; 4]); // kettle off
        p.extend(vec![200.0; 4]); // heater off

        let events = disaggregate(&p, &library, 300.0, 100.0);
        assert_eq!(events.len(), 4, "events {events:?}");
        assert_eq!(events[0].appliance.as_deref(), Some("heater"));
        assert!(events[0].delta_w > 0.0);
        assert_eq!(events[1].appliance.as_deref(), Some("kettle"));
        assert_eq!(events[2].appliance.as_deref(), Some("kettle")); // turn-off (|Δ| matches)
        assert!(events[2].delta_w < 0.0);
        assert_eq!(events[3].appliance.as_deref(), Some("heater"));
    }

    #[test]
    fn baseline_noise_below_threshold_is_ignored() {
        let library = [("heater", 1500.0)];
        let p = vec![200.0, 210.0, 195.0, 205.0, 198.0]; // small fluctuations
        assert!(disaggregate(&p, &library, 300.0, 100.0).is_empty());
    }

    #[test]
    fn a_step_matching_no_signature_is_detected_but_unlabeled() {
        // A 700 W step is well above min_step but within tol of neither 150 nor
        // 2000 W → the event is reported with appliance = None.
        let library = [("fridge", 150.0), ("kettle", 2000.0)];
        let p = vec![200.0, 200.0, 900.0, 900.0]; // +700 W
        let events = disaggregate(&p, &library, 300.0, 100.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].appliance, None);
        assert!((events[0].delta_w - 700.0).abs() < 1e-9);
    }
}
