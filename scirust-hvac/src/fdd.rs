//! Air-Handling-Unit Fault Detection and Diagnostics (ASHRAE Guideline 36-style).
//!
//! Each rule is a physical residual: the mixed-air temperature must be the
//! mass-weighted blend of return and outdoor air; with cooling on, the supply
//! must drop across the coil; and when free cooling is available the economizer
//! should be open. A residual beyond tolerance flags a specific fault.

use serde::{Deserialize, Serialize};

/// One AHU operating snapshot (temperatures in the same unit, e.g. °C).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AhuReading {
    pub return_temp: f64,
    pub outdoor_temp: f64,
    pub mixed_temp: f64,
    pub supply_temp: f64,
    /// Commanded outdoor-air fraction in `[0, 1]`.
    pub oa_fraction: f64,
    /// Whether mechanical cooling is enabled.
    pub cooling_on: bool,
}

/// A detected AHU fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AhuFault {
    /// Mixed-air temperature inconsistent with the damper position (stuck/leaking damper).
    MixingError,
    /// Cooling commanded but the coil produces no temperature drop (stuck valve).
    CoolingCoilStuck,
    /// Free cooling available but the economizer damper is closed.
    EconomizerFault,
}

/// Diagnose an AHU reading; `temp_tol` is the mixing residual tolerance.
pub fn diagnose_ahu(r: &AhuReading, temp_tol: f64) -> Vec<AhuFault> {
    let mut faults = Vec::new();
    let expected_mix = r.oa_fraction * r.outdoor_temp + (1.0 - r.oa_fraction) * r.return_temp;
    if (r.mixed_temp - expected_mix).abs() > temp_tol
    {
        faults.push(AhuFault::MixingError);
    }
    if r.cooling_on && r.supply_temp > r.mixed_temp - 0.5
    {
        // Cooling on but supply did not drop below the mixed-air temperature.
        faults.push(AhuFault::CoolingCoilStuck);
    }
    if r.cooling_on && r.outdoor_temp < r.return_temp - 2.0 && r.oa_fraction < 0.5
    {
        // Outdoor air is usefully cool yet the economizer is mostly closed.
        faults.push(AhuFault::EconomizerFault);
    }
    faults
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_ahu_has_no_faults() {
        // Free cooling available, so the economizer is open (80% OA):
        // mix = 0.8·15 + 0.2·24 = 16.8; mechanical cooling drops supply to 13.
        let r = AhuReading {
            return_temp: 24.0,
            outdoor_temp: 15.0,
            mixed_temp: 16.8,
            supply_temp: 13.0,
            oa_fraction: 0.8,
            cooling_on: true,
        };
        assert!(diagnose_ahu(&r, 0.5).is_empty());
    }

    #[test]
    fn flags_a_mixing_damper_fault() {
        let mut r = AhuReading {
            return_temp: 24.0,
            outdoor_temp: 15.0,
            mixed_temp: 24.0, // damper stuck closed -> mix == return
            supply_temp: 13.0,
            oa_fraction: 0.5, // commands 50% OA
            cooling_on: true,
        };
        assert!(diagnose_ahu(&r, 0.5).contains(&AhuFault::MixingError));
        r.mixed_temp = 0.5 * 15.0 + 0.5 * 24.0; // consistent -> clears
        assert!(!diagnose_ahu(&r, 0.5).contains(&AhuFault::MixingError));
    }

    #[test]
    fn flags_a_stuck_cooling_coil() {
        let r = AhuReading {
            return_temp: 24.0,
            outdoor_temp: 28.0,
            mixed_temp: 25.0,
            supply_temp: 25.0, // no drop despite cooling on
            oa_fraction: 0.2,
            cooling_on: true,
        };
        assert!(diagnose_ahu(&r, 0.5).contains(&AhuFault::CoolingCoilStuck));
    }

    #[test]
    fn flags_a_closed_economizer_when_free_cooling_is_available() {
        let r = AhuReading {
            return_temp: 24.0,
            outdoor_temp: 16.0, // 8° cooler than return -> free cooling
            mixed_temp: 22.4,
            supply_temp: 13.0,
            oa_fraction: 0.2, // economizer mostly closed
            cooling_on: true,
        };
        assert!(diagnose_ahu(&r, 0.5).contains(&AhuFault::EconomizerFault));
    }
}
