//! ISO 10816-3 / ISO 20816 broadband vibration severity zones.
//!
//! Turns a broadband RMS velocity (mm/s) into a normalised evaluation zone
//! **A / B / C / D** for the machine's group and support class — a *compliance
//! verdict*, not a raw number. Unlike a single fixed threshold table, the zone
//! boundaries depend on machine size (group) and foundation (rigid/flexible),
//! exactly as the standard prescribes.
//!
//! Boundary values are the ISO 10816-3 Annex evaluation-zone limits; for a
//! specific installation confirm the machine's group and support class.

use crate::detectors::FaultSeverity;
use serde::{Deserialize, Serialize};

/// ISO 10816-3 machine group, by rated power / shaft height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MachineGroup {
    /// Group 1: large machines, 300 kW – 50 MW (shaft height ≥ 315 mm).
    Group1,
    /// Group 2: medium machines, 15 – 300 kW (shaft height 160 – 315 mm).
    Group2,
}

/// Foundation / support class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Support {
    /// Rigid support (foundation natural frequency well above running speed).
    Rigid,
    /// Flexible support.
    Flexible,
}

/// ISO evaluation zone for broadband vibration severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VibrationZone {
    /// A — typically newly commissioned machines.
    A,
    /// B — acceptable for unrestricted long-term operation.
    B,
    /// C — unsatisfactory for long-term operation; restricted running, plan action.
    C,
    /// D — severe enough to cause damage.
    D,
}

impl VibrationZone {
    /// Single-letter label.
    pub fn label(self) -> &'static str {
        match self
        {
            VibrationZone::A => "A",
            VibrationZone::B => "B",
            VibrationZone::C => "C",
            VibrationZone::D => "D",
        }
    }

    /// Map to the coarse [`FaultSeverity`] used elsewhere in the crate.
    /// A/B are acceptable (Normal), C is unsatisfactory (Danger), D risks damage
    /// (Critical).
    pub fn severity(self) -> FaultSeverity {
        match self
        {
            VibrationZone::A | VibrationZone::B => FaultSeverity::Normal,
            VibrationZone::C => FaultSeverity::Danger,
            VibrationZone::D => FaultSeverity::Critical,
        }
    }
}

/// ISO 10816-3 broadband severity evaluator (RMS velocity in mm/s).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Iso10816 {
    ab: f64,
    bc: f64,
    cd: f64,
}

impl Iso10816 {
    /// Standard zone boundaries (mm/s RMS) for the group and support class,
    /// per ISO 10816-3.
    pub fn new(group: MachineGroup, support: Support) -> Self {
        let (ab, bc, cd) = match (group, support)
        {
            (MachineGroup::Group1, Support::Rigid) => (2.3, 4.5, 7.1),
            (MachineGroup::Group1, Support::Flexible) => (3.5, 7.1, 11.0),
            (MachineGroup::Group2, Support::Rigid) => (1.4, 2.8, 4.5),
            (MachineGroup::Group2, Support::Flexible) => (2.3, 4.5, 7.1),
        };
        Self { ab, bc, cd }
    }

    /// Construct from explicit boundaries (e.g. manufacturer-specified), in
    /// mm/s RMS, with `ab ≤ bc ≤ cd`.
    pub fn with_boundaries(ab: f64, bc: f64, cd: f64) -> Self {
        Self { ab, bc, cd }
    }

    /// The `(A/B, B/C, C/D)` zone boundaries in mm/s RMS.
    pub fn boundaries(&self) -> (f64, f64, f64) {
        (self.ab, self.bc, self.cd)
    }

    /// Classify a broadband RMS velocity (mm/s) into an evaluation zone.
    /// Boundaries are inclusive of the lower zone (a reading exactly at A/B is B).
    pub fn evaluate(&self, rms_mms: f64) -> VibrationZone {
        if rms_mms < self.ab
        {
            VibrationZone::A
        }
        else if rms_mms < self.bc
        {
            VibrationZone::B
        }
        else if rms_mms < self.cd
        {
            VibrationZone::C
        }
        else
        {
            VibrationZone::D
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_boundaries_match_iso_10816_3() {
        // (group, support) -> (A/B, B/C, C/D) per ISO 10816-3 Annex.
        let cases = [
            (MachineGroup::Group1, Support::Rigid, (2.3, 4.5, 7.1)),
            (MachineGroup::Group1, Support::Flexible, (3.5, 7.1, 11.0)),
            (MachineGroup::Group2, Support::Rigid, (1.4, 2.8, 4.5)),
            (MachineGroup::Group2, Support::Flexible, (2.3, 4.5, 7.1)),
        ];
        for (g, s, want) in cases
        {
            assert_eq!(Iso10816::new(g, s).boundaries(), want, "{g:?}/{s:?}");
        }
    }

    #[test]
    fn classifies_each_band_for_group2_rigid() {
        // Group 2 rigid: A/B=1.4, B/C=2.8, C/D=4.5.
        let iso = Iso10816::new(MachineGroup::Group2, Support::Rigid);
        assert_eq!(iso.evaluate(1.0), VibrationZone::A);
        assert_eq!(iso.evaluate(2.0), VibrationZone::B);
        assert_eq!(iso.evaluate(3.5), VibrationZone::C);
        assert_eq!(iso.evaluate(6.0), VibrationZone::D);
        // Boundaries fall into the upper zone.
        assert_eq!(iso.evaluate(1.4), VibrationZone::B);
        assert_eq!(iso.evaluate(2.8), VibrationZone::C);
        assert_eq!(iso.evaluate(4.5), VibrationZone::D);
    }

    #[test]
    fn zone_is_monotonic_in_amplitude() {
        let iso = Iso10816::new(MachineGroup::Group1, Support::Flexible);
        let mut prev = VibrationZone::A;
        let mut rms = 0.0;
        while rms <= 15.0
        {
            let z = iso.evaluate(rms);
            assert!(z as u8 >= prev as u8, "zone decreased at {rms}");
            prev = z;
            rms += 0.1;
        }
        assert_eq!(iso.evaluate(20.0), VibrationZone::D);
    }

    #[test]
    fn severity_mapping() {
        assert_eq!(VibrationZone::A.severity(), FaultSeverity::Normal);
        assert_eq!(VibrationZone::B.severity(), FaultSeverity::Normal);
        assert_eq!(VibrationZone::C.severity(), FaultSeverity::Danger);
        assert_eq!(VibrationZone::D.severity(), FaultSeverity::Critical);
    }
}
