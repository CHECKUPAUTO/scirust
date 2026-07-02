//! A full Safety Instrumented Function (SIF) loop: sensor subsystem → logic
//! solver subsystem → final-element subsystem, each with its own voting
//! architecture. Standard IEC 61511/ISA-TR84.00.02 SIL-verification
//! practice: the loop's total `PFDavg` is the **sum** of its subsystems'
//! `PFDavg` (they fail independently of each other by construction — a
//! sensor fault doesn't make the final element more likely to fail), and
//! the achieved SIL is the band that total falls into.

use crate::error::SisResult;
use crate::voting::Architecture;
use scirust_reliability::Sil;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subsystem {
    pub name: String,
    pub architecture: Architecture,
    /// Dangerous-undetected failure rate, per hour.
    pub lambda_du: f64,
    /// Common-cause fraction (0.0 if not applicable/known).
    pub beta: f64,
    /// Proof-test interval, hours.
    pub t1: f64,
}

impl Subsystem {
    pub fn new(
        name: impl Into<String>,
        architecture: Architecture,
        lambda_du: f64,
        beta: f64,
        t1: f64,
    ) -> Self {
        Self {
            name: name.into(),
            architecture,
            lambda_du,
            beta,
            t1,
        }
    }

    pub fn pfd_avg(&self) -> SisResult<f64> {
        self.architecture
            .pfd_avg(self.lambda_du, self.t1, self.beta)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SifLoop {
    pub name: String,
    pub subsystems: Vec<Subsystem>,
}

impl SifLoop {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            subsystems: Vec::new(),
        }
    }

    pub fn add_subsystem(&mut self, subsystem: Subsystem) -> &mut Self {
        self.subsystems.push(subsystem);
        self
    }

    /// Total `PFDavg` of the loop: the sum of every subsystem's `PFDavg`.
    pub fn total_pfd_avg(&self) -> SisResult<f64> {
        let mut total = 0.0;
        for s in &self.subsystems
        {
            total += s.pfd_avg()?;
        }
        Ok(total)
    }

    /// The SIL band the loop's total `PFDavg` falls into.
    pub fn achieved_sil(&self) -> SisResult<Sil> {
        Ok(scirust_reliability::sil_from_pfd(self.total_pfd_avg()?))
    }

    /// Per-subsystem `PFDavg` breakdown, in subsystem order — useful for
    /// identifying which subsystem dominates the loop's total.
    pub fn breakdown(&self) -> SisResult<Vec<(String, f64)>> {
        self.subsystems
            .iter()
            .map(|s| Ok((s.name.clone(), s.pfd_avg()?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn typical_loop() -> SifLoop {
        // A realistic 3-subsystem SIF: 2oo3 pressure transmitters, a 1oo1
        // logic solver, and 1oo2 final-element (shutdown valves).
        let mut sif = SifLoop::new("HP Trip SIF-101");
        sif.add_subsystem(Subsystem::new(
            "Sensors (2oo3 PT)",
            Architecture::TWO_OO3,
            5e-7,
            0.02,
            8760.0,
        ))
        .add_subsystem(Subsystem::new(
            "Logic Solver (1oo1)",
            Architecture::OO1,
            1e-7,
            0.0,
            8760.0,
        ))
        .add_subsystem(Subsystem::new(
            "Final Elements (1oo2 XV)",
            Architecture::OO2,
            2e-6,
            0.05,
            8760.0,
        ));
        sif
    }

    #[test]
    fn total_pfd_is_sum_of_subsystems() {
        let sif = typical_loop();
        let breakdown = sif.breakdown().unwrap();
        let manual_sum: f64 = breakdown.iter().map(|(_, pfd)| pfd).sum();
        assert_relative_eq!(sif.total_pfd_avg().unwrap(), manual_sum, epsilon = 1e-15);
    }

    #[test]
    fn achieved_sil_matches_total_pfd_band() {
        let sif = typical_loop();
        let total = sif.total_pfd_avg().unwrap();
        assert_eq!(
            sif.achieved_sil().unwrap(),
            scirust_reliability::sil_from_pfd(total)
        );
    }

    #[test]
    fn empty_loop_has_zero_pfd_and_sil4() {
        let sif = SifLoop::new("empty");
        assert_eq!(sif.total_pfd_avg().unwrap(), 0.0);
        assert_eq!(sif.achieved_sil().unwrap(), Sil::Sil4);
    }

    #[test]
    fn breakdown_preserves_subsystem_order_and_names() {
        let sif = typical_loop();
        let names: Vec<String> = sif
            .breakdown()
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(
            names,
            vec![
                "Sensors (2oo3 PT)".to_string(),
                "Logic Solver (1oo1)".to_string(),
                "Final Elements (1oo2 XV)".to_string(),
            ]
        );
    }

    #[test]
    fn adding_a_weaker_subsystem_can_only_worsen_or_hold_sil() {
        let mut sif = typical_loop();
        let sil_before = sif.achieved_sil().unwrap();
        // A poorly-maintained 2oo2 final element (worst MooN member) with a
        // long proof-test interval dominates the total.
        sif.add_subsystem(Subsystem::new(
            "Legacy interlock (2oo2)",
            Architecture::TWO_OO2,
            5e-6,
            0.0,
            26280.0,
        ));
        let sil_after = sif.achieved_sil().unwrap();
        assert!(
            sil_after <= sil_before,
            "{sil_after:?} should not exceed {sil_before:?}"
        );
    }
}
