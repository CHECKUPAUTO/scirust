use serde::{Deserialize, Serialize};

/// Automotive Safety Integrity Level (ASIL A..D).
///
/// ASIL D is the most stringent (e.g., braking, steering).
/// ASIL A is the least stringent (e.g., comfort features).
/// QM = Quality Managed, no safety relevance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AsilLevel {
    QM,
    A,
    B,
    C,
    D,
}

impl AsilLevel {
    pub fn label(&self) -> &'static str {
        match self
        {
            AsilLevel::QM => "QM",
            AsilLevel::A => "ASIL-A",
            AsilLevel::B => "ASIL-B",
            AsilLevel::C => "ASIL-C",
            AsilLevel::D => "ASIL-D",
        }
    }

    /// Required MC/DC coverage percentage.
    pub fn required_mcdc_coverage(&self) -> f32 {
        match self
        {
            AsilLevel::QM | AsilLevel::A => 0.0,
            AsilLevel::B => 50.0,
            AsilLevel::C => 75.0,
            AsilLevel::D => 100.0,
        }
    }

    /// Required fault injection test count.
    pub fn required_fault_injection_count(&self) -> usize {
        match self
        {
            AsilLevel::QM => 0,
            AsilLevel::A => 10,
            AsilLevel::B => 50,
            AsilLevel::C => 100,
            AsilLevel::D => 200,
        }
    }

    /// Maximum allowed WCET (worst-case execution time) budget factor.
    /// Safety-critical code must complete within this fraction of nominal time.
    pub fn max_wcet_factor(&self) -> f64 {
        match self
        {
            AsilLevel::QM => 10.0,
            AsilLevel::A => 5.0,
            AsilLevel::B => 3.0,
            AsilLevel::C => 2.0,
            AsilLevel::D => 1.5,
        }
    }
}

/// A safety goal defined per ISO 26262 hazard analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyGoal {
    pub id: String,
    pub description: String,
    pub asil_level: AsilLevel,
    /// Safe state to transition to on fault
    pub safe_state: String,
    /// Fault tolerant time interval (ms)
    pub ftti_ms: u32,
}

/// Configuration for a safety-critical component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsilConfig {
    pub component_id: String,
    pub asil_level: AsilLevel,
    /// Enable dual-core lockstep
    pub dual_lockstep: bool,
    /// Watchdog timeout in ms
    pub watchdog_timeout_ms: u32,
    /// Maximum allowed inference latency in ms
    pub max_latency_ms: f64,
    /// Redundancy factor (1 = single, 2 = dual, 3 = triple)
    pub redundancy: u8,
}

impl Default for AsilConfig {
    fn default() -> Self {
        Self {
            component_id: "default".to_string(),
            asil_level: AsilLevel::QM,
            dual_lockstep: false,
            watchdog_timeout_ms: 100,
            max_latency_ms: 50.0,
            redundancy: 1,
        }
    }
}

impl AsilConfig {
    /// Create a configuration for a given ASIL level with appropriate defaults.
    pub fn for_level(component_id: &str, level: AsilLevel) -> Self {
        let (lockstep, redundancy, watchdog, latency) = match level
        {
            AsilLevel::QM => (false, 1, 100, 50.0),
            AsilLevel::A => (false, 1, 50, 30.0),
            AsilLevel::B => (false, 2, 30, 20.0),
            AsilLevel::C => (true, 2, 20, 10.0),
            AsilLevel::D => (true, 3, 10, 5.0),
        };
        Self {
            component_id: component_id.to_string(),
            asil_level: level,
            dual_lockstep: lockstep,
            watchdog_timeout_ms: watchdog,
            max_latency_ms: latency,
            redundancy,
        }
    }

    /// Check if a given latency satisfies the safety requirement.
    pub fn check_latency(&self, measured_latency_ms: f64) -> bool {
        measured_latency_ms <= self.max_latency_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asil_ordering() {
        assert!(AsilLevel::D > AsilLevel::C);
        assert!(AsilLevel::C > AsilLevel::B);
        assert!(AsilLevel::B > AsilLevel::A);
        assert!(AsilLevel::A > AsilLevel::QM);
    }

    #[test]
    fn test_mcdc_coverage() {
        assert_eq!(AsilLevel::D.required_mcdc_coverage(), 100.0);
        assert_eq!(AsilLevel::A.required_mcdc_coverage(), 0.0);
        assert_eq!(AsilLevel::B.required_mcdc_coverage(), 50.0);
    }

    #[test]
    fn test_fault_injection_count() {
        assert_eq!(AsilLevel::D.required_fault_injection_count(), 200);
        assert_eq!(AsilLevel::QM.required_fault_injection_count(), 0);
    }

    #[test]
    fn test_config_for_level() {
        let cfg = AsilConfig::for_level("brake-ctrl", AsilLevel::D);
        assert!(cfg.dual_lockstep);
        assert_eq!(cfg.redundancy, 3);
        assert_eq!(cfg.watchdog_timeout_ms, 10);
        assert!((cfg.max_latency_ms - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_check_latency() {
        let cfg = AsilConfig::for_level("test", AsilLevel::C);
        assert!(cfg.check_latency(8.0)); // under limit
        assert!(!cfg.check_latency(15.0)); // over limit
    }

    #[test]
    fn test_asil_label() {
        assert_eq!(AsilLevel::D.label(), "ASIL-D");
        assert_eq!(AsilLevel::QM.label(), "QM");
    }
}
