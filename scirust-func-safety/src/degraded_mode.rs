use serde::{Deserialize, Serialize};

/// Degradation level for graceful degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DegradationLevel {
    /// Full functionality — all systems operational
    Level0,
    /// Reduced performance — lower sample rate, simpler model
    Level1,
    /// Safety mode — conservative fallback, alerts raised
    Level2,
    /// Emergency stop — safe state, production halted
    Level3,
}

impl DegradationLevel {
    pub fn label(&self) -> &'static str {
        match self
        {
            DegradationLevel::Level0 => "Level 0 - Full",
            DegradationLevel::Level1 => "Level 1 - Reduced",
            DegradationLevel::Level2 => "Level 2 - Safety",
            DegradationLevel::Level3 => "Level 3 - Emergency",
        }
    }
}

/// Action to take at each degradation level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationAction {
    pub level: DegradationLevel,
    pub description: String,
    pub reduce_sample_rate: bool,
    pub fallback_model: bool,
    pub alert_operators: bool,
    pub halt_production: bool,
}

impl DegradationAction {
    pub fn for_level(level: DegradationLevel) -> Self {
        match level
        {
            DegradationLevel::Level0 => Self {
                level,
                description: "Normal operation".to_string(),
                reduce_sample_rate: false,
                fallback_model: false,
                alert_operators: false,
                halt_production: false,
            },
            DegradationLevel::Level1 => Self {
                level,
                description: "Reduced performance — lower sample rate".to_string(),
                reduce_sample_rate: true,
                fallback_model: false,
                alert_operators: false,
                halt_production: false,
            },
            DegradationLevel::Level2 => Self {
                level,
                description: "Safety mode — fallback model active".to_string(),
                reduce_sample_rate: true,
                fallback_model: true,
                alert_operators: true,
                halt_production: false,
            },
            DegradationLevel::Level3 => Self {
                level,
                description: "Emergency stop — production halted".to_string(),
                reduce_sample_rate: true,
                fallback_model: true,
                alert_operators: true,
                halt_production: true,
            },
        }
    }
}

/// Controller for graceful degradation.
///
/// Monitors confidence scores and sensor health, transitioning through
/// degradation levels as needed. Implements the "safe state" concept
/// required by ISO 26262.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradedModeController {
    pub current_level: DegradationLevel,
    /// Confidence threshold for Level 0 → Level 1
    pub confidence_warn: f32,
    /// Confidence threshold for Level 1 → Level 2
    pub confidence_critical: f32,
    /// Confidence threshold for Level 2 → Level 3
    pub confidence_emergency: f32,
    /// Sensor failure count
    pub sensor_failures: u32,
    /// Max sensor failures before degradation
    pub max_sensor_failures: u32,
    /// Time in current level (ms)
    time_in_level_ms: f64,
    /// Minimum time to stay in a level before transitioning (hysteresis)
    pub min_dwell_time_ms: f64,
    /// History of level transitions
    pub transition_history: Vec<(DegradationLevel, f64)>,
}

impl DegradedModeController {
    pub fn new() -> Self {
        Self {
            current_level: DegradationLevel::Level0,
            confidence_warn: 0.85,
            confidence_critical: 0.5,
            confidence_emergency: 0.2,
            sensor_failures: 0,
            max_sensor_failures: 2,
            time_in_level_ms: 0.0,
            min_dwell_time_ms: 1000.0,
            transition_history: vec![(DegradationLevel::Level0, 0.0)],
        }
    }

    /// Update the controller with the latest inference confidence and sensor status.
    ///
    /// `confidence`: model output confidence (0..1)
    /// `sensor_failures`: count of currently-failed sensors
    /// `dt_ms`: elapsed time since last update (ms)
    pub fn update(
        &mut self,
        confidence: f32,
        sensor_failures: u32,
        dt_ms: f64,
    ) -> DegradationAction {
        self.time_in_level_ms += dt_ms;

        let new_level = self.compute_level(confidence, sensor_failures);

        // Apply hysteresis: don't transition too quickly
        if new_level != self.current_level && self.time_in_level_ms >= self.min_dwell_time_ms
        {
            self.current_level = new_level;
            self.time_in_level_ms = 0.0;
            self.transition_history.push((new_level, dt_ms));
        }

        DegradationAction::for_level(self.current_level)
    }

    fn compute_level(&self, confidence: f32, sensor_failures: u32) -> DegradationLevel {
        // Sensor failures override confidence
        if sensor_failures >= self.max_sensor_failures * 2
        {
            return DegradationLevel::Level3;
        }
        if sensor_failures >= self.max_sensor_failures
        {
            return DegradationLevel::Level2;
        }

        // Confidence-based degradation
        if confidence < self.confidence_emergency
        {
            DegradationLevel::Level3
        }
        else if confidence < self.confidence_critical
        {
            DegradationLevel::Level2
        }
        else if confidence < self.confidence_warn
        {
            DegradationLevel::Level1
        }
        else
        {
            DegradationLevel::Level0
        }
    }

    /// Force a specific degradation level (manual override).
    pub fn force_level(&mut self, level: DegradationLevel) {
        self.current_level = level;
        self.time_in_level_ms = 0.0;
        self.transition_history.push((level, 0.0));
    }

    /// Reset to Level 0 (normal operation).
    pub fn reset(&mut self) {
        self.current_level = DegradationLevel::Level0;
        self.time_in_level_ms = 0.0;
        self.sensor_failures = 0;
    }

    /// Check if production should continue.
    pub fn production_active(&self) -> bool {
        !matches!(self.current_level, DegradationLevel::Level3)
    }
}

impl Default for DegradedModeController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_level_is_0() {
        let ctrl = DegradedModeController::new();
        assert_eq!(ctrl.current_level, DegradationLevel::Level0);
        assert!(ctrl.production_active());
    }

    #[test]
    fn test_degradation_from_confidence() {
        let mut ctrl = DegradedModeController::new();
        ctrl.min_dwell_time_ms = 0.0; // disable hysteresis for test

        let action = ctrl.update(0.9, 0, 1.0);
        assert_eq!(ctrl.current_level, DegradationLevel::Level0);
        assert!(!action.alert_operators);

        let action = ctrl.update(0.7, 0, 1.0);
        assert_eq!(ctrl.current_level, DegradationLevel::Level1);
        assert!(action.reduce_sample_rate);

        let action = ctrl.update(0.3, 0, 1.0);
        assert_eq!(ctrl.current_level, DegradationLevel::Level2);
        assert!(action.alert_operators);
        assert!(action.fallback_model);

        let action = ctrl.update(0.1, 0, 1.0);
        assert_eq!(ctrl.current_level, DegradationLevel::Level3);
        assert!(action.halt_production);
        assert!(!ctrl.production_active());
    }

    #[test]
    fn test_degradation_from_sensor_failures() {
        let mut ctrl = DegradedModeController::new();
        ctrl.min_dwell_time_ms = 0.0;
        ctrl.max_sensor_failures = 2;

        // Normal confidence but sensor failures
        let _action = ctrl.update(0.95, 2, 1.0);
        assert_eq!(ctrl.current_level, DegradationLevel::Level2);

        ctrl.reset();
        let _action = ctrl.update(0.95, 4, 1.0);
        assert_eq!(ctrl.current_level, DegradationLevel::Level3);
    }

    #[test]
    fn test_hysteresis() {
        let mut ctrl = DegradedModeController::new();
        ctrl.min_dwell_time_ms = 1000.0;

        // First update: confidence drops, but dwell time not met
        ctrl.update(0.3, 0, 100.0);
        // Should still be Level0 because dwell time < 1000ms
        assert_eq!(ctrl.current_level, DegradationLevel::Level0);

        // After enough time, should transition
        ctrl.update(0.3, 0, 1000.0);
        assert_eq!(ctrl.current_level, DegradationLevel::Level2);
    }

    #[test]
    fn test_force_level() {
        let mut ctrl = DegradedModeController::new();
        ctrl.force_level(DegradationLevel::Level3);
        assert_eq!(ctrl.current_level, DegradationLevel::Level3);
    }

    #[test]
    fn test_reset() {
        let mut ctrl = DegradedModeController::new();
        ctrl.force_level(DegradationLevel::Level3);
        ctrl.reset();
        assert_eq!(ctrl.current_level, DegradationLevel::Level0);
    }

    #[test]
    fn test_transition_history() {
        let mut ctrl = DegradedModeController::new();
        ctrl.min_dwell_time_ms = 0.0;
        ctrl.update(0.3, 0, 1.0);
        ctrl.update(0.1, 0, 1.0);
        assert!(ctrl.transition_history.len() >= 3); // initial + 2 transitions
    }

    #[test]
    fn test_degradation_action_for_level() {
        let action = DegradationAction::for_level(DegradationLevel::Level3);
        assert!(action.halt_production);
        assert!(action.alert_operators);
        assert!(action.fallback_model);

        let action = DegradationAction::for_level(DegradationLevel::Level0);
        assert!(!action.halt_production);
    }
}
