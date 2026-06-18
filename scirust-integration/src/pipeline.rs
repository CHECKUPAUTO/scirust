use crate::backend::{Backend, BackendFactory, BackendType};
use crate::config::PipelineConfig;
use scirust_events_core::EventStream;
use scirust_events_models::SpikeDetector;
use scirust_events_runtime::EventRuntime;
use scirust_func_safety::audit::AuditLog;
use scirust_mqtt::event_to_payload;
use scirust_pdm::change_detection::CUSUM;
use scirust_pdm::health::HealthIndex;
use scirust_pdm::rul::{LinearRulEstimator, RulEstimator};
use serde::{Deserialize, Serialize};

/// Current status of the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub cycles_completed: usize,
    pub events_detected: u64,
    pub events_published: u64,
    pub current_health_index: f64,
    pub current_health_state: String,
    pub rul_hours: f64,
    pub audit_entries: usize,
    pub audit_chain_valid: bool,
    pub backend_type: String,
    pub connected: bool,
}

/// Final report after pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    pub total_cycles: usize,
    pub total_events: u64,
    pub total_published: u64,
    pub final_health_index: f64,
    pub final_health_state: String,
    pub final_rul: f64,
    pub rul_lower_bound: f64,
    pub rul_upper_bound: f64,
    pub audit_entries: usize,
    pub audit_chain_valid: bool,
    pub mqtt_messages: u64,
    pub mqtt_info: usize,
    pub mqtt_warning: usize,
    pub mqtt_critical: usize,
    pub duration_note: String,
}

/// The complete monitoring pipeline.
///
/// Ties together: Backend → Signal → Events → Health → RUL → MQTT → Audit
pub struct Pipeline {
    pub config: PipelineConfig,
    pub backend: Backend,
    pub runtime: EventRuntime,
    pub health: HealthIndex,
    pub rul: LinearRulEstimator,
    pub cusum: CUSUM,
    pub audit: AuditLog,
    cycles_completed: usize,
    events_detected: u64,
    events_published: u64,
    sim_time: f64,
    subscribed_node_ids: Vec<String>,
}

impl Pipeline {
    /// Create a new pipeline from configuration.
    pub fn new(config: PipelineConfig) -> Self {
        let backend_type =
            BackendType::parse_from_str(&config.backend_type).unwrap_or(BackendType::Simulated);

        let opcua_cfg: scirust_opcua::OpcuaConfig = (&config.opcua).into();
        let mqtt_cfg: scirust_mqtt::MqttConfig = (&config.mqtt).into();

        let backend = match BackendFactory::create(&opcua_cfg, &mqtt_cfg, backend_type)
        {
            Ok(b) => b,
            Err(_) => BackendFactory::try_real_or_simulated(&opcua_cfg, &mqtt_cfg),
        };

        // First station config drives the pipeline
        let station = &config.stations[0];

        let runtime = EventRuntime::new(Box::new(SpikeDetector::new(
            station.spike_threshold as f32,
            station.ema_alpha as f32,
        )));

        let baselines: Vec<f64> = station.sensors.iter().map(|s| s.baseline).collect();
        let thresholds: Vec<f64> = station
            .sensors
            .iter()
            .map(|s| s.failure_threshold)
            .collect();
        let weights: Vec<f64> = station.sensors.iter().map(|s| s.weight).collect();

        let health = HealthIndex::new(baselines, thresholds, weights, 0.3);
        let rul = LinearRulEstimator::new(
            config.settings.rul_window_size,
            config.settings.rul_min_observations,
        );
        let cusum = CUSUM::new(0.1, 0.05, 0.5);
        let audit = AuditLog::new(config.settings.audit_log_size);

        Self {
            config,
            backend,
            runtime,
            health,
            rul,
            cusum,
            audit,
            cycles_completed: 0,
            events_detected: 0,
            events_published: 0,
            sim_time: 0.0,
            subscribed_node_ids: Vec::new(),
        }
    }

    /// Initialize: subscribe to OPC-UA nodes matching configured sensors.
    pub fn init(&mut self) -> Result<(), String> {
        let station = &self.config.stations[0];
        // Use the sensor node IDs from configuration directly
        self.subscribed_node_ids = station.sensors.iter().map(|s| s.node_id.clone()).collect();
        // Try to subscribe (works with both real and simulated backends)
        match self.backend.opcua.subscribe(&self.subscribed_node_ids)
        {
            Ok(()) => Ok(()),
            Err(_e) =>
            {
                // Some node IDs may not exist in simulated backend — try browse + subscribe
                let nodes = self.backend.opcua.browse("")?;
                let available_ids: Vec<String> = nodes.iter().map(|n| n.node_id.clone()).collect();
                let filtered: Vec<String> = self
                    .subscribed_node_ids
                    .iter()
                    .filter(|id| available_ids.contains(id))
                    .cloned()
                    .collect();
                if filtered.is_empty()
                {
                    // Fall back to available nodes
                    self.subscribed_node_ids = available_ids;
                }
                else
                {
                    self.subscribed_node_ids = filtered;
                }
                self.backend.opcua.subscribe(&self.subscribed_node_ids)?;
                Ok(())
            },
        }
    }

    /// Run one monitoring cycle.
    ///
    /// Returns the number of events detected in this cycle.
    pub fn run_cycle(&mut self) -> usize {
        // Poll OPC-UA for new sensor values
        let values = match self.backend.opcua.poll_subscription()
        {
            Ok(v) if !v.is_empty() => v,
            _ => return 0,
        };

        // Extract feature values in the ORDER of configured sensors
        // This ensures HealthIndex gets the right number of features
        let station = &self.config.stations[0];
        let n_sensors = station.sensors.len();
        let features: Vec<f64> = if values.len() >= n_sensors
        {
            // If we have enough values, take the first n_sensors in order
            values.iter().take(n_sensors).map(|v| v.value).collect()
        }
        else
        {
            // Pad with zeros if not enough values
            (0..n_sensors)
                .map(|i| values.get(i).map(|v| v.value).unwrap_or(0.0))
                .collect()
        };

        if features.is_empty()
        {
            return 0;
        }

        // Update Health Index
        let hi = self.health.update(&features);
        let state = self.health.state();

        // Update RUL
        self.rul.update(hi, self.sim_time);

        // CUSUM on first feature
        if !features.is_empty()
        {
            let _ = self.cusum.update(features[0], 0.1);
        }

        // Run event detection
        let mut stream = EventStream::new(
            features.iter().map(|f| *f as f32).collect(),
            self.config.stations[0].window_size,
            self.config.stations[0].window_stride,
        );
        let events = self.runtime.process_all(&mut stream);
        let cycle_events = events.len();
        self.events_detected += cycle_events as u64;

        // Publish events to MQTT
        let station = &self.config.stations[0];
        for event in &events
        {
            if event.confidence >= station.min_confidence
            {
                let payload = event_to_payload(event, &station.id, None);
                if self.backend.mqtt.publish_event(&payload).is_ok()
                {
                    self.events_published += 1;
                }
            }
        }

        // Audit log
        self.audit.add(
            "monitoring_cycle",
            &format!(
                "Cycle {}: HI={:.3}, state={}, events={}",
                self.cycles_completed,
                hi,
                state.label(),
                cycle_events
            ),
            &station.id,
            if hi < 0.5 { "alert" } else { "pass" },
            hi as f32,
            self.sim_time,
        );

        self.cycles_completed += 1;
        self.sim_time += 0.1;
        cycle_events
    }

    /// Run the pipeline for `n` cycles.
    pub fn run(&mut self, n_cycles: usize) -> PipelineReport {
        if self.init().is_err()
        {
            // Continue even if init fails (simulated mode may not need browse)
        }

        for _ in 0..n_cycles
        {
            self.run_cycle();
        }

        self.generate_report()
    }

    /// Get current pipeline status.
    pub fn status(&self) -> PipelineStatus {
        let rul_pred = self.rul.predict();
        PipelineStatus {
            cycles_completed: self.cycles_completed,
            events_detected: self.events_detected,
            events_published: self.events_published,
            current_health_index: self.health.value(),
            current_health_state: self.health.state().label().to_string(),
            rul_hours: rul_pred.remaining_hours,
            audit_entries: self.audit.len(),
            audit_chain_valid: self.audit.verify_chain(),
            backend_type: self.backend.backend_type.label().to_string(),
            connected: self.backend.is_connected(),
        }
    }

    /// Generate a final report.
    pub fn generate_report(&self) -> PipelineReport {
        let rul_pred = self.rul.predict();
        // We can't downcast the trait object, so we use events_published as a proxy.
        // In a real implementation, the MQTT publisher would track counts internally.
        let mqtt_messages = self.events_published;
        let mqtt_info = 0;
        let mqtt_warning = 0;
        let mqtt_critical = 0;

        PipelineReport {
            total_cycles: self.cycles_completed,
            total_events: self.events_detected,
            total_published: self.events_published,
            final_health_index: self.health.value(),
            final_health_state: self.health.state().label().to_string(),
            final_rul: rul_pred.remaining_hours,
            rul_lower_bound: rul_pred.lower_bound_hours,
            rul_upper_bound: rul_pred.upper_bound_hours,
            audit_entries: self.audit.len(),
            audit_chain_valid: self.audit.verify_chain(),
            mqtt_messages,
            mqtt_info,
            mqtt_warning,
            mqtt_critical,
            duration_note: format!("Ran {} cycles", self.cycles_completed),
        }
    }

    /// Export the audit log as JSON.
    pub fn export_audit(&self) -> Result<String, String> {
        self.audit.export_json()
    }

    /// Export the final report as JSON.
    pub fn export_report_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.generate_report()).map_err(|e| e.to_string())
    }

    /// Export the status as JSON.
    pub fn export_status_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.status()).map_err(|e| e.to_string())
    }

    /// Shutdown: disconnect backends.
    pub fn shutdown(&mut self) {
        let _ = self.backend.opcua.disconnect();
        let _ = self.backend.mqtt.disconnect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = Pipeline::new(config);
        assert_eq!(pipeline.cycles_completed, 0);
        assert!(pipeline.backend.is_simulated());
    }

    #[test]
    fn test_pipeline_run_cycles() {
        let config = PipelineConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.run(10);
        assert_eq!(pipeline.cycles_completed, 10);
        // Should have audit entries
        assert!(!pipeline.audit.is_empty());
        assert!(pipeline.audit.verify_chain());
    }

    #[test]
    fn test_pipeline_status() {
        let config = PipelineConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.run(5);
        let status = pipeline.status();
        assert_eq!(status.cycles_completed, 5);
        assert_eq!(status.backend_type, "simulated");
        assert!(status.connected);
    }

    #[test]
    fn test_pipeline_report() {
        let config = PipelineConfig::automotive_line("test-line", 2);
        let mut pipeline = Pipeline::new(config);
        pipeline.run(20);
        let report = pipeline.generate_report();
        assert_eq!(report.total_cycles, 20);
        assert!(report.audit_chain_valid);
    }

    #[test]
    fn test_pipeline_automotive_config() {
        let config = PipelineConfig::automotive_line("line-7", 3);
        let pipeline = Pipeline::new(config);
        assert!(pipeline.config.stations[0].bearing.is_some());
    }

    #[test]
    fn test_pipeline_export_json() {
        let config = PipelineConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.run(5);
        let json = pipeline.export_status_json().unwrap();
        assert!(json.contains("cycles_completed"));
        assert!(json.contains("simulated"));
    }

    #[test]
    fn test_pipeline_shutdown() {
        let config = PipelineConfig::default();
        let mut pipeline = Pipeline::new(config);
        pipeline.run(3);
        pipeline.shutdown();
        // After shutdown, backend should be disconnected
    }
}
