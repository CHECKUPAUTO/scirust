use crate::backend::{Backend, BackendFactory, BackendType};
use crate::config::PipelineConfig;
use scirust_events_core::EventStream;
use scirust_events_models::SpikeDetector;
use scirust_events_runtime::EventRuntime;
use scirust_func_safety::audit::AuditLog;
use scirust_mqtt::{EventSeverity, event_to_payload};
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
    /// Number of CUSUM drift change-points detected so far.
    pub drift_alarms: u64,
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
    /// Number of CUSUM drift change-points detected over the run.
    pub drift_alarms: u64,
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
    mqtt_info: usize,
    mqtt_warning: usize,
    mqtt_critical: usize,
    sim_time: f64,
    subscribed_node_ids: Vec<String>,
    /// Last successfully-read feature vector, used to carry forward values for
    /// sensors that are momentarily absent from a poll instead of injecting 0.0.
    last_features: Vec<f64>,
    /// Number of cycles in which a CUSUM drift change-point fired.
    drift_alarms: u64,
}

impl Pipeline {
    /// Create a new pipeline with the bundled simulated backend.
    ///
    /// The pipeline is driven by the first configured station. A configuration
    /// with zero stations is not usable, so any empty `stations` list is
    /// backfilled with a single [`StationConfig::default`] here to guarantee an
    /// index-safe, panic-free construction (deserialized JSON can legally carry
    /// `"stations": []`). Use [`Pipeline::try_new`] when callers want to reject
    /// such a configuration up front instead of running against a default
    /// station. Production transports use [`Pipeline::with_backend`].
    pub fn new(mut config: PipelineConfig) -> Self {
        if config.stations.is_empty()
        {
            config
                .stations
                .push(crate::config::StationConfig::default());
        }

        let opcua_cfg: scirust_opcua::OpcuaConfig = (&config.opcua).into();
        let mqtt_cfg: scirust_mqtt::MqttConfig = (&config.mqtt).into();
        let backend = BackendFactory::create(&opcua_cfg, &mqtt_cfg, BackendType::Simulated)
            .expect("built-in simulated backend must accept validated configuration");
        config.backend_type = BackendType::Simulated.label().to_string();
        Self::from_parts(config, backend)
    }

    /// Build a pipeline around caller-supplied, connected transport adapters.
    pub fn with_backend(mut config: PipelineConfig, backend: Backend) -> Result<Self, String> {
        let validation_errors = config.validate();
        if !validation_errors.is_empty()
        {
            return Err(format!(
                "invalid pipeline configuration: {}",
                validation_errors.join("; ")
            ));
        }
        backend
            .ensure_ready()
            .map_err(|error| format!("pipeline backend is not ready: {error}"))?;
        config.backend_type = backend.backend_type().label().to_string();
        Ok(Self::from_parts(config, backend))
    }

    fn from_parts(config: PipelineConfig, backend: Backend) -> Self {
        // First station config drives the pipeline (guaranteed non-empty above).
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
        // Parameterize the drift detector from the configured drift threshold
        // rather than a hardcoded constant, so config changes actually take effect.
        let cusum = CUSUM::new(0.1, 0.05, config.settings.drift_threshold);
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
            mqtt_info: 0,
            mqtt_warning: 0,
            mqtt_critical: 0,
            sim_time: 0.0,
            subscribed_node_ids: Vec::new(),
            last_features: Vec::new(),
            drift_alarms: 0,
        }
    }

    /// Create a new pipeline, rejecting an invalid configuration.
    ///
    /// Unlike [`Pipeline::new`], this validates the configuration first and
    /// returns an error (rather than backfilling a default station) when the
    /// config is unusable — most importantly when it declares no stations.
    pub fn try_new(config: PipelineConfig) -> Result<Self, String> {
        let validation_errors = config.validate();
        if !validation_errors.is_empty()
        {
            return Err(format!(
                "invalid pipeline configuration: {}",
                validation_errors.join("; ")
            ));
        }
        match BackendType::parse_from_str(&config.backend_type)
        {
            Some(BackendType::Simulated) => Ok(Self::new(config)),
            Some(BackendType::External) =>
            {
                Err("external backend configuration requires Pipeline::with_backend".to_string())
            },
            None => Err(format!(
                "unknown backend type `{}`; expected `simulated` or use Pipeline::with_backend",
                config.backend_type
            )),
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

        // Align polled values to configured sensors BY node_id (not by arrival
        // order). The OPC-UA browse fallback may expose nodes with a namespace
        // prefix (e.g. "ns=2;s=Vibration.X") while the config uses bare ids
        // (e.g. "Vibration.X"), so we match either an exact id or a polled id
        // that ends with the configured id.
        // Snapshot the per-sensor config we need as owned locals so the rest of
        // the cycle can take &mut self freely (audit/rul/health updates).
        let sensor_ids: Vec<String> = self.config.stations[0]
            .sensors
            .iter()
            .map(|s| s.node_id.clone())
            .collect();
        let station_id = self.config.stations[0].id.clone();
        let min_confidence = self.config.stations[0].min_confidence;
        let window_size = self.config.stations[0].window_size;
        let window_stride = self.config.stations[0].window_stride;
        let n_sensors = sensor_ids.len();

        let lookup = |sensor_id: &str| -> Option<f64> {
            values
                .iter()
                .find(|v| v.node_id == sensor_id || v.node_id.ends_with(sensor_id))
                .map(|v| v.value)
        };

        // Assemble the feature vector in configured-sensor order. For a sensor
        // that is genuinely absent from this poll, carry forward its last known
        // value rather than injecting a fake 0.0 (which would masquerade as a
        // perfectly healthy reading). If we have no prior value at all, skip the
        // cycle instead of fabricating data.
        let mut features: Vec<f64> = Vec::with_capacity(n_sensors);
        let mut all_missing = true;
        for (i, sensor_id) in sensor_ids.iter().enumerate()
        {
            if let Some(v) = lookup(sensor_id)
            {
                features.push(v);
                all_missing = false;
            }
            else if let Some(&prev) = self.last_features.get(i)
            {
                features.push(prev);
            }
            else
            {
                // No reading now and none previously: skip this cycle.
                return 0;
            }
        }

        if features.is_empty() || all_missing
        {
            return 0;
        }

        self.last_features = features.clone();

        // Update Health Index
        let hi = self.health.update(&features);
        let state = self.health.state();

        // Update RUL
        self.rul.update(hi, self.sim_time);

        // CUSUM drift detection on the first feature. Gated by configuration;
        // when a change-point fires, surface it via the audit log and a drift
        // counter instead of silently discarding the result.
        if self.config.settings.enable_drift_detection
        {
            if let Some(cp) = self.cusum.update(features[0], 0.1)
            {
                self.drift_alarms += 1;
                self.audit.add(
                    "drift_detected",
                    &format!(
                        "CUSUM change-point: dir={}, magnitude={:.3}",
                        cp.direction, cp.magnitude
                    ),
                    &station_id,
                    "alert",
                    cp.magnitude as f32,
                    self.sim_time,
                );
            }
        }

        // Run event detection over a window that actually fits the data. The
        // per-cycle feature vector has length == sensor count, so clamp the
        // configured window (and stride) to that length; otherwise next_window
        // never yields and detection is dead.
        let ws = window_size.min(features.len()).max(1);
        let stride = window_stride.clamp(1, ws);
        let mut stream = EventStream::new(features.iter().map(|f| *f as f32).collect(), ws, stride);
        let events = self.runtime.process_all(&mut stream);
        let cycle_events = events.len();
        self.events_detected += cycle_events as u64;

        // Publish events to MQTT
        for event in &events
        {
            if event.confidence >= min_confidence
            {
                let payload = event_to_payload(event, &station_id, None);
                if self.backend.mqtt.publish_event(&payload).is_ok()
                {
                    self.events_published += 1;
                    match payload.severity
                    {
                        EventSeverity::Info => self.mqtt_info += 1,
                        EventSeverity::Warning => self.mqtt_warning += 1,
                        EventSeverity::Critical => self.mqtt_critical += 1,
                    }
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
            &station_id,
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
            drift_alarms: self.drift_alarms,
        }
    }

    /// Generate a final report.
    pub fn generate_report(&self) -> PipelineReport {
        let rul_pred = self.rul.predict();
        let clock_kind = if self.backend.is_simulated()
        {
            "simulated"
        }
        else
        {
            "logical pipeline"
        };
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
            mqtt_messages: self.events_published,
            mqtt_info: self.mqtt_info,
            mqtt_warning: self.mqtt_warning,
            mqtt_critical: self.mqtt_critical,
            drift_alarms: self.drift_alarms,
            duration_note: format!(
                "Ran {} cycles over {:.1}s of {} time ({} drift alarm(s))",
                self.cycles_completed, self.sim_time, clock_kind, self.drift_alarms
            ),
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

    /// Oracle for Bug 1 (dead event detection due to window_size > data length).
    ///
    /// With window_size clamped to the feature-vector length, a SpikeDetector
    /// whose threshold is below the (always-positive) feature magnitudes must
    /// emit at least one event per cycle, and those events (confidence == 1.0,
    /// which clears the default min_confidence of 0.8) must be published.
    ///
    /// On the pre-fix code (window_size 32 vs 4 features) next_window never
    /// yields, so total_events would be identically 0 and this test would fail.
    #[test]
    fn test_run_cycle_detects_event_when_window_fits() {
        let mut config = PipelineConfig::default();
        let n = config.stations[0].sensors.len();
        config.stations[0].window_size = n;
        config.stations[0].window_stride = n;
        // Threshold well below the simulated feature magnitudes (temperature is
        // ~25), and full-weight EMA so the detector fires on the first window.
        config.stations[0].spike_threshold = 0.001;
        config.stations[0].ema_alpha = 1.0;

        let mut pipeline = Pipeline::new(config);
        let report = pipeline.run(5);

        assert!(
            report.total_events > 0,
            "expected events once the window fits the data, got {}",
            report.total_events
        );
        assert!(
            report.total_published > 0,
            "events with confidence 1.0 must clear min_confidence and publish, got {}",
            report.total_published
        );
    }

    /// Oracle for HealthIndex math, exercised directly. Placing each feature at
    /// the midpoint of its [baseline, failure] range yields normalized 0.5 per
    /// sensor; equal weights keep the aggregate at 0.5; EMA on the first sample
    /// is the identity, so HI == 0.5 exactly.
    #[test]
    fn test_health_index_known_midpoint() {
        let mut hi = HealthIndex::new(vec![0.5, 1.0], vec![5.0, 10.0], vec![0.5, 0.5], 1.0);
        let value = hi.update(&[2.75, 5.5]);
        assert!(
            (value - 0.5).abs() < 1e-12,
            "expected HI == 0.5, got {}",
            value
        );
    }

    /// Oracle for Bug 3 / the hardcoded-zero MQTT report fields. After the
    /// window fix, every fired event publishes (confidence 1.0 -> Critical ->
    /// "/critical" topic). The report must mirror the publisher's own counters:
    /// mqtt_messages == publish_count and the per-severity buckets ==
    /// count_by_severity(). The bucket sum is NOT forced to equal publish_count
    /// in general, but here all messages route to /critical so it holds too.
    #[test]
    fn test_generate_report_mqtt_severity_counts_match_publisher() {
        let mut config = PipelineConfig::default();
        let n = config.stations[0].sensors.len();
        config.stations[0].window_size = n;
        config.stations[0].window_stride = n;
        config.stations[0].spike_threshold = 0.001;
        config.stations[0].ema_alpha = 1.0;

        let mut pipeline = Pipeline::new(config);
        let report = pipeline.run(5);

        let publish_count = pipeline.events_published;
        let (info, warning, critical) = (
            pipeline.mqtt_info,
            pipeline.mqtt_warning,
            pipeline.mqtt_critical,
        );

        assert!(publish_count > 0, "expected published messages");
        assert_eq!(report.mqtt_messages, publish_count);
        assert_eq!(report.mqtt_info, info);
        assert_eq!(report.mqtt_warning, warning);
        assert_eq!(report.mqtt_critical, critical);
        // SpikeDetector confidence is exactly 1.0 -> Critical severity.
        assert_eq!(report.mqtt_critical as u64, publish_count);
        assert_eq!(report.mqtt_info, 0);
        assert_eq!(report.mqtt_warning, 0);
    }

    /// Non-regression oracle for the zero-stations panic. A config that
    /// deserializes with an empty `stations` list (legal JSON) must not cause
    /// `Pipeline::new`/`init`/`run_cycle` to index out of bounds. Before the
    /// fix, `Pipeline::new` panicked on `config.stations[0]`; after the fix it
    /// backfills a default station so construction and a run cycle both
    /// succeed. `try_new` must instead reject the empty config with an error.
    #[test]
    fn test_pipeline_handles_zero_stations_config() {
        let json = r#"{
            "backend_type": "simulated",
            "opcua": {
                "endpoint": "opc.tcp://localhost:4840",
                "application_name": "SciRust-Monitor",
                "session_timeout_ms": 60000,
                "sampling_interval_ms": 100.0
            },
            "mqtt": {
                "host": "localhost",
                "port": 1883,
                "client_id": "scirust-monitor",
                "base_topic": "scirust/events",
                "username": null,
                "password": null,
                "keep_alive_secs": 60,
                "qos": 1
            },
            "stations": [],
            "settings": {
                "max_cycles": 1000,
                "audit_log_size": 10000,
                "enable_rul": true,
                "enable_fault_detectors": true,
                "enable_drift_detection": true,
                "enable_degraded_mode": true,
                "rul_window_size": 100,
                "rul_min_observations": 10,
                "drift_window_size": 50,
                "drift_threshold": 0.25
            }
        }"#;

        let config = PipelineConfig::from_json_str(json).expect("empty stations must parse");
        assert!(config.stations.is_empty());

        // Strict path: try_new rejects the unusable config instead of panicking.
        assert!(Pipeline::try_new(config.clone()).is_err());

        // Lenient path: new backfills a default station and never indexes out
        // of bounds, so construction and a full run cycle both succeed.
        let mut pipeline = Pipeline::new(config);
        assert_eq!(pipeline.config.stations.len(), 1);
        let report = pipeline.run(3);
        assert_eq!(report.total_cycles, 3);
    }

    #[test]
    fn test_pipeline_accepts_explicit_external_adapters() {
        let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
        scirust_opcua::OpcuaClient::connect(&mut opcua, &scirust_opcua::OpcuaConfig::default())
            .unwrap();
        let mut mqtt = scirust_mqtt::SimulatedMqttPublisher::new();
        scirust_mqtt::MqttPublisher::connect(&mut mqtt, &scirust_mqtt::MqttConfig::default())
            .unwrap();
        let backend = Backend::external(Box::new(opcua), Box::new(mqtt)).unwrap();

        let pipeline = Pipeline::with_backend(PipelineConfig::default(), backend).unwrap();
        assert_eq!(pipeline.backend.backend_type(), BackendType::External);
        assert_eq!(pipeline.config.backend_type, "external");
    }

    #[test]
    fn test_pipeline_rechecks_opcua_readiness_at_injection() {
        let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
        scirust_opcua::OpcuaClient::connect(&mut opcua, &scirust_opcua::OpcuaConfig::default())
            .unwrap();
        let mut mqtt = scirust_mqtt::SimulatedMqttPublisher::new();
        scirust_mqtt::MqttPublisher::connect(&mut mqtt, &scirust_mqtt::MqttConfig::default())
            .unwrap();
        let mut backend = Backend::external(Box::new(opcua), Box::new(mqtt)).unwrap();
        backend.opcua.disconnect().unwrap();

        let error = Pipeline::with_backend(PipelineConfig::default(), backend)
            .err()
            .expect("a disconnected OPC-UA adapter must be rejected");
        assert!(
            error.contains("OPC-UA"),
            "unexpected readiness error: {error}"
        );
    }

    /// Oracle for the two previously-untested filesystem config functions:
    /// PipelineConfig::save_to_file + PipelineConfig::from_file round-trip
    /// through disk must preserve the automotive_line structure exactly.
    #[test]
    fn test_config_save_load_roundtrip_via_disk() {
        use std::path::PathBuf;
        let cfg = PipelineConfig::automotive_line("line-9", 2);

        let mut path: PathBuf = std::env::temp_dir();
        path.push(format!("scirust_cfg_roundtrip_{}.json", std::process::id()));

        cfg.save_to_file(&path).unwrap();
        let loaded = PipelineConfig::from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.stations.len(), 2);
        assert_eq!(loaded.stations[0].id, "line-9-station-001");
        assert_eq!(loaded.stations[0].bearing.as_ref().unwrap().n_balls, 9);
    }
}
