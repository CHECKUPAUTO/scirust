use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Complete pipeline configuration.
///
/// Can be loaded from a TOML-like JSON file or created programmatically.
/// This is the single source of truth for an industrial monitoring deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Backend type (simulated, opcua, mqtt, file_replay)
    pub backend_type: String,
    /// OPC-UA connection settings
    pub opcua: OpcuaBackendConfig,
    /// MQTT broker settings
    pub mqtt: MqttBackendConfig,
    /// Monitoring stations (one per machine/cell)
    pub stations: Vec<StationConfig>,
    /// Global settings
    pub settings: GlobalSettings,
}

/// OPC-UA backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcuaBackendConfig {
    pub endpoint: String,
    pub application_name: String,
    pub session_timeout_ms: u32,
    pub sampling_interval_ms: f64,
}

impl Default for OpcuaBackendConfig {
    fn default() -> Self {
        Self {
            endpoint: "opc.tcp://localhost:4840".to_string(),
            application_name: "SciRust-Monitor".to_string(),
            session_timeout_ms: 60_000,
            sampling_interval_ms: 100.0,
        }
    }
}

impl From<&OpcuaBackendConfig> for scirust_opcua::OpcuaConfig {
    fn from(c: &OpcuaBackendConfig) -> Self {
        scirust_opcua::OpcuaConfig {
            endpoint: c.endpoint.clone(),
            application_name: c.application_name.clone(),
            session_timeout_ms: c.session_timeout_ms,
            sampling_interval_ms: c.sampling_interval_ms,
        }
    }
}

/// MQTT backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqttBackendConfig {
    pub host: String,
    pub port: u16,
    pub client_id: String,
    pub base_topic: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub keep_alive_secs: u16,
    pub qos: u8,
}

impl Default for MqttBackendConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 1883,
            client_id: "scirust-monitor".to_string(),
            base_topic: "scirust/events".to_string(),
            username: None,
            password: None,
            keep_alive_secs: 60,
            qos: 1,
        }
    }
}

impl From<&MqttBackendConfig> for scirust_mqtt::MqttConfig {
    fn from(c: &MqttBackendConfig) -> Self {
        scirust_mqtt::MqttConfig {
            host: c.host.clone(),
            port: c.port,
            client_id: c.client_id.clone(),
            base_topic: c.base_topic.clone(),
            username: c.username.clone(),
            password: c.password.clone(),
            keep_alive_secs: c.keep_alive_secs,
            qos: c.qos,
        }
    }
}

/// Configuration for a single monitoring station.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub sensors: Vec<SensorConfig>,
    /// Minimum confidence to publish an event
    pub min_confidence: f32,
    /// SpikeDetector threshold
    pub spike_threshold: f64,
    /// SpikeDetector EMA alpha
    pub ema_alpha: f64,
    /// EventStream window size
    pub window_size: usize,
    /// EventStream window stride
    pub window_stride: usize,
    /// Bearing geometry (if applicable)
    pub bearing: Option<BearingConfig>,
    /// Shaft frequency in Hz (if applicable)
    pub shaft_freq: Option<f64>,
    /// ASIL safety level
    pub asil_level: Option<String>,
}

impl Default for StationConfig {
    fn default() -> Self {
        Self {
            id: "station-001".to_string(),
            name: "Default Monitoring Station".to_string(),
            description: "Spindle motor vibration monitoring".to_string(),
            sensors: vec![
                SensorConfig::new("Vibration.X", "m/s²", "Accelerometer X-axis"),
                SensorConfig::new("Vibration.Y", "m/s²", "Accelerometer Y-axis"),
                SensorConfig::new("Vibration.Z", "m/s²", "Accelerometer Z-axis"),
                SensorConfig::new("Temperature.Motor", "°C", "Motor winding temperature"),
            ],
            min_confidence: 0.8,
            spike_threshold: 1.0,
            ema_alpha: 0.8,
            window_size: 32,
            window_stride: 8,
            bearing: None,
            shaft_freq: None,
            asil_level: None,
        }
    }
}

/// Individual sensor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    pub node_id: String,
    pub unit: String,
    pub description: String,
    /// Baseline (healthy) value for Health Index
    pub baseline: f64,
    /// Failure threshold for Health Index
    pub failure_threshold: f64,
    /// Weight in Health Index (0..1)
    pub weight: f64,
}

impl SensorConfig {
    pub fn new(node_id: &str, unit: &str, description: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            unit: unit.to_string(),
            description: description.to_string(),
            baseline: 0.1,
            failure_threshold: 5.0,
            weight: 0.25,
        }
    }

    pub fn with_baseline(mut self, baseline: f64, failure_threshold: f64, weight: f64) -> Self {
        self.baseline = baseline;
        self.failure_threshold = failure_threshold;
        self.weight = weight;
        self
    }
}

/// Bearing geometry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BearingConfig {
    pub pitch_diameter: f64,
    pub ball_diameter: f64,
    pub n_balls: usize,
    pub contact_angle_deg: f64,
}

impl From<&BearingConfig> for scirust_signal::bearing::BearingGeometry {
    fn from(c: &BearingConfig) -> Self {
        scirust_signal::bearing::BearingGeometry {
            pitch_diameter: c.pitch_diameter,
            ball_diameter: c.ball_diameter,
            n_balls: c.n_balls,
            contact_angle_deg: c.contact_angle_deg,
        }
    }
}

/// Global pipeline settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSettings {
    pub max_cycles: usize,
    pub audit_log_size: usize,
    pub enable_rul: bool,
    pub enable_fault_detectors: bool,
    pub enable_drift_detection: bool,
    pub enable_degraded_mode: bool,
    pub rul_window_size: usize,
    pub rul_min_observations: usize,
    pub drift_window_size: usize,
    pub drift_threshold: f64,
}

impl Default for GlobalSettings {
    fn default() -> Self {
        Self {
            max_cycles: 1000,
            audit_log_size: 10_000,
            enable_rul: true,
            enable_fault_detectors: true,
            enable_drift_detection: true,
            enable_degraded_mode: true,
            rul_window_size: 100,
            rul_min_observations: 10,
            drift_window_size: 50,
            drift_threshold: 0.25,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            backend_type: "simulated".to_string(),
            opcua: OpcuaBackendConfig::default(),
            mqtt: MqttBackendConfig::default(),
            stations: vec![StationConfig::default()],
            settings: GlobalSettings::default(),
        }
    }
}

impl PipelineConfig {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Cannot read config file {}: {}", path.display(), e))?;
        Self::from_json_str(&content)
    }

    /// Parse configuration from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Config parse error: {}", e))
    }

    /// Save configuration to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("Serialize error: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("Cannot write config file {}: {}", path.display(), e))
    }

    /// Generate a default configuration for a specific use case.
    pub fn automotive_line(line_id: &str, n_stations: usize) -> Self {
        let stations: Vec<StationConfig> = (0..n_stations)
            .map(|i| {
                let id = format!("{}-station-{:03}", line_id, i + 1);
                StationConfig {
                    id,
                    name: format!("Station {} — Spindle Motor", i + 1),
                    description: "CNC spindle vibration + temperature monitoring".to_string(),
                    sensors: vec![
                        SensorConfig::new("Vibration.X", "m/s²", "Accelerometer X-axis")
                            .with_baseline(0.1, 5.0, 0.25),
                        SensorConfig::new("Vibration.Y", "m/s²", "Accelerometer Y-axis")
                            .with_baseline(0.1, 5.0, 0.25),
                        SensorConfig::new("Vibration.Z", "m/s²", "Accelerometer Z-axis")
                            .with_baseline(0.1, 5.0, 0.25),
                        SensorConfig::new("Temperature.Motor", "°C", "Motor winding temp")
                            .with_baseline(40.0, 120.0, 0.25),
                    ],
                    bearing: Some(BearingConfig {
                        pitch_diameter: 39.04,
                        ball_diameter: 7.94,
                        n_balls: 9,
                        contact_angle_deg: 0.0,
                    }),
                    shaft_freq: Some(29.53),
                    asil_level: Some("B".to_string()),
                    ..StationConfig::default()
                }
            })
            .collect();

        Self {
            backend_type: "simulated".to_string(),
            stations,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.stations.is_empty()
        {
            errors.push("No monitoring stations defined".to_string());
        }
        for (i, station) in self.stations.iter().enumerate()
        {
            if station.sensors.is_empty()
            {
                errors.push(format!("Station {} ({}) has no sensors", i, station.id));
            }
            let total_weight: f64 = station.sensors.iter().map(|s| s.weight).sum();
            if (total_weight - 1.0).abs() > 0.01
            {
                errors.push(format!(
                    "Station {} sensor weights sum to {:.3} (should be ~1.0)",
                    station.id, total_weight
                ));
            }
            if station.window_size == 0
            {
                errors.push(format!("Station {} window_size is 0", station.id));
            }
            if station.window_stride == 0
            {
                errors.push(format!("Station {} window_stride is 0", station.id));
            }
        }
        if self.settings.max_cycles == 0
        {
            errors.push("max_cycles is 0".to_string());
        }
        errors
    }

    /// Check if the configuration is valid.
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = PipelineConfig::default();
        assert!(!cfg.stations.is_empty());
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_automotive_line_config() {
        let cfg = PipelineConfig::automotive_line("line-3", 5);
        assert_eq!(cfg.stations.len(), 5);
        assert!(cfg.is_valid());
        assert!(cfg.stations[0].bearing.is_some());
        assert!(cfg.stations[0].shaft_freq.is_some());
    }

    #[test]
    fn test_config_validation_catches_errors() {
        let mut cfg = PipelineConfig::default();
        cfg.stations[0].sensors.clear();
        let errors = cfg.validate();
        assert!(errors.iter().any(|e| e.contains("no sensors")));
    }

    #[test]
    fn test_config_validation_weight_mismatch() {
        let mut cfg = PipelineConfig::default();
        for s in &mut cfg.stations[0].sensors
        {
            s.weight = 0.1; // total = 0.4, not ~1.0
        }
        let errors = cfg.validate();
        assert!(errors.iter().any(|e| e.contains("weights sum to")));
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let cfg = PipelineConfig::automotive_line("line-1", 3);
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed = PipelineConfig::from_json_str(&json).unwrap();
        assert_eq!(parsed.stations.len(), 3);
        assert_eq!(parsed.stations[0].id, "line-1-station-001");
    }

    #[test]
    fn test_bearing_config_conversion() {
        let bc = BearingConfig {
            pitch_diameter: 39.04,
            ball_diameter: 7.94,
            n_balls: 9,
            contact_angle_deg: 0.0,
        };
        let geo: scirust_signal::bearing::BearingGeometry = (&bc).into();
        assert_eq!(geo.n_balls, 9);
        assert!((geo.pitch_diameter - 39.04).abs() < 1e-10);
    }

    #[test]
    fn test_opcua_config_conversion() {
        let cfg = OpcuaBackendConfig::default();
        let opcua_cfg: scirust_opcua::OpcuaConfig = (&cfg).into();
        assert_eq!(opcua_cfg.endpoint, cfg.endpoint);
    }

    #[test]
    fn test_mqtt_config_conversion() {
        let cfg = MqttBackendConfig::default();
        let mqtt_cfg: scirust_mqtt::MqttConfig = (&cfg).into();
        assert_eq!(mqtt_cfg.host, cfg.host);
        assert_eq!(mqtt_cfg.port, cfg.port);
    }

    #[test]
    fn test_empty_stations_invalid() {
        let cfg = PipelineConfig {
            stations: vec![],
            ..Default::default()
        };
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_zero_window_size_invalid() {
        let mut cfg = PipelineConfig::default();
        cfg.stations[0].window_size = 0;
        assert!(!cfg.is_valid());
    }
}
