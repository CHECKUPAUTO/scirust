use scirust_mqtt::MqttPublisher;
use scirust_opcua::OpcuaClient;
use serde::{Deserialize, Serialize};

/// Type of industrial backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BackendType {
    /// Simulated data — no real PLC or broker needed (default, for development)
    #[default]
    Simulated,
    /// Real OPC-UA PLC connection
    OpcUa,
    /// Real MQTT broker connection
    Mqtt,
    /// File-based replay (CSV, Parquet)
    FileReplay,
}

impl BackendType {
    pub fn label(&self) -> &'static str {
        match self
        {
            BackendType::Simulated => "simulated",
            BackendType::OpcUa => "opcua",
            BackendType::Mqtt => "mqtt",
            BackendType::FileReplay => "file_replay",
        }
    }

    pub fn parse_from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str()
        {
            "simulated" | "sim" | "test" => Some(BackendType::Simulated),
            "opcua" | "opc-ua" | "plc" => Some(BackendType::OpcUa),
            "mqtt" | "broker" => Some(BackendType::Mqtt),
            "file" | "replay" | "csv" => Some(BackendType::FileReplay),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self
        {
            BackendType::Simulated => "Simulated sensors — no external hardware required",
            BackendType::OpcUa => "Real OPC-UA PLC/SCADA connection (requires opcua crate)",
            BackendType::Mqtt => "Real MQTT broker connection (requires rumqttc crate)",
            BackendType::FileReplay => "Replay from CSV/log files",
        }
    }

    pub fn requires_external_crate(&self) -> Option<&'static str> {
        match self
        {
            BackendType::OpcUa => Some("opcua"),
            BackendType::Mqtt => Some("rumqttc"),
            _ => None,
        }
    }
}

/// Unified backend abstraction.
///
/// Wraps OPC-UA client + MQTT publisher into a single interface.
/// The actual implementation (real vs simulated) is selected at construction
/// time by the `BackendFactory`.
pub struct Backend {
    pub backend_type: BackendType,
    pub opcua: Box<dyn scirust_opcua::OpcuaClient>,
    pub mqtt: Box<dyn scirust_mqtt::MqttPublisher>,
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend")
            .field("backend_type", &self.backend_type)
            .finish()
    }
}

impl Backend {
    /// Get the backend type.
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Check if this is a simulated backend.
    pub fn is_simulated(&self) -> bool {
        self.backend_type == BackendType::Simulated
    }

    /// Check if this backend is ready (connected).
    pub fn is_connected(&self) -> bool {
        self.mqtt.is_connected()
    }
}

/// Factory for creating backends based on configuration.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend from a configuration.
    ///
    /// In simulated mode, returns simulated OPC-UA + MQTT clients.
    /// In real mode, would return real protocol clients (requires feature flags).
    pub fn create(
        opcua_config: &scirust_opcua::OpcuaConfig,
        mqtt_config: &scirust_mqtt::MqttConfig,
        backend_type: BackendType,
    ) -> Result<Backend, String> {
        match backend_type
        {
            BackendType::Simulated =>
            {
                let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
                opcua
                    .connect(opcua_config)
                    .map_err(|e| format!("OPC-UA connect error: {}", e))?;
                let mut mqtt = scirust_mqtt::SimulatedMqttPublisher::new();
                mqtt.connect(mqtt_config)
                    .map_err(|e| format!("MQTT connect error: {}", e))?;
                Ok(Backend {
                    backend_type: BackendType::Simulated,
                    opcua: Box::new(opcua),
                    mqtt: Box::new(mqtt),
                })
            },
            BackendType::OpcUa =>
            {
                #[cfg(not(feature = "real-opcua"))]
                {
                    Err("Real OPC-UA backend requires the 'real-opcua' feature.\n\
                         Add to Cargo.toml:\n  scirust-integration = { features = [\"real-opcua\"] }\n\
                         And add the `opcua` crate as a dependency.\n\
                         Falling back to simulated mode is recommended for development.".to_string())
                }
                #[cfg(feature = "real-opcua")]
                {
                    // Real implementation would go here:
                    // let opcua = RealOpcuaClient::new(opcua_config);
                    // ...
                    Err(
                        "real-opcua feature is enabled but implementation is not yet available. \
                         Use simulated mode for now."
                            .to_string(),
                    )
                }
            },
            BackendType::Mqtt =>
            {
                #[cfg(not(feature = "real-mqtt"))]
                {
                    Err("Real MQTT backend requires the 'real-mqtt' feature.\n\
                         Add to Cargo.toml:\n  scirust-integration = { features = [\"real-mqtt\"] }\n\
                         And add the `rumqttc` crate as a dependency.\n\
                         Falling back to simulated mode is recommended for development.".to_string())
                }
                #[cfg(feature = "real-mqtt")]
                {
                    Err(
                        "real-mqtt feature is enabled but implementation is not yet available. \
                         Use simulated mode for now."
                            .to_string(),
                    )
                }
            },
            BackendType::FileReplay =>
            {
                Err("File replay backend not yet implemented. Use simulated mode.".to_string())
            },
        }
    }

    /// Create a simulated backend (always available, no feature flags needed).
    pub fn simulated() -> Backend {
        let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
        let _ = opcua.connect(&scirust_opcua::OpcuaConfig::default());
        let mut mqtt = scirust_mqtt::SimulatedMqttPublisher::new();
        let _ = mqtt.connect(&scirust_mqtt::MqttConfig::default());
        Backend {
            backend_type: BackendType::Simulated,
            opcua: Box::new(opcua),
            mqtt: Box::new(mqtt),
        }
    }

    /// Try to create a real backend, falling back to simulated on failure.
    pub fn try_real_or_simulated(
        opcua_config: &scirust_opcua::OpcuaConfig,
        mqtt_config: &scirust_mqtt::MqttConfig,
    ) -> Backend {
        match Self::create(opcua_config, mqtt_config, BackendType::OpcUa)
        {
            Ok(b) => b,
            Err(_) =>
            {
                // Fall back to simulated
                let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
                let _ = opcua.connect(opcua_config);
                let mut mqtt = scirust_mqtt::SimulatedMqttPublisher::new();
                let _ = mqtt.connect(mqtt_config);
                Backend {
                    backend_type: BackendType::Simulated,
                    opcua: Box::new(opcua),
                    mqtt: Box::new(mqtt),
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_parse_from_str() {
        assert_eq!(
            BackendType::parse_from_str("simulated"),
            Some(BackendType::Simulated)
        );
        assert_eq!(
            BackendType::parse_from_str("sim"),
            Some(BackendType::Simulated)
        );
        assert_eq!(
            BackendType::parse_from_str("opcua"),
            Some(BackendType::OpcUa)
        );
        assert_eq!(
            BackendType::parse_from_str("opc-ua"),
            Some(BackendType::OpcUa)
        );
        assert_eq!(BackendType::parse_from_str("plc"), Some(BackendType::OpcUa));
        assert_eq!(BackendType::parse_from_str("mqtt"), Some(BackendType::Mqtt));
        assert_eq!(
            BackendType::parse_from_str("broker"),
            Some(BackendType::Mqtt)
        );
        assert_eq!(BackendType::parse_from_str("unknown"), None);
    }

    #[test]
    fn test_backend_factory_simulated() {
        let backend = BackendFactory::simulated();
        assert!(backend.is_simulated());
        assert!(backend.is_connected());
    }

    #[test]
    fn test_backend_factory_create_simulated() {
        let opcua_cfg = scirust_opcua::OpcuaConfig::default();
        let mqtt_cfg = scirust_mqtt::MqttConfig::default();
        let backend =
            BackendFactory::create(&opcua_cfg, &mqtt_cfg, BackendType::Simulated).unwrap();
        assert!(backend.is_simulated());
    }

    #[test]
    fn test_backend_factory_real_opcua_without_feature() {
        let opcua_cfg = scirust_opcua::OpcuaConfig::default();
        let mqtt_cfg = scirust_mqtt::MqttConfig::default();
        let result = BackendFactory::create(&opcua_cfg, &mqtt_cfg, BackendType::OpcUa);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("real-opcua"));
    }

    #[test]
    fn test_backend_factory_real_mqtt_without_feature() {
        let opcua_cfg = scirust_opcua::OpcuaConfig::default();
        let mqtt_cfg = scirust_mqtt::MqttConfig::default();
        let result = BackendFactory::create(&opcua_cfg, &mqtt_cfg, BackendType::Mqtt);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("real-mqtt"));
    }

    #[test]
    fn test_backend_factory_try_real_or_simulated() {
        let opcua_cfg = scirust_opcua::OpcuaConfig::default();
        let mqtt_cfg = scirust_mqtt::MqttConfig::default();
        let backend = BackendFactory::try_real_or_simulated(&opcua_cfg, &mqtt_cfg);
        assert!(backend.is_simulated()); // falls back to simulated
    }

    #[test]
    fn test_backend_type_requires_external_crate() {
        assert_eq!(BackendType::OpcUa.requires_external_crate(), Some("opcua"));
        assert_eq!(BackendType::Mqtt.requires_external_crate(), Some("rumqttc"));
        assert_eq!(BackendType::Simulated.requires_external_crate(), None);
    }
}
