use scirust_mqtt::{MqttPublisher, SimulatedMqttPublisher};
use scirust_opcua::OpcuaClient;
use serde::{Deserialize, Serialize};

/// Origin of the clients used by an industrial pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BackendType {
    /// Deterministic in-memory clients; no PLC or broker is required.
    #[default]
    Simulated,
    /// Caller-supplied [`OpcuaClient`] and [`MqttPublisher`] adapters.
    External,
}

impl BackendType {
    pub fn label(&self) -> &'static str {
        match self
        {
            Self::Simulated => "simulated",
            Self::External => "external",
        }
    }

    pub fn parse_from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str()
        {
            "simulated" | "sim" | "test" => Some(Self::Simulated),
            "external" | "custom" | "injected" => Some(Self::External),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self
        {
            Self::Simulated => "Simulated sensors and publisher; no external hardware required",
            Self::External => "Caller-supplied OPC-UA and MQTT transport adapters",
        }
    }
}

/// Unified pair of OPC-UA and MQTT clients.
///
/// SciRust owns the protocol-neutral traits. Production transports are injected
/// with [`Backend::external`], so this crate never pretends to open a PLC or
/// broker when only its in-memory implementations are present.
pub struct Backend {
    pub backend_type: BackendType,
    pub opcua: Box<dyn OpcuaClient>,
    pub mqtt: Box<dyn MqttPublisher>,
}

/// Point-in-time transport readiness for an industrial backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendHealth {
    pub opcua_connected: bool,
    pub mqtt_connected: bool,
}

impl BackendHealth {
    /// Both transports are required for the read-process-publish pipeline.
    pub fn is_ready(&self) -> bool {
        self.opcua_connected && self.mqtt_connected
    }
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend")
            .field("backend_type", &self.backend_type)
            .field("connected", &self.is_connected())
            .finish()
    }
}

impl Backend {
    /// Build a production backend from already-created transport adapters.
    ///
    /// The adapters may be backed by any OPC-UA/MQTT implementation. They are
    /// expected to be connected by the caller, because construction details and
    /// credentials are transport-specific.
    pub fn external(
        opcua: Box<dyn OpcuaClient>,
        mqtt: Box<dyn MqttPublisher>,
    ) -> Result<Self, String> {
        let backend = Self {
            backend_type: BackendType::External,
            opcua,
            mqtt,
        };
        backend.ensure_ready()?;
        Ok(backend)
    }

    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    pub fn is_simulated(&self) -> bool {
        self.backend_type == BackendType::Simulated
    }

    pub fn is_connected(&self) -> bool {
        self.health().is_ready()
    }

    /// Return the current readiness of each required transport.
    pub fn health(&self) -> BackendHealth {
        BackendHealth {
            opcua_connected: self.opcua.is_connected(),
            mqtt_connected: self.mqtt.is_connected(),
        }
    }

    /// Fail with a component-specific error when the pipeline is not ready.
    pub fn ensure_ready(&self) -> Result<(), String> {
        let health = self.health();
        if !health.opcua_connected
        {
            return Err("OPC-UA adapter is not connected".to_string());
        }
        if !health.mqtt_connected
        {
            return Err("MQTT adapter is not connected".to_string());
        }
        Ok(())
    }
}

/// Factory for the transport implementations bundled with this crate.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a bundled backend from configuration.
    ///
    /// Only the in-memory backend is constructible without a transport-specific
    /// dependency. Use [`Backend::external`] for production adapters.
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
                    .map_err(|error| format!("OPC-UA connect error: {error}"))?;
                let mut mqtt = SimulatedMqttPublisher::new();
                mqtt.connect(mqtt_config)
                    .map_err(|error| format!("MQTT connect error: {error}"))?;
                Ok(Backend {
                    backend_type: BackendType::Simulated,
                    opcua: Box::new(opcua),
                    mqtt: Box::new(mqtt),
                })
            },
            BackendType::External => Err(
                "external transports cannot be inferred from configuration; inject connected adapters with Backend::external"
                    .to_string(),
            ),
        }
    }

    /// Create a simulated backend (always available).
    pub fn simulated() -> Backend {
        Self::create(
            &scirust_opcua::OpcuaConfig::default(),
            &scirust_mqtt::MqttConfig::default(),
            BackendType::Simulated,
        )
        .expect("built-in simulated clients must accept their default configuration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_type_parsing_only_advertises_constructible_modes() {
        assert_eq!(
            BackendType::parse_from_str("sim"),
            Some(BackendType::Simulated)
        );
        assert_eq!(
            BackendType::parse_from_str("external"),
            Some(BackendType::External)
        );
        assert_eq!(BackendType::parse_from_str("opcua"), None);
        assert_eq!(BackendType::parse_from_str("mqtt"), None);
        assert_eq!(BackendType::parse_from_str("file_replay"), None);
    }

    #[test]
    fn simulated_factory_returns_connected_clients() {
        let backend = BackendFactory::simulated();
        assert!(backend.is_simulated());
        assert!(backend.is_connected());
        assert_eq!(
            backend.health(),
            BackendHealth {
                opcua_connected: true,
                mqtt_connected: true,
            }
        );
    }

    #[test]
    fn external_adapters_are_injected_instead_of_faked() {
        let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
        opcua
            .connect(&scirust_opcua::OpcuaConfig::default())
            .unwrap();
        let mut mqtt = SimulatedMqttPublisher::new();
        mqtt.connect(&scirust_mqtt::MqttConfig::default()).unwrap();

        let backend = Backend::external(Box::new(opcua), Box::new(mqtt)).unwrap();
        assert_eq!(backend.backend_type(), BackendType::External);
        assert!(!backend.is_simulated());
        assert!(backend.is_connected());
    }

    #[test]
    fn configuration_cannot_silently_fabricate_external_transports() {
        let result = BackendFactory::create(
            &scirust_opcua::OpcuaConfig::default(),
            &scirust_mqtt::MqttConfig::default(),
            BackendType::External,
        );
        assert!(result.unwrap_err().contains("Backend::external"));
    }

    #[test]
    fn external_backend_rejects_disconnected_opcua_even_when_mqtt_is_ready() {
        let opcua = scirust_opcua::SimulatedOpcuaClient::new();
        let mut mqtt = SimulatedMqttPublisher::new();
        mqtt.connect(&scirust_mqtt::MqttConfig::default()).unwrap();

        let error = Backend::external(Box::new(opcua), Box::new(mqtt)).unwrap_err();
        assert!(
            error.contains("OPC-UA"),
            "unexpected readiness error: {error}"
        );
    }

    #[test]
    fn external_backend_still_rejects_disconnected_mqtt() {
        let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
        opcua
            .connect(&scirust_opcua::OpcuaConfig::default())
            .unwrap();
        let mqtt = SimulatedMqttPublisher::new();

        let error = Backend::external(Box::new(opcua), Box::new(mqtt)).unwrap_err();
        assert!(
            error.contains("MQTT"),
            "unexpected readiness error: {error}"
        );
    }

    #[test]
    fn backend_health_tracks_opcua_disconnect_after_injection() {
        let mut opcua = scirust_opcua::SimulatedOpcuaClient::new();
        opcua
            .connect(&scirust_opcua::OpcuaConfig::default())
            .unwrap();
        let mut mqtt = SimulatedMqttPublisher::new();
        mqtt.connect(&scirust_mqtt::MqttConfig::default()).unwrap();
        let mut backend = Backend::external(Box::new(opcua), Box::new(mqtt)).unwrap();

        backend.opcua.disconnect().unwrap();

        assert!(!backend.is_connected());
        assert_eq!(
            backend.health(),
            BackendHealth {
                opcua_connected: false,
                mqtt_connected: true,
            }
        );
        assert!(backend.ensure_ready().unwrap_err().contains("OPC-UA"));
    }
}
