//! SciRust Integration Kit
//!
//! Provides a protocol-neutral backend abstraction for industrial systems.
//! The deterministic simulated backend is bundled; production PLC and broker
//! transports are supplied explicitly as `OpcuaClient` / `MqttPublisher`
//! adapters and injected with `Backend::external` + `Pipeline::with_backend`.
//!
//! ## Quick Start
//! ```text
//! use scirust_integration::{Pipeline, PipelineConfig};
//!
//! let config = PipelineConfig::from_file("monitoring.json")
//!     .unwrap_or_default();
//! let mut pipeline = Pipeline::new(config);
//! pipeline.run(100);  // 100 monitoring cycles
//! ```
//!
//! ## Architecture
//! ```text
//! [Injected OPC-UA/MQTT clients or simulation] → [Signal Processing] → [Event Detection]
//! → [Health Index + RUL] → [Fault Detectors] → [MQTT Publish] → [Audit Log]
//! ```

pub mod backend;
pub mod config;
pub mod pipeline;
pub mod templates;

pub use backend::{Backend, BackendFactory, BackendHealth, BackendType};
pub use config::{
    MqttBackendConfig, OpcuaBackendConfig, PipelineConfig, SensorConfig, StationConfig,
};
pub use pipeline::{Pipeline, PipelineReport, PipelineStatus};
pub use templates::{CodeTemplate, TemplateKind, generate_project};
