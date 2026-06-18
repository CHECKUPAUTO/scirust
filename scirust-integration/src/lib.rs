//! SciRust Integration Kit
//!
//! Simplifies connecting SciRust to real industrial systems.
//! Provides a unified backend abstraction that works with either simulated
//! data (for development) or real PLCs/brokers (for production), selected
//! via feature flags or runtime configuration.
//!
//! ## Quick Start
//! ```text
//! use scirust_integration::{Pipeline, PipelineConfig};
//!
//! let config = PipelineConfig::from_file("monitoring.toml")
//!     .unwrap_or_default();
//! let mut pipeline = Pipeline::new(config);
//! pipeline.run(100);  // 100 monitoring cycles
//! ```
//!
//! ## Architecture
//! ```text
//! [Backend (OPC-UA/MQTT/Simulated)] → [Signal Processing] → [Event Detection]
//! → [Health Index + RUL] → [Fault Detectors] → [MQTT Publish] → [Audit Log]
//! ```

pub mod backend;
pub mod config;
pub mod pipeline;
pub mod templates;

pub use backend::{Backend, BackendFactory, BackendType};
pub use config::{
    MqttBackendConfig, OpcuaBackendConfig, PipelineConfig, SensorConfig, StationConfig,
};
pub use pipeline::{Pipeline, PipelineReport, PipelineStatus};
pub use templates::{CodeTemplate, TemplateKind, generate_project};
