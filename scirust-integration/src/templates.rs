use crate::config::PipelineConfig;

/// Type of code template to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    /// Minimal monitoring example
    Minimal,
    /// Full automotive line with all features
    AutomotiveLine,
    /// Bearing fault detection focus
    BearingFault,
    /// Predictive maintenance with RUL
    PredictiveMaintenance,
}

impl TemplateKind {
    pub fn parse_from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str()
        {
            "minimal" | "basic" => Some(TemplateKind::Minimal),
            "automotive" | "line" => Some(TemplateKind::AutomotiveLine),
            "bearing" | "bearing-fault" => Some(TemplateKind::BearingFault),
            "pdm" | "rul" | "predictive" => Some(TemplateKind::PredictiveMaintenance),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self
        {
            TemplateKind::Minimal => "minimal",
            TemplateKind::AutomotiveLine => "automotive-line",
            TemplateKind::BearingFault => "bearing-fault",
            TemplateKind::PredictiveMaintenance => "predictive-maintenance",
        }
    }

    pub fn description(&self) -> &'static str {
        match self
        {
            TemplateKind::Minimal =>
            {
                "Minimal monitoring: 1 station, simulated backend, spike detection"
            },
            TemplateKind::AutomotiveLine =>
            {
                "Full automotive line: multiple stations, bearing diagnostics, RUL, MQTT, audit"
            },
            TemplateKind::BearingFault =>
            {
                "Bearing fault detection: FFT envelope, BPFO/BPFI/BSF detection"
            },
            TemplateKind::PredictiveMaintenance =>
            {
                "Predictive maintenance: Health Index, RUL estimation, CUSUM"
            },
        }
    }
}

/// A code template ready to be written to disk.
pub struct CodeTemplate {
    pub filename: String,
    pub content: String,
    pub description: String,
}

/// Generate a complete project scaffold.
///
/// Creates:
/// - `Cargo.toml` with the right dependencies
/// - `src/main.rs` with the monitoring code
/// - `config.json` with a default pipeline configuration
/// - `README.md` with usage instructions
pub fn generate_project(kind: TemplateKind, project_name: &str) -> Vec<CodeTemplate> {
    let mut templates = Vec::new();

    // Cargo.toml
    templates.push(CodeTemplate {
        filename: "Cargo.toml".to_string(),
        content: generate_cargo_toml(project_name),
        description: "Project manifest with SciRust industrial dependencies".to_string(),
    });

    // config.json
    let config = match kind
    {
        TemplateKind::Minimal => PipelineConfig::default(),
        TemplateKind::AutomotiveLine => PipelineConfig::automotive_line("line-1", 3),
        TemplateKind::BearingFault =>
        {
            let mut cfg = PipelineConfig::default();
            cfg.stations[0].bearing = Some(crate::config::BearingConfig {
                pitch_diameter: 39.04,
                ball_diameter: 7.94,
                n_balls: 9,
                contact_angle_deg: 0.0,
            });
            cfg.stations[0].shaft_freq = Some(29.53);
            cfg
        },
        TemplateKind::PredictiveMaintenance => PipelineConfig::default(),
    };
    let config_json = serde_json::to_string_pretty(&config).unwrap_or_default();
    templates.push(CodeTemplate {
        filename: "config.json".to_string(),
        content: config_json,
        description: "Pipeline configuration".to_string(),
    });

    // src/main.rs
    templates.push(CodeTemplate {
        filename: "src/main.rs".to_string(),
        content: match kind
        {
            TemplateKind::Minimal => generate_minimal_main(project_name),
            TemplateKind::AutomotiveLine => generate_automotive_main(project_name),
            TemplateKind::BearingFault => generate_bearing_main(project_name),
            TemplateKind::PredictiveMaintenance => generate_pdm_main(project_name),
        },
        description: "Main monitoring application".to_string(),
    });

    // README.md
    templates.push(CodeTemplate {
        filename: "README.md".to_string(),
        content: generate_readme(project_name, kind),
        description: "Project documentation".to_string(),
    });

    // .gitignore
    templates.push(CodeTemplate {
        filename: ".gitignore".to_string(),
        content: "/target\n*.log\n".to_string(),
        description: "Git ignore file".to_string(),
    });

    templates
}

fn generate_cargo_toml(project_name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[dependencies]
scirust-integration = {{ path = "../scirust-integration" }}
scirust-signal = {{ path = "../scirust-signal" }}
scirust-pdm = {{ path = "../scirust-pdm" }}
scirust-opcua = {{ path = "../scirust-opcua" }}
scirust-mqtt = {{ path = "../scirust-mqtt" }}
serde_json = "1"

# For production, implement OpcuaClient / MqttPublisher with your selected
# transport crates, then inject the connected clients through Backend::external
# and Pipeline::with_backend.
# opcua = "0.13"
# rumqttc = "0.24"
"#,
        name = project_name
    )
}

fn generate_minimal_main(_name: &str) -> String {
    r#"//! Minimal Industrial Monitor
//!
//! Reads sensor data (simulated by default), detects events,
//! and publishes alerts to MQTT.

use scirust_integration::{Pipeline, PipelineConfig};
use std::path::Path;

fn main() {
    println!("=== Minimal Industrial Monitor ===\n");

    // Load config (or use defaults)
    let config = PipelineConfig::from_file(Path::new("config.json"))
        .unwrap_or_default();

    println!("Backend: {}", config.backend_type);
    println!("Stations: {}", config.stations.len());

    // Create and run pipeline
    let mut pipeline = Pipeline::new(config);
    let report = pipeline.run(100);

    // Print report
    println!("\n=== Report ===");
    println!("Cycles: {}", report.total_cycles);
    println!("Events detected: {}", report.total_events);
    println!("Events published: {}", report.total_published);
    println!("Health Index: {:.3} ({})", report.final_health_index, report.final_health_state);
    println!("RUL: {:.1} hours", report.final_rul);
    println!("Audit entries: {}", report.audit_entries);
    println!("Audit chain valid: {}", report.audit_chain_valid);

    pipeline.shutdown();
}
"#
    .to_string()
}

fn generate_automotive_main(_name: &str) -> String {
    r#"//! Automotive Production Line Monitor
//!
//! Full pipeline: OPC-UA → Signal → Events → Health → RUL → Bearing faults → MQTT → Audit

use scirust_integration::{Pipeline, PipelineConfig};
use scirust_signal::bearing::{BearingGeometry, bpfo, bpfi, bsf};
use std::path::Path;

fn main() {
    println!("=== Automotive Line Monitor ===\n");

    let config = PipelineConfig::from_file(Path::new("config.json"))
        .unwrap_or_else(|_| {
            println!("Using default automotive configuration");
            PipelineConfig::automotive_line("line-1", 3)
        });

    // Print station info
    for station in &config.stations {
        println!("Station: {} ({})", station.id, station.name);
        println!("  Sensors: {}", station.sensors.len());
        if let Some(b) = &station.bearing {
            let geo = BearingGeometry {
                pitch_diameter: b.pitch_diameter,
                ball_diameter: b.ball_diameter,
                n_balls: b.n_balls,
                contact_angle_deg: b.contact_angle_deg,
            };
            let shaft = station.shaft_freq.unwrap_or(29.53);
            println!("  Bearing: BPFO={:.1}Hz, BPFI={:.1}Hz, BSF={:.1}Hz",
                bpfo(&geo, shaft), bpfi(&geo, shaft), bsf(&geo, shaft));
        }
        if let Some(asil) = &station.asil_level {
            println!("  Safety: ASIL-{}", asil);
        }
        println!();
    }

    // Run pipeline
    let mut pipeline = Pipeline::new(config);
    let report = pipeline.run(500);

    // Detailed report
    println!("=== Final Report ===");
    println!("Total cycles: {}", report.total_cycles);
    println!("Events detected: {}", report.total_events);
    println!("Events published: {}", report.total_published);
    println!("Final Health: {:.3} ({})", report.final_health_index, report.final_health_state);
    println!("RUL: {:.1}h (CI: {:.1} - {:.1})",
        report.final_rul, report.rul_lower_bound, report.rul_upper_bound);
    println!("Audit: {} entries, chain valid: {}",
        report.audit_entries, report.audit_chain_valid);

    // Export report
    if let Ok(json) = pipeline.export_report_json() {
        println!("\nJSON Report:\n{}", json);
    }

    pipeline.shutdown();
}
"#
    .to_string()
}

fn generate_bearing_main(_name: &str) -> String {
    r#"//! Bearing Fault Detection Monitor
//!
//! Focuses on bearing fault frequency detection using envelope analysis.

use scirust_integration::{Pipeline, PipelineConfig};
use scirust_signal::bearing::{BearingGeometry, bpfo, bpfi, bsf, ftf};
use scirust_pdm::detectors::{BearingFaultDetector, FaultType};
use std::path::Path;

fn main() {
    println!("=== Bearing Fault Detection Monitor ===\n");

    let config = PipelineConfig::from_file(Path::new("config.json"))
        .unwrap_or_default();

    // Print bearing fault frequencies
    if let Some(bc) = &config.stations[0].bearing {
        let geo = BearingGeometry {
            pitch_diameter: bc.pitch_diameter,
            ball_diameter: bc.ball_diameter,
            n_balls: bc.n_balls,
            contact_angle_deg: bc.contact_angle_deg,
        };
        let shaft = config.stations[0].shaft_freq.unwrap_or(29.53);
        println!("Bearing fault frequencies (shaft = {:.2} Hz):", shaft);
        println!("  BPFO (outer race): {:.2} Hz", bpfo(&geo, shaft));
        println!("  BPFI (inner race): {:.2} Hz", bpfi(&geo, shaft));
        println!("  BSF  (ball spin):  {:.2} Hz", bsf(&geo, shaft));
        println!("  FTF  (cage):       {:.2} Hz", ftf(&geo, shaft));
        println!();
    }

    let mut pipeline = Pipeline::new(config);
    let report = pipeline.run(200);

    println!("=== Report ===");
    println!("Cycles: {}", report.total_cycles);
    println!("Events: {}", report.total_events);
    println!("Health: {:.3} ({})", report.final_health_index, report.final_health_state);

    pipeline.shutdown();
}
"#
    .to_string()
}

fn generate_pdm_main(_name: &str) -> String {
    r#"//! Predictive Maintenance Monitor
//!
//! Focuses on Health Index tracking and RUL estimation.

use scirust_integration::{Pipeline, PipelineConfig};
use scirust_pdm::health::HealthState;
use std::path::Path;

fn main() {
    println!("=== Predictive Maintenance Monitor ===\n");

    let config = PipelineConfig::from_file(Path::new("config.json"))
        .unwrap_or_default();

    println!("RUL enabled: {}", config.settings.enable_rul);
    println!("Drift detection: {}", config.settings.enable_drift_detection);
    println!("Max cycles: {}", config.settings.max_cycles);
    println!();

    let mut pipeline = Pipeline::new(config);
    let report = pipeline.run(500);

    println!("=== Predictive Maintenance Report ===");
    println!("Cycles: {}", report.total_cycles);
    println!("Health Index: {:.3} ({})", report.final_health_index, report.final_health_state);
    println!("RUL: {:.1} hours", report.final_rul);
    println!("RUL 95% CI: [{:.1}, {:.1}]", report.rul_lower_bound, report.rul_upper_bound);
    println!("Events: {} detected, {} published", report.total_events, report.total_published);
    println!("Audit: {} entries, valid: {}", report.audit_entries, report.audit_chain_valid);

    // Health state interpretation
    let state = HealthState::from_index(report.final_health_index);
    match state {
        HealthState::Good => println!("\n=> System healthy, no action needed"),
        HealthState::Degraded => println!("\n=> Slight degradation — monitor closely"),
        HealthState::Warning => println!("\n=> Significant degradation — plan maintenance"),
        HealthState::Critical => println!("\n=> CRITICAL — immediate maintenance required"),
        HealthState::Failed => println!("\n=> Component failed — production halted"),
    }

    pipeline.shutdown();
}
"#
    .to_string()
}

fn generate_readme(project_name: &str, kind: TemplateKind) -> String {
    format!(
        r#"# {name}

{description}

## Quick Start

```bash
# Run with simulated data (default)
cargo run

# Production PLC/broker clients are injected through Backend::external and
# Pipeline::with_backend; the default executable remains fully simulated.
```

## Configuration

Edit `config.json` to configure:
- Backend type (`simulated`; injected clients are marked `external`)
- OPC-UA endpoint
- MQTT broker
- Monitoring stations and sensors
- Health Index baselines and thresholds

## Files

- `Cargo.toml` — Project dependencies
- `config.json` — Pipeline configuration
- `src/main.rs` — Monitoring application
- `README.md` — This file

## Generated by

SciRust Industrial Integration Kit
Template: {template}
"#,
        name = project_name,
        description = kind.description(),
        template = kind.label()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_kind_parse_from_str() {
        assert_eq!(
            TemplateKind::parse_from_str("minimal"),
            Some(TemplateKind::Minimal)
        );
        assert_eq!(
            TemplateKind::parse_from_str("automotive"),
            Some(TemplateKind::AutomotiveLine)
        );
        assert_eq!(
            TemplateKind::parse_from_str("bearing"),
            Some(TemplateKind::BearingFault)
        );
        assert_eq!(
            TemplateKind::parse_from_str("pdm"),
            Some(TemplateKind::PredictiveMaintenance)
        );
        assert_eq!(TemplateKind::parse_from_str("unknown"), None);
    }

    #[test]
    fn test_generate_project_minimal() {
        let templates = generate_project(TemplateKind::Minimal, "test-project");
        assert_eq!(templates.len(), 5);
        assert!(templates.iter().any(|t| t.filename == "Cargo.toml"));
        assert!(templates.iter().any(|t| t.filename == "config.json"));
        assert!(templates.iter().any(|t| t.filename == "src/main.rs"));
        assert!(templates.iter().any(|t| t.filename == "README.md"));
        assert!(templates.iter().any(|t| t.filename == ".gitignore"));
    }

    #[test]
    fn test_generate_project_automotive() {
        let templates = generate_project(TemplateKind::AutomotiveLine, "auto-line");
        let main = templates
            .iter()
            .find(|t| t.filename == "src/main.rs")
            .unwrap();
        assert!(main.content.contains("Automotive"));
        assert!(main.content.contains("bearing"));
    }

    #[test]
    fn test_generate_project_bearing() {
        let templates = generate_project(TemplateKind::BearingFault, "bearing-test");
        let main = templates
            .iter()
            .find(|t| t.filename == "src/main.rs")
            .unwrap();
        assert!(main.content.contains("BPFO"));
        assert!(main.content.contains("BPFI"));
    }

    #[test]
    fn test_generate_project_pdm() {
        let templates = generate_project(TemplateKind::PredictiveMaintenance, "pdm-test");
        let main = templates
            .iter()
            .find(|t| t.filename == "src/main.rs")
            .unwrap();
        assert!(main.content.contains("RUL"));
        assert!(main.content.contains("Health Index"));
    }

    #[test]
    fn test_cargo_toml_contains_real_backend_hints() {
        let templates = generate_project(TemplateKind::Minimal, "test");
        let cargo = templates
            .iter()
            .find(|t| t.filename == "Cargo.toml")
            .unwrap();
        // Generated projects document the explicit adapter-injection path.
        assert!(cargo.content.contains("Backend::external"));
        assert!(cargo.content.contains("Pipeline::with_backend"));
        assert!(cargo.content.contains("opcua"));
        assert!(cargo.content.contains("rumqttc"));
    }

    #[test]
    fn test_config_json_is_valid() {
        let templates = generate_project(TemplateKind::AutomotiveLine, "test");
        let config = templates
            .iter()
            .find(|t| t.filename == "config.json")
            .unwrap();
        let parsed: PipelineConfig = serde_json::from_str(&config.content).unwrap();
        assert!(parsed.is_valid());
    }

    #[test]
    fn test_template_descriptions() {
        assert!(TemplateKind::Minimal.description().contains("Minimal"));
        assert!(TemplateKind::AutomotiveLine.description().contains("Full"));
        assert!(TemplateKind::BearingFault.description().contains("Bearing"));
        assert!(
            TemplateKind::PredictiveMaintenance
                .description()
                .contains("RUL")
        );
    }
}
