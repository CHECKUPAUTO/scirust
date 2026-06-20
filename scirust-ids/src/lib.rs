//! SciRust Intrusion Detection System
//!
//! Système de détection d'intrusions réseau complet basé sur le traitement
//! du signal, la détection statistique de changements, les classifieurs ML,
//! la corrélation d'alertes et l'export SIEM.
//!
//! Zero external dependencies beyond SciRust workspace crates.
//!
//! ## Modules
//! - **flow** — Modèle de flux réseau (Flow, FlowWindow)
//! - **protocols** — Analyse de protocoles et extraction de features
//! - **capture** — Abstraction de capture réseau (trait, simulated backend)
//! - **parsers** — Décodeurs protocoles: HTTP, DNS, SSH
//! - **detectors** — Détecteurs d'attaques:
//!   - `PortScanDetector` — Scan de ports (vertical, horizontal, complet)
//!   - `DdosDetector` — SYN flood, RST flood, volumétrique, applicatif
//!   - `BruteForceDetector` — Force brute, dictionary, credential stuffing
//!   - `DnsTunnelDetector` — Tunneling DNS (exfiltration)
//!   - `BeaconDetector` — Beaconing C2 (Command and Control)
//! - **correlator** — Corrélation d'alertes (multi-attack, escalation, coordonnée)
//! - **learner** — Détection d'anomalies par autoencodeur ML
//! - **siem** — Export SIEM (JSON, NDJSON, CEF, Syslog, LEEF)
//! - **engine** — Moteur IDS intégré orchestrant tous les sous-systèmes
//! - **alert** — Système d'alertes avec recommandations d'action
//!
//! ## Architecture
//! ```text
//! Capture (pcap/simulated) -> RawPacket -> FlowWindow
//!     |
//!     v
//! IdsEngine
//!     |-- Parsers (HTTP/DNS/SSH) -> ParsedPayload
//!     |-- Detectors (5x) -> DetectorResult
//!     |-- AnomalyModel (ML) -> anomaly_score
//!     |-- CUSUM/PageHinkley -> change detection
//!     |
//!     v
//! AlertCorrelator -> CorrelationResult
//!     |
//!     v
//! SiemExporter -> JSON/CEF/Syslog/NDJSON/LEEF
//! ```
//!
//! ## Exemple d'utilisation
//! ```rust
//! use scirust_ids::*;
//!
//! let mut engine = IdsEngine::with_defaults();
//! let mut window = FlowWindow::new(0.0, 60.0);
//! // ... remplir window avec des flux ...
//! let report = engine.analyze(&window, 1000.0);
//!
//! // Corréler les alertes
//! let mut correlator = AlertCorrelator::with_defaults();
//! let correlations = correlator.add_results(&report.results, report.timestamp);
//!
//! // Exporter en SIEM
//! let mut siem = SiemExporter::with_defaults();
//! siem.push_results(&report.results, report.timestamp, "my-ids");
//! ```

pub mod alert;
pub mod capture;
pub mod correlator;
pub mod detectors;
pub mod engine;
pub mod flow;
pub mod hashchain;
pub mod learner;
pub mod ot;
pub mod ot_dnp3;
pub mod ot_firmware;
pub mod ot_goose;
pub mod ot_plc;
pub mod parsers;
pub mod protocols;
pub mod siem;

pub use alert::{Alert, AlertLog, AlertSeverity};
pub use capture::{CaptureConfig, NetworkCapture, RawPacket, SimulatedCapture};
pub use correlator::{AlertCorrelator, CorrelationResult, CorrelationType};
pub use detectors::{
    BeaconDetector, BruteForceDetector, DdosDetector, DetectorResult, DnsTunnelDetector,
    PortScanDetector,
};
pub use engine::{IdsConfig, IdsEngine, IdsReport};
pub use flow::{Flow, FlowDirection, FlowWindow, Protocol};
pub use learner::{AnomalyModel, IdsLearner, LearnerConfig};
pub use ot::{ModbusEvent, ModbusProfile, OtGuard};
pub use ot_dnp3::{Dnp3Event, Dnp3Guard, Dnp3Profile};
pub use ot_firmware::{Attestation, FirmwareBaseline};
pub use ot_goose::{GooseFrame, GooseMonitor, GooseVerdict};
pub use ot_plc::{PlcBaseline, PlcRung, PlcVerdict};
pub use parsers::{MultiParser, ParsedPayload, dns::DnsParser, http::HttpParser, ssh::SshParser};
pub use protocols::{
    count_by_key, flow_features, identify_protocol_from_payload, inter_arrival_times,
    shannon_entropy, window_features,
};
pub use siem::{SiemConfig, SiemEvent, SiemExporter, SiemFormat};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pipeline() {
        let mut engine = IdsEngine::with_defaults();
        let mut window = FlowWindow::new(0.0, 60.0);

        for port in 1..=30
        {
            let mut f = Flow::new("attacker.com", "10.0.0.2", 40000 + port, port);
            f.start_time = port as f64 * 0.1;
            f.end_time = f.start_time + 0.05;
            f.packets_out = 1;
            window.push(f);
        }

        let report = engine.analyze(&window, 1000.0);
        assert!(report.alert_count > 0);
        assert!(report.results.iter().any(|r| r.detector == "port_scan"));
    }

    #[test]
    fn test_full_production_pipeline() {
        // 1. Capture
        let mut cap = SimulatedCapture::new();
        let config = CaptureConfig::default();
        for i in 0..40
        {
            let pkt = SimulatedCapture::fake_tcp_packet(
                "attacker.com",
                "10.0.0.2",
                40000 + i,
                1 + i,
                0x02,
                0,
                i as f64,
            );
            cap.inject(vec![pkt]);
        }
        cap.start(&config).unwrap();
        let window = cap.capture_window(&config, 0.0).unwrap();

        // 2. Engine analysis
        let mut engine = IdsEngine::with_defaults();
        let report = engine.analyze(&window, 1000.0);
        assert!(report.alert_count > 0);

        // 3. Correlation
        let mut correlator = AlertCorrelator::with_defaults();
        let _correlations = correlator.add_results(&report.results, report.timestamp);
        // May or may not correlate depending on alert count

        // 4. SIEM export
        let mut siem = SiemExporter::with_defaults();
        siem.push_results(&report.results, report.timestamp, "test-ids");
        let output = siem.flush().unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_learner_integration() {
        let config = LearnerConfig {
            input_features: 10,
            hidden_size: 8,
            epochs: 50,
            ..Default::default()
        };
        let mut learner = IdsLearner::new(config);

        // Entraîner sur du trafic normal
        let normal: Vec<Vec<f64>> = (0..200)
            .map(|i| (0..10).map(|j| ((i + j) as f64 * 0.05).sin()).collect())
            .collect();
        let test_normal = normal[100].clone();
        learner.add_normal_samples(normal);
        learner.train();
        assert!(learner.is_trained());

        // Trafic normal: score plus bas que l'anomalie
        let normal_score = learner.score(&test_normal);
        let anomaly_sample = vec![1000.0; 10];
        let anomaly_score = learner.score(&anomaly_sample);
        assert!(
            normal_score < anomaly_score,
            "normal score {} should be < anomaly score {}",
            normal_score,
            anomaly_score
        );
        // L'anomalie doit être détectée
        assert!(learner.detect(&anomaly_sample).is_some());
    }

    #[test]
    fn test_alert_log_integration() {
        let mut engine = IdsEngine::with_defaults();
        let mut log = AlertLog::with_defaults();
        let mut window = FlowWindow::new(0.0, 60.0);

        for port in 1..=25
        {
            let mut f = Flow::new("attacker", "10.0.0.2", 40000 + port, port);
            f.start_time = port as f64 * 0.1;
            f.end_time = f.start_time + 0.05;
            f.packets_out = 1;
            window.push(f);
        }

        let report = engine.analyze(&window, 1000.0);
        let count = log.push_results(&report.results, report.timestamp);
        assert!(count > 0);
        assert!(!log.is_empty());
    }
}
