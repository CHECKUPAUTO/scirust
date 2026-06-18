pub mod beacon;
pub mod brute_force;
pub mod ddos;
pub mod dns_tunnel;
pub mod port_scan;

pub use beacon::BeaconDetector;
pub use brute_force::BruteForceDetector;
pub use ddos::DdosDetector;
pub use dns_tunnel::DnsTunnelDetector;
pub use port_scan::PortScanDetector;

use serde::{Deserialize, Serialize};

/// Résultat d'un détecteur d'intrusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorResult {
    /// Nom du détecteur
    pub detector: String,
    /// Étiquette anglaise
    pub label_en: String,
    /// Étiquette française
    pub label_fr: String,
    /// Score de confiance (0.0 à 1.0)
    pub confidence: f32,
    /// Sévérité: INFO, WARNING, CRITICAL
    pub severity: String,
    /// IP source
    pub source_ip: String,
    /// IP destination
    pub destination_ip: String,
    /// Détails textuels
    pub details: String,
}

impl DetectorResult {
    pub fn is_critical(&self) -> bool {
        self.severity == "CRITICAL"
    }

    pub fn is_warning(&self) -> bool {
        self.severity == "WARNING"
    }

    pub fn to_event(&self, id: u64, timestamp: f64) -> scirust_events_core::Event {
        scirust_events_core::Event {
            id,
            timestamp,
            label_en: self.label_en.clone(),
            label_fr: self.label_fr.clone(),
            confidence: self.confidence,
            data_snapshot: None,
        }
    }
}
