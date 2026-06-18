use crate::detectors::DetectorResult;
use crate::flow::FlowWindow;
use serde::{Deserialize, Serialize};

/// Configuration du détecteur de beaconing C2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaconConfig {
    /// Nombre minimal de connexions entre une paire src/dst
    pub min_connections: usize,
    /// Seuil de régularité (coefficient de variation des intervalles < threshold = régulier)
    pub regularity_threshold: f64,
    /// Seuil d'entropie des intervalles (faible = régulier)
    pub interval_entropy_bins: usize,
    /// Seuil minimal de confiance
    pub min_confidence: f32,
    /// Seuil de payload size variance (faible = beaconing)
    pub payload_variance_threshold: f64,
}

impl Default for BeaconConfig {
    fn default() -> Self {
        Self {
            min_connections: 5,
            regularity_threshold: 0.3,
            interval_entropy_bins: 10,
            min_confidence: 0.7,
            payload_variance_threshold: 0.5,
        }
    }
}

/// Détecteur de beaconing C2 (Command and Control).
///
/// Les malwares communiquent souvent avec leur serveur C2 via des
/// connexions régulières (beaconing). Ce détecteur identifie les
/// patterns temporels réguliers en analysant les intervalles
/// inter-connexions entre paires src/dst.
#[derive(Debug, Clone)]
pub struct BeaconDetector {
    pub config: BeaconConfig,
    detected_count: u64,
}

impl BeaconDetector {
    pub fn new(config: BeaconConfig) -> Self {
        Self {
            config,
            detected_count: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(BeaconConfig::default())
    }

    /// Analyser une fenêtre de flux pour détecter le beaconing C2.
    pub fn analyze(&mut self, window: &FlowWindow) -> Vec<DetectorResult> {
        let mut results = Vec::new();

        // Grouper par (src_ip, dst_ip)
        let mut pairs: std::collections::HashMap<(String, String), Vec<&crate::flow::Flow>> =
            std::collections::HashMap::new();
        for f in &window.flows
        {
            pairs
                .entry((f.src_ip.clone(), f.dst_ip.clone()))
                .or_default()
                .push(f);
        }

        for ((src_ip, dst_ip), flows) in &pairs
        {
            if flows.len() < self.config.min_connections
            {
                continue;
            }

            // Trier par start_time
            let mut times: Vec<f64> = flows.iter().map(|f| f.start_time).collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculer les intervalles inter-connexions
            let intervals: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
            if intervals.is_empty()
            {
                continue;
            }

            let mut indicators = 0u32;
            let mut score = 0.0f32;

            // Indicateur 1: Régularité des intervalles (coefficient de variation)
            let mean_interval: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let variance: f64 = intervals
                .iter()
                .map(|&x| (x - mean_interval).powi(2))
                .sum::<f64>()
                / intervals.len() as f64;
            let std_dev = variance.sqrt();
            let cv = if mean_interval > f64::EPSILON
            {
                std_dev / mean_interval
            }
            else
            {
                f64::INFINITY
            };

            if cv < self.config.regularity_threshold && mean_interval > 0.1
            {
                indicators += 1;
                score += 0.35;
            }

            // Indicateur 2: Entropie des intervalles (faible = régulier)
            let entropy = crate::protocols::shannon_entropy(
                &intervals
                    .iter()
                    .map(|&x| (x * 100.0) as u64)
                    .collect::<Vec<_>>(),
            );
            let normalized_entropy = if mean_interval > f64::EPSILON
            {
                entropy / (mean_interval * 100.0 + 1.0).log2()
            }
            else
            {
                1.0
            };

            if normalized_entropy < 0.5
            {
                indicators += 1;
                score += 0.25;
            }

            // Indicateur 3: Taille constante des payloads (beacon = même taille)
            let payload_sizes: Vec<f64> = flows.iter().map(|f| f.bytes_out as f64).collect();
            let mean_payload: f64 = payload_sizes.iter().sum::<f64>() / payload_sizes.len() as f64;
            let payload_var: f64 = payload_sizes
                .iter()
                .map(|&x| (x - mean_payload).powi(2))
                .sum::<f64>()
                / payload_sizes.len() as f64;
            let payload_cv = if mean_payload > f64::EPSILON
            {
                payload_var.sqrt() / mean_payload
            }
            else
            {
                1.0
            };

            if payload_cv < self.config.payload_variance_threshold && mean_payload > 10.0
            {
                indicators += 1;
                score += 0.25;
            }

            // Indicateur 4: Nombre de connexions (plus = plus confiant)
            if flows.len() >= 10
            {
                indicators += 1;
                score += 0.15;
            }

            let confidence = score.min(1.0);
            if confidence >= self.config.min_confidence && indicators >= 2
            {
                self.detected_count += 1;
                results.push(DetectorResult {
                    detector: "beacon".to_string(),
                    label_en: "c2_beaconing".to_string(),
                    label_fr: "balisage_c2".to_string(),
                    confidence,
                    severity: if confidence >= 0.9 {
                        "CRITICAL".to_string()
                    } else if confidence >= 0.8 {
                        "WARNING".to_string()
                    } else {
                        "INFO".to_string()
                    },
                    source_ip: src_ip.clone(),
                    destination_ip: dst_ip.clone(),
                    details: format!(
                        "connections={} mean_interval={:.2}s cv={:.3} entropy={:.2} payload_cv={:.3}",
                        flows.len(), mean_interval, cv, normalized_entropy, payload_cv
                    ),
                });
            }
        }

        results
    }

    pub fn detected_count(&self) -> u64 {
        self.detected_count
    }

    pub fn reset(&mut self) {
        self.detected_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Flow;

    fn make_beacon_window() -> FlowWindow {
        let mut window = FlowWindow::new(0.0, 300.0);
        // Connexions régulières toutes les 30 secondes
        for i in 0..10
        {
            let mut f = Flow::new("10.0.0.1", "192.168.1.100", 40000 + i, 443);
            f.bytes_out = 256; // taille constante
            f.bytes_in = 512;
            f.packets_out = 1;
            f.packets_in = 1;
            f.start_time = i as f64 * 30.0; // régulier
            f.end_time = f.start_time + 0.5;
            window.push(f);
        }
        window
    }

    #[test]
    fn test_detect_beacon() {
        let window = make_beacon_window();
        let mut det = BeaconDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(!results.is_empty(), "should detect beaconing");
        assert_eq!(results[0].label_en, "c2_beaconing");
    }

    #[test]
    fn test_no_beacon_random_traffic() {
        let mut window = FlowWindow::new(0.0, 300.0);
        for i in 0..10
        {
            let mut f = Flow::new("10.0.0.1", "192.168.1.100", 40000 + i, 80);
            f.bytes_out = (100 + i * 50) as u64; // taille variable
            f.bytes_in = (200 + i * 100) as u64;
            f.packets_out = 1;
            f.packets_in = 1;
            // Intervalles aléatoires
            f.start_time = i as f64 * (10.0 + (i as f64 * 3.7 % 15.0));
            f.end_time = f.start_time + 1.0;
            window.push(f);
        }
        let mut det = BeaconDetector::with_defaults();
        let results = det.analyze(&window);
        // Les intervalles aléatoires ne devraient pas déclencher
        // (sauf si hasard, mais très improbable avec ces paramètres)
        assert!(results.is_empty() || results[0].confidence < 0.8);
    }
}
