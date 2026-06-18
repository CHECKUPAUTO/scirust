use crate::detectors::DetectorResult;
use crate::flow::FlowWindow;
use serde::{Deserialize, Serialize};

/// Configuration du détecteur DDoS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DdosConfig {
    /// Nombre minimal de sources uniques pour un DDoS distribué
    pub min_sources: usize,
    /// Nombre minimal de flux par fenêtre
    pub min_flows: usize,
    /// Seuil de taux de SYN pour SYN flood
    pub syn_flood_threshold: f64,
    /// Seuil de taux de RST pour RST flood
    pub rst_flood_threshold: f64,
    /// Seuil de taux d'erreurs pour attaque applicative
    pub error_rate_threshold: f64,
    /// Seuil minimal de confiance
    pub min_confidence: f32,
}

impl Default for DdosConfig {
    fn default() -> Self {
        Self {
            min_sources: 10,
            min_flows: 50,
            syn_flood_threshold: 0.7,
            rst_flood_threshold: 0.3,
            error_rate_threshold: 0.5,
            min_confidence: 0.7,
        }
    }
}

/// Type d'attaque DDoS détectée.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DdosType {
    /// SYN flood: majorité de SYN sans handshake complet
    SynFlood,
    /// RST flood: nombre anormal de RST
    _RSTFlood,
    /// Volumetric: volume de trafic anormalement élevé
    Volumetric,
    /// Application layer: taux d'erreurs HTTP élevé
    ApplicationLayer,
    /// Mixte
    Mixed,
}

impl DdosType {
    pub fn label_en(&self) -> &'static str {
        match self
        {
            DdosType::SynFlood => "syn_flood",
            DdosType::_RSTFlood => "rst_flood",
            DdosType::Volumetric => "volumetric_ddos",
            DdosType::ApplicationLayer => "application_layer_ddos",
            DdosType::Mixed => "mixed_ddos",
        }
    }

    pub fn label_fr(&self) -> &'static str {
        match self
        {
            DdosType::SynFlood => "inondation_syn",
            DdosType::_RSTFlood => "inondation_rst",
            DdosType::Volumetric => "ddos_volumétrique",
            DdosType::ApplicationLayer => "ddos_couche_applicative",
            DdosType::Mixed => "ddos_mixte",
        }
    }
}

/// Détecteur d'attaques DDoS.
///
/// Identifie les SYN floods, RST floods, attaques volumétriques
/// et attaques en couche applicative par analyse des flux agrégés.
#[derive(Debug, Clone)]
pub struct DdosDetector {
    pub config: DdosConfig,
    /// Seuil de volume (octets) pour détection volumétrique (auto-calibré)
    volume_threshold: f64,
    /// Moyenne mobile du volume
    volume_ema: f64,
    /// Alpha pour l'EMA du volume
    ema_alpha: f64,
    /// Nombre de detections
    detected_count: u64,
}

impl DdosDetector {
    pub fn new(config: DdosConfig) -> Self {
        Self {
            config,
            volume_threshold: f64::MAX,
            volume_ema: 0.0,
            ema_alpha: 0.1,
            detected_count: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DdosConfig::default())
    }

    pub fn analyze(&mut self, window: &FlowWindow) -> Vec<DetectorResult> {
        let mut results = Vec::new();

        if window.len() < self.config.min_flows
        {
            self.update_volume_baseline(window);
            return results;
        }

        let unique_sources = window.unique_sources();
        let total_bytes = window.flows.iter().map(|f| f.total_bytes()).sum::<u64>() as f64;
        let syn_rate = window.syn_rate();
        let rst_rate = window.rst_rate();
        let error_rate = window.error_rate();

        // Mise à jour du seuil volumétrique
        self.update_volume_baseline(window);

        let mut is_ddos = false;
        let mut ddos_types = Vec::new();
        let mut confidence: f32 = 0.0;

        // SYN flood
        if syn_rate > self.config.syn_flood_threshold && unique_sources >= self.config.min_sources
        {
            is_ddos = true;
            ddos_types.push(DdosType::SynFlood);
            confidence = confidence.max(0.5 + syn_rate as f32 * 0.4);
        }

        // RST flood
        if rst_rate > self.config.rst_flood_threshold
        {
            is_ddos = true;
            ddos_types.push(DdosType::_RSTFlood);
            confidence = confidence.max(0.5 + rst_rate as f32 * 0.4);
        }

        // Volumetric
        if self.volume_ema > 0.0 && total_bytes > self.volume_threshold
        {
            is_ddos = true;
            ddos_types.push(DdosType::Volumetric);
            let ratio = total_bytes / self.volume_ema;
            confidence = confidence.max(0.5 + (ratio as f32 * 0.1).min(0.5));
        }

        // Application layer
        if error_rate > self.config.error_rate_threshold && unique_sources >= 5
        {
            is_ddos = true;
            ddos_types.push(DdosType::ApplicationLayer);
            confidence = confidence.max(0.5 + error_rate as f32 * 0.3);
        }

        if is_ddos && confidence >= self.config.min_confidence
        {
            let ddos_type = if ddos_types.len() > 1
            {
                DdosType::Mixed
            }
            else
            {
                ddos_types[0]
            };

            self.detected_count += 1;
            let top_source = window
                .connections_per_source()
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(ip, _)| ip)
                .unwrap_or_default();

            results.push(DetectorResult {
                detector: "ddos".to_string(),
                label_en: ddos_type.label_en().to_string(),
                label_fr: ddos_type.label_fr().to_string(),
                confidence: confidence.min(1.0),
                severity: if confidence >= 0.9
                {
                    "CRITICAL".to_string()
                }
                else if confidence >= 0.8
                {
                    "WARNING".to_string()
                }
                else
                {
                    "INFO".to_string()
                },
                source_ip: top_source,
                destination_ip: window
                    .flows
                    .first()
                    .map(|f| f.dst_ip.clone())
                    .unwrap_or_default(),
                details: format!(
                    "sources={} syn_rate={:.2} rst_rate={:.2} err_rate={:.2} volume={:.0}",
                    unique_sources, syn_rate, rst_rate, error_rate, total_bytes
                ),
            });
        }

        results
    }

    fn update_volume_baseline(&mut self, window: &FlowWindow) {
        let total_bytes = window.flows.iter().map(|f| f.total_bytes()).sum::<u64>() as f64;
        if self.volume_ema < f64::EPSILON
        {
            self.volume_ema = total_bytes;
        }
        else
        {
            self.volume_ema =
                self.ema_alpha * total_bytes + (1.0 - self.ema_alpha) * self.volume_ema;
        }
        // Seuil = 3x la moyenne mobile
        self.volume_threshold = self.volume_ema * 3.0;
    }

    pub fn detected_count(&self) -> u64 {
        self.detected_count
    }

    pub fn reset(&mut self) {
        self.detected_count = 0;
        self.volume_ema = 0.0;
        self.volume_threshold = f64::MAX;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Flow;

    fn make_ddos_window() -> FlowWindow {
        let mut window = FlowWindow::new(0.0, 10.0);
        // 20 sources différentes, chacune avec 10 connexions SYN
        for src in 0..20
        {
            for _ in 0..10
            {
                let mut f = Flow::new(
                    &format!("10.0.{}.1", src),
                    "10.0.0.2",
                    40000 + src as u16,
                    80,
                );
                f.syn_count = 5;
                f.packets_out = 6;
                f.packets_in = 1;
                f.bytes_out = 60;
                f.bytes_in = 60;
                f.start_time = 1.0;
                f.end_time = 2.0;
                window.push(f);
            }
        }
        window
    }

    #[test]
    fn test_detect_syn_flood() {
        let window = make_ddos_window();
        let mut det = DdosDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(!results.is_empty(), "should detect SYN flood");
        assert_eq!(results[0].label_en, "syn_flood");
    }

    #[test]
    fn test_no_ddos_normal() {
        let mut window = FlowWindow::new(0.0, 10.0);
        for i in 0..5
        {
            let mut f = Flow::new(&format!("10.0.0.{}", i), "10.0.0.2", 40000, 80);
            f.packets_out = 10;
            f.packets_in = 10;
            f.bytes_out = 1000;
            f.bytes_in = 5000;
            f.start_time = i as f64;
            f.end_time = i as f64 + 1.0;
            window.push(f);
        }
        let mut det = DdosDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(results.is_empty(), "normal traffic should not trigger");
    }
}
