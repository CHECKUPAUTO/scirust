use crate::detectors::DetectorResult;
use crate::flow::FlowWindow;
use serde::{Deserialize, Serialize};

/// Configuration du détecteur de DNS tunneling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsTunnelConfig {
    /// Taille maximale d'un nom de domaine DNS normal (octets)
    pub max_normal_domain_len: usize,
    /// Nombre minimal de requêtes DNS par fenêtre
    pub min_dns_queries: usize,
    /// Seuil de ratio longueur moyenne des requêtes / longueur normale
    pub length_ratio_threshold: f64,
    /// Seuil d'entropie des noms de domaine (1.0 = aléatoire pur)
    pub entropy_threshold: f64,
    /// Nombre minimal de sous-domaines uniques
    pub min_unique_subdomains: usize,
    /// Seuil minimal de confiance
    pub min_confidence: f32,
}

impl Default for DnsTunnelConfig {
    fn default() -> Self {
        Self {
            max_normal_domain_len: 30,
            min_dns_queries: 10,
            length_ratio_threshold: 2.0,
            entropy_threshold: 3.5,
            min_unique_subdomains: 5,
            min_confidence: 0.7,
        }
    }
}

/// Détecteur de DNS tunneling.
///
/// Le DNS tunneling exfiltre des données en les cachant dans des
/// requêtes DNS (noms de domaine très longs et aléatoires).
/// Ce détecteur analyse les caractéristiques statistiques des
/// flux DNS dans une fenêtre temporelle.
#[derive(Debug, Clone)]
pub struct DnsTunnelDetector {
    pub config: DnsTunnelConfig,
    detected_count: u64,
}

impl DnsTunnelDetector {
    pub fn new(config: DnsTunnelConfig) -> Self {
        Self {
            config,
            detected_count: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DnsTunnelConfig::default())
    }

    /// Analyser les flux DNS dans une fenêtre.
    ///
    /// Comme nous n'avons pas accès au payload DNS dans le modèle Flow,
    /// nous utilisons des proxies: taille moyenne des paquets, nombre
    /// de sous-domaines, asymétrie du trafic, et patterns temporels.
    pub fn analyze(&mut self, window: &FlowWindow) -> Vec<DetectorResult> {
        let mut results = Vec::new();

        // Filtrer les flux DNS
        let dns_flows: Vec<&crate::flow::Flow> = window
            .flows
            .iter()
            .filter(|f| f.protocol == crate::flow::Protocol::Dns)
            .collect();

        if dns_flows.len() < self.config.min_dns_queries
        {
            return results;
        }

        // Grouper par destination (serveur DNS)
        let mut by_server: std::collections::HashMap<String, Vec<&crate::flow::Flow>> =
            std::collections::HashMap::new();
        for &f in &dns_flows
        {
            by_server.entry(f.dst_ip.clone()).or_default().push(f);
        }

        for (server_ip, flows) in &by_server
        {
            let total_queries = flows.len();
            if total_queries < self.config.min_dns_queries
            {
                continue;
            }

            // Proxy 1: taille moyenne des paquets sortants
            let avg_out_size: f64 =
                flows.iter().map(|f| f.bytes_out as f64).sum::<f64>() / total_queries as f64;
            let avg_in_size: f64 =
                flows.iter().map(|f| f.bytes_in as f64).sum::<f64>() / total_queries as f64;

            // Proxy 2: nombre de sources différentes (DNS tunneling = une source)
            let unique_sources = flows
                .iter()
                .map(|f| f.src_ip.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len();

            // Proxy 3: asymétrie (tunnel = gros queries, petites réponses ou inversement)
            let asymmetry = if avg_out_size > f64::EPSILON
            {
                avg_in_size / avg_out_size
            }
            else
            {
                0.0
            };

            // Proxy 4: nombre de ports sources uniques (indique sous-domaines différents)
            let unique_src_ports = flows
                .iter()
                .map(|f| f.src_port)
                .collect::<std::collections::HashSet<_>>()
                .len();

            // Proxy 5: volume total anormal pour du DNS
            let total_volume: u64 = flows.iter().map(|f| f.total_bytes()).sum();
            let avg_volume_per_query = total_volume as f64 / total_queries as f64;

            // Score composite
            let mut indicators = 0u32;
            let mut score = 0.0f32;

            // Gros paquets DNS (normal < 512 octets, tunnel > 512)
            if avg_out_size > 512.0
            {
                indicators += 1;
                score += 0.25;
            }

            // Asymétrie anormale (tunnel: réponses beaucoup plus grandes)
            if asymmetry > 5.0 || asymmetry < 0.2
            {
                indicators += 1;
                score += 0.2;
            }

            // Une seule source (typique du tunneling)
            if unique_sources <= 2 && total_queries > 20
            {
                indicators += 1;
                score += 0.15;
            }

            // Volume anormal
            if avg_volume_per_query > 1000.0
            {
                indicators += 1;
                score += 0.2;
            }

            // Beaucoup de ports sources (proxy pour sous-domaines)
            if unique_src_ports > self.config.min_unique_subdomains
            {
                indicators += 1;
                score += 0.2;
            }

            let confidence = score.min(1.0);
            if confidence >= self.config.min_confidence && indicators >= 2
            {
                self.detected_count += 1;
                let top_source = flows
                    .iter()
                    .max_by_key(|f| f.total_bytes())
                    .map(|f| f.src_ip.clone())
                    .unwrap_or_default();

                results.push(DetectorResult {
                    detector: "dns_tunnel".to_string(),
                    label_en: "dns_tunneling".to_string(),
                    label_fr: "tunneling_dns".to_string(),
                    confidence,
                    severity: if confidence >= 0.9 {
                        "CRITICAL".to_string()
                    } else if confidence >= 0.8 {
                        "WARNING".to_string()
                    } else {
                        "INFO".to_string()
                    },
                    source_ip: top_source,
                    destination_ip: server_ip.clone(),
                    details: format!(
                        "queries={} avg_out_bytes={:.0} avg_in_bytes={:.0} asymmetry={:.1} unique_src_ports={}",
                        total_queries, avg_out_size, avg_in_size, asymmetry, unique_src_ports
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

    fn make_dns_tunnel_window() -> FlowWindow {
        let mut window = FlowWindow::new(0.0, 60.0);
        // Simuler 30 requêtes DNS avec gros payload sortant (tunnel)
        for i in 0..30
        {
            let mut f = Flow::new("10.0.0.1", "8.8.8.8", 50000 + i as u16, 53);
            f.protocol = crate::flow::Protocol::Dns;
            f.bytes_out = 800; // gros payload sortant (données exfiltrées)
            f.bytes_in = 100; // petite réponse
            f.packets_out = 1;
            f.packets_in = 1;
            f.start_time = i as f64 * 2.0;
            f.end_time = f.start_time + 0.1;
            window.push(f);
        }
        window
    }

    #[test]
    fn test_detect_dns_tunnel() {
        let window = make_dns_tunnel_window();
        let mut det = DnsTunnelDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(!results.is_empty(), "should detect DNS tunneling");
        assert_eq!(results[0].label_en, "dns_tunneling");
    }

    #[test]
    fn test_no_tunnel_normal_dns() {
        let mut window = FlowWindow::new(0.0, 60.0);
        for i in 0..20
        {
            let mut f = Flow::new("10.0.0.1", "8.8.8.8", 40000, 53);
            f.protocol = crate::flow::Protocol::Dns;
            f.bytes_out = 60; // taille normale
            f.bytes_in = 200; // réponse normale
            f.packets_out = 1;
            f.packets_in = 1;
            f.start_time = i as f64 * 3.0;
            f.end_time = f.start_time + 0.05;
            window.push(f);
        }
        let mut det = DnsTunnelDetector::with_defaults();
        let results = det.analyze(&window);
        assert!(results.is_empty(), "normal DNS should not trigger");
    }
}
