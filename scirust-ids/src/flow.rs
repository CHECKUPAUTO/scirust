use serde::{Deserialize, Serialize};

/// Protocole réseau identifié.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Dns,
    Http,
    Https,
    Ssh,
    Ftp,
    Smtp,
    Telnet,
    Unknown(u16),
}

impl Protocol {
    /// Numéro de port standard associé au protocole.
    pub fn default_port(&self) -> u16 {
        match self
        {
            Protocol::Tcp => 0,
            Protocol::Udp => 0,
            Protocol::Icmp => 1,
            Protocol::Dns => 53,
            Protocol::Http => 80,
            Protocol::Https => 443,
            Protocol::Ssh => 22,
            Protocol::Ftp => 21,
            Protocol::Smtp => 25,
            Protocol::Telnet => 23,
            Protocol::Unknown(p) => *p,
        }
    }

    /// Identification à partir du numéro de port.
    pub fn from_port(port: u16) -> Self {
        match port
        {
            53 => Protocol::Dns,
            80 => Protocol::Http,
            443 => Protocol::Https,
            22 => Protocol::Ssh,
            21 => Protocol::Ftp,
            25 => Protocol::Smtp,
            23 => Protocol::Telnet,
            _ => Protocol::Unknown(port),
        }
    }
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Protocol::Tcp => write!(f, "TCP"),
            Protocol::Udp => write!(f, "UDP"),
            Protocol::Icmp => write!(f, "ICMP"),
            Protocol::Dns => write!(f, "DNS"),
            Protocol::Http => write!(f, "HTTP"),
            Protocol::Https => write!(f, "HTTPS"),
            Protocol::Ssh => write!(f, "SSH"),
            Protocol::Ftp => write!(f, "FTP"),
            Protocol::Smtp => write!(f, "SMTP"),
            Protocol::Telnet => write!(f, "Telnet"),
            Protocol::Unknown(p) => write!(f, "Unknown({})", p),
        }
    }
}

/// Direction du flux réseau.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlowDirection {
    Inbound,
    Outbound,
    Lateral,
}

/// Un flux réseau (connexion entre deux endpoints).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flow {
    /// Adresse IP source
    pub src_ip: String,
    /// Adresse IP destination
    pub dst_ip: String,
    /// Port source
    pub src_port: u16,
    /// Port destination
    pub dst_port: u16,
    /// Protocole identifié
    pub protocol: Protocol,
    /// Nombre d'octets émis (source -> destination)
    pub bytes_out: u64,
    /// Nombre d'octets reçus (destination -> source)
    pub bytes_in: u64,
    /// Nombre de paquets émis
    pub packets_out: u64,
    /// Nombre de paquets reçus
    pub packets_in: u64,
    /// Horodatage de début (secondes epoch)
    pub start_time: f64,
    /// Horodatage de fin
    pub end_time: f64,
    /// Nombre de tentatives TCP SYN (pour détection scan)
    pub syn_count: u32,
    /// Nombre de RST (reset)
    pub rst_count: u32,
    /// Nombre d'erreurs applicatives (HTTP 4xx/5xx, etc.)
    pub error_count: u32,
    /// Direction par rapport à l'hôte surveillé
    pub direction: FlowDirection,
    /// Payload size moyen (0 si non disponible)
    pub avg_payload_size: f64,
}

impl Flow {
    pub fn new(src_ip: &str, dst_ip: &str, src_port: u16, dst_port: u16) -> Self {
        Self {
            src_ip: src_ip.to_string(),
            dst_ip: dst_ip.to_string(),
            src_port,
            dst_port,
            protocol: Protocol::from_port(dst_port),
            bytes_out: 0,
            bytes_in: 0,
            packets_out: 0,
            packets_in: 0,
            start_time: 0.0,
            end_time: 0.0,
            syn_count: 0,
            rst_count: 0,
            error_count: 0,
            direction: FlowDirection::Lateral,
            avg_payload_size: 0.0,
        }
    }

    /// Durée du flux en secondes.
    pub fn duration(&self) -> f64 {
        (self.end_time - self.start_time).max(0.0)
    }

    /// Nombre total d'octets.
    pub fn total_bytes(&self) -> u64 {
        self.bytes_out + self.bytes_in
    }

    /// Nombre total de paquets.
    pub fn total_packets(&self) -> u64 {
        self.packets_out + self.packets_in
    }

    /// Débit en octets/seconde.
    pub fn throughput_bps(&self) -> f64 {
        let d = self.duration();
        if d < f64::EPSILON
        {
            0.0
        }
        else
        {
            self.total_bytes() as f64 / d
        }
    }

    /// Ratio bytes_in / bytes_out (asymétrie).
    pub fn asymmetry(&self) -> f64 {
        if self.bytes_out == 0
        {
            if self.bytes_in == 0
            {
                0.0
            }
            else
            {
                f64::INFINITY
            }
        }
        else
        {
            self.bytes_in as f64 / self.bytes_out as f64
        }
    }
}

/// Fenêtre de flux pour analyse temporelle.
#[derive(Debug, Clone)]
pub struct FlowWindow {
    /// Flux dans cette fenêtre
    pub flows: Vec<Flow>,
    /// Début de la fenêtre (epoch seconds)
    pub start_time: f64,
    /// Fin de la fenêtre
    pub end_time: f64,
}

impl FlowWindow {
    pub fn new(start_time: f64, end_time: f64) -> Self {
        Self {
            flows: Vec::new(),
            start_time,
            end_time,
        }
    }

    pub fn push(&mut self, flow: Flow) {
        self.flows.push(flow);
    }

    /// Nombre total de flux.
    pub fn len(&self) -> usize {
        self.flows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.flows.is_empty()
    }

    /// Nombre de sources uniques.
    pub fn unique_sources(&self) -> usize {
        use std::collections::HashSet;
        self.flows
            .iter()
            .map(|f| f.src_ip.as_str())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Nombre de destinations uniques.
    pub fn unique_destinations(&self) -> usize {
        use std::collections::HashSet;
        self.flows
            .iter()
            .map(|f| f.dst_ip.as_str())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Nombre de ports uniques ciblés.
    pub fn unique_dst_ports(&self) -> usize {
        use std::collections::HashSet;
        self.flows
            .iter()
            .map(|f| f.dst_port)
            .collect::<HashSet<_>>()
            .len()
    }

    /// Nombre de ports uniques sources.
    pub fn unique_src_ports(&self) -> usize {
        use std::collections::HashSet;
        self.flows
            .iter()
            .map(|f| f.src_port)
            .collect::<HashSet<_>>()
            .len()
    }

    /// Taux de SYN (SYN flood indicator).
    pub fn syn_rate(&self) -> f64 {
        let total_packets: u64 = self.flows.iter().map(|f| f.total_packets()).sum();
        let total_syn: u64 = self.flows.iter().map(|f| f.syn_count as u64).sum();
        if total_packets == 0
        {
            0.0
        }
        else
        {
            total_syn as f64 / total_packets as f64
        }
    }

    /// Taux de RST.
    pub fn rst_rate(&self) -> f64 {
        let total_packets: u64 = self.flows.iter().map(|f| f.total_packets()).sum();
        let total_rst: u64 = self.flows.iter().map(|f| f.rst_count as u64).sum();
        if total_packets == 0
        {
            0.0
        }
        else
        {
            total_rst as f64 / total_packets as f64
        }
    }

    /// Nombre total d'erreurs.
    pub fn total_errors(&self) -> u32 {
        self.flows.iter().map(|f| f.error_count).sum()
    }

    /// Taux d'erreurs.
    pub fn error_rate(&self) -> f64 {
        let total = self.flows.len() as f64;
        if total < f64::EPSILON
        {
            0.0
        }
        else
        {
            self.total_errors() as f64 / total
        }
    }

    /// Extraire le nombre de connexions par source IP.
    pub fn connections_per_source(&self) -> std::collections::HashMap<String, usize> {
        let mut map = std::collections::HashMap::new();
        for f in &self.flows
        {
            *map.entry(f.src_ip.clone()).or_insert(0) += 1;
        }
        map
    }

    /// Extraire le nombre de destinations par source IP.
    pub fn destinations_per_source(
        &self,
    ) -> std::collections::HashMap<String, std::collections::HashSet<String>> {
        let mut map: std::collections::HashMap<String, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        for f in &self.flows
        {
            map.entry(f.src_ip.clone())
                .or_default()
                .insert(f.dst_ip.clone());
        }
        map
    }

    /// Extraire les intervalles inter-connexions pour une paire src/dst.
    pub fn inter_arrival_times(&self, src_ip: &str, dst_ip: &str) -> Vec<f64> {
        let mut times: Vec<f64> = self
            .flows
            .iter()
            .filter(|f| f.src_ip == src_ip && f.dst_ip == dst_ip)
            .map(|f| f.start_time)
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        times.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_from_port() {
        assert_eq!(Protocol::from_port(80), Protocol::Http);
        assert_eq!(Protocol::from_port(443), Protocol::Https);
        assert_eq!(Protocol::from_port(22), Protocol::Ssh);
        assert_eq!(Protocol::from_port(53), Protocol::Dns);
        assert!(matches!(Protocol::from_port(9999), Protocol::Unknown(9999)));
    }

    #[test]
    fn test_flow_metrics() {
        let mut f = Flow::new("10.0.0.1", "10.0.0.2", 45000, 80);
        f.bytes_out = 1000;
        f.bytes_in = 5000;
        f.packets_out = 10;
        f.packets_in = 5;
        f.start_time = 100.0;
        f.end_time = 110.0;

        assert_eq!(f.total_bytes(), 6000);
        assert_eq!(f.total_packets(), 15);
        assert!((f.duration() - 10.0).abs() < f64::EPSILON);
        assert!((f.throughput_bps() - 600.0).abs() < f64::EPSILON);
        assert!((f.asymmetry() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_flow_window() {
        let mut w = FlowWindow::new(0.0, 60.0);
        let mut f1 = Flow::new("10.0.0.1", "10.0.0.2", 40000, 80);
        f1.start_time = 1.0;
        f1.end_time = 2.0;
        w.push(f1);

        let mut f2 = Flow::new("10.0.0.3", "10.0.0.2", 40001, 80);
        f2.start_time = 3.0;
        f2.end_time = 4.0;
        w.push(f2);

        assert_eq!(w.len(), 2);
        assert_eq!(w.unique_sources(), 2);
        assert_eq!(w.unique_destinations(), 1);
        assert_eq!(w.unique_dst_ports(), 1);
    }

    #[test]
    fn test_empty_window() {
        let w = FlowWindow::new(0.0, 60.0);
        assert!(w.is_empty());
        assert_eq!(w.unique_sources(), 0);
        assert!((w.syn_rate() - 0.0).abs() < f64::EPSILON);
    }
}
