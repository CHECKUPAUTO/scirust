use crate::flow::{Flow, FlowDirection, FlowWindow, Protocol};
use serde::{Deserialize, Serialize};

/// Configuration de la capture réseau.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    /// Interface réseau (ex: "eth0", "any")
    pub interface: String,
    /// Filtre BPF (Berkeley Packet Filter)
    pub bpf_filter: String,
    /// Nombre maximal de paquets par fenêtre
    pub max_packets_per_window: usize,
    /// Durée de la fenêtre de capture en secondes
    pub window_duration_secs: f64,
    /// Mode promiscuité
    pub promiscuous: bool,
    /// Timeout de lecture en millisecondes
    pub read_timeout_ms: u32,
    /// Taille maximale du snaplen (octets par paquet)
    pub snaplen: u32,
    /// Réseau local (plage d'adresses privées)
    pub local_networks: Vec<String>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            interface: "any".to_string(),
            bpf_filter: "tcp or udp or icmp".to_string(),
            max_packets_per_window: 10000,
            window_duration_secs: 60.0,
            promiscuous: true,
            read_timeout_ms: 100,
            snaplen: 65535,
            local_networks: vec![
                "10.0.0.0/8".to_string(),
                "172.16.0.0/12".to_string(),
                "192.168.0.0/16".to_string(),
            ],
        }
    }
}

/// Paquet brut capturé.
#[derive(Debug, Clone)]
pub struct RawPacket {
    /// Horodatage epoch en secondes (avec fractional)
    pub timestamp: f64,
    /// Longueur du paquet sur le fil (wire length)
    pub wire_len: usize,
    /// Longueur capturée (<= wire_len si snaplen)
    pub captured_len: usize,
    /// En-tête Ethernet (14 octets)
    pub eth_header: Vec<u8>,
    /// En-tête IP (20-60 octets)
    pub ip_header: Vec<u8>,
    /// En-tête TCP/UDP/ICMP
    pub transport_header: Vec<u8>,
    /// Payload applicatif
    pub payload: Vec<u8>,
    /// Flags TCP (SYN, ACK, RST, FIN, etc.)
    pub tcp_flags: u8,
    /// Numéro de séquence TCP
    pub tcp_seq: u32,
    /// Numéro d'acquittement TCP
    pub tcp_ack: u32,
    /// Fenêtre TCP
    pub tcp_window: u16,
}

impl RawPacket {
    /// Extraire l'adresse IP source depuis l'en-tête IP.
    pub fn src_ip(&self) -> String {
        if self.ip_header.len() >= 20
        {
            format!(
                "{}.{}.{}.{}",
                self.ip_header[12], self.ip_header[13], self.ip_header[14], self.ip_header[15]
            )
        }
        else
        {
            "0.0.0.0".to_string()
        }
    }

    /// Extraire l'adresse IP destination depuis l'en-tête IP.
    pub fn dst_ip(&self) -> String {
        if self.ip_header.len() >= 20
        {
            format!(
                "{}.{}.{}.{}",
                self.ip_header[16], self.ip_header[17], self.ip_header[18], self.ip_header[19]
            )
        }
        else
        {
            "0.0.0.0".to_string()
        }
    }

    /// Protocole IP (6=TCP, 17=UDP, 1=ICMP).
    pub fn ip_protocol(&self) -> u8 {
        if !self.ip_header.is_empty()
        {
            self.ip_header[9]
        }
        else
        {
            0
        }
    }

    /// Extraire le port source depuis l'en-tête transport.
    pub fn src_port(&self) -> u16 {
        if self.transport_header.len() >= 2
        {
            u16::from_be_bytes([self.transport_header[0], self.transport_header[1]])
        }
        else
        {
            0
        }
    }

    /// Extraire le port destination depuis l'en-tête transport.
    pub fn dst_port(&self) -> u16 {
        if self.transport_header.len() >= 4
        {
            u16::from_be_bytes([self.transport_header[2], self.transport_header[3]])
        }
        else
        {
            0
        }
    }

    /// Est-ce un paquet SYN (sans ACK)?
    pub fn is_syn(&self) -> bool {
        (self.tcp_flags & 0x02) != 0 && (self.tcp_flags & 0x10) == 0
    }

    /// Est-ce un paquet RST?
    pub fn is_rst(&self) -> bool {
        (self.tcp_flags & 0x04) != 0
    }

    /// Est-ce un paquet ACK?
    pub fn is_ack(&self) -> bool {
        (self.tcp_flags & 0x10) != 0
    }

    /// Est-ce un paquet FIN?
    pub fn is_fin(&self) -> bool {
        (self.tcp_flags & 0x01) != 0
    }
}

/// Convertir un RawPacket en Flow (agrégation nécessaire).
pub fn raw_packet_to_flow(packet: &RawPacket) -> Flow {
    let protocol = match packet.ip_protocol()
    {
        6 => Protocol::from_port(packet.dst_port()),
        17 =>
        {
            if packet.dst_port() == 53
            {
                Protocol::Dns
            }
            else
            {
                Protocol::Udp
            }
        },
        1 => Protocol::Icmp,
        _ => Protocol::Unknown(packet.ip_protocol() as u16),
    };

    let direction = FlowDirection::Lateral;

    let mut flow = Flow::new(
        &packet.src_ip(),
        &packet.dst_ip(),
        packet.src_port(),
        packet.dst_port(),
    );
    flow.protocol = protocol;
    flow.bytes_out = packet.captured_len as u64;
    flow.start_time = packet.timestamp;
    flow.end_time = packet.timestamp;
    flow.packets_out = 1;
    flow.direction = direction;
    flow.syn_count = if packet.is_syn() { 1 } else { 0 };
    flow.rst_count = if packet.is_rst() { 1 } else { 0 };
    flow.avg_payload_size = packet.payload.len() as f64;
    flow
}

/// Agréger une liste de paquets bruts en un FlowWindow.
pub fn packets_to_window(packets: &[RawPacket], window_start: f64, window_end: f64) -> FlowWindow {
    let mut window = FlowWindow::new(window_start, window_end);
    for pkt in packets
    {
        if pkt.timestamp >= window_start && pkt.timestamp < window_end
        {
            window.push(raw_packet_to_flow(pkt));
        }
    }
    window
}

/// Trait d'abstraction pour la capture réseau.
///
/// Permet de remplacer la capture réelle (pcap) par un backend simulé
/// pour les tests et le développement.
pub trait NetworkCapture {
    /// Démarrer la capture sur l'interface spécifiée.
    fn start(&mut self, config: &CaptureConfig) -> Result<(), String>;

    /// Arrêter la capture.
    fn stop(&mut self) -> Result<(), String>;

    /// Lire le prochain paquet (timeout géré par le backend).
    fn next_packet(&mut self) -> Result<Option<RawPacket>, String>;

    /// Lire tous les paquets disponibles (jusqu'au timeout).
    fn read_all(&mut self) -> Result<Vec<RawPacket>, String> {
        let mut packets = Vec::new();
        loop
        {
            match self.next_packet()
            {
                Ok(Some(pkt)) => packets.push(pkt),
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(packets)
    }

    /// Lire les paquets pendant une durée donnée et retourner un FlowWindow.
    fn capture_window(
        &mut self,
        config: &CaptureConfig,
        window_start: f64,
    ) -> Result<FlowWindow, String> {
        let window_end = window_start + config.window_duration_secs;
        let packets = self.read_all()?;
        Ok(packets_to_window(&packets, window_start, window_end))
    }

    /// Vérifier si la capture est active.
    fn is_running(&self) -> bool;

    /// Nombre de paquets capturés depuis le début.
    fn packet_count(&self) -> u64;

    /// Nombre de paquets丢弃és (drop).
    fn drop_count(&self) -> u64;
}

// ---------------------------------------------------------------------------
// Simulated Capture Backend
// ---------------------------------------------------------------------------

/// Backend de capture simulé pour tests et développement.
pub struct SimulatedCapture {
    running: bool,
    packets: Vec<RawPacket>,
    cursor: usize,
    packet_count: u64,
    drop_count: u64,
}

impl SimulatedCapture {
    pub fn new() -> Self {
        Self {
            running: false,
            packets: Vec::new(),
            cursor: 0,
            packet_count: 0,
            drop_count: 0,
        }
    }

    /// Injecter des paquets simulés pour les tests.
    pub fn inject(&mut self, packets: Vec<RawPacket>) {
        self.packets.extend(packets);
    }

    /// Générer un paquet TCP simulé.
    pub fn fake_tcp_packet(
        src_ip: &str,
        dst_ip: &str,
        src_port: u16,
        dst_port: u16,
        flags: u8,
        payload_len: usize,
        timestamp: f64,
    ) -> RawPacket {
        let ip_src: Vec<u8> = src_ip.split('.').filter_map(|s| s.parse().ok()).collect();
        let ip_dst: Vec<u8> = dst_ip.split('.').filter_map(|s| s.parse().ok()).collect();

        let mut ip_header = vec![0x45, 0x00]; // IPv4, IHL=5, DSCP=0
        let total_len = 20 + 20 + payload_len;
        ip_header.extend_from_slice(&(total_len as u16).to_be_bytes());
        ip_header.extend_from_slice(&[0x00, 0x00]); // identification
        ip_header.extend_from_slice(&[0x40, 0x00]); // flags + fragment
        ip_header.push(0x40); // TTL=64
        ip_header.push(0x06); // protocol=TCP
        ip_header.extend_from_slice(&[0x00, 0x00]); // checksum placeholder
        if ip_src.len() == 4
        {
            ip_header.extend_from_slice(&ip_src);
        }
        else
        {
            ip_header.extend_from_slice(&[10, 0, 0, 1]);
        }
        if ip_dst.len() == 4
        {
            ip_header.extend_from_slice(&ip_dst);
        }
        else
        {
            ip_header.extend_from_slice(&[10, 0, 0, 2]);
        }

        let mut transport = Vec::new();
        transport.extend_from_slice(&src_port.to_be_bytes());
        transport.extend_from_slice(&dst_port.to_be_bytes());
        transport.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // seq
        transport.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // ack
        transport.push(0x50); // data offset
        transport.push(flags);
        transport.extend_from_slice(&[0x00, 0x00]); // window
        transport.extend_from_slice(&[0x00, 0x00]); // checksum
        transport.extend_from_slice(&[0x00, 0x00]); // urgent

        let eth_header = vec![
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00,
        ];

        RawPacket {
            timestamp,
            wire_len: 14 + total_len,
            captured_len: 14 + total_len,
            eth_header,
            ip_header,
            transport_header: transport,
            payload: vec![0u8; payload_len],
            tcp_flags: flags,
            tcp_seq: 0,
            tcp_ack: 0,
            tcp_window: 65535,
        }
    }
}

impl Default for SimulatedCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkCapture for SimulatedCapture {
    fn start(&mut self, _config: &CaptureConfig) -> Result<(), String> {
        self.running = true;
        self.cursor = 0;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), String> {
        self.running = false;
        Ok(())
    }

    fn next_packet(&mut self) -> Result<Option<RawPacket>, String> {
        if !self.running
        {
            return Err("Capture not started".to_string());
        }
        if self.cursor < self.packets.len()
        {
            let pkt = self.packets[self.cursor].clone();
            self.cursor += 1;
            self.packet_count += 1;
            Ok(Some(pkt))
        }
        else
        {
            Ok(None)
        }
    }

    fn is_running(&self) -> bool {
        self.running
    }

    fn packet_count(&self) -> u64 {
        self.packet_count
    }

    fn drop_count(&self) -> u64 {
        self.drop_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_packet_ip_extraction() {
        let pkt = SimulatedCapture::fake_tcp_packet(
            "192.168.1.100",
            "10.0.0.1",
            45000,
            80,
            0x02, // SYN
            0,
            1000.0,
        );
        assert_eq!(pkt.src_ip(), "192.168.1.100");
        assert_eq!(pkt.dst_ip(), "10.0.0.1");
        assert_eq!(pkt.src_port(), 45000);
        assert_eq!(pkt.dst_port(), 80);
        assert!(pkt.is_syn());
        assert!(!pkt.is_rst());
    }

    #[test]
    fn test_simulated_capture_lifecycle() {
        let mut cap = SimulatedCapture::new();
        let config = CaptureConfig::default();

        cap.start(&config).unwrap();
        assert!(cap.is_running());

        let pkt =
            SimulatedCapture::fake_tcp_packet("10.0.0.1", "10.0.0.2", 40000, 80, 0x02, 0, 1.0);
        cap.inject(vec![pkt]);

        let next = cap.next_packet().unwrap();
        assert!(next.is_some());
        assert_eq!(cap.packet_count(), 1);

        // No more packets
        let none = cap.next_packet().unwrap();
        assert!(none.is_none());

        cap.stop().unwrap();
        assert!(!cap.is_running());
    }

    #[test]
    fn test_raw_packet_to_flow() {
        let pkt =
            SimulatedCapture::fake_tcp_packet("10.0.0.1", "10.0.0.2", 40000, 80, 0x02, 100, 1000.0);
        let flow = raw_packet_to_flow(&pkt);
        assert_eq!(flow.src_ip, "10.0.0.1");
        assert_eq!(flow.dst_ip, "10.0.0.2");
        assert_eq!(flow.src_port, 40000);
        assert_eq!(flow.dst_port, 80);
        assert_eq!(flow.protocol, Protocol::Http);
        assert_eq!(flow.syn_count, 1);
        assert_eq!(flow.bytes_out, pkt.captured_len as u64);
    }

    #[test]
    fn test_packets_to_window() {
        let mut packets = Vec::new();
        for i in 0..5
        {
            packets.push(SimulatedCapture::fake_tcp_packet(
                "10.0.0.1",
                "10.0.0.2",
                40000 + i,
                80,
                0x10, // ACK
                0,
                100.0 + i as f64 * 10.0,
            ));
        }
        let window = packets_to_window(&packets, 100.0, 200.0);
        assert_eq!(window.len(), 5);
    }
}
