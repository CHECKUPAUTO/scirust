use crate::flow::{Flow, Protocol};

/// Extraction de caractéristiques réseau à partir d'un flux.
///
/// Retourne un vecteur de features numériques exploitables par les détecteurs.
/// Ordre: [bytes_out, bytes_in, packets_out, packets_in, duration,
///         syn_count, rst_count, error_count, payload_size, asymmetry]
pub fn flow_features(flow: &Flow) -> Vec<f64> {
    vec![
        flow.bytes_out as f64,
        flow.bytes_in as f64,
        flow.packets_out as f64,
        flow.packets_in as f64,
        flow.duration(),
        flow.syn_count as f64,
        flow.rst_count as f64,
        flow.error_count as f64,
        flow.avg_payload_size,
        flow.asymmetry().min(100.0),
    ]
}

/// Extraction de features à partir d'une fenêtre de flux.
///
/// Caractéristiques agrégées: nombre de flux, sources/destinations uniques,
/// ports ciblés, taux SYN/RST, taux d'erreurs, volume total.
pub fn window_features(window: &crate::flow::FlowWindow) -> Vec<f64> {
    let total_bytes: u64 = window.flows.iter().map(|f| f.total_bytes()).sum();
    let total_packets: u64 = window.flows.iter().map(|f| f.total_packets()).sum();
    let avg_duration: f64 = if window.flows.is_empty()
    {
        0.0
    }
    else
    {
        window.flows.iter().map(|f| f.duration()).sum::<f64>() / window.flows.len() as f64
    };

    vec![
        window.len() as f64,
        window.unique_sources() as f64,
        window.unique_destinations() as f64,
        window.unique_dst_ports() as f64,
        window.syn_rate(),
        window.rst_rate(),
        window.error_rate(),
        total_bytes as f64,
        total_packets as f64,
        avg_duration,
    ]
}

/// Identification de protocole au-delà du port (inspection basique de payload).
pub fn identify_protocol_from_payload(payload: &[u8], dst_port: u16) -> Protocol {
    if payload.len() < 3
    {
        return Protocol::from_port(dst_port);
    }
    // HTTP methods
    let is_http = payload.starts_with(b"GET ")
        || payload.starts_with(b"POST ")
        || payload.starts_with(b"PUT ")
        || payload.starts_with(b"DELETE ")
        || payload.starts_with(b"HEAD ")
        || payload.starts_with(b"OPTIONS ")
        || payload.starts_with(b"CONNECT ");
    if is_http
    {
        return Protocol::Http;
    }
    // DNS detection (port 53 + standard DNS header flags)
    if dst_port == 53 && payload.len() >= 12
    {
        let flags = payload[2];
        // QR bit set = response, Opcode in upper nibble
        if (flags & 0x80) != 0 || (flags & 0x78) == 0
        {
            return Protocol::Dns;
        }
    }
    // SSH banner
    if payload.starts_with(b"SSH-")
    {
        return Protocol::Ssh;
    }
    // SMTP banner
    if payload.starts_with(b"220 ")
    {
        return Protocol::Smtp;
    }
    // FTP banner
    if payload.starts_with(b"220 ") && dst_port == 21
    {
        return Protocol::Ftp;
    }

    Protocol::from_port(dst_port)
}

/// Calculer les intervalles d'arrivée inter-paquets pour une série de flux
/// triés par timestamp.
pub fn inter_arrival_times(flows: &[Flow]) -> Vec<f64> {
    let mut times: Vec<f64> = flows.iter().map(|f| f.start_time).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Calculer l'entropie de Shannon sur une distribution de valeurs discrètes.
pub fn shannon_entropy(values: &[u64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    let total: u64 = values.iter().sum();
    if total == 0
    {
        return 0.0;
    }
    let n = total as f64;
    values
        .iter()
        .filter(|&&v| v > 0)
        .map(|&v| {
            let p = v as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Comptage par catégorie: regroupe les flux par clé (IP, port, etc.)
/// et retourne les comptes.
pub fn count_by_key(flows: &[Flow], key: &str) -> std::collections::HashMap<String, usize> {
    let mut map = std::collections::HashMap::new();
    for f in flows
    {
        let k = match key
        {
            "src_ip" => f.src_ip.clone(),
            "dst_ip" => f.dst_ip.clone(),
            "src_port" => f.src_port.to_string(),
            "dst_port" => f.dst_port.to_string(),
            "protocol" => f.protocol.to_string(),
            "direction" => format!("{:?}", f.direction),
            _ => continue,
        };
        *map.entry(k).or_insert(0) += 1;
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_features_length() {
        let flow = Flow::new("10.0.0.1", "10.0.0.2", 40000, 80);
        let features = flow_features(&flow);
        assert_eq!(features.len(), 10);
    }

    #[test]
    fn test_shannon_entropy() {
        // Distribution uniforme → entropie maximale
        let uniform = vec![1, 1, 1, 1];
        let h = shannon_entropy(&uniform);
        assert!((h - 2.0).abs() < 1e-10);

        // Distribution concentrée → entropie faible
        let concentrated = vec![100, 0, 0, 0];
        let h2 = shannon_entropy(&concentrated);
        assert!((h2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_identify_protocol_ssh() {
        let payload = b"SSH-2.0-OpenSSH_8.9";
        assert_eq!(identify_protocol_from_payload(payload, 2222), Protocol::Ssh);
    }

    #[test]
    fn test_identify_protocol_http() {
        let payload = b"GET /index.html HTTP/1.1\r\n";
        assert_eq!(
            identify_protocol_from_payload(payload, 8080),
            Protocol::Http
        );
    }

    #[test]
    fn test_inter_arrival_times() {
        let flows = [Flow::new("a", "b", 1, 2), Flow::new("a", "b", 1, 2)];
        let mut f1 = flows[0].clone();
        f1.start_time = 1.0;
        let mut f2 = flows[1].clone();
        f2.start_time = 3.0;
        let iats = inter_arrival_times(&[f1, f2]);
        assert_eq!(iats.len(), 1);
        assert!((iats[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_count_by_key() {
        let flows = vec![
            Flow::new("10.0.0.1", "10.0.0.2", 1, 80),
            Flow::new("10.0.0.1", "10.0.0.3", 2, 80),
            Flow::new("10.0.0.4", "10.0.0.2", 3, 443),
        ];
        let counts = count_by_key(&flows, "src_ip");
        assert_eq!(counts.get("10.0.0.1"), Some(&2));
        assert_eq!(counts.get("10.0.0.4"), Some(&1));
    }
}
