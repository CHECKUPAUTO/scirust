use super::{ParsedPayload, PayloadParser};
use crate::flow::Protocol;

/// Parseur DNS (requete et reponse).
///
/// Extrait: type de requete, nom de domaine, nombre de records.
/// Detecte: noms de domaine anormalement longs, types de records suspects,
///          requetes NXDOMAIN massives.
pub struct DnsParser {
    /// Longueur maximale normale d'un nom de domaine
    pub max_domain_len: usize,
    /// Nombre max de labels (sous-domaines)
    pub max_labels: usize,
}

impl DnsParser {
    pub fn new() -> Self {
        Self {
            max_domain_len: 253,
            max_labels: 10,
        }
    }
}

impl Default for DnsParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Types DNS communs.
#[derive(Debug, Clone, Copy)]
pub enum DnsType {
    A,
    AAAA,
    MX,
    NS,
    CNAME,
    TXT,
    SOA,
    PTR,
    SRV,
    Unknown(u16),
}

impl DnsType {
    pub fn from_u16(val: u16) -> Self {
        match val
        {
            1 => DnsType::A,
            28 => DnsType::AAAA,
            15 => DnsType::MX,
            2 => DnsType::NS,
            5 => DnsType::CNAME,
            16 => DnsType::TXT,
            6 => DnsType::SOA,
            12 => DnsType::PTR,
            33 => DnsType::SRV,
            _ => DnsType::Unknown(val),
        }
    }

    pub fn label(&self) -> &'static str {
        match self
        {
            DnsType::A => "A",
            DnsType::AAAA => "AAAA",
            DnsType::MX => "MX",
            DnsType::NS => "NS",
            DnsType::CNAME => "CNAME",
            DnsType::TXT => "TXT",
            DnsType::SOA => "SOA",
            DnsType::PTR => "PTR",
            DnsType::SRV => "SRV",
            DnsType::Unknown(_) => "Unknown",
        }
    }
}

/// Resultat du parsing DNS.
#[derive(Debug, Clone)]
pub struct DnsParsed {
    /// ID de transaction
    pub transaction_id: u16,
    /// Flags
    pub flags: u16,
    /// Est-ce une reponse?
    pub is_response: bool,
    /// Code de reponse (0=NOERROR, 3=NXDOMAIN, etc.)
    pub response_code: u8,
    /// Nombre de requetes
    pub query_count: u16,
    /// Nombre de reponses
    pub answer_count: u16,
    /// Noms de domaines requis
    pub queries: Vec<String>,
    /// Types DNS demandes
    pub query_types: Vec<DnsType>,
}

impl PayloadParser for DnsParser {
    fn protocol(&self) -> Protocol {
        Protocol::Dns
    }

    fn parse(&self, payload: &[u8], _src_port: u16, dst_port: u16) -> Option<ParsedPayload> {
        // Le port 53 est obligatoire pour DNS
        if dst_port != 53 && _src_port != 53
        {
            return None;
        }

        // En-tete DNS minimal: 12 octets
        if payload.len() < 12
        {
            return None;
        }

        let mut parsed = ParsedPayload::new(Protocol::Dns);

        let tx_id = u16::from_be_bytes([payload[0], payload[1]]);
        let flags = u16::from_be_bytes([payload[2], payload[3]]);
        let qd_count = u16::from_be_bytes([payload[4], payload[5]]);
        let an_count = u16::from_be_bytes([payload[6], payload[7]]);

        let is_response = (flags & 0x8000) != 0;
        let rcode = (flags & 0x000F) as u8;

        parsed.command = if is_response
        {
            "DNS_RESPONSE"
        }
        else
        {
            "DNS_QUERY"
        }
        .to_string();

        // Extraire les noms de domaines des queries
        let mut offset = 12;
        let mut domains = Vec::new();
        let mut query_types = Vec::new();

        for _ in 0..qd_count
        {
            if offset >= payload.len()
            {
                break;
            }

            // Parser le nom de domaine (labels)
            let mut domain_parts = Vec::new();
            let mut label_len = payload[offset] as usize;
            let _name_start = offset;

            while label_len > 0 && offset < payload.len()
            {
                offset += 1;
                if offset + label_len > payload.len()
                {
                    break;
                }
                let label = &payload[offset..offset + label_len];
                if let Ok(s) = std::str::from_utf8(label)
                {
                    domain_parts.push(s.to_string());
                }
                offset += label_len;
                if offset >= payload.len()
                {
                    break;
                }
                label_len = payload[offset] as usize;
            }

            let domain = domain_parts.join(".");
            if !domain.is_empty()
            {
                domains.push(domain.clone());
                // Anomalie: domaine trop long
                if domain.len() > self.max_domain_len
                {
                    parsed
                        .anomalies
                        .push(format!("long_domain:{}", domain.len()));
                }
                // Anomalie: trop de labels
                if domain_parts.len() > self.max_labels
                {
                    parsed
                        .anomalies
                        .push(format!("deep_subdomain:{}", domain_parts.len()));
                }
                // Anomalie: caracteres hex en base32hex (tunneling)
                if domain_parts.iter().any(|label| {
                    label.len() > 20
                        && label
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                })
                {
                    parsed.anomalies.push("suspicious_label_length".to_string());
                }
            }

            // Type DNS
            offset += 1; // skip null byte
            if offset + 4 <= payload.len()
            {
                let qtype = u16::from_be_bytes([payload[offset], payload[offset + 1]]);
                query_types.push(DnsType::from_u16(qtype));
                offset += 4; // type + class
            }
        }

        parsed.payload_size = payload.len();
        parsed
            .headers
            .push(("tx_id".to_string(), tx_id.to_string()));
        parsed
            .headers
            .push(("flags".to_string(), format!("0x{:04x}", flags)));
        parsed
            .headers
            .push(("qd_count".to_string(), qd_count.to_string()));
        parsed
            .headers
            .push(("an_count".to_string(), an_count.to_string()));

        // NXDOMAIN massif = tentative de DNS Rebinding ou tunneling
        if is_response && rcode == 3
        {
            parsed.anomalies.push("nxdomain".to_string());
        }

        // TXT queries = potentiel tunneling
        if query_types.iter().any(|t| matches!(t, DnsType::TXT))
        {
            parsed.anomalies.push("txt_query".to_string());
        }

        let query_str = domains.join(", ");
        parsed.uri = query_str;
        parsed
            .headers
            .push(("domains".to_string(), domains.join("; ")));
        parsed.headers.push((
            "types".to_string(),
            query_types
                .iter()
                .map(|t| t.label())
                .collect::<Vec<_>>()
                .join(","),
        ));

        Some(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dns_query(domain: &str) -> Vec<u8> {
        let mut payload = Vec::new();
        // Header: ID=0x1234, flags=0x0100 (standard query), QD=1, AN=0
        payload.extend_from_slice(&[0x12, 0x34]); // transaction id
        payload.extend_from_slice(&[0x01, 0x00]); // flags
        payload.extend_from_slice(&[0x00, 0x01]); // questions
        payload.extend_from_slice(&[0x00, 0x00]); // answers
        payload.extend_from_slice(&[0x00, 0x00]); // authority
        payload.extend_from_slice(&[0x00, 0x00]); // additional

        // Domain name labels
        for label in domain.split('.')
        {
            payload.push(label.len() as u8);
            payload.extend_from_slice(label.as_bytes());
        }
        payload.push(0); // null terminator

        // Type A (1), Class IN (1)
        payload.extend_from_slice(&[0x00, 0x01]); // type
        payload.extend_from_slice(&[0x00, 0x01]); // class

        payload
    }

    #[test]
    fn test_parse_dns_query() {
        let payload = make_dns_query("example.com");
        let parser = DnsParser::new();
        let result = parser.parse(payload.as_slice(), 40000, 53).unwrap();
        assert_eq!(result.command, "DNS_QUERY");
        assert!(result.uri.contains("example.com"));
    }

    #[test]
    fn test_parse_dns_not_port_53() {
        let payload = make_dns_query("example.com");
        let parser = DnsParser::new();
        assert!(parser.parse(payload.as_slice(), 40000, 80).is_none());
    }

    #[test]
    fn test_parse_dns_too_short() {
        let parser = DnsParser::new();
        assert!(parser.parse(&[0u8; 5], 40000, 53).is_none());
    }
}
