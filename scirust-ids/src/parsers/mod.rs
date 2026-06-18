pub mod dns;
pub mod http;
pub mod ssh;

pub use dns::DnsParser;
pub use http::HttpParser;
pub use ssh::SshParser;

use crate::flow::Protocol;

/// Résultat du parsing d'un paquet applicatif.
#[derive(Debug, Clone)]
pub struct ParsedPayload {
    /// Protocole identifié
    pub protocol: Protocol,
    /// Méthode/Commande (GET, POST, SSH-2.0, etc.)
    pub command: String,
    /// URI ou chemin (pour HTTP)
    pub uri: String,
    /// En-têtes extraits (clé -> valeur)
    pub headers: Vec<(String, String)>,
    /// Code de réponse (pour HTTP: 200, 404, etc.)
    pub response_code: Option<u16>,
    /// Taille du payload original
    pub payload_size: usize,
    /// Indicateurs d'anomalie
    pub anomalies: Vec<String>,
}

impl ParsedPayload {
    pub fn new(protocol: Protocol) -> Self {
        Self {
            protocol,
            command: String::new(),
            uri: String::new(),
            headers: Vec::new(),
            response_code: None,
            payload_size: 0,
            anomalies: Vec::new(),
        }
    }

    pub fn has_anomalies(&self) -> bool {
        !self.anomalies.is_empty()
    }
}

/// Trait pour les parseurs de protocoles.
pub trait PayloadParser {
    /// Tenter de parser le payload. Retourne Some si le protocole est reconnu.
    fn parse(&self, payload: &[u8], src_port: u16, dst_port: u16) -> Option<ParsedPayload>;

    /// Protocole supporté par ce parseur.
    fn protocol(&self) -> Protocol;
}

/// Parser composite qui essaie tous les parseurs en séquence.
pub struct MultiParser {
    parsers: Vec<Box<dyn PayloadParser>>,
}

impl MultiParser {
    pub fn new() -> Self {
        Self {
            parsers: Vec::new(),
        }
    }

    pub fn add_parser(&mut self, parser: Box<dyn PayloadParser>) {
        self.parsers.push(parser);
    }

    pub fn parse(&self, payload: &[u8], src_port: u16, dst_port: u16) -> Option<ParsedPayload> {
        for parser in &self.parsers
        {
            if let Some(result) = parser.parse(payload, src_port, dst_port)
            {
                return Some(result);
            }
        }
        None
    }
}

impl Default for MultiParser {
    fn default() -> Self {
        let mut mp = Self::new();
        mp.add_parser(Box::new(HttpParser::new()));
        mp.add_parser(Box::new(DnsParser::new()));
        mp.add_parser(Box::new(SshParser::new()));
        mp
    }
}
