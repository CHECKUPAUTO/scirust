//! Portée d'autorisation de découverte, sur le modèle **zones et conduits**
//! de l'ISA/IEC 62443 : aucune sonde n'est envoyée sur le réseau avant
//! qu'une [`ScopeAuthorization`] signée n'ait validé la cible contre une
//! liste blanche de plages CIDR, une liste blanche de protocoles, une
//! fenêtre de validité temporelle, et un niveau de sécurité de zone —
//! plutôt que de faire confiance à un simple booléen « scan autorisé ».
//!
//! Motivation (voir `README.md` pour les sources) : les automates
//! programmables industriels embarquent souvent des piles TCP/IP minimales
//! qui peuvent planter sous un scan générique — la doctrine NIST SP 800-82
//! est de préférer la découverte passive/native au protocole, et de
//! réserver tout sondage actif à une fenêtre de maintenance explicitement
//! autorisée par l'exploitant de l'installation. Cette structure encode
//! cette autorisation comme une donnée vérifiable plutôt qu'une convention.

use crate::hmac::hmac_sha256_hex;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr};

/// Plage CIDR IPv4 (IPv6 non supporté pour l'instant — voir `README.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cidr {
    network: u32,
    prefix_len: u8,
}

impl Cidr {
    pub fn parse(s: &str) -> Result<Self, String> {
        let (addr_part, prefix_part) = s
            .split_once('/')
            .ok_or_else(|| format!("CIDR '{s}' is missing a /prefix"))?;
        let addr: Ipv4Addr = addr_part
            .parse()
            .map_err(|_| format!("CIDR '{s}': '{addr_part}' is not a valid IPv4 address"))?;
        let prefix_len: u8 = prefix_part
            .parse()
            .map_err(|_| format!("CIDR '{s}': '{prefix_part}' is not a valid prefix length"))?;
        if prefix_len > 32
        {
            return Err(format!("CIDR '{s}': prefix length {prefix_len} > 32"));
        }
        Ok(Self {
            network: u32::from(addr),
            prefix_len,
        })
    }

    pub fn contains(&self, ip: Ipv4Addr) -> bool {
        if self.prefix_len == 0
        {
            return true;
        }
        let mask = u32::MAX << (32 - self.prefix_len);
        (u32::from(ip) & mask) == (self.network & mask)
    }
}

/// Portée d'autorisation signée pour une campagne de découverte.
///
/// `sign`/`verify` utilisent HMAC-SHA256 avec une clé pré-partagée entre
/// l'opérateur qui autorise et l'agent qui exécute la découverte — voir
/// `src/hmac.rs` pour les limites de ce modèle (pas de PKI, pas de
/// révocation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeAuthorization {
    pub operator: String,
    /// Étiquette de zone IEC 62443 (ex. "line3-plc-zone").
    pub zone: String,
    /// Niveau de sécurité cible de la zone, 0 (SL0) à 4 (SL4).
    pub zone_security_level: u8,
    pub allowed_cidrs: Vec<String>,
    pub allowed_protocols: Vec<String>,
    pub valid_from_unix: u64,
    pub valid_until_unix: u64,
    /// Si `false` (par défaut), toute zone SL3+ est refusée même si elle
    /// apparaît par ailleurs dans la portée — un dépassement explicite est
    /// requis pour sonder une zone de sécurité critique.
    #[serde(default)]
    pub allow_high_security_zone: bool,
    #[serde(default)]
    pub signature_hex: String,
}

impl ScopeAuthorization {
    fn canonical_payload(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}",
            self.operator,
            self.zone,
            self.zone_security_level,
            self.allowed_cidrs.join(","),
            self.allowed_protocols.join(","),
            self.valid_from_unix,
            self.valid_until_unix,
            self.allow_high_security_zone
        )
    }

    /// Signe cette portée avec la clé donnée, en remplaçant
    /// `signature_hex`.
    pub fn sign(mut self, key: &[u8]) -> Self {
        self.signature_hex = hmac_sha256_hex(key, self.canonical_payload().as_bytes());
        self
    }

    pub fn signature_valid(&self, key: &[u8]) -> bool {
        // Constant-time comparison: compute the expected HMAC and fold it
        // against the stored hex with XOR+OR so the result does not leak timing
        // information about how many leading characters matched (matches the
        // approach used by `scirust_trader::wallet::WalletAuthorization`).
        let expected = hmac_sha256_hex(key, self.canonical_payload().as_bytes());
        if self.signature_hex.is_empty()
        {
            return false;
        }
        // Length mismatch is itself a channel, but a valid HMAC hex is always
        // 64 chars; reject early only on an obviously wrong length (this does
        // not reveal anything about a correctly-formed signature).
        if expected.len() != self.signature_hex.len()
        {
            return false;
        }
        expected
            .bytes()
            .zip(self.signature_hex.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    }

    /// Vérifie que `ip`/`protocol` sont couverts par cette portée, à
    /// l'instant `now_unix` (injecté par l'appelant — voir la note de
    /// déterminisme du module racine). Renvoie `Err(raison)` sinon, jamais
    /// de panique sur une entrée malformée.
    pub fn authorize(
        &self,
        now_unix: u64,
        key: &[u8],
        ip: IpAddr,
        protocol: &str,
    ) -> Result<(), String> {
        if !self.signature_valid(key)
        {
            return Err("scope authorization signature is invalid or missing".to_string());
        }
        if now_unix < self.valid_from_unix || now_unix > self.valid_until_unix
        {
            return Err(format!(
                "scope authorization is not valid at t={now_unix} (window [{}, {}])",
                self.valid_from_unix, self.valid_until_unix
            ));
        }
        if self.zone_security_level >= 3 && !self.allow_high_security_zone
        {
            return Err(format!(
                "zone '{}' is SL{} (high security) and allow_high_security_zone is false",
                self.zone, self.zone_security_level
            ));
        }
        if !self.allowed_protocols.iter().any(|p| p == protocol)
        {
            return Err(format!(
                "protocol '{protocol}' is not in the allowed protocol list"
            ));
        }
        let ipv4 = match ip
        {
            IpAddr::V4(v4) => v4,
            IpAddr::V6(_) => return Err("only IPv4 targets are currently supported".to_string()),
        };
        let mut parsed_any_error = None;
        let in_scope = self.allowed_cidrs.iter().any(|c| match Cidr::parse(c)
        {
            Ok(cidr) => cidr.contains(ipv4),
            Err(e) =>
            {
                parsed_any_error.get_or_insert(e);
                false
            },
        });
        if let Some(e) = parsed_any_error
        {
            if !in_scope
            {
                return Err(format!("malformed CIDR in scope configuration: {e}"));
            }
        }
        if !in_scope
        {
            return Err(format!(
                "{ip} is not within any allowed CIDR range for zone '{}'",
                self.zone
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = b"test-preshared-key";

    fn valid_scope() -> ScopeAuthorization {
        ScopeAuthorization {
            operator: "alice@example.com".to_string(),
            zone: "line3-plc-zone".to_string(),
            zone_security_level: 1,
            allowed_cidrs: vec!["192.168.1.0/24".to_string()],
            allowed_protocols: vec!["opcua".to_string(), "modbus".to_string()],
            valid_from_unix: 1000,
            valid_until_unix: 2000,
            allow_high_security_zone: false,
            signature_hex: String::new(),
        }
        .sign(KEY)
    }

    #[test]
    fn cidr_contains_matches_expected_range() {
        let cidr = Cidr::parse("192.168.1.0/24").unwrap();
        assert!(cidr.contains("192.168.1.42".parse().unwrap()));
        assert!(!cidr.contains("192.168.2.1".parse().unwrap()));
    }

    #[test]
    fn cidr_zero_prefix_matches_everything() {
        let cidr = Cidr::parse("0.0.0.0/0").unwrap();
        assert!(cidr.contains("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn cidr_rejects_malformed_input() {
        assert!(Cidr::parse("not-an-ip/24").is_err());
        assert!(Cidr::parse("192.168.1.0/33").is_err());
        assert!(Cidr::parse("192.168.1.0").is_err());
    }

    #[test]
    fn authorize_accepts_in_scope_request() {
        let scope = valid_scope();
        let ip: IpAddr = "192.168.1.42".parse().unwrap();
        assert!(scope.authorize(1500, KEY, ip, "opcua").is_ok());
    }

    #[test]
    fn authorize_rejects_out_of_range_ip() {
        let scope = valid_scope();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(scope.authorize(1500, KEY, ip, "opcua").is_err());
    }

    #[test]
    fn authorize_rejects_disallowed_protocol() {
        let scope = valid_scope();
        let ip: IpAddr = "192.168.1.42".parse().unwrap();
        assert!(scope.authorize(1500, KEY, ip, "mdns").is_err());
    }

    #[test]
    fn authorize_rejects_expired_window() {
        let scope = valid_scope();
        let ip: IpAddr = "192.168.1.42".parse().unwrap();
        assert!(scope.authorize(9999, KEY, ip, "opcua").is_err());
        assert!(scope.authorize(1, KEY, ip, "opcua").is_err());
    }

    #[test]
    fn authorize_rejects_tampered_scope() {
        let mut scope = valid_scope();
        scope.allowed_cidrs.push("0.0.0.0/0".to_string()); // élargi après signature
        let ip: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(scope.authorize(1500, KEY, ip, "opcua").is_err());
    }

    #[test]
    fn authorize_rejects_wrong_key() {
        let scope = valid_scope();
        let ip: IpAddr = "192.168.1.42".parse().unwrap();
        assert!(scope.authorize(1500, b"wrong-key", ip, "opcua").is_err());
    }

    #[test]
    fn authorize_rejects_high_security_zone_by_default() {
        let mut scope = ScopeAuthorization {
            zone_security_level: 3,
            ..valid_scope()
        };
        scope = scope.sign(KEY); // re-sign after mutating a signed field
        let ip: IpAddr = "192.168.1.42".parse().unwrap();
        assert!(scope.authorize(1500, KEY, ip, "opcua").is_err());
    }

    #[test]
    fn authorize_allows_high_security_zone_with_explicit_override() {
        let mut scope = ScopeAuthorization {
            zone_security_level: 3,
            allow_high_security_zone: true,
            ..valid_scope()
        };
        scope = scope.sign(KEY);
        let ip: IpAddr = "192.168.1.42".parse().unwrap();
        assert!(scope.authorize(1500, KEY, ip, "opcua").is_ok());
    }

    #[test]
    fn authorize_rejects_ipv6_target() {
        let scope = valid_scope();
        let ip: IpAddr = "::1".parse().unwrap();
        assert!(scope.authorize(1500, KEY, ip, "opcua").is_err());
    }
}
