//! Moteur d'orchestration : vérifie l'autorisation de portée *avant tout
//! envoi de paquet*, sonde le protocole demandé si la cible est autorisée,
//! et journalise chaque tentative — dans la portée ou refusée — dans le
//! journal d'audit hash-chaîné. C'est le seul point d'entrée destiné aux
//! appelants (CLI, outil MCP) : il n'existe aucun chemin qui sonde le
//! réseau sans passer par la vérification de portée.

use crate::audit::AuditLog;
use crate::protocols::{bacnet, ethernet_ip, mdns, modbus, opcua, snmp};
use crate::scope::ScopeAuthorization;
use serde::Serialize;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Protocol {
    OpcUa,
    ModbusTcp,
    Mdns,
    BacnetIp,
    Snmp,
    EtherNetIp,
}

impl Protocol {
    pub fn as_str(&self) -> &'static str {
        match self
        {
            Protocol::OpcUa => "opcua",
            Protocol::ModbusTcp => "modbus",
            Protocol::Mdns => "mdns",
            Protocol::BacnetIp => "bacnet",
            Protocol::Snmp => "snmp",
            Protocol::EtherNetIp => "ethernet_ip",
        }
    }

    pub fn default_port(&self) -> u16 {
        match self
        {
            Protocol::OpcUa => opcua::DEFAULT_PORT,
            Protocol::ModbusTcp => modbus::DEFAULT_PORT,
            Protocol::Mdns => 5353,
            Protocol::BacnetIp => bacnet::DEFAULT_PORT,
            Protocol::Snmp => snmp::DEFAULT_PORT,
            Protocol::EtherNetIp => ethernet_ip::DEFAULT_PORT,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s
        {
            "opcua" => Ok(Protocol::OpcUa),
            "modbus" => Ok(Protocol::ModbusTcp),
            "mdns" => Ok(Protocol::Mdns),
            "bacnet" => Ok(Protocol::BacnetIp),
            "snmp" => Ok(Protocol::Snmp),
            "ethernet_ip" => Ok(Protocol::EtherNetIp),
            other => Err(format!(
                "unknown protocol '{other}' (expected opcua, modbus, mdns, bacnet, snmp, or ethernet_ip)"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DiscoveryOutcome {
    Found { summary: String },
    NotFound { reason: String },
    Refused { reason: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryResult {
    pub target: String,
    pub protocol: String,
    pub outcome: DiscoveryOutcome,
}

/// Orchestrateur de découverte : une portée signée, une clé pour la
/// vérifier, et un journal d'audit qui grandit à chaque tentative.
pub struct DiscoveryEngine {
    scope: ScopeAuthorization,
    hmac_key: Vec<u8>,
    timeout: Duration,
    audit: AuditLog,
}

impl DiscoveryEngine {
    pub fn new(scope: ScopeAuthorization, hmac_key: Vec<u8>, timeout: Duration) -> Self {
        Self {
            scope,
            hmac_key,
            timeout,
            audit: AuditLog::new(),
        }
    }

    pub fn audit_log(&self) -> &AuditLog {
        &self.audit
    }

    /// Sonde une cible unique. `now_unix` est le temps courant, injecté par
    /// l'appelant plutôt que lu en interne — voir `crate::scope` pour la
    /// justification (testabilité déterministe de la fenêtre de validité).
    pub fn probe_one(&mut self, ip: IpAddr, protocol: Protocol, now_unix: u64) -> DiscoveryResult {
        let target = ip.to_string();

        if let Err(reason) = self
            .scope
            .authorize(now_unix, &self.hmac_key, ip, protocol.as_str())
        {
            self.audit.record(
                &self.scope.operator,
                &self.scope.zone,
                &target,
                protocol.as_str(),
                "refused",
            );
            return DiscoveryResult {
                target,
                protocol: protocol.as_str().to_string(),
                outcome: DiscoveryOutcome::Refused { reason },
            };
        }

        let addr = SocketAddr::new(ip, protocol.default_port());
        let outcome = match protocol
        {
            Protocol::OpcUa => match opcua::probe(
                addr,
                &format!("opc.tcp://{ip}:{}", protocol.default_port()),
                self.timeout,
            )
            {
                Ok(ack) => DiscoveryOutcome::Found {
                    summary: format!(
                        "OPC-UA UACP endpoint (protocol version {})",
                        ack.protocol_version
                    ),
                },
                Err(reason) => DiscoveryOutcome::NotFound { reason },
            },
            Protocol::ModbusTcp => match modbus::probe(addr, 0xFF, self.timeout)
            {
                Ok(id) => DiscoveryOutcome::Found {
                    summary: format!(
                        "Modbus device (vendor={}, product={})",
                        id.vendor_name.as_deref().unwrap_or("unknown"),
                        id.product_code.as_deref().unwrap_or("unknown")
                    ),
                },
                Err(reason) => DiscoveryOutcome::NotFound { reason },
            },
            Protocol::Mdns => match mdns::probe(addr, "_services._dns-sd._udp.local", self.timeout)
            {
                Ok(names) if !names.is_empty() => DiscoveryOutcome::Found {
                    summary: format!("{} mDNS service(s): {}", names.len(), names.join(", ")),
                },
                Ok(_) => DiscoveryOutcome::NotFound {
                    reason: "no services advertised".to_string(),
                },
                Err(reason) => DiscoveryOutcome::NotFound { reason },
            },
            Protocol::BacnetIp => match bacnet::probe(addr, self.timeout)
            {
                Ok(i_am) => DiscoveryOutcome::Found {
                    summary: format!(
                        "BACnet/IP device instance {} (object type {})",
                        i_am.instance, i_am.object_type
                    ),
                },
                Err(reason) => DiscoveryOutcome::NotFound { reason },
            },
            Protocol::Snmp => match snmp::probe(addr, "public", self.timeout)
            {
                Ok(description) => DiscoveryOutcome::Found {
                    summary: format!("SNMP sysDescr: {description}"),
                },
                Err(reason) => DiscoveryOutcome::NotFound { reason },
            },
            Protocol::EtherNetIp => match ethernet_ip::probe(addr, self.timeout)
            {
                Ok(id) => DiscoveryOutcome::Found {
                    summary: format!(
                        "EtherNet/IP device: {} (vendor=0x{:04x}, product_code={})",
                        id.product_name, id.vendor_id, id.product_code
                    ),
                },
                Err(reason) => DiscoveryOutcome::NotFound { reason },
            },
        };

        let outcome_label = match &outcome
        {
            DiscoveryOutcome::Found { .. } => "found",
            DiscoveryOutcome::NotFound { .. } => "not_found",
            DiscoveryOutcome::Refused { .. } => "refused",
        };
        self.audit.record(
            &self.scope.operator,
            &self.scope.zone,
            &target,
            protocol.as_str(),
            outcome_label,
        );

        DiscoveryResult {
            target,
            protocol: protocol.as_str().to_string(),
            outcome,
        }
    }

    pub fn scan(&mut self, targets: &[(IpAddr, Protocol)], now_unix: u64) -> Vec<DiscoveryResult> {
        targets
            .iter()
            .map(|&(ip, proto)| self.probe_one(ip, proto, now_unix))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::ScopeAuthorization;

    const KEY: &[u8] = b"engine-test-key";

    fn scope() -> ScopeAuthorization {
        ScopeAuthorization {
            operator: "bob@example.com".to_string(),
            zone: "test-zone".to_string(),
            zone_security_level: 1,
            // TEST-NET-1 (RFC 5737) : réservé à la documentation, jamais
            // routé sur un vrai réseau — évite toute collision avec le
            // réseau réel de l'environnement qui exécute ce test.
            allowed_cidrs: vec!["192.0.2.0/24".to_string()],
            allowed_protocols: vec![
                "opcua".to_string(),
                "modbus".to_string(),
                "mdns".to_string(),
            ],
            valid_from_unix: 0,
            valid_until_unix: 4_000_000_000,
            allow_high_security_zone: false,
            signature_hex: String::new(),
        }
        .sign(KEY)
    }

    #[test]
    fn out_of_scope_target_is_refused_without_any_network_io() {
        let mut engine = DiscoveryEngine::new(scope(), KEY.to_vec(), Duration::from_millis(200));
        let result = engine.probe_one("10.0.0.1".parse().unwrap(), Protocol::OpcUa, 100);
        assert!(matches!(result.outcome, DiscoveryOutcome::Refused { .. }));
        assert_eq!(engine.audit_log().len(), 1);
        assert_eq!(engine.audit_log().entries()[0].outcome, "refused");
    }

    #[test]
    fn in_scope_target_with_no_listener_is_not_found_not_refused() {
        // 192.0.2.254 is in-scope but never routable (TEST-NET-1) — nothing
        // will ever answer, so this exercises the "authorized but
        // unreachable" path distinctly from "refused".
        let mut engine = DiscoveryEngine::new(scope(), KEY.to_vec(), Duration::from_millis(200));
        let result = engine.probe_one("192.0.2.254".parse().unwrap(), Protocol::ModbusTcp, 100);
        assert!(matches!(result.outcome, DiscoveryOutcome::NotFound { .. }));
        assert_eq!(engine.audit_log().entries()[0].outcome, "not_found");
    }

    #[test]
    fn scan_records_one_audit_entry_per_target() {
        let mut engine = DiscoveryEngine::new(scope(), KEY.to_vec(), Duration::from_millis(200));
        let targets = [
            ("10.0.0.1".parse().unwrap(), Protocol::OpcUa),
            ("192.0.2.254".parse().unwrap(), Protocol::Mdns),
        ];
        let results = engine.scan(&targets, 100);
        assert_eq!(results.len(), 2);
        assert_eq!(engine.audit_log().len(), 2);
        assert!(engine.audit_log().verify_chain());
    }

    #[test]
    fn protocol_parse_roundtrips() {
        assert_eq!(Protocol::parse("opcua").unwrap(), Protocol::OpcUa);
        assert_eq!(Protocol::parse("modbus").unwrap(), Protocol::ModbusTcp);
        assert_eq!(Protocol::parse("mdns").unwrap(), Protocol::Mdns);
        assert!(Protocol::parse("ftp").is_err());
    }
}
