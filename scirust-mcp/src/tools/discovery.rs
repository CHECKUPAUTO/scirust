//! Outil MCP pour `scirust-discovery`.
//!
//! La clé HMAC qui autorise une portée de découverte n'est **jamais** un
//! argument d'appel d'outil : elle vit côté serveur, dans la variable
//! d'environnement `SCIRUST_DISCOVERY_KEY`. Un agent ne peut donc jamais
//! s'auto-autoriser en fabriquant une portée signée à l'intérieur de la
//! conversation elle-même — seul l'opérateur qui a déployé ce serveur MCP
//! (et qui connaît la clé utilisée pour signer une portée hors bande)
//! contrôle si la découverte est possible du tout.

use crate::registry::McpTool;
use scirust_discovery::{DiscoveryEngine, Protocol, ScopeAuthorization};
use serde_json::json;
use std::net::IpAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn discovery_tools() -> Vec<McpTool> {
    vec![scan_tool()]
}

fn scan_tool() -> McpTool {
    McpTool {
        name: "discovery_scan".to_string(),
        description: "Safely probe OT/IT network targets for known industrial protocols \
            (OPC-UA, Modbus TCP, mDNS) using protocol-native handshakes only — never a generic \
            port scan. Requires a signed `scope` (see scirust-discovery::ScopeAuthorization) \
            covering every target's IP/protocol, and the server operator to have set the \
            SCIRUST_DISCOVERY_KEY environment variable to the key that scope was signed with. \
            Every attempt — in scope, unreachable, or refused — is appended to a SHA-256 \
            hash-chained audit log."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "scope": {
                    "type": "object",
                    "description": "a signed ScopeAuthorization: {operator, zone, zone_security_level, allowed_cidrs, allowed_protocols, valid_from_unix, valid_until_unix, allow_high_security_zone, signature_hex}",
                },
                "targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ip": { "type": "string", "description": "IPv4 address" },
                            "protocol": { "type": "string", "enum": ["opcua", "modbus", "mdns"] },
                        },
                        "required": ["ip", "protocol"],
                    },
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "per-target probe timeout in milliseconds (default 2000)",
                },
            },
            "required": ["scope", "targets"],
        }),
        handler: Box::new(|args| {
            let key = std::env::var("SCIRUST_DISCOVERY_KEY").map_err(|_| {
                "discovery is disabled: the server operator has not set SCIRUST_DISCOVERY_KEY"
                    .to_string()
            })?;
            let scope: ScopeAuthorization =
                serde_json::from_value(args.get("scope").cloned().ok_or("missing `scope`")?)
                    .map_err(|e| format!("invalid `scope`: {e}"))?;

            let targets_json = args
                .get("targets")
                .and_then(|v| v.as_array())
                .ok_or("missing `targets` array")?;
            let mut targets = Vec::with_capacity(targets_json.len());
            for (i, t) in targets_json.iter().enumerate()
            {
                let ip_str = t
                    .get("ip")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("targets[{i}]: missing `ip`"))?;
                let ip: IpAddr = ip_str
                    .parse()
                    .map_err(|_| format!("targets[{i}]: '{ip_str}' is not a valid IP address"))?;
                let proto_str = t
                    .get("protocol")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("targets[{i}]: missing `protocol`"))?;
                let proto = Protocol::parse(proto_str).map_err(|e| format!("targets[{i}]: {e}"))?;
                targets.push((ip, proto));
            }

            let timeout_ms = args
                .get("timeout_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(2000);
            let now_unix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let mut engine =
                DiscoveryEngine::new(scope, key.into_bytes(), Duration::from_millis(timeout_ms));
            let results = engine.scan(&targets, now_unix);
            Ok(json!({
                "results": results,
                "audit_chain_valid": engine.audit_log().verify_chain(),
                "audit_entries": engine.audit_log().len(),
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `SCIRUST_DISCOVERY_KEY` is process-global state; serialize the tests
    // that touch it so they don't race across the test harness's threads.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn refuses_without_server_side_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("SCIRUST_DISCOVERY_KEY");
        let tool = scan_tool();
        let result = (tool.handler)(json!({ "scope": {}, "targets": [] }));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SCIRUST_DISCOVERY_KEY"));
    }

    #[test]
    fn refuses_out_of_scope_target_but_still_reports_audit_chain() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("SCIRUST_DISCOVERY_KEY", "test-key-for-mcp-tool");
        let scope = ScopeAuthorization {
            operator: "alice".to_string(),
            zone: "zone-a".to_string(),
            zone_security_level: 1,
            allowed_cidrs: vec!["192.0.2.0/24".to_string()],
            allowed_protocols: vec!["opcua".to_string()],
            valid_from_unix: 0,
            valid_until_unix: 4_000_000_000,
            allow_high_security_zone: false,
            signature_hex: String::new(),
        }
        .sign(b"test-key-for-mcp-tool");

        let tool = scan_tool();
        let result = (tool.handler)(json!({
            "scope": scope,
            "targets": [{ "ip": "10.0.0.1", "protocol": "opcua" }],
        }))
        .unwrap();
        assert_eq!(result["audit_entries"], json!(1));
        assert_eq!(result["audit_chain_valid"], json!(true));
        assert_eq!(result["results"][0]["outcome"]["status"], json!("refused"));
        std::env::remove_var("SCIRUST_DISCOVERY_KEY");
    }

    #[test]
    fn rejects_invalid_ip_in_targets() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("SCIRUST_DISCOVERY_KEY", "another-test-key");
        let scope = ScopeAuthorization {
            operator: "alice".to_string(),
            zone: "zone-a".to_string(),
            zone_security_level: 1,
            allowed_cidrs: vec!["192.0.2.0/24".to_string()],
            allowed_protocols: vec!["opcua".to_string()],
            valid_from_unix: 0,
            valid_until_unix: 4_000_000_000,
            allow_high_security_zone: false,
            signature_hex: String::new(),
        }
        .sign(b"another-test-key");
        let tool = scan_tool();
        let result = (tool.handler)(json!({
            "scope": scope,
            "targets": [{ "ip": "not-an-ip", "protocol": "opcua" }],
        }));
        assert!(result.is_err());
        std::env::remove_var("SCIRUST_DISCOVERY_KEY");
    }
}
