//! Types JSON-RPC 2.0 minimaux pour le transport stdio du Model Context
//! Protocol (MCP) — <https://modelcontextprotocol.io>.
//!
//! Le transport stdio du MCP place un objet JSON par ligne, sans caractère
//! de nouvelle ligne à l'intérieur d'un message, et **sans** framing
//! `Content-Length` (contrairement au LSP, avec lequel il est parfois
//! confondu).

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const JSONRPC_VERSION: &str = "2.0";
/// Version du protocole MCP annoncée par ce serveur lors de `initialize`.
pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

pub const PARSE_ERROR: i64 = -32700;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;

#[derive(Debug, Clone, Deserialize)]
pub struct RpcRequest {
    #[serde(default)]
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl RpcResponse {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_request() {
        let req: RpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#).unwrap();
        assert_eq!(req.method, "ping");
        assert_eq!(req.id, Some(Value::from(1)));
    }

    #[test]
    fn parses_notification_without_id() {
        let req: RpcRequest =
            serde_json::from_str(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .unwrap();
        assert_eq!(req.id, None);
    }

    #[test]
    fn response_serializes_without_null_fields() {
        let resp = RpcResponse::ok(Value::from(1), serde_json::json!({"a": 1}));
        let text = serde_json::to_string(&resp).unwrap();
        assert!(!text.contains("error"));
    }
}
