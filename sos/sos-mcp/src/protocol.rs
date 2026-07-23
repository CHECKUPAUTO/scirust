//! Minimal JSON-RPC 2.0 types for the Model Context Protocol's stdio
//! transport (<https://modelcontextprotocol.io>).
//!
//! MCP's stdio transport places one JSON object per line, with no
//! `Content-Length` framing (unlike LSP, with which it is sometimes
//! confused).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The JSON-RPC version string every message declares.
pub const JSONRPC_VERSION: &str = "2.0";
/// The MCP protocol version this server announces during `initialize`.
pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Standard JSON-RPC error code: the request could not be parsed as JSON.
pub const PARSE_ERROR: i64 = -32700;
/// Standard JSON-RPC error code: no handler for the requested method.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// Standard JSON-RPC error code: the method's parameters were invalid.
pub const INVALID_PARAMS: i64 = -32602;

/// An incoming JSON-RPC request (or notification, if `id` is absent).
#[derive(Debug, Clone, Deserialize)]
pub struct RpcRequest {
    /// The JSON-RPC version (unchecked; present for wire compatibility).
    #[serde(default)]
    pub jsonrpc: String,
    /// The request id, echoed back in the response. Absent for a
    /// notification, which expects no response.
    #[serde(default)]
    pub id: Option<Value>,
    /// The method name to dispatch on.
    pub method: String,
    /// The method's parameters.
    #[serde(default)]
    pub params: Value,
}

/// An outgoing JSON-RPC response: exactly one of `result`/`error` is set.
#[derive(Debug, Clone, Serialize)]
pub struct RpcResponse {
    /// Always [`JSONRPC_VERSION`].
    pub jsonrpc: &'static str,
    /// Echoes the request's id.
    pub id: Value,
    /// The method's result, on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// The failure, on error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// A JSON-RPC error object.
#[derive(Debug, Clone, Serialize)]
pub struct RpcError {
    /// The JSON-RPC (or MCP-specific) error code.
    pub code: i64,
    /// A human-readable description.
    pub message: String,
}

impl RpcResponse {
    /// A successful response.
    #[must_use]
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            result: Some(result),
            error: None,
        }
    }

    /// An error response.
    #[must_use]
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
