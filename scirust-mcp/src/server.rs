//! Boucle de service MCP sur stdio : une requête JSON-RPC 2.0 par ligne en
//! entrée, une réponse par ligne en sortie.

use crate::audit::AuditLog;
use crate::protocol::{
    INVALID_PARAMS, MCP_PROTOCOL_VERSION, METHOD_NOT_FOUND, PARSE_ERROR, RpcRequest, RpcResponse,
};
use crate::registry::ToolRegistry;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

pub struct McpServer {
    registry: ToolRegistry,
    audit: AuditLog,
}

impl McpServer {
    pub fn new(registry: ToolRegistry) -> Self {
        Self {
            registry,
            audit: AuditLog::new(),
        }
    }

    pub fn audit_log(&self) -> &AuditLog {
        &self.audit
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Traite une requête déjà désérialisée. Renvoie `None` pour une
    /// notification (pas d'`id`) — le protocole n'attend aucune réponse.
    pub fn handle(&mut self, req: RpcRequest) -> Option<RpcResponse> {
        let id = req.id.clone();
        let is_notification = id.is_none();

        let result: Result<Value, (i64, String)> = match req.method.as_str()
        {
            "initialize" => Ok(json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": { "name": "scirust-mcp", "version": env!("CARGO_PKG_VERSION") },
                "capabilities": { "tools": {} },
            })),
            "notifications/initialized" | "notifications/cancelled" => return None,
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": self.registry.list_json() })),
            "resources/list" => Ok(json!({ "resources": [] })),
            "prompts/list" => Ok(json!({ "prompts": [] })),
            "tools/call" => self.handle_tool_call(&req.params),
            other => Err((METHOD_NOT_FOUND, format!("unknown method: {other}"))),
        };

        if is_notification
        {
            return None;
        }
        let id = id.unwrap_or(Value::Null);
        Some(match result
        {
            Ok(value) => RpcResponse::ok(id, value),
            Err((code, message)) => RpcResponse::err(id, code, message),
        })
    }

    fn handle_tool_call(&mut self, params: &Value) -> Result<Value, (i64, String)> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or((INVALID_PARAMS, "missing `name`".to_string()))?;
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        match self.registry.call(name, arguments.clone())
        {
            Ok(value) =>
            {
                self.audit.record(name, &arguments, "ok", &value);
                Ok(json!({
                    "content": [{ "type": "text", "text": value.to_string() }],
                    "isError": false,
                }))
            },
            Err(message) =>
            {
                let err_value = json!({ "error": message });
                self.audit.record(name, &arguments, "error", &err_value);
                // Une erreur d'exécution d'outil est un résultat MCP normal
                // (isError: true), pas une erreur de protocole JSON-RPC —
                // c'est ainsi que le client sait la présenter au modèle
                // plutôt que de la traiter comme une panne du transport.
                Ok(json!({
                    "content": [{ "type": "text", "text": message }],
                    "isError": true,
                }))
            },
        }
    }
}

/// Lance la boucle stdio bloquante.
pub fn run_stdio(mut server: McpServer) -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines()
    {
        let line = line?;
        if line.trim().is_empty()
        {
            continue;
        }
        let response = match serde_json::from_str::<RpcRequest>(&line)
        {
            Ok(req) => server.handle(req),
            Err(e) => Some(RpcResponse::err(
                Value::Null,
                PARSE_ERROR,
                format!("parse error: {e}"),
            )),
        };
        if let Some(resp) = response
        {
            let text = serde_json::to_string(&resp)?;
            writeln!(stdout, "{text}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::McpTool;

    fn req(id: Option<i64>, method: &str, params: Value) -> RpcRequest {
        RpcRequest {
            jsonrpc: "2.0".to_string(),
            id: id.map(Value::from),
            method: method.to_string(),
            params,
        }
    }

    fn test_server() -> McpServer {
        let mut registry = ToolRegistry::new();
        registry.register(McpTool {
            name: "echo".to_string(),
            description: "echoes its input".to_string(),
            input_schema: json!({"type": "object"}),
            handler: Box::new(Ok),
        });
        registry.register(McpTool {
            name: "always_fails".to_string(),
            description: "always returns an error".to_string(),
            input_schema: json!({"type": "object"}),
            handler: Box::new(|_args| Err("boom".to_string())),
        });
        McpServer::new(registry)
    }

    #[test]
    fn initialize_reports_protocol_version() {
        let mut server = test_server();
        let resp = server
            .handle(req(Some(1), "initialize", json!({})))
            .unwrap();
        assert_eq!(
            resp.result.unwrap()["protocolVersion"],
            Value::String(MCP_PROTOCOL_VERSION.to_string())
        );
    }

    #[test]
    fn notification_gets_no_response() {
        let mut server = test_server();
        let resp = server.handle(req(None, "notifications/initialized", json!({})));
        assert!(resp.is_none());
    }

    #[test]
    fn tools_list_reflects_registry() {
        let mut server = test_server();
        let resp = server
            .handle(req(Some(1), "tools/list", json!({})))
            .unwrap();
        let tools = resp.result.unwrap()["tools"].clone();
        assert_eq!(tools.as_array().unwrap().len(), 2);
    }

    #[test]
    fn tools_call_success_is_audited() {
        let mut server = test_server();
        let resp = server
            .handle(req(
                Some(1),
                "tools/call",
                json!({ "name": "echo", "arguments": { "x": 42 } }),
            ))
            .unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], Value::Bool(false));
        assert_eq!(server.audit_log().len(), 1);
        assert!(server.audit_log().verify_chain());
    }

    #[test]
    fn tools_call_failure_is_reported_as_mcp_result_not_protocol_error() {
        let mut server = test_server();
        let resp = server
            .handle(req(
                Some(1),
                "tools/call",
                json!({ "name": "always_fails" }),
            ))
            .unwrap();
        assert!(
            resp.error.is_none(),
            "tool failure must not be a JSON-RPC error"
        );
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], Value::Bool(true));
        assert_eq!(server.audit_log().len(), 1);
        assert_eq!(server.audit_log().entries()[0].outcome, "error");
    }

    #[test]
    fn unknown_method_is_a_protocol_error() {
        let mut server = test_server();
        let resp = server
            .handle(req(Some(1), "nonexistent/method", json!({})))
            .unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn tools_call_missing_name_is_invalid_params() {
        let mut server = test_server();
        let resp = server
            .handle(req(Some(1), "tools/call", json!({})))
            .unwrap();
        assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
    }
}
