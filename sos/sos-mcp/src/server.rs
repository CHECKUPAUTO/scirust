//! The MCP service loop over stdio: one JSON-RPC 2.0 request per input line,
//! one response per output line.
//!
//! Every tool call — success or failure — is attested into a
//! [`CcosChain`](sos_ccos::CcosChain): the same tamper-evident
//! `input → output → chain` hash link `sos-ccos` uses for cognitive acts.
//! Reusing it here rather than a bespoke audit log is not a coincidence — an
//! MCP tool call **is** a cognitive act in RFC-0002's framing (an external
//! agent driving the OS), so it gets the same attestation mechanism, not a
//! second one.

use std::io::{self, BufRead, Write};

use serde_json::{Value, json};
use sos_ccos::CcosChain;

use crate::protocol::{
    INVALID_PARAMS, MCP_PROTOCOL_VERSION, METHOD_NOT_FOUND, PARSE_ERROR, RpcRequest, RpcResponse,
};
use crate::registry::ToolRegistry;

/// A request line longer than this is rejected rather than buffered without
/// bound.
const MAX_REQUEST_BYTES: usize = 1024 * 1024;

/// The MCP server: a tool registry plus the attestation chain of every call.
pub struct McpServer {
    registry: ToolRegistry,
    chain: CcosChain,
}

impl McpServer {
    /// Construct a server over `registry`.
    #[must_use]
    pub fn new(registry: ToolRegistry) -> Self {
        Self {
            registry,
            chain: CcosChain::new(),
        }
    }

    /// The attestation chain of every tool call handled so far.
    #[must_use]
    pub fn chain(&self) -> &CcosChain {
        &self.chain
    }

    /// The wrapped tool registry.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Handle one already-deserialized request. Returns `None` for a
    /// notification (no `id`) — the protocol expects no response.
    pub fn handle(&mut self, req: RpcRequest) -> Option<RpcResponse> {
        let id = req.id.clone();
        let is_notification = id.is_none();

        let result: Result<Value, (i64, String)> = match req.method.as_str()
        {
            "initialize" => Ok(json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": { "name": "sos-mcp", "version": env!("CARGO_PKG_VERSION") },
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
            .ok_or((INVALID_PARAMS, "missing `name`".to_owned()))?;
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let outcome = self.registry.call(name, arguments.clone());
        let attested = match &outcome
        {
            Ok(value) => json!({ "outcome": "ok", "name": name, "value": value }),
            Err(message) => json!({ "outcome": "error", "name": name, "message": message }),
        };
        self.chain.append(
            arguments.to_string().as_bytes(),
            attested.to_string().as_bytes(),
        );

        match outcome
        {
            Ok(value) => Ok(json!({
                "content": [{ "type": "text", "text": value.to_string() }],
                "isError": false,
            })),
            Err(message) =>
            {
                // A tool's own failure is a normal MCP *result* (`isError:
                // true`), not a JSON-RPC protocol error — that is how the
                // client knows to show it to the model rather than treat it
                // as a transport fault.
                Ok(json!({
                    "content": [{ "type": "text", "text": message }],
                    "isError": true,
                }))
            },
        }
    }
}

/// Run the blocking stdio service loop.
///
/// # Errors
/// [`io::Error`] if reading stdin or writing stdout fails.
pub fn run_stdio(server: McpServer) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_transport(server, stdin.lock(), stdout.lock())
}

enum BoundedLine {
    Eof,
    Line(Vec<u8>),
    TooLong,
}

fn read_bounded_line<R: BufRead>(reader: &mut R) -> io::Result<BoundedLine> {
    let mut line = Vec::new();
    let mut too_long = false;
    loop
    {
        let available = reader.fill_buf()?;
        if available.is_empty()
        {
            return if line.is_empty() && !too_long
            {
                Ok(BoundedLine::Eof)
            }
            else if too_long
            {
                Ok(BoundedLine::TooLong)
            }
            else
            {
                Ok(BoundedLine::Line(line))
            };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |index| index + 1);
        let content_len = newline.unwrap_or(available.len());
        if !too_long
        {
            if line.len().saturating_add(content_len) > MAX_REQUEST_BYTES
            {
                too_long = true;
                line.clear();
            }
            else
            {
                line.extend_from_slice(&available[..content_len]);
            }
        }
        reader.consume(consumed);
        if newline.is_some()
        {
            if too_long
            {
                return Ok(BoundedLine::TooLong);
            }
            if line.last() == Some(&b'\r')
            {
                line.pop();
            }
            return Ok(BoundedLine::Line(line));
        }
    }
}

fn run_transport<R: BufRead, W: Write>(
    mut server: McpServer,
    mut input: R,
    mut output: W,
) -> io::Result<()> {
    loop
    {
        let response = match read_bounded_line(&mut input)?
        {
            BoundedLine::Eof => break,
            BoundedLine::Line(line) if line.iter().all(u8::is_ascii_whitespace) => continue,
            BoundedLine::Line(line) => match serde_json::from_slice::<RpcRequest>(&line)
            {
                Ok(req) => server.handle(req),
                Err(e) => Some(RpcResponse::err(
                    Value::Null,
                    PARSE_ERROR,
                    format!("parse error: {e}"),
                )),
            },
            BoundedLine::TooLong => Some(RpcResponse::err(
                Value::Null,
                PARSE_ERROR,
                format!("request exceeds the {MAX_REQUEST_BYTES}-byte limit"),
            )),
        };
        if let Some(resp) = response
        {
            let text = serde_json::to_string(&resp)?;
            writeln!(output, "{text}")?;
            output.flush()?;
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
            jsonrpc: "2.0".to_owned(),
            id: id.map(Value::from),
            method: method.to_owned(),
            params,
        }
    }

    fn test_server() -> McpServer {
        let mut registry = ToolRegistry::new();
        registry.register(McpTool {
            name: "echo".to_owned(),
            description: "echoes its input".to_owned(),
            input_schema: json!({"type": "object"}),
            handler: Box::new(Ok),
        });
        registry.register(McpTool {
            name: "always_fails".to_owned(),
            description: "always returns an error".to_owned(),
            input_schema: json!({"type": "object"}),
            handler: Box::new(|_args| Err("boom".to_owned())),
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
            Value::String(MCP_PROTOCOL_VERSION.to_owned())
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
    fn tools_call_success_is_attested() {
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
        assert_eq!(server.chain().len(), 1);
        server.chain().verify().unwrap();
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
        assert_eq!(server.chain().len(), 1);
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

    #[test]
    fn transport_rejects_oversized_line_and_recovers_for_next_request() {
        let mut input = vec![b' '; MAX_REQUEST_BYTES + 1];
        input.push(b'\n');
        input.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n");
        let mut output = Vec::new();
        run_transport(test_server(), io::Cursor::new(input), &mut output).unwrap();
        let responses: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["error"]["code"], json!(PARSE_ERROR));
        assert_eq!(responses[1]["id"], json!(2));
    }
}
