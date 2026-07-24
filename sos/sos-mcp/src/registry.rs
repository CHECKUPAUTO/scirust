//! The MCP tool registry.
//!
//! Each tool exposes a name, description, JSON input schema (so a client —
//! agent or script — knows how to call it without guessing), and a
//! synchronous handler `Value -> Result<Value, String>`. This is the server's
//! single extension point: a new SOS syscall registers here and becomes
//! immediately callable by any MCP client.

use serde_json::Value;

/// A tool's handler: takes its JSON arguments, returns its result or an
/// error message.
pub type ToolHandler = Box<dyn Fn(Value) -> Result<Value, String> + Send + Sync>;

/// One registered MCP tool.
pub struct McpTool {
    /// The tool's name, as clients call it.
    pub name: String,
    /// A human-readable description (surfaced to the calling agent).
    pub description: String,
    /// A JSON Schema describing the tool's expected arguments.
    pub input_schema: Value,
    /// The synchronous handler.
    pub handler: ToolHandler,
}

/// A registry of MCP tools, indexed by name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<McpTool>,
}

impl ToolRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool.
    ///
    /// # Panics
    /// If a tool with the same name is already registered — a name collision
    /// is a server configuration error, not a runtime case to handle silently.
    pub fn register(&mut self, tool: McpTool) {
        assert!(
            !self.tools.iter().any(|t| t.name == tool.name),
            "duplicate MCP tool name: {}",
            tool.name
        );
        self.tools.push(tool);
    }

    /// The `tools/list` JSON payload: every tool's name, description, and
    /// input schema.
    #[must_use]
    pub fn list_json(&self) -> Value {
        Value::Array(
            self.tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    })
                })
                .collect(),
        )
    }

    /// Call a tool by name.
    ///
    /// # Errors
    /// A message naming the tool if none is registered under `name`, or the
    /// tool handler's own error message.
    pub fn call(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("unknown tool: {name}"))?;
        (tool.handler)(arguments)
    }

    /// Every registered tool's name.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    /// How many tools are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dummy_tool(name: &str) -> McpTool {
        McpTool {
            name: name.to_owned(),
            description: "a dummy tool".to_owned(),
            input_schema: json!({"type": "object"}),
            handler: Box::new(|_args| Ok(json!({"ok": true}))),
        }
    }

    #[test]
    fn register_and_call() {
        let mut reg = ToolRegistry::new();
        reg.register(dummy_tool("echo"));
        assert_eq!(reg.len(), 1);
        let result = reg.call("echo", json!({})).unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[test]
    fn call_unknown_tool_errors() {
        let reg = ToolRegistry::new();
        assert!(reg.call("nope", json!({})).is_err());
    }

    #[test]
    #[should_panic(expected = "duplicate MCP tool name")]
    fn duplicate_registration_panics() {
        let mut reg = ToolRegistry::new();
        reg.register(dummy_tool("echo"));
        reg.register(dummy_tool("echo"));
    }

    #[test]
    fn list_json_includes_schema() {
        let mut reg = ToolRegistry::new();
        reg.register(dummy_tool("echo"));
        let list = reg.list_json();
        assert_eq!(list[0]["name"], "echo");
        assert_eq!(list[0]["inputSchema"]["type"], "object");
    }
}
