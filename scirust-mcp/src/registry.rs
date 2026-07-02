//! Registre d'outils MCP.
//!
//! Chaque outil expose un nom, une description, un schéma JSON d'entrée
//! (pour que le client — LLM ou script — sache le construire sans deviner),
//! et un handler synchrone `Value -> Result<Value, String>`. C'est le point
//! d'extension unique du serveur : un nouveau domaine SciRust s'enregistre
//! ici et devient immédiatement appelable par n'importe quel agent MCP.

use serde_json::Value;

pub type ToolHandler = Box<dyn Fn(Value) -> Result<Value, String> + Send + Sync>;

pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub handler: ToolHandler,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<McpTool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enregistre un outil. Panique si son nom est déjà pris — une
    /// collision de noms est une erreur de configuration du serveur, pas un
    /// cas d'exécution à gérer silencieusement.
    pub fn register(&mut self, tool: McpTool) {
        assert!(
            !self.tools.iter().any(|t| t.name == tool.name),
            "duplicate MCP tool name: {}",
            tool.name
        );
        self.tools.push(tool);
    }

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

    pub fn call(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("unknown tool: {name}"))?;
        (tool.handler)(arguments)
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

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
            name: name.to_string(),
            description: "a dummy tool".to_string(),
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
