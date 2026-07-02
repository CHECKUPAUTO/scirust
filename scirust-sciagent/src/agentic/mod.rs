pub mod guard;
pub mod tools;
pub use guard::{ConformalGuard, GuardVerdict};
pub use tools::Tool;
pub use tools::ToolResult;

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum AgentAction {
    Call { tool: String, params: HashMap<String, String> },
    Respond { text: String },
    Abstain,
}

#[derive(Debug)]
pub struct AgentTurn {
    pub action: AgentAction,
    pub result: Option<String>,
}

pub struct AgentRouter {
    tools: Vec<Tool>,
}

impl AgentRouter {
    pub fn new() -> Self {
        Self {
            tools: Tool::builtins(),
        }
    }

    pub fn parse_action(&self, text: &str) -> AgentAction {
        // Try to parse a JSON tool call from the model output
        if let Some(json) = extract_json(text) {
            if let Some(action) = self.parse_tool_call(&json) {
                return action;
            }
        }

        // Check for abstain keywords
        let lower = text.to_lowercase();
        if lower.contains("abstain") || lower.contains("i don't know") || lower.contains("pas sûr") {
            return AgentAction::Abstain;
        }

        AgentAction::Respond { text: text.to_string() }
    }

    pub fn parse_tool_call(&self, json: &serde_json::Value) -> Option<AgentAction> {
        let name = json.get("name").and_then(|v| v.as_str())?;
        let params: HashMap<String, String> = json
            .get("params")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if self.tools.iter().any(|t| t.name == name) {
            Some(AgentAction::Call {
                tool: name.to_string(),
                params,
            })
        } else {
            None
        }
    }

    pub fn execute(&self, action: &AgentAction) -> String {
        match action {
            AgentAction::Call { tool, params } => {
                for t in &self.tools {
                    if t.name == *tool {
                        return (t.execute)(params.clone());
                    }
                }
                format!("Unknown tool: {tool}")
            }
            AgentAction::Respond { text } => text.clone(),
            AgentAction::Abstain => {
                "I abstain — confidence below threshold.".to_string()
            }
        }
    }
}

fn extract_json(text: &str) -> Option<serde_json::Value> {
    // Find JSON in markdown code blocks or bare JSON
    let candidates = [
        text,
        &text.trim(),
    ];

    for &candidate in &candidates {
        // Try to strip ```json ... ``` markers
        let cleaned = if let Some(start) = candidate.find("```json") {
            let start = start + 7;
            let end = candidate[start..].find("```").map(|e| start + e).unwrap_or(candidate.len());
            &candidate[start..end]
        } else if let Some(start) = candidate.find("```") {
            let start = start + 3;
            let end = candidate[start..].find("```").map(|e| start + e).unwrap_or(candidate.len());
            &candidate[start..end]
        } else {
            candidate
        };

        if let Ok(val) = serde_json::from_str::<serde_json::Value>(cleaned.trim()) {
            if val.is_object() && val.get("name").is_some() {
                return Some(val);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_call_json() {
        let router = AgentRouter::new();
        let text = r#"{"name": "search", "params": {"pattern": "Muon", "path": "scirust-core"}}"#;
        let action = router.parse_action(text);
        assert_eq!(
            action,
            AgentAction::Call {
                tool: "search".to_string(),
                params: [("pattern".to_string(), "Muon".to_string()), ("path".to_string(), "scirust-core".to_string())]
                    .iter().cloned().collect(),
            }
        );
    }

    #[test]
    fn test_parse_tool_call_markdown() {
        let router = AgentRouter::new();
        let text = "I'll search for that:\n```json\n{\"name\": \"grep\", \"params\": {\"pattern\": \"PcgEngine\"}}\n```";
        let action = router.parse_action(text);
        assert_eq!(
            action,
            AgentAction::Call {
                tool: "grep".to_string(),
                params: [("pattern".to_string(), "PcgEngine".to_string())]
                    .iter().cloned().collect(),
            }
        );
    }

    #[test]
    fn test_parse_respond() {
        let router = AgentRouter::new();
        let action = router.parse_action("Hello, I can help with that.");
        assert_eq!(action, AgentAction::Respond { text: "Hello, I can help with that.".to_string() });
    }

    #[test]
    fn test_parse_abstain() {
        let router = AgentRouter::new();
        let action = router.parse_action("I abstain from answering this.");
        assert_eq!(action, AgentAction::Abstain);
    }

    #[test]
    fn test_execute_search() {
        let router = AgentRouter::new();
        let action = AgentAction::Call {
            tool: "search".to_string(),
            params: [("pattern".to_string(), "NdMuon".to_string()), ("path".to_string(), "/root/scirust/scirust-core".to_string())]
                .iter().cloned().collect(),
        };
        let result = router.execute(&action);
        assert!(result.contains("NdMuon"), "Search should find NdMuon, got: {result}");
    }
}
