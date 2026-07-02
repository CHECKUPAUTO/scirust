//! Adapte les outils de développement de `scirust_sciagent::agentic`
//! (`search`, `grep`, `read`, `explain`, `build`, `test`, `status`) au
//! format MCP, avec le préfixe `dev_`. Ce sont les outils que le SLM
//! `scirust-sciagent` utilisait déjà en interne ; les exposer aussi en MCP
//! les rend appelables par n'importe quel autre agent sans dupliquer leur
//! implémentation.

use crate::registry::McpTool;
use scirust_sciagent::agentic::tools::Tool;
use serde_json::{Value, json};
use std::collections::HashMap;

fn value_to_string(v: &Value) -> String {
    match v
    {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn dev_tools() -> Vec<McpTool> {
    Tool::builtins()
        .into_iter()
        .map(|t| {
            let properties: serde_json::Map<String, Value> = t
                .parameters
                .iter()
                .map(|p| {
                    (
                        p.name.to_string(),
                        json!({ "type": p.param_type, "description": p.description }),
                    )
                })
                .collect();
            let required: Vec<Value> = t
                .parameters
                .iter()
                .filter(|p| p.required)
                .map(|p| Value::String(p.name.to_string()))
                .collect();
            let execute = t.execute;
            McpTool {
                name: format!("dev_{}", t.name),
                description: t.description.to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }),
                handler: Box::new(move |args: Value| {
                    let map: HashMap<String, String> = match args
                    {
                        Value::Object(o) => o
                            .iter()
                            .map(|(k, v)| (k.clone(), value_to_string(v)))
                            .collect(),
                        Value::Null => HashMap::new(),
                        _ => return Err("arguments must be a JSON object".to_string()),
                    };
                    Ok(Value::String((execute)(map)))
                }),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_all_builtin_dev_tools() {
        let tools = dev_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"dev_search"));
        assert!(names.contains(&"dev_status"));
        assert_eq!(tools.len(), Tool::builtins().len());
    }

    #[test]
    fn dev_status_runs_without_arguments() {
        let tools = dev_tools();
        let status = tools.iter().find(|t| t.name == "dev_status").unwrap();
        let result = (status.handler)(json!({}));
        assert!(result.is_ok());
    }

    #[test]
    fn dev_search_finds_known_symbol() {
        let tools = dev_tools();
        let search = tools.iter().find(|t| t.name == "dev_search").unwrap();
        // Répertoire de cette crate elle-même — n'importe quel checkout du
        // workspace l'a, pas besoin de connaître la racine du workspace.
        let here = env!("CARGO_MANIFEST_DIR");
        let result = (search.handler)(json!({
            "pattern": "fn dev_tools",
            "path": format!("{here}/src"),
        }))
        .unwrap();
        assert!(result.as_str().unwrap().contains("dev_tools"));
    }
}
