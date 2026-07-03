//! Outil MCP pour `scirust-fab::r2r` : un pas de contrôleur EWMA
//! run-to-run — donne la recette recommandée pour le prochain lot compte
//! tenu du run observé, l'usage naturel pour un agent qui pilote une
//! boucle de recette de fabrication.

use crate::registry::McpTool;
use scirust_fab::EwmaR2rController;
use serde_json::{Value, json};

fn get_f64(v: &Value, field: &str) -> Result<f64, String> {
    v.get(field)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("missing or non-numeric `{field}`"))
}

pub fn fab_tools() -> Vec<McpTool> {
    vec![r2r_update_tool()]
}

fn r2r_update_tool() -> McpTool {
    McpTool {
        name: "fab_r2r_update".to_string(),
        description: "EWMA run-to-run recipe control (Sachs, Hu & Ingolfsson 1995): given the \
            target output, process gain, EWMA smoothing factor, the current drift estimate, and \
            the recipe/output of the last run, returns the updated drift estimate and the \
            recommended recipe for the next run."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "target": { "type": "number" },
                "gain": { "type": "number", "description": "process gain (b-hat), must be non-zero" },
                "lambda": { "type": "number", "description": "EWMA forgetting factor, (0, 1]" },
                "current_drift_estimate": { "type": "number", "description": "a-hat before this update (0.0 if none yet)" },
                "applied_recipe": { "type": "number", "description": "the recipe actually used for the observed run" },
                "measured_output": { "type": "number", "description": "the measured output of that run" },
            },
            "required": ["target", "gain", "lambda", "current_drift_estimate", "applied_recipe", "measured_output"],
        }),
        handler: Box::new(|args| {
            let target = get_f64(&args, "target")?;
            let gain = get_f64(&args, "gain")?;
            let lambda = get_f64(&args, "lambda")?;
            let current_drift_estimate = get_f64(&args, "current_drift_estimate")?;
            let applied_recipe = get_f64(&args, "applied_recipe")?;
            let measured_output = get_f64(&args, "measured_output")?;

            let mut controller =
                EwmaR2rController::new(target, gain, lambda, current_drift_estimate);
            let next_recipe = controller.update(applied_recipe, measured_output);

            Ok(json!({
                "drift_estimate": controller.drift_estimate(),
                "next_recipe": next_recipe,
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2r_update_tool_matches_the_worked_example() {
        let tool = r2r_update_tool();
        let result = (tool.handler)(json!({
            "target": 500.0,
            "gain": 5.0,
            "lambda": 0.3,
            "current_drift_estimate": 0.0,
            "applied_recipe": 100.0,
            "measured_output": 510.0,
        }))
        .unwrap();
        assert!((result["drift_estimate"].as_f64().unwrap() - 3.0).abs() < 1e-9);
        assert!((result["next_recipe"].as_f64().unwrap() - 99.4).abs() < 1e-9);
    }

    #[test]
    fn r2r_update_tool_rejects_missing_fields() {
        let tool = r2r_update_tool();
        let result = (tool.handler)(json!({ "target": 500.0 }));
        assert!(result.is_err());
    }
}
