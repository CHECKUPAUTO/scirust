//! Outil MCP pour `scirust_biomed::control::barrier` : le filtre de
//! sécurité par fonction barrière de contrôle (CBF-QP) — donne à un agent
//! superviseur la dose sûre la plus proche d'une dose désirée, sans
//! exposer l'algèbre du filtre.
//!
//! **Avertissement non-clinique** : voir `scirust-biomed`'s
//! `control::barrier` module doc — ceci démontre une technique de
//! contrôle certifiable, ce n'est pas un dispositif médical validé.

use crate::registry::McpTool;
use scirust_biomed::{GlucoseModel, cbf_safe_dose};
use serde_json::{Value, json};

fn get_f64(v: &Value, field: &str) -> Result<f64, String> {
    v.get(field)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("missing or non-numeric `{field}`"))
}

pub fn biomed_tools() -> Vec<McpTool> {
    vec![cbf_safe_dose_tool()]
}

fn cbf_safe_dose_tool() -> McpTool {
    McpTool {
        name: "biomed_cbf_safe_dose".to_string(),
        description: "Control-Barrier-Function safety filter (Ames et al., IEEE TAC 2017) for a \
            simplified affine glucose-dynamics model: given the desired insulin dose, returns the \
            closest dose that provably keeps the modeled glucose trajectory above a safety floor. \
            NOT a clinically validated dosing algorithm — see scirust-biomed::control::barrier's \
            module doc for the full non-clinical-use caveat."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "reversion_rate": { "type": "number", "description": "model parameter a (1/min)" },
                "basal_target": { "type": "number", "description": "model basal glucose target" },
                "insulin_sensitivity": { "type": "number", "description": "model parameter k, must be > 0" },
                "glucose": { "type": "number", "description": "current glucose reading" },
                "glucose_floor": { "type": "number", "description": "hypoglycemic safety floor" },
                "alpha": { "type": "number", "description": "class-K barrier gain, must be > 0" },
                "u_desired": { "type": "number", "description": "desired insulin infusion rate" },
                "u_max": { "type": "number", "description": "pump maximum infusion rate" },
            },
            "required": ["reversion_rate", "basal_target", "insulin_sensitivity", "glucose", "glucose_floor", "alpha", "u_desired", "u_max"],
        }),
        handler: Box::new(|args| {
            let model = GlucoseModel {
                reversion_rate: get_f64(&args, "reversion_rate")?,
                basal_target: get_f64(&args, "basal_target")?,
                insulin_sensitivity: get_f64(&args, "insulin_sensitivity")?,
            };
            let glucose = get_f64(&args, "glucose")?;
            let glucose_floor = get_f64(&args, "glucose_floor")?;
            let alpha = get_f64(&args, "alpha")?;
            let u_desired = get_f64(&args, "u_desired")?;
            let u_max = get_f64(&args, "u_max")?;

            let safe = cbf_safe_dose(model, glucose, glucose_floor, alpha, u_desired, u_max);

            Ok(json!({
                "units_per_hour": safe.units_per_hour,
                "constrained": safe.constrained,
                "barrier_violated_at_zero_dose": safe.barrier_violated_at_zero_dose,
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbf_safe_dose_tool_caps_an_aggressive_dose() {
        let tool = cbf_safe_dose_tool();
        let result = (tool.handler)(json!({
            "reversion_rate": 0.02,
            "basal_target": 100.0,
            "insulin_sensitivity": 3.0,
            "glucose": 180.0,
            "glucose_floor": 70.0,
            "alpha": 0.05,
            "u_desired": 4.0,
            "u_max": 10.0,
        }))
        .unwrap();
        assert!((result["units_per_hour"].as_f64().unwrap() - 1.3).abs() < 1e-9);
        assert_eq!(result["constrained"], json!(true));
    }

    #[test]
    fn cbf_safe_dose_tool_rejects_missing_fields() {
        let tool = cbf_safe_dose_tool();
        assert!((tool.handler)(json!({ "glucose": 100.0 })).is_err());
    }
}
