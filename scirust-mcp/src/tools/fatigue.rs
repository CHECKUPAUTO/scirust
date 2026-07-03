//! Outil MCP pour `scirust-fatigue` : comptage rainflow (ASTM E1049) et
//! dommage cumulé de Miner en un seul appel — donne directement à un
//! agent le dommage prédit pour un historique de charge et une courbe
//! S-N, sans exposer les cycles intermédiaires si ce n'est pas nécessaire.

use crate::registry::McpTool;
use scirust_fatigue::{PowerLawSnCurve, aggregate_by_range, count_cycles, miner_damage};
use serde_json::json;

pub fn fatigue_tools() -> Vec<McpTool> {
    vec![rainflow_damage_tool()]
}

fn rainflow_damage_tool() -> McpTool {
    McpTool {
        name: "fatigue_rainflow_damage".to_string(),
        description: "ASTM E1049 rainflow cycle counting followed by Palmgren-Miner cumulative \
            damage: given a raw load history and a power-law S-N curve (N = coefficient * \
            range^-exponent, caller-supplied — no material curve is assumed), returns the counted \
            cycles (range, mean, count) and the total damage D = sum(n_i/N_i). D >= 1.0 predicts \
            failure."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "series": { "type": "array", "items": { "type": "number" }, "description": "raw load history" },
                "sn_coefficient": { "type": "number" },
                "sn_exponent": { "type": "number" },
            },
            "required": ["series", "sn_coefficient", "sn_exponent"],
        }),
        handler: Box::new(|args| {
            let series: Vec<f64> = args
                .get("series")
                .and_then(|v| v.as_array())
                .ok_or("missing `series`")?
                .iter()
                .map(|x| x.as_f64().ok_or("`series` contains a non-numeric entry"))
                .collect::<Result<_, _>>()?;
            let coefficient = args
                .get("sn_coefficient")
                .and_then(|v| v.as_f64())
                .ok_or("missing `sn_coefficient`")?;
            let exponent = args
                .get("sn_exponent")
                .and_then(|v| v.as_f64())
                .ok_or("missing `sn_exponent`")?;

            let cycles = count_cycles(&series);
            let aggregated = aggregate_by_range(&cycles);
            let curve = PowerLawSnCurve {
                coefficient,
                exponent,
            };
            let damage = miner_damage(&aggregated, &curve);

            Ok(json!({
                "cycles": cycles.iter().map(|c| json!({
                    "range": c.range,
                    "mean": c.mean,
                    "count": c.count,
                    "start_index": c.start_index,
                    "end_index": c.end_index,
                })).collect::<Vec<_>>(),
                "damage": damage,
                "failure_predicted": damage >= 1.0,
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rainflow_damage_tool_matches_the_worked_example() {
        let tool = rainflow_damage_tool();
        let result = (tool.handler)(json!({
            "series": [2.0, -1.0, 5.0, -2.0, 3.0, -3.0, 4.0, -1.0],
            "sn_coefficient": 1.0e6,
            "sn_exponent": 3.0,
        }))
        .unwrap();
        assert_eq!(result["cycles"].as_array().unwrap().len(), 6);
        let damage = result["damage"].as_f64().unwrap();
        assert!(damage > 0.0);
        assert_eq!(result["failure_predicted"], json!(false));
    }

    #[test]
    fn rainflow_damage_tool_rejects_missing_fields() {
        let tool = rainflow_damage_tool();
        assert!((tool.handler)(json!({ "series": [1.0, 2.0] })).is_err());
    }
}
