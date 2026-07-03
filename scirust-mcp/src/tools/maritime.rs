//! Outil MCP pour `scirust-maritime` : évaluation combinée du risque de
//! collision (CPA/TCPA) et classification COLREG de la rencontre, en un
//! seul appel — le cas d'usage naturel d'un agent de surveillance
//! maritime qui doit décider "est-ce que ce contact est dangereux, et
//! quelle règle s'applique ?" d'un coup.

use crate::registry::McpTool;
use scirust_maritime::{
    classify_encounter, cpa_tcpa, is_collision_risk, relative_bearing_deg, velocity_from_heading,
};
use serde_json::{Value, json};

fn parse_xy(v: Option<&Value>, field: &str) -> Result<(f64, f64), String> {
    let arr = v
        .ok_or_else(|| format!("missing `{field}`"))?
        .as_array()
        .ok_or_else(|| format!("`{field}` must be a [x, y] array"))?;
    if arr.len() != 2
    {
        return Err(format!("`{field}` must have exactly 2 elements"));
    }
    let x = arr[0]
        .as_f64()
        .ok_or_else(|| format!("`{field}[0]` must be numeric"))?;
    let y = arr[1]
        .as_f64()
        .ok_or_else(|| format!("`{field}[1]` must be numeric"))?;
    Ok((x, y))
}

fn parse_f64(v: &Value, obj_field: &str, field: &str) -> Result<f64, String> {
    v.get(field)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| format!("`{obj_field}.{field}` missing or not numeric"))
}

pub fn maritime_tools() -> Vec<McpTool> {
    vec![collision_risk_tool()]
}

fn collision_risk_tool() -> McpTool {
    McpTool {
        name: "maritime_collision_risk".to_string(),
        description: "Assess collision risk between own ship and a target track: computes CPA \
            (closest point of approach) and TCPA (time to CPA) from position/heading/speed, \
            classifies the COLREG encounter type (head-on/crossing/overtaking) from the relative \
            bearing, and flags whether the situation meets caller-supplied risk thresholds."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "own": {
                    "type": "object",
                    "properties": {
                        "position": { "type": "array", "items": { "type": "number" }, "description": "[east, north], nm" },
                        "heading_deg": { "type": "number", "description": "compass heading, 0=north, clockwise" },
                        "speed": { "type": "number", "description": "knots" },
                    },
                    "required": ["position", "heading_deg", "speed"],
                },
                "target": {
                    "type": "object",
                    "properties": {
                        "position": { "type": "array", "items": { "type": "number" } },
                        "heading_deg": { "type": "number" },
                        "speed": { "type": "number" },
                    },
                    "required": ["position", "heading_deg", "speed"],
                },
                "cpa_threshold": { "type": "number", "description": "nm; risk flagged if CPA is within this" },
                "tcpa_max": { "type": "number", "description": "hours; risk flagged only if TCPA is within this" },
            },
            "required": ["own", "target", "cpa_threshold", "tcpa_max"],
        }),
        handler: Box::new(|args| {
            let own = args.get("own").ok_or("missing `own`")?;
            let target = args.get("target").ok_or("missing `target`")?;

            let own_pos = parse_xy(own.get("position"), "own.position")?;
            let own_heading = parse_f64(own, "own", "heading_deg")?;
            let own_speed = parse_f64(own, "own", "speed")?;
            let target_pos = parse_xy(target.get("position"), "target.position")?;
            let target_heading = parse_f64(target, "target", "heading_deg")?;
            let target_speed = parse_f64(target, "target", "speed")?;

            let cpa_threshold = args
                .get("cpa_threshold")
                .and_then(|v| v.as_f64())
                .ok_or("missing `cpa_threshold`")?;
            let tcpa_max = args
                .get("tcpa_max")
                .and_then(|v| v.as_f64())
                .ok_or("missing `tcpa_max`")?;

            let own_vel = velocity_from_heading(own_heading, own_speed);
            let target_vel = velocity_from_heading(target_heading, target_speed);
            let result = cpa_tcpa(own_pos, own_vel, target_pos, target_vel);

            let bearing = relative_bearing_deg(own_pos, own_heading, target_pos);
            let encounter = classify_encounter(bearing);

            Ok(json!({
                "cpa_nm": result.cpa,
                "tcpa_hours": result.tcpa,
                "relative_bearing_deg": bearing,
                "encounter_type": format!("{encounter:?}"),
                "is_collision_risk": is_collision_risk(result, cpa_threshold, tcpa_max),
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_risk_tool_matches_the_two_ship_worked_example() {
        let tool = collision_risk_tool();
        let result = (tool.handler)(json!({
            "own": { "position": [1.0, 1.0], "heading_deg": 45.0, "speed": 6.0 },
            "target": { "position": [9.0, 8.0], "heading_deg": 270.0, "speed": 6.0 },
            "cpa_threshold": 5.0,
            "tcpa_max": 2.0,
        }))
        .unwrap();
        let cpa = result["cpa_nm"].as_f64().unwrap();
        let tcpa_min = result["tcpa_hours"].as_f64().unwrap() * 60.0;
        assert!((cpa - 3.4057).abs() < 1e-3, "cpa {cpa}");
        assert!((tcpa_min - 54.497).abs() < 1e-2, "tcpa_min {tcpa_min}");
        assert_eq!(result["is_collision_risk"], json!(true));
    }

    #[test]
    fn collision_risk_tool_rejects_missing_fields() {
        let tool = collision_risk_tool();
        let result = (tool.handler)(json!({
            "own": { "position": [0.0, 0.0], "heading_deg": 0.0, "speed": 5.0 },
        }));
        assert!(result.is_err());
    }
}
