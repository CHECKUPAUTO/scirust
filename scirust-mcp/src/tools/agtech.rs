//! Outil MCP pour `scirust-agtech` : pipeline complet de nettoyage de
//! carte de rendement (filtres global + local, puis interpolation IDW en
//! option) en un seul appel — l'usage naturel pour un agent qui reçoit
//! un fichier de rendement brut et doit produire une carte reproductible.

use crate::registry::McpTool;
use scirust_agtech::{YieldPoint, global_filter, idw_interpolate, local_filter};
use serde_json::{Value, json};

fn parse_points(v: Option<&Value>) -> Result<Vec<YieldPoint>, String> {
    v.ok_or("missing `points`")?
        .as_array()
        .ok_or("`points` must be an array")?
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let x = p
                .get("x")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("points[{i}].x missing"))?;
            let y = p
                .get("y")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("points[{i}].y missing"))?;
            let yield_value = p
                .get("yield")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("points[{i}].yield missing"))?;
            Ok(YieldPoint { x, y, yield_value })
        })
        .collect()
}

pub fn agtech_tools() -> Vec<McpTool> {
    vec![clean_yield_map_tool()]
}

fn clean_yield_map_tool() -> McpTool {
    McpTool {
        name: "agtech_clean_yield_map".to_string(),
        description: "Reproducible yield-map cleaning pipeline (Sudduth & Drummond 2007): applies \
            an explicit global outlier filter (k std devs from the field mean) and a local \
            outlier filter (k std devs from each point's k-nearest-neighbor mean) — a point \
            survives only if it passes both. Optionally interpolates the cleaned data at query \
            points by inverse-distance weighting (IDW)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": { "type": "number" },
                            "y": { "type": "number" },
                            "yield": { "type": "number" },
                        },
                        "required": ["x", "y", "yield"],
                    },
                },
                "global_k_std": { "type": "number", "description": "e.g. 3.0" },
                "local_k_neighbors": { "type": "integer", "description": "e.g. 8" },
                "local_k_std": { "type": "number", "description": "e.g. 2.0" },
                "query_points": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } },
                    "description": "optional [x, y] points to interpolate after cleaning",
                },
                "idw_power": { "type": "number", "description": "default 2.0" },
                "idw_k_neighbors": { "type": "integer", "description": "default 8" },
            },
            "required": ["points", "global_k_std", "local_k_neighbors", "local_k_std"],
        }),
        handler: Box::new(|args| {
            let points = parse_points(args.get("points"))?;
            let global_k_std = args
                .get("global_k_std")
                .and_then(|v| v.as_f64())
                .ok_or("missing `global_k_std`")?;
            let local_k_neighbors = args
                .get("local_k_neighbors")
                .and_then(|v| v.as_u64())
                .ok_or("missing `local_k_neighbors`")? as usize;
            let local_k_std = args
                .get("local_k_std")
                .and_then(|v| v.as_f64())
                .ok_or("missing `local_k_std`")?;

            let global_kept: std::collections::HashSet<usize> =
                global_filter(&points, global_k_std).into_iter().collect();
            let local_kept: std::collections::HashSet<usize> =
                local_filter(&points, local_k_neighbors, local_k_std)
                    .into_iter()
                    .collect();
            let mut kept: Vec<usize> = global_kept.intersection(&local_kept).copied().collect();
            kept.sort_unstable();
            let rejected: Vec<usize> = (0..points.len()).filter(|i| !kept.contains(i)).collect();

            let cleaned: Vec<YieldPoint> = kept.iter().map(|&i| points[i]).collect();

            let mut out = json!({
                "kept_indices": kept,
                "rejected_indices": rejected,
            });

            if let Some(queries) = args.get("query_points").and_then(|v| v.as_array())
            {
                let power = args
                    .get("idw_power")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(2.0);
                let k = args
                    .get("idw_k_neighbors")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8) as usize;
                let mut interpolated = Vec::with_capacity(queries.len());
                for (i, q) in queries.iter().enumerate()
                {
                    let arr = q
                        .as_array()
                        .ok_or_else(|| format!("query_points[{i}] must be [x, y]"))?;
                    if arr.len() != 2
                    {
                        return Err(format!("query_points[{i}] must have exactly 2 elements"));
                    }
                    let x = arr[0]
                        .as_f64()
                        .ok_or_else(|| format!("query_points[{i}][0] must be numeric"))?;
                    let y = arr[1]
                        .as_f64()
                        .ok_or_else(|| format!("query_points[{i}][1] must be numeric"))?;
                    let value = idw_interpolate(&cleaned, (x, y), power, k);
                    interpolated.push(json!({ "query": [x, y], "value": value }));
                }
                out["interpolated"] = Value::Array(interpolated);
            }

            Ok(out)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_yield_map_tool_rejects_the_gross_outlier() {
        let tool = clean_yield_map_tool();
        let result = (tool.handler)(json!({
            "points": [
                {"x": 0.0, "y": 0.0, "yield": 8.0},
                {"x": 1.0, "y": 0.0, "yield": 9.0},
                {"x": 2.0, "y": 0.0, "yield": 10.0},
                {"x": 3.0, "y": 0.0, "yield": 9.5},
                {"x": 4.0, "y": 0.0, "yield": 8.5},
                {"x": 5.0, "y": 0.0, "yield": 50.0},
            ],
            "global_k_std": 2.0,
            "local_k_neighbors": 3,
            "local_k_std": 2.0,
        }))
        .unwrap();
        let rejected = result["rejected_indices"].as_array().unwrap();
        assert!(rejected.contains(&json!(5)));
    }

    #[test]
    fn clean_yield_map_tool_interpolates_query_points() {
        // A smooth trend with enough points that the local filter (which
        // needs real neighborhood variance to compare against) doesn't
        // reject anything, then a query exactly on a known point to check
        // the interpolation plumbing without depending on the weighted-sum
        // arithmetic (already covered directly in scirust-agtech's own
        // idw tests).
        let tool = clean_yield_map_tool();
        let result = (tool.handler)(json!({
            "points": [
                {"x": 0.0, "y": 0.0, "yield": 10.0},
                {"x": 1.0, "y": 0.0, "yield": 12.0},
                {"x": 2.0, "y": 0.0, "yield": 14.0},
                {"x": 3.0, "y": 0.0, "yield": 16.0},
                {"x": 4.0, "y": 0.0, "yield": 18.0},
            ],
            "global_k_std": 3.0,
            "local_k_neighbors": 2,
            "local_k_std": 5.0,
            "query_points": [[2.0, 0.0]],
            "idw_power": 2.0,
            "idw_k_neighbors": 2,
        }))
        .unwrap();
        assert_eq!(result["rejected_indices"].as_array().unwrap().len(), 0);
        let interpolated = result["interpolated"].as_array().unwrap();
        let value = interpolated[0]["value"].as_f64().unwrap();
        assert!((value - 14.0).abs() < 1e-9, "value {value}");
    }
}
