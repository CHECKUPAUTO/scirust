//! Outil MCP pour `scirust-grid::state_estimation` : estimation d'état de
//! réseau électrique par moindres carrés pondérés avec détection de
//! données aberrantes en un seul appel — utile à un agent de supervision
//! réseau qui veut à la fois l'état estimé et un diagnostic de confiance
//! sur les mesures.

use crate::registry::McpTool;
use scirust_grid::{
    GridError, chi_squared_test, largest_normalized_residual_test, wls_state_estimate,
};
use scirust_solvers::linalg::Matrix;
use serde_json::{Value, json};

fn parse_matrix(v: Option<&Value>, field: &str) -> Result<Matrix, String> {
    let rows = v
        .ok_or_else(|| format!("missing `{field}`"))?
        .as_array()
        .ok_or_else(|| format!("`{field}` must be a 2D array"))?;
    if rows.is_empty()
    {
        return Err(format!("`{field}` must be non-empty"));
    }
    let ncols = rows[0]
        .as_array()
        .ok_or_else(|| format!("`{field}` rows must themselves be arrays"))?
        .len();
    let mut data = Vec::with_capacity(rows.len() * ncols);
    for (i, row) in rows.iter().enumerate()
    {
        let row = row
            .as_array()
            .ok_or_else(|| format!("{field} row {i} is not an array"))?;
        if row.len() != ncols
        {
            return Err(format!(
                "{field} row {i} has a different column count (ragged matrix)"
            ));
        }
        for x in row
        {
            data.push(
                x.as_f64()
                    .ok_or_else(|| format!("{field} row {i} contains a non-numeric entry"))?,
            );
        }
    }
    Ok(Matrix::from_row_major(rows.len(), ncols, data))
}

fn parse_vector(v: Option<&Value>, field: &str) -> Result<Vec<f64>, String> {
    v.ok_or_else(|| format!("missing `{field}`"))?
        .as_array()
        .ok_or_else(|| format!("`{field}` must be an array"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| format!("`{field}` contains a non-numeric entry"))
        })
        .collect()
}

fn describe_error(e: GridError) -> String {
    e.to_string()
}

pub fn grid_tools() -> Vec<McpTool> {
    vec![state_estimate_tool()]
}

fn state_estimate_tool() -> McpTool {
    McpTool {
        name: "grid_state_estimate".to_string(),
        description: "Weighted-least-squares power-system state estimation (Abur & Expósito) \
            with bad-data detection: given the measurement Jacobian `h`, measurements `z`, and \
            per-measurement weights, returns the estimated state, residuals, and (if \
            `bad_data_threshold` is given) the largest-normalized-residual diagnosis identifying \
            the most suspect measurement."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "h": { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "measurement Jacobian, m rows x n state variables" },
                "z": { "type": "array", "items": { "type": "number" }, "description": "measurement vector, length m" },
                "weights": { "type": "array", "items": { "type": "number" }, "description": "per-measurement weights (1/sigma^2), length m" },
                "bad_data_threshold": { "type": "number", "description": "optional: normalized-residual threshold (e.g. 3.0) to flag a suspect measurement" },
            },
            "required": ["h", "z", "weights"],
        }),
        handler: Box::new(|args| {
            let h = parse_matrix(args.get("h"), "h")?;
            let z = parse_vector(args.get("z"), "z")?;
            let weights = parse_vector(args.get("weights"), "weights")?;

            let estimate = wls_state_estimate(&h, &z, &weights).map_err(describe_error)?;
            let mut out = json!({
                "x": estimate.x,
                "residuals": estimate.residuals,
                "objective": estimate.objective,
            });

            if let Some(threshold) = args.get("bad_data_threshold").and_then(|v| v.as_f64())
            {
                let report =
                    largest_normalized_residual_test(&h, &weights, &estimate.residuals, threshold)
                        .map_err(describe_error)?;
                out["bad_data"] = json!({
                    "normalized_residuals": report.normalized_residuals,
                    "suspect_index": report.suspect_index,
                    "chi_squared_flagged": chi_squared_test(estimate.objective, threshold * threshold),
                });
            }

            Ok(out)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_estimate_tool_matches_the_three_bus_worked_example() {
        let tool = state_estimate_tool();
        let result = (tool.handler)(json!({
            "h": [[-10.0, 0.0], [0.0, -10.0], [20.0, -10.0]],
            "z": [0.202, 0.348, -0.045],
            "weights": [1.0, 1.0, 1.0],
        }))
        .unwrap();
        let x = result["x"].as_array().unwrap();
        assert!((x[0].as_f64().unwrap() - (-0.019833333333)).abs() < 1e-9);
        assert!((x[1].as_f64().unwrap() - (-0.034983333333)).abs() < 1e-9);
        assert!(result.get("bad_data").is_none());
    }

    #[test]
    fn state_estimate_tool_reports_bad_data_when_threshold_given() {
        let tool = state_estimate_tool();
        let result = (tool.handler)(json!({
            "h": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
            "z": [3.0, 2.0, 10.0, 1.0], // measurement 2 corrupted (true value 5.0)
            "weights": [1.0, 1.0, 1.0, 1.0],
            "bad_data_threshold": 2.5,
        }))
        .unwrap();
        assert_eq!(result["bad_data"]["suspect_index"], json!(2));
    }

    #[test]
    fn state_estimate_tool_rejects_dimension_mismatch() {
        let tool = state_estimate_tool();
        let result = (tool.handler)(json!({
            "h": [[1.0, 0.0], [0.0, 1.0]],
            "z": [1.0],
            "weights": [1.0, 1.0],
        }));
        assert!(result.is_err());
    }
}
