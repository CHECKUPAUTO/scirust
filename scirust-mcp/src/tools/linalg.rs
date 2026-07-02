//! Outils MCP pour les nouvelles primitives d'algèbre linéaire de
//! `scirust-solvers` (décomposition en valeurs propres symétrique, SVD
//! générale, GMRES) — entrée/sortie JSON structurée plutôt que du texte,
//! pour qu'un agent puisse enchaîner les appels sans reparser une sortie
//! CLI.

use crate::registry::McpTool;
use scirust_solvers::Tolerance;
use scirust_solvers::linalg::{Matrix, eigen_symmetric, gmres, svd};
use serde_json::{Value, json};

fn parse_matrix(v: Option<&Value>) -> Result<Matrix, String> {
    let v = v.ok_or("missing `a`")?;
    let rows = v.as_array().ok_or("`a` must be a 2D array")?;
    if rows.is_empty()
    {
        return Err("`a` must be non-empty".to_string());
    }
    let ncols = rows[0]
        .as_array()
        .ok_or("`a` rows must themselves be arrays")?
        .len();
    let mut data = Vec::with_capacity(rows.len() * ncols);
    for (i, row) in rows.iter().enumerate()
    {
        let row = row
            .as_array()
            .ok_or_else(|| format!("row {i} is not an array"))?;
        if row.len() != ncols
        {
            return Err(format!(
                "row {i} has {} columns, expected {ncols} (ragged matrix)",
                row.len()
            ));
        }
        for x in row
        {
            data.push(
                x.as_f64()
                    .ok_or_else(|| format!("row {i} contains a non-numeric entry"))?,
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

fn matrix_to_json(m: &Matrix) -> Value {
    let (r, c) = m.shape();
    Value::Array(
        (0..r)
            .map(|i| Value::Array((0..c).map(|j| json!(m[(i, j)])).collect()))
            .collect(),
    )
}

pub fn linalg_tools() -> Vec<McpTool> {
    vec![eigen_tool(), svd_tool(), gmres_tool()]
}

fn eigen_tool() -> McpTool {
    McpTool {
        name: "linalg_eigen_symmetric".to_string(),
        description: "Symmetric dense eigenvalue decomposition (Householder tridiagonalization \
            + implicit QL with Wilkinson shift, Golub & Van Loan). Input: a square symmetric \
            matrix `a`. Returns eigenvalues (ascending) and orthonormal eigenvectors (columns)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } },
                    "description": "square symmetric matrix, row-major nested array",
                }
            },
            "required": ["a"],
        }),
        handler: Box::new(|args| {
            let a = parse_matrix(args.get("a"))?;
            let eig = eigen_symmetric(&a).map_err(|e| e.to_string())?;
            Ok(json!({
                "eigenvalues": eig.eigenvalues,
                "eigenvectors": matrix_to_json(&eig.eigenvectors),
            }))
        }),
    }
}

fn svd_tool() -> McpTool {
    McpTool {
        name: "linalg_svd".to_string(),
        description: "General dense SVD via one-sided Jacobi rotations (Hestenes). Input: any \
            (m, n) matrix `a`. Returns thin U, singular values `s` (descending), and V such \
            that a ≈ U · diag(s) · Vᵀ."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } },
                    "description": "(m, n) matrix, row-major nested array",
                }
            },
            "required": ["a"],
        }),
        handler: Box::new(|args| {
            let a = parse_matrix(args.get("a"))?;
            let s = svd(&a).map_err(|e| e.to_string())?;
            Ok(json!({
                "u": matrix_to_json(&s.u),
                "s": s.s,
                "v": matrix_to_json(&s.v),
            }))
        }),
    }
}

fn gmres_tool() -> McpTool {
    McpTool {
        name: "linalg_gmres".to_string(),
        description: "GMRES(m) restarted solver for A·x = b — works on nonsymmetric systems, \
            unlike conjugate gradient. Input: square matrix `a`, vector `b`, optional \
            `restart` (Krylov subspace size before restart; default min(n, 30)). Returns the \
            solution, iteration count, and final residual."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } },
                    "description": "square (n, n) matrix, row-major nested array",
                },
                "b": {
                    "type": "array",
                    "items": { "type": "number" },
                    "description": "right-hand side, length n",
                },
                "restart": { "type": "integer", "minimum": 1 },
            },
            "required": ["a", "b"],
        }),
        handler: Box::new(|args| {
            let a = parse_matrix(args.get("a"))?;
            let b = parse_vector(args.get("b"), "b")?;
            let n = b.len();
            if a.shape() != (n, n)
            {
                let (rows, cols) = a.shape();
                return Err(format!(
                    "`a` is {rows}x{cols} but must be {n}x{n} to match `b` (length {n})"
                ));
            }
            let restart = args
                .get("restart")
                .and_then(|v| v.as_u64())
                .map(|r| r as usize)
                .unwrap_or_else(|| n.clamp(1, 30));
            let sol = gmres(
                |x, y| y.copy_from_slice(&a.matvec(x).expect("dimensions already validated")),
                &b,
                vec![0.0; n],
                restart,
                Tolerance::default(),
            )
            .map_err(|e| e.to_string())?;
            Ok(json!({
                "x": sol.value,
                "iterations": sol.info.iterations,
                "residual": sol.info.residual,
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn eigen_tool_on_identity() {
        let tool = eigen_tool();
        let result = (tool.handler)(json!({ "a": [[1.0, 0.0], [0.0, 1.0]] })).unwrap();
        let evs = result["eigenvalues"].as_array().unwrap();
        assert_relative_eq!(evs[0].as_f64().unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(evs[1].as_f64().unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn eigen_tool_rejects_non_symmetric() {
        let tool = eigen_tool();
        let result = (tool.handler)(json!({ "a": [[1.0, 2.0], [0.0, 1.0]] }));
        assert!(result.is_err());
    }

    #[test]
    fn svd_tool_on_diagonal() {
        let tool = svd_tool();
        let result = (tool.handler)(json!({ "a": [[2.0, 0.0], [0.0, 5.0]] })).unwrap();
        let s = result["s"].as_array().unwrap();
        assert_relative_eq!(s[0].as_f64().unwrap(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(s[1].as_f64().unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn gmres_tool_solves_nonsymmetric_system() {
        let tool = gmres_tool();
        let result = (tool.handler)(json!({
            "a": [[4.0, 1.0], [2.0, 3.0]],
            "b": [6.0, 7.0],
        }))
        .unwrap();
        let x = result["x"].as_array().unwrap();
        // A·x = b ⇒ x = (1.1, 1.6)
        assert_relative_eq!(x[0].as_f64().unwrap(), 1.1, epsilon = 1e-6);
        assert_relative_eq!(x[1].as_f64().unwrap(), 1.6, epsilon = 1e-6);
    }

    #[test]
    fn gmres_tool_rejects_dimension_mismatch() {
        let tool = gmres_tool();
        let result = (tool.handler)(json!({ "a": [[1.0, 0.0], [0.0, 1.0]], "b": [1.0, 2.0, 3.0] }));
        assert!(result.is_err());
    }

    #[test]
    fn eigen_tool_rejects_missing_field() {
        let tool = eigen_tool();
        assert!((tool.handler)(json!({})).is_err());
    }
}
