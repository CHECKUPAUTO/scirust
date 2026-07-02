//! MCP tools for `scirust-sis`: IEC 61511 Safety Instrumented Function (SIF)
//! loop verification and proof-test interval sizing — structured JSON in
//! and out, so an agent can iterate on a design (try another architecture,
//! shorten a proof-test interval) without reparsing text.

use crate::registry::McpTool;
use scirust_sis::{Architecture, SifLoop, Subsystem};
use serde_json::{Value, json};

fn parse_architecture(v: &Value) -> Result<Architecture, String> {
    let m = v
        .get("m")
        .and_then(|x| x.as_u64())
        .ok_or("subsystem.architecture: missing `m`")? as u8;
    let n = v
        .get("n")
        .and_then(|x| x.as_u64())
        .ok_or("subsystem.architecture: missing `n`")? as u8;
    Architecture::new(m, n).map_err(|e| e.to_string())
}

fn architecture_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "m": { "type": "integer", "minimum": 1, "description": "votes required to trip" },
            "n": { "type": "integer", "minimum": 1, "description": "total channels" },
        },
        "required": ["m", "n"],
        "description": "MooN voting architecture; only 1oo1, 1oo2, 2oo2, 2oo3, 1oo3 have a known PFDavg formula",
    })
}

pub fn sis_tools() -> Vec<McpTool> {
    vec![sif_loop_tool(), proof_test_interval_tool()]
}

fn sif_loop_tool() -> McpTool {
    McpTool {
        name: "sis_verify_sif_loop".to_string(),
        description: "Verify a Safety Instrumented Function (SIF) loop against IEC 61511/61508: \
            given its sensor/logic-solver/final-element subsystems (each with a MooN voting \
            architecture, dangerous-undetected failure rate, common-cause beta, and proof-test \
            interval), returns the total PFDavg (sum across subsystems, standard \
            ISA-TR84.00.02 practice), the achieved SIL band, and a per-subsystem breakdown \
            showing which subsystem dominates."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "subsystems": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "architecture": architecture_schema(),
                            "lambda_du": { "type": "number", "description": "dangerous-undetected failure rate, per hour" },
                            "beta": { "type": "number", "description": "common-cause fraction, 0.0-1.0" },
                            "t1": { "type": "number", "description": "proof-test interval, hours" },
                        },
                        "required": ["name", "architecture", "lambda_du", "beta", "t1"],
                    },
                },
            },
            "required": ["subsystems"],
        }),
        handler: Box::new(|args| {
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("SIF loop")
                .to_string();
            let subsystems_json = args
                .get("subsystems")
                .and_then(|v| v.as_array())
                .ok_or("missing `subsystems` array")?;
            if subsystems_json.is_empty()
            {
                return Err("`subsystems` must contain at least one subsystem".to_string());
            }

            let mut sif = SifLoop::new(name);
            for (i, s) in subsystems_json.iter().enumerate()
            {
                let sub_name = s
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("subsystems[{i}]: missing `name`"))?;
                let architecture = parse_architecture(
                    s.get("architecture")
                        .ok_or_else(|| format!("subsystems[{i}]: missing `architecture`"))?,
                )
                .map_err(|e| format!("subsystems[{i}]: {e}"))?;
                let lambda_du = s
                    .get("lambda_du")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| format!("subsystems[{i}]: missing `lambda_du`"))?;
                let beta = s
                    .get("beta")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| format!("subsystems[{i}]: missing `beta`"))?;
                let t1 = s
                    .get("t1")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| format!("subsystems[{i}]: missing `t1`"))?;
                sif.add_subsystem(Subsystem::new(sub_name, architecture, lambda_du, beta, t1));
            }

            let breakdown = sif.breakdown().map_err(|e| e.to_string())?;
            let total = sif.total_pfd_avg().map_err(|e| e.to_string())?;
            let sil = sif.achieved_sil().map_err(|e| e.to_string())?;
            Ok(json!({
                "name": sif.name,
                "total_pfd_avg": total,
                "achieved_sil": format!("{sil:?}"),
                "breakdown": breakdown.into_iter().map(|(n, pfd)| json!({"name": n, "pfd_avg": pfd})).collect::<Vec<_>>(),
            }))
        }),
    }
}

fn proof_test_interval_tool() -> McpTool {
    McpTool {
        name: "sis_size_proof_test_interval".to_string(),
        description: "Size the longest IEC 61511 proof-test interval (hours) that still meets a \
            target PFDavg for a given MooN architecture, dangerous-undetected failure rate, and \
            common-cause beta — the inverse of computing PFDavg from a known interval. Useful for \
            answering 'how rarely can we proof-test this loop and still claim SIL2?'."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "architecture": architecture_schema(),
                "lambda_du": { "type": "number" },
                "beta": { "type": "number" },
                "target_pfd_avg": { "type": "number", "description": "e.g. 1e-3 for the SIL2 ceiling" },
            },
            "required": ["architecture", "lambda_du", "beta", "target_pfd_avg"],
        }),
        handler: Box::new(|args| {
            let architecture =
                parse_architecture(args.get("architecture").ok_or("missing `architecture`")?)?;
            let lambda_du = args
                .get("lambda_du")
                .and_then(|v| v.as_f64())
                .ok_or("missing `lambda_du`")?;
            let beta = args
                .get("beta")
                .and_then(|v| v.as_f64())
                .ok_or("missing `beta`")?;
            let target = args
                .get("target_pfd_avg")
                .and_then(|v| v.as_f64())
                .ok_or("missing `target_pfd_avg`")?;

            let t1 = scirust_sis::max_proof_test_interval(architecture, lambda_du, beta, target)
                .map_err(|e| e.to_string())?;
            let achieved_pfd = architecture
                .pfd_avg(lambda_du, t1, beta)
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "max_proof_test_interval_hours": t1,
                "achieved_pfd_avg": achieved_pfd,
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sif_loop_tool_computes_total_and_sil() {
        let tool = sif_loop_tool();
        let result = (tool.handler)(json!({
            "name": "Test SIF",
            "subsystems": [
                { "name": "Sensors", "architecture": {"m": 2, "n": 3}, "lambda_du": 5e-7, "beta": 0.02, "t1": 8760.0 },
                { "name": "Logic", "architecture": {"m": 1, "n": 1}, "lambda_du": 1e-7, "beta": 0.0, "t1": 8760.0 },
            ],
        }))
        .unwrap();
        assert_eq!(result["breakdown"].as_array().unwrap().len(), 2);
        assert!(result["total_pfd_avg"].as_f64().unwrap() > 0.0);
        assert!(result["achieved_sil"].as_str().unwrap().starts_with("Sil"));
    }

    #[test]
    fn sif_loop_tool_rejects_empty_subsystems() {
        let tool = sif_loop_tool();
        let result = (tool.handler)(json!({ "subsystems": [] }));
        assert!(result.is_err());
    }

    #[test]
    fn proof_test_interval_tool_roundtrips() {
        let tool = proof_test_interval_tool();
        let result = (tool.handler)(json!({
            "architecture": {"m": 1, "n": 2},
            "lambda_du": 1e-3,
            "beta": 0.1,
            "target_pfd_avg": 0.32,
        }))
        .unwrap();
        let t1 = result["max_proof_test_interval_hours"].as_f64().unwrap();
        assert!((t1 - 1000.0).abs() < 1e-3, "t1 {t1}");
    }

    #[test]
    fn proof_test_interval_tool_rejects_unsupported_architecture() {
        let tool = proof_test_interval_tool();
        let result = (tool.handler)(json!({
            "architecture": {"m": 2, "n": 4},
            "lambda_du": 1e-3,
            "beta": 0.1,
            "target_pfd_avg": 1e-3,
        }));
        assert!(result.is_err());
    }
}
