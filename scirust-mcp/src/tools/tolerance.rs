//! Outils MCP pour `scirust-tolerance` : capabilité inertielle d'un lot et
//! répartition d'une chaîne de cotes (tolérancement inertiel de Pillet).
//! Donne directement à un agent l'inertie, les indices `Cp/Cpk/Cpm/Cpi`, le
//! taux de non-conformité, et la répartition des inerties composants — sans
//! réimplémenter les formules côté agent.

use crate::registry::McpTool;
use scirust_tolerance::capability::CapabilitySummary;
use scirust_tolerance::chain::{
    Allocation, Contributor, allocate, assembly_inertia_statistical, assembly_inertia_worst_case,
};
use scirust_tolerance::form::FormBatch;
use scirust_tolerance::inertia::{Inertia, InertiaCone, i_max_from_tolerance};
use scirust_tolerance::modal::{ModalBasis, modal_inertias};
use scirust_tolerance::sampling::design_plan;
use scirust_tolerance::spatial::{
    Feature, Torsor, inertia_decomposition, surface_inertia_from_torsors,
};
use serde_json::json;

pub fn tolerance_tools() -> Vec<McpTool> {
    vec![
        inertial_capability_tool(),
        chain_allocate_tool(),
        acceptance_plan_tool(),
        form_modal_tool(),
        torsor_3d_tool(),
    ]
}

/// Parse a JSON array of 3-element numeric arrays into `Vec<[f64; 3]>`.
fn vec3_array(args: &serde_json::Value, key: &str) -> Result<Vec<[f64; 3]>, String> {
    args.get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing `{key}`"))?
        .iter()
        .map(|row| {
            let a = row
                .as_array()
                .ok_or(format!("`{key}` must be an array of [x,y,z]"))?;
            if a.len() != 3
            {
                return Err(format!("`{key}` rows must have 3 numbers"));
            }
            Ok([
                a[0].as_f64().ok_or("non-numeric")?,
                a[1].as_f64().ok_or("non-numeric")?,
                a[2].as_f64().ok_or("non-numeric")?,
            ])
        })
        .collect()
}

fn f64_field(args: &serde_json::Value, key: &str) -> Result<f64, String> {
    args.get(key)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("missing or non-numeric `{key}`"))
}

fn f64_array(args: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    args.get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing `{key}`"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| format!("`{key}` has a non-numeric entry"))
        })
        .collect()
}

fn inertial_capability_tool() -> McpTool {
    McpTool {
        name: "tolerance_inertial_capability".to_string(),
        description: "Inertial tolerancing (Pillet) capability of a batch: given a measurement \
            sample, a target and a bilateral spec [lsl, usl], returns the inertia I=sqrt(delta^2 + \
            sigma^2) (RMS deviation from target), the classical indices Cp/Cpk/Cpm, the inertial \
            index Cpi=I_max/I (>=1 means inside the inertia cone), and the predicted non-conformity \
            in ppm. I_max defaults to the Cp=1 budget (usl-lsl)/6 unless overridden."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "sample": { "type": "array", "items": { "type": "number" }, "description": "measurements" },
                "target": { "type": "number" },
                "lsl": { "type": "number", "description": "lower spec limit" },
                "usl": { "type": "number", "description": "upper spec limit" },
                "i_max": { "type": "number", "description": "optional inertia budget; default (usl-lsl)/6" },
            },
            "required": ["sample", "target", "lsl", "usl"],
        }),
        handler: Box::new(|args| {
            let sample = f64_array(&args, "sample")?;
            if sample.is_empty()
            {
                return Err("`sample` must be non-empty".to_string());
            }
            let target = f64_field(&args, "target")?;
            let lsl = f64_field(&args, "lsl")?;
            let usl = f64_field(&args, "usl")?;
            if usl <= lsl
            {
                return Err("`usl` must be greater than `lsl`".to_string());
            }
            let i_max = args
                .get("i_max")
                .and_then(|v| v.as_f64())
                .unwrap_or_else(|| i_max_from_tolerance(usl - lsl, 1.0));

            let s = CapabilitySummary::from_sample(&sample, lsl, usl, target, i_max);
            let cone = InertiaCone::new(i_max);
            let inertia = Inertia::from_sample(&sample, target);
            Ok(json!({
                "mean": s.mean,
                "sigma": s.sigma,
                "off_centering": inertia.off_centering,
                "inertia": s.inertia,
                "i_max": i_max,
                "cp": s.cp,
                "cpk": s.cpk,
                "cpm": s.cpm,
                "cpi": s.cpi,
                "ppm": s.ppm,
                "inside_inertia_cone": cone.accepts(&inertia),
                "conforming": s.cpi >= 1.0,
            }))
        }),
    }
}

fn chain_allocate_tool() -> McpTool {
    McpTool {
        name: "tolerance_chain_allocate".to_string(),
        description: "Inertial tolerance-chain allocation: distribute a linear assembly's tolerance \
            interval R_Y over its component influence coefficients (alpha_i, +/-1 for a plain stack) \
            as per-component inertia budgets I_i. Methods: `worst_case` (I_Y/sum|alpha|), \
            `statistical` (I_Y/sqrt(sum alpha^2)), `guaranteed_cpk` (statistical tightened by \
            ICC=sqrt(cpk^2+n/9) to guarantee a Cpk on the resultant). The assembly inertia budget is \
            I_Y=R_Y/6. Returns per-component I_i and the recombined assembly inertia (statistical & \
            worst-case) as a check."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "assembly_tolerance": { "type": "number", "description": "R_Y, the functional-requirement tolerance interval" },
                "coefficients": { "type": "array", "items": { "type": "number" }, "description": "influence coefficients alpha_i" },
                "method": { "type": "string", "enum": ["worst_case", "statistical", "guaranteed_cpk"] },
                "cpk": { "type": "number", "description": "target Cpk for `guaranteed_cpk` (default 1.0)" },
            },
            "required": ["assembly_tolerance", "coefficients", "method"],
        }),
        handler: Box::new(|args| {
            let r_y = f64_field(&args, "assembly_tolerance")?;
            let coeffs = f64_array(&args, "coefficients")?;
            if coeffs.is_empty()
            {
                return Err("`coefficients` must be non-empty".to_string());
            }
            let method_name = args
                .get("method")
                .and_then(|v| v.as_str())
                .ok_or("missing `method`")?;
            let method = match method_name
            {
                "worst_case" => Allocation::WorstCase,
                "statistical" => Allocation::Statistical,
                "guaranteed_cpk" =>
                {
                    let cpk = args.get("cpk").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    Allocation::GuaranteedCpk(cpk)
                },
                other => return Err(format!("unknown method `{other}`")),
            };
            let i_y = i_max_from_tolerance(r_y, 1.0);
            let budgets = allocate(i_y, &coeffs, &method).map_err(|e| e.to_string())?;

            let contributors: Vec<Contributor> = coeffs
                .iter()
                .zip(&budgets)
                .enumerate()
                .map(|(i, (a, b))| Contributor::new(format!("X{}", i + 1), *a, *b))
                .collect();

            Ok(json!({
                "assembly_inertia_budget": i_y,
                "component_inertias": budgets,
                "recombined_statistical": assembly_inertia_statistical(&contributors),
                "recombined_worst_case": assembly_inertia_worst_case(&contributors),
            }))
        }),
    }
}

fn acceptance_plan_tool() -> McpTool {
    McpTool {
        name: "tolerance_acceptance_plan".to_string(),
        description: "Design an inertial acceptance-sampling plan (Pillet & Maire): find the \
            smallest sample size n and acceptance factor k (accept the batch if the sampled inertia \
            Î <= k*I_max) that accepts a good batch at I_max with probability >= 1-alpha (producer \
            risk) and accepts a bad batch at ratio_bad*I_max with probability <= beta (consumer \
            risk). Both evaluated at the fully-dispersed worst-case split via the non-central \
            chi-square law. Returns n, k, and the operating-characteristic probabilities at the \
            good and bad inertia."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "alpha": { "type": "number", "description": "producer risk (e.g. 0.05)" },
                "beta": { "type": "number", "description": "consumer risk (e.g. 0.10)" },
                "ratio_bad": { "type": "number", "description": "bad-batch inertia as a multiple of I_max, > 1" },
                "max_n": { "type": "integer", "description": "largest sample size to try (default 500)" },
            },
            "required": ["alpha", "beta", "ratio_bad"],
        }),
        handler: Box::new(|args| {
            let alpha = f64_field(&args, "alpha")?;
            let beta = f64_field(&args, "beta")?;
            let ratio_bad = f64_field(&args, "ratio_bad")?;
            if !(alpha > 0.0 && alpha < 1.0) || !(beta > 0.0 && beta < 1.0)
            {
                return Err("`alpha` and `beta` must lie in (0, 1)".to_string());
            }
            if ratio_bad <= 1.0
            {
                return Err("`ratio_bad` must be greater than 1".to_string());
            }
            let max_n = args
                .get("max_n")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(500);
            match design_plan(alpha, beta, ratio_bad, max_n)
            {
                None => Ok(json!({
                    "feasible": false,
                    "reason": "no plan within max_n meets both risks",
                })),
                Some(plan) =>
                {
                    // Evaluate the OC at the good (I_max) and bad (ratio_bad*I_max)
                    // inertia; the result is scale-free, so use I_max = 1.
                    let good = plan.probability_of_acceptance_at(1.0, 1.0, 0.0);
                    let bad = plan.probability_of_acceptance_at(1.0, ratio_bad, 0.0);
                    Ok(json!({
                        "feasible": true,
                        "n": plan.n,
                        "factor_k": plan.factor,
                        "p_accept_good": good,
                        "p_accept_bad": bad,
                    }))
                },
            }
        }),
    }
}

fn form_modal_tool() -> McpTool {
    McpTool {
        name: "tolerance_form_modal".to_string(),
        description: "Surface / form inertial tolerancing with modal decomposition (Adragna, \
            Pillet, Samper). Given a batch of surface measurements (rows = parts, columns = points \
            measured against nominal 0), returns the surface inertia I_S (RMS of every deviation \
            from nominal), the worst point, and — via an orthonormal DCT modal basis — the \
            per-mode inertias I_k, which for the complete basis (num_modes = all points, the default) \
            partition the surface inertia (sum I_k^2 = m*I_S^2); with fewer modes the sum is smaller. \
            Low modes are physical: mode 0 = size/mean offset, 1 = tilt, 2 = ovality/curvature, etc."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "measurements": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } },
                    "description": "rows = parts, columns = point deviations from nominal",
                },
                "num_modes": { "type": "integer", "description": "modes to report (default = all points)" },
            },
            "required": ["measurements"],
        }),
        handler: Box::new(|args| {
            let rows = args
                .get("measurements")
                .and_then(|v| v.as_array())
                .ok_or("missing `measurements`")?;
            let parts: Vec<Vec<f64>> = rows
                .iter()
                .map(|r| {
                    r.as_array()
                        .ok_or("`measurements` must be an array of arrays".to_string())?
                        .iter()
                        .map(|x| x.as_f64().ok_or("non-numeric measurement".to_string()))
                        .collect::<Result<Vec<f64>, String>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let batch = FormBatch::new(parts).ok_or("empty or ragged `measurements`")?;
            let m = batch.points();
            let k = args
                .get("num_modes")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(m);
            let basis = ModalBasis::dct(m, k);
            let modal = modal_inertias(&basis, batch.deviations());
            let i_s = batch.surface_inertia();
            let (worst_idx, worst) = batch.worst_point().ok_or("no points")?;

            Ok(json!({
                "surface_inertia": i_s,
                "worst_point": { "index": worst_idx, "inertia": worst.value() },
                "modal_inertias": modal.iter().enumerate().map(|(mode, i)| json!({
                    "mode": mode,
                    "inertia": i.value(),
                    "off_centering": i.off_centering,
                })).collect::<Vec<_>>(),
                // Partition check (exact only for a complete basis k = m).
                "modal_energy_sum": modal.iter().map(|i| i.mean_squared_deviation()).sum::<f64>(),
                "surface_energy": m as f64 * i_s * i_s,
            }))
        }),
    }
}

fn torsor_3d_tool() -> McpTool {
    McpTool {
        name: "tolerance_3d_surface_inertia".to_string(),
        description: "3D inertial tolerancing by small-displacement torsors (Adragna, Samper, \
            Pillet). Given a nominal feature sampled as points (positions OM relative to the working \
            origin) with outward unit normals, and a batch of per-part torsors (translation T + \
            small rotation R), returns the surface inertia I_S — the RMS normal deviation e=T·n+R·(OM×n) \
            over all points and parts — and its split into location (translation), orientation \
            (rotation), and coupling contributions to I_S² (the statistical combination of location \
            and orientation)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "points":      { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "OM positions [x,y,z] per sample point" },
                "normals":     { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "outward unit normals [x,y,z] per sample point" },
                "translations":{ "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "per-part translation T [x,y,z]" },
                "rotations":   { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "per-part small rotation R [x,y,z]" },
            },
            "required": ["points", "normals", "translations", "rotations"],
        }),
        handler: Box::new(|args| {
            let points = vec3_array(&args, "points")?;
            let normals = vec3_array(&args, "normals")?;
            let translations = vec3_array(&args, "translations")?;
            let rotations = vec3_array(&args, "rotations")?;
            if points.len() != normals.len() || points.is_empty()
            {
                return Err("`points` and `normals` must be non-empty and equal length".to_string());
            }
            if translations.len() != rotations.len() || translations.is_empty()
            {
                return Err(
                    "`translations` and `rotations` must be non-empty and equal length".to_string(),
                );
            }
            let feature = Feature::new(points.into_iter().zip(normals).collect());
            let torsors: Vec<Torsor> = translations
                .into_iter()
                .zip(rotations)
                .map(|(t, r)| Torsor::new(t, r))
                .collect();
            let i_s = surface_inertia_from_torsors(&feature, &torsors);
            let d = inertia_decomposition(&feature, &torsors);
            Ok(json!({
                "surface_inertia": i_s,
                "decomposition_of_i_s_squared": {
                    "location": d.location,
                    "orientation": d.orientation,
                    "coupling": d.coupling,
                    "total": d.total(),
                },
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inertial_capability_tool_reports_conforming_batch() {
        let tool = inertial_capability_tool();
        let out = (tool.handler)(json!({
            "sample": [10.0, 10.02, 9.98, 10.01, 9.99],
            "target": 10.0,
            "lsl": 9.4,
            "usl": 10.6,
        }))
        .unwrap();
        assert!(out["cpi"].as_f64().unwrap() > 1.0);
        assert_eq!(out["conforming"], json!(true));
        assert!(out["ppm"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn chain_allocate_tool_matches_paper_table2() {
        let tool = chain_allocate_tool();
        let out = (tool.handler)(json!({
            "assembly_tolerance": 1.0,
            "coefficients": [1.0, -1.0, -1.0, -1.0, -1.0],
            "method": "statistical",
        }))
        .unwrap();
        let i0 = out["component_inertias"].as_array().unwrap()[0]
            .as_f64()
            .unwrap();
        assert!((i0 - 0.0745).abs() < 1e-3);
        // Statistical allocation recombines back to the R_Y/6 budget.
        assert!((out["recombined_statistical"].as_f64().unwrap() - 1.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn chain_allocate_tool_rejects_unknown_method() {
        let tool = chain_allocate_tool();
        assert!(
            (tool.handler)(json!({
                "assembly_tolerance": 1.0,
                "coefficients": [1.0, -1.0],
                "method": "bogus",
            }))
            .is_err()
        );
    }

    #[test]
    fn acceptance_plan_tool_designs_a_feasible_plan() {
        let tool = acceptance_plan_tool();
        let out = (tool.handler)(json!({
            "alpha": 0.05,
            "beta": 0.10,
            "ratio_bad": 2.0,
        }))
        .unwrap();
        assert_eq!(out["feasible"], json!(true));
        assert!(out["n"].as_u64().unwrap() >= 2);
        assert!(out["p_accept_good"].as_f64().unwrap() >= 0.94);
        assert!(out["p_accept_bad"].as_f64().unwrap() <= 0.11);
    }

    #[test]
    fn acceptance_plan_tool_rejects_bad_ratio() {
        let tool = acceptance_plan_tool();
        assert!((tool.handler)(json!({ "alpha": 0.05, "beta": 0.1, "ratio_bad": 0.9 })).is_err());
    }

    #[test]
    fn form_modal_tool_partitions_surface_inertia() {
        let tool = form_modal_tool();
        let out = (tool.handler)(json!({
            "measurements": [
                [0.10, -0.05, 0.20, 0.00],
                [-0.10, 0.05, 0.10, 0.10],
                [0.00, 0.15, -0.10, 0.05],
            ],
        }))
        .unwrap();
        // Complete DCT basis ⇒ modal energy sum equals m·I_S².
        let esum = out["modal_energy_sum"].as_f64().unwrap();
        let etot = out["surface_energy"].as_f64().unwrap();
        assert!((esum - etot).abs() < 1e-9);
        assert_eq!(out["modal_inertias"].as_array().unwrap().len(), 4);
        assert!(out["surface_inertia"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn form_modal_tool_rejects_ragged_input() {
        let tool = form_modal_tool();
        assert!((tool.handler)(json!({ "measurements": [[0.1, 0.2], [0.1]] })).is_err());
    }

    #[test]
    fn torsor_3d_tool_reports_inertia_and_decomposition() {
        let tool = torsor_3d_tool();
        let out = (tool.handler)(json!({
            "points":  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "normals": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translations": [[0.02, -0.01, 0.03], [-0.01, 0.0, 0.01]],
            "rotations":    [[0.01, 0.0, -0.005], [0.0, 0.005, 0.0]],
        }))
        .unwrap();
        assert!(out["surface_inertia"].as_f64().unwrap() > 0.0);
        let d = &out["decomposition_of_i_s_squared"];
        let total = d["total"].as_f64().unwrap();
        let sum = d["location"].as_f64().unwrap()
            + d["orientation"].as_f64().unwrap()
            + d["coupling"].as_f64().unwrap();
        assert!((total - sum).abs() < 1e-12);
    }

    #[test]
    fn torsor_3d_tool_rejects_length_mismatch() {
        let tool = torsor_3d_tool();
        assert!(
            (tool.handler)(json!({
                "points": [[1.0, 0.0, 0.0]],
                "normals": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "translations": [[0.0, 0.0, 0.0]],
                "rotations": [[0.0, 0.0, 0.0]],
            }))
            .is_err()
        );
    }
}
