//! Outils MCP pour `scirust-tolerance` : capabilité inertielle d'un lot et
//! répartition d'une chaîne de cotes (tolérancement inertiel de Pillet).
//! Donne directement à un agent l'inertie, les indices `Cp/Cpk/Cpm/Cpi`, le
//! taux de non-conformité, et la répartition des inerties composants — sans
//! réimplémenter les formules côté agent.

use crate::registry::McpTool;
use scirust_tolerance::attribution::attribute;
use scirust_tolerance::capability::{
    CapabilitySummary, cp_confidence_interval, cpk_confidence_interval,
};
use scirust_tolerance::chain::{
    Allocation, Contributor, ContributorState, allocate, assembly_inertia_statistical,
    assembly_inertia_worst_case,
};
use scirust_tolerance::correlated::{correlated_inertia, uniform_correlation};
use scirust_tolerance::distfit::{best_fit, percentile_capability};
use scirust_tolerance::drift::{cpk_to_ppk, long_term_ppm, long_term_sigma};
use scirust_tolerance::form::FormBatch;
use scirust_tolerance::geometry::{
    cylindricity, cylindricity_inertia, flatness, flatness_inertia, roundness, roundness_inertia,
    straightness, straightness_inertia,
};
use scirust_tolerance::inertia::{Inertia, InertiaCone, i_max_from_tolerance};
use scirust_tolerance::interval::tolerance_interval;
use scirust_tolerance::modal::{ModalBasis, modal_inertias};
use scirust_tolerance::montecarlo::{Distribution, linear, simulate};
use scirust_tolerance::msa::gage_rr;
use scirust_tolerance::nonnormal::{clements_capability, nonnormal_ppm};
use scirust_tolerance::optimize::{Component, Requirement, optimize};
use scirust_tolerance::position::{
    CompositePosition, FeatureType, datum_shift, positional_inertia, resultant_condition,
    total_position_tolerance, true_position, virtual_condition,
};
use scirust_tolerance::process::{Combination, ProcessOption, allocate_discrete};
use scirust_tolerance::sampling::design_plan;
use scirust_tolerance::sensitivity::{contributions, dual_contributions};
use scirust_tolerance::sixsigma::{dpmo, dpmo_from_sigma, dpu, process_report, sigma_from_yield};
use scirust_tolerance::spatial::{
    Feature, Torsor, inertia_decomposition, surface_inertia_from_torsors,
};
use scirust_tolerance::variables::design_variables_plan;
use serde_json::json;

pub fn tolerance_tools() -> Vec<McpTool> {
    vec![
        inertial_capability_tool(),
        chain_allocate_tool(),
        acceptance_plan_tool(),
        form_modal_tool(),
        torsor_3d_tool(),
        optimize_cost_tool(),
        nonnormal_tool(),
        position_tool(),
        monte_carlo_tool(),
        geometry_tool(),
        sensitivity_tool(),
        discrete_allocate_tool(),
        drift_tool(),
        correlated_tool(),
        gage_rr_tool(),
        tolerance_interval_tool(),
        dual_sensitivity_tool(),
        distribution_fit_tool(),
        gdt_tool(),
        capability_ci_tool(),
        variables_plan_tool(),
        six_sigma_tool(),
        attribution_tool(),
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

fn optimize_cost_tool() -> McpTool {
    McpTool {
        name: "tolerance_optimize_cost".to_string(),
        description: "Minimum-cost inertial tolerance synthesis under several functional \
            requirements at once (the 'calcul optimal' of inertial tolerancing). Minimises total \
            manufacturing cost Σ bᵢ·Iᵢ^(-rᵢ) (reciprocal-power cost-tolerance model) subject to each \
            requirement's statistical inertia √(Σ αₖᵢ² Iᵢ²) ≤ i_max_k, by convex Lagrangian dual \
            ascent. Returns the optimal per-component inertias, the total cost, and for each \
            requirement the achieved inertia, whether it is binding, and its shadow price."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "cost": { "type": "number", "description": "cost coefficient bᵢ > 0" },
                            "exponent": { "type": "number", "description": "cost exponent rᵢ > 0" },
                        },
                        "required": ["cost", "exponent"],
                    },
                },
                "requirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "coeffs": { "type": "array", "items": { "type": "number" }, "description": "influence coefficients αₖᵢ, one per component" },
                            "i_max": { "type": "number", "description": "max resultant inertia" },
                        },
                        "required": ["coeffs", "i_max"],
                    },
                },
            },
            "required": ["components", "requirements"],
        }),
        handler: Box::new(|args| {
            let comps_json = args
                .get("components")
                .and_then(|v| v.as_array())
                .ok_or("missing `components`")?;
            let components: Vec<Component> = comps_json
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let cost = c.get("cost").and_then(|v| v.as_f64()).ok_or("component `cost` missing")?;
                    let exp = c
                        .get("exponent")
                        .and_then(|v| v.as_f64())
                        .ok_or("component `exponent` missing")?;
                    let name = c
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("X{i}"));
                    Ok(Component::new(name, cost, exp))
                })
                .collect::<Result<Vec<_>, String>>()?;

            let reqs_json = args
                .get("requirements")
                .and_then(|v| v.as_array())
                .ok_or("missing `requirements`")?;
            let requirements: Vec<Requirement> = reqs_json
                .iter()
                .enumerate()
                .map(|(k, r)| {
                    let coeffs: Vec<f64> = r
                        .get("coeffs")
                        .and_then(|v| v.as_array())
                        .ok_or("requirement `coeffs` missing")?
                        .iter()
                        .map(|x| x.as_f64().ok_or("non-numeric coeff".to_string()))
                        .collect::<Result<_, _>>()?;
                    let i_max = r.get("i_max").and_then(|v| v.as_f64()).ok_or("requirement `i_max` missing")?;
                    let name = r
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("Y{k}"));
                    Ok(Requirement::new(name, coeffs, i_max))
                })
                .collect::<Result<Vec<_>, String>>()?;

            let res = optimize(&components, &requirements).map_err(|e| e.to_string())?;
            Ok(json!({
                "inertias": res.inertias,
                "total_cost": res.total_cost,
                "converged": res.converged,
                "requirements": requirements.iter().enumerate().map(|(k, r)| json!({
                    "name": r.name,
                    "achieved": res.achieved[k],
                    "binding": res.binding[k],
                    "shadow_price": res.multipliers[k],
                })).collect::<Vec<_>>(),
            }))
        }),
    }
}

fn nonnormal_tool() -> McpTool {
    McpTool {
        name: "tolerance_nonnormal_capability".to_string(),
        description: "Non-normal statistical tolerancing from the first four moments (mean, sd, \
            skewness, excess kurtosis). Returns the predicted non-conformity in ppm (Cornish-Fisher \
            tail inversion) and the Clements (1989) percentile capability Cp/Cpk for skewed data. \
            Both reduce to the classical normal results when skew=0, kurtosis=0. Valid for moderate \
            non-normality and spec limits within the distribution bulk (a few sigma)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "mean": { "type": "number" },
                "sd": { "type": "number", "description": "standard deviation > 0" },
                "skewness": { "type": "number" },
                "excess_kurtosis": { "type": "number", "description": "kurtosis − 3" },
                "lsl": { "type": "number" },
                "usl": { "type": "number" },
            },
            "required": ["mean", "sd", "skewness", "excess_kurtosis", "lsl", "usl"],
        }),
        handler: Box::new(|args| {
            let mean = f64_field(&args, "mean")?;
            let sd = f64_field(&args, "sd")?;
            let skew = f64_field(&args, "skewness")?;
            let exk = f64_field(&args, "excess_kurtosis")?;
            let lsl = f64_field(&args, "lsl")?;
            let usl = f64_field(&args, "usl")?;
            if sd <= 0.0 || usl <= lsl
            {
                return Err("need sd > 0 and usl > lsl".to_string());
            }
            let c = clements_capability(mean, sd, skew, exk, lsl, usl);
            Ok(json!({
                "ppm": nonnormal_ppm(mean, sd, skew, exk, lsl, usl),
                "clements_cp": c.cp,
                "clements_cpk": c.cpk,
                "clements_cpu": c.cpu,
                "clements_cpl": c.cpl,
                "median": c.median,
            }))
        }),
    }
}

fn position_tool() -> McpTool {
    McpTool {
        name: "tolerance_position".to_string(),
        description: "GD&T / ISO positional tolerancing. Given an axis offset (dx, dy) from true \
            position, a stated diametral position tolerance, and optional MMC size data, returns the \
            true position deviation 2*sqrt(dx^2+dy^2), the total tolerance including MMC bonus, \
            whether the feature conforms, and the positional inertia sqrt(Ix^2+Iy^2). Set \
            `feature` to \"internal\" (hole) or \"external\" (pin) with `actual_size`/`mmc_size` for \
            the bonus."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "dx": { "type": "number", "description": "X offset from true position" },
                "dy": { "type": "number", "description": "Y offset from true position" },
                "stated_tol": { "type": "number", "description": "stated diametral Ø position tolerance" },
                "feature": { "type": "string", "enum": ["internal", "external"], "description": "for MMC bonus" },
                "actual_size": { "type": "number" },
                "mmc_size": { "type": "number" },
                "ix": { "type": "number", "description": "optional per-axis inertia X for positional inertia" },
                "iy": { "type": "number", "description": "optional per-axis inertia Y" },
            },
            "required": ["dx", "dy", "stated_tol"],
        }),
        handler: Box::new(|args| {
            let dx = f64_field(&args, "dx")?;
            let dy = f64_field(&args, "dy")?;
            let stated = f64_field(&args, "stated_tol")?;
            let tp = true_position(dx, dy);
            // Total tolerance with optional MMC bonus.
            let total = match (
                args.get("feature").and_then(|v| v.as_str()),
                args.get("actual_size").and_then(|v| v.as_f64()),
                args.get("mmc_size").and_then(|v| v.as_f64()),
            )
            {
                (Some(f), Some(actual), Some(mmc)) =>
                {
                    let ft = match f
                    {
                        "internal" => FeatureType::Internal,
                        "external" => FeatureType::External,
                        other => return Err(format!("unknown feature `{other}`")),
                    };
                    total_position_tolerance(stated, actual, mmc, ft)
                },
                _ => stated,
            };
            let mut out = json!({
                "true_position": tp,
                "total_tolerance": total,
                "conforms": tp <= total,
            });
            if let (Some(ix), Some(iy)) =
                (args.get("ix").and_then(|v| v.as_f64()), args.get("iy").and_then(|v| v.as_f64()))
            {
                out["positional_inertia"] = json!(positional_inertia(ix, iy));
            }
            Ok(out)
        }),
    }
}

fn parse_distribution(v: &serde_json::Value) -> Result<Distribution, String> {
    let kind = v
        .get("type")
        .and_then(|x| x.as_str())
        .ok_or("component `type` missing")?;
    let f = |k: &str| {
        v.get(k)
            .and_then(|x| x.as_f64())
            .ok_or_else(|| format!("distribution field `{k}` missing"))
    };
    match kind
    {
        "normal" => Ok(Distribution::Normal {
            mean: f("mean")?,
            sd: f("sd")?,
        }),
        "uniform" => Ok(Distribution::Uniform {
            lo: f("lo")?,
            hi: f("hi")?,
        }),
        "triangular" => Ok(Distribution::Triangular {
            lo: f("lo")?,
            mode: f("mode")?,
            hi: f("hi")?,
        }),
        other => Err(format!("unknown distribution `{other}`")),
    }
}

/// Parse `points` (array of numeric arrays) with a required per-row dimension.
fn parse_points(args: &serde_json::Value, dim: usize) -> Result<Vec<Vec<f64>>, String> {
    args.get("points")
        .and_then(|v| v.as_array())
        .ok_or("missing `points`")?
        .iter()
        .map(|row| {
            let a = row
                .as_array()
                .ok_or("`points` must be an array of arrays")?;
            if a.len() != dim
            {
                return Err(format!("each point must have {dim} coordinates"));
            }
            a.iter()
                .map(|x| {
                    x.as_f64()
                        .ok_or_else(|| "non-numeric coordinate".to_string())
                })
                .collect()
        })
        .collect()
}

fn monte_carlo_tool() -> McpTool {
    McpTool {
        name: "tolerance_monte_carlo".to_string(),
        description: "Monte-Carlo tolerance simulation of a linear assembly Y = Σ coeff_i·X_i. Each \
            component X_i is drawn from its distribution (normal {mean,sd}, uniform {lo,hi}, or \
            triangular {lo,mode,hi}); returns the response mean, sigma, inertia about target, \
            non-conformity ppm, yield, and the 0.135/50/99.865% percentiles. Deterministic in `seed`."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": { "type": "object" },
                    "description": "per-component laws, e.g. {\"type\":\"normal\",\"mean\":10,\"sd\":0.1}",
                },
                "coeffs": { "type": "array", "items": { "type": "number" }, "description": "influence coefficients α_i" },
                "target": { "type": "number" },
                "lsl": { "type": "number" },
                "usl": { "type": "number" },
                "trials": { "type": "integer", "description": "number of trials (default 100000)" },
                "seed": { "type": "integer", "description": "RNG seed (default 1)" },
            },
            "required": ["components", "coeffs", "target", "lsl", "usl"],
        }),
        handler: Box::new(|args| {
            let comps_json = args
                .get("components")
                .and_then(|v| v.as_array())
                .ok_or("missing `components`")?;
            let comps: Vec<Distribution> =
                comps_json.iter().map(parse_distribution).collect::<Result<_, _>>()?;
            let coeffs = f64_array(&args, "coeffs")?;
            if coeffs.len() != comps.len() || comps.is_empty()
            {
                return Err("`coeffs` must be non-empty and match `components` length".to_string());
            }
            let target = f64_field(&args, "target")?;
            let lsl = f64_field(&args, "lsl")?;
            let usl = f64_field(&args, "usl")?;
            if usl <= lsl
            {
                return Err("`usl` must exceed `lsl`".to_string());
            }
            let n = args
                .get("trials")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(100_000);
            let seed = args.get("seed").and_then(|v| v.as_u64()).unwrap_or(1);
            let res = simulate(&comps, |xs| linear(&coeffs, xs), target, lsl, usl, n, seed);
            Ok(json!({
                "mean": res.mean,
                "sigma": res.sigma,
                "inertia": res.inertia,
                "ppm": res.ppm,
                "yield": res.yield_fraction,
                "min": res.min,
                "max": res.max,
                "p_low": res.p_low,
                "median": res.median,
                "p_high": res.p_high,
                "trials": res.n,
            }))
        }),
    }
}

fn geometry_tool() -> McpTool {
    McpTool {
        name: "tolerance_geometry".to_string(),
        description: "ISO 1101 form characteristics from measured points: `straightness` / \
            `roundness` (2D points [x,y]) or `flatness` / `cylindricity` (3D points [x,y,z]). \
            Returns the peak-to-valley zone value (least-squares reference feature) and the inertial \
            RMS deviation. For orientation zones use the crate's parallelism/perpendicularity."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "characteristic": { "type": "string", "enum": ["straightness", "roundness", "flatness", "cylindricity"] },
                "points": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } },
                    "description": "[x,y] rows for straightness/roundness; [x,y,z] for flatness/cylindricity",
                },
            },
            "required": ["characteristic", "points"],
        }),
        handler: Box::new(|args| {
            let characteristic = args
                .get("characteristic")
                .and_then(|v| v.as_str())
                .ok_or("missing `characteristic`")?;
            let (value, inertia) = match characteristic
            {
                "straightness" =>
                {
                    let p: Vec<[f64; 2]> =
                        parse_points(&args, 2)?.iter().map(|r| [r[0], r[1]]).collect();
                    (straightness(&p), straightness_inertia(&p))
                },
                "roundness" =>
                {
                    let p: Vec<[f64; 2]> =
                        parse_points(&args, 2)?.iter().map(|r| [r[0], r[1]]).collect();
                    (roundness(&p), roundness_inertia(&p))
                },
                "flatness" =>
                {
                    let p: Vec<[f64; 3]> =
                        parse_points(&args, 3)?.iter().map(|r| [r[0], r[1], r[2]]).collect();
                    (flatness(&p), flatness_inertia(&p))
                },
                "cylindricity" =>
                {
                    let p: Vec<[f64; 3]> =
                        parse_points(&args, 3)?.iter().map(|r| [r[0], r[1], r[2]]).collect();
                    (cylindricity(&p), cylindricity_inertia(&p))
                },
                other => return Err(format!("unknown characteristic `{other}`")),
            };
            Ok(json!({ "characteristic": characteristic, "zone_value": value, "inertia": inertia }))
        }),
    }
}

fn sensitivity_tool() -> McpTool {
    McpTool {
        name: "tolerance_sensitivity".to_string(),
        description: "Rank a tolerance chain's components by their share of the assembly inertia: \
            c_i = α_i²·I_i² / Σ α_j²·I_j² (sums to 1). Points at the few characteristics worth \
            tightening and the many already negligible. Returns contributions sorted largest-first."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "coefficients": { "type": "array", "items": { "type": "number" }, "description": "influence coefficients α_i" },
                "inertias": { "type": "array", "items": { "type": "number" }, "description": "component inertia budgets I_i" },
            },
            "required": ["coefficients", "inertias"],
        }),
        handler: Box::new(|args| {
            let coeffs = f64_array(&args, "coefficients")?;
            let inertias = f64_array(&args, "inertias")?;
            if coeffs.len() != inertias.len() || coeffs.is_empty()
            {
                return Err(
                    "`coefficients` and `inertias` must be non-empty and equal length".to_string(),
                );
            }
            let cs: Vec<Contributor> = coeffs
                .iter()
                .zip(&inertias)
                .enumerate()
                .map(|(i, (a, inertia))| Contributor::new(format!("X{}", i + 1), *a, *inertia))
                .collect();
            let cons = contributions(&cs);
            Ok(json!({
                "contributions": cons.iter().map(|c| json!({
                    "name": c.name,
                    "fraction": c.fraction,
                    "inertia_contribution": c.inertia_contribution,
                })).collect::<Vec<_>>(),
            }))
        }),
    }
}

fn discrete_allocate_tool() -> McpTool {
    McpTool {
        name: "tolerance_discrete_allocate".to_string(),
        description: "Minimum-cost discrete-process tolerance allocation (multiple-choice knapsack): \
            pick one process {inertia,cost} per component so the assembly inertia (statistical \
            √(Σα²I²) or worst_case Σ|α|I) stays within `budget` at least cost. Returns the chosen \
            option index per component, the total cost, and the achieved inertia — or feasible=false."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "coefficients": { "type": "array", "items": { "type": "number" } },
                "options": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "object" } },
                    "description": "per-component menu of {inertia, cost}",
                },
                "budget": { "type": "number", "description": "assembly inertia budget" },
                "method": { "type": "string", "enum": ["statistical", "worst_case"] },
            },
            "required": ["coefficients", "options", "budget"],
        }),
        handler: Box::new(|args| {
            let coeffs = f64_array(&args, "coefficients")?;
            let opts_json = args
                .get("options")
                .and_then(|v| v.as_array())
                .ok_or("missing `options`")?;
            let options: Vec<Vec<ProcessOption>> = opts_json
                .iter()
                .map(|comp| {
                    comp.as_array()
                        .ok_or("`options` rows must be arrays".to_string())?
                        .iter()
                        .map(|o| {
                            let inertia = o
                                .get("inertia")
                                .and_then(|v| v.as_f64())
                                .ok_or("option `inertia` missing".to_string())?;
                            let cost = o
                                .get("cost")
                                .and_then(|v| v.as_f64())
                                .ok_or("option `cost` missing".to_string())?;
                            Ok(ProcessOption::new(inertia, cost))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let budget = f64_field(&args, "budget")?;
            let method = match args.get("method").and_then(|v| v.as_str()).unwrap_or("statistical")
            {
                "statistical" => Combination::Statistical,
                "worst_case" => Combination::WorstCase,
                other => return Err(format!("unknown method `{other}`")),
            };
            match allocate_discrete(&coeffs, &options, budget, method)
            {
                Some(a) => Ok(json!({
                    "feasible": true,
                    "selection": a.selection,
                    "total_cost": a.total_cost,
                    "assembly_inertia": a.assembly_inertia,
                })),
                None => Ok(json!({
                    "feasible": false,
                    "reason": "no process selection meets the budget",
                })),
            }
        }),
    }
}

fn drift_tool() -> McpTool {
    McpTool {
        name: "tolerance_drift".to_string(),
        description: "Short-vs-long-term capability under process drift. From the within (short-term) \
            sigma and a uniform mean-drift half-width d, returns the long-term sigma √(σ²+d²/3) and \
            the long-term non-conformity ppm vs [lsl,usl]. If `cpk` is given, also the 1.5σ-shifted \
            long-term Ppk = Cpk − 0.5."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "mean": { "type": "number" },
                "sigma_st": { "type": "number", "description": "short-term (within) standard deviation" },
                "drift": { "type": "number", "description": "uniform mean-drift half-width d" },
                "lsl": { "type": "number" },
                "usl": { "type": "number" },
                "cpk": { "type": "number", "description": "optional short-term Cpk for the 1.5σ-shift Ppk" },
            },
            "required": ["mean", "sigma_st", "drift", "lsl", "usl"],
        }),
        handler: Box::new(|args| {
            let mean = f64_field(&args, "mean")?;
            let sigma_st = f64_field(&args, "sigma_st")?;
            let drift = f64_field(&args, "drift")?;
            let lsl = f64_field(&args, "lsl")?;
            let usl = f64_field(&args, "usl")?;
            let mut out = json!({
                "long_term_sigma": long_term_sigma(sigma_st, drift),
                "long_term_ppm": long_term_ppm(mean, sigma_st, drift, lsl, usl),
            });
            if let Some(cpk) = args.get("cpk").and_then(|v| v.as_f64())
            {
                out["long_term_ppk"] = json!(cpk_to_ppk(cpk, 1.5));
            }
            Ok(out)
        }),
    }
}

fn correlated_tool() -> McpTool {
    McpTool {
        name: "tolerance_correlated".to_string(),
        description: "Correlated statistical assembly inertia I_Y = √(Σ_ij α_i α_j ρ_ij I_i I_j) — \
            the chain combination when components share a fixture/tool. Give a single common \
            correlation `rho` (applied to every pair). Reduces to the independent √(Σα²I²) at rho=0; \
            returns both for comparison."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "coefficients": { "type": "array", "items": { "type": "number" } },
                "inertias": { "type": "array", "items": { "type": "number" } },
                "rho": { "type": "number", "description": "common pairwise correlation in [-1,1] (default 0)" },
            },
            "required": ["coefficients", "inertias"],
        }),
        handler: Box::new(|args| {
            let coeffs = f64_array(&args, "coefficients")?;
            let inertias = f64_array(&args, "inertias")?;
            if coeffs.len() != inertias.len() || coeffs.is_empty()
            {
                return Err("`coefficients` and `inertias` must be non-empty and equal length"
                    .to_string());
            }
            let rho = args.get("rho").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let corr = uniform_correlation(coeffs.len(), rho);
            let independent: Vec<Contributor> = coeffs
                .iter()
                .zip(&inertias)
                .map(|(a, i)| Contributor::new("x", *a, *i))
                .collect();
            Ok(json!({
                "correlated_inertia": correlated_inertia(&coeffs, &inertias, &corr),
                "independent_inertia": assembly_inertia_statistical(&independent),
                "rho": rho,
            }))
        }),
    }
}

fn gage_rr_tool() -> McpTool {
    McpTool {
        name: "tolerance_gage_rr".to_string(),
        description: "Crossed Gage R&R (ANOVA method, AIAG MSA): from a balanced study \
            measurements[part][operator][replicate], separates the variance into repeatability \
            (equipment), reproducibility (appraiser) and part-to-part, and returns %R&R (study \
            variation), %contribution, %tolerance (if a tolerance is given), the number of distinct \
            categories (ndc) and the AIAG verdict (acceptable/marginal/unacceptable)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "measurements": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } },
                    "description": "measurements[part][operator][replicate], balanced (p≥2, o≥2, r≥2)",
                },
                "tolerance": { "type": "number", "description": "spec width USL−LSL for %tolerance (optional)" },
            },
            "required": ["measurements"],
        }),
        handler: Box::new(|args| {
            let rows = args
                .get("measurements")
                .and_then(|v| v.as_array())
                .ok_or("missing `measurements`")?;
            let data: Vec<Vec<Vec<f64>>> = rows
                .iter()
                .map(|part| {
                    part.as_array()
                        .ok_or("`measurements` must be 3-deep".to_string())?
                        .iter()
                        .map(|cell| {
                            cell.as_array()
                                .ok_or("`measurements` must be 3-deep".to_string())?
                                .iter()
                                .map(|x| x.as_f64().ok_or("non-numeric reading".to_string()))
                                .collect::<Result<Vec<f64>, String>>()
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tol = args.get("tolerance").and_then(|v| v.as_f64());
            match gage_rr(&data, tol)
            {
                None => Err("unbalanced or too-small study (need p≥2, o≥2, r≥2)".to_string()),
                Some(g) => Ok(json!({
                    "repeatability_var": g.repeatability_var,
                    "reproducibility_var": g.reproducibility_var,
                    "grr_var": g.grr_var,
                    "part_var": g.part_var,
                    "total_var": g.total_var,
                    "pct_study_rr": g.pct_study_rr,
                    "pct_contribution": g.pct_contribution,
                    "pct_tolerance": g.pct_tolerance,
                    "ndc": g.ndc,
                    "verdict": format!("{:?}", g.verdict),
                })),
            }
        }),
    }
}

fn tolerance_interval_tool() -> McpTool {
    McpTool {
        name: "tolerance_statistical_interval".to_string(),
        description: "Two-sided statistical tolerance interval (normal theory, ISO 16269-6): from a \
            sample mean, sd and size n, returns limits x̄±k·s that contain at least proportion \
            `coverage` of the population with confidence `confidence`, plus whether they fit inside \
            an optional [lsl, usl] spec. Unlike x̄±3s this accounts for the finite sample."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "mean": { "type": "number" },
                "sd": { "type": "number", "description": "sample standard deviation (n−1)" },
                "n": { "type": "integer", "description": "sample size (≥2)" },
                "coverage": { "type": "number", "description": "population proportion p (e.g. 0.99)" },
                "confidence": { "type": "number", "description": "confidence 1−α (e.g. 0.95)" },
                "lsl": { "type": "number", "description": "optional lower spec for conformance" },
                "usl": { "type": "number", "description": "optional upper spec for conformance" },
            },
            "required": ["mean", "sd", "n", "coverage", "confidence"],
        }),
        handler: Box::new(|args| {
            let mean = f64_field(&args, "mean")?;
            let sd = f64_field(&args, "sd")?;
            let n = args.get("n").and_then(|v| v.as_u64()).ok_or("missing `n`")? as usize;
            let p = f64_field(&args, "coverage")?;
            let conf = f64_field(&args, "confidence")?;
            let ti = tolerance_interval(mean, sd, n, p, conf)
                .ok_or("invalid inputs (need n≥2 and coverage/confidence in (0,1))")?;
            let mut out = json!({ "lower": ti.lower, "upper": ti.upper, "k": ti.k });
            if let (Some(lsl), Some(usl)) = (
                args.get("lsl").and_then(|v| v.as_f64()),
                args.get("usl").and_then(|v| v.as_f64()),
            )
            {
                out["covers_spec"] = json!(ti.covers_spec(lsl, usl));
            }
            Ok(out)
        }),
    }
}

fn dual_sensitivity_tool() -> McpTool {
    McpTool {
        name: "tolerance_dual_sensitivity".to_string(),
        description:
            "Dual sensitivity (GeoFactor + mean-vs-variance split, à la 3DCS/CETOL): from \
            component states (coeff α, off-centering δ, sigma σ), returns per component its \
            geometric magnification |α|, its contribution to the assembly MEAN shift (α·δ, summing \
            to δ_Y) and its share of the assembly VARIANCE (α²σ²/σ_Y², summing to 1) — so a part \
            that needs re-centring is told apart from one that needs its spread reduced."
                .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "coefficients": { "type": "array", "items": { "type": "number" } },
                "off_centerings": { "type": "array", "items": { "type": "number" }, "description": "δ_i per component" },
                "sigmas": { "type": "array", "items": { "type": "number" }, "description": "σ_i per component" },
            },
            "required": ["coefficients", "off_centerings", "sigmas"],
        }),
        handler: Box::new(|args| {
            let coeffs = f64_array(&args, "coefficients")?;
            let deltas = f64_array(&args, "off_centerings")?;
            let sigmas = f64_array(&args, "sigmas")?;
            if coeffs.len() != deltas.len() || coeffs.len() != sigmas.len() || coeffs.is_empty()
            {
                return Err(
                    "`coefficients`, `off_centerings`, `sigmas` must be equal, non-empty"
                        .to_string(),
                );
            }
            let states: Vec<ContributorState> = (0..coeffs.len())
                .map(|i| {
                    ContributorState::new(format!("X{}", i + 1), coeffs[i], deltas[i], sigmas[i])
                })
                .collect();
            let dual = dual_contributions(&states);
            Ok(json!({
                "components": dual.iter().map(|d| json!({
                    "name": d.name,
                    "geo_factor": d.geo_factor,
                    "mean_contribution": d.mean_contribution,
                    "variance_fraction": d.variance_fraction,
                })).collect::<Vec<_>>(),
            }))
        }),
    }
}

fn distribution_fit_tool() -> McpTool {
    McpTool {
        name: "tolerance_distribution_fit".to_string(),
        description: "Fit the best distribution (normal/lognormal/Rayleigh/Weibull by \
            log-likelihood, Q-DAS/ISO 22514 style) to a data sample and report percentile capability \
            Cp=(USL−LSL)/(X99.865−X0.135), Cpk with the fitted median — the correct capability for \
            skewed processes where the normal Cp/Cpk lie."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "sample": { "type": "array", "items": { "type": "number" } },
                "lsl": { "type": "number" },
                "usl": { "type": "number" },
            },
            "required": ["sample", "lsl", "usl"],
        }),
        handler: Box::new(|args| {
            let sample = f64_array(&args, "sample")?;
            let lsl = f64_field(&args, "lsl")?;
            let usl = f64_field(&args, "usl")?;
            if usl <= lsl
            {
                return Err("`usl` must exceed `lsl`".to_string());
            }
            let dist = best_fit(&sample).ok_or("could not fit any distribution")?;
            let c = percentile_capability(&dist, lsl, usl);
            Ok(json!({
                "distribution": format!("{dist:?}"),
                "cp": c.cp,
                "cpk": c.cpk,
                "cpu": c.cpu,
                "cpl": c.cpl,
                "median": c.median,
            }))
        }),
    }
}

fn gdt_tool() -> McpTool {
    McpTool {
        name: "tolerance_gdt".to_string(),
        description: "Advanced GD&T (ASME Y14.5) boundaries: virtual condition (MMC ∓ geo_tol) and \
            resultant condition of a feature-of-size, datum shift (departure of a datum feature from \
            its MMB), and composite-position conformance (two-tier PLTZF/FRTZF). Set `feature` to \
            \"internal\" (hole) or \"external\" (pin)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string", "enum": ["virtual_condition", "resultant_condition", "datum_shift", "composite"] },
                "feature": { "type": "string", "enum": ["internal", "external"] },
                "mmc_size": { "type": "number" },
                "lmc_size": { "type": "number", "description": "for resultant_condition" },
                "geo_tol": { "type": "number" },
                "actual_datum_size": { "type": "number", "description": "for datum_shift" },
                "mmb_size": { "type": "number", "description": "for datum_shift" },
                "pltzf": { "type": "number", "description": "composite upper zone" },
                "frtzf": { "type": "number", "description": "composite lower zone" },
                "loc_dx": { "type": "number" }, "loc_dy": { "type": "number" },
                "pat_dx": { "type": "number" }, "pat_dy": { "type": "number" },
            },
            "required": ["operation"],
        }),
        handler: Box::new(|args| {
            let op = args
                .get("operation")
                .and_then(|v| v.as_str())
                .ok_or("missing `operation`")?;
            let feat = || -> Result<FeatureType, String> {
                match args.get("feature").and_then(|v| v.as_str())
                {
                    Some("internal") => Ok(FeatureType::Internal),
                    Some("external") => Ok(FeatureType::External),
                    _ => Err("need `feature` = \"internal\" or \"external\"".to_string()),
                }
            };
            match op
            {
                "virtual_condition" => Ok(json!({
                    "virtual_condition": virtual_condition(f64_field(&args, "mmc_size")?, f64_field(&args, "geo_tol")?, feat()?),
                })),
                "resultant_condition" => Ok(json!({
                    "resultant_condition": resultant_condition(
                        f64_field(&args, "mmc_size")?,
                        f64_field(&args, "lmc_size")?,
                        f64_field(&args, "geo_tol")?,
                        feat()?,
                    ),
                })),
                "datum_shift" => Ok(json!({
                    "datum_shift": datum_shift(f64_field(&args, "actual_datum_size")?, f64_field(&args, "mmb_size")?, feat()?),
                })),
                "composite" =>
                {
                    let comp = CompositePosition::new(f64_field(&args, "pltzf")?, f64_field(&args, "frtzf")?);
                    let (ldx, ldy) = (f64_field(&args, "loc_dx")?, f64_field(&args, "loc_dy")?);
                    let (pdx, pdy) = (f64_field(&args, "pat_dx")?, f64_field(&args, "pat_dy")?);
                    Ok(json!({
                        "pattern_true_position": true_position(ldx, ldy),
                        "refinement_true_position": true_position(pdx, pdy),
                        "conforms": comp.conforms(ldx, ldy, pdx, pdy),
                    }))
                },
                other => Err(format!("unknown operation `{other}`")),
            }
        }),
    }
}

fn capability_ci_tool() -> McpTool {
    McpTool {
        name: "tolerance_capability_ci".to_string(),
        description:
            "Confidence intervals on capability indices: the exact χ² interval for Cp and \
            the Bissell large-sample interval for Cpk, from a point estimate and the sample size n \
            at a given confidence. Reports the uncertainty a single Cp/Cpk number hides."
                .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "cp": { "type": "number", "description": "Cp point estimate (optional)" },
                "cpk": { "type": "number", "description": "Cpk point estimate (optional)" },
                "n": { "type": "integer", "description": "sample size (≥2)" },
                "confidence": { "type": "number", "description": "confidence 1−α (e.g. 0.95)" },
            },
            "required": ["n", "confidence"],
        }),
        handler: Box::new(|args| {
            let n = args
                .get("n")
                .and_then(|v| v.as_u64())
                .ok_or("missing `n`")? as usize;
            let conf = f64_field(&args, "confidence")?;
            let mut out = json!({});
            if let Some(cp) = args.get("cp").and_then(|v| v.as_f64())
            {
                let (lo, hi) =
                    cp_confidence_interval(cp, n, conf).ok_or("invalid inputs for Cp CI")?;
                out["cp_ci"] = json!([lo, hi]);
            }
            if let Some(cpk) = args.get("cpk").and_then(|v| v.as_f64())
            {
                let (lo, hi) =
                    cpk_confidence_interval(cpk, n, conf).ok_or("invalid inputs for Cpk CI")?;
                out["cpk_ci"] = json!([lo, hi]);
            }
            if out.as_object().map(|o| o.is_empty()).unwrap_or(true)
            {
                return Err("provide at least one of `cp` or `cpk`".to_string());
            }
            Ok(out)
        }),
    }
}

fn variables_plan_tool() -> McpTool {
    McpTool {
        name: "tolerance_variables_plan".to_string(),
        description: "Acceptance sampling by variables (ISO 3951 / MIL-STD-414 Form-k): designs a \
            single-sampling plan (sample size n, acceptance constant k) whose OC passes through the \
            producer point (aql, 1−alpha) and consumer point (rql, beta). Accept the lot when \
            (limit−mean)/spread ≥ k. Reports the achieved acceptance probabilities and, if a spec \
            [lsl, usl] is given, the maximum process sigma (MSD) a centred lot may carry. Set \
            known_sigma=false for the s-method (unknown σ), which needs a larger sample."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "aql": { "type": "number", "description": "acceptable quality level (fraction nonconforming, e.g. 0.01)" },
                "rql": { "type": "number", "description": "rejectable quality level (fraction, > aql, e.g. 0.08)" },
                "alpha": { "type": "number", "description": "producer's risk (e.g. 0.05)" },
                "beta": { "type": "number", "description": "consumer's risk (e.g. 0.10)" },
                "known_sigma": { "type": "boolean", "description": "true=σ-method (default), false=s-method" },
                "lsl": { "type": "number", "description": "lower spec limit (optional, for MSD)" },
                "usl": { "type": "number", "description": "upper spec limit (optional, for MSD)" },
            },
            "required": ["aql", "rql", "alpha", "beta"],
        }),
        handler: Box::new(|args| {
            let aql = f64_field(&args, "aql")?;
            let rql = f64_field(&args, "rql")?;
            let alpha = f64_field(&args, "alpha")?;
            let beta = f64_field(&args, "beta")?;
            let known = args
                .get("known_sigma")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let plan = design_variables_plan(aql, rql, alpha, beta, known)
                .ok_or("invalid inputs (need 0<aql<rql<1, risks in (0,1))")?;
            let mut out = json!({
                "sample_size": plan.sample_size,
                "acceptance_constant": plan.acceptance_constant,
                "known_sigma": plan.known_sigma,
                "p_accept_aql": plan.probability_of_acceptance(aql),
                "p_accept_rql": plan.probability_of_acceptance(rql),
            });
            if let (Some(lsl), Some(usl)) =
                (args.get("lsl").and_then(|v| v.as_f64()), args.get("usl").and_then(|v| v.as_f64()))
            {
                out["max_process_sigma"] = json!(plan.max_process_sigma(lsl, usl));
            }
            Ok(out)
        }),
    }
}

fn six_sigma_tool() -> McpTool {
    McpTool {
        name: "tolerance_six_sigma".to_string(),
        description: "Six-Sigma yield accounting. Provide `step_yields` (per-step throughput \
            yields) for a rolled-throughput-yield report (RTY, normalised yield, total DPU, sigma \
            level, DPMO); and/or `defects`+`units`(+`opportunities`) for DPU/DPMO; and/or `sigma` \
            for its DPMO; and/or `yield` for its sigma level. `shift` is the sigma-level shift \
            (default 1.5 short-term; use 0 for long-term Z-bench)."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "step_yields": { "type": "array", "items": { "type": "number" }, "description": "per-step throughput yields in [0,1]" },
                "defects": { "type": "number", "description": "observed defect count" },
                "units": { "type": "number", "description": "units inspected" },
                "opportunities": { "type": "number", "description": "defect opportunities per unit" },
                "sigma": { "type": "number", "description": "process sigma level → DPMO" },
                "yield": { "type": "number", "description": "yield fraction → sigma level" },
                "shift": { "type": "number", "description": "sigma shift (default 1.5)" },
            },
        }),
        handler: Box::new(|args| {
            let shift = args.get("shift").and_then(|v| v.as_f64()).unwrap_or(1.5);
            let mut out = json!({});
            if let Some(ys) = args.get("step_yields").and_then(|v| v.as_array())
            {
                let yields: Vec<f64> = ys.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
                let rep = process_report(&yields, shift).ok_or("empty `step_yields`")?;
                out["rolled_throughput_yield"] = json!(rep.rolled_throughput_yield);
                out["normalized_yield"] = json!(rep.normalized_yield);
                out["total_dpu"] = json!(rep.total_dpu);
                out["sigma_level"] = json!(rep.sigma_level);
                out["dpmo"] = json!(rep.dpmo);
            }
            if let (Some(d), Some(u)) = (
                args.get("defects").and_then(|v| v.as_f64()),
                args.get("units").and_then(|v| v.as_f64()),
            )
            {
                out["dpu"] = json!(dpu(d, u));
                if let Some(o) = args.get("opportunities").and_then(|v| v.as_f64())
                {
                    out["dpmo_measured"] = json!(dpmo(d, u, o));
                }
            }
            if let Some(s) = args.get("sigma").and_then(|v| v.as_f64())
            {
                out["dpmo_from_sigma"] = json!(dpmo_from_sigma(s, shift));
            }
            if let Some(y) = args.get("yield").and_then(|v| v.as_f64())
            {
                out["sigma_from_yield"] = json!(sigma_from_yield(y, shift));
            }
            if out.as_object().map(|o| o.is_empty()).unwrap_or(true)
            {
                return Err(
                    "provide at least one of step_yields / defects+units / sigma / yield"
                        .to_string(),
                );
            }
            Ok(out)
        }),
    }
}

fn attribution_tool() -> McpTool {
    McpTool {
        name: "tolerance_attribution".to_string(),
        description: "Data-driven root-cause attribution: fits the measured assembly characteristic \
            `assembly` to co-measured component series `columns` (columns[component][observation]) by \
            least squares and decomposes the explained variance. Returns each component's fitted \
            sensitivity (empirical ∂y/∂x) and signed contribution share (summing to R²), the model R² \
            and the unexplained remainder — the tell-tale of a cause outside the measured set."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": { "type": "array", "items": { "type": "number" } },
                    "description": "columns[component][observation]; every column length n = assembly length",
                },
                "assembly": { "type": "array", "items": { "type": "number" }, "description": "measured assembly characteristic, length n" },
                "names": { "type": "array", "items": { "type": "string" }, "description": "component names (optional; default x1..xk)" },
            },
            "required": ["columns", "assembly"],
        }),
        handler: Box::new(|args| {
            let cols_json = args
                .get("columns")
                .and_then(|v| v.as_array())
                .ok_or("missing `columns`")?;
            let columns: Vec<Vec<f64>> = cols_json
                .iter()
                .map(|c| {
                    c.as_array()
                        .ok_or("`columns` must be 2-deep".to_string())?
                        .iter()
                        .map(|x| x.as_f64().ok_or("non-numeric reading".to_string()))
                        .collect::<Result<Vec<f64>, String>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let assembly = f64_array(&args, "assembly")?;
            let default_names: Vec<String> =
                (1..=columns.len()).map(|i| format!("x{i}")).collect();
            let names: Vec<String> = match args.get("names").and_then(|v| v.as_array())
            {
                Some(a) => a
                    .iter()
                    .map(|s| s.as_str().unwrap_or("").to_string())
                    .collect(),
                None => default_names,
            };
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let rep = attribute(&name_refs, &columns, &assembly)
                .ok_or("bad shape / collinear / too few observations (need n ≥ k+2, Var(y)>0)")?;
            Ok(json!({
                "intercept": rep.intercept,
                "r_squared": rep.r_squared,
                "unexplained": rep.unexplained,
                "components": rep.components.iter().map(|c| json!({
                    "name": c.name,
                    "sensitivity": c.sensitivity,
                    "contribution": c.contribution,
                })).collect::<Vec<_>>(),
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

    #[test]
    fn optimize_cost_tool_solves_two_requirements() {
        let tool = optimize_cost_tool();
        let out = (tool.handler)(json!({
            "components": [
                { "name": "A", "cost": 1.0, "exponent": 2.0 },
                { "name": "B", "cost": 3.0, "exponent": 2.0 },
                { "name": "C", "cost": 2.0, "exponent": 3.0 },
            ],
            "requirements": [
                { "name": "Y1", "coeffs": [1.0, -1.0, 1.0], "i_max": 0.06 },
                { "name": "Y2", "coeffs": [1.0, 1.0, 0.0], "i_max": 0.05 },
            ],
        }))
        .unwrap();
        assert_eq!(out["converged"], json!(true));
        assert_eq!(out["inertias"].as_array().unwrap().len(), 3);
        assert!(out["total_cost"].as_f64().unwrap() > 0.0);
        // Every requirement is feasible.
        for r in out["requirements"].as_array().unwrap()
        {
            assert!(r["achieved"].as_f64().unwrap() <= 0.06 * (1.0 + 1e-6));
        }
    }

    #[test]
    fn optimize_cost_tool_rejects_unconstrained_component() {
        let tool = optimize_cost_tool();
        assert!(
            (tool.handler)(json!({
                "components": [{ "cost": 1.0, "exponent": 2.0 }],
                "requirements": [{ "coeffs": [0.0], "i_max": 0.1 }],
            }))
            .is_err()
        );
    }

    #[test]
    fn nonnormal_tool_reduces_to_normal_when_symmetric() {
        let tool = nonnormal_tool();
        let out = (tool.handler)(json!({
            "mean": 10.5,
            "sd": 1.0,
            "skewness": 0.0,
            "excess_kurtosis": 0.0,
            "lsl": 7.0,
            "usl": 13.0,
        }))
        .unwrap();
        // Symmetric ⇒ Clements Cp = (USL−LSL)/6σ = 1.0, median = mean.
        assert!((out["clements_cp"].as_f64().unwrap() - 1.0).abs() < 1e-3);
        assert!((out["median"].as_f64().unwrap() - 10.5).abs() < 1e-6);
        assert!(out["ppm"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn nonnormal_tool_skew_fattens_the_tail() {
        let tool = nonnormal_tool();
        let base = json!({ "mean": 0.0, "sd": 1.0, "lsl": -3.0, "usl": 3.0 });
        let mut sym = base.clone();
        sym["skewness"] = json!(0.0);
        sym["excess_kurtosis"] = json!(0.0);
        let mut skewed = base;
        skewed["skewness"] = json!(1.0);
        skewed["excess_kurtosis"] = json!(2.0);
        let p_sym = (tool.handler)(sym).unwrap()["ppm"].as_f64().unwrap();
        let p_skew = (tool.handler)(skewed).unwrap()["ppm"].as_f64().unwrap();
        assert!(p_skew > p_sym);
    }

    #[test]
    fn nonnormal_tool_rejects_bad_spec() {
        let tool = nonnormal_tool();
        assert!(
            (tool.handler)(json!({
                "mean": 0.0, "sd": 0.0, "skewness": 0.0,
                "excess_kurtosis": 0.0, "lsl": -1.0, "usl": 1.0,
            }))
            .is_err()
        );
    }

    #[test]
    fn position_tool_reports_true_position_and_bonus() {
        let tool = position_tool();
        let out = (tool.handler)(json!({
            "dx": 0.03,
            "dy": 0.04,
            "stated_tol": 0.1,
            "feature": "internal",
            "actual_size": 10.2,
            "mmc_size": 10.0,
            "ix": 0.03,
            "iy": 0.04,
        }))
        .unwrap();
        // (0.03, 0.04) ⇒ radius 0.05 ⇒ true position Ø 0.10.
        assert!((out["true_position"].as_f64().unwrap() - 0.10).abs() < 1e-12);
        // Bonus 0.2 ⇒ total 0.30; 0.10 ≤ 0.30 conforms.
        assert!((out["total_tolerance"].as_f64().unwrap() - 0.30).abs() < 1e-12);
        assert_eq!(out["conforms"], json!(true));
        // I_pos = √(0.03²+0.04²) = 0.05.
        assert!((out["positional_inertia"].as_f64().unwrap() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn position_tool_without_mmc_uses_stated_tol() {
        let tool = position_tool();
        let out = (tool.handler)(json!({ "dx": 0.1, "dy": 0.0, "stated_tol": 0.15 })).unwrap();
        // Ø 0.20 > 0.15 ⇒ does not conform; no positional_inertia key.
        assert!((out["true_position"].as_f64().unwrap() - 0.20).abs() < 1e-12);
        assert_eq!(out["total_tolerance"], json!(0.15));
        assert_eq!(out["conforms"], json!(false));
        assert!(out.get("positional_inertia").is_none());
    }

    #[test]
    fn position_tool_rejects_unknown_feature() {
        let tool = position_tool();
        assert!(
            (tool.handler)(json!({
                "dx": 0.0, "dy": 0.0, "stated_tol": 0.1,
                "feature": "bogus", "actual_size": 10.1, "mmc_size": 10.0,
            }))
            .is_err()
        );
    }

    #[test]
    fn monte_carlo_tool_matches_linear_normal() {
        let tool = monte_carlo_tool();
        let out = (tool.handler)(json!({
            "components": [
                { "type": "normal", "mean": 10.0, "sd": 0.10 },
                { "type": "normal", "mean": 4.0, "sd": 0.08 },
            ],
            "coeffs": [1.0, -1.0],
            "target": 6.0,
            "lsl": 5.0,
            "usl": 7.0,
            "trials": 200000,
            "seed": 7,
        }))
        .unwrap();
        // Y = X1 − X2 ⇒ mean 6, σ = √(0.01+0.0064) ≈ 0.128.
        assert!((out["mean"].as_f64().unwrap() - 6.0).abs() < 0.01);
        assert!((out["sigma"].as_f64().unwrap() - (0.0164f64).sqrt()).abs() < 0.01);
        assert!(out["yield"].as_f64().unwrap() > 0.99);
    }

    #[test]
    fn geometry_tool_reports_zero_for_perfect_circle() {
        let tool = geometry_tool();
        let pts: Vec<[f64; 2]> = (0..16)
            .map(|k| {
                let t = k as f64 / 16.0 * std::f64::consts::TAU;
                [2.0 + t.cos(), -1.0 + t.sin()]
            })
            .collect();
        let out = (tool.handler)(json!({ "characteristic": "roundness", "points": pts })).unwrap();
        assert!(out["zone_value"].as_f64().unwrap() < 1e-6);
        // Wrong dimensionality is rejected.
        assert!(
            (tool.handler)(json!({ "characteristic": "flatness", "points": [[0.0, 0.0]] }))
                .is_err()
        );
    }

    #[test]
    fn sensitivity_tool_ranks_and_sums_to_one() {
        let tool = sensitivity_tool();
        let out = (tool.handler)(json!({
            "coefficients": [1.0, 2.0, 1.0],
            "inertias": [0.10, 0.05, 0.02],
        }))
        .unwrap();
        let cons = out["contributions"].as_array().unwrap();
        let sum: f64 = cons.iter().map(|c| c["fraction"].as_f64().unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn discrete_allocate_tool_selects_within_budget() {
        let tool = discrete_allocate_tool();
        let out = (tool.handler)(json!({
            "coefficients": [1.0, -1.0],
            "options": [
                [{ "inertia": 0.10, "cost": 1.0 }, { "inertia": 0.05, "cost": 3.0 }],
                [{ "inertia": 0.10, "cost": 1.0 }, { "inertia": 0.05, "cost": 3.0 }],
            ],
            "budget": 0.20,
            "method": "worst_case",
        }))
        .unwrap();
        assert_eq!(out["feasible"], json!(true));
        assert!((out["total_cost"].as_f64().unwrap() - 2.0).abs() < 1e-12);
        // Impossible budget ⇒ infeasible.
        let bad = (tool.handler)(json!({
            "coefficients": [1.0, -1.0],
            "options": [
                [{ "inertia": 0.10, "cost": 1.0 }],
                [{ "inertia": 0.10, "cost": 1.0 }],
            ],
            "budget": 0.05,
            "method": "worst_case",
        }))
        .unwrap();
        assert_eq!(bad["feasible"], json!(false));
    }

    #[test]
    fn drift_tool_inflates_sigma_and_ppm() {
        let tool = drift_tool();
        let out = (tool.handler)(json!({
            "mean": 0.0, "sigma_st": 0.3, "drift": 0.6, "lsl": -1.0, "usl": 1.0, "cpk": 1.5,
        }))
        .unwrap();
        // σ_lt = √(0.09 + 0.12) = √0.21.
        assert!((out["long_term_sigma"].as_f64().unwrap() - 0.21f64.sqrt()).abs() < 1e-9);
        assert!((out["long_term_ppk"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn correlated_tool_reduces_to_independent_at_zero() {
        let tool = correlated_tool();
        let out = (tool.handler)(json!({
            "coefficients": [1.0, -1.0, 0.5],
            "inertias": [0.10, 0.08, 0.20],
            "rho": 0.0,
        }))
        .unwrap();
        let a = out["correlated_inertia"].as_f64().unwrap();
        let b = out["independent_inertia"].as_f64().unwrap();
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn gage_rr_tool_reports_components_and_verdict() {
        let tool = gage_rr_tool();
        let out = (tool.handler)(json!({
            "measurements": [
                [[10.0, 10.0], [10.1, 10.1]],
                [[12.0, 12.0], [12.1, 12.1]],
                [[8.0, 8.0], [8.1, 8.1]],
            ],
            "tolerance": 6.0,
        }))
        .unwrap();
        assert!(out["repeatability_var"].as_f64().unwrap() < 1e-9);
        assert!(out["part_var"].as_f64().unwrap() > 0.0);
        assert!(out["verdict"].is_string());
        // Unbalanced ⇒ error.
        assert!((tool.handler)(json!({ "measurements": [[[1.0]]] })).is_err());
    }

    #[test]
    fn tolerance_interval_tool_spec_coverage() {
        let tool = tolerance_interval_tool();
        let out = (tool.handler)(json!({
            "mean": 10.0, "sd": 0.5, "n": 30, "coverage": 0.99, "confidence": 0.95,
            "lsl": 5.0, "usl": 15.0,
        }))
        .unwrap();
        assert!(out["k"].as_f64().unwrap() > 1.959);
        assert_eq!(out["covers_spec"], json!(true));
    }

    #[test]
    fn dual_sensitivity_tool_splits_mean_and_variance() {
        let tool = dual_sensitivity_tool();
        let out = (tool.handler)(json!({
            "coefficients": [1.0, 1.0],
            "off_centerings": [0.2, 0.0],
            "sigmas": [0.01, 0.10],
        }))
        .unwrap();
        let comps = out["components"].as_array().unwrap();
        let vsum: f64 = comps
            .iter()
            .map(|c| c["variance_fraction"].as_f64().unwrap())
            .sum();
        assert!((vsum - 1.0).abs() < 1e-12);
        // Second component dominates variance; first dominates the mean.
        assert!(comps[1]["variance_fraction"].as_f64().unwrap() > 0.98);
        assert!(comps[0]["mean_contribution"].as_f64().unwrap().abs() > 0.1);
    }

    #[test]
    fn distribution_fit_tool_reduces_to_normal() {
        let tool = distribution_fit_tool();
        // Roughly symmetric data ⇒ fit + finite capability.
        let out = (tool.handler)(json!({
            "sample": [9.8, 10.1, 10.0, 9.9, 10.2, 9.95, 10.05, 9.85, 10.15, 10.0],
            "lsl": 9.0, "usl": 11.0,
        }))
        .unwrap();
        assert!(out["cp"].as_f64().unwrap() > 0.0);
        assert!(out["distribution"].is_string());
    }

    #[test]
    fn gdt_tool_virtual_condition_and_composite() {
        let tool = gdt_tool();
        let vc = (tool.handler)(json!({
            "operation": "virtual_condition", "feature": "internal", "mmc_size": 10.0, "geo_tol": 0.2,
        }))
        .unwrap();
        assert!((vc["virtual_condition"].as_f64().unwrap() - 9.8).abs() < 1e-12);
        let comp = (tool.handler)(json!({
            "operation": "composite", "pltzf": 0.4, "frtzf": 0.1,
            "loc_dx": 0.15, "loc_dy": 0.0, "pat_dx": 0.04, "pat_dy": 0.0,
        }))
        .unwrap();
        assert_eq!(comp["conforms"], json!(true));
        assert!((tool.handler)(json!({ "operation": "bogus" })).is_err());
    }

    #[test]
    fn capability_ci_tool_brackets_estimates() {
        let tool = capability_ci_tool();
        let out =
            (tool.handler)(json!({ "cp": 1.33, "cpk": 1.2, "n": 50, "confidence": 0.95 })).unwrap();
        let cp_ci = out["cp_ci"].as_array().unwrap();
        assert!(cp_ci[0].as_f64().unwrap() < 1.33 && 1.33 < cp_ci[1].as_f64().unwrap());
        // No index provided ⇒ error.
        assert!((tool.handler)(json!({ "n": 50, "confidence": 0.95 })).is_err());
    }

    #[test]
    fn variables_plan_tool_designs_and_reports_msd() {
        let tool = variables_plan_tool();
        let out = (tool.handler)(json!({
            "aql": 0.01, "rql": 0.08, "alpha": 0.05, "beta": 0.10,
            "lsl": 10.0, "usl": 20.0,
        }))
        .unwrap();
        assert!(out["sample_size"].as_u64().unwrap() >= 2);
        assert!(out["acceptance_constant"].as_f64().unwrap() > 1.0);
        assert!(out["p_accept_aql"].as_f64().unwrap() > 0.9);
        assert!(out["p_accept_rql"].as_f64().unwrap() < 0.15);
        // MSD = (usl−lsl)/(2k).
        let k = out["acceptance_constant"].as_f64().unwrap();
        assert!((out["max_process_sigma"].as_f64().unwrap() - 10.0 / (2.0 * k)).abs() < 1e-9);
        // aql ≥ rql is rejected.
        assert!(
            (tool.handler)(json!({ "aql": 0.08, "rql": 0.01, "alpha": 0.05, "beta": 0.10 }))
                .is_err()
        );
    }

    #[test]
    fn six_sigma_tool_rolls_yield_and_converts() {
        let tool = six_sigma_tool();
        let out = (tool.handler)(json!({ "step_yields": [0.99, 0.99, 0.99, 0.99, 0.99] })).unwrap();
        // Five 99% steps roll up below any single step.
        assert!(out["rolled_throughput_yield"].as_f64().unwrap() < 0.99);
        assert!((out["normalized_yield"].as_f64().unwrap() - 0.99).abs() < 1e-9);
        // 6σ short-term ⇒ ≈3.4 DPMO.
        let d = (tool.handler)(json!({ "sigma": 6.0 })).unwrap();
        assert!((d["dpmo_from_sigma"].as_f64().unwrap() - 3.4).abs() < 0.1);
        // Nothing supplied ⇒ error.
        assert!((tool.handler)(json!({})).is_err());
    }

    #[test]
    fn attribution_tool_recovers_sensitivities() {
        let tool = attribution_tool();
        // y = 2·x1 − x2 exactly over 6 observations.
        let x1 = [0.0, 1.0, 2.0, 3.0, 1.5, -1.0];
        let x2 = [1.0, 0.0, 1.0, -1.0, 2.0, 0.5];
        let y: Vec<f64> = (0..6).map(|i| 2.0 * x1[i] - x2[i]).collect();
        let out = (tool.handler)(json!({
            "columns": [x1, x2],
            "assembly": y,
            "names": ["a", "b"],
        }))
        .unwrap();
        assert!(out["r_squared"].as_f64().unwrap() > 0.999_999);
        let comps = out["components"].as_array().unwrap();
        assert_eq!(comps[0]["name"], json!("a"));
        assert!((comps[0]["sensitivity"].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((comps[1]["sensitivity"].as_f64().unwrap() + 1.0).abs() < 1e-6);
        // Contributions sum to R².
        let sum: f64 = comps
            .iter()
            .map(|c| c["contribution"].as_f64().unwrap())
            .sum();
        assert!((sum - out["r_squared"].as_f64().unwrap()).abs() < 1e-7);
        // Too few observations ⇒ error.
        assert!(
            (tool.handler)(json!({ "columns": [[1.0, 2.0]], "assembly": [1.0, 2.0] })).is_err()
        );
    }
}
