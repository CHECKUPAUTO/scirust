//! MCP tools for the reproducible trading-research layer.
//!
//! These tools expose validation evidence to agents without silently selecting
//! a profitable strategy. Inputs are explicit and outputs are serializable so
//! they can be attached to experiment manifests and audit logs.

use crate::registry::McpTool;
use serde::Serialize;
use serde_json::{Value, json};

use scirust_trader::comparison::{
    CandidateEvidence, ComparisonMetric, compare_candidates, comparison_csv,
};
use scirust_trader::ml_dataset::TimeSeriesMlDataset;
use scirust_trader::research_validation::{ExperimentManifest, cost_stress};
use scirust_trader::rl_market::RlExperimentPlan;
use scirust_trader::stat_validation::{
    cscv_probability_of_backtest_overfitting, deflated_sharpe_ratio,
};
use scirust_trader::validation_cv::{PurgedCvConfig, purged_kfold};

fn to_value<T: Serialize>(value: &T) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|error| error.to_string())
}

fn required_f64(args: &Value, key: &str) -> Result<f64, String> {
    args.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or invalid `{key}`"))
}

fn required_u64(args: &Value, key: &str) -> Result<u64, String> {
    args.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid `{key}`"))
}

fn required_usize(args: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(required_u64(args, key)?).map_err(|_| format!("`{key}` exceeds usize"))
}

fn required_vec_f64(args: &Value, key: &str) -> Result<Vec<f64>, String> {
    serde_json::from_value(
        args.get(key)
            .cloned()
            .ok_or_else(|| format!("missing `{key}`"))?,
    )
    .map_err(|error| format!("invalid `{key}`: {error}"))
}

fn parse_dataset(args: &Value) -> Result<TimeSeriesMlDataset, String> {
    serde_json::from_value(
        args.get("dataset")
            .cloned()
            .ok_or_else(|| "missing `dataset`".to_string())?,
    )
    .map_err(|error| format!("invalid `dataset`: {error}"))
}

pub fn trader_research_tools() -> Vec<McpTool> {
    vec![
        purged_cv_tool(),
        dsr_tool(),
        pbo_tool(),
        cost_stress_tool(),
        temporal_rl_plan_tool(),
        compare_tool(),
        manifest_tool(),
    ]
}

fn purged_cv_tool() -> McpTool {
    McpTool {
        name: "trader_research_purged_cv".to_string(),
        description: "Build contiguous purged/embargoed CV folds from a time-stamped forward-labelled dataset. Returns exact train/test/purged/embargoed indices; it does not fit or rank a strategy.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["dataset", "n_splits", "embargo_rows"],
            "properties": {
                "dataset": {"type": "object", "description": "TimeSeriesMlDataset JSON including feature_provenance and rows with ts_ms, feature_available_ts_ms and target_ts_ms"},
                "n_splits": {"type": "integer", "minimum": 2},
                "embargo_rows": {"type": "integer", "minimum": 0}
            }
        }),
        handler: Box::new(|args| {
            let dataset = parse_dataset(&args)?;
            let config = PurgedCvConfig {
                n_splits: required_usize(&args, "n_splits")?,
                embargo_rows: required_usize(&args, "embargo_rows")?,
            };
            let folds = purged_kfold(&dataset, config).map_err(|error| format!("{error:?}"))?;
            to_value(&folds)
        }),
    }
}

fn dsr_tool() -> McpTool {
    McpTool {
        name: "trader_research_dsr".to_string(),
        description: "Compute a Deflated Sharpe Ratio probability from an observed Sharpe, sample moments and an explicit effective independent-trial count. No trial-count inference is performed.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["observed_sharpe", "n_observations", "skewness", "raw_kurtosis", "independent_trials", "cross_trial_sharpe_sd"],
            "properties": {
                "observed_sharpe": {"type": "number"},
                "n_observations": {"type": "integer", "minimum": 2},
                "skewness": {"type": "number"},
                "raw_kurtosis": {"type": "number", "description": "raw kurtosis; Gaussian = 3"},
                "independent_trials": {"type": "integer", "minimum": 1},
                "cross_trial_sharpe_sd": {"type": "number", "minimum": 0}
            }
        }),
        handler: Box::new(|args| {
            let report = deflated_sharpe_ratio(
                required_f64(&args, "observed_sharpe")?,
                required_usize(&args, "n_observations")?,
                required_f64(&args, "skewness")?,
                required_f64(&args, "raw_kurtosis")?,
                required_usize(&args, "independent_trials")?,
                required_f64(&args, "cross_trial_sharpe_sd")?,
            )
            .map_err(|error| format!("{error:?}"))?;
            to_value(&report)
        }),
    }
}

fn pbo_tool() -> McpTool {
    McpTool {
        name: "trader_research_pbo".to_string(),
        description: "Estimate Probability of Backtest Overfitting with combinatorially symmetric cross-validation over a rectangular strategy-return matrix.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["strategy_returns", "slices"],
            "properties": {
                "strategy_returns": {"type": "array", "description": "matrix [strategy][observation] of realized returns"},
                "slices": {"type": "integer", "minimum": 2, "description": "even number of contiguous time slices"}
            }
        }),
        handler: Box::new(|args| {
            let matrix: Vec<Vec<f64>> = serde_json::from_value(
                args.get("strategy_returns")
                    .cloned()
                    .ok_or_else(|| "missing `strategy_returns`".to_string())?,
            )
            .map_err(|error| format!("invalid `strategy_returns`: {error}"))?;
            let report = cscv_probability_of_backtest_overfitting(
                &matrix,
                required_usize(&args, "slices")?,
            )
            .map_err(|error| format!("{error:?}"))?;
            to_value(&report)
        }),
    }
}

fn cost_stress_tool() -> McpTool {
    McpTool {
        name: "trader_research_cost_stress".to_string(),
        description: "Reprice one fixed gross-return/turnover path over an explicit transaction-cost grid. This isolates cost sensitivity and does not resimulate fills or market impact.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["gross_returns", "turnover", "cost_grid_bps"],
            "properties": {
                "gross_returns": {"type": "array", "items": {"type": "number"}},
                "turnover": {"type": "array", "items": {"type": "number", "minimum": 0}},
                "cost_grid_bps": {"type": "array", "items": {"type": "number", "minimum": 0}}
            }
        }),
        handler: Box::new(|args| {
            let gross_returns = required_vec_f64(&args, "gross_returns")?;
            let turnover = required_vec_f64(&args, "turnover")?;
            let grid = required_vec_f64(&args, "cost_grid_bps")?;
            let report = cost_stress(&gross_returns, &turnover, &grid)
                .ok_or_else(|| "invalid cost-stress inputs".to_string())?;
            to_value(&report)
        }),
    }
}

fn temporal_rl_plan_tool() -> McpTool {
    McpTool {
        name: "trader_research_rl_plan".to_string(),
        description: "Create a chronological RL train/validation/holdout experiment plan with label-overlap checks. It returns the plan only and does not train on holdout data.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["dataset", "train_fraction", "validation_fraction", "seed", "transaction_cost_bps_per_unit_turnover"],
            "properties": {
                "dataset": {"type": "object"},
                "train_fraction": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "validation_fraction": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                "seed": {"type": "integer", "minimum": 0},
                "transaction_cost_bps_per_unit_turnover": {"type": "number", "minimum": 0}
            }
        }),
        handler: Box::new(|args| {
            let dataset = parse_dataset(&args)?;
            let split = dataset
                .time_split(
                    required_f64(&args, "train_fraction")? as f32,
                    required_f64(&args, "validation_fraction")? as f32,
                )
                .map_err(|error| format!("{error:?}"))?;
            let plan = RlExperimentPlan::from_time_split(
                &dataset,
                split,
                required_u64(&args, "seed")?,
                required_f64(&args, "transaction_cost_bps_per_unit_turnover")?,
            )
            .map_err(|error| format!("{error:?}"))?;
            to_value(&plan)
        }),
    }
}

fn compare_tool() -> McpTool {
    McpTool {
        name: "trader_research_compare".to_string(),
        description: "Build the final multi-metric candidate comparison with explicit metric directions, per-metric ranks and a Pareto front. No hidden weighting or composite profitability score is used.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["metrics", "candidates"],
            "properties": {
                "metrics": {"type": "array", "description": "ComparisonMetric objects: name, direction (HigherBetter|LowerBetter), unit"},
                "candidates": {"type": "array", "description": "CandidateEvidence objects with identical metric sets and evidence_refs"}
            }
        }),
        handler: Box::new(|args| {
            let metrics: Vec<ComparisonMetric> = serde_json::from_value(
                args.get("metrics")
                    .cloned()
                    .ok_or_else(|| "missing `metrics`".to_string())?,
            )
            .map_err(|error| format!("invalid `metrics`: {error}"))?;
            let candidates: Vec<CandidateEvidence> = serde_json::from_value(
                args.get("candidates")
                    .cloned()
                    .ok_or_else(|| "missing `candidates`".to_string())?,
            )
            .map_err(|error| format!("invalid `candidates`: {error}"))?;
            let report = compare_candidates(metrics, candidates)
                .map_err(|error| format!("{error:?}"))?;
            let csv = comparison_csv(&report);
            Ok(json!({"report": report, "csv": csv}))
        }),
    }
}

fn manifest_tool() -> McpTool {
    McpTool {
        name: "trader_research_manifest".to_string(),
        description: "Validate and SHA-256 fingerprint an explicit trading-research ExperimentManifest. Invalid or incomplete manifests are rejected rather than filled with hidden defaults.".to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["manifest"],
            "properties": {"manifest": {"type": "object"}}
        }),
        handler: Box::new(|args| {
            let manifest: ExperimentManifest = serde_json::from_value(
                args.get("manifest")
                    .cloned()
                    .ok_or_else(|| "missing `manifest`".to_string())?,
            )
            .map_err(|error| format!("invalid `manifest`: {error}"))?;
            let fingerprint = manifest
                .fingerprint()
                .ok_or_else(|| "manifest failed validation".to_string())?;
            Ok(json!({"manifest": manifest, "fingerprint": fingerprint}))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_trader::comparison::MetricDirection;
    use scirust_trader::ml_dataset::{FeatureProvenance, MlRow};
    use std::collections::BTreeMap;

    fn dataset_value() -> Value {
        to_value(&TimeSeriesMlDataset {
            feature_provenance: vec![FeatureProvenance {
                name: "x".into(),
                source: "test".into(),
                transformation: "identity".into(),
            }],
            rows: (0..12)
                .map(|i| MlRow {
                    ts_ms: i * 10,
                    feature_available_ts_ms: i * 10,
                    target_ts_ms: i * 10 + 5,
                    features: vec![i as f32],
                    target: i as f32 / 100.0,
                })
                .collect(),
        })
        .unwrap()
    }

    #[test]
    fn tool_names_are_unique() {
        let tools = trader_research_tools();
        let mut names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), tools.len());
    }

    #[test]
    fn purged_cv_tool_returns_folds() {
        let tool = purged_cv_tool();
        let result = (tool.handler)(json!({
            "dataset": dataset_value(),
            "n_splits": 3,
            "embargo_rows": 1
        }))
        .unwrap();
        assert_eq!(result.as_array().unwrap().len(), 3);
    }

    #[test]
    fn rl_plan_tool_keeps_holdout_separate() {
        let tool = temporal_rl_plan_tool();
        let result = (tool.handler)(json!({
            "dataset": dataset_value(),
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "seed": 42,
            "transaction_cost_bps_per_unit_turnover": 5.0
        }))
        .unwrap();
        assert_eq!(result["train"]["start"], 0);
        assert_eq!(result["holdout"]["end"], 12);
    }

    #[test]
    fn comparison_tool_returns_report_and_csv() {
        let tool = compare_tool();
        let metrics = vec![ComparisonMetric {
            name: "holdout_sharpe".into(),
            direction: MetricDirection::HigherBetter,
            unit: "ratio".into(),
        }];
        let candidates = vec![CandidateEvidence {
            candidate_id: "baseline".into(),
            family: "linear".into(),
            metrics: BTreeMap::from([("holdout_sharpe".into(), 0.4)]),
            evidence_refs: vec!["baseline.json".into()],
        }];
        let result = (tool.handler)(json!({"metrics": metrics, "candidates": candidates})).unwrap();
        assert_eq!(result["report"]["pareto_front"][0], "baseline");
        assert!(result["csv"].as_str().unwrap().contains("baseline.json"));
    }
}
