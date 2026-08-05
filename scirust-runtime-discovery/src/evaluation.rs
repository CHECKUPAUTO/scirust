use crate::{FeatureHypothesis, ProposalReview, SignalHistory, evaluate_expression};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureEvaluation {
    pub hypothesis_id: String,
    pub expression: String,
    pub ablation_group: String,
    pub evaluated_rows: usize,
    pub skipped_rows: usize,
    pub safe_rows: usize,
    pub unsafe_rows: usize,
    pub safe_mean: f64,
    pub unsafe_mean: f64,
    pub standardized_mean_difference: f64,
    pub unsafe_auc: f64,
    pub discrimination_auc: f64,
    pub unsafe_direction: String,
    pub profitable_safe_rows: usize,
    pub evaluation_errors: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationGroupSummary {
    pub group: String,
    pub hypotheses: usize,
    pub best_hypothesis_id: String,
    pub best_discrimination_auc: f64,
    pub best_absolute_standardized_mean_difference: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetEvaluationReport {
    pub schema_version: u32,
    pub experiment_id: String,
    pub source_dataset: String,
    pub rows_read: usize,
    pub prompts: usize,
    pub semantic_unsafe_rows: usize,
    pub semantically_safe_rows: usize,
    pub profitable_semantically_safe_rows: usize,
    pub evaluations: Vec<FeatureEvaluation>,
    pub ablation_groups: Vec<AblationGroupSummary>,
}

#[derive(Debug, Clone)]
struct DatasetRow {
    prompt_id: u64,
    ordinal: u64,
    signals: BTreeMap<String, f64>,
    semantic_unsafe: bool,
    profitable_safe: bool,
}

pub fn evaluate_review_on_jsonl(
    review: &ProposalReview,
    dataset_path: &Path,
) -> Result<DatasetEvaluationReport, String> {
    let mut rows = read_rows(dataset_path)?;
    rows.sort_by_key(|row| (row.prompt_id, row.ordinal));

    let prompts = rows.iter().map(|row| row.prompt_id).collect::<BTreeSet<_>>().len();
    let semantic_unsafe_rows = rows.iter().filter(|row| row.semantic_unsafe).count();
    let profitable_semantically_safe_rows = rows.iter().filter(|row| row.profitable_safe).count();
    let hypotheses = review
        .accepted
        .iter()
        .map(|ranked| &ranked.hypothesis)
        .collect::<Vec<_>>();

    let mut observations: BTreeMap<String, Vec<(f64, bool, bool)>> = hypotheses
        .iter()
        .map(|hypothesis| (hypothesis.id.clone(), Vec::new()))
        .collect();
    let mut errors: BTreeMap<String, BTreeMap<String, usize>> = hypotheses
        .iter()
        .map(|hypothesis| (hypothesis.id.clone(), BTreeMap::new()))
        .collect();
    let mut prompt_history: BTreeMap<u64, Vec<BTreeMap<String, f64>>> = BTreeMap::new();

    for row in &rows {
        let previous = prompt_history.entry(row.prompt_id).or_default();
        let past = previous.iter().rev().cloned().collect::<Vec<_>>();
        let history = SignalHistory::new(row.signals.clone(), past);

        for hypothesis in &hypotheses {
            match evaluate_expression(&hypothesis.expression, &history) {
                Ok(value) => observations
                    .get_mut(&hypothesis.id)
                    .expect("hypothesis observation entry")
                    .push((value, row.semantic_unsafe, row.profitable_safe)),
                Err(error) => {
                    *errors
                        .get_mut(&hypothesis.id)
                        .expect("hypothesis error entry")
                        .entry(error)
                        .or_insert(0) += 1;
                },
            }
        }
        previous.push(row.signals.clone());
    }

    let mut evaluations = hypotheses
        .iter()
        .map(|hypothesis| {
            summarize_feature(
                hypothesis,
                observations.remove(&hypothesis.id).unwrap_or_default(),
                errors.remove(&hypothesis.id).unwrap_or_default(),
                rows.len(),
            )
        })
        .collect::<Vec<_>>();
    evaluations.sort_by(|left, right| {
        right
            .discrimination_auc
            .partial_cmp(&left.discrimination_auc)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .standardized_mean_difference
                    .abs()
                    .partial_cmp(&left.standardized_mean_difference.abs())
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.hypothesis_id.cmp(&right.hypothesis_id))
    });

    let ablation_groups = summarize_ablation_groups(&evaluations);
    Ok(DatasetEvaluationReport {
        schema_version: 1,
        experiment_id: review.experiment_id.clone(),
        source_dataset: dataset_path.display().to_string(),
        rows_read: rows.len(),
        prompts,
        semantic_unsafe_rows,
        semantically_safe_rows: rows.len().saturating_sub(semantic_unsafe_rows),
        profitable_semantically_safe_rows,
        evaluations,
        ablation_groups,
    })
}

fn read_rows(path: &Path) -> Result<Vec<DatasetRow>, String> {
    let file = File::open(path)
        .map_err(|error| format!("cannot open dataset {}: {error}", path.display()))?;
    let mut rows = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("cannot read line {}: {error}", line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| format!("invalid JSON at line {}: {error}", line_index + 1))?;
        rows.push(parse_row(&value, line_index + 1)?);
    }
    if rows.is_empty() {
        return Err("dataset contains no rows".to_string());
    }
    Ok(rows)
}

fn parse_row(value: &Value, line: usize) -> Result<DatasetRow, String> {
    let prompt_id = value
        .get("dataset_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("line {line}: missing dataset_index"))?;
    let candidate = value
        .get("candidate")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("line {line}: missing candidate object"))?;
    let ordinal = candidate
        .get("ordinal")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("line {line}: missing candidate ordinal"))?;

    let labels = value
        .get("labels")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("line {line}: missing labels object"))?;
    let prediction_invariant = bool_field(labels, "prediction_invariant", line)?;
    let correctness_invariant = bool_field(labels, "correctness_invariant", line)?;
    let quality_regression = bool_field(labels, "quality_regression", line)?;
    let semantic_unsafe = !prediction_invariant || !correctness_invariant || quality_regression;
    let refresh_saving = value
        .pointer("/effects/refresh_cost_improvement")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("line {line}: missing refresh_cost_improvement"))?;

    let mut signals = BTreeMap::new();
    flatten_numeric_object(candidate.get("features"), &mut signals);
    flatten_numeric_object(candidate.get("runtime_context"), &mut signals);
    flatten_numeric_object(value.get("generation_profile"), &mut signals);
    for name in ["skip_margin", "refresh_cost", "votes", "ordinal", "layer_id"] {
        if let Some(number) = candidate.get(name).and_then(Value::as_f64) {
            signals.insert(name.to_string(), number);
        }
    }
    add_derived_signals(&mut signals);

    Ok(DatasetRow {
        prompt_id,
        ordinal,
        signals,
        semantic_unsafe,
        profitable_safe: !semantic_unsafe && refresh_saving > 0.0,
    })
}

fn bool_field(
    object: &serde_json::Map<String, Value>,
    name: &str,
    line: usize,
) -> Result<bool, String> {
    object
        .get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("line {line}: missing boolean label `{name}`"))
}

fn flatten_numeric_object(value: Option<&Value>, target: &mut BTreeMap<String, f64>) {
    if let Some(object) = value.and_then(Value::as_object) {
        for (name, value) in object {
            if let Some(number) = value.as_f64() {
                target.insert(name.clone(), number);
            }
        }
    }
}

fn add_derived_signals(signals: &mut BTreeMap<String, f64>) {
    let sequence = signals.get("sequence_length").copied().unwrap_or(0.0).max(1.0);
    let generation = signals.get("generation_length").copied().unwrap_or(0.0).max(1.0);
    let max_new = signals.get("max_new_tokens").copied().unwrap_or(0.0).max(1.0);
    let derived = [
        ("generation_progress", signals.get("generation_step").copied().unwrap_or(0.0) / max_new),
        ("nfe_progress", signals.get("nfe").copied().unwrap_or(0.0) / max_new),
        ("remaining_masked_fraction", signals.get("remaining_masked_tokens").copied().unwrap_or(0.0) / generation),
        ("query_masked_fraction", signals.get("query_masked_tokens").copied().unwrap_or(0.0) / generation),
        ("tracked_token_fraction", signals.get("tracked_tokens").copied().unwrap_or(0.0) / sequence),
        ("runtime_window_fraction", signals.get("window_length").copied().unwrap_or(0.0) / generation),
        ("profile_gamma", signals.get("gamma").copied().unwrap_or(0.0)),
        ("profile_track_fraction", signals.get("track_num").copied().unwrap_or(0.0) / sequence),
        ("profile_max_new_tokens_fraction", signals.get("max_new_tokens").copied().unwrap_or(0.0) / sequence),
        ("profile_window_fraction", signals.get("window_length").copied().unwrap_or(0.0) / generation),
    ];
    for (name, value) in derived {
        signals.insert(name.to_string(), value.clamp(0.0, 1.0));
    }
}

fn summarize_feature(
    hypothesis: &FeatureHypothesis,
    observations: Vec<(f64, bool, bool)>,
    evaluation_errors: BTreeMap<String, usize>,
    total_rows: usize,
) -> FeatureEvaluation {
    let safe = observations
        .iter()
        .filter(|(_, unsafe_label, _)| !unsafe_label)
        .map(|(value, _, _)| *value)
        .collect::<Vec<_>>();
    let unsafe_values = observations
        .iter()
        .filter(|(_, unsafe_label, _)| *unsafe_label)
        .map(|(value, _, _)| *value)
        .collect::<Vec<_>>();
    let safe_mean = mean(&safe);
    let unsafe_mean = mean(&unsafe_values);
    let standardized_mean_difference = standardized_difference(&safe, &unsafe_values);
    let unsafe_auc = auc(&observations);
    let discrimination_auc = unsafe_auc.max(1.0 - unsafe_auc);

    FeatureEvaluation {
        hypothesis_id: hypothesis.id.clone(),
        expression: hypothesis.expression.clone(),
        ablation_group: hypothesis.ablation_group.clone(),
        evaluated_rows: observations.len(),
        skipped_rows: total_rows.saturating_sub(observations.len()),
        safe_rows: safe.len(),
        unsafe_rows: unsafe_values.len(),
        safe_mean,
        unsafe_mean,
        standardized_mean_difference,
        unsafe_auc,
        discrimination_auc,
        unsafe_direction: if unsafe_auc >= 0.5 { "higher" } else { "lower" }.to_string(),
        profitable_safe_rows: observations.iter().filter(|(_, _, profitable)| *profitable).count(),
        evaluation_errors,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn variance(values: &[f64], center: f64) -> f64 {
    if values.len() < 2 {
        0.0
    } else {
        values.iter().map(|value| (value - center).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }
}

fn standardized_difference(left: &[f64], right: &[f64]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let left_mean = mean(left);
    let right_mean = mean(right);
    let denominator = ((variance(left, left_mean) + variance(right, right_mean)) / 2.0).sqrt();
    if denominator <= f64::EPSILON { 0.0 } else { (right_mean - left_mean) / denominator }
}

fn auc(observations: &[(f64, bool, bool)]) -> f64 {
    let positives = observations.iter().filter(|(_, label, _)| *label).count();
    let negatives = observations.len().saturating_sub(positives);
    if positives == 0 || negatives == 0 {
        return 0.5;
    }
    let mut ranked = observations.iter().map(|(value, label, _)| (*value, *label)).collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let mut rank_sum = 0.0;
    let mut index = 0;
    while index < ranked.len() {
        let mut end = index + 1;
        while end < ranked.len() && ranked[end].0 == ranked[index].0 {
            end += 1;
        }
        let average_rank = ((index + 1 + end) as f64) / 2.0;
        rank_sum += ranked[index..end]
            .iter()
            .filter(|(_, label)| *label)
            .count() as f64
            * average_rank;
        index = end;
    }
    (rank_sum - positives as f64 * (positives as f64 + 1.0) / 2.0)
        / (positives as f64 * negatives as f64)
}

fn summarize_ablation_groups(evaluations: &[FeatureEvaluation]) -> Vec<AblationGroupSummary> {
    let mut groups: BTreeMap<String, Vec<&FeatureEvaluation>> = BTreeMap::new();
    for evaluation in evaluations {
        groups.entry(evaluation.ablation_group.clone()).or_default().push(evaluation);
    }
    groups
        .into_iter()
        .map(|(group, values)| {
            let best = values
                .iter()
                .copied()
                .max_by(|left, right| {
                    left.discrimination_auc
                        .partial_cmp(&right.discrimination_auc)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| right.hypothesis_id.cmp(&left.hypothesis_id))
                })
                .expect("non-empty group");
            AblationGroupSummary {
                group,
                hypotheses: values.len(),
                best_hypothesis_id: best.hypothesis_id.clone(),
                best_discrimination_auc: best.discrimination_auc,
                best_absolute_standardized_mean_difference: values
                    .iter()
                    .map(|value| value.standardized_mean_difference.abs())
                    .fold(0.0, f64::max),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureFamily, RankedHypothesis, RuntimeCost, TemporalAvailability};
    use std::io::Write;

    #[test]
    fn evaluates_current_and_temporal_features() {
        let path = std::env::temp_dir().join(format!("runtime-discovery-{}.jsonl", std::process::id()));
        let mut file = File::create(&path).unwrap();
        for (ordinal, drift, unsafe_label) in [(1, 0.1, false), (2, 0.2, false), (3, 0.9, true)] {
            writeln!(file, "{{\"dataset_index\":1,\"candidate\":{{\"ordinal\":{ordinal},\"skip_margin\":0.2,\"refresh_cost\":0.9,\"features\":{{\"drift\":{drift},\"cache_age\":0.1}}}},\"labels\":{{\"prediction_invariant\":{},\"correctness_invariant\":{},\"quality_regression\":{unsafe_label}}},\"effects\":{{\"refresh_cost_improvement\":0.1}}}}", !unsafe_label, !unsafe_label).unwrap();
        }
        let make = |id: &str, expression: &str| RankedHypothesis {
            hypothesis: FeatureHypothesis {
                id: id.to_string(), name: id.to_string(), family: FeatureFamily::TemporalSlope,
                expression: expression.to_string(), required_signals: vec!["drift".to_string()],
                temporal_availability: TemporalAvailability::PastOnly,
                runtime_cost: RuntimeCost::constant(3), rationale: "test".to_string(),
                expected_failure_mode: "test".to_string(), ablation_group: "test".to_string(), deterministic: true,
            }, score: 1.0, novelty_score: 1.0, cost_score: 1.0, observability_score: 1.0,
        };
        let review = ProposalReview { schema_version: 1, experiment_id: "test".to_string(), proposer: "test".to_string(), accepted: vec![make("level", "drift"), make("delta", "drift[t] - drift[t-1]")], rejected: Vec::new() };
        let report = evaluate_review_on_jsonl(&review, &path).unwrap();
        assert_eq!(report.rows_read, 3);
        assert_eq!(report.semantic_unsafe_rows, 1);
        assert_eq!(report.evaluations.len(), 2);
        assert_eq!(report.evaluations[0].discrimination_auc, 1.0);
        std::fs::remove_file(path).unwrap();
    }
}
