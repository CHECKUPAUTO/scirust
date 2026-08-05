use crate::{DatasetEvaluationReport, ProposalReview};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TasteFeedbackKind {
    Accepted,
    Rejected,
    Confirmed,
    Contradicted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificTasteEvent {
    pub schema_version: u16,
    pub preference_id: String,
    pub scope: String,
    pub kind: TasteFeedbackKind,
    pub confidence_bps: u16,
    pub evidence_ids: Vec<String>,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificTasteProfile {
    pub schema_version: u16,
    pub experiment_id: String,
    pub events: Vec<ScientificTasteEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SciAgentResearchTarget {
    pub hypothesis_id: String,
    pub expression: String,
    pub discrimination_auc: f64,
    pub standardized_mean_difference: f64,
    pub coverage: f64,
    pub instruction: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SciAgentResearchBrief {
    pub schema_version: u16,
    pub experiment_id: String,
    pub objective: String,
    pub hard_constraints: Vec<String>,
    pub validated_preferences: Vec<String>,
    pub rejected_patterns: Vec<String>,
    pub targets: Vec<SciAgentResearchTarget>,
}

pub fn derive_scientific_taste(
    report: &DatasetEvaluationReport,
) -> ScientificTasteProfile {
    let mut events = Vec::new();
    for evaluation in &report.evaluations {
        let coverage = if report.rows_read == 0 {
            0.0
        } else {
            evaluation.evaluated_rows as f64 / report.rows_read as f64
        };
        let discrimination = evaluation.discrimination_auc;
        let effect = evaluation.standardized_mean_difference.abs();

        let (kind, confidence_bps, rationale) = if discrimination >= 0.65
            && coverage >= 0.80
            && effect >= 0.20
        {
            (
                TasteFeedbackKind::Confirmed,
                score_to_bps(discrimination, effect, coverage),
                "feature separates semantic risk with broad development coverage",
            )
        } else if discrimination >= 0.58 && coverage >= 0.70 {
            (
                TasteFeedbackKind::Accepted,
                score_to_bps(discrimination, effect, coverage),
                "feature is promising and remains in development quarantine",
            )
        } else if discrimination <= 0.52 && effect < 0.10 {
            (
                TasteFeedbackKind::Rejected,
                8_000,
                "feature adds negligible univariate discrimination",
            )
        } else {
            (
                TasteFeedbackKind::Contradicted,
                5_000,
                "feature evidence is mixed or coverage is insufficient",
            )
        };

        events.push(ScientificTasteEvent {
            schema_version: 1,
            preference_id: format!("feature:{}", evaluation.hypothesis_id),
            scope: "project:elastic-cache".to_string(),
            kind,
            confidence_bps,
            evidence_ids: vec![format!(
                "dataset:{}:rows:{}",
                report.experiment_id, report.rows_read
            )],
            rationale: rationale.to_string(),
        });
    }
    events.sort_by(|left, right| left.preference_id.cmp(&right.preference_id));
    ScientificTasteProfile {
        schema_version: 1,
        experiment_id: report.experiment_id.clone(),
        events,
    }
}

pub fn build_sciagent_research_brief(
    review: &ProposalReview,
    report: &DatasetEvaluationReport,
    maximum_targets: usize,
) -> SciAgentResearchBrief {
    let mut evaluations = report.evaluations.clone();
    evaluations.sort_by(|left, right| {
        right
            .discrimination_auc
            .partial_cmp(&left.discrimination_auc)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.hypothesis_id.cmp(&right.hypothesis_id))
    });

    let targets = evaluations
        .into_iter()
        .take(maximum_targets)
        .map(|evaluation| {
            let coverage = if report.rows_read == 0 {
                0.0
            } else {
                evaluation.evaluated_rows as f64 / report.rows_read as f64
            };
            SciAgentResearchTarget {
                hypothesis_id: evaluation.hypothesis_id,
                expression: evaluation.expression,
                discrimination_auc: evaluation.discrimination_auc,
                standardized_mean_difference: evaluation.standardized_mean_difference,
                coverage,
                instruction: "propose cheaper, more observable refinements and interactions that preserve causal timing"
                    .to_string(),
            }
        })
        .collect();

    let validated_preferences = vec![
        "prefer decision-time or past-only signals".to_string(),
        "prefer deterministic constant-time expressions".to_string(),
        "separate semantic safety from trajectory stability".to_string(),
        "reuse existing development data before requesting new GPU runs".to_string(),
        "require provenance, ablation groups, and explicit failure modes".to_string(),
    ];
    let rejected_patterns = review
        .rejected
        .iter()
        .map(|item| format!("{}: {}", item.id, item.reason))
        .collect();

    SciAgentResearchBrief {
        schema_version: 1,
        experiment_id: report.experiment_id.clone(),
        objective: "discover runtime features that reduce semantic false-safe decisions under strict cost and leakage constraints"
            .to_string(),
        hard_constraints: vec![
            "never use future-dependent or task-outcome-dependent signals".to_string(),
            "never grant model-generated proposals runtime capabilities".to_string(),
            "do not read untouched confirmation data during development".to_string(),
            "hard safety constraints are lexicographic and non-compensable".to_string(),
        ],
        validated_preferences,
        rejected_patterns,
        targets,
    }
}

fn score_to_bps(discrimination: f64, effect: f64, coverage: f64) -> u16 {
    let score = (0.55 * discrimination + 0.25 * effect.min(1.0) + 0.20 * coverage)
        .clamp(0.0, 1.0);
    (score * 10_000.0).round() as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AblationGroupSummary, FeatureEvaluation};
    use std::collections::BTreeMap;

    fn report() -> DatasetEvaluationReport {
        DatasetEvaluationReport {
            schema_version: 1,
            experiment_id: "test".to_string(),
            source_dataset: "dataset.jsonl".to_string(),
            rows_read: 100,
            prompts: 20,
            semantic_unsafe_rows: 10,
            semantically_safe_rows: 90,
            profitable_semantically_safe_rows: 70,
            evaluations: vec![FeatureEvaluation {
                hypothesis_id: "h1".to_string(),
                expression: "abs(drift)".to_string(),
                ablation_group: "g".to_string(),
                evaluated_rows: 100,
                skipped_rows: 0,
                safe_rows: 90,
                unsafe_rows: 10,
                safe_mean: 0.1,
                unsafe_mean: 0.4,
                standardized_mean_difference: 0.8,
                unsafe_auc: 0.72,
                discrimination_auc: 0.72,
                unsafe_direction: "higher".to_string(),
                profitable_safe_rows: 70,
                evaluation_errors: BTreeMap::new(),
            }],
            ablation_groups: vec![AblationGroupSummary {
                group: "g".to_string(),
                hypotheses: 1,
                best_hypothesis_id: "h1".to_string(),
                best_discrimination_auc: 0.72,
                best_absolute_standardized_mean_difference: 0.8,
            }],
        }
    }

    #[test]
    fn strong_feature_becomes_confirmed_taste_event() {
        let profile = derive_scientific_taste(&report());
        assert_eq!(profile.events[0].kind, TasteFeedbackKind::Confirmed);
    }
}
