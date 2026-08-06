use crate::schema::{
    ComputeClass, DiscoveryRequest, FeatureCatalog, FeatureFamily, FeatureHypothesis,
    RejectedHypothesis, RuntimeCost, TemporalAvailability,
};
use std::collections::BTreeSet;

fn slug(input: &str) -> String {
    let mut output = String::new();
    let mut previous_separator = false;
    for character in input.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(character.to_ascii_lowercase());
            previous_separator = false;
        } else if !previous_separator && !output.is_empty() {
            output.push('_');
            previous_separator = true;
        }
    }
    while output.ends_with('_') {
        output.pop();
    }
    output
}

fn unary(
    signal: &str,
    suffix: &str,
    family: FeatureFamily,
    expression: String,
    rationale: &str,
    failure_mode: &str,
    cost: RuntimeCost,
) -> FeatureHypothesis {
    FeatureHypothesis {
        id: format!("{}_{}", slug(signal), suffix),
        name: format!("{signal} {suffix}"),
        family,
        expression,
        required_signals: vec![signal.to_string()],
        temporal_availability: TemporalAvailability::CurrentDecision,
        runtime_cost: cost,
        rationale: rationale.to_string(),
        expected_failure_mode: failure_mode.to_string(),
        ablation_group: format!("{}_transforms", slug(signal)),
        deterministic: true,
    }
}

fn temporal(signal: &str, suffix: &str, expression: String, state: u32) -> FeatureHypothesis {
    FeatureHypothesis {
        id: format!("{}_{}", slug(signal), suffix),
        name: format!("{signal} {suffix}"),
        family: FeatureFamily::TemporalDelta,
        expression,
        required_signals: vec![signal.to_string()],
        temporal_availability: TemporalAvailability::PastOnly,
        runtime_cost: RuntimeCost {
            compute_class: ComputeClass::Constant,
            estimated_scalar_ops: 4,
            persistent_state_scalars: state,
            temporary_state_scalars: 1,
        },
        rationale: "Detects abrupt local changes hidden by the current feature level.".to_string(),
        expected_failure_mode: "A cache skip becomes unsafe after a sudden state transition.".to_string(),
        ablation_group: format!("{}_temporal", slug(signal)),
        deterministic: true,
    }
}

fn interaction(left: &str, right: &str) -> FeatureHypothesis {
    FeatureHypothesis {
        id: format!("{}_x_{}", slug(left), slug(right)),
        name: format!("{left} × {right}"),
        family: FeatureFamily::CrossSignalInteraction,
        expression: format!("{left} * {right}"),
        required_signals: vec![left.to_string(), right.to_string()],
        temporal_availability: TemporalAvailability::CurrentDecision,
        runtime_cost: RuntimeCost::constant(1),
        rationale: "Represents conditional risk that is invisible in either signal alone.".to_string(),
        expected_failure_mode: "A moderate signal becomes dangerous only under a second runtime condition.".to_string(),
        ablation_group: "pairwise_interactions".to_string(),
        deterministic: true,
    }
}

pub fn generate_catalog(request: &DiscoveryRequest) -> Result<FeatureCatalog, String> {
    request.validate()?;

    let available: BTreeSet<&str> = request
        .base_features
        .iter()
        .chain(request.available_signals.iter())
        .map(String::as_str)
        .collect();

    let mut hypotheses = Vec::new();
    let mut rejected = Vec::new();

    for signal in &request.base_features {
        hypotheses.push(unary(
            signal,
            "square",
            FeatureFamily::DistributionShape,
            format!("{signal} * {signal}"),
            "Exposes nonlinear risk while preserving sign-independent magnitude.",
            "Unsafe regions may occur at both positive and negative extremes.",
            RuntimeCost::constant(1),
        ));
        hypotheses.push(unary(
            signal,
            "abs",
            FeatureFamily::DistributionShape,
            format!("abs({signal})"),
            "Separates magnitude from direction.",
            "The sign can be unstable while the magnitude predicts divergence.",
            RuntimeCost::constant(1),
        ));
        hypotheses.push(temporal(
            signal,
            "delta_1",
            format!("{signal}[t] - {signal}[t-1]"),
            1,
        ));
        hypotheses.push(temporal(
            signal,
            "ema_residual_4",
            format!("{signal}[t] - ema_4({signal})"),
            1,
        ));
    }

    let interaction_priority = [
        ("drift", "cache_age"),
        ("drift", "head_std"),
        ("drift", "skip_margin"),
        ("worsening", "cache_age"),
        ("untracked_mass", "head_std"),
        ("remaining_masked_fraction", "profile_gamma"),
        ("generation_progress", "drift"),
        ("generation_progress", "skip_margin"),
        ("tracked_token_fraction", "untracked_mass"),
        ("nfe_progress", "head_std"),
    ];

    for (left, right) in interaction_priority {
        if available.contains(left) && available.contains(right) {
            hypotheses.push(interaction(left, right));
        } else {
            rejected.push(RejectedHypothesis {
                id: format!("{}_x_{}", slug(left), slug(right)),
                reason: format!("missing required runtime signal `{left}` or `{right}`"),
            });
        }
    }

    let distribution_signals = [
        ("logit_entropy", ComputeClass::LinearTokens),
        ("top1_top2_margin", ComputeClass::LinearTokens),
        ("topk_jaccard", ComputeClass::LinearTokens),
        ("attention_entropy", ComputeClass::LinearTokens),
        ("inter_head_disagreement", ComputeClass::LinearHeads),
        ("q_cache_residual", ComputeClass::LinearTokens),
        ("k_cache_residual", ComputeClass::LinearTokens),
        ("v_cache_residual", ComputeClass::LinearTokens),
    ];

    for (signal, compute_class) in distribution_signals {
        if available.contains(signal) {
            hypotheses.push(FeatureHypothesis {
                id: slug(signal),
                name: signal.replace('_', " "),
                family: FeatureFamily::Stability,
                expression: signal.to_string(),
                required_signals: vec![signal.to_string()],
                temporal_availability: TemporalAvailability::CurrentDecision,
                runtime_cost: RuntimeCost {
                    compute_class,
                    estimated_scalar_ops: 0,
                    persistent_state_scalars: 0,
                    temporary_state_scalars: 4,
                },
                rationale: "Directly measures output-distribution or cache-state stability.".to_string(),
                expected_failure_mode: "Attention similarity remains high while token probabilities or cached states diverge.".to_string(),
                ablation_group: "model_state_stability".to_string(),
                deterministic: true,
            });
        } else {
            rejected.push(RejectedHypothesis {
                id: slug(signal),
                reason: format!("runtime instrumentation does not expose `{signal}` yet"),
            });
        }
    }

    hypotheses.sort_by(|left, right| left.id.cmp(&right.id));
    hypotheses.dedup_by(|left, right| left.id == right.id);
    hypotheses.truncate(request.max_hypotheses);

    let catalog = FeatureCatalog {
        schema_version: 1,
        experiment_id: request.experiment_id.clone(),
        evidence_boundary: request.evidence_boundary.clone(),
        hypotheses,
        rejected,
    };
    catalog.validate()?;
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EvidenceBoundary;

    fn request() -> DiscoveryRequest {
        DiscoveryRequest {
            schema_version: 1,
            experiment_id: "elastic-cache-development".to_string(),
            base_features: vec![
                "drift".to_string(),
                "cache_age".to_string(),
                "skip_margin".to_string(),
            ],
            available_signals: vec!["logit_entropy".to_string()],
            observed_false_positive_ids: vec!["holdout-history-1".to_string()],
            evidence_boundary: EvidenceBoundary::default(),
            max_hypotheses: 128,
        }
    }

    #[test]
    fn repeated_catalogs_are_identical() {
        let left = generate_catalog(&request()).unwrap();
        let right = generate_catalog(&request()).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn generated_hypotheses_are_runtime_safe() {
        let catalog = generate_catalog(&request()).unwrap();
        assert!(!catalog.hypotheses.is_empty());
        assert!(catalog
            .hypotheses
            .iter()
            .all(|hypothesis| hypothesis.temporal_availability.is_runtime_safe()));
    }

    #[test]
    fn missing_instrumentation_is_reported_not_invented() {
        let catalog = generate_catalog(&request()).unwrap();
        assert!(catalog
            .rejected
            .iter()
            .any(|item| item.id == "top1_top2_margin"));
    }
}
