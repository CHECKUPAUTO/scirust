use super::data::{CandidateRecord, RUNTIME_FEATURES};
use scirust_sequential::{FeatureFn, LinearChainCRF};
use serde::Serialize;
use std::sync::Arc;

const TAG_SAFE: usize = 0;
const TAG_UNSAFE: usize = 1;
const TAGS: usize = 2;
const CRF_FEATURES: usize = 1 + RUNTIME_FEATURES + TAGS * TAGS;

#[derive(Debug, Clone, Serialize)]
pub struct SequentialRiskReport {
    pub model: String,
    pub tags: Vec<String>,
    pub feature_count: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2_penalty: f64,
    pub training_negative_log_likelihood: f64,
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SequentialRiskModel {
    crf: LinearChainCRF,
    feature_table: Arc<Vec<[f64; RUNTIME_FEATURES]>>,
    pub report: SequentialRiskReport,
}

fn feature_functions(
    table: &Arc<Vec<[f64; RUNTIME_FEATURES]>>,
) -> Vec<Box<FeatureFn>> {
    let mut features: Vec<Box<FeatureFn>> = Vec::with_capacity(CRF_FEATURES);
    features.push(Box::new(|_, current, _| {
        if current == TAG_UNSAFE { 1.0 } else { 0.0 }
    }));
    for feature_index in 0..RUNTIME_FEATURES {
        let table = Arc::clone(table);
        features.push(Box::new(move |_, current, observation| {
            if current == TAG_UNSAFE {
                table[observation][feature_index]
            } else {
                0.0
            }
        }));
    }
    for previous in 0..TAGS {
        for current in 0..TAGS {
            features.push(Box::new(move |observed_previous, observed_current, _| {
                if observed_previous == Some(previous) && observed_current == current {
                    1.0
                } else {
                    0.0
                }
            }));
        }
    }
    debug_assert_eq!(features.len(), CRF_FEATURES);
    features
}

fn tags_for_sequence(rows: &[CandidateRecord], sequence: &[usize]) -> Vec<usize> {
    sequence
        .iter()
        .map(|index| {
            if rows[*index].strict_unsafe {
                TAG_UNSAFE
            } else {
                TAG_SAFE
            }
        })
        .collect()
}

fn dataset_nll(
    crf: &LinearChainCRF,
    table: &Arc<Vec<[f64; RUNTIME_FEATURES]>>,
    rows: &[CandidateRecord],
    sequences: &[Vec<usize>],
) -> f64 {
    if sequences.is_empty() {
        return f64::INFINITY;
    }
    let features = feature_functions(table);
    sequences
        .iter()
        .map(|sequence| crf.nll(sequence, &tags_for_sequence(rows, sequence), &features))
        .sum::<f64>()
        / sequences.len() as f64
}

pub fn fit_sequential_risk(
    rows: &[CandidateRecord],
    standardized_features: Vec<[f64; RUNTIME_FEATURES]>,
    training_sequences: &[Vec<usize>],
    epochs: usize,
    learning_rate: f64,
    l2_penalty: f64,
) -> Result<SequentialRiskModel, String> {
    if training_sequences.is_empty() {
        return Err("cannot fit the sequential risk model without sequences".to_string());
    }
    if epochs == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("invalid CRF optimization configuration".to_string());
    }
    if !l2_penalty.is_finite() || l2_penalty < 0.0 {
        return Err("CRF L2 penalty must be finite and non-negative".to_string());
    }
    if standardized_features.len() != rows.len() {
        return Err("CRF feature table length does not match trajectory rows".to_string());
    }

    let table = Arc::new(standardized_features);
    let unsafe_count = training_sequences
        .iter()
        .flatten()
        .filter(|index| rows[**index].strict_unsafe)
        .count();
    let total = training_sequences.iter().map(Vec::len).sum::<usize>();
    if total == 0 {
        return Err("CRF training sequences are empty".to_string());
    }
    let prior = ((unsafe_count as f64 + 0.5) / (total as f64 + 1.0))
        .clamp(1e-6, 1.0 - 1e-6);
    let mut crf = LinearChainCRF::new(TAGS, CRF_FEATURES);
    crf.weights[0] = (prior / (1.0 - prior)).ln();
    let features = feature_functions(&table);

    for epoch in 0..epochs {
        let mut gradient = vec![0.0; CRF_FEATURES];
        for sequence in training_sequences {
            if sequence.is_empty() {
                continue;
            }
            let gold = tags_for_sequence(rows, sequence);
            let sequence_gradient = crf.gradient(sequence, &gold, &features);
            for (target, value) in gradient.iter_mut().zip(sequence_gradient) {
                *target += value;
            }
        }
        let count = training_sequences.len() as f64;
        let rate = learning_rate / (1.0 + epoch as f64 / 50.0).sqrt();
        for (index, weight) in crf.weights.iter_mut().enumerate() {
            let regularization = if index < 1 + RUNTIME_FEATURES {
                l2_penalty * *weight
            } else {
                0.25 * l2_penalty * *weight
            };
            let value = (gradient[index] / count + regularization).clamp(-100.0, 100.0);
            *weight = (*weight - rate * value).clamp(-24.0, 24.0);
        }
        if crf.weights.iter().any(|weight| !weight.is_finite()) {
            return Err(format!("CRF optimization became non-finite at epoch {epoch}"));
        }
    }

    let training_nll = dataset_nll(&crf, &table, rows, training_sequences);
    if !training_nll.is_finite() {
        return Err("CRF training produced a non-finite likelihood".to_string());
    }
    let report = SequentialRiskReport {
        model: "scirust-sequential linear-chain CRF".to_string(),
        tags: vec!["safe".to_string(), "strict_unsafe".to_string()],
        feature_count: CRF_FEATURES,
        epochs,
        learning_rate,
        l2_penalty,
        training_negative_log_likelihood: training_nll,
        weights: crf.weights.clone(),
    };
    Ok(SequentialRiskModel {
        crf,
        feature_table: table,
        report,
    })
}

impl SequentialRiskModel {
    pub fn unsafe_probabilities(
        &self,
        sequences: &[Vec<usize>],
        row_count: usize,
    ) -> Result<Vec<f64>, String> {
        let features = feature_functions(&self.feature_table);
        let mut probabilities = vec![f64::NAN; row_count];
        for sequence in sequences {
            if sequence.is_empty() {
                continue;
            }
            let (alpha, beta, log_partition) = self.crf.forward_backward(sequence, &features);
            if !log_partition.is_finite() {
                return Err("CRF produced a non-finite partition function".to_string());
            }
            for (step, row_index) in sequence.iter().enumerate() {
                let log_probability = alpha[step * TAGS + TAG_UNSAFE]
                    + beta[step * TAGS + TAG_UNSAFE]
                    - log_partition;
                probabilities[*row_index] = log_probability.exp().clamp(0.0, 1.0);
            }
        }
        Ok(probabilities)
    }
}
