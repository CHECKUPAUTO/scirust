use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const RUNTIME_FEATURES: usize = 11;
pub const RUNTIME_FEATURE_NAMES: [&str; RUNTIME_FEATURES] = [
    "drift",
    "worsening",
    "head_std",
    "cache_age",
    "untracked_mass",
    "layer_fraction",
    "drift_age",
    "refresh_cost",
    "skip_margin",
    "vote_fraction",
    "candidate_ordinal",
];

#[derive(Debug, Clone, Deserialize)]
struct RawFeatures {
    drift: f64,
    worsening: f64,
    head_std: f64,
    cache_age: f64,
    untracked_mass: f64,
    layer_fraction: f64,
    drift_age: f64,
    refresh_cost: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RawCandidate {
    ordinal: usize,
    layer_id: usize,
    skip_margin: f64,
    refresh_cost: f64,
    votes: Option<usize>,
    features: RawFeatures,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRun {
    correct: bool,
    prediction: Option<String>,
    gold: String,
    elapsed_seconds: f64,
    decisions: usize,
    refreshes: usize,
    refresh_cost: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RawLabels {
    exact_response_invariant: bool,
    prediction_invariant: bool,
    correctness_invariant: bool,
    quality_regression: bool,
    quality_improvement: bool,
    decision_count_invariant: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RawEffects {
    decision_delta: i64,
    decision_ratio: f64,
    latency_improvement: f64,
    refresh_cost_improvement: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRow {
    schema_version: u32,
    split: String,
    dataset_index: usize,
    generation_seed: u64,
    candidate: RawCandidate,
    baseline: RawRun,
    single_skip: RawRun,
    labels: RawLabels,
    effects: RawEffects,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateRecord {
    pub prompt_id: usize,
    pub generation_seed: u64,
    pub ordinal: usize,
    pub layer_id: usize,
    pub raw_features: [f64; RUNTIME_FEATURES],
    pub skip_margin: f64,
    pub votes: usize,
    pub strict_unsafe: bool,
    pub semantic_unsafe: bool,
    pub trajectory_unsafe: bool,
    pub quality_regression: bool,
    pub quality_improvement: bool,
    pub prediction_changed: bool,
    pub response_changed: bool,
    pub decision_count_changed: bool,
    pub baseline_correct: bool,
    pub branch_correct: bool,
    pub baseline_prediction: Option<String>,
    pub branch_prediction: Option<String>,
    pub gold: String,
    pub baseline_decisions: usize,
    pub branch_decisions: usize,
    pub branch_skips: usize,
    pub decision_delta: i64,
    pub decision_ratio: f64,
    pub latency_improvement: f64,
    pub refresh_cost_improvement: f64,
    pub saved_refresh_cost: f64,
}

impl CandidateRecord {
    fn from_raw(raw: RawRow) -> Result<Self, String> {
        if raw.schema_version != 1
        {
            return Err(format!(
                "unsupported trajectory row schema {}",
                raw.schema_version
            ));
        }
        if raw.split != "gsm8k_train"
        {
            return Err(format!(
                "trajectory discovery accepts GSM8K train rows only, found `{}`",
                raw.split
            ));
        }
        let finite = [
            raw.candidate.skip_margin,
            raw.candidate.refresh_cost,
            raw.candidate.features.drift,
            raw.candidate.features.worsening,
            raw.candidate.features.head_std,
            raw.candidate.features.cache_age,
            raw.candidate.features.untracked_mass,
            raw.candidate.features.layer_fraction,
            raw.candidate.features.drift_age,
            raw.candidate.features.refresh_cost,
            raw.baseline.elapsed_seconds,
            raw.single_skip.elapsed_seconds,
            raw.baseline.refresh_cost,
            raw.single_skip.refresh_cost,
            raw.effects.decision_ratio,
            raw.effects.latency_improvement,
            raw.effects.refresh_cost_improvement,
        ];
        if finite.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "non-finite trajectory value for prompt {} candidate {}",
                raw.dataset_index, raw.candidate.ordinal
            ));
        }
        if raw.candidate.ordinal == 0
        {
            return Err("candidate ordinals must be one-based".to_string());
        }
        if raw.baseline.decisions == 0 || raw.single_skip.decisions == 0
        {
            return Err("trajectory rows must contain at least one decision".to_string());
        }
        if raw.baseline.refreshes != raw.baseline.decisions
        {
            return Err("baseline row is not an always-refresh trajectory".to_string());
        }
        if raw.single_skip.refreshes > raw.single_skip.decisions
        {
            return Err("branch refresh count exceeds its decision count".to_string());
        }
        if raw.candidate.refresh_cost.to_bits() != raw.candidate.features.refresh_cost.to_bits()
        {
            return Err("candidate refresh-cost fields disagree".to_string());
        }

        let votes = raw.candidate.votes.unwrap_or(5);
        let response_changed = !raw.labels.exact_response_invariant;
        let prediction_changed = !raw.labels.prediction_invariant;
        let decision_count_changed = !raw.labels.decision_count_invariant;
        let semantic_unsafe = prediction_changed
            || !raw.labels.correctness_invariant
            || raw.labels.quality_regression;
        let trajectory_unsafe = response_changed || decision_count_changed;
        let strict_unsafe = semantic_unsafe || trajectory_unsafe;
        let ordinal_feature = (raw.candidate.ordinal as f64 / 4.0).clamp(0.0, 1.0);
        let vote_fraction = (votes as f64 / 5.0).clamp(0.0, 1.0);
        let raw_features = [
            raw.candidate.features.drift,
            raw.candidate.features.worsening,
            raw.candidate.features.head_std,
            raw.candidate.features.cache_age,
            raw.candidate.features.untracked_mass,
            raw.candidate.features.layer_fraction,
            raw.candidate.features.drift_age,
            raw.candidate.features.refresh_cost,
            raw.candidate.skip_margin,
            vote_fraction,
            ordinal_feature,
        ];

        Ok(Self {
            prompt_id: raw.dataset_index,
            generation_seed: raw.generation_seed,
            ordinal: raw.candidate.ordinal,
            layer_id: raw.candidate.layer_id,
            raw_features,
            skip_margin: raw.candidate.skip_margin,
            votes,
            strict_unsafe,
            semantic_unsafe,
            trajectory_unsafe,
            quality_regression: raw.labels.quality_regression,
            quality_improvement: raw.labels.quality_improvement,
            prediction_changed,
            response_changed,
            decision_count_changed,
            baseline_correct: raw.baseline.correct,
            branch_correct: raw.single_skip.correct,
            baseline_prediction: raw.baseline.prediction,
            branch_prediction: raw.single_skip.prediction,
            gold: raw.baseline.gold,
            baseline_decisions: raw.baseline.decisions,
            branch_decisions: raw.single_skip.decisions,
            branch_skips: raw.single_skip.decisions - raw.single_skip.refreshes,
            decision_delta: raw.effects.decision_delta,
            decision_ratio: raw.effects.decision_ratio,
            latency_improvement: raw.effects.latency_improvement,
            refresh_cost_improvement: raw.effects.refresh_cost_improvement,
            saved_refresh_cost: raw.baseline.refresh_cost - raw.single_skip.refresh_cost,
        })
    }
}

pub fn read_candidate_jsonl(path: &Path) -> Result<Vec<CandidateRecord>, String> {
    let file = File::open(path)
        .map_err(|error| format!("cannot open trajectory dataset {}: {error}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for (line_index, line) in reader.lines().enumerate()
    {
        let line = line.map_err(|error| {
            format!(
                "cannot read trajectory dataset {} line {}: {error}",
                path.display(),
                line_index + 1
            )
        })?;
        if line.trim().is_empty()
        {
            continue;
        }
        let raw: RawRow = serde_json::from_str(&line).map_err(|error| {
            format!(
                "invalid trajectory JSON at {} line {}: {error}",
                path.display(),
                line_index + 1
            )
        })?;
        let row = CandidateRecord::from_raw(raw)?;
        if !seen.insert((row.prompt_id, row.ordinal))
        {
            return Err(format!(
                "duplicate prompt/candidate pair ({}, {})",
                row.prompt_id, row.ordinal
            ));
        }
        rows.push(row);
    }
    if rows.is_empty()
    {
        return Err("trajectory candidate dataset is empty".to_string());
    }
    rows.sort_by_key(|row| (row.prompt_id, row.ordinal));
    Ok(rows)
}

#[derive(Debug, Clone, Serialize)]
pub struct Standardizer {
    pub mean: [f64; RUNTIME_FEATURES],
    pub scale: [f64; RUNTIME_FEATURES],
}

impl Standardizer {
    pub fn fit(rows: &[CandidateRecord], indices: &[usize]) -> Result<Self, String> {
        if indices.is_empty()
        {
            return Err("cannot fit a feature standardizer on an empty split".to_string());
        }
        let mut mean = [0.0; RUNTIME_FEATURES];
        for &index in indices
        {
            for (slot, value) in mean.iter_mut().zip(rows[index].raw_features)
            {
                *slot += value;
            }
        }
        for value in &mut mean
        {
            *value /= indices.len() as f64;
        }
        let mut variance = [0.0; RUNTIME_FEATURES];
        for &index in indices
        {
            for feature in 0..RUNTIME_FEATURES
            {
                let delta = rows[index].raw_features[feature] - mean[feature];
                variance[feature] += delta * delta;
            }
        }
        let mut scale = [1.0; RUNTIME_FEATURES];
        for feature in 0..RUNTIME_FEATURES
        {
            scale[feature] = (variance[feature] / indices.len() as f64).sqrt();
            if !scale[feature].is_finite() || scale[feature] < 1e-12
            {
                scale[feature] = 1.0;
            }
        }
        Ok(Self { mean, scale })
    }

    pub fn transform(&self, features: &[f64; RUNTIME_FEATURES]) -> [f64; RUNTIME_FEATURES] {
        let mut result = [0.0; RUNTIME_FEATURES];
        for index in 0..RUNTIME_FEATURES
        {
            result[index] =
                ((features[index] - self.mean[index]) / self.scale[index]).clamp(-12.0, 12.0);
        }
        result
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptSplit {
    pub train_prompts: Vec<usize>,
    pub validation_prompts: Vec<usize>,
    pub holdout_prompts: Vec<usize>,
    #[serde(skip)]
    pub train_rows: Vec<usize>,
    #[serde(skip)]
    pub validation_rows: Vec<usize>,
    #[serde(skip)]
    pub holdout_rows: Vec<usize>,
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn split_by_prompt(rows: &[CandidateRecord], seed: u64) -> Result<PromptSplit, String> {
    let mut prompts: Vec<usize> = rows
        .iter()
        .map(|row| row.prompt_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if prompts.len() < 10
    {
        return Err("trajectory discovery requires at least ten distinct prompts".to_string());
    }
    prompts.sort_by_key(|prompt| (mix64(*prompt as u64 ^ seed), *prompt));
    let train_end = (prompts.len() * 3) / 5;
    let validation_end = (prompts.len() * 4) / 5;
    if train_end == 0 || validation_end <= train_end || validation_end >= prompts.len()
    {
        return Err("cannot form non-empty train/validation/holdout prompt splits".to_string());
    }
    let train_prompts = prompts[..train_end].to_vec();
    let validation_prompts = prompts[train_end..validation_end].to_vec();
    let holdout_prompts = prompts[validation_end..].to_vec();
    let train_set: BTreeSet<usize> = train_prompts.iter().copied().collect();
    let validation_set: BTreeSet<usize> = validation_prompts.iter().copied().collect();
    let holdout_set: BTreeSet<usize> = holdout_prompts.iter().copied().collect();
    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    let mut holdout_rows = Vec::new();
    for (index, row) in rows.iter().enumerate()
    {
        if train_set.contains(&row.prompt_id)
        {
            train_rows.push(index);
        }
        else if validation_set.contains(&row.prompt_id)
        {
            validation_rows.push(index);
        }
        else if holdout_set.contains(&row.prompt_id)
        {
            holdout_rows.push(index);
        }
        else
        {
            return Err("internal prompt-split assignment failure".to_string());
        }
    }
    Ok(PromptSplit {
        train_prompts,
        validation_prompts,
        holdout_prompts,
        train_rows,
        validation_rows,
        holdout_rows,
    })
}

pub fn sequences_for_indices(rows: &[CandidateRecord], selected: &[usize]) -> Vec<Vec<usize>> {
    let selected: BTreeSet<usize> = selected.iter().copied().collect();
    let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate()
    {
        if selected.contains(&index)
        {
            grouped.entry(row.prompt_id).or_default().push(index);
        }
    }
    let mut sequences: Vec<Vec<usize>> = grouped.into_values().collect();
    for sequence in &mut sequences
    {
        sequence.sort_by_key(|index| rows[*index].ordinal);
    }
    sequences
}

#[derive(Debug, Clone, Serialize)]
pub struct LabelSummary {
    pub rows: usize,
    pub prompts: usize,
    pub strict_unsafe: usize,
    pub semantic_unsafe: usize,
    pub trajectory_unsafe: usize,
    pub quality_regressions: usize,
    pub prediction_changes: usize,
    pub response_changes: usize,
    pub decision_count_changes: usize,
}

pub fn summarize_labels(rows: &[CandidateRecord], indices: &[usize]) -> LabelSummary {
    let prompts = indices
        .iter()
        .map(|index| rows[*index].prompt_id)
        .collect::<BTreeSet<_>>()
        .len();
    LabelSummary {
        rows: indices.len(),
        prompts,
        strict_unsafe: indices
            .iter()
            .filter(|index| rows[**index].strict_unsafe)
            .count(),
        semantic_unsafe: indices
            .iter()
            .filter(|index| rows[**index].semantic_unsafe)
            .count(),
        trajectory_unsafe: indices
            .iter()
            .filter(|index| rows[**index].trajectory_unsafe)
            .count(),
        quality_regressions: indices
            .iter()
            .filter(|index| rows[**index].quality_regression)
            .count(),
        prediction_changes: indices
            .iter()
            .filter(|index| rows[**index].prediction_changed)
            .count(),
        response_changes: indices
            .iter()
            .filter(|index| rows[**index].response_changed)
            .count(),
        decision_count_changes: indices
            .iter()
            .filter(|index| rows[**index].decision_count_changed)
            .count(),
    }
}
