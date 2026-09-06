//! Reproducible robustness summaries and experiment manifests.
//!
//! These helpers aggregate evidence that is already produced by backtests or
//! model evaluations. They do not choose a strategy and they do not reinterpret
//! a failed stress test as success.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ParameterStabilityReport {
    pub samples: usize,
    pub mean_score: f64,
    pub median_score: f64,
    pub sample_std: f64,
    pub worst_score: f64,
    pub best_score: f64,
    pub positive_fraction: f64,
    pub range: f64,
}

pub fn parameter_stability(scores: &[f64]) -> Option<ParameterStabilityReport> {
    if scores.is_empty() || scores.iter().any(|x| !x.is_finite())
    {
        return None;
    }
    let mut ordered = scores.to_vec();
    ordered.sort_by(f64::total_cmp);
    let n = ordered.len();
    let mean = ordered.iter().sum::<f64>() / n as f64;
    let median = if n.is_multiple_of(2)
    {
        (ordered[n / 2 - 1] + ordered[n / 2]) / 2.0
    }
    else
    {
        ordered[n / 2]
    };
    let sample_std = if n < 2
    {
        0.0
    }
    else
    {
        let ss = ordered.iter().map(|x| (*x - mean).powi(2)).sum::<f64>();
        (ss / (n - 1) as f64).sqrt()
    };
    let positive = ordered.iter().filter(|x| **x > 0.0).count();
    Some(ParameterStabilityReport {
        samples: n,
        mean_score: mean,
        median_score: median,
        sample_std,
        worst_score: ordered[0],
        best_score: ordered[n - 1],
        positive_fraction: positive as f64 / n as f64,
        range: ordered[n - 1] - ordered[0],
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegimeMetrics {
    pub observations: usize,
    pub mean_return: f64,
    pub cumulative_simple_return: f64,
    pub sharpe: f64,
    pub positive_fraction: f64,
}

pub fn regime_robustness(
    returns: &[f64],
    regimes: &[String],
) -> Option<BTreeMap<String, RegimeMetrics>> {
    if returns.is_empty()
        || returns.len() != regimes.len()
        || returns.iter().any(|r| !r.is_finite())
        || regimes.iter().any(|r| r.is_empty())
    {
        return None;
    }
    let mut buckets: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (ret, regime) in returns.iter().copied().zip(regimes)
    {
        buckets.entry(regime.clone()).or_default().push(ret);
    }
    let mut report = BTreeMap::new();
    for (regime, values) in buckets
    {
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let sample_std = if n < 2
        {
            0.0
        }
        else
        {
            let ss = values.iter().map(|x| (*x - mean).powi(2)).sum::<f64>();
            (ss / (n - 1) as f64).sqrt()
        };
        let sharpe = if sample_std <= f64::EPSILON
        {
            0.0
        }
        else
        {
            mean / sample_std
        };
        let positive = values.iter().filter(|x| **x > 0.0).count();
        report.insert(
            regime,
            RegimeMetrics {
                observations: n,
                mean_return: mean,
                cumulative_simple_return: values.iter().sum(),
                sharpe,
                positive_fraction: positive as f64 / n as f64,
            },
        );
    }
    Some(report)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostStressPoint {
    pub cost_bps_per_unit_turnover: f64,
    pub cumulative_net_return: f64,
    pub mean_net_return: f64,
    pub net_sharpe: f64,
}

/// Reprice a fixed gross-return/turnover path under declared transaction costs.
/// This intentionally does not alter fills or market impact; those belong to
/// execution simulation. It isolates sensitivity to the cost assumption alone.
pub fn cost_stress(
    gross_returns: &[f64],
    turnover: &[f64],
    cost_grid_bps: &[f64],
) -> Option<Vec<CostStressPoint>> {
    if gross_returns.is_empty()
        || gross_returns.len() != turnover.len()
        || gross_returns.iter().any(|x| !x.is_finite())
        || turnover.iter().any(|x| !x.is_finite() || *x < 0.0)
        || cost_grid_bps.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return None;
    }
    let n = gross_returns.len();
    let mut output = Vec::with_capacity(cost_grid_bps.len());
    for cost_bps in cost_grid_bps
    {
        let rate = *cost_bps / 10_000.0;
        let net: Vec<f64> = gross_returns
            .iter()
            .zip(turnover)
            .map(|(gross, turn)| *gross - *turn * rate)
            .collect();
        let mean = net.iter().sum::<f64>() / n as f64;
        let sample_std = if n < 2
        {
            0.0
        }
        else
        {
            let ss = net.iter().map(|x| (*x - mean).powi(2)).sum::<f64>();
            (ss / (n - 1) as f64).sqrt()
        };
        output.push(CostStressPoint {
            cost_bps_per_unit_turnover: *cost_bps,
            cumulative_net_return: net.iter().sum(),
            mean_net_return: mean,
            net_sharpe: if sample_std <= f64::EPSILON
            {
                0.0
            }
            else
            {
                mean / sample_std
            },
        });
    }
    Some(output)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerturbationPoint {
    pub name: String,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerturbationReport {
    pub baseline_score: f64,
    pub points: Vec<PerturbationPoint>,
    pub worst_score: f64,
    /// `baseline_score - worst_score`; positive values mean degradation.
    pub worst_degradation: f64,
    pub fraction_not_worse_than_baseline: f64,
}

pub fn perturbation_report(
    baseline_score: f64,
    points: Vec<PerturbationPoint>,
) -> Option<PerturbationReport> {
    if !baseline_score.is_finite()
        || points.is_empty()
        || points
            .iter()
            .any(|point| point.name.is_empty() || !point.score.is_finite())
    {
        return None;
    }
    let worst = points
        .iter()
        .map(|point| point.score)
        .min_by(f64::total_cmp)?;
    let not_worse = points
        .iter()
        .filter(|point| point.score >= baseline_score)
        .count();
    Some(PerturbationReport {
        baseline_score,
        worst_score: worst,
        worst_degradation: baseline_score - worst,
        fraction_not_worse_than_baseline: not_worse as f64 / points.len() as f64,
        points,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentManifest {
    pub schema_version: u32,
    pub experiment_id: String,
    pub created_at_ms: i64,
    pub seed: u64,
    pub code_ref: String,
    pub data_fingerprint: String,
    pub observation_start_ms: i64,
    pub observation_end_ms: i64,
    pub candidate_count: usize,
    pub config: serde_json::Value,
    pub evidence_artifacts: Vec<String>,
}

impl ExperimentManifest {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    pub fn validate(&self) -> bool {
        self.schema_version == Self::CURRENT_SCHEMA_VERSION
            && !self.experiment_id.is_empty()
            && !self.code_ref.is_empty()
            && !self.data_fingerprint.is_empty()
            && self.observation_start_ms <= self.observation_end_ms
            && self.candidate_count > 0
            && self
                .evidence_artifacts
                .iter()
                .all(|artifact| !artifact.is_empty())
    }

    /// SHA-256 over the serialized manifest. With the same manifest value and
    /// serde_json version this is deterministic and can be stored beside the
    /// experiment artifacts for integrity checks.
    pub fn fingerprint(&self) -> Option<String> {
        if !self.validate()
        {
            return None;
        }
        let bytes = serde_json::to_vec(self).ok()?;
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Some(format!("{:x}", hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stability_exposes_fragile_parameter_surface() {
        let report = parameter_stability(&[1.0, 0.9, -2.0, 1.1, 1.0]).unwrap();
        assert_eq!(report.samples, 5);
        assert!(report.range > 3.0);
        assert!((report.positive_fraction - 0.8).abs() < 1e-12);
    }

    #[test]
    fn regimes_are_reported_separately() {
        let returns = [0.02, 0.01, -0.03, -0.01];
        let regimes = vec![
            "trend".to_string(),
            "trend".to_string(),
            "range".to_string(),
            "range".to_string(),
        ];
        let report = regime_robustness(&returns, &regimes).unwrap();
        assert!(report["trend"].mean_return > 0.0);
        assert!(report["range"].mean_return < 0.0);
    }

    #[test]
    fn higher_costs_reduce_same_fixed_path() {
        let gross = [0.01, 0.01, -0.005, 0.02];
        let turnover = [1.0, 0.5, 2.0, 1.0];
        let report = cost_stress(&gross, &turnover, &[0.0, 10.0, 50.0]).unwrap();
        assert!(report[0].cumulative_net_return > report[1].cumulative_net_return);
        assert!(report[1].cumulative_net_return > report[2].cumulative_net_return);
    }

    #[test]
    fn perturbations_preserve_named_evidence() {
        let report = perturbation_report(
            1.0,
            vec![
                PerturbationPoint {
                    name: "delay+50ms".into(),
                    score: 0.8,
                },
                PerturbationPoint {
                    name: "fee+5bps".into(),
                    score: 0.6,
                },
            ],
        )
        .unwrap();
        assert_eq!(report.worst_score, 0.6);
        assert!((report.worst_degradation - 0.4).abs() < 1e-12);
    }

    #[test]
    fn manifest_fingerprint_is_repeatable_and_sensitive() {
        let mut manifest = ExperimentManifest {
            schema_version: ExperimentManifest::CURRENT_SCHEMA_VERSION,
            experiment_id: "exp-1".into(),
            created_at_ms: 100,
            seed: 42,
            code_ref: "abc123".into(),
            data_fingerprint: "data456".into(),
            observation_start_ms: 0,
            observation_end_ms: 99,
            candidate_count: 12,
            config: serde_json::json!({"cost_bps": 5.0, "folds": 6}),
            evidence_artifacts: vec!["scores.json".into()],
        };
        let first = manifest.fingerprint().unwrap();
        assert_eq!(first, manifest.fingerprint().unwrap());
        manifest.seed = 43;
        assert_ne!(first, manifest.fingerprint().unwrap());
    }
}
