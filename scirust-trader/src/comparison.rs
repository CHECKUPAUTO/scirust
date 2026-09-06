//! Auditable multi-metric comparison of trading candidates.
//!
//! The comparison deliberately avoids an implicit weighted score. Every metric
//! declares whether larger or smaller values are preferred. Candidates are
//! ranked per metric and a Pareto front preserves trade-offs instead of hiding
//! them behind one arbitrary scalar objective.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricDirection {
    HigherBetter,
    LowerBetter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonMetric {
    pub name: String,
    pub direction: MetricDirection,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvidence {
    pub candidate_id: String,
    pub family: String,
    pub metrics: BTreeMap<String, f64>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricRank {
    pub metric: String,
    /// Competition rank (1 is best). Equal values receive equal ranks.
    pub rank: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateComparisonRow {
    pub candidate_id: String,
    pub family: String,
    pub metrics: BTreeMap<String, f64>,
    pub ranks: Vec<MetricRank>,
    pub pareto_optimal: bool,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub metrics: Vec<ComparisonMetric>,
    pub rows: Vec<CandidateComparisonRow>,
    pub pareto_front: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonError {
    NoMetrics,
    NoCandidates,
    DuplicateMetric(String),
    DuplicateCandidate(String),
    EmptyMetricName,
    EmptyCandidateId,
    EmptyFamily,
    MetricSetMismatch {
        candidate_id: String,
    },
    NonFiniteMetric {
        candidate_id: String,
        metric: String,
    },
}

fn validate(
    metrics: &[ComparisonMetric],
    candidates: &[CandidateEvidence],
) -> Result<(), ComparisonError> {
    if metrics.is_empty()
    {
        return Err(ComparisonError::NoMetrics);
    }
    if candidates.is_empty()
    {
        return Err(ComparisonError::NoCandidates);
    }

    let mut metric_names = BTreeSet::new();
    for metric in metrics
    {
        if metric.name.is_empty()
        {
            return Err(ComparisonError::EmptyMetricName);
        }
        if !metric_names.insert(metric.name.clone())
        {
            return Err(ComparisonError::DuplicateMetric(metric.name.clone()));
        }
    }

    let expected: BTreeSet<String> = metrics.iter().map(|metric| metric.name.clone()).collect();
    let mut candidate_ids = BTreeSet::new();
    for candidate in candidates
    {
        if candidate.candidate_id.is_empty()
        {
            return Err(ComparisonError::EmptyCandidateId);
        }
        if candidate.family.is_empty()
        {
            return Err(ComparisonError::EmptyFamily);
        }
        if !candidate_ids.insert(candidate.candidate_id.clone())
        {
            return Err(ComparisonError::DuplicateCandidate(
                candidate.candidate_id.clone(),
            ));
        }
        let actual: BTreeSet<String> = candidate.metrics.keys().cloned().collect();
        if actual != expected
        {
            return Err(ComparisonError::MetricSetMismatch {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        for (name, value) in &candidate.metrics
        {
            if !value.is_finite()
            {
                return Err(ComparisonError::NonFiniteMetric {
                    candidate_id: candidate.candidate_id.clone(),
                    metric: name.clone(),
                });
            }
        }
    }
    Ok(())
}

fn preferred_or_equal(direction: MetricDirection, lhs: f64, rhs: f64) -> bool {
    match direction
    {
        MetricDirection::HigherBetter => lhs >= rhs,
        MetricDirection::LowerBetter => lhs <= rhs,
    }
}

fn strictly_preferred(direction: MetricDirection, lhs: f64, rhs: f64) -> bool {
    match direction
    {
        MetricDirection::HigherBetter => lhs > rhs,
        MetricDirection::LowerBetter => lhs < rhs,
    }
}

fn dominates(
    lhs: &CandidateEvidence,
    rhs: &CandidateEvidence,
    metrics: &[ComparisonMetric],
) -> bool {
    let mut strictly_better = false;
    for metric in metrics
    {
        let lhs_value = lhs.metrics[&metric.name];
        let rhs_value = rhs.metrics[&metric.name];
        if !preferred_or_equal(metric.direction, lhs_value, rhs_value)
        {
            return false;
        }
        strictly_better |= strictly_preferred(metric.direction, lhs_value, rhs_value);
    }
    strictly_better
}

fn metric_ranks(
    metric: &ComparisonMetric,
    candidates: &[CandidateEvidence],
) -> BTreeMap<String, usize> {
    let mut ordered: Vec<(&str, f64)> = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.candidate_id.as_str(),
                candidate.metrics[&metric.name],
            )
        })
        .collect();
    ordered.sort_by(|a, b| {
        let value_order = match metric.direction
        {
            MetricDirection::HigherBetter => b.1.total_cmp(&a.1),
            MetricDirection::LowerBetter => a.1.total_cmp(&b.1),
        };
        value_order.then_with(|| a.0.cmp(b.0))
    });

    let mut result = BTreeMap::new();
    let mut previous_value: Option<f64> = None;
    let mut previous_rank = 0usize;
    for (index, (candidate_id, value)) in ordered.into_iter().enumerate()
    {
        let rank = match previous_value
        {
            Some(previous) if previous == value => previous_rank,
            _ => index + 1,
        };
        result.insert(candidate_id.to_string(), rank);
        previous_value = Some(value);
        previous_rank = rank;
    }
    result
}

pub fn compare_candidates(
    metrics: Vec<ComparisonMetric>,
    candidates: Vec<CandidateEvidence>,
) -> Result<ComparisonReport, ComparisonError> {
    validate(&metrics, &candidates)?;

    let rank_maps: Vec<(String, BTreeMap<String, usize>)> = metrics
        .iter()
        .map(|metric| (metric.name.clone(), metric_ranks(metric, &candidates)))
        .collect();

    let pareto_flags: Vec<bool> = (0..candidates.len())
        .map(|candidate_index| {
            !(0..candidates.len()).any(|other_index| {
                other_index != candidate_index
                    && dominates(
                        &candidates[other_index],
                        &candidates[candidate_index],
                        &metrics,
                    )
            })
        })
        .collect();

    let mut rows = Vec::with_capacity(candidates.len());
    for (candidate_index, candidate) in candidates.iter().enumerate()
    {
        let ranks = rank_maps
            .iter()
            .map(|(metric, ranks)| MetricRank {
                metric: metric.clone(),
                rank: ranks[&candidate.candidate_id],
            })
            .collect();
        rows.push(CandidateComparisonRow {
            candidate_id: candidate.candidate_id.clone(),
            family: candidate.family.clone(),
            metrics: candidate.metrics.clone(),
            ranks,
            pareto_optimal: pareto_flags[candidate_index],
            evidence_refs: candidate.evidence_refs.clone(),
        });
    }
    rows.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));

    let pareto_front = rows
        .iter()
        .filter(|row| row.pareto_optimal)
        .map(|row| row.candidate_id.clone())
        .collect();

    Ok(ComparisonReport {
        metrics,
        rows,
        pareto_front,
    })
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r')
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    }
    else
    {
        value.to_string()
    }
}

/// Deterministic CSV rendering of the final comparison matrix.
pub fn comparison_csv(report: &ComparisonReport) -> String {
    let mut output = String::new();
    output.push_str("candidate_id,family,pareto_optimal");
    for metric in &report.metrics
    {
        output.push(',');
        output.push_str(&csv_escape(&metric.name));
        output.push_str(",rank:");
        output.push_str(&csv_escape(&metric.name));
    }
    output.push_str(",evidence_refs\n");

    for row in &report.rows
    {
        output.push_str(&csv_escape(&row.candidate_id));
        output.push(',');
        output.push_str(&csv_escape(&row.family));
        output.push(',');
        output.push_str(if row.pareto_optimal { "true" } else { "false" });
        for metric in &report.metrics
        {
            output.push(',');
            output.push_str(&row.metrics[&metric.name].to_string());
            output.push(',');
            let rank = row
                .ranks
                .iter()
                .find(|rank| rank.metric == metric.name)
                .map(|rank| rank.rank)
                .unwrap_or(0);
            output.push_str(&rank.to_string());
        }
        output.push(',');
        output.push_str(&csv_escape(&row.evidence_refs.join(";")));
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(name: &str, direction: MetricDirection) -> ComparisonMetric {
        ComparisonMetric {
            name: name.into(),
            direction,
            unit: "unit".into(),
        }
    }

    fn candidate(id: &str, ret: f64, drawdown: f64) -> CandidateEvidence {
        CandidateEvidence {
            candidate_id: id.into(),
            family: "test".into(),
            metrics: BTreeMap::from([("return".into(), ret), ("drawdown".into(), drawdown)]),
            evidence_refs: vec![format!("{id}.json")],
        }
    }

    #[test]
    fn dominated_candidate_is_not_on_pareto_front() {
        let report = compare_candidates(
            vec![
                metric("return", MetricDirection::HigherBetter),
                metric("drawdown", MetricDirection::LowerBetter),
            ],
            vec![candidate("A", 0.20, 0.10), candidate("B", 0.10, 0.20)],
        )
        .unwrap();
        assert_eq!(report.pareto_front, vec!["A"]);
    }

    #[test]
    fn tradeoffs_remain_visible_on_pareto_front() {
        let report = compare_candidates(
            vec![
                metric("return", MetricDirection::HigherBetter),
                metric("drawdown", MetricDirection::LowerBetter),
            ],
            vec![candidate("A", 0.20, 0.20), candidate("B", 0.10, 0.05)],
        )
        .unwrap();
        assert_eq!(report.pareto_front, vec!["A", "B"]);
    }

    #[test]
    fn equal_metric_values_receive_equal_rank() {
        let report = compare_candidates(
            vec![
                metric("return", MetricDirection::HigherBetter),
                metric("drawdown", MetricDirection::LowerBetter),
            ],
            vec![candidate("B", 0.20, 0.20), candidate("A", 0.20, 0.10)],
        )
        .unwrap();
        let rank_a = report.rows[0]
            .ranks
            .iter()
            .find(|rank| rank.metric == "return")
            .unwrap()
            .rank;
        let rank_b = report.rows[1]
            .ranks
            .iter()
            .find(|rank| rank.metric == "return")
            .unwrap()
            .rank;
        assert_eq!(rank_a, rank_b);
    }

    #[test]
    fn csv_is_deterministic_and_contains_evidence() {
        let report = compare_candidates(
            vec![
                metric("return", MetricDirection::HigherBetter),
                metric("drawdown", MetricDirection::LowerBetter),
            ],
            vec![candidate("A", 0.20, 0.10)],
        )
        .unwrap();
        let a = comparison_csv(&report);
        let b = comparison_csv(&report);
        assert_eq!(a, b);
        assert!(a.contains("A.json"));
        assert!(a.contains("pareto_optimal"));
    }
}
