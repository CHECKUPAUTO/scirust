//! Deterministic selection from already-ranked candidates with accepted evidence.
//!
//! Static ranking decides which candidates are worth measuring. This layer never
//! invents candidates and never accepts timing without correctness evidence.

use crate::{
    ElasticAutoTuner, ElasticEvidence, ElasticEvidenceError, ElasticObjective, RankedCandidate,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElasticSelectionError {
    NoRankedCandidates,
    NoMeasuredCandidate,
    InvalidEvidence(ElasticEvidenceError),
    NonDeterministicCandidate,
    DuplicateEvidence,
}

impl core::fmt::Display for ElasticSelectionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::NoRankedCandidates => write!(f, "no statically qualified candidates were supplied"),
            Self::NoMeasuredCandidate => write!(f, "no measured evidence matched a ranked candidate"),
            Self::InvalidEvidence(error) => write!(f, "invalid measured evidence: {error}"),
            Self::NonDeterministicCandidate =>
            {
                write!(f, "deterministic-only selection received a non-deterministic candidate")
            },
            Self::DuplicateEvidence => write!(f, "multiple evidence records exist for one candidate"),
        }
    }
}

impl std::error::Error for ElasticSelectionError {}

impl ElasticAutoTuner {
    /// Select one measured candidate from the statically ranked set.
    ///
    /// `MinLatency`, `MaxThroughput`, and `DeterministicOnly` minimize measured
    /// latency because every candidate solves the same problem class. The
    /// temporary-memory objective minimizes scratch first. Balanced selection
    /// avoids mixing nanoseconds and bytes: it restricts selection to the
    /// latency/memory Pareto frontier and then uses the existing objective-aware
    /// static rank as the deterministic tie-breaker.
    pub fn select_measured_evidence(
        &self,
        ranked: &[RankedCandidate],
        evidence: &[ElasticEvidence],
    ) -> Result<ElasticEvidence, ElasticSelectionError> {
        if ranked.is_empty()
        {
            return Err(ElasticSelectionError::NoRankedCandidates);
        }

        let mut matched = Vec::new();
        for (rank_index, ranked_candidate) in ranked.iter().enumerate()
        {
            let mut records = evidence
                .iter()
                .filter(|record| record.candidate == ranked_candidate.candidate);
            let Some(record) = records.next() else {
                continue;
            };
            if records.next().is_some()
            {
                return Err(ElasticSelectionError::DuplicateEvidence);
            }
            record
                .validate()
                .map_err(ElasticSelectionError::InvalidEvidence)?;
            if self.config().objective == ElasticObjective::DeterministicOnly
                && !record.candidate.deterministic
            {
                return Err(ElasticSelectionError::NonDeterministicCandidate);
            }
            matched.push((rank_index, record));
        }

        if matched.is_empty()
        {
            return Err(ElasticSelectionError::NoMeasuredCandidate);
        }

        let selected = match self.config().objective
        {
            ElasticObjective::MinTemporaryMemory => matched
                .into_iter()
                .min_by(|left, right| memory_key(*left).cmp(&memory_key(*right)))
                .unwrap(),
            ElasticObjective::BalancedLatencyMemory => select_balanced(&matched),
            ElasticObjective::MinLatency
            | ElasticObjective::MaxThroughput
            | ElasticObjective::DeterministicOnly => matched
                .into_iter()
                .min_by(|left, right| latency_key(*left).cmp(&latency_key(*right)))
                .unwrap(),
        };

        Ok(selected.1.clone())
    }
}

fn latency_key(entry: (usize, &ElasticEvidence)) -> (u64, u64, u64, u64, u64, usize) {
    let (rank, evidence) = entry;
    (
        evidence.measurement.median_ns,
        evidence.measurement.p95_ns,
        evidence.measurement.p99_ns,
        evidence.measurement.mad_ns,
        evidence.candidate.temporary_bytes,
        rank,
    )
}

fn memory_key(entry: (usize, &ElasticEvidence)) -> (u64, u64, u64, u64, u64, usize) {
    let (rank, evidence) = entry;
    (
        evidence.candidate.temporary_bytes,
        evidence.measurement.median_ns,
        evidence.measurement.p95_ns,
        evidence.measurement.p99_ns,
        evidence.measurement.mad_ns,
        rank,
    )
}

fn select_balanced<'a>(matched: &[(usize, &'a ElasticEvidence)]) -> (usize, &'a ElasticEvidence) {
    matched
        .iter()
        .copied()
        .filter(|candidate| {
            !matched.iter().copied().any(|other| {
                dominates(other.1, candidate.1)
            })
        })
        .min_by_key(|entry| entry.0)
        .unwrap()
}

fn dominates(left: &ElasticEvidence, right: &ElasticEvidence) -> bool {
    let no_slower = left.measurement.median_ns <= right.measurement.median_ns;
    let no_larger = left.candidate.temporary_bytes <= right.candidate.temporary_bytes;
    let strictly_better = left.measurement.median_ns < right.measurement.median_ns
        || left.candidate.temporary_bytes < right.candidate.temporary_bytes;
    no_slower && no_larger && strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElasticCandidate, ElasticConfig, ElasticMeasurement, ElasticParameter};

    fn candidate(id: i64, temporary_bytes: u64, deterministic: bool) -> ElasticCandidate {
        ElasticCandidate::new(
            "sgemm-f32",
            b"rev1".to_vec(),
            [ElasticParameter {
                name: "path".into(),
                value: id,
            }],
            deterministic,
            temporary_bytes,
        )
        .unwrap()
    }

    fn evidence(candidate: ElasticCandidate, median_ns: u64) -> ElasticEvidence {
        ElasticEvidence::validated(
            candidate,
            vec![1],
            ElasticMeasurement {
                sample_count: 9,
                median_ns,
                p95_ns: median_ns + 10,
                p99_ns: median_ns + 20,
                mad_ns: 2,
            },
        )
        .unwrap()
    }

    fn ranked(candidates: &[ElasticCandidate]) -> Vec<RankedCandidate> {
        candidates
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, candidate)| RankedCandidate {
                estimated_cost_units: index as u64,
                candidate,
            })
            .collect()
    }

    #[test]
    fn latency_objective_uses_measurement_not_static_rank() {
        let candidates = [candidate(0, 0, true), candidate(1, 4096, true)];
        let ranked = ranked(&candidates);
        let measured = [
            evidence(candidates[0].clone(), 120),
            evidence(candidates[1].clone(), 80),
        ];
        let selected = ElasticAutoTuner::new(ElasticConfig::default())
            .select_measured_evidence(&ranked, &measured)
            .unwrap();
        assert_eq!(selected.candidate, candidates[1]);
    }

    #[test]
    fn memory_objective_prefers_smaller_scratch() {
        let candidates = [candidate(0, 0, true), candidate(1, 4096, true)];
        let ranked = ranked(&candidates);
        let measured = [
            evidence(candidates[0].clone(), 120),
            evidence(candidates[1].clone(), 80),
        ];
        let tuner = ElasticAutoTuner::new(ElasticConfig {
            objective: ElasticObjective::MinTemporaryMemory,
            ..ElasticConfig::default()
        });
        let selected = tuner.select_measured_evidence(&ranked, &measured).unwrap();
        assert_eq!(selected.candidate, candidates[0]);
    }

    #[test]
    fn balanced_objective_uses_pareto_frontier_then_static_rank() {
        let candidates = [
            candidate(0, 1024, true),
            candidate(1, 4096, true),
            candidate(2, 2048, true),
        ];
        let ranked = ranked(&candidates);
        let measured = [
            evidence(candidates[0].clone(), 120),
            evidence(candidates[1].clone(), 80),
            evidence(candidates[2].clone(), 140),
        ];
        let tuner = ElasticAutoTuner::new(ElasticConfig {
            objective: ElasticObjective::BalancedLatencyMemory,
            ..ElasticConfig::default()
        });
        let selected = tuner.select_measured_evidence(&ranked, &measured).unwrap();
        assert_eq!(selected.candidate, candidates[0]);
    }

    #[test]
    fn duplicate_evidence_fails_closed() {
        let candidates = [candidate(0, 0, true)];
        let ranked = ranked(&candidates);
        let record = evidence(candidates[0].clone(), 100);
        let tuner = ElasticAutoTuner::new(ElasticConfig::default());
        assert_eq!(
            tuner.select_measured_evidence(&ranked, &[record.clone(), record]),
            Err(ElasticSelectionError::DuplicateEvidence)
        );
    }
}
