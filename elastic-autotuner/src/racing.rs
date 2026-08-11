//! Deterministic racing / successive-halving support.
//!
//! This layer consumes already-collected, correctness-qualified evidence. It
//! does not benchmark kernels itself; it decides which candidates survive to a
//! higher-budget measurement round.

use crate::measurement_protocol::{ElasticMeasurementProtocol, ElasticMeasurementProtocolError};
use crate::{
    ElasticAutoTuner, ElasticEvidence, ElasticEvidenceError, ElasticObjective, RankedCandidate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticRacingPolicy {
    /// Keep approximately `ceil(n / survivor_divisor)` candidates each round.
    pub survivor_divisor: u32,
    /// Never reduce below this many survivors while `n` is larger.
    pub min_survivors: usize,
    /// Multiply measured iterations by this factor at each subsequent round.
    pub measurement_growth: u32,
}

impl Default for ElasticRacingPolicy {
    fn default() -> Self {
        Self {
            survivor_divisor: 2,
            min_survivors: 1,
            measurement_growth: 2,
        }
    }
}

impl ElasticRacingPolicy {
    pub fn validate(self) -> Result<(), ElasticRacingError> {
        if self.survivor_divisor < 2
        {
            return Err(ElasticRacingError::InvalidSurvivorDivisor);
        }
        if self.min_survivors == 0
        {
            return Err(ElasticRacingError::ZeroMinimumSurvivors);
        }
        if self.measurement_growth == 0
        {
            return Err(ElasticRacingError::ZeroMeasurementGrowth);
        }
        Ok(())
    }

    /// Deterministically derive the measurement protocol for a racing round.
    /// Round zero returns the base budget unchanged.
    pub fn protocol_for_round(
        self,
        base: ElasticMeasurementProtocol,
        round: u32,
    ) -> Result<ElasticMeasurementProtocol, ElasticRacingError> {
        self.validate()?;
        base.validate()
            .map_err(ElasticRacingError::MeasurementProtocol)?;

        let mut measured_iterations = base.measured_iterations;
        for _ in 0..round
        {
            measured_iterations = measured_iterations
                .checked_mul(self.measurement_growth)
                .ok_or(ElasticRacingError::MeasurementBudgetOverflow)?;
        }

        Ok(ElasticMeasurementProtocol {
            measured_iterations,
            ..base
        })
    }

    /// Run one deterministic elimination round over a fully measured active set.
    ///
    /// Every ranked candidate must have exactly one valid evidence record. Input
    /// evidence ordering is irrelevant. Returned candidates are ordered by the
    /// round's objective-specific measured ranking and are ready for the next
    /// higher-budget round.
    pub fn race_round(
        self,
        tuner: &ElasticAutoTuner,
        ranked: &[RankedCandidate],
        evidence: &[ElasticEvidence],
    ) -> Result<Vec<RankedCandidate>, ElasticRacingError> {
        self.validate()?;
        if ranked.is_empty()
        {
            return Err(ElasticRacingError::NoCandidates);
        }

        let mut records = Vec::with_capacity(ranked.len());
        for (rank_index, ranked_candidate) in ranked.iter().enumerate()
        {
            let mut matches = evidence
                .iter()
                .filter(|record| record.candidate == ranked_candidate.candidate);
            let Some(record) = matches.next()
            else
            {
                return Err(ElasticRacingError::MissingEvidence { rank_index });
            };
            if matches.next().is_some()
            {
                return Err(ElasticRacingError::DuplicateEvidence { rank_index });
            }
            record
                .validate()
                .map_err(ElasticRacingError::InvalidEvidence)?;
            if tuner.config().objective == ElasticObjective::DeterministicOnly
                && !record.candidate.deterministic
            {
                return Err(ElasticRacingError::NonDeterministicCandidate { rank_index });
            }
            records.push((rank_index, ranked_candidate.clone(), record));
        }

        match tuner.config().objective
        {
            ElasticObjective::MinTemporaryMemory => records.sort_by_key(memory_key),
            ElasticObjective::BalancedLatencyMemory =>
            {
                let snapshot = records.clone();
                records.sort_by(|left, right| {
                    balanced_key(left, &snapshot).cmp(&balanced_key(right, &snapshot))
                });
            },
            ElasticObjective::MinLatency
            | ElasticObjective::MaxThroughput
            | ElasticObjective::DeterministicOnly => records.sort_by_key(latency_key),
        }

        let divisor = usize::try_from(self.survivor_divisor)
            .map_err(|_| ElasticRacingError::SurvivorCountOverflow)?;
        let target = ranked.len().div_ceil(divisor);
        let survivors = target.max(self.min_survivors).min(ranked.len());
        records.truncate(survivors);
        Ok(records
            .into_iter()
            .map(|(_, candidate, _)| candidate)
            .collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticRacingError {
    InvalidSurvivorDivisor,
    ZeroMinimumSurvivors,
    ZeroMeasurementGrowth,
    MeasurementBudgetOverflow,
    SurvivorCountOverflow,
    NoCandidates,
    MissingEvidence { rank_index: usize },
    DuplicateEvidence { rank_index: usize },
    InvalidEvidence(ElasticEvidenceError),
    NonDeterministicCandidate { rank_index: usize },
    MeasurementProtocol(ElasticMeasurementProtocolError),
}

impl core::fmt::Display for ElasticRacingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::InvalidSurvivorDivisor =>
            {
                write!(f, "racing survivor divisor must be at least two")
            },
            Self::ZeroMinimumSurvivors => write!(f, "racing must retain at least one survivor"),
            Self::ZeroMeasurementGrowth => write!(f, "measurement growth factor must be non-zero"),
            Self::MeasurementBudgetOverflow =>
            {
                write!(f, "racing measurement budget overflowed u32")
            },
            Self::SurvivorCountOverflow =>
            {
                write!(f, "racing survivor divisor does not fit this target")
            },
            Self::NoCandidates => write!(f, "racing requires at least one active candidate"),
            Self::MissingEvidence { rank_index } =>
            {
                write!(
                    f,
                    "missing racing evidence for ranked candidate {rank_index}"
                )
            },
            Self::DuplicateEvidence { rank_index } =>
            {
                write!(
                    f,
                    "duplicate racing evidence for ranked candidate {rank_index}"
                )
            },
            Self::InvalidEvidence(error) => write!(f, "invalid racing evidence: {error}"),
            Self::NonDeterministicCandidate { rank_index } => write!(
                f,
                "deterministic-only racing received non-deterministic candidate {rank_index}"
            ),
            Self::MeasurementProtocol(error) =>
            {
                write!(f, "invalid racing measurement protocol: {error}")
            },
        }
    }
}

impl std::error::Error for ElasticRacingError {}

type RaceRecord<'a> = (usize, RankedCandidate, &'a ElasticEvidence);

fn latency_key(record: &RaceRecord<'_>) -> (u64, u64, u64, u64, u64, usize) {
    (
        record.2.measurement.median_ns,
        record.2.measurement.p95_ns,
        record.2.measurement.p99_ns,
        record.2.measurement.mad_ns,
        record.2.candidate.temporary_bytes,
        record.0,
    )
}

fn memory_key(record: &RaceRecord<'_>) -> (u64, u64, u64, u64, u64, usize) {
    (
        record.2.candidate.temporary_bytes,
        record.2.measurement.median_ns,
        record.2.measurement.p95_ns,
        record.2.measurement.p99_ns,
        record.2.measurement.mad_ns,
        record.0,
    )
}

fn balanced_key(record: &RaceRecord<'_>, all: &[RaceRecord<'_>]) -> (usize, usize, u64, u64) {
    let domination_count = all
        .iter()
        .filter(|other| dominates(other.2, record.2))
        .count();
    (
        domination_count,
        record.0,
        record.2.measurement.median_ns,
        record.2.candidate.temporary_bytes,
    )
}

fn dominates(left: &ElasticEvidence, right: &ElasticEvidence) -> bool {
    let no_slower = left.measurement.median_ns <= right.measurement.median_ns;
    let no_larger = left.candidate.temporary_bytes <= right.candidate.temporary_bytes;
    let strict = left.measurement.median_ns < right.measurement.median_ns
        || left.candidate.temporary_bytes < right.candidate.temporary_bytes;
    no_slower && no_larger && strict
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement_protocol::{
        ElasticResidenceMode, ElasticSynchronizationBoundary, ElasticTimingSource,
    };
    use crate::{ElasticCandidate, ElasticConfig, ElasticMeasurement, ElasticParameter};

    fn candidate(id: i64, scratch: u64) -> ElasticCandidate {
        ElasticCandidate::new(
            "sgemm-f32",
            b"rev1".to_vec(),
            [ElasticParameter {
                name: "path".into(),
                value: id,
            }],
            true,
            scratch,
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

    fn evidence(candidate: ElasticCandidate, median_ns: u64) -> ElasticEvidence {
        ElasticEvidence::validated(
            candidate,
            vec![1],
            ElasticMeasurement {
                sample_count: 5,
                median_ns,
                p95_ns: median_ns + 5,
                p99_ns: median_ns + 10,
                mad_ns: 1,
            },
        )
        .unwrap()
    }

    #[test]
    fn racing_is_independent_of_evidence_input_order() {
        let candidates = [
            candidate(0, 0),
            candidate(1, 1024),
            candidate(2, 2048),
            candidate(3, 4096),
        ];
        let ranked = ranked(&candidates);
        let forward = [
            evidence(candidates[0].clone(), 120),
            evidence(candidates[1].clone(), 80),
            evidence(candidates[2].clone(), 100),
            evidence(candidates[3].clone(), 90),
        ];
        let reverse = [
            forward[3].clone(),
            forward[2].clone(),
            forward[1].clone(),
            forward[0].clone(),
        ];
        let tuner = ElasticAutoTuner::new(ElasticConfig::default());
        let policy = ElasticRacingPolicy::default();
        let a = policy.race_round(&tuner, &ranked, &forward).unwrap();
        let b = policy.race_round(&tuner, &ranked, &reverse).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].candidate, candidates[1]);
        assert_eq!(a[1].candidate, candidates[3]);
    }

    #[test]
    fn protocol_budget_grows_deterministically() {
        let base = ElasticMeasurementProtocol::new(
            2,
            5,
            ElasticTimingSource::HostWallClock,
            ElasticResidenceMode::Resident,
            ElasticSynchronizationBoundary::PerIteration,
        );
        let policy = ElasticRacingPolicy::default();
        assert_eq!(
            policy
                .protocol_for_round(base, 0)
                .unwrap()
                .measured_iterations,
            5
        );
        assert_eq!(
            policy
                .protocol_for_round(base, 1)
                .unwrap()
                .measured_iterations,
            10
        );
        assert_eq!(
            policy
                .protocol_for_round(base, 3)
                .unwrap()
                .measured_iterations,
            40
        );
    }

    #[test]
    fn missing_evidence_fails_closed() {
        let candidates = [candidate(0, 0), candidate(1, 0)];
        let ranked = ranked(&candidates);
        let measured = [evidence(candidates[0].clone(), 10)];
        let error = ElasticRacingPolicy::default()
            .race_round(
                &ElasticAutoTuner::new(ElasticConfig::default()),
                &ranked,
                &measured,
            )
            .unwrap_err();
        assert_eq!(error, ElasticRacingError::MissingEvidence { rank_index: 1 });
    }

    #[test]
    fn survivor_floor_is_respected() {
        let candidates = [
            candidate(0, 0),
            candidate(1, 0),
            candidate(2, 0),
            candidate(3, 0),
            candidate(4, 0),
        ];
        let ranked = ranked(&candidates);
        let measured: Vec<_> = candidates
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, candidate)| evidence(candidate, 100 + index as u64))
            .collect();
        let policy = ElasticRacingPolicy {
            survivor_divisor: 4,
            min_survivors: 3,
            measurement_growth: 2,
        };
        let survivors = policy
            .race_round(
                &ElasticAutoTuner::new(ElasticConfig::default()),
                &ranked,
                &measured,
            )
            .unwrap();
        assert_eq!(survivors.len(), 3);
    }
}
