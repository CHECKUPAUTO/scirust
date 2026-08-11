//! Validation helpers for externally collected autotuning evidence.
//!
//! Kernel families own correctness proofs and timing collection. ElasticAutoTuner
//! owns the policy boundary that decides whether those opaque proofs are coherent
//! enough to be promoted into a reusable execution plan.

use crate::{ElasticCandidate, ElasticEvidence, ElasticEvidenceError, ElasticMeasurement};

impl ElasticEvidence {
    /// Construct evidence only when its correctness and timing records satisfy
    /// the structural invariants required by plan promotion.
    pub fn validated(
        candidate: ElasticCandidate,
        correctness_evidence: Vec<u8>,
        measurement: ElasticMeasurement,
    ) -> Result<Self, ElasticEvidenceError> {
        validate_measurement(measurement)?;
        if correctness_evidence.is_empty()
        {
            return Err(ElasticEvidenceError::MissingCorrectnessEvidence);
        }
        Ok(Self {
            candidate,
            correctness_evidence,
            measurement,
        })
    }

    /// Revalidate evidence received from storage or another process boundary.
    pub fn validate(&self) -> Result<(), ElasticEvidenceError> {
        if self.correctness_evidence.is_empty()
        {
            return Err(ElasticEvidenceError::MissingCorrectnessEvidence);
        }
        validate_measurement(self.measurement)
    }
}

fn validate_measurement(measurement: ElasticMeasurement) -> Result<(), ElasticEvidenceError> {
    if measurement.sample_count == 0
    {
        return Err(ElasticEvidenceError::NoMeasurements);
    }
    if measurement.median_ns > measurement.p95_ns || measurement.p95_ns > measurement.p99_ns
    {
        return Err(ElasticEvidenceError::NonMonotonicQuantiles);
    }
    if measurement.mad_ns > measurement.p99_ns
    {
        return Err(ElasticEvidenceError::MadExceedsP99);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElasticParameter;

    fn candidate() -> ElasticCandidate {
        ElasticCandidate::new(
            "sgemm-f32",
            b"rev1".to_vec(),
            [ElasticParameter {
                name: "path".to_string(),
                value: 0,
            }],
            true,
            0,
        )
        .unwrap()
    }

    fn measurement() -> ElasticMeasurement {
        ElasticMeasurement {
            sample_count: 21,
            median_ns: 100,
            p95_ns: 120,
            p99_ns: 140,
            mad_ns: 5,
        }
    }

    #[test]
    fn accepted_evidence_round_trips_validation() {
        let evidence =
            ElasticEvidence::validated(candidate(), vec![1, 2, 3], measurement()).unwrap();
        assert_eq!(evidence.validate(), Ok(()));
    }

    #[test]
    fn empty_correctness_proof_fails_closed() {
        assert_eq!(
            ElasticEvidence::validated(candidate(), Vec::new(), measurement()),
            Err(ElasticEvidenceError::MissingCorrectnessEvidence)
        );
    }

    #[test]
    fn zero_samples_are_rejected() {
        let mut measurement = measurement();
        measurement.sample_count = 0;
        assert_eq!(
            ElasticEvidence::validated(candidate(), vec![1], measurement),
            Err(ElasticEvidenceError::NoMeasurements)
        );
    }

    #[test]
    fn quantiles_must_be_monotonic() {
        let mut measurement = measurement();
        measurement.p95_ns = 150;
        assert_eq!(
            ElasticEvidence::validated(candidate(), vec![1], measurement),
            Err(ElasticEvidenceError::NonMonotonicQuantiles)
        );
    }

    #[test]
    fn pathological_mad_is_rejected() {
        let mut measurement = measurement();
        measurement.mad_ns = 141;
        assert_eq!(
            ElasticEvidence::validated(candidate(), vec![1], measurement),
            Err(ElasticEvidenceError::MadExceedsP99)
        );
    }
}
