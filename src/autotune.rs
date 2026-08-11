//! Evidence-driven autotuning for concrete SciRust kernels.
//!
//! This module is available with the `autotune` feature. It keeps the generic
//! [`ElasticAutoTuner`] independent from kernel implementations while providing
//! a first complete integration for SciRust's real `f32` SGEMM candidates.
//!
//! No function in this module reads a clock. Callers collect timing samples under
//! an explicit [`ElasticMeasurementProtocol`], while SciRust owns correctness
//! qualification, canonical correctness evidence, measured selection and exact
//! prepared-kernel reconstruction.

pub use elastic_autotuner::measurement_protocol::{
    ElasticMeasurementProtocol, ElasticMeasurementProtocolError, ElasticResidenceMode,
    ElasticSynchronizationBoundary, ElasticTimingSource,
};
pub use elastic_autotuner::{
    ElasticAutoTuner, ElasticCandidate, ElasticCandidateError, ElasticConfig, ElasticEvidence,
    ElasticEvidenceError, ElasticExecutionPlan, ElasticHardwareProfile, ElasticMode,
    ElasticObjective, ElasticParameter, ElasticProblemClass, ElasticSelectionError,
    RankedCandidate,
};

use elastic_autotuner::{ElasticConstraintSolver, ElasticCostModel, ElasticSearchSpace};
use scirust_compute::probe_host_cpu;
use scirust_simd::matrix::candidate_evidence::{
    GemmCorrectnessEvidenceError, encode_gemm_correctness_evidence,
};
use scirust_simd::matrix::candidate_plan::{CandidateGemmPlanError, CandidateGemmPlanF32};
use scirust_simd::matrix::candidate_qualification::{
    GemmQualificationError, GemmQualificationInput, GemmQualificationPolicy,
    GemmQualificationReport, qualify_gemm_candidate_f32,
};
use scirust_simd::matrix::gemm_candidates::{
    GemmCandidateDescriptor, GemmCandidateError, GemmProblemSignature,
    available_gemm_candidates_f32,
};
use scirust_simd::matrix::gemm_plan::{GemmExecutionPath, GemmPlanError, GemmPlanF32};

const SGEMM_ELASTIC_FAMILY: &str = "scirust-sgemm-f32";
const SGEMM_KERNEL_REVISION: &[u8] = b"scirust-simd-sgemm-candidate-v1";

/// Complete ElasticAutoTuner adapter for one fixed row-major `f32` GEMM shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SgemmElasticAdapter {
    problem: GemmProblemSignature,
    problem_class: ElasticProblemClass,
    preferred_path: GemmExecutionPath,
}

impl SgemmElasticAdapter {
    /// Prepare the tuning boundary for `m×k · k×n` on the current host.
    pub fn new(m: usize, k: usize, n: usize) -> Result<Self, SgemmElasticError> {
        let problem = GemmProblemSignature::new(m, k, n)
            .map_err(SgemmElasticError::CandidateShape)?;
        let class_key = problem
            .class_key()
            .map_err(SgemmElasticError::CandidateShape)?;
        let preferred_path = GemmPlanF32::prepare(m, k, n)
            .map_err(SgemmElasticError::PreferredPlan)?
            .path();
        Ok(Self {
            problem,
            problem_class: ElasticProblemClass::new(SGEMM_ELASTIC_FAMILY, class_key.to_vec()),
            preferred_path,
        })
    }

    pub const fn problem(&self) -> GemmProblemSignature {
        self.problem
    }

    pub fn problem_class(&self) -> &ElasticProblemClass {
        &self.problem_class
    }

    /// Probe the CPU executing this process and turn the result into the canonical
    /// hardware identity used by ElasticAutoTuner plan/evidence caches.
    pub fn host_hardware_profile(&self) -> Result<ElasticHardwareProfile, SgemmElasticError> {
        ElasticHardwareProfile::from_capabilities(&probe_host_cpu())
            .map_err(SgemmElasticError::HardwareProfile)
    }

    /// Return the concrete SGEMM implementations executable on this host, encoded
    /// as canonical Elastic candidates.
    pub fn elastic_candidates(&self) -> Result<Vec<ElasticCandidate>, SgemmElasticError> {
        available_gemm_candidates_f32(self.problem)
            .into_iter()
            .map(elastic_candidate_from_descriptor)
            .collect()
    }

    /// Rank the concrete host-executable SGEMM candidates using ElasticAutoTuner's
    /// static phase. Measured evidence can later override this heuristic ordering.
    pub fn rank_candidates(
        &self,
        tuner: &ElasticAutoTuner,
        hardware: &ElasticHardwareProfile,
    ) -> Vec<RankedCandidate> {
        tuner.rank_candidates(
            hardware,
            &self.problem_class,
            self,
            self,
            self,
        )
    }

    /// Qualify one exact candidate against SciRust's scalar oracle and bind its
    /// caller-collected timing samples to the declared measurement protocol.
    ///
    /// Rejected/non-finite numerical results can never become positive evidence.
    pub fn qualify_and_measure(
        &self,
        candidate: &ElasticCandidate,
        input: GemmQualificationInput<'_>,
        policy: GemmQualificationPolicy,
        protocol: ElasticMeasurementProtocol,
        samples_ns: &[u64],
        scratch: &mut [u64],
    ) -> Result<(GemmQualificationReport, ElasticEvidence), SgemmElasticError> {
        let descriptor = self.descriptor_for_candidate(candidate)?;
        let report = qualify_gemm_candidate_f32(self.problem, descriptor, input, policy)
            .map_err(SgemmElasticError::Qualification)?;
        let correctness_evidence =
            encode_gemm_correctness_evidence(self.problem, policy, report)
                .map_err(SgemmElasticError::CorrectnessEvidence)?;
        let measurement = protocol
            .summarize(samples_ns, scratch)
            .map_err(SgemmElasticError::MeasurementProtocol)?;
        let evidence = ElasticEvidence::validated(
            candidate.clone(),
            correctness_evidence,
            measurement,
        )
        .map_err(SgemmElasticError::Evidence)?;
        Ok((report, evidence))
    }

    /// Select measured evidence under `tuner`'s objective and reconstruct the
    /// exact currently-executable SciRust SGEMM plan it identifies.
    pub fn select_plan(
        &self,
        tuner: &ElasticAutoTuner,
        hardware: ElasticHardwareProfile,
        evidence: &[ElasticEvidence],
    ) -> Result<SgemmElasticPlan, SgemmElasticError> {
        let ranked = self.rank_candidates(tuner, &hardware);
        let selected = tuner
            .select_measured_evidence(&ranked, evidence)
            .map_err(SgemmElasticError::Selection)?;
        let descriptor = self.descriptor_for_candidate(&selected.candidate)?;
        let elastic_plan = tuner
            .plan_from_evidence(hardware, self.problem_class.clone(), selected)
            .map_err(SgemmElasticError::Evidence)?;
        let kernel_plan = CandidateGemmPlanF32::prepare(self.problem, descriptor)
            .map_err(SgemmElasticError::KernelPlan)?;
        Ok(SgemmElasticPlan {
            elastic_plan,
            kernel_plan,
        })
    }

    /// Resolve an Elastic candidate back to the exact descriptor currently
    /// executable on this host. Stale, modified or foreign candidates fail closed.
    pub fn descriptor_for_candidate(
        &self,
        candidate: &ElasticCandidate,
    ) -> Result<GemmCandidateDescriptor, SgemmElasticError> {
        for descriptor in available_gemm_candidates_f32(self.problem)
        {
            let encoded = elastic_candidate_from_descriptor(descriptor)?;
            if &encoded == candidate
            {
                return Ok(descriptor);
            }
        }
        Err(SgemmElasticError::UnknownCandidate)
    }
}

impl ElasticSearchSpace for SgemmElasticAdapter {
    fn candidates(
        &self,
        _hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        output: &mut Vec<ElasticCandidate>,
    ) {
        if problem != &self.problem_class
        {
            return;
        }
        for descriptor in available_gemm_candidates_f32(self.problem)
        {
            if let Ok(candidate) = elastic_candidate_from_descriptor(descriptor)
            {
                output.push(candidate);
            }
        }
    }
}

impl ElasticConstraintSolver for SgemmElasticAdapter {
    fn is_valid(
        &self,
        _hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        candidate: &ElasticCandidate,
    ) -> bool {
        problem == &self.problem_class && self.descriptor_for_candidate(candidate).is_ok()
    }
}

impl ElasticCostModel for SgemmElasticAdapter {
    fn estimated_cost_units(
        &self,
        _hardware: &ElasticHardwareProfile,
        problem: &ElasticProblemClass,
        candidate: &ElasticCandidate,
        objective: ElasticObjective,
    ) -> u64 {
        if problem != &self.problem_class
        {
            return u64::MAX;
        }
        let Ok(descriptor) = self.descriptor_for_candidate(candidate) else {
            return u64::MAX;
        };
        match objective
        {
            ElasticObjective::MinTemporaryMemory => candidate.temporary_bytes,
            ElasticObjective::MinLatency
            | ElasticObjective::MaxThroughput
            | ElasticObjective::BalancedLatencyMemory
            | ElasticObjective::DeterministicOnly =>
            {
                u64::from(descriptor.path != self.preferred_path)
            },
        }
    }
}

/// A selected Elastic execution plan bound back to the exact prepared SGEMM
/// implementation that can be executed repeatedly without candidate replanning.
#[derive(Debug)]
pub struct SgemmElasticPlan {
    elastic_plan: ElasticExecutionPlan,
    kernel_plan: CandidateGemmPlanF32,
}

impl SgemmElasticPlan {
    pub fn elastic_plan(&self) -> &ElasticExecutionPlan {
        &self.elastic_plan
    }

    pub const fn candidate(&self) -> GemmCandidateDescriptor {
        self.kernel_plan.candidate()
    }

    pub const fn path(&self) -> GemmExecutionPath {
        self.kernel_plan.candidate().path
    }

    pub fn workspace_identities(&self) -> Option<(usize, usize)> {
        self.kernel_plan.workspace_identities()
    }

    /// Execute `C = alpha·A·B + beta·C` using the exact selected candidate.
    pub fn execute(
        &mut self,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
    ) -> Result<(), SgemmElasticError> {
        self.kernel_plan
            .execute(alpha, a, b, beta, c)
            .map_err(SgemmElasticError::KernelPlan)
    }
}

#[derive(Debug)]
pub enum SgemmElasticError {
    CandidateShape(GemmCandidateError),
    PreferredPlan(GemmPlanError),
    HardwareProfile(scirust_compute::ProfileEncodingError),
    CandidateEncoding(ElasticCandidateError),
    TemporaryBytesTooLarge,
    UnknownCandidate,
    Qualification(GemmQualificationError),
    CorrectnessEvidence(GemmCorrectnessEvidenceError),
    MeasurementProtocol(ElasticMeasurementProtocolError),
    Evidence(ElasticEvidenceError),
    Selection(ElasticSelectionError),
    KernelPlan(CandidateGemmPlanError),
}

impl core::fmt::Display for SgemmElasticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::CandidateShape(error) => write!(f, "invalid SGEMM problem: {error}"),
            Self::PreferredPlan(error) => write!(f, "cannot prepare SGEMM reference plan: {error}"),
            Self::HardwareProfile(error) => write!(f, "cannot encode host hardware profile: {error}"),
            Self::CandidateEncoding(error) => write!(f, "cannot encode SGEMM candidate: {error}"),
            Self::TemporaryBytesTooLarge =>
            {
                write!(f, "SGEMM temporary-byte requirement does not fit canonical u64")
            },
            Self::UnknownCandidate =>
            {
                write!(f, "Elastic candidate is not executable by this SGEMM adapter")
            },
            Self::Qualification(error) => write!(f, "SGEMM correctness qualification failed: {error}"),
            Self::CorrectnessEvidence(error) =>
            {
                write!(f, "SGEMM correctness evidence encoding failed: {error}")
            },
            Self::MeasurementProtocol(error) =>
            {
                write!(f, "SGEMM measurement protocol failed: {error}")
            },
            Self::Evidence(error) => write!(f, "Elastic evidence rejected: {error}"),
            Self::Selection(error) => write!(f, "Elastic measured selection failed: {error}"),
            Self::KernelPlan(error) => write!(f, "selected SGEMM plan preparation/execution failed: {error}"),
        }
    }
}

impl std::error::Error for SgemmElasticError {}

fn elastic_candidate_from_descriptor(
    descriptor: GemmCandidateDescriptor,
) -> Result<ElasticCandidate, SgemmElasticError> {
    let temporary_bytes = u64::try_from(descriptor.temporary_bytes)
        .map_err(|_| SgemmElasticError::TemporaryBytesTooLarge)?;
    ElasticCandidate::new(
        SGEMM_ELASTIC_FAMILY,
        SGEMM_KERNEL_REVISION.to_vec(),
        descriptor
            .tuning_parameters()
            .into_iter()
            .map(|(name, value)| ElasticParameter {
                name: name.to_string(),
                value,
            }),
        descriptor.deterministic,
        temporary_bytes,
    )
    .map_err(SgemmElasticError::CandidateEncoding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_host_candidate_round_trips_through_elastic_identity() {
        let adapter = SgemmElasticAdapter::new(17, 23, 19).unwrap();
        let candidates = adapter.elastic_candidates().unwrap();
        assert!(!candidates.is_empty());
        for candidate in candidates
        {
            let descriptor = adapter.descriptor_for_candidate(&candidate).unwrap();
            assert_eq!(elastic_candidate_from_descriptor(descriptor).unwrap(), candidate);
        }
    }

    #[test]
    fn scalar_candidate_can_be_qualified_and_measured_without_reading_a_clock() {
        let adapter = SgemmElasticAdapter::new(3, 5, 4).unwrap();
        let candidates = adapter.elastic_candidates().unwrap();
        let scalar = &candidates[0];
        let a: Vec<f32> = (0..15).map(|index| index as f32 * 0.02 - 0.1).collect();
        let b: Vec<f32> = (0..20).map(|index| index as f32 * 0.015 - 0.08).collect();
        let initial_c = vec![0.0_f32; 12];
        let protocol = ElasticMeasurementProtocol::new(
            2,
            5,
            ElasticTimingSource::HostWallClock,
            ElasticResidenceMode::Resident,
            ElasticSynchronizationBoundary::PerIteration,
        );
        let samples = [101_u64, 99, 100, 102, 98];
        let mut scratch = [0_u64; 5];
        let (report, evidence) = adapter
            .qualify_and_measure(
                scalar,
                GemmQualificationInput {
                    alpha: 1.0,
                    a: &a,
                    b: &b,
                    beta: 0.0,
                    initial_c: &initial_c,
                },
                GemmQualificationPolicy::default(),
                protocol,
                &samples,
                &mut scratch,
            )
            .unwrap();
        assert!(report.accepted);
        assert!(report.finite);
        assert!(!evidence.correctness_evidence.is_empty());
        assert_eq!(evidence.measurement.median_ns, 100);
    }

    #[test]
    fn modified_candidate_fails_closed() {
        let adapter = SgemmElasticAdapter::new(4, 4, 4).unwrap();
        let mut candidate = adapter.elastic_candidates().unwrap().remove(0);
        candidate.kernel_revision.push(0xff);
        assert!(matches!(
            adapter.descriptor_for_candidate(&candidate),
            Err(SgemmElasticError::UnknownCandidate)
        ));
    }

    #[test]
    fn full_selection_reconstructs_an_executable_plan() {
        let adapter = SgemmElasticAdapter::new(2, 3, 2).unwrap();
        let hardware = adapter.host_hardware_profile().unwrap();
        let tuner = ElasticAutoTuner::new(ElasticConfig::default());
        let candidates = adapter.elastic_candidates().unwrap();
        let a = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b = [0.2_f32, -0.1, 0.3, 0.4, -0.2, 0.5];
        let initial_c = [0.0_f32; 4];
        let protocol = ElasticMeasurementProtocol::new(
            1,
            3,
            ElasticTimingSource::HostWallClock,
            ElasticResidenceMode::Resident,
            ElasticSynchronizationBoundary::PerIteration,
        );
        let mut evidence = Vec::new();
        for (index, candidate) in candidates.iter().enumerate()
        {
            let base = 100_u64 + index as u64 * 10;
            let samples = [base, base + 1, base + 2];
            let mut scratch = [0_u64; 3];
            let (_, record) = adapter
                .qualify_and_measure(
                    candidate,
                    GemmQualificationInput {
                        alpha: 1.0,
                        a: &a,
                        b: &b,
                        beta: 0.0,
                        initial_c: &initial_c,
                    },
                    GemmQualificationPolicy::default(),
                    protocol,
                    &samples,
                    &mut scratch,
                )
                .unwrap();
            evidence.push(record);
        }
        let mut plan = adapter.select_plan(&tuner, hardware, &evidence).unwrap();
        let identities = plan.workspace_identities();
        let mut c = initial_c;
        plan.execute(1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!(c.iter().all(|value| value.is_finite()));
        assert_eq!(plan.workspace_identities(), identities);
    }
}
