//! Canonical persistent execution-plan records for ElasticAutoTuner.
//!
//! The format is dependency-free, length-prefixed, versioned and protected by
//! CRC-32/IEEE against accidental corruption. CRC is an integrity check, not an
//! authentication or cryptographic signature. Scientific correctness evidence
//! and provenance remain explicit fields in the record.

use crate::measurement_protocol::{
    ElasticMeasurementProtocol, ElasticMeasurementProtocolError, ElasticResidenceMode,
    ElasticSynchronizationBoundary, ElasticTimingSource,
};
use crate::{
    ELASTIC_SCHEMA_VERSION, ElasticCandidate, ElasticEvidence, ElasticEvidenceError,
    ElasticExecutionPlan, ElasticHardwareProfile, ElasticMeasurement, ElasticObjective,
    ElasticParameter, ElasticProblemClass,
};

pub const ELASTIC_PERSISTENCE_SCHEMA_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"ELAUTO01";
const MAX_RECORD_BYTES: usize = 64 * 1024 * 1024;
const MAX_FIELD_BYTES: usize = 16 * 1024 * 1024;
const MAX_PARAMETERS: usize = 4096;
const MAX_INVALIDATION_DEPENDENCIES: usize = 4096;

/// Fully reproducible persisted selection/evidence record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElasticPersistedPlan {
    pub schema_version: u32,
    pub plan: ElasticExecutionPlan,
    pub measurement_protocol: ElasticMeasurementProtocol,
    pub selected: bool,
    /// Caller-supplied timestamp. The autotuner never reads a clock itself.
    pub recorded_unix_ns: u64,
    /// Caller-defined deterministic provenance payload.
    pub provenance: Vec<u8>,
    /// Canonically sorted/deduplicated dependency identities that invalidate the
    /// record when their semantics or implementation revisions change.
    pub invalidation_dependencies: Vec<Vec<u8>>,
}

impl ElasticPersistedPlan {
    pub fn new(
        plan: ElasticExecutionPlan,
        measurement_protocol: ElasticMeasurementProtocol,
        selected: bool,
        recorded_unix_ns: u64,
        provenance: Vec<u8>,
        mut invalidation_dependencies: Vec<Vec<u8>>,
    ) -> Result<Self, ElasticPersistenceError> {
        invalidation_dependencies.sort();
        invalidation_dependencies.dedup();
        let record = Self {
            schema_version: ELASTIC_PERSISTENCE_SCHEMA_VERSION,
            plan,
            measurement_protocol,
            selected,
            recorded_unix_ns,
            provenance,
            invalidation_dependencies,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), ElasticPersistenceError> {
        if self.schema_version != ELASTIC_PERSISTENCE_SCHEMA_VERSION
        {
            return Err(ElasticPersistenceError::UnsupportedPersistenceSchema {
                expected: ELASTIC_PERSISTENCE_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.plan.schema_version != ELASTIC_SCHEMA_VERSION
        {
            return Err(ElasticPersistenceError::UnsupportedPlanSchema {
                expected: ELASTIC_SCHEMA_VERSION,
                actual: self.plan.schema_version,
            });
        }
        if self.plan.hardware.schema_version != ELASTIC_SCHEMA_VERSION
        {
            return Err(ElasticPersistenceError::UnsupportedHardwareSchema {
                expected: ELASTIC_SCHEMA_VERSION,
                actual: self.plan.hardware.schema_version,
            });
        }
        self.plan
            .evidence
            .validate()
            .map_err(ElasticPersistenceError::InvalidEvidence)?;
        self.measurement_protocol
            .validate()
            .map_err(ElasticPersistenceError::InvalidMeasurementProtocol)?;
        if self.measurement_protocol.measured_iterations
            != self.plan.evidence.measurement.sample_count
        {
            return Err(ElasticPersistenceError::MeasurementProtocolMismatch {
                protocol_samples: self.measurement_protocol.measured_iterations,
                evidence_samples: self.plan.evidence.measurement.sample_count,
            });
        }
        validate_field_len(self.plan.hardware.canonical_bytes().len())?;
        validate_field_len(self.plan.problem.family().len())?;
        validate_field_len(self.plan.problem.class_key().len())?;
        validate_field_len(self.plan.evidence.candidate.kernel_family.len())?;
        validate_field_len(self.plan.evidence.candidate.kernel_revision.len())?;
        validate_field_len(self.plan.evidence.correctness_evidence.len())?;
        validate_field_len(self.provenance.len())?;
        if self.provenance.is_empty()
        {
            return Err(ElasticPersistenceError::EmptyProvenance);
        }
        if self.plan.evidence.candidate.parameters().len() > MAX_PARAMETERS
        {
            return Err(ElasticPersistenceError::TooManyParameters);
        }
        for parameter in self.plan.evidence.candidate.parameters()
        {
            validate_field_len(parameter.name.len())?;
        }
        if self.invalidation_dependencies.len() > MAX_INVALIDATION_DEPENDENCIES
        {
            return Err(ElasticPersistenceError::TooManyInvalidationDependencies);
        }
        let mut previous: Option<&[u8]> = None;
        for dependency in &self.invalidation_dependencies
        {
            validate_field_len(dependency.len())?;
            if dependency.is_empty()
            {
                return Err(ElasticPersistenceError::EmptyInvalidationDependency);
            }
            if previous.is_some_and(|value| value >= dependency.as_slice())
            {
                return Err(ElasticPersistenceError::NonCanonicalInvalidationDependencies);
            }
            previous = Some(dependency);
        }
        Ok(())
    }

    /// Encode this record using the canonical v1 binary format.
    pub fn encode(&self) -> Result<Vec<u8>, ElasticPersistenceError> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        push_u32(&mut out, self.schema_version);
        push_u64(&mut out, self.recorded_unix_ns);
        push_u8(&mut out, u8::from(self.selected));
        push_u8(&mut out, objective_code(self.plan.objective));
        push_u32(&mut out, self.plan.schema_version);

        push_u32(&mut out, self.plan.hardware.schema_version);
        push_bytes(&mut out, self.plan.hardware.canonical_bytes())?;

        push_string(&mut out, self.plan.problem.family())?;
        push_bytes(&mut out, self.plan.problem.class_key())?;

        let candidate = &self.plan.evidence.candidate;
        push_string(&mut out, &candidate.kernel_family)?;
        push_bytes(&mut out, &candidate.kernel_revision)?;
        push_u8(&mut out, u8::from(candidate.deterministic));
        push_u64(&mut out, candidate.temporary_bytes);
        push_count(&mut out, candidate.parameters().len())?;
        for parameter in candidate.parameters()
        {
            push_string(&mut out, &parameter.name)?;
            push_i64(&mut out, parameter.value);
        }

        push_bytes(&mut out, &self.plan.evidence.correctness_evidence)?;
        encode_measurement(&mut out, self.plan.evidence.measurement);
        encode_protocol(&mut out, self.measurement_protocol);
        push_bytes(&mut out, &self.provenance)?;
        push_count(&mut out, self.invalidation_dependencies.len())?;
        for dependency in &self.invalidation_dependencies
        {
            push_bytes(&mut out, dependency)?;
        }

        if out.len() + 4 > MAX_RECORD_BYTES
        {
            return Err(ElasticPersistenceError::RecordTooLarge);
        }
        let checksum = crc32_ieee(&out);
        push_u32(&mut out, checksum);
        Ok(out)
    }

    /// Decode and fully validate a canonical persisted record.
    pub fn decode(bytes: &[u8]) -> Result<Self, ElasticPersistenceError> {
        if bytes.len() < MAGIC.len() + 4 + 4
        {
            return Err(ElasticPersistenceError::TruncatedRecord);
        }
        if bytes.len() > MAX_RECORD_BYTES
        {
            return Err(ElasticPersistenceError::RecordTooLarge);
        }
        let payload_len = bytes.len() - 4;
        let payload = &bytes[..payload_len];
        let expected_checksum = u32::from_le_bytes(
            bytes[payload_len..]
                .try_into()
                .map_err(|_| ElasticPersistenceError::TruncatedRecord)?,
        );
        let actual_checksum = crc32_ieee(payload);
        if actual_checksum != expected_checksum
        {
            return Err(ElasticPersistenceError::ChecksumMismatch);
        }

        let mut reader = Reader::new(payload);
        if reader.read_exact(MAGIC.len())? != MAGIC
        {
            return Err(ElasticPersistenceError::InvalidMagic);
        }
        let schema_version = reader.read_u32()?;
        if schema_version != ELASTIC_PERSISTENCE_SCHEMA_VERSION
        {
            return Err(ElasticPersistenceError::UnsupportedPersistenceSchema {
                expected: ELASTIC_PERSISTENCE_SCHEMA_VERSION,
                actual: schema_version,
            });
        }
        let recorded_unix_ns = reader.read_u64()?;
        let selected = decode_bool(reader.read_u8()?)?;
        let objective = decode_objective(reader.read_u8()?)?;
        let plan_schema_version = reader.read_u32()?;

        let hardware_schema_version = reader.read_u32()?;
        let hardware_bytes = reader.read_bytes()?;
        let hardware = ElasticHardwareProfile {
            schema_version: hardware_schema_version,
            canonical_bytes: hardware_bytes,
        };

        let problem = ElasticProblemClass::new(reader.read_string()?, reader.read_bytes()?);

        let kernel_family = reader.read_string()?;
        let kernel_revision = reader.read_bytes()?;
        let deterministic = decode_bool(reader.read_u8()?)?;
        let temporary_bytes = reader.read_u64()?;
        let parameter_count = reader.read_count(MAX_PARAMETERS)?;
        let mut parameters = Vec::with_capacity(parameter_count);
        for _ in 0..parameter_count
        {
            parameters.push(ElasticParameter {
                name: reader.read_string()?,
                value: reader.read_i64()?,
            });
        }
        let candidate = ElasticCandidate::new(
            kernel_family,
            kernel_revision,
            parameters,
            deterministic,
            temporary_bytes,
        )
        .map_err(|_| ElasticPersistenceError::InvalidCandidate)?;

        let correctness_evidence = reader.read_bytes()?;
        let measurement = decode_measurement(&mut reader)?;
        let measurement_protocol = decode_protocol(&mut reader)?;
        let provenance = reader.read_bytes()?;
        let dependency_count = reader.read_count(MAX_INVALIDATION_DEPENDENCIES)?;
        let mut invalidation_dependencies = Vec::with_capacity(dependency_count);
        for _ in 0..dependency_count
        {
            invalidation_dependencies.push(reader.read_bytes()?);
        }
        if !reader.is_empty()
        {
            return Err(ElasticPersistenceError::TrailingBytes);
        }

        let record = Self {
            schema_version,
            plan: ElasticExecutionPlan {
                schema_version: plan_schema_version,
                hardware,
                problem,
                objective,
                evidence: ElasticEvidence {
                    candidate,
                    correctness_evidence,
                    measurement,
                },
            },
            measurement_protocol,
            selected,
            recorded_unix_ns,
            provenance,
            invalidation_dependencies,
        };
        record.validate()?;
        Ok(record)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticPersistenceError {
    UnsupportedPersistenceSchema { expected: u32, actual: u32 },
    UnsupportedPlanSchema { expected: u32, actual: u32 },
    UnsupportedHardwareSchema { expected: u32, actual: u32 },
    InvalidEvidence(ElasticEvidenceError),
    InvalidMeasurementProtocol(ElasticMeasurementProtocolError),
    MeasurementProtocolMismatch { protocol_samples: u32, evidence_samples: u32 },
    EmptyProvenance,
    EmptyInvalidationDependency,
    NonCanonicalInvalidationDependencies,
    FieldTooLarge,
    RecordTooLarge,
    TooManyParameters,
    TooManyInvalidationDependencies,
    CountTooLarge,
    TruncatedRecord,
    InvalidMagic,
    ChecksumMismatch,
    InvalidBoolean,
    InvalidObjective,
    InvalidTimingSource,
    InvalidResidenceMode,
    InvalidSynchronizationBoundary,
    InvalidUtf8,
    InvalidCandidate,
    TrailingBytes,
}

impl core::fmt::Display for ElasticPersistenceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self
        {
            Self::UnsupportedPersistenceSchema { expected, actual } => write!(
                f,
                "persistence schema mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedPlanSchema { expected, actual } =>
            {
                write!(f, "plan schema mismatch: expected {expected}, got {actual}")
            },
            Self::UnsupportedHardwareSchema { expected, actual } => write!(
                f,
                "hardware profile schema mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidEvidence(error) => write!(f, "invalid persisted evidence: {error}"),
            Self::InvalidMeasurementProtocol(error) =>
            {
                write!(f, "invalid persisted measurement protocol: {error}")
            },
            Self::MeasurementProtocolMismatch {
                protocol_samples,
                evidence_samples,
            } => write!(
                f,
                "measurement protocol/evidence sample mismatch: protocol {protocol_samples}, evidence {evidence_samples}"
            ),
            Self::EmptyProvenance => write!(f, "persisted plan provenance must not be empty"),
            Self::EmptyInvalidationDependency =>
            {
                write!(f, "invalidation dependency identities must not be empty")
            },
            Self::NonCanonicalInvalidationDependencies => write!(
                f,
                "invalidation dependencies must be strictly sorted and deduplicated"
            ),
            Self::FieldTooLarge => write!(f, "persisted field exceeds the format size limit"),
            Self::RecordTooLarge => write!(f, "persisted record exceeds the format size limit"),
            Self::TooManyParameters => write!(f, "persisted candidate has too many parameters"),
            Self::TooManyInvalidationDependencies =>
            {
                write!(f, "persisted record has too many invalidation dependencies")
            },
            Self::CountTooLarge => write!(f, "persisted count does not fit the canonical u32 format"),
            Self::TruncatedRecord => write!(f, "persisted record is truncated"),
            Self::InvalidMagic => write!(f, "persisted record has an invalid magic header"),
            Self::ChecksumMismatch => write!(f, "persisted record CRC32 integrity check failed"),
            Self::InvalidBoolean => write!(f, "persisted record contains an invalid boolean tag"),
            Self::InvalidObjective => write!(f, "persisted record contains an invalid objective tag"),
            Self::InvalidTimingSource =>
            {
                write!(f, "persisted record contains an invalid timing-source tag")
            },
            Self::InvalidResidenceMode =>
            {
                write!(f, "persisted record contains an invalid residence-mode tag")
            },
            Self::InvalidSynchronizationBoundary => write!(
                f,
                "persisted record contains an invalid synchronization-boundary tag"
            ),
            Self::InvalidUtf8 => write!(f, "persisted string field is not valid UTF-8"),
            Self::InvalidCandidate => write!(f, "persisted candidate is not canonical"),
            Self::TrailingBytes => write!(f, "persisted record contains trailing payload bytes"),
        }
    }
}

impl std::error::Error for ElasticPersistenceError {}

fn validate_field_len(len: usize) -> Result<(), ElasticPersistenceError> {
    if len > MAX_FIELD_BYTES
    {
        return Err(ElasticPersistenceError::FieldTooLarge);
    }
    Ok(())
}

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_count(out: &mut Vec<u8>, value: usize) -> Result<(), ElasticPersistenceError> {
    let value = u32::try_from(value).map_err(|_| ElasticPersistenceError::CountTooLarge)?;
    push_u32(out, value);
    Ok(())
}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), ElasticPersistenceError> {
    validate_field_len(value.len())?;
    push_count(out, value.len())?;
    out.extend_from_slice(value);
    Ok(())
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), ElasticPersistenceError> {
    push_bytes(out, value.as_bytes())
}

fn encode_measurement(out: &mut Vec<u8>, measurement: ElasticMeasurement) {
    push_u32(out, measurement.sample_count);
    push_u64(out, measurement.median_ns);
    push_u64(out, measurement.p95_ns);
    push_u64(out, measurement.p99_ns);
    push_u64(out, measurement.mad_ns);
}

fn decode_measurement(reader: &mut Reader<'_>) -> Result<ElasticMeasurement, ElasticPersistenceError> {
    Ok(ElasticMeasurement {
        sample_count: reader.read_u32()?,
        median_ns: reader.read_u64()?,
        p95_ns: reader.read_u64()?,
        p99_ns: reader.read_u64()?,
        mad_ns: reader.read_u64()?,
    })
}

fn encode_protocol(out: &mut Vec<u8>, protocol: ElasticMeasurementProtocol) {
    push_u32(out, protocol.schema_version);
    push_u32(out, protocol.warmup_iterations);
    push_u32(out, protocol.measured_iterations);
    push_u8(out, timing_code(protocol.timing_source));
    push_u8(out, residence_code(protocol.residence_mode));
    push_u8(out, synchronization_code(protocol.synchronization));
}

fn decode_protocol(
    reader: &mut Reader<'_>,
) -> Result<ElasticMeasurementProtocol, ElasticPersistenceError> {
    Ok(ElasticMeasurementProtocol {
        schema_version: reader.read_u32()?,
        warmup_iterations: reader.read_u32()?,
        measured_iterations: reader.read_u32()?,
        timing_source: decode_timing(reader.read_u8()?)?,
        residence_mode: decode_residence(reader.read_u8()?)?,
        synchronization: decode_synchronization(reader.read_u8()?)?,
    })
}

const fn objective_code(objective: ElasticObjective) -> u8 {
    match objective {
        ElasticObjective::MinLatency => 0,
        ElasticObjective::MaxThroughput => 1,
        ElasticObjective::MinTemporaryMemory => 2,
        ElasticObjective::BalancedLatencyMemory => 3,
        ElasticObjective::DeterministicOnly => 4,
    }
}

fn decode_objective(code: u8) -> Result<ElasticObjective, ElasticPersistenceError> {
    match code {
        0 => Ok(ElasticObjective::MinLatency),
        1 => Ok(ElasticObjective::MaxThroughput),
        2 => Ok(ElasticObjective::MinTemporaryMemory),
        3 => Ok(ElasticObjective::BalancedLatencyMemory),
        4 => Ok(ElasticObjective::DeterministicOnly),
        _ => Err(ElasticPersistenceError::InvalidObjective),
    }
}

const fn timing_code(value: ElasticTimingSource) -> u8 {
    match value {
        ElasticTimingSource::HostWallClock => 0,
        ElasticTimingSource::DeviceTimestamp => 1,
    }
}

fn decode_timing(code: u8) -> Result<ElasticTimingSource, ElasticPersistenceError> {
    match code {
        0 => Ok(ElasticTimingSource::HostWallClock),
        1 => Ok(ElasticTimingSource::DeviceTimestamp),
        _ => Err(ElasticPersistenceError::InvalidTimingSource),
    }
}

const fn residence_code(value: ElasticResidenceMode) -> u8 {
    match value {
        ElasticResidenceMode::Resident => 0,
        ElasticResidenceMode::TransferInclusive => 1,
    }
}

fn decode_residence(code: u8) -> Result<ElasticResidenceMode, ElasticPersistenceError> {
    match code {
        0 => Ok(ElasticResidenceMode::Resident),
        1 => Ok(ElasticResidenceMode::TransferInclusive),
        _ => Err(ElasticPersistenceError::InvalidResidenceMode),
    }
}

const fn synchronization_code(value: ElasticSynchronizationBoundary) -> u8 {
    match value {
        ElasticSynchronizationBoundary::PerIteration => 0,
        ElasticSynchronizationBoundary::BatchEnd => 1,
    }
}

fn decode_synchronization(
    code: u8,
) -> Result<ElasticSynchronizationBoundary, ElasticPersistenceError> {
    match code {
        0 => Ok(ElasticSynchronizationBoundary::PerIteration),
        1 => Ok(ElasticSynchronizationBoundary::BatchEnd),
        _ => Err(ElasticPersistenceError::InvalidSynchronizationBoundary),
    }
}

fn decode_bool(code: u8) -> Result<bool, ElasticPersistenceError> {
    match code {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(ElasticPersistenceError::InvalidBoolean),
    }
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes
    {
        crc ^= u32::from(byte);
        for _ in 0..8
        {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], ElasticPersistenceError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ElasticPersistenceError::TruncatedRecord)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ElasticPersistenceError::TruncatedRecord)?;
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, ElasticPersistenceError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, ElasticPersistenceError> {
        let bytes: [u8; 4] = self
            .read_exact(4)?
            .try_into()
            .map_err(|_| ElasticPersistenceError::TruncatedRecord)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, ElasticPersistenceError> {
        let bytes: [u8; 8] = self
            .read_exact(8)?
            .try_into()
            .map_err(|_| ElasticPersistenceError::TruncatedRecord)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64, ElasticPersistenceError> {
        let bytes: [u8; 8] = self
            .read_exact(8)?
            .try_into()
            .map_err(|_| ElasticPersistenceError::TruncatedRecord)?;
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_count(&mut self, maximum: usize) -> Result<usize, ElasticPersistenceError> {
        let value = usize::try_from(self.read_u32()?)
            .map_err(|_| ElasticPersistenceError::CountTooLarge)?;
        if value > maximum
        {
            return Err(ElasticPersistenceError::CountTooLarge);
        }
        Ok(value)
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, ElasticPersistenceError> {
        let len = self.read_count(MAX_FIELD_BYTES)?;
        Ok(self.read_exact(len)?.to_vec())
    }

    fn read_string(&mut self) -> Result<String, ElasticPersistenceError> {
        String::from_utf8(self.read_bytes()?).map_err(|_| ElasticPersistenceError::InvalidUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement_protocol::{
        ElasticResidenceMode, ElasticSynchronizationBoundary, ElasticTimingSource,
    };
    use scirust_compute::{DeviceCapabilities, HardwareCapabilities};

    fn fixture() -> ElasticPersistedPlan {
        let capabilities =
            HardwareCapabilities::from_device_capabilities(&DeviceCapabilities::reference_cpu());
        let hardware = ElasticHardwareProfile::from_capabilities(&capabilities).unwrap();
        let candidate = ElasticCandidate::new(
            "sgemm-f32",
            b"kernel-rev-1".to_vec(),
            [
                ElasticParameter {
                    name: "path".into(),
                    value: 0,
                },
                ElasticParameter {
                    name: "mr".into(),
                    value: 8,
                },
            ],
            true,
            4096,
        )
        .unwrap();
        let evidence = ElasticEvidence::validated(
            candidate,
            vec![1, 2, 3, 4],
            ElasticMeasurement {
                sample_count: 5,
                median_ns: 100,
                p95_ns: 120,
                p99_ns: 130,
                mad_ns: 4,
            },
        )
        .unwrap();
        let plan = ElasticExecutionPlan {
            schema_version: ELASTIC_SCHEMA_VERSION,
            hardware,
            problem: ElasticProblemClass::new("sgemm-f32", b"m256-k256-n256".to_vec()),
            objective: ElasticObjective::MinLatency,
            evidence,
        };
        let protocol = ElasticMeasurementProtocol::new(
            2,
            5,
            ElasticTimingSource::HostWallClock,
            ElasticResidenceMode::Resident,
            ElasticSynchronizationBoundary::PerIteration,
        );
        ElasticPersistedPlan::new(
            plan,
            protocol,
            true,
            123_456_789,
            b"runner=x86-test;toolchain=1.89".to_vec(),
            vec![
                b"scirust-simd/rev-2".to_vec(),
                b"kernel-schema/v1".to_vec(),
                b"scirust-simd/rev-2".to_vec(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn canonical_record_round_trips_byte_identically() {
        let record = fixture();
        let encoded = record.encode().unwrap();
        let decoded = ElasticPersistedPlan::decode(&encoded).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(decoded.encode().unwrap(), encoded);
        assert_eq!(decoded.invalidation_dependencies.len(), 2);
    }

    #[test]
    fn corruption_fails_integrity_check() {
        let mut encoded = fixture().encode().unwrap();
        encoded[20] ^= 0x01;
        assert_eq!(
            ElasticPersistedPlan::decode(&encoded),
            Err(ElasticPersistenceError::ChecksumMismatch)
        );
    }

    #[test]
    fn unknown_schema_fails_even_with_recomputed_crc() {
        let mut encoded = fixture().encode().unwrap();
        encoded[8..12].copy_from_slice(&99_u32.to_le_bytes());
        let payload_len = encoded.len() - 4;
        let checksum = crc32_ieee(&encoded[..payload_len]);
        encoded[payload_len..].copy_from_slice(&checksum.to_le_bytes());
        assert_eq!(
            ElasticPersistedPlan::decode(&encoded),
            Err(ElasticPersistenceError::UnsupportedPersistenceSchema {
                expected: ELASTIC_PERSISTENCE_SCHEMA_VERSION,
                actual: 99,
            })
        );
    }

    #[test]
    fn protocol_and_evidence_sample_counts_must_match() {
        let mut record = fixture();
        record.measurement_protocol.measured_iterations = 7;
        assert_eq!(
            record.validate(),
            Err(ElasticPersistenceError::MeasurementProtocolMismatch {
                protocol_samples: 7,
                evidence_samples: 5,
            })
        );
    }

    #[test]
    fn crc_matches_standard_check_vector() {
        assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
    }
}
