//! Explicit, versioned canonical binary encoding for content-integrity digests.
//!
//! The archive content digest is a SHA-256 over an **explicit** canonical byte
//! encoding produced here — never over JSON. The encoding is fully specified so
//! it is stable and reproducible across builds and platforms:
//!
//! * a [`DIGEST_FORMAT_VERSION`] independent of the archive schema version leads
//!   the stream, so any encoding change is versioned;
//! * every composite type is prefixed with a distinct domain-separation tag;
//! * integers are fixed-width little-endian; `usize` is converted to `u64`
//!   through a checked conversion;
//! * enum variants use stable numeric tags;
//! * strings and vectors are length-prefixed (`u64`) then their ordered bytes /
//!   elements;
//! * `f32`/`f64` are encoded through `to_bits`, so `-0.0` and `+0.0` (which the
//!   archive model treats as distinct values) encode differently;
//! * non-finite floats are rejected in fields where they are not valid.
//!
//! This is a **content-integrity / corruption-detection** digest. It is not
//! keyed, so it does not authenticate an archive against a party able to edit
//! both the archive and its digest; authenticity would require a trusted
//! signature or MAC (not provided here).

use sha2::{Digest, Sha256};

use super::archive::HallOfFameEntry;
use super::cost::CostReport;
use super::experiment::{GenerationRecord, PopulationSummary};
use super::fitness::FitnessReport;
use super::generate::{GenerationConfig, OperatorSet};
use super::ir::{TensorInstruction, TensorProgram};
use super::population::{EvolutionConfig, TournamentConfig};
use super::problem::{CaseFixture, ProblemLimits, SuccessCriteria, TensorFixture, TensorProblem};

/// Version of the canonical digest encoding. Bump when the encoding changes.
pub const DIGEST_FORMAT_VERSION: u32 = 1;

// Domain-separation tags for composite types.
const TAG_ARCHIVE: u8 = 0x01;
const TAG_PROBLEM: u8 = 0x02;
const TAG_CASE: u8 = 0x03;
const TAG_FIXTURE: u8 = 0x04;
const TAG_LIMITS: u8 = 0x05;
const TAG_EVOLUTION: u8 = 0x06;
const TAG_GEN_CONFIG: u8 = 0x07;
const TAG_OPERATORS: u8 = 0x08;
const TAG_TOURNAMENT: u8 = 0x09;
const TAG_SUCCESS: u8 = 0x0A;
const TAG_RECORD: u8 = 0x0B;
const TAG_COST: u8 = 0x0C;
const TAG_SUMMARY: u8 = 0x0D;
const TAG_HOF: u8 = 0x0E;
const TAG_FITNESS: u8 = 0x0F;
const TAG_PROGRAM: u8 = 0x10;
const TAG_INSTRUCTION: u8 = 0x11;

/// A failure while producing the canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DigestError {
    /// A non-finite float appeared in a field required to be finite.
    NonFiniteFloat,
    /// A `usize` value did not fit into `u64` (only possible on exotic targets).
    LengthOverflow,
}

/// A deterministic canonical byte encoder.
struct CanonicalEncoder {
    bytes: Vec<u8>,
}

impl CanonicalEncoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn tag(&mut self, tag: u8) {
        self.bytes.push(tag);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u128(&mut self, value: u128) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) -> Result<(), DigestError> {
        let value = u64::try_from(value).map_err(|_| DigestError::LengthOverflow)?;
        self.u64(value);
        Ok(())
    }

    fn bool(&mut self, value: bool) {
        self.bytes.push(value as u8);
    }

    fn f32(&mut self, value: f32) -> Result<(), DigestError> {
        if !value.is_finite()
        {
            return Err(DigestError::NonFiniteFloat);
        }
        self.u32(value.to_bits());
        Ok(())
    }

    fn f64(&mut self, value: f64) -> Result<(), DigestError> {
        if !value.is_finite()
        {
            return Err(DigestError::NonFiniteFloat);
        }
        self.u64(value.to_bits());
        Ok(())
    }

    fn str(&mut self, value: &str) -> Result<(), DigestError> {
        self.usize(value.len())?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn usize_vec(&mut self, values: &[usize]) -> Result<(), DigestError> {
        self.usize(values.len())?;
        for &value in values
        {
            self.usize(value)?;
        }
        Ok(())
    }

    fn digest_hex(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.bytes);
        to_hex(&hasher.finalize())
    }
}

fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes
    {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Compute the content-integrity digest (hex SHA-256) of the deterministic
/// archive fields. The digest field itself is excluded.
#[allow(clippy::too_many_arguments)]
pub fn archive_content_digest(
    schema_version: u32,
    crate_version: &str,
    problem: &TensorProblem,
    seed: u64,
    success: bool,
    generations_executed: usize,
    history: &[GenerationRecord],
    final_population: &PopulationSummary,
    hall_of_fame_capacity: usize,
    hall_of_fame: &[HallOfFameEntry],
    best: &HallOfFameEntry,
) -> Result<String, DigestError> {
    let mut encoder = CanonicalEncoder::new();

    encoder.tag(TAG_ARCHIVE);
    encoder.u32(DIGEST_FORMAT_VERSION);
    encoder.u32(schema_version);
    encoder.str(crate_version)?;
    encode_problem(&mut encoder, problem)?;
    encoder.u64(seed);
    encoder.bool(success);
    encoder.usize(generations_executed)?;

    encoder.usize(history.len())?;
    for record in history
    {
        encode_record(&mut encoder, record)?;
    }

    encode_summary(&mut encoder, final_population)?;

    encoder.usize(hall_of_fame_capacity)?;
    encoder.usize(hall_of_fame.len())?;
    for entry in hall_of_fame
    {
        encode_hof_entry(&mut encoder, entry)?;
    }

    encode_hof_entry(&mut encoder, best)?;

    Ok(encoder.digest_hex())
}

fn encode_problem(
    encoder: &mut CanonicalEncoder,
    problem: &TensorProblem,
) -> Result<(), DigestError> {
    encoder.tag(TAG_PROBLEM);
    encoder.str(&problem.id)?;
    encoder.str(&problem.description)?;
    encoder.usize(problem.cases.len())?;
    for case in &problem.cases
    {
        encode_case(encoder, case)?;
    }
    encode_limits(encoder, &problem.limits)?;
    encode_evolution(encoder, &problem.evolution)?;
    encoder.u64(problem.seed);
    encode_success(encoder, &problem.success)
}

fn encode_case(encoder: &mut CanonicalEncoder, case: &CaseFixture) -> Result<(), DigestError> {
    encoder.tag(TAG_CASE);
    encoder.usize(case.inputs.len())?;
    for fixture in &case.inputs
    {
        encode_fixture(encoder, fixture)?;
    }
    encode_fixture(encoder, &case.expected)
}

fn encode_fixture(
    encoder: &mut CanonicalEncoder,
    fixture: &TensorFixture,
) -> Result<(), DigestError> {
    encoder.tag(TAG_FIXTURE);
    encoder.usize_vec(&fixture.shape)?;
    encoder.usize(fixture.data.len())?;
    for &value in &fixture.data
    {
        encoder.f32(value)?;
    }
    Ok(())
}

fn encode_limits(
    encoder: &mut CanonicalEncoder,
    limits: &ProblemLimits,
) -> Result<(), DigestError> {
    encoder.tag(TAG_LIMITS);
    encoder.usize(limits.max_instructions)?;
    encoder.usize(limits.max_rank)?;
    encoder.usize(limits.max_elements_per_tensor)?;
    encoder.usize(limits.max_total_register_elements)
}

fn encode_evolution(
    encoder: &mut CanonicalEncoder,
    evolution: &EvolutionConfig,
) -> Result<(), DigestError> {
    encoder.tag(TAG_EVOLUTION);
    encode_gen_config(encoder, &evolution.generation)?;
    encoder.usize(evolution.population_size)?;
    encoder.usize(evolution.generations)?;
    encoder.usize(evolution.elitism)?;
    encode_tournament(encoder, &evolution.tournament)?;
    encoder.f32(evolution.scale_magnitude)?;
    encoder.f64(evolution.crossover_probability)?;
    encoder.f64(evolution.mutation_probability)
}

fn encode_gen_config(
    encoder: &mut CanonicalEncoder,
    config: &GenerationConfig,
) -> Result<(), DigestError> {
    encoder.tag(TAG_GEN_CONFIG);
    encoder.usize(config.input_shapes.len())?;
    for shape in &config.input_shapes
    {
        encoder.usize_vec(shape)?;
    }
    encoder.usize(config.min_instructions)?;
    encoder.usize(config.max_instructions)?;
    encode_operators(encoder, &config.operators);
    encoder.f32(config.scale_magnitude)
}

fn encode_operators(encoder: &mut CanonicalEncoder, operators: &OperatorSet) {
    encoder.tag(TAG_OPERATORS);
    encoder.bool(operators.add);
    encoder.bool(operators.matmul);
    encoder.bool(operators.transpose);
    encoder.bool(operators.relu);
    encoder.bool(operators.scale);
}

fn encode_tournament(
    encoder: &mut CanonicalEncoder,
    tournament: &TournamentConfig,
) -> Result<(), DigestError> {
    encoder.tag(TAG_TOURNAMENT);
    encoder.usize(tournament.size)
}

fn encode_success(
    encoder: &mut CanonicalEncoder,
    success: &SuccessCriteria,
) -> Result<(), DigestError> {
    encoder.tag(TAG_SUCCESS);
    encode_option_f64(encoder, success.max_loss)?;
    encode_option_usize(encoder, success.max_active_instructions)?;
    encode_option_u64(encoder, success.max_estimated_flops);
    encode_option_u64(encoder, success.max_peak_live_elements);
    Ok(())
}

fn encode_option_f64(
    encoder: &mut CanonicalEncoder,
    value: Option<f64>,
) -> Result<(), DigestError> {
    match value
    {
        None => encoder.u8(0),
        Some(value) =>
        {
            encoder.u8(1);
            encoder.f64(value)?;
        },
    }
    Ok(())
}

fn encode_option_usize(
    encoder: &mut CanonicalEncoder,
    value: Option<usize>,
) -> Result<(), DigestError> {
    match value
    {
        None => encoder.u8(0),
        Some(value) =>
        {
            encoder.u8(1);
            encoder.usize(value)?;
        },
    }
    Ok(())
}

fn encode_option_u64(encoder: &mut CanonicalEncoder, value: Option<u64>) {
    match value
    {
        None => encoder.u8(0),
        Some(value) =>
        {
            encoder.u8(1);
            encoder.u64(value);
        },
    }
}

fn encode_record(
    encoder: &mut CanonicalEncoder,
    record: &GenerationRecord,
) -> Result<(), DigestError> {
    encoder.tag(TAG_RECORD);
    encoder.usize(record.generation)?;
    encoder.f64(record.best_loss)?;
    encode_cost(encoder, &record.best_cost)?;
    encoder.u128(record.best_fingerprint);
    encoder.usize(record.valid_individuals)?;
    encoder.usize(record.invalid_individuals)?;
    encoder.usize(record.exact_solutions)?;
    encoder.usize(record.pareto_front_size)?;
    encoder.usize(record.diversity)
}

fn encode_cost(encoder: &mut CanonicalEncoder, cost: &CostReport) -> Result<(), DigestError> {
    encoder.tag(TAG_COST);
    encoder.usize(cost.active_instructions)?;
    encoder.u64(cost.estimated_flops);
    encoder.u64(cost.total_active_elements);
    encoder.u64(cost.peak_live_elements);
    encoder.u64(cost.generated_intermediate_bytes);
    encoder.usize(cost.dead_instructions)?;
    encoder.f64(cost.bloat_ratio)
}

fn encode_summary(
    encoder: &mut CanonicalEncoder,
    summary: &PopulationSummary,
) -> Result<(), DigestError> {
    encoder.tag(TAG_SUMMARY);
    encoder.usize(summary.size)?;
    encoder.usize(summary.valid)?;
    encoder.usize(summary.invalid)?;
    encoder.usize(summary.distinct)?;
    encoder.u128(summary.best_fingerprint);
    Ok(())
}

fn encode_hof_entry(
    encoder: &mut CanonicalEncoder,
    entry: &HallOfFameEntry,
) -> Result<(), DigestError> {
    encoder.tag(TAG_HOF);
    encode_program(encoder, &entry.program)?;
    encode_fitness(encoder, &entry.fitness)?;
    encoder.u128(entry.fingerprint);
    encoder.usize(entry.generation)?;
    encoder.u64(entry.seed);
    Ok(())
}

fn encode_fitness(
    encoder: &mut CanonicalEncoder,
    fitness: &FitnessReport,
) -> Result<(), DigestError> {
    encoder.tag(TAG_FITNESS);
    encoder.f64(fitness.loss)?;
    encoder.usize(fitness.failed_cases)?;
    encode_cost(encoder, &fitness.cost)?;
    encoder.bool(fitness.evaluated);
    encoder.u128(fitness.fingerprint);
    Ok(())
}

fn encode_program(
    encoder: &mut CanonicalEncoder,
    program: &TensorProgram,
) -> Result<(), DigestError> {
    encoder.tag(TAG_PROGRAM);
    encoder.usize(program.instructions.len())?;
    for instruction in &program.instructions
    {
        encode_instruction(encoder, instruction)?;
    }
    encoder.usize(program.output)
}

fn encode_instruction(
    encoder: &mut CanonicalEncoder,
    instruction: &TensorInstruction,
) -> Result<(), DigestError> {
    encoder.tag(TAG_INSTRUCTION);
    match *instruction
    {
        TensorInstruction::Input { input } =>
        {
            encoder.u8(0);
            encoder.usize(input)?;
        },
        TensorInstruction::Add { lhs, rhs } =>
        {
            encoder.u8(1);
            encoder.usize(lhs)?;
            encoder.usize(rhs)?;
        },
        TensorInstruction::MatMul { lhs, rhs } =>
        {
            encoder.u8(2);
            encoder.usize(lhs)?;
            encoder.usize(rhs)?;
        },
        TensorInstruction::Transpose2d { src } =>
        {
            encoder.u8(3);
            encoder.usize(src)?;
        },
        TensorInstruction::Relu { src } =>
        {
            encoder.u8(4);
            encoder.usize(src)?;
        },
        TensorInstruction::Scale { src, factor } =>
        {
            encoder.u8(5);
            encoder.usize(src)?;
            encoder.f32(factor)?;
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorInstruction;

    #[test]
    fn f32_distinguishes_signed_zero() {
        let mut positive = CanonicalEncoder::new();
        positive.f32(0.0).unwrap();
        let mut negative = CanonicalEncoder::new();
        negative.f32(-0.0).unwrap();
        assert_ne!(positive.bytes, negative.bytes);
    }

    #[test]
    fn non_finite_floats_are_rejected() {
        let mut encoder = CanonicalEncoder::new();
        assert_eq!(encoder.f32(f32::NAN), Err(DigestError::NonFiniteFloat));
        assert_eq!(encoder.f64(f64::INFINITY), Err(DigestError::NonFiniteFloat));
    }

    /// A fixed program exercising every instruction variant and a signed factor.
    fn known_answer_program() -> TensorProgram {
        TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Transpose2d { src: 0 },
                TensorInstruction::MatMul { lhs: 0, rhs: 1 },
                TensorInstruction::Scale {
                    src: 2,
                    factor: -0.5,
                },
                TensorInstruction::Relu { src: 3 },
                TensorInstruction::Add { lhs: 3, rhs: 4 },
            ],
            5,
        )
    }

    #[test]
    fn pinned_canonical_byte_known_answer() {
        // Pinned SHA-256 of the canonical encoding of a fixed program, prefixed
        // by the digest-format version. If the encoding changes without bumping
        // DIGEST_FORMAT_VERSION, this known-answer test fails.
        let mut encoder = CanonicalEncoder::new();
        encoder.u32(DIGEST_FORMAT_VERSION);
        encode_program(&mut encoder, &known_answer_program()).unwrap();
        assert_eq!(
            encoder.digest_hex(),
            "6225b4a90c4becd095d4913f25fbd05a07eb0fd628bbb9728faeefdb5db1d090"
        );
    }

    #[test]
    fn program_encoding_reacts_to_structure_and_output() {
        let mut base = CanonicalEncoder::new();
        let program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.0,
                },
            ],
            1,
        );
        encode_program(&mut base, &program).unwrap();

        let mut changed_output = CanonicalEncoder::new();
        encode_program(
            &mut changed_output,
            &TensorProgram::new(program.instructions.clone(), 0),
        )
        .unwrap();
        assert_ne!(base.bytes, changed_output.bytes);
    }
}
