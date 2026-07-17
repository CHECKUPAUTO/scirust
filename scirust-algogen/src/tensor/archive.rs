//! Deterministic hall of fame, versioned archive, content digest and replay.
//!
//! An [`ExperimentArchive`] captures everything needed to reproduce and audit a
//! run: the schema version, the crate version, the problem, the seed, the
//! configuration (inside the problem), the generation history, a final
//! population summary, the hall of fame, the best solution and a content digest.
//! The digest is a SHA-256 over a canonical byte serialisation of the
//! deterministic content only; no wall-clock time or timestamp participates in
//! archive equality or the digest. [`replay`] recomputes fitness, costs and the
//! digest, detecting tampering or corruption.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::canonical::program_fingerprint;
use super::experiment::{ExperimentError, GenerationRecord, PopulationSummary};
use super::fitness::{FitnessReport, evaluate_program};
use super::ir::TensorProgram;
use super::population::rank;
use super::problem::TensorProblem;

/// Version of the archive schema.
pub const ARCHIVE_SCHEMA_VERSION: u32 = 1;

/// One retained solution with full provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HallOfFameEntry {
    pub program: TensorProgram,
    pub fitness: FitnessReport,
    pub fingerprint: u128,
    pub generation: usize,
    pub seed: u64,
}

/// A deterministic, capacity-bounded hall of fame.
///
/// Entries are deduplicated by structural fingerprint (the earliest discovery of
/// a program is kept), ordered best-first by the same total order as ranking,
/// and evicted deterministically down to the capacity.
#[derive(Debug, Clone, PartialEq)]
pub struct HallOfFame {
    entries: Vec<HallOfFameEntry>,
    capacity: usize,
}

impl HallOfFame {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
        }
    }

    /// Consider a candidate for inclusion.
    pub fn consider(
        &mut self,
        program: &TensorProgram,
        fitness: FitnessReport,
        generation: usize,
        seed: u64,
    ) {
        if self.capacity == 0
        {
            return;
        }
        // Deduplicate by fingerprint, keeping the earliest discovery.
        if self
            .entries
            .iter()
            .any(|entry| entry.fingerprint == fitness.fingerprint)
        {
            return;
        }

        self.entries.push(HallOfFameEntry {
            program: program.clone(),
            fitness,
            fingerprint: fitness.fingerprint,
            generation,
            seed,
        });
        self.sort_and_truncate();
    }

    fn sort_and_truncate(&mut self) {
        let reports: Vec<FitnessReport> = self.entries.iter().map(|entry| entry.fitness).collect();
        let order = rank(&reports);
        let mut reordered: Vec<HallOfFameEntry> = order
            .iter()
            .map(|&index| self.entries[index].clone())
            .collect();
        reordered.truncate(self.capacity);
        self.entries = reordered;
    }

    pub fn entries(&self) -> &[HallOfFameEntry] {
        &self.entries
    }

    pub fn into_entries(self) -> Vec<HallOfFameEntry> {
        self.entries
    }
}

/// A versioned, reproducible record of a discovery experiment.
///
/// All fields are deterministic; equality and the digest never involve
/// wall-clock time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentArchive {
    pub schema_version: u32,
    pub crate_version: String,
    pub problem: TensorProblem,
    pub seed: u64,
    pub success: bool,
    pub generations_executed: usize,
    pub history: Vec<GenerationRecord>,
    pub final_population: PopulationSummary,
    pub hall_of_fame: Vec<HallOfFameEntry>,
    pub best: HallOfFameEntry,
    /// SHA-256 (hex) over the canonical bytes of every field above.
    pub digest: String,
}

impl ExperimentArchive {
    /// Build an archive, computing its content digest.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        problem: TensorProblem,
        seed: u64,
        success: bool,
        generations_executed: usize,
        history: Vec<GenerationRecord>,
        final_population: PopulationSummary,
        hall_of_fame: Vec<HallOfFameEntry>,
        best: HallOfFameEntry,
    ) -> Result<Self, ExperimentError> {
        let digest = compute_digest(
            ARCHIVE_SCHEMA_VERSION,
            env!("CARGO_PKG_VERSION"),
            &problem,
            seed,
            success,
            generations_executed,
            &history,
            &final_population,
            &hall_of_fame,
            &best,
        )?;

        Ok(Self {
            schema_version: ARCHIVE_SCHEMA_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            problem,
            seed,
            success,
            generations_executed,
            history,
            final_population,
            hall_of_fame,
            best,
            digest,
        })
    }

    /// Recompute the content digest from the stored fields.
    pub fn recompute_digest(&self) -> Result<String, ExperimentError> {
        compute_digest(
            self.schema_version,
            &self.crate_version,
            &self.problem,
            self.seed,
            self.success,
            self.generations_executed,
            &self.history,
            &self.final_population,
            &self.hall_of_fame,
            &self.best,
        )
    }
}

/// SHA-256 over a canonical JSON byte serialisation of the deterministic
/// content. serde_json serialises structs in field order with no maps, so the
/// bytes are canonical; the program-bearing fields additionally fix their own
/// structure through serde. The digest field itself is excluded.
#[allow(clippy::too_many_arguments)]
fn compute_digest(
    schema_version: u32,
    crate_version: &str,
    problem: &TensorProblem,
    seed: u64,
    success: bool,
    generations_executed: usize,
    history: &[GenerationRecord],
    final_population: &PopulationSummary,
    hall_of_fame: &[HallOfFameEntry],
    best: &HallOfFameEntry,
) -> Result<String, ExperimentError> {
    let view = (
        schema_version,
        crate_version,
        problem,
        seed,
        success,
        generations_executed,
        history,
        final_population,
        hall_of_fame,
        best,
    );
    let bytes = serde_json::to_vec(&view)
        .map_err(|error| ExperimentError::Serialization(error.to_string()))?;

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(to_hex(&hasher.finalize()))
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

/// A single replay discrepancy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplayMismatch {
    /// The recomputed content digest differs from the stored digest.
    Digest,
    /// A stored program's fingerprint does not match its structure.
    Program { fingerprint: u128 },
    /// A stored fitness does not match re-evaluation.
    Fitness { fingerprint: u128 },
}

/// The result of replaying an archive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub digest_ok: bool,
    pub recomputed_digest: String,
    pub entries_checked: usize,
    pub mismatches: Vec<ReplayMismatch>,
}

impl ReplayReport {
    /// Whether the archive is intact: digest matches and nothing mismatched.
    pub fn is_intact(&self) -> bool {
        self.digest_ok && self.mismatches.is_empty()
    }
}

/// Replay an archive: recompute the digest and re-evaluate every stored program,
/// reporting any tampering or corruption.
pub fn replay(archive: &ExperimentArchive) -> Result<ReplayReport, ExperimentError> {
    let recomputed_digest = archive.recompute_digest()?;
    let digest_ok = recomputed_digest == archive.digest;

    let mut mismatches = Vec::new();
    if !digest_ok
    {
        mismatches.push(ReplayMismatch::Digest);
    }

    let dataset = archive.problem.dataset()?;
    let limits = archive.problem.verification_limits();

    let mut entries_checked = 0usize;
    for entry in archive
        .hall_of_fame
        .iter()
        .chain(std::iter::once(&archive.best))
    {
        entries_checked += 1;

        // A changed program has a different fingerprint.
        if program_fingerprint(&entry.program) != entry.fingerprint
        {
            mismatches.push(ReplayMismatch::Program {
                fingerprint: entry.fingerprint,
            });
            continue;
        }

        // Re-evaluation must reproduce the stored fitness exactly.
        let recomputed = evaluate_program(&entry.program, &dataset, limits);
        if recomputed != entry.fitness
        {
            mismatches.push(ReplayMismatch::Fitness {
                fingerprint: entry.fingerprint,
            });
        }
    }

    Ok(ReplayReport {
        digest_ok,
        recomputed_digest,
        entries_checked,
        mismatches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::benchmarks;
    use crate::tensor::experiment::{RunOptions, run_experiment};
    use crate::tensor::{TensorInstruction, TensorProgram};

    fn archive() -> ExperimentArchive {
        run_experiment(&benchmarks::relu(), RunOptions::default()).unwrap()
    }

    #[test]
    fn archive_round_trips_through_serde() {
        let archive = archive();
        let json = serde_json::to_string(&archive).unwrap();
        let decoded: ExperimentArchive = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, archive);
    }

    #[test]
    fn digest_is_stable() {
        let archive = archive();
        assert_eq!(archive.digest, archive.recompute_digest().unwrap());
    }

    #[test]
    fn changing_one_instruction_changes_the_digest() {
        let mut archive = archive();
        let original = archive.recompute_digest().unwrap();
        // Alter a single instruction of the best program.
        archive
            .best
            .program
            .instructions
            .push(TensorInstruction::Relu { src: 0 });
        assert_ne!(original, archive.recompute_digest().unwrap());
    }

    #[test]
    fn replay_accepts_an_intact_archive() {
        let archive = archive();
        let report = replay(&archive).unwrap();
        assert!(report.is_intact(), "mismatches: {:?}", report.mismatches);
        assert!(report.entries_checked >= 1);
    }

    #[test]
    fn replay_detects_altered_fitness() {
        let mut archive = archive();
        archive.best.fitness.loss += 1.0;
        let report = replay(&archive).unwrap();
        assert!(!report.is_intact());
        assert!(report.mismatches.contains(&ReplayMismatch::Digest));
        assert!(
            report
                .mismatches
                .iter()
                .any(|m| matches!(m, ReplayMismatch::Fitness { .. }))
        );
    }

    #[test]
    fn replay_detects_altered_program() {
        let mut archive = archive();
        archive.best.program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 123.0,
                },
            ],
            1,
        );
        let report = replay(&archive).unwrap();
        assert!(!report.is_intact());
        assert!(
            report
                .mismatches
                .iter()
                .any(|m| matches!(m, ReplayMismatch::Program { .. }))
        );
    }

    #[test]
    fn hall_of_fame_deduplicates_and_evicts_deterministically() {
        // Distinct programs with distinct fitness; capacity forces eviction.
        let dataset = benchmarks::relu().dataset().unwrap();
        let limits = benchmarks::relu().verification_limits();

        let programs = [
            TensorProgram::new(vec![TensorInstruction::Input { input: 0 }], 0),
            TensorProgram::new(
                vec![
                    TensorInstruction::Input { input: 0 },
                    TensorInstruction::Relu { src: 0 },
                ],
                1,
            ),
            TensorProgram::new(
                vec![
                    TensorInstruction::Input { input: 0 },
                    TensorInstruction::Scale {
                        src: 0,
                        factor: 2.0,
                    },
                ],
                1,
            ),
        ];

        let mut hall = HallOfFame::new(2);
        for (generation, program) in programs.iter().enumerate()
        {
            let fitness = evaluate_program(program, &dataset, limits);
            hall.consider(program, fitness, generation, 0);
            // Re-considering the same program is a no-op (dedup).
            hall.consider(program, fitness, generation + 100, 0);
        }

        assert_eq!(hall.entries().len(), 2);
        // The best (lowest loss) is the Relu program; it must be retained.
        let relu_fp = program_fingerprint(&programs[1]);
        assert_eq!(hall.entries()[0].fingerprint, relu_fp);

        // Deterministic: repeating the exact sequence yields the same hall.
        let mut again = HallOfFame::new(2);
        for (generation, program) in programs.iter().enumerate()
        {
            let fitness = evaluate_program(program, &dataset, limits);
            again.consider(program, fitness, generation, 0);
        }
        assert_eq!(hall, again);
    }
}
