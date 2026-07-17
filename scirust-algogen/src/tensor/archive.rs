//! Deterministic hall of fame, versioned archive, content-integrity digest and
//! archive verification.
//!
//! An [`ExperimentArchive`] captures everything needed to reproduce and audit a
//! run's stored solutions: schema and digest-format versions, the crate version,
//! the problem, the seed, the configuration (inside the problem), the generation
//! history, a final-population summary, the hall of fame, the best solution and
//! a content digest.
//!
//! # Integrity, not authenticity
//!
//! The digest is an **unkeyed** SHA-256 over an explicit canonical binary
//! encoding (see [`super::digest`]). It provides **content integrity /
//! corruption detection**: it detects accidental corruption, truncation or stale
//! content, and any edit to the archived content. It does **not** authenticate an
//! archive against a party able to edit both the archive and its digest — that
//! would require a trusted signature or MAC, which this phase deliberately does
//! not provide. No wall-clock time or timestamp participates in archive equality
//! or the digest.
//!
//! [`verify_archive`] re-derives the digest and re-evaluates every stored
//! program, and additionally checks a set of structural invariants. It verifies
//! the **stored solutions and summary invariants**, not the complete hidden
//! evolutionary trajectory (the archive does not store every generation's full
//! population), so it is honestly named "archive verification" rather than a full
//! replay.

use std::iter::once;

use serde::{Deserialize, Serialize};

use super::canonical::{canonical_equal, program_fingerprint};
use super::digest::{DIGEST_FORMAT_VERSION, archive_content_digest};
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
/// Entries are deduplicated by **canonical program bytes** (the authoritative
/// identity — not by fingerprint, which could collide), keeping the earliest
/// discovery of a program; they are ordered best-first by the same total order
/// as ranking and evicted deterministically down to the capacity.
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

    pub fn capacity(&self) -> usize {
        self.capacity
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
        // Deduplicate by authoritative canonical identity, not by fingerprint:
        // two structurally distinct programs are never collapsed even if their
        // fingerprints collide.
        if self
            .entries
            .iter()
            .any(|entry| canonical_equal(&entry.program, program))
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
        let programs: Vec<TensorProgram> = self
            .entries
            .iter()
            .map(|entry| entry.program.clone())
            .collect();
        let reports: Vec<FitnessReport> = self.entries.iter().map(|entry| entry.fitness).collect();
        let order = rank(&programs, &reports);
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
    pub digest_format_version: u32,
    pub crate_version: String,
    pub problem: TensorProblem,
    pub seed: u64,
    pub success: bool,
    pub generations_executed: usize,
    pub history: Vec<GenerationRecord>,
    pub final_population: PopulationSummary,
    pub hall_of_fame_capacity: usize,
    pub hall_of_fame: Vec<HallOfFameEntry>,
    pub best: HallOfFameEntry,
    /// SHA-256 (hex) over the canonical bytes of every field above (excluding
    /// this digest). An unkeyed content-integrity digest, not an authenticator.
    pub digest: String,
}

impl ExperimentArchive {
    /// Build an archive, computing its content-integrity digest.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        problem: TensorProblem,
        seed: u64,
        success: bool,
        generations_executed: usize,
        history: Vec<GenerationRecord>,
        final_population: PopulationSummary,
        hall_of_fame_capacity: usize,
        hall_of_fame: Vec<HallOfFameEntry>,
        best: HallOfFameEntry,
    ) -> Result<Self, ExperimentError> {
        let digest = archive_content_digest(
            ARCHIVE_SCHEMA_VERSION,
            env!("CARGO_PKG_VERSION"),
            &problem,
            seed,
            success,
            generations_executed,
            &history,
            &final_population,
            hall_of_fame_capacity,
            &hall_of_fame,
            &best,
        )?;

        Ok(Self {
            schema_version: ARCHIVE_SCHEMA_VERSION,
            digest_format_version: DIGEST_FORMAT_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            problem,
            seed,
            success,
            generations_executed,
            history,
            final_population,
            hall_of_fame_capacity,
            hall_of_fame,
            best,
            digest,
        })
    }

    /// Recompute the content digest from the stored fields.
    pub fn recompute_digest(&self) -> Result<String, ExperimentError> {
        archive_content_digest(
            self.schema_version,
            &self.crate_version,
            &self.problem,
            self.seed,
            self.success,
            self.generations_executed,
            &self.history,
            &self.final_population,
            self.hall_of_fame_capacity,
            &self.hall_of_fame,
            &self.best,
        )
        .map_err(ExperimentError::Digest)
    }
}

/// A structured archive-verification issue. One variant per invariant so callers
/// can react precisely rather than to a single generic mismatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationIssue {
    UnsupportedSchemaVersion {
        found: u32,
        supported: u32,
    },
    UnsupportedDigestVersion {
        found: u32,
        supported: u32,
    },
    DigestComputationFailed,
    DigestMismatch,
    ProblemInvalid,
    SeedMismatch {
        archive_seed: u64,
        problem_seed: u64,
    },
    GenerationsHistoryLengthMismatch {
        generations_executed: usize,
        history_len: usize,
    },
    NonMonotonicHistory {
        position: usize,
        expected: usize,
        found: usize,
    },
    PopulationSummaryInconsistent,
    FinalSummaryMismatchesLastRecord,
    HallOfFameCapacityExceeded {
        capacity: usize,
        length: usize,
    },
    HallOfFameProgramFingerprintMismatch {
        index: usize,
    },
    HallOfFameDuplicate {
        first: usize,
        second: usize,
    },
    HallOfFameOrderInvalid,
    BestNotInHallOfFame,
    BestProgramFingerprintMismatch,
    SuccessFlagInconsistent {
        success: bool,
        best_meets_criteria: bool,
    },
    ProgramFitnessMismatch {
        fingerprint: u128,
    },
}

/// The result of verifying an archive.
///
/// Verifies stored solutions and summary invariants, not the complete hidden
/// evolutionary trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveVerification {
    pub digest_ok: bool,
    pub recomputed_digest: Option<String>,
    pub programs_checked: usize,
    pub issues: Vec<VerificationIssue>,
}

impl ArchiveVerification {
    /// Whether every checked invariant holds (including the digest).
    pub fn is_intact(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Verify an archive: recompute the digest, re-evaluate every stored program and
/// check structural invariants, returning a structured issue per failing check.
pub fn verify_archive(archive: &ExperimentArchive) -> ArchiveVerification {
    let mut issues = Vec::new();

    // 1. Supported schema version.
    if archive.schema_version != ARCHIVE_SCHEMA_VERSION
    {
        issues.push(VerificationIssue::UnsupportedSchemaVersion {
            found: archive.schema_version,
            supported: ARCHIVE_SCHEMA_VERSION,
        });
    }
    // 2. Supported digest-format version.
    if archive.digest_format_version != DIGEST_FORMAT_VERSION
    {
        issues.push(VerificationIssue::UnsupportedDigestVersion {
            found: archive.digest_format_version,
            supported: DIGEST_FORMAT_VERSION,
        });
    }

    // Digest recomputation and comparison.
    let recomputed_digest = archive.recompute_digest().ok();
    let digest_ok = match &recomputed_digest
    {
        Some(digest) => *digest == archive.digest,
        None => false,
    };
    match &recomputed_digest
    {
        None => issues.push(VerificationIssue::DigestComputationFailed),
        Some(_) if !digest_ok => issues.push(VerificationIssue::DigestMismatch),
        Some(_) =>
        {},
    }

    // 3. Problem validation.
    let problem_ok = archive.problem.validate().is_ok();
    if !problem_ok
    {
        issues.push(VerificationIssue::ProblemInvalid);
    }

    // 4. Seed consistency.
    if archive.seed != archive.problem.seed
    {
        issues.push(VerificationIssue::SeedMismatch {
            archive_seed: archive.seed,
            problem_seed: archive.problem.seed,
        });
    }

    // 5. generations_executed against history length.
    if archive.history.len() != archive.generations_executed + 1
    {
        issues.push(VerificationIssue::GenerationsHistoryLengthMismatch {
            generations_executed: archive.generations_executed,
            history_len: archive.history.len(),
        });
    }

    // 6. Monotonic, gap-free generation indices.
    for (position, record) in archive.history.iter().enumerate()
    {
        if record.generation != position
        {
            issues.push(VerificationIssue::NonMonotonicHistory {
                position,
                expected: position,
                found: record.generation,
            });
            break;
        }
    }

    // 7. Final population summary internal consistency.
    let summary = &archive.final_population;
    if summary.valid + summary.invalid != summary.size || summary.distinct > summary.size
    {
        issues.push(VerificationIssue::PopulationSummaryInconsistent);
    }
    // 14. Final summary against the last recorded generation.
    if let Some(last) = archive.history.last()
    {
        if summary.valid != last.valid_individuals
            || summary.invalid != last.invalid_individuals
            || summary.distinct != last.diversity
            || summary.best_fingerprint != last.best_fingerprint
        {
            issues.push(VerificationIssue::FinalSummaryMismatchesLastRecord);
        }
    }

    // 8. Hall-of-fame entry fingerprints against full program structure.
    for (index, entry) in archive.hall_of_fame.iter().enumerate()
    {
        if program_fingerprint(&entry.program) != entry.fingerprint
        {
            issues.push(VerificationIssue::HallOfFameProgramFingerprintMismatch { index });
        }
    }

    // 9. No accidental duplicate programs in the hall of fame.
    for first in 0..archive.hall_of_fame.len()
    {
        for second in (first + 1)..archive.hall_of_fame.len()
        {
            if canonical_equal(
                &archive.hall_of_fame[first].program,
                &archive.hall_of_fame[second].program,
            )
            {
                issues.push(VerificationIssue::HallOfFameDuplicate { first, second });
            }
        }
    }

    // 10. Ordering and capacity policy.
    if archive.hall_of_fame.len() > archive.hall_of_fame_capacity
    {
        issues.push(VerificationIssue::HallOfFameCapacityExceeded {
            capacity: archive.hall_of_fame_capacity,
            length: archive.hall_of_fame.len(),
        });
    }
    {
        let programs: Vec<TensorProgram> = archive
            .hall_of_fame
            .iter()
            .map(|entry| entry.program.clone())
            .collect();
        let reports: Vec<FitnessReport> = archive
            .hall_of_fame
            .iter()
            .map(|entry| entry.fitness)
            .collect();
        let order = rank(&programs, &reports);
        if order != (0..archive.hall_of_fame.len()).collect::<Vec<_>>()
        {
            issues.push(VerificationIssue::HallOfFameOrderInvalid);
        }
    }

    // 11. Best entry present in the hall of fame (when it retains entries).
    if archive.hall_of_fame_capacity >= 1
        && !archive.hall_of_fame.is_empty()
        && !archive
            .hall_of_fame
            .iter()
            .any(|entry| canonical_equal(&entry.program, &archive.best.program))
    {
        issues.push(VerificationIssue::BestNotInHallOfFame);
    }
    if program_fingerprint(&archive.best.program) != archive.best.fingerprint
    {
        issues.push(VerificationIssue::BestProgramFingerprintMismatch);
    }

    // 12. Success flag against the explicit success criteria.
    if problem_ok
    {
        let best_meets = archive.problem.success.is_met(&archive.best.fitness);
        if best_meets != archive.success
        {
            issues.push(VerificationIssue::SuccessFlagInconsistent {
                success: archive.success,
                best_meets_criteria: best_meets,
            });
        }
    }

    // 13. Recompute fitness and structural costs for every archived program.
    let mut programs_checked = 0usize;
    if problem_ok
    {
        if let Ok(dataset) = archive.problem.dataset()
        {
            let limits = archive.problem.verification_limits();
            for entry in archive.hall_of_fame.iter().chain(once(&archive.best))
            {
                programs_checked += 1;
                let recomputed = evaluate_program(&entry.program, &dataset, limits);
                if recomputed != entry.fitness
                {
                    issues.push(VerificationIssue::ProgramFitnessMismatch {
                        fingerprint: entry.fingerprint,
                    });
                }
            }
        }
    }

    ArchiveVerification {
        digest_ok,
        recomputed_digest,
        programs_checked,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::benchmarks;
    use crate::tensor::experiment::{RunOptions, run_experiment};
    use crate::tensor::{TensorInstruction, TensorProgram, evaluate_program};

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
    fn digest_is_stable_and_survives_pretty_print_round_trip() {
        let archive = archive();
        assert_eq!(archive.digest, archive.recompute_digest().unwrap());

        // Pretty vs compact JSON must not change the content digest.
        let pretty = serde_json::to_string_pretty(&archive).unwrap();
        let from_pretty: ExperimentArchive = serde_json::from_str(&pretty).unwrap();
        assert_eq!(from_pretty.recompute_digest().unwrap(), archive.digest);

        let compact = serde_json::to_string(&archive).unwrap();
        let from_compact: ExperimentArchive = serde_json::from_str(&compact).unwrap();
        assert_eq!(from_compact.digest, archive.digest);
    }

    #[test]
    fn changing_relevant_content_changes_the_digest() {
        let archive = archive();
        let original = archive.recompute_digest().unwrap();

        // A scalar in the best fitness.
        let mut altered = archive.clone();
        altered.best.fitness.loss += 1.0;
        assert_ne!(original, altered.recompute_digest().unwrap());

        // A program output register (isolated on a fixed two-instruction
        // program so exactly the output field differs between the two digests).
        let mut altered = archive.clone();
        altered.best.program = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Relu { src: 0 },
            ],
            0,
        );
        let output_zero = altered.recompute_digest().unwrap();
        altered.best.program.output = 1;
        let output_one = altered.recompute_digest().unwrap();
        assert_ne!(output_zero, output_one);

        // A single tensor element in the problem.
        let mut altered = archive.clone();
        altered.problem.cases[0].expected.data[0] += 1.0;
        assert_ne!(original, altered.recompute_digest().unwrap());

        // The history.
        let mut altered = archive.clone();
        altered.history[0].diversity += 1;
        assert_ne!(original, altered.recompute_digest().unwrap());

        // One instruction in the best program.
        let mut altered = archive.clone();
        altered
            .best
            .program
            .instructions
            .push(TensorInstruction::Relu { src: 0 });
        assert_ne!(original, altered.recompute_digest().unwrap());
    }

    #[test]
    fn hall_of_fame_order_change_changes_digest_when_meaningful() {
        let mut archive = archive();
        if archive.hall_of_fame.len() >= 2
        {
            let original = archive.recompute_digest().unwrap();
            archive.hall_of_fame.swap(0, 1);
            assert_ne!(original, archive.recompute_digest().unwrap());
        }
    }

    #[test]
    fn verify_accepts_an_intact_archive() {
        let archive = archive();
        let report = verify_archive(&archive);
        assert!(report.is_intact(), "issues: {:?}", report.issues);
        assert!(report.digest_ok);
        assert!(report.programs_checked >= 1);
    }

    #[test]
    fn verify_detects_altered_fitness() {
        let mut archive = archive();
        archive.best.fitness.loss += 1.0;
        let report = verify_archive(&archive);
        assert!(!report.is_intact());
        assert!(report.issues.contains(&VerificationIssue::DigestMismatch));
        assert!(
            report
                .issues
                .iter()
                .any(|issue| matches!(issue, VerificationIssue::ProgramFitnessMismatch { .. }))
        );
    }

    #[test]
    fn verify_detects_altered_program() {
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
        let report = verify_archive(&archive);
        assert!(!report.is_intact());
        assert!(
            report
                .issues
                .contains(&VerificationIssue::BestProgramFingerprintMismatch)
        );
    }

    #[test]
    fn verify_detects_history_gap() {
        let mut archive = archive();
        if let Some(record) = archive.history.get_mut(0)
        {
            record.generation = 5;
        }
        let report = verify_archive(&archive);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| matches!(issue, VerificationIssue::NonMonotonicHistory { .. }))
        );
    }

    #[test]
    fn hall_of_fame_deduplicates_by_canonical_bytes_and_evicts() {
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
            // Re-considering the same program is a no-op.
            hall.consider(program, fitness, generation + 100, 0);
        }

        assert_eq!(hall.entries().len(), 2);
        let relu_fp = program_fingerprint(&programs[1]);
        assert_eq!(hall.entries()[0].fingerprint, relu_fp);

        // Deterministic across repetition.
        let mut again = HallOfFame::new(2);
        for (generation, program) in programs.iter().enumerate()
        {
            let fitness = evaluate_program(program, &dataset, limits);
            again.consider(program, fitness, generation, 0);
        }
        assert_eq!(hall, again);
    }

    #[test]
    fn distinct_programs_with_equal_fingerprints_are_not_deduplicated() {
        // Inject two distinct programs whose FitnessReport fingerprint fields are
        // forced equal; canonical-bytes dedup must keep both.
        let dataset = benchmarks::relu().dataset().unwrap();
        let limits = benchmarks::relu().verification_limits();

        let a = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 1.0,
                },
            ],
            1,
        );
        let b = TensorProgram::new(
            vec![
                TensorInstruction::Input { input: 0 },
                TensorInstruction::Scale {
                    src: 0,
                    factor: 2.0,
                },
            ],
            1,
        );
        let mut fitness_a = evaluate_program(&a, &dataset, limits);
        let mut fitness_b = evaluate_program(&b, &dataset, limits);
        // Force a fingerprint collision at the report level.
        fitness_a.fingerprint = 0xC0FFEE;
        fitness_b.fingerprint = 0xC0FFEE;

        let mut hall = HallOfFame::new(8);
        hall.consider(&a, fitness_a, 0, 0);
        hall.consider(&b, fitness_b, 1, 0);
        assert_eq!(
            hall.entries().len(),
            2,
            "distinct programs must both survive"
        );
    }
}
