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
//! [`verify_archive`] re-derives the digest, re-evaluates stored programs and
//! checks structural invariants; it does not regenerate hidden populations.
//! [`replay_experiment`] is the separate, stronger operation that reruns every
//! generation from the archived deterministic problem, configuration and seed.

use std::iter::once;

use serde::{Deserialize, Serialize};

use super::canonical::{canonical_equal, program_fingerprint};
use super::digest::{
    DIGEST_FORMAT_VERSION, DigestError, archive_canonical_bytes, archive_content_digest,
};
use super::experiment::{
    ExperimentError, GenerationRecord, PopulationSummary, RunOptions, run_experiment,
};
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

        let fingerprint = program_fingerprint(program);
        let mut fitness = fitness;
        fitness.fingerprint = fingerprint;
        self.entries.push(HallOfFameEntry {
            program: program.clone(),
            fitness,
            fingerprint,
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
            DIGEST_FORMAT_VERSION,
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
            self.digest_format_version,
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

    /// Produce the explicit canonical archive bytes covered by [`Self::digest`].
    ///
    /// JSON formatting never participates. The digest field itself is excluded.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ExperimentError> {
        archive_canonical_bytes(
            self.digest_format_version,
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

/// Location of a stored program referenced by a verification issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchiveEntryLocation {
    HallOfFame(usize),
    Best,
}

/// A structured archive-verification issue. Each invariant has a distinct
/// variant so callers never have to infer a semantic failure from a generic
/// digest mismatch.
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
    CanonicalEncodingInvalid {
        error: DigestError,
    },
    DigestMismatch,
    ProblemInvalid,
    SeedMismatch {
        archive_seed: u64,
        problem_seed: u64,
    },
    GenerationCountInconsistent {
        executed: usize,
        budget: usize,
        success: bool,
    },
    HistoryLengthMismatch {
        expected: Option<usize>,
        found: usize,
    },
    HistoryDoesNotStartAtZero {
        found: usize,
    },
    HistoryGenerationGap {
        position: usize,
        expected: usize,
        found: usize,
    },
    FinalHistoryGenerationMismatch {
        expected: usize,
        found: Option<usize>,
    },
    PopulationSizeMismatch {
        archived: usize,
        configured: usize,
    },
    PopulationCountsMismatch {
        size: usize,
        valid: usize,
        invalid: usize,
    },
    PopulationDistinctExceedsSize {
        size: usize,
        distinct: usize,
    },
    GenerationCountsMismatch {
        generation: usize,
        population_size: usize,
        valid: usize,
        invalid: usize,
    },
    ExactSolutionsExceedValid {
        generation: usize,
        exact_solutions: usize,
        valid: usize,
    },
    GenerationSummaryCountOutOfRange {
        generation: usize,
        field: String,
        value: usize,
        population_size: usize,
    },
    FinalSummaryMismatchesLastRecord,
    HallOfFameCapacityExceeded {
        capacity: usize,
        length: usize,
    },
    HallOfFameMustBeEmptyAtZeroCapacity {
        length: usize,
    },
    HallOfFameMissingAtPositiveCapacity {
        capacity: usize,
    },
    ProgramFingerprintMismatch {
        location: ArchiveEntryLocation,
        stored: u128,
        recomputed: u128,
    },
    FitnessFingerprintMismatch {
        location: ArchiveEntryLocation,
        stored: u128,
        recomputed: u128,
    },
    EntrySeedMismatch {
        location: ArchiveEntryLocation,
        stored: u64,
        archive_seed: u64,
    },
    EntryGenerationOutOfRange {
        location: ArchiveEntryLocation,
        generation: usize,
        generations_executed: usize,
    },
    HallOfFameDuplicate {
        first: usize,
        second: usize,
    },
    HallOfFameFingerprintCollision {
        first: usize,
        second: usize,
        fingerprint: u128,
    },
    HallOfFameOrderInvalid,
    BestDoesNotMatchHallOfFameLeader,
    StoredFitnessMismatch {
        location: ArchiveEntryLocation,
    },
    StoredStructuralCostMismatch {
        location: ArchiveEntryLocation,
    },
    SuccessFlagInconsistent {
        success: bool,
        best_meets_criteria: bool,
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

    // 3-4. Canonical encoding validity and digest match.
    let recomputed_digest = match archive.recompute_digest()
    {
        Ok(digest) => Some(digest),
        Err(ExperimentError::Digest(error)) =>
        {
            issues.push(VerificationIssue::CanonicalEncodingInvalid { error });
            None
        },
        Err(_) => unreachable!("digest recomputation only returns digest errors"),
    };
    let digest_ok = match &recomputed_digest
    {
        Some(digest) => *digest == archive.digest,
        None => false,
    };
    match &recomputed_digest
    {
        None =>
        {},
        Some(_) if !digest_ok => issues.push(VerificationIssue::DigestMismatch),
        Some(_) =>
        {},
    }

    // 5. Problem validation.
    let problem_ok = archive.problem.validate().is_ok();
    if !problem_ok
    {
        issues.push(VerificationIssue::ProblemInvalid);
    }

    // 6. Seed consistency.
    if archive.seed != archive.problem.seed
    {
        issues.push(VerificationIssue::SeedMismatch {
            archive_seed: archive.seed,
            problem_seed: archive.problem.seed,
        });
    }

    // 7. Executed generations must be compatible with the budget and early
    // stopping policy.
    let budget = archive.problem.evolution.generations;
    if archive.generations_executed > budget
        || (!archive.success && archive.generations_executed != budget)
    {
        issues.push(VerificationIssue::GenerationCountInconsistent {
            executed: archive.generations_executed,
            budget,
            success: archive.success,
        });
    }

    // 8-11. History length and generation indices.
    let expected_history_len = archive.generations_executed.checked_add(1);
    if expected_history_len != Some(archive.history.len())
    {
        issues.push(VerificationIssue::HistoryLengthMismatch {
            expected: expected_history_len,
            found: archive.history.len(),
        });
    }
    if let Some(first) = archive.history.first()
    {
        if first.generation != 0
        {
            issues.push(VerificationIssue::HistoryDoesNotStartAtZero {
                found: first.generation,
            });
        }
    }
    for (position, record) in archive.history.iter().enumerate()
    {
        if record.generation != position
        {
            issues.push(VerificationIssue::HistoryGenerationGap {
                position,
                expected: position,
                found: record.generation,
            });
        }
    }
    let final_generation = archive.history.last().map(|record| record.generation);
    if final_generation != Some(archive.generations_executed)
    {
        issues.push(VerificationIssue::FinalHistoryGenerationMismatch {
            expected: archive.generations_executed,
            found: final_generation,
        });
    }

    // 12-14. Population and per-generation summary counts.
    let summary = &archive.final_population;
    let configured_size = archive.problem.evolution.population_size;
    if summary.size != configured_size
    {
        issues.push(VerificationIssue::PopulationSizeMismatch {
            archived: summary.size,
            configured: configured_size,
        });
    }
    if summary.valid.checked_add(summary.invalid) != Some(summary.size)
    {
        issues.push(VerificationIssue::PopulationCountsMismatch {
            size: summary.size,
            valid: summary.valid,
            invalid: summary.invalid,
        });
    }
    if summary.distinct > summary.size
    {
        issues.push(VerificationIssue::PopulationDistinctExceedsSize {
            size: summary.size,
            distinct: summary.distinct,
        });
    }
    for record in &archive.history
    {
        if record
            .valid_individuals
            .checked_add(record.invalid_individuals)
            != Some(configured_size)
        {
            issues.push(VerificationIssue::GenerationCountsMismatch {
                generation: record.generation,
                population_size: configured_size,
                valid: record.valid_individuals,
                invalid: record.invalid_individuals,
            });
        }
        if record.exact_solutions > record.valid_individuals
        {
            issues.push(VerificationIssue::ExactSolutionsExceedValid {
                generation: record.generation,
                exact_solutions: record.exact_solutions,
                valid: record.valid_individuals,
            });
        }
        for (field, value) in [
            ("pareto_front_size", record.pareto_front_size),
            ("diversity", record.diversity),
        ]
        {
            if value > configured_size
            {
                issues.push(VerificationIssue::GenerationSummaryCountOutOfRange {
                    generation: record.generation,
                    field: field.to_string(),
                    value,
                    population_size: configured_size,
                });
            }
        }
    }

    // 24. Final summary against the last recorded generation.
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

    // 15-18. Hall-of-fame identity, fingerprint, order and capacity.
    for (index, entry) in archive.hall_of_fame.iter().enumerate()
    {
        verify_entry(
            archive,
            ArchiveEntryLocation::HallOfFame(index),
            entry,
            &mut issues,
        );
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
            else if archive.hall_of_fame[first].fingerprint
                == archive.hall_of_fame[second].fingerprint
            {
                issues.push(VerificationIssue::HallOfFameFingerprintCollision {
                    first,
                    second,
                    fingerprint: archive.hall_of_fame[first].fingerprint,
                });
            }
        }
    }

    if archive.hall_of_fame.len() > archive.hall_of_fame_capacity
    {
        issues.push(VerificationIssue::HallOfFameCapacityExceeded {
            capacity: archive.hall_of_fame_capacity,
            length: archive.hall_of_fame.len(),
        });
    }
    if archive.hall_of_fame_capacity == 0 && !archive.hall_of_fame.is_empty()
    {
        issues.push(VerificationIssue::HallOfFameMustBeEmptyAtZeroCapacity {
            length: archive.hall_of_fame.len(),
        });
    }
    if archive.hall_of_fame_capacity > 0 && archive.hall_of_fame.is_empty()
    {
        issues.push(VerificationIssue::HallOfFameMissingAtPositiveCapacity {
            capacity: archive.hall_of_fame_capacity,
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

    // 19-22. Best provenance/policy and all stored solution data.
    verify_entry(
        archive,
        ArchiveEntryLocation::Best,
        &archive.best,
        &mut issues,
    );
    if archive.hall_of_fame_capacity > 0
        && !archive
            .hall_of_fame
            .first()
            .is_some_and(|leader| entries_equal(leader, &archive.best))
    {
        issues.push(VerificationIssue::BestDoesNotMatchHallOfFameLeader);
    }

    // 23. Success flag against the explicit success criteria.
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

    let mut programs_checked = 0usize;
    if problem_ok
    {
        if let Ok(dataset) = archive.problem.dataset()
        {
            let limits = archive.problem.verification_limits();
            for (location, entry) in archive
                .hall_of_fame
                .iter()
                .enumerate()
                .map(|(index, entry)| (ArchiveEntryLocation::HallOfFame(index), entry))
                .chain(once((ArchiveEntryLocation::Best, &archive.best)))
            {
                programs_checked += 1;
                let recomputed = evaluate_program(&entry.program, &dataset, limits);
                if !fitness_values_equal(&recomputed, &entry.fitness)
                {
                    issues.push(VerificationIssue::StoredFitnessMismatch { location });
                }
                if !costs_equal(&recomputed.cost, &entry.fitness.cost)
                {
                    issues.push(VerificationIssue::StoredStructuralCostMismatch { location });
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

fn verify_entry(
    archive: &ExperimentArchive,
    location: ArchiveEntryLocation,
    entry: &HallOfFameEntry,
    issues: &mut Vec<VerificationIssue>,
) {
    let recomputed = program_fingerprint(&entry.program);
    if entry.fingerprint != recomputed
    {
        issues.push(VerificationIssue::ProgramFingerprintMismatch {
            location,
            stored: entry.fingerprint,
            recomputed,
        });
    }
    if entry.fitness.fingerprint != recomputed
    {
        issues.push(VerificationIssue::FitnessFingerprintMismatch {
            location,
            stored: entry.fitness.fingerprint,
            recomputed,
        });
    }
    if entry.seed != archive.seed
    {
        issues.push(VerificationIssue::EntrySeedMismatch {
            location,
            stored: entry.seed,
            archive_seed: archive.seed,
        });
    }
    if entry.generation > archive.generations_executed
    {
        issues.push(VerificationIssue::EntryGenerationOutOfRange {
            location,
            generation: entry.generation,
            generations_executed: archive.generations_executed,
        });
    }
}

fn fitness_values_equal(left: &FitnessReport, right: &FitnessReport) -> bool {
    left.loss.to_bits() == right.loss.to_bits()
        && left.failed_cases == right.failed_cases
        && left.evaluated == right.evaluated
}

fn fitness_equal(left: &FitnessReport, right: &FitnessReport) -> bool {
    fitness_values_equal(left, right)
        && costs_equal(&left.cost, &right.cost)
        && left.fingerprint == right.fingerprint
}

fn entries_equal(left: &HallOfFameEntry, right: &HallOfFameEntry) -> bool {
    canonical_equal(&left.program, &right.program)
        && fitness_equal(&left.fitness, &right.fitness)
        && left.fingerprint == right.fingerprint
        && left.generation == right.generation
        && left.seed == right.seed
}

fn records_equal(left: &GenerationRecord, right: &GenerationRecord) -> bool {
    left.generation == right.generation
        && left.best_loss.to_bits() == right.best_loss.to_bits()
        && costs_equal(&left.best_cost, &right.best_cost)
        && left.best_fingerprint == right.best_fingerprint
        && left.valid_individuals == right.valid_individuals
        && left.invalid_individuals == right.invalid_individuals
        && left.exact_solutions == right.exact_solutions
        && left.pareto_front_size == right.pareto_front_size
        && left.diversity == right.diversity
}

fn costs_equal(left: &super::cost::CostReport, right: &super::cost::CostReport) -> bool {
    left.active_instructions == right.active_instructions
        && left.estimated_flops == right.estimated_flops
        && left.total_active_elements == right.total_active_elements
        && left.peak_live_elements == right.peak_live_elements
        && left.generated_intermediate_bytes == right.generated_intermediate_bytes
        && left.dead_instructions == right.dead_instructions
        && left.bloat_ratio.to_bits() == right.bloat_ratio.to_bits()
}

/// A mismatch found by rerunning the complete deterministic experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentReplayMismatch {
    CanonicalContent,
    GenerationsExecuted,
    History,
    BestProgram,
    BestFitness,
    BestProvenance,
    FinalPopulation,
    HallOfFame,
    Success,
    Digest,
}

/// Result of a true seed-to-final-archive deterministic experiment rerun.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentReplay {
    pub verification: ArchiveVerification,
    pub replayed_digest: String,
    pub mismatches: Vec<ExperimentReplayMismatch>,
}

impl ExperimentReplay {
    /// Whether both stored-archive verification and the full rerun agree.
    pub fn is_reproduced(&self) -> bool {
        self.verification.is_intact() && self.mismatches.is_empty()
    }
}

/// Rerun the complete evolutionary experiment from the archived problem,
/// configuration, seed, early-stop criteria and Hall-of-Fame capacity, then
/// compare every deterministic archived result.
///
/// This is stronger than [`verify_archive`]: it regenerates every population in
/// the trajectory. It is guaranteed only for a supported archive produced by
/// the same deterministic engine semantics; it is not an authenticity check.
pub fn replay_experiment(archive: &ExperimentArchive) -> Result<ExperimentReplay, ExperimentError> {
    let verification = verify_archive(archive);
    let replayed = run_experiment(
        &archive.problem,
        RunOptions {
            hall_of_fame_capacity: archive.hall_of_fame_capacity,
            parallel: false,
        },
    )?;
    let mut mismatches = Vec::new();

    if archive.canonical_bytes()? != replayed.canonical_bytes()?
    {
        mismatches.push(ExperimentReplayMismatch::CanonicalContent);
    }
    if archive.generations_executed != replayed.generations_executed
    {
        mismatches.push(ExperimentReplayMismatch::GenerationsExecuted);
    }
    if archive.history.len() != replayed.history.len()
        || !archive
            .history
            .iter()
            .zip(&replayed.history)
            .all(|(stored, rerun)| records_equal(stored, rerun))
    {
        mismatches.push(ExperimentReplayMismatch::History);
    }
    if !canonical_equal(&archive.best.program, &replayed.best.program)
    {
        mismatches.push(ExperimentReplayMismatch::BestProgram);
    }
    if !fitness_equal(&archive.best.fitness, &replayed.best.fitness)
    {
        mismatches.push(ExperimentReplayMismatch::BestFitness);
    }
    if archive.best.generation != replayed.best.generation
        || archive.best.seed != replayed.best.seed
        || archive.best.fingerprint != replayed.best.fingerprint
    {
        mismatches.push(ExperimentReplayMismatch::BestProvenance);
    }
    if archive.final_population != replayed.final_population
    {
        mismatches.push(ExperimentReplayMismatch::FinalPopulation);
    }
    if archive.hall_of_fame.len() != replayed.hall_of_fame.len()
        || !archive
            .hall_of_fame
            .iter()
            .zip(&replayed.hall_of_fame)
            .all(|(stored, rerun)| entries_equal(stored, rerun))
    {
        mismatches.push(ExperimentReplayMismatch::HallOfFame);
    }
    if archive.success != replayed.success
    {
        mismatches.push(ExperimentReplayMismatch::Success);
    }
    if archive.digest != replayed.digest
    {
        mismatches.push(ExperimentReplayMismatch::Digest);
    }

    Ok(ExperimentReplay {
        verification,
        replayed_digest: replayed.digest,
        mismatches,
    })
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
                .any(|issue| matches!(issue, VerificationIssue::StoredFitnessMismatch { .. }))
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
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            VerificationIssue::ProgramFingerprintMismatch {
                location: ArchiveEntryLocation::Best,
                ..
            }
        )));
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
                .any(|issue| matches!(issue, VerificationIssue::HistoryGenerationGap { .. }))
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
        assert_ne!(hall.entries()[0].fingerprint, hall.entries()[1].fingerprint);
    }

    fn small_known_answer_archive() -> ExperimentArchive {
        let mut problem = benchmarks::identity();
        problem.seed = 0x0102_0304_0506_0708;
        problem.evolution.population_size = 1;
        problem.evolution.generations = 0;
        problem.evolution.elitism = 0;
        problem.success = crate::tensor::SuccessCriteria::max_loss(-1.0);
        run_experiment(
            &problem,
            RunOptions {
                hall_of_fame_capacity: 1,
                parallel: false,
            },
        )
        .unwrap()
    }

    #[test]
    fn canonical_archive_known_answer() {
        let archive = small_known_answer_archive();
        let bytes = archive.canonical_bytes().unwrap();
        let expected_prefix = [
            super::super::digest::DIGEST_MAGIC,
            &[0x01],
            &DIGEST_FORMAT_VERSION.to_le_bytes(),
            &ARCHIVE_SCHEMA_VERSION.to_le_bytes(),
        ]
        .concat();
        assert_eq!(&bytes[..expected_prefix.len()], expected_prefix);
        assert_eq!(bytes.len(), 1122);
        assert_eq!(
            archive.digest,
            "9f2bedb314dc1955fedec7beb4a0c837c72ce1e964633fec5b7b9aff2ec39a28"
        );
    }

    #[test]
    fn canonical_bytes_and_digest_repeat_exactly() {
        let archive = small_known_answer_archive();
        assert_eq!(
            archive.canonical_bytes().unwrap(),
            archive.canonical_bytes().unwrap()
        );
        assert_eq!(
            archive.recompute_digest().unwrap(),
            archive.recompute_digest().unwrap()
        );
    }

    #[test]
    fn seed_and_best_entry_changes_affect_digest() {
        let archive = small_known_answer_archive();
        let original = archive.recompute_digest().unwrap();

        let mut changed_seed = archive.clone();
        changed_seed.seed ^= 1;
        assert_ne!(original, changed_seed.recompute_digest().unwrap());

        let mut changed_best = archive.clone();
        changed_best.best.generation += 1;
        assert_ne!(original, changed_best.recompute_digest().unwrap());
    }

    #[test]
    fn unsupported_digest_format_and_non_finite_values_are_rejected() {
        let mut archive = small_known_answer_archive();
        archive.digest_format_version += 1;
        assert!(matches!(
            archive.canonical_bytes(),
            Err(ExperimentError::Digest(
                super::super::digest::DigestError::UnsupportedFormat { .. }
            ))
        ));

        for bits in [0x7fc0_0001, 0xffc0_1234]
        {
            let mut non_finite = small_known_answer_archive();
            non_finite.problem.cases[0].inputs[0].data[0] = f32::from_bits(bits);
            assert!(matches!(
                non_finite.canonical_bytes(),
                Err(ExperimentError::Digest(
                    super::super::digest::DigestError::NonFiniteFloat
                ))
            ));
        }
    }

    #[test]
    fn verification_reports_independent_count_invariants() {
        let mut archive = archive();
        archive.final_population.valid = archive.final_population.size;
        archive.final_population.invalid = 1;
        archive.final_population.distinct = archive.final_population.size + 1;
        archive.history[0].exact_solutions = archive.history[0].valid_individuals + 1;
        archive.digest = archive.recompute_digest().unwrap();

        let report = verify_archive(&archive);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| matches!(issue, VerificationIssue::PopulationCountsMismatch { .. }))
        );
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            VerificationIssue::PopulationDistinctExceedsSize { .. }
        )));
        assert!(
            report
                .issues
                .iter()
                .any(|issue| matches!(issue, VerificationIssue::ExactSolutionsExceedValid { .. }))
        );
    }

    #[test]
    fn verification_detects_duplicate_and_fingerprint_collision_separately() {
        let mut collision = archive();
        assert!(collision.hall_of_fame.len() >= 2);
        collision.hall_of_fame[1].fingerprint = collision.hall_of_fame[0].fingerprint;
        collision.hall_of_fame[1].fitness.fingerprint = collision.hall_of_fame[0].fingerprint;
        collision.digest = collision.recompute_digest().unwrap();
        let report = verify_archive(&collision);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            VerificationIssue::HallOfFameFingerprintCollision { .. }
        )));

        let mut duplicate = archive();
        duplicate.hall_of_fame[1] = duplicate.hall_of_fame[0].clone();
        duplicate.digest = duplicate.recompute_digest().unwrap();
        let report = verify_archive(&duplicate);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| matches!(issue, VerificationIssue::HallOfFameDuplicate { .. }))
        );
    }

    #[test]
    fn verification_detects_order_and_best_policy_mismatches() {
        let mut archive = archive();
        assert!(archive.hall_of_fame.len() >= 2);
        archive.hall_of_fame.swap(0, 1);
        archive.digest = archive.recompute_digest().unwrap();
        let report = verify_archive(&archive);
        assert!(
            report
                .issues
                .contains(&VerificationIssue::HallOfFameOrderInvalid)
        );
        assert!(
            report
                .issues
                .contains(&VerificationIssue::BestDoesNotMatchHallOfFameLeader)
        );
    }

    #[test]
    fn full_experiment_replay_reproduces_trajectory_and_detects_rehashed_edits() {
        let archive = small_known_answer_archive();
        let replay = replay_experiment(&archive).unwrap();
        assert!(
            replay.is_reproduced(),
            "mismatches: {:?}",
            replay.mismatches
        );

        let mut edited = archive.clone();
        edited.history[0].diversity += 1;
        edited.digest = edited.recompute_digest().unwrap();
        let replay = replay_experiment(&edited).unwrap();
        assert!(replay.verification.digest_ok);
        assert!(
            replay
                .mismatches
                .contains(&ExperimentReplayMismatch::History)
        );
        assert!(
            replay
                .mismatches
                .contains(&ExperimentReplayMismatch::CanonicalContent)
        );
    }
}
