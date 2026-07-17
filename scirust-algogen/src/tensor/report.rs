//! Deterministic, human-readable reporting of experiment archives.
//!
//! Reports contain only deterministic experiment data — problem, seed, success,
//! generations, best program, correctness, structural costs, hall of fame and
//! (optionally) the replay result. No wall-clock timing appears here;
//! observational benchmark data is kept separate from this deterministic view.

use std::fmt::Write;

use super::archive::{ExperimentArchive, ReplayReport};
use super::ir::TensorInstruction;

/// Render a deterministic text report. When `replay` is supplied its verdict is
/// appended.
pub fn text_report(archive: &ExperimentArchive, replay: Option<&ReplayReport>) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "experiment: {}", archive.problem.id);
    let _ = writeln!(out, "  {}", archive.problem.description);
    let _ = writeln!(out, "schema version: {}", archive.schema_version);
    let _ = writeln!(out, "crate version: {}", archive.crate_version);
    let _ = writeln!(out, "seed: {}", archive.seed);
    let _ = writeln!(out, "success: {}", archive.success);
    let _ = writeln!(
        out,
        "generations executed: {}",
        archive.generations_executed
    );
    let _ = writeln!(out, "digest: {}", archive.digest);

    let best = &archive.best;
    let _ = writeln!(out, "best (generation {}):", best.generation);
    let _ = writeln!(out, "  fingerprint: {:032x}", best.fingerprint);
    let _ = writeln!(out, "  loss: {}", best.fitness.loss);
    let _ = writeln!(out, "  failed cases: {}", best.fitness.failed_cases);
    let cost = &best.fitness.cost;
    let _ = writeln!(out, "  active instructions: {}", cost.active_instructions);
    let _ = writeln!(out, "  estimated flops: {}", cost.estimated_flops);
    let _ = writeln!(out, "  peak live elements: {}", cost.peak_live_elements);
    let _ = writeln!(
        out,
        "  total active elements: {}",
        cost.total_active_elements
    );
    let _ = writeln!(
        out,
        "  generated intermediate bytes: {}",
        cost.generated_intermediate_bytes
    );
    let _ = writeln!(out, "  dead instructions: {}", cost.dead_instructions);

    let _ = writeln!(out, "  program:");
    for (index, instruction) in best.program.instructions.iter().enumerate()
    {
        let marker = if index == best.program.output
        {
            " <- output"
        }
        else
        {
            ""
        };
        let _ = writeln!(out, "    {index}: {}{marker}", describe(instruction));
    }

    let _ = writeln!(
        out,
        "hall of fame ({} entries):",
        archive.hall_of_fame.len()
    );
    for (rank, entry) in archive.hall_of_fame.iter().enumerate()
    {
        let _ = writeln!(
            out,
            "  #{rank}: loss={} active={} flops={} fp={:032x}",
            entry.fitness.loss,
            entry.fitness.cost.active_instructions,
            entry.fitness.cost.estimated_flops,
            entry.fingerprint
        );
    }

    if let Some(replay) = replay
    {
        let _ = writeln!(
            out,
            "replay: intact={} digest_ok={} entries_checked={} mismatches={}",
            replay.is_intact(),
            replay.digest_ok,
            replay.entries_checked,
            replay.mismatches.len()
        );
    }

    out
}

/// Render the archive as a JSON string (deterministic field order).
pub fn json_report(archive: &ExperimentArchive) -> Result<String, String> {
    serde_json::to_string_pretty(archive).map_err(|error| error.to_string())
}

fn describe(instruction: &TensorInstruction) -> String {
    match *instruction
    {
        TensorInstruction::Input { input } => format!("Input({input})"),
        TensorInstruction::Add { lhs, rhs } => format!("Add({lhs}, {rhs})"),
        TensorInstruction::MatMul { lhs, rhs } => format!("MatMul({lhs}, {rhs})"),
        TensorInstruction::Transpose2d { src } => format!("Transpose2d({src})"),
        TensorInstruction::Relu { src } => format!("Relu({src})"),
        TensorInstruction::Scale { src, factor } => format!("Scale({src}, {factor})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::archive::replay;
    use crate::tensor::benchmarks;
    use crate::tensor::experiment::{RunOptions, run_experiment};

    #[test]
    fn text_report_is_deterministic_and_mentions_key_fields() {
        let archive = run_experiment(&benchmarks::identity(), RunOptions::default()).unwrap();
        let replay = replay(&archive).unwrap();

        let first = text_report(&archive, Some(&replay));
        let second = text_report(&archive, Some(&replay));
        assert_eq!(first, second);

        assert!(first.contains("experiment: identity"));
        assert!(first.contains("digest:"));
        assert!(first.contains("replay: intact=true"));
    }

    #[test]
    fn json_report_round_trips() {
        let archive = run_experiment(&benchmarks::relu(), RunOptions::default()).unwrap();
        let json = json_report(&archive).unwrap();
        let decoded: ExperimentArchive = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, archive);
    }
}
