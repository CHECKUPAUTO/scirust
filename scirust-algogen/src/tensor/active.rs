//! Backward liveness analysis for tensor programs.

use super::ir::TensorProgram;

/// Identify the instructions that contribute to the selected output.
///
/// The returned vector has one entry per instruction. A `true` entry means
/// that the instruction is part of the output's dependency graph.
///
/// Untrusted programs must first be passed to `verify_program`. For a malformed
/// output index, this function returns an all-false map rather than panicking.
pub fn analyze_active(program: &TensorProgram) -> Vec<bool> {
    let mut active = vec![false; program.instructions.len()];

    if program.output >= program.instructions.len()
    {
        return active;
    }

    active[program.output] = true;

    for node in (0..program.instructions.len()).rev()
    {
        if !active[node]
        {
            continue;
        }

        program.instructions[node].for_each_source(|source| {
            if source < node
            {
                active[source] = true;
            }
        });
    }

    active
}
