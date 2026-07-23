//! `sos ask` — run a curiosity sweep over the knowledge graph.

use sos_curiosity::{BeCurious, Budget, Curiosity, CuriosityPolicy};
use sos_knowledge::KnowledgeGraph;

use crate::args::Args;
use crate::error::Result;
use crate::store;

/// Run `sos ask [path] [--limit <n>] [--w-contradiction <n>] [--w-novelty <n>]
/// [--w-info-gain <n>] [--w-inv-cost <n>]`: build the knowledge graph from the
/// store's `Edge` objects, sweep it for open questions, and print the ranked
/// agenda (highest priority first).
///
/// # Errors
/// [`crate::error::CliError::Store`]/[`crate::error::CliError::Knowledge`] if
/// the store cannot be opened or its edges cannot be parsed.
pub fn run(path: Option<&str>, args: &Args) -> Result<String> {
    let root = store::resolve_root(path)?;
    let s = store::open(&root)?;
    let kg = KnowledgeGraph::build(&s)?;

    let default = CuriosityPolicy::default();
    let policy = CuriosityPolicy {
        w_info_gain: args.flag_i64("w-info-gain", default.w_info_gain)?,
        w_novelty: args.flag_i64("w-novelty", default.w_novelty)?,
        w_contradiction: args.flag_i64("w-contradiction", default.w_contradiction)?,
        w_inv_cost: args.flag_i64("w-inv-cost", default.w_inv_cost)?,
        ..default
    };
    let limit = args.flag_i64("limit", 20)?.max(0) as usize;

    let engine = Curiosity::new(&kg);
    let agenda = engine.sweep(&policy, &Budget::new(limit));

    if agenda.is_empty()
    {
        return Ok("(no open questions found)".to_owned());
    }

    let mut out = String::new();
    for (i, sq) in agenda.iter().enumerate()
    {
        out.push_str(&format!(
            "{}. [{}] priority={} — {}\n   subject: {}\n",
            i + 1,
            sq.question.strategy.code(),
            sq.priority.total,
            sq.question.prompt,
            sq.question
                .subject
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out.pop();
    Ok(out)
}
