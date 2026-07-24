//! `sos know` — query the knowledge graph: neighbors, relations between two
//! objects, and shortest paths.

use sos_core::ObjectId;
use sos_knowledge::{Knowledge, KnowledgeGraph, Relation};

use crate::args::Args;
use crate::error::{CliError, Result};
use crate::store;

/// Run `sos know <path> <subcommand> ...`, dispatching to the knowledge-graph
/// queries built from the store's `Edge` objects.
///
/// Subcommands:
/// - `neighbors <id> <relation>` — objects `id` points to via `relation`.
/// - `in-neighbors <id> <relation>` — objects that point to `id` via `relation`.
/// - `related <from> <to>` — every relation directly connecting the two.
/// - `path <from> <to> [relation]` — the shortest path, optionally restricted
///   to one relation type.
///
/// # Errors
/// [`CliError::Usage`] for a malformed subcommand; [`CliError::Knowledge`] if
/// the store's edges cannot be parsed.
pub fn run(path: Option<&str>, args: &Args) -> Result<String> {
    let root = store::resolve_root(path)?;
    let s = store::open(&root)?;
    let kg = KnowledgeGraph::build(&s)?;

    let sub = args.positional(0, "know subcommand")?;
    match sub
    {
        "neighbors" =>
        {
            let id = parse_id(args.positional(1, "id")?)?;
            let relation = parse_relation(args.positional(2, "relation")?);
            Ok(format_ids(&kg.neighbors(id, &relation)))
        },
        "in-neighbors" =>
        {
            let id = parse_id(args.positional(1, "id")?)?;
            let relation = parse_relation(args.positional(2, "relation")?);
            Ok(format_ids(&kg.in_neighbors(id, &relation)))
        },
        "related" =>
        {
            let from = parse_id(args.positional(1, "from")?)?;
            let to = parse_id(args.positional(2, "to")?)?;
            let relations = kg.related(from, to);
            if relations.is_empty()
            {
                Ok("(no direct relation)".to_owned())
            }
            else
            {
                Ok(relations
                    .iter()
                    .map(Relation::code)
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        },
        "path" =>
        {
            let from = parse_id(args.positional(1, "from")?)?;
            let to = parse_id(args.positional(2, "to")?)?;
            let relation = args.positional_opt(3).map(parse_relation);
            match kg.path(from, to, relation.as_ref())
            {
                Some(p) => Ok(format_ids(&p)),
                None => Ok("(no path)".to_owned()),
            }
        },
        other => Err(CliError::Usage(format!(
            "unknown `sos know` subcommand `{other}` (expected neighbors, in-neighbors, related, path)"
        ))),
    }
}

fn parse_id(s: &str) -> Result<ObjectId> {
    crate::args::parse_object_id(s)
}

/// Parse a relation code (e.g. `"is-a"`, `"contradicts"`) into a [`Relation`];
/// anything not among the built-in codes becomes [`Relation::Custom`].
fn parse_relation(s: &str) -> Relation {
    match s
    {
        "is-a" => Relation::IsA,
        "generalizes" => Relation::Generalizes,
        "specializes" => Relation::Specializes,
        "derives-from" => Relation::DerivesFrom,
        "implies" => Relation::Implies,
        "equivalent-to" => Relation::EquivalentTo,
        "contradicts" => Relation::Contradicts,
        "supported-by" => Relation::SupportedBy,
        "refuted-by" => Relation::RefutedBy,
        "constrained-by" => Relation::ConstrainedBy,
        "has-dimension" => Relation::HasDimension,
        "measures" => Relation::Measures,
        "cites" => Relation::Cites,
        "instance-of" => Relation::InstanceOf,
        "supersedes" => Relation::Supersedes,
        "analogous-to" => Relation::AnalogousTo,
        "limit-of" => Relation::LimitOf,
        other => Relation::Custom(other.to_owned()),
    }
}

fn format_ids(ids: &[ObjectId]) -> String {
    if ids.is_empty()
    {
        "(none)".to_owned()
    }
    else
    {
        ids.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }
}
