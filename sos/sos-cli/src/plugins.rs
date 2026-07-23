//! `sos plugins` — list or find plugins by role and domain.
//!
//! [`sos_registry::Registry`] is a pure in-memory index — it has no
//! persistence of its own and no relationship to the object store (registering
//! a plugin is not a store operation). This command's own, honest convention
//! is a JSON array of [`PluginDescriptor`] (which already derives
//! `Serialize`/`Deserialize`) as the descriptors file — the same
//! "consume already-known data from a file" shape `sos plan` uses for
//! candidates, since there is no other persistence format to point at.

use sos_registry::{DomainTag, PluginDescriptor, Registry, Role};

use crate::args::Args;
use crate::error::Result;

/// Run `sos plugins <descriptors.json> [--role <role>] [--domain <tag>]`.
///
/// Without `--role`, lists every descriptor. With `--role`, finds descriptors
/// of that role (optionally further filtered to `--domain`).
///
/// # Errors
/// [`crate::error::CliError::Io`]/[`crate::error::CliError::Serde`] if the
/// descriptors file cannot be read or parsed.
pub fn run(args: &Args) -> Result<String> {
    let path = args.positional(0, "descriptors.json")?;
    let bytes = std::fs::read(path)?;
    let descriptors: Vec<PluginDescriptor> = serde_json::from_slice(&bytes)?;

    let mut registry = Registry::new();
    for d in descriptors
    {
        registry.register(d);
    }

    let matches = match args.flag("role")
    {
        Some(role) =>
        {
            let role = parse_role(role);
            let domain = args.flag("domain").map(|d| DomainTag(d.to_owned()));
            registry.find(&role, domain.as_ref())
        },
        None => registry.all(),
    };

    if matches.is_empty()
    {
        return Ok("(no matching plugins)".to_owned());
    }

    let mut out = String::new();
    for d in matches
    {
        out.push_str(&format!(
            "{} v{} [{}] level={} domains={}\n",
            d.name.0,
            d.version,
            role_code(&d.role),
            d.level.code(),
            d.domains
                .iter()
                .map(|t| t.0.as_str())
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    out.pop();
    Ok(out)
}

/// Parse a role code into a [`Role`]; anything unrecognized becomes
/// [`Role::Custom`].
fn parse_role(s: &str) -> Role {
    match s
    {
        "knowledge" => Role::Knowledge,
        "reasoning" => Role::Reasoning,
        "curiosity" => Role::Curiosity,
        "simulation" => Role::Simulation,
        "planning" => Role::Planning,
        "publication" => Role::Publication,
        "memory" => Role::Memory,
        "hypothesis-generator" => Role::HypothesisGenerator,
        "predictor" => Role::Predictor,
        "experiment-designer" => Role::ExperimentDesigner,
        "executor" => Role::Executor,
        "evidence-extractor" => Role::EvidenceExtractor,
        "statistical-evaluator" => Role::StatisticalEvaluator,
        "hypothesis-ranker" => Role::HypothesisRanker,
        "theory-reviser" => Role::TheoryReviser,
        other => Role::Custom(other.to_owned()),
    }
}

/// A short display code for a role (mirrors the parser's vocabulary).
fn role_code(role: &Role) -> String {
    match role
    {
        Role::Custom(s) => format!("custom:{s}"),
        other => format!("{other:?}"),
    }
}
