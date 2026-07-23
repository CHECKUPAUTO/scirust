//! `sos plan` — recommend the next experiment from a set of candidate designs.
//!
//! Computing the expected-information-gain [`Estimate`] behind each
//! [`Candidate`] is `sos-scirust`'s job (Invariant VIII); this command
//! **consumes** already-computed estimates from a JSON file and runs the
//! deterministic [`GreedyPlanner`] over them — exactly the boundary
//! `sos-planner` itself draws.

use sos_core::ObjectId;
use sos_planner::{Candidate, GreedyPlanner, Planner, StopVerdict, UtilityPolicy};

use crate::args::Args;
use crate::error::{CliError, Result};

/// Run `sos plan <candidates.json> [--floor <milli>] [--budget <milli>]`.
///
/// `candidates.json` is a JSON array of [`Candidate`] (each already carrying
/// its own [`sos_planner::Estimate`] and [`sos_planner::Cost`]). Without
/// `--budget`, ranks by `EIG / cost` ([`UtilityPolicy::EigPerCost`]); with it,
/// ranks by raw EIG subject to the cost budget
/// ([`UtilityPolicy::EigBudgeted`], excluding any design over budget).
/// `--floor` sets the significance floor in millibits (default `0`).
///
/// # Errors
/// [`CliError::Io`]/[`CliError::Serde`] if the candidates file cannot be read
/// or parsed; [`CliError::Planner`] if there are no candidates.
pub fn run(args: &Args) -> Result<String> {
    let path = args.positional(0, "candidates.json")?;
    let bytes = std::fs::read(path)?;
    let candidates: Vec<Candidate> = serde_json::from_slice(&bytes)?;

    let floor = args.flag_i64("floor", 0)?;
    let policy = match args.flag("budget")
    {
        Some(b) =>
        {
            let budget = b
                .parse()
                .map_err(|_| CliError::Usage(format!("--budget must be an integer, got `{b}`")))?;
            UtilityPolicy::EigBudgeted { budget }
        },
        None => UtilityPolicy::EigPerCost,
    };

    let plan = GreedyPlanner::new().recommend(&candidates, policy, floor)?;
    Ok(render(&plan))
}

/// A human-readable rendering of a recommendation.
fn render(plan: &sos_planner::Plan) -> String {
    let mut out = String::new();
    out.push_str(&format!("Policy: {}\n", plan.policy.code()));
    out.push_str(&format!(
        "EIG floor: {} millibits\n\n",
        plan.eig_floor_milli
    ));
    out.push_str("Ranked designs:\n");
    for (i, d) in plan.ranked.iter().enumerate()
    {
        out.push_str(&format!(
            "  {}. {} — EIG={}±{} millibits, cost={}, utility={}\n",
            i + 1,
            format_experiment(d.experiment),
            d.eig.bits_milli,
            d.eig.se_milli,
            d.cost.total(),
            d.utility
        ));
    }
    match plan.verdict
    {
        StopVerdict::Recommend(id) =>
        {
            out.push_str(&format!("\nRecommend: {}", format_experiment(id)))
        },
        StopVerdict::InformationExhausted =>
        {
            out.push_str("\nRecommend: information exhausted — no design clears the floor");
        },
    }
    out
}

/// A shortened, readable rendering of an experiment's object id.
fn format_experiment(id: ObjectId) -> String {
    id.to_string()
}
