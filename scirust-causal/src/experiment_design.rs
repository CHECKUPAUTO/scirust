//! **Experimental design**: which intervention would most reduce the
//! equivalence class?
//!
//! # The question this answers, and the one it does not
//!
//! `crate::equivalence_class` returns a [`Cpdag`] whose undirected edges mark
//! what observational data cannot settle. `crate::scm` showed that gap is not
//! cosmetic: two members of one class answer the same counterfactual `3` and
//! `1`. The obvious follow-up is *what experiment would close it* — and that
//! question, unlike the ones the earlier phases ask, has an exact answer that
//! needs no data at all.
//!
//! Under a perfect intervention on `v`, every edge incident to `v` becomes
//! orientable: cutting `v` off from its parents changes the distribution of
//! its children and leaves its parents alone, so comparing the observational
//! and interventional distributions reveals which neighbours are which. That
//! is a statement about the *graph*, so the number of edges it settles can be
//! computed before running anything.
//!
//! What this module does **not** do is tell you an experiment is worth
//! running. It has no cost model, no notion of ethics or feasibility beyond
//! the target list a caller supplies, and no opinion about sample size. It
//! reports how much structural ambiguity an experiment would remove. Whether
//! that is worth the cost is not a question data answers.
//!
//! # Outcome-dependence, and what is guaranteed anyway
//!
//! Orienting the edges *at* the target happens whatever the experiment finds.
//! What follows from them through Meek's rules does not: `x -> v` and
//! `v -> x` propagate differently. So each candidate reports two numbers —
//! a **guaranteed** count that holds for every possible outcome, and an
//! **optimistic** count for the luckiest one. Ranking uses the guarantee,
//! because a plan justified by its best case is not a plan.
//!
//! The guaranteed figure is computed by enumerating the possible outcomes
//! rather than by arguing about them, which costs `2^k` in the number of
//! undirected edges at the target. Past
//! [`ExperimentDesignConfig::max_outcome_enumeration`] the enumeration is
//! abandoned and the candidate falls back to the count of edges at the target
//! alone — still a true lower bound, since propagation only ever adds — with
//! a warning saying so. A capped candidate is never silently ranked as if it
//! had been fully evaluated.
//!
//! # Assumptions
//!
//! The input CPDAG is taken as correct, and everything `crate::equivalence_class`
//! assumed to produce it is inherited: causal sufficiency, faithfulness,
//! acyclicity, correct conditional-independence decisions. Added here: that
//! an intervention is **perfect** (hard, on the named target only, severing
//! it from its parents without touching any other mechanism) and that its
//! outcome is read without error. Real experiments are none of these things.
//! A plan is a statement about an idealisation; its value is that the
//! idealisation is stated rather than assumed away.

use crate::assumptions::CausalAssumption;
use crate::certificate::{CausalCertificate, IdentifiabilityStatus};
use crate::cpdag::Cpdag;
use crate::error::CausalError;
use crate::orientation::apply_meek_rules_with_background;

/// How to search for an experiment.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExperimentDesignConfig {
    /// Variables that may actually be intervened on. `None` means every
    /// variable is available — usually false in practice, which is why this
    /// is explicit rather than assumed.
    pub feasible_targets: Option<Vec<usize>>,
    /// Cap on the `2^k` outcome enumeration per candidate. A candidate whose
    /// target carries more undirected edges than `log2` of this is scored by
    /// its guaranteed edges-at-target only, and says so.
    pub max_outcome_enumeration: usize,
}

impl Default for ExperimentDesignConfig {
    fn default() -> Self {
        Self {
            feasible_targets: None,
            max_outcome_enumeration: 1024,
        }
    }
}

/// One candidate experiment, fully scored.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExperimentCandidate {
    pub target: usize,
    /// Undirected edges incident to `target`. These orient whatever the
    /// experiment finds, so this is a floor under every other count here.
    pub edges_at_target: usize,
    /// Newly directed edges under the **worst** outcome, including Meek
    /// propagation. Equals `edges_at_target` when the enumeration was capped.
    pub guaranteed_orientations: usize,
    /// Newly directed edges under the **best** outcome. Equals
    /// `edges_at_target` when the enumeration was capped.
    pub optimistic_orientations: usize,
    /// Whether the outcome enumeration ran to completion. When `false`, the
    /// two counts above are lower bounds rather than the true extremes.
    pub fully_enumerated: bool,
    /// Whether this experiment leaves nothing undirected, whatever it finds.
    pub resolves_everything: bool,
}

/// What the planner concluded.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExperimentPlanOutcome {
    /// A feasible experiment is guaranteed to orient at least one edge.
    Recommended { target: usize },
    /// The graph is already fully directed. No experiment is needed — which
    /// is a result, not a failure to find one.
    AlreadyDetermined,
    /// Something is still undirected but no feasible target touches it. The
    /// remaining ambiguity is out of reach of the experiments allowed.
    NoInformativeExperiment,
}

/// A ranked experiment plan.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExperimentPlan {
    /// Undirected edges remaining **after** closing the supplied graph under
    /// the orientation rules — the ambiguity an experiment could actually
    /// address.
    pub undirected_before: usize,
    /// Edges the supplied graph forced on its own, before any experiment. A
    /// CPDAG from `crate::equivalence_class` is already closed, so this is
    /// zero; a hand-built graph may not be. These are deliberately **not**
    /// credited to any candidate: an experiment gets no score for what the
    /// graph already implied.
    pub forced_by_graph_alone: usize,
    /// Every feasible candidate, best first. Ranked by guaranteed
    /// orientations, then optimistic, then ascending index — a total order,
    /// so the ranking never depends on iteration accidents.
    pub candidates: Vec<ExperimentCandidate>,
    pub outcome: ExperimentPlanOutcome,
    pub certificate: CausalCertificate,
    pub warnings: Vec<String>,
}

/// A worst-case sequence of experiments.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExperimentSequence {
    /// Intervention targets in order.
    pub steps: Vec<usize>,
    /// Undirected edges left when the sequence stopped.
    pub undirected_remaining: usize,
    /// Whether the sequence orients the whole graph.
    pub fully_oriented: bool,
    pub certificate: CausalCertificate,
    pub warnings: Vec<String>,
}

fn assumptions() -> Vec<CausalAssumption> {
    vec![
        CausalAssumption::Acyclicity,
        CausalAssumption::CausalSufficiency,
        CausalAssumption::Faithfulness,
        CausalAssumption::Other(
            "interventions are perfect: hard, on the named target only, severing it from its \
             parents without altering any other mechanism, and read without error"
                .to_string(),
        ),
    ]
}

const SCOPE_NOTE: &str = "This plans what an experiment would REVEAL about graph structure; it is \
                          not a claim about any effect size, nor that the experiment is worth its \
                          cost. It assumes the input CPDAG is correct and that interventions are \
                          perfect (hard, target-only, read without error).";

/// Undirected edges incident to `node`.
fn undirected_at(cpdag: &Cpdag, node: usize) -> Vec<usize> {
    cpdag
        .neighbors(node)
        .into_iter()
        .filter(|&other| cpdag.is_undirected(node, other))
        .collect()
}

/// Applies one outcome — a choice of direction for each undirected edge at
/// `target` — then propagates. Bit `i` of `assignment` set means the edge
/// points *into* the target.
fn apply_outcome(cpdag: &Cpdag, target: usize, neighbors: &[usize], assignment: u32) -> Cpdag {
    let mut world = cpdag.clone();
    for (bit, &other) in neighbors.iter().enumerate()
    {
        if assignment >> bit & 1 == 1
        {
            world.orient(other, target);
        }
        else
        {
            world.orient(target, other);
        }
    }
    apply_meek_rules_with_background(&mut world);
    world
}

/// Scores one candidate target against the current graph.
fn score(
    cpdag: &Cpdag,
    target: usize,
    undirected_before: usize,
    max_outcome_enumeration: usize,
) -> ExperimentCandidate {
    let neighbors = undirected_at(cpdag, target);
    let edges_at_target = neighbors.len();
    let combinations = 1u64 << edges_at_target;

    if edges_at_target == 0 || combinations > max_outcome_enumeration as u64
    {
        return ExperimentCandidate {
            target,
            edges_at_target,
            guaranteed_orientations: edges_at_target,
            optimistic_orientations: edges_at_target,
            fully_enumerated: edges_at_target == 0,
            resolves_everything: edges_at_target > 0 && edges_at_target == undirected_before,
        };
    }

    let mut worst = usize::MAX;
    let mut best = 0usize;
    let mut worst_leaves_none = true;
    for assignment in 0..combinations as u32
    {
        let world = apply_outcome(cpdag, target, &neighbors, assignment);
        let remaining = world.undirected_edges().len();
        let oriented = undirected_before - remaining;
        if oriented < worst
        {
            worst = oriented;
        }
        if oriented > best
        {
            best = oriented;
        }
        if remaining > 0
        {
            worst_leaves_none = false;
        }
    }

    ExperimentCandidate {
        target,
        edges_at_target,
        guaranteed_orientations: worst,
        optimistic_orientations: best,
        fully_enumerated: true,
        resolves_everything: worst_leaves_none,
    }
}

/// Validates and normalizes the feasible-target list.
fn feasible(cpdag: &Cpdag, config: &ExperimentDesignConfig) -> Result<Vec<usize>, CausalError> {
    match &config.feasible_targets
    {
        None => Ok((0..cpdag.n_nodes()).collect()),
        Some(listed) =>
        {
            let mut out: Vec<usize> = Vec::with_capacity(listed.len());
            for &index in listed
            {
                if index >= cpdag.n_nodes()
                {
                    return Err(CausalError::UnknownVariableIndex { index });
                }
                if !out.contains(&index)
                {
                    out.push(index);
                }
            }
            out.sort_unstable();
            Ok(out)
        },
    }
}

/// Ranks the feasible single-variable experiments for `cpdag`.
///
/// See the module docs for what the two counts on each candidate mean and why
/// the ranking uses the guaranteed one.
///
/// # Errors
///
/// [`CausalError::InvalidConfiguration`] if `max_outcome_enumeration` is
/// zero; [`CausalError::UnknownVariableIndex`] for an out-of-range feasible
/// target; [`CausalError::InvalidContract`] if the certificate cannot be
/// built.
pub fn plan_next_experiment(
    cpdag: &Cpdag,
    config: &ExperimentDesignConfig,
) -> Result<ExperimentPlan, CausalError> {
    if config.max_outcome_enumeration == 0
    {
        return Err(CausalError::InvalidConfiguration {
            name: "max_outcome_enumeration",
            value: 0.0,
        });
    }
    let targets = feasible(cpdag, config)?;
    let mut warnings = Vec::new();

    // Close the input first. Anything the orientation rules force from the
    // graph alone is not something an experiment reveals, and crediting it to
    // one would inflate every count below. A CPDAG from discovery is already
    // closed, so this is a no-op there; a hand-built graph may not be.
    let mut cpdag = cpdag.clone();
    let supplied_undirected = cpdag.undirected_edges().len();
    apply_meek_rules_with_background(&mut cpdag);
    let undirected_before = cpdag.undirected_edges().len();
    let forced_by_graph_alone = supplied_undirected - undirected_before;
    if forced_by_graph_alone > 0
    {
        warnings.push(format!(
            "the supplied graph was not closed under the orientation rules: {forced_by_graph_alone}              edge(s) are forced by it alone and are excluded from every candidate's score"
        ));
    }
    let cpdag = &cpdag;

    let mut candidates: Vec<ExperimentCandidate> = targets
        .iter()
        .map(|&target| {
            score(
                cpdag,
                target,
                undirected_before,
                config.max_outcome_enumeration,
            )
        })
        .collect();
    for candidate in &candidates
    {
        if !candidate.fully_enumerated && candidate.edges_at_target > 0
        {
            warnings.push(format!(
                "target {} carries {} undirected edges, whose {} outcomes exceed the enumeration \
                 cap of {}; it is scored by edges-at-target alone, a lower bound",
                candidate.target,
                candidate.edges_at_target,
                1u64 << candidate.edges_at_target,
                config.max_outcome_enumeration,
            ));
        }
    }
    candidates.sort_by(|a, b| {
        b.guaranteed_orientations
            .cmp(&a.guaranteed_orientations)
            .then(b.optimistic_orientations.cmp(&a.optimistic_orientations))
            .then(a.target.cmp(&b.target))
    });

    let best = candidates
        .first()
        .filter(|candidate| candidate.guaranteed_orientations > 0);
    let (outcome, status, evidence) = match (undirected_before, best)
    {
        (0, _) => (
            ExperimentPlanOutcome::AlreadyDetermined,
            IdentifiabilityStatus::Identifiable,
            "the graph is fully directed; no experiment would orient anything".to_string(),
        ),
        (remaining, None) => (
            ExperimentPlanOutcome::NoInformativeExperiment,
            IdentifiabilityStatus::NotIdentifiable,
            format!(
                "{remaining} undirected edge(s) remain, and no feasible target is guaranteed to \
                 orient any of them"
            ),
        ),
        (remaining, Some(candidate)) => (
            ExperimentPlanOutcome::Recommended {
                target: candidate.target,
            },
            IdentifiabilityStatus::Identifiable,
            format!(
                "{} feasible target(s) scored against {remaining} undirected edge(s); \
                 intervening on {} orients at least {} of them whatever the outcome, and at most \
                 {}",
                candidates.len(),
                candidate.target,
                candidate.guaranteed_orientations,
                candidate.optimistic_orientations,
            ),
        ),
    };

    let mut builder = CausalCertificate::builder(
        format!("which single-variable experiment most reduces the equivalence class of a {}-node CPDAG?", cpdag.n_nodes()),
        status,
        assumptions(),
        evidence,
    )
    .with_sensitivity(SCOPE_NOTE);
    if let ExperimentPlanOutcome::Recommended { .. } = outcome
    {
        for candidate in candidates.iter().skip(1).take(4)
        {
            builder = builder.with_unresolved_alternative(format!(
                "target {} would guarantee {} orientation(s)",
                candidate.target, candidate.guaranteed_orientations
            ));
        }
    }

    Ok(ExperimentPlan {
        undirected_before,
        forced_by_graph_alone,
        candidates,
        outcome,
        certificate: builder.finalize()?,
        warnings,
    })
}

/// Plans experiments one after another, each time assuming the **worst**
/// outcome the previous one could have had.
///
/// The result is therefore an upper bound: this many experiments suffice
/// however they turn out. It is **greedy, not minimal** — at each step it
/// takes the locally best target, and no claim is made that a shorter
/// sequence does not exist. Finding the shortest is a set-cover-flavoured
/// problem and is not attempted here.
///
/// # Errors
///
/// As [`plan_next_experiment`].
pub fn greedy_experiment_sequence(
    cpdag: &Cpdag,
    config: &ExperimentDesignConfig,
) -> Result<ExperimentSequence, CausalError> {
    let mut world = cpdag.clone();
    apply_meek_rules_with_background(&mut world);
    let mut steps = Vec::new();
    let mut warnings = Vec::new();
    let mut capped = false;

    loop
    {
        let plan = plan_next_experiment(&world, config)?;
        capped |= plan.candidates.iter().any(|c| !c.fully_enumerated);
        let ExperimentPlanOutcome::Recommended { target } = plan.outcome
        else
        {
            break;
        };
        // Advance to the worst outcome this experiment could produce, so the
        // sequence's length is a guarantee rather than a hope.
        let neighbors = undirected_at(&world, target);
        let combinations = 1u64 << neighbors.len();
        let mut worst: Option<Cpdag> = None;
        if combinations <= config.max_outcome_enumeration as u64
        {
            for assignment in 0..combinations as u32
            {
                let outcome_world = apply_outcome(&world, target, &neighbors, assignment);
                let replace = worst.as_ref().is_none_or(|current| {
                    outcome_world.undirected_edges().len() > current.undirected_edges().len()
                });
                if replace
                {
                    worst = Some(outcome_world);
                }
            }
        }
        else
        {
            // Cannot enumerate: advance by the outcome-independent part only
            // (edges at the target, no propagation credited).
            worst = Some(apply_outcome(&world, target, &neighbors, 0));
        }
        let Some(next) = worst
        else
        {
            break;
        };
        if next.undirected_edges().len() >= world.undirected_edges().len()
        {
            warnings.push(format!(
                "target {target} was recommended but its worst outcome removes no undirected \
                 edge; stopping rather than looping"
            ));
            break;
        }
        world = next;
        steps.push(target);
    }

    let undirected_remaining = world.undirected_edges().len();
    let fully_oriented = undirected_remaining == 0;
    if capped
    {
        warnings.push(
            "at least one step hit the outcome-enumeration cap; the sequence is still valid but \
             may be longer than necessary"
                .to_string(),
        );
    }

    let status = if fully_oriented
    {
        IdentifiabilityStatus::Identifiable
    }
    else if capped
    {
        IdentifiabilityStatus::Inconclusive
    }
    else
    {
        IdentifiabilityStatus::NotIdentifiable
    };
    let evidence = if fully_oriented
    {
        format!(
            "{} experiment(s) orient the whole graph, whatever each one finds",
            steps.len()
        )
    }
    else
    {
        format!(
            "{} experiment(s) exhausted the feasible targets with {undirected_remaining} \
             undirected edge(s) still open",
            steps.len()
        )
    };

    let mut builder = CausalCertificate::builder(
        format!(
            "how many feasible single-variable experiments orient a {}-node CPDAG?",
            cpdag.n_nodes()
        ),
        status,
        assumptions(),
        evidence,
    )
    .with_sensitivity(format!(
        "{SCOPE_NOTE} The sequence is greedy and worst-case: it is an upper bound on the \
         experiments needed, not a proof that no shorter sequence exists."
    ));
    if !fully_oriented
    {
        builder = builder.with_unresolved_alternative(format!(
            "{undirected_remaining} edge(s) stay undirected under every feasible experiment"
        ));
    }

    Ok(ExperimentSequence {
        steps,
        undirected_remaining,
        fully_oriented,
        certificate: builder.finalize()?,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(cpdag: &Cpdag) -> ExperimentPlan {
        plan_next_experiment(cpdag, &ExperimentDesignConfig::default()).unwrap()
    }

    #[test]
    fn a_single_undirected_edge_is_resolved_by_either_endpoint() {
        let g = Cpdag::from_edges(2, &[], &[(0, 1)]).unwrap();
        let result = plan(&g);
        assert_eq!(
            result.outcome,
            ExperimentPlanOutcome::Recommended { target: 0 }
        );
        assert_eq!(result.undirected_before, 1);
        assert!(result.candidates.iter().all(|c| c.resolves_everything));
        assert_eq!(result.candidates[0].guaranteed_orientations, 1);
    }

    #[test]
    fn a_fully_directed_graph_needs_no_experiment() {
        let g = Cpdag::from_edges(3, &[(0, 1), (1, 2)], &[]).unwrap();
        let result = plan(&g);
        assert_eq!(result.outcome, ExperimentPlanOutcome::AlreadyDetermined);
        assert_eq!(
            result.certificate.status(),
            IdentifiabilityStatus::Identifiable
        );
    }

    #[test]
    fn an_out_of_reach_ambiguity_is_reported_not_papered_over() {
        // The undirected edge is between 2 and 3, but only 0 and 1 may be
        // intervened on.
        let g = Cpdag::from_edges(4, &[(0, 1)], &[(2, 3)]).unwrap();
        let config = ExperimentDesignConfig {
            feasible_targets: Some(vec![0, 1]),
            ..Default::default()
        };
        let result = plan_next_experiment(&g, &config).unwrap();
        assert_eq!(
            result.outcome,
            ExperimentPlanOutcome::NoInformativeExperiment
        );
        assert_eq!(
            result.certificate.status(),
            IdentifiabilityStatus::NotIdentifiable
        );
    }

    #[test]
    fn the_hub_of_a_star_beats_a_leaf() {
        // 0 at the centre, undirected to 1, 2, 3.
        let g = Cpdag::from_edges(4, &[], &[(0, 1), (0, 2), (0, 3)]).unwrap();
        let result = plan(&g);
        assert_eq!(
            result.outcome,
            ExperimentPlanOutcome::Recommended { target: 0 }
        );
        let hub = &result.candidates[0];
        assert_eq!(hub.target, 0);
        assert_eq!(hub.edges_at_target, 3);
        assert_eq!(hub.guaranteed_orientations, 3);
        assert!(hub.resolves_everything);
        // A leaf orients its own edge for certain, and may propagate further.
        let leaf = result.candidates.iter().find(|c| c.target == 1).unwrap();
        assert_eq!(leaf.edges_at_target, 1);
        assert_eq!(leaf.guaranteed_orientations, 1);
        assert!(leaf.optimistic_orientations >= 1);
    }

    #[test]
    fn propagation_is_credited_only_when_every_outcome_delivers_it() {
        // The undirected triangle is the smallest graph where the guarantee
        // and the best case come apart, so it is worth stating exactly why.
        //
        // Intervening on 0 orients 0-1 and 0-2 whatever happens. Whether 1-2
        // follows depends on how they land:
        //
        //   0 -> 1, 2 -> 0  and  1 -> 0, 0 -> 2  chain through 0, so R2 forces
        //                                        the third edge: 3 oriented.
        //   0 -> 1, 0 -> 2  and  1 -> 0, 2 -> 0  do not. R1 needs the outer
        //                                        pair non-adjacent and they
        //                                        are adjacent; R2 needs a
        //                                        chain and there is none. The
        //                                        third edge stays genuinely
        //                                        free: 2 oriented.
        //
        // So the guarantee is 2 and the best case is 3, and this experiment
        // must NOT be reported as resolving everything.
        let g = Cpdag::from_edges(3, &[], &[(0, 1), (0, 2), (1, 2)]).unwrap();
        let result = plan(&g);
        for candidate in &result.candidates
        {
            assert_eq!(candidate.edges_at_target, 2);
            assert_eq!(candidate.guaranteed_orientations, 2);
            assert_eq!(candidate.optimistic_orientations, 3);
            assert!(!candidate.resolves_everything);
        }
    }

    #[test]
    fn ranking_is_a_total_order_and_does_not_depend_on_input_order() {
        let a = Cpdag::from_edges(4, &[], &[(0, 1), (0, 2), (0, 3)]).unwrap();
        let b = Cpdag::from_edges(4, &[], &[(0, 3), (0, 2), (0, 1)]).unwrap();
        assert_eq!(plan(&a).candidates, plan(&b).candidates);
    }

    #[test]
    fn the_enumeration_cap_downgrades_to_a_lower_bound_and_says_so() {
        let g = Cpdag::from_edges(4, &[], &[(0, 1), (0, 2), (0, 3)]).unwrap();
        let config = ExperimentDesignConfig {
            max_outcome_enumeration: 4, // 2^3 = 8 > 4
            ..Default::default()
        };
        let result = plan_next_experiment(&g, &config).unwrap();
        let hub = result.candidates.iter().find(|c| c.target == 0).unwrap();
        assert!(!hub.fully_enumerated);
        assert_eq!(hub.guaranteed_orientations, hub.edges_at_target);
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("enumeration cap"))
        );
    }

    #[test]
    fn a_greedy_sequence_orients_a_chain_and_is_worst_case() {
        let g = Cpdag::from_edges(4, &[], &[(0, 1), (1, 2), (2, 3)]).unwrap();
        let sequence = greedy_experiment_sequence(&g, &ExperimentDesignConfig::default()).unwrap();
        assert!(sequence.fully_oriented);
        assert_eq!(sequence.undirected_remaining, 0);
        assert!(!sequence.steps.is_empty());
        assert_eq!(
            sequence.certificate.status(),
            IdentifiabilityStatus::Identifiable
        );
    }

    #[test]
    fn a_sequence_that_cannot_finish_says_so_rather_than_looping() {
        let g = Cpdag::from_edges(4, &[], &[(0, 1), (2, 3)]).unwrap();
        let config = ExperimentDesignConfig {
            feasible_targets: Some(vec![0, 1]),
            ..Default::default()
        };
        let sequence = greedy_experiment_sequence(&g, &config).unwrap();
        assert!(!sequence.fully_oriented);
        assert_eq!(sequence.undirected_remaining, 1);
        assert_eq!(
            sequence.certificate.status(),
            IdentifiabilityStatus::NotIdentifiable
        );
    }

    #[test]
    fn malformed_configuration_and_targets_are_typed_errors() {
        let g = Cpdag::from_edges(2, &[], &[(0, 1)]).unwrap();
        assert!(matches!(
            plan_next_experiment(
                &g,
                &ExperimentDesignConfig {
                    max_outcome_enumeration: 0,
                    ..Default::default()
                }
            ),
            Err(CausalError::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            plan_next_experiment(
                &g,
                &ExperimentDesignConfig {
                    feasible_targets: Some(vec![7]),
                    ..Default::default()
                }
            ),
            Err(CausalError::UnknownVariableIndex { index: 7 })
        ));
    }

    #[test]
    fn orientations_the_graph_already_forces_are_not_credited_to_an_experiment() {
        // R4's premises: 0-1, 0-2, 0-3 undirected, 2 -> 3 -> 1, with 1 and 2
        // non-adjacent. R4 forces 0 -> 1 from the graph alone, so the real
        // ambiguity is 2 edges, not the 3 that were written down.
        let g = Cpdag::from_edges(4, &[(2, 3), (3, 1)], &[(0, 1), (0, 2), (0, 3)]).unwrap();
        let result = plan(&g);
        assert_eq!(result.forced_by_graph_alone, 1, "R4 should close 0 -> 1");
        assert_eq!(result.undirected_before, 2);
        assert!(result.warnings.iter().any(|w| w.contains("not closed")));
        // No candidate may claim the edge the graph itself settled.
        for candidate in &result.candidates
        {
            assert!(candidate.guaranteed_orientations <= 2);
        }
    }

    #[test]
    fn a_closed_graph_reports_nothing_forced_and_no_warning() {
        let g = Cpdag::from_edges(3, &[], &[(0, 1), (0, 2), (1, 2)]).unwrap();
        let result = plan(&g);
        assert_eq!(result.forced_by_graph_alone, 0);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn the_certificate_never_carries_an_effect_estimate() {
        let g = Cpdag::from_edges(2, &[], &[(0, 1)]).unwrap();
        let result = plan(&g);
        assert_eq!(result.certificate.estimate(), None);
        assert!(
            result
                .certificate
                .sensitivity_note()
                .unwrap()
                .contains("not a claim about any effect size")
        );
    }
}
