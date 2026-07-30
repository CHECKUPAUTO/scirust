//! # Hypothesis harness: regional causal memory vs. retrieval baselines
//!
//! Tests, *without an LLM*, the mechanism behind the research hypothesis:
//!
//! > An agent cannot solve a task whose required causal context is absent from
//! > its window. Does **regional causal memory** place that context in the
//! > window on long, multi-file tasks better than relevance-ranked retrieval —
//! > and does it stay robust when the *query is lexically misleading*?
//!
//! This is a **simulation under an explicit, falsifiable oracle**, not an LLM
//! evaluation (see the paper §9 for the proposed real-LLM study).
//!
//! ## What each strategy is given
//!
//! Real retrieval pipelines (RAG, GraphRAG) locate code from a **text query**
//! and are therefore vulnerable to lexical *decoys*. CCOS, an OS-style memory,
//! instead **anchors on the workspace signal** — the file/region currently being
//! edited or implicated by a failing test — which is structural, not lexical.
//! We model this explicitly: every task carries a `query` (a bag of tokens, which
//! in the *noisy* scenario is engineered so a decoy in an unrelated subsystem is
//! the best lexical match) and an `anchor` (the active file node).
//!
//! - `rag-dense`      — top-budget nodes by query similarity (classical RAG);
//! - `rag-hybrid`     — query similarity blended with the causal score;
//! - `graphrag-1hop`  — best lexical hit + its 1-hop neighbours;
//! - `graphrag-bfs`   — unbounded graph expansion from the best lexical hit;
//! - `ccos-from-query`— CCOS region of the best **lexical** hit (ablation: CCOS
//!   that trusts the query);
//! - `ccos-region`    — CCOS region of the **anchor** (the workspace signal).
//!
//! Two scenarios are run: **clean** (query points at the target) and **noisy**
//! (a trap decoy out-scores the target lexically). The success oracle is
//! `required causal set ⊆ window`. Everything is seeded and deterministic.

use crate::context_region::file_of;
use crate::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use crate::region_engine::ContextRegionEngine;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::{BTreeSet, VecDeque};

/// Token cost per selected node (matches the rest of CCOS).
const TOKENS_PER_NODE: usize = 128;

const STRATEGIES: [&str; 6] = [
    "rag-dense",
    "rag-hybrid",
    "graphrag-1hop",
    "graphrag-bfs",
    "ccos-from-query",
    "ccos-region",
];

/// Configuration for one experiment scenario.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// RNG seed (determinism).
    pub seed: u64,
    /// Number of tasks to sample.
    pub tasks: usize,
    /// Node budget of the context window (tokens = budget × 128).
    pub budget: usize,
    /// Independent subsystems (modular causal clusters).
    pub subsystems: usize,
    /// Files per subsystem (also the causal-chain length).
    pub files_per_subsystem: usize,
    /// Decoy (high-score, off-topic) symbols per file.
    pub decoys_per_file: usize,
    /// Task diameters to sweep.
    pub diameters: Vec<u32>,
    /// When true, the query is engineered so a decoy in an unrelated subsystem
    /// is the best lexical match (the "trap").
    pub noisy: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        ExperimentConfig {
            seed: 42,
            tasks: 400,
            budget: 30,
            subsystems: 8,
            files_per_subsystem: 9,
            decoys_per_file: 1,
            diameters: vec![1, 2, 3, 4],
            noisy: false,
        }
    }
}

/// Aggregated outcome for one strategy at one diameter (or overall).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StrategyResult {
    pub strategy: String,
    pub tasks: usize,
    pub successes: usize,
    pub success_rate: f32,
    pub mean_coverage: f32,
    pub mean_tokens: f32,
}

/// Full experiment report for one scenario.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentReport {
    pub seed: u64,
    pub budget_tokens: usize,
    pub n_tasks: usize,
    pub noisy: bool,
    /// `(diameter, [results per strategy])`.
    pub per_diameter: Vec<(u32, Vec<StrategyResult>)>,
    pub overall: Vec<StrategyResult>,
}

struct Synth {
    graph: MemoryGraph,
    /// Lexical tokens per node id (file token + a unique random name token).
    tokens: std::collections::HashMap<String, BTreeSet<String>>,
    /// Chains, one per subsystem (ordered symbol ids).
    chains: Vec<Vec<String>>,
    /// Decoy node ids, in generation order (subsystem-major).
    decoys: Vec<String>,
}

/// The subsystem key of a node id (`s{N}` from `…s{N}_f{j}…`).
fn subsystem_of(id: &str) -> String {
    file_of(id).split('_').next().unwrap_or("").to_string()
}

/// Generate a modular synthetic repository: independent subsystems, each its own
/// cross-file causal chain plus high-score decoys, with NO edges between
/// subsystems. Causal structure is decoupled from lexical structure.
fn generate_repo(cfg: &ExperimentConfig, rng: &mut StdRng) -> Synth {
    let mut graph = MemoryGraph::new(0.0, usize::MAX);
    let mut tokens: std::collections::HashMap<String, BTreeSet<String>> = Default::default();
    let add = |g: &mut MemoryGraph,
               toks: &mut std::collections::HashMap<String, BTreeSet<String>>,
               id: &str,
               file_tok: &str,
               name_tok: &str| {
        g.upsert_node(id.into(), id.into(), String::new(), NodeType::Symbol);
        let mut t = BTreeSet::new();
        t.insert(file_tok.to_string());
        t.insert(name_tok.to_string());
        toks.insert(id.to_string(), t);
    };
    let rand_tok = |rng: &mut StdRng| -> String {
        let n: u32 = rng.gen();
        format!("t{n:08x}")
    };

    let mut chains: Vec<Vec<String>> = Vec::new();
    let mut decoys: Vec<String> = Vec::new();
    for s in 0..cfg.subsystems {
        let l = cfg.files_per_subsystem;
        for j in 0..l {
            let ftok = format!("s{s}_f{j}");
            add(
                &mut graph,
                &mut tokens,
                &format!("file:{ftok}.rs"),
                &ftok,
                &ftok,
            );
        }
        let mut chain: Vec<String> = Vec::new();
        for j in 0..l {
            let ftok = format!("s{s}_f{j}");
            let name = rand_tok(rng);
            let id = format!("sym:s{s}_f{j}.rs:{name}");
            add(&mut graph, &mut tokens, &id, &ftok, &name);
            graph.add_edge(
                format!("file:{ftok}.rs").into(),
                id.clone().into(),
                0.6,
                EdgeType::Contains,
            );
            if let Some(prev) = chain.last() {
                graph.add_edge(
                    prev.clone().into(),
                    id.clone().into(),
                    0.9,
                    EdgeType::DependsOn,
                );
            }
            chain.push(id);
        }
        chains.push(chain);
        for j in 0..l {
            let ftok = format!("s{s}_f{j}");
            for d in 0..cfg.decoys_per_file {
                let name = rand_tok(rng);
                let id = format!("sym:s{s}_f{j}.rs:decoy_{d}_{name}");
                add(&mut graph, &mut tokens, &id, &ftok, &name);
                graph.add_edge(
                    format!("file:{ftok}.rs").into(),
                    id.clone().into(),
                    0.6,
                    EdgeType::Contains,
                );
                for _ in 0..40 {
                    graph.upsert_node(
                        id.clone().into(),
                        id.clone(),
                        String::new(),
                        NodeType::Symbol,
                    );
                }
                decoys.push(id);
            }
        }
    }

    Synth {
        graph,
        tokens,
        chains,
        decoys,
    }
}

struct Task {
    required: BTreeSet<String>,
    diameter: u32,
    /// The lexical query (token bag) the RAG/GraphRAG strategies see.
    query: BTreeSet<String>,
    /// The structural anchor CCOS activates from (the active file node).
    anchor: String,
}

/// Draw tasks. In the noisy scenario the query is built so a decoy in a
/// *different* subsystem out-scores the target lexically (a deterministic trap,
/// so the same tasks are used in both scenarios).
fn generate_tasks(synth: &Synth, cfg: &ExperimentConfig, rng: &mut StdRng) -> Vec<Task> {
    let mut tasks = Vec::new();
    let empty = BTreeSet::new();
    for _ in 0..cfg.tasks {
        let chain = &synth.chains[rng.gen_range(0..synth.chains.len())];
        let requested = cfg.diameters[rng.gen_range(0..cfg.diameters.len())];
        let max_r = ((chain.len() - 1) / 2) as u32;
        let rr = requested.min(max_r).max(1) as usize;
        let i = rng.gen_range(rr..=chain.len() - 1 - rr);
        let target = chain[i].clone();
        let required: BTreeSet<String> = chain[i - rr..=i + rr].iter().cloned().collect();

        let target_toks = synth.tokens.get(&target).unwrap_or(&empty);
        let name_tok: String = target_toks
            .iter()
            .find(|t| !t.starts_with('s') || !t.contains("_f"))
            .cloned()
            .unwrap_or_default();
        let anchor = format!("file:{}", file_of(&target));

        let query = if !cfg.noisy {
            target_toks.clone()
        } else {
            // Trap: the first decoy from a different subsystem out-scores the
            // target by sharing both of the query's non-target tokens.
            let tsub = subsystem_of(&target);
            let trap = synth
                .decoys
                .iter()
                .find(|d| subsystem_of(d) != tsub)
                .cloned()
                .unwrap_or_default();
            let mut q = BTreeSet::new();
            q.insert(name_tok); // the user names the target symbol…
            if let Some(tt) = synth.tokens.get(&trap) {
                q.extend(tt.iter().cloned()); // …but a distractor term collides with a decoy
            }
            q
        };

        tasks.push(Task {
            required,
            diameter: rr as u32,
            query,
            anchor,
        });
    }
    tasks
}

/// Jaccard similarity between a node's tokens and the query token bag.
fn sim(synth: &Synth, node: &str, query: &BTreeSet<String>) -> f32 {
    let empty = BTreeSet::new();
    let a = synth.tokens.get(node).unwrap_or(&empty);
    let inter = a.intersection(query).count();
    let union = a.union(query).count();
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

/// The node most similar to the query (ties broken by smallest id).
fn best_lexical_hit(synth: &Synth, query: &BTreeSet<String>) -> String {
    synth
        .graph
        .nodes
        .keys()
        .map(|k| (sim(synth, &k.0, query), k.0.clone()))
        .max_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.cmp(&a.1))
        })
        .map(|(_, id)| id)
        .unwrap_or_default()
}

fn region_of(g: &MemoryGraph, entry: &str, budget: usize) -> BTreeSet<String> {
    let mut engine = ContextRegionEngine::new();
    let mut sink = crate::event_log::EventLog::new("exp".into());
    engine.initialize_regions(g, &mut sink);
    let Some(rid) = engine.region_of(entry) else {
        return BTreeSet::new();
    };
    let mut members: Vec<String> = engine.regions[&rid].members.clone();
    members.sort_by(|a, b| {
        let sa = g
            .nodes
            .get(&NodeId(a.clone()))
            .map(|n| g.compute_node_score(n))
            .unwrap_or(0.0);
        let sb = g
            .nodes
            .get(&NodeId(b.clone()))
            .map(|n| g.compute_node_score(n))
            .unwrap_or(0.0);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    members.truncate(budget);
    members.into_iter().collect()
}

/// Select up to `budget` node ids under one strategy.
fn select(strategy: &str, synth: &Synth, task: &Task, budget: usize) -> BTreeSet<String> {
    let g = &synth.graph;
    match strategy {
        "rag-dense" => rank_take(g, budget, |id| {
            (sim(synth, id, &task.query) as f64, id.to_string())
        }),
        "rag-hybrid" => rank_take(g, budget, |id| {
            let s = sim(synth, id, &task.query) as f64;
            let score = g
                .nodes
                .get(&NodeId(id.to_string()))
                .map(|n| g.compute_node_score(n))
                .unwrap_or(0.0);
            (0.5 * s + 0.5 * score, id.to_string())
        }),
        "graphrag-1hop" => {
            let seed = best_lexical_hit(synth, &task.query);
            let mut sel: BTreeSet<String> = BTreeSet::new();
            sel.insert(seed.clone());
            for e in &g.edges {
                if e.source.0 == seed {
                    sel.insert(e.target.0.clone());
                } else if e.target.0 == seed {
                    sel.insert(e.source.0.clone());
                }
            }
            if sel.len() > budget {
                let mut v: Vec<String> = sel.into_iter().collect();
                v.sort_by(|a, b| {
                    sim(synth, b, &task.query)
                        .partial_cmp(&sim(synth, a, &task.query))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(b))
                });
                v.truncate(budget);
                v.into_iter().collect()
            } else {
                let fill = rank_take(g, budget, |id| {
                    (sim(synth, id, &task.query) as f64, id.to_string())
                });
                for id in fill {
                    if sel.len() >= budget {
                        break;
                    }
                    sel.insert(id);
                }
                sel
            }
        }
        "graphrag-bfs" => bfs_from(g, &best_lexical_hit(synth, &task.query), budget),
        "ccos-from-query" => region_of(g, &best_lexical_hit(synth, &task.query), budget),
        "ccos-region" => region_of(g, &task.anchor, budget),
        _ => BTreeSet::new(),
    }
}

/// Unbounded BFS from `seed` over undirected edges, taking nodes in BFS order
/// until `budget` is reached.
fn bfs_from(g: &MemoryGraph, seed: &str, budget: usize) -> BTreeSet<String> {
    let mut sel: BTreeSet<String> = BTreeSet::new();
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    visited.insert(seed.to_string());
    queue.push_back(seed.to_string());
    while let Some(n) = queue.pop_front() {
        if sel.len() >= budget {
            break;
        }
        sel.insert(n.clone());
        let mut nbs: Vec<String> = Vec::new();
        for e in &g.edges {
            if e.source.0 == n {
                nbs.push(e.target.0.clone());
            } else if e.target.0 == n {
                nbs.push(e.source.0.clone());
            }
        }
        nbs.sort();
        for nb in nbs {
            if visited.insert(nb.clone()) {
                queue.push_back(nb);
            }
        }
    }
    sel
}

/// Rank all nodes by a `(score, tiebreak)` key (descending, then id) and take
/// the top `budget`.
fn rank_take<F>(g: &MemoryGraph, budget: usize, key: F) -> BTreeSet<String>
where
    F: Fn(&str) -> (f64, String),
{
    let mut scored: Vec<(f64, String)> = g.nodes.keys().map(|k| key(&k.0)).collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    scored.into_iter().take(budget).map(|(_, id)| id).collect()
}

/// Failure-propagated copy of the graph (CCOS's causal signal), per task.
fn with_failure(graph: &MemoryGraph, target: &str, depth: u32) -> MemoryGraph {
    let mut g = graph.clone();
    g.set_failure_relevance(&NodeId(target.to_string()), 0.95);
    g.propagate_failure(&NodeId(target.to_string()), 0, depth);
    g
}

/// Run one scenario (clean or noisy, per `cfg.noisy`).
pub fn run_experiment(cfg: &ExperimentConfig) -> ExperimentReport {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let base = generate_repo(cfg, &mut rng);
    let tasks = generate_tasks(&base, cfg, &mut rng);

    // strategy → diameter → (successes, coverage_sum, token_sum, n)
    let mut acc: std::collections::HashMap<(String, u32), (usize, f32, f32, usize)> =
        Default::default();
    for task in &tasks {
        // The anchor target for failure propagation = the node behind the anchor
        // file (its chain symbol); approximate via the required set's centre.
        let centre = task.required.iter().next().cloned().unwrap_or_default();
        let g = with_failure(&base.graph, &centre, task.diameter + 1);
        let synth = Synth {
            graph: g,
            tokens: base.tokens.clone(),
            chains: base.chains.clone(),
            decoys: base.decoys.clone(),
        };
        for strat in STRATEGIES {
            let sel = select(strat, &synth, task, cfg.budget);
            let hit = task.required.intersection(&sel).count();
            let coverage = hit as f32 / task.required.len() as f32;
            let success = usize::from(hit == task.required.len());
            let e = acc
                .entry((strat.to_string(), task.diameter))
                .or_insert((0, 0.0, 0.0, 0));
            e.0 += success;
            e.1 += coverage;
            e.2 += (sel.len() * TOKENS_PER_NODE) as f32;
            e.3 += 1;
        }
    }

    let summarise = |succ: usize, cov: f32, tok: f32, n: usize| StrategyResult {
        strategy: String::new(),
        tasks: n,
        successes: succ,
        success_rate: if n == 0 { 0.0 } else { succ as f32 / n as f32 },
        mean_coverage: if n == 0 { 0.0 } else { cov / n as f32 },
        mean_tokens: if n == 0 { 0.0 } else { tok / n as f32 },
    };

    let mut per_diameter = Vec::new();
    for &d in &cfg.diameters {
        let mut row = Vec::new();
        for strat in STRATEGIES {
            if let Some((s, c, t, n)) = acc.get(&(strat.to_string(), d)) {
                if *n > 0 {
                    let mut r = summarise(*s, *c, *t, *n);
                    r.strategy = strat.to_string();
                    row.push(r);
                }
            }
        }
        if !row.is_empty() {
            per_diameter.push((d, row));
        }
    }

    let mut overall = Vec::new();
    for strat in STRATEGIES {
        let (mut s, mut c, mut t, mut n) = (0usize, 0.0f32, 0.0f32, 0usize);
        for &d in &cfg.diameters {
            if let Some((ss, cc, tt, nn)) = acc.get(&(strat.to_string(), d)) {
                s += ss;
                c += cc;
                t += tt;
                n += nn;
            }
        }
        if n > 0 {
            let mut r = summarise(s, c, t, n);
            r.strategy = strat.to_string();
            overall.push(r);
        }
    }

    ExperimentReport {
        seed: cfg.seed,
        budget_tokens: cfg.budget * TOKENS_PER_NODE,
        n_tasks: tasks.len(),
        noisy: cfg.noisy,
        per_diameter,
        overall,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_is_deterministic() {
        let cfg = ExperimentConfig {
            tasks: 100,
            ..Default::default()
        };
        assert_eq!(run_experiment(&cfg).overall, run_experiment(&cfg).overall);
    }

    #[test]
    fn all_strategies_are_evaluated() {
        let report = run_experiment(&ExperimentConfig {
            tasks: 80,
            ..Default::default()
        });
        assert_eq!(report.overall.len(), 6);
    }

    fn rate(report: &ExperimentReport, strat: &str) -> f32 {
        report
            .overall
            .iter()
            .find(|r| r.strategy == strat)
            .unwrap()
            .success_rate
    }

    #[test]
    fn clean_scenario_region_and_strong_graph_succeed() {
        let r = run_experiment(&ExperimentConfig {
            tasks: 300,
            noisy: false,
            ..Default::default()
        });
        assert!(
            rate(&r, "ccos-region") > 0.95,
            "region must solve clean tasks"
        );
        assert!(
            rate(&r, "graphrag-bfs") > 0.95,
            "strong graph baseline ties on clean tasks"
        );
        assert!(
            rate(&r, "rag-dense") < 0.10,
            "lexical RAG fails on cross-file tasks"
        );
    }

    #[test]
    fn noisy_scenario_only_anchored_region_survives() {
        let r = run_experiment(&ExperimentConfig {
            tasks: 300,
            noisy: true,
            ..Default::default()
        });
        // The anchored region ignores the misleading query → still solves tasks.
        assert!(
            rate(&r, "ccos-region") > 0.95,
            "anchored region must survive query noise"
        );
        // Everything that trusts the lexical query is fooled by the trap.
        assert!(
            rate(&r, "graphrag-bfs") < 0.20,
            "graph BFS seeded on a decoy fails"
        );
        assert!(
            rate(&r, "ccos-from-query") < 0.20,
            "CCOS that trusts the query is fooled too — the differentiator is the anchor"
        );
    }
}
