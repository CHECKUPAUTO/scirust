//! # Post-mortem — an interactive time-travel debugger for an agent session
//!
//! The [`agent_session`](crate::agent_session) module makes a cognitive timeline
//! *replayable*; this module makes it *walkable by hand*. Point it at a persisted
//! workspace (the `<workspace>.oplog` written by `ccos mcp`) — or the built-in
//! [`demo_session`] of a session that drifts — and step through the agent's
//! history: move a cursor across the timeline, inspect the recalled context window
//! at any past step, and diff two points to see how the working set drifted as
//! failures accumulated.
//!
//! It is a thin REPL over [`AgentSession::replay_to`] /
//! [`AgentSession::recall_what_if`]: every command reconstructs state
//! deterministically from the op-log, so the post-mortem is exact and side-effect
//! free (it never mutates the session).
//!
//! Run with `ccos postmortem [workspace.ccos]`. Commands: `timeline`, `goto N`,
//! `next`/`prev`, `recall [budget]`, `around <anchor> [budget]`, `task <text…>`,
//! `diff A B`, `stats`, `help`, `quit`.

use crate::agent_session::AgentSession;
use crate::external_memory::{ExternalMemory, Recall, RecallWindow};
use crate::memory::{MemoryGraph, NodeId};
use chrono::Utc;

/// What a [`Debugger::command`] decides the REPL should do next.
pub enum Outcome {
    /// Print this text (may be empty — print nothing) and keep going.
    Print(String),
    /// Leave the debugger.
    Quit,
}

const HELP: &str = "\
commands:
  timeline | tl            the cognitive journal (▸ marks the cursor)
  goto N   | g N           move the cursor to step N (time-travel position)
  next | n   /  prev | p    move the cursor one step
  recall [budget] | r      working-set window as of the cursor
  around <anchor> [budget] | a   region window anchored on a node/file, at the cursor
  task <text…> | t         free-text recall at the cursor
  diff A B | d             which files entered/left the working set between A and B
  energy A B | e           node-level Δscore + failure-pressure between A and B
  missing <node> [budget] | m   the step a node is first evicted from the budgeted window
  cause <node> | c         which recorded op moved a node's score most (drift attribution)
  impact <node> | i        do(X): the nodes a change to X would force (its dependents)
  blame <node> | bl        the candidate causes of X — what it depends on (dual of impact)
  stats | s                memory counts at the cursor
  help | h | ?             this help
  quit | q                 leave";

/// An interactive walk over an [`AgentSession`]'s recorded timeline.
pub struct Debugger {
    session: AgentSession,
    /// Current time-travel position (logical step `0..=len`).
    cursor: usize,
}

impl Debugger {
    /// Open a debugger positioned at the end of the timeline ("now").
    pub fn new(session: AgentSession) -> Self {
        let cursor = session.len();
        Debugger { session, cursor }
    }

    /// The current cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Parse and run one REPL command line.
    pub fn command(&mut self, line: &str) -> Outcome {
        let mut it = line.split_whitespace();
        let Some(cmd) = it.next() else {
            return Outcome::Print(String::new());
        };
        let rest: Vec<&str> = it.collect();
        let num = |s: &str| s.parse::<usize>().ok();
        let len = self.session.len();

        match cmd {
            "quit" | "q" | "exit" => Outcome::Quit,
            "help" | "h" | "?" => Outcome::Print(HELP.to_string()),
            "timeline" | "tl" => Outcome::Print(self.render_timeline()),
            "len" => Outcome::Print(format!("{len} steps")),
            "stats" | "s" => Outcome::Print(self.render_at_cursor()),
            "goto" | "g" => match rest.first().and_then(|s| num(s)) {
                Some(n) => {
                    self.cursor = n.min(len);
                    Outcome::Print(self.render_at_cursor())
                }
                None => Outcome::Print("usage: goto <step>".to_string()),
            },
            "next" | "n" => {
                self.cursor = (self.cursor + 1).min(len);
                Outcome::Print(self.render_at_cursor())
            }
            "prev" | "p" => {
                self.cursor = self.cursor.saturating_sub(1);
                Outcome::Print(self.render_at_cursor())
            }
            "recall" | "r" => {
                let budget = rest.first().and_then(|s| num(s)).unwrap_or(2048);
                Outcome::Print(self.render_recall(Recall::working_set(), budget))
            }
            "around" | "a" => match rest.first() {
                Some(anchor) => {
                    let budget = rest.get(1).and_then(|s| num(s)).unwrap_or(2048);
                    Outcome::Print(self.render_recall(Recall::around(*anchor), budget))
                }
                None => Outcome::Print("usage: around <anchor> [budget]".to_string()),
            },
            "task" | "t" => {
                if rest.is_empty() {
                    Outcome::Print("usage: task <text…>".to_string())
                } else {
                    Outcome::Print(self.render_recall(Recall::task(rest.join(" ")), 2048))
                }
            }
            "diff" | "d" => {
                match (
                    rest.first().and_then(|s| num(s)),
                    rest.get(1).and_then(|s| num(s)),
                ) {
                    (Some(a), Some(b)) => Outcome::Print(self.render_diff(a, b)),
                    _ => Outcome::Print("usage: diff <step-a> <step-b>".to_string()),
                }
            }
            "energy" | "pressure" | "e" => {
                match (
                    rest.first().and_then(|s| num(s)),
                    rest.get(1).and_then(|s| num(s)),
                ) {
                    (Some(a), Some(b)) => Outcome::Print(self.render_energy(a, b)),
                    _ => Outcome::Print("usage: energy <step-a> <step-b>".to_string()),
                }
            }
            "missing" | "m" => match rest.first() {
                Some(node) => {
                    let budget = rest.get(1).and_then(|s| num(s)).unwrap_or(2048);
                    Outcome::Print(self.render_missing(node, budget))
                }
                None => Outcome::Print("usage: missing <node> [budget]".to_string()),
            },
            "cause" | "c" => match rest.first() {
                Some(node) => Outcome::Print(self.render_cause(node)),
                None => Outcome::Print("usage: cause <node>".to_string()),
            },
            "impact" | "i" => match rest.first() {
                Some(node) => Outcome::Print(self.render_impact(node)),
                None => Outcome::Print("usage: impact <node>".to_string()),
            },
            "blame" | "bl" => match rest.first() {
                Some(node) => Outcome::Print(self.render_blame(node)),
                None => Outcome::Print("usage: blame <node>".to_string()),
            },
            other => Outcome::Print(format!("unknown command '{other}' (try 'help')")),
        }
    }

    /// The journal, with a marker on the line at the cursor.
    fn render_timeline(&self) -> String {
        let mut out = String::new();
        for line in self.session.timeline() {
            // Lines read "t=<n>  <op>"; mark the one at the cursor.
            let at_cursor = line
                .strip_prefix("t=")
                .and_then(|r| r.split_whitespace().next())
                .and_then(|n| n.parse::<usize>().ok())
                == Some(self.cursor);
            out.push_str(if at_cursor { "▸ " } else { "  " });
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(&format!(
            "(cursor at step {} of {})",
            self.cursor,
            self.session.len()
        ));
        out
    }

    /// Memory counts at the cursor.
    fn render_at_cursor(&self) -> String {
        let st = self.session.replay_to(self.cursor).stats();
        format!(
            "cursor → t={}/{}   nodes={} edges={} files={}",
            self.cursor,
            self.session.len(),
            st.nodes,
            st.edges,
            st.files
        )
    }

    /// A recalled window as of the cursor (a time-travel what-if).
    fn render_recall(&self, recall: Recall, budget: usize) -> String {
        let win = self.session.recall_what_if(self.cursor, &recall, budget);
        let mut out = format!(
            "t={}/{}  {}  ({} items, ~{} tokens)\n",
            self.cursor,
            self.session.len(),
            win.strategy,
            win.items.len(),
            win.tokens
        );
        for it in win.items.iter().take(12) {
            out.push_str(&format!(
                "  {:.3}  {:<8}  {}\n",
                it.score,
                trunc(&it.kind, 8),
                it.uri
            ));
        }
        out.pop(); // drop the trailing newline
        out
    }

    /// How the working set drifted between two steps (files that entered / left).
    fn render_diff(&self, a: usize, b: usize) -> String {
        let files = |step: usize| -> std::collections::BTreeSet<String> {
            window_files(
                &self
                    .session
                    .recall_what_if(step, &Recall::working_set(), 8192),
            )
        };
        let (fa, fb) = (files(a), files(b));
        let entered: Vec<&String> = fb.difference(&fa).collect();
        let left: Vec<&String> = fa.difference(&fb).collect();
        let fmt = |v: &[&String]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            }
        };
        format!(
            "working set drift  t={a} → t={b}\n  entered: {}\n  left:    {}",
            fmt(&entered),
            fmt(&left)
        )
    }

    /// Node-level **energy/pressure drift**: how each AST node's causal score (and
    /// failure pressure) moved between two steps — the migration of "heat" through
    /// the graph as failures propagate. Sorted by the magnitude of the score delta.
    fn render_energy(&self, a: usize, b: usize) -> String {
        let (ma, mb) = (self.session.replay_to(a), self.session.replay_to(b));
        let (ga, gb) = (ma.graph(), mb.graph());
        let score = |g: &MemoryGraph, id: &NodeId| {
            g.nodes
                .get(id)
                .map(|n| g.compute_node_score(n))
                .unwrap_or(0.0)
        };
        let fail = |g: &MemoryGraph, id: &NodeId| {
            g.nodes.get(id).map(|n| n.failure_relevance).unwrap_or(0.0)
        };

        let mut ids: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
        ids.extend(ga.nodes.keys().cloned());
        ids.extend(gb.nodes.keys().cloned());

        let mut rows: Vec<(f64, String)> = ids
            .iter()
            .map(|id| {
                let d = score(gb, id) - score(ga, id);
                (
                    d,
                    format!(
                        "{:+.3}  {:<30}  fail {:.2}→{:.2}",
                        d,
                        trunc(&id.0, 30),
                        fail(ga, id),
                        fail(gb, id)
                    ),
                )
            })
            .filter(|(d, _)| d.abs() > 1e-9)
            .collect();
        rows.sort_by(|x, y| {
            y.0.abs()
                .partial_cmp(&x.0.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut out = format!("node energy/pressure drift  t={a} → t={b}  (Δscore, top movers)\n");
        if rows.is_empty() {
            out.push_str("  (no change)");
            return out;
        }
        for (_, line) in rows.iter().take(12) {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
        out.pop();
        out
    }

    /// **Eviction watchpoint**: scan the timeline for the first step where `node`
    /// is in the graph but has dropped out of the budgeted working-set window — the
    /// moment the budget squeezed it out of context. Composes with compaction: if
    /// the eviction lies in folded history, it says so rather than guessing.
    fn render_missing(&self, node: &str, budget: usize) -> String {
        let target = normalize_id(node);
        let id = NodeId(target.clone());
        let (floor, len) = (self.session.floor(), self.session.len());

        let mut first_in_graph: Option<usize> = None;
        let mut first_missing: Option<usize> = None;
        let mut strip = String::new();
        for step in floor..=len {
            let mem = self.session.replay_to(step);
            let in_graph = mem.graph().nodes.contains_key(&id);
            let in_window = in_graph
                && mem
                    .recall(&Recall::working_set(), budget)
                    .items
                    .iter()
                    .any(|i| i.uri == target);
            if in_graph && first_in_graph.is_none() {
                first_in_graph = Some(step);
            }
            if in_graph && !in_window && first_missing.is_none() {
                first_missing = Some(step);
            }
            strip.push(if !in_graph {
                '·'
            } else if in_window {
                '●'
            } else {
                '○'
            });
        }

        let mut out = format!("MISSING scan: {target}   (budget {budget} tokens, working-set)\n");
        match first_in_graph {
            None => {
                out.push_str("  node never appears in the timeline");
                return out;
            }
            Some(j) if j == floor && floor > 0 => {
                out.push_str(&format!(
                    "  in graph as of the compaction floor t={floor} (earlier history folded in)\n"
                ));
            }
            Some(j) => out.push_str(&format!("  in graph since t={j}\n")),
        }

        match first_missing {
            None => {
                out.push_str("  ✓ never MISSING — stays in every budgeted window once ingested\n")
            }
            Some(k) if k == floor && floor > 0 => {
                out.push_str(&format!(
                    "  ⚠ already MISSING at the compaction floor (t≤{floor}) — the eviction is in \
                     folded history; only the baseline state survives below the floor\n"
                ));
            }
            Some(k) => {
                out.push_str(&format!("  ⚠ first MISSING at t={k}\n"));
                if let Some(op) = self.op_at(k) {
                    out.push_str(&format!("     trigger: {op}\n"));
                }
                out.push_str(&self.eviction_detail(k, &target, budget));
            }
        }

        out.push_str(&format!("  t{floor}..t{len}:  {}\n", strip_view(&strip)));
        out.push_str("  legend: ● in window   ○ in graph but MISSING   · not yet ingested");
        out
    }

    /// The rank/token detail at the eviction step: how far the node was outranked
    /// and how many tokens short of the budget it fell.
    fn eviction_detail(&self, step: usize, target: &str, budget: usize) -> String {
        let mem = self.session.replay_to(step);
        let kept = mem.recall(&Recall::working_set(), budget);
        let full = mem.recall(&Recall::working_set(), usize::MAX);
        let rank = full
            .items
            .iter()
            .position(|i| i.uri == target)
            .map(|p| p + 1);
        // Tokens needed to reach the node in score order (chars/4, as recall counts).
        let mut needed = 0usize;
        for it in &full.items {
            needed += it.content.chars().count() / 4;
            if it.uri == target {
                break;
            }
        }
        let gap = needed.saturating_sub(budget);
        format!(
            "     ranked #{}/{}; the window kept {} nodes (~{}/{} tok); the node sat at ~{} tok — short by ~{}\n",
            rank.map(|r| r.to_string()).unwrap_or_else(|| "?".into()),
            full.items.len(),
            kept.items.len(),
            kept.tokens,
            budget,
            needed,
            gap
        )
    }

    /// Attribute a node's causal-score drift to the recorded operation that moved it most — the
    /// CUSUM change-point over its replayed trajectory ([`AgentSession::attribute_drift`]). This is
    /// the auditable "which op did this" that the `missing` watchpoint's raw trigger line only
    /// guessed at: a tested change-point, not the op that merely happened at the eviction step.
    fn render_cause(&self, node: &str) -> String {
        let id = normalize_id(node);
        match self.session.attribute_drift(&id) {
            Some(c) => format!(
                "drift attribution — {id}\n  {} {:+.3} at step {}\n  cause: {}\n  (CUSUM {:.3})",
                if c.delta >= 0.0 {
                    "score rose"
                } else {
                    "score fell"
                },
                c.delta,
                c.step,
                c.op,
                c.cusum,
            ),
            None => format!(
                "{id}: no attributable drift (flat trajectory, or the break is below the compaction floor)"
            ),
        }
    }

    /// `do(X)` at the cursor: the nodes a change to `node` would structurally force — its dependents,
    /// ranked by attenuated impact ([`MemoryGraph::intervene`]). The interventional counterpart of the
    /// `missing`/`cause` views: not "what is near X" but "what X *forces*". A probabilistic retriever
    /// has no do-operator to answer this.
    fn render_impact(&self, node: &str) -> String {
        let id = normalize_id(node);
        let mem = self.session.replay_to(self.cursor);
        let impact = mem.graph().intervene(&NodeId(id.clone()), 1.0, 0.75, 4);
        if impact.is_empty() {
            return format!(
                "do({id}) at t={}: nothing depends on it — no downstream impact",
                self.cursor
            );
        }
        let shown = impact.len().min(15);
        let mut out = format!(
            "do({id}) at t={} — nodes a change would force (impact, top {shown} of {}):\n",
            self.cursor,
            impact.len()
        );
        for (n, v) in impact.iter().take(shown) {
            out.push_str(&format!("  {v:>6.3}  {}\n", n.0));
        }
        out
    }

    /// `blame(X)` at the cursor: the dependencies of `node` — what it relies on, the candidate root
    /// causes of a failure in it ([`MemoryGraph::blame`]). The dual of `impact`, and the principled
    /// form of "the culprit is in a different file": an upstream, low-similarity dependency.
    fn render_blame(&self, node: &str) -> String {
        let id = normalize_id(node);
        let mem = self.session.replay_to(self.cursor);
        let causes = mem.graph().blame(&NodeId(id.clone()), 0.75, 4);
        if causes.is_empty() {
            return format!(
                "blame({id}) at t={}: it depends on nothing — no candidate cause",
                self.cursor
            );
        }
        let shown = causes.len().min(15);
        let mut out = format!(
            "blame({id}) at t={} — candidate causes it depends on (weight, top {shown} of {}):\n",
            self.cursor,
            causes.len()
        );
        for (n, v) in causes.iter().take(shown) {
            out.push_str(&format!("  {v:>6.3}  {}\n", n.0));
        }
        out
    }

    /// The journal line for a given logical step (e.g. `t=6  SignalFailure(…)`).
    fn op_at(&self, step: usize) -> Option<String> {
        self.session.timeline().into_iter().find(|l| {
            l.strip_prefix("t=")
                .and_then(|r| r.split_whitespace().next())
                .and_then(|n| n.parse::<usize>().ok())
                == Some(step)
        })
    }
}

/// Prefix a bare path with `file:`; leave known node-id prefixes untouched.
fn normalize_id(s: &str) -> String {
    const PREFIXES: [&str; 5] = ["file:", "sym:", "mod:", "use:", "dep:"];
    if PREFIXES.iter().any(|p| s.starts_with(p)) {
        s.to_string()
    } else {
        format!("file:{s}")
    }
}

/// Show a status strip, keeping the last 72 steps if it is longer.
fn strip_view(s: &str) -> String {
    let n = s.chars().count();
    if n <= 72 {
        s.to_string()
    } else {
        format!("…{}", s.chars().skip(n - 72).collect::<String>())
    }
}

/// The `file:` nodes of a window.
fn window_files(win: &RecallWindow) -> std::collections::BTreeSet<String> {
    win.items
        .iter()
        .filter(|i| i.uri.starts_with("file:"))
        .map(|i| i.uri.clone())
        .collect()
}

/// Truncate for column display.
fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

/// A built-in session that **drifts**: a small import chain plus an unrelated file,
/// then a failure on the entrypoint and a page-fault on the deep cause — so the
/// working set visibly migrates toward `db.rs` as the post-mortem walks forward.
pub fn demo_session() -> AgentSession {
    let mut s = AgentSession::new();
    s.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
    s.ingest(
        "src/repo.rs",
        "use crate::db;\npub fn fetch() -> i64 { db::timeout() }\n",
    );
    s.ingest(
        "src/api.rs",
        "use crate::repo;\npub fn handle() -> i64 { repo::fetch() }\n",
    );
    s.ingest(
        "src/util.rs",
        "pub fn fmt_date() -> &'static str { \"\" }\n",
    );
    s.recall(Recall::working_set(), 2048);
    // The drift begins: the entrypoint is failing…
    let _ = s.signal_failure("file:src/api.rs", 3);
    s.recall(Recall::around("file:src/api.rs"), 2048);
    // …and a panic points at the deep cause, pulling the hot set toward db.rs.
    let panic = "thread 'main' panicked at src/db.rs:1:14:\nattempt to add with overflow\n";
    s.page_fault(panic, 800);
    s.ingest(
        "src/api.rs",
        "use crate::repo;\npub fn handle() -> i64 { repo::fetch() + 1 }\n",
    );
    s
}

/// A one-shot, analytics-ready JSON snapshot of a session's **field record**:
/// stats, hash-chain integrity, the cognitive timeline, and the current working
/// set. The non-interactive counterpart to the REPL — run it on a cron or after a
/// field run to **archive** a session (each export captures the timeline *before*
/// the next compaction folds older steps into the baseline). The raw
/// `<workspace>.oplog` remains the structured source of truth for deeper analysis.
pub fn export(session: &AgentSession, workspace: &str, budget: usize) -> serde_json::Value {
    serde_json::json!({
        "ccos_version": env!("CARGO_PKG_VERSION"),
        "workspace": workspace,
        "generated_at": Utc::now().to_rfc3339(),
        "timeline_len": session.len(),
        "compaction_floor": session.floor(),
        "stats": session.memory().stats(),
        "integrity": session.memory().verify(),
        "timeline": session.timeline(),
        "working_set": session.memory().recall(&Recall::working_set(), budget),
    })
}

/// Run the interactive REPL until `quit`/EOF. Prompts and the banner go to stderr;
/// command output goes to stdout (so a piped session captures clean results).
pub fn serve(session: AgentSession) {
    use std::io::{BufRead, Write};
    const PROMPT: &str = "ccos⏪ ";
    let mut dbg = Debugger::new(session);
    eprintln!(
        "CCOS post-mortem — interactive time-travel debugger ({} steps). Type 'help'.",
        dbg.session.len()
    );
    eprint!("{PROMPT}");
    let _ = std::io::stderr().flush();

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        match dbg.command(&line) {
            Outcome::Quit => break,
            Outcome::Print(text) => {
                let mut out = stdout.lock();
                if !text.is_empty() {
                    let _ = writeln!(out, "{text}");
                }
                let _ = out.flush();
            }
        }
        eprint!("{PROMPT}");
        let _ = std::io::stderr().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(o: Outcome) -> String {
        match o {
            Outcome::Print(s) => s,
            Outcome::Quit => "<quit>".to_string(),
        }
    }

    #[test]
    fn demo_session_has_the_expected_length() {
        // 5 ingests/recalls + failure + recall + page_fault: a non-trivial timeline.
        assert!(demo_session().len() >= 8);
    }

    #[test]
    fn cursor_starts_at_now() {
        let d = Debugger::new(demo_session());
        assert_eq!(d.cursor(), d.session.len());
    }

    #[test]
    fn goto_then_recall_reflects_that_point_in_history() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 1")); // only db.rs ingested so far
        let r = out(d.command("recall 2000"));
        assert!(r.contains("file:src/db.rs"), "step-1 window has db.rs: {r}");
        assert!(
            !r.contains("file:src/api.rs"),
            "step-1 predates api.rs: {r}"
        );
    }

    #[test]
    fn diff_shows_the_working_set_drift() {
        let mut d = Debugger::new(demo_session());
        let end = d.session.len();
        let report = out(d.command(&format!("diff 1 {end}")));
        assert!(report.contains("entered:"));
        // By the end the import chain has joined the hot set.
        assert!(
            report.contains("file:src/api.rs") || report.contains("file:src/repo.rs"),
            "drift surfaces the dependents: {report}"
        );
    }

    #[test]
    fn energy_shows_pressure_rising_on_the_cause() {
        let mut d = Debugger::new(demo_session());
        let end = d.session.len();
        // Before the failure (t=4) vs after the page-fault (end): the deep cause
        // db.rs should gain energy and failure pressure.
        let report = out(d.command(&format!("energy 4 {end}")));
        assert!(
            report.contains("file:src/db.rs"),
            "db.rs surfaces as a mover: {report}"
        );
        assert!(
            report.contains("fail 0.00→"),
            "pressure column present: {report}"
        );
    }

    #[test]
    fn cause_attributes_drift_to_a_recorded_op() {
        let mut d = Debugger::new(demo_session());
        let report = out(d.command("cause src/db.rs"));
        assert!(
            report.contains("drift attribution") && report.contains("cause:"),
            "reports the culprit op: {report}"
        );
        assert!(report.contains("step"), "names the timeline step: {report}");
        // Bare `cause` prints usage; an unknown node reports no attributable drift.
        assert_eq!(out(d.command("cause")), "usage: cause <node>");
        assert!(out(d.command("cause src/ghost.rs")).contains("no attributable drift"));
    }

    #[test]
    fn impact_lists_the_dependents_of_a_node() {
        let mut d = Debugger::new(demo_session());
        let report = out(d.command("impact src/db.rs"));
        assert!(
            report.contains("do(file:src/db.rs)"),
            "framed as an intervention: {report}"
        );
        // repo imports db directly and api imports it transitively, so a change to db forces both.
        assert!(
            report.contains("file:src/repo.rs"),
            "the direct dependent surfaces: {report}"
        );
        assert!(
            report.contains("file:src/api.rs"),
            "the transitive dependent surfaces: {report}"
        );
        // Bare `impact` prints usage; a top-level node forces nothing downstream.
        assert_eq!(out(d.command("impact")), "usage: impact <node>");
        assert!(out(d.command("impact src/api.rs")).contains("no downstream impact"));
    }

    #[test]
    fn blame_lists_the_dependencies_of_a_node() {
        let mut d = Debugger::new(demo_session());
        // api depends on repo (directly) and db (transitively) — its candidate causes.
        let report = out(d.command("blame src/api.rs"));
        assert!(
            report.contains("blame(file:src/api.rs)"),
            "framed as blame: {report}"
        );
        assert!(
            report.contains("file:src/repo.rs") && report.contains("file:src/db.rs"),
            "dependencies surface as candidate causes: {report}"
        );
        // db is a leaf: it depends on nothing.
        assert!(out(d.command("blame src/db.rs")).contains("no candidate cause"));
        assert_eq!(out(d.command("blame")), "usage: blame <node>");
    }

    #[test]
    fn missing_detects_eviction_under_a_tight_budget() {
        let mut d = Debugger::new(demo_session());
        // budget 1 ⇒ only the single hottest node fits; the deep cause is not always
        // it (early on the scores are flat / dep:crate leads), so it is evicted.
        let r = out(d.command("missing src/db.rs 1"));
        assert!(r.contains("in graph since t=1"), "{r}");
        assert!(
            r.contains("first MISSING at t="),
            "tight budget evicts it: {r}"
        );
        assert!(r.contains("short by ~"), "reports the token gap: {r}");
    }

    #[test]
    fn missing_handles_generous_budget_and_unknown_node() {
        let mut d = Debugger::new(demo_session());
        // A budget large enough for the whole graph: never evicted.
        assert!(out(d.command("missing src/db.rs 100000")).contains("never MISSING"));
        // A node that was never ingested.
        assert!(out(d.command("missing nope.rs")).contains("never appears"));
    }

    #[test]
    fn missing_composes_with_the_compaction_floor() {
        let mut s = demo_session();
        s.compact(2); // fold all but the last 2 ops: floor = len - 2 > 0
        let mut d = Debugger::new(s);
        // db.rs was ingested at t=1, far below the floor — the scan must report it
        // relative to the floor rather than inventing a sub-floor step.
        let r = out(d.command("missing src/db.rs 1"));
        assert!(
            r.contains("compaction floor"),
            "stays honest about folded history: {r}"
        );
    }

    #[test]
    fn next_and_prev_move_the_cursor() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 2"));
        out(d.command("next"));
        assert_eq!(d.cursor(), 3);
        out(d.command("prev"));
        assert_eq!(d.cursor(), 2);
    }

    #[test]
    fn goto_clamps_past_the_end() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 99999"));
        assert_eq!(d.cursor(), d.session.len());
    }

    #[test]
    fn timeline_marks_the_cursor() {
        let mut d = Debugger::new(demo_session());
        out(d.command("goto 1"));
        let tl = out(d.command("timeline"));
        assert!(tl.contains("▸ t=1"), "cursor marked at t=1: {tl}");
        assert!(tl.contains("cursor at step 1"));
    }

    #[test]
    fn quit_and_unknown_commands() {
        let mut d = Debugger::new(demo_session());
        assert!(matches!(d.command("quit"), Outcome::Quit));
        assert!(out(d.command("frobnicate")).contains("unknown command"));
        assert!(out(d.command("help")).contains("time-travel position"));
    }

    #[test]
    fn export_produces_a_structured_field_record() {
        let mut s = AgentSession::new();
        s.ingest("src/db.rs", "pub fn q() {}\n");
        let v = export(&s, "workspace.ccos", 2048);
        assert_eq!(v["workspace"], "workspace.ccos");
        assert!(v["stats"]["nodes"].as_u64().unwrap() >= 1);
        assert_eq!(v["integrity"]["valid"], true);
        assert!(v["working_set"]["items"].is_array());
        assert!(v["timeline"]
            .as_array()
            .unwrap()
            .iter()
            .any(|l| l.as_str().unwrap().contains("Ingest(src/db.rs)")));
    }
}
