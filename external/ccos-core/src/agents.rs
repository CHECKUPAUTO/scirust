//! # Multi-Agent Execution Layer (CCOS v0.3)
//!
//! Specialized agents analyze a shared context and emit results. Every result
//! is funneled through the [`GuardLayer`](crate::guard) and then the
//! [`EventLog`](crate::event_log), so agent output is validated and auditable,
//! and a session of agent runs replays identically.
//!
//! Agents are intentionally deterministic (pure static heuristics over the
//! context, no randomness) so the produced event stream is reproducible.

use crate::event_log::{EventLog, EventPayload, EventType};
use crate::guard::{GuardConfig, GuardLayer};
use crate::util::sha256_hex;
use serde::{Deserialize, Serialize};

/// The specialization of an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    Coder,
    Reviewer,
    Security,
}

impl AgentRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentRole::Coder => "coder",
            AgentRole::Reviewer => "reviewer",
            AgentRole::Security => "security",
        }
    }
}

/// The result of an agent analysis (its `output` is JSON).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentAnalysis {
    pub agent_id: String,
    pub role: AgentRole,
    pub output: String,
    pub confidence: f64,
    pub guard_passed: bool,
}

/// A context-analyzing agent.
pub trait Agent {
    fn id(&self) -> &str;
    fn role(&self) -> AgentRole;
    /// Load the context the agent will reason over.
    fn receive_context(&mut self, context: String);
    /// Produce a (deterministic) analysis of the current context.
    fn analyze(&self) -> AgentAnalysis;
    /// Append the (already guarded) analysis to the event log.
    fn emit_event(&self, analysis: &AgentAnalysis, log: &mut EventLog) {
        log.append(
            EventType::AgentAction,
            EventPayload::Custom {
                key: format!("agent:{}:{}", self.role().as_str(), self.id()),
                value: analysis.output.clone(),
            },
        );
    }
}

/// Runs agents end-to-end: `receive_context → analyze → guard → emit_event`.
#[derive(Debug, Clone)]
pub struct AgentExecutor {
    guard: GuardLayer,
}

impl AgentExecutor {
    pub fn new() -> Self {
        Self {
            guard: GuardLayer::new(GuardConfig::from_env()),
        }
    }

    /// Execute a single agent over `context`, guard its output, log a
    /// `GuardCheck` and an `AgentAction` event, and return the guarded analysis.
    pub fn execute(
        &self,
        agent: &mut dyn Agent,
        context: &str,
        log: &mut EventLog,
    ) -> AgentAnalysis {
        agent.receive_context(context.to_string());
        let mut analysis = agent.analyze();

        let guarded = self.guard.validate_and_sanitize(&analysis.output);
        analysis.guard_passed = guarded.passed;
        analysis.output = guarded.sanitized_output;

        log.append(
            EventType::GuardCheck,
            EventPayload::GuardCheck {
                input_hash: sha256_hex(&analysis.output),
                passed: guarded.passed,
                score: guarded.reliability_score,
                warnings: guarded.warnings,
            },
        );
        agent.emit_event(&analysis, log);
        analysis
    }

    /// Execute several agents over the same context, in order.
    pub fn execute_all(
        &self,
        agents: &mut [Box<dyn Agent>],
        context: &str,
        log: &mut EventLog,
    ) -> Vec<AgentAnalysis> {
        agents
            .iter_mut()
            .map(|a| self.execute(a.as_mut(), context, log))
            .collect()
    }
}

impl Default for AgentExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Concrete agents ─────────────────────────────────────────────────

/// Counts code structure (functions, structs, impls) in the context.
#[derive(Debug, Clone, Default)]
pub struct CoderAgent {
    pub id: String,
    pub context: String,
    pub confidence: f64,
}

impl CoderAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            context: String::new(),
            confidence: 0.0,
        }
    }
}

impl Agent for CoderAgent {
    fn id(&self) -> &str {
        &self.id
    }
    fn role(&self) -> AgentRole {
        AgentRole::Coder
    }
    fn receive_context(&mut self, context: String) {
        self.context = context;
    }
    fn analyze(&self) -> AgentAnalysis {
        let functions = count_occurrences(&self.context, "fn ");
        let structs = count_occurrences(&self.context, "struct ");
        let impls = count_occurrences(&self.context, "impl ");
        let lines = self.context.lines().count();
        let confidence = (0.6 + 0.1 * ((functions + structs) as f64).min(3.0)).min(1.0);
        let output = serde_json::json!({
            "role": "coder",
            "lines": lines,
            "functions": functions,
            "structs": structs,
            "impls": impls,
        })
        .to_string();
        AgentAnalysis {
            agent_id: self.id.clone(),
            role: AgentRole::Coder,
            output,
            confidence,
            guard_passed: false,
        }
    }
}

/// Flags maintainability issues: very long lines, `unwrap`s, `TODO`s.
#[derive(Debug, Clone, Default)]
pub struct ReviewerAgent {
    pub id: String,
    pub context: String,
    pub confidence: f64,
}

impl ReviewerAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            context: String::new(),
            confidence: 0.0,
        }
    }
}

impl Agent for ReviewerAgent {
    fn id(&self) -> &str {
        &self.id
    }
    fn role(&self) -> AgentRole {
        AgentRole::Reviewer
    }
    fn receive_context(&mut self, context: String) {
        self.context = context;
    }
    fn analyze(&self) -> AgentAnalysis {
        let long_lines = self.context.lines().filter(|l| l.len() > 100).count();
        let unwraps = count_occurrences(&self.context, ".unwrap(");
        let todos = count_occurrences(&self.context, "TODO");
        let issues = long_lines + unwraps + todos;
        let confidence = if issues == 0 { 0.8 } else { 0.6 };
        let output = serde_json::json!({
            "role": "reviewer",
            "long_lines": long_lines,
            "unwraps": unwraps,
            "todos": todos,
            "issues": issues,
        })
        .to_string();
        AgentAnalysis {
            agent_id: self.id.clone(),
            role: AgentRole::Reviewer,
            output,
            confidence,
            guard_passed: false,
        }
    }
}

/// Scans for risky patterns (unsafe, shelling out, hardcoded secrets, etc.).
#[derive(Debug, Clone, Default)]
pub struct SecurityAgent {
    pub id: String,
    pub context: String,
    pub confidence: f64,
}

impl SecurityAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            context: String::new(),
            confidence: 0.0,
        }
    }

    const PATTERNS: &'static [&'static str] = &[
        "unsafe ",
        "system(",
        "/bin/",
        "rm -rf",
        "sk-",
        "password",
        "secret_key",
        "process::Command",
    ];
}

impl Agent for SecurityAgent {
    fn id(&self) -> &str {
        &self.id
    }
    fn role(&self) -> AgentRole {
        AgentRole::Security
    }
    fn receive_context(&mut self, context: String) {
        self.context = context;
    }
    fn analyze(&self) -> AgentAnalysis {
        let risks: Vec<&str> = Self::PATTERNS
            .iter()
            .copied()
            .filter(|p| self.context.contains(p))
            .collect();
        let confidence = if risks.is_empty() { 0.6 } else { 0.9 };
        let output = serde_json::json!({
            "role": "security",
            "risk_count": risks.len(),
            "risks": risks,
        })
        .to_string();
        AgentAnalysis {
            agent_id: self.id.clone(),
            role: AgentRole::Security,
            output,
            confidence,
            guard_passed: false,
        }
    }
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "fn main() {}\nstruct S;\nimpl S { fn go(&self) { let x = foo().unwrap(); } }\n// TODO: fix\nlet cmd = std::process::Command::new(\"/bin/sh\");";

    #[test]
    fn agents_produce_valid_guarded_json() {
        let executor = AgentExecutor::new();
        let mut log = EventLog::new("agents".into());
        let mut agents: Vec<Box<dyn Agent>> = vec![
            Box::new(CoderAgent::new("c1")),
            Box::new(ReviewerAgent::new("r1")),
            Box::new(SecurityAgent::new("s1")),
        ];
        let results = executor.execute_all(&mut agents, SAMPLE, &mut log);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(
                r.guard_passed,
                "agent JSON must pass the guard: {}",
                r.output
            );
            assert!(serde_json::from_str::<serde_json::Value>(&r.output).is_ok());
        }
        // Each agent emits a GuardCheck + an AgentAction event.
        assert_eq!(log.events_by_type(EventType::AgentAction).len(), 3);
        assert_eq!(log.events_by_type(EventType::GuardCheck).len(), 3);
    }

    #[test]
    fn security_agent_flags_risky_patterns() {
        let mut agent = SecurityAgent::new("s1");
        agent.receive_context(SAMPLE.to_string());
        let analysis = agent.analyze();
        let v: serde_json::Value = serde_json::from_str(&analysis.output).unwrap();
        assert!(
            v["risk_count"].as_u64().unwrap() >= 2,
            "must flag /bin/ and Command"
        );
        assert_eq!(analysis.confidence, 0.9);
    }

    #[test]
    fn agents_are_deterministic_for_replay() {
        let executor = AgentExecutor::new();

        let run = || {
            let mut log = EventLog::new("det".into());
            let mut agents: Vec<Box<dyn Agent>> = vec![
                Box::new(CoderAgent::new("c1")),
                Box::new(ReviewerAgent::new("r1")),
                Box::new(SecurityAgent::new("s1")),
            ];
            let results = executor.execute_all(&mut agents, SAMPLE, &mut log);
            let outputs: Vec<String> = results.iter().map(|r| r.output.clone()).collect();
            (outputs, log.event_count())
        };

        let (a_out, a_events) = run();
        let (b_out, b_events) = run();
        assert_eq!(a_out, b_out, "agent outputs must be deterministic");
        assert_eq!(a_events, b_events, "event count must match across runs");
    }

    #[test]
    fn coder_agent_counts_structure() {
        let mut agent = CoderAgent::new("c1");
        agent.receive_context(SAMPLE.to_string());
        let v: serde_json::Value = serde_json::from_str(&agent.analyze().output).unwrap();
        assert!(v["functions"].as_u64().unwrap() >= 2);
        assert_eq!(v["structs"].as_u64().unwrap(), 1);
    }
}
