use super::policy_store::ApprovalPolicyStore;
use super::tool_runtime::{ToolCall, ToolPolicy};
use super::tools::Tool;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

const SUBJECT_KEYS: &[&str] = &[
    "command",
    "file_path",
    "path",
    "source_path",
    "destination_path",
    "crate",
    "test",
    "pattern",
];

/// Per-call authorization decision, matching the DeepSeek harness vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    Allow,
    Ask,
    Deny,
}

impl PermissionDecision {
    /// Unknown or empty values conservatively resolve to `Ask`.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str()
        {
            "allow" => Self::Allow,
            "deny" => Self::Deny,
            _ => Self::Ask,
        }
    }
}

/// Closed result vocabulary for one exact one-shot approval request.
///
/// `AllowedOnce` authorizes only the current tool call and is never inserted
/// into the session approval cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOutcome {
    AllowedOnce,
    Rejected,
    Cancelled,
    Unavailable,
}

impl ApprovalOutcome {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str()
        {
            "allowed-once" => Self::AllowedOnce,
            "rejected" => Self::Rejected,
            "cancelled" => Self::Cancelled,
            "unavailable" => Self::Unavailable,
            _ => Self::Unavailable,
        }
    }

    pub fn label(self) -> &'static str {
        match self
        {
            Self::AllowedOnce => "allowed-once",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Session approval policy matching the DeepSeek harness vocabulary.
///
/// `Ask` (the default) delegates approval-required operations to the configured
/// approval service and fails closed when none is available. `Never`
/// deterministically rejects any operation that would require a NEW approval,
/// before any approver or sandbox approval service is invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalPolicy {
    #[default]
    Ask,
    Never,
}

impl ApprovalPolicy {
    /// Wire-safe vocabulary label.
    pub fn label(self) -> &'static str {
        match self
        {
            Self::Ask => "ask",
            Self::Never => "never",
        }
    }

    /// Parse an untrusted policy label; unknown values fail closed to `Never`.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str()
        {
            "ask" => Self::Ask,
            "never" => Self::Never,
            _ => Self::Never,
        }
    }
}

/// One tool-family rule with an optional subject selector.
///
/// Accepted forms are `Tool`, `Tool(glob)`, and the compatibility form
/// `Tool=literal`. Globs support `*` and `?` and intentionally allow `*` to
/// cross path separators so command/path prefixes behave as operators expect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRule {
    pub tool: String,
    pub subject: Option<String>,
    pub literal: bool,
}

impl PermissionRule {
    pub fn parse(input: &str) -> Option<Self> {
        let input = input.trim();
        if input.is_empty()
        {
            return None;
        }

        if let Some(equal) = input.find('=')
        {
            let paren = input.find('(');
            if equal > 0 && paren.is_none_or(|paren| equal < paren)
            {
                let tool = input[..equal].trim();
                if tool.is_empty()
                {
                    return None;
                }
                return Some(Self {
                    tool: tool.to_string(),
                    subject: Some(input[equal + 1..].to_string()),
                    literal: true,
                });
            }
        }

        if input.ends_with(')')
        {
            if let Some(open) = input.find('(')
            {
                let tool = input[..open].trim();
                if tool.is_empty()
                {
                    return None;
                }
                return Some(Self {
                    tool: tool.to_string(),
                    subject: Some(input[open + 1..input.len() - 1].to_string()),
                    literal: false,
                });
            }
        }

        Some(Self {
            tool: input.to_string(),
            subject: None,
            literal: false,
        })
    }

    fn matches(&self, tool: &str, subject: &str) -> bool {
        if !self.tool.eq_ignore_ascii_case(tool)
        {
            return false;
        }
        match self.subject.as_deref()
        {
            None => true,
            Some(_) if subject.is_empty() => false,
            Some(pattern) if self.literal => pattern == subject,
            Some(pattern) => glob_matches(pattern, subject),
        }
    }
}

/// Pure, deterministic authorization policy.
///
/// Rule precedence is `deny > ask > allow > fallback`. Read-only tools fall
/// back to `Allow`; writers fall back to `mode`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionPolicy {
    pub mode: PermissionDecision,
    pub allow: Vec<PermissionRule>,
    pub ask: Vec<PermissionRule>,
    pub deny: Vec<PermissionRule>,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            mode: PermissionDecision::Ask,
            allow: Vec::new(),
            ask: Vec::new(),
            deny: Vec::new(),
        }
    }
}

impl PermissionPolicy {
    pub fn new(
        mode: PermissionDecision,
        allow: Vec<PermissionRule>,
        ask: Vec<PermissionRule>,
        deny: Vec<PermissionRule>,
    ) -> Self {
        Self {
            mode,
            allow,
            ask,
            deny,
        }
    }

    /// Build the policy from SciAgent environment variables.
    ///
    /// Rules are separated by semicolons or newlines:
    /// `SCIAGENT_PERMISSION_ALLOW`, `SCIAGENT_PERMISSION_ASK`, and
    /// `SCIAGENT_PERMISSION_DENY`. `SCIAGENT_PERMISSION_MODE` defaults to
    /// `ask` for writer tools. Malformed empty rules are ignored.
    pub fn from_env() -> Self {
        Self {
            mode: PermissionDecision::parse(
                &std::env::var("SCIAGENT_PERMISSION_MODE").unwrap_or_default(),
            ),
            allow: env_rules("SCIAGENT_PERMISSION_ALLOW"),
            ask: env_rules("SCIAGENT_PERMISSION_ASK"),
            deny: env_rules("SCIAGENT_PERMISSION_DENY"),
        }
    }

    pub fn decide(&self, call: &ToolCall, read_only: bool) -> PermissionDecision {
        let subjects = call_subjects(call);
        if subjects.is_empty()
        {
            return self.decide_subject(&call.tool, read_only, "");
        }

        let mut decision = PermissionDecision::Allow;
        for subject in subjects
        {
            match self.decide_subject(&call.tool, read_only, &subject)
            {
                PermissionDecision::Deny => return PermissionDecision::Deny,
                PermissionDecision::Ask => decision = PermissionDecision::Ask,
                PermissionDecision::Allow =>
                {},
            }
        }
        decision
    }

    fn explicitly_asks(&self, call: &ToolCall) -> bool {
        let subjects = call_subjects(call);
        if subjects.is_empty()
        {
            return matches_any(&self.ask, &call.tool, "");
        }
        subjects
            .iter()
            .any(|subject| matches_any(&self.ask, &call.tool, subject))
    }

    pub fn decide_subject(&self, tool: &str, read_only: bool, subject: &str) -> PermissionDecision {
        if matches_any(&self.deny, tool, subject)
        {
            PermissionDecision::Deny
        }
        else if matches_any(&self.ask, tool, subject)
        {
            PermissionDecision::Ask
        }
        else if matches_any(&self.allow, tool, subject) || read_only
        {
            PermissionDecision::Allow
        }
        else
        {
            self.mode
        }
    }
}

/// One operator response to an `Ask` decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalChoice {
    /// Permit this call only; do not remember it.
    Once,
    /// Permit this tool family for the lifetime of this gate/session.
    Session,
    /// Persist a tool-wide allow rule through the explicitly installed store.
    Always,
    /// Refuse this call; do not remember the refusal.
    Decline,
}

/// Interactive resolver that can distinguish one-shot and session approval.
pub trait ScopedToolApprover: Send + Sync {
    fn approve(
        &self,
        call: &ToolCall,
        tool: &Tool,
        subject: &str,
    ) -> Result<ApprovalChoice, String>;
}

/// Opt-in persistence seam for an explicit `Always` approval.
///
/// SciAgent never chooses a storage location itself. Front-ends may install
/// a store that persists the supplied config-style bare allow rule. Store
/// failure is authorization failure and leaves no session grant behind.
pub trait PermissionRuleStore: Send + Sync {
    fn remember_allow(&self, rule: &str) -> Result<(), String>;
}

/// One-shot resolver with a closed, fail-closed outcome vocabulary.
pub trait ToolApprover: Send + Sync {
    fn approve(
        &self,
        call: &ToolCall,
        tool: &Tool,
        subject: &str,
    ) -> Result<ApprovalOutcome, String>;
}

/// Runtime permission gate installed as a [`ToolPolicy`].
///
/// Session approval follows the DeepSeek harness contract: choosing `Session`
/// remembers the normalized tool name, not the individual parameters. The
/// memory is process-local, shared only by clones of this gate, and never
/// persisted unless the operator explicitly chooses `Always` and a
/// [`PermissionRuleStore`] is installed. Static `Deny` and explicit `Ask` rules
/// keep their precedence over remembered grants.
#[derive(Clone)]
pub struct PermissionGate {
    policy: PermissionPolicy,
    approver: Option<Arc<dyn ToolApprover>>,
    scoped_approver: Option<Arc<dyn ScopedToolApprover>>,
    rule_store: Option<Arc<dyn PermissionRuleStore>>,
    approval_policy: Arc<RwLock<ApprovalPolicy>>,
    approval_policy_store: Option<Arc<dyn ApprovalPolicyStore>>,
    session_approved_tools: Arc<RwLock<HashSet<String>>>,
}

impl PermissionGate {
    pub fn new(policy: PermissionPolicy) -> Self {
        Self {
            policy,
            approver: None,
            scoped_approver: None,
            rule_store: None,
            approval_policy: Arc::new(RwLock::new(ApprovalPolicy::Ask)),
            approval_policy_store: None,
            session_approved_tools: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub fn from_env() -> Self {
        Self::new(PermissionPolicy::from_env())
    }

    /// Install the typed one-shot approver. No outcome is remembered for the session.
    pub fn with_approver(policy: PermissionPolicy, approver: Arc<dyn ToolApprover>) -> Self {
        Self {
            policy,
            approver: Some(approver),
            scoped_approver: None,
            rule_store: None,
            approval_policy: Arc::new(RwLock::new(ApprovalPolicy::Ask)),
            approval_policy_store: None,
            session_approved_tools: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Install an approver that can return `Once`, `Session`, `Always`, or `Decline`.
    pub fn with_scoped_approver(
        policy: PermissionPolicy,
        approver: Arc<dyn ScopedToolApprover>,
    ) -> Self {
        Self {
            policy,
            approver: None,
            scoped_approver: Some(approver),
            rule_store: None,
            approval_policy: Arc::new(RwLock::new(ApprovalPolicy::Ask)),
            approval_policy_store: None,
            session_approved_tools: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub fn policy(&self) -> &PermissionPolicy {
        &self.policy
    }

    /// Current approval policy shared by every clone of this gate.
    ///
    /// Lock failure is authorization failure and fails closed.
    pub fn approval_policy(&self) -> Result<ApprovalPolicy, String> {
        Ok(*self
            .approval_policy
            .read()
            .map_err(|_| "approval policy state is unavailable; refusing approval".to_string())?)
    }

    /// Change the approval policy for this gate/session and all of its clones.
    /// When a durable [`ApprovalPolicyStore`] is installed, the change is
    /// appended there first; a failed append refuses the change so a policy
    /// switch never exists only in memory. Lock failure is authorization
    /// failure and refuses the change.
    pub fn set_approval_policy(&self, policy: ApprovalPolicy) -> Result<(), String> {
        if let Some(store) = self.approval_policy_store.as_ref()
        {
            store.append(policy, "runtime")?;
        }
        *self
            .approval_policy
            .write()
            .map_err(|_| "approval policy state is unavailable; refusing approval".to_string())? =
            policy;
        Ok(())
    }

    /// Install the durable policy store and restore the effective policy from
    /// it. A corrupt or unverifiable log fails closed (`Never`) rather than
    /// silently keeping `Ask`.
    pub fn with_approval_policy_store(
        mut self,
        store: Arc<dyn ApprovalPolicyStore>,
    ) -> Result<Self, String> {
        let effective = store.effective()?;
        self.approval_policy_store = Some(store);
        if let Some(policy) = effective
        {
            *self.approval_policy.write().map_err(|_| {
                "approval policy state is unavailable; refusing approval".to_string()
            })? = policy;
        }
        Ok(self)
    }

    /// The installed durable policy store, if any.
    pub fn approval_policy_store(&self) -> Option<Arc<dyn ApprovalPolicyStore>> {
        self.approval_policy_store.clone()
    }

    /// Install the opt-in persistence sink used only for `ApprovalChoice::Always`.
    pub fn with_rule_store(mut self, store: Arc<dyn PermissionRuleStore>) -> Self {
        self.rule_store = Some(store);
        self
    }

    /// Remember one tool family for this in-memory session.
    pub fn approve_tool_for_session(&self, tool: &str) -> Result<(), String> {
        let key = session_tool_key(tool)?;
        self.session_approved_tools
            .write()
            .map_err(|_| {
                "session approval memory is unavailable; refusing to mutate it".to_string()
            })?
            .insert(key);
        Ok(())
    }

    /// Remove one remembered tool family. Returns whether a grant existed.
    pub fn revoke_tool_for_session(&self, tool: &str) -> Result<bool, String> {
        let key = session_tool_key(tool)?;
        Ok(self
            .session_approved_tools
            .write()
            .map_err(|_| {
                "session approval memory is unavailable; refusing to mutate it".to_string()
            })?
            .remove(&key))
    }

    /// Forget every remembered session approval.
    pub fn clear_session_approvals(&self) -> Result<(), String> {
        self.session_approved_tools
            .write()
            .map_err(|_| {
                "session approval memory is unavailable; refusing to mutate it".to_string()
            })?
            .clear();
        Ok(())
    }

    /// Query a remembered tool family. Lock failure is fail-closed.
    pub fn is_tool_approved_for_session(&self, tool: &str) -> Result<bool, String> {
        let key = session_tool_key(tool)?;
        Ok(self
            .session_approved_tools
            .read()
            .map_err(|_| {
                "session approval memory is unavailable; refusing remembered approval".to_string()
            })?
            .contains(&key))
    }

    /// Return a stable, sorted snapshot suitable for a supervision UI.
    pub fn session_approved_tools(&self) -> Result<Vec<String>, String> {
        let mut tools = self
            .session_approved_tools
            .read()
            .map_err(|_| {
                "session approval memory is unavailable; refusing remembered approval".to_string()
            })?
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        tools.sort();
        Ok(tools)
    }
}

impl ToolPolicy for PermissionGate {
    fn before_execute(&self, call: &ToolCall, tool: &Tool) -> Result<(), String> {
        match self.policy.decide(call, tool_is_read_only(tool))
        {
            PermissionDecision::Allow => Ok(()),
            PermissionDecision::Deny => Err(
                "denied by permission policy; do not retry this tool call unchanged".to_string(),
            ),
            PermissionDecision::Ask =>
            {
                if !self.policy.explicitly_asks(call)
                    && self.is_tool_approved_for_session(&call.tool)?
                {
                    return Ok(());
                }

                if self.approval_policy()? == ApprovalPolicy::Never
                {
                    return Err(
                        "approval policy is set to never; refusing to request approval".to_string(),
                    );
                }

                let subject = primary_subject(call);
                if let Some(approver) = self.scoped_approver.as_ref()
                {
                    return match approver.approve(call, tool, &subject)?
                    {
                        ApprovalChoice::Once => Ok(()),
                        ApprovalChoice::Session =>
                        {
                            self.approve_tool_for_session(&call.tool)?;
                            Ok(())
                        },
                        ApprovalChoice::Always =>
                        {
                            let rule = session_tool_key(&call.tool)?;
                            let Some(store) = self.rule_store.as_ref()
                            else
                            {
                                return Err(
                                    "persistent approval requested but no permission rule store is configured; refusing to execute"
                                        .to_string(),
                                );
                            };
                            store.remember_allow(&rule).map_err(|error| {
                                format!("failed to persist approval rule: {error}")
                            })?;
                            self.approve_tool_for_session(&rule)?;
                            Ok(())
                        },
                        ApprovalChoice::Decline => Err(
                            "the user declined this tool call; do not retry it unchanged"
                                .to_string(),
                        ),
                    };
                }

                let Some(approver) = self.approver.as_ref()
                else
                {
                    return Err(approval_unavailable_error());
                };
                let outcome = approver
                    .approve(call, tool, &subject)
                    .unwrap_or(ApprovalOutcome::Unavailable);
                match outcome
                {
                    ApprovalOutcome::AllowedOnce => Ok(()),
                    ApprovalOutcome::Rejected => Err(
                        "the user declined this tool call; do not retry it unchanged".to_string(),
                    ),
                    ApprovalOutcome::Cancelled => Err(
                        "approval for this tool call was cancelled; do not retry it unchanged"
                            .to_string(),
                    ),
                    ApprovalOutcome::Unavailable => Err(approval_unavailable_error()),
                }
            },
        }
    }
}

fn approval_unavailable_error() -> String {
    "approval required but no approval service is available; refusing to execute".to_string()
}

fn session_tool_key(tool: &str) -> Result<String, String> {
    let tool = tool.trim();
    if tool.is_empty()
    {
        return Err("cannot remember approval for an empty tool name".to_string());
    }
    Ok(tool.to_ascii_lowercase())
}

/// Built-ins with no side effects outside observation are readers. Unknown or
/// future tools default to writer classification, which is conservative.
fn tool_is_read_only(tool: &Tool) -> bool {
    matches!(tool.name, "search" | "grep" | "read" | "explain" | "status")
}

fn env_rules(name: &str) -> Vec<PermissionRule> {
    std::env::var(name)
        .ok()
        .into_iter()
        .flat_map(|value| {
            value
                .split([';', '\n'])
                .filter_map(PermissionRule::parse)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn matches_any(rules: &[PermissionRule], tool: &str, subject: &str) -> bool {
    rules.iter().any(|rule| rule.matches(tool, subject))
}

fn primary_subject(call: &ToolCall) -> String {
    call_subjects(call).into_iter().next().unwrap_or_default()
}

fn call_subjects(call: &ToolCall) -> Vec<String> {
    let source = call
        .params
        .get("source_path")
        .filter(|value| !value.is_empty());
    let destination = call
        .params
        .get("destination_path")
        .filter(|value| !value.is_empty());
    if let (Some(source), Some(destination)) = (source, destination)
    {
        let mut subjects = vec![source.clone()];
        if destination != source
        {
            subjects.push(destination.clone());
        }
        return subjects;
    }

    for key in SUBJECT_KEYS
    {
        if let Some(value) = call.params.get(*key)
        {
            if !value.is_empty()
            {
                return vec![value.clone()];
            }
        }
    }
    Vec::new()
}

fn glob_matches(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    let mut pattern_index = 0usize;
    let mut value_index = 0usize;
    let mut star_pattern = None;
    let mut star_value = 0usize;

    while value_index < value.len()
    {
        if pattern_index < pattern.len()
            && (pattern[pattern_index] == b'?' || pattern[pattern_index] == value[value_index])
        {
            pattern_index += 1;
            value_index += 1;
        }
        else if pattern_index < pattern.len() && pattern[pattern_index] == b'*'
        {
            star_pattern = Some(pattern_index);
            star_value = value_index;
            pattern_index += 1;
        }
        else if let Some(star) = star_pattern
        {
            pattern_index = star + 1;
            star_value += 1;
            value_index = star_value;
        }
        else
        {
            return false;
        }
    }

    while pattern_index < pattern.len() && pattern[pattern_index] == b'*'
    {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn call(tool: &str, params: &[(&str, &str)]) -> ToolCall {
        ToolCall::new(
            "test",
            tool,
            params
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect::<HashMap<_, _>>(),
        )
    }

    fn rule(text: &str) -> PermissionRule {
        PermissionRule::parse(text).expect("valid rule")
    }

    fn noop(_params: HashMap<String, String>) -> String {
        "ok".to_string()
    }

    fn tool(name: &'static str) -> Tool {
        Tool {
            name,
            description: "permission test tool",
            parameters: Vec::new(),
            execute: noop,
        }
    }

    #[test]
    fn parses_bare_glob_and_literal_rules() {
        assert_eq!(
            rule("build"),
            PermissionRule {
                tool: "build".to_string(),
                subject: None,
                literal: false,
            }
        );
        assert_eq!(
            rule("test(scirust-*)").subject.as_deref(),
            Some("scirust-*")
        );
        let literal = rule("build=scirust-core*");
        assert!(literal.literal);
        assert!(literal.matches("build", "scirust-core*"));
        assert!(!literal.matches("build", "scirust-core-extra"));
    }

    #[test]
    fn glob_matches_across_path_separators() {
        assert!(glob_matches(
            "docs/*/contract?.md",
            "docs/api/v1/contract2.md"
        ));
        assert!(!glob_matches("docs/*/contract?.md", "src/api/contract2.md"));
    }

    #[test]
    fn precedence_is_deny_then_ask_then_allow_then_fallback() {
        let policy = PermissionPolicy::new(
            PermissionDecision::Ask,
            vec![rule("build(scirust-*)")],
            vec![rule("build(scirust-core)")],
            vec![rule("build(scirust-core)")],
        );
        assert_eq!(
            policy.decide(&call("build", &[("crate", "scirust-core")]), false),
            PermissionDecision::Deny
        );
        assert_eq!(
            policy.decide(&call("build", &[("crate", "scirust-stats")]), false),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.decide(&call("build", &[("crate", "external")]), false),
            PermissionDecision::Ask
        );
    }

    #[test]
    fn readers_default_allow_and_unknown_writers_use_mode() {
        let policy = PermissionPolicy::default();
        assert_eq!(
            policy.decide(&call("read", &[("path", "README.md")]), true),
            PermissionDecision::Allow
        );
        assert_eq!(
            policy.decide(&call("custom", &[]), false),
            PermissionDecision::Ask
        );
    }

    #[test]
    fn both_move_endpoints_must_be_safe() {
        let policy = PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            vec![rule("move_file(/etc/*)")],
        );
        assert_eq!(
            policy.decide(
                &call(
                    "move_file",
                    &[("source_path", "src/a"), ("destination_path", "/etc/a")],
                ),
                false,
            ),
            PermissionDecision::Deny
        );
    }

    #[test]
    fn subject_specific_rule_does_not_match_subjectless_call() {
        let policy = PermissionPolicy::new(
            PermissionDecision::Ask,
            vec![rule("build(scirust-core)")],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(
            policy.decide(&call("build", &[]), false),
            PermissionDecision::Ask
        );
    }

    #[test]
    fn explicit_deny_blocks_before_execution() {
        let gate = PermissionGate::new(PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            vec![rule("build")],
        ));
        let invocation = call("build", &[("crate", "scirust-core")]);
        let error = gate
            .before_execute(&invocation, &tool("build"))
            .expect_err("deny must block");
        assert!(error.contains("denied by permission policy"));
    }

    #[test]
    fn ask_without_approver_fails_closed() {
        let gate = PermissionGate::new(PermissionPolicy::default());
        let error = gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect_err("ask without approver must fail closed");
        assert!(
            error.contains("no approval service is available"),
            "{error}"
        );
    }

    #[test]
    fn approval_policy_vocabulary_matches_harness() {
        assert_eq!(ApprovalPolicy::Ask.label(), "ask");
        assert_eq!(ApprovalPolicy::Never.label(), "never");
        assert_eq!(ApprovalPolicy::default(), ApprovalPolicy::Ask);
        assert_eq!(ApprovalPolicy::parse("ask"), ApprovalPolicy::Ask);
        assert_eq!(ApprovalPolicy::parse("never"), ApprovalPolicy::Never);
        assert_eq!(ApprovalPolicy::parse("NEVER"), ApprovalPolicy::Never);
        assert_eq!(ApprovalPolicy::parse("unknown"), ApprovalPolicy::Never);

        let gate = PermissionGate::new(PermissionPolicy::default());
        assert_eq!(gate.approval_policy().unwrap(), ApprovalPolicy::Ask);
    }

    #[test]
    fn never_policy_is_shared_and_blocks_before_approver() {
        let approver = one_shot_approver(ApprovalOutcome::AllowedOnce);
        let gate = PermissionGate::with_approver(PermissionPolicy::default(), approver.clone());
        let clone = gate.clone();

        clone.set_approval_policy(ApprovalPolicy::Never).unwrap();
        assert_eq!(gate.approval_policy().unwrap(), ApprovalPolicy::Never);

        let error = gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect_err("never must reject before the approver");

        assert!(error.contains("approval policy is set to never"), "{error}");
        assert_eq!(approver.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn never_policy_blocks_explicit_ask_with_scoped_approver() {
        let policy = PermissionPolicy::new(
            PermissionDecision::Ask,
            Vec::new(),
            vec![rule("build(secret-*)")],
            Vec::new(),
        );
        let approver = Arc::new(SequenceScopedApprover::new([ApprovalChoice::Once]));
        let gate = PermissionGate::with_scoped_approver(policy, approver.clone());
        gate.approve_tool_for_session("build").unwrap();
        gate.set_approval_policy(ApprovalPolicy::Never).unwrap();

        let error = gate
            .before_execute(&call("build", &[("crate", "secret-model")]), &tool("build"))
            .expect_err("explicit ask must still require approval under never");

        assert!(error.contains("approval policy is set to never"), "{error}");
        assert_eq!(approver.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn never_policy_leaves_static_allow_deny_and_remembered_grants_intact() {
        // Static Allow stays a pass even under Never.
        let allow_gate = PermissionGate::new(PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        allow_gate
            .set_approval_policy(ApprovalPolicy::Never)
            .unwrap();
        allow_gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect("static allow must remain a pass under never");

        // Static Deny stays an absolute block even under Never.
        let deny_gate = PermissionGate::new(PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            vec![rule("build")],
        ));
        deny_gate
            .set_approval_policy(ApprovalPolicy::Never)
            .unwrap();
        let error = deny_gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect_err("static deny must remain absolute under never");
        assert!(error.contains("denied by permission policy"), "{error}");

        // A remembered session grant prevents a NEW approval request, so it
        // remains valid under Never (no implicit revocation).
        let remembered_gate = PermissionGate::new(PermissionPolicy::default());
        remembered_gate.approve_tool_for_session("build").unwrap();
        remembered_gate
            .set_approval_policy(ApprovalPolicy::Never)
            .unwrap();
        remembered_gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect("remembered session grant must satisfy an ordinary ask under never");
    }

    #[test]
    fn approval_policy_lock_poisoning_fails_closed() {
        let gate = PermissionGate::new(PermissionPolicy::default());
        let clone = gate.clone();
        // Poison the shared policy lock.
        let handle = std::thread::spawn(move || {
            let _guard = clone.approval_policy.write().unwrap();
            panic!("poison the approval policy lock");
        });
        assert!(handle.join().is_err());

        let error = gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect_err("poisoned policy state must fail closed");
        assert!(error.contains("unavailable"), "{error}");
        assert!(gate.set_approval_policy(ApprovalPolicy::Never).is_err());
    }

    #[test]
    fn durable_store_restores_effective_policy_on_gate() {
        let store = Arc::new(super::super::policy_store::MemoryApprovalPolicyStore::default());
        store.append(ApprovalPolicy::Never, "deployment").unwrap();
        let gate = PermissionGate::new(PermissionPolicy::default())
            .with_approval_policy_store(store.clone())
            .expect("store must load");
        assert_eq!(gate.approval_policy().unwrap(), ApprovalPolicy::Never);
        assert!(gate.approval_policy_store().is_some());
    }

    #[test]
    fn durable_set_policy_appends_before_switching() {
        let store = Arc::new(super::super::policy_store::MemoryApprovalPolicyStore::default());
        let gate = PermissionGate::new(PermissionPolicy::default())
            .with_approval_policy_store(store.clone())
            .unwrap();
        gate.set_approval_policy(ApprovalPolicy::Never).unwrap();
        assert_eq!(gate.approval_policy().unwrap(), ApprovalPolicy::Never);
        assert_eq!(store.effective().unwrap(), Some(ApprovalPolicy::Never));
    }

    #[test]
    fn durable_store_failure_fails_closed_and_keeps_old_policy() {
        struct FailingStore;
        impl ApprovalPolicyStore for FailingStore {
            fn append(&self, _policy: ApprovalPolicy, _source: &str) -> Result<u64, String> {
                Err("policy store unavailable".to_string())
            }
            fn effective(&self) -> Result<Option<ApprovalPolicy>, String> {
                Ok(None)
            }
            fn events(
                &self,
            ) -> Result<Vec<super::super::policy_store::ApprovalPolicyEvent>, String> {
                Ok(Vec::new())
            }
        }
        let store = Arc::new(FailingStore);
        let gate = PermissionGate::new(PermissionPolicy::default())
            .with_approval_policy_store(store.clone())
            .unwrap();
        assert_eq!(gate.approval_policy().unwrap(), ApprovalPolicy::Ask);
        let error = gate
            .set_approval_policy(ApprovalPolicy::Never)
            .expect_err("a failed durable append must refuse the switch");
        assert!(error.contains("policy store unavailable"), "{error}");
        assert_eq!(gate.approval_policy().unwrap(), ApprovalPolicy::Ask);
    }

    struct CountingApprover {
        calls: AtomicUsize,
        outcome: ApprovalOutcome,
        fail: bool,
    }

    impl ToolApprover for CountingApprover {
        fn approve(
            &self,
            _call: &ToolCall,
            _tool: &Tool,
            _subject: &str,
        ) -> Result<ApprovalOutcome, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail
            {
                Err("approval transport failed".to_string())
            }
            else
            {
                Ok(self.outcome)
            }
        }
    }

    fn one_shot_approver(outcome: ApprovalOutcome) -> Arc<CountingApprover> {
        Arc::new(CountingApprover {
            calls: AtomicUsize::new(0),
            outcome,
            fail: false,
        })
    }

    #[test]
    fn approval_outcome_vocabulary_matches_harness() {
        for (value, outcome) in [
            ("allowed-once", ApprovalOutcome::AllowedOnce),
            ("rejected", ApprovalOutcome::Rejected),
            ("cancelled", ApprovalOutcome::Cancelled),
            ("unavailable", ApprovalOutcome::Unavailable),
        ]
        {
            assert_eq!(ApprovalOutcome::parse(value), outcome);
            assert_eq!(outcome.label(), value);
        }
        assert_eq!(
            ApprovalOutcome::parse("unexpected-answer"),
            ApprovalOutcome::Unavailable
        );
    }

    #[test]
    fn rejected_cancelled_and_unavailable_one_shot_outcomes_fail_closed() {
        for (outcome, needle) in [
            (ApprovalOutcome::Rejected, "declined"),
            (ApprovalOutcome::Cancelled, "cancelled"),
            (
                ApprovalOutcome::Unavailable,
                "no approval service is available",
            ),
        ]
        {
            let approver = one_shot_approver(outcome);
            let gate = PermissionGate::with_approver(PermissionPolicy::default(), approver.clone());
            let error = gate
                .before_execute(&call("test", &[("crate", "scirust-core")]), &tool("test"))
                .expect_err("non-allow one-shot outcome must block");
            assert!(error.contains(needle), "{error}");
            assert_eq!(approver.calls.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn one_shot_approver_error_normalizes_to_unavailable() {
        let approver = Arc::new(CountingApprover {
            calls: AtomicUsize::new(0),
            outcome: ApprovalOutcome::AllowedOnce,
            fail: true,
        });
        let gate = PermissionGate::with_approver(PermissionPolicy::default(), approver.clone());
        let error = gate
            .before_execute(&call("test", &[("crate", "scirust-core")]), &tool("test"))
            .expect_err("approver transport error must fail closed");
        assert!(
            error.contains("no approval service is available"),
            "{error}"
        );
        assert_eq!(approver.calls.load(Ordering::SeqCst), 1);
    }

    struct SequenceScopedApprover {
        calls: AtomicUsize,
        choices: Mutex<VecDeque<ApprovalChoice>>,
    }

    impl SequenceScopedApprover {
        fn new(choices: impl IntoIterator<Item = ApprovalChoice>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                choices: Mutex::new(choices.into_iter().collect()),
            }
        }
    }

    impl ScopedToolApprover for SequenceScopedApprover {
        fn approve(
            &self,
            _call: &ToolCall,
            _tool: &Tool,
            _subject: &str,
        ) -> Result<ApprovalChoice, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.choices
                .lock()
                .map_err(|_| "test approval queue poisoned".to_string())?
                .pop_front()
                .ok_or_else(|| "no test approval choice remains".to_string())
        }
    }

    #[test]
    fn session_choice_caches_only_the_tool_family() {
        let approver = Arc::new(SequenceScopedApprover::new([
            ApprovalChoice::Session,
            ApprovalChoice::Once,
        ]));
        let gate =
            PermissionGate::with_scoped_approver(PermissionPolicy::default(), approver.clone());

        gate.before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect("first build should receive session approval");
        gate.before_execute(
            &call("build", &[("crate", "scirust-stats")]),
            &tool("build"),
        )
        .expect("same tool should use remembered session approval");
        gate.before_execute(&call("test", &[("crate", "scirust-core")]), &tool("test"))
            .expect("different tool should request its own approval");

        assert_eq!(approver.calls.load(Ordering::SeqCst), 2);
        assert_eq!(gate.session_approved_tools().unwrap(), vec!["build"]);
    }

    #[test]
    fn one_shot_and_decline_are_never_cached() {
        let approver = Arc::new(SequenceScopedApprover::new([
            ApprovalChoice::Once,
            ApprovalChoice::Decline,
        ]));
        let gate =
            PermissionGate::with_scoped_approver(PermissionPolicy::default(), approver.clone());

        gate.before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect("one-shot approval should permit the first call");
        let error = gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect_err("second call should prompt and be declined");

        assert!(error.contains("declined"), "{error}");
        assert_eq!(approver.calls.load(Ordering::SeqCst), 2);
        assert!(gate.session_approved_tools().unwrap().is_empty());
    }

    #[test]
    fn static_deny_overrides_remembered_session_approval() {
        let policy = PermissionPolicy::new(
            PermissionDecision::Ask,
            Vec::new(),
            Vec::new(),
            vec![rule("build(secret-*)")],
        );
        let gate = PermissionGate::new(policy);
        gate.approve_tool_for_session("BUILD").unwrap();

        gate.before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect("remembered build approval should satisfy an ordinary ask");
        let error = gate
            .before_execute(&call("build", &[("crate", "secret-model")]), &tool("build"))
            .expect_err("explicit deny must dominate remembered approval");
        assert!(error.contains("denied by permission policy"), "{error}");
    }

    #[test]
    fn session_approvals_are_revocable_clearable_and_process_local() {
        let gate = PermissionGate::new(PermissionPolicy::default());
        let clone = gate.clone();
        gate.approve_tool_for_session("Build").unwrap();
        gate.approve_tool_for_session("test").unwrap();

        assert_eq!(
            clone.session_approved_tools().unwrap(),
            vec!["build".to_string(), "test".to_string()]
        );
        assert!(clone.revoke_tool_for_session("BUILD").unwrap());
        assert!(!gate.is_tool_approved_for_session("build").unwrap());
        clone.clear_session_approvals().unwrap();
        assert!(gate.session_approved_tools().unwrap().is_empty());

        let fresh_gate = PermissionGate::new(PermissionPolicy::default());
        assert!(!fresh_gate.is_tool_approved_for_session("test").unwrap());
    }

    #[test]
    fn typed_allowed_once_is_never_cached_as_session_approval() {
        let approver = one_shot_approver(ApprovalOutcome::AllowedOnce);
        let gate = PermissionGate::with_approver(PermissionPolicy::default(), approver.clone());

        gate.before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .unwrap();
        gate.before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .unwrap();

        assert_eq!(approver.calls.load(Ordering::SeqCst), 2);
        assert!(gate.session_approved_tools().unwrap().is_empty());
    }

    #[test]
    fn read_only_builtins_never_prompt_without_explicit_rule() {
        let approver = one_shot_approver(ApprovalOutcome::Rejected);
        let gate = PermissionGate::with_approver(PermissionPolicy::default(), approver.clone());
        gate.before_execute(&call("read", &[("path", "README.md")]), &tool("read"))
            .expect("reader should be allowed by fallback");
        assert_eq!(approver.calls.load(Ordering::SeqCst), 0);
    }

    struct MemoryRuleStore {
        rules: Mutex<Vec<String>>,
        fail: bool,
    }

    impl MemoryRuleStore {
        fn new(fail: bool) -> Self {
            Self {
                rules: Mutex::new(Vec::new()),
                fail,
            }
        }

        fn rules(&self) -> Vec<String> {
            self.rules.lock().unwrap().clone()
        }
    }

    impl PermissionRuleStore for MemoryRuleStore {
        fn remember_allow(&self, rule: &str) -> Result<(), String> {
            if self.fail
            {
                return Err("rule store unavailable".to_string());
            }
            self.rules
                .lock()
                .map_err(|_| "rule store poisoned".to_string())?
                .push(rule.to_string());
            Ok(())
        }
    }

    #[test]
    fn persistent_choice_requires_an_explicit_store() {
        let approver = Arc::new(SequenceScopedApprover::new([ApprovalChoice::Always]));
        let gate = PermissionGate::with_scoped_approver(PermissionPolicy::default(), approver);
        let error = gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect_err("Always without a store must fail closed");
        assert!(error.contains("no permission rule store"), "{error}");
        assert!(gate.session_approved_tools().unwrap().is_empty());
    }

    #[test]
    fn persistent_choice_stores_bare_rule_and_grants_current_session() {
        let approver = Arc::new(SequenceScopedApprover::new([ApprovalChoice::Always]));
        let store = Arc::new(MemoryRuleStore::new(false));
        let gate =
            PermissionGate::with_scoped_approver(PermissionPolicy::default(), approver.clone())
                .with_rule_store(store.clone());

        gate.before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect("persistent approval should authorize the call");
        gate.before_execute(
            &call("build", &[("crate", "scirust-stats")]),
            &tool("build"),
        )
        .expect("persistent approval should grant the current session");

        assert_eq!(store.rules(), vec!["build"]);
        assert_eq!(gate.session_approved_tools().unwrap(), vec!["build"]);
        assert_eq!(approver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn persistent_store_failure_fails_closed_without_session_grant() {
        let approver = Arc::new(SequenceScopedApprover::new([ApprovalChoice::Always]));
        let gate = PermissionGate::with_scoped_approver(PermissionPolicy::default(), approver)
            .with_rule_store(Arc::new(MemoryRuleStore::new(true)));
        let error = gate
            .before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect_err("store failure must fail closed");
        assert!(error.contains("failed to persist approval rule"), "{error}");
        assert!(gate.session_approved_tools().unwrap().is_empty());
    }

    #[test]
    fn static_ask_overrides_remembered_session_approval() {
        let policy = PermissionPolicy::new(
            PermissionDecision::Ask,
            Vec::new(),
            vec![rule("build(secret-*)")],
            Vec::new(),
        );
        let gate = PermissionGate::new(policy);
        gate.approve_tool_for_session("build").unwrap();

        gate.before_execute(&call("build", &[("crate", "scirust-core")]), &tool("build"))
            .expect("fallback ask may be satisfied by remembered approval");
        let error = gate
            .before_execute(&call("build", &[("crate", "secret-model")]), &tool("build"))
            .expect_err("explicit ask must still require approval");
        assert!(
            error.contains("no approval service is available"),
            "{error}"
        );
    }
}
