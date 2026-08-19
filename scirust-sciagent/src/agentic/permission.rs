use super::tool_runtime::{ToolCall, ToolPolicy};
use super::tools::Tool;
use std::sync::Arc;

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

/// Interactive resolver for `Ask` decisions.
pub trait ToolApprover: Send + Sync {
    fn approve(&self, call: &ToolCall, tool: &Tool, subject: &str) -> Result<bool, String>;
}

/// Runtime permission gate installed as a [`ToolPolicy`].
///
/// A missing approver fails closed for `Ask`, matching the DeepSeek user-approval
/// seam. Explicit `Deny` rules always block before the tool callback and before
/// sandbox execution.
pub struct PermissionGate {
    policy: PermissionPolicy,
    approver: Option<Arc<dyn ToolApprover>>,
}

impl PermissionGate {
    pub fn new(policy: PermissionPolicy) -> Self {
        Self {
            policy,
            approver: None,
        }
    }

    pub fn from_env() -> Self {
        Self::new(PermissionPolicy::from_env())
    }

    pub fn with_approver(policy: PermissionPolicy, approver: Arc<dyn ToolApprover>) -> Self {
        Self {
            policy,
            approver: Some(approver),
        }
    }

    pub fn policy(&self) -> &PermissionPolicy {
        &self.policy
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
                let Some(approver) = self.approver.as_ref()
                else
                {
                    return Err(
                        "approval required but no approval service is available; refusing to execute"
                            .to_string(),
                    );
                };
                let subject = primary_subject(call);
                if approver.approve(call, tool, &subject)?
                {
                    Ok(())
                }
                else
                {
                    Err("the user declined this tool call; do not retry it unchanged".to_string())
                }
            },
        }
    }
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
    use std::collections::HashMap;
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

    struct CountingApprover {
        calls: AtomicUsize,
        allow: bool,
    }

    impl ToolApprover for CountingApprover {
        fn approve(&self, _call: &ToolCall, _tool: &Tool, _subject: &str) -> Result<bool, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.allow)
        }
    }

    #[test]
    fn approver_resolves_ask_before_execution() {
        let approver = Arc::new(CountingApprover {
            calls: AtomicUsize::new(0),
            allow: false,
        });
        let gate = PermissionGate::with_approver(PermissionPolicy::default(), approver.clone());
        let error = gate
            .before_execute(&call("test", &[("crate", "scirust-core")]), &tool("test"))
            .expect_err("declined approval must block");
        assert!(error.contains("declined"));
        assert_eq!(approver.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn read_only_builtins_never_prompt_without_explicit_rule() {
        let approver = Arc::new(CountingApprover {
            calls: AtomicUsize::new(0),
            allow: false,
        });
        let gate = PermissionGate::with_approver(PermissionPolicy::default(), approver.clone());
        gate.before_execute(&call("read", &[("path", "README.md")]), &tool("read"))
            .expect("reader should be allowed by fallback");
        assert_eq!(approver.calls.load(Ordering::SeqCst), 0);
    }
}
