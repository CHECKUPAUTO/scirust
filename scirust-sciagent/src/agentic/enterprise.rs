//! CCOS Enterprise hardening — identity, RBAC and multi-tenant isolation.
//!
//! This module layers composable authorization on top of the existing
//! [`PermissionGate`](super::permission::PermissionGate): an
//! [`EnterprisePolicyGate`] implements the same [`ToolPolicy`] seam and is
//! chained BEFORE the permission gate, so an enterprise `Deny` is absolute
//! regardless of what the gate would have decided. Identity is explicit
//! ([`EnterpriseIdentity`]) and every workspace root is bound to exactly one
//! tenant: a rule or a tool call that names another tenant's workspace is
//! denied before execution.
//!
//! Isolation invariants:
//! - a workspace path belongs to exactly one tenant;
//! - no rule may grant access to another tenant's workspace;
//! - unknown identity fields fail closed (denied);
//! - an empty role set grants nothing.

use super::tool_runtime::{ToolCall, ToolPolicy};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Strict tenant identifier: 1..=64 ASCII alphanumeric plus `-` and `_`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TenantId(String);

/// Strict organization identifier inside a tenant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrgId(String);

/// Strict project identifier inside an organization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProjectId(String);

/// Strict workspace identifier inside a project.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkspaceId(String);

fn validate_id(value: &str, what: &str) -> Result<(), String> {
    let ok = !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if ok
    {
        Ok(())
    }
    else
    {
        Err(format!(
            "invalid {what} id {value:?}: expected 1..=64 ASCII alphanumeric, '-' or '_'"
        ))
    }
}

macro_rules! id_type {
    ($name:ident, $what:literal) => {
        impl $name {
            /// Validate and construct. Unknown or malformed values are
            /// rejected so no downstream code can act on a bogus tenant.
            pub fn parse(value: &str) -> Result<Self, String> {
                validate_id(value, $what)?;
                Ok(Self(value.to_string()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_type!(TenantId, "tenant");
id_type!(OrgId, "organization");
id_type!(ProjectId, "project");
id_type!(WorkspaceId, "workspace");

/// Full identity of one acting subject in the enterprise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnterpriseIdentity {
    pub tenant: TenantId,
    pub org: OrgId,
    pub project: ProjectId,
    pub workspace: WorkspaceId,
    pub subject: String,
}

impl EnterpriseIdentity {
    pub fn new(
        tenant: TenantId,
        org: OrgId,
        project: ProjectId,
        workspace: WorkspaceId,
        subject: impl Into<String>,
    ) -> Self {
        Self {
            tenant,
            org,
            project,
            workspace,
            subject: subject.into(),
        }
    }

    /// Canonical workspace root bound to this identity.
    pub fn workspace_root(&self) -> PathBuf {
        PathBuf::from("/workspaces")
            .join(self.tenant.as_str())
            .join(self.org.as_str())
            .join(self.project.as_str())
            .join(self.workspace.as_str())
    }
}

/// One RBAC rule: an action on a tool allowed for a role inside a tenant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EnterpriseRule {
    pub tenant: TenantId,
    pub role: String,
    pub tool: String,
    pub action: EnterpriseAction,
}

/// Closed action vocabulary. Unknown actions fail closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EnterpriseAction {
    Read,
    Write,
    Execute,
}

impl EnterpriseAction {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str()
        {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "execute" => Ok(Self::Execute),
            _ => Err(format!("unknown enterprise action {value:?}")),
        }
    }

    pub fn label(self) -> &'static str {
        match self
        {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
        }
    }
}

/// Enterprise authorization decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnterpriseDecision {
    Allow,
    Deny,
}

/// Composable RBAC gate implementing the `ToolPolicy` seam.
///
/// Chain it BEFORE the `PermissionGate` in the tool runtime: its `Deny` is
/// absolute and stops execution, its `Allow` merely passes the decision on to
/// the next gate (the permission gate keeps its own Allow/Ask/Deny semantics
/// on top). Fail closed: no matching rule => Deny.
#[derive(Debug, Clone)]
pub struct EnterprisePolicyGate {
    identity: EnterpriseIdentity,
    roles: BTreeSet<String>,
    rules: BTreeSet<EnterpriseRule>,
}

impl EnterprisePolicyGate {
    pub fn new(identity: EnterpriseIdentity, roles: BTreeSet<String>) -> Self {
        Self {
            identity,
            roles,
            rules: BTreeSet::new(),
        }
    }

    /// Add one rule scoped to this tenant. A rule for another tenant is
    /// refused: no cross-tenant grant can ever be registered.
    pub fn add_rule(&mut self, rule: EnterpriseRule) -> Result<(), String> {
        if rule.tenant != self.identity.tenant
        {
            return Err(format!(
                "refusing cross-tenant rule for tenant {}",
                rule.tenant
            ));
        }
        self.rules.insert(rule);
        Ok(())
    }

    pub fn roles(&self) -> &BTreeSet<String> {
        &self.roles
    }

    /// The tool's requested action, derived from its reserved metadata.
    fn requested_action(call: &ToolCall) -> Result<EnterpriseAction, String> {
        match call.params.get("action").map(String::as_str)
        {
            Some(action) => EnterpriseAction::parse(action),
            // No explicit action defaults to Execute (tool invocation).
            None => Ok(EnterpriseAction::Execute),
        }
    }

    fn authorize(&self, call: &ToolCall) -> Result<(), String> {
        let action = Self::requested_action(call)?;
        let allowed = self.roles.iter().any(|role| {
            self.rules.iter().any(|rule| {
                rule.tenant == self.identity.tenant
                    && rule.role == *role
                    && rule.action == action
                    && (rule.tool == "*" || rule.tool == call.tool)
            })
        });
        if allowed
        {
            Ok(())
        }
        else
        {
            Err(format!(
                "enterprise policy denies {} on {} for subject {}",
                action.label(),
                call.tool,
                self.identity.subject
            ))
        }
    }

    /// Reject any tool call whose parameters reference a workspace path that
    /// escapes this identity's tenant root.
    pub fn validate_workspace_paths(&self, call: &ToolCall) -> Result<(), String> {
        let root = self.identity.workspace_root();
        for (key, value) in &call.params
        {
            if key.ends_with("_path") || key == "path" || key == "workspace"
            {
                self.check_path(root.as_path(), Path::new(value), key)?;
            }
        }
        Ok(())
    }

    fn check_path(&self, root: &Path, path: &Path, key: &str) -> Result<(), String> {
        let absolute = if path.is_absolute()
        {
            path.to_path_buf()
        }
        else
        {
            root.join(path)
        };
        if absolute.starts_with(root)
        {
            Ok(())
        }
        else
        {
            Err(format!(
                "enterprise isolation denies {key} path {}: outside tenant workspace root {}",
                absolute.display(),
                root.display()
            ))
        }
    }
}

impl ToolPolicy for EnterprisePolicyGate {
    fn before_execute(&self, call: &ToolCall, _tool: &super::tools::Tool) -> Result<(), String> {
        self.validate_workspace_paths(call)?;
        self.authorize(call)
    }

    fn approve_sandbox_escalation(
        &self,
        _call: &ToolCall,
        _tool: &super::tools::Tool,
        _request: &super::sandbox_approval::SandboxApprovalRequest,
    ) -> Result<(), String> {
        // Enterprise gates never approve sandbox escalation directly; the
        // permission gate owns that decision. Failing closed here is safe.
        Err("enterprise policy does not approve sandbox escalation".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::sandbox_approval::SandboxPermission;
    use std::collections::HashMap;

    fn noop(_params: HashMap<String, String>) -> String {
        "ok".to_string()
    }

    fn tool(name: &'static str) -> crate::agentic::tools::Tool {
        crate::agentic::tools::Tool {
            name,
            description: "enterprise test tool",
            parameters: Vec::new(),
            execute: noop,
        }
    }

    fn tenant(value: &str) -> TenantId {
        TenantId::parse(value).unwrap()
    }

    fn identity(subject: &str) -> EnterpriseIdentity {
        EnterpriseIdentity::new(
            tenant("acme"),
            OrgId::parse("core").unwrap(),
            ProjectId::parse("tensor").unwrap(),
            WorkspaceId::parse("ws1").unwrap(),
            subject,
        )
    }

    fn call(tool: &str, params: &[(&str, &str)]) -> ToolCall {
        let mut map = HashMap::new();
        for (key, value) in params
        {
            map.insert((*key).to_string(), (*value).to_string());
        }
        ToolCall::new("call-1", tool, map)
    }

    #[test]
    fn id_validation_rejects_malformed_values() {
        assert!(TenantId::parse("acme").is_ok());
        assert!(TenantId::parse("").is_err());
        assert!(TenantId::parse(&"a".repeat(65)).is_err());
        assert!(TenantId::parse("bad space").is_err());
        assert!(TenantId::parse("bad/slash").is_err());
    }

    #[test]
    fn no_rules_grants_nothing() {
        let gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["dev".to_string()]));
        let error = gate
            .before_execute(&call("build", &[]), &tool("build"))
            .expect_err("empty rule set must deny");
        assert!(error.contains("denies"), "{error}");
    }

    #[test]
    fn matching_role_and_tool_allows() {
        let mut gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["dev".to_string()]));
        gate.add_rule(EnterpriseRule {
            tenant: tenant("acme"),
            role: "dev".to_string(),
            tool: "build".to_string(),
            action: EnterpriseAction::Execute,
        })
        .unwrap();
        assert!(
            gate.before_execute(&call("build", &[]), &tool("build"))
                .is_ok()
        );
    }

    #[test]
    fn wrong_role_denies() {
        let mut gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["qa".to_string()]));
        gate.add_rule(EnterpriseRule {
            tenant: tenant("acme"),
            role: "dev".to_string(),
            tool: "build".to_string(),
            action: EnterpriseAction::Execute,
        })
        .unwrap();
        let error = gate
            .before_execute(&call("build", &[]), &tool("build"))
            .expect_err("qa role must not match dev rule");
        assert!(error.contains("denies"), "{error}");
    }

    #[test]
    fn cross_tenant_rule_is_refused() {
        let mut gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["dev".to_string()]));
        let error = gate
            .add_rule(EnterpriseRule {
                tenant: tenant("other"),
                role: "dev".to_string(),
                tool: "build".to_string(),
                action: EnterpriseAction::Execute,
            })
            .expect_err("cross-tenant rule must be refused at registration");
        assert!(error.contains("cross-tenant"), "{error}");
    }

    #[test]
    fn workspace_path_escape_is_denied() {
        let gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["dev".to_string()]));
        let error = gate
            .validate_workspace_paths(&call("read", &[("path", "/etc/passwd")]))
            .expect_err("path escape must be denied");
        assert!(error.contains("isolation"), "{error}");
    }

    #[test]
    fn workspace_path_inside_root_is_allowed() {
        let gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["dev".to_string()]));
        let root = identity("alice").workspace_root();
        let inside = root.join("sub/file.rs");
        assert!(
            gate.validate_workspace_paths(&call("read", &[("path", inside.to_str().unwrap())]))
                .is_ok()
        );
        // Relative paths resolve inside the root.
        assert!(
            gate.validate_workspace_paths(&call("read", &[("path", "sub/file.rs")]))
                .is_ok()
        );
    }

    #[test]
    fn sandbox_escalation_never_approved_by_enterprise() {
        let gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["dev".to_string()]));
        let request = super::super::sandbox_approval::SandboxApprovalRequest {
            call_id: "call-1".to_string(),
            tool: "build".to_string(),
            current: SandboxPermission::ReadOnly,
            requested: SandboxPermission::DangerFullAccess,
            justification: String::new(),
        };
        assert!(
            gate.approve_sandbox_escalation(&call("build", &[]), &tool("build"), &request)
                .is_err()
        );
    }

    #[test]
    fn unknown_action_fails_closed() {
        let gate =
            EnterprisePolicyGate::new(identity("alice"), BTreeSet::from(["dev".to_string()]));
        let error = gate
            .before_execute(&call("build", &[("action", "sudo")]), &tool("build"))
            .expect_err("unknown action must fail closed");
        assert!(error.contains("unknown enterprise action"), "{error}");
    }

    #[test]
    fn workspace_root_is_tenant_scoped() {
        let a = identity("alice").workspace_root();
        let b = EnterpriseIdentity::new(
            tenant("other"),
            OrgId::parse("core").unwrap(),
            ProjectId::parse("tensor").unwrap(),
            WorkspaceId::parse("ws1").unwrap(),
            "bob",
        )
        .workspace_root();
        assert_ne!(a, b);
        assert!(a.starts_with("/workspaces/acme"));
        assert!(b.starts_with("/workspaces/other"));
    }
}
