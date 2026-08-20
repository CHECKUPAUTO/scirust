//! Safe child-agent delegation — monotonic privilege ceilings.
//!
//! Fundamental rule: a child's privileges are a SUBSET of its parent's.
//! [`DelegationContext`] carries the ceilings a child may rely on; deriving a
//! child context from a parent context can only narrow them, never widen
//! them. Nested delegation preserves monotonicity by construction.

use super::permission::ApprovalPolicy;
use super::sandbox_approval::SandboxPermission;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Component, Path};

/// Resource ceilings for one delegated agent.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResourceBudget {
    pub wall_time_seconds: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub max_processes: Option<u32>,
    pub gpu_allowed: bool,
}

impl ResourceBudget {
    /// A budget fits inside a ceiling when every field is at most the
    /// ceiling's (None = unbounded ceiling).
    fn fits_in(&self, ceiling: &ResourceBudget) -> bool {
        fn limit_fits<T: PartialOrd>(requested: Option<T>, ceiling: Option<T>) -> bool {
            match (requested, ceiling)
            {
                (_, None) => true,
                (Some(value), Some(limit)) => value <= limit,
                (None, Some(_)) => false,
            }
        }

        let time_ok = limit_fits(self.wall_time_seconds, ceiling.wall_time_seconds);
        let memory_ok = limit_fits(self.memory_bytes, ceiling.memory_bytes);
        let processes_ok = limit_fits(self.max_processes, ceiling.max_processes);
        let gpu_ok = !self.gpu_allowed || ceiling.gpu_allowed;
        time_ok && memory_ok && processes_ok && gpu_ok
    }
}

fn normalized_workspace_components(value: &str) -> Option<Vec<OsString>> {
    if value.is_empty()
    {
        return None;
    }

    let mut components = Vec::new();
    for component in Path::new(value).components()
    {
        match component
        {
            Component::ParentDir => return None,
            Component::CurDir => {},
            Component::Prefix(prefix) => components.push(prefix.as_os_str().to_os_string()),
            Component::RootDir => components.push(OsString::from(std::path::MAIN_SEPARATOR_STR)),
            Component::Normal(part) => components.push(part.to_os_string()),
        }
    }
    (!components.is_empty()).then_some(components)
}

fn workspace_root_is_within(root: &str, parent: &str) -> bool {
    let Some(root) = normalized_workspace_components(root)
    else
    {
        return false;
    };
    let Some(parent) = normalized_workspace_components(parent)
    else
    {
        return false;
    };

    root.len() >= parent.len()
        && root
            .iter()
            .zip(parent.iter())
            .all(|(root_component, parent_component)| root_component == parent_component)
}

/// Secret capability handle: an opaque id the child may reference; the
/// secret itself is never copied into the child context.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SecretCapability {
    pub id: String,
}

/// Full delegation context of one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelegationContext {
    pub session_id: String,
    pub parent_session_id: Option<String>,
    pub allowed_tools: Option<BTreeSet<String>>,
    pub sandbox_ceiling: SandboxPermission,
    pub approval_policy: ApprovalPolicy,
    pub resource_budget: ResourceBudget,
    pub secret_capabilities: BTreeSet<SecretCapability>,
    pub workspace_roots: Vec<String>,
}

impl DelegationContext {
    pub fn root(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            parent_session_id: None,
            allowed_tools: None,
            sandbox_ceiling: SandboxPermission::DangerFullAccess,
            approval_policy: ApprovalPolicy::Ask,
            resource_budget: ResourceBudget::default(),
            secret_capabilities: BTreeSet::new(),
            workspace_roots: Vec::new(),
        }
    }

    /// Derive a child context that can only narrow the parent's ceilings.
    ///
    /// Returns an error for any requested widening; the parent context is
    /// never modified.
    pub fn derive_child(&self, request: ChildRequest) -> Result<Self, DelegationError> {
        let ChildRequest {
            child_session_id,
            allowed_tools,
            sandbox_ceiling,
            approval_policy,
            resource_budget,
            secret_capabilities,
            workspace_roots,
        } = request;
        // Tools: the child may only receive a subset of the parent's set.
        if let Some(parent_tools) = self.allowed_tools.as_ref()
        {
            if let Some(child_tools) = allowed_tools.as_ref()
            {
                if !child_tools.is_subset(parent_tools)
                {
                    return Err(DelegationError::ToolWidening);
                }
            }
            else
            {
                // "No tool list" would mean "all parent tools" — but an
                // explicit subset is required for child contexts.
                return Err(DelegationError::ToolWidening);
            }
        }

        // Sandbox: the child ceiling cannot exceed the parent's.
        if !sandbox_ceiling.le(&self.sandbox_ceiling)
        {
            return Err(DelegationError::SandboxWidening);
        }

        // Approval policy: Never is stricter than Ask. A child may keep Never
        // or move to Never; it may not move from Never to Ask.
        if self.approval_policy == ApprovalPolicy::Never && approval_policy == ApprovalPolicy::Ask
        {
            return Err(DelegationError::ApprovalPolicyWidening);
        }

        // Resources: the child budget must fit inside the parent's.
        if !resource_budget.fits_in(&self.resource_budget)
        {
            return Err(DelegationError::ResourceWidening);
        }

        // Secrets: the child never receives a capability the parent does not
        // own.
        for capability in &secret_capabilities
        {
            if !self.secret_capabilities.contains(capability)
            {
                return Err(DelegationError::UnauthorizedSecret(capability.id.clone()));
            }
        }

        // Workspace roots: compare path components rather than raw string
        // prefixes. Any `..` component fails closed, so a child cannot turn
        // `/workspace/../etc` into an apparent descendant of `/workspace`.
        for root in &workspace_roots
        {
            if !self
                .workspace_roots
                .iter()
                .any(|parent| workspace_root_is_within(root, parent))
            {
                return Err(DelegationError::WorkspaceWidening(root.clone()));
            }
        }

        Ok(Self {
            session_id: child_session_id,
            parent_session_id: Some(self.session_id.clone()),
            allowed_tools,
            sandbox_ceiling,
            approval_policy,
            resource_budget,
            secret_capabilities,
            workspace_roots,
        })
    }
}

/// One requested child context, validated against the parent ceilings.
#[derive(Debug, Clone)]
pub struct ChildRequest {
    pub child_session_id: String,
    pub allowed_tools: Option<BTreeSet<String>>,
    pub sandbox_ceiling: SandboxPermission,
    pub approval_policy: ApprovalPolicy,
    pub resource_budget: ResourceBudget,
    pub secret_capabilities: BTreeSet<SecretCapability>,
    pub workspace_roots: Vec<String>,
}

impl ChildRequest {
    pub fn new(
        child_session_id: impl Into<String>,
        allowed_tools: Option<BTreeSet<String>>,
        sandbox_ceiling: SandboxPermission,
        approval_policy: ApprovalPolicy,
        resource_budget: ResourceBudget,
        secret_capabilities: BTreeSet<SecretCapability>,
        workspace_roots: Vec<String>,
    ) -> Self {
        Self {
            child_session_id: child_session_id.into(),
            allowed_tools,
            sandbox_ceiling,
            approval_policy,
            resource_budget,
            secret_capabilities,
            workspace_roots,
        }
    }
}

impl SandboxPermission {
    /// Partial order: read-only < workspace-write < danger-full-access.
    fn le(&self, other: &SandboxPermission) -> bool {
        matches!(
            (self, other),
            (Self::ReadOnly, _)
                | (
                    Self::WorkspaceWrite,
                    Self::WorkspaceWrite | Self::DangerFullAccess
                )
                | (Self::DangerFullAccess, Self::DangerFullAccess)
        )
    }
}

/// Closed delegation failure vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DelegationError {
    ToolWidening,
    SandboxWidening,
    ApprovalPolicyWidening,
    ResourceWidening,
    UnauthorizedSecret(String),
    WorkspaceWidening(String),
}

impl std::fmt::Display for DelegationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::ToolWidening => write!(f, "child may not add tools beyond the parent set"),
            Self::SandboxWidening => write!(f, "child may not widen the sandbox ceiling"),
            Self::ApprovalPolicyWidening =>
            {
                write!(f, "child may not widen the approval policy (never -> ask)")
            },
            Self::ResourceWidening => write!(f, "child may not enlarge the resource ceiling"),
            Self::UnauthorizedSecret(id) =>
            {
                write!(
                    f,
                    "child may not inherit unauthorized secret capability {id:?}"
                )
            },
            Self::WorkspaceWidening(root) =>
            {
                write!(
                    f,
                    "child workspace root {root:?} is outside the parent roots"
                )
            },
        }
    }
}

impl std::error::Error for DelegationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(names: &[&str]) -> Option<BTreeSet<String>> {
        Some(names.iter().map(|n| n.to_string()).collect())
    }

    fn secret(id: &str) -> SecretCapability {
        SecretCapability { id: id.to_string() }
    }

    /// Build a child request with the given session and ceilings.
    fn request(
        session: &str,
        tools: Option<BTreeSet<String>>,
        sandbox: SandboxPermission,
        policy: ApprovalPolicy,
        budget: ResourceBudget,
        secrets: BTreeSet<SecretCapability>,
        roots: Vec<String>,
    ) -> ChildRequest {
        ChildRequest::new(session, tools, sandbox, policy, budget, secrets, roots)
    }

    /// Root context with an allowed-tools ceiling, for tests.
    fn rooted(session: &str, tools: Option<BTreeSet<String>>) -> DelegationContext {
        DelegationContext {
            session_id: session.to_string(),
            parent_session_id: None,
            allowed_tools: tools,
            sandbox_ceiling: SandboxPermission::DangerFullAccess,
            approval_policy: ApprovalPolicy::Ask,
            resource_budget: ResourceBudget::default(),
            secret_capabilities: BTreeSet::new(),
            workspace_roots: Vec::new(),
        }
    }

    /// Root context that already OWNS the given secrets and workspace roots
    /// (a deployment bootstrap), for tests.
    fn owning_root(
        session: &str,
        secrets: BTreeSet<SecretCapability>,
        workspace_roots: Vec<String>,
    ) -> DelegationContext {
        DelegationContext {
            session_id: session.to_string(),
            parent_session_id: None,
            allowed_tools: None,
            sandbox_ceiling: SandboxPermission::DangerFullAccess,
            approval_policy: ApprovalPolicy::Ask,
            resource_budget: ResourceBudget::default(),
            secret_capabilities: secrets,
            workspace_roots,
        }
    }

    #[test]
    fn child_cannot_add_a_tool() {
        let parent = rooted("parent", tools(&["read", "grep"]));
        let error = parent
            .derive_child(request(
                "child",
                tools(&["read", "build"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .expect_err("adding a tool must fail");
        assert_eq!(error, DelegationError::ToolWidening);
    }

    #[test]
    fn child_cannot_widen_sandbox() {
        let parent = rooted("parent", None);
        // Parent ceiling is danger-full-access; a child may not exceed it —
        // but it cannot be wider anyway. Use a stricter parent: build a
        // parent with workspace-write ceiling.
        let parent = parent
            .derive_child(request(
                "parent2",
                None,
                SandboxPermission::WorkspaceWrite,
                ApprovalPolicy::Ask,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .unwrap();
        let error = parent
            .derive_child(request(
                "child",
                None,
                SandboxPermission::DangerFullAccess,
                ApprovalPolicy::Ask,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .expect_err("widening the sandbox ceiling must fail");
        assert_eq!(error, DelegationError::SandboxWidening);
    }

    #[test]
    fn child_cannot_widen_approval_policy() {
        let parent = rooted("parent", None);
        let parent = parent
            .derive_child(request(
                "parent2",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .unwrap();
        let error = parent
            .derive_child(request(
                "child",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Ask,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .expect_err("never -> ask must fail");
        assert_eq!(error, DelegationError::ApprovalPolicyWidening);
    }

    #[test]
    fn child_cannot_inherit_unauthorized_secret() {
        let parent = owning_root("parent", BTreeSet::from([secret("api-key")]), Vec::new());
        let parent = parent
            .derive_child(request(
                "parent2",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::from([secret("api-key")]),
                Vec::new(),
            ))
            .unwrap();
        let error = parent
            .derive_child(request(
                "child",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::from([secret("other-key")]),
                Vec::new(),
            ))
            .expect_err("unauthorized secret must fail");
        assert_eq!(
            error,
            DelegationError::UnauthorizedSecret("other-key".to_string())
        );
    }

    #[test]
    fn child_cannot_enlarge_resource_ceiling() {
        let parent = rooted("parent", None);
        let parent = parent
            .derive_child(request(
                "parent2",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget {
                    wall_time_seconds: Some(60),
                    memory_bytes: None,
                    max_processes: Some(4),
                    gpu_allowed: false,
                },
                BTreeSet::new(),
                Vec::new(),
            ))
            .unwrap();
        let error = parent
            .derive_child(request(
                "child",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget {
                    wall_time_seconds: Some(120),
                    memory_bytes: None,
                    max_processes: Some(2),
                    gpu_allowed: false,
                },
                BTreeSet::new(),
                Vec::new(),
            ))
            .expect_err("larger wall time must fail");
        assert_eq!(error, DelegationError::ResourceWidening);
    }

    #[test]
    fn child_cannot_drop_finite_resource_ceiling() {
        let finite_budgets = [
            ResourceBudget {
                wall_time_seconds: Some(60),
                ..ResourceBudget::default()
            },
            ResourceBudget {
                memory_bytes: Some(1024),
                ..ResourceBudget::default()
            },
            ResourceBudget {
                max_processes: Some(4),
                ..ResourceBudget::default()
            },
        ];

        for resource_budget in finite_budgets
        {
            let mut parent = rooted("parent", None);
            parent.resource_budget = resource_budget;
            let error = parent
                .derive_child(request(
                    "child",
                    None,
                    SandboxPermission::ReadOnly,
                    ApprovalPolicy::Never,
                    ResourceBudget::default(),
                    BTreeSet::new(),
                    Vec::new(),
                ))
                .expect_err("removing a finite parent resource ceiling must fail");
            assert_eq!(error, DelegationError::ResourceWidening);
        }
    }

    #[test]
    fn nested_children_remain_constrained() {
        let parent = rooted("parent", tools(&["read", "grep", "status"]));
        let child = parent
            .derive_child(request(
                "child",
                tools(&["read"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget {
                    wall_time_seconds: Some(30),
                    memory_bytes: None,
                    max_processes: None,
                    gpu_allowed: false,
                },
                BTreeSet::new(),
                Vec::new(),
            ))
            .unwrap();
        // The grandchild is derived from the child, not the root: it can only
        // narrow the child's ceilings.
        let grandchild = child
            .derive_child(request(
                "grandchild",
                tools(&["read"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget {
                    wall_time_seconds: Some(10),
                    memory_bytes: None,
                    max_processes: None,
                    gpu_allowed: false,
                },
                BTreeSet::new(),
                Vec::new(),
            ))
            .expect("narrowing must succeed");
        assert_eq!(grandchild.parent_session_id.as_deref(), Some("child"));
        let error = grandchild
            .derive_child(request(
                "great-grandchild",
                tools(&["read", "build"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .expect_err("nested widening must fail");
        assert_eq!(error, DelegationError::ToolWidening);
    }

    #[test]
    fn child_termination_does_not_alter_parent() {
        let parent = rooted("parent", tools(&["read"]));
        let child = parent
            .derive_child(request(
                "child",
                tools(&["read"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .unwrap();
        // Dropping the child leaves the parent untouched.
        drop(child);
        let another = parent
            .derive_child(request(
                "child2",
                tools(&["read"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                Vec::new(),
            ))
            .expect("parent must remain usable after child termination");
        assert_eq!(another.parent_session_id.as_deref(), Some("parent"));
    }

    #[test]
    fn workspace_roots_must_stay_inside_parent() {
        let parent = owning_root("parent", BTreeSet::new(), vec!["/workspace".to_string()]);
        let parent = parent
            .derive_child(request(
                "parent2",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                vec!["/workspace".to_string()],
            ))
            .unwrap();
        let child = parent
            .derive_child(request(
                "child",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                vec!["/workspace/./sub/".to_string()],
            ))
            .expect("normalized subdirectory must be allowed");
        assert_eq!(child.workspace_roots, vec!["/workspace/./sub/"]);
        let error = parent
            .derive_child(request(
                "child2",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                vec!["/elsewhere".to_string()],
            ))
            .expect_err("outside root must fail");
        assert_eq!(
            error,
            DelegationError::WorkspaceWidening("/elsewhere".to_string())
        );
    }

    #[test]
    fn workspace_roots_reject_parent_dir_escape() {
        let parent = owning_root("parent", BTreeSet::new(), vec!["/workspace".to_string()]);
        let error = parent
            .derive_child(request(
                "child",
                None,
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::new(),
                vec!["/workspace/../etc".to_string()],
            ))
            .expect_err("parent-dir traversal must never widen a workspace root");
        assert_eq!(
            error,
            DelegationError::WorkspaceWidening("/workspace/../etc".to_string())
        );
    }

    #[test]
    fn replay_preserves_delegation() {
        let parent = owning_root(
            "parent",
            BTreeSet::from([secret("k1")]),
            vec!["/w".to_string()],
        );
        let parent = parent
            .derive_child(request(
                "parent2",
                tools(&["read"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget::default(),
                BTreeSet::from([secret("k1")]),
                vec!["/w".to_string()],
            ))
            .unwrap();
        let child = parent
            .derive_child(request(
                "child",
                tools(&["read"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget {
                    wall_time_seconds: Some(30),
                    memory_bytes: None,
                    max_processes: None,
                    gpu_allowed: false,
                },
                BTreeSet::from([secret("k1")]),
                vec!["/w".to_string()],
            ))
            .unwrap();
        // "Replay": derive the same child again from the same parent — the
        // result must be identical.
        let replay = parent
            .derive_child(request(
                "child",
                tools(&["read"]),
                SandboxPermission::ReadOnly,
                ApprovalPolicy::Never,
                ResourceBudget {
                    wall_time_seconds: Some(30),
                    memory_bytes: None,
                    max_processes: None,
                    gpu_allowed: false,
                },
                BTreeSet::from([secret("k1")]),
                vec!["/w".to_string()],
            ))
            .unwrap();
        assert_eq!(child, replay);
    }
}
