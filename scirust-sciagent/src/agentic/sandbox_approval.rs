use super::permission::{ApprovalOutcome, PermissionGate};
use super::tool_runtime::{ToolCall, ToolPolicy};
use super::tools::Tool;
use std::fmt;
use std::sync::Arc;

/// Sandbox permission vocabulary accepted by one-shot escalation requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxPermission
{
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl SandboxPermission
{
    pub fn parse(value: &str) -> Result<Self, SandboxApprovalError>
    {
        match value.trim()
        {
            "read-only" => Ok(Self::ReadOnly),
            "workspace-write" => Ok(Self::WorkspaceWrite),
            "danger-full-access" => Ok(Self::DangerFullAccess),
            other => Err(SandboxApprovalError::InvalidSandboxPermission(
                other.to_string(),
            )),
        }
    }

    pub fn label(self) -> &'static str
    {
        match self
        {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }

    pub fn can_escalate_to(self, requested: Self) -> bool
    {
        matches!(
            (self, requested),
            (Self::ReadOnly, Self::WorkspaceWrite | Self::DangerFullAccess)
                | (Self::WorkspaceWrite, Self::DangerFullAccess)
        )
    }
}

/// A self-contained request to widen one exact tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxApprovalRequest
{
    pub call_id: String,
    pub tool: String,
    pub current: SandboxPermission,
    pub requested: SandboxPermission,
    pub justification: String,
}

impl SandboxApprovalRequest
{
    pub fn new(
        call_id: impl Into<String>,
        tool: impl Into<String>,
        current: SandboxPermission,
        requested: SandboxPermission,
        justification: impl Into<String>,
    ) -> Result<Self, SandboxApprovalError>
    {
        let call_id = call_id.into();
        if call_id.trim().is_empty()
        {
            return Err(SandboxApprovalError::MissingCallId);
        }

        let tool = tool.into();
        if tool.trim().is_empty()
        {
            return Err(SandboxApprovalError::MissingTool);
        }

        let justification = justification.into();
        let justification = justification.trim();
        if justification.is_empty()
        {
            return Err(SandboxApprovalError::EmptyJustification);
        }

        if !current.can_escalate_to(requested)
        {
            return Err(SandboxApprovalError::NotAnEscalation { current, requested });
        }

        Ok(Self {
            call_id,
            tool,
            current,
            requested,
            justification: justification.to_string(),
        })
    }

    /// Parse optional metadata while requiring permission and justification to
    /// be both absent or both present.
    pub fn from_metadata(
        call_id: impl Into<String>,
        tool: impl Into<String>,
        current: SandboxPermission,
        sandbox_permissions: Option<&str>,
        justification: Option<&str>,
    ) -> Result<Option<Self>, SandboxApprovalError>
    {
        match (sandbox_permissions, justification)
        {
            (None, None) => Ok(None),
            (Some(_), None) => Err(SandboxApprovalError::MissingJustification),
            (None, Some(_)) => Err(SandboxApprovalError::MissingSandboxPermissions),
            (Some(requested), Some(justification)) => Self::new(
                call_id,
                tool,
                current,
                SandboxPermission::parse(requested)?,
                justification,
            )
            .map(Some),
        }
    }
}

/// Synchronous seam for the UI/session layer that owns sandbox escalation.
pub trait SandboxApprovalService: Send + Sync
{
    fn request_approval(
        &self,
        request: &SandboxApprovalRequest,
    ) -> Result<ApprovalOutcome, String>;
}

/// Safe default when no interactive escalation channel is installed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSandboxApprovalService;

impl SandboxApprovalService for NoSandboxApprovalService
{
    fn request_approval(
        &self,
        _request: &SandboxApprovalRequest,
    ) -> Result<ApprovalOutcome, String>
    {
        Ok(ApprovalOutcome::Unavailable)
    }
}

/// Composite runtime policy that keeps ordinary tool permission and sandbox
/// escalation on separate approval channels.
///
/// A remembered [`PermissionGate`] session grant can satisfy an ordinary tool
/// `Ask`, but it never grants a sandbox widening request. Widening is always
/// resolved independently and only `AllowedOnce` permits the current call.
#[derive(Clone)]
pub struct SandboxPermissionGate
{
    permission: PermissionGate,
    sandbox_approver: Option<Arc<dyn SandboxApprovalService>>,
}

impl SandboxPermissionGate
{
    pub fn new(permission: PermissionGate) -> Self
    {
        Self {
            permission,
            sandbox_approver: None,
        }
    }

    pub fn with_approval_service(
        permission: PermissionGate,
        sandbox_approver: Arc<dyn SandboxApprovalService>,
    ) -> Self
    {
        Self {
            permission,
            sandbox_approver: Some(sandbox_approver),
        }
    }

    pub fn permission_gate(&self) -> &PermissionGate
    {
        &self.permission
    }
}

impl ToolPolicy for SandboxPermissionGate
{
    fn before_execute(&self, call: &ToolCall, tool: &Tool) -> Result<(), String>
    {
        self.permission.before_execute(call, tool)
    }

    fn approve_sandbox_escalation(
        &self,
        _call: &ToolCall,
        _tool: &Tool,
        request: &SandboxApprovalRequest,
    ) -> Result<(), String>
    {
        let Some(approver) = self.sandbox_approver.as_ref()
        else
        {
            return Err(
                "sandbox escalation approval required but no sandbox approval service is available; refusing to execute"
                    .to_string(),
            );
        };

        let outcome = approver
            .request_approval(request)
            .unwrap_or(ApprovalOutcome::Unavailable);
        match outcome
        {
            ApprovalOutcome::AllowedOnce => Ok(()),
            ApprovalOutcome::Rejected => Err(
                "the user rejected this sandbox escalation; do not retry it unchanged".to_string(),
            ),
            ApprovalOutcome::Cancelled => Err(
                "sandbox escalation approval was cancelled; do not retry it unchanged".to_string(),
            ),
            ApprovalOutcome::Unavailable => {
                Err("sandbox escalation approval is unavailable; refusing to execute".to_string())
            },
        }
    }

    fn after_execute(&self, call: &ToolCall, tool: &Tool, output: &str)
    {
        self.permission.after_execute(call, tool, output);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxApprovalError
{
    InvalidSandboxPermission(String),
    MissingCallId,
    MissingTool,
    MissingSandboxPermissions,
    MissingJustification,
    EmptyJustification,
    NotAnEscalation {
        current: SandboxPermission,
        requested: SandboxPermission,
    },
}

impl fmt::Display for SandboxApprovalError
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self
        {
            Self::InvalidSandboxPermission(value) => write!(
                f,
                "Invalid sandbox permission {value:?}; expected read-only, workspace-write, or danger-full-access"
            ),
            Self::MissingCallId => write!(f, "Sandbox approval request is missing a tool call id"),
            Self::MissingTool => write!(f, "Sandbox approval request is missing a tool name"),
            Self::MissingSandboxPermissions | Self::MissingJustification => write!(
                f,
                "sandbox_permissions and justification must be supplied together"
            ),
            Self::EmptyJustification => {
                write!(f, "Sandbox approval justification must not be empty")
            },
            Self::NotAnEscalation { current, requested } => write!(
                f,
                "Sandbox permission {:?} cannot escalate to {:?}",
                current.label(),
                requested.label()
            ),
        }
    }
}

impl std::error::Error for SandboxApprovalError {}

#[cfg(test)]
mod tests
{
    use super::*;
    use super::super::permission::{PermissionDecision, PermissionPolicy};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn request(
        current: SandboxPermission,
        requested: SandboxPermission,
    ) -> Result<SandboxApprovalRequest, SandboxApprovalError>
    {
        SandboxApprovalRequest::new(
            "call-7",
            "build",
            current,
            requested,
            "The exact call needs a wider filesystem policy",
        )
    }

    fn noop(_params: HashMap<String, String>) -> String
    {
        "ok".to_string()
    }

    fn synthetic_tool() -> Tool
    {
        Tool {
            name: "build",
            description: "sandbox approval test tool",
            parameters: Vec::new(),
            execute: noop,
        }
    }

    struct FixedSandboxApprover
    {
        calls: AtomicUsize,
        outcome: ApprovalOutcome,
        fail: bool,
    }

    impl SandboxApprovalService for FixedSandboxApprover
    {
        fn request_approval(
            &self,
            _request: &SandboxApprovalRequest,
        ) -> Result<ApprovalOutcome, String>
        {
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

    fn allow_tool_permission() -> PermissionGate
    {
        PermissionGate::new(PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
    }

    #[test]
    fn permission_vocabulary_is_exact()
    {
        assert_eq!(
            SandboxPermission::parse("read-only").unwrap(),
            SandboxPermission::ReadOnly
        );
        assert_eq!(
            SandboxPermission::parse("workspace-write").unwrap(),
            SandboxPermission::WorkspaceWrite
        );
        assert_eq!(
            SandboxPermission::parse("danger-full-access").unwrap(),
            SandboxPermission::DangerFullAccess
        );
        assert!(SandboxPermission::parse("danger_full_access").is_err());
        assert!(SandboxPermission::parse("DANGER-FULL-ACCESS").is_err());
    }

    #[test]
    fn widening_matrix_is_strict()
    {
        use SandboxPermission::{DangerFullAccess, ReadOnly, WorkspaceWrite};

        assert!(ReadOnly.can_escalate_to(WorkspaceWrite));
        assert!(ReadOnly.can_escalate_to(DangerFullAccess));
        assert!(WorkspaceWrite.can_escalate_to(DangerFullAccess));
        assert!(!ReadOnly.can_escalate_to(ReadOnly));
        assert!(!WorkspaceWrite.can_escalate_to(ReadOnly));
        assert!(!WorkspaceWrite.can_escalate_to(WorkspaceWrite));
        assert!(!DangerFullAccess.can_escalate_to(ReadOnly));
        assert!(!DangerFullAccess.can_escalate_to(WorkspaceWrite));
        assert!(!DangerFullAccess.can_escalate_to(DangerFullAccess));
    }

    #[test]
    fn metadata_pair_is_atomic()
    {
        let current = SandboxPermission::ReadOnly;
        assert_eq!(
            SandboxApprovalRequest::from_metadata(
                "call-1",
                "build",
                current,
                Some("workspace-write"),
                None,
            )
            .unwrap_err(),
            SandboxApprovalError::MissingJustification
        );
        assert_eq!(
            SandboxApprovalRequest::from_metadata(
                "call-1",
                "build",
                current,
                None,
                Some("need output"),
            )
            .unwrap_err(),
            SandboxApprovalError::MissingSandboxPermissions
        );
    }

    #[test]
    fn empty_justification_is_refused()
    {
        let error = SandboxApprovalRequest::new(
            "call-1",
            "build",
            SandboxPermission::ReadOnly,
            SandboxPermission::WorkspaceWrite,
            "   ",
        )
        .unwrap_err();
        assert_eq!(error, SandboxApprovalError::EmptyJustification);
    }

    #[test]
    fn non_widening_requests_are_refused()
    {
        assert!(request(
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::ReadOnly,
        )
        .is_err());
        assert!(request(
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::WorkspaceWrite,
        )
        .is_err());
    }

    #[test]
    fn absent_service_is_unavailable()
    {
        let request = request(
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
        )
        .unwrap();
        assert_eq!(
            NoSandboxApprovalService.request_approval(&request).unwrap(),
            ApprovalOutcome::Unavailable
        );
    }

    #[test]
    fn composite_gate_fails_closed_without_sandbox_service()
    {
        let gate = SandboxPermissionGate::new(allow_tool_permission());
        let call = ToolCall::new("call-7", "build", HashMap::new());
        let request = request(
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
        )
        .unwrap();
        let error = gate
            .approve_sandbox_escalation(&call, &synthetic_tool(), &request)
            .expect_err("missing sandbox approval service must fail closed");
        assert!(error.contains("no sandbox approval service"), "{error}");
    }

    #[test]
    fn composite_gate_accepts_only_allowed_once()
    {
        for (outcome, allowed) in [
            (ApprovalOutcome::AllowedOnce, true),
            (ApprovalOutcome::Rejected, false),
            (ApprovalOutcome::Cancelled, false),
            (ApprovalOutcome::Unavailable, false),
        ]
        {
            let approver = Arc::new(FixedSandboxApprover {
                calls: AtomicUsize::new(0),
                outcome,
                fail: false,
            });
            let gate = SandboxPermissionGate::with_approval_service(
                allow_tool_permission(),
                approver.clone(),
            );
            let call = ToolCall::new("call-7", "build", HashMap::new());
            let request = request(
                SandboxPermission::WorkspaceWrite,
                SandboxPermission::DangerFullAccess,
            )
            .unwrap();
            assert_eq!(
                gate.approve_sandbox_escalation(&call, &synthetic_tool(), &request)
                    .is_ok(),
                allowed
            );
            assert_eq!(approver.calls.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn sandbox_approval_transport_error_fails_closed()
    {
        let approver = Arc::new(FixedSandboxApprover {
            calls: AtomicUsize::new(0),
            outcome: ApprovalOutcome::AllowedOnce,
            fail: true,
        });
        let gate =
            SandboxPermissionGate::with_approval_service(allow_tool_permission(), approver.clone());
        let call = ToolCall::new("call-7", "build", HashMap::new());
        let request = request(
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
        )
        .unwrap();
        let error = gate
            .approve_sandbox_escalation(&call, &synthetic_tool(), &request)
            .expect_err("transport failure must fail closed");
        assert!(error.contains("unavailable"), "{error}");
        assert_eq!(approver.calls.load(Ordering::SeqCst), 1);
    }
}
