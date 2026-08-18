use std::fmt;

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
    pub fn parse(value: &str) -> Result<Self, ApprovalError>
    {
        match value.trim()
        {
            "read-only" => Ok(Self::ReadOnly),
            "workspace-write" => Ok(Self::WorkspaceWrite),
            "danger-full-access" => Ok(Self::DangerFullAccess),
            other => Err(ApprovalError::InvalidSandboxPermission(other.to_string())),
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
pub struct ApprovalRequest
{
    pub call_id: String,
    pub tool: String,
    pub current: SandboxPermission,
    pub requested: SandboxPermission,
    pub justification: String,
}

impl ApprovalRequest
{
    pub fn new(
        call_id: impl Into<String>,
        tool: impl Into<String>,
        current: SandboxPermission,
        requested: SandboxPermission,
        justification: impl Into<String>,
    ) -> Result<Self, ApprovalError>
    {
        let call_id = call_id.into();
        if call_id.trim().is_empty()
        {
            return Err(ApprovalError::MissingCallId);
        }

        let tool = tool.into();
        if tool.trim().is_empty()
        {
            return Err(ApprovalError::MissingTool);
        }

        let justification = justification.into();
        let justification = justification.trim();
        if justification.is_empty()
        {
            return Err(ApprovalError::EmptyJustification);
        }

        if !current.can_escalate_to(requested)
        {
            return Err(ApprovalError::NotAnEscalation { current, requested });
        }

        Ok(Self {
            call_id,
            tool,
            current,
            requested,
            justification: justification.to_string(),
        })
    }

    /// Parse optional escalation metadata while enforcing that permission and
    /// justification are either both absent or both present.
    pub fn from_metadata(
        call_id: impl Into<String>,
        tool: impl Into<String>,
        current: SandboxPermission,
        sandbox_permissions: Option<&str>,
        justification: Option<&str>,
    ) -> Result<Option<Self>, ApprovalError>
    {
        match (sandbox_permissions, justification)
        {
            (None, None) => Ok(None),
            (Some(_), None) => Err(ApprovalError::MissingJustification),
            (None, Some(_)) => Err(ApprovalError::MissingSandboxPermissions),
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

/// Result vocabulary returned by an approval channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOutcome
{
    AllowedOnce,
    Rejected,
    Cancelled,
    Unavailable,
}

/// Synchronous seam for whichever UI/session layer owns user approval.
pub trait ApprovalService
{
    fn request_approval(&self, request: &ApprovalRequest) -> ApprovalOutcome;
}

/// Safe default while no interactive approval channel is installed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoApprovalService;

impl ApprovalService for NoApprovalService
{
    fn request_approval(&self, _request: &ApprovalRequest) -> ApprovalOutcome
    {
        ApprovalOutcome::Unavailable
    }
}

/// Ask for approval for this exact request. An allowed result is deliberately
/// not converted into a reusable capability: callers must invoke this function
/// again for every later tool call.
pub fn require_one_shot_approval(
    service: &dyn ApprovalService,
    request: &ApprovalRequest,
) -> Result<(), ApprovalError>
{
    match service.request_approval(request)
    {
        ApprovalOutcome::AllowedOnce => Ok(()),
        ApprovalOutcome::Rejected => Err(ApprovalError::Rejected),
        ApprovalOutcome::Cancelled => Err(ApprovalError::Cancelled),
        ApprovalOutcome::Unavailable => Err(ApprovalError::Unavailable),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalError
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
    Rejected,
    Cancelled,
    Unavailable,
}

impl fmt::Display for ApprovalError
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self
        {
            Self::InvalidSandboxPermission(value) => write!(
                f,
                "Invalid sandbox permission {value:?}; expected read-only, workspace-write, or danger-full-access"
            ),
            Self::MissingCallId => write!(f, "Approval request is missing a tool call id"),
            Self::MissingTool => write!(f, "Approval request is missing a tool name"),
            Self::MissingSandboxPermissions => write!(
                f,
                "sandbox_permissions and justification must be supplied together"
            ),
            Self::MissingJustification => write!(
                f,
                "sandbox_permissions and justification must be supplied together"
            ),
            Self::EmptyJustification => write!(f, "Approval justification must not be empty"),
            Self::NotAnEscalation { current, requested } => write!(
                f,
                "Sandbox permission {:?} cannot escalate to {:?}",
                current.label(),
                requested.label()
            ),
            Self::Rejected => write!(f, "Sandbox escalation was rejected"),
            Self::Cancelled => write!(f, "Sandbox escalation was cancelled"),
            Self::Unavailable => write!(f, "Sandbox escalation approval is unavailable"),
        }
    }
}

impl std::error::Error for ApprovalError {}

#[cfg(test)]
mod tests
{
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingApproval
    {
        calls: AtomicUsize,
        outcome: ApprovalOutcome,
    }

    impl CountingApproval
    {
        fn new(outcome: ApprovalOutcome) -> Self
        {
            Self {
                calls: AtomicUsize::new(0),
                outcome,
            }
        }
    }

    impl ApprovalService for CountingApproval
    {
        fn request_approval(&self, _request: &ApprovalRequest) -> ApprovalOutcome
        {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.outcome
        }
    }

    fn request() -> ApprovalRequest
    {
        ApprovalRequest::new(
            "call-7",
            "build",
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
            "Dependency build requires access outside the workspace",
        )
        .unwrap()
    }

    #[test]
    fn permission_vocabulary_matches_harness()
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
    fn metadata_must_pair_permission_and_justification()
    {
        let missing_justification = ApprovalRequest::from_metadata(
            "call-1",
            "build",
            SandboxPermission::ReadOnly,
            Some("workspace-write"),
            None,
        )
        .unwrap_err();
        assert_eq!(missing_justification, ApprovalError::MissingJustification);

        let missing_permission = ApprovalRequest::from_metadata(
            "call-1",
            "build",
            SandboxPermission::ReadOnly,
            None,
            Some("needs build output"),
        )
        .unwrap_err();
        assert_eq!(
            missing_permission,
            ApprovalError::MissingSandboxPermissions
        );
    }

    #[test]
    fn empty_justification_is_refused()
    {
        let error = ApprovalRequest::new(
            "call-1",
            "build",
            SandboxPermission::ReadOnly,
            SandboxPermission::WorkspaceWrite,
            "   ",
        )
        .unwrap_err();
        assert_eq!(error, ApprovalError::EmptyJustification);
    }

    #[test]
    fn non_widening_request_is_refused_before_approval()
    {
        let error = ApprovalRequest::new(
            "call-1",
            "build",
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::ReadOnly,
            "not actually wider",
        )
        .unwrap_err();
        assert!(matches!(error, ApprovalError::NotAnEscalation { .. }));
    }

    #[test]
    fn missing_approval_channel_fails_closed()
    {
        assert_eq!(
            require_one_shot_approval(&NoApprovalService, &request()),
            Err(ApprovalError::Unavailable)
        );
    }

    #[test]
    fn allowed_once_is_not_cached()
    {
        let service = CountingApproval::new(ApprovalOutcome::AllowedOnce);
        let request = request();
        require_one_shot_approval(&service, &request).unwrap();
        require_one_shot_approval(&service, &request).unwrap();
        assert_eq!(service.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn rejected_cancelled_and_unavailable_are_fail_closed()
    {
        for (outcome, expected) in [
            (ApprovalOutcome::Rejected, ApprovalError::Rejected),
            (ApprovalOutcome::Cancelled, ApprovalError::Cancelled),
            (ApprovalOutcome::Unavailable, ApprovalError::Unavailable),
        ]
        {
            let service = CountingApproval::new(outcome);
            assert_eq!(require_one_shot_approval(&service, &request()), Err(expected));
        }
    }
}
