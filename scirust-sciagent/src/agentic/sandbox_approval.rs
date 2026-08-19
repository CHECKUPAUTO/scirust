use super::permission::ApprovalOutcome;
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
}
