use super::approval_request::ApprovalRequestId;
use super::permission::{ApprovalChoice, ApprovalOutcome, ScopedToolApprover, ToolApprover};
use super::sandbox_approval::{SandboxApprovalRequest, SandboxApprovalService};
use super::tool_runtime::ToolCall;
use super::tools::Tool;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Approval channel whose lifecycle is being observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalChannel {
    Tool,
    Sandbox,
}

/// Lifecycle phase for one correlated approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalLifecycle {
    Requested,
    Resolved,
}

/// Closed resolution vocabulary shared by supervision clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalResolution {
    AllowedOnce,
    AllowedSession,
    AllowedPersistent,
    Rejected,
    Cancelled,
    Unavailable,
}

/// One machine-readable approval lifecycle event.
///
/// `request_id` is the strong [`ApprovalRequestId`] shared by a Requested /
/// Resolved pair; `call_id` remains the correlation to the actual tool
/// invocation. Sandbox events additionally carry the requested permission and
/// justification. Tool events may carry a subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalAuditEvent {
    pub sequence: u64,
    pub channel: ApprovalChannel,
    pub lifecycle: ApprovalLifecycle,
    pub request_id: Option<ApprovalRequestId>,
    pub call_id: String,
    pub tool: String,
    pub subject: Option<String>,
    pub requested_permission: Option<String>,
    pub justification: Option<String>,
    pub resolution: Option<ApprovalResolution>,
}

impl ApprovalAuditEvent {
    pub fn tool_requested(call_id: &str, tool: &str, subject: &str) -> Self {
        Self {
            sequence: 0,
            channel: ApprovalChannel::Tool,
            lifecycle: ApprovalLifecycle::Requested,
            request_id: None,
            call_id: call_id.to_string(),
            tool: tool.to_string(),
            subject: (!subject.is_empty()).then(|| subject.to_string()),
            requested_permission: None,
            justification: None,
            resolution: None,
        }
    }

    pub fn tool_requested_with_id(
        request_id: ApprovalRequestId,
        call_id: &str,
        tool: &str,
        subject: &str,
    ) -> Self {
        let mut event = Self::tool_requested(call_id, tool, subject);
        event.request_id = Some(request_id);
        event
    }

    pub fn tool_resolved(
        call_id: &str,
        tool: &str,
        subject: &str,
        resolution: ApprovalResolution,
    ) -> Self {
        let mut event = Self::tool_requested(call_id, tool, subject);
        event.lifecycle = ApprovalLifecycle::Resolved;
        event.resolution = Some(resolution);
        event
    }

    pub fn tool_resolved_with_id(
        request_id: ApprovalRequestId,
        call_id: &str,
        tool: &str,
        subject: &str,
        resolution: ApprovalResolution,
    ) -> Self {
        let mut event = Self::tool_resolved(call_id, tool, subject, resolution);
        event.request_id = Some(request_id);
        event
    }

    pub fn sandbox_requested(request: &SandboxApprovalRequest) -> Self {
        Self {
            sequence: 0,
            channel: ApprovalChannel::Sandbox,
            lifecycle: ApprovalLifecycle::Requested,
            request_id: None,
            call_id: request.call_id.clone(),
            tool: request.tool.clone(),
            subject: None,
            requested_permission: Some(request.requested.label().to_string()),
            justification: Some(request.justification.clone()),
            resolution: None,
        }
    }

    pub fn sandbox_requested_with_id(
        request_id: ApprovalRequestId,
        request: &SandboxApprovalRequest,
    ) -> Self {
        let mut event = Self::sandbox_requested(request);
        event.request_id = Some(request_id);
        event
    }

    pub fn sandbox_resolved(
        request: &SandboxApprovalRequest,
        resolution: ApprovalResolution,
    ) -> Self {
        let mut event = Self::sandbox_requested(request);
        event.lifecycle = ApprovalLifecycle::Resolved;
        event.resolution = Some(resolution);
        event
    }

    pub fn sandbox_resolved_with_id(
        request_id: ApprovalRequestId,
        request: &SandboxApprovalRequest,
        resolution: ApprovalResolution,
    ) -> Self {
        let mut event = Self::sandbox_resolved(request, resolution);
        event.request_id = Some(request_id);
        event
    }
}

/// Observation seam for approval lifecycle events.
///
/// Wrappers in this module treat recording as part of the approval contract:
/// once a sink is installed, a recording failure fails the approval closed.
pub trait ApprovalAuditSink: Send + Sync {
    fn record(&self, event: ApprovalAuditEvent) -> Result<(), String>;
}

/// Bounded, process-local journal suitable for a supervision UI.
pub struct InMemoryApprovalAudit {
    capacity: usize,
    next_sequence: AtomicU64,
    events: Mutex<VecDeque<ApprovalAuditEvent>>,
}

impl InMemoryApprovalAudit {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0
        {
            return Err("approval audit capacity must be greater than zero".to_string());
        }
        Ok(Self {
            capacity,
            next_sequence: AtomicU64::new(1),
            events: Mutex::new(VecDeque::with_capacity(capacity)),
        })
    }

    pub fn snapshot(&self) -> Result<Vec<ApprovalAuditEvent>, String> {
        Ok(self
            .events
            .lock()
            .map_err(|_| "approval audit journal is unavailable".to_string())?
            .iter()
            .cloned()
            .collect())
    }

    pub fn clear(&self) -> Result<(), String> {
        self.events
            .lock()
            .map_err(|_| "approval audit journal is unavailable".to_string())?
            .clear();
        Ok(())
    }
}

impl ApprovalAuditSink for InMemoryApprovalAudit {
    fn record(&self, mut event: ApprovalAuditEvent) -> Result<(), String> {
        event.sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        let mut events = self
            .events
            .lock()
            .map_err(|_| "approval audit journal is unavailable".to_string())?;
        if events.len() == self.capacity
        {
            events.pop_front();
        }
        events.push_back(event);
        Ok(())
    }
}

/// One-shot tool approver wrapper that emits a correlated request/resolution pair.
pub struct AuditedToolApprover {
    inner: Arc<dyn ToolApprover>,
    audit: Arc<dyn ApprovalAuditSink>,
}

impl AuditedToolApprover {
    pub fn new(inner: Arc<dyn ToolApprover>, audit: Arc<dyn ApprovalAuditSink>) -> Self {
        Self { inner, audit }
    }
}

impl ToolApprover for AuditedToolApprover {
    fn approve(
        &self,
        call: &ToolCall,
        tool: &Tool,
        subject: &str,
    ) -> Result<ApprovalOutcome, String> {
        let request_id = ApprovalRequestId::generate();
        self.audit
            .record(ApprovalAuditEvent::tool_requested_with_id(
                request_id.clone(),
                &call.id,
                &call.tool,
                subject,
            ))?;
        let outcome = match self.inner.approve(call, tool, subject)
        {
            Ok(outcome) => outcome,
            Err(error) =>
            {
                self.audit
                    .record(ApprovalAuditEvent::tool_resolved_with_id(
                        request_id,
                        &call.id,
                        &call.tool,
                        subject,
                        ApprovalResolution::Unavailable,
                    ))?;
                return Err(error);
            },
        };
        self.audit
            .record(ApprovalAuditEvent::tool_resolved_with_id(
                request_id,
                &call.id,
                &call.tool,
                subject,
                resolution_from_outcome(outcome),
            ))?;
        Ok(outcome)
    }
}

/// Session-aware tool approver wrapper with the same correlated lifecycle.
pub struct AuditedScopedToolApprover {
    inner: Arc<dyn ScopedToolApprover>,
    audit: Arc<dyn ApprovalAuditSink>,
}

impl AuditedScopedToolApprover {
    pub fn new(inner: Arc<dyn ScopedToolApprover>, audit: Arc<dyn ApprovalAuditSink>) -> Self {
        Self { inner, audit }
    }
}

impl ScopedToolApprover for AuditedScopedToolApprover {
    fn approve(
        &self,
        call: &ToolCall,
        tool: &Tool,
        subject: &str,
    ) -> Result<ApprovalChoice, String> {
        let request_id = ApprovalRequestId::generate();
        self.audit
            .record(ApprovalAuditEvent::tool_requested_with_id(
                request_id.clone(),
                &call.id,
                &call.tool,
                subject,
            ))?;
        let choice = match self.inner.approve(call, tool, subject)
        {
            Ok(choice) => choice,
            Err(error) =>
            {
                self.audit
                    .record(ApprovalAuditEvent::tool_resolved_with_id(
                        request_id,
                        &call.id,
                        &call.tool,
                        subject,
                        ApprovalResolution::Unavailable,
                    ))?;
                return Err(error);
            },
        };
        let resolution = match choice
        {
            ApprovalChoice::Once => ApprovalResolution::AllowedOnce,
            ApprovalChoice::Session => ApprovalResolution::AllowedSession,
            ApprovalChoice::Always => ApprovalResolution::AllowedPersistent,
            ApprovalChoice::Decline => ApprovalResolution::Rejected,
        };
        self.audit
            .record(ApprovalAuditEvent::tool_resolved_with_id(
                request_id, &call.id, &call.tool, subject, resolution,
            ))?;
        Ok(choice)
    }
}

/// Sandbox escalation service wrapper that preserves the exact one-shot result.
pub struct AuditedSandboxApprovalService {
    inner: Arc<dyn SandboxApprovalService>,
    audit: Arc<dyn ApprovalAuditSink>,
}

impl AuditedSandboxApprovalService {
    pub fn new(inner: Arc<dyn SandboxApprovalService>, audit: Arc<dyn ApprovalAuditSink>) -> Self {
        Self { inner, audit }
    }
}

impl SandboxApprovalService for AuditedSandboxApprovalService {
    fn request_approval(
        &self,
        request: &SandboxApprovalRequest,
    ) -> Result<ApprovalOutcome, String> {
        let request_id = ApprovalRequestId::generate();
        self.audit
            .record(ApprovalAuditEvent::sandbox_requested_with_id(
                request_id.clone(),
                request,
            ))?;
        let outcome = match self.inner.request_approval(request)
        {
            Ok(outcome) => outcome,
            Err(error) =>
            {
                self.audit
                    .record(ApprovalAuditEvent::sandbox_resolved_with_id(
                        request_id,
                        request,
                        ApprovalResolution::Unavailable,
                    ))?;
                return Err(error);
            },
        };
        self.audit
            .record(ApprovalAuditEvent::sandbox_resolved_with_id(
                request_id,
                request,
                resolution_from_outcome(outcome),
            ))?;
        Ok(outcome)
    }
}

fn resolution_from_outcome(outcome: ApprovalOutcome) -> ApprovalResolution {
    match outcome
    {
        ApprovalOutcome::AllowedOnce => ApprovalResolution::AllowedOnce,
        ApprovalOutcome::Rejected => ApprovalResolution::Rejected,
        ApprovalOutcome::Cancelled => ApprovalResolution::Cancelled,
        ApprovalOutcome::Unavailable => ApprovalResolution::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::super::sandbox_approval::SandboxPermission;
    use super::*;
    use std::collections::HashMap;

    fn noop(_params: HashMap<String, String>) -> String {
        "ok".to_string()
    }

    fn tool() -> Tool {
        Tool {
            name: "build",
            description: "audit test tool",
            parameters: Vec::new(),
            execute: noop,
        }
    }

    struct AllowOnce;

    impl ToolApprover for AllowOnce {
        fn approve(
            &self,
            _call: &ToolCall,
            _tool: &Tool,
            _subject: &str,
        ) -> Result<ApprovalOutcome, String> {
            Ok(ApprovalOutcome::AllowedOnce)
        }
    }

    impl SandboxApprovalService for AllowOnce {
        fn request_approval(
            &self,
            _request: &SandboxApprovalRequest,
        ) -> Result<ApprovalOutcome, String> {
            Ok(ApprovalOutcome::AllowedOnce)
        }
    }

    struct AllowSession;

    impl ScopedToolApprover for AllowSession {
        fn approve(
            &self,
            _call: &ToolCall,
            _tool: &Tool,
            _subject: &str,
        ) -> Result<ApprovalChoice, String> {
            Ok(ApprovalChoice::Session)
        }
    }

    struct AllowPersistent;

    impl ScopedToolApprover for AllowPersistent {
        fn approve(
            &self,
            _call: &ToolCall,
            _tool: &Tool,
            _subject: &str,
        ) -> Result<ApprovalChoice, String> {
            Ok(ApprovalChoice::Always)
        }
    }

    #[test]
    fn bounded_journal_is_ordered_and_evicts_oldest() {
        let audit = InMemoryApprovalAudit::new(2).unwrap();
        audit
            .record(ApprovalAuditEvent::tool_requested("c1", "build", "core"))
            .unwrap();
        audit
            .record(ApprovalAuditEvent::tool_resolved(
                "c1",
                "build",
                "core",
                ApprovalResolution::AllowedOnce,
            ))
            .unwrap();
        let request = SandboxApprovalRequest::new(
            "c2",
            "test",
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
            "needs external fixture",
        )
        .unwrap();
        audit
            .record(ApprovalAuditEvent::sandbox_requested(&request))
            .unwrap();
        let events = audit.snapshot().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, 2);
        assert_eq!(events[1].sequence, 3);
        assert_eq!(events[1].call_id, "c2");
    }

    #[test]
    fn audited_one_shot_tool_approval_records_pair() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedToolApprover::new(Arc::new(AllowOnce), audit.clone());
        let call = ToolCall::new("call-1", "build", HashMap::new());
        assert_eq!(
            approver.approve(&call, &tool(), "scirust-core").unwrap(),
            ApprovalOutcome::AllowedOnce
        );
        let events = audit.snapshot().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].lifecycle, ApprovalLifecycle::Requested);
        assert_eq!(events[1].lifecycle, ApprovalLifecycle::Resolved);
        assert_eq!(events[1].resolution, Some(ApprovalResolution::AllowedOnce));
    }

    #[test]
    fn audited_scoped_tool_approval_records_session_resolution() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedScopedToolApprover::new(Arc::new(AllowSession), audit.clone());
        let call = ToolCall::new("call-2", "build", HashMap::new());
        assert_eq!(
            approver.approve(&call, &tool(), "scirust-core").unwrap(),
            ApprovalChoice::Session
        );
        let events = audit.snapshot().unwrap();
        assert_eq!(
            events[1].resolution,
            Some(ApprovalResolution::AllowedSession)
        );
    }

    #[test]
    fn audited_scoped_tool_approval_records_persistent_resolution() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedScopedToolApprover::new(Arc::new(AllowPersistent), audit.clone());
        let call = ToolCall::new("call-persist", "build", HashMap::new());
        assert_eq!(
            approver.approve(&call, &tool(), "scirust-core").unwrap(),
            ApprovalChoice::Always
        );
        let events = audit.snapshot().unwrap();
        assert_eq!(
            events[1].resolution,
            Some(ApprovalResolution::AllowedPersistent)
        );
    }

    #[test]
    fn audited_sandbox_approval_records_requested_mode_and_justification() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedSandboxApprovalService::new(Arc::new(AllowOnce), audit.clone());
        let request = SandboxApprovalRequest::new(
            "call-3",
            "build",
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
            "needs external linker",
        )
        .unwrap();
        assert_eq!(
            approver.request_approval(&request).unwrap(),
            ApprovalOutcome::AllowedOnce
        );
        let events = audit.snapshot().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0].requested_permission.as_deref(),
            Some("danger-full-access")
        );
        assert_eq!(
            events[0].justification.as_deref(),
            Some("needs external linker")
        );
    }

    #[test]
    fn audited_pair_shares_one_request_id() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedToolApprover::new(Arc::new(AllowOnce), audit.clone());
        let call = ToolCall::new("call-1", "build", HashMap::new());
        approver.approve(&call, &tool(), "scirust-core").unwrap();
        let events = audit.snapshot().unwrap();
        assert_eq!(events.len(), 2);
        let requested = &events[0];
        let resolved = &events[1];
        assert_eq!(requested.lifecycle, ApprovalLifecycle::Requested);
        assert_eq!(resolved.lifecycle, ApprovalLifecycle::Resolved);
        let request_id = requested.request_id.as_ref().expect("requested id");
        assert_eq!(
            resolved.request_id.as_ref(),
            Some(request_id),
            "the pair must share the same ApprovalRequestId"
        );
        assert_eq!(requested.call_id, "call-1");
        assert_eq!(resolved.call_id, "call-1");
        assert!(request_id.is_valid());
    }

    #[test]
    fn same_call_twice_gets_two_distinct_request_ids() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedToolApprover::new(Arc::new(AllowOnce), audit.clone());
        let call = ToolCall::new("call-1", "build", HashMap::new());
        approver.approve(&call, &tool(), "scirust-core").unwrap();
        approver.approve(&call, &tool(), "scirust-core").unwrap();
        let events = audit.snapshot().unwrap();
        let first = events[0].request_id.as_ref().unwrap();
        let second = events[2].request_id.as_ref().unwrap();
        assert_ne!(first, second, "each new question needs a fresh id");
        assert_eq!(events[0].call_id, events[2].call_id);
    }

    #[test]
    fn audited_sandbox_pair_shares_one_request_id() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedSandboxApprovalService::new(Arc::new(AllowOnce), audit.clone());
        let request = SandboxApprovalRequest::new(
            "call-3",
            "build",
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
            "needs external linker",
        )
        .unwrap();
        approver.request_approval(&request).unwrap();
        let events = audit.snapshot().unwrap();
        let requested = events[0].request_id.as_ref().unwrap();
        assert_eq!(events[1].request_id.as_ref(), Some(requested));
        assert_eq!(events[0].call_id, "call-3");
    }

    #[test]
    fn sandbox_pair_preserves_call_id_and_tool() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let approver = AuditedSandboxApprovalService::new(Arc::new(AllowOnce), audit.clone());
        let request = SandboxApprovalRequest::new(
            "call-9",
            "build",
            SandboxPermission::ReadOnly,
            SandboxPermission::WorkspaceWrite,
            "needs wider fs",
        )
        .unwrap();
        approver.request_approval(&request).unwrap();
        let events = audit.snapshot().unwrap();
        assert_eq!(events[0].call_id, "call-9");
        assert_eq!(events[0].tool, "build");
        assert_eq!(events[1].call_id, "call-9");
        assert_eq!(events[1].tool, "build");
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert!(InMemoryApprovalAudit::new(0).is_err());
    }

    #[test]
    fn clear_removes_all_events() {
        let audit = InMemoryApprovalAudit::new(4).unwrap();
        audit
            .record(ApprovalAuditEvent::tool_requested("c1", "build", ""))
            .unwrap();
        audit.clear().unwrap();
        assert!(audit.snapshot().unwrap().is_empty());
    }
}
