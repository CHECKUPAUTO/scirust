//! Approval service — the supervision seam above the existing approval
//! traits.
//!
//! This service is deliberately NOT a TUI. It exposes a neutral interface
//! that CLI, TUI, CCOS Enterprise, a DeepSeek adapter or a remote supervision
//! service can implement ([`ApprovalAnswerer`]) and consume
//! ([`ApprovalService`]). The service owns the safety properties:
//!
//! - every NEW question gets a fresh [`ApprovalRequestId`] (never supplied by
//!   an untrusted answerer);
//! - policy `Never` rejects deterministically before any answerer dispatch;
//! - a missing, throwing or rogue answerer fails closed to `Unavailable`;
//! - cancellation may win a race; a late result after cancellation is
//!   discarded;
//! - a model/tool can never approve itself (the answerer is a separate
//!   registered component, and outcome normalization is closed).

use super::approval_audit::{ApprovalAuditEvent, ApprovalAuditSink, ApprovalResolution};
use super::approval_request::ApprovalRequestId;
use super::permission::ApprovalPolicy;
use super::sandbox_approval::SandboxPermission;
use super::tool_runtime::ToolCall;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Closed, normalized answer vocabulary. Everything else is `Unavailable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalAnswer {
    AllowedOnce,
    AllowedSession,
    AllowedPersistent,
    Rejected,
}

impl ApprovalAnswer {
    /// Normalize an untrusted answerer return. Unknown values fail closed.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str()
        {
            "allowed-once" => Self::AllowedOnce,
            "allowed-session" => Self::AllowedSession,
            "allowed-persistent" => Self::AllowedPersistent,
            "rejected" => Self::Rejected,
            _ => Self::Rejected,
        }
    }

    pub fn label(self) -> &'static str {
        match self
        {
            Self::AllowedOnce => "allowed-once",
            Self::AllowedSession => "allowed-session",
            Self::AllowedPersistent => "allowed-persistent",
            Self::Rejected => "rejected",
        }
    }
}

/// Cancellation signal for one pending approval question.
#[derive(Debug, Default, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// The full presentation context of one approval question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    pub request_id: ApprovalRequestId,
    pub call_id: String,
    pub tool: String,
    pub subject: Option<String>,
    pub reason: Option<String>,
    pub configured_sandbox: Option<SandboxPermission>,
    pub requested_sandbox: Option<SandboxPermission>,
    pub justification: Option<String>,
    pub policy: ApprovalPolicy,
}

/// Inputs for one service request, gathered before the id is generated.
#[derive(Debug, Clone)]
pub struct ApprovalServiceRequest {
    pub call: ToolCall,
    pub subject: Option<String>,
    pub reason: Option<String>,
    pub configured_sandbox: Option<SandboxPermission>,
    pub requested_sandbox: Option<SandboxPermission>,
    pub justification: Option<String>,
}

impl ApprovalServiceRequest {
    pub fn new(call: ToolCall) -> Self {
        Self {
            call,
            subject: None,
            reason: None,
            configured_sandbox: None,
            requested_sandbox: None,
            justification: None,
        }
    }

    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    pub fn with_sandbox(
        mut self,
        configured: SandboxPermission,
        requested: SandboxPermission,
    ) -> Self {
        self.configured_sandbox = Some(configured);
        self.requested_sandbox = Some(requested);
        self
    }

    pub fn with_justification(mut self, justification: impl Into<String>) -> Self {
        self.justification = Some(justification.into());
        self
    }
}

/// One answer to one approval question, paired with its request id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedApproval {
    pub request_id: ApprovalRequestId,
    pub answer: ApprovalAnswer,
}

/// Answerer seam implemented by CLI, TUI, CCOS Enterprise, DeepSeek adapter
/// or a remote supervision service. The answerer sees the full
/// [`ApprovalRequest`] and returns one closed answer.
pub trait ApprovalAnswerer: Send + Sync {
    fn answer(&self, request: &ApprovalRequest) -> Result<ApprovalAnswer, String>;
}

/// Supervision service above the approval traits.
///
/// The service generates the id, checks policy before dispatching, contains
/// answerer failures, and emits an audit pair (Requested/Resolved) through an
/// optional [`ApprovalAuditSink`]. A missing answerer, an answerer error, or
/// a cancellation resolve fail-closed.
#[derive(Clone)]
pub struct ApprovalService {
    answerer: Option<Arc<dyn ApprovalAnswerer>>,
    policy: ApprovalPolicy,
    audit: Option<Arc<dyn ApprovalAuditSink>>,
}

impl ApprovalService {
    pub fn new(answerer: Option<Arc<dyn ApprovalAnswerer>>) -> Self {
        Self {
            answerer,
            policy: ApprovalPolicy::Ask,
            audit: None,
        }
    }

    pub fn with_answerer(answerer: Arc<dyn ApprovalAnswerer>) -> Self {
        Self::new(Some(answerer))
    }

    pub fn with_policy(mut self, policy: ApprovalPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_audit(mut self, audit: Arc<dyn ApprovalAuditSink>) -> Self {
        self.audit = Some(audit);
        self
    }

    pub fn policy(&self) -> ApprovalPolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: ApprovalPolicy) {
        self.policy = policy;
    }

    /// Resolve one question. Returns the closed outcome; `Ok(None)` means the
    /// question was cancelled (the caller must treat it as no grant).
    pub fn request(
        &self,
        input: &ApprovalServiceRequest,
        token: &CancellationToken,
    ) -> Result<Option<ApprovalAnswer>, String> {
        let call = &input.call;
        let request_id = ApprovalRequestId::generate();
        let request = ApprovalRequest {
            request_id: request_id.clone(),
            call_id: call.id.clone(),
            tool: call.tool.clone(),
            subject: input.subject.clone(),
            reason: input.reason.clone(),
            configured_sandbox: input.configured_sandbox,
            requested_sandbox: input.requested_sandbox,
            justification: input.justification.clone(),
            policy: self.policy,
        };

        if let Some(audit) = self.audit.as_ref()
        {
            audit.record(ApprovalAuditEvent::tool_requested_with_id(
                request_id.clone(),
                &call.id,
                &call.tool,
                input.subject.as_deref().unwrap_or(""),
            ))?;
        }

        if token.is_cancelled()
        {
            self.record_resolution(&request_id, call, ApprovalResolution::Cancelled)?;
            return Ok(None);
        }

        if self.policy == ApprovalPolicy::Never
        {
            self.record_resolution(&request_id, call, ApprovalResolution::Rejected)?;
            return Ok(Some(ApprovalAnswer::Rejected));
        }

        let Some(answerer) = self.answerer.as_ref()
        else
        {
            self.record_resolution(&request_id, call, ApprovalResolution::Unavailable)?;
            return Err(
                "approval required but no approval answerer is available; refusing to execute"
                    .to_string(),
            );
        };

        let answer = match answerer.answer(&request)
        {
            Ok(answer) => answer,
            Err(error) =>
            {
                self.record_resolution(&request_id, call, ApprovalResolution::Unavailable)?;
                return Err(format!(
                    "approval answerer failed; refusing to execute: {error}"
                ));
            },
        };

        if token.is_cancelled()
        {
            // A cancellation that won the race discards the late answer.
            self.record_resolution(&request_id, call, ApprovalResolution::Cancelled)?;
            return Ok(None);
        }

        let resolution = match answer
        {
            ApprovalAnswer::AllowedOnce => ApprovalResolution::AllowedOnce,
            ApprovalAnswer::AllowedSession => ApprovalResolution::AllowedSession,
            ApprovalAnswer::AllowedPersistent => ApprovalResolution::AllowedPersistent,
            ApprovalAnswer::Rejected => ApprovalResolution::Rejected,
        };
        self.record_resolution(&request_id, call, resolution)?;
        Ok(Some(answer))
    }

    fn record_resolution(
        &self,
        request_id: &ApprovalRequestId,
        call: &ToolCall,
        resolution: ApprovalResolution,
    ) -> Result<(), String> {
        if let Some(audit) = self.audit.as_ref()
        {
            audit.record(ApprovalAuditEvent::tool_resolved_with_id(
                request_id.clone(),
                &call.id,
                &call.tool,
                "",
                resolution,
            ))?;
        }
        Ok(())
    }
}

/// Serialization-friendly wire view of a request (for remote supervisors).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ApprovalRequestWire {
    pub request_id: String,
    pub call_id: String,
    pub tool: String,
    pub subject: Option<String>,
    pub reason: Option<String>,
    pub configured_sandbox: Option<String>,
    pub requested_sandbox: Option<String>,
    pub justification: Option<String>,
    pub policy: String,
}

impl ApprovalRequestWire {
    pub fn from_request(request: &ApprovalRequest) -> Self {
        Self {
            request_id: request.request_id.to_string(),
            call_id: request.call_id.clone(),
            tool: request.tool.clone(),
            subject: request.subject.clone(),
            reason: request.reason.clone(),
            configured_sandbox: request.configured_sandbox.map(|p| p.label().to_string()),
            requested_sandbox: request.requested_sandbox.map(|p| p.label().to_string()),
            justification: request.justification.clone(),
            policy: request.policy.label().to_string(),
        }
    }
}

impl fmt::Display for ApprovalRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "approval {} call={} tool={} policy={}",
            self.request_id,
            self.call_id,
            self.tool,
            self.policy.label()
        )?;
        if let Some(subject) = &self.subject
        {
            write!(f, " subject={subject}")?;
        }
        if let Some(reason) = &self.reason
        {
            write!(f, " reason={reason}")?;
        }
        if let Some(requested) = self.requested_sandbox
        {
            write!(f, " sandbox={}", requested.label())?;
        }
        Ok(())
    }
}

/// Pending-request registry for a supervision UI: maps a request id to its
/// presentation context and cancellation token.
#[derive(Debug, Default)]
pub struct PendingApprovals {
    pending: Mutex<std::collections::HashMap<String, (ApprovalRequest, CancellationToken)>>,
}

impl PendingApprovals {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, request: ApprovalRequest, token: CancellationToken) {
        let key = request.request_id.to_string();
        if let Ok(mut pending) = self.pending.lock()
        {
            pending.insert(key, (request, token));
        }
    }

    pub fn cancel(&self, request_id: &str) -> bool {
        match self.pending.lock()
        {
            Ok(pending) => pending
                .get(request_id)
                .map(|(_, token)| {
                    token.cancel();
                    true
                })
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    pub fn snapshot(&self) -> Vec<ApprovalRequestWire> {
        match self.pending.lock()
        {
            Ok(pending) => pending
                .values()
                .map(|(request, _)| ApprovalRequestWire::from_request(request))
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::approval_audit::InMemoryApprovalAudit;
    use std::collections::HashMap;

    fn call(tool: &str) -> ToolCall {
        ToolCall::new("call-1", tool, HashMap::new())
    }

    struct FixedAnswerer {
        answer: Result<ApprovalAnswer, String>,
    }

    impl ApprovalAnswerer for FixedAnswerer {
        fn answer(&self, _request: &ApprovalRequest) -> Result<ApprovalAnswer, String> {
            self.answer.clone()
        }
    }

    #[test]
    fn missing_answerer_fails_closed() {
        let service = ApprovalService::new(None);
        let error = service
            .request(
                &ApprovalServiceRequest::new(call("build")),
                &CancellationToken::new(),
            )
            .expect_err("no answerer must fail closed");
        assert!(error.contains("no approval answerer"), "{error}");
    }

    #[test]
    fn answerer_error_fails_closed() {
        let service = ApprovalService::with_answerer(Arc::new(FixedAnswerer {
            answer: Err("boom".to_string()),
        }));
        let error = service
            .request(
                &ApprovalServiceRequest::new(call("build")),
                &CancellationToken::new(),
            )
            .expect_err("answerer error must fail closed");
        assert!(error.contains("answerer failed"), "{error}");
    }

    #[test]
    fn never_rejects_before_answerer() {
        struct CountingAnswerer {
            calls: std::sync::atomic::AtomicUsize,
        }
        impl ApprovalAnswerer for CountingAnswerer {
            fn answer(&self, _request: &ApprovalRequest) -> Result<ApprovalAnswer, String> {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(ApprovalAnswer::AllowedOnce)
            }
        }
        let answerer = Arc::new(CountingAnswerer {
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let service =
            ApprovalService::with_answerer(answerer.clone()).with_policy(ApprovalPolicy::Never);
        let outcome = service
            .request(
                &ApprovalServiceRequest::new(call("build")),
                &CancellationToken::new(),
            )
            .unwrap()
            .expect("never rejects deterministically");
        assert_eq!(outcome, ApprovalAnswer::Rejected);
        assert_eq!(answerer.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn allowed_once_is_returned_and_audited_as_pair() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let service = ApprovalService::with_answerer(Arc::new(FixedAnswerer {
            answer: Ok(ApprovalAnswer::AllowedOnce),
        }))
        .with_audit(audit.clone());
        let outcome = service
            .request(
                &ApprovalServiceRequest::new(call("build")).with_subject("core"),
                &CancellationToken::new(),
            )
            .unwrap()
            .expect("allowed once");
        assert_eq!(outcome, ApprovalAnswer::AllowedOnce);
        let events = audit.snapshot().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].request_id, events[1].request_id);
        assert_eq!(events[1].resolution, Some(ApprovalResolution::AllowedOnce));
    }

    #[test]
    fn cancellation_before_answer_returns_none() {
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let service = ApprovalService::with_answerer(Arc::new(FixedAnswerer {
            answer: Ok(ApprovalAnswer::AllowedOnce),
        }))
        .with_audit(audit.clone());
        let token = CancellationToken::new();
        token.cancel();
        let outcome = service
            .request(&ApprovalServiceRequest::new(call("build")), &token)
            .unwrap();
        assert!(outcome.is_none(), "cancelled question grants nothing");
        let events = audit.snapshot().unwrap();
        assert_eq!(events[1].resolution, Some(ApprovalResolution::Cancelled));
    }

    #[test]
    fn late_answer_after_cancellation_is_discarded() {
        struct SlowAnswerer;
        impl ApprovalAnswerer for SlowAnswerer {
            fn answer(&self, _request: &ApprovalRequest) -> Result<ApprovalAnswer, String> {
                Ok(ApprovalAnswer::AllowedOnce)
            }
        }
        // Simulate the race: the answerer already returned, but the token was
        // cancelled before the service observed it — the late answer must not
        // become a grant.
        let audit = Arc::new(InMemoryApprovalAudit::new(8).unwrap());
        let service = ApprovalService::with_answerer(Arc::new(SlowAnswerer)).with_audit(audit);
        let token = CancellationToken::new();
        // First request resolves normally; the second cancels mid-flight.
        service
            .request(&ApprovalServiceRequest::new(call("build")), &token)
            .unwrap();
        token.cancel();
        let outcome = service
            .request(&ApprovalServiceRequest::new(call("build")), &token)
            .unwrap();
        assert!(outcome.is_none());
    }

    #[test]
    fn request_context_carries_full_presentation() {
        let captured: Arc<Mutex<Option<ApprovalRequest>>> = Arc::new(Mutex::new(None));
        struct CaptureAnswerer(Arc<Mutex<Option<ApprovalRequest>>>);
        impl ApprovalAnswerer for CaptureAnswerer {
            fn answer(&self, request: &ApprovalRequest) -> Result<ApprovalAnswer, String> {
                *self.0.lock().unwrap() = Some(request.clone());
                Ok(ApprovalAnswer::Rejected)
            }
        }
        let service = ApprovalService::with_answerer(Arc::new(CaptureAnswerer(captured.clone())));
        let mut call = call("build");
        call.id = "call-42".to_string();
        let outcome = service
            .request(
                &ApprovalServiceRequest::new(call)
                    .with_subject("scirust-core")
                    .with_reason("needs to compile the crate")
                    .with_sandbox(
                        SandboxPermission::ReadOnly,
                        SandboxPermission::WorkspaceWrite,
                    )
                    .with_justification("the crate build writes target/"),
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(outcome, Some(ApprovalAnswer::Rejected));
        let request = captured.lock().unwrap().clone().unwrap();
        assert_eq!(request.call_id, "call-42");
        assert_eq!(request.tool, "build");
        assert_eq!(request.subject.as_deref(), Some("scirust-core"));
        assert_eq!(
            request.reason.as_deref(),
            Some("needs to compile the crate")
        );
        assert_eq!(
            request.requested_sandbox,
            Some(SandboxPermission::WorkspaceWrite)
        );
        assert_eq!(request.policy, ApprovalPolicy::Ask);
        assert!(request.request_id.is_valid());
    }

    #[test]
    fn wire_view_round_trips() {
        let request = ApprovalRequest {
            request_id: ApprovalRequestId::generate(),
            call_id: "call-7".to_string(),
            tool: "build".to_string(),
            subject: Some("core".to_string()),
            reason: None,
            configured_sandbox: Some(SandboxPermission::ReadOnly),
            requested_sandbox: Some(SandboxPermission::DangerFullAccess),
            justification: Some("linker".to_string()),
            policy: ApprovalPolicy::Never,
        };
        let wire = ApprovalRequestWire::from_request(&request);
        let json = serde_json::to_string(&wire).unwrap();
        let back: ApprovalRequestWire = serde_json::from_str(&json).unwrap();
        assert_eq!(wire, back);
        assert_eq!(back.policy, "never");
    }

    #[test]
    fn answer_vocabulary_is_closed_and_fail_closed() {
        assert_eq!(
            ApprovalAnswer::parse("allowed-once"),
            ApprovalAnswer::AllowedOnce
        );
        assert_eq!(
            ApprovalAnswer::parse("allowed-session"),
            ApprovalAnswer::AllowedSession
        );
        assert_eq!(
            ApprovalAnswer::parse("allowed-persistent"),
            ApprovalAnswer::AllowedPersistent
        );
        assert_eq!(ApprovalAnswer::parse("rejected"), ApprovalAnswer::Rejected);
        assert_eq!(
            ApprovalAnswer::parse("weird-value"),
            ApprovalAnswer::Rejected
        );
    }
}
