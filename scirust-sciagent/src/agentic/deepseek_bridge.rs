//! DeepSeek Harness bridge — first-class interoperability with
//! `@deepseek-ai/dsh-*` frontends/orchestrators.
//!
//! The bridge NEVER lets a DeepSeek client call shell/process callbacks
//! directly. Every tool call travels the mandatory chain:
//!
//! ```text
//! DeepSeek adapter
//!   -> versioned wire contract (this module)
//!   -> typed ToolCall
//!   -> ToolRuntime validation
//!   -> PermissionGate
//!   -> ApprovalService
//!   -> sandbox escalation gate
//!   -> hardened execution
//! ```
//!
//! All wire types are serde-serializable and versioned. Unknown or malformed
//! values fail closed (`BridgeError::UnknownValue`) rather than broadening
//! privileges.

use super::approval_request::ApprovalRequestId;
use super::approval_service::{
    ApprovalAnswer, ApprovalRequestWire, ApprovalService, ApprovalServiceRequest, CancellationToken,
};
use super::enterprise::EnterpriseIdentity;
use super::enterprise_audit::EnterpriseAuditSink;
use super::permission::ApprovalPolicy;
use super::sandbox_approval::SandboxPermission;
use super::tool_runtime::{ToolCall, ToolRuntime, ToolRuntimeError};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Wire schema version of the bridge contract.
pub const BRIDGE_SCHEMA_VERSION: u16 = 1;

/// One exported tool definition (name, description, typed params, version).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ParameterDefinition>,
    pub version: u16,
}

/// One typed parameter of a tool definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterDefinition {
    pub name: String,
    pub required: bool,
}

/// One tool call from a DeepSeek-style client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallWire {
    pub call_id: String,
    pub tool: String,
    pub params: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_permissions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub justification: Option<String>,
}

/// Closed bridge outcomes for approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BridgeApprovalOutcome {
    AllowedOnce,
    AllowedSession,
    AllowedPersistent,
    Rejected,
    Cancelled,
    Unavailable,
}

/// Streaming event vocabulary of the bridge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum BridgeEvent {
    CallAnnounced {
        call_id: String,
        tool: String,
    },
    CallStarted {
        call_id: String,
        tool: String,
    },
    ApprovalRequested {
        request: ApprovalRequestWire,
    },
    ApprovalResolved {
        request_id: String,
        outcome: BridgeApprovalOutcome,
    },
    ExecutionStarted {
        call_id: String,
    },
    ExecutionEnded {
        call_id: String,
        output: String,
    },
    ExecutionError {
        call_id: String,
        error: BridgeError,
    },
}

/// Typed, versioned bridge errors. Unknown values fail closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BridgeError {
    UnsupportedSchema(u16),
    UnknownTool(String),
    InvalidParameters(String),
    PolicyDenied(String),
    ApprovalUnavailable,
    SandboxEscalationDenied(String),
    GovernanceDenied(String),
    ExecutionFailed(String),
    UnknownValue(String),
    Cancelled,
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::UnsupportedSchema(v) => write!(f, "unsupported bridge schema version {v}"),
            Self::UnknownTool(t) => write!(f, "unknown tool {t:?}"),
            Self::InvalidParameters(p) => write!(f, "invalid tool parameters: {p}"),
            Self::PolicyDenied(r) => write!(f, "denied by policy: {r}"),
            Self::ApprovalUnavailable => write!(f, "approval unavailable; refusing to execute"),
            Self::SandboxEscalationDenied(r) => write!(f, "sandbox escalation refused: {r}"),
            Self::GovernanceDenied(r) => write!(f, "resource governance refused: {r}"),
            Self::ExecutionFailed(e) => write!(f, "execution failed: {e}"),
            Self::UnknownValue(v) => write!(f, "unknown value {v:?}; failing closed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<ToolRuntimeError> for BridgeError {
    fn from(error: ToolRuntimeError) -> Self {
        match error
        {
            ToolRuntimeError::UnknownTool(tool) => Self::UnknownTool(tool),
            ToolRuntimeError::DuplicateTool(_) | ToolRuntimeError::DuplicateParameter { .. } =>
            {
                Self::UnknownValue(error.to_string())
            },
            ToolRuntimeError::MissingRequiredParameter { tool, parameter }
            | ToolRuntimeError::UndeclaredParameter { tool, parameter } =>
            {
                Self::InvalidParameters(format!("{tool}: {parameter}"))
            },
            ToolRuntimeError::ReservedParameter { tool, parameter } =>
            {
                Self::InvalidParameters(format!("{tool}: {parameter}"))
            },
            ToolRuntimeError::PolicyDenied { tool, reason } =>
            {
                Self::PolicyDenied(format!("{tool}: {reason}"))
            },
            ToolRuntimeError::SandboxEscalationDenied { tool, reason } =>
            {
                Self::SandboxEscalationDenied(format!("{tool}: {reason}"))
            },
            ToolRuntimeError::GovernanceDenied { tool, reason } =>
            {
                Self::GovernanceDenied(format!("{tool}: {reason}"))
            },
        }
    }
}

/// The DeepSeek Harness bridge over a hardened SciAgent runtime.
pub struct DeepSeekBridge<P> {
    runtime: ToolRuntime<P>,
    approval: ApprovalService,
    enterprise_audit: Option<Arc<dyn EnterpriseAuditSink>>,
    identity: Option<EnterpriseIdentity>,
    session_id: String,
}

impl<P: super::tool_runtime::ToolPolicy> DeepSeekBridge<P> {
    pub fn new(runtime: ToolRuntime<P>, approval: ApprovalService) -> Self {
        Self {
            runtime,
            approval,
            enterprise_audit: None,
            identity: None,
            session_id: String::new(),
        }
    }

    /// Automatically emit one correlated enterprise audit event per executed
    /// call. Fail-closed: when the sink refuses the record, a successful
    /// execution is reported as failed instead of returning an unaudited
    /// result to the model.
    pub fn with_enterprise_audit(
        mut self,
        sink: Arc<dyn EnterpriseAuditSink>,
        identity: EnterpriseIdentity,
        session_id: impl Into<String>,
    ) -> Self {
        self.enterprise_audit = Some(sink);
        self.identity = Some(identity);
        self.session_id = session_id.into();
        self
    }

    /// Export the tool definitions of the underlying runtime.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.runtime
            .tools()
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.name.to_string(),
                description: tool.description.to_string(),
                parameters: tool
                    .parameters
                    .iter()
                    .map(|param| ParameterDefinition {
                        name: param.name.to_string(),
                        required: param.required,
                    })
                    .collect(),
                version: BRIDGE_SCHEMA_VERSION,
            })
            .collect()
    }

    /// The session's effective approval policy, exposed to the model-facing
    /// runtime context.
    pub fn approval_policy(&self) -> ApprovalPolicy {
        self.approval.policy()
    }

    /// Execute one wire tool call through the mandatory chain, emitting
    /// streaming events. The call is validated by the ToolRuntime before any
    /// policy hook; approval flows through the ApprovalService; sandbox
    /// escalation is enforced by the runtime policy.
    pub fn execute(
        &self,
        wire: ToolCallWire,
        token: &CancellationToken,
        events: &dyn Fn(BridgeEvent),
    ) -> Result<String, BridgeError> {
        if wire.call_id.trim().is_empty()
        {
            return Err(BridgeError::UnknownValue("empty call_id".to_string()));
        }
        if wire.tool.trim().is_empty()
        {
            return Err(BridgeError::UnknownValue("empty tool name".to_string()));
        }
        // The sandbox permission/justification pair must be supplied together;
        // checked before any runtime validation so malformed metadata can
        // never be interpreted as a non-escalation.
        if wire.sandbox_permissions.is_some() != wire.justification.is_some()
        {
            return Err(BridgeError::UnknownValue(
                "sandbox_permissions and justification must be supplied together".to_string(),
            ));
        }
        let call = ToolCall::new(wire.call_id.clone(), wire.tool.clone(), wire.params.clone())
            .with_sandbox_metadata(wire.sandbox_permissions.clone(), wire.justification.clone());

        events(BridgeEvent::CallAnnounced {
            call_id: call.id.clone(),
            tool: call.tool.clone(),
        });

        // ToolRuntime validation (unknown tool, undeclared/missing params).
        self.runtime
            .validate_call(&call)
            .map_err(BridgeError::from)?;

        events(BridgeEvent::CallStarted {
            call_id: call.id.clone(),
            tool: call.tool.clone(),
        });

        // Resolve sandbox metadata into the approval input before execution.
        // The ApprovalRequested event itself is emitted only after the
        // ApprovalService has generated the authoritative request id.
        let (configured, requested) = match wire.sandbox_permissions.as_deref()
        {
            None => (None, None),
            Some(requested) =>
            {
                let requested_perm = SandboxPermission::parse(requested).map_err(|_| {
                    BridgeError::UnknownValue(format!("sandbox permission {requested:?}"))
                })?;
                let configured_perm = super::sandbox::configured_permission()
                    .map_err(BridgeError::SandboxEscalationDenied)?;
                (Some(configured_perm), Some(requested_perm))
            },
        };

        // ApprovalService (id generation, policy, answerer, cancellation).
        let mut input = ApprovalServiceRequest::new(call.clone())
            .with_reason(format!("tool {} requested by DeepSeek client", call.tool));
        if let Some(requested) = requested
        {
            input =
                input.with_sandbox(configured.unwrap_or(SandboxPermission::ReadOnly), requested);
        }
        if let Some(justification) = wire.justification.as_deref()
        {
            input = input.with_justification(justification);
        }

        let observed_request_id = RefCell::new(None::<String>);
        let outcome = self.approval.request_resolved(&input, token, &|request| {
            let request = ApprovalRequestWire::from_request(request);
            *observed_request_id.borrow_mut() = Some(request.request_id.clone());
            events(BridgeEvent::ApprovalRequested { request });
        });
        let observed_request_id = observed_request_id.into_inner();

        let outcome = match outcome
        {
            Ok(outcome) => outcome,
            Err(_error) =>
            {
                if let Some(request_id) = observed_request_id
                {
                    events(BridgeEvent::ApprovalResolved {
                        request_id,
                        outcome: BridgeApprovalOutcome::Unavailable,
                    });
                }
                return Err(BridgeError::ApprovalUnavailable);
            },
        };

        let observed_request_id = observed_request_id.ok_or_else(|| {
            BridgeError::UnknownValue(
                "approval service resolved without exposing a request id".to_string(),
            )
        })?;

        match outcome
        {
            None =>
            {
                events(BridgeEvent::ApprovalResolved {
                    request_id: observed_request_id,
                    outcome: BridgeApprovalOutcome::Cancelled,
                });
                Err(BridgeError::Cancelled)
            },
            Some(resolved) =>
            {
                let request_id = resolved.request_id.to_string();
                if request_id != observed_request_id
                {
                    return Err(BridgeError::UnknownValue(
                        "approval request/resolution correlation mismatch".to_string(),
                    ));
                }
                let bridge_outcome = match resolved.answer
                {
                    ApprovalAnswer::AllowedOnce => BridgeApprovalOutcome::AllowedOnce,
                    ApprovalAnswer::AllowedSession => BridgeApprovalOutcome::AllowedSession,
                    ApprovalAnswer::AllowedPersistent => BridgeApprovalOutcome::AllowedPersistent,
                    ApprovalAnswer::Rejected => BridgeApprovalOutcome::Rejected,
                };
                events(BridgeEvent::ApprovalResolved {
                    request_id,
                    outcome: bridge_outcome,
                });

                if resolved.answer == ApprovalAnswer::Rejected
                {
                    self.audit_execution(
                        &call,
                        wire.sandbox_permissions.as_deref(),
                        Some(&resolved.request_id),
                        "rejected",
                        "",
                    )?;
                    return Err(BridgeError::PolicyDenied(
                        "approval rejected by the user".to_string(),
                    ));
                }

                let approved_request_id = resolved.request_id.clone();
                events(BridgeEvent::ExecutionStarted {
                    call_id: call.id.clone(),
                });
                let result = self.runtime.execute(&call);
                match result
                {
                    Ok(output) =>
                    {
                        // No unaudited success leaves the bridge: a refusing
                        // sink turns the executed call into an error even
                        // though side effects already happened.
                        self.audit_execution(
                            &call,
                            wire.sandbox_permissions.as_deref(),
                            Some(&approved_request_id),
                            "executed",
                            &crate::sha256::sha256_hex(output.as_bytes()),
                        )?;
                        events(BridgeEvent::ExecutionEnded {
                            call_id: call.id.clone(),
                            output: output.clone(),
                        });
                        Ok(output)
                    },
                    Err(error) =>
                    {
                        // The execution already failed; auditing its failure
                        // is best-effort and never masks the original error.
                        let _ = self.audit_execution(
                            &call,
                            wire.sandbox_permissions.as_deref(),
                            Some(&approved_request_id),
                            "failed",
                            "",
                        );
                        let bridge_error = BridgeError::from(error);
                        events(BridgeEvent::ExecutionError {
                            call_id: call.id.clone(),
                            error: bridge_error.clone(),
                        });
                        Err(bridge_error)
                    },
                }
            },
        }
    }

    /// Emit one correlated enterprise event per executed (or refused) call
    /// when [`DeepSeekBridge::with_enterprise_audit`] wired a sink.
    fn audit_execution(
        &self,
        call: &ToolCall,
        sandbox_permissions: Option<&str>,
        request_id: Option<&ApprovalRequestId>,
        decision: &str,
        output_digest: &str,
    ) -> Result<(), BridgeError> {
        let (Some(sink), Some(identity)) = (&self.enterprise_audit, &self.identity)
        else
        {
            return Ok(());
        };
        if self.session_id.is_empty()
        {
            return Err(BridgeError::ExecutionFailed(
                "enterprise audit is wired but the session id is empty".to_string(),
            ));
        }
        let sandbox = sandbox_permissions.unwrap_or("not-requested").to_string();
        sink.record_execution(
            &identity.tenant,
            &identity.subject,
            &self.session_id,
            request_id,
            &call.id,
            &call.tool,
            &sandbox,
            decision,
            output_digest,
        )
        .map_err(|error| {
            BridgeError::ExecutionFailed(format!("enterprise audit unavailable: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::approval_service::{ApprovalAnswerer, ApprovalRequest};
    use crate::agentic::permission::{
        ApprovalOutcome, PermissionDecision, PermissionGate, PermissionPolicy,
    };
    use crate::agentic::sandbox_approval::SandboxPermissionGate;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct AllowAnswerer {
        calls: AtomicUsize,
        answer: ApprovalAnswer,
    }

    impl ApprovalAnswerer for AllowAnswerer {
        fn answer(&self, _request: &ApprovalRequest) -> Result<ApprovalAnswer, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.answer)
        }
    }

    /// Approver stub so escalated tool calls can complete execution in tests.
    struct AlwaysAllowSandboxApprover;

    impl crate::agentic::sandbox_approval::SandboxApprovalService for AlwaysAllowSandboxApprover {
        fn request_approval(
            &self,
            _request: &crate::agentic::sandbox_approval::SandboxApprovalRequest,
        ) -> Result<ApprovalOutcome, String> {
            Ok(ApprovalOutcome::AllowedOnce)
        }
    }

    fn test_runtime() -> ToolRuntime<SandboxPermissionGate> {
        let gate = PermissionGate::new(PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        let policy = SandboxPermissionGate::with_approval_service(
            gate,
            Arc::new(AlwaysAllowSandboxApprover),
        );
        ToolRuntime::new(crate::agentic::tools::Tool::builtins(), policy).unwrap()
    }

    fn test_bridge_with_answer(answer: ApprovalAnswer) -> DeepSeekBridge<SandboxPermissionGate> {
        let approval = ApprovalService::with_answerer(Arc::new(AllowAnswerer {
            calls: AtomicUsize::new(0),
            answer,
        }));
        DeepSeekBridge::new(test_runtime(), approval)
    }

    fn test_bridge() -> DeepSeekBridge<SandboxPermissionGate> {
        test_bridge_with_answer(ApprovalAnswer::AllowedOnce)
    }

    fn enterprise_identity() -> crate::agentic::enterprise::EnterpriseIdentity {
        use crate::agentic::enterprise::{OrgId, ProjectId, TenantId, WorkspaceId};
        crate::agentic::enterprise::EnterpriseIdentity::new(
            TenantId::parse("acme").unwrap(),
            OrgId::parse("org-1").unwrap(),
            ProjectId::parse("proj-1").unwrap(),
            WorkspaceId::parse("ws-1").unwrap(),
            "alice",
        )
    }

    fn temp_enterprise_log() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!(
            "scirust-bridge-audit-{}-{unique}.jsonl",
            std::process::id()
        ));
        path
    }

    #[test]
    fn tool_definitions_are_exported() {
        let bridge = test_bridge();
        let definitions = bridge.tool_definitions();
        assert!(!definitions.is_empty());
        assert!(definitions.iter().any(|d| d.name == "read"));
        for definition in definitions
        {
            assert_eq!(definition.version, BRIDGE_SCHEMA_VERSION);
            assert!(!definition.name.is_empty());
            assert!(!definition.description.is_empty());
        }
    }

    #[test]
    fn unknown_tool_fails_closed() {
        let bridge = test_bridge();
        let events = RefCell::new(Vec::new());
        let error = bridge
            .execute(
                ToolCallWire {
                    call_id: "call-1".to_string(),
                    tool: "nonexistent".to_string(),
                    params: HashMap::new(),
                    sandbox_permissions: None,
                    justification: None,
                },
                &CancellationToken::new(),
                &|event| events.borrow_mut().push(event),
            )
            .expect_err("unknown tool must fail closed");
        assert!(matches!(error, BridgeError::UnknownTool(_)));
        assert!(
            events
                .borrow()
                .iter()
                .any(|e| matches!(e, BridgeEvent::CallAnnounced { .. }))
        );
    }

    #[test]
    fn missing_required_parameter_fails_closed() {
        let bridge = test_bridge();
        let error = bridge
            .execute(
                ToolCallWire {
                    call_id: "call-2".to_string(),
                    tool: "read".to_string(),
                    params: HashMap::new(),
                    sandbox_permissions: None,
                    justification: None,
                },
                &CancellationToken::new(),
                &|_| {},
            )
            .expect_err("missing required parameter must fail closed");
        assert!(matches!(error, BridgeError::InvalidParameters(_)));
    }

    #[test]
    fn sandbox_metadata_must_be_paired() {
        let bridge = test_bridge();
        let error = bridge
            .execute(
                ToolCallWire {
                    call_id: "call-3".to_string(),
                    tool: "build".to_string(),
                    params: HashMap::new(),
                    sandbox_permissions: Some("workspace-write".to_string()),
                    justification: None,
                },
                &CancellationToken::new(),
                &|_| {},
            )
            .expect_err("unpaired sandbox metadata must fail closed");
        assert!(matches!(error, BridgeError::UnknownValue(_)));
    }

    #[test]
    fn empty_call_id_fails_closed() {
        let bridge = test_bridge();
        let error = bridge
            .execute(
                ToolCallWire {
                    call_id: "".to_string(),
                    tool: "read".to_string(),
                    params: HashMap::new(),
                    sandbox_permissions: None,
                    justification: None,
                },
                &CancellationToken::new(),
                &|_| {},
            )
            .expect_err("empty call_id must fail closed");
        assert!(matches!(error, BridgeError::UnknownValue(_)));
    }

    #[test]
    fn successful_execution_emits_full_stream() {
        let bridge = test_bridge();
        let events = RefCell::new(Vec::new());
        let mut params = HashMap::new();
        params.insert("path".to_string(), "Cargo.toml".to_string());
        let output = bridge
            .execute(
                ToolCallWire {
                    call_id: "call-4".to_string(),
                    tool: "read".to_string(),
                    params,
                    sandbox_permissions: None,
                    justification: None,
                },
                &CancellationToken::new(),
                &|event| events.borrow_mut().push(event),
            )
            .expect("read must execute");
        assert!(output.contains("[package]"));
        let kinds: Vec<&str> = events
            .borrow()
            .iter()
            .map(|e| match e
            {
                BridgeEvent::CallAnnounced { .. } => "announced",
                BridgeEvent::CallStarted { .. } => "started",
                BridgeEvent::ApprovalRequested { .. } => "approval_requested",
                BridgeEvent::ApprovalResolved { .. } => "approval_resolved",
                BridgeEvent::ExecutionStarted { .. } => "execution_started",
                BridgeEvent::ExecutionEnded { .. } => "execution_ended",
                BridgeEvent::ExecutionError { .. } => "execution_error",
            })
            .collect();
        assert!(kinds.contains(&"announced"));
        assert!(kinds.contains(&"started"));
        assert!(kinds.contains(&"approval_requested"));
        assert!(kinds.contains(&"approval_resolved"));
        assert!(kinds.contains(&"execution_started"));
        assert!(kinds.contains(&"execution_ended"));
    }

    #[test]
    fn approval_stream_uses_real_correlated_request_id() {
        let bridge = test_bridge();
        let events = RefCell::new(Vec::new());
        let mut params = HashMap::new();
        params.insert("path".to_string(), "Cargo.toml".to_string());
        bridge
            .execute(
                ToolCallWire {
                    call_id: "call-correlated".to_string(),
                    tool: "read".to_string(),
                    params,
                    sandbox_permissions: None,
                    justification: None,
                },
                &CancellationToken::new(),
                &|event| events.borrow_mut().push(event),
            )
            .unwrap();

        let events = events.borrow();
        let requested_id = events
            .iter()
            .find_map(|event| match event
            {
                BridgeEvent::ApprovalRequested { request } => Some(request.request_id.clone()),
                _ => None,
            })
            .expect("approval request event");
        let (resolved_id, outcome) = events
            .iter()
            .find_map(|event| match event
            {
                BridgeEvent::ApprovalResolved {
                    request_id,
                    outcome,
                } => Some((request_id.clone(), *outcome)),
                _ => None,
            })
            .expect("approval resolution event");
        assert!(!requested_id.is_empty());
        assert_eq!(requested_id, resolved_id);
        assert_eq!(outcome, BridgeApprovalOutcome::AllowedOnce);
    }

    #[test]
    fn approval_scope_is_preserved_on_wire() {
        for (answer, expected) in [
            (
                ApprovalAnswer::AllowedSession,
                BridgeApprovalOutcome::AllowedSession,
            ),
            (
                ApprovalAnswer::AllowedPersistent,
                BridgeApprovalOutcome::AllowedPersistent,
            ),
        ]
        {
            let bridge = test_bridge_with_answer(answer);
            let events = RefCell::new(Vec::new());
            let mut params = HashMap::new();
            params.insert("path".to_string(), "Cargo.toml".to_string());
            bridge
                .execute(
                    ToolCallWire {
                        call_id: format!("scope-{}", answer.label()),
                        tool: "read".to_string(),
                        params,
                        sandbox_permissions: None,
                        justification: None,
                    },
                    &CancellationToken::new(),
                    &|event| events.borrow_mut().push(event),
                )
                .unwrap();
            let actual = events
                .into_inner()
                .into_iter()
                .find_map(|event| match event
                {
                    BridgeEvent::ApprovalResolved { outcome, .. } => Some(outcome),
                    _ => None,
                })
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn sandbox_justification_reaches_answerer() {
        let captured: Arc<Mutex<Option<ApprovalRequest>>> = Arc::new(Mutex::new(None));
        struct CaptureAnswerer(Arc<Mutex<Option<ApprovalRequest>>>);
        impl ApprovalAnswerer for CaptureAnswerer {
            fn answer(&self, request: &ApprovalRequest) -> Result<ApprovalAnswer, String> {
                *self.0.lock().unwrap() = Some(request.clone());
                Ok(ApprovalAnswer::AllowedOnce)
            }
        }
        let approval = ApprovalService::with_answerer(Arc::new(CaptureAnswerer(captured.clone())));
        let bridge = DeepSeekBridge::new(test_runtime(), approval);
        let mut params = HashMap::new();
        // `grep` executes through the process sandbox (unlike `read`), so the
        // escalation metadata survives the full announce -> approve -> execute
        // path and the justification reaches the answerer verbatim.
        params.insert("pattern".to_string(), "version".to_string());
        params.insert("path".to_string(), "Cargo.toml".to_string());
        bridge
            .execute(
                ToolCallWire {
                    call_id: "call-justification".to_string(),
                    tool: "grep".to_string(),
                    params,
                    sandbox_permissions: Some("danger-full-access".to_string()),
                    justification: Some("full access escalation requested".to_string()),
                },
                &CancellationToken::new(),
                &|_| {},
            )
            .unwrap();
        let request = captured.lock().unwrap().clone().unwrap();
        assert_eq!(
            request.justification.as_deref(),
            Some("full access escalation requested")
        );
        assert!(request.request_id.is_valid());
    }

    #[test]
    fn wire_round_trip() {
        let wire = ToolCallWire {
            call_id: "call-9".to_string(),
            tool: "build".to_string(),
            params: HashMap::from([("crate".to_string(), "core".to_string())]),
            sandbox_permissions: Some("workspace-write".to_string()),
            justification: Some("need target".to_string()),
        };
        let json = serde_json::to_string(&wire).unwrap();
        let back: ToolCallWire = serde_json::from_str(&json).unwrap();
        assert_eq!(wire, back);
    }

    #[test]
    fn unknown_wire_value_fails_closed() {
        let json = r#"{"event":"execution_ended","call_id":"c1","output":"x"}"#;
        let parsed: Result<BridgeEvent, _> = serde_json::from_str(json);
        assert!(parsed.is_ok());
        let bad = r#"{"event":"unknown_event"}"#;
        assert!(serde_json::from_str::<BridgeEvent>(bad).is_err());
    }

    // -- Automatic enterprise audit emission --------------------------------

    fn read_wire(call_id: &str) -> ToolCallWire {
        let mut params = HashMap::new();
        params.insert("path".to_string(), "Cargo.toml".to_string());
        ToolCallWire {
            call_id: call_id.to_string(),
            tool: "read".to_string(),
            params,
            sandbox_permissions: None,
            justification: None,
        }
    }

    #[test]
    fn every_executed_call_is_audited_with_full_correlation() {
        use crate::agentic::enterprise_audit::FileEnterpriseAuditTrail;

        let path = temp_enterprise_log();
        let bridge = test_bridge().with_enterprise_audit(
            Arc::new(FileEnterpriseAuditTrail::new(&path)),
            enterprise_identity(),
            "session-77",
        );
        let output = bridge
            .execute(
                read_wire("call-audit-1"),
                &CancellationToken::new(),
                &|_| {},
            )
            .expect("read must execute");
        let store = FileEnterpriseAuditTrail::new(&path);
        let events = store.replay().unwrap();
        assert_eq!(events.len(), 1, "exactly one correlated event");
        let event = &events[0];
        assert_eq!(event.tenant.as_str(), "acme");
        assert_eq!(event.subject, "alice");
        assert_eq!(event.session_id, "session-77");
        assert_eq!(event.call_id.as_deref(), Some("call-audit-1"));
        assert_eq!(event.tool.as_deref(), Some("read"));
        assert_eq!(event.sandbox.as_deref(), Some("not-requested"));
        assert_eq!(event.decision.as_deref(), Some("executed"));
        assert_eq!(
            event.execution_digest.as_deref(),
            Some(crate::sha256::sha256_hex(output.as_bytes()).as_str())
        );
        assert!(event.request_id.is_some(), "approval id must correlate");
        assert_eq!(
            event.prev_hash,
            crate::agentic::enterprise_audit::ENTERPRISE_AUDIT_GENESIS
        );
        store.verify().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn refusing_audit_sink_fails_closed_after_execution() {
        use crate::agentic::enterprise_audit::FileEnterpriseAuditTrail;

        // A directory cannot serve as an append-only log: the sink refuses.
        let dir = temp_enterprise_log();
        std::fs::create_dir_all(&dir).unwrap();
        let bridge = test_bridge().with_enterprise_audit(
            Arc::new(FileEnterpriseAuditTrail::new(&dir)),
            enterprise_identity(),
            "session-77",
        );
        let error = bridge
            .execute(
                read_wire("call-audit-2"),
                &CancellationToken::new(),
                &|_| {},
            )
            .expect_err("an unaudited success must not reach the model");
        assert!(
            matches!(error, BridgeError::ExecutionFailed(ref reason)
                if reason.contains("enterprise audit unavailable")),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_wired_sink_behaviour_is_unchanged() {
        let bridge = test_bridge();
        let output = bridge
            .execute(
                read_wire("call-noaudit"),
                &CancellationToken::new(),
                &|_| {},
            )
            .expect("no sink wired: execution must behave as before");
        assert!(output.contains("[package]"));
    }
}
