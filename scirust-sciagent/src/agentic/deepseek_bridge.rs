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

use super::approval_service::{
    ApprovalAnswer, ApprovalRequestWire, ApprovalService, ApprovalServiceRequest, CancellationToken,
};
use super::permission::ApprovalPolicy;
use super::sandbox_approval::SandboxPermission;
use super::tool_runtime::{ToolCall, ToolRuntime, ToolRuntimeError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        }
    }
}

/// The DeepSeek Harness bridge over a hardened SciAgent runtime.
pub struct DeepSeekBridge<P> {
    runtime: ToolRuntime<P>,
    approval: ApprovalService,
}

impl<P: super::tool_runtime::ToolPolicy> DeepSeekBridge<P> {
    pub fn new(runtime: ToolRuntime<P>, approval: ApprovalService) -> Self {
        Self { runtime, approval }
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

        // Resolve sandbox metadata into an approval input before execution.
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
                let justification = wire.justification.as_deref().unwrap_or_default();
                events(BridgeEvent::ApprovalRequested {
                    request: ApprovalRequestWire {
                        request_id: String::new(),
                        call_id: call.id.clone(),
                        tool: call.tool.clone(),
                        subject: None,
                        reason: Some("sandbox escalation".to_string()),
                        configured_sandbox: Some(configured_perm.label().to_string()),
                        requested_sandbox: Some(requested_perm.label().to_string()),
                        justification: Some(justification.to_string()),
                        policy: self.approval.policy().label().to_string(),
                    },
                });
                (Some(configured_perm), Some(requested_perm))
            },
        };

        // ApprovalService (id generation, policy, answerer, cancellation).
        let input = ApprovalServiceRequest::new(call.clone())
            .with_reason(format!("tool {} requested by DeepSeek client", call.tool));
        let input = if let Some(requested) = requested
        {
            input.with_sandbox(configured.unwrap_or(SandboxPermission::ReadOnly), requested)
        }
        else
        {
            input
        };

        // Build the full presentation request for the answerer by asking the
        // service; it generates the id and enforces policy.
        let outcome = self
            .approval
            .request(&input, token)
            .map_err(|_error| BridgeError::ApprovalUnavailable)?;

        match outcome
        {
            None =>
            {
                events(BridgeEvent::ApprovalResolved {
                    request_id: String::new(),
                    outcome: BridgeApprovalOutcome::Cancelled,
                });
                Err(BridgeError::Cancelled)
            },
            Some(ApprovalAnswer::Rejected) =>
            {
                events(BridgeEvent::ApprovalResolved {
                    request_id: String::new(),
                    outcome: BridgeApprovalOutcome::Rejected,
                });
                Err(BridgeError::PolicyDenied(
                    "approval rejected by the user".to_string(),
                ))
            },
            Some(_) =>
            {
                events(BridgeEvent::ApprovalResolved {
                    request_id: String::new(),
                    outcome: BridgeApprovalOutcome::AllowedOnce,
                });
                events(BridgeEvent::ExecutionStarted {
                    call_id: call.id.clone(),
                });
                let result = self.runtime.execute(&call);
                match result
                {
                    Ok(output) =>
                    {
                        events(BridgeEvent::ExecutionEnded {
                            call_id: call.id.clone(),
                            output: output.clone(),
                        });
                        Ok(output)
                    },
                    Err(error) =>
                    {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::approval_service::{ApprovalAnswerer, ApprovalRequest};
    use crate::agentic::permission::{PermissionDecision, PermissionGate, PermissionPolicy};
    use crate::agentic::sandbox_approval::SandboxPermissionGate;
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct AllowAnswerer {
        calls: AtomicUsize,
    }

    impl ApprovalAnswerer for AllowAnswerer {
        fn answer(&self, _request: &ApprovalRequest) -> Result<ApprovalAnswer, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ApprovalAnswer::AllowedOnce)
        }
    }

    fn test_bridge() -> DeepSeekBridge<SandboxPermissionGate> {
        let gate = PermissionGate::new(PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
        let policy = SandboxPermissionGate::new(gate);
        let runtime = ToolRuntime::new(crate::agentic::tools::Tool::builtins(), policy).unwrap();
        let approval = ApprovalService::with_answerer(Arc::new(AllowAnswerer {
            calls: AtomicUsize::new(0),
        }));
        DeepSeekBridge::new(runtime, approval)
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
        assert!(kinds.contains(&"execution_started"));
        assert!(kinds.contains(&"execution_ended"));
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
}
