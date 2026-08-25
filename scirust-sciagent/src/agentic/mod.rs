pub mod approval_audit;
pub mod approval_request;
pub mod approval_service;
pub mod budgets;
pub mod deepseek_bridge;
pub mod delegation;
pub mod durable_audit;
pub mod enforcement;
pub mod enterprise;
pub mod enterprise_audit;
pub mod guard;
pub mod permission;
pub mod policy_store;
mod sandbox;
pub mod sandbox_approval;
pub mod secrets;
pub mod tool_runtime;
pub mod tools;
pub use approval_audit::{
    ApprovalAuditEvent, ApprovalAuditSink, ApprovalChannel, ApprovalLifecycle, ApprovalResolution,
    AuditedSandboxApprovalService, AuditedScopedToolApprover, AuditedToolApprover,
    InMemoryApprovalAudit,
};
pub use approval_request::{APPROVAL_REQUEST_ID_CHARS, ApprovalRequestId, ApprovalRequestIdError};
pub use approval_service::{
    ApprovalAnswer, ApprovalAnswerer, ApprovalRequest, ApprovalRequestWire, ApprovalService,
    ApprovalServiceRequest, CancellationToken, PendingApprovals, ResolvedApproval,
};
pub use budgets::{
    EgressPolicy, EgressTarget, FullResourceBackend, NoResourceBackend, ResourceBackend,
    ResourceLimits,
};
pub use deepseek_bridge::{
    BRIDGE_SCHEMA_VERSION, BridgeApprovalOutcome, BridgeError, BridgeEvent, DeepSeekBridge,
    ParameterDefinition, ToolCallWire, ToolDefinition,
};
pub use delegation::{
    ChildRequest, DelegationContext, DelegationError, ResourceBudget, SecretCapability,
};
pub use durable_audit::{AUDIT_GENESIS_HASH, DurableAuditEntry, FileApprovalAudit};
pub use enforcement::ExecutionConstraints;
pub use enterprise::{
    EnterpriseAction, EnterpriseDecision, EnterpriseIdentity, EnterprisePolicyGate, EnterpriseRule,
    OrgId, ProjectId, TenantId, WorkspaceId,
};
pub use enterprise_audit::{ENTERPRISE_AUDIT_GENESIS, EnterpriseAuditEvent, EnterpriseAuditTrail};
pub use guard::{ConformalGuard, GuardVerdict};
pub use permission::{
    ApprovalChoice, ApprovalOutcome, ApprovalPolicy, PermissionDecision, PermissionGate,
    PermissionPolicy, PermissionRule, PermissionRuleStore, ScopedToolApprover, ToolApprover,
};
pub use policy_store::{
    ApprovalPolicyEvent, ApprovalPolicyStore, FileApprovalPolicyStore, MemoryApprovalPolicyStore,
    POLICY_GENESIS_HASH,
};
pub use sandbox_approval::{
    NoSandboxApprovalService, SandboxApprovalError, SandboxApprovalRequest, SandboxApprovalService,
    SandboxPermission, SandboxPermissionGate,
};
pub use secrets::{SecretGrant, SecretHandle, SecretId, SecretStore};
pub use tool_runtime::{
    AllowAllPolicy, EGRESS_POLICY_METADATA, JUSTIFICATION_METADATA, RESOURCE_LIMITS_METADATA,
    SANDBOX_PERMISSIONS_METADATA, ToolCall, ToolPolicy, ToolRuntime, ToolRuntimeError,
};
pub use tools::Tool;
pub use tools::ToolResult;

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum AgentAction {
    Call {
        tool: String,
        params: HashMap<String, String>,
    },
    Respond {
        text: String,
    },
    Abstain,
}

#[derive(Debug)]
pub struct AgentTurn {
    pub action: AgentAction,
    pub result: Option<String>,
}

pub struct AgentRouter {
    runtime: ToolRuntime<SandboxPermissionGate>,
}

impl Default for AgentRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentRouter {
    pub fn new() -> Self {
        Self::with_permission_gate(PermissionGate::from_env())
    }

    pub fn with_permission_policy(policy: PermissionPolicy) -> Self {
        Self::with_permission_gate(PermissionGate::new(policy))
    }

    pub fn with_permission_gate(gate: PermissionGate) -> Self {
        Self::with_runtime_policy(SandboxPermissionGate::new(gate))
    }

    /// Install independent ordinary-tool and sandbox-widening approval seams.
    /// A session grant held by `gate` never grants sandbox escalation.
    pub fn with_sandbox_approval_service(
        gate: PermissionGate,
        service: Arc<dyn SandboxApprovalService>,
    ) -> Self {
        Self::with_runtime_policy(SandboxPermissionGate::with_approval_service(gate, service))
    }

    fn with_runtime_policy(policy: SandboxPermissionGate) -> Self {
        let mut tools = Tool::builtins();
        sandbox::install_process_sandbox(&mut tools);
        Self {
            runtime: ToolRuntime::new(tools, policy)
                .expect("built-in SciAgent tool contracts must be valid"),
        }
    }

    pub fn parse_action(&self, text: &str) -> AgentAction {
        // Try to parse a JSON tool call from the model output
        if let Some(json) = extract_json(text)
        {
            if let Some(action) = self.parse_tool_call(&json)
            {
                return action;
            }
        }

        // Check for abstain keywords
        let lower = text.to_lowercase();
        if lower.contains("abstain") || lower.contains("i don't know") || lower.contains("pas sûr")
        {
            return AgentAction::Abstain;
        }

        AgentAction::Respond {
            text: text.to_string(),
        }
    }

    pub fn parse_tool_call(&self, json: &serde_json::Value) -> Option<AgentAction> {
        let name = json.get("name").and_then(|v| v.as_str())?;
        let mut params: HashMap<String, String> = json
            .get("params")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default();

        for key in [
            SANDBOX_PERMISSIONS_METADATA,
            JUSTIFICATION_METADATA,
            RESOURCE_LIMITS_METADATA,
            EGRESS_POLICY_METADATA,
        ]
        {
            if let Some(value) = json.get(key).and_then(|value| value.as_str())
            {
                if let Some(existing) = params.get(key)
                {
                    if existing != value
                    {
                        return None;
                    }
                }
                params.insert(key.to_string(), value.to_string());
            }
        }

        if self.runtime.has_tool(name)
        {
            Some(AgentAction::Call {
                tool: name.to_string(),
                params,
            })
        }
        else
        {
            None
        }
    }

    pub fn execute(&self, action: &AgentAction) -> String {
        match action
        {
            AgentAction::Call { tool, params } => self
                .runtime
                .execute_named(tool, params.clone())
                .unwrap_or_else(|error| error.to_string()),
            AgentAction::Respond { text } => text.clone(),
            AgentAction::Abstain => "I abstain — confidence below threshold.".to_string(),
        }
    }
}

fn extract_json(text: &str) -> Option<serde_json::Value> {
    // Find JSON in markdown code blocks or bare JSON
    let candidates = [text, text.trim()];

    for &candidate in &candidates
    {
        // Try to strip ```json ... ``` markers
        let cleaned = if let Some(start) = candidate.find("```json")
        {
            let start = start + 7;
            let end = candidate[start..]
                .find("```")
                .map(|e| start + e)
                .unwrap_or(candidate.len());
            &candidate[start..end]
        }
        else if let Some(start) = candidate.find("```")
        {
            let start = start + 3;
            let end = candidate[start..]
                .find("```")
                .map(|e| start + e)
                .unwrap_or(candidate.len());
            &candidate[start..end]
        }
        else
        {
            candidate
        };

        if let Ok(val) = serde_json::from_str::<serde_json::Value>(cleaned.trim())
        {
            if val.is_object() && val.get("name").is_some()
            {
                return Some(val);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_call_json() {
        let router = AgentRouter::new();
        let text = r#"{"name": "search", "params": {"pattern": "Muon", "path": "scirust-core"}}"#;
        let action = router.parse_action(text);
        assert_eq!(
            action,
            AgentAction::Call {
                tool: "search".to_string(),
                params: [
                    ("pattern".to_string(), "Muon".to_string()),
                    ("path".to_string(), "scirust-core".to_string())
                ]
                .iter()
                .cloned()
                .collect(),
            }
        );
    }

    #[test]
    fn test_parse_tool_call_markdown() {
        let router = AgentRouter::new();
        let text = "I'll search for that:\n```json\n{\"name\": \"grep\", \"params\": {\"pattern\": \"PcgEngine\"}}\n```";
        let action = router.parse_action(text);
        assert_eq!(
            action,
            AgentAction::Call {
                tool: "grep".to_string(),
                params: [("pattern".to_string(), "PcgEngine".to_string())]
                    .iter()
                    .cloned()
                    .collect(),
            }
        );
    }

    #[test]
    fn test_parse_tool_call_sandbox_metadata() {
        let router = AgentRouter::new();
        let text = r#"{"name":"build","params":{"crate":"scirust-core"},"sandbox_permissions":"danger-full-access","justification":"requires one exact wider call"}"#;
        let action = router.parse_action(text);
        let AgentAction::Call { tool, params } = action
        else
        {
            panic!("expected tool call");
        };
        assert_eq!(tool, "build");
        assert_eq!(
            params.get(SANDBOX_PERMISSIONS_METADATA).map(String::as_str),
            Some("danger-full-access")
        );
        assert_eq!(
            params.get(JUSTIFICATION_METADATA).map(String::as_str),
            Some("requires one exact wider call")
        );
    }

    #[test]
    fn conflicting_sandbox_metadata_is_not_parsed_as_a_tool_call() {
        let router = AgentRouter::new();
        let text = r#"{"name":"build","params":{"crate":"scirust-core","sandbox_permissions":"workspace-write"},"sandbox_permissions":"danger-full-access","justification":"conflict"}"#;
        assert_eq!(
            router.parse_action(text),
            AgentAction::Respond {
                text: text.to_string()
            }
        );
    }

    #[test]
    fn test_parse_respond() {
        let router = AgentRouter::new();
        let action = router.parse_action("Hello, I can help with that.");
        assert_eq!(
            action,
            AgentAction::Respond {
                text: "Hello, I can help with that.".to_string()
            }
        );
    }

    #[test]
    fn test_parse_abstain() {
        let router = AgentRouter::new();
        let action = router.parse_action("I abstain from answering this.");
        assert_eq!(action, AgentAction::Abstain);
    }

    #[test]
    fn test_execute_search() {
        let router = AgentRouter::new();
        let action = AgentAction::Call {
            tool: "search".to_string(),
            params: [
                ("pattern".to_string(), "NdMuon".to_string()),
                (
                    "path".to_string(),
                    format!("{}/scirust-core", super::tools::workspace_root()),
                ),
            ]
            .iter()
            .cloned()
            .collect(),
        };
        let result = router.execute(&action);
        assert!(
            result.contains("NdMuon"),
            "Search should find NdMuon, got: {result}"
        );
    }

    #[test]
    fn router_enforces_required_tool_parameters() {
        let router = AgentRouter::new();
        let action = AgentAction::Call {
            tool: "search".to_string(),
            params: HashMap::new(),
        };
        let result = router.execute(&action);
        assert!(result.contains("Missing required parameter"), "{result}");
    }

    #[test]
    fn router_permission_policy_denies_before_tool_execution() {
        let router = AgentRouter::with_permission_policy(PermissionPolicy::new(
            PermissionDecision::Allow,
            Vec::new(),
            Vec::new(),
            vec![PermissionRule::parse("search").expect("valid rule")],
        ));
        let action = AgentAction::Call {
            tool: "search".to_string(),
            params: [("pattern".to_string(), "NdMuon".to_string())]
                .into_iter()
                .collect(),
        };
        let result = router.execute(&action);
        assert!(result.contains("denied by permission policy"), "{result}");
    }
}
