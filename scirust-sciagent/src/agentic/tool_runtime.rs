use super::enforcement::{ExecutionConstraints, probed_backend};
use super::sandbox_approval::SandboxApprovalRequest;
use super::tools::{Tool, ToolParam};
use std::collections::{HashMap, HashSet};

pub const SANDBOX_PERMISSIONS_METADATA: &str = "sandbox_permissions";
pub const JUSTIFICATION_METADATA: &str = "justification";
pub const RESOURCE_LIMITS_METADATA: &str = "resource_limits";
pub const EGRESS_POLICY_METADATA: &str = "egress_policy";

const RESERVED_METADATA: &[&str] = &[
    SANDBOX_PERMISSIONS_METADATA,
    JUSTIFICATION_METADATA,
    RESOURCE_LIMITS_METADATA,
    EGRESS_POLICY_METADATA,
];

/// A validated tool invocation presented to runtime policy hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub tool: String,
    pub params: HashMap<String, String>,
    pub sandbox_permissions: Option<String>,
    pub justification: Option<String>,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        tool: impl Into<String>,
        params: HashMap<String, String>,
    ) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            params,
            sandbox_permissions: None,
            justification: None,
        }
    }

    pub fn with_sandbox_metadata(
        mut self,
        sandbox_permissions: Option<String>,
        justification: Option<String>,
    ) -> Self {
        self.sandbox_permissions = sandbox_permissions;
        self.justification = justification;
        self
    }
}

/// Policy seam around tool execution. Policies may refuse a call before any
/// side effect and observe the completed output afterwards.
pub trait ToolPolicy {
    fn before_execute(&self, _call: &ToolCall, _tool: &Tool) -> Result<(), String> {
        Ok(())
    }

    /// Sandbox widening is a distinct privilege and therefore fails closed
    /// unless the active policy explicitly implements an approval channel.
    fn approve_sandbox_escalation(
        &self,
        _call: &ToolCall,
        _tool: &Tool,
        _request: &SandboxApprovalRequest,
    ) -> Result<(), String> {
        Err("sandbox escalation is not supported by this tool policy".to_string())
    }

    fn after_execute(&self, _call: &ToolCall, _tool: &Tool, _output: &str) {}
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAllPolicy;

impl ToolPolicy for AllowAllPolicy {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolRuntimeError {
    DuplicateTool(String),
    DuplicateParameter { tool: String, parameter: String },
    UnknownTool(String),
    MissingRequiredParameter { tool: String, parameter: String },
    UndeclaredParameter { tool: String, parameter: String },
    ReservedParameter { tool: String, parameter: String },
    PolicyDenied { tool: String, reason: String },
    SandboxEscalationDenied { tool: String, reason: String },
    GovernanceDenied { tool: String, reason: String },
}

impl std::fmt::Display for ToolRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::DuplicateTool(tool) => write!(f, "Duplicate tool registration: {tool}"),
            Self::DuplicateParameter { tool, parameter } =>
            {
                write!(f, "Duplicate parameter {parameter:?} in tool {tool:?}")
            },
            Self::UnknownTool(tool) => write!(f, "Unknown tool: {tool}"),
            Self::MissingRequiredParameter { tool, parameter } => write!(
                f,
                "Missing required parameter {parameter:?} for tool {tool:?}"
            ),
            Self::UndeclaredParameter { tool, parameter } =>
            {
                write!(f, "Undeclared parameter {parameter:?} for tool {tool:?}")
            },
            Self::ReservedParameter { tool, parameter } => write!(
                f,
                "Parameter {parameter:?} is reserved runtime metadata for tool {tool:?}"
            ),
            Self::PolicyDenied { tool, reason } =>
            {
                write!(f, "Tool {tool:?} denied by policy: {reason}")
            },
            Self::SandboxEscalationDenied { tool, reason } => write!(
                f,
                "Sandbox escalation for tool {tool:?} was refused: {reason}"
            ),
            Self::GovernanceDenied { tool, reason } =>
            {
                write!(f, "Tool {tool:?} refused by resource governance: {reason}")
            },
        }
    }
}

impl std::error::Error for ToolRuntimeError {}

/// Contract-validating runtime around the existing hardened SciAgent tools.
///
/// Built-in path checks, process limits, and secret stripping remain intact;
/// process-spawning callbacks are additionally routed through the sandbox seam.
/// This layer centralises schema validation and pre/post execution policy.
pub struct ToolRuntime<P = AllowAllPolicy> {
    tools: Vec<Tool>,
    policy: P,
}

impl ToolRuntime<AllowAllPolicy> {
    pub fn builtins() -> Self {
        let mut tools = Tool::builtins();
        super::sandbox::install_process_sandbox(&mut tools);
        Self::new(tools, AllowAllPolicy).expect("built-in SciAgent tool contracts must be valid")
    }
}

impl Default for ToolRuntime<AllowAllPolicy> {
    fn default() -> Self {
        Self::builtins()
    }
}

impl<P: ToolPolicy> ToolRuntime<P> {
    pub fn new(tools: Vec<Tool>, policy: P) -> Result<Self, ToolRuntimeError> {
        validate_registry(&tools)?;
        Ok(Self { tools, policy })
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|tool| tool.name == name)
    }

    pub fn tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|tool| tool.name == name)
    }

    pub fn validate_call(&self, call: &ToolCall) -> Result<&Tool, ToolRuntimeError> {
        let tool = self
            .tool(&call.tool)
            .ok_or_else(|| ToolRuntimeError::UnknownTool(call.tool.clone()))?;
        validate_parameters(tool, &call.params)?;
        Ok(tool)
    }

    /// Execute with explicit resource governance.
    ///
    /// The constraints travel as reserved call metadata, are checked for
    /// enforceability against the probed kernel backend BEFORE the policy
    /// hook runs, and reach the sandboxed spawn path where they become real
    /// rlimits, CPU pinning and a seccomp deny-all socket filter. A tool
    /// that does not execute through that path refuses governance instead
    /// of silently running unbounded.
    pub fn execute_governed(
        &self,
        call: &ToolCall,
        constraints: &ExecutionConstraints,
    ) -> Result<String, ToolRuntimeError> {
        let mut governed = call.clone();
        for (key, value) in constraints.to_metadata()
        {
            governed.params.insert(key, value);
        }
        self.execute(&governed)
    }

    fn requested_constraints(
        &self,
        call: &ToolCall,
    ) -> Result<Option<ExecutionConstraints>, String> {
        ExecutionConstraints::from_metadata(
            call.params
                .get(RESOURCE_LIMITS_METADATA)
                .map(String::as_str),
            call.params.get(EGRESS_POLICY_METADATA).map(String::as_str),
        )
    }

    pub fn execute(&self, call: &ToolCall) -> Result<String, ToolRuntimeError> {
        let tool = self.validate_call(call)?;

        // Governance gate: declared limits must be enforceable before any
        // side effect. This is the H-5 contract — a limit the active kernel
        // backend cannot enforce denies the call instead of running
        // unbounded.
        let constraints = self.requested_constraints(call).map_err(|reason| {
            ToolRuntimeError::GovernanceDenied {
                tool: call.tool.clone(),
                reason,
            }
        })?;
        if let Some(constraints) = &constraints
        {
            if !super::sandbox::tool_supports_sandbox(&call.tool)
            {
                return Err(ToolRuntimeError::GovernanceDenied {
                    tool: call.tool.clone(),
                    reason: "this tool does not execute through the governed process-spawn path"
                        .to_string(),
                });
            }
            constraints
                .ensure_enforceable(probed_backend())
                .map_err(|reason| ToolRuntimeError::GovernanceDenied {
                    tool: call.tool.clone(),
                    reason,
                })?;
        }

        self.policy.before_execute(call, tool).map_err(|reason| {
            ToolRuntimeError::PolicyDenied {
                tool: call.tool.clone(),
                reason,
            }
        })?;

        let escalation = sandbox_request(call)?;
        let mut params = call.params.clone();
        if let Some(request) = escalation.as_ref()
        {
            self.policy
                .approve_sandbox_escalation(call, tool, request)
                .map_err(|reason| ToolRuntimeError::SandboxEscalationDenied {
                    tool: call.tool.clone(),
                    reason,
                })?;
            super::sandbox::install_one_shot_override(&mut params, request.requested);
        }

        let output = (tool.execute)(params);
        self.policy.after_execute(call, tool, &output);
        Ok(output)
    }

    pub fn execute_named(
        &self,
        tool: &str,
        mut params: HashMap<String, String>,
    ) -> Result<String, ToolRuntimeError> {
        let sandbox_permissions = params.remove(SANDBOX_PERMISSIONS_METADATA);
        let justification = params.remove(JUSTIFICATION_METADATA);
        self.execute(
            &ToolCall::new("legacy", tool, params)
                .with_sandbox_metadata(sandbox_permissions, justification),
        )
    }
}

fn sandbox_request(call: &ToolCall) -> Result<Option<SandboxApprovalRequest>, ToolRuntimeError> {
    if call.sandbox_permissions.is_none() && call.justification.is_none()
    {
        return Ok(None);
    }
    if !super::sandbox::tool_supports_sandbox(&call.tool)
    {
        return Err(ToolRuntimeError::SandboxEscalationDenied {
            tool: call.tool.clone(),
            reason: "this tool does not execute through the process sandbox".to_string(),
        });
    }
    let current = super::sandbox::configured_permission().map_err(|reason| {
        ToolRuntimeError::SandboxEscalationDenied {
            tool: call.tool.clone(),
            reason,
        }
    })?;
    SandboxApprovalRequest::from_metadata(
        call.id.clone(),
        call.tool.clone(),
        current,
        call.sandbox_permissions.as_deref(),
        call.justification.as_deref(),
    )
    .map_err(|error| ToolRuntimeError::SandboxEscalationDenied {
        tool: call.tool.clone(),
        reason: error.to_string(),
    })
}

fn validate_registry(tools: &[Tool]) -> Result<(), ToolRuntimeError> {
    let mut names = HashSet::new();
    for tool in tools
    {
        if !names.insert(tool.name)
        {
            return Err(ToolRuntimeError::DuplicateTool(tool.name.to_string()));
        }
        let mut params = HashSet::new();
        for parameter in &tool.parameters
        {
            if RESERVED_METADATA.contains(&parameter.name)
            {
                return Err(ToolRuntimeError::ReservedParameter {
                    tool: tool.name.to_string(),
                    parameter: parameter.name.to_string(),
                });
            }
            if !params.insert(parameter.name)
            {
                return Err(ToolRuntimeError::DuplicateParameter {
                    tool: tool.name.to_string(),
                    parameter: parameter.name.to_string(),
                });
            }
        }
    }
    Ok(())
}

fn validate_parameters(
    tool: &Tool,
    params: &HashMap<String, String>,
) -> Result<(), ToolRuntimeError> {
    for ToolParam { name, required, .. } in &tool.parameters
    {
        if *required && params.get(*name).is_none_or(String::is_empty)
        {
            return Err(ToolRuntimeError::MissingRequiredParameter {
                tool: tool.name.to_string(),
                parameter: (*name).to_string(),
            });
        }
    }

    for name in params.keys()
    {
        // Reserved runtime metadata travels inside params but is owned by
        // the runtime: the governance gate parses it after schema
        // validation, so it is never an "undeclared parameter".
        if RESERVED_METADATA.contains(&name.as_str())
        {
            continue;
        }
        if !tool
            .parameters
            .iter()
            .any(|parameter| parameter.name == name.as_str())
        {
            return Err(ToolRuntimeError::UndeclaredParameter {
                tool: tool.name.to_string(),
                parameter: name.clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static OBSERVATIONS: AtomicUsize = AtomicUsize::new(0);

    fn panic_tool(_params: HashMap<String, String>) -> String {
        panic!("tool callback must not run")
    }

    fn ok_tool(_params: HashMap<String, String>) -> String {
        "ok".to_string()
    }

    fn synthetic_tool(execute: fn(HashMap<String, String>) -> String) -> Tool {
        Tool {
            name: "synthetic",
            description: "synthetic runtime test tool",
            parameters: vec![
                ToolParam {
                    name: "required",
                    param_type: "string",
                    description: "required input",
                    required: true,
                },
                ToolParam {
                    name: "optional",
                    param_type: "string",
                    description: "optional input",
                    required: false,
                },
            ],
            execute,
        }
    }

    struct DenyPolicy;

    impl ToolPolicy for DenyPolicy {
        fn before_execute(&self, call: &ToolCall, _tool: &Tool) -> Result<(), String> {
            if call.tool == "synthetic"
            {
                Err("synthetic denial".to_string())
            }
            else
            {
                Ok(())
            }
        }
    }

    struct ObservePolicy;

    impl ToolPolicy for ObservePolicy {
        fn after_execute(&self, _call: &ToolCall, _tool: &Tool, output: &str) {
            assert_eq!(output, "ok");
            OBSERVATIONS.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn params() -> HashMap<String, String> {
        [("required".to_string(), "value".to_string())]
            .into_iter()
            .collect()
    }

    #[test]
    fn builtins_have_unique_contracts() {
        let runtime = ToolRuntime::builtins();
        assert!(runtime.has_tool("search"));
        assert!(runtime.has_tool("status"));
    }

    #[test]
    fn unknown_tool_is_refused() {
        let runtime = ToolRuntime::builtins();
        let error = runtime
            .execute_named("does-not-exist", HashMap::new())
            .expect_err("unknown tool must fail");
        assert_eq!(
            error,
            ToolRuntimeError::UnknownTool("does-not-exist".to_string())
        );
    }

    #[test]
    fn missing_required_parameter_is_refused_before_callback() {
        let runtime = ToolRuntime::new(vec![synthetic_tool(panic_tool)], AllowAllPolicy).unwrap();
        let error = runtime
            .execute_named("synthetic", HashMap::new())
            .expect_err("required parameter must be enforced");
        assert!(matches!(
            error,
            ToolRuntimeError::MissingRequiredParameter { .. }
        ));
    }

    #[test]
    fn empty_required_parameter_is_refused_before_callback() {
        let runtime = ToolRuntime::new(vec![synthetic_tool(panic_tool)], AllowAllPolicy).unwrap();
        let mut call_params = params();
        call_params.insert("required".to_string(), String::new());
        let error = runtime
            .execute_named("synthetic", call_params)
            .expect_err("empty required parameter must be enforced");
        assert!(matches!(
            error,
            ToolRuntimeError::MissingRequiredParameter { .. }
        ));
    }

    #[test]
    fn undeclared_parameter_is_refused_before_callback() {
        let runtime = ToolRuntime::new(vec![synthetic_tool(panic_tool)], AllowAllPolicy).unwrap();
        let mut call_params = params();
        call_params.insert("surprise".to_string(), "value".to_string());
        let error = runtime
            .execute_named("synthetic", call_params)
            .expect_err("undeclared parameter must fail");
        assert!(matches!(
            error,
            ToolRuntimeError::UndeclaredParameter { .. }
        ));
    }

    #[test]
    fn policy_denial_prevents_callback() {
        let runtime = ToolRuntime::new(vec![synthetic_tool(panic_tool)], DenyPolicy).unwrap();
        let error = runtime
            .execute_named("synthetic", params())
            .expect_err("policy must deny");
        assert_eq!(
            error,
            ToolRuntimeError::PolicyDenied {
                tool: "synthetic".to_string(),
                reason: "synthetic denial".to_string(),
            }
        );
    }

    #[test]
    fn successful_execution_runs_post_hook() {
        OBSERVATIONS.store(0, Ordering::SeqCst);
        let runtime = ToolRuntime::new(vec![synthetic_tool(ok_tool)], ObservePolicy).unwrap();
        assert_eq!(runtime.execute_named("synthetic", params()).unwrap(), "ok");
        assert_eq!(OBSERVATIONS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn allow_all_policy_cannot_approve_sandbox_escalation() {
        use super::super::sandbox_approval::{SandboxApprovalRequest, SandboxPermission};

        let policy = AllowAllPolicy;
        let request = SandboxApprovalRequest::new(
            "call-1",
            "status",
            SandboxPermission::WorkspaceWrite,
            SandboxPermission::DangerFullAccess,
            "one exact status call needs an explicit wider policy",
        )
        .unwrap();
        let tool = Tool {
            name: "status",
            description: "synthetic status",
            parameters: Vec::new(),
            execute: panic_tool,
        };
        let call = ToolCall::new("call-1", "status", HashMap::new());
        assert!(
            policy
                .approve_sandbox_escalation(&call, &tool, &request)
                .is_err()
        );
    }

    #[test]
    fn reserved_metadata_cannot_be_registered_as_tool_parameter() {
        let mut tool = synthetic_tool(ok_tool);
        tool.parameters.push(ToolParam {
            name: SANDBOX_PERMISSIONS_METADATA,
            param_type: "string",
            description: "must stay runtime metadata",
            required: false,
        });
        let error = match ToolRuntime::new(vec![tool], AllowAllPolicy)
        {
            Ok(_) => panic!("reserved runtime metadata must not enter a tool schema"),
            Err(error) => error,
        };
        assert!(matches!(error, ToolRuntimeError::ReservedParameter { .. }));
    }

    #[test]
    fn duplicate_tool_registration_is_refused() {
        let result = ToolRuntime::new(
            vec![synthetic_tool(ok_tool), synthetic_tool(ok_tool)],
            AllowAllPolicy,
        );
        let error = match result
        {
            Ok(_) => panic!("duplicate tools must fail"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            ToolRuntimeError::DuplicateTool("synthetic".to_string())
        );
    }

    // -- H-5 resource governance -------------------------------------------

    static GOVERNED_RUNS: AtomicUsize = AtomicUsize::new(0);

    fn governed_status_tool(_params: HashMap<String, String>) -> String {
        GOVERNED_RUNS.fetch_add(1, Ordering::SeqCst);
        "ran".to_string()
    }

    fn status_tool() -> Tool {
        Tool {
            name: "status",
            description: "governance probe tool",
            parameters: Vec::new(),
            execute: governed_status_tool,
        }
    }

    /// A governed `status` tool whose execution is a hard failure: denial
    /// paths must never reach it, so any callback invocation fails the test.
    fn forbidden_status_tool(_params: HashMap<String, String>) -> String {
        panic!("tool callback must not run under refused governance")
    }

    fn forbidden_status_tool_tool() -> Tool {
        Tool {
            name: "status",
            description: "governance probe tool",
            parameters: Vec::new(),
            execute: forbidden_status_tool,
        }
    }

    fn file_size_constraints() -> crate::agentic::ExecutionConstraints {
        use crate::agentic::ExecutionConstraints;
        use crate::agentic::budgets::{EgressPolicy, ResourceLimits};
        ExecutionConstraints {
            limits: ResourceLimits {
                max_file_size_bytes: Some(4096),
                ..ResourceLimits::default()
            },
            egress: EgressPolicy::deny_all(),
        }
    }

    #[test]
    fn governed_call_runs_when_limits_are_enforceable() {
        if !crate::agentic::enforcement::probed_backend().supports_egress_deny_all()
        {
            return;
        }
        GOVERNED_RUNS.store(0, Ordering::SeqCst);
        let runtime = ToolRuntime::new(vec![status_tool()], AllowAllPolicy).unwrap();
        let output = runtime
            .execute_governed(
                &ToolCall::new("g1", "status", HashMap::new()),
                &file_size_constraints(),
            )
            .expect("enforceable governance must run");
        assert_eq!(output, "ran");
        assert_eq!(GOVERNED_RUNS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unenforceable_gpu_limit_denies_before_execution() {
        let mut constraints = file_size_constraints();
        constraints.limits.gpu_memory_bytes = Some(1 << 30);
        let runtime = ToolRuntime::new(vec![forbidden_status_tool_tool()], AllowAllPolicy).unwrap();
        let error = runtime
            .execute_governed(&ToolCall::new("g2", "status", HashMap::new()), &constraints)
            .expect_err("gpu memory caps are not kernel-enforceable and must refuse");
        assert!(
            matches!(error, ToolRuntimeError::GovernanceDenied { ref reason, .. }
                if reason.contains("gpu memory")),
            "{error}"
        );
    }

    #[test]
    fn egress_allow_list_denies_on_kernel_backend() {
        use crate::agentic::ExecutionConstraints;
        use crate::agentic::budgets::{EgressPolicy, EgressTarget};
        let constraints = ExecutionConstraints {
            egress: EgressPolicy::with_targets(vec![EgressTarget::new("example.com", vec![443])]),
            ..ExecutionConstraints::default()
        };
        let runtime = ToolRuntime::new(vec![forbidden_status_tool_tool()], AllowAllPolicy).unwrap();
        let error = match runtime
            .execute_governed(&ToolCall::new("g3", "status", HashMap::new()), &constraints)
        {
            Err(error) => error,
            // Only reachable on hypothetical backends that enforce
            // host-level filtering; nothing to prove there.
            Ok(_) => return,
        };
        assert!(
            matches!(error, ToolRuntimeError::GovernanceDenied { ref reason, .. }
                if reason.contains("allow-list")),
            "{error}"
        );
    }

    #[test]
    fn tools_outside_the_spawn_path_refuse_governance() {
        if !crate::agentic::enforcement::probed_backend().supports_egress_deny_all()
        {
            return;
        }
        let runtime = ToolRuntime::new(vec![synthetic_tool(panic_tool)], AllowAllPolicy).unwrap();
        let error = runtime
            .execute_governed(
                &ToolCall::new("g4", "synthetic", params()),
                &file_size_constraints(),
            )
            .expect_err("non-sandboxed tools must refuse declared-but-unenforceable limits");
        assert!(
            matches!(error, ToolRuntimeError::GovernanceDenied { ref reason, .. }
                if reason.contains("process-spawn path")),
            "{error}"
        );
    }

    #[test]
    fn malformed_inline_metadata_is_a_governance_denial() {
        let runtime = ToolRuntime::builtins();
        let mut call_params = HashMap::new();
        call_params.insert(RESOURCE_LIMITS_METADATA.to_string(), "not-json".to_string());
        let error = runtime
            .execute_named("status", call_params)
            .expect_err("malformed governance metadata must refuse");
        assert!(
            matches!(error, ToolRuntimeError::GovernanceDenied { ref reason, .. }
                if reason.contains("malformed resource_limits")),
            "{error}"
        );
    }

    #[test]
    fn governance_metadata_names_stay_reserved_in_schemas() {
        for reserved in [RESOURCE_LIMITS_METADATA, EGRESS_POLICY_METADATA]
        {
            let mut tool = synthetic_tool(ok_tool);
            tool.parameters.push(ToolParam {
                name: reserved,
                param_type: "string",
                description: "must stay runtime metadata",
                required: false,
            });
            let error = match ToolRuntime::new(vec![tool], AllowAllPolicy)
            {
                Ok(_) => panic!("reserved metadata {reserved} must not enter a tool schema"),
                Err(error) => error,
            };
            assert!(matches!(error, ToolRuntimeError::ReservedParameter { .. }));
        }
    }

    #[test]
    fn reserved_metadata_params_are_not_undeclared_parameters() {
        let runtime = ToolRuntime::new(vec![synthetic_tool(ok_tool)], DenyPolicy).unwrap();
        let mut call_params = params();
        call_params.insert(
            EGRESS_POLICY_METADATA.to_string(),
            r#"{"allow": []}"#.to_string(),
        );
        // Either denial proves schema validation tolerated the reserved key;
        // an UndeclaredParameter error would mean governance metadata cannot
        // travel inside call parameters at all.
        let error = runtime
            .execute_named("synthetic", call_params)
            .expect_err("policy still denies the call");
        assert!(
            matches!(
                error,
                ToolRuntimeError::PolicyDenied { .. } | ToolRuntimeError::GovernanceDenied { .. }
            ),
            "{error}"
        );
    }
}
