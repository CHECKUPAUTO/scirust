use super::tools::{Tool, ToolParam};
use std::collections::{HashMap, HashSet};

/// A validated tool invocation presented to runtime policy hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub tool: String,
    pub params: HashMap<String, String>,
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
        }
    }
}

/// Policy seam around tool execution. Policies may refuse a call before any
/// side effect and observe the completed output afterwards.
pub trait ToolPolicy {
    fn before_execute(&self, _call: &ToolCall, _tool: &Tool) -> Result<(), String> {
        Ok(())
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
    PolicyDenied { tool: String, reason: String },
}

impl std::fmt::Display for ToolRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::DuplicateTool(tool) => write!(f, "Duplicate tool registration: {tool}"),
            Self::DuplicateParameter { tool, parameter } => {
                write!(f, "Duplicate parameter {parameter:?} in tool {tool:?}")
            },
            Self::UnknownTool(tool) => write!(f, "Unknown tool: {tool}"),
            Self::MissingRequiredParameter { tool, parameter } => write!(
                f,
                "Missing required parameter {parameter:?} for tool {tool:?}"
            ),
            Self::UndeclaredParameter { tool, parameter } => {
                write!(f, "Undeclared parameter {parameter:?} for tool {tool:?}")
            },
            Self::PolicyDenied { tool, reason } => {
                write!(f, "Tool {tool:?} denied by policy: {reason}")
            },
        }
    }
}

impl std::error::Error for ToolRuntimeError {}

/// Contract-validating runtime around the existing hardened SciAgent tools.
///
/// The built-in tool callbacks continue to own filesystem/process confinement;
/// this layer centralises schema validation and pre/post execution policy.
pub struct ToolRuntime<P = AllowAllPolicy> {
    tools: Vec<Tool>,
    policy: P,
}

impl ToolRuntime<AllowAllPolicy> {
    pub fn builtins() -> Self {
        Self::new(Tool::builtins(), AllowAllPolicy)
            .expect("built-in SciAgent tool contracts must be valid")
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

    pub fn execute(&self, call: &ToolCall) -> Result<String, ToolRuntimeError> {
        let tool = self.validate_call(call)?;
        self.policy
            .before_execute(call, tool)
            .map_err(|reason| ToolRuntimeError::PolicyDenied {
                tool: call.tool.clone(),
                reason,
            })?;
        let output = (tool.execute)(call.params.clone());
        self.policy.after_execute(call, tool, &output);
        Ok(output)
    }

    pub fn execute_named(
        &self,
        tool: &str,
        params: HashMap<String, String>,
    ) -> Result<String, ToolRuntimeError> {
        self.execute(&ToolCall::new("legacy", tool, params))
    }
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
        assert_eq!(
            runtime.execute_named("synthetic", params()).unwrap(),
            "ok"
        );
        assert_eq!(OBSERVATIONS.load(Ordering::SeqCst), 1);
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
}
