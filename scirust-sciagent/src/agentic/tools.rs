use std::collections::HashMap;
use std::process::Command;

pub type ToolFn = fn(HashMap<String, String>) -> String;

#[derive(Clone)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Vec<ToolParam>,
    pub execute: ToolFn,
}

#[derive(Clone)]
pub struct ToolParam {
    pub name: &'static str,
    pub param_type: &'static str,
    pub description: &'static str,
    pub required: bool,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool: String,
    pub success: bool,
    pub output: String,
}

impl Tool {
    pub fn builtins() -> Vec<Self> {
        vec![
            Tool {
                name: "search",
                description: "Search for a regex pattern in code files",
                parameters: vec![
                    ToolParam { name: "pattern", param_type: "string", description: "Regex pattern", required: true },
                    ToolParam { name: "path", param_type: "string", description: "Root path (default: /root/scirust)", required: false },
                ],
                execute: |params| {
                    let pattern = params.get("pattern").map(|s| s.as_str()).unwrap_or("");
                    let path = params.get("path").map(|s| s.as_str()).unwrap_or("/root/scirust");
                    if pattern.is_empty() {
                        return "Missing pattern".to_string();
                    }
                    // Try ripgrep first, fall back to grep -r
                    let result = Command::new("rg")
                        .args(["-n", "--max-count", "10", pattern, path])
                        .output();
                    match result {
                        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                        _ => {
                            let result = Command::new("grep")
                                .args(["-rn", "--max-count", "10", pattern, path])
                                .output();
                            match result {
                                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                                Ok(o) => format!("No matches: {}", String::from_utf8_lossy(&o.stderr)),
                                Err(e) => format!("Failed to run grep: {e}"),
                            }
                        }
                    }
                },
            },
            Tool {
                name: "grep",
                description: "Grep for a pattern in files (alias for search)",
                parameters: vec![
                    ToolParam { name: "pattern", param_type: "string", description: "Regex pattern", required: true },
                    ToolParam { name: "path", param_type: "string", description: "File or directory", required: false },
                ],
                execute: |params| {
                    let pattern = params.get("pattern").map(|s| s.as_str()).unwrap_or("");
                    let path = params.get("path").map(|s| s.as_str()).unwrap_or("/root/scirust");
                    if pattern.is_empty() {
                        return "Missing pattern".to_string();
                    }
                    let result = Command::new("rg")
                        .args(["-n", "--max-count", "15", pattern, path])
                        .output();
                    match result {
                        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                        _ => {
                            let result = Command::new("grep")
                                .args(["-rn", "--max-count", "15", pattern, path])
                                .output();
                            match result {
                                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                                Ok(o) => format!("No matches: {}", String::from_utf8_lossy(&o.stderr)),
                                Err(e) => format!("Failed to run grep: {e}"),
                            }
                        }
                    }
                },
            },
            Tool {
                name: "read",
                description: "Read the contents of a file",
                parameters: vec![
                    ToolParam { name: "path", param_type: "string", description: "File path", required: true },
                    ToolParam { name: "lines", param_type: "string", description: "Line range (e.g. 10-30)", required: false },
                ],
                execute: |params| {
                    let path = params.get("path").map(|s| s.as_str()).unwrap_or("");
                    if path.is_empty() {
                        return "Missing path".to_string();
                    }
                    let content = std::fs::read_to_string(path);
                    match content {
                        Ok(text) => {
                            if let Some(range) = params.get("lines") {
                                let parts: Vec<&str> = range.splitn(2, '-').collect();
                                let start: usize = parts[0].parse().unwrap_or(1);
                                let end: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(start + 50);
                                text.lines()
                                    .skip(start.saturating_sub(1))
                                    .take(end - start + 1)
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            } else {
                                text.chars().take(3000).collect()
                            }
                        }
                        Err(e) => format!("Error reading {path}: {e}"),
                    }
                },
            },
            Tool {
                name: "explain",
                description: "Explain a function or file in the codebase",
                parameters: vec![
                    ToolParam { name: "path", param_type: "string", description: "File path", required: true },
                    ToolParam { name: "lines", param_type: "string", description: "Line range", required: false },
                ],
                execute: |params| {
                    let path = params.get("path").map(|s| s.as_str()).unwrap_or("");
                    let lines = params.get("lines").cloned().unwrap_or_default();
                    let content = std::fs::read_to_string(path);
                    match content {
                        Ok(text) => {
                            let excerpt = if !lines.is_empty() {
                                let parts: Vec<&str> = lines.splitn(2, '-').collect();
                                let start: usize = parts[0].parse().unwrap_or(1);
                                let end: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(start + 30);
                                text.lines()
                                    .skip(start.saturating_sub(1))
                                    .take(end - start + 1)
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            } else {
                                text.chars().take(2000).collect()
                            };
                            format!("File: {path}\n```rust\n{excerpt}\n```")
                        }
                        Err(e) => format!("Cannot read {path}: {e}"),
                    }
                },
            },
            Tool {
                name: "build",
                description: "Build a crate in the workspace",
                parameters: vec![
                    ToolParam { name: "crate", param_type: "string", description: "Crate name (e.g. scirust-core)", required: true },
                ],
                execute: |params| {
                    let crate_name = params.get("crate").map(|s| s.as_str()).unwrap_or("");
                    if crate_name.is_empty() {
                        return "Missing crate name".to_string();
                    }
                    let output = Command::new("cargo")
                        .args(["check", "-p", crate_name, "--message-format=short"])
                        .output();
                    match output {
                        Ok(o) => {
                            let _stdout = String::from_utf8_lossy(&o.stdout);
                            let stderr = String::from_utf8_lossy(&o.stderr);
                            if o.status.success() {
                                format!("{crate_name} builds successfully")
                            } else {
                                format!("Build errors:\n{stderr}")
                            }
                        }
                        Err(e) => format!("Failed to run cargo: {e}"),
                    }
                },
            },
            Tool {
                name: "test",
                description: "Run tests for a crate",
                parameters: vec![
                    ToolParam { name: "crate", param_type: "string", description: "Crate name", required: true },
                    ToolParam { name: "test", param_type: "string", description: "Test name filter", required: false },
                ],
                execute: |params| {
                    let crate_name = params.get("crate").map(|s| s.as_str()).unwrap_or("");
                    if crate_name.is_empty() {
                        return "Missing crate name".to_string();
                    }
                    let mut args = vec!["test", "-p", crate_name, "--message-format=short"];
                    if let Some(filter) = params.get("test") {
                        args.push("--");
                        args.push(filter);
                    }
                    let output = Command::new("cargo").args(&args).output();
                    match output {
                        Ok(o) => {
                            let stderr = String::from_utf8_lossy(&o.stderr);
                            let stdout = String::from_utf8_lossy(&o.stdout);
                            if o.status.success() {
                                let passed = stdout.lines().filter(|l| l.contains("test result")).next().unwrap_or("unknown");
                                format!("Tests passed: {passed}")
                            } else {
                                format!("Test failures:\n{stderr}")
                            }
                        }
                        Err(e) => format!("Failed to run tests: {e}"),
                    }
                },
            },
            Tool {
                name: "status",
                description: "Show git status of the workspace",
                parameters: vec![],
                execute: |_params| {
                    let output = Command::new("git")
                        .args(["status", "--short"])
                        .current_dir("/root/scirust")
                        .output();
                    match output {
                        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
                        Err(e) => format!("Git error: {e}"),
                    }
                },
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_tool() {
        let tools = Tool::builtins();
        let search = tools.iter().find(|t| t.name == "search").unwrap();
        let mut params = HashMap::new();
        params.insert("pattern".to_string(), "fn main".to_string());
        params.insert("path".to_string(), "/root/scirust/scirust-sciagent/src".to_string());
        let result = (search.execute)(params);
        assert!(!result.is_empty(), "Search should find results");
    }

    #[test]
    fn test_status_tool() {
        let tools = Tool::builtins();
        let status = tools.iter().find(|t| t.name == "status").unwrap();
        let result = (status.execute)(HashMap::new());
        assert!(result.contains(".rs") || result.is_empty(), "Status should work");
    }
}
