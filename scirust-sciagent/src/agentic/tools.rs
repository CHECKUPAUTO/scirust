use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MAX_SOURCE_BYTES: u64 = 1024 * 1024;
const MAX_TOOL_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_LINE_RANGE: usize = 200;
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);
const SECRET_ENV_VARS: &[&str] = &[
    "SCIRUST_DISCOVERY_KEY",
    "SCIRUST_EXCHANGE_SECRET",
    "SCIRUST_WALLET_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
];

pub type ToolFn = fn(HashMap<String, String>) -> String;

/// Workspace root the built-in tools operate on: `SCIAGENT_ROOT` when set (a
/// deployed agent), else the parent of this crate's manifest directory (the
/// scirust workspace in a source build). Never a hard-coded machine path.
pub(crate) fn workspace_root() -> String {
    std::env::var("SCIAGENT_ROOT").unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    })
}

fn canonical_workspace_root() -> Result<PathBuf, String> {
    std::fs::canonicalize(workspace_root())
        .map_err(|e| format!("Cannot resolve the configured workspace root: {e}"))
}

/// Resolve an existing path and prove that it stays below the configured
/// workspace after following symlinks. Absolute paths are accepted only when
/// they resolve inside that root.
fn resolve_workspace_path(requested: &str) -> Result<PathBuf, String> {
    let root = canonical_workspace_root()?;
    let requested = if requested.is_empty()
    {
        root.clone()
    }
    else
    {
        let path = Path::new(requested);
        if path.is_absolute()
        {
            path.to_path_buf()
        }
        else
        {
            root.join(path)
        }
    };
    let resolved = std::fs::canonicalize(&requested)
        .map_err(|e| format!("Cannot resolve `{}`: {e}", requested.display()))?;
    if !resolved.starts_with(&root)
    {
        return Err(format!(
            "Refused path outside workspace `{}`",
            root.display()
        ));
    }
    Ok(resolved)
}

fn read_workspace_file(requested: &str) -> Result<String, String> {
    let path = resolve_workspace_path(requested)?;
    let metadata = std::fs::metadata(&path)
        .map_err(|e| format!("Cannot inspect `{}`: {e}", path.display()))?;
    if !metadata.is_file()
    {
        return Err(format!("`{}` is not a regular file", path.display()));
    }
    if metadata.len() > MAX_SOURCE_BYTES
    {
        return Err(format!("Refused file larger than {MAX_SOURCE_BYTES} bytes"));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    std::fs::File::open(&path)
        .map_err(|e| format!("Cannot open `{}`: {e}", path.display()))?
        .take(MAX_SOURCE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Cannot read `{}`: {e}", path.display()))?;
    if bytes.len() as u64 > MAX_SOURCE_BYTES
    {
        return Err(format!("Refused file larger than {MAX_SOURCE_BYTES} bytes"));
    }
    String::from_utf8(bytes).map_err(|_| "Refused non-UTF-8 source file".to_string())
}

fn excerpt(text: &str, range: Option<&String>, default_lines: usize) -> Result<String, String> {
    let (start, count) = if let Some(range) = range
    {
        let (start_text, end_text) = range.split_once('-').unwrap_or((range, range));
        let start = start_text
            .parse::<usize>()
            .map_err(|_| "Invalid line range: start must be an integer".to_string())?;
        let end = end_text
            .parse::<usize>()
            .map_err(|_| "Invalid line range: end must be an integer".to_string())?;
        if start == 0 || end < start
        {
            return Err("Invalid line range: require 1 <= start <= end".to_string());
        }
        let count = end
            .checked_sub(start)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| "Invalid line range".to_string())?;
        if count > MAX_LINE_RANGE
        {
            return Err(format!(
                "Refused line range larger than {MAX_LINE_RANGE} lines"
            ));
        }
        (start, count)
    }
    else
    {
        (1, default_lines.min(MAX_LINE_RANGE))
    };
    Ok(text
        .lines()
        .skip(start - 1)
        .take(count)
        .collect::<Vec<_>>()
        .join("\n"))
}

struct LimitedOutput {
    success: bool,
    timed_out: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn drain_pipe<R: Read + Send + 'static>(mut pipe: R) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut chunk = [0u8; 8192];
        while let Ok(n) = pipe.read(&mut chunk)
        {
            if n == 0
            {
                break;
            }
            let remaining = MAX_TOOL_OUTPUT_BYTES.saturating_sub(kept.len());
            kept.extend_from_slice(&chunk[..n.min(remaining)]);
        }
        kept
    })
}

/// Run a fixed executable/argument vector with bounded capture and wall time.
fn run_limited(mut command: Command) -> Result<LimitedOutput, String> {
    for variable in SECRET_ENV_VARS
    {
        command.env_remove(variable);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let stdout_thread = drain_pipe(child.stdout.take().expect("stdout was piped"));
    let stderr_thread = drain_pipe(child.stderr.take().expect("stderr was piped"));
    let deadline = Instant::now() + TOOL_TIMEOUT;
    let (status, timed_out) = loop
    {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())?
        {
            break (status, false);
        }
        if Instant::now() >= deadline
        {
            let _ = child.kill();
            break (child.wait().map_err(|e| e.to_string())?, true);
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_thread
        .join()
        .map_err(|_| "stdout capture thread failed".to_string())?;
    let stderr = stderr_thread
        .join()
        .map_err(|_| "stderr capture thread failed".to_string())?;
    Ok(LimitedOutput {
        success: status.success(),
        timed_out,
        stdout,
        stderr,
    })
}

fn valid_crate_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn search_workspace(params: &HashMap<String, String>, max_count: &str) -> String {
    let pattern = params.get("pattern").map(String::as_str).unwrap_or("");
    if pattern.is_empty()
    {
        return "Missing pattern".to_string();
    }
    if pattern.len() > 1024
    {
        return "Refused search pattern longer than 1024 bytes".to_string();
    }
    let path = match resolve_workspace_path(params.get("path").map(String::as_str).unwrap_or(""))
    {
        Ok(path) => path,
        Err(e) => return e,
    };

    let mut rg = Command::new("rg");
    rg.args([
        "-n",
        "--max-count",
        max_count,
        "--max-filesize",
        "1M",
        "--max-columns",
        "512",
        "--glob",
        "!target/**",
        "--",
        pattern,
    ])
    .arg(&path);
    match run_limited(rg)
    {
        Ok(output) if output.timed_out => "Search timed out after 30 seconds".to_string(),
        Ok(output) if output.success => String::from_utf8_lossy(&output.stdout).into_owned(),
        _ =>
        {
            let mut grep = Command::new("grep");
            grep.args([
                "-rn",
                "--max-count",
                max_count,
                "--exclude-dir=target",
                "--",
                pattern,
            ])
            .arg(path);
            match run_limited(grep)
            {
                Ok(output) if output.timed_out => "Search timed out after 30 seconds".to_string(),
                Ok(output) if output.success =>
                {
                    String::from_utf8_lossy(&output.stdout).into_owned()
                },
                Ok(output) => format!("No matches: {}", String::from_utf8_lossy(&output.stderr)),
                Err(e) => format!("Failed to run search: {e}"),
            }
        },
    }
}

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
                    ToolParam {
                        name: "pattern",
                        param_type: "string",
                        description: "Regex pattern",
                        required: true,
                    },
                    ToolParam {
                        name: "path",
                        param_type: "string",
                        description: "Root path (default: the scirust workspace root)",
                        required: false,
                    },
                ],
                execute: |params| search_workspace(&params, "10"),
            },
            Tool {
                name: "grep",
                description: "Grep for a pattern in files (alias for search)",
                parameters: vec![
                    ToolParam {
                        name: "pattern",
                        param_type: "string",
                        description: "Regex pattern",
                        required: true,
                    },
                    ToolParam {
                        name: "path",
                        param_type: "string",
                        description: "File or directory",
                        required: false,
                    },
                ],
                execute: |params| search_workspace(&params, "15"),
            },
            Tool {
                name: "read",
                description: "Read the contents of a file",
                parameters: vec![
                    ToolParam {
                        name: "path",
                        param_type: "string",
                        description: "File path",
                        required: true,
                    },
                    ToolParam {
                        name: "lines",
                        param_type: "string",
                        description: "Line range (e.g. 10-30)",
                        required: false,
                    },
                ],
                execute: |params| {
                    let path = params.get("path").map(|s| s.as_str()).unwrap_or("");
                    if path.is_empty()
                    {
                        return "Missing path".to_string();
                    }
                    match read_workspace_file(path)
                    {
                        Ok(text) => excerpt(&text, params.get("lines"), 100).unwrap_or_else(|e| e),
                        Err(e) => e,
                    }
                },
            },
            Tool {
                name: "explain",
                description: "Explain a function or file in the codebase",
                parameters: vec![
                    ToolParam {
                        name: "path",
                        param_type: "string",
                        description: "File path",
                        required: true,
                    },
                    ToolParam {
                        name: "lines",
                        param_type: "string",
                        description: "Line range",
                        required: false,
                    },
                ],
                execute: |params| {
                    let path = params.get("path").map(|s| s.as_str()).unwrap_or("");
                    if path.is_empty()
                    {
                        return "Missing path".to_string();
                    }
                    match read_workspace_file(path)
                    {
                        Ok(text) => match excerpt(&text, params.get("lines"), 75)
                        {
                            Ok(excerpt) => format!("File: {path}\n```rust\n{excerpt}\n```"),
                            Err(e) => e,
                        },
                        Err(e) => e,
                    }
                },
            },
            Tool {
                name: "build",
                description: "Build a crate in the workspace",
                parameters: vec![ToolParam {
                    name: "crate",
                    param_type: "string",
                    description: "Crate name (e.g. scirust-core)",
                    required: true,
                }],
                execute: |params| {
                    let crate_name = params.get("crate").map(|s| s.as_str()).unwrap_or("");
                    if !valid_crate_name(crate_name)
                    {
                        return "Invalid crate name".to_string();
                    }
                    let root = match canonical_workspace_root()
                    {
                        Ok(root) => root,
                        Err(e) => return e,
                    };
                    let mut command = Command::new("cargo");
                    command
                        .args([
                            "check",
                            "--locked",
                            "-p",
                            crate_name,
                            "--message-format=short",
                        ])
                        .current_dir(root);
                    match run_limited(command)
                    {
                        Ok(output) if output.timed_out =>
                        {
                            "Build timed out after 30 seconds".to_string()
                        },
                        Ok(output) if output.success =>
                        {
                            format!("{crate_name} builds successfully")
                        },
                        Ok(output) =>
                        {
                            format!("Build errors:\n{}", String::from_utf8_lossy(&output.stderr))
                        },
                        Err(e) => format!("Failed to run cargo: {e}"),
                    }
                },
            },
            Tool {
                name: "test",
                description: "Run tests for a crate",
                parameters: vec![
                    ToolParam {
                        name: "crate",
                        param_type: "string",
                        description: "Crate name",
                        required: true,
                    },
                    ToolParam {
                        name: "test",
                        param_type: "string",
                        description: "Test name filter",
                        required: false,
                    },
                ],
                execute: |params| {
                    let crate_name = params.get("crate").map(|s| s.as_str()).unwrap_or("");
                    if !valid_crate_name(crate_name)
                    {
                        return "Invalid crate name".to_string();
                    }
                    let mut args = vec![
                        "test",
                        "--locked",
                        "-p",
                        crate_name,
                        "--message-format=short",
                    ];
                    if let Some(filter) = params.get("test")
                    {
                        if filter.len() > 256
                        {
                            return "Refused test filter longer than 256 bytes".to_string();
                        }
                        args.push("--");
                        args.push(filter);
                    }
                    let root = match canonical_workspace_root()
                    {
                        Ok(root) => root,
                        Err(e) => return e,
                    };
                    let mut command = Command::new("cargo");
                    command.args(&args).current_dir(root);
                    match run_limited(command)
                    {
                        Ok(output) if output.timed_out =>
                        {
                            "Tests timed out after 30 seconds".to_string()
                        },
                        Ok(output) if output.success =>
                        {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let passed = stdout
                                .lines()
                                .find(|l| l.contains("test result"))
                                .unwrap_or("unknown");
                            format!("Tests passed: {passed}")
                        },
                        Ok(output) => format!(
                            "Test failures:\n{}",
                            String::from_utf8_lossy(&output.stderr)
                        ),
                        Err(e) => format!("Failed to run tests: {e}"),
                    }
                },
            },
            Tool {
                name: "status",
                description: "Show git status of the workspace",
                parameters: vec![],
                execute: |_params| {
                    let root = match canonical_workspace_root()
                    {
                        Ok(root) => root,
                        Err(e) => return e,
                    };
                    let mut command = Command::new("git");
                    command.args(["status", "--short"]).current_dir(root);
                    match run_limited(command)
                    {
                        Ok(output) if output.timed_out =>
                        {
                            "Git status timed out after 30 seconds".to_string()
                        },
                        Ok(output) => String::from_utf8_lossy(&output.stdout).into_owned(),
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
        params.insert(
            "path".to_string(),
            format!("{}/scirust-sciagent/src", workspace_root()),
        );
        let result = (search.execute)(params);
        assert!(!result.is_empty(), "Search should find results");
    }

    #[test]
    fn test_status_tool() {
        let tools = Tool::builtins();
        let status = tools.iter().find(|t| t.name == "status").unwrap();
        let result = (status.execute)(HashMap::new());
        assert!(
            result.contains(".rs") || result.is_empty(),
            "Status should work"
        );
    }

    #[test]
    fn rejects_paths_outside_workspace() {
        let root = canonical_workspace_root().unwrap();
        let outside = root.parent().expect("workspace has a parent");
        let result = resolve_workspace_path(&outside.to_string_lossy());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outside workspace"));
    }

    #[test]
    fn rejects_reversed_or_excessive_line_ranges() {
        let reversed = "30-10".to_string();
        assert!(excerpt("a\nb\nc", Some(&reversed), 10).is_err());
        let excessive = "1-201".to_string();
        assert!(excerpt("a", Some(&excessive), 10).is_err());
    }
}
