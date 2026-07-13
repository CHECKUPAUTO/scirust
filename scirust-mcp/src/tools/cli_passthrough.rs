//! Échappatoire générique : exécute le binaire CLI `scirust` (voir
//! `scirust-cli`) comme sous-processus, pour exposer d'un coup toutes ses
//! commandes (`linsolve`, `solve`, `diff`, `integrate`, `ode`, `certify`,
//! `conformal`, `evo`, `analyze`, ...) sans réimplémenter chacune comme un
//! outil MCP dédié. Préférer un outil dédié (ex. `linalg_eigen_symmetric`)
//! quand il existe : il renvoie du JSON structuré au lieu de texte à
//! reparser, et documente son schéma d'entrée précisément.
//!
//! Résolution du binaire, dans l'ordre : `SCIRUST_BIN` (chemin explicite),
//! puis `scirust` sur `PATH`, puis `cargo run -p scirust-cli --` en dernier
//! recours (lent, mais fonctionne depuis un checkout source sans install
//! préalable).

use crate::registry::McpTool;
use serde_json::json;
use std::io::Read;
use std::path::{Component, Path};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MAX_ARGS: usize = 64;
const MAX_ARG_BYTES: usize = 4096;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;
const CLI_TIMEOUT: Duration = Duration::from_secs(60);
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

fn is_on_path(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(bin).is_file()))
        .unwrap_or(false)
}

fn development_root() -> std::path::PathBuf {
    std::env::var_os("SCIAGENT_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        })
}

fn resolve_command(args: &[String]) -> Command {
    if let Ok(bin) = std::env::var("SCIRUST_BIN")
    {
        let mut cmd = Command::new(bin);
        cmd.args(args);
        cmd.current_dir(development_root());
        return cmd;
    }
    if is_on_path("scirust")
    {
        let mut cmd = Command::new("scirust");
        cmd.args(args);
        cmd.current_dir(development_root());
        return cmd;
    }
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet", "-p", "scirust-cli", "--"]);
    cmd.args(args);
    cmd.current_dir(development_root());
    cmd
}

fn safe_argument(argument: &str) -> bool {
    argument.len() <= MAX_ARG_BYTES
        && !Path::new(argument).is_absolute()
        && !Path::new(argument)
            .components()
            .any(|component| component == Component::ParentDir)
}

fn drain_pipe<R: Read + Send + 'static>(mut pipe: R) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut chunk = [0u8; 8192];
        while let Ok(count) = pipe.read(&mut chunk)
        {
            if count == 0
            {
                break;
            }
            let remaining = MAX_OUTPUT_BYTES.saturating_sub(kept.len());
            kept.extend_from_slice(&chunk[..count.min(remaining)]);
        }
        kept
    })
}

fn run_bounded(mut command: Command) -> Result<(Option<i32>, Vec<u8>, Vec<u8>), String> {
    for variable in SECRET_ENV_VARS
    {
        command.env_remove(variable);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let stdout_thread = drain_pipe(child.stdout.take().expect("stdout was piped"));
    let stderr_thread = drain_pipe(child.stderr.take().expect("stderr was piped"));
    let deadline = Instant::now() + CLI_TIMEOUT;
    let status = loop
    {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())?
        {
            break status;
        }
        if Instant::now() >= deadline
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err("scirust CLI timed out after 60 seconds".to_string());
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_thread
        .join()
        .map_err(|_| "stdout capture thread failed".to_string())?;
    let stderr = stderr_thread
        .join()
        .map_err(|_| "stderr capture thread failed".to_string())?;
    Ok((status.code(), stdout, stderr))
}

pub fn cli_tool() -> McpTool {
    McpTool {
        name: "scirust_cli".to_string(),
        description: "Run any `scirust` CLI subcommand (run `scirust_cli` with args=[\"help\"] \
            to list them) — linsolve, solve, diff, integrate, ode, certify, conformal, evo, \
            analyze, and more. Input: `args`, the argument list without the leading `scirust`. \
            Returns the captured exit code, stdout, and stderr."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "e.g. [\"linsolve\", \"2,1;1,3\", \"3,5\"]",
                }
            },
            "required": ["args"],
        }),
        handler: Box::new(|args| {
            let arg_list: Vec<String> = args
                .get("args")
                .and_then(|v| v.as_array())
                .ok_or("missing `args` array")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| "`args` entries must be strings".to_string())
                })
                .collect::<Result<_, _>>()?;
            if arg_list.len() > MAX_ARGS
            {
                return Err(format!("refused more than {MAX_ARGS} CLI arguments"));
            }
            if let Some(argument) = arg_list.iter().find(|argument| !safe_argument(argument))
            {
                return Err(format!(
                    "refused absolute, parent-relative, or oversized CLI argument: `{argument}`"
                ));
            }
            let (exit_code, stdout, stderr) = run_bounded(resolve_command(&arg_list))
                .map_err(|e| format!("failed to run the scirust CLI: {e}"))?;
            Ok(json!({
                "exit_code": exit_code,
                "stdout": String::from_utf8_lossy(&stdout),
                "stderr": String::from_utf8_lossy(&stderr),
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_args() {
        let tool = cli_tool();
        assert!((tool.handler)(json!({})).is_err());
    }

    #[test]
    fn rejects_non_string_args() {
        let tool = cli_tool();
        assert!((tool.handler)(json!({ "args": [1, 2] })).is_err());
    }

    #[test]
    fn rejects_paths_that_can_escape_development_workspace() {
        let tool = cli_tool();
        assert!((tool.handler)(json!({ "args": ["analyze", "../secret"] })).is_err());
        let absolute = if cfg!(windows)
        {
            "C:\\Windows\\win.ini"
        }
        else
        {
            "/etc/passwd"
        };
        assert!((tool.handler)(json!({ "args": ["analyze", absolute] })).is_err());
    }
}
