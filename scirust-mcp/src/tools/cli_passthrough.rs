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
use std::process::Command;

fn is_on_path(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(bin).is_file()))
        .unwrap_or(false)
}

fn resolve_command(args: &[String]) -> Command {
    if let Ok(bin) = std::env::var("SCIRUST_BIN")
    {
        let mut cmd = Command::new(bin);
        cmd.args(args);
        return cmd;
    }
    if is_on_path("scirust")
    {
        let mut cmd = Command::new("scirust");
        cmd.args(args);
        return cmd;
    }
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet", "-p", "scirust-cli", "--"]);
    cmd.args(args);
    cmd
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
            let output = resolve_command(&arg_list)
                .output()
                .map_err(|e| format!("failed to run the scirust CLI: {e}"))?;
            Ok(json!({
                "exit_code": output.status.code(),
                "stdout": String::from_utf8_lossy(&output.stdout),
                "stderr": String::from_utf8_lossy(&output.stderr),
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
}
