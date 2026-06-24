//! # `scirust-rsi-mcp` — an MCP server that evolves algorithms
//!
//! A minimal [Model Context Protocol](https://modelcontextprotocol.io) server
//! (JSON-RPC over stdio) that exposes **one tool**, `evolve_algorithm`, backed
//! by [`scirust_rsi::progevo`]. An LLM client (Claude Desktop, Claude Code, or
//! any MCP-capable app) connects to it; the user then just asks, in plain
//! language, to evolve a program that fits some input→output examples — the
//! model translates that into a tool call and relays the result.
//!
//! Build & run:
//! ```text
//! cargo build -p scirust-rsi --bin scirust-rsi-mcp --features mcp --release
//! ./target/release/scirust-rsi-mcp        # speaks MCP on stdin/stdout
//! ```
//!
//! The evolution itself runs entirely locally and offline — no model and no API
//! key are needed inside the tool; scirust only proposes and *selects* programs
//! under its bounded, elitist, reproducible guarantees.

use scirust_rsi::{Guard, progevo};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

const PROTOCOL_VERSION: &str = "2024-11-05";

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines()
    {
        let line = match line
        {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty()
        {
            continue;
        }
        let req: Value = match serde_json::from_str(&line)
        {
            Ok(v) => v,
            Err(_) => continue, // ignore malformed frames
        };

        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");

        let response = match method
        {
            "initialize" => Some(ok(
                &id,
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "serverInfo": {
                        "name": "scirust-rsi",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                }),
            )),
            "tools/list" => Some(ok(&id, json!({ "tools": [tool_schema()] }))),
            "tools/call" => Some(handle_call(&id, req.get("params"))),
            "ping" => Some(ok(&id, json!({}))),
            // Notifications (no id) get no response; unknown calls get an error.
            _ if id.is_some() => Some(error(&id, -32601, &format!("method not found: {method}"))),
            _ => None,
        };

        if let Some(resp) = response
        {
            if let Ok(s) = serde_json::to_string(&resp)
            {
                let _ = writeln!(stdout, "{s}");
                let _ = stdout.flush();
            }
        }
    }
}

/// JSON-RPC success envelope.
fn ok(id: &Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// JSON-RPC error envelope.
fn error(id: &Option<Value>, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// The declaration of the single `evolve_algorithm` tool.
fn tool_schema() -> Value {
    json!({
        "name": "evolve_algorithm",
        "description":
            "Evolve a small arithmetic program over one input `x` (reverse-Polish \
             tokens: x, numbers, + - * /) so it reproduces the given input→output \
             examples. Uses scirust's bounded, elitist, reproducible evolutionary \
             search. Optionally start from the user's own program (`seed_program`); \
             the result never scores worse than that seed. Returns the evolved \
             program, its mean-squared error, and an audit trail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "description": "Input→output pairs the program must fit, e.g. [[1,2],[2,4],[3,6]].",
                    "items": {
                        "type": "array",
                        "items": { "type": "number" },
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "minItems": 1
                },
                "seed_program": {
                    "type": "string",
                    "description": "Optional starting program in RPN (default \"x\")."
                },
                "max_iters": {
                    "type": "integer",
                    "description": "Iteration cap (default 1500).",
                    "minimum": 1
                },
                "samples": {
                    "type": "integer",
                    "description": "Candidates proposed per round, best-of-n (default 32).",
                    "minimum": 1
                },
                "seed": {
                    "type": "integer",
                    "description": "RNG seed for reproducibility (default 0)."
                }
            },
            "required": ["examples"]
        }
    })
}

/// Run the `evolve_algorithm` tool and wrap the result as MCP tool content.
fn handle_call(id: &Option<Value>, params: Option<&Value>) -> Value {
    let args = params.and_then(|p| p.get("arguments"));
    let name = params
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if name != "evolve_algorithm"
    {
        return tool_error(id, &format!("unknown tool: {name}"));
    }

    let examples = match parse_examples(args.and_then(|a| a.get("examples")))
    {
        Ok(e) if !e.is_empty() => e,
        Ok(_) =>
        {
            return tool_error(
                id,
                "`examples` must contain at least one [input, output] pair",
            );
        },
        Err(e) => return tool_error(id, &e),
    };

    let seed_program = args
        .and_then(|a| a.get("seed_program"))
        .and_then(Value::as_str)
        .unwrap_or("x")
        .to_string();
    let max_iters = args
        .and_then(|a| a.get("max_iters"))
        .and_then(Value::as_u64)
        .unwrap_or(1500) as usize;
    let samples = args
        .and_then(|a| a.get("samples"))
        .and_then(Value::as_u64)
        .unwrap_or(32) as usize;
    let seed = args
        .and_then(|a| a.get("seed"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    // `target` rarely fires (the Occam length penalty keeps fitness just below
    // 0), so `patience` is what stops the run cleanly once it stops improving.
    let guard = Guard::new()
        .max_iters(max_iters)
        .patience((max_iters / 5).max(100))
        .target(-1e-9);
    let out = progevo::evolve(&examples, &seed_program, samples, seed, &guard);

    let mut text = String::new();
    text.push_str(&format!(
        "Evolved program (reverse-Polish): {}\n",
        out.program
    ));
    text.push_str(&format!(
        "Mean-squared error: {:.6} (0 = perfect fit)\n",
        out.mse
    ));
    text.push_str(&format!(
        "Iterations: {}   improvements adopted: {}   stop: {:?}\n",
        out.report.iterations, out.report.accepted, out.report.stop_reason
    ));
    text.push_str(&format!(
        "Non-regressing (monotone): {}   started from: \"{}\"\n\n",
        out.report.is_monotone(),
        seed_program
    ));
    text.push_str("input -> predicted (expected)\n");
    for &(x, y) in examples.iter().take(16)
    {
        let p = progevo::eval_rpn(&out.program, x)
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "invalid".into());
        text.push_str(&format!("  {x} -> {p}  (expected {y})\n"));
    }
    if examples.len() > 16
    {
        text.push_str(&format!("  … and {} more\n", examples.len() - 16));
    }

    ok(
        id,
        json!({ "content": [{ "type": "text", "text": text }], "isError": false }),
    )
}

/// MCP tool-level error (returned as content with `isError: true`, per spec, so
/// the model sees the message rather than a transport failure).
fn tool_error(id: &Option<Value>, message: &str) -> Value {
    ok(
        id,
        json!({ "content": [{ "type": "text", "text": format!("Error: {message}") }], "isError": true }),
    )
}

/// Parse `examples` from either `[[in,out],…]` or `[{input,output},…]`.
fn parse_examples(v: Option<&Value>) -> Result<Vec<(f64, f64)>, String> {
    let arr = v
        .and_then(Value::as_array)
        .ok_or_else(|| "`examples` must be an array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate()
    {
        let pair = if let Some(pair) = item.as_array()
        {
            let a = pair.first().and_then(Value::as_f64);
            let b = pair.get(1).and_then(Value::as_f64);
            match (a, b)
            {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(format!("example {i} must be [input, output] numbers")),
            }
        }
        else if let Some(obj) = item.as_object()
        {
            let a = obj.get("input").and_then(Value::as_f64);
            let b = obj.get("output").and_then(Value::as_f64);
            match (a, b)
            {
                (Some(a), Some(b)) => (a, b),
                _ =>
                {
                    return Err(format!(
                        "example {i} must have numeric `input` and `output`"
                    ));
                },
            }
        }
        else
        {
            return Err(format!("example {i} must be an array or object"));
        };
        out.push(pair);
    }
    Ok(out)
}
