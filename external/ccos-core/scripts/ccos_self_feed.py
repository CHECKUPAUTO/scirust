#!/usr/bin/env python3
"""CCOS self-feed hook — the "hardware intercept" for an agent's causal memory.

A Claude Code **PostToolUse** hook: after every tool the agent runs, it
transparently feeds the *side effect* into CCOS, so the agent's working memory —
and its replayable ``.oplog`` — build with zero cognitive overhead. The agent
never has to *think* about calling ``ingest``/``page_fault``; like a hardware MMU,
the feeding is invisible and automatic.

  - Read / Edit / Write of a source file        ->  ccos ``ingest``
  - Bash ``cargo test|build|check|clippy`` that errors  ->  ccos ``page_fault``

Each event opens a short-lived ``ccos mcp <workspace>`` session that loads the
workspace, applies one operation, checkpoints (snapshot + ``.oplog``), and exits —
so the timeline is post-mortem-able with ``ccos postmortem <workspace>``.

Wire it up in your Claude Code settings (see ``docs/SELF_ANALYSIS.md``):

    "hooks": {
      "PostToolUse": [
        { "matcher": "Read|Edit|Write|Bash",
          "hooks": [{ "type": "command",
                      "command": "python3 scripts/ccos_self_feed.py" }] }
      ]
    }

Env: ``CCOS_BIN`` (default ``<cwd>/target/release/ccos``),
``CCOS_WORKSPACE`` (default ``<cwd>/workspace.ccos``). Always exits 0 — a memory
hook must never block or fail the agent.
"""
import json
import os
import subprocess
import sys

SOURCE_EXTS = (
    ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go",
    ".java", ".c", ".h", ".cpp", ".hpp", ".rb", ".swift",
)
# Markers that a build/test command actually failed (page-fault-worthy).
ERROR_MARKERS = ("error[", "error:", "panicked at", "test result: FAILED", "FAILED")


def mcp_call(ccos_bin, workspace, name, arguments):
    """Fire a one-shot MCP session: open the workspace, apply one op, checkpoint."""
    msgs = [
        {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        },
        {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    ]
    payload = "".join(json.dumps(m) + "\n" for m in msgs)
    try:
        subprocess.run(
            [ccos_bin, "mcp", workspace],
            input=payload, text=True, capture_output=True, timeout=20,
        )
    except Exception as exc:  # never propagate — this is a best-effort hook
        print(f"ccos-self-feed: {exc}", file=sys.stderr)


def output_text(tool_response):
    """Pull real (newline-bearing) text out of a Bash tool_response."""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        parts = [
            tool_response[k]
            for k in ("stdout", "stderr", "output", "content", "result")
            if isinstance(tool_response.get(k), str)
        ]
        return "\n".join(parts) if parts else json.dumps(tool_response)
    return str(tool_response)


def main():
    try:
        event = json.load(sys.stdin)
    except Exception:
        return

    cwd = event.get("cwd") or os.getcwd()
    ccos_bin = os.environ.get("CCOS_BIN", os.path.join(cwd, "target/release/ccos"))
    workspace = os.environ.get("CCOS_WORKSPACE", os.path.join(cwd, "workspace.ccos"))
    if not os.path.exists(ccos_bin):
        return  # not built yet (cargo build --release) — silently do nothing

    tool = event.get("tool_name", "")
    tin = event.get("tool_input", {}) or {}
    tout = event.get("tool_response", {})

    # A source file was read or written: fold it into the causal graph.
    if tool in ("Read", "Edit", "Write", "NotebookEdit"):
        path = tin.get("file_path") or tin.get("notebook_path") or ""
        if path.endswith(SOURCE_EXTS) and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except Exception:
                return
            uri = os.path.relpath(path, cwd)
            mcp_call(ccos_bin, workspace, "ingest", {"uri": uri, "source": source})

    # A build/test command failed: feed the trace back as a context page fault.
    elif tool == "Bash":
        cmd = tin.get("command") or ""
        if any(k in cmd for k in ("cargo test", "cargo build", "cargo check", "cargo clippy")):
            text = output_text(tout)
            if any(m in text for m in ERROR_MARKERS):
                mcp_call(ccos_bin, workspace, "page_fault", {"output": text, "budget": 1024})


if __name__ == "__main__":
    main()
    sys.exit(0)
