#!/usr/bin/env python3
"""CCOS Campaign J — build two EQUAL-budget contexts for the sufficiency test (Q7).

The decisive question the paper's Phase 4 asks: at the *same* token budget, does the
CCOS causal window help a local LLM *resolve* a bug better/cheaper than a naive dump?

This helper builds the two contexts so the comparison is fair and reproducible; the
model call + grading is yours (Thor's local LLM + `cargo test`).

  context_ccos.txt     : ingest the crate, pressure the failing file (signal_failure, or
                         page_fault if you pass the cargo output), recall AROUND it at the
                         budget, linearised — the causal region.
  context_baseline.txt : the failing file's source + its directory siblings, concatenated
                         up to the SAME budget — the "no graph, just dump files" baseline.

Usage:
    python3 scripts/ccos_resolution_context.py <crate_src_dir> <failing_file.rs> \
        [--budget 4096] [--cargo-output red.txt] [--ccos ./target/release/ccos] \
        [--outdir corpus_J/<bug>]

Then, for EACH context, with the SAME prompt:
    "Here is the context. The test in <failing_file> fails. Return a unified diff that fixes it."
    -> apply the diff -> `cargo test` -> record resolved? and the context's token count.
Bring back: bug id, resolved_ccos, resolved_baseline, tokens_ccos, tokens_baseline.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile


def toks(s):
    return max(1, len(s) // 4)


def load_flat_src(crate_src):
    files = {}
    for p in sorted(glob.glob(os.path.join(crate_src, "*.rs"))):
        with open(p, encoding="utf-8", errors="replace") as f:
            files["src/" + os.path.basename(p)] = f.read()
    return files


def ccos_context(ccos, files, failing_uri, budget, cargo_output, plain=False):
    """Ingest, pressure the failing file, recall around it; linearise the window."""
    ws = tempfile.mkdtemp(prefix="ccos_J_")
    reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}]
    rid = 2
    for uri, src in files.items():
        reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                     "params": {"name": "ingest", "arguments": {"uri": uri, "source": src}}})
        rid += 1
    if cargo_output:
        reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                     "params": {"name": "page_fault",
                                "arguments": {"output": cargo_output, "budget": budget}}})
    else:
        reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                     "params": {"name": "signal_failure",
                                "arguments": {"node": "file:" + failing_uri, "depth": 3}}})
    rid += 1
    reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                 "params": {"name": "recall", "arguments": {"strategy": "around",
                            "anchor": "file:" + failing_uri, "budget": budget}}})
    rec = rid
    inp = "\n".join(json.dumps(r) for r in reqs) + "\n"
    p = subprocess.run([ccos, "mcp", ws], input=inp, capture_output=True, text=True, timeout=600)
    win = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("id") == rec:
            win = json.loads(m["result"]["content"][0]["text"])
    if plain:
        # Plain mode: emit the window as ordinary multi-file source (`// path` + code),
        # no `[kind score]` annotations — a weak model (≤~3B) reads `// sym:` headers as
        # code and miscompiles (Campaign J2 finding). Group nothing; one marker per item.
        out = []
        for it in win["items"]:
            path = it["uri"].split(":", 2)[1] if ":" in it["uri"] else it["uri"]
            out.append("// %s\n%s" % (path, it["content"]))
        return "\n\n".join(out), win["tokens"]
    out = ["// CCOS causal context around %s (~%d tokens, %d items)"
           % (failing_uri, win["tokens"], len(win["items"]))]
    for it in win["items"]:
        out.append("\n// %s  [%s score=%.3f]\n%s" % (it["uri"], it["kind"], it["score"], it["content"]))
    return "\n".join(out), win["tokens"]


def baseline_context(crate_src, files, failing_uri, budget):
    """The failing file first, then its src/ siblings, concatenated and HARD-capped to the
    same token budget (a naive agent dumps files until the budget, truncating the last)."""
    order = [failing_uri] + sorted(u for u in files if u != failing_uri)
    blob = "\n\n".join("// %s\n%s" % (uri, files[uri]) for uri in order)
    cap = budget * 4  # chars; toks() is len/4
    if len(blob) > cap:
        blob = blob[:cap]
    return blob, toks(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crate_src")
    ap.add_argument("failing_file", help="e.g. writer.rs (a file under crate_src)")
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--cargo-output", default="", help="path to captured red `cargo test` output")
    ap.add_argument("--ccos", default="./target/release/ccos")
    ap.add_argument("--outdir", default="corpus_J/bug")
    ap.add_argument("--plain", action="store_true",
                    help="emit the CCOS context as plain `// path` + code (no [kind score] "
                         "annotations) — for weak models that misread the annotations")
    args = ap.parse_args()

    if not os.path.exists(args.ccos):
        sys.exit("CCOS binary not found at %s — run `cargo build --release`." % args.ccos)
    files = load_flat_src(args.crate_src)
    failing_uri = "src/" + os.path.basename(args.failing_file)
    if failing_uri not in files:
        sys.exit("%s not found among flat src files" % failing_uri)
    cargo = ""
    if args.cargo_output:
        with open(args.cargo_output, encoding="utf-8", errors="replace") as f:
            cargo = f.read()

    ccos_txt, ccos_tok = ccos_context(args.ccos, files, failing_uri, args.budget, cargo, args.plain)
    base_txt, base_tok = baseline_context(args.crate_src, files, failing_uri, args.budget)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "context_ccos.txt"), "w") as f:
        f.write(ccos_txt)
    with open(os.path.join(args.outdir, "context_baseline.txt"), "w") as f:
        f.write(base_txt)

    print("failing file : %s" % failing_uri)
    print("budget       : %d tokens" % args.budget)
    print("context_ccos     : %d tokens  -> %s/context_ccos.txt" % (ccos_tok, args.outdir))
    print("context_baseline : %d tokens  -> %s/context_baseline.txt" % (base_tok, args.outdir))
    print("\nNext (yours): feed EACH to your local LLM with the SAME fix prompt, apply the")
    print("diff, run `cargo test`, and record resolved? + the token count for each.")


if __name__ == "__main__":
    main()
