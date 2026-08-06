#!/usr/bin/env python3
"""CCOS field probe — structural Campaign H/I on ANY Rust crate (model-free).

Measures the triad + budget-balancing on a real crate, independent of CCOS:
  #1 granularity  : full-region token blow-up vs unique all-src  (want ~1x, not 15x)
  #2 flood        : nodes pressured by signal_failure on one file (want a small fraction)
  #3 coverage     : of an anchor's real `use crate::` deps, how many land in the window,
                    at what token cost (% of all-src), with how much distant noise.

Usage:
    python3 scripts/ccos_campaign_probe.py <crate_src_dir> [--budget 2048] [--depth 3]
        [--anchors a.rs,b.rs] [--ccos ./target/release/ccos] [--out corpus_I/<crate>.json]

Notes:
  * Ingests only the FLAT `src/*.rs` files — CCOS's cross-file linking is strongest on a
    flat layout (sub-module `mod.rs` resolution is a separate, known limitation). Point it
    at a crate whose modules are mostly `src/<name>.rs`.
  * One fresh CCOS session per anchor (re-ingest), so failure pressure never bleeds between
    measurements. Deterministic.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def toks(s):
    return max(1, len(s) // 4)


def file_of(uri):
    m = re.search(r"src/[A-Za-z0-9_]+\.rs", uri)
    return m.group(0) if m else None


def load_flat_src(crate_src):
    files = {}
    for p in sorted(glob.glob(os.path.join(crate_src, "*.rs"))):
        with open(p, encoding="utf-8", errors="replace") as f:
            files["src/" + os.path.basename(p)] = f.read()
    return files


def real_deps(files, uri):
    """Flat files X for which the anchor has `use crate::X...` — single, multi-segment
    (`crate::X::Y` → X), and single-line grouped (`crate::{X, Y::z}` → X, Y) imports."""
    deps = set()

    def add(seg):
        cand = "src/%s.rs" % seg.strip()
        if cand in files and cand != uri:
            deps.add(cand)

    for m in re.finditer(r"use\s+crate::([^;]+);", files[uri]):
        body = m.group(1).strip()
        if body.startswith("{"):
            for part in body[1:].rstrip("}").split(","):
                root = part.strip().split("::")[0].strip()
                if root and root not in ("self", "*"):
                    add(root)
        else:
            add(body.split("::")[0])
    return deps


def drive(ccos, files, calls, workdir):
    """Run one CCOS MCP session: ingest every file, then `calls` (list of tool params).
    Returns {tag: parsed_json_result}."""
    ws = tempfile.mkdtemp(dir=workdir)
    reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}]
    rid = 2
    for uri, src in files.items():
        reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                     "params": {"name": "ingest", "arguments": {"uri": uri, "source": src}}})
        rid += 1
    ids = {}
    for tag, params in calls:
        reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": params})
        ids[tag] = rid
        rid += 1
    inp = "\n".join(json.dumps(r) for r in reqs) + "\n"
    p = subprocess.run([ccos, "mcp", ws], input=inp, capture_output=True, text=True, timeout=600)
    out = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(m.get("id"), int):
            out[m["id"]] = m
    shutil.rmtree(ws, ignore_errors=True)
    res = {}
    for tag, i in ids.items():
        if i in out and "result" in out[i]:
            res[tag] = json.loads(out[i]["result"]["content"][0]["text"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crate_src", help="path to the crate's src/ directory")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--anchors", default="", help="comma-separated file names, e.g. expr.rs,ty.rs")
    ap.add_argument("--ccos", default="./target/release/ccos")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not os.path.exists(args.ccos):
        sys.exit("CCOS binary not found at %s — run `cargo build --release` first." % args.ccos)
    files = load_flat_src(args.crate_src)
    if not files:
        sys.exit("no flat src/*.rs found under %s" % args.crate_src)
    all_t = sum(toks(s) for s in files.values())
    workdir = tempfile.mkdtemp(prefix="ccos_probe_")

    if args.anchors:
        anchors = ["src/" + a.strip() for a in args.anchors.split(",")]
    else:
        cand = [(len(real_deps(files, u)), u) for u in files]
        anchors = [u for n, u in sorted(cand, reverse=True) if n >= 2][:5]
    if not anchors:
        sys.exit("no anchor files with >=2 flat deps; pass --anchors explicitly")

    report = {"crate_src": args.crate_src, "files": len(files), "all_src_tokens": all_t,
              "budget": args.budget, "depth": args.depth, "anchors": {}}

    # #1 duplication — full region around the first anchor at an effectively infinite budget.
    dup = drive(args.ccos, files,
                [("r", {"name": "recall", "arguments": {"strategy": "around",
                  "anchor": "file:" + anchors[0], "budget": 50_000_000}})], workdir)
    region_tok = dup["r"]["tokens"] if "r" in dup else 0
    report["duplication_factor"] = round(region_tok / all_t, 3) if all_t else 0

    print("CCOS field probe — %s" % args.crate_src)
    print("  %d flat files, all-src = %d tokens" % (len(files), all_t))
    print("  [#1] full-region blow-up = %.2fx  (want ~1x, NOT 15x)\n" % report["duplication_factor"])
    print("  %-16s %-6s %-9s %-12s %-11s %-9s %s" %
          ("anchor", "deps", "affected", "deps_in_win", "window_tok", "%all-src", "noise"))

    for a in anchors:
        rd = real_deps(files, a)
        res = drive(args.ccos, files, [
            ("sf", {"name": "signal_failure", "arguments": {"node": "file:" + a, "depth": args.depth}}),
            ("rc", {"name": "recall", "arguments": {"strategy": "around",
             "anchor": "file:" + a, "budget": args.budget}}),
        ], workdir)
        aff = res.get("sf", {}).get("affected", -1)
        w = res.get("rc", {"items": [], "tokens": 0})
        refs = {file_of(it["uri"]) for it in w["items"]}
        refs.discard(None)
        cov = sorted(d for d in rd if d in refs)
        noise = sorted(f for f in refs if f not in rd and f != a)
        pct = 100.0 * w["tokens"] / all_t if all_t else 0
        report["anchors"][a] = {"deps": sorted(rd), "deps_in_window": cov,
                                "affected": aff, "window_tokens": w["tokens"],
                                "pct_all_src": round(pct, 2), "noise_files": noise}
        print("  %-16s %-6d %-9s %d/%-10d %-11d %-9.1f %d" %
              (a.replace("src/", ""), len(rd), aff, len(cov), len(rd),
               w["tokens"], pct, len(noise)))

    shutil.rmtree(workdir, ignore_errors=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote %s" % args.out)
    print("\nRead: deps_in_win should be ~all of them; window_tok a few %% of all-src; "
          "noise ~0. If deps_in_win is low, raise --budget (big files need more) — that is "
          "the budget-scaling signal, see docs/DESIGN_recall_budget.md.")


if __name__ == "__main__":
    main()
