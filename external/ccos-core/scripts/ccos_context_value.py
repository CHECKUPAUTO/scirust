#!/usr/bin/env python3
"""CCOS context-assembly value — the BROAD case (model-free).

Campaign J measured *resolution* on bug-fix-with-pretest bugs, which are rare (~1-2% of real
fixes). But the underlying need — "I'm working on file X; are X's cross-file dependencies in
my budget?" — is ubiquitous. This measures exactly that, over EVERY file of a real crate:

  for each file X with >=1 real `use crate::` dep, at a fixed budget B:
    ccos_cov   = fraction of X's real deps whose content is in `recall around X`
    naive_cov  = fraction in a naive dump (X truncated to B, then siblings) — what opening
                 the file gives you

No LLM, no failure signal (you're just working on X): pure recall. Reports the aggregate gap,
split by anchor-file size (small files: the naive dump fits siblings too; big files: it can't,
and only CCOS carries the deps).

Usage: python3 scripts/ccos_context_value.py <crate_src_dir> [--budget 2048]
       [--ccos ./target/release/ccos] [--out report.json]
"""
import argparse
import glob
import json
import os
import re
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


def naive_dump_files(files, anchor, budget):
    """Files whose content survives a naive dump (anchor first, then siblings) truncated to
    `budget`. A file is 'covered' if any of its bytes land before the cap."""
    order = [anchor] + sorted(u for u in files if u != anchor)
    covered, used, cap = set(), 0, budget * 4
    for uri in order:
        if used >= cap:
            break
        covered.add(uri)            # at least its marker/start lands before the cap
        used += len("// %s\n%s\n\n" % (uri, files[uri]))
    return covered


def ccos_recalls(ccos, files, anchors, budget):
    """One session: ingest every file, then `recall around` each anchor (no failure signal —
    you're just working on the file). Returns {anchor: set(files referenced in window)}."""
    ws = tempfile.mkdtemp(prefix="ccos_val_")
    reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}]
    rid = 2
    for uri, src in files.items():
        reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                     "params": {"name": "ingest", "arguments": {"uri": uri, "source": src}}})
        rid += 1
    ids = {}
    for a in anchors:
        reqs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                     "params": {"name": "recall", "arguments": {"strategy": "around",
                                "anchor": "file:" + a, "budget": budget}}})
        ids[a] = rid
        rid += 1
    inp = "\n".join(json.dumps(r) for r in reqs) + "\n"
    p = subprocess.run([ccos, "mcp", ws], input=inp, capture_output=True, text=True, timeout=900)
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
    res = {}
    for a, i in ids.items():
        win = json.loads(out[i]["result"]["content"][0]["text"])
        refs = {file_of(it["uri"]) for it in win["items"]}
        refs.discard(None)
        res[a] = refs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crate_src")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--ccos", default="./target/release/ccos")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not os.path.exists(args.ccos):
        sys.exit("CCOS binary not found at %s — run `cargo build --release`." % args.ccos)
    files = load_flat_src(args.crate_src)
    deps = {u: real_deps(files, u) for u in files}
    anchors = [u for u in files if deps[u]]
    if not anchors:
        sys.exit("no files with flat cross-file deps under %s" % args.crate_src)

    ccos = ccos_recalls(args.ccos, files, anchors, args.budget)

    rows = []
    for a in anchors:
        d = deps[a]
        c = len(d & ccos[a])
        n = len(d & naive_dump_files(files, a, args.budget))
        rows.append((a, toks(files[a]), len(d), c, n))

    tot_d = sum(r[2] for r in rows)
    tot_c = sum(r[3] for r in rows)
    tot_n = sum(r[4] for r in rows)
    big = [r for r in rows if r[1] > args.budget]      # anchor alone exceeds the budget
    small = [r for r in rows if r[1] <= args.budget]

    def cov(rs, idx):
        dd = sum(r[2] for r in rs)
        return (sum(r[idx] for r in rs) / dd) if dd else 0.0

    print("context-assembly value — %s" % args.crate_src)
    print("  %d files, %d with cross-file deps, budget=%d\n" % (len(files), len(anchors), args.budget))
    print("  cross-file dep coverage (deps in window / deps needed):")
    print("    CCOS  : %.0f%%   naive dump: %.0f%%   (over %d dep-edges)" %
          (100 * tot_c / tot_d, 100 * tot_n / tot_d, tot_d))
    print("  split by anchor size:")
    print("    big files (> budget, n=%d): CCOS %.0f%% vs naive %.0f%%   <- the case that matters" %
          (len(big), 100 * cov(big, 3), 100 * cov(big, 4)))
    print("    small files (<= budget, n=%d): CCOS %.0f%% vs naive %.0f%%" %
          (len(small), 100 * cov(small, 3), 100 * cov(small, 4)))
    if args.out:
        json.dump({"crate": args.crate_src, "budget": args.budget,
                   "ccos_cov": tot_c / tot_d, "naive_cov": tot_n / tot_d,
                   "rows": [{"file": r[0], "tok": r[1], "deps": r[2],
                             "ccos": r[3], "naive": r[4]} for r in rows]},
                  open(args.out, "w"), indent=2)
        print("\nwrote %s" % args.out)
    print("\nReading: where the naive dump already carries the deps (small files), CCOS adds "
          "little. The value is on big files — where opening the file truncates its deps away "
          "and only CCOS keeps them. Cross-file refs are everywhere, so this is the broad case "
          "the bug-fix test (rare) under-measured.")


if __name__ == "__main__":
    main()
