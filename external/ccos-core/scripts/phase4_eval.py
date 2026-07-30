#!/usr/bin/env python3
"""Phase 4 — the SUFFICIENT condition (prototype; the real run needs a model).

The necessary-condition metric (`R_cov`, see `validate.py`) showed CCOS *ties* a
lexical RAG at recovering a fix's files. The only way CCOS can still justify
itself is the sufficient condition: **does an LLM fix more real bugs when given
CCOS's causal context than an equal-token-budget of lexical RAG chunks?** This
harness asks exactly that.

Method (SWE-bench-flavoured): for a bug-fix commit, check out the *parent*
(buggy) tree; build the agent's context two ways at an equal token budget —
  * CCOS: the causal region around the target file (`ccos memory` recall);
  * RAG : the top files by TF-IDF cosine to the target;
ask the model to rewrite the buggy file; apply it; run the crate's tests; score
pass/fail. Compare CCOS vs RAG resolved-rate.

STATUS: prototype. `--dry-run` builds and prints both contexts (validated
offline). The model call (Ollama) and `cargo test` grading are wired but only
exercised where a model and a Rust toolchain exist (the Jetson). Simplifications,
stated plainly:
  * targets single-file `src/*.rs` fixes (the common case);
  * grading is "the crate's tests pass after the patch" — not strict
    FAIL_TO_PASS test isolation (a stronger protocol; future work);
  * the model rewrites the whole target file (no diff application).

Usage:
  python scripts/phase4_eval.py --repo /tmp/fd --dry-run --limit 1
  OLLAMA_ENDPOINT=http://localhost:11434 OLLAMA_MODEL=qwen2.5:7b-instruct \
      python scripts/phase4_eval.py --repo /tmp/fd --limit 20 --budget 6000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

KEYWORDS = ["fix", "bug", "crash", "panic", "regression", "issue"]


def git(repo: Path, *args: str, timeout: float = 60.0) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def mine_single_file_fixes(
    repo: Path, subdir: str, limit: int, scan: int, max_target_tokens: int
) -> list[dict]:
    """Fix commits whose change is exactly one `subdir/*.rs` file present at the
    parent and small enough for the model to rewrite (≤ `max_target_tokens`) —
    so the budget leaves room for differentiating context."""
    grep = [a for k in KEYWORDS for a in ("--grep", k)]
    log = git(repo, "log", "--no-merges", "-i", *grep, "--pretty=format:%H%x1f%P%x1f%s", "-n", str(scan))
    out = []
    for line in log.splitlines():
        commit, parents, subject = (line.split("\x1f", 2) + ["", ""])[:3]
        parent = parents.split()[0] if parents.strip() else ""
        if not parent:
            continue
        changed = [
            f
            for f in git(repo, "diff", "--name-only", parent, commit, "--", subdir).splitlines()
            if f.endswith(".rs")
        ]
        if len(changed) != 1:
            continue
        tgt = changed[0]
        try:
            src = git(repo, "show", f"{parent}:{tgt}")
        except RuntimeError:
            continue
        if len(src) // 4 > max_target_tokens:  # too big to rewrite / saturates budget
            continue
        out.append({"commit": commit, "parent": parent, "subject": subject.strip(), "target": tgt})
        if len(out) >= limit:
            break
    return out


def add_worktree(repo: Path, rev: str) -> Path:
    wt = Path(tempfile.mkdtemp(prefix="ccos-p4-"))
    git(repo, "worktree", "add", "--detach", "--force", str(wt), rev, timeout=120.0)
    return wt


def remove_worktree(repo: Path, wt: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)], capture_output=True)
    subprocess.run(["rm", "-rf", str(wt)], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "prune"], capture_output=True)


def read_sources(worktree: Path, subdir: str) -> dict[str, str]:
    base = Path(worktree)
    out: dict[str, str] = {}
    for p in (base / subdir).rglob("*.rs"):
        try:
            out[str(p.relative_to(base))] = p.read_text(errors="ignore")
        except OSError:
            pass
    return out


# --- context builders (equal token budget; ≈4 chars/token) ----------------- #
def _toks(t: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]+", t)


def ccos_context(ccos_bin: str, worktree: Path, subdir: str, target: str, budget: int) -> list[str]:
    """Files in the causal region around the target (via the ccos memory façade),
    target first."""
    srcs = read_sources(worktree, subdir)
    reqs = [{"op": "ingest", "uri": u, "source": s} for u, s in srcs.items()]
    reqs.append({"op": "recall", "strategy": "around", "anchor": f"file:{target}", "budget": budget})
    inp = "\n".join(json.dumps(r) for r in reqs)
    path = tempfile.mktemp(suffix=".ccos")
    try:
        out = subprocess.run([ccos_bin, "memory", "--path", path], input=inp, capture_output=True, text=True, timeout=120)
        lines = [json.loads(x) for x in out.stdout.splitlines() if x.strip()]
        win = lines[-1] if lines else {"items": []}
        files = [i["uri"][len("file:") :] for i in win.get("items", []) if i["uri"].startswith("file:")]
    finally:
        Path(path).exists() and os.remove(path)
    # target first, then the rest in recall order
    ordered = [target] + [f for f in files if f != target]
    return _budget_files(ordered, srcs, budget)


def rag_context(worktree: Path, subdir: str, target: str, budget: int) -> list[str]:
    srcs = read_sources(worktree, subdir)
    docs = {p: Counter(_toks(t)) for p, t in srcs.items()}
    n = max(len(docs), 1)
    df: Counter = Counter()
    for c in docs.values():
        df.update(c.keys())
    idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}

    def vec(c):
        v = {t: (1 + math.log(x)) * idf[t] for t, x in c.items()}
        return v, math.sqrt(sum(w * w for w in v.values())) or 1.0

    vecs = {p: vec(c) for p, c in docs.items()}
    if target not in vecs:
        return _budget_files([target], srcs, budget)
    (qv, qn) = vecs[target]

    def cos(p):
        v, vn = vecs[p]
        s, l = (qv, v) if len(qv) <= len(v) else (v, qv)
        return sum(w * l.get(t, 0.0) for t, w in s.items()) / (qn * vn)

    ranked = sorted((p for p in vecs if p != target), key=lambda p: (-cos(p), p))
    return _budget_files([target] + ranked, srcs, budget)


def _budget_files(ordered: list[str], srcs: dict[str, str], budget: int) -> list[str]:
    out, used = [], 0
    for f in ordered:
        t = len(srcs.get(f, "")) // 4
        if used + t > budget and out:
            break
        used += t
        out.append(f)
    return out


def build_prompt(
    subject: str, target: str, files: list[str], srcs: dict[str, str], error: str | None = None
) -> str:
    body = ["You are fixing a bug in a Rust crate.", f"Bug report: {subject}", ""]
    if error:
        body += [
            "Your previous patch did NOT pass `cargo test`. Output (tail):",
            "```",
            error[-1500:],
            "```",
            "",
        ]
    body.append("Relevant files:")
    for f in files:
        body.append(f"\n// ===== {f} =====\n{srcs.get(f, '')}")
    body.append(
        f"\nThe bug is in `{target}`. Output the COMPLETE corrected contents of "
        f"`{target}` and nothing else (no markdown fences, no prose)."
    )
    return "\n".join(body)


def ask_ollama(prompt: str) -> str | None:
    endpoint = os.environ.get("OLLAMA_ENDPOINT")
    if not endpoint:
        return None
    body = json.dumps(
        {"model": os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct"), "prompt": prompt,
         "stream": False, "options": {"temperature": 0}}
    ).encode()
    req = urllib.request.Request(f"{endpoint}/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["response"]
    except Exception as e:  # noqa: BLE001
        print(f"  (LLM error: {e})", file=sys.stderr)
        return None


def cargo_test_output(worktree: Path, timeout: float) -> tuple[bool, str]:
    """Run the crate's tests; return (passed, combined stdout+stderr). RUST_BACKTRACE
    is on so panics carry their source locations (the page-fault signal)."""
    try:
        p = subprocess.run(
            ["cargo", "test", "--quiet"],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "RUST_BACKTRACE": "1"},
        )
        return p.returncode == 0, (p.stdout + "\n" + p.stderr)
    except subprocess.TimeoutExpired:
        return False, "cargo test timed out"


def ccos_trace_files(ccos_bin: str, output: str) -> list[str]:
    """The faulting source files of a cargo/panic output, via `ccos trace`."""
    try:
        out = subprocess.run(
            [ccos_bin, "trace"], input=output, capture_output=True, text=True, timeout=30
        ).stdout
        return json.loads(out).get("files", [])
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return []


def ccos_page_fault_context(
    ccos_bin: str, worktree: Path, subdir: str, fault_files: list[str], budget: int
) -> list[str]:
    """A *context page fault*: inject failure pressure on the faulting files and
    recall a refreshed window around them (via `ccos memory`)."""
    srcs = read_sources(worktree, subdir)
    reqs = [{"op": "ingest", "uri": u, "source": s} for u, s in srcs.items()]
    for ff in fault_files:
        reqs.append({"op": "failure", "node": f"file:{ff}", "depth": 2})
    if fault_files:
        reqs.append({"op": "recall", "strategy": "around",
                     "anchor": f"file:{fault_files[0]}", "budget": budget})
    else:
        reqs.append({"op": "recall", "strategy": "working_set", "budget": budget})
    inp = "\n".join(json.dumps(r) for r in reqs)
    path = tempfile.mktemp(suffix=".ccos")
    try:
        out = subprocess.run([ccos_bin, "memory", "--path", path],
                             input=inp, capture_output=True, text=True, timeout=120).stdout
        win = next((json.loads(x) for x in reversed(out.splitlines()) if '"items"' in x), {"items": []})
        files = [i["uri"][len("file:"):] for i in win.get("items", []) if i["uri"].startswith("file:")]
    except (subprocess.SubprocessError, json.JSONDecodeError):
        files = list(fault_files)
    finally:
        Path(path).exists() and os.remove(path)
    return _budget_files(files, srcs, budget)


def attempt(repo, sc, strat, ccos_bin, subdir, budget, dry_run, test_timeout, max_attempts) -> str:
    """Return 'pass' / 'fail' / 'skip' for one (scenario, strategy), with a
    compiler-in-the-loop retry: on a failing `cargo test`, a context page fault
    enriches the window from the error before the next attempt."""
    wt = add_worktree(repo, sc["parent"])
    target = sc["target"]
    try:
        srcs = read_sources(wt, subdir)
        files = (
            ccos_context(ccos_bin, wt, subdir, target, budget)
            if strat == "ccos"
            else rag_context(wt, subdir, target, budget)
        )
        ctx_tokens = sum(len(srcs.get(f, "")) for f in files) // 4
        if dry_run:
            print(f"    [{strat}] {len(files)} files, ~{ctx_tokens} tok: {files[:6]}")
            return "skip", ctx_tokens

        error = None
        for att in range(max_attempts):
            prompt = build_prompt(sc["subject"], target, files, srcs, error)
            reply = ask_ollama(prompt)
            if not reply:
                return "skip", ctx_tokens
            (Path(wt) / target).write_text(reply)
            ok, output = cargo_test_output(wt, test_timeout)
            if ok:
                if att:
                    print(f"    [{strat}] passed on attempt {att + 1}", file=sys.stderr)
                return "pass", ctx_tokens
            # Context page fault: enrich the window from the failure for the retry.
            error = output
            fault = ccos_trace_files(ccos_bin, output)
            if strat == "ccos":
                files = (
                    ccos_page_fault_context(ccos_bin, wt, subdir, fault, budget)
                    if fault
                    else files
                )
            elif fault:
                files = rag_context(wt, subdir, fault[0], budget)
            srcs = read_sources(wt, subdir)
        return "fail", ctx_tokens
    finally:
        remove_worktree(repo, wt)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Phase 4 — sufficient-condition prototype (CCOS vs RAG).")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--subdir", default="src")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--budget", type=int, default=8000, help="context token budget")
    ap.add_argument("--max-target-tokens", type=int, default=1200, help="skip larger target files")
    ap.add_argument("--ccos-bin", default=str(Path(__file__).resolve().parents[1] / "target/release/ccos"))
    ap.add_argument("--test-timeout", type=float, default=900.0)
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="compiler-in-the-loop retries (page fault between attempts)")
    ap.add_argument("--dry-run", action="store_true", help="build + print contexts, no model, no tests")
    args = ap.parse_args(argv)

    scen = mine_single_file_fixes(
        args.repo, args.subdir, args.limit, scan=max(args.limit * 30, 400),
        max_target_tokens=args.max_target_tokens,
    )
    print(f"[mine] {len(scen)} single-file fix scenario(s)", file=sys.stderr)
    tally = {"ccos": Counter(), "rag": Counter()}
    toks: dict[str, list[int]] = {"ccos": [], "rag": []}
    for i, sc in enumerate(scen, 1):
        print(f"[{i}/{len(scen)}] {sc['commit'][:8]} {sc['target']}: {sc['subject'][:60]}", file=sys.stderr)
        for strat in ("ccos", "rag"):
            try:
                status, ct = attempt(args.repo, sc, strat, args.ccos_bin, args.subdir,
                                     args.budget, args.dry_run, args.test_timeout, args.max_attempts)
                tally[strat][status] += 1
                toks[strat].append(ct)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"    [{strat}] skip: {e}", file=sys.stderr)
                tally[strat]["skip"] += 1

    if not args.dry_run:
        print("\n=== Phase 4 — resolved-rate + context efficiency (equal resolution → fewer tokens?) ===")
        for strat in ("ccos", "rag"):
            t = tally[strat]
            graded = t["pass"] + t["fail"]
            rate = t["pass"] / graded if graded else 0.0
            mean_tok = sum(toks[strat]) / len(toks[strat]) if toks[strat] else 0.0
            print(f"  {strat:5}: {t['pass']}/{graded} pass ({rate:.0%})   "
                  f"mean context {mean_tok:.0f} tok   skipped {t['skip']}")

    # Efficiency report — works in --dry-run too (tokens need no model).
    print("\n=== Context efficiency (mean tokens of the assembled window) ===")
    means = {}
    for strat in ("ccos", "rag"):
        ts = toks[strat]
        means[strat] = sum(ts) / len(ts) if ts else 0.0
        print(f"  {strat:5}: mean {means[strat]:.0f} tok over {len(ts)} scenarios")
    if means["ccos"] and means["rag"]:
        print(f"  → CCOS assembles {means['rag'] / means['ccos']:.1f}x fewer context tokens "
              f"than the budget-filling RAG (it stops at the causal region).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
