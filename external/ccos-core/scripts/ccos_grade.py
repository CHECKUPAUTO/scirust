#!/usr/bin/env python3
"""CCOS Campaign J grader — record the GROUND TRUTH from `cargo test`, not a heuristic.

Thor's first run mis-scored a passing CCOS fix as `resolved=False` because it inferred
resolution from which file was patched. Ground truth is the test suite. This grader runs
`cargo test` in an already-patched crate, sums the real pass/fail counts, and records what
the patch actually changed (via git). Apply the model's diff/code-block however you like,
then call this.

Usage:
    python3 scripts/ccos_grade.py <crate_dir> --bug JM3 --context ccos \
        [--tokens 2351] [--out corpus_J/JM3/ccos_result.json]

`resolved` is true iff `cargo test` ran at least one test and none failed.
"""
import argparse
import json
import os
import re
import subprocess
import sys


def cargo_test(crate_dir):
    p = subprocess.run(["cargo", "test"], cwd=crate_dir,
                       capture_output=True, text=True, timeout=300)
    return p.stdout + "\n" + p.stderr


def parse_results(output):
    """Sum `test result: <ok|FAILED>. N passed; M failed` across all blocks (lib, doc, …)."""
    passed = failed = 0
    seen = False
    for m in re.finditer(r"test result:\s+\w+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed", output):
        passed += int(m.group(1))
        failed += int(m.group(2))
        seen = True
    compile_error = "error[E" in output or "error: could not compile" in output
    return passed, failed, seen, compile_error


def patched_files(crate_dir):
    """Files changed vs git HEAD (best-effort; empty if not a git repo)."""
    try:
        p = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=crate_dir,
                           capture_output=True, text=True, timeout=30)
        return [f for f in p.stdout.split("\n") if f.strip()]
    except (subprocess.SubprocessError, OSError):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crate_dir")
    ap.add_argument("--bug", required=True)
    ap.add_argument("--context", required=True, choices=["ccos", "baseline"])
    ap.add_argument("--tokens", type=int, default=0, help="input tokens of the context used")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not os.path.isdir(args.crate_dir):
        sys.exit("not a directory: %s" % args.crate_dir)
    out = cargo_test(args.crate_dir)
    passed, failed, ran, compile_error = parse_results(out)
    resolved = ran and failed == 0 and not compile_error

    result = {
        "bug": args.bug,
        "context": args.context,
        "resolved": resolved,
        "tests_passed": passed,
        "tests_failed": failed,
        "compile_error": compile_error,
        "patched_files": patched_files(args.crate_dir),
        "tokens": args.tokens,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    # Non-zero exit on an unresolved bug, so a loop can branch on it.
    sys.exit(0 if resolved else 1)


if __name__ == "__main__":
    main()
