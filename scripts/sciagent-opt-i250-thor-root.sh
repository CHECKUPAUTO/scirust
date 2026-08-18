#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "I250 production optimization must run from the Thor root shell so the production SCIAGENT checkpoint is visible." >&2
  exit 77
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$repo_root" ]; then
  echo "run this script from a dedicated SciRust Git worktree" >&2
  exit 78
fi
cd "$repo_root"

if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "the optimization worktree must be clean before the campaign starts" >&2
  git status --short >&2
  exit 79
fi

: "${SCIAGENT_CKPT:?set SCIAGENT_CKPT to the trained SCIAGENT checkpoint visible from the root shell}"
if [ ! -d "$SCIAGENT_CKPT" ]; then
  echo "SCIAGENT_CKPT is not an accessible directory: $SCIAGENT_CKPT" >&2
  exit 80
fi

lock="${SCIRUST_THOR_GPU_LOCK:-/dev/nvidia0}"
if [ ! -c "$lock" ] || [ ! -r "$lock" ] || [ ! -w "$lock" ]; then
  echo "Thor GPU lock target must be a readable/writable character device: $lock" >&2
  exit 81
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required" >&2
  exit 82
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the physical-Thor idle check" >&2
  exit 83
fi

# Use the same physical device inode as the existing SciRust production trainer and
# hardware qualification workflows. Never kill or preempt an existing owner.
exec 9>"$lock"
if ! flock -n -x 9; then
  echo "Thor GPU lock is busy; leaving the active training/qualification owner untouched." >&2
  exit 75
fi

if ! compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)"; then
  echo "cannot prove that the Thor GPU is idle" >&2
  exit 84
fi
if [ -n "${compute_apps//[[:space:]]/}" ]; then
  echo "Thor GPU has active compute processes despite the acquired lock; refusing concurrent optimization:" >&2
  printf '%s\n' "$compute_apps" >&2
  exit 85
fi

manifest="${SCIAGENT_OPT_MANIFEST:-scirust-sciagent/examples/optimization_tasks/i250_cuda_decode.json}"
run_root="${SCIAGENT_OPT_RUN_ROOT:-.sciagent-opt}"
if [ ! -f "$manifest" ]; then
  echo "optimization manifest not found: $manifest" >&2
  exit 86
fi

export SCIRUST_THOR_GPU_LOCK="$lock"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

cargo +nightly-2026-07-02 run \
  -p scirust-sciagent --bin sciagent-optimize -- \
  run \
  --manifest "$manifest" \
  --workspace "$repo_root" \
  --run-root "$run_root" \
  --json | tee "$repo_root/sciagent-i250-optimization-report.json"

report="$repo_root/sciagent-i250-optimization-report.json"
python3 - "$report" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
print(f"final_decision={report['final_decision']}")
print(f"baseline_median_ns={report['baseline']['median_ns']}")
print(f"best_verified_speedup={report.get('best_verified_speedup')}")
print(f"iterations={len(report.get('iterations', []))}")
print(f"failures={len(report.get('failures', []))}")
if report['final_decision'] != 'promote':
    raise SystemExit(2)
PY

echo "Promoted candidate remains as a worktree diff for review; no commit or push was performed automatically."
git status --short
git diff --stat
