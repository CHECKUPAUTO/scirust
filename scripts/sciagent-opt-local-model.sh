#!/usr/bin/env bash
set -euo pipefail

stage="${1:-generate}"
: "${SCIAGENT_CKPT:?SCIAGENT_CKPT must point to a trained SCIAGENT checkpoint}"
: "${SCIAGENT_OPT_CONTEXT:?}"
: "${SCIAGENT_OPT_SKILL_PATH:?}"
: "${SCIAGENT_OPT_RUN_DIR:?}"
: "${SCIAGENT_OPT_ITERATION:?}"

prompt_file="${SCIAGENT_OPT_RUN_DIR}/agent-${SCIAGENT_OPT_ITERATION}-${stage}-prompt.txt"
raw_file="${SCIAGENT_OPT_RUN_DIR}/agent-${SCIAGENT_OPT_ITERATION}-${stage}-raw.txt"
patch_file="${SCIAGENT_OPT_RUN_DIR}/agent-${SCIAGENT_OPT_ITERATION}-${stage}.patch"
source_limit="${SCIAGENT_OPT_SOURCE_BYTES:-48000}"

python3 - "$SCIAGENT_OPT_CONTEXT" "$SCIAGENT_OPT_SKILL_PATH" "$prompt_file" "$source_limit" "$SCIAGENT_OPT_RUN_DIR" <<'PY'
import json
import pathlib
import sys

context_path = pathlib.Path(sys.argv[1])
skill_path = pathlib.Path(sys.argv[2])
prompt_path = pathlib.Path(sys.argv[3])
source_limit = max(1024, int(sys.argv[4]))
run_dir = pathlib.Path(sys.argv[5])
evidence_limit = min(source_limit, 16000)
ctx = json.loads(context_path.read_text())

parts = []
parts.append("You are running one SCIAGENT evidence-driven optimization iteration.\n")
parts.append(skill_path.read_text())
parts.append("\n\n# Machine-readable iteration context\n")
parts.append(json.dumps(ctx, indent=2, sort_keys=True))


def append_artifact(title, path, limit=evidence_limit):
    path = pathlib.Path(path)
    if not path.is_file():
        return
    data = path.read_bytes()
    if len(data) > limit:
        data = data[-limit:]
        prefix = "[artifact tail; earlier bytes omitted]\n"
    else:
        prefix = ""
    text = data.decode(errors="replace")
    parts.append(f"\n## {title}: {path}\n```text\n{prefix}{text}\n```\n")


parts.append("\n\n# Prior failure evidence\n")
for failure in ctx.get("failures", [])[-8:]:
    parts.append(json.dumps(failure, sort_keys=True) + "\n")
    log_path = failure.get("log_path")
    if log_path:
        append_artifact(f"failure log ({failure.get('stage', 'unknown')})", log_path)

parts.append("\n\n# Latest verification, benchmark, and profiling evidence\n")
artifacts = [run_dir / "verify.json", run_dir / "candidate.json"]
artifacts.extend(sorted(run_dir.glob("profile-*.log"))[-4:])
artifacts.extend(sorted(run_dir.glob("profile-*.note"))[-4:])
artifacts.extend(sorted(run_dir.glob("profile-*.stats.log"))[-4:])
seen = set()
for artifact in artifacts:
    key = str(artifact)
    if key in seen:
        continue
    seen.add(key)
    append_artifact("evidence", artifact)

parts.append("\n\n# Allowed implementation sources\n")
for raw in ctx.get("allowed_paths", []):
    path = pathlib.Path(raw)
    if path.is_file():
        data = path.read_text(errors="replace")
        if len(data.encode()) > source_limit:
            encoded = data.encode()[:source_limit]
            data = encoded.decode(errors="replace")
            suffix = "\n/* SOURCE TRUNCATED BY OPTIMIZATION ADAPTER */\n"
        else:
            suffix = ""
        parts.append(f"\n## {raw}\n```rust\n{data}{suffix}```\n")
    elif path.is_dir():
        for child in sorted(path.rglob("*.rs"))[:12]:
            data = child.read_text(errors="replace")
            encoded = data.encode()[:source_limit]
            data = encoded.decode(errors="replace")
            parts.append(f"\n## {child.as_posix()}\n```rust\n{data}\n```\n")

parts.append(
    "\n# Output protocol\n"
    "Return ONLY a unified git patch. The first output line must start with `diff --git `. "
    "Do not use Markdown fences. Change only allowed paths. Do not edit tests, benchmarks, "
    "verification code, CI, manifests, or documentation. Preserve public semantics and focus on "
    "one evidence-based optimization hypothesis.\n"
)
prompt_path.write_text("".join(parts))
PY

cargo run --quiet -p scirust-sciagent --bin sciagent -- \
  --checkpoint "$SCIAGENT_CKPT" \
  --max-tokens "${SCIAGENT_OPT_MAX_NEW:-4096}" \
  --temperature 0 \
  generate "$(cat "$prompt_file")" > "$raw_file"

awk '
  /^diff --git / { started=1 }
  started && $0 != "```" { print }
' "$raw_file" > "$patch_file"

if ! grep -q '^diff --git ' "$patch_file"; then
  echo "SCIAGENT did not emit a unified git patch; raw output: $raw_file" >&2
  exit 65
fi

python3 - "$SCIAGENT_OPT_CONTEXT" "$patch_file" <<'PY'
import json
import pathlib
import re
import sys

ctx = json.loads(pathlib.Path(sys.argv[1]).read_text())
patch = pathlib.Path(sys.argv[2]).read_text(errors="replace")
allowed = [pathlib.PurePosixPath(p) for p in ctx.get("allowed_paths", [])]
changed = []
for line in patch.splitlines():
    match = re.match(r"^diff --git a/(.+) b/(.+)$", line)
    if match:
        changed.append(pathlib.PurePosixPath(match.group(2)))

if not changed:
    raise SystemExit("patch contains no changed paths")

def is_allowed(path):
    for base in allowed:
        if path == base or base in path.parents:
            return True
    return False

rejected = [str(path) for path in changed if not is_allowed(path)]
if rejected:
    raise SystemExit("patch touches forbidden paths: " + ", ".join(rejected))
PY

git apply --check "$patch_file"
git apply "$patch_file"
git diff --check
