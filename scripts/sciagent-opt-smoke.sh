#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
case "$stage" in
  baseline)
    printf '%s\n' '{"median_ns":1000.0}' > "${SCIAGENT_OPT_BASELINE_METRICS:?}"
    ;;
  generate)
    test -f "${SCIAGENT_OPT_CONTEXT:?}"
    test -f "${SCIAGENT_OPT_SKILL_PATH:?}"
    ;;
  compile)
    # The Rust compile gate is exercised separately by the workflow before the
    # end-to-end runner smoke test. This stage proves command orchestration.
    true
    ;;
  verify)
    printf '%s\n' '{"passed":true,"max_abs_error":1.0e-9,"max_rel_error":1.0e-9,"notes":"smoke verifier"}' > "${SCIAGENT_OPT_VERIFY_METRICS:?}"
    ;;
  benchmark)
    printf '%s\n' '{"median_ns":800.0}' > "${SCIAGENT_OPT_CANDIDATE_METRICS:?}"
    ;;
  profile)
    printf '%s\n' 'smoke profile' > "${SCIAGENT_OPT_RUN_DIR:?}/profile.txt"
    ;;
  rewrite)
    test "${SCIAGENT_OPT_ITERATION:?}" -ge 2
    ;;
  *)
    echo "unknown smoke stage: $stage" >&2
    exit 64
    ;;
esac
