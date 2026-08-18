#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
iteration="${SCIAGENT_OPT_ITERATION:-0}"
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
    # end-to-end runner smoke test. Optional failure injection proves recovery.
    if [ "${SCIAGENT_OPT_SMOKE_FAIL_COMPILE_ITER:-0}" = "$iteration" ]; then
      echo "intentional smoke compile failure at iteration $iteration" >&2
      exit 42
    fi
    ;;
  verify)
    if [ "${SCIAGENT_OPT_SMOKE_FAIL_VERIFY_ITER:-0}" = "$iteration" ]; then
      printf '%s\n' '{"passed":false,"max_abs_error":1.0,"max_rel_error":1.0,"notes":"intentional smoke verification failure"}' > "${SCIAGENT_OPT_VERIFY_METRICS:?}"
    else
      printf '%s\n' '{"passed":true,"max_abs_error":1.0e-9,"max_rel_error":1.0e-9,"notes":"smoke verifier"}' > "${SCIAGENT_OPT_VERIFY_METRICS:?}"
    fi
    ;;
  benchmark)
    min_iteration="${SCIAGENT_OPT_SMOKE_BENCHMARK_MIN_ITER:-1}"
    if [ "$iteration" -lt "$min_iteration" ]; then
      echo "benchmark was invoked too early at iteration $iteration; expected >= $min_iteration" >&2
      exit 43
    fi
    printf '%s\n' '{"median_ns":800.0}' > "${SCIAGENT_OPT_CANDIDATE_METRICS:?}"
    ;;
  profile)
    printf '%s\n' 'smoke profile' > "${SCIAGENT_OPT_RUN_DIR:?}/profile-${iteration}.log"
    ;;
  rewrite)
    test "$iteration" -ge 2
    ;;
  *)
    echo "unknown smoke stage: $stage" >&2
    exit 64
    ;;
esac
