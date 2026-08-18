#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
prompt_len="${SCIAGENT_OPT_I250_PROMPT:-128}"
new_tokens="${SCIAGENT_OPT_I250_NEW:-64}"
samples="${SCIAGENT_OPT_I250_SAMPLES:-5}"

if ! [[ "$samples" =~ ^[0-9]+$ ]] || [ "$samples" -lt 3 ]; then
  echo "SCIAGENT_OPT_I250_SAMPLES must be an integer >= 3" >&2
  exit 64
fi

run_bench() {
  local prompt="$1"
  local new="$2"
  SCIAGENT_DECODE_PROMPT="$prompt" \
  SCIAGENT_DECODE_NEW="$new" \
    cargo run --quiet --release -p scirust-sciagent --features cuda --example cuda_decode_bench
}

field_from_line() {
  local key="$1"
  awk -v wanted="$key" '{
    for (i = 1; i <= NF; ++i) {
      split($i, kv, "=")
      if (kv[1] == wanted) {
        print kv[2]
        exit
      }
    }
  }'
}

measure_to() {
  local destination="$1"
  local label="$2"
  local tmp
  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' RETURN

  local i
  for i in $(seq 1 "$samples"); do
    local output line tps parity
    output="$(run_bench "$prompt_len" "$new_tokens")"
    printf '%s\n' "$output" > "${SCIAGENT_OPT_RUN_DIR:?}/${label}-sample-${i}.log"
    line="$(printf '%s\n' "$output" | grep '^SCIAGENT_I250_DECODE ' | tail -n 1)"
    if [ -z "$line" ]; then
      echo "missing SCIAGENT_I250_DECODE benchmark record" >&2
      exit 65
    fi
    tps="$(printf '%s\n' "$line" | field_from_line fast_tok_s)"
    parity="$(printf '%s\n' "$line" | field_from_line parity)"
    if [ "$parity" != "true" ]; then
      echo "I250 fast path lost exact cached-oracle token parity" >&2
      exit 66
    fi
    awk -v tps="$tps" 'BEGIN {
      if (!(tps > 0)) exit 1;
      printf "%.9f\n", 1000000000.0 / tps;
    }' >> "$tmp"
  done

  local median
  median="$(sort -n "$tmp" | awk '{ values[NR] = $1 } END {
    if (NR % 2 == 1) print values[(NR + 1) / 2];
    else print (values[NR / 2] + values[NR / 2 + 1]) / 2.0;
  }')"
  printf '{"median_ns":%.9f}\n' "$median" > "$destination"
  rm -f "$tmp"
  trap - RETURN
}

case "$stage" in
  baseline)
    measure_to "${SCIAGENT_OPT_BASELINE_METRICS:?}" baseline
    ;;
  generate|rewrite)
    exec bash scripts/sciagent-opt-local-model.sh "$stage"
    ;;
  compile)
    cargo check -p scirust-sciagent --features cuda --example cuda_decode_bench
    ;;
  verify)
    output="$(run_bench "$prompt_len" "${SCIAGENT_OPT_I250_VERIFY_NEW:-8}")"
    printf '%s\n' "$output" > "${SCIAGENT_OPT_RUN_DIR:?}/verify-${SCIAGENT_OPT_ITERATION:?}.log"
    line="$(printf '%s\n' "$output" | grep '^SCIAGENT_I250_DECODE ' | tail -n 1)"
    parity="$(printf '%s\n' "$line" | field_from_line parity)"
    lm_parity="$(printf '%s\n' "$line" | field_from_line lm_parity)"
    dense_parity="$(printf '%s\n' "$line" | field_from_line dense_parity)"
    if [ "$parity" = "true" ] && [ "$lm_parity" = "true" ] && [ "$dense_parity" = "true" ]; then
      printf '%s\n' '{"passed":true,"notes":"exact generated-token parity against the B49 cached oracle and canonical full-logits baselines"}' > "${SCIAGENT_OPT_VERIFY_METRICS:?}"
    else
      printf '%s\n' '{"passed":false,"notes":"I250 decode parity failure"}' > "${SCIAGENT_OPT_VERIFY_METRICS:?}"
      exit 67
    fi
    ;;
  benchmark)
    measure_to "${SCIAGENT_OPT_CANDIDATE_METRICS:?}" candidate-${SCIAGENT_OPT_ITERATION:?}
    ;;
  profile)
    cargo build --quiet --release -p scirust-sciagent --features cuda --example cuda_decode_bench
    binary="target/release/examples/cuda_decode_bench"
    if command -v nsys >/dev/null 2>&1; then
      base="${SCIAGENT_OPT_RUN_DIR:?}/nsys-${SCIAGENT_OPT_ITERATION:?}"
      SCIAGENT_DECODE_PROMPT="$prompt_len" \
      SCIAGENT_DECODE_NEW="${SCIAGENT_OPT_I250_PROFILE_NEW:-16}" \
        nsys profile \
          --force-overwrite=true \
          -o "$base" \
          "$binary" \
          > "${SCIAGENT_OPT_RUN_DIR}/profile-${SCIAGENT_OPT_ITERATION}.log" 2>&1

      report="$(find "$SCIAGENT_OPT_RUN_DIR" -maxdepth 1 -type f -name "nsys-${SCIAGENT_OPT_ITERATION}*.nsys-rep" -print -quit)"
      if [ -n "$report" ]; then
        nsys stats "$report" \
          > "${SCIAGENT_OPT_RUN_DIR}/profile-${SCIAGENT_OPT_ITERATION}.stats.log" 2>&1 || true
      else
        printf '%s\n' 'nsys completed but no .nsys-rep file was found for textual stats extraction' \
          > "${SCIAGENT_OPT_RUN_DIR}/profile-${SCIAGENT_OPT_ITERATION}.note"
      fi
    else
      run_bench "$prompt_len" "${SCIAGENT_OPT_I250_PROFILE_NEW:-16}" \
        > "${SCIAGENT_OPT_RUN_DIR:?}/profile-${SCIAGENT_OPT_ITERATION:?}.log"
      printf '%s\n' 'nsys unavailable; retained benchmark evidence instead of a fabricated profiler result' \
        > "${SCIAGENT_OPT_RUN_DIR}/profile-${SCIAGENT_OPT_ITERATION}.note"
    fi
    ;;
  *)
    echo "unknown I250 optimization stage: $stage" >&2
    exit 64
    ;;
esac
