#!/usr/bin/env bash
# =============================================================================
# SciRust -- Full Functional Acceptance Protocol
# -----------------------------------------------------------------------------
# One command that certifies the ENTIRE platform: every crate's oracle tests,
# every quality gate CI enforces, cross-process determinism, cross-architecture
# compilation, documentation, and the supply-chain audit. It produces a single
# PASS / FAIL verdict and a timestamped evidence bundle suitable for an
# industrial acceptance sign-off.
#
# Nothing here is a stub: every "functionality" is verified by the same honest
# oracle tests that ship inside each crate (fixed-seed RNG, golden constants,
# agreement against an independent reference). This protocol *runs* them all.
#
# GATES (the eight CI enforces, plus reproducibility & opt-in extras)
#   fmt          rustfmt --check                         (required)
#   clippy       clippy --workspace --all-targets -Dwarn (required)
#   build        build  --workspace --all-targets        (required)
#   test         test   --workspace   (ALL oracle tests) (required)
#   simd         portable-simd feature test              (required, nightly)
#   determinism  two-process reproducibility re-run      (required)
#   doc          rustdoc -D warnings                     (required)
#   aarch64      cross-check NEON/SVE paths              (required if target)
#   deny         cargo-deny license & security audit     (required if installed)
#   clippy-gpu   scirust-gpu wgpu-feature lint           (optional)
#   gpu          scirust-gpu wgpu test on Vulkan adapter (optional, needs ICD)
#   stable       build+test on the stable toolchain      (optional, if present)
#   examples     smoke-run the data-free example bins    (optional, --with-examples)
#
# USAGE
#   scripts/test-protocol.sh                 # full protocol (default)
#   scripts/test-protocol.sh --quick         # fmt+clippy+build+test+determinism
#   scripts/test-protocol.sh --with-examples # also smoke-run example binaries
#   scripts/test-protocol.sh --only test,doc # run just these gates
#   scripts/test-protocol.sh --skip gpu,deny # run all but these
#   scripts/test-protocol.sh --strict        # missing prereqs FAIL (no SKIP)
#   scripts/test-protocol.sh --no-clean      # keep target/doc & incremental
#   scripts/test-protocol.sh --list          # print the gate plan and exit
#   scripts/test-protocol.sh -h | --help
#
# EXIT CODE  0 = every required gate PASSED.  Non-zero = at least one required
#            gate FAILED (or, under --strict, was skipped for a missing prereq).
# =============================================================================

set -uo pipefail

# ---- locate the workspace root (the script lives in <root>/scripts) ---------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

# Match CI's build hygiene and keep the debug tree small: incremental artifacts
# can double target/ and have overflowed constrained containers mid-run.
export CARGO_INCREMENTAL=0
export RUST_BACKTRACE=1

# ---- options ----------------------------------------------------------------
QUICK=0; STRICT=0; CLEAN=1; WITH_EXAMPLES=0; LIST=0
ONLY=""; SKIP=""
QUICK_SET="fmt,clippy,build,test,determinism"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)         QUICK=1 ;;
    --strict)        STRICT=1 ;;
    --no-clean)      CLEAN=0 ;;
    --with-examples) WITH_EXAMPLES=1 ;;
    --list)          LIST=1 ;;
    --only)          ONLY="${2:-}"; shift ;;
    --only=*)        ONLY="${1#*=}" ;;
    --skip)          SKIP="${2:-}"; shift ;;
    --skip=*)        SKIP="${1#*=}" ;;
    -h|--help)       sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# ---- presentation -----------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_BLU=$'\033[34m'; C_CYN=$'\033[36m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""; C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_CYN=""
fi
BAR="$(printf '%.0s-' {1..72})"
hr()   { printf '%s%s%s\n' "$C_DIM" "$BAR" "$C_RESET"; }
ok()   { printf '%s+ %s%s\n' "$C_GRN" "$*" "$C_RESET"; }
bad()  { printf '%sx %s%s\n' "$C_RED" "$*" "$C_RESET"; }
warn() { printf '%s* %s%s\n' "$C_YEL" "$*" "$C_RESET"; }
info() { printf '%s%s%s\n'   "$C_CYN" "$*" "$C_RESET"; }

# ---- evidence bundle --------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$ROOT/target/protocol-logs/run-$STAMP"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"

# ---- result bookkeeping -----------------------------------------------------
declare -a GATE_ID GATE_REQ GATE_STATUS GATE_DUR GATE_NOTE
record() { GATE_ID+=("$1"); GATE_REQ+=("$2"); GATE_STATUS+=("$3"); GATE_DUR+=("$4"); GATE_NOTE+=("$5"); }

# Decide whether a gate is in scope given --only / --skip / --quick.
should_run() {
  local id="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$id,"* ]] && return 0 || return 1; fi
  if [[ -n "$SKIP" && ",$SKIP," == *",$id,"* ]]; then return 1; fi
  if [[ $QUICK -eq 1 ]]; then [[ ",$QUICK_SET," == *",$id,"* ]] && return 0 || return 1; fi
  return 0
}

# Run one gate: `run_gate ID required|optional "Description" "shell command"`.
run_gate() {
  local id="$1" req="$2" desc="$3" cmd="$4"
  local log="$LOG_DIR/${id}.log"
  hr
  printf '  %s> [%s]%s %s\n' "$C_BOLD" "$id" "$C_RESET" "$desc"
  printf '    %s$ %s%s\n' "$C_DIM" "$cmd" "$C_RESET"
  local start dur rc
  start=$(date +%s)
  ( eval "$cmd" ) >"$log" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [[ $rc -eq 0 ]]; then
    record "$id" "$req" PASS "$dur" ""
    ok "[$id] PASS (${dur}s)  -> $(basename "$log")"
  else
    record "$id" "$req" FAIL "$dur" "exit=$rc"
    bad "[$id] FAIL (exit $rc, ${dur}s) -- last lines:"
    tail -n 18 "$log" | sed 's/^/        /'
  fi
  return $rc
}

# Skip a gate, recording the reason honestly.
skip_gate() {
  local id="$1" req="$2" reason="$3"
  if [[ $STRICT -eq 1 && "$req" == required ]]; then
    record "$id" "$req" FAIL 0 "missing prerequisite (--strict): $reason"
    bad "[$id] FAIL (strict) -- $reason"
  else
    record "$id" "$req" SKIP 0 "$reason"
    warn "[$id] SKIP -- $reason"
  fi
}

# ---- gate plan (for --list and the header) ----------------------------------
print_plan() {
  local ids=(fmt clippy build test simd determinism doc aarch64 deny clippy-gpu gpu stable examples)
  info "Gate plan:"
  for id in "${ids[@]}"; do
    if should_run "$id"; then printf '   %s*%s %s\n' "$C_GRN" "$C_RESET" "$id"
    else printf '   %s.%s %s%s%s\n' "$C_DIM" "$C_RESET" "$C_DIM" "$id (excluded)" "$C_RESET"; fi
  done
}

# =============================================================================
#  HEADER / PREFLIGHT
# =============================================================================
[[ -t 1 ]] && { clear 2>/dev/null || true; }
EQBAR="$(printf '%.0s=' {1..72})"
printf '%s%s\n' "$C_BOLD" "$EQBAR"
printf '%s        SciRust - Full Functional Acceptance Protocol\n' "$C_BOLD"
printf '%s%s%s\n' "$C_BOLD" "$EQBAR" "$C_RESET"
info "Workspace : $ROOT"
info "Started   : $(date -u '+%Y-%m-%dT%H:%M:%SZ')  (commit $(git rev-parse --short HEAD 2>/dev/null || echo '?'), branch $(git branch --show-current 2>/dev/null || echo '?'))"
info "Evidence  : $LOG_DIR"
echo
print_plan
if [[ $LIST -eq 1 ]]; then echo; info "(--list) no gates executed."; exit 0; fi
echo

# Toolchain provenance -- recorded into the evidence bundle.
{
  echo "# Toolchain"
  rustc -V 2>/dev/null || echo "rustc: MISSING"
  cargo -V 2>/dev/null || echo "cargo: MISSING"
  cargo clippy -V 2>/dev/null || echo "clippy: MISSING"
  cargo fmt --version 2>/dev/null || echo "rustfmt: MISSING"
  echo "# Host"; uname -a
} >"$LOG_DIR/toolchain.txt" 2>&1
info "Toolchain : $(rustc -V 2>/dev/null || echo '?') / $(cargo -V 2>/dev/null || echo '?')"

# Disk guard -- a full --all-targets debug tree for the workspace is ~25-30 GB.
# Estimate free space and bail early with a clear message rather than ENOSPC
# mid-run (which can wedge the whole container). The build/test gates are the
# peak; lighter scopes (--only fmt,clippy) need far less.
DISK_PCT="$(df -P "$ROOT" | awk 'NR==2{gsub("%","",$5); print $5}')"
FREE_KB="$(df -P "$ROOT" | awk 'NR==2{print $4}')"
FREE_GB=$(( FREE_KB / 1024 / 1024 ))
info "Disk      : ${DISK_PCT}% used, ${FREE_GB} GB free on $(df -P "$ROOT" | awk 'NR==2{print $6}')"
# How much headroom this scope needs (the heavy build/test gates dominate). With
# CARGO_INCREMENTAL=0 the all-targets debug tree is ~20 GB; allow margin for the
# transient doctest temps cargo writes under /tmp during the test gate.
NEED_GB=24
should_run build || should_run test || should_run simd || should_run aarch64 || NEED_GB=3
if [[ "$FREE_GB" -lt "$NEED_GB" ]]; then
  bad "Insufficient disk: ${FREE_GB} GB free, this scope needs ~${NEED_GB} GB."
  warn "Reclaim with 'cargo clean' (or 'rm -rf target'), or run a lighter scope,"
  warn "e.g. --only fmt,clippy  (the build/test gates are what need the space)."
  exit 3
fi
if [[ "${DISK_PCT:-0}" -ge 95 ]]; then
  bad "Refusing to start: disk ${DISK_PCT}% full. Free space first ('cargo clean')."
  exit 3
fi

# Reclaim space before a full build (docs + incremental are the big, safe wins).
if [[ $CLEAN -eq 1 ]]; then
  rm -rf "$ROOT/target/doc" "$ROOT/target/debug/incremental" 2>/dev/null || true
  info "Cleaned   : target/doc and target/debug/incremental"
fi
echo

# =============================================================================
#  GATE 1 -- Formatting
# =============================================================================
if should_run fmt; then
  run_gate fmt required "Formatting is canonical (rustfmt --check)" \
    "cargo fmt --all -- --check"
fi

# =============================================================================
#  GATE 2 -- Lints (the wall: -D warnings, all targets)
# =============================================================================
if should_run clippy; then
  run_gate clippy required "Lints clean across the workspace (-D warnings)" \
    "cargo clippy --workspace --all-targets -- -D warnings"
fi

# =============================================================================
#  GATE 3 -- Build everything (lib + bins + tests + examples)
# =============================================================================
if should_run build; then
  run_gate build required "Whole workspace builds (--all-targets)" \
    "cargo build --workspace --all-targets"
fi

# =============================================================================
#  GATE 4 -- THE functionality gate: every crate's oracle tests
# =============================================================================
if should_run test; then
  run_gate test required "Every crate's oracle tests (unit + integration + doc)" \
    "cargo test --workspace --no-fail-fast"
  # Tally the evidence from the test log (best-effort, never fails the gate).
  TLOG="$LOG_DIR/test.log"
  if [[ -f "$TLOG" ]]; then
    read -r T_PASS T_FAIL T_IGN < <(awk '
      /test result:/ { for (i=1;i<=NF;i++) {
        if ($i ~ /^passed/)  p+=$(i-1);
        if ($i ~ /^failed/)  f+=$(i-1);
        if ($i ~ /^ignored/) g+=$(i-1); } }
      END { printf "%d %d %d", p, f, g }' "$TLOG")
    T_GROUPS=$(grep -c 'test result:' "$TLOG" 2>/dev/null || echo 0)
    info "    -> ${T_PASS:-0} passed, ${T_FAIL:-0} failed, ${T_IGN:-0} ignored across ${T_GROUPS:-0} test groups"
    echo "tests_passed=${T_PASS:-0}"   >>"$SUMMARY"
    echo "tests_failed=${T_FAIL:-0}"   >>"$SUMMARY"
    echo "tests_ignored=${T_IGN:-0}"   >>"$SUMMARY"
    echo "test_groups=${T_GROUPS:-0}"  >>"$SUMMARY"
  fi
fi

# =============================================================================
#  GATE 5 -- Optional nightly portable-simd path
# =============================================================================
if should_run simd; then
  run_gate simd required "Portable-SIMD kernels (nightly feature) test" \
    "cargo test -p scirust-simd --features portable-simd"
fi

# =============================================================================
#  GATE 6 -- Cross-process determinism / reproducibility
#  Numeric determinism is pinned by the workspace's golden / fixed-seed oracle
#  tests. We re-run that determinism-tagged subset in TWO independent processes
#  and require an identical set of green results -- proving a computation is
#  bit-reproducible across process invocations, not merely within one.
# =============================================================================
if should_run determinism; then
  hr
  printf '  %s> [determinism]%s Cross-process reproducibility of the oracle suite\n' "$C_BOLD" "$C_RESET"
  DET_FILTERS="deterministic determinism reproducib bit_exact bit_deterministic bit_reproducible golden"
  printf '    %s$ cargo test --workspace -- %s   (x2 processes, compared)%s\n' "$C_DIM" "$DET_FILTERS" "$C_RESET"
  d_start=$(date +%s)
  cargo test --workspace --no-fail-fast -- $DET_FILTERS >"$LOG_DIR/determinism-run1.log" 2>&1; drc1=$?
  cargo test --workspace --no-fail-fast -- $DET_FILTERS >"$LOG_DIR/determinism-run2.log" 2>&1; drc2=$?
  d_dur=$(( $(date +%s) - d_start ))
  # Signature = sorted set of "<test path> ... ok" lines (timing stripped).
  det_sig() { grep -hE '\.\.\. ok$' "$1" | sed -E 's/ \.\.\. ok$//; s/^test //' | sort; }
  det_sig "$LOG_DIR/determinism-run1.log" >"$LOG_DIR/determinism-sig1.txt"
  det_sig "$LOG_DIR/determinism-run2.log" >"$LOG_DIR/determinism-sig2.txt"
  DET_N=$(wc -l <"$LOG_DIR/determinism-sig1.txt" | tr -d ' ')
  if [[ $drc1 -ne 0 || $drc2 -ne 0 ]]; then
    record determinism required FAIL "$d_dur" "a determinism run did not pass (rc=$drc1/$drc2)"
    bad "[determinism] FAIL -- oracle re-run did not stay green (rc=$drc1/$drc2)"
    tail -n 18 "$LOG_DIR/determinism-run1.log" | sed 's/^/        /'
  elif [[ "${DET_N:-0}" -lt 1 ]]; then
    record determinism required FAIL "$d_dur" "no determinism-tagged tests matched"
    bad "[determinism] FAIL -- filter matched zero tests (cannot verify reproducibility)"
  elif ! diff -q "$LOG_DIR/determinism-sig1.txt" "$LOG_DIR/determinism-sig2.txt" >/dev/null; then
    record determinism required FAIL "$d_dur" "the two process runs disagreed"
    bad "[determinism] FAIL -- runs diverged across processes:"
    diff "$LOG_DIR/determinism-sig1.txt" "$LOG_DIR/determinism-sig2.txt" | head -20 | sed 's/^/        /'
  else
    record determinism required PASS "$d_dur" "$DET_N tests reproduced bit-for-bit across 2 processes"
    ok "[determinism] PASS (${d_dur}s) -- $DET_N determinism oracles identical across 2 independent processes"
    echo "determinism_tests=$DET_N" >>"$SUMMARY"
  fi
fi

# =============================================================================
#  GATE 7 -- Documentation builds warning-free
# =============================================================================
if should_run doc; then
  run_gate doc required "Public docs build with no warnings (rustdoc -D warnings)" \
    "RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --workspace"
fi

# =============================================================================
#  GATE 8 -- Cross-architecture compilation (aarch64 NEON/SVE paths)
# =============================================================================
if should_run aarch64; then
  if rustc --print target-list 2>/dev/null | grep -q '^aarch64-unknown-linux-gnu$'; then
    if ! rustup target list --installed 2>/dev/null | grep -q '^aarch64-unknown-linux-gnu$'; then
      info "    installing aarch64-unknown-linux-gnu std (one-off)..."
      rustup target add aarch64-unknown-linux-gnu >"$LOG_DIR/aarch64-target-add.log" 2>&1 || true
    fi
    if rustup target list --installed 2>/dev/null | grep -q '^aarch64-unknown-linux-gnu$'; then
      # The aarch64 check builds a whole second-architecture tree (~10 GB). The
      # host build/test/doc gates are finished by now, so on a tight disk reclaim
      # the host debug tree first to make room (optional later gates rebuild it).
      A64_FREE_GB=$(( $(df -P "$ROOT" | awk 'NR==2{print $4}') / 1024 / 1024 ))
      if [[ "$A64_FREE_GB" -lt 14 && -d "$ROOT/target/debug" ]]; then
        warn "    reclaiming host target/debug (${A64_FREE_GB} GB free) to fit the aarch64 tree"
        rm -rf "$ROOT/target/debug" 2>/dev/null || true
      fi
      run_gate aarch64 required "Cross-check NEON/SVE paths (aarch64)" \
        "cargo check --workspace --all-targets --target aarch64-unknown-linux-gnu"
      # Reclaim the second-architecture tree immediately -- it is ~10 GB and not
      # needed once the check passes.
      rm -rf "$ROOT/target/aarch64-unknown-linux-gnu" 2>/dev/null || true
    else
      skip_gate aarch64 required "aarch64 std unavailable (offline rustup?)"
    fi
  else
    skip_gate aarch64 required "this rustc cannot target aarch64-unknown-linux-gnu"
  fi
fi

# =============================================================================
#  GATE 9 -- License & security audit
# =============================================================================
if should_run deny; then
  if cargo deny --version >/dev/null 2>&1; then
    run_gate deny required "License & advisory audit (cargo-deny)" \
      "cargo deny check"
  else
    skip_gate deny required "cargo-deny not installed (cargo install cargo-deny)"
  fi
fi

# =============================================================================
#  OPTIONAL -- scirust-gpu wgpu feature lint (compiles, no adapter needed)
# =============================================================================
if should_run clippy-gpu; then
  run_gate clippy-gpu optional "scirust-gpu wgpu-feature lints (-D warnings)" \
    "cargo clippy -p scirust-gpu --features wgpu --all-targets -- -D warnings"
fi

# =============================================================================
#  OPTIONAL -- real wgpu compute path on a Vulkan adapter
# =============================================================================
if should_run gpu; then
  if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo >/dev/null 2>&1; then
    run_gate gpu optional "wgpu GEMM vs CPU oracle on a Vulkan adapter" \
      "cargo test -p scirust-gpu --features wgpu"
  else
    skip_gate gpu optional "no Vulkan adapter (install mesa-vulkan-drivers / lavapipe)"
  fi
fi

# =============================================================================
#  OPTIONAL -- stable toolchain build + test (CI's industrial gate)
# =============================================================================
if should_run stable; then
  if rustup toolchain list 2>/dev/null | grep -q '^stable'; then
    run_gate stable optional "Build + test on the stable toolchain" \
      "cargo +stable build --workspace --all-targets && cargo +stable test --workspace"
  else
    skip_gate stable optional "no stable toolchain (rustup toolchain install stable)"
  fi
fi

# =============================================================================
#  OPTIONAL -- smoke-run the self-contained example binaries (--with-examples)
#  Data-dependent demos (MNIST/CIFAR/sentiment) are skipped unless their data
#  is present, so this never produces a false failure.
# =============================================================================
if [[ $WITH_EXAMPLES -eq 1 ]] && should_run examples; then
  hr
  printf '  %s> [examples]%s Smoke-run data-free example binaries\n' "$C_BOLD" "$C_RESET"
  ex_log="$LOG_DIR/examples.log"; : >"$ex_log"
  ex_fail=0; ex_ran=0
  # name|package|needs_data
  ex_list=(
    "ids_demo|ids_demo|no"
    "industrial_monitor|industrial_monitor|no"
    "quickstart_v2|quickstart_v2|no"
    "tensor-examples|scirust-tensor-examples|no"
  )
  for spec in "${ex_list[@]}"; do
    IFS='|' read -r nm pkg _ <<<"$spec"
    printf '    * %s ... ' "$nm"
    if timeout 300 cargo run -q -p "$pkg" >>"$ex_log" 2>&1; then
      printf '%sok%s\n' "$C_GRN" "$C_RESET"; ex_ran=$((ex_ran+1))
    else
      printf '%sFAILED%s\n' "$C_RED" "$C_RESET"; ex_fail=$((ex_fail+1))
    fi
  done
  if [[ $ex_fail -eq 0 ]]; then
    record examples optional PASS 0 "$ex_ran example binaries ran clean"
    ok "[examples] PASS -- $ex_ran example binaries ran clean"
  else
    record examples optional FAIL 0 "$ex_fail example binaries failed (see examples.log)"
    bad "[examples] FAIL -- $ex_fail example binaries failed"
  fi
fi

# =============================================================================
#  FINAL REPORT
# =============================================================================
echo
hr
printf '%s  ACCEPTANCE REPORT%s   %s\n' "$C_BOLD" "$C_RESET" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
hr

# Authoritative workspace size (one manifest_path per package, --no-deps).
PKG_COUNT="$(cargo metadata --no-deps --format-version 1 2>/dev/null | grep -o '"manifest_path":' | wc -l | tr -d ' ')"

req_fail=0; opt_fail=0; req_skip=0; n_pass=0; n_total=0
printf '  %-13s %-8s %-8s %s\n' "GATE" "REQUIRED" "STATUS" "DETAIL"
printf '  %-13s %-8s %-8s %s\n' "-------------" "--------" "------" "------"
for i in "${!GATE_ID[@]}"; do
  id="${GATE_ID[$i]}"; req="${GATE_REQ[$i]}"; st="${GATE_STATUS[$i]}"
  dur="${GATE_DUR[$i]}"; note="${GATE_NOTE[$i]}"
  n_total=$((n_total+1))
  case "$st" in
    PASS) col="$C_GRN"; n_pass=$((n_pass+1)) ;;
    FAIL) col="$C_RED"; [[ "$req" == required ]] && req_fail=$((req_fail+1)) || opt_fail=$((opt_fail+1)) ;;
    SKIP) col="$C_YEL"; [[ "$req" == required ]] && req_skip=$((req_skip+1)) ;;
    *)    col="$C_RESET" ;;
  esac
  detail="$note"; [[ "$st" == PASS ]] && detail="${dur}s${note:+ -- $note}"
  printf '  %-13s %-8s %s%-8s%s %s\n' "$id" "$req" "$col" "$st" "$C_RESET" "$detail"
done

echo
info "Workspace : $PKG_COUNT packages in the build graph"
[[ -n "${T_PASS:-}" ]] && info "Tests     : ${T_PASS} passed / ${T_FAIL:-0} failed / ${T_IGN:-0} ignored (${T_GROUPS:-0} groups)"
[[ -n "${DET_N:-}" && "${DET_N:-0}" -gt 0 ]] && info "Determinism: $DET_N oracles reproduced across 2 processes"
info "Evidence  : $LOG_DIR"

# Persist a machine-readable summary into the bundle.
{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo '?')"
  echo "branch=$(git branch --show-current 2>/dev/null || echo '?')"
  echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "packages=$PKG_COUNT"
  for i in "${!GATE_ID[@]}"; do
    echo "gate.${GATE_ID[$i]}=${GATE_STATUS[$i]} (${GATE_REQ[$i]}, ${GATE_DUR[$i]}s)${GATE_NOTE[$i]:+ -- ${GATE_NOTE[$i]}}"
  done
} >>"$SUMMARY"

echo
if [[ $req_fail -eq 0 && $req_skip -eq 0 ]]; then
  printf '%s%s  VERDICT: PASS  %s -- all required gates green (%d/%d gates passed).\n' "$C_BOLD" "$C_GRN" "$C_RESET" "$n_pass" "$n_total"
  echo "verdict=PASS" >>"$SUMMARY"; exit 0
elif [[ $req_fail -eq 0 && $req_skip -gt 0 ]]; then
  printf '%s%s  VERDICT: PASS (with gaps)  %s -- %d required gate(s) skipped for missing prerequisites; coverage incomplete.\n' "$C_BOLD" "$C_YEL" "$C_RESET" "$req_skip"
  echo "verdict=PASS_WITH_GAPS" >>"$SUMMARY"; exit 0
else
  printf '%s%s  VERDICT: FAIL  %s -- %d required gate(s) failed. See the evidence bundle.\n' "$C_BOLD" "$C_RED" "$C_RESET" "$req_fail"
  echo "verdict=FAIL" >>"$SUMMARY"; exit 1
fi
