#!/usr/bin/env bash
# =============================================================================
# SciRust - NVIDIA Jetson (aarch64) Acceptance Protocol
# -----------------------------------------------------------------------------
# Run THIS on the Jetson. Unlike x86 CI - which only *cross-compiles* the
# aarch64 NEON/SVE paths - a Jetson is a native aarch64 host, so the build and
# test gates actually EXECUTE those ARM kernels (NEON int8, aarch64 sgemv, the
# SIMD reductions). This script is the authoritative on-device validation: it
# runs every crate's oracle tests natively on ARM, re-proves cross-process
# determinism, builds the docs, and prints one PASS / FAIL verdict plus a
# timestamped evidence bundle.
#
# It is self-contained: copy it onto the Jetson and run it from anywhere; it
# locates the scirust workspace (or you point it at one with --repo).
#
# QUICK START (on the Jetson)
#   # 1. clone the repo if you have not already
#   git clone https://github.com/Memorithm/scirust.git
#   # 2. install the nightly Rust toolchain (the repo pins nightly)
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   rustup toolchain install nightly -c rustfmt -c clippy
#   # 3. run the protocol
#   ./test-protocol-jetson.sh --repo ./scirust
#
# USAGE
#   test-protocol-jetson.sh [--repo PATH] [--quick] [--jobs N]
#                           [--with-gpu] [--only a,b] [--skip a,b]
#                           [--min-disk GB] [--list] [-h|--help]
#
#   --repo PATH    path to the scirust workspace (default: autodetect)
#   --quick        fmt + clippy + build + test + determinism only
#   --jobs N       cargo build jobs (default: memory-aware, to avoid OOM)
#   --with-gpu     also run the wgpu GPU gate (needs a Vulkan adapter)
#   --only a,b     run only these gates
#   --skip a,b     run all gates except these
#   --min-disk GB  required free space for the build/test scope (default 22)
#   --list         print the platform report + gate plan, then exit
#
# EXIT CODE  0 = every required gate PASSED.  Non-zero = a required gate FAILED.
# =============================================================================

set -uo pipefail

# ---- options ----------------------------------------------------------------
REPO=""; QUICK=0; WITH_GPU=0; LIST=0; ONLY=""; SKIP=""; JOBS=""; MIN_DISK=22
QUICK_SET="fmt,clippy,build,test,determinism"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)     REPO="${2:-}"; shift ;;
    --repo=*)   REPO="${1#*=}" ;;
    --quick)    QUICK=1 ;;
    --with-gpu) WITH_GPU=1 ;;
    --jobs)     JOBS="${2:-}"; shift ;;
    --jobs=*)   JOBS="${1#*=}" ;;
    --min-disk) MIN_DISK="${2:-}"; shift ;;
    --min-disk=*) MIN_DISK="${1#*=}" ;;
    --only)     ONLY="${2:-}"; shift ;;
    --only=*)   ONLY="${1#*=}" ;;
    --skip)     SKIP="${2:-}"; shift ;;
    --skip=*)   SKIP="${1#*=}" ;;
    --list)     LIST=1 ;;
    -h|--help)  sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# ---- presentation -----------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_CYN=$'\033[36m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""; C_RED=""; C_GRN=""; C_YEL=""; C_CYN=""
fi
BAR="$(printf '%.0s-' {1..72})"
hr()   { printf '%s%s%s\n' "$C_DIM" "$BAR" "$C_RESET"; }
ok()   { printf '%s+ %s%s\n' "$C_GRN" "$*" "$C_RESET"; }
bad()  { printf '%sx %s%s\n' "$C_RED" "$*" "$C_RESET"; }
warn() { printf '%s* %s%s\n' "$C_YEL" "$*" "$C_RESET"; }
info() { printf '%s%s%s\n'   "$C_CYN" "$*" "$C_RESET"; }

# ---- locate the workspace ---------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
is_workspace() { [[ -f "$1/Cargo.toml" && -d "$1/scirust-core" ]]; }
if [[ -z "$REPO" ]]; then
  if   [[ -n "${SCIRUST_HOME:-}" ]] && is_workspace "$SCIRUST_HOME"; then REPO="$SCIRUST_HOME"
  elif is_workspace "$PWD";                then REPO="$PWD"
  elif is_workspace "$SCRIPT_DIR/..";      then REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
  elif is_workspace "$PWD/scirust";        then REPO="$PWD/scirust"
  fi
fi
if [[ -z "$REPO" ]] || ! is_workspace "$REPO"; then
  bad "Could not find the scirust workspace."
  warn "Run from inside the repo, or pass --repo /path/to/scirust, or:"
  warn "  git clone https://github.com/Memorithm/scirust.git && \\"
  warn "  $0 --repo ./scirust"
  exit 2
fi
REPO="$(cd "$REPO" && pwd)"
cd "$REPO"

# Keep the debug tree small (matches CI); incremental artifacts can double it
# and Jetson storage is often limited.
export CARGO_INCREMENTAL=0
export RUST_BACKTRACE=1

# Memory-aware job count: a full workspace build can OOM a small Jetson. Allow
# roughly one rustc per 2 GB of RAM, capped at the core count.
NPROC="$(nproc 2>/dev/null || echo 4)"
MEM_GB="$(awk '/MemTotal/{printf "%d",$2/1024/1024}' /proc/meminfo 2>/dev/null || echo 8)"
if [[ -z "$JOBS" ]]; then
  JOBS=$(( MEM_GB / 2 )); [[ "$JOBS" -lt 1 ]] && JOBS=1
  [[ "$JOBS" -gt "$NPROC" ]] && JOBS="$NPROC"
fi
export CARGO_BUILD_JOBS="$JOBS"

# ---- evidence bundle --------------------------------------------------------
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$REPO/target/jetson-protocol-logs/run-$STAMP"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"

# ---- result bookkeeping -----------------------------------------------------
declare -a GATE_ID GATE_REQ GATE_STATUS GATE_DUR GATE_NOTE
record() { GATE_ID+=("$1"); GATE_REQ+=("$2"); GATE_STATUS+=("$3"); GATE_DUR+=("$4"); GATE_NOTE+=("$5"); }
should_run() {
  local id="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$id,"* ]] && return 0 || return 1; fi
  if [[ -n "$SKIP" && ",$SKIP," == *",$id,"* ]]; then return 1; fi
  if [[ $QUICK -eq 1 ]]; then [[ ",$QUICK_SET," == *",$id,"* ]] && return 0 || return 1; fi
  return 0
}
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
    tail -n 20 "$log" | sed 's/^/        /'
  fi
  return $rc
}
skip_gate() { record "$1" "$2" SKIP 0 "$3"; warn "[$1] SKIP -- $3"; }

# =============================================================================
#  PLATFORM REPORT
# =============================================================================
[[ -t 1 ]] && { clear 2>/dev/null || true; }
EQBAR="$(printf '%.0s=' {1..72})"
printf '%s%s\n' "$C_BOLD" "$EQBAR"
printf '%s   SciRust - NVIDIA Jetson (aarch64) Acceptance Protocol\n' "$C_BOLD"
printf '%s%s%s\n' "$C_BOLD" "$EQBAR" "$C_RESET"

ARCH="$(uname -m)"
JETSON_MODEL=""; [[ -r /proc/device-tree/model ]] && JETSON_MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
L4T="$( (grep -o 'R[0-9]\+ (release).*REVISION: [0-9.]\+' /etc/nv_tegra_release 2>/dev/null | head -1) || true)"
CUDA_VER="$( (nvcc --version 2>/dev/null | grep -o 'release [0-9.]\+' | head -1) || true)"
[[ -z "$CUDA_VER" && -f /usr/local/cuda/version.txt ]] && CUDA_VER="$(cat /usr/local/cuda/version.txt)"
NEON="$( (grep -m1 -o 'asimd' /proc/cpuinfo 2>/dev/null) || true)"
VULKAN="no"; command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo >/dev/null 2>&1 && VULKAN="yes"
FREE_GB=$(( $(df -P "$REPO" | awk 'NR==2{print $4}') / 1024 / 1024 ))

info "Workspace : $REPO  (commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?'))"
info "Host      : ${JETSON_MODEL:-unknown device} [$ARCH]"
info "L4T       : ${L4T:-not detected (is this a Jetson?)}"
info "CUDA      : ${CUDA_VER:-not found}   (note: SciRust's CUDA path is archived; GPU gate uses wgpu/Vulkan)"
info "CPU       : ${NPROC} cores, NEON/ASIMD: ${NEON:-unknown}, RAM: ${MEM_GB} GB, build jobs: ${JOBS}"
info "Vulkan    : ${VULKAN}   |   Disk free: ${FREE_GB} GB on $(df -P "$REPO" | awk 'NR==2{print $6}')"
info "Toolchain : $(rustc -V 2>/dev/null || echo 'rustc MISSING') / $(cargo -V 2>/dev/null || echo 'cargo MISSING')"
info "Evidence  : $LOG_DIR"
{
  echo "host=${JETSON_MODEL:-unknown} arch=$ARCH l4t=${L4T:-?} cuda=${CUDA_VER:-none}"
  echo "cores=$NPROC mem_gb=$MEM_GB jobs=$JOBS vulkan=$VULKAN neon=${NEON:-?}"
  rustc -V 2>/dev/null; cargo -V 2>/dev/null; uname -a
} >"$LOG_DIR/platform.txt" 2>&1

if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  warn "This host is '$ARCH', not aarch64 -- the ARM kernels will NOT be exercised natively."
  warn "Run this on the Jetson itself for an on-device validation."
fi

# ---- gate plan --------------------------------------------------------------
echo
PLAN=(fmt clippy build test neon simd determinism doc)
[[ $WITH_GPU -eq 1 ]] && PLAN+=(gpu)
PLAN+=(deny)
info "Gate plan:"
for id in "${PLAN[@]}"; do
  if should_run "$id"; then printf '   %s*%s %s\n' "$C_GRN" "$C_RESET" "$id"
  else printf '   %s.%s %s%s%s\n' "$C_DIM" "$C_RESET" "$C_DIM" "$id (excluded)" "$C_RESET"; fi
done
if [[ $LIST -eq 1 ]]; then echo; info "(--list) no gates executed."; exit 0; fi
echo

# ---- preflight: toolchain + disk -------------------------------------------
if ! cargo -V >/dev/null 2>&1; then
  bad "No 'cargo' on PATH. Install the nightly Rust toolchain (see --help) and retry."
  exit 3
fi
case "$(rustc -V 2>/dev/null)" in
  *nightly*) : ;;
  *) warn "rustc is not nightly. The repo pins nightly (rust-toolchain.toml); some"
     warn "gates (portable-simd) need it. Install: rustup toolchain install nightly" ;;
esac
NEED=$MIN_DISK
should_run build || should_run test || NEED=3
if [[ "$FREE_GB" -lt "$NEED" ]]; then
  bad "Insufficient disk: ${FREE_GB} GB free, this scope needs ~${NEED} GB."
  warn "Free space ('cargo clean'), use a larger volume, or run --quick / --only fmt,clippy."
  exit 3
fi
rm -rf "$REPO/target/doc" "$REPO/target/debug/incremental" 2>/dev/null || true

# =============================================================================
#  GATES
# =============================================================================
if should_run fmt; then
  run_gate fmt required "Formatting is canonical (rustfmt --check)" \
    "cargo fmt --all -- --check"
fi
if should_run clippy; then
  run_gate clippy required "Lints clean across the workspace (-D warnings)" \
    "cargo clippy --workspace --all-targets -- -D warnings"
fi
if should_run build; then
  run_gate build required "Native aarch64 build of the whole workspace (--all-targets)" \
    "cargo build --workspace --all-targets"
fi
if should_run test; then
  run_gate test required "Every crate's oracle tests, executed natively on ARM" \
    "cargo test --workspace --no-fail-fast"
  TLOG="$LOG_DIR/test.log"
  if [[ -f "$TLOG" ]]; then
    read -r T_PASS T_FAIL T_IGN < <(awk '
      /test result:/ { for (i=1;i<=NF;i++) {
        if ($i ~ /^passed/)  p+=$(i-1);
        if ($i ~ /^failed/)  f+=$(i-1);
        if ($i ~ /^ignored/) g+=$(i-1); } }
      END { printf "%d %d %d", p, f, g }' "$TLOG")
    T_GROUPS=$(grep -c 'test result:' "$TLOG" 2>/dev/null || echo 0)
    info "    -> ${T_PASS:-0} passed, ${T_FAIL:-0} failed, ${T_IGN:-0} ignored across ${T_GROUPS:-0} groups"
    { echo "tests_passed=${T_PASS:-0}"; echo "tests_failed=${T_FAIL:-0}"
      echo "tests_ignored=${T_IGN:-0}"; echo "test_groups=${T_GROUPS:-0}"; } >>"$SUMMARY"
  fi
fi

# NEON / aarch64 highlight: re-run just the ARM-relevant oracle tests and report
# how many executed natively. This is the on-device evidence that the NEON int8
# kernels and aarch64 SIMD paths actually RAN (not merely cross-compiled).
if should_run neon; then
  hr
  printf '  %s> [neon]%s ARM kernel oracles executed natively (NEON int8 / aarch64 SIMD)\n' "$C_BOLD" "$C_RESET"
  NEON_FILTERS="neon aarch64 int8 simd quant fixed_point"
  nlog="$LOG_DIR/neon.log"
  if cargo test --workspace --no-fail-fast -- $NEON_FILTERS >"$nlog" 2>&1; then
    n_pass=$(grep -hoE 'result: ok\. [0-9]+ passed' "$nlog" | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
    if [[ "${n_pass:-0}" -gt 0 ]]; then
      record neon required PASS 0 "$n_pass ARM-relevant tests ran natively"
      ok "[neon] PASS -- $n_pass NEON/aarch64/int8/SIMD oracle tests executed on this ARM host"
    else
      record neon optional SKIP 0 "no ARM-tagged tests matched (covered by the test gate)"
      warn "[neon] none matched the filter -- the full 'test' gate still ran them"
    fi
  else
    record neon required FAIL 0 "an ARM kernel test failed"
    bad "[neon] FAIL -- an ARM-relevant test failed:"; tail -n 20 "$nlog" | sed 's/^/        /'
  fi
fi

if should_run simd; then
  run_gate simd required "Portable-SIMD kernels (nightly feature)" \
    "cargo test -p scirust-simd --features portable-simd"
fi

if should_run determinism; then
  hr
  printf '  %s> [determinism]%s Cross-process reproducibility of the oracle suite\n' "$C_BOLD" "$C_RESET"
  DET_FILTERS="deterministic determinism reproducib bit_exact bit_deterministic bit_reproducible golden"
  d_start=$(date +%s)
  cargo test --workspace --no-fail-fast -- $DET_FILTERS >"$LOG_DIR/determinism-run1.log" 2>&1; drc1=$?
  cargo test --workspace --no-fail-fast -- $DET_FILTERS >"$LOG_DIR/determinism-run2.log" 2>&1; drc2=$?
  d_dur=$(( $(date +%s) - d_start ))
  det_sig() { grep -hE '\.\.\. ok$' "$1" | sed -E 's/ \.\.\. ok$//; s/^test //' | sort; }
  det_sig "$LOG_DIR/determinism-run1.log" >"$LOG_DIR/determinism-sig1.txt"
  det_sig "$LOG_DIR/determinism-run2.log" >"$LOG_DIR/determinism-sig2.txt"
  DET_N=$(wc -l <"$LOG_DIR/determinism-sig1.txt" | tr -d ' ')
  if [[ $drc1 -ne 0 || $drc2 -ne 0 ]]; then
    record determinism required FAIL "$d_dur" "a determinism run did not pass (rc=$drc1/$drc2)"
    bad "[determinism] FAIL -- oracle re-run did not stay green (rc=$drc1/$drc2)"
  elif [[ "${DET_N:-0}" -lt 1 ]]; then
    record determinism required FAIL "$d_dur" "no determinism-tagged tests matched"
    bad "[determinism] FAIL -- filter matched zero tests"
  elif ! diff -q "$LOG_DIR/determinism-sig1.txt" "$LOG_DIR/determinism-sig2.txt" >/dev/null; then
    record determinism required FAIL "$d_dur" "the two process runs disagreed"
    bad "[determinism] FAIL -- runs diverged across processes:"
    diff "$LOG_DIR/determinism-sig1.txt" "$LOG_DIR/determinism-sig2.txt" | head -20 | sed 's/^/        /'
  else
    record determinism required PASS "$d_dur" "$DET_N tests reproduced across 2 processes"
    ok "[determinism] PASS (${d_dur}s) -- $DET_N oracle tests identical across 2 independent processes"
  fi
fi

if should_run doc; then
  run_gate doc required "Public docs build with no warnings" \
    "RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --workspace"
fi

# Optional: the real wgpu GPU path (Vulkan). The Jetson's CUDA cuBLAS path is
# archived and NOT part of today's build (CudaBackend returns Unavailable).
if should_run gpu; then
  if [[ "$VULKAN" == "yes" ]]; then
    run_gate gpu optional "wgpu GEMM vs CPU oracle on the Jetson's Vulkan adapter" \
      "cargo test -p scirust-gpu --features wgpu"
  else
    skip_gate gpu optional "no Vulkan adapter found (install the Jetson Vulkan ICD)"
  fi
fi

# Optional: license & advisory audit.
if should_run deny; then
  if cargo deny --version >/dev/null 2>&1; then
    run_gate deny optional "License & advisory audit (cargo-deny)" "cargo deny check"
  else
    skip_gate deny optional "cargo-deny not installed (cargo install cargo-deny)"
  fi
fi

# =============================================================================
#  FINAL REPORT
# =============================================================================
echo; hr
printf '%s  JETSON ACCEPTANCE REPORT%s   %s\n' "$C_BOLD" "$C_RESET" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
hr
req_fail=0; req_skip=0; n_pass=0; n_total=0
printf '  %-13s %-8s %-8s %s\n' "GATE" "REQUIRED" "STATUS" "DETAIL"
printf '  %-13s %-8s %-8s %s\n' "-------------" "--------" "------" "------"
for i in "${!GATE_ID[@]}"; do
  id="${GATE_ID[$i]}"; req="${GATE_REQ[$i]}"; st="${GATE_STATUS[$i]}"; dur="${GATE_DUR[$i]}"; note="${GATE_NOTE[$i]}"
  n_total=$((n_total+1))
  case "$st" in
    PASS) col="$C_GRN"; n_pass=$((n_pass+1)) ;;
    FAIL) col="$C_RED"; [[ "$req" == required ]] && req_fail=$((req_fail+1)) ;;
    SKIP) col="$C_YEL"; [[ "$req" == required ]] && req_skip=$((req_skip+1)) ;;
    *)    col="$C_RESET" ;;
  esac
  detail="$note"; [[ "$st" == PASS ]] && detail="${dur}s${note:+ -- $note}"
  printf '  %-13s %-8s %s%-8s%s %s\n' "$id" "$req" "$col" "$st" "$C_RESET" "$detail"
done
echo
[[ -n "${T_PASS:-}" ]] && info "Tests     : ${T_PASS} passed / ${T_FAIL:-0} failed / ${T_IGN:-0} ignored (${T_GROUPS:-0} groups)"
[[ -n "${DET_N:-}" ]] && info "Determinism: ${DET_N} oracles reproduced across 2 processes"
info "Evidence  : $LOG_DIR"
{
  echo "commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?')"
  echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  for i in "${!GATE_ID[@]}"; do
    echo "gate.${GATE_ID[$i]}=${GATE_STATUS[$i]} (${GATE_REQ[$i]}, ${GATE_DUR[$i]}s)${GATE_NOTE[$i]:+ -- ${GATE_NOTE[$i]}}"
  done
} >>"$SUMMARY"
echo
if [[ $req_fail -eq 0 && $req_skip -eq 0 ]]; then
  printf '%s%s  VERDICT: PASS  %s -- all required gates green on this Jetson (%d/%d).\n' "$C_BOLD" "$C_GRN" "$C_RESET" "$n_pass" "$n_total"
  echo "verdict=PASS" >>"$SUMMARY"; exit 0
elif [[ $req_fail -eq 0 ]]; then
  printf '%s%s  VERDICT: PASS (with gaps)  %s -- %d required gate(s) skipped for missing prerequisites.\n' "$C_BOLD" "$C_YEL" "$C_RESET" "$req_skip"
  echo "verdict=PASS_WITH_GAPS" >>"$SUMMARY"; exit 0
else
  printf '%s%s  VERDICT: FAIL  %s -- %d required gate(s) failed. See the evidence bundle.\n' "$C_BOLD" "$C_RED" "$C_RESET" "$req_fail"
  echo "verdict=FAIL" >>"$SUMMARY"; exit 1
fi
