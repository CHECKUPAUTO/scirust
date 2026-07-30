#!/usr/bin/env sh
# One-shot build + install for CCOS.
#
# Builds the release binary with the deployment features, installs it to PREFIX
# (default /usr/local/bin), and runs `ccos doctor`. Override with env vars:
#
#   PREFIX=/opt/bin CCOS_FEATURES=llm,license,learned-embed sh scripts/install.sh
#
# The `ccos` binary REQUIRES the `llm` feature, so `llm` must stay in CCOS_FEATURES.
set -eu

PREFIX="${PREFIX:-/usr/local/bin}"
FEATURES="${CCOS_FEATURES:-llm,license}"
BIN="target/release/ccos"

echo "==> building ccos (release, --features ${FEATURES})"
cargo build --release --features "${FEATURES}"

if [ ! -x "${BIN}" ]; then
  echo "error: build produced no binary at ${BIN}" >&2
  echo "       the 'ccos' bin requires the 'llm' feature — keep 'llm' in CCOS_FEATURES." >&2
  exit 1
fi

echo "==> installing to ${PREFIX}/ccos"
if [ -w "${PREFIX}" ]; then
  install -m 755 "${BIN}" "${PREFIX}/ccos"
else
  sudo install -m 755 "${BIN}" "${PREFIX}/ccos"
fi

echo "==> ccos doctor"
"${PREFIX}/ccos" doctor
