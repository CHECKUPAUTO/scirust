#!/usr/bin/env bash
# Fetches the phase-728 industrial evaluation datasets into data/industrial/
# (git-ignored), verifying every archive and extracted file against pinned
# SHA-256 checksums. This script is the ONLY sanctioned download path:
# nothing in the library, tests, or evaluation binary ever touches the
# network, and `cargo test` never invokes this script. Absent data makes
# integration tests skip loudly, not download silently.
#
# Datasets, sources and licenses (full details: scirust-srcc-bench/DATASETS.md):
#
# - NASA C-MAPSS turbofan degradation (run-to-failure), FD001 and FD003
#   subsets. Source: NASA Prognostics Center of Excellence data repository.
#   License: U.S. Government work, public domain. Cite: A. Saxena,
#   K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for
#   Aircraft Engine Run-to-Failure Simulation", PHM 2008. FD001 and FD003
#   are distinct fault-mode subsets of the same simulation and serve as the
#   two PdM workloads (the replication pair for the SRCC questions).
#
# - SECOM semiconductor manufacturing (real process/yield data).
#   Source: UCI Machine Learning Repository, dataset 179.
#   License: CC BY 4.0. Cite: M. McCann, A. Johnston, "SECOM", UCI ML
#   Repository, 2008.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/data/industrial"
WORK="$TARGET/.download"

CMAPSS_URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CMAPSS_ZIP_SHA="c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
SECOM_URL="https://archive.ics.uci.edu/static/public/179/secom.zip"
SECOM_ZIP_SHA="eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"

# Extracted-file checksums (the identities the evaluation binary verifies).
TRAIN_FD001_SHA="963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
TEST_FD001_SHA="3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
RUL_FD001_SHA="a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
TRAIN_FD003_SHA="2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2"
TEST_FD003_SHA="299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790"
RUL_FD003_SHA="df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb"
SECOM_DATA_SHA="20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c"
SECOM_LABELS_SHA="126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03"

check() {
    local file="$1" expected="$2"
    echo "$expected  $file" | sha256sum -c - >/dev/null
    echo "verified $file"
}

mkdir -p "$WORK" "$TARGET/cmapss" "$TARGET/secom"

echo "== C-MAPSS (NASA, public domain) =="
curl -sS --fail -o "$WORK/cmapss.zip" "$CMAPSS_URL"
check "$WORK/cmapss.zip" "$CMAPSS_ZIP_SHA"
unzip -o -q "$WORK/cmapss.zip" -d "$WORK/cmapss_outer"
unzip -o -q "$WORK/cmapss_outer/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" \
    -d "$WORK/cmapss_inner"
cp "$WORK/cmapss_inner/train_FD001.txt" \
   "$WORK/cmapss_inner/test_FD001.txt" \
   "$WORK/cmapss_inner/RUL_FD001.txt" \
   "$WORK/cmapss_inner/train_FD003.txt" \
   "$WORK/cmapss_inner/test_FD003.txt" \
   "$WORK/cmapss_inner/RUL_FD003.txt" \
   "$TARGET/cmapss/"
check "$TARGET/cmapss/train_FD001.txt" "$TRAIN_FD001_SHA"
check "$TARGET/cmapss/test_FD001.txt" "$TEST_FD001_SHA"
check "$TARGET/cmapss/RUL_FD001.txt" "$RUL_FD001_SHA"
check "$TARGET/cmapss/train_FD003.txt" "$TRAIN_FD003_SHA"
check "$TARGET/cmapss/test_FD003.txt" "$TEST_FD003_SHA"
check "$TARGET/cmapss/RUL_FD003.txt" "$RUL_FD003_SHA"

echo "== SECOM (UCI 179, CC BY 4.0) =="
curl -sS --fail -o "$WORK/secom.zip" "$SECOM_URL"
check "$WORK/secom.zip" "$SECOM_ZIP_SHA"
unzip -o -q "$WORK/secom.zip" -d "$WORK/secom_extract"
cp "$WORK/secom_extract/secom.data" "$WORK/secom_extract/secom_labels.data" "$TARGET/secom/"
check "$TARGET/secom/secom.data" "$SECOM_DATA_SHA"
check "$TARGET/secom/secom_labels.data" "$SECOM_LABELS_SHA"

rm -rf "$WORK"

echo "done: datasets in $TARGET"
