#!/usr/bin/env bash
# =============================================================================
# SciRust — preuve cross-platform de la voie f32 portable (portable_f32)
# -----------------------------------------------------------------------------
# À exécuter SUR CHAQUE plate-forme cible (x86_64 Debian, Jetson/aarch64, …).
# Rejoue le contrat de portabilité de `scirust-core/src/portable_f32.rs` via le
# binaire `proof_portable_f32` : goldens bit-à-bit, empreintes FNV-1a des
# balayages de l'espace des bits f32 (contrat pas 65 537, dense pas 257, et
# exhaustif pas 1 avec --full), softmax et GEMM composites. Les empreintes
# attendues sont COMMISES dans le dépôt : un `verdict=PASS` sur une machine
# prouve qu'elle reproduit bit à bit les résultats calculés sur x86-64.
#
# DÉMARRAGE RAPIDE (sur la machine cible)
#   git clone https://github.com/Memorithm/scirust.git
#   cd scirust                      # (ou git fetch && git checkout <branche>)
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   scripts/proof-portable-f32.sh --full
#
# USAGE
#   proof-portable-f32.sh [--repo PATH] [--full] [-h|--help]
#
#   --repo PATH   chemin du workspace scirust (défaut : parent du script)
#   --full        ajoute le balayage EXHAUSTIF (2³² entrées par fonction,
#                 quelques minutes en release)
#
# SORTIE : bundle d'évidence horodaté  proof-portable-f32-<UTC>/
#   platform.txt      machine, OS, rustc, commit
#   report.txt        sortie brute du binaire (lignes canoniques + contexte #)
#   canonical.sha256  SHA-256 des lignes canoniques — doit être IDENTIQUE
#                     sur toutes les plates-formes (pour un même mode --full)
# Le bundle reste sur la machine (`.gitignore`d) ; le verdict et le SHA sont
# reportés dans LIVESTATE.md.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FULL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --full) FULL="--full"; shift ;;
        -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "argument inconnu : $1 (voir --help)" >&2; exit 2 ;;
    esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/proof-portable-f32-$STAMP"
mkdir -p "$OUT"

{
    echo "date_utc=$STAMP"
    echo "uname=$(uname -a)"
    command -v rustc >/dev/null && echo "rustc=$(rustc -V)"
    [[ -r /etc/os-release ]] && echo "os=$(. /etc/os-release && echo "$PRETTY_NAME")"
    echo "commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'hors dépôt git')"
    echo "mode=${FULL:-standard}"
} > "$OUT/platform.txt"

echo "== build (release) =="
cargo build --release -p scirust-core \
    --bin proof_portable_f32 --bin proof_portable_training \
    --manifest-path "$REPO/Cargo.toml"

echo "== preuve : fonctions portables =="
set +e
"$REPO/target/release/proof_portable_f32" $FULL | tee "$OUT/report.txt"
RC=${PIPESTATUS[0]}
set -e

echo "== preuve : entraînement portable (MLP + Adam + CE, contrat poids finaux) =="
set +e
"$REPO/target/release/proof_portable_training" | tee "$OUT/report-training.txt"
RC2=${PIPESTATUS[0]}
set -e
[[ $RC -eq 0 ]] && RC=$RC2

grep -v '^#' "$OUT/report.txt" "$OUT/report-training.txt" | sha256sum | tee "$OUT/canonical.sha256"
echo "bundle : $OUT"
if [[ $RC -eq 0 ]]; then
    echo "VERDICT : PASS — cette machine reproduit bit à bit le contrat commis."
else
    echo "VERDICT : FAIL — divergence bit à bit détectée (voir report.txt)." >&2
fi
exit "$RC"
