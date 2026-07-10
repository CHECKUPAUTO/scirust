#!/usr/bin/env bash
# =============================================================================
# SciRust — O1 sur Jetson (aarch64) : coût du déterminisme, volet ARM
# -----------------------------------------------------------------------------
# À exécuter SUR le Jetson. Complète la mesure x86 du banc
# `bench_reduction_overhead` (réduction en ordre de worker figé — le pattern
# de DataParallelTrainer — contre accumulation en ordre d'arrivée) avec le
# volet aarch64 exigé par paper/PAPER_PLAN.md (table claims → évidence,
# ligne O1), et exécute au passage les deux évidences natives ARM voisines :
#   - Q3 : noyau NEON int8 bit-exact contre la référence scalaire ;
#   - R4 : fingerprint du forward invariant au nombre de threads (1/2/4/8).
#
# Le script N'ALTÈRE JAMAIS la configuration du Jetson sans demande
# explicite : l'épinglage des horloges (nvpmodel + jetson_clocks) n'est
# tenté qu'avec --pin-clocks, et l'état courant est de toute façon consigné
# dans le bundle d'évidence.
#
# DÉMARRAGE RAPIDE (sur le Jetson)
#   # 1. cloner le dépôt et se placer sur la branche de la PR #268
#   git clone https://github.com/Memorithm/scirust.git
#   cd scirust && git checkout claude/new-session-n8bf71
#   # 2. installer la toolchain Rust nightly si absente
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   rustup toolchain install nightly
#   # 3. lancer le banc (horloges épinglées pour un wall-clock stable)
#   sudo scripts/bench-o1-jetson.sh --pin-clocks
#
# USAGE
#   bench-o1-jetson.sh [--repo PATH] [--runs N] [--pin-clocks] [-h|--help]
#
#   --repo PATH    chemin du workspace scirust (défaut : parent du script)
#   --runs N       nombre d'exécutions du banc (défaut : 3 — stabilité
#                  run-to-run visible dans l'évidence)
#   --pin-clocks   épingle les horloges AVANT la mesure (nvpmodel -m 0 puis
#                  jetson_clocks ; requiert root). Sans ce drapeau le banc
#                  tourne dans le mode courant, qui est seulement consigné.
#
# SORTIE : bundle d'évidence horodaté  bench-o1-jetson-<UTC>/
#   platform.txt   modèle, noyau, cœurs, L4T, nvpmodel, rustc, commit
#   run-N.md       sortie brute du banc (tableau Markdown, empreintes)
#   tests.txt      résultats Q3 (NEON) et R4 (fingerprint multi-thread)
# Coller le tableau médian dans paper/PAPER_PLAN.md §4 (ligne O1, volet
# Jetson) et consigner une ligne dans LIVESTATE.md.
#
# CODE DE SORTIE : 0 = banc exécuté et tests natifs verts ; ≠ 0 sinon.
# =============================================================================

set -uo pipefail

REPO=""
RUNS=3
PIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)       REPO="${2:-}"; shift ;;
    --repo=*)     REPO="${1#*=}" ;;
    --runs)       RUNS="${2:-}"; shift ;;
    --runs=*)     RUNS="${1#*=}" ;;
    --pin-clocks) PIN=1 ;;
    -h|--help)    sed -n '2,46p' "$0"; exit 0 ;;
    *) echo "argument inconnu : $1 (voir --help)" >&2; exit 2 ;;
  esac
  shift
done

# ---- localisation du workspace ----------------------------------------------
if [[ -z "$REPO" ]]; then
  REPO="$(cd "$(dirname "$0")/.." && pwd)"
fi
if [[ ! -f "$REPO/Cargo.toml" ]]; then
  echo "workspace scirust introuvable : $REPO (utiliser --repo PATH)" >&2
  exit 2
fi
cd "$REPO"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "AVERTISSEMENT : machine $(uname -m), pas aarch64 — ce volet vise le Jetson." >&2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/bench-o1-jetson-$STAMP"
mkdir -p "$OUT"

# ---- épinglage d'horloges (explicite seulement) -------------------------------
if [[ "$PIN" == "1" ]]; then
  if [[ "$(id -u)" != "0" ]]; then
    echo "--pin-clocks requiert root (sudo) — abandon plutôt qu'une mesure ambiguë." >&2
    exit 2
  fi
  command -v nvpmodel >/dev/null 2>&1 && nvpmodel -m 0 || echo "nvpmodel absent — ignoré" >&2
  command -v jetson_clocks >/dev/null 2>&1 && jetson_clocks || echo "jetson_clocks absent — ignoré" >&2
fi

# ---- rapport plateforme (consigné, jamais supposé) ----------------------------
{
  echo "== SciRust O1 — volet Jetson =="
  echo "date-utc : $STAMP"
  echo "commit   : $(git rev-parse HEAD 2>/dev/null || echo inconnu)"
  echo "branche  : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo inconnue)"
  echo "modele   : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo inconnu)"
  echo "noyau    : $(uname -a)"
  echo "coeurs   : $(nproc)"
  echo "memoire  : $(grep MemTotal /proc/meminfo 2>/dev/null || echo inconnue)"
  echo "l4t      : $(head -1 /etc/nv_tegra_release 2>/dev/null || echo absent)"
  echo "nvpmodel : $(nvpmodel -q 2>/dev/null | tr '\n' ' ' || echo absent)"
  echo "rustc    : $(rustc --version 2>/dev/null || echo absent)"
  echo "pin      : $PIN (1 = nvpmodel -m 0 + jetson_clocks appliqués par ce script)"
} | tee "$OUT/platform.txt"

# ---- build release -------------------------------------------------------------
echo "== build (release) =="
if ! cargo build --release -p scirust-core --bin bench_reduction_overhead; then
  echo "échec de build — voir la sortie ci-dessus." >&2
  exit 1
fi

# ---- banc O1 : RUNS exécutions ---------------------------------------------------
status=0
for i in $(seq 1 "$RUNS"); do
  echo "== banc O1 — run $i/$RUNS =="
  if ! ./target/release/bench_reduction_overhead | tee "$OUT/run-$i.md"; then
    echo "run $i : échec" >&2
    status=1
  fi
done

# ---- évidences natives ARM voisines (Q3 NEON, R4 fingerprint) -------------------
echo "== tests natifs ARM (Q3 + R4) =="
{
  echo "-- Q3 : NEON int8 bit-exact (scirust-core) --"
  cargo test --release -p scirust-core neon_matches_scalar_bit_exact 2>&1 | tail -4
  echo "-- R4 : fingerprint invariant aux threads (scirust-runtime) --"
  cargo test --release -p scirust-runtime --test fingerprint_thread_invariance 2>&1 | tail -4
} | tee "$OUT/tests.txt"
# Gate strict sur ARM seulement : Q3 est cfg(aarch64) (0 test sur x86, où ce
# script ne sert qu'au test de fumée) ; sur Jetson les DEUX doivent passer.
if [[ "$(uname -m)" == "aarch64" ]]; then
  [[ "$(grep -c "test result: ok. 1 passed" "$OUT/tests.txt")" -ge 2 ]] || status=1
else
  echo "(hors aarch64 : résultats des tests natifs informatifs seulement)"
fi

echo
echo "Bundle d'évidence : $OUT"
echo "À reporter : tableau médian de run-*.md → paper/PAPER_PLAN.md §4 (O1,"
echo "volet Jetson) + une ligne LIVESTATE.md (date, commit, modèle, chiffres)."
exit "$status"
