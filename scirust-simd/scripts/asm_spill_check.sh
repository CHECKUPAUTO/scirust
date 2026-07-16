#!/usr/bin/env bash
# scirust-simd/scripts/asm_spill_check.sh
#
# Régression assembleur AArch64 pour les noyaux hypercomplexes.
#
# Compile les sondes autonomes (feature `asm-probe`,
# src/hypercomplex/asm_probe.rs) vers `aarch64-unknown-linux-gnu` en n'émettant
# QUE l'assembleur (`--emit asm`, pas d'édition de liens — aucun linker ARM
# requis sur un hôte x86_64), puis compte, dans chaque boucle chaude, les
# chargements/stockages de registres VECTORIELS (q/v/d) référençant `sp` —
# c.-à-d. les spills/reloads de pile.
#
# La pression registre — donc le nombre de spills — dépend du MODÈLE
# D'ORDONNANCEMENT du cœur cible. Le script balaie donc plusieurs `target-cpu`
# et affiche une matrice : les cœurs out-of-order visés (Neoverse N1/V1 =
# Graviton 2/3, Apple Silicon) doivent afficher 0 spill de boucle chaude pour
# le sédénion ; `generic` et les petits cœurs in-order (Cortex-A72) en gardent
# quelques-uns (pression intrinsèque des 16 produits de Hamilton).
#
# IMPORTANT : le `target-cpu` est passé via RUSTFLAGS (et non en argument
# `cargo rustc -- -C target-cpu`) car cargo n'inclut pas fiablement ce dernier
# dans son empreinte de recompilation → sinon des .s périmés seraient réanalysés.
#
# Usage :
#   scirust-simd/scripts/asm_spill_check.sh            # matrice multi-CPU
#   TARGET_CPUS="neoverse-n1 apple-m1" scirust-simd/scripts/asm_spill_check.sh
#   DETAIL_CPU=neoverse-n1 scirust-simd/scripts/asm_spill_check.sh  # + dump détaillé
#
# Variables d'environnement :
#   TOOLCHAIN    (défaut nightly-2026-07-02)
#   TARGET_CPUS  (défaut : generic neoverse-n1 neoverse-v1 apple-m1 cortex-a72)
#   DETAIL_CPU   (défaut : neoverse-n1) — CPU pour le dump détaillé par kernel
#
# Prérequis : rustup target add aarch64-unknown-linux-gnu --toolchain <TOOLCHAIN>
#
# Code de sortie : 0 si l'analyse aboutit (comptage informatif, pas un gate dur).

set -euo pipefail

TARGET="aarch64-unknown-linux-gnu"
TOOLCHAIN="${TOOLCHAIN:-nightly-2026-07-02}"
TARGET_CPUS="${TARGET_CPUS:-generic neoverse-n1 neoverse-v1 apple-m1 cortex-a72}"
DETAIL_CPU="${DETAIL_CPU:-neoverse-n1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$CRATE_DIR")"
cd "$WORKSPACE_DIR"

TARGET_DIR="$(cargo "+$TOOLCHAIN" metadata --format-version 1 --no-deps \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"

DEPS_DIR="$TARGET_DIR/$TARGET/release/deps"
PROBE_SRC="$CRATE_DIR/src/hypercomplex/asm_probe.rs"

# Purge les .s accumulés : sinon un cache-hit cargo (aucun .s réémis) laisserait
# `ls -t` renvoyer un .s périmé d'un AUTRE target-cpu.
rm -f "$DEPS_DIR/scirust_simd-"*.s 2>/dev/null || true

# Compile les sondes pour un target-cpu et renvoie le chemin du .s émis.
# On `touch` la source des sondes pour FORCER la réémission de l'assembleur à
# chaque appel (sinon un cache-hit ne réécrirait pas le .s) ; le fichier le plus
# récent est alors sans ambiguïté celui du target-cpu courant.
emit_asm() {
    local cpu="$1"
    touch "$PROBE_SRC"
    # --emit asm remplace l'émission par défaut (link) : rustc s'arrête à
    # l'assembleur, aucun linker AArch64 requis. codegen-units=1 → un seul .s.
    # target-cpu via RUSTFLAGS (empreinte de recompilation fiable).
    RUSTFLAGS="-C target-cpu=$cpu" cargo "+$TOOLCHAIN" rustc -p scirust-simd \
        --features asm-probe \
        --target "$TARGET" --release --quiet \
        -- -C codegen-units=1 --emit asm >/dev/null 2>&1
    ls -t "$DEPS_DIR/scirust_simd-"*.s 2>/dev/null | head -1
}

echo "[asm_spill_check] cible=$TARGET toolchain=$TOOLCHAIN"
echo "[asm_spill_check] target-cpus: $TARGET_CPUS"
echo
printf "%-14s | %-9s | %-9s | %-9s   (spills de boucle chaude, vectoriels)\n" \
    "target-cpu" "quat_mul" "oct_mul" "sed_mul"
printf -- "-------------------------------------------------------------\n"
for cpu in $TARGET_CPUS; do
    asm="$(emit_asm "$cpu")"
    if [[ -z "$asm" || ! -f "$asm" ]]; then
        printf "%-14s | %s\n" "$cpu" "ERREUR: pas de .s émis"
        continue
    fi
    line="$(python3 "$SCRIPT_DIR/asm_spill_check.py" --matrix-row "$asm")"
    printf "%-14s | %s\n" "$cpu" "$line"
done

echo
echo "[asm_spill_check] dump détaillé (target-cpu=$DETAIL_CPU) :"
echo
asm="$(emit_asm "$DETAIL_CPU")"
python3 "$SCRIPT_DIR/asm_spill_check.py" "$asm"
