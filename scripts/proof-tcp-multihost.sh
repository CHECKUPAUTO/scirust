#!/usr/bin/env bash
# =============================================================================
# SciRust — preuve de l'all-reduce à arbre fixe sur TCP RÉEL, entre MACHINES
# PHYSIQUES SÉPARÉES (pas seulement 127.0.0.1) — scirust-core/src/tree_allreduce.rs
# -----------------------------------------------------------------------------
# Complète scripts/proof-portable-f32.sh (calcul portable) sur l'axe transport
# réseau : chaque rang génère sa contribution localement (Philox, seed+rang :
# reproductible sur toute machine), participe au protocole TCP réel, et le
# rang 0 recalcule la référence EN-PROCESS pour comparer bit à bit — la
# preuve est AUTO-VÉRIFIANTE (aucune empreinte à récolter à l'avance). Si les
# rangs tournent sur des architectures différentes, un `verdict=PASS` prouve
# le bit-exact inter-architectures à travers un vrai réseau (pas seulement
# en mémoire partagée / boucle locale).
#
# Ce script lance UN rang (un appel = un processus = un rang). Pour une
# preuve à 2 machines, exécuter les commandes correspondantes sur CHAQUE
# machine — voir l'EXEMPLE ci-dessous.
#
# USAGE
#   proof-tcp-multihost.sh --rank R --n N --seed SEED
#       [--my-addr HOST:PORT] [--parent-addr HOST:PORT]
#       [--dim D] [--combine fixed|exact] [--repo PATH]
#
#   --rank R          indice du rang (0 = racine)
#   --n N             nombre total de rangs
#   --seed SEED       graine partagée par TOUS les rangs (même valeur partout)
#   --my-addr H:P     requis SSI ce rang a des enfants (2R+1 < N) : adresse de
#                     *bind* du listener (ex. 0.0.0.0:9000 pour écouter sur
#                     toutes les interfaces)
#   --parent-addr H:P requis SSI R > 0 : adresse EXTERNE joignable du parent
#                     (ex. 192.168.1.10:9000 — l'IP réelle de la machine du
#                     rang parent, PAS 0.0.0.0)
#   --dim D           taille du vecteur par rang (défaut 8)
#   --combine MODE    fixed (somme f32 ordre d'arbre) ou exact (Kulisch,
#                     indépendant même de la topologie) — défaut exact
#   --repo PATH       chemin du workspace scirust (défaut : parent du script)
#
# EXEMPLE — 3 rangs sur 2 machines (arbre : rang 0 racine, enfants 1 et 2) :
#
#   Machine A (IP 192.168.1.10) — rang 0 (racine, écoute sur :9000) :
#     scripts/proof-tcp-multihost.sh --rank 0 --n 3 --seed 42 \
#         --my-addr 0.0.0.0:9000
#
#   Machine A — rang 1 (feuille, second terminal, même machine) :
#     scripts/proof-tcp-multihost.sh --rank 1 --n 3 --seed 42 \
#         --parent-addr 192.168.1.10:9000
#
#   Machine B (Jetson, IP 192.168.1.20) — rang 2 (feuille) :
#     scripts/proof-tcp-multihost.sh --rank 2 --n 3 --seed 42 \
#         --parent-addr 192.168.1.10:9000
#
# Démarrer la racine (rang 0) EN PREMIER : les rangs non-racine réessaient la
# connexion pendant ~30 s si le parent n'écoute pas encore, mais un ordre de
# démarrage racine-d'abord évite l'attente. `--seed` DOIT être identique sur
# les trois commandes ; les adresses `--my-addr`/`--parent-addr` doivent
# désigner un port réellement joignable entre les deux machines (pare-feu,
# NAT — utiliser l'IP du réseau local, pas 127.0.0.1, entre deux machines).
#
# Le PASS/FAIL est imprimé UNIQUEMENT par le rang 0 (les autres rangs ne font
# qu'envoyer leur contribution) : c'est lui qui recalcule la référence et
# compare. Reporter le verdict et les adresses/architectures utilisées dans
# LIVESTATE.md.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

echo "== build (release) =="
cargo build --release -p scirust-core --bin proof_tcp_multihost \
    --manifest-path "$REPO/Cargo.toml"

echo "== plate-forme =="
echo "uname=$(uname -a)"
command -v rustc >/dev/null && echo "rustc=$(rustc -V)"
echo "commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'hors dépôt git')"

echo "== rang =="
exec "$REPO/target/release/proof_tcp_multihost" "${ARGS[@]}"
