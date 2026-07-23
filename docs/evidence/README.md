# Évidence brute archivée — chantier « déterminisme comme évidence » (2026-07-10)

Pièces de provenance des chiffres cités par `docs/DEAD_GUARDS_STUDY.md`,
`paper/PAPER_PLAN.md` (table claims → évidence) et le draft
`paper/correctness26/main.tex`. Archivées ici le 2026-07-11 lors de la
clôture du chantier (soumission Correctness reportée — pas en 2026).

## `dead-guards/` — campagne de minage (Lot 2, verdict NO-GO)

Les 22 rapports Markdown par dépôt, produits par
`epsilon-audit --mine <dépôt> --out <rapport>` le 2026-07-10, **tels quels** :
chacun est scellé par son `Report-SHA256` (hachage du corps — toute
altération est détectable) et reproductible bit-à-bit sur un arbre source
identique. `SHAS.txt` liste, pour chaque dépôt : nom, SHA de commit cloné
(`--depth 1`), URL, et sous-répertoires du clone sparse le cas échéant.
Synthèse et revue manuelle des candidats : `docs/DEAD_GUARDS_STUDY.md`.

Provenance : générés dans le conteneur de session (x86-64) au commit
`ecf575b3` de l'outil ; copiés ici sans modification (les scellés SHA-256
en font foi).

## `o1-bench/` — banc « coût du déterminisme » (protocole O1)

Sorties brutes de `bench_reduction_overhead`
(`scirust-core/src/bin/bench_reduction_overhead.rs`) :

- `x86-20260710.md` — deux exécutions sur l'hôte x86-64 du conteneur de
  session (4 cœurs). Provenance : sortie de terminal de la session,
  archivée par l'agent qui l'a exécutée.
- `jetson-20260710T094509Z.md` et `jetson-20260710T114542Z.md` — les deux
  protocoles complets sur **NVIDIA Jetson AGX Thor Developer Kit**
  (14 cœurs, L4T R38.4.0, MAXN), exécutés par l'opérateur via
  `scripts/bench-o1-jetson.sh` aux commits `0c2f1bf` puis `014795f`.
  Provenance : **retranscription de la sortie de terminal collée par
  l'opérateur en session** — les bundles originaux
  (`bench-o1-jetson-<UTC>/`) restent sur la machine Jetson (`~/scirust/
  scirust/`), git-ignorés par conception ; en cas de doute, ils font foi.

Résultat clé re-vérifiable dans ces fichiers : les 4 empreintes de la
réduction en ordre figé (`0x60daf62cf2cb2c29`, `0x9bf7c3f3e9b18898`,
`0xd5b8e15fc7c028e6`, `0x7e99a9d050da4d55` à 1/2/4/8 threads) sont
**bit-identiques entre x86-64 et aarch64**, sur des runs indépendants.
