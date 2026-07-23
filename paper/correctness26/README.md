# Soumission Correctness — ARCHIVÉ (non soumis en 2026)

> **Statut (2026-07-11, décision utilisateur)** : soumission reportée — le
> draft n'a pas été soumis à Correctness '26 et est archivé en l'état,
> prêt pour une édition future (CFP analogue attendu vers juin-juillet de
> chaque année). **Chiffres arrêtés aux commits `0c2f1bf`/`014795f` du
> 2026-07-10** (évidence brute : `docs/evidence/`) ; avant toute soumission
> future, re-mesurer O1 (x86 + Jetson), re-vérifier la bibliographie, et
> reprendre les TODO ci-dessous. Voir `paper/PAPER_PLAN.md` §7.

Draft du papier « Determinism as Certification Evidence: A Fully Auditable
Rust Stack for Bit-Reproducible Training and Quantized Edge Inference ».

- **Deadline visée à l'époque** : 23 juillet 2026 (notification :
  1er septembre 2026) — non tenue par décision, pas par retard.
- **Format** : ACM `acmart` option `sigconf` ; papier régulier = 7 à 8 pages
  de contenu (tout compris **sauf** les références) ; repli papier court =
  4 pages. CFP : <https://correctness-workshop.github.io/2026/>.
- **Sources** : `main.tex` + `references.bib` (références vérifiées le
  2026-07-10 — ne rien ajouter sans vérification).

## Compiler

Aucune chaîne TeX n'est requise dans le dépôt ; deux options :

```bash
# Option 1 — machine locale avec TeX Live :
cd paper/correctness26 && latexmk -pdf main.tex

# Option 2 — Overleaf : importer main.tex + references.bib tels quels
# (la classe acmart est fournie par Overleaf).
```

## Contenu et discipline

Chaque claim du papier est adossée à la table claims → évidence de
`paper/PAPER_PLAN.md` (tests CI T1-T4/R1-R4/Q1-Q3/S1-S3/A1, protocoles
O1-O2). Les chiffres de la section « cost of determinism » proviennent des
runs consignés dans `LIVESTATE.md` (x86-64 4 cœurs + Jetson AGX Thor 14
cœurs, MAXN, horloges épinglées, 3 runs × 30 reps par plateforme).

## TODO avant soumission (marqués `TODO` dans main.tex)

1. Affiliation exacte de l'auteur.
2. Lien d'artefact public pour les relecteurs (ou déclaration d'accès).
3. Vérifier au CFP si la soumission doit être anonyme
   (`[sigconf,review,anonymous]`).
4. Relecture de longueur après compilation : viser ≤ 8 pages hors
   références (couper d'abord dans §2 et §7 si dépassement).
