# Travaux connexes

> Section rédigée pour être citable telle quelle dans le paper (Lot 3).
> Base bibliographique volontairement restreinte aux références vérifiées ;
> aucune référence inventée. L'étude empirique « dead guards »
> (`docs/DEAD_GUARDS_STUDY.md`) ayant conclu NO-GO, cette section ne comporte
> pas de sous-section « prévalence mesurée de la classe de bug ».

## 1. Reproductibilité flottante classique

Les pièges de l'arithmétique flottante sont documentés depuis longtemps.
Goldberg (1991, *What Every Computer Scientist Should Know About
Floating-Point Arithmetic*) établit le socle : l'addition flottante n'est pas
associative, l'arrondi dépend du format intermédiaire, et deux évaluations
mathématiquement équivalentes peuvent différer bit à bit. Monniaux (2008,
*The pitfalls of verifying floating-point computations*) montre que ces
écarts ne sont pas seulement algébriques mais **matériels et
compilatoires** : registres x87 à 80 bits, contraction FMA, modes
flush-to-zero (FTZ/DAZ), options fast-math — le *même* code source produit
des résultats différents selon la plateforme et le compilateur. Ces deux
références fondent notre position : la reproductibilité bit-à-bit n'est
jamais un acquis par défaut, c'est une propriété qu'il faut **construire puis
prouver par exécution**.

Côté construction, ReproBLAS (Demmel & Nguyen) est la fondation classique :
des sommations **reproductibles indépendantes de l'ordre** des opérandes,
qui rendent le résultat d'une réduction insensible à l'ordonnancement
parallèle. SciRust adopte l'autre voie possible vers le même invariant :
plutôt que de rendre la somme insensible à l'ordre, **figer l'ordre**
(réduction séquentielle en ordre de worker fixe, orthogonalisations
séquentielles, budgets d'itération fixes). Les deux approches livrent un
résultat bit-identique face au parallélisme ; la nôtre est plus simple à
auditer — le prix est un point de séquentialisation, mesuré plutôt que nié.

## 2. Déterminisme pour l'apprentissage profond

Les frameworks établis traitent le déterminisme comme un **mode
best-effort**. `torch.use_deterministic_algorithms` (PyTorch) force des
noyaux déterministes *à configuration fixée* : même machine, même version,
même nombre de threads. La garantie est run-to-run ; elle n'est **ni
bit-identique entre nombres de threads différents, ni entre plateformes**,
et certaines opérations n'ont simplement pas de variante déterministe.
EasyScale (arXiv:2208.14228) étend la portée au distribué élastique :
entraînement **bit-identique** sous variation du nombre de GPU hétérogènes,
en préservant l'état par worker logique et en figeant l'ordre effectif des
réductions — la preuve que l'invariance au degré de parallélisme est
atteignable à l'échelle, au prix d'une ingénierie dédiée.

RepDL (Microsoft Research, 2025, arXiv:2510.09180) est, à notre
connaissance, le travail le plus fort sur l'axe portabilité : entraînement et
inférence **bit-à-bit reproductibles entre plateformes** (CPU/GPU
différents), obtenus par (a) arrondi correct des opérations, dans la lignée
des bibliothèques à la MPFR/RLIBM, et (b) invariance d'ordre (sommations
séquentielles figées, graphes fixés). Le positionnement respectif mérite
d'être énoncé sans détour. **Sur l'axe cross-platform float32, RepDL est plus
fort que la voie f32 *sanitized* de SciRust**, qui n'est déterministe
qu'intra-architecture. Réciproquement, RepDL est une **surcouche d'un runtime
PyTorch** — un TCB C++/Python de plusieurs millions de lignes hors du
périmètre d'audit — limitée à un sous-ensemble d'opérations float32, sans
basse précision (bf16/int8 explicitement hors périmètre), et son rapport
technique ne comporte pas de section d'évaluation (ni benchmark, ni overhead
mesuré). SciRust occupe le créneau complémentaire : une pile **100 % Rust
auditable de bout en bout, sans FFI dans le chemin de calcul**, où le
déterminisme n'est pas seulement une propriété d'exécution mais une **pièce
d'évidence** — fingerprints d'inférence 64 bits, journaux hash-chaînés,
reconstruction par manifeste, chaque claim adossée à un test CI — avec un
pipeline int8 entièrement entier (bit-exact cross-platform par construction,
noyau NEON validé sur ARM embarqué) que l'approche « arrondi correct f32 »
ne couvre pas. Les deux travaux sont ainsi moins concurrents
qu'orthogonaux : RepDL renforce le *noyau numérique* d'un écosystème
existant ; SciRust reconstruit la *chaîne de confiance* entière au-dessus
d'un noyau volontairement plus simple.

## 3. Divergences GPU inter-constructeurs : pourquoi une voie « sanitized »

La comparaison systématique NVIDIA/AMD (arXiv:2410.09172) documente des
différences numériques entre GPU de constructeurs différents exécutant le
même calcul, aggravées par le fait que les **exceptions flottantes ne sont
pas signalées** sur GPU : underflow, arrondis divergents et comportements
sous-normaux passent silencieusement. C'est précisément le régime que la
voie 3 de SciRust neutralise *à l'intérieur d'une architecture* :
`sanitize_f32` écrase tout sous-normal (seuil = `f32::MIN_POSITIVE`,
aligné par test sur la constante σ de `scirust-sigma`), retirant du chemin
de calcul la classe de valeurs dont le traitement diffère le plus entre
matériels et modes (FTZ/DAZ des drivers). La campagne de minage de
`docs/DEAD_GUARDS_STUDY.md` confirme le réalisme de ce modèle de menace —
9 des 22 dépôts numériques majeurs scannés activent fast-math ou contrôlent
FTZ dans leurs builds — tout en constatant honnêtement (verdict NO-GO)
que la classe de bug « garde sous-normale » n'y a pas été observée : la
motivation de la voie sanitized est la *portabilité des comportements*, pas
une prévalence de bugs chez autrui. Pour les régimes où l'identité
inter-plateformes est exigée, SciRust fournit les voies entière et virgule
fixe, bit-exactes cross-platform par construction ; la convergence de la
voie f32 vers la portabilité totale (transcendantales correctement arrondies
en Rust pur) est un travail futur explicite, en dialogue avec l'approche
RepDL.

## Références

- D. Goldberg, *What Every Computer Scientist Should Know About
  Floating-Point Arithmetic*, ACM Computing Surveys, 1991.
- D. Monniaux, *The pitfalls of verifying floating-point computations*,
  ACM TOPLAS, 2008.
- J. Demmel, H. D. Nguyen, *ReproBLAS — Reproducible BLAS* (sommation
  reproductible indépendante de l'ordre).
- PyTorch, `torch.use_deterministic_algorithms` (documentation officielle) —
  déterminisme run-to-run à configuration fixée.
- EasyScale, arXiv:2208.14228 — entraînement élastique bit-identique sur
  GPU hétérogènes.
- RepDL (Microsoft Research), arXiv:2510.09180, 2025 —
  github.com/microsoft/RepDL.
- arXiv:2410.09172 — différences numériques NVIDIA vs AMD ; exceptions FP
  non signalées sur GPU.
