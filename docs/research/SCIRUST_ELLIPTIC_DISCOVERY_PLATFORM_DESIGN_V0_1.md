# Plateforme de découverte elliptique — conception v0.1

## Statut

- Type : document de conception et d’audit ; aucun code de production n’est ajouté par ce livrable.
- Dépôt : Memorithm/scirust.
- Branche de travail : `research/elliptic-discovery-platform`.
- Base distante auditée : `origin/master` à `3d6615c8e4784149d0b8b97e9f58631edf5e6f90` (2026-08-01).
- État initial de cette branche : propre après création depuis `origin/master`.

Cette plateforme a pour seul objet la recherche mathématique reproductible sur des courbes
elliptiques jouets, des instances générées localement et des jeux de recherche explicitement
autorisés. Elle ne doit jamais accepter, dériver, rechercher ou comparer une adresse Bitcoin,
une clé publique tierce, une cible blockchain réelle, ni une donnée qui en tient lieu.

Une relation testée est une hypothèse. Elle ne peut pas être décrite comme une découverte
nouvelle sans franchir les portes de validation définies dans « Protocole de falsification ».

## Méthode d’audit et conventions du workspace

L’audit a porté notamment sur :

- `scirust-hypercrypto` et sa documentation de recherche ;
- `scirust-symbolic` ;
- `scirust-neuro-symbolic` ;
- `scirust-solvers` ;
- `scirust-evo` ;
- `scirust-core` ;
- les voisins directement nécessaires à la décision : `scirust-modalg`, `scirust-sim` et
  `scirust-algogen`.

Les conventions observées à la racine sont : Rust édition 2021 pour la majorité des crates
existants, MSRV 1.89, résolveur Cargo v2, outil de CI `nightly-2026-07-02`, `rustfmt`
(maximum 100 colonnes) et Clippy avec `-D warnings`. Les commandes de référence sont
documentées dans [CONTRIBUTING.md](../../CONTRIBUTING.md) et dans
[.github/workflows/ci.yml](../../.github/workflows/ci.yml).

La future implémentation doit être Rust pur, sans FFI, et son propre crate doit porter
`#![forbid(unsafe_code)]`. Une dépendance ne sera ajoutée qu’après avoir vérifié qu’une
abstraction existante ne couvre pas le besoin.

## État Git initial observé

| Vérification | Commande exécutée | Résultat observé |
|---|---|---|
| Récupération de la branche distante | `git fetch origin --prune` | `origin/master` a avancé de `25f272a` à `3d6615c`. |
| Création isolée | `git worktree add -b research/elliptic-discovery-platform … origin/master` | Branche créée au commit `3d6615c`, avec suivi de `origin/master`. |
| État du worktree créé | `git status --short --branch` | `## research/elliptic-discovery-platform...origin/master`, sans fichier modifié. |
| Cohérence initiale du diff | `git diff --check` | Succès, aucune sortie. |

Les worktrees préexistants contenant des modifications non liées ont été préservés. Toutes les
modifications de ce travail se limitent au worktree isolé ci-dessus.

## Lecture de scirust-hypercrypto

Le crate [scirust-hypercrypto](../../scirust-hypercrypto/) est un harnais de falsification
expérimental pour une construction hypercomplexe de permutation à clés. Son README et
[sa spécification v0.1](../../scirust-hypercrypto/docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md)
délimitent explicitement une cible de recherche, non un primitive cryptographique de production.

Ses qualités réutilisables sont méthodologiques, non structurelles :

- analyse déterministe, algèbre exacte et contrôles négatifs ;
- couverture explicitement annotée `Exhaustive` ou `Sampled { count, seed }` dans
  [analysis/util.rs](../../scirust-hypercrypto/src/analysis/util.rs) ;
- verdicts et indicateurs de contrôle dans
  [analysis/battery.rs](../../scirust-hypercrypto/src/analysis/battery.rs) ;
- rapports canoniques et empreintes SHA-256 dans
  [analysis/report.rs](../../scirust-hypercrypto/src/analysis/report.rs) ;
- dépendance directe à `scirust-modalg` pour l’algèbre exacte.

Sa façade `algebra.rs` ré-exporte des entiers de mots modulaires, quaternions, octonions et
matrices modulaires propres à son domaine. Ajouter des courbes elliptiques à cette façade
mélangerait deux objets de recherche sans invariant partagé, ferait croire à une composante
cryptographique opérationnelle, et forcerait une dépendance inversée artificielle.

## Décision d’architecture

**Décision : créer ultérieurement un nouveau crate générique
`scirust-elliptic-discovery`; ne pas étendre `scirust-hypercrypto`.**

| Option | Avantage | Problème déterminant | Décision |
|---|---|---|---|
| Étendre `scirust-hypercrypto` | Réutilisation apparente des rapports de falsification | Domaine, API et objectif cryptographique expérimental incompatibles ; couplage trompeur | Écartée |
| Étendre `scirust-modalg` | Réutilisation maximale de l’arithmétique | `modalg` est une bibliothèque algébrique générique ; y placer l’orchestration d’expériences, le catalogue et les garde-fous créerait un mélange de couches | Écartée |
| Nouveau `scirust-elliptic-discovery` | Périmètre, sécurité et reproductibilité explicites ; dépendances minimes vers les fondations existantes | Nouveau manifeste et tests à maintenir | Retenue |

Le nouveau crate sera un consommateur fin de `scirust-modalg`, pas un remplacement. Il pourra
reprendre les idées de rapport de `scirust-hypercrypto`, sans import de son API métier.

Le manifeste proposé, à n’ajouter qu’en phase 1, est :

    [package]
    name = "scirust-elliptic-discovery"
    version = "0.1.0"
    edition = "2021"
    publish = false
    rust-version = "1.89"

    [dependencies]
    scirust-modalg = { path = "../scirust-modalg" }
    scirust-sim = { path = "../scirust-sim", default-features = false }

Aucune dépendance externe nouvelle n’est prévue. `scirust-sim` ne serait utilisé que pour son
générateur déterministe à état explicite ; aucun tirage flottant ne ferait partie d’un calcul
algébrique ou d’une décision de validité.

## Inventaire des composants réutilisables

| Besoin | Composant existant | Évaluation et emploi prévu |
|---|---|---|
| Entiers multi-précision exacts | `scirust-modalg::bigint::BigInt` | Entier signé arbitraire, opérations exactes, PGCD et conversion décimale. À réserver aux bornes ou certificats qui dépassent `u64`. |
| Théorie des nombres et module premier | `scirust-modalg::numtheory` | `is_prime` déterministe sur `u64`, `pow_mod`, `mulmod`, `inv_mod`, factorisation et diviseurs. Fondation à réutiliser pour \(\mathbb F_p\). |
| Corps finis et polynômes | `scirust-modalg::poly::Poly`, `extfield::ExtField` | Polynômes canoniques sur \(\mathrm{GF}(p)\), test d’irréductibilité, extensions et Frobenius exacts. Réutilisables pour une tentative de justification symbolique ; v0.1 des courbes reste sur \(\mathbb F_p\). |
| Calcul symbolique | `scirust-symbolic` | Expressions, différentiation et simplification présentes, mais constantes et évaluation en `f64`. Ne convient ni à la base algébrique ni à une preuve. Aucun lien direct prévu. |
| Raisonnement neuro-symbolique | `scirust-neuro-symbolic` | CSP/SAT/Datalog/e-graph disponibles, mais domaines entiers ou flottants, conteneurs hashés et absence de certificats de corps fini. Inspiration de conception seulement. |
| Solveurs | `scirust-solvers` | Racines polynomiales et interface unifiée majoritairement en `f64`. Inadapté aux égalités exactes dans \(\mathbb F_p\). Aucun lien direct prévu. |
| Recherche évolutionnaire | `scirust-evo` | Routines déterministes sous graine, mais génotypes et fitness en `f64`, avec `rand`/Rayon. Peut inspirer une phase exploratoire séparée, jamais valider une relation. |
| Reproductibilité | `scirust-sim::SplitMix64` | Générateur pur Rust public, graine explicite et vecteurs de référence. Candidat direct, limité à l’échantillonnage non algébrique. |
| Empreintes et rejouabilité | `scirust-algogen` et `scirust-hypercrypto` | Identité canonique, archive de campagne et rapports ordonnés sont des précédents de conception. Ne pas dépendre d’`algogen` car ses programmes sont flottants. |
| Réductions reproductibles | `scirust-core::reproducible` | Vise les réductions flottantes reproductibles ; hors besoin pour la base exacte. `scirust-core` contient aussi des zones `unsafe` et des backends FFI/BLAS ; aucun lien direct. |

Les observations importantes sont les suivantes :

1. `scirust-modalg` couvre déjà les primitives exactes utiles. Il serait risqué de réécrire un
   inverse modulaire, une factorisation ou un test de primalité.
2. Aucun composant audité ne fournit aujourd’hui une arithmétique complète de courbes
   elliptiques jouets sur \(\mathbb F_p\), avec énumération, ordre, catalogue de symétries et
   falsification certifiée.
3. Les composants symboliques, solveurs et évolutionnaires audités emploient des flottants.
   Ils ne doivent pas participer à l’établissement d’une égalité, d’un contre-exemple ou d’un
   statut de découverte.
4. `scirust-core` est une dépendance trop large pour ce sous-système : son inventaire inclut
   des chemins `unsafe`, des appels système et des backends optionnels. Cela violerait le
   périmètre Rust pur/aucune FFI de ce crate.

## Risques de duplication et parades

| Risque | Parade obligatoire |
|---|---|
| Réimplémenter \(\mathbb F_p\) au-dessus d’opérations naïves et diverger de `modalg` | Une mince façade de type peut déléguer les opérations à `numtheory`; tests croisés sur le domaine jouet. |
| Créer un second PRNG maison | Réutiliser `scirust-sim::SplitMix64` ou documenter pourquoi une API existante est insuffisante. Graine, algorithme et version font partie du rapport. |
| Dupliquer les rapports de HyperCrypto sans ordre canonique | Reprendre le principe : structures ordonnées, encodage de longueur explicite, empreinte du corpus et du programme de recherche. |
| Confondre recherche de motif et preuve | Séparer strictement générateur de candidats, falsificateur, classificateur et tentative de preuve. |
| Étendre implicitement le domaine à des clés réelles | Des types dédiés `ToyPrime`, `ToyCurve` et `LocalResearchCase`; aucune API de décodage SEC 1, adresse, clé publique ou RPC. |
| Dépendre de `scirust-core`, `symbolic`, `solvers` ou `evo` pour accélérer v0.1 | Interdit tant qu’un audit de sûreté, d’exactitude et de MSRV ne justifie pas une interface discrète et exacte. |

## Modèle algébrique minimal

La phase initiale se limite aux courbes courtes de Weierstrass sur un corps premier jouet :

\[
E_{a,b}/\mathbb F_p : y^2 = x^3 + ax + b,
\qquad p \text{ premier impair},
\qquad 4a^3 + 27b^2 \not\equiv 0 \pmod p.
\]

Le domaine v0.1 est borné à \(5 \le p \le 4093\). Cette limite rend l’énumération exacte
praticable, ne prétend pas représenter une courbe de production et ne doit pas être relevée
sans nouvelle analyse de coût et de protocole.

Les futurs types publics doivent être intentionnellement étroits :

- `ToyPrime` : premier vérifié, impair, dans la borne de recherche ;
- `Fp` : résidu canonique \([0,p-1]\), sans flottant ;
- `ToyCurve` : paramètres \((p,a,b)\) avec discriminant non nul ;
- `ToyPoint` : point à l’infini ou coordonnées validées sur **sa propre** `ToyCurve` ;
- `LocalResearchCase` : graine, domaine, source locale et autorisation explicite ;
- `ExperimentId` : empreinte canonique du manifeste, de la graine et du corpus.

Aucun de ces types ne doit implémenter un parseur de clé publique, une désérialisation SEC 1,
un import d’adresse, une URL de chaîne, un client réseau ou une conversion implicite depuis des
octets externes.

### Arithmétique et énumération exactes

1. Vérifier `p` avec le test déterministe de `scirust-modalg`.
2. Réduire `a` et `b` de façon canonique, puis refuser le discriminant nul.
3. Construire une table ordonnée \(r \mapsto [y]\) de tous les \(y^2 \bmod p\).
4. Parcourir les \(x\) croissants, calculer \(x^3+ax+b\), et émettre les solutions dans
   l’ordre \((x,y)\), précédées du point à l’infini.
5. Poser \(\#E(\mathbb F_p)\) égal au nombre de points ainsi énumérés. La borne de Hasse est un
   contrôle de cohérence, jamais un substitut à l’énumération.
6. Calculer l’ordre d’un point \(P\) en partant de \(\#E\), en factorisant ce nombre avec
   `scirust-modalg`, puis en divisant par chaque facteur seulement si
   \((\#E/q)P=\mathcal O\).

L’addition, le doublement, l’inverse et les cas particuliers (\(\mathcal O\), opposés,
tangente verticale) seront testés contre cette énumération exhaustive. Les inverses de
dénominateurs sont fournis par l’arithmétique modulaire existante ; aucun calcul réel, aucune
tolérance et aucun logarithme discret ne sont nécessaires.

## Architecture proposée du futur crate

    scirust-elliptic-discovery/
      src/
        lib.rs              # surface publique, forbid unsafe
        scope.rs            # garde-fous LocalResearchCase
        field.rs            # ToyPrime et Fp, façade exacte sur modalg
        curve.rs            # ToyCurve, ToyPoint, lois de groupe
        enumerate.rs        # liste exacte et ordre canonique
        orders.rs           # ordres et certificats de division
        invariant.rs        # invariants exacts observables
        grammar.rs          # langage fini de relations candidates
        catalog.rs          # propriétés connues et signatures
        classify.rs         # Known / Artifact / Refuted / Candidate
        falsify.rs          # recherche ordonnée du premier contre-exemple
        proof.rs            # tentative symbolique exacte et certificat
        experiment.rs       # graines, corpus, partitions, manifeste
        canonical.rs        # sérialisation canonique et empreintes
        report.rs           # rapport stable et relecture
      tests/
        field_and_curve.rs
        exhaustive_small.rs
        known_catalog.rs
        counterexamples.rs
        reproducibility.rs

Les modules sont orientés dans un seul sens : `scope` et `field` fondent `curve`;
`curve` fonde `enumerate` et `orders`; les candidats n’accèdent qu’aux invariants
immuables ; le falsificateur et le classificateur produisent un rapport mais ne modifient ni le
corpus ni le catalogue.

## Langage de recherche et classification

Un candidat est une expression typée, bornée et entièrement exacte sur des points et des scalaires.
Le noyau initial comprend : identité, négation, addition, doublement, multiplication scalaire
par un entier borné, coordonnées valides, \(j\), discriminant, ordre de point et cardinal du
groupe. Les opérateurs partiels retournent `Undefined` plutôt que d’inventer une valeur.

Chaque résultat porte l’un des statuts suivants :

| Statut | Sens |
|---|---|
| `Refuted` | Un contre-exemple canonique est archivé. |
| `Known` | Correspond à une propriété du catalogue, avec référence. |
| `RepresentationArtifact` | Disparaît après normalisation de représentation ou changement de coordonnées/encodage. |
| `NeedsLiteratureReview` | Motif persistant mais hors catalogue suffisamment vérifié. |
| `Inconclusive` | Couverture insuffisante ou résultat non déterminable. |
| `CandidateUnclassified` | A franchi toutes les portes automatiques, mais n’est **pas** une découverte nouvelle. |

Il n’existe volontairement aucun statut `New` ou `Discovered`. Une conclusion humaine
éventuelle exige une revue de littérature et une preuve ou justification indépendante.

## Catalogue initial de propriétés connues

Le moteur doit reconnaître et exclure au minimum les familles suivantes avant de classer un
motif comme candidat non classifié.

| Famille connue | Signature exacte ou testable | Classification |
|---|---|---|
| Négation et identité | \(-(x,y)=(x,-y)\), \(P+(-P)=\mathcal O\), \(-\mathcal O=\mathcal O\) | `Known` |
| Linéarité de groupe | \(m(nP)=(mn)P\), associativité, divisibilité de l’ordre | `Known` |
| Automorphismes \(j=0\) | Pour \(E:y^2=x^3+b\), \((x,y)\mapsto(\zeta x,y)\) si \(\zeta^3=1\) | `Known` |
| Racines cubiques de l’unité | \(\zeta^3=1\), \(\zeta\ne1\), seulement lorsque le corps contient la racine non triviale | `Known` ou conditionnelle |
| Cas \(j=1728\) voisin | Automorphismes supplémentaires de \(y^2=x^3+ax\) lorsque le corps contient les constantes requises | `Known` |
| Endomorphismes de type GLV | Relation \(\phi(P)=[\lambda]P\) provenant d’un endomorphisme connu et de son polynôme minimal | `Known` ; jamais présenté comme nouveau |
| Changements de coordonnées | Isomorphisme \(x=u^2x', y=u^3y'\) et transformation correspondante de \(a,b\) | `RepresentationArtifact` ou `Known` |
| Twists et classes de \(j\) | Même invariant \(j\) sans identité automatique de groupe de points | `Known`; éviter les conclusions transversales abusives |
| Symétries d’encodage | Choix de signe de \(y\), point à l’infini, ordre ou forme de coordonnées | `RepresentationArtifact` |
| Artefacts de sous-corpus | Motif vrai seulement pour \(j=0\), une congruence de \(p\), ou un ordre fixé | `RepresentationArtifact` ou `Refuted` après partition indépendante |

L’automorphisme \(j=0\) se vérifie directement :
\((\zeta x)^3+b=\zeta^3x^3+b=x^3+b\).
L’existence de racines cubiques non triviales, les classes exceptionnelles \(j=0\) et
\(j=1728\), et les twists devront être marqués comme conditions de corpus, non comme
exceptions découvertes. Les références de base sont [Silverman, Arithmetic of Elliptic Curves](https://www.math.brown.edu/johsilve/AECHome.html)
et la documentation des [courbes elliptiques sur corps finis de SageMath](https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/ell_finite_field.html).

Les endomorphismes GLV sont une technique connue d’accélération de multiplication scalaire, pas
un signal de structure inédite ; voir [Gallant, Lambert et Vanstone, CRYPTO 2001](https://link.springer.com/chapter/10.1007/3-540-44647-8_11).
Les représentations comprimées doivent être traitées comme des encodages : SEC 1 décrit les
formats de représentation de points, et non une nouvelle symétrie algébrique
([SEC 1 v2](https://www.secg.org/sec1-v2.pdf)). Le v0.1 n’implémentera toutefois aucun de ces
formats.

## Protocole de falsification

Une relation ne franchit les portes suivantes que dans cet ordre :

1. **G0 — Domaine autorisé.** Chaque cas est `LocalResearchCase`, étiqueté jouet et généré
   localement ; tout autre type d’entrée est impossible à représenter dans l’API.
2. **G1 — Base exacte.** L’évaluation utilise seulement des entiers, \(\mathbb F_p\) et les lois
   de groupe vérifiées ; aucune valeur flottante ni heuristique ne décide le verdict.
3. **G2 — Exhaustivité petite.** Tester toutes les courbes non singulières et tous les points
   nécessaires sur le corpus exhaustif défini ci-dessous.
4. **G3 — Jeu indépendant.** Tester un corpus séparé, déterminé par une graine et un manifeste
   distincts ; ne jamais choisir l’échantillon après avoir vu le candidat.
5. **G4 — Contre-exemple.** Énumérer les entrées dans un ordre canonique et archiver le premier
   contre-exemple \((p,a,b,\text{tuple de points},\text{expression})\), s’il existe.
6. **G5 — Catalogue.** Comparer la relation et ses normalisations au catalogue ci-dessus,
   notamment sous négation, isomorphismes et encodages.
7. **G6 — Montée en taille et justification.** Passer l’échelle définie ci-dessous, puis tenter
   une identité polynomiale exacte, un certificat de calcul fini ou une justification
   symbolique. Un échec garde le statut `CandidateUnclassified` ou `Inconclusive`.

Les contrôles négatifs obligatoires comprennent : une formule de négation volontairement fausse,
un doublement avec signe erroné, une propriété valide seulement à \(j=0\) présentée à tort comme
universelle, une symétrie de signe d’encodage, et une expression surajustée au corpus
d’apprentissage. Ils doivent être réfutés par le falsificateur.

Les résultats de G2 et G3 sont distincts. Une propriété qui passe G2 mais échoue G3 est
`Refuted`; une propriété qui ne reçoit pas de couverture suffisante est `Inconclusive`.
Aucune réussite expérimentale, même exhaustive sur la borne jouet, ne vaut une généralisation
hors de son domaine.

## Corpus déterministes

| Ensemble | Construction | Objet |
|---|---|---|
| `ExhaustiveSmall` | Tous les \((a,b)\) non singuliers pour \(p\in\{5,7,11,13\}\), tous les points et tous les tuples dans le budget déclaré | Falsification complète sur petite taille |
| `IndependentHoldout` | Primes \(17\) à \(97\), partitions déterministes par graine, courbes locales et points énumérés exactement | Validation indépendante |
| `ScaleLadder` | \(p\in\{127,251,509,1021,2039,4093\}\), toutes les courbes sélectionnées par manifeste, énumération exacte des points ; tuples d’arité élevée sous couverture explicitement chiffrée | Étude de montée en taille |

Le rapport doit contenir, pour chaque ensemble : algorithme de sélection, graine, version du
crate, bornes, nombre de courbes examinées, nombre de points/tuples évalués, ordre de parcours,
empreinte canonique et verdict. Toute parallélisation future doit produire le même premier
contre-exemple et le même rapport qu’une exécution séquentielle.

## Invariants de sécurité, d’exactitude et de reproductibilité

| Invariant | Exigence vérifiable |
|---|---|
| Recherche autorisée uniquement | Aucune API n’accepte adresses, clés publiques, encodages SEC 1, endpoints réseau ou données blockchain. |
| Instances locales | Toutes les courbes proviennent de paramètres jouets validés ou d’un générateur local avec graine. |
| Exactitude | Aucun `f32`, `f64`, epsilon, racine numérique, hash map ordonnée implicitement ou tirage caché dans le chemin de verdict. |
| Reproductibilité | Graine, version, manifeste, limites, ordre canonique et empreinte sont présents dans chaque rapport. |
| Déterminisme | Les conteneurs de sortie sont ordonnés ; le premier contre-exemple est défini par ordre lexicographique. |
| Pureté Rust | `#![forbid(unsafe_code)]`, aucune FFI, aucun backend BLAS, aucune E/S réseau. |
| Transparence | Les hypothèses, contrôles négatifs, limites de couverture et échecs de preuve sont rapportés, jamais supprimés. |
| Non-attribution | Les statuts ne comportent pas « nouveau » ; le catalogue connu est consulté avant tout candidat. |

## Feuille de route

### Phase 0 — Audit et contrat de recherche

Terminer ce document, garder le dépôt sans changement de production, faire relire le périmètre
et le catalogue. Critère de sortie : accord sur les types d’entrée interdits et les portes G0–G6.

### Phase 1 — Noyau exact minimal

Créer le crate, l’ajouter au workspace, implémenter `ToyPrime`, `Fp`, `ToyCurve`,
`ToyPoint`, énumération et ordres, avec références croisées contre `scirust-modalg`.
Critère de sortie : lois de groupe et corpus `ExhaustiveSmall` passent sans flottants.

### Phase 2 — Harnais expérimental

Ajouter manifeste canonique, corpus local, `SplitMix64` à graine explicite, rapport stable,
empreinte et premier contre-exemple. Critère de sortie : deux exécutions identiques produisent
des octets identiques.

### Phase 3 — Catalogue et contrôles

Implémenter les règles de reconnaissance de négation, \(j=0\), racines cubiques, changements de
coordonnées, artefacts d’encodage et GLV connu. Ajouter les contrôles négatifs. Critère de sortie :
chaque contrôle est classé correctement.

### Phase 4 — Génération de candidats et falsification

Ajouter une grammaire finie typée, une recherche exhaustive à budget fixe, les partitions
d’apprentissage/validation et la montée en taille. Critère de sortie : aucun candidat ne peut
éviter G2–G6.

### Phase 5 — Justification et revue

Ajouter les certificats/identités exacts faisables avec `modalg::poly`, une exportation de
rapport lisible et une procédure de revue de littérature humaine. Critère de sortie : les
rapports séparent formellement preuve, contre-exemple, propriété connue et hypothèse.

### Phase 6 — Exécution et rejeu locaux

Ajouter une frontière d’exécution sur `SearchPlan`, un reçu canonique complet et un rejeu
strict qui détecte toute divergence. Aucun décodeur de données externes n’est ajouté. Critère
de sortie : les mêmes plan et version produisent les mêmes octets, et toute altération du reçu
est détectée. La spécification détaillée est
[`SCIRUST_ELLIPTIC_DISCOVERY_EXECUTION_REPLAY_V0_1.md`](SCIRUST_ELLIPTIC_DISCOVERY_EXECUTION_REPLAY_V0_1.md).

## Validations à appliquer

Après l’ajout du crate, exécuter au minimum :

    cargo +nightly-2026-07-02 fmt --all -- --check
    cargo +nightly-2026-07-02 clippy -p scirust-elliptic-discovery --all-targets --locked -- -D warnings
    cargo +nightly-2026-07-02 test -p scirust-elliptic-discovery --locked
    cargo +1.89.0 check -p scirust-elliptic-discovery --locked
    git diff --check

Pour le présent livrable documentaire, la validation applicable est `git diff --check` et une
inspection du diff. Dans l’environnement d’audit actuel, `cargo` et `rustup` ne sont pas
installés dans `PATH`; aucune validation Cargo ne doit donc être présentée comme exécutée ou
réussie.

## Références

- [Documentation de contribution du workspace](../../CONTRIBUTING.md)
- [CI du workspace](../../.github/workflows/ci.yml)
- [Spécification HyperCrypto v0.1](../../scirust-hypercrypto/docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md)
- [Rapport de falsification HyperCrypto phase 1](../../scirust-hypercrypto/docs/research/SCIRUST_HYPERCRYPTO_FALSIFICATION_PHASE1.md)
- [J. H. Silverman, The Arithmetic of Elliptic Curves](https://www.math.brown.edu/johsilve/AECHome.html)
- [R. Gallant, R. Lambert et S. Vanstone, Faster Point Multiplication on Elliptic Curves with Efficient Endomorphisms](https://link.springer.com/chapter/10.1007/3-540-44647-8_11)
- [Standards for Efficient Cryptography, SEC 1 v2](https://www.secg.org/sec1-v2.pdf)
- [SageMath — Elliptic curves over finite fields](https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/ell_finite_field.html)
