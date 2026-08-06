# SciRust Elliptic Discovery — orchestration de campagne v0.1

## Statut

Ce document définit la phase 8 avant toute modification du code de production.
Il complète les phases 0 à 7 sans étendre leur domaine : seules les courbes
jouets intégrées, générées localement sur des corps premiers bornés, sont
représentables.

## Constat vérifiable

Les composants nécessaires à une campagne existent, mais restent séparés :

- `run_control` exécute un contrôle obligatoire sur un corpus ;
- `run_search` retourne les évaluations détaillées des candidats ;
- `execute_local` produit un reçu de recherche, mais ne lance pas les contrôles
  et ne conserve que les empreintes et un résumé ;
- `attempt_justification` tente une preuve exacte ;
- `review_candidate` applique la frontière de revue humaine ;
- `ReviewReport` produit un rapport lisible pour un candidat.

Ainsi, aucun point d'entrée public unique ne garantit actuellement qu'une
recherche, ses contrôles, ses justifications et ses revues en attente décrivent
la même exécution. La phase 8 ferme cette lacune d'orchestration. Elle n'ajoute
ni nouvelle grammaire, ni nouveau corpus, ni déclaration de nouveauté.

## Objectif

Ajouter une frontière locale `execute_campaign(SearchPlan)` qui produit un
artefact complet, déterministe et rejouable contenant :

1. le reçu de recherche de phase 6 ;
2. les six contrôles obligatoires dans un ordre canonique ;
3. toutes les évaluations détaillées dans l'ordre de génération ;
4. une tentative de justification exacte pour chaque candidat ;
5. une revue humaine explicitement en attente ;
6. un rapport Markdown stable et une empreinte SHA-256 de campagne.

## Décisions de conception

### Construction unique des corpus et résultats

Une campagne construit `ResearchCorpora` une seule fois. Les contrôles utilisent
`ExhaustiveSmall`; la recherche utilise les trois partitions existantes. Le
reçu est construit à partir des corpus et évaluations déjà calculés, sans
relancer silencieusement la recherche.

L'ordre d'exécution est fixe :

1. génération des corpus ;
2. contrôles obligatoires ;
3. génération et falsification des candidats ;
4. construction du reçu ;
5. tentative de justification ;
6. création des revues `Pending` ;
7. encodage canonique et rapport lisible.

### Contrôles obligatoires et attentes

La séquence canonique et ses résultats attendus sont :

| Ordre | Contrôle | Statut attendu | Contre-exemple attendu |
|---:|---|---|---|
| 0 | `TrueNegation` | `Known` | non |
| 1 | `FalseNegationKeepsY` | `Refuted` | oui |
| 2 | `FalseDoublingSign` | `Refuted` | oui |
| 3 | `JZeroClaimedUniversal` | `Refuted` | oui |
| 4 | `EncodingSignClaimedNovel` | `RepresentationArtifact` | non |
| 5 | `OverfitAZero` | `Refuted` | oui |

`CampaignRun::controls_valid` exige exactement cette séquence. Une campagne
dont un contrôle diverge reste inspectable, mais ne peut pas être présentée
comme une exécution de référence valide.

### Frontière de revue humaine

La campagne appelle `attempt_justification` pour chaque évaluation, puis
`review_candidate` avec `LiteratureReview::pending()`. Elle ne fabrique jamais
de reviewer, de source ou de décision humaine.

Par conséquent, aucune exécution automatisée de campagne ne peut attribuer
`CandidateUnclassified`. Une relation qui nécessite une revue demeure
`NeedsLiteratureReview` jusqu'à une intervention humaine séparée et auditable.

### Artefact canonique

`CampaignRun::canonical_bytes` utilise le domaine :

    SCIRUST-ELLIPTIC-DISCOVERY/CAMPAIGN/V1

L'encodage contient, dans cet ordre :

1. la version de schéma ;
2. les octets canoniques du reçu d'exécution ;
3. chaque contrôle, sa classification et son contre-exemple éventuel ;
4. pour chaque candidat, son empreinte d'évaluation et le rapport de revue en
   attente correspondant.

Les longueurs sont explicites et tous les entiers sont en ordre big-endian via
`CanonicalEncoder`. L'empreinte de campagne est le SHA-256 de ces octets.

L'ajout de cette surface publique porte le crate à `0.3.0`. Les domaines V2 du
plan, des évaluations et du reçu restent inchangés : ils encodent déjà la
version du crate. Le nouvel artefact reçoit son propre domaine `CAMPAIGN/V1`.

La phase n'ajoute aucun décodeur de données externes. Le rejeu reçoit un objet
`CampaignRun` déjà construit par l'API locale et recalcule la campagne à partir
du `SearchPlan` validé qu'il contient.

### Rapport lisible

`CampaignReport` produit un Markdown déterministe comprenant :

- empreintes du plan, du reçu et de la campagne ;
- validité et résultats des contrôles ;
- résumé exact des statuts automatisés ;
- une section ordonnée par candidat avec couverture, contre-exemple,
  catalogue, justification et état de revue ;
- un avertissement explicite qu'aucun statut ne constitue une découverte.

Le rapport est une vue de l'artefact, jamais une source d'autorité distincte.

### Rejeu strict

`replay_campaign` recalcule `execute_campaign(expected.plan())`, compare les
octets canoniques complets et conserve l'observation recalculée même en cas de
divergence. Une suppression, permutation ou altération d'un résultat doit donc
être détectée.

## Réutilisation et absence de duplication

- `ResearchCorpora` et `run_search` restent les seules sources des corpus et
  évaluations ;
- la construction interne du reçu est extraite d'`execute_local` et réutilisée ;
- l'encodage canonique des classifications et contre-exemples est partagé avec
  `execution` ;
- `ReviewReport` reste la représentation lisible détaillée d'un candidat ;
- aucune dépendance n'est ajoutée.

## Invariants de sécurité et de reproductibilité

| Invariant | Garantie de phase 8 |
|---|---|
| Entrées | Seul un `SearchPlan` validé est accepté. |
| Corpus | Seulement les trois corpus jouets intégrés. |
| Contrôles | Six contrôles, ordre et attentes fixes. |
| Exactitude | Aucun flottant, epsilon ou heuristique dans le verdict. |
| Reproductibilité | Même plan et même version donnent les mêmes octets. |
| Revue | Toujours `Pending` lors de l'exécution automatisée. |
| Non-nouveauté | Aucun statut `New` ou `Discovered`; aucune revue inventée. |
| Pureté Rust | `unsafe` interdit, aucune FFI, aucun réseau. |
| Ressources | Les bornes de `SearchPlan` et de phase 7 restent applicables. |

## Tests de sortie

La phase est terminée lorsque :

- deux campagnes identiques produisent les mêmes octets et la même empreinte ;
- deux graines distinctes produisent des empreintes distinctes ;
- les six contrôles sont présents dans l'ordre et valides ;
- le reçu de campagne est identique au reçu local construit sur le même plan ;
- chaque candidat possède une justification et une revue `Pending` ;
- aucune campagne automatisée ne produit `CandidateUnclassified` ;
- une altération de la campagne est détectée au rejeu ;
- le rapport contient les contrôles, la couverture, les contre-exemples, les
  justifications et l'avertissement de non-nouveauté ;
- formatage, Clippy, tests, MSRV et vérifications du workspace passent.

## Hors périmètre

- CLI, lecture ou écriture de fichiers et désérialisation d'artefacts ;
- courbes fournies par l'utilisateur ou corpus externes ;
- adresses, clés, SEC 1, réseau, RPC ou blockchain ;
- nouvelle grammaire mathématique ou nouveaux certificats ;
- revue de littérature automatique ;
- déclaration de découverte, de nouveauté ou de portée hors domaine jouet.
