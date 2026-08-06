# RFC-0001 — SciRust Compute Contracts

- **Status:** Draft
- **Crate:** `scirust-compute`
- **MSRV:** Rust 1.89

## Motivation

SciRust possède déjà plusieurs chemins de calcul CPU, SIMD, WGPU et CUDA,
mais leurs contrats sont séparés et incompatibles.

`scirust-compute` introduit un vocabulaire commun sans remplacer les
implémentations existantes.

## Decision

Le crate définit uniquement :

- les types scalaires et métadonnées tensor ;
- les périphériques et espaces mémoire ;
- les modules noyaux et configurations de lancement ;
- les liaisons de buffers ;
- les erreurs communes ;
- le trait `ComputeBackend`.

Le crate reste sans dépendance et compatible `no_std`.

## Dependency rule

Les backends dépendent de `scirust-compute`.

`scirust-compute` ne dépend ni de `scirust-core`, ni de `scirust-gpu`,
ni de `scirust-cuda`, ni de `scirust-simd`.

## Non-goals

Cette phase ne remplace pas CUDA ou WGPU, ne crée pas un nouveau moteur tensor,
ne modifie aucune API existante et n'implémente ni autograd ni ordonnanceur.
