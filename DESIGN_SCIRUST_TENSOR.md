# Design Specification: scirust-tensor

## 1. Architecture de crates/modules

Le module **scirust-tensor** est structuré comme une suite de crates spécialisées, garantissant une séparation claire entre la logique d'algèbre, la planification et l'exécution.

*   **scirust-tensor-core** : Définit le type `TensorND` (N-dimensionnel), les types de formes (Shapes) et les primitives de manipulation de strides. C'est le socle commun sans dépendances lourdes.
*   **scirust-tensor-einsum** : Contient le parseur de signatures de style Einstein (ex: `"ij,jk->ik"`) et la logique de réduction en opérations de contraction binaires.
*   **scirust-tensor-contraction** : Implémente le **Contraction Planner**. Il décide de l'ordre optimal des multiplications pour minimiser les FLOPs et la mémoire. Contient les kernels CPU/SIMD de base.
*   **scirust-tensor-compile** : Le "compilateur de graphe". Il transforme une suite d'opérations en un graphe d'exécution optimisé (élimination de redondances, fusion d'opérateurs).
*   **scirust-tensor-runtime** : Moteur d'exécution léger. Il gère l'allocation de buffers et l'exécution des graphes compilés, compatible avec le format SRT1.
*   **scirust-tensor-examples** : Démonstrations (Transformer Multi-Head Attention via einsum).

**Dépendances** : `runtime` -> `compile` -> `contraction` -> `einsum` -> `core`.
**Hardware** : Core/Einsum/Planner sont 100% CPU. Les kernels de `contraction` et le `runtime` sont extensibles au GPU.

## 2. Types Rust principaux

```rust
use std::collections::HashMap;

/// Tenseur N-dimensionnel avec gestion explicite des strides pour le déterminisme.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorND {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

/// Représentation d'une opération einsum analysée.
pub struct EinsumPattern {
    pub inputs: Vec<Vec<char>>,
    pub output: Vec<char>,
}

/// Un plan de contraction est une séquence d'étapes de calcul.
pub struct ContractionPlan {
    pub steps: Vec<ContractionStep>,
}

pub enum ContractionStep {
    /// Multiplication de deux tenseurs avec réorganisation optionnelle.
    Contract {
        left: usize,
        right: usize,
        indices_left: Vec<char>,
        indices_right: Vec<char>,
        out_indices: Vec<char>,
    },
    /// Opération unitaire (ex: somme sur un axe).
    Reduce {
        input: usize,
        axis: usize,
    },
}

/// Noeud d'un graphe d'opérations optimisées.
pub enum TensorOp {
    MatMul(usize, usize),
    Add(usize, usize),
    ReLU(usize),
    Fused(FusedOp),
}

/// Noeud d'un graphe d'opérations fusionnées.
pub enum FusedOp {
    /// MatMul + Add Bias + ReLU fusionné en une seule passe mémoire.
    LinearReLU {
        input_idx: usize,
        weight_idx: usize,
        bias_idx: usize,
    },
    /// Contraction optimisée.
    OptimizedContraction(ContractionPlan),
}

pub struct TensorGraph {
    pub ops: Vec<TensorOp>,
    pub buffers: Vec<TensorND>,
}
```

## 3. Pipeline complet d'algèbre tensorielle

1.  **Parsing de signature** : La chaîne `"bij,bjk->bik"` est transformée en `EinsumPattern`. On vérifie la cohérence des dimensions.
2.  **Contraction Planning** : Pour plus de 2 tenseurs, un algorithme glouton (ou exhaustif pour les petits graphes) calcule le coût en FLOPs de chaque ordre possible (ex: `(A*B)*C` vs `A*(B*C)`).
3.  **Construction du Graphe** : Les opérations (matmul, transpose, add) sont insérées dans un `TensorGraph`.
4.  **Optimisation et Fusion** :
    *   **Permute Fusion** : Si une permutation d'axes précède un MatMul, on fusionne les deux en manipulant les strides dans le kernel GEMM.
    *   **Operator Fusion** : On identifie les patterns `Linear -> Bias -> ReLU` et on les remplace par un seul kernel `FusedOp`.
5.  **Exécution CPU** : Utilisation de `scirust-simd` pour des kernels tillés (blocking) garantissant le déterminisme bit-à-bit (sommation fixe).
6.  **Exécution GPU** : Si disponible, dispatch vers des shaders WGSL (`wgpu`) ou des kernels Tensor Cores (`cuBLAS`).

## 4. Version MVP (v1)

*   **Fonctionnalités** : Einsum binaire, transposition automatique, kernels CPU optimisés via Rayon (déterministe).
*   **API Rust** :
```rust
use scirust_tensor_einsum::einsum;

let a = TensorND::rand(&[10, 20]);
let b = TensorND::rand(&[20, 30]);

// C[i, k] = sum_j A[i, j] * B[j, k]
let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
```
*   **Métriques attendues** : Overhead de parsing < 1ms, latence GEMM CPU compétitive avec `scirust-core`.

## 5. Version avancée (v2)

*   **Planner Automatique** : Support d'einsum à N tenseurs (ex: `"ij,jk,kl->il"`) avec recherche du chemin de contraction optimal.
*   **Compilation XLA-like** : Génération d'un plan d'exécution statique réutilisable pour l'inférence.
*   **Fusion automatique d'opérateurs** : Heuristique de fusion multi-couches.
*   **Support GPU complet** : Kernels wgpu optimisés pour les contractions binaires.
*   **JIT de kernels** : Utilisation de `scirust-rustc-driver` pour compiler des kernels fusionnés à la volée.

## 6. Métriques à suivre

*   **Performance** : GFLOPS, latence p50/p99.
*   **Mémoire** : Nombre de buffers intermédiaires économisés par fusion.
*   **Optimisation** : Nombre d'opérations éliminées du graphe initial.
*   **Déterminisme** : Fingerprint (hash) de la sortie bit-à-bit identique sur 1 et N threads.

## 7. Déterminisme et SRT1

*   **Ordre de réduction** : Toutes les réductions (sommes) utilisent un ordre fixe (généralement croissant par index) pour éviter les instabilités du flottant liées à l'associativité.
*   **Frozen Graph** : Le graphe optimisé est sérialisé dans le format **SRT1**, incluant les formes et les types de kernels choisis.
*   **Validation par Oracle** : Chaque plan de contraction complexe est validé par un oracle "naïf" (boucles imbriquées) lors des tests d'intégration.

## 8. Quantification int8 et QSR1

*   **Quantification des tenseurs** : Stockage des échelles (scales) et des points zéro (zero points) dans le format QSR1.
*   **Einsum int8** : Accumulation systématique en `i32` pour éviter l'overflow, suivie d'une requantisation fixe déterministe.
*   **Validation** : Chaque opération quantifiée est comparée à son équivalent f32 via oracle avec une erreur bornée.

## 9. Risques techniques

*   **Compilateur XLA-like complexe** : Difficulté de gérer tous les cas de fusion. *Mitigation : Commencer par des patterns prédéfinis.*
*   **Fusion sur GPU** : Nécessite l'écriture manuelle de kernels WGSL complexes. *Mitigation : Utiliser des templates de kernels.*
*   **Déterminisme cross-architecture** : Hors scope, focus sur la stabilité cross-thread sur une même machine.

## 10. Checklist de validation

*   [ ] Tests unitaires pour einsum (parsing et exécution).
*   [ ] Tests d'intégration pour contraction planner (FLOPs optimality).
*   [ ] Tests de déterminisme (fingerprint stable).
*   [ ] Validation par oracle CPU (loop-based reference).
*   [ ] Benchmarks de performance (GFLOPS).
