# SciRust Memory Wall — Architecture des 5 Piliers

## Vue d'ensemble

Ce document décrit l'architecture complète pour surmonter le **Memory Wall** (goulot d'étranglement de la bande passante mémoire) dans SciRust, cible les architectures ARM64 (Jetson AGX Thor) et x86 (64 cœurs).

## Problématique

Dans les SSM (Mamba), LLMs, et le trading haute fréquence :

| Métrique | Avant | Cible |
|----------|-------|-------|
| Alllocations/intmédiaire | 1-3 per layer norm | 0 (fusion) |
| CPU↔GPU copy per matmul | 2 (h2d + d2h) | 0 (zero-copy pinned) |
| Arena alloc latency | O(n) linear scan | O(1) pointer bump |
| L2 cache hit rate (matmul tiled) | ~65% | >95% |
| Bandwidth effective (quant int8) | ~1× | ~4× |

## Structure des modules

```
scirust/
├── scirust-arena/                    # Pilier 3: Arena Allocators
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                    # Arena public API
│       ├── allocator.rs              # PinnedArena impl
│       ├── slab.rs                   # Slab allocator for SSM states
│       └── aligned.rs                # AlignedVec + alignment utilities
├── scirust-fusion/                   # Pilier 1: AST Kernel Fusion
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                    # Public API
│       ├── graph.rs                  # OpGraph — graphe de dépendance
│       ├── fusion.rs                 # FusionPass — détection + fusion
│       ├── kernel.rs                 # FusedKernel — génération de code
│       └── patterns.rs               # Built-in fusion patterns
├── scirust-core/ (modifié)
│   └── src/
│       ├── tensor/
│       │   ├── pinned.rs             # Pilier 2: PinnedMemory
│       │   └── mem_pool.rs           # Pilier 2: MemoryPool
│       ├── simd/
│       │   ├── tiling.rs             # Pilier 4: Cache-Aware Tiling
│       │   └── neon.rs               # Pilier 4: ARM64 NEON/SVE
│       ├── quant/
│       │   ├── mod.rs                # Pilier 5: quantification module
│       │   ├── int8.rs               # Pilier 5: int8 quant + dequant SIMD
│       │   ├── bf16.rs               # Pilier 5: bf16 <-> f32
│       │   └── int4.rs               # Pilier 5: int4 unpacking
│       └── nn/
│           └── fused_ops.rs          # Pilier 1: Fused matmul+silu+layernorm
└── scirust-simd/ (modifié)
    └── src/
        ├── sve.rs                    # Pilier 4: ARM SVE intrinsics
        └── matrix/tiling_dispatch.rs # Pilier 4: Tiling dispatch
```

## Détails par pilier

### Pilier 1: AST Kernel Fusion

**Objectif**: Éviter les allers-retours en RAM en fondant les opérateurs consécutifs.

**Algorithme de détection**:
1. Construire un `OpGraph` depuis le MIR (via le rustc driver) ou depuis le forward pass (via le tracing runtime)
2. Rechercher les motifs (patterns) canoniques:
   - `MatMul → SiLU` (linear activation)
   - `MatMul → SiLU → LayerNorm` (MLP block)
   - `MatMul → LayerNorm` (pre-LN transformer)
   - `MatMul → MatMul → Add` (two-layer MLP)
   - `Conv2d → ReLU → Pool` (conv block)
3. Pour chaque motif détecté, générer un `FusedKernel` qui:
   - Calcule mean/var en un passage (LayerNorm)
   - Applique SiLU/GELU sans intermédiaire
   - Accumule les produits matriciels dans des registres accum

### Pilier 2: Mémoire Unifiée (Zero-Copy)

**PinnedMemory** — mémoire alignée 64 octets, pinée en espace utilisateur:
- Sur ARM64 (Jetson): utilise `mmap(MAP_ANONYMOUS | MAP_POPULATE)` avec `mlock()`
- Sur x86: utilise `posix_memalign` + `mlock()`
- Compatible CUDA Unified Memory (`cudaHostRegister`) et GPU Direct

**MemoryPool** — pool de tenseurs à taille fixe:
- Réduit la fragmentation pour les batches de taille constante
- Réutilise les blocs déjà alloués

### Pilier 3: Arena Allocators

**PinnedArena** — allocation par bump pointer:
- Pré-alloue un grand bloc (128-byte aligned)
- `alloc::<T>()` = déplacement d'un pointeur (O(1))
- `reset()` = remise à zéro du pointeur (O(1))
- **No Drop, no free** — toutes les allocations sont dealloquées ensemble

**Slab** — pour les états SSM:
- Stocke les états cachés (c, h̃) des cellules Mamba
- Accès par index (O(1))
- Supporte le garbage collection mark-and-sweep pour les séquences de longueur variable

### Pilier 4: Auto-Vectorisation SIMD "Cache-Aware"

**Tiling** — pour le matmul:
- Analyse la taille du cache L2 de la machine cible
- Adapte la taille des blocs (tile) pour qu'ils tiennent dans L2
- x86: AVX-512 (16x f32 par tile), AVX2 (8x f32)
- ARM64: NEON (4x f32), SVE (scalable vector length)

**Cache profilage**:
- Détecte la taille L2 au runtime via `/sys/devices/system/cpu/cpu0/cache/`
- Sur Jetson AGX Thor: L2 = 4MB per cluster
- Sur x86: L2 ≈ 256KB-1MB per core

### Pilier 5: Primitives de Quantification Natives

**QuantizedTensor** — stockage quantifié:
- `int8`: 4× compression (f32 → int8)
- `bf16`: 2× compression (f32 → bf16)
- `int4`: 8× compression (f32 → int4 packed)

**Décompression on-the-fly**:
- int8 → f32 dans registres SIMD (AVX2: 8-lanes, NEON: 4-lanes)
- bf16 → f32: conversion directe
- int4 packed → int8 → f32: unpack + sign-extend

**Calcul en quantifié**:
- int8 × int8 → int32 (accumulate in int32)
- Fused dequant + matmul: déquantiser directement dans le produit

## Compatibilité

| Plateforme | SIMD | Arena | Fusion | Quant | Pinned |
|------------|------|-------|--------|-------|--------|
| x86_64 (AVX-512) | AVX-512 | ✓ | ✓ | ✓ | ✓ |
| x86_64 (AVX2) | AVX2 | ✓ | ✓ | ✓ | ✓ |
| ARM64 (NEON) | NEON | ✓ | ✓ | ✓ | ✓ |
| ARM64 (SVE) | SVE | ✓ | ✓ | ✓ | ✓ |
| Jetson AGX Thor | NEON+SVE | ✓ | ✓ | ✓ | ✓ |
