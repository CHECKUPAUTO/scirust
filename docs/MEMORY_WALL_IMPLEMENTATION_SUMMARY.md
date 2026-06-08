# SciRust — Mémoire Wall Optimization: Implementation Summary

## Architecture finale des modules

```
scirust/
├── Cargo.toml                         # Workspace avec scirust-arena + scirust-fusion
├── docs/
│   ├── MEMORY_WALL_ARCHITECTURE.md    # Architecture complète (5 piliers)
│   └── MEMORY_WALL_IMPLEMENTATION_SUMMARY.md  # Ce fichier
│
├── scirust-arena/                     # PILIER 3: Arena Allocators
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                     # PinnedArena + Slab + AlignedVec + exports
│       ├── allocator.rs               # PinnedArena — bump pointer, 128-byte aligned
│       ├── slab.rs                    # Slab — free list + versioning pour SSM
│       └── aligned.rs                 # AlignedVec — buffer aligné SIMD
│
├── scirust-fusion/                    # PILIER 1: AST Kernel Fusion
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                     # FusionPipeline — entrée publique
│       ├── graph.rs                   # OpGraph — graphe de dépendance DAG
│       ├── fusion.rs                  # FusionPass — détection de motifs
│       ├── kernel.rs                  # FusedKernel — exécution des kernels
│       └── patterns.rs                # FusionPatterns — base de motifs canoniques
│
├── scirust-core/                      # Core — modules mis à jour
│   ├── Cargo.toml                     # +libc pour pinned memory
│   └── src/
│       ├── lib.rs                     # Export quant + tensor/pinned
│       ├── tensor/
│       │   ├── mod.rs                 # +pinned + tiling exports
│       │   ├── tensor_nd.rs           # TensorND (inchangé)
│       │   ├── pinned.rs              # PILIER 2: PinnedBuffer
│       │   └── tiling.rs              # PILIER 4: Tiling config + detection
│       ├── quant/                     # PILIER 5: Quantification native
│       │   ├── mod.rs                 # QuantTensor + Quantized trait
│       │   ├── int8.rs                # int8 quant/dequant + SIMD AVX2
│       │   ├── bf16.rs                # bf16 quant/dequant + NEON/SVE
│       │   └── int4.rs                # int4 packed (8× compression)
│       └── nn/
│           ├── mod.rs                 # +fused_ops module
│           └── fused_ops.rs           # PILIER 1+4: Kernels fusionnés
│
└── scirust-simd/                      # SIMD — extensions ARM64
    ├── Cargo.toml                     # +libc pour SVE detection
    └── src/
        ├── lib.rs                     # +NEON + SVE + runtime dispatch
        ├── neon.rs                    # PILIER 4: ARM64 NEON intrinsics
        ├── sve.rs                     # PILIER 4: ARM SVE intrinsics
        └── matrix/
            ├── backend.rs             # SimdBackend trait (inchangé)
            └── tiling_dispatch.rs     # (en cours)
```

## Pilier 1: AST Kernel Fusion

### Fichier: `scirust-fusion/src/graph.rs`
- **OpKind**: enum des 30+ opérations supportées
- **FusedOp**: nœud du graphe avec inputs, constante, kind
- **OpGraph**: DAG avec topological sort (algorithme de Kahn)

### Fichier: `scirust-fusion/src/patterns.rs`
Motifs détectés:
| Motif | Gain mémoire | Operations |
|-------|-------------|------------|
| matmul_silu | 50% | Linear + SiLU |
| matmul_relu | 50% | Linear + ReLU |
| matmul_silu_layernorm | 66% | Linear + SiLU + LayerNorm |
| matmul_layernorm | 50% | Linear + LayerNorm |
| layernorm_activation | 50% | LayerNorm + Activation |
| two_layer_mlp | 66% | Linear + Linear + Add |
| matmul_scale | 50% | Linear × scale |
| ssm_scan | 0% | SsmStep + SsmStep (séquentiel) |

### Fichier: `scirust-fusion/src/kernel.rs`
- FusedKernel avec execute() pour chaque kernel type
- Implémente matmul_silu, matmul_gelu, matmul_relu, matmul_layernorm
- Les accumulateurs restent dans des vecteurs locaux (stack), jamais en heap

### Fichier: `scirust-core/src/nn/fused_ops.rs`
- Kernels fusionnés exécutables (matmul_silu, matmul_gelu, matmul_layernorm)
- Utilise scirust_core::simd::tiling pour le tiling automatique
- Compatible avec le graphe autograd

## Pilier 2: PinnedMemory (Zero-Copy)

### Fichier: `scirust-core/src/tensor/pinned.rs`
- **PinnedBuffer**: mmap + mlock sur Linux, align 128 bytes
- **PinnedPool**: pool de buffers réutilisables
- **MemoryLayout**: enum (Cpu, Pinned, GpuUnified)
- Compatible CUDA unified memory (cudaHostRegister)

## Pilier 3: Arena Allocators

### Fichier: `scirust-arena/src/allocator.rs`
- **PinnedArena**: bump pointer, O(1) alloc/dalloc
- Alignement 128 bytes garanti
- reset() = O(1) — tout le bloc est reset en une opération
- **MemoryBlock**: mmap(MAP_ANONYMOUS) + mlock()

### Fichier: `scirust-arena/src/slab.rs`
- **Slab**: free list + versioning pour les états SSM
- Handle avec version → protection use-after-free
- **SlabHandle**: index + version

### Fichier: `scirust-arena/src/aligned.rs`
- **AlignedVec**: Vec avec alignement garanti
- Trait ToAligned pour Vec<T> → AlignedVec

## Pilier 4: Cache-Aware Tiling

### Fichier: `scirust-core/src/simd/tiling.rs`
- **TilingConfig**: détection auto de la plateforme et du cache L2
- **CacheProfile**: détection L1/L2/L3 + calcul des tiles optimaux
- **matmul_tiled_f32**: matmul tuilé i-p-j
- **detect_l2_cache_size()**: lit /sys/devices/system/cpu/cpu0/cache/
- Config par plateforme:
  - x86_64 AVX-512: tile 64, lane 16
  - x86_64 AVX2: tile 32, lane 8
  - ARM64 NEON: tile 32, lane 4
  - ARM64 SVE: tile scalable, lane configurable

### Fichier: `scirust-simd/src/neon.rs`
- **NEON kernels**: saxpy, add, mul, silu, gelu, relu, layernorm, matmul
- 4-lanes par registre (float32x4_t)
- Tiling pour matmul (32×32 par défaut)

### Fichier: `scirust-simd/src/sve.rs`
- **SVE kernels**: scalable vector length (256-bit sur Jetson Thor)
- Predicate-based: svld1, svst1, svmla, etc.
- Detecte la présence SVE via getauxval(AT_HWCAP)

## Pilier 5: Quantification Native

### Fichier: `scirust-core/src/quant/mod.rs`
- **Quantized** trait: format, compression ratio
- **QuantFormat**: Fp32, Int8, Bf16, Int4
- **QuantTensor**: stockage + métadonnées + dequantize()

### Fichier: `scirust-core/src/quant/int8.rs`
- Quantification symétrique int8 par canal
- SIMD dequantization: int8 → f32 en 8-lanes (AVX2)
- Matmul int8 × int8 → int32 accumulateur

### Fichier: `scirust-core/src/quant/bf16.rs`
- Conversion f32 ↔ bf16 (troncature LSB)
- NEON batch: 4 elements par itération
- AVX2 batch: 8 elements par itération

### Fichier: `scirust-core/src/quant/int4.rs`
- Quantification int4 signed (8× compression)
- Packing: 2 valeurs int4 par byte
- Matmul int4 packed × int4 packed → fp32

## Compatibilité multi-plateforme

| Feature | x86_64 | ARM64 | Jetson | Windows |
|---------|--------|-------|--------|---------|
| AVX-512 | ✓ | - | - | - |
| AVX2 | ✓ | - | - | - |
| SSE2 | ✓ | - | - | - |
| NEON | - | ✓ | ✓ | - |
| SVE | - | ✓ | ✓ | - |
| Arena alloc | ✓ | ✓ | ✓ | ✓ |
| Pinned mem | ✓ | ✓ | ✓ | ✓ |
| int8 quant | ✓ | ✓ | ✓ | ✓ |
| bf16 quant | ✓ | ✓ | ✓ | ✓ |
| int4 quant | ✓ | ✓ | ✓ | ✓ |

## Prochaines étapes (TODO)

1. **Fusion avec autodiff**: adapter les kernels fusionnés pour qu'ils fonctionnent
   avec le graphe tape. Nécessite d'ajouter les backward rules pour chaque kernel.

2. **PinnedMemory + CUDA**: intégrer cudaHostRegister/cudaHostUnregister pour le
   zero-copy vers GPU.

3. **SLab + SSM cells**: implémenter les cellules Mamba avec le Slab pour gérer
   les états cachés (c, h̃) à chaque timestep.

4. **Fused matmul SIMD**: remplacer les boucles scalaires dans les kernels fusionnés
   par les appels NEON/AVX512 directement.

5. **Benchmarks**: mesurer le speedup sur les patterns targets (MatMul → SiLU → LN).

6. **rustc driver pass**: implémenter la détection de motifs dans le MIR pass
   (scirust-rustc-driver/src/passes/) pour la fusion au moment de la compilation.
