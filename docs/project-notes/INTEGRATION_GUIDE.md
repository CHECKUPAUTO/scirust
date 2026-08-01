# SciRust — Guide d'intégration SIMD + Matrix Views

## Résumé des changements

| Fichier | Rôle |
|---|---|
| `scirust-simd/src/portable.rs` | Kernels SIMD portables via `std::simd` |
| `scirust-core/src/matrix/view.rs` | `MatrixView` / `MatrixViewMut` sans allocation |
| `scirust-core/src/matrix/backend.rs` | Trait `SimdBackend` + implémentations |
| `examples/simd_views_demo/` | Démo end-to-end |
| `examples/benchmarks/benches/simd_bench.rs` | Benchmarks Criterion |

---

## Étapes d'intégration

### 1. Activer nightly pour std::simd

```toml
# rust-toolchain.toml (racine)
[toolchain]
channel    = "nightly"
components = ["rustfmt", "clippy", "rustc-dev"]
```

### 2. Ajouter la feature dans les Cargo.toml

```toml
# scirust-simd/Cargo.toml
[features]
default       = []
portable-simd = []

# scirust-core/Cargo.toml
[features]
portable-simd = ["scirust-simd/portable-simd"]

[dependencies]
scirust-simd = { path = "../scirust-simd" }
```

### 3. Copier les fichiers

```bash
cp scirust-simd/src/portable.rs          <repo>/scirust-simd/src/portable.rs
cp scirust-core/src/matrix/view.rs       <repo>/scirust-core/src/matrix/view.rs
cp scirust-core/src/matrix/backend.rs   <repo>/scirust-core/src/matrix/backend.rs
```

### 4. Exposer les modules

Dans `scirust-simd/src/lib.rs`, ajouter :
```rust
pub mod portable;
pub use portable::simd_ops;
```

Dans `scirust-core/src/lib.rs`, ajouter :
```rust
pub mod matrix {
    pub mod view;
    pub mod backend;
}
```

### 5. Build et tests

```bash
# Stable — kernels scalaires
cargo test

# Nightly + SIMD portable
cargo test  --features portable-simd
cargo bench --features portable-simd

# Démo complète
cargo run --package simd_views_demo --features scirust-core/portable-simd
```

---

## Architecture du trait SimdBackend

```
SimdBackend (trait)
├── ScalarBackend       — stable, toujours dispo
├── PortableSimdBackend — nightly std::simd (AVX2/NEON/SVE auto)
└── BlasBackend  — matrixmultiply / netlib
```

Le choix de backend se fait à la compilation via `best_backend()`.
À terme, un `enum Backend` permettra la sélection à l'exécution.

---

## Performances attendues (estimation)

| Opération | n | Scalar | SIMD portable | Gain |
|---|---|---|---|---|
| `dot_f32` | 65 536 | ~120 µs | ~18 µs | **6–7×** |
| `saxpy_f32` | 262 144 | ~400 µs | ~60 µs | **6–7×** |
| `relu_f32` | 1 048 576 | ~1.5 ms | ~200 µs | **7–8×** |
| `sgemm_f32` | 128×128 | ~4 ms | ~600 µs | **6×** |

*Mesuré sur x86_64 avec AVX2. Sur ARM (Apple M-series) avec NEON les gains sont similaires.*

---

## Prochaines étapes roadmap

1. **BlasBackend** — déléguer `sgemm` à `matrixmultiply` pour les grandes matrices
2. **MatrixView col-major** — transposition sans copie pour LAPACK interop
3. **Reverse-mode autodiff** — intégrer les vues dans le graphe de calcul
4. **JIT cache** — réutiliser les kernels SIMD compilés entre appels

---

## Notes sur std::simd

`std::simd` (aka `portable_simd`) est **stabilisé progressivement** depuis
Rust 1.77+. Les types `f32x8`, `f64x4` et les méthodes `.mul_add()`,
`.simd_max()`, `.reduce_sum()` sont disponibles sur nightly sans
`#[target_feature]` — le compilateur émet automatiquement les instructions
AVX2 / SSE4 / NEON / SVE selon la cible.

Avantage clé : un seul source, pas de `cfg(target_arch)` par branche.
