# LIVESTATE — scirust

> Fichier de bord partagé entre agents.
> Dernière mise à jour : 2026-06-10

## HEAD
- **Hash:** `cf738ae`
- **Message:** sync: scirust-arena + fused_ops + memory wall docs + perf fixes
- **Auteur:** Claude Code (via Ollama)
- **Date:** 2026-06-08

## Branche active
- master

## Statut
- Workspace Rust 14+ crates (scirust-tn ajouté au workspace)
- 334 scirust-core tests pass, 22 scirust-simd tests pass, 14 scirust-tn tests pass
- CI/CD pipeline (fmt, clippy, build-test, deny, coverage) — Phase 0 terminée
- GPU autodiff (GpuEngine trait, MatMulGpu backward) — Phase 2a terminée
- TT-Linear autograd through cores (Op::TtContract) — Phase 2b terminée

## Changements récents (session 2026-06-10)
### Phase 0 — CI/Linting
- rustfmt.toml, clippy.toml, deny.toml créés
- Edition 2021→2024, modules manquants supprimés, doctests réparés
- Safety docs sur neon.rs, portable.rs, dispatch.rs

### Phase 3 — SGEMV kernels
- AVX2, SSE2, NEON SGEMV kernels dans scirust-simd (plus de fallback ScalarBackend)
- 22 tests cross-backend (exhaustif 0–20, known-value, tall/wide, non-puissance-de-2)
- Property tests TT-Linear (forward match dense)

### Phase 4 — Benchmarks
- TT-Linear vs dense Linear forward benchmark (examples/benchmarks/benches/nn_ops.rs)

### Bug fixes TT-Linear
- `register_params` : toujours re-enregistrer (pas de cache stale)
- `forward` : `out.add(b_var)` → `out.add_bias(b_var)`
- `impl TTLinear` : accolade manquante (rustfmt avait drop)
- `#[derive(Clone)]` sur TTLinear

### assert! → Result migration
- 16 `try_*` variants sur Var (try_add, try_matmul, try_softmax, try_conv2d, try_layer_norm, etc.)
- Tous les modules nn migrés (linear, lstm, gnn, attention, conv2d, batch_norm, loss, etc.)
- Les méthodes originales sont des wrappers thin (delegate → unwrap)

### TT code dedup
- scirust-tn dépend maintenant de scirust-core (non-optionnel)
- 4 modules dupliqués supprimés : tensor.rs, factorize.rs, ops/, tt/decompose.rs (~950 lignes)
- scirust-tn réexporte depuis scirust_core::tn
- scirust-tn ajouté au workspace

### TT-Linear Phase 2 — On-tape contraction avec gradient cores
- Nouveau `Op::TtContract` : opération fusionnée qui reconstruit W, calcule x@W+b, sauve x+W
- Backward exact : interleave(grad_W) → projection left-right par core → gradients cores
- `Var::try_tt_contract` / `Var::tt_contract` : interface publique
- `TTLinear::forward` réécrit pour utiliser `tt_contract` on-tape
- Test `tt_backward_gradient_flows` activé et passe — gradients flow through cores
- `interleave_weight` rendu pub(crate) pour réutilisation dans le backward

## Notes
- mixed FR/EN comments dans le code
- Cargo.lock dans .gitignore mais versionné
- Nightly-only (portable_simd)
- TT d>2 backward non encore implémenté (gradients zéro pour d>2 — cas rare)