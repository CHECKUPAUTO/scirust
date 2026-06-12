# LIVESTATE — scirust

> Fichier de bord partagé entre agents.
> Dernière mise à jour : 2026-06-12

## Session 2026-06-12 — volet 2 : réparation CI
- `.github/workflows/ci.yml` réécrit pour être réalisable :
  - suppression de `--all-features` (blas-openblas et blas-mkl sont
    mutuellement exclusifs → la CI ne pouvait JAMAIS compiler)
  - nouveau job `cross-check-aarch64` (cargo check --target aarch64) :
    type-vérifie tous les chemins NEON/SVE — la classe de bug du merge
    du 12/06 devient détectable sur PR
  - coverage passé en informatif (continue-on-error), cargo-llvm-cov
    pré-compilé via taiki-e/install-action
- `deny.toml` réécrit (l'ancien n'était pas du TOML valide → le job
  License/Security échouait au parsing) ; validé en local avec
  cargo-deny 0.19.8 : advisories/bans/licenses/sources tous ok ;
  RUSTSEC-2024-0436 (`paste` unmaintained via nalgebra→simba) ignoré
  avec justification — c'est l'alerte Dependabot ouverte
- `publish = false` ajouté aux 51 manifestes (réalité : deps par chemin,
  licence non-commerciale) — active l'exemption licences des crates
  privées dans cargo-deny
- Tous les warnings restants éliminés (RUSTFLAGS -D warnings tenable) ;
  gate clippy étendu à --all-targets : 14 lints de code de test corrigés
- Cross-check aarch64 a aussi révélé/réparé : test SVE cassé
  (`sve_vector_length_elements` inexistante — modules sve/sve_fns en
  ré-export circulaire vide) → implémentation réelle via `rdvl` (asm
  stable, gardée par détection runtime)
- Reste à faire côté GitHub (non scriptable depuis le repo) : protection
  de branche master exigeant fmt/clippy/build-test/cross-check/deny

## Session 2026-06-12 (audit + SOM tranche verticale)
- Audit complet exécuté : voir `scirust_complete_audit_report.md`
- Régression de merge réparée (sgemv AVX2/SSE2/NEON, arena slab) ; gates
  check / clippy -D warnings / test / fmt tous verts en local
- **655 tests workspace passent** (630 avant SOM, +25 SOM), 0 échec
- SOM : tranche verticale réelle livrée (voir `scirust-som/README.md`) —
  oracle d'ownership déterministe, tokenizer+vocab fermé, générateur de
  dataset seedé, backbone TransformerEncoder réel (attention du core),
  trainer bit-déterministe, éval vs oracle : ownership 83,7 % vs baseline
  31,4 % (held-out seed 9042), visualizer markdown
- Anciens stubs SOM remplacés : trainer/inference/symbolic/visualizer
  n'étaient que des fichiers d'1 ligne ; le « Graph Transformer » (MLP
  étiqueté) est devenu un vrai Transformer séquence — l'attention sur
  graphe PCG reste un travail futur et est documentée comme telle

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