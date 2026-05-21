# scirust-burn-bridge

Pont d'inférence entre [Burn](https://burn.dev/) (framework deep learning Rust) et les boucles SciRust (algorithmes non-différentiables : évolution, RL, MCTS, Monte-Carlo).

## Statut

🟢 **v0.0.1 — squelette fonctionnel** (Phase 0).

API stable pour le cas d'usage minimal : *« évaluer un `burn::Module` depuis une boucle SciRust sans la pénalité de l'autodiff »*.

Pas encore stable :
- Évaluation batchée parallèle (rayon) — cible v0.1
- Détection compile-time du backend `Autodiff<_>` interdit — cible v0.1
- Support GPU via Wgpu/Cuda — cible v0.2 (le code est déjà générique sur `Backend`, faut juste tester)

## Quick reference

```rust
use scirust_burn_bridge::{InferenceOnly, Policy};
use burn::backend::NdArray;

type B = NdArray<f32>;

// 1. Implémenter Policy<B> pour ton réseau
impl<BB: Backend> Policy<BB> for MyMlp<BB> {
    type Input = Tensor<BB, 2>;
    type Output = Tensor<BB, 2>;
    fn forward(&self, input: Tensor<BB, 2>) -> Tensor<BB, 2> { /* ... */ }
}

// 2. Wrapper et évaluer
let bridge = InferenceOnly::new(my_mlp, device);
let output = bridge.eval(input);
```

## Tests

```bash
cargo test -p scirust-burn-bridge          # tests unitaires + intégration
cargo run --release -p scirust-burn-bridge --example eval_population
cargo run --release -p scirust-burn-bridge --bench forward
```

## Cible de performance Phase 0

≥ **1 000 000 forwards/s** en single-thread sur petit MLP (4→8→2, NdArray, f32) sur un CPU moderne.

Si non atteint après optimisation du profil release, ouvrir une issue avec la sortie de `bench forward` + `lscpu`.

## Garantie philosophique

**Ce crate ne tracke jamais de gradient.** Si tu vois un usage avec `Autodiff<_>` quelque part, c'est un bug.

## Licence

Apache-2.0 OR MIT (au choix de l'utilisateur).
