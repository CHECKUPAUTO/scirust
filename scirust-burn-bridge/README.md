# scirust-burn-bridge

Inference bridge between [Burn](https://burn.dev/) (Rust deep learning framework) and the SciRust loops (non-differentiable algorithms: evolution, RL, MCTS, Monte-Carlo).

## Status

🟢 **v0.0.1 — functional skeleton** (Phase 0).

Stable API for the minimal use case: *"evaluate a `burn::Module` from a SciRust loop without the autodiff penalty"*.

Not yet stable:
- Parallel batched evaluation (rayon) — target v0.1
- Compile-time detection of the forbidden `Autodiff<_>` backend — target v0.1
- GPU support via Wgpu/Cuda — target v0.2 (the code is already generic over `Backend`, just needs testing)

## Quick reference

```rust
use scirust_burn_bridge::{InferenceOnly, Policy};
use burn::backend::NdArray;

type B = NdArray<f32>;

// 1. Implement Policy<B> for your network
impl<BB: Backend> Policy<BB> for MyMlp<BB> {
    type Input = Tensor<BB, 2>;
    type Output = Tensor<BB, 2>;
    fn forward(&self, input: Tensor<BB, 2>) -> Tensor<BB, 2> { /* ... */ }
}

// 2. Wrap and evaluate
let bridge = InferenceOnly::new(my_mlp, device);
let output = bridge.eval(input);
```

## Tests

```bash
cargo test -p scirust-burn-bridge          # unit + integration tests
cargo run --release -p scirust-burn-bridge --example eval_population
cargo run --release -p scirust-burn-bridge --bench forward
```

## Phase 0 performance target

≥ **1,000,000 forwards/s** single-threaded on a small MLP (4→8→2, NdArray, f32) on a modern CPU.

If not reached after optimizing the release profile, open an issue with the output of `bench forward` + `lscpu`.

## Philosophical guarantee

**This crate never tracks gradients.** If you see a use with `Autodiff<_>` anywhere, that's a bug.

## License

Apache-2.0 OR MIT (at the user's choice).
