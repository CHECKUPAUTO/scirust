# Quickstart — Train your first model

## Goal

Train a 2-class classifier on synthetic 2D data in **5 minutes**.

## What you'll learn

1. How to construct a model with `Sequential`
2. How to define and run a forward pass
3. How to compute gradients with `backward()`
4. How to update weights with an optimizer

## The full example

Save this as `examples/quickstart/src/main.rs`:

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear, ReLU,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::{Loss, strict::CrossEntropyLoss};

fn main() {
    // Random number generator with fixed seed for reproducibility
    let mut rng = PcgEngine::new(42);

    // 2 → 8 → 2 MLP
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng).with_name("fc1"))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng).with_name("fc2"));

    // 4 toy points: two clusters
    let x_train = Tensor::from_vec(
        vec![1.0, 1.0,    // class 0
             2.0, 2.0,    // class 0
            -1.0, -1.0,   // class 1
            -2.0, -2.0],  // class 1
        4, 2,
    );
    let y_train = Tensor::from_vec(
        vec![1.0, 0.0,   // one-hot class 0
             1.0, 0.0,
             0.0, 1.0,
             0.0, 1.0],
        4, 2,
    );

    let mut optimizer = Adam::new(0.05);

    for epoch in 0..100 {
        // Each step: build a fresh tape (the AD record)
        let tape = Tape::new();
        let xv = tape.input(x_train.clone());
        let yv = tape.input(y_train.clone());

        // Forward: model produces logits
        let logits = model.forward(&tape, xv);

        // Loss
        let loss = CrossEntropyLoss.forward(logits, yv);

        // Backward: populates gradients on the tape
        loss.backward();

        // Optimizer step: updates the weights using the gradients
        optimizer.step(&model.parameter_indices(), &tape);

        // Sync: copy updated weights from tape back into the model
        model.sync(&tape);

        if epoch % 20 == 0 {
            let loss_val = tape.value(loss.idx()).data[0];
            println!("epoch {epoch:3}: loss = {loss_val:.4}");
        }
    }

    // Inference on the same data
    let tape = Tape::new();
    let xv = tape.input(x_train.clone());
    let logits = model.forward(&tape, xv);
    let probs = tape.value(logits.idx());

    println!("\nFinal predictions:");
    for i in 0..4 {
        let row = &probs.data[i*2..(i+1)*2];
        let pred_class = if row[0] > row[1] { 0 } else { 1 };
        println!("  point ({:.1}, {:.1}) → class {pred_class} (logits: {row:?})",
                 x_train.data[i*2], x_train.data[i*2+1]);
    }
}
```

## Run it

```bash
cargo run --release
```

Expected output:

```
epoch   0: loss = 0.7234
epoch  20: loss = 0.1281
epoch  40: loss = 0.0234
epoch  60: loss = 0.0080
epoch  80: loss = 0.0036

Final predictions:
  point (1.0, 1.0) → class 0 (logits: [3.2, -3.1])
  point (2.0, 2.0) → class 0 (logits: [5.1, -4.9])
  point (-1.0, -1.0) → class 1 (logits: [-3.0, 3.3])
  point (-2.0, -2.0) → class 1 (logits: [-4.8, 5.0])
```

## Key concepts

**Tape** — The "tape" is the autograd record. Every operation on a `Var`
adds a node. Calling `loss.backward()` walks the tape backwards, computing
gradients via the chain rule.

**Var** vs **Tensor** — `Var<'t>` is a handle into the tape (an index +
a lifetime). `Tensor` is the actual data buffer. You access `Tensor` via
`tape.value(var.idx())` when needed.

**Sequential** — A simple way to chain layers. Each layer implements the
`Module` trait. You can also write your own custom layers.

**One tape per step** — Tapes don't accumulate across batches. Build a
new `Tape::new()` for each forward pass, run backward, optimizer step,
sync, then drop. The tape is cheap to construct.

## Next steps

- [`MNIST.md`](MNIST.md) — Train on real MNIST data with a CNN
- [`GPU.md`](GPU.md) — Move computation to GPU with one method call
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — How the autograd tape works internally
