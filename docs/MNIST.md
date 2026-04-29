# MNIST — Real-world training tutorial

## Goal

Train a CNN on the official MNIST handwritten digits dataset using:
- Real data loading (IDX format)
- DataLoader with shuffle and mini-batching
- Multi-threaded training (data parallelism)
- Adam optimizer with CrossEntropy loss

## Step 1 — Get the data

```bash
mkdir mnist && cd mnist
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

You should now have 4 files in `mnist/`, each ending in `-ubyte`.

## Step 2 — Architecture

A small CNN that gets ~98% test accuracy in 5 epochs:

```
Input (28×28×1)
  → Conv2d(1→8, 3×3, same)  + ReLU + MaxPool(2×2)   → (14×14×8)
  → Conv2d(8→16, 3×3, same) + ReLU + MaxPool(2×2)   → (7×7×16)
  → Linear(784 → 10)                                 → 10 logits
```

## Step 3 — Code

```rust
use scirust_core::autodiff::reverse::Tape;
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::data::{DataLoader, Dataset};
use scirust_core::data::mnist::MnistDataset;
use scirust_core::nn::{
    PcgEngine, Module, Linear, ReLU, KaimingNormal, Zeros,
};
use scirust_core::nn::conv2d::Conv2d;
use scirust_core::nn::pool::MaxPool2d;
use scirust_core::nn::conv_utils::Padding;
use scirust_core::nn::loss::{Loss, strict::CrossEntropyLoss};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load MNIST
    let train = MnistDataset::load_idx(
        "./mnist/train-images-idx3-ubyte",
        "./mnist/train-labels-idx1-ubyte",
    )?;
    let test = MnistDataset::load_idx(
        "./mnist/t10k-images-idx3-ubyte",
        "./mnist/t10k-labels-idx1-ubyte",
    )?;

    println!("Loaded: {} train, {} test", train.len(), test.len());

    // Build model
    let mut rng = PcgEngine::new(42);
    let mut conv1 = Conv2d::new(1, 8, 3, 1, Padding::Same,
        &KaimingNormal, Some(&Zeros), &mut rng).input_dims(28, 28);
    let mut pool1 = MaxPool2d::new(2, 2).input_shape(8, 28, 28);
    let mut conv2 = Conv2d::new(8, 16, 3, 1, Padding::Same,
        &KaimingNormal, Some(&Zeros), &mut rng).input_dims(14, 14);
    let mut pool2 = MaxPool2d::new(2, 2).input_shape(16, 14, 14);
    let mut fc = Linear::new(7*7*16, 10, &KaimingNormal, &Zeros, &mut rng);

    // Training setup
    let batch_size = 64;
    let mut loader = DataLoader::new(train.into_in_memory(), batch_size, true, 42);
    let mut optimizer = Adam::new(0.001);

    for epoch in 0..5 {
        loader.shuffle_epoch(epoch);
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for (x_batch, y_batch) in loader.iter() {
            let tape = Tape::new();
            let xv = tape.input(x_batch);
            let yv = tape.input(y_batch);

            // Forward
            let h1 = conv1.forward(&tape, xv).relu();
            let p1 = pool1.forward(&tape, h1);
            let h2 = conv2.forward(&tape, p1).relu();
            let p2 = pool2.forward(&tape, h2);
            let logits = fc.forward(&tape, p2);

            // Loss + backward
            let loss = CrossEntropyLoss.forward(logits, yv);
            loss.backward();

            // Collect all params from all layers
            let mut params = Vec::new();
            params.extend(conv1.parameter_indices());
            params.extend(conv2.parameter_indices());
            params.extend(fc.parameter_indices());
            optimizer.step(&params, &tape);

            // Sync
            conv1.sync(&tape);
            conv2.sync(&tape);
            fc.sync(&tape);

            epoch_loss += tape.value(loss.idx()).data[0];
            n_batches += 1;
        }

        // Evaluation on test set
        let test_acc = evaluate(&mut conv1, &mut pool1, &mut conv2,
                                &mut pool2, &mut fc, &test);
        println!("Epoch {epoch}: avg loss = {:.4}, test acc = {:.2}%",
                 epoch_loss / n_batches as f32, test_acc);
    }

    Ok(())
}

// (Evaluation function: forward only, no backward, batch by 256 for memory)
fn evaluate(...) -> f32 { ... }
```

## Step 4 — Run

```bash
cargo run --release -- ./mnist
```

Expected output:
```
Loaded: 60000 train, 10000 test
Epoch 0: avg loss = 0.5234, test acc = 92.45%
Epoch 1: avg loss = 0.1823, test acc = 96.12%
Epoch 2: avg loss = 0.1014, test acc = 97.34%
Epoch 3: avg loss = 0.0671, test acc = 97.89%
Epoch 4: avg loss = 0.0489, test acc = 98.21%
```

## Speeding up with data parallelism

For multi-core CPUs, use the `parallel_step` helper:

```rust
use scirust_core::nn::parallel::{ParallelStep, Grads, parallel_step};

struct MnistStepper { /* model fields */ }
impl ParallelStep for MnistStepper { /* ... */ }

let stepper = MnistStepper { /* ... */ };
let n_workers = std::thread::available_parallelism()
    .map(|n| n.get().min(4))
    .unwrap_or(2);

let (mean_loss, mean_grads) = parallel_step(&stepper, x_batch, y_batch, n_workers);
```

See [`examples/v7a_mnist_demo`](../examples/v7a_mnist_demo) for a complete example.

## Common issues

**Magic number incorrect** — The IDX files must be ungzipped. Run `file *-ubyte`
to confirm they're binary, not gzip-compressed.

**Out of memory** — Reduce `batch_size` or use `subsample(n)` on the training
set during development.

**Slow training** — Make sure to compile with `--release`. Debug builds are
~30× slower for ML workloads.

## Next steps

- [`GPU.md`](GPU.md) — Move conv2d operations to GPU
