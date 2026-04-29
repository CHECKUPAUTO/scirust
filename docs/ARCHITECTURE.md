# Architecture — How the autograd tape works

## The big picture

SciRust uses **reverse-mode autograd** with an explicit tape, like PyTorch's
older `Variable` API or modern Zygote.jl. The key idea: every operation
records a node on a tape, and `backward()` walks the tape in reverse to
compute gradients via the chain rule.

```
         Forward                    Backward
         ───────                    ────────

x = tape.input(...)                 ←  grad of x accumulated last
y = x * 2                           ←  grad of y → grad of x via chain rule
z = y + 1                           ←  grad of z → grad of y via chain rule
loss = z.sum()                      ←  grad of loss = 1 (scalar seed)
loss.backward()                     ↑  walk reverse, propagate

let g = tape.grad(x.idx());
```

## Components

### `Tape`

A `Tape` owns three parallel `Vec`s indexed by node ID:

```rust
struct Tape {
    nodes:  RefCell<Vec<Node>>,        // op + saved data per node
    values: RefCell<Vec<DeviceTensor>>, // forward result of each node (CPU or GPU)
    grads:  RefCell<Vec<Tensor>>,      // gradient buffer (always CPU in v9)
}
```

A `Node` records the op variant and any saved tensors needed for backward:

```rust
struct Node {
    op:    Op,
    shape: (usize, usize),
    saved: SavedData,    // mask, indices, im2col GPU tensor, etc.
}
```

### `Var<'t>`

A `Var<'t>` is a lightweight handle into the tape. The `'t` lifetime ties
it to the tape it came from, preventing dangling references at compile time.

```rust
pub struct Var<'t> {
    tape: &'t Tape,
    idx:  usize,
}
```

You don't store `Var`s long-term. They're created during a forward pass,
consumed by ops, and discarded when the tape is dropped.

### `Op`

`Op` is the enum of every operation that can be on the tape:

```rust
pub enum Op {
    Input,
    Add(usize, usize),       // (lhs, rhs) indices in the tape
    Mul(usize, usize),
    MatMul(usize, usize),
    Relu(usize),
    Conv2dForward { input_idx, weight_idx, bias_idx, config },
    MaxAxis(usize, u8),
    // ... ~30 ops total
}
```

Each variant stores indices (`usize`) into the tape, not references. This
keeps the Op `Copy` and avoids self-referential structs.

## Forward pass — how it builds the tape

When you call `x.relu()`:

```rust
impl<'t> Var<'t> {
    pub fn relu(self) -> Var<'t> {
        // 1. Read current value
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();

        // 2. Compute output
        let mut out = a.clone();
        for x in out.data.iter_mut() { *x = x.max(0.0); }

        // 3. Push a new node on the tape
        let idx = self.tape.push(
            Op::Relu(self.idx),                  // remember which op + parent
            DeviceTensor::cpu(out),              // store the result
        );

        // 4. Return a new Var pointing at the new node
        Var { tape: self.tape, idx }
    }
}
```

Notice that `relu` *consumes* `self` (takes by value). This forces a linear
flow: each Var is used at most once unless explicitly cloned. It's the same
pattern Rust's iterators use.

## Backward pass — how it propagates gradients

`loss.backward()` does this:

```rust
pub fn backward(&self) {
    // 1. Seed: grad of loss = ones
    let n = self.tape.nodes.borrow().len();
    self.tape.grads.borrow_mut()[self.idx] = Tensor::ones(...);

    // 2. Walk in reverse topological order (= reverse insertion order
    //    because we always push children after parents)
    for i in (0..n).rev() {
        let op = self.tape.nodes.borrow()[i].op.clone();
        let grad_out = self.tape.grads.borrow()[i].clone();

        propagate(self.tape, i, op, &grad_out);
    }
}
```

`propagate` is a big match on `Op` that distributes the gradient to parents:

```rust
fn propagate(tape: &Tape, i: usize, op: Op, grad_out: &Tensor) {
    match op {
        Op::Add(a, b) => {
            // d(a+b)/da = 1, d(a+b)/db = 1
            accumulate(tape, a, grad_out);
            accumulate(tape, b, grad_out);
        }
        Op::Mul(a, b) => {
            // d(a*b)/da = b, d(a*b)/db = a
            let a_val = tape.values.borrow()[a].as_cpu().clone();
            let b_val = tape.values.borrow()[b].as_cpu().clone();
            // grad_a = grad_out * b_val  (Hadamard)
            let mut grad_a = grad_out.clone();
            for j in 0..grad_a.data.len() { grad_a.data[j] *= b_val.data[j]; }
            accumulate(tape, a, &grad_a);
            // ... same for b
        }
        Op::Relu(a) => {
            let a_val = tape.values.borrow()[a].as_cpu().clone();
            let mut grad_a = grad_out.clone();
            for j in 0..grad_a.data.len() {
                if a_val.data[j] <= 0.0 { grad_a.data[j] = 0.0; }
            }
            accumulate(tape, a, &grad_a);
        }
        // ... ~30 more match arms, one per op
    }
}
```

`accumulate` adds the incoming gradient to the existing one (because a node
might be used by multiple downstream ops):

```rust
fn accumulate(tape: &Tape, idx: usize, grad: &Tensor) {
    let mut grads = tape.grads.borrow_mut();
    for i in 0..grads[idx].data.len() {
        grads[idx].data[i] += grad.data[i];
    }
}
```

## SavedData — auxiliary storage per node

Some ops need information beyond their inputs and output for the backward
pass. Examples:

- **Dropout** keeps a binary mask
- **MaxAxis** stores the argmax indices
- **Conv2d on GPU** keeps the im2col tensor in VRAM

We provide a closed enum `SavedData`:

```rust
pub enum SavedData {
    None,
    Tensor(Tensor),
    Mask { bits: Vec<u8>, len: usize },     // 1 bit per element
    Indices(Vec<u32>),
    GpuTensor(Arc<GpuTensor>),               // VRAM-resident
}
```

This is type-safe (no `Box<dyn Any>` downcast), memory-efficient (Mask saves
32× over Vec<f32>), and explicitly limited (can't store arbitrary types).

## DeviceTensor — CPU/GPU mix

Since v8, the tape can store either CPU or GPU tensors:

```rust
pub enum DeviceTensor {
    Cpu(Tensor),
    Gpu(Arc<GpuTensor>),
}
```

CPU ops use `.as_cpu()` to extract their input. If a GPU tensor is passed by
mistake, they panic with a clear message rather than producing wrong results.
Transitions are explicit: `var.to_gpu(&ctx)` or `var.to_cpu(&ctx)`.

## Lifetime discipline

The `'t` lifetime on `Var<'t>` means you cannot:
- Use a Var after its tape is dropped (compile error)
- Mix Vars from different tapes in the same op (compile error)

This makes data parallelism trivial: each thread owns its tape, and the
compiler statically prevents cross-thread Var leaks.

## What's not on the tape

- **The optimizer** (Adam, SGD) reads gradients and updates weights, but
  doesn't add nodes. It works on `Vec<usize>` of parameter indices.
- **Running statistics** in BatchNorm. These update on the module
  imperatively, not through the autograd graph (they're not differentiable).
- **Random masks** in Dropout. The mask is generated, stored in
  `SavedData::Mask`, and applied — the AD treats it as a constant.

## Common patterns

### Custom layer

Implement the `Module` trait:

```rust
struct MyLayer { weight: Tensor, last_w_idx: Option<usize> }

impl Module for MyLayer {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let w = tape.input(self.weight.clone());
        self.last_w_idx = Some(w.idx());
        // ... ops using w and input
    }
    fn parameter_indices(&self) -> Vec<usize> {
        self.last_w_idx.into_iter().collect()
    }
    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx { self.weight = tape.value(i); }
    }
    // ...
}
```

### Custom loss

Implement the `Loss` trait:

```rust
impl Loss for MyLoss {
    fn forward<'t>(&self, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        // build the loss expression on the tape, return scalar Var
    }
}
```

The autograd handles the gradient automatically — you never write a custom
backward unless you're adding a new low-level op.

## Performance notes

- **Tape allocation is cheap**: a new `Tape::new()` per training step is fine.
- **Cloning a Var is cheap**: just copies a pointer + index.
- **Cloning a Tensor is expensive**: it copies all the data. Avoid in hot paths.
- **Gradients are always CPU** in v9. GPU activation chains are supported,
  but the backward eventually populates CPU grad buffers.
