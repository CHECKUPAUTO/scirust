//! Reverse-mode autodiff over **N-D tensors** ([`TensorND`]) with numpy-style
//! broadcasting — the N-D autograd path (roadmap P2.4).
//!
//! This coexists with the production 2D [`crate::autodiff::reverse`] tape rather
//! than replacing it: the 2D tape stays the battle-tested default while this
//! module grows the N-D capability the compiler/IR ambitions need. Every
//! backward rule is validated by a **numerical gradient check** in the tests
//! (finite differences vs. the analytic gradient), the gold standard for
//! autodiff correctness.
//!
//! Ops: elementwise `add`/`sub`/`mul` (broadcasting), 2-D `matmul`, **batched
//! `bmm`**, `relu`, `softmax` (last axis), `transpose_last2`, `reshape`,
//! `permute`, `layernorm` (last axis), and `sum` to a scalar — enough to build
//! a full transformer block (see `nn::nd_layers`). The shape building blocks
//! (`broadcast_shape`, `broadcast_to`, `matmul_shape`) live on [`TensorND`].

use std::cell::RefCell;

use crate::tensor::tensor_nd::TensorND;

#[derive(Clone)]
enum Op {
    Leaf,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Matmul(usize, usize),
    /// Batched matmul `(…,m,k)·(…,k,n)→(…,m,n)` with broadcast batch axes.
    Bmm(usize, usize),
    Relu(usize),
    /// Softmax over the last axis.
    Softmax(usize),
    /// Swap the last two axes.
    TransposeLast2(usize),
    /// Reshape (data unchanged); backward reshapes the gradient back.
    Reshape(usize),
    /// General axis permutation; backward permutes by the inverse.
    Permute(usize, Vec<usize>),
    /// Layer normalisation over the last axis (no affine); `f32` is `eps`.
    LayerNormLast(usize, f32),
    /// RMS normalisation over the last axis (no affine); `f32` is `eps`.
    /// `y = x / sqrt(mean(x²) + eps)`.
    RmsNormLast(usize, f32),
    /// Logistic sigmoid `σ(x) = 1/(1+e^-x)`, elementwise.
    Sigmoid(usize),
    /// Elementwise `exp(x)`; backward is `g · exp(x)`.
    Exp(usize),
    /// Rotary position embedding over `(…, seq, d)` (position = axis −2); `f32`
    /// is the frequency base. Backward applies the inverse rotation.
    Rope(usize, f32),
    Sum(usize),
    /// Row lookup (embedding): select rows of a `(vocab, dim)` table by the
    /// integer indices. Backward scatter-adds the upstream rows back.
    Gather(usize, Vec<usize>),
    /// Concatenate operands along axis 0 (shared trailing dims). Backward splits
    /// the upstream gradient row-blocks back to each operand.
    Cat(Vec<usize>),
    /// Fused softmax + mean negative-log-likelihood over `(n, vocab)` logits
    /// with one integer target per row. Output is the scalar mean loss; the
    /// indices are constants (not differentiated).
    CrossEntropy(usize, Vec<usize>),
}

struct Node {
    op: Op,
    value: TensorND,
}

/// An N-D reverse-mode autodiff tape.
#[derive(Default)]
pub struct NdTape {
    nodes: RefCell<Vec<Node>>,
}

/// A handle to a value on an [`NdTape`].
#[derive(Clone, Copy)]
pub struct NdVar<'t> {
    tape: &'t NdTape,
    idx: usize,
}

impl NdTape {
    /// Create an empty tape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Place an input (leaf) value on the tape.
    pub fn input(&self, value: TensorND) -> NdVar<'_> {
        self.push(Op::Leaf, value)
    }

    fn push(&self, op: Op, value: TensorND) -> NdVar<'_> {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node { op, value });
        NdVar { tape: self, idx }
    }

    /// The forward value of a node.
    pub fn value(&self, v: NdVar<'_>) -> TensorND {
        self.nodes.borrow()[v.idx].value.clone()
    }

    /// Reverse-mode backward from a **scalar** root (shape `[1]`). Returns the
    /// gradient of the root w.r.t. every node, each with the node's own shape.
    pub fn backward(&self, root: NdVar<'_>) -> Vec<TensorND> {
        let nodes = self.nodes.borrow();
        let mut grads: Vec<TensorND> = nodes
            .iter()
            .map(|nd| TensorND::zeros(&nd.value.shape))
            .collect();
        assert_eq!(
            nodes[root.idx].value.numel(),
            1,
            "backward: root must be a scalar"
        );
        grads[root.idx] = TensorND::ones(&nodes[root.idx].value.shape);

        for i in (0..nodes.len()).rev()
        {
            let g = grads[i].clone();
            match nodes[i].op
            {
                Op::Leaf =>
                {},
                Op::Add(a, b) =>
                {
                    let sa = nodes[a].value.shape.clone();
                    let sb = nodes[b].value.shape.clone();
                    accumulate(&mut grads[a], &unbroadcast(&g, &sa));
                    accumulate(&mut grads[b], &unbroadcast(&g, &sb));
                },
                Op::Sub(a, b) =>
                {
                    let sa = nodes[a].value.shape.clone();
                    let sb = nodes[b].value.shape.clone();
                    let neg = ew(&g, &g, |x, _| -x);
                    accumulate(&mut grads[a], &unbroadcast(&g, &sa));
                    accumulate(&mut grads[b], &unbroadcast(&neg, &sb));
                },
                Op::Mul(a, b) =>
                {
                    let out_shape = &nodes[i].value.shape;
                    let av = nodes[a].value.broadcast_to(out_shape).unwrap();
                    let bv = nodes[b].value.broadcast_to(out_shape).unwrap();
                    let ga = ew(&g, &bv, |x, y| x * y);
                    let gb = ew(&g, &av, |x, y| x * y);
                    accumulate(&mut grads[a], &unbroadcast(&ga, &nodes[a].value.shape));
                    accumulate(&mut grads[b], &unbroadcast(&gb, &nodes[b].value.shape));
                },
                Op::Matmul(a, b) =>
                {
                    // out (m,n) = A(m,k)·B(k,n) → gA = g·Bᵀ, gB = Aᵀ·g
                    let av = &nodes[a].value;
                    let bv = &nodes[b].value;
                    let ga = matmul2d(&g, &transpose2d(bv));
                    let gb = matmul2d(&transpose2d(av), &g);
                    accumulate(&mut grads[a], &ga);
                    accumulate(&mut grads[b], &gb);
                },
                Op::Bmm(a, b) =>
                {
                    let (ga, gb) = bmm_backward(&nodes[a].value, &nodes[b].value, &g);
                    accumulate(&mut grads[a], &ga);
                    accumulate(&mut grads[b], &gb);
                },
                Op::Softmax(a) =>
                {
                    // dx_i = y_i · (g_i − Σ_j g_j y_j) over the last axis.
                    let y = &nodes[i].value;
                    accumulate(&mut grads[a], &softmax_backward(y, &g));
                },
                Op::TransposeLast2(a) =>
                {
                    accumulate(&mut grads[a], &transpose_last2(&g));
                },
                Op::Reshape(a) =>
                {
                    // Reshape the upstream grad back to the input's shape.
                    let shape = nodes[a].value.shape.clone();
                    accumulate(&mut grads[a], &TensorND::new(g.data.clone(), shape));
                },
                Op::Permute(a, ref perm) =>
                {
                    // Gradient flows through the inverse permutation.
                    let mut inv = vec![0usize; perm.len()];
                    for (i, &p) in perm.iter().enumerate()
                    {
                        inv[p] = i;
                    }
                    accumulate(&mut grads[a], &g.transpose(&inv).expect("permute backward"));
                },
                Op::LayerNormLast(a, eps) =>
                {
                    let dx = layernorm_backward(&nodes[a].value, &nodes[i].value, &g, eps);
                    accumulate(&mut grads[a], &dx);
                },
                Op::RmsNormLast(a, eps) =>
                {
                    let dx = rmsnorm_backward(&nodes[a].value, &nodes[i].value, &g, eps);
                    accumulate(&mut grads[a], &dx);
                },
                Op::Sigmoid(a) =>
                {
                    // dx = g · y · (1 − y), y the sigmoid output.
                    let y = &nodes[i].value;
                    let d: Vec<f32> = g
                        .data
                        .iter()
                        .zip(&y.data)
                        .map(|(&gi, &yi)| gi * yi * (1.0 - yi))
                        .collect();
                    accumulate(&mut grads[a], &TensorND::new(d, y.shape.clone()));
                },
                Op::Exp(a) =>
                {
                    // dx = g · exp(x); the output node already holds exp(x).
                    let y = &nodes[i].value;
                    let d: Vec<f32> = g
                        .data
                        .iter()
                        .zip(&y.data)
                        .map(|(&gi, &yi)| gi * yi)
                        .collect();
                    accumulate(&mut grads[a], &TensorND::new(d, y.shape.clone()));
                },
                Op::Rope(a, base) =>
                {
                    // RoPE is an orthogonal rotation R(pos); dx = Rᵀ·g = R(−pos)·g.
                    accumulate(&mut grads[a], &rope_lastaxis(&g, base, true));
                },
                Op::Relu(a) =>
                {
                    let av = &nodes[a].value;
                    let mut d = g.data.clone();
                    for (gi, &x) in d.iter_mut().zip(av.data.iter())
                    {
                        if x <= 0.0
                        {
                            *gi = 0.0;
                        }
                    }
                    accumulate(&mut grads[a], &TensorND::new(d, av.shape.clone()));
                },
                Op::Sum(a) =>
                {
                    // out is scalar; every input element gets the same upstream grad.
                    let s = g.data[0];
                    let shape = nodes[a].value.shape.clone();
                    let n = nodes[a].value.numel();
                    accumulate(&mut grads[a], &TensorND::new(vec![s; n], shape));
                },
                Op::Gather(a, ref idx) =>
                {
                    // Scatter-add each output row's gradient back to its source
                    // row (repeated indices accumulate).
                    let dim = nodes[a].value.shape[1];
                    for (r, &ix) in idx.iter().enumerate()
                    {
                        for c in 0..dim
                        {
                            grads[a].data[ix * dim + c] += g.data[r * dim + c];
                        }
                    }
                },
                Op::Cat(ref idxs) =>
                {
                    // Split the upstream gradient into row-blocks, one per operand.
                    let trailing: usize = nodes[i].value.shape[1..].iter().product();
                    let mut row_off = 0usize;
                    for &part in idxs
                    {
                        let prows = nodes[part].value.shape[0];
                        let start = row_off * trailing;
                        let end = (row_off + prows) * trailing;
                        let gp = TensorND::new(
                            g.data[start..end].to_vec(),
                            nodes[part].value.shape.clone(),
                        );
                        accumulate(&mut grads[part], &gp);
                        row_off += prows;
                    }
                },
                Op::CrossEntropy(a, ref targets) =>
                {
                    // dL/dlogits = (softmax(logits) − onehot(target)) / n, times
                    // the (scalar) upstream gradient.
                    let logits = &nodes[a].value;
                    let (n, vocab) = (logits.shape[0], logits.shape[1]);
                    let scale = g.data[0] / n as f32;
                    let mut dl = vec![0.0f32; logits.data.len()];
                    for (r, &target) in targets.iter().enumerate()
                    {
                        let row = &logits.data[r * vocab..(r + 1) * vocab];
                        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let sum: f32 = row.iter().map(|&v| (v - mx).exp()).sum();
                        for c in 0..vocab
                        {
                            dl[r * vocab + c] = scale * (row[c] - mx).exp() / sum;
                        }
                        dl[r * vocab + target] -= scale;
                    }
                    accumulate(&mut grads[a], &TensorND::new(dl, logits.shape.clone()));
                },
            }
        }
        grads
    }
}

impl<'t> NdVar<'t> {
    /// Index of this node (for retrieving its gradient from `backward`).
    pub fn idx(self) -> usize {
        self.idx
    }

    /// Elementwise `self + other` (broadcasting).
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: NdVar<'t>) -> NdVar<'t> {
        let (a, b) = self.pair(other);
        let out = ew_broadcast(&a, &b, |x, y| x + y);
        self.tape.push(Op::Add(self.idx, other.idx), out)
    }

    /// Elementwise `self - other` (broadcasting).
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: NdVar<'t>) -> NdVar<'t> {
        let (a, b) = self.pair(other);
        let out = ew_broadcast(&a, &b, |x, y| x - y);
        self.tape.push(Op::Sub(self.idx, other.idx), out)
    }

    /// Elementwise `self * other` (broadcasting).
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: NdVar<'t>) -> NdVar<'t> {
        let (a, b) = self.pair(other);
        let out = ew_broadcast(&a, &b, |x, y| x * y);
        self.tape.push(Op::Mul(self.idx, other.idx), out)
    }

    /// 2-D matrix product `self @ other`.
    pub fn matmul(self, other: NdVar<'t>) -> NdVar<'t> {
        let (a, b) = self.pair(other);
        let out = matmul2d(&a, &b);
        self.tape.push(Op::Matmul(self.idx, other.idx), out)
    }

    /// Batched matrix product `(…,m,k) · (…,k,n) → (…,m,n)` with the leading
    /// batch axes broadcast — the N-D capability the 2-D tape cannot express
    /// (e.g. per-head attention scores). Both operands need `ndim ≥ 2`.
    pub fn bmm(self, other: NdVar<'t>) -> NdVar<'t> {
        let (a, b) = self.pair(other);
        let out = batched_matmul(&a, &b);
        self.tape.push(Op::Bmm(self.idx, other.idx), out)
    }

    /// Softmax over the last axis (e.g. attention weights over keys).
    pub fn softmax(self) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let out = softmax_lastaxis(&a);
        self.tape.push(Op::Softmax(self.idx), out)
    }

    /// Swap the last two axes (e.g. `Kᵀ` inside attention).
    pub fn transpose_last2(self) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let out = transpose_last2(&a);
        self.tape.push(Op::TransposeLast2(self.idx), out)
    }

    /// Reshape to `shape` (same number of elements; data order preserved).
    pub fn reshape(self, shape: &[usize]) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        assert_eq!(
            a.data.len(),
            shape.iter().product::<usize>(),
            "reshape: element count mismatch"
        );
        let out = TensorND::new(a.data.clone(), shape.to_vec());
        self.tape.push(Op::Reshape(self.idx), out)
    }

    /// Layer normalisation over the last axis (zero mean, unit variance), with
    /// no affine — the `gamma`/`beta` of a `LayerNorm` layer are applied as a
    /// separate `mul`/`add`.
    pub fn layernorm(self, eps: f32) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let out = layernorm_lastaxis(&a, eps);
        self.tape.push(Op::LayerNormLast(self.idx, eps), out)
    }

    /// RMS normalisation over the last axis (`y = x / √(mean(x²)+eps)`), with no
    /// affine — the `gamma` of an [`NdRmsNorm`](crate::nn::nd_layers::NdRmsNorm)
    /// is applied as a separate `mul`. Cheaper than `layernorm` (no centring);
    /// the LLaMA-family normalisation.
    pub fn rmsnorm(self, eps: f32) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let out = rmsnorm_lastaxis(&a, eps);
        self.tape.push(Op::RmsNormLast(self.idx, eps), out)
    }

    /// Elementwise logistic sigmoid `σ(x) = 1/(1+e^-x)` (e.g. the gate of SiLU /
    /// SwiGLU).
    pub fn sigmoid(self) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let data: Vec<f32> = a.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let out = TensorND::new(data, a.shape.clone());
        self.tape.push(Op::Sigmoid(self.idx), out)
    }

    /// Elementwise `exp(x)` — the discretisation `exp(Δ·A)` of a state-space
    /// model (e.g. Mamba's selective scan) and positive reparametrisations.
    pub fn exp(self) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let data: Vec<f32> = a.data.iter().map(|&x| x.exp()).collect();
        let out = TensorND::new(data, a.shape.clone());
        self.tape.push(Op::Exp(self.idx), out)
    }

    /// **Rotary position embedding** (Su et al., RoFormer 2021) over a
    /// `(…, seq, d)` tensor: position is the second-to-last axis, and each
    /// adjacent pair `(x_{2p}, x_{2p+1})` of the last axis (which must be even)
    /// is rotated by `pos · base^(−2p/d)`. Applied to queries/keys, it makes the
    /// attention score depend only on the **relative** position. `base` is
    /// typically `10000`.
    pub fn rope(self, base: f32) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let out = rope_lastaxis(&a, base, false);
        self.tape.push(Op::Rope(self.idx, base), out)
    }

    /// Permute the axes (e.g. `(seq, heads, d) → (heads, seq, d)` for attention).
    pub fn permute(self, perm: &[usize]) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let out = a.transpose(perm).expect("permute: invalid axes");
        self.tape.push(Op::Permute(self.idx, perm.to_vec()), out)
    }

    /// The shape of this node's value.
    pub fn shape(self) -> Vec<usize> {
        self.tape.nodes.borrow()[self.idx].value.shape.clone()
    }

    /// Elementwise ReLU.
    pub fn relu(self) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let data = a.data.iter().map(|&x| x.max(0.0)).collect();
        let out = TensorND::new(data, a.shape.clone());
        self.tape.push(Op::Relu(self.idx), out)
    }

    /// Sum of all elements → scalar (shape `[1]`).
    pub fn sum(self) -> NdVar<'t> {
        let a = self.tape.nodes.borrow()[self.idx].value.clone();
        let s: f32 = a.data.iter().sum();
        self.tape
            .push(Op::Sum(self.idx), TensorND::new(vec![s], vec![1]))
    }

    /// Embedding lookup: `self` is a `(vocab, dim)` table; returns the
    /// `(indices.len(), dim)` stack of the selected rows. Gradients
    /// scatter-add back to the table (repeated indices accumulate).
    pub fn gather(self, indices: &[usize]) -> NdVar<'t> {
        let w = self.tape.nodes.borrow()[self.idx].value.clone();
        assert_eq!(w.ndim(), 2, "gather: table must be 2-D (vocab, dim)");
        let (vocab, dim) = (w.shape[0], w.shape[1]);
        let mut data = Vec::with_capacity(indices.len() * dim);
        for &ix in indices
        {
            assert!(
                ix < vocab,
                "gather: index {ix} out of range (vocab {vocab})"
            );
            data.extend_from_slice(&w.data[ix * dim..(ix + 1) * dim]);
        }
        let out = TensorND::new(data, vec![indices.len(), dim]);
        self.tape.push(Op::Gather(self.idx, indices.to_vec()), out)
    }

    /// Concatenate `self` with `rest` along axis 0 — all parts must share their
    /// trailing dims. Lets a recurrence (e.g. DeltaNet) assemble per-timestep
    /// `(1, d)` outputs into a single `(seq, d)` tensor on the tape. Backward
    /// splits the upstream gradient row-blocks back to each part.
    pub fn cat0(self, rest: &[NdVar<'t>]) -> NdVar<'t> {
        let nodes = self.tape.nodes.borrow();
        let first = &nodes[self.idx].value;
        let trailing: Vec<usize> = first.shape[1..].to_vec();
        let mut rows = first.shape[0];
        let mut data = first.data.clone();
        let mut idxs = vec![self.idx];
        for p in rest
        {
            let v = &nodes[p.idx].value;
            assert_eq!(v.shape[1..], trailing[..], "cat0: trailing dims must match");
            rows += v.shape[0];
            data.extend_from_slice(&v.data);
            idxs.push(p.idx);
        }
        drop(nodes);
        let mut shape = vec![rows];
        shape.extend_from_slice(&trailing);
        let out = TensorND::new(data, shape);
        self.tape.push(Op::Cat(idxs), out)
    }

    /// Fused softmax + mean negative-log-likelihood: `self` is `(n, vocab)`
    /// logits, `targets` holds one class index per row. Returns the scalar mean
    /// loss, computed with the log-sum-exp trick for numerical stability.
    pub fn cross_entropy(self, targets: &[usize]) -> NdVar<'t> {
        let logits = self.tape.nodes.borrow()[self.idx].value.clone();
        assert_eq!(
            logits.ndim(),
            2,
            "cross_entropy: logits must be 2-D (n, vocab)"
        );
        let (n, vocab) = (logits.shape[0], logits.shape[1]);
        assert_eq!(targets.len(), n, "cross_entropy: one target per row");
        let mut loss = 0.0f32;
        for (r, &t) in targets.iter().enumerate()
        {
            let row = &logits.data[r * vocab..(r + 1) * vocab];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + row.iter().map(|&v| (v - mx).exp()).sum::<f32>().ln();
            assert!(
                t < vocab,
                "cross_entropy: target {t} out of range (vocab {vocab})"
            );
            loss += lse - row[t];
        }
        loss /= n as f32;
        self.tape.push(
            Op::CrossEntropy(self.idx, targets.to_vec()),
            TensorND::new(vec![loss], vec![1]),
        )
    }

    fn pair(self, other: NdVar<'t>) -> (TensorND, TensorND) {
        let nodes = self.tape.nodes.borrow();
        (
            nodes[self.idx].value.clone(),
            nodes[other.idx].value.clone(),
        )
    }
}

// --- TensorND helpers (autodiff-local) -----------------------------------

/// Elementwise op on two equally-shaped tensors.
fn ew(a: &TensorND, b: &TensorND, f: impl Fn(f32, f32) -> f32) -> TensorND {
    debug_assert_eq!(a.shape, b.shape);
    let data = a.data.iter().zip(&b.data).map(|(&x, &y)| f(x, y)).collect();
    TensorND::new(data, a.shape.clone())
}

/// Elementwise op with numpy broadcasting.
fn ew_broadcast(a: &TensorND, b: &TensorND, f: impl Fn(f32, f32) -> f32) -> TensorND {
    let out =
        TensorND::broadcast_shape(&a.shape, &b.shape).expect("ew_broadcast: incompatible shapes");
    ew(
        &a.broadcast_to(&out).unwrap(),
        &b.broadcast_to(&out).unwrap(),
        f,
    )
}

/// `acc += g` (same shape).
fn accumulate(acc: &mut TensorND, g: &TensorND) {
    debug_assert_eq!(acc.shape, g.shape);
    for (a, &x) in acc.data.iter_mut().zip(&g.data)
    {
        *a += x;
    }
}

/// Reduce a broadcasted gradient back to `target` shape (sum over the axes that
/// broadcasting expanded), the transpose of `broadcast_to`.
fn unbroadcast(grad: &TensorND, target: &[usize]) -> TensorND {
    let mut g = grad.clone();
    // Collapse extra leading axes that `target` does not have.
    while g.shape.len() > target.len()
    {
        g = sum_axis(&g, 0, false);
    }
    // Sum over axes that were size-1 in the source (kept as size 1).
    for (axis, &t) in target.iter().enumerate()
    {
        if t == 1 && g.shape[axis] != 1
        {
            g = sum_axis(&g, axis, true);
        }
    }
    debug_assert_eq!(g.shape, target);
    g
}

/// Sum a tensor along one axis. `keepdim` keeps the axis as size 1; otherwise it
/// is removed.
fn sum_axis(t: &TensorND, axis: usize, keepdim: bool) -> TensorND {
    let mut out_shape = t.shape.clone();
    out_shape[axis] = 1;
    let out_strides = strides_of(&out_shape);
    let mut out = vec![0.0f32; out_shape.iter().product()];
    let in_strides = strides_of(&t.shape);
    for (flat, &v) in t.data.iter().enumerate()
    {
        // Decode the multi-index, zero out `axis`, re-encode into the output.
        let mut rem = flat;
        let mut out_flat = 0usize;
        for (ax, &st) in in_strides.iter().enumerate()
        {
            let idx = rem / st;
            rem %= st;
            let oidx = if ax == axis { 0 } else { idx };
            out_flat += oidx * out_strides[ax];
        }
        out[out_flat] += v;
    }
    if keepdim
    {
        TensorND::new(out, out_shape)
    }
    else
    {
        let squeezed: Vec<usize> = out_shape
            .iter()
            .enumerate()
            .filter(|(ax, _)| *ax != axis)
            .map(|(_, &d)| d)
            .collect();
        TensorND::new(out, squeezed)
    }
}

fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev()
    {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn transpose2d(t: &TensorND) -> TensorND {
    assert_eq!(t.ndim(), 2, "transpose2d: need a 2-D tensor");
    let (r, c) = (t.shape[0], t.shape[1]);
    let mut data = vec![0.0f32; r * c];
    for i in 0..r
    {
        for j in 0..c
        {
            data[j * r + i] = t.data[i * c + j];
        }
    }
    TensorND::new(data, vec![c, r])
}

fn matmul2d(a: &TensorND, b: &TensorND) -> TensorND {
    assert_eq!(a.ndim(), 2, "matmul2d: lhs must be 2-D");
    assert_eq!(b.ndim(), 2, "matmul2d: rhs must be 2-D");
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul2d: inner dims disagree");
    let mut data = vec![0.0f32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0.0f32;
            for p in 0..k
            {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            data[i * n + j] = acc;
        }
    }
    TensorND::new(data, vec![m, n])
}

/// Softmax over the last (contiguous) axis.
fn softmax_lastaxis(t: &TensorND) -> TensorND {
    let last = t.shape[t.ndim() - 1].max(1);
    let outer = t.data.len() / last;
    let mut out = vec![0.0f32; t.data.len()];
    for o in 0..outer
    {
        let base = o * last;
        let mut mx = f32::NEG_INFINITY;
        for i in 0..last
        {
            mx = mx.max(t.data[base + i]);
        }
        let mut sum = 0.0f32;
        for i in 0..last
        {
            let e = (t.data[base + i] - mx).exp();
            out[base + i] = e;
            sum += e;
        }
        for i in 0..last
        {
            out[base + i] /= sum;
        }
    }
    TensorND::new(out, t.shape.clone())
}

/// Backward of [`softmax_lastaxis`]: `dx_i = y_i·(g_i − Σ_j g_j y_j)`.
fn softmax_backward(y: &TensorND, g: &TensorND) -> TensorND {
    let last = y.shape[y.ndim() - 1].max(1);
    let outer = y.data.len() / last;
    let mut dx = vec![0.0f32; y.data.len()];
    for o in 0..outer
    {
        let base = o * last;
        let mut dot = 0.0f32;
        for i in 0..last
        {
            dot += g.data[base + i] * y.data[base + i];
        }
        for i in 0..last
        {
            dx[base + i] = y.data[base + i] * (g.data[base + i] - dot);
        }
    }
    TensorND::new(dx, y.shape.clone())
}

/// Layer normalisation over the last axis (no affine).
fn layernorm_lastaxis(t: &TensorND, eps: f32) -> TensorND {
    let d = t.shape[t.ndim() - 1].max(1);
    let outer = t.data.len() / d;
    let mut out = vec![0.0f32; t.data.len()];
    for o in 0..outer
    {
        let base = o * d;
        let row = &t.data[base..base + d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for i in 0..d
        {
            out[base + i] = (row[i] - mean) * rstd;
        }
    }
    TensorND::new(out, t.shape.clone())
}

/// Backward of [`layernorm_lastaxis`]:
/// `dx_i = rstd·(g_i − mean(g) − y_i·mean(g·y))` over the last axis.
fn layernorm_backward(x: &TensorND, y: &TensorND, g: &TensorND, eps: f32) -> TensorND {
    let d = x.shape[x.ndim() - 1].max(1);
    let outer = x.data.len() / d;
    let mut dx = vec![0.0f32; x.data.len()];
    for o in 0..outer
    {
        let base = o * d;
        let row = &x.data[base..base + d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        let mean_g = g.data[base..base + d].iter().sum::<f32>() / d as f32;
        let mean_gy = (0..d)
            .map(|i| g.data[base + i] * y.data[base + i])
            .sum::<f32>()
            / d as f32;
        for i in 0..d
        {
            dx[base + i] = rstd * (g.data[base + i] - mean_g - y.data[base + i] * mean_gy);
        }
    }
    TensorND::new(dx, x.shape.clone())
}

/// RMS normalisation over the last axis (no affine): `y = x / √(mean(x²)+eps)`.
fn rmsnorm_lastaxis(t: &TensorND, eps: f32) -> TensorND {
    let d = t.shape[t.ndim() - 1].max(1);
    let outer = t.data.len() / d;
    let mut out = vec![0.0f32; t.data.len()];
    for o in 0..outer
    {
        let base = o * d;
        let row = &t.data[base..base + d];
        let ms = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let r = (ms + eps).sqrt();
        for i in 0..d
        {
            out[base + i] = row[i] / r;
        }
    }
    TensorND::new(out, t.shape.clone())
}

/// Backward of [`rmsnorm_lastaxis`]: `dx_i = (g_i − y_i·mean(g·y)) / r`, with
/// `r = √(mean(x²)+eps)` and `y` the forward output.
fn rmsnorm_backward(x: &TensorND, y: &TensorND, g: &TensorND, eps: f32) -> TensorND {
    let d = x.shape[x.ndim() - 1].max(1);
    let outer = x.data.len() / d;
    let mut dx = vec![0.0f32; x.data.len()];
    for o in 0..outer
    {
        let base = o * d;
        let ms = x.data[base..base + d].iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let r = (ms + eps).sqrt();
        let mean_gy = (0..d)
            .map(|i| g.data[base + i] * y.data[base + i])
            .sum::<f32>()
            / d as f32;
        for i in 0..d
        {
            dx[base + i] = (g.data[base + i] - y.data[base + i] * mean_gy) / r;
        }
    }
    TensorND::new(dx, x.shape.clone())
}

/// Rotary position embedding over `(…, seq, d)` (position = axis −2). Each pair
/// `(x_{2p}, x_{2p+1})` is rotated by `pos · base^(−2p/d)`. `inverse` rotates by
/// the negative angle (the transpose), used by the backward pass.
fn rope_lastaxis(t: &TensorND, base: f32, inverse: bool) -> TensorND {
    let nd = t.ndim();
    assert!(nd >= 2, "rope: need ndim >= 2 (…, seq, d)");
    let d = t.shape[nd - 1];
    let seq = t.shape[nd - 2];
    assert!(d % 2 == 0, "rope: last axis must be even");
    let m = t.data.len() / (seq * d).max(1);
    let mut out = vec![0.0f32; t.data.len()];
    let sign = if inverse { -1.0 } else { 1.0 };
    for outer in 0..m
    {
        for s in 0..seq
        {
            let row = (outer * seq + s) * d;
            for p in 0..d / 2
            {
                let theta = base.powf(-2.0 * p as f32 / d as f32);
                let ang = sign * s as f32 * theta;
                let (sin, cos) = ang.sin_cos();
                let (a, b) = (t.data[row + 2 * p], t.data[row + 2 * p + 1]);
                out[row + 2 * p] = a * cos - b * sin;
                out[row + 2 * p + 1] = a * sin + b * cos;
            }
        }
    }
    TensorND::new(out, t.shape.clone())
}

/// Swap the last two axes (`(…,a,b) → (…,b,a)`); its own inverse.
fn transpose_last2(t: &TensorND) -> TensorND {
    let nd = t.ndim();
    assert!(nd >= 2, "transpose_last2: need ndim >= 2");
    let (a, b) = (t.shape[nd - 2], t.shape[nd - 1]);
    let outer = t.data.len() / (a * b).max(1);
    let mut out = vec![0.0f32; t.data.len()];
    for o in 0..outer
    {
        let base = o * a * b;
        for i in 0..a
        {
            for j in 0..b
            {
                out[base + j * a + i] = t.data[base + i * b + j];
            }
        }
    }
    let mut shape = t.shape.clone();
    shape.swap(nd - 2, nd - 1);
    TensorND::new(out, shape)
}

/// Map a flat index in `out_batch` to the corresponding flat batch offset in a
/// (possibly smaller / size-1-broadcast) `target_batch`.
fn project_batch(out_flat: usize, out_batch: &[usize], target_batch: &[usize]) -> usize {
    if target_batch.is_empty()
    {
        return 0;
    }
    let out_strides = strides_of(out_batch);
    let tgt_strides = strides_of(target_batch);
    let off = out_batch.len() - target_batch.len();
    let mut rem = out_flat;
    let mut tgt_flat = 0usize;
    for (ax, &os) in out_strides.iter().enumerate()
    {
        let idx = rem / os;
        rem %= os;
        if ax >= off
        {
            let ta = ax - off;
            let tidx = if target_batch[ta] == 1 { 0 } else { idx };
            tgt_flat += tidx * tgt_strides[ta];
        }
    }
    tgt_flat
}

/// Batched matmul `(…,m,k)·(…,k,n)→(…,m,n)` with broadcast batch axes.
fn batched_matmul(a: &TensorND, b: &TensorND) -> TensorND {
    let (an, bn) = (a.ndim(), b.ndim());
    assert!(an >= 2 && bn >= 2, "bmm: both operands need ndim >= 2");
    let (m, k) = (a.shape[an - 2], a.shape[an - 1]);
    let (k2, n) = (b.shape[bn - 2], b.shape[bn - 1]);
    assert_eq!(k, k2, "bmm: inner dims disagree");
    let a_batch = &a.shape[..an - 2];
    let b_batch = &b.shape[..bn - 2];
    let out_batch = TensorND::broadcast_shape(a_batch, b_batch).expect("bmm: batch broadcast");
    let bsz: usize = out_batch.iter().product::<usize>().max(1);
    let mut data = vec![0.0f32; bsz * m * n];
    for bi in 0..bsz
    {
        let a_off = project_batch(bi, &out_batch, a_batch) * m * k;
        let b_off = project_batch(bi, &out_batch, b_batch) * k * n;
        let o_off = bi * m * n;
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a.data[a_off + i * k + p] * b.data[b_off + p * n + j];
                }
                data[o_off + i * n + j] = acc;
            }
        }
    }
    let mut shape = out_batch;
    shape.push(m);
    shape.push(n);
    TensorND::new(data, shape)
}

/// Gradients of [`batched_matmul`]: `gA = g·Bᵀ`, `gB = Aᵀ·g` per batch,
/// accumulated back into the (broadcast) operand shapes.
fn bmm_backward(a: &TensorND, b: &TensorND, g: &TensorND) -> (TensorND, TensorND) {
    let (an, bn) = (a.ndim(), b.ndim());
    let (m, k) = (a.shape[an - 2], a.shape[an - 1]);
    let n = b.shape[bn - 1];
    let a_batch = &a.shape[..an - 2];
    let b_batch = &b.shape[..bn - 2];
    let out_batch = TensorND::broadcast_shape(a_batch, b_batch).unwrap();
    let bsz: usize = out_batch.iter().product::<usize>().max(1);
    let mut ga = vec![0.0f32; a.data.len()];
    let mut gb = vec![0.0f32; b.data.len()];
    for bi in 0..bsz
    {
        let a_off = project_batch(bi, &out_batch, a_batch) * m * k;
        let b_off = project_batch(bi, &out_batch, b_batch) * k * n;
        let g_off = bi * m * n;
        // gA[i,p] += sum_j g[i,j] * b[p,j]   (g·Bᵀ)
        for i in 0..m
        {
            for p in 0..k
            {
                let mut acc = 0.0f32;
                for j in 0..n
                {
                    acc += g.data[g_off + i * n + j] * b.data[b_off + p * n + j];
                }
                ga[a_off + i * k + p] += acc;
            }
        }
        // gB[p,j] += sum_i a[i,p] * g[i,j]   (Aᵀ·g)
        for p in 0..k
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for i in 0..m
                {
                    acc += a.data[a_off + i * k + p] * g.data[g_off + i * n + j];
                }
                gb[b_off + p * n + j] += acc;
            }
        }
    }
    (
        TensorND::new(ga, a.shape.clone()),
        TensorND::new(gb, b.shape.clone()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `loss = sum( relu(X·W + b) * V )` and return the scalar loss.
    /// `b` is `(1, out)` and broadcasts over the batch — exercising add/mul
    /// broadcasting, matmul, relu and sum in one graph.
    fn forward_loss(x: &[f32], w: &[f32], b: &[f32], v: &[f32]) -> f32 {
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x.to_vec(), vec![2, 3]));
        let wv = tape.input(TensorND::new(w.to_vec(), vec![3, 4]));
        let bv = tape.input(TensorND::new(b.to_vec(), vec![1, 4]));
        let vv = tape.input(TensorND::new(v.to_vec(), vec![2, 4]));
        let loss = xv.matmul(wv).add(bv).relu().mul(vv).sum();
        tape.value(loss).data[0]
    }

    #[test]
    fn numerical_gradient_check() {
        // Inputs chosen so the relu pre-activations straddle 0 (a real test of
        // the relu gradient) but avoid exact-zero kinks.
        let x = [0.5, -0.4, 0.3, -0.2, 0.6, 0.1];
        let w = [
            0.2, -0.5, 0.4, 0.1, -0.3, 0.7, 0.6, -0.1, 0.2, 0.3, -0.4, 0.5,
        ];
        let b = [0.05, -0.1, 0.2, -0.15];
        let v = [0.7, -0.6, 0.5, 0.4, -0.3, 0.2, 0.8, -0.9];

        // Analytic gradients via backward.
        let tape = NdTape::new();
        let xv = tape.input(TensorND::new(x.to_vec(), vec![2, 3]));
        let wv = tape.input(TensorND::new(w.to_vec(), vec![3, 4]));
        let bv = tape.input(TensorND::new(b.to_vec(), vec![1, 4]));
        let vv = tape.input(TensorND::new(v.to_vec(), vec![2, 4]));
        let loss = xv.matmul(wv).add(bv).relu().mul(vv).sum();
        let grads = tape.backward(loss);
        let (gx, gw, gb, gv) = (
            grads[xv.idx()].clone(),
            grads[wv.idx()].clone(),
            grads[bv.idx()].clone(),
            grads[vv.idx()].clone(),
        );

        // Central finite differences for each input tensor.
        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for k in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[k] += eps;
                dn[k] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                let ana = analytic.data[k];
                assert!(
                    (num - ana).abs() < 2e-2,
                    "grad mismatch at {k}: numeric {num}, analytic {ana}"
                );
            }
        };

        check(&gx, &x, &|p| forward_loss(p, &w, &b, &v));
        check(&gw, &w, &|p| forward_loss(&x, p, &b, &v));
        check(&gb, &b, &|p| forward_loss(&x, &w, p, &v));
        check(&gv, &v, &|p| forward_loss(&x, &w, &b, p));

        // The bias broadcast must reduce correctly: gb has the bias shape.
        assert_eq!(gb.shape, vec![1, 4]);
    }

    /// Multi-head **scaled-dot-product attention** expressed entirely on the
    /// N-D tape — `softmax(Q·Kᵀ/√d)·V` over `(heads, seq, d)` — with its
    /// gradients checked against finite differences. This is the milestone the
    /// 2-D tape cannot reach: it proves the N-D autograd handles batched matmul,
    /// last-axis transpose, scaling and softmax together, correctly.
    #[test]
    fn nd_multihead_attention_gradient_check() {
        let (h, seq, d) = (2usize, 3, 4);
        let shape = vec![h, seq, d];
        let n = h * seq * d;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11 - 0.5).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07 + 0.3).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05 - 0.2).sin()).collect();
        let scale = 1.0 / (d as f32).sqrt();

        let attn_loss = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
            let t = NdTape::new();
            let qv = t.input(TensorND::new(q.to_vec(), shape.clone()));
            let kv = t.input(TensorND::new(k.to_vec(), shape.clone()));
            let vv = t.input(TensorND::new(v.to_vec(), shape.clone()));
            let sc = t.input(TensorND::new(vec![scale], vec![1]));
            let out = qv.bmm(kv.transpose_last2()).mul(sc).softmax().bmm(vv);
            t.value(out.sum()).data[0]
        };

        let t = NdTape::new();
        let qv = t.input(TensorND::new(q.clone(), shape.clone()));
        let kv = t.input(TensorND::new(k.clone(), shape.clone()));
        let vv = t.input(TensorND::new(v.clone(), shape.clone()));
        let sc = t.input(TensorND::new(vec![scale], vec![1]));
        let out = qv.bmm(kv.transpose_last2()).mul(sc).softmax().bmm(vv);
        assert_eq!(t.value(out).shape, vec![h, seq, d]);
        let grads = t.backward(out.sum());
        let (gq, gk, gv) = (
            grads[qv.idx()].clone(),
            grads[kv.idx()].clone(),
            grads[vv.idx()].clone(),
        );

        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for kk in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[kk] += eps;
                dn[kk] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[kk]).abs() < 2e-2,
                    "attention grad {kk}: numeric {num}, analytic {}",
                    analytic.data[kk]
                );
            }
        };
        check(&gq, &q, &|p| attn_loss(p, &k, &v));
        check(&gk, &k, &|p| attn_loss(&q, p, &v));
        check(&gv, &v, &|p| attn_loss(&q, &k, p));
    }

    /// LayerNorm over the last axis: its input gradient matches finite
    /// differences (loss = sum(layernorm(x)·v) to make the gradient non-trivial).
    #[test]
    fn nd_layernorm_gradient_check() {
        let shape = vec![2usize, 5];
        let n = 10;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let eps = 1e-5f32;

        let loss_of = |xd: &[f32]| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), shape.clone()));
            let vv = t.input(TensorND::new(v.clone(), shape.clone()));
            t.value(xv.layernorm(eps).mul(vv).sum()).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), shape.clone()));
        let vv = t.input(TensorND::new(v.clone(), shape.clone()));
        let grads = t.backward(xv.layernorm(eps).mul(vv).sum());
        let gx = grads[xv.idx()].clone();

        let fd = 1e-3f32;
        for k in 0..n
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += fd;
            dn[k] -= fd;
            let num = (loss_of(&up) - loss_of(&dn)) / (2.0 * fd);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "layernorm grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// RMSNorm over the last axis: input gradient matches finite differences
    /// (loss = sum(rmsnorm(x)·v)).
    #[test]
    fn nd_rmsnorm_gradient_check() {
        let shape = vec![2usize, 5];
        let n = 10;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1.0).sin() + 0.4).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let eps = 1e-6f32;

        let loss_of = |xd: &[f32]| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), shape.clone()));
            let vv = t.input(TensorND::new(v.clone(), shape.clone()));
            t.value(xv.rmsnorm(eps).mul(vv).sum()).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), shape.clone()));
        let vv = t.input(TensorND::new(v.clone(), shape.clone()));
        let grads = t.backward(xv.rmsnorm(eps).mul(vv).sum());
        let gx = grads[xv.idx()].clone();

        let fd = 1e-3f32;
        for k in 0..n
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += fd;
            dn[k] -= fd;
            let num = (loss_of(&up) - loss_of(&dn)) / (2.0 * fd);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "rmsnorm grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// Sigmoid: forward sanity (`σ(0)=0.5`) and input gradient vs finite
    /// differences (loss = sum(sigmoid(x)·v)).
    #[test]
    fn nd_sigmoid_forward_and_gradient_check() {
        let n = 6;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 1.5).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 0.2).cos()).collect();

        let t0 = NdTape::new();
        let z = t0.input(TensorND::new(vec![0.0], vec![1]));
        assert!((t0.value(z.sigmoid()).data[0] - 0.5).abs() < 1e-7);

        let loss_of = |xd: &[f32]| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), vec![n]));
            let vv = t.input(TensorND::new(v.clone(), vec![n]));
            t.value(xv.sigmoid().mul(vv).sum()).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![n]));
        let vv = t.input(TensorND::new(v.clone(), vec![n]));
        let grads = t.backward(xv.sigmoid().mul(vv).sum());
        let gx = grads[xv.idx()].clone();

        let fd = 1e-3f32;
        for k in 0..n
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += fd;
            dn[k] -= fd;
            let num = (loss_of(&up) - loss_of(&dn)) / (2.0 * fd);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "sigmoid grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// RoPE input gradient matches finite differences (loss = sum(rope(x)·v)).
    #[test]
    fn nd_rope_gradient_check() {
        let (seq, d) = (3usize, 4usize);
        let shape = vec![seq, d];
        let x: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.21 - 0.5).sin())
            .collect();
        let v: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.13).cos()).collect();
        let base = 10000.0f32;

        let loss_of = |xd: &[f32]| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xd.to_vec(), shape.clone()));
            let vv = t.input(TensorND::new(v.clone(), shape.clone()));
            t.value(xv.rope(base).mul(vv).sum()).data[0]
        };

        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), shape.clone()));
        let vv = t.input(TensorND::new(v.clone(), shape.clone()));
        let grads = t.backward(xv.rope(base).mul(vv).sum());
        let gx = grads[xv.idx()].clone();

        let fd = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += fd;
            dn[k] -= fd;
            let num = (loss_of(&up) - loss_of(&dn)) / (2.0 * fd);
            assert!(
                (num - gx.data[k]).abs() < 2e-2,
                "rope grad {k}: numeric {num}, analytic {}",
                gx.data[k]
            );
        }
    }

    /// RoPE is an orthogonal rotation: it preserves each row's L2 norm.
    #[test]
    fn nd_rope_preserves_norm() {
        let (seq, d) = (4usize, 6usize);
        let x: Vec<f32> = (0..seq * d)
            .map(|i| (i as f32 * 0.3 - 1.0).sin() + 0.2)
            .collect();
        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![seq, d]));
        let y = t.value(xv.rope(10000.0)).data;
        for s in 0..seq
        {
            let nx: f32 = x[s * d..(s + 1) * d].iter().map(|v| v * v).sum();
            let ny: f32 = y[s * d..(s + 1) * d].iter().map(|v| v * v).sum();
            assert!((nx - ny).abs() < 1e-4, "row {s}: norm {nx} -> {ny}");
        }
    }

    /// The defining RoPE property: with identical query rows and identical key
    /// rows, the score `q'_i · k'_j` depends only on the **relative** offset
    /// `i − j`, so equal offsets give equal scores.
    #[test]
    fn nd_rope_relative_position() {
        let (seq, d) = (5usize, 4usize);
        let qrow = [0.5f32, -0.3, 0.8, 0.1];
        let krow = [0.2f32, 0.7, -0.4, 0.6];
        let q: Vec<f32> = (0..seq).flat_map(|_| qrow).collect();
        let k: Vec<f32> = (0..seq).flat_map(|_| krow).collect();

        let t = NdTape::new();
        let qv = t
            .value(t.input(TensorND::new(q, vec![seq, d])).rope(10000.0))
            .data;
        let kv = t
            .value(t.input(TensorND::new(k, vec![seq, d])).rope(10000.0))
            .data;
        let dot =
            |i: usize, j: usize| -> f32 { (0..d).map(|c| qv[i * d + c] * kv[j * d + c]).sum() };
        // offset +1: (1,0), (2,1), (3,2), (4,3) all equal.
        let s = dot(1, 0);
        for (i, j) in [(2, 1), (3, 2), (4, 3)]
        {
            assert!((dot(i, j) - s).abs() < 1e-4, "offset +1 not constant");
        }
        // offset −2: (0,2) == (2,4).
        assert!(
            (dot(0, 2) - dot(2, 4)).abs() < 1e-4,
            "offset -2 not constant"
        );
    }

    #[test]
    fn broadcast_add_reduces_gradient() {
        // c = a(2,3) + b(1,3); d(loss) = sum(c). dL/db must be summed over the
        // broadcast (batch) axis → each equals the batch size (2).
        let tape = NdTape::new();
        let a = tape.input(TensorND::zeros(&[2, 3]));
        let b = tape.input(TensorND::ones(&[1, 3]));
        let loss = a.add(b).sum();
        let grads = tape.backward(loss);
        assert_eq!(grads[b.idx()].shape, vec![1, 3]);
        assert_eq!(grads[b.idx()].data, vec![2.0, 2.0, 2.0]);
        assert_eq!(grads[a.idx()].data, vec![1.0; 6]);
    }

    #[test]
    fn matmul_forward_matches_hand_value() {
        let tape = NdTape::new();
        let a = tape.input(TensorND::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ));
        let b = tape.input(TensorND::new(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![3, 2],
        ));
        let c = a.matmul(b);
        // row0·cols: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]; row1: [10, 11]
        assert_eq!(tape.value(c).data, vec![4.0, 5.0, 10.0, 11.0]);
    }

    /// Batched matmul with a **broadcast** batch axis: forward matches a manual
    /// per-batch product, and the gradients match finite differences (including
    /// the batch-1 operand whose gradient accumulates over every batch).
    #[test]
    fn bmm_forward_and_gradient_check() {
        let a_shape = vec![2usize, 2, 3]; // batch 2
        let b_shape = vec![1usize, 3, 2]; // batch 1 → broadcasts over the 2
        let a: Vec<f32> = (0..12).map(|i| (i as f32 * 0.2 - 1.0).sin()).collect();
        let b: Vec<f32> = (0..6).map(|i| (i as f32 * 0.3 + 0.5).cos()).collect();

        // Forward correctness on one cell of batch 0.
        let tape = NdTape::new();
        let av = tape.input(TensorND::new(a.clone(), a_shape.clone()));
        let bv = tape.input(TensorND::new(b.clone(), b_shape.clone()));
        let out = tape.value(av.bmm(bv));
        assert_eq!(out.shape, vec![2, 2, 2]);
        let manual000 = a[0] * b[0] + a[1] * b[2] + a[2] * b[4];
        assert!((out.data[0] - manual000).abs() < 1e-5);

        // loss = sum(bmm(a, b)); gradient-check both operands.
        let loss_of = |aa: &[f32], bb: &[f32]| -> f32 {
            let t = NdTape::new();
            let xa = t.input(TensorND::new(aa.to_vec(), a_shape.clone()));
            let xb = t.input(TensorND::new(bb.to_vec(), b_shape.clone()));
            t.value(xa.bmm(xb).sum()).data[0]
        };
        let t = NdTape::new();
        let xa = t.input(TensorND::new(a.clone(), a_shape.clone()));
        let xb = t.input(TensorND::new(b.clone(), b_shape.clone()));
        let grads = t.backward(xa.bmm(xb).sum());
        let (ga, gb) = (grads[xa.idx()].clone(), grads[xb.idx()].clone());

        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for kk in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[kk] += eps;
                dn[kk] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[kk]).abs() < 2e-2,
                    "bmm grad {kk}: numeric {num}, analytic {}",
                    analytic.data[kk]
                );
            }
        };
        check(&ga, &a, &|p| loss_of(p, &b));
        check(&gb, &b, &|p| loss_of(&a, p));
        // The broadcast operand keeps its own (batch-1) shape.
        assert_eq!(gb.shape, vec![1, 3, 2]);
    }

    /// `cat0` concatenates along axis 0 (forward), and the gradient splits the
    /// upstream row-blocks back to each part (checked against finite differences).
    #[test]
    fn nd_cat0_forward_and_gradient_check() {
        let a = vec![1.0f32, -2.0, 0.5, 3.0, -1.0, 0.7]; // (2,3)
        let b = vec![0.2f32, -0.4, 0.9]; // (1,3)

        let t0 = NdTape::new();
        let av0 = t0.input(TensorND::new(a.clone(), vec![2, 3]));
        let bv0 = t0.input(TensorND::new(b.clone(), vec![1, 3]));
        let c = t0.value(av0.cat0(&[bv0]));
        assert_eq!(c.shape, vec![3, 3]);
        assert_eq!(&c.data[0..6], &a[..]);
        assert_eq!(&c.data[6..9], &b[..]);

        // loss = sum(cat²); gradient-check both parts.
        let loss_of = |aa: &[f32], bb: &[f32]| -> f32 {
            let t = NdTape::new();
            let xa = t.input(TensorND::new(aa.to_vec(), vec![2, 3]));
            let xb = t.input(TensorND::new(bb.to_vec(), vec![1, 3]));
            let cc = xa.cat0(&[xb]);
            t.value(cc.mul(cc).sum()).data[0]
        };
        let t = NdTape::new();
        let xa = t.input(TensorND::new(a.clone(), vec![2, 3]));
        let xb = t.input(TensorND::new(b.clone(), vec![1, 3]));
        let cc = xa.cat0(&[xb]);
        let grads = t.backward(cc.mul(cc).sum());
        let (ga, gb) = (grads[xa.idx()].clone(), grads[xb.idx()].clone());

        let eps = 1e-3f32;
        let check = |analytic: &TensorND, base: &[f32], rebuild: &dyn Fn(&[f32]) -> f32| {
            for k in 0..base.len()
            {
                let mut up = base.to_vec();
                let mut dn = base.to_vec();
                up[k] += eps;
                dn[k] -= eps;
                let num = (rebuild(&up) - rebuild(&dn)) / (2.0 * eps);
                assert!(
                    (num - analytic.data[k]).abs() < 2e-2,
                    "cat0 grad {k}: numeric {num}, analytic {}",
                    analytic.data[k]
                );
            }
        };
        check(&ga, &a, &|p| loss_of(p, &b));
        check(&gb, &b, &|p| loss_of(&a, p));
    }

    /// `exp` forward and gradient (`d/dx exp = exp`) vs finite differences.
    #[test]
    fn nd_exp_forward_and_gradient_check() {
        let x = vec![0.3f32, -0.7, 1.1, -0.2, 0.5, -1.4];
        let t0 = NdTape::new();
        let xv0 = t0.input(TensorND::new(x.clone(), vec![2, 3]));
        let e = t0.value(xv0.exp());
        for (got, &xi) in e.data.iter().zip(&x)
        {
            assert!((got - xi.exp()).abs() < 1e-6);
        }
        let loss_of = |xx: &[f32]| -> f32 {
            let t = NdTape::new();
            let xv = t.input(TensorND::new(xx.to_vec(), vec![2, 3]));
            t.value(xv.exp().sum()).data[0]
        };
        let t = NdTape::new();
        let xv = t.input(TensorND::new(x.clone(), vec![2, 3]));
        let grads = t.backward(xv.exp().sum());
        let g = grads[xv.idx()].clone();
        let eps = 1e-3f32;
        for k in 0..x.len()
        {
            let mut up = x.clone();
            let mut dn = x.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up) - loss_of(&dn)) / (2.0 * eps);
            assert!(
                (num - g.data[k]).abs() < 2e-2,
                "exp grad {k}: {num} vs {}",
                g.data[k]
            );
        }
    }

    /// Embedding `gather`: forward selects the right rows, and the gradient
    /// w.r.t. the table matches finite differences. The index list repeats a
    /// row (tests scatter-add) and omits two rows (whose gradient must be 0).
    #[test]
    fn nd_gather_gradient_check() {
        let (vocab, dim) = (5usize, 3usize);
        let idx = vec![2usize, 0, 2, 4]; // row 2 repeated; rows 1 and 3 unused
        let w: Vec<f32> = (0..vocab * dim)
            .map(|i| (i as f32 * 0.2 - 0.5).sin())
            .collect();
        let v: Vec<f32> = (0..idx.len() * dim)
            .map(|i| (i as f32 * 0.3).cos())
            .collect();

        // Forward picks the correct rows.
        let t0 = NdTape::new();
        let w0 = t0.input(TensorND::new(w.clone(), vec![vocab, dim]));
        let picked = t0.value(w0.gather(&idx));
        assert_eq!(picked.shape, vec![idx.len(), dim]);
        assert_eq!(&picked.data[0..dim], &w[2 * dim..3 * dim]);

        let loss_of = |wd: &[f32]| -> f32 {
            let t = NdTape::new();
            let wv = t.input(TensorND::new(wd.to_vec(), vec![vocab, dim]));
            let vv = t.input(TensorND::new(v.clone(), vec![idx.len(), dim]));
            t.value(wv.gather(&idx).mul(vv).sum()).data[0]
        };

        let t = NdTape::new();
        let wv = t.input(TensorND::new(w.clone(), vec![vocab, dim]));
        let vv = t.input(TensorND::new(v.clone(), vec![idx.len(), dim]));
        let grads = t.backward(wv.gather(&idx).mul(vv).sum());
        let gw = grads[wv.idx()].clone();
        assert_eq!(gw.shape, vec![vocab, dim]);

        let eps = 1e-3f32;
        for k in 0..w.len()
        {
            let mut up = w.clone();
            let mut dn = w.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up) - loss_of(&dn)) / (2.0 * eps);
            assert!(
                (num - gw.data[k]).abs() < 2e-2,
                "gather grad {k}: numeric {num}, analytic {}",
                gw.data[k]
            );
        }
        // Rows never gathered (1 and 3) receive zero gradient.
        assert_eq!(&gw.data[dim..2 * dim], &[0.0, 0.0, 0.0]);
        assert_eq!(&gw.data[3 * dim..4 * dim], &[0.0, 0.0, 0.0]);
    }

    /// Fused softmax cross-entropy: gradient w.r.t. the logits matches finite
    /// differences, and each row's gradient sums to ~0 (a property of
    /// `softmax − onehot`).
    #[test]
    fn nd_cross_entropy_gradient_check() {
        let (n, vocab) = (3usize, 4usize);
        let targets = vec![1usize, 3, 0];
        let logits: Vec<f32> = (0..n * vocab)
            .map(|i| (i as f32 * 0.17 - 0.5).sin())
            .collect();

        let loss_of = |ld: &[f32]| -> f32 {
            let t = NdTape::new();
            let lv = t.input(TensorND::new(ld.to_vec(), vec![n, vocab]));
            t.value(lv.cross_entropy(&targets)).data[0]
        };

        let t = NdTape::new();
        let lv = t.input(TensorND::new(logits.clone(), vec![n, vocab]));
        let grads = t.backward(lv.cross_entropy(&targets));
        let gl = grads[lv.idx()].clone();

        let eps = 1e-3f32;
        for k in 0..logits.len()
        {
            let mut up = logits.clone();
            let mut dn = logits.clone();
            up[k] += eps;
            dn[k] -= eps;
            let num = (loss_of(&up) - loss_of(&dn)) / (2.0 * eps);
            assert!(
                (num - gl.data[k]).abs() < 2e-2,
                "cross-entropy grad {k}: numeric {num}, analytic {}",
                gl.data[k]
            );
        }
        for r in 0..n
        {
            let s: f32 = gl.data[r * vocab..(r + 1) * vocab].iter().sum();
            assert!(s.abs() < 1e-6, "row {r} gradient should sum to 0, got {s}");
        }
    }

    /// Sanity: with uniform logits the softmax is uniform, so the cross-entropy
    /// equals `ln(vocab)` regardless of the target.
    #[test]
    fn nd_cross_entropy_uniform_is_ln_vocab() {
        let (n, vocab) = (2usize, 7usize);
        let t = NdTape::new();
        let lv = t.input(TensorND::new(vec![0.0f32; n * vocab], vec![n, vocab]));
        let loss = t.value(lv.cross_entropy(&[0, 3])).data[0];
        assert!((loss - (vocab as f32).ln()).abs() < 1e-5, "loss {loss}");
    }
}
