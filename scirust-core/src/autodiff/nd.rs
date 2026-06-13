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
//! Supported ops (MVP): elementwise `add`/`mul` with broadcasting, 2-D `matmul`,
//! `relu`, and `sum` to a scalar. The building blocks (`broadcast_shape`,
//! `broadcast_to`, `matmul_shape`) live on [`TensorND`].

use std::cell::RefCell;

use crate::tensor::tensor_nd::TensorND;

#[derive(Clone)]
enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    Matmul(usize, usize),
    Relu(usize),
    Sum(usize),
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
}
