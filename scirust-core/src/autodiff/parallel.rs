// scirust-core/src/autodiff/parallel.rs
// Phase 4: Data Parallelism Engine — Send + Sync tape wrapper
//
// ParallelTape is a Send + Sync tape that stores the computation graph
// behind Arc<RwLock> for safe sharing across threads.
// Gradients are stored as scalar f64 values (summed from full tensor grads).

use std::sync::{Arc, RwLock};
use super::reverse::{Node, Op, Tensor, SavedData};

/// A Send + Sync tape wrapper.
///
/// - `nodes`     : the computation graph (Arc<RwLock> for thread-safety)
/// - `values`    : forward tensor values (needed during backward)
/// - `grads`     : scalar f64 gradients (one per node, set by backward())
///
/// ParallelTape automatically implements Send + Sync because all fields
/// use Arc<RwLock<…>> of types that are themselves Send + Sync.
#[derive(Debug, Clone)]
pub struct ParallelTape {
    nodes: Arc<RwLock<Vec<Node>>>,
    values: Arc<RwLock<Vec<Tensor>>>,
    grads: Arc<RwLock<Vec<f64>>>,
}

impl ParallelTape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            values: Arc::new(RwLock::new(Vec::new())),
            grads: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Append a node and return its index.
    /// Initialises the corresponding value slot with zeros
    /// and the gradient slot with 0.0.
    pub fn alloc_node(&self, node: Node) -> usize {
        let mut nodes = self.nodes.write().expect("ParallelTape nodes lock poisoned");
        let mut vals = self.values.write().expect("ParallelTape values lock poisoned");
        let mut grads = self.grads.write().expect("ParallelTape grads lock poisoned");
        let idx = nodes.len();
        let (r, c) = node.shape;
        nodes.push(node);
        vals.push(Tensor::zeros(r, c));
        grads.push(0.0);
        idx
    }

    /// Set the forward value of node `idx`.
    pub fn set_value(&self, idx: usize, data: &[f32]) {
        let mut vals = self.values.write().expect("ParallelTape values lock poisoned");
        let len = vals[idx].data.len();
        assert_eq!(data.len(), len, "set_value size mismatch");
        vals[idx].data.copy_from_slice(data);
    }

    /// Get the forward value of node `idx`.
    pub fn value(&self, idx: usize) -> Tensor {
        self.values.read().expect("ParallelTape values lock poisoned")[idx].clone()
    }

    /// Get the scalar gradient of node `idx`.
    pub fn grad(&self, idx: usize) -> f64 {
        self.grads.read().expect("ParallelTape grads lock poisoned")[idx]
    }

    /// Return all scalar gradients.
    pub fn grads(&self) -> Vec<f64> {
        self.grads.read().expect("ParallelTape grads lock poisoned").clone()
    }

    /// Return the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.read().expect("ParallelTape nodes lock poisoned").len()
    }

    /// Run backward from `output_idx`, computing scalar gradients
    /// for every node.  The algorithm is identical to the sequential
    /// [`Tape::backward`](super::reverse::Tape::backward) but stores
    /// the result as a single `f64` per node (the sum of the full
    /// tensor gradient).
    pub fn backward(&self, output_idx: usize) {
        let nodes = self.nodes.read().expect("ParallelTape nodes lock poisoned");
        let values = self.values.read().expect("ParallelTape values lock poisoned");
        let n = nodes.len();
        assert!(output_idx < n, "backward: idx {} out of bounds ({} nodes)", output_idx, n);

        // ---- full tensor gradients (local, not shared) ----
        let mut t_grads: Vec<Tensor> = (0..n)
            .map(|i| Tensor::zeros(nodes[i].shape.0, nodes[i].shape.1))
            .collect();

        // seed
        {
            let (r, c) = nodes[output_idx].shape;
            t_grads[output_idx] = Tensor::from_vec(vec![1.0f32; r * c], r, c);
        }

        // ---- reverse pass ----
        for i in (0..=output_idx).rev() {
            let g = t_grads[i].clone();
            // skip dead gradients
            if g.data.iter().all(|&x| x == 0.0) {
                continue;
            }

            match nodes[i].op {
                Op::Input => {}

                Op::Add(a, b) => {
                    t_grads[a] = t_grads[a].add(&g);
                    t_grads[b] = t_grads[b].add(&g);
                }
                Op::Sub(a, b) => {
                    t_grads[a] = t_grads[a].add(&g);
                    t_grads[b] = t_grads[b].sub(&g);
                }
                Op::Mul(a, b) => {
                    t_grads[a] = t_grads[a].add(&g.hadamard(&values[b]));
                    t_grads[b] = t_grads[b].add(&g.hadamard(&values[a]));
                }
                Op::Div(a, b) => {
                    let av = &values[a];
                    let bv = &values[b];
                    let b_recip = bv.reciprocal();
                    let a_over_b2 = av.hadamard(&b_recip.hadamard(&b_recip));
                    t_grads[a] = t_grads[a].add(&g.hadamard(&b_recip));
                    t_grads[b] = t_grads[b].sub(&g.hadamard(&a_over_b2));
                }

                Op::AddBroadcast(a, b) => {
                    let av = &values[a];
                    let bv = &values[b];
                    t_grads[a] = t_grads[a].add(&g);
                    if bv.rows == 1 && bv.cols == av.cols {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[c] += g.data[r * g.cols + c];
                            }
                        }
                        t_grads[b] = t_grads[b].add(&db);
                    } else if bv.rows == av.rows && bv.cols == 1 {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[r] += g.data[r * g.cols + c];
                            }
                        }
                        t_grads[b] = t_grads[b].add(&db);
                    } else {
                        t_grads[b] = t_grads[b].add(&g);
                    }
                }
                Op::SubBroadcast(a, b) => {
                    let av = &values[a];
                    let bv = &values[b];
                    t_grads[a] = t_grads[a].add(&g);
                    if bv.rows == 1 && bv.cols == av.cols {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[c] += g.data[r * g.cols + c];
                            }
                        }
                        t_grads[b] = t_grads[b].sub(&db);
                    } else if bv.rows == av.rows && bv.cols == 1 {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[r] += g.data[r * g.cols + c];
                            }
                        }
                        t_grads[b] = t_grads[b].sub(&db);
                    } else {
                        t_grads[b] = t_grads[b].sub(&g);
                    }
                }
                Op::MulBroadcast(a, b) => {
                    let av = &values[a];
                    let bv = &values[b];
                    t_grads[a] = t_grads[a].add(&g.hadamard(&bv.broadcast_to(g.rows, g.cols)));
                    if bv.rows == 1 && bv.cols == av.cols {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[c] += g.data[r * g.cols + c] * av.data[r * av.cols + c];
                            }
                        }
                        t_grads[b] = t_grads[b].add(&db);
                    } else if bv.rows == av.rows && bv.cols == 1 {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[r] += g.data[r * g.cols + c] * av.data[r * av.cols + c];
                            }
                        }
                        t_grads[b] = t_grads[b].add(&db);
                    } else {
                        t_grads[b] = t_grads[b].add(&g.hadamard(&av));
                    }
                }
                Op::DivBroadcast(a, b) => {
                    let av = &values[a];
                    let bv = &values[b];
                    let b_recip = bv.reciprocal();
                    t_grads[a] = t_grads[a].add(&g.hadamard(&b_recip.broadcast_to(g.rows, g.cols)));
                    if bv.rows == 1 && bv.cols == av.cols {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[c] += g.data[r * g.cols + c] * (-av.data[r * av.cols + c] / (bv.data[c] * bv.data[c]));
                            }
                        }
                        t_grads[b] = t_grads[b].add(&db);
                    } else if bv.rows == av.rows && bv.cols == 1 {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows {
                            for c in 0..g.cols {
                                db.data[r] += g.data[r * g.cols + c] * (-av.data[r * av.cols + c] / (bv.data[r] * bv.data[r]));
                            }
                        }
                        t_grads[b] = t_grads[b].add(&db);
                    } else {
                        t_grads[b] = t_grads[b].sub(&g.hadamard(&av.hadamard(&bv.reciprocal().hadamard(&bv.reciprocal()))));
                    }
                }

                Op::MatMul(a, b) => {
                    let av = &values[a];
                    let bv = &values[b];
                    let ga = g.matmul(&bv.transpose());
                    let gb = av.transpose().matmul(&g);
                    t_grads[a] = t_grads[a].add(&ga);
                    t_grads[b] = t_grads[b].add(&gb);
                }

                Op::Scale { input, scalar } => {
                    t_grads[input] = t_grads[input].add(&g.scale(scalar));
                }
                Op::Neg(a) => {
                    t_grads[a] = t_grads[a].sub(&g);
                }
                Op::Exp(a) => {
                    let av = &values[a];
                    t_grads[a] = t_grads[a].add(&g.hadamard(&av.exp()));
                }
                Op::Log(a) => {
                    let av = &values[a];
                    t_grads[a] = t_grads[a].add(&g.hadamard(&av.reciprocal()));
                }
                Op::Sqrt(a) => {
                    let av = &values[a];
                    let two_sqrt = av.sqrt().scale(2.0);
                    t_grads[a] = t_grads[a].add(&g.hadamard(&two_sqrt.reciprocal()));
                }
                Op::Reciprocal(a) => {
                    let av = &values[a];
                    let mut denom = av.hadamard(av);
                    for d in &mut denom.data {
                        *d = 1.0 / (*d + 1e-10);
                    }
                    let minus_one_over_x2 = denom.scale(-1.0);
                    t_grads[a] = t_grads[a].add(&g.hadamard(&minus_one_over_x2));
                }
                Op::Pow { base, exp } => {
                    let av = &values[base];
                    let deriv = av.pow(exp - 1.0).scale(exp);
                    t_grads[base] = t_grads[base].add(&g.hadamard(&deriv));
                }
                Op::ReLU(a) => {
                    let av = &values[a];
                    let mut mask = Tensor::zeros(av.rows, av.cols);
                    for j in 0..av.data.len() {
                        mask.data[j] = if av.data[j] > 0.0 { 1.0 } else { 0.0 };
                    }
                    t_grads[a] = t_grads[a].add(&g.hadamard(&mask));
                }
                Op::Sigmoid(a) => {
                    let av = &values[a];
                    let sig = av.sigmoid();
                    let ones = Tensor::from_vec(vec![1.0f32; sig.data.len()], sig.rows, sig.cols);
                    let deriv = sig.hadamard(&ones.sub(&sig));
                    t_grads[a] = t_grads[a].add(&g.hadamard(&deriv));
                }
                Op::Tanh(a) => {
                    let av = &values[a];
                    let t = av.tanh();
                    let ones = Tensor::from_vec(vec![1.0f32; t.data.len()], t.rows, t.cols);
                    let deriv = ones.sub(&t.hadamard(&t));
                    t_grads[a] = t_grads[a].add(&g.hadamard(&deriv));
                }

                Op::Sum(a) => {
                    let av = &values[a];
                    t_grads[a] = t_grads[a].add(&g.broadcast_to(av.rows, av.cols));
                }
                Op::SumAxis(a, _axis) => {
                    let av = &values[a];
                    t_grads[a] = t_grads[a].add(&g.broadcast_to(av.rows, av.cols));
                }
                Op::MeanAxis(a, axis) => {
                    let av = &values[a];
                    let n = if axis == 0 { av.rows } else { av.cols } as f32;
                    t_grads[a] = t_grads[a].add(&g.scale(1.0 / n).broadcast_to(av.rows, av.cols));
                }
                Op::VarAxis(a, axis) => {
                    let av = &values[a];
                    let n = if axis == 0 { av.rows } else { av.cols } as f32;
                    let mean = av.mean_axis(axis);
                    let diff = av.sub(&mean.broadcast_to(av.rows, av.cols));
                    let two_over_n = 2.0 / n;
                    t_grads[a] = t_grads[a].add(&g.scale(two_over_n).broadcast_to(av.rows, av.cols).hadamard(&diff));
                }
                Op::MaxAxis(a, axis) => {
                    let av = &values[a];
                    let max_v = av.max_axis(axis);
                    let mut mask = Tensor::zeros(av.rows, av.cols);
                    if axis == 0 {
                        for c in 0..av.cols {
                            let m = max_v.data[c];
                            for r in 0..av.rows {
                                if (av.data[r * av.cols + c] - m).abs() < 1e-6 {
                                    mask.data[r * av.cols + c] = 1.0;
                                }
                            }
                        }
                    } else {
                        for r in 0..av.rows {
                            let m = max_v.data[r];
                            for c in 0..av.cols {
                                if (av.data[r * av.cols + c] - m).abs() < 1e-6 {
                                    mask.data[r * av.cols + c] = 1.0;
                                }
                            }
                        }
                    }
                    t_grads[a] = t_grads[a].add(&g.broadcast_to(av.rows, av.cols).hadamard(&mask));
                }
                Op::Softmax { input, axis } => {
                    let av = &values[input];
                    let sm = av.softmax(axis);
                    let g_broadcast = g.broadcast_to(av.rows, av.cols);
                    let gs = g_broadcast.hadamard(&sm);
                    let sum_gs = gs.sum_axis(axis);
                    let diff = gs.sub(&sm.hadamard(&sum_gs.broadcast_to(av.rows, av.cols)));
                    t_grads[input] = t_grads[input].add(&diff);
                }
                Op::LogSoftmax { input, axis } => {
                    let av = &values[input];
                    let sm = av.softmax(axis);
                    let g_broadcast = g.broadcast_to(av.rows, av.cols);
                    let sum_g = g_broadcast.sum_axis(axis);
                    let diff = g_broadcast.sub(&sm.hadamard(&sum_g.broadcast_to(av.rows, av.cols)));
                    t_grads[input] = t_grads[input].add(&diff);
                }

                Op::Transpose2d(a) => {
                    t_grads[a] = t_grads[a].add(&g.transpose());
                }

                Op::Concat { input_indices, row_counts } => {
                    let cols = nodes[input_indices[0]].shape.1;
                    let mut off = 0;
                    for k in 0..3 {
                        let a = input_indices[k];
                        if a == 0 && row_counts[k] == 0 {
                            continue;
                        }
                        let n = row_counts[k];
                        for r in 0..n {
                            for c in 0..cols {
                                t_grads[a].data[r * cols + c] += g.data[(off + r) * cols + c];
                            }
                        }
                        off += n;
                    }
                }
                Op::Slice { input_idx, start, len } => {
                    let c = values[input_idx].cols;
                    for r in 0..len {
                        for col in 0..c {
                            t_grads[input_idx].data[(start + r) * c + col] += g.data[r * c + col];
                        }
                    }
                }
                Op::SliceCols { input_idx, start, len } => {
                    let c = values[input_idx].cols;
                    for r in 0..values[input_idx].rows {
                        for col in 0..len {
                            t_grads[input_idx].data[r * c + (start + col)] += g.data[r * len + col];
                        }
                    }
                }

                Op::Embedding { table_idx, n_tokens: _ } => {
                    let vocab = values[table_idx].rows;
                    let d = values[table_idx].cols;
                    if let SavedData::Indices(ref indices) = nodes[i].saved {
                        for (i_tok, &idx_u) in indices.iter().enumerate() {
                            let idx_usize = idx_u as usize;
                            if idx_usize >= vocab {
                                continue; // safety guard
                            }
                            for j in 0..d {
                                t_grads[table_idx].data[idx_usize * d + j] += g.data[i_tok * d + j];
                            }
                        }
                    }
                }
                Op::Linear { input_idx, weight_idx, bias_idx } => {
                    let iv = &values[input_idx];
                    let wv = &values[weight_idx];
                    t_grads[input_idx] = t_grads[input_idx].add(&g.matmul(&wv.transpose()));
                    t_grads[weight_idx] = t_grads[weight_idx].add(&iv.transpose().matmul(&g));
                    // bias grad = sum over rows (only if no saved data)
                    if matches!(nodes[i].saved, SavedData::None) {
                        let bias_g = g.sum_axis(0);
                        t_grads[bias_idx] = t_grads[bias_idx].add(&bias_g);
                    }
                }
                Op::CausalMask { input_idx, seq_len } => {
                    let av = &values[input_idx];
                    let mut mask = Tensor::zeros(av.rows, av.cols);
                    for r in 0..av.rows {
                        for c in 0..av.cols {
                            let col_in_seq = c % seq_len;
                            let row_in_seq = r % seq_len;
                            mask.data[r * av.cols + c] = if col_in_seq > row_in_seq { 0.0 } else { 1.0 };
                        }
                    }
                    t_grads[input_idx] = t_grads[input_idx].add(&g.hadamard(&mask));
                }
                Op::Dropout { input_idx, mask_idx, .. } => {
                    let mv = &values[mask_idx];
                    t_grads[input_idx] = t_grads[input_idx].add(&g.hadamard(mv));
                    t_grads[mask_idx] = t_grads[mask_idx].add(&g.hadamard(&values[input_idx]));
                }
                Op::MaxPool2d { input_idx, c, h, w, kernel, stride } => {
                    let av = &values[input_idx];
                    let h_out = (h - kernel) / stride + 1;
                    let w_out = (w - kernel) / stride + 1;
                    let mut grad_in = Tensor::zeros(av.rows, av.cols);
                    for b in 0..av.rows {
                        for ch in 0..c {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let mut m = -f32::INFINITY;
                                    let mut mh = 0usize;
                                    let mut mw = 0usize;
                                    for kh in 0..kernel {
                                        for kw in 0..kernel {
                                            let ih = oh * stride + kh;
                                            let iw = ow * stride + kw;
                                            let idx_in = b * c * h * w + ch * h * w + ih * w + iw;
                                            let v = av.data[idx_in];
                                            if v > m {
                                                m = v;
                                                mh = ih;
                                                mw = iw;
                                            }
                                        }
                                    }
                                    let idx_out = b * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                                    let idx_in_max = b * c * h * w + ch * h * w + mh * w + mw;
                                    grad_in.data[idx_in_max] += g.data[idx_out];
                                }
                            }
                        }
                    }
                    t_grads[input_idx] = t_grads[input_idx].add(&grad_in);
                }
                Op::BatchNorm { input_idx, gamma_idx, beta_idx } => {
                    let gv = &values[gamma_idx];
                    let g_b = g.broadcast_to(values[input_idx].rows, values[input_idx].cols);
                    t_grads[input_idx] = t_grads[input_idx].add(&g_b.hadamard(gv));
                    t_grads[gamma_idx] = t_grads[gamma_idx].add(&g.sum_axis(0));
                    t_grads[beta_idx] = t_grads[beta_idx].add(&g.sum_axis(0));
                }
                Op::LayerNorm { input_idx, gamma_idx, beta_idx, .. } => {
                    let gv = &values[gamma_idx];
                    let g_b = g.broadcast_to(values[input_idx].rows, values[input_idx].cols);
                    t_grads[input_idx] = t_grads[input_idx].add(&g_b.hadamard(gv));
                    t_grads[gamma_idx] = t_grads[gamma_idx].add(&g.sum_axis(0));
                    t_grads[beta_idx] = t_grads[beta_idx].add(&g.sum_axis(0));
                }
                Op::Conv2dForward { input, weight, bias, batch, in_c, h, w, out_c, kernel, stride, pad } => {
                    let input_t = &values[input];
                    let weight_t = &values[weight];
                    let h_out = (h + 2 * pad - kernel) / stride + 1;
                    let w_out = (w + 2 * pad - kernel) / stride + 1;

                    if let Some(b_idx) = bias {
                        let mut db = Tensor::zeros(1, out_c);
                        for b_i in 0..batch {
                            for oc in 0..out_c {
                                for oh in 0..h_out {
                                    for ow in 0..w_out {
                                        let out_idx =
                                            b_i * out_c * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                                        db.data[oc] += g.data[out_idx];
                                    }
                                }
                            }
                        }
                        t_grads[b_idx] = t_grads[b_idx].add(&db);
                    }

                    let mut dw = Tensor::zeros(weight_t.rows, weight_t.cols);
                    let mut dx = Tensor::zeros(input_t.rows, input_t.cols);
                    for b_i in 0..batch {
                        for oc in 0..out_c {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let out_idx =
                                        b_i * out_c * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                                    let grad_out = g.data[out_idx];
                                    for ic in 0..in_c {
                                        for kh in 0..kernel {
                                            for kw in 0..kernel {
                                                let ih = oh as isize * stride as isize + kh as isize - pad as isize;
                                                let iw = ow as isize * stride as isize + kw as isize - pad as isize;
                                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                                    let ih_u = ih as usize;
                                                    let iw_u = iw as usize;
                                                    let in_idx =
                                                        b_i * in_c * h * w + ic * h * w + ih_u * w + iw_u;
                                                    let w_idx = oc * in_c * kernel * kernel
                                                        + ic * kernel * kernel
                                                        + kh * kernel
                                                        + kw;
                                                    dw.data[w_idx] += grad_out * input_t.data[in_idx];
                                                    dx.data[in_idx] += grad_out * weight_t.data[w_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    t_grads[weight] = t_grads[weight].add(&dw);
                    t_grads[input] = t_grads[input].add(&dx);
                }
                Op::Reshape(input_idx, old_rows, old_cols) => {
                    t_grads[input_idx] = t_grads[input_idx].add(&g.reshape(old_rows, old_cols));
                }
            }
        }

        // ---- reduce tensor grads to scalar f64 ----
        {
            let mut grads = self.grads.write().expect("ParallelTape grads lock poisoned");
            for i in 0..n {
                grads[i] = t_grads[i].sum() as f64;
            }
        }
    }

    /// Reset all gradients to zero.
    pub fn reset(&self) {
        let mut grads = self.grads.write().expect("ParallelTape grads lock poisoned");
        for g in grads.iter_mut() {
            *g = 0.0;
        }
    }
}

impl Default for ParallelTape {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================== //
//  Tests                                                             //
// ================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<ParallelTape>();
        assert_sync::<ParallelTape>();
    }

    #[test]
    fn test_alloc_backward_scale() {
        // f(x) = x * 2, df/dx = 2
        let tape = ParallelTape::new();
        let x = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let y = tape.alloc_node(Node {
            op: Op::Scale { input: x, scalar: 2.0 },
            shape: (1, 1),
            saved: SavedData::None,
        });
        tape.set_value(x, &[3.0]);
        tape.set_value(y, &[6.0]);

        tape.backward(y);
        assert!((tape.grad(x) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_tape_add() {
        // f(a,b) = a + b, df/da = df/db = 1
        let tape = ParallelTape::new();
        let a = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let b = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let y = tape.alloc_node(Node {
            op: Op::Add(a, b),
            shape: (1, 1),
            saved: SavedData::None,
        });
        tape.set_value(a, &[5.0]);
        tape.set_value(b, &[3.0]);
        tape.set_value(y, &[8.0]);

        tape.backward(y);
        assert!((tape.grad(a) - 1.0).abs() < 1e-6, "grad a = {}", tape.grad(a));
        assert!((tape.grad(b) - 1.0).abs() < 1e-6, "grad b = {}", tape.grad(b));
    }

    #[test]
    fn test_parallel_tape_mul() {
        // f(a,b) = a * b, df/da = b_val, df/db = a_val
        // a=3, b=4 => df/da = 4, df/db = 3
        let tape = ParallelTape::new();
        let a = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let b = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let y = tape.alloc_node(Node {
            op: Op::Mul(a, b),
            shape: (1, 1),
            saved: SavedData::None,
        });
        tape.set_value(a, &[3.0]);
        tape.set_value(b, &[4.0]);
        tape.set_value(y, &[12.0]);

        tape.backward(y);
        assert!((tape.grad(a) - 4.0).abs() < 1e-6, "grad a = {}", tape.grad(a));
        assert!((tape.grad(b) - 3.0).abs() < 1e-6, "grad b = {}", tape.grad(b));
    }

    #[test]
    fn test_parallel_tape_sequential_parity() {
        // Build the same graph (x*2 + 1) on both tapes and compare grads
        use crate::autodiff::reverse::Tape;

        // Sequential
        let seq = Tape::new();
        let sx = seq.input(Tensor::from_vec(vec![3.0], 1, 1));
        let sx_idx = sx.idx();
        let sy = sx.scale(2.0).add(seq.input(Tensor::from_vec(vec![1.0], 1, 1)));
        sy.backward();
        let seq_grad: f64 = seq.grad(sx_idx).sum() as f64;

        // Parallel: build graph manually
        let p = ParallelTape::new();
        let px = p.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let pc = p.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        let ps = p.alloc_node(Node {
            op: Op::Scale { input: px, scalar: 2.0 },
            shape: (1, 1),
            saved: SavedData::None,
        });
        let py = p.alloc_node(Node {
            op: Op::Add(ps, pc),
            shape: (1, 1),
            saved: SavedData::None,
        });

        p.set_value(px, &[3.0]);
        p.set_value(pc, &[1.0]);
        p.set_value(ps, &[6.0]);
        p.set_value(py, &[7.0]);

        p.backward(py);
        let p_grad = p.grad(px);

        assert!(
            (p_grad - seq_grad).abs() < 1e-5,
            "seq_grad={} p_grad={}",
            seq_grad,
            p_grad
        );
    }

    #[test]
    fn test_reset() {
        let tape = ParallelTape::new();
        let x = tape.alloc_node(Node {
            op: Op::Input,
            shape: (1, 1),
            saved: SavedData::None,
        });
        tape.set_value(x, &[1.0]);
        let y = tape.alloc_node(Node {
            op: Op::Scale { input: x, scalar: 2.0 },
            shape: (1, 1),
            saved: SavedData::None,
        });
        tape.set_value(y, &[2.0]);
        tape.backward(y);
        assert!((tape.grad(x) - 2.0).abs() < 1e-6);
        tape.reset();
        assert!((tape.grad(x)).abs() < 1e-12);
    }
}
