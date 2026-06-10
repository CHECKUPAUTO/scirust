// scirust-core/src/autodiff/reverse.rs
// Reverse-mode autodiff — compatible V10A/V11

use crate::nn::conv_utils::im2col_raw;
use crate::nn::rng::PcgEngine;
use crate::tensor::tensor_nd::TensorND;
use crate::tn::tt_decompose::{interleave_weight, reconstruct_matrix, TTCores};
use matrixmultiply::sgemm;
use std::cell::RefCell;

// ================================================================== //
//  GpuEngine trait — plug-in GPU acceleration without circular deps   //
// ================================================================== //

/// GPU engine trait for hardware-accelerated matmul in backward pass.
/// Implemented externally (e.g. by `scirust-gpu`) to avoid circular dependencies.
pub trait GpuEngine: std::fmt::Debug {
    /// GEMM: C = alpha * op(A) * op(B) + beta * C
    ///
    /// All matrices f32 row-major.
    /// `transpose_a` / `transpose_b`: if true, the corresponding matrix is
    /// implicitly transposed before multiplication.
    ///
    /// When not transposed:
    ///   - op(A) is `m × k`, stored in `a` (length m*k)
    ///   - op(B) is `k × n`, stored in `b` (length k*n)
    ///   - C is `m × n`, stored in `c` (length m*n)
    fn gemm(
        &self,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        transpose_a: bool,
        transpose_b: bool,
    );
}

// ================================================================== //
//  Tensor — 2D dense row-major                                       //
// ================================================================== //

#[derive(Debug, Clone)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }
    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "Tensor::from_vec size mismatch");
        Self { rows, cols, data }
    }
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Tensor::add shape mismatch");
        let mut out = self.clone();
        out.add_assign(other);
        out
    }
    pub fn add_assign(&mut self, other: &Tensor) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor::add_assign shape mismatch"
        );
        for (a, b) in self.data.iter_mut().zip(&other.data)
        {
            *a += b;
        }
    }
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Tensor::sub shape mismatch");
        let mut out = self.clone();
        out.sub_assign(other);
        out
    }
    pub fn sub_assign(&mut self, other: &Tensor) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor::sub_assign shape mismatch"
        );
        for (a, b) in self.data.iter_mut().zip(&other.data)
        {
            *a -= b;
        }
    }
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.hadamard(other)
    }
    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "Tensor::div shape mismatch");
        let mut out = self.clone();
        for (a, b) in out.data.iter_mut().zip(&other.data)
        {
            *a /= b;
        }
        out
    }
    pub fn hadamard(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor::hadamard shape mismatch"
        );
        let mut out = self.clone();
        out.hadamard_assign(other);
        out
    }
    pub fn hadamard_assign(&mut self, other: &Tensor) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor::hadamard_assign shape mismatch"
        );
        for (a, b) in self.data.iter_mut().zip(&other.data)
        {
            *a *= b;
        }
    }
    pub fn neg(&self) -> Tensor {
        self.scale(-1.0)
    }
    pub fn reciprocal(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = 1.0 / *x;
        }
        out
    }
    pub fn exp(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.exp();
        }
        out
    }
    pub fn log(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.ln();
        }
        out
    }
    pub fn sqrt(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.sqrt();
        }
        out
    }
    pub fn pow(&self, exp: f32) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.powf(exp);
        }
        out
    }
    pub fn sigmoid(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
        out
    }
    pub fn tanh(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.tanh();
        }
        out
    }
    pub fn sin(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.sin();
        }
        out
    }
    pub fn cos(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.cos();
        }
        out
    }
    pub fn tan(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.tan();
        }
        out
    }
    pub fn sinh(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.sinh();
        }
        out
    }
    pub fn cosh(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.cosh();
        }
        out
    }
    pub fn log10(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.log10();
        }
        out
    }
    pub fn asin(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.asin();
        }
        out
    }
    pub fn acos(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.acos();
        }
        out
    }
    pub fn atan(&self) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x = x.atan();
        }
        out
    }
    pub fn atan2(&self, x: &Tensor) -> Tensor {
        assert_eq!(self.shape(), x.shape(), "atan2: shape mismatch");
        let mut out = self.clone();
        for i in 0..self.data.len()
        {
            out.data[i] = self.data[i].atan2(x.data[i]);
        }
        out
    }
    pub fn scale(&self, s: f32) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data
        {
            *x *= s;
        }
        out
    }
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }
    pub fn sum_axis(&self, axis: u8) -> Tensor {
        if axis == 0
        {
            let mut out = Tensor::zeros(1, self.cols);
            for r in 0..self.rows
            {
                let row_off = r * self.cols;
                for c in 0..self.cols
                {
                    out.data[c] += self.data[row_off + c];
                }
            }
            out
        }
        else
        {
            let mut out = Tensor::zeros(self.rows, 1);
            for r in 0..self.rows
            {
                let mut s = 0.0f32;
                for c in 0..self.cols
                {
                    s += self.data[r * self.cols + c];
                }
                out.data[r] = s;
            }
            out
        }
    }
    pub fn mean_axis(&self, axis: u8) -> Tensor {
        let n = if axis == 0 { self.rows } else { self.cols } as f32;
        self.sum_axis(axis).scale(1.0 / n)
    }
    pub fn var_axis(&self, axis: u8) -> Tensor {
        let mean = self.mean_axis(axis);
        let diff = self.sub(&mean.broadcast_to(self.rows, self.cols));
        let sq = diff.hadamard(&diff);
        sq.mean_axis(axis)
    }
    pub fn max_axis(&self, axis: u8) -> Tensor {
        if axis == 0
        {
            let mut out = Tensor::zeros(1, self.cols);
            if self.rows > 0
            {
                out.data.copy_from_slice(&self.data[0..self.cols]);
                for r in 1..self.rows
                {
                    let row_off = r * self.cols;
                    for c in 0..self.cols
                    {
                        out.data[c] = out.data[c].max(self.data[row_off + c]);
                    }
                }
            }
            out
        }
        else
        {
            let mut out = Tensor::zeros(self.rows, 1);
            for r in 0..self.rows
            {
                let mut m = self.data[r * self.cols];
                for c in 1..self.cols
                {
                    m = m.max(self.data[r * self.cols + c]);
                }
                out.data[r] = m;
            }
            out
        }
    }
    pub fn softmax(&self, axis: u8) -> Tensor {
        let max = self.max_axis(axis);
        let shifted = self.sub(&max.broadcast_to(self.rows, self.cols));
        let exp = shifted.exp();
        let sum = exp.sum_axis(axis);
        exp.div(&sum.broadcast_to(self.rows, self.cols))
    }
    pub fn transpose(&self) -> Tensor {
        let mut out = Tensor::zeros(self.cols, self.rows);
        for r in 0..self.rows
        {
            for c in 0..self.cols
            {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.cols, other.rows,
            "matmul: inner dim mismatch {}x{} @ {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut out = Tensor::zeros(self.rows, other.cols);
        unsafe {
            sgemm(
                self.rows,
                self.cols,
                other.cols,
                1.0,
                self.data.as_ptr(),
                self.cols as isize,
                1,
                other.data.as_ptr(),
                other.cols as isize,
                1,
                0.0,
                out.data.as_mut_ptr(),
                out.cols as isize,
                1,
            );
        }
        out
    }

    pub fn reshape(&self, rows: usize, cols: usize) -> Tensor {
        assert_eq!(self.data.len(), rows * cols, "reshape: size mismatch");
        Tensor {
            rows,
            cols,
            data: self.data.clone(),
        }
    }
    pub fn broadcast_to(&self, rows: usize, cols: usize) -> Tensor {
        if self.rows == rows && self.cols == cols
        {
            return self.clone();
        }
        if self.rows == 1 && self.cols == cols
        {
            let mut out = Tensor::zeros(rows, cols);
            for r in 0..rows
            {
                for c in 0..cols
                {
                    out.data[r * cols + c] = self.data[c];
                }
            }
            out
        }
        else if self.rows == rows && self.cols == 1
        {
            let mut out = Tensor::zeros(rows, cols);
            for r in 0..rows
            {
                for c in 0..cols
                {
                    out.data[r * cols + c] = self.data[r];
                }
            }
            out
        }
        else if self.rows == 1 && self.cols == 1
        {
            Tensor::from_vec(vec![self.data[0]; rows * cols], rows, cols)
        }
        else
        {
            panic!(
                "broadcast_to: incompatible shapes ({},{}) -> ({},{})",
                self.rows, self.cols, rows, cols
            );
        }
    }
}

impl std::ops::Index<(usize, usize)> for Tensor {
    type Output = f32;
    fn index(&self, (row, col): (usize, usize)) -> &f32 {
        assert!(
            row < self.rows,
            "row {} out of bounds (rows={})",
            row,
            self.rows
        );
        assert!(
            col < self.cols,
            "col {} out of bounds (cols={})",
            col,
            self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f32 {
        assert!(
            row < self.rows,
            "row {} out of bounds (rows={})",
            row,
            self.rows
        );
        assert!(
            col < self.cols,
            "col {} out of bounds (cols={})",
            col,
            self.cols
        );
        &mut self.data[row * self.cols + col]
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::zeros(1, 1)
    }
}

// ================================================================== //
//  DeviceTensor                                                      //
// ================================================================== //

#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub inner: Tensor,
}

impl DeviceTensor {
    pub fn as_cpu(&self) -> &Tensor {
        &self.inner
    }
    pub fn cpu(t: Tensor) -> Self {
        Self { inner: t }
    }
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    pub fn scalar_value(&self) -> f32 {
        self.inner.data.iter().sum::<f32>()
    }
}

// ================================================================== //
//  SavedData                                                         //
// ================================================================== //

#[derive(Debug, Clone)]
pub enum SavedData {
    None,
    Mask(Tensor),
    Indices(Vec<u32>),
    Im2Col(Tensor),
    ConvInputShape {
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
    },
    /// Cached normalized input (x-μ)/σ for LayerNorm backward
    LayerNormNormed(Tensor),
    /// Cached normalized input for BatchNorm backward
    BatchNormNormed(Tensor),
    /// Cached flash attention online-softmax state: m (running max) and l (running sum)
    FlashAttentionState {
        m: Tensor,
        l: Tensor,
    },
    /// Cached input and reconstructed weight for TtContract backward
    TtContractState {
        input: Tensor,
        weight: Tensor,
    },
}

// ================================================================== //
//  Op                                                                //
// ================================================================== //

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Input,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    AddBroadcast(usize, usize),
    SubBroadcast(usize, usize),
    MulBroadcast(usize, usize),
    DivBroadcast(usize, usize),
    MatMul(usize, usize),
    MatMulGpu(usize, usize),
    Scale {
        input: usize,
        scalar: f32,
    },
    Neg(usize),
    Exp(usize),
    Log(usize),
    Sqrt(usize),
    Reciprocal(usize),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    Sinh(usize),
    Cosh(usize),
    Log10(usize),
    Asin(usize),
    Acos(usize),
    Atan(usize),
    Atan2(usize, usize),
    Pow {
        base: usize,
        exp: f32,
    },
    ReLU(usize),
    Sigmoid(usize),
    Tanh(usize),
    Sum(usize),
    SumAxis(usize, u8),
    MeanAxis(usize, u8),
    VarAxis(usize, u8),
    MaxAxis(usize, u8),
    Broadcast {
        input: usize,
        rows: usize,
        cols: usize,
    },
    Softmax {
        input: usize,
        axis: u8,
    },
    LogSoftmax {
        input: usize,
        axis: u8,
    },
    Transpose2d(usize),
    Concat {
        input_indices: [usize; 3],
        row_counts: [usize; 3],
    },
    Slice {
        input_idx: usize,
        start: usize,
        len: usize,
    },
    SliceCols {
        input_idx: usize,
        start: usize,
        len: usize,
    },
    Embedding {
        table_idx: usize,
        n_tokens: usize,
    },
    Linear {
        input_idx: usize,
        weight_idx: usize,
        bias_idx: usize,
    },
    CausalMask {
        input_idx: usize,
        seq_len: usize,
    },
    Dropout {
        input_idx: usize,
        mask_idx: usize,
        p: f32,
    },
    MaxPool2d {
        input_idx: usize,
        c: usize,
        h: usize,
        w: usize,
        kernel: usize,
        stride: usize,
    },
    BatchNorm {
        input_idx: usize,
        gamma_idx: usize,
        beta_idx: usize,
    },
    FlashAttention {
        q: usize,
        k: usize,
        v: usize,
        mask: Option<usize>,
        batch: usize,
        n_heads: usize,
        seq_len: usize,
        d_head: usize,
        scale: f32,
        block_size: usize,
    },
    LayerNorm {
        input_idx: usize,
        gamma_idx: usize,
        beta_idx: usize,
        eps: f32,
    },
    Conv2dForward {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
    },
    Conv2dTransposeForward {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        output_padding: usize,
    },
    Reshape(usize, usize, usize),
    FakeQuantize {
        input: usize,
        scale: f32,
        zero_point: i32,
    },
    TtContract {
        input_idx: usize,
        core_indices: [usize; 8],
        num_cores: usize,
        bias_idx: Option<usize>,
        in_dims: [usize; 8],
        out_dims: [usize; 8],
        ranks: [usize; 9],
        d: usize,
    },
}

// ================================================================== //
//  Node                                                              //
// ================================================================== //

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub shape: (usize, usize),
    pub saved: SavedData,
}

// ================================================================== //
//  Tape                                                              //
// ================================================================== //

#[derive(Debug)]
pub struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>,
    pub(crate) values: RefCell<Vec<DeviceTensor>>,
    pub(crate) grads: RefCell<Vec<Tensor>>,
    grad_enabled: RefCell<bool>,
    pub(crate) gpu_engine: RefCell<Option<Box<dyn GpuEngine>>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
            values: RefCell::new(Vec::new()),
            grads: RefCell::new(Vec::new()),
            grad_enabled: RefCell::new(true),
            gpu_engine: RefCell::new(None),
        }
    }

    /// Attach a GPU engine for accelerated backward passes.
    pub fn with_gpu_engine(self, engine: impl GpuEngine + 'static) -> Self {
        *self.gpu_engine.borrow_mut() = Some(Box::new(engine));
        self
    }

    /// Replace the GPU engine at runtime.
    pub fn set_gpu_engine(&self, engine: impl GpuEngine + 'static) {
        *self.gpu_engine.borrow_mut() = Some(Box::new(engine));
    }

    /// Remove the GPU engine, falling back to CPU-only.
    pub fn clear_gpu_engine(&self) {
        *self.gpu_engine.borrow_mut() = None;
    }

    pub fn set_grad_enabled(&self, enabled: bool) {
        *self.grad_enabled.borrow_mut() = enabled;
    }

    pub fn is_grad_enabled(&self) -> bool {
        *self.grad_enabled.borrow()
    }

    pub fn no_grad<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let prev = self.is_grad_enabled();
        self.set_grad_enabled(false);
        let result = f();
        self.set_grad_enabled(prev);
        result
    }

    pub fn num_parameters(&self) -> usize {
        0
    }

    pub fn input(&self, t: Tensor) -> Var<'_> {
        let idx = self.push_with_saved(Op::Input, DeviceTensor::cpu(t.clone()), SavedData::None);
        self.values.borrow_mut()[idx] = DeviceTensor::cpu(t);
        Var { tape: self, idx }
    }

    pub fn push_with_saved(&self, op: Op, value: DeviceTensor, saved: SavedData) -> usize {
        let shape = value.shape();
        if !self.is_grad_enabled()
        {
            // Forward seul : on pousse un Input inerte (pas de graph)
            let mut nodes = self.nodes.borrow_mut();
            let idx = nodes.len();
            nodes.push(Node {
                op: Op::Input,
                shape,
                saved: SavedData::None,
            });
            self.values.borrow_mut().push(value);
            self.grads
                .borrow_mut()
                .push(Tensor::zeros(shape.0, shape.1));
            return idx;
        }
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node { op, shape, saved });
        self.values.borrow_mut().push(value);
        self.grads
            .borrow_mut()
            .push(Tensor::zeros(shape.0, shape.1));
        idx
    }

    pub fn value(&self, idx: usize) -> Tensor {
        self.values.borrow()[idx].as_cpu().clone()
    }

    pub fn shape(&self, idx: usize) -> (usize, usize) {
        self.values.borrow()[idx].shape()
    }

    pub fn zeros(&self, rows: usize, cols: usize) -> Var<'_> {
        self.input(Tensor::zeros(rows, cols))
    }

    pub fn grad(&self, idx: usize) -> Tensor {
        self.grads.borrow()[idx].clone()
    }

    /// Clipping par valeur : chaque element du gradient est borne dans [-max, max].
    pub fn clip_grad_value(&self, max: f32) {
        let mut grads = self.grads.borrow_mut();
        for g in grads.iter_mut()
        {
            for v in g.data.iter_mut()
            {
                *v = v.clamp(-max, max);
            }
        }
    }

    /// Clipping par norme globale (Pascanu et al., 2013) :
    /// si ||g|| > max_norm, on rescale tous les gradients par max_norm / ||g||.
    pub fn clip_grad_norm(&self, max_norm: f32) {
        let mut grads = self.grads.borrow_mut();
        let mut total_norm_sq = 0.0f32;
        for g in grads.iter()
        {
            for v in g.data.iter()
            {
                total_norm_sq += v * v;
            }
        }
        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm && total_norm > 1e-12
        {
            let scale = max_norm / total_norm;
            for g in grads.iter_mut()
            {
                for v in g.data.iter_mut()
                {
                    *v *= scale;
                }
            }
        }
    }

    pub fn set_value(&self, idx: usize, value: Tensor) {
        self.values.borrow_mut()[idx] = DeviceTensor::cpu(value);
    }

    pub fn backward(&self, idx: usize) {
        let nodes = self.nodes.borrow();
        let values = self.values.borrow();
        let mut grads = self.grads.borrow_mut();
        let n = nodes.len();
        assert!(idx < n, "backward: idx {} out of bounds ({} nodes)", idx, n);

        // reset gradients
        for i in 0..n
        {
            let (r, c) = nodes[i].shape;
            grads[i] = Tensor::zeros(r, c);
        }

        // seed
        let (r, c) = nodes[idx].shape;
        grads[idx] = Tensor::from_vec(vec![1.0; r * c], r, c);

        for i in (0..=idx).rev()
        {
            let g = grads[i].clone();
            if g.data.iter().all(|&x| x == 0.0)
            {
                continue;
            }

            match nodes[i].op
            {
                Op::Input =>
                {},
                Op::Add(a, b) =>
                {
                    grads[a].add_assign(&g);
                    grads[b].add_assign(&g);
                },
                Op::Sub(a, b) =>
                {
                    grads[a].add_assign(&g);
                    grads[b].sub_assign(&g);
                },
                Op::Mul(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();
                    grads[a].add_assign(&g.hadamard(bv));
                    grads[b].add_assign(&g.hadamard(av));
                },
                Op::Div(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();
                    let b_recip = bv.reciprocal();
                    let a_over_b2 = av.hadamard(&b_recip.hadamard(&b_recip));
                    grads[a].add_assign(&g.hadamard(&b_recip));
                    grads[b].sub_assign(&g.hadamard(&a_over_b2));
                },
                Op::AddBroadcast(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();
                    grads[a].add_assign(&g);
                    if bv.rows == 1 && bv.cols == av.cols
                    {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[c] += g.data[off + c];
                            }
                        }
                        grads[b].add_assign(&db);
                    }
                    else if bv.rows == av.rows && bv.cols == 1
                    {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[r] += g.data[off + c];
                            }
                        }
                        grads[b].add_assign(&db);
                    }
                    else if bv.rows == 1 && bv.cols == 1
                    {
                        let s: f32 = g.data.iter().sum();
                        grads[b].add_assign(&Tensor::from_vec(vec![s], 1, 1));
                    }
                    else
                    {
                        grads[b].add_assign(&g);
                    }
                },
                Op::SubBroadcast(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();
                    grads[a].add_assign(&g);
                    if bv.rows == 1 && bv.cols == av.cols
                    {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[c] += g.data[off + c];
                            }
                        }
                        grads[b].sub_assign(&db);
                    }
                    else if bv.rows == av.rows && bv.cols == 1
                    {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[r] += g.data[off + c];
                            }
                        }
                        grads[b].sub_assign(&db);
                    }
                    else if bv.rows == 1 && bv.cols == 1
                    {
                        let s: f32 = g.data.iter().sum();
                        grads[b].sub_assign(&Tensor::from_vec(vec![s], 1, 1));
                    }
                    else
                    {
                        grads[b].sub_assign(&g);
                    }
                },
                Op::MulBroadcast(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();
                    grads[a].add_assign(&g.hadamard(&bv.broadcast_to(g.rows, g.cols)));
                    if bv.rows == 1 && bv.cols == av.cols
                    {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[c] += g.data[off + c] * av.data[off + c];
                            }
                        }
                        grads[b].add_assign(&db);
                    }
                    else if bv.rows == av.rows && bv.cols == 1
                    {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[r] += g.data[off + c] * av.data[off + c];
                            }
                        }
                        grads[b].add_assign(&db);
                    }
                    else if bv.rows == 1 && bv.cols == 1
                    {
                        let s: f32 = g
                            .data
                            .iter()
                            .zip(av.data.iter())
                            .map(|(&gi, &ai)| gi * ai)
                            .sum();
                        grads[b].add_assign(&Tensor::from_vec(vec![s], 1, 1));
                    }
                    else
                    {
                        grads[b].add_assign(&g.hadamard(&av.broadcast_to(g.rows, g.cols)));
                    }
                },
                Op::DivBroadcast(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();
                    let b_recip = bv.reciprocal();
                    grads[a].add_assign(&g.hadamard(&b_recip.broadcast_to(g.rows, g.cols)));
                    if bv.rows == 1 && bv.cols == av.cols
                    {
                        let mut db = Tensor::zeros(1, bv.cols);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[c] -=
                                    g.data[off + c] * av.data[off + c] / (bv.data[c] * bv.data[c]);
                            }
                        }
                        grads[b].add_assign(&db);
                    }
                    else if bv.rows == av.rows && bv.cols == 1
                    {
                        let mut db = Tensor::zeros(bv.rows, 1);
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                db.data[r] -=
                                    g.data[off + c] * av.data[off + c] / (bv.data[r] * bv.data[r]);
                            }
                        }
                        grads[b].add_assign(&db);
                    }
                    else if bv.rows == 1 && bv.cols == 1
                    {
                        let s: f32 = g
                            .data
                            .iter()
                            .zip(av.data.iter())
                            .map(|(&gi, &ai)| -gi * ai / (bv.data[0] * bv.data[0]))
                            .sum();
                        grads[b].add_assign(&Tensor::from_vec(vec![s], 1, 1));
                    }
                    else
                    {
                        let a_over_b2 =
                            av.hadamard(&b_recip.hadamard(&b_recip).broadcast_to(g.rows, g.cols));
                        grads[b].sub_assign(&g.hadamard(&a_over_b2));
                    }
                },
                Op::MatMul(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();

                    // ga = g @ b.T
                    // (M x N) @ (K x N).T -> (M x K)
                    let ga = &mut grads[a];
                    unsafe {
                        sgemm(
                            g.rows,
                            g.cols,
                            bv.rows,
                            1.0,
                            g.data.as_ptr(),
                            g.cols as isize,
                            1,
                            bv.data.as_ptr(),
                            1,
                            bv.cols as isize,
                            1.0,
                            ga.data.as_mut_ptr(),
                            ga.cols as isize,
                            1,
                        );
                    }

                    // gb = a.T @ g
                    // (M x K).T @ (M x N) -> (K x N)
                    let gb = &mut grads[b];
                    unsafe {
                        sgemm(
                            av.cols,
                            av.rows,
                            g.cols,
                            1.0,
                            av.data.as_ptr(),
                            1,
                            av.cols as isize,
                            g.data.as_ptr(),
                            g.cols as isize,
                            1,
                            1.0,
                            gb.data.as_mut_ptr(),
                            gb.cols as isize,
                            1,
                        );
                    }
                },
                Op::MatMulGpu(a, b) =>
                {
                    let av = &values[a].as_cpu();
                    let bv = &values[b].as_cpu();
                    let m = av.rows;          // M
                    let k = av.cols;          // K = bv.rows
                    let n = bv.cols;          // N
                    debug_assert_eq!(bv.rows, k);

                    // Try GPU engine first
                    if let Some(ref engine) = *self.gpu_engine.borrow()
                    {
                        let ga = &mut grads[a];
                        // ga += g @ b.T  (M×K = M×N * K×N^T)
                        let mut ga_data = ga.data.clone();
                        engine.gemm(1.0, g.data.as_slice(), bv.data.as_slice(), 1.0, &mut ga_data, m, n, k, false, true);
                        ga.data = ga_data;

                        let gb = &mut grads[b];
                        // gb += a.T @ g  (K×N = M×K^T * M×N)
                        let mut gb_data = gb.data.clone();
                        engine.gemm(1.0, av.data.as_slice(), g.data.as_slice(), 1.0, &mut gb_data, k, m, n, true, false);
                        gb.data = gb_data;
                    }
                    else
                    {
                        // CPU fallback
                        let ga = &mut grads[a];
                        unsafe {
                            sgemm(
                                g.rows,
                                g.cols,
                                bv.rows,
                                1.0,
                                g.data.as_ptr(),
                                g.cols as isize,
                                1,
                                bv.data.as_ptr(),
                                1,
                                bv.cols as isize,
                                1.0,
                                ga.data.as_mut_ptr(),
                                ga.cols as isize,
                                1,
                            );
                        }
                        let gb = &mut grads[b];
                        unsafe {
                            sgemm(
                                av.cols,
                                av.rows,
                                g.cols,
                                1.0,
                                av.data.as_ptr(),
                                1,
                                av.cols as isize,
                                g.data.as_ptr(),
                                g.cols as isize,
                                1,
                                1.0,
                                gb.data.as_mut_ptr(),
                                gb.cols as isize,
                                1,
                            );
                        }
                    }
                },
                Op::Scale { input, scalar } =>
                {
                    grads[input].add_assign(&g.scale(scalar));
                },
                Op::Neg(a) =>
                {
                    grads[a].sub_assign(&g);
                },
                Op::Exp(a) =>
                {
                    // dL/dx = g * exp(x) = g * value(node_i)
                    let val = &values[i].as_cpu();
                    grads[a].add_assign(&g.hadamard(val));
                },
                Op::Log(a) =>
                {
                    let av = &values[a].as_cpu();
                    grads[a].add_assign(&g.hadamard(&av.reciprocal()));
                },
                Op::Sqrt(a) =>
                {
                    let av = &values[a].as_cpu();
                    let two_sqrt = av.sqrt().scale(2.0);
                    grads[a] = grads[a].add(&g.hadamard(&two_sqrt.reciprocal()));
                },
                Op::Reciprocal(a) =>
                {
                    let av = &values[a].as_cpu();
                    let mut denom = av.hadamard(av);
                    for d in &mut denom.data
                    {
                        *d = 1.0 / (*d + 1e-10);
                    }
                    let minus_one_over_x2 = denom.scale(-1.0);
                    grads[a] = grads[a].add(&g.hadamard(&minus_one_over_x2));
                },
                Op::Sin(a) =>
                {
                    let av = values[a].as_cpu();
                    grads[a] = grads[a].add(&g.hadamard(&av.cos()));
                },
                Op::Cos(a) =>
                {
                    let av = values[a].as_cpu();
                    grads[a] = grads[a].sub(&g.hadamard(&av.sin()));
                },
                Op::Tan(a) =>
                {
                    let av = values[a].as_cpu();
                    let cos_v = av.cos();
                    grads[a] = grads[a].add(&g.hadamard(&cos_v.hadamard(&cos_v).reciprocal()));
                },
                Op::Sinh(a) =>
                {
                    let av = values[a].as_cpu();
                    grads[a] = grads[a].add(&g.hadamard(&av.cosh()));
                },
                Op::Cosh(a) =>
                {
                    let av = values[a].as_cpu();
                    grads[a] = grads[a].add(&g.hadamard(&av.sinh()));
                },
                Op::Log10(a) =>
                {
                    let av = values[a].as_cpu();
                    let ln10 = std::f32::consts::LN_10;
                    grads[a] = grads[a].add(&g.hadamard(&av.reciprocal().scale(1.0 / ln10)));
                },
                Op::Asin(a) =>
                {
                    let av = values[a].as_cpu();
                    let ones = Tensor::from_vec(vec![1.0f32; av.data.len()], av.rows, av.cols);
                    let denom = ones.sub(&av.hadamard(av)).sqrt();
                    grads[a] = grads[a].add(&g.hadamard(&denom.reciprocal()));
                },
                Op::Acos(a) =>
                {
                    let av = values[a].as_cpu();
                    let ones = Tensor::from_vec(vec![1.0f32; av.data.len()], av.rows, av.cols);
                    let denom = ones.sub(&av.hadamard(av)).sqrt();
                    grads[a] = grads[a].sub(&g.hadamard(&denom.reciprocal()));
                },
                Op::Atan(a) =>
                {
                    let av = values[a].as_cpu();
                    let ones = Tensor::from_vec(vec![1.0f32; av.data.len()], av.rows, av.cols);
                    let denom = ones.add(&av.hadamard(av));
                    grads[a] = grads[a].add(&g.hadamard(&denom.reciprocal()));
                },
                Op::Atan2(a, b) =>
                {
                    let yv = values[a].as_cpu();
                    let xv = values[b].as_cpu();
                    let denom = xv.hadamard(xv).add(&yv.hadamard(yv));
                    // add epsilon guard element-wise for numerical stability at (0,0)
                    let mut denom_safe = denom.clone();
                    for d in &mut denom_safe.data
                    {
                        *d += 1e-10;
                    }
                    let deriv_y = xv.hadamard(&denom_safe.reciprocal());
                    let deriv_x = yv.hadamard(&denom_safe.reciprocal()).neg();
                    grads[a] = grads[a].add(&g.hadamard(&deriv_y));
                    grads[b] = grads[b].add(&g.hadamard(&deriv_x));
                },
                Op::Pow { base, exp } =>
                {
                    let av = &values[base].as_cpu();
                    let deriv = av.pow(exp - 1.0).scale(exp);
                    grads[base] = grads[base].add(&g.hadamard(&deriv));
                },
                Op::ReLU(a) =>
                {
                    let av = &values[a].as_cpu();
                    let ga = &mut grads[a];
                    for j in 0..av.data.len()
                    {
                        if av.data[j] > 0.0
                        {
                            ga.data[j] += g.data[j];
                        }
                    }
                },
                Op::Sigmoid(a) =>
                {
                    // dL/dx = g * sig(x) * (1 - sig(x)) = g * val * (1 - val)
                    let sig = &values[i].as_cpu();
                    for j in 0..sig.data.len()
                    {
                        let s = sig.data[j];
                        grads[a].data[j] += g.data[j] * s * (1.0 - s);
                    }
                },
                Op::Tanh(a) =>
                {
                    // dL/dx = g * (1 - tanh(x)^2) = g * (1 - val^2)
                    let t = &values[i].as_cpu();
                    for j in 0..t.data.len()
                    {
                        let val = t.data[j];
                        grads[a].data[j] += g.data[j] * (1.0 - val * val);
                    }
                },
                Op::Sum(a) =>
                {
                    let av = &values[a].as_cpu();
                    grads[a] = grads[a].add(&g.broadcast_to(av.rows, av.cols));
                },
                Op::SumAxis(a, _axis) =>
                {
                    let av = &values[a].as_cpu();
                    grads[a] = grads[a].add(&g.broadcast_to(av.rows, av.cols));
                },
                Op::MeanAxis(a, axis) =>
                {
                    let av = &values[a].as_cpu();
                    let n = if axis == 0 { av.rows } else { av.cols } as f32;
                    grads[a] = grads[a].add(&g.scale(1.0 / n).broadcast_to(av.rows, av.cols));
                },
                Op::VarAxis(a, axis) =>
                {
                    let av = &values[a].as_cpu();
                    let n = if axis == 0 { av.rows } else { av.cols } as f32;
                    let mean = av.mean_axis(axis);
                    let diff = av.sub(&mean.broadcast_to(av.rows, av.cols));
                    let two_over_n = 2.0 / n;
                    grads[a] = grads[a].add(
                        &g.scale(two_over_n)
                            .broadcast_to(av.rows, av.cols)
                            .hadamard(&diff),
                    );
                },
                Op::MaxAxis(a, axis) =>
                {
                    let av = &values[a].as_cpu();
                    let max_v = av.max_axis(axis);
                    let mut mask = Tensor::zeros(av.rows, av.cols);
                    if axis == 0
                    {
                        for c in 0..av.cols
                        {
                            let m = max_v.data[c];
                            for r in 0..av.rows
                            {
                                if (av.data[r * av.cols + c] - m).abs() < 1e-6
                                {
                                    mask.data[r * av.cols + c] = 1.0;
                                }
                            }
                        }
                    }
                    else
                    {
                        for r in 0..av.rows
                        {
                            let m = max_v.data[r];
                            for c in 0..av.cols
                            {
                                if (av.data[r * av.cols + c] - m).abs() < 1e-6
                                {
                                    mask.data[r * av.cols + c] = 1.0;
                                }
                            }
                        }
                    }
                    grads[a] = grads[a].add(&g.broadcast_to(av.rows, av.cols).hadamard(&mask));
                },
                Op::Broadcast { input, rows, cols } =>
                {
                    let av = &values[input].as_cpu();
                    let g_sum = if av.rows == rows && av.cols == cols
                    {
                        g.clone()
                    }
                    else if av.rows == 1 && av.cols == cols
                    {
                        g.sum_axis(0)
                    }
                    else if av.rows == rows && av.cols == 1
                    {
                        g.sum_axis(1)
                    }
                    else if av.rows == 1 && av.cols == 1
                    {
                        Tensor::from_vec(vec![g.sum()], 1, 1)
                    }
                    else
                    {
                        panic!(
                            "Broadcast backward: unsupported shape ({},{}) -> ({},{})",
                            av.rows, av.cols, rows, cols
                        );
                    };
                    grads[input] = grads[input].add(&g_sum);
                },
                Op::Softmax { input, axis } =>
                {
                    let av = &values[input].as_cpu();
                    let sm = av.softmax(axis);
                    let g_broadcast = g.broadcast_to(av.rows, av.cols);
                    let gs = g_broadcast.hadamard(&sm);
                    let sum_gs = gs.sum_axis(axis);
                    let diff = gs.sub(&sm.hadamard(&sum_gs.broadcast_to(av.rows, av.cols)));
                    grads[input] = grads[input].add(&diff);
                },
                Op::LogSoftmax { input, axis } =>
                {
                    let av = &values[input].as_cpu();
                    let sm = av.softmax(axis);
                    let g_broadcast = g.broadcast_to(av.rows, av.cols);
                    let sum_g = g_broadcast.sum_axis(axis);
                    let diff = g_broadcast.sub(&sm.hadamard(&sum_g.broadcast_to(av.rows, av.cols)));
                    grads[input] = grads[input].add(&diff);
                },
                Op::Transpose2d(a) =>
                {
                    grads[a] = grads[a].add(&g.transpose());
                },
                Op::Concat {
                    input_indices,
                    row_counts,
                } =>
                {
                    let mut off = 0;
                    for k in 0..3
                    {
                        let a = input_indices[k];
                        if a == 0 && row_counts[k] == 0
                        {
                            continue;
                        }
                        let av = &values[a].as_cpu();
                        let n = av.rows;
                        let c = av.cols;
                        for r in 0..n
                        {
                            for col in 0..c
                            {
                                grads[a].data[r * c + col] += g.data[(off + r) * c + col];
                            }
                        }
                        off += n;
                    }
                },
                Op::Slice {
                    input_idx,
                    start,
                    len,
                } =>
                {
                    let av = &values[input_idx].as_cpu();
                    let c = av.cols;
                    for r in 0..len
                    {
                        for col in 0..c
                        {
                            grads[input_idx].data[(start + r) * c + col] += g.data[r * c + col];
                        }
                    }
                },
                Op::SliceCols {
                    input_idx,
                    start,
                    len,
                } =>
                {
                    let av = &values[input_idx].as_cpu();
                    let c = av.cols;
                    for r in 0..av.rows
                    {
                        for col in 0..len
                        {
                            grads[input_idx].data[r * c + (start + col)] += g.data[r * len + col];
                        }
                    }
                },
                Op::Embedding {
                    table_idx,
                    n_tokens: _,
                } =>
                {
                    let table = &values[table_idx].as_cpu();
                    let vocab = table.rows;
                    let d = table.cols;
                    if let SavedData::Indices(ref indices) = nodes[i].saved
                    {
                        for (i_tok, &idx_u) in indices.iter().enumerate()
                        {
                            let idx_usize = idx_u as usize;
                            assert!(
                                idx_usize < vocab,
                                "Embedding backward: index {} >= vocab {}",
                                idx_usize,
                                vocab
                            );
                            for j in 0..d
                            {
                                grads[table_idx].data[idx_usize * d + j] += g.data[i_tok * d + j];
                            }
                        }
                    }
                },
                Op::Linear {
                    input_idx,
                    weight_idx,
                    bias_idx,
                } =>
                {
                    let iv = &values[input_idx].as_cpu();
                    let wv = &values[weight_idx].as_cpu();

                    // d_input = g @ w.T
                    let gi = &mut grads[input_idx];
                    unsafe {
                        sgemm(
                            g.rows,
                            g.cols,
                            wv.rows,
                            1.0,
                            g.data.as_ptr(),
                            g.cols as isize,
                            1,
                            wv.data.as_ptr(),
                            1,
                            wv.cols as isize,
                            1.0,
                            gi.data.as_mut_ptr(),
                            gi.cols as isize,
                            1,
                        );
                    }

                    // d_weight = input.T @ g
                    let gw = &mut grads[weight_idx];
                    unsafe {
                        sgemm(
                            iv.cols,
                            iv.rows,
                            g.cols,
                            1.0,
                            iv.data.as_ptr(),
                            1,
                            iv.cols as isize,
                            g.data.as_ptr(),
                            g.cols as isize,
                            1,
                            1.0,
                            gw.data.as_mut_ptr(),
                            gw.cols as isize,
                            1,
                        );
                    }

                    if let SavedData::None = nodes[i].saved
                    {
                        // bias grad = sum over batch dim (rows)
                        let gb = &mut grads[bias_idx];
                        for r in 0..g.rows
                        {
                            let off = r * g.cols;
                            for c in 0..g.cols
                            {
                                gb.data[c] += g.data[off + c];
                            }
                        }
                    }
                },
                Op::CausalMask { input_idx, seq_len } =>
                {
                    let av = &values[input_idx].as_cpu();
                    let mut mask = Tensor::zeros(av.rows, av.cols);
                    for r in 0..av.rows
                    {
                        for c in 0..av.cols
                        {
                            let col_in_seq = c % seq_len;
                            let row_in_seq = r % seq_len;
                            if col_in_seq > row_in_seq
                            {
                                mask.data[r * av.cols + c] = 0.0;
                            }
                            else
                            {
                                mask.data[r * av.cols + c] = 1.0;
                            }
                        }
                    }
                    grads[input_idx] = grads[input_idx].add(&g.hadamard(&mask));
                },
                Op::Dropout {
                    input_idx,
                    mask_idx,
                    ..
                } =>
                {
                    let mv = &values[mask_idx].as_cpu();
                    let iv = &values[input_idx].as_cpu();
                    if input_idx == mask_idx
                    {
                        let gi = &mut grads[input_idx];
                        for j in 0..gi.data.len()
                        {
                            gi.data[j] += g.data[j] * (mv.data[j] + iv.data[j]);
                        }
                    }
                    else if input_idx < mask_idx
                    {
                        let (left, right) = grads.split_at_mut(mask_idx);
                        let gi = &mut left[input_idx];
                        let gm = &mut right[0];
                        for j in 0..gi.data.len()
                        {
                            gi.data[j] += g.data[j] * mv.data[j];
                            gm.data[j] += g.data[j] * iv.data[j];
                        }
                    }
                    else
                    {
                        let (left, right) = grads.split_at_mut(input_idx);
                        let gm = &mut left[mask_idx];
                        let gi = &mut right[0];
                        for j in 0..gi.data.len()
                        {
                            gi.data[j] += g.data[j] * mv.data[j];
                            gm.data[j] += g.data[j] * iv.data[j];
                        }
                    }
                },
                Op::MaxPool2d {
                    input_idx,
                    c,
                    h,
                    w,
                    kernel,
                    stride,
                } =>
                {
                    let av = &values[input_idx].as_cpu();
                    let h_out = (h - kernel) / stride + 1;
                    let w_out = (w - kernel) / stride + 1;
                    let mut grad_in = Tensor::zeros(av.rows, av.cols);
                    for b in 0..av.rows
                    {
                        for ch in 0..c
                        {
                            for oh in 0..h_out
                            {
                                for ow in 0..w_out
                                {
                                    let mut m = -f32::INFINITY;
                                    let mut mh = 0usize;
                                    let mut mw = 0usize;
                                    for kh in 0..kernel
                                    {
                                        for kw in 0..kernel
                                        {
                                            let ih = oh * stride + kh;
                                            let iw = ow * stride + kw;
                                            let idx_in = b * c * h * w + ch * h * w + ih * w + iw;
                                            let v = av.data[idx_in];
                                            if v > m
                                            {
                                                m = v;
                                                mh = ih;
                                                mw = iw;
                                            }
                                        }
                                    }
                                    let idx_out = b * c * h_out * w_out
                                        + ch * h_out * w_out
                                        + oh * w_out
                                        + ow;
                                    let idx_in_max = b * c * h * w + ch * h * w + mh * w + mw;
                                    grad_in.data[idx_in_max] += g.data[idx_out];
                                }
                            }
                        }
                    }
                    grads[input_idx] = grads[input_idx].add(&grad_in);
                },
                Op::BatchNorm {
                    input_idx,
                    gamma_idx,
                    beta_idx,
                } =>
                {
                    // Analytic backward for BatchNorm (same as LayerNorm but per-channel):
                    // y = gamma * (x - mu)/sigma + beta
                    // dL/dx = (gamma / sigma) * (dL/dy - mean(dL/dy) - x_norm * mean(dL/dy * x_norm))
                    //
                    // Approximation: dL/dx = gamma * (dL/dy - mean(dL/dy)) / sigma
                    // Full fix requires caching x_norm in SavedData::BatchNormNormed.
                    let input = &values[input_idx].as_cpu();
                    let (rows, cols) = input.shape();
                    let mut grad_x = Tensor::zeros(rows, cols);
                    let g_v = &values[gamma_idx].as_cpu();

                    for r in 0..rows
                    {
                        let mut mean = 0.0f32;
                        let mut var = 0.0f32;
                        for c in 0..cols
                        {
                            mean += input.data[r * cols + c];
                        }
                        mean /= cols as f32;
                        for c in 0..cols
                        {
                            let d = input.data[r * cols + c] - mean;
                            var += d * d;
                        }
                        var = var / cols as f32 + 1e-5f32;
                        let sigma = var.sqrt();

                        let mut g_mean = 0.0f32;
                        for c in 0..cols
                        {
                            g_mean += g.data[r * cols + c];
                        }
                        g_mean /= cols as f32;

                        for c in 0..cols
                        {
                            grad_x.data[r * cols + c] =
                                g_v.data[c] * (g.data[r * cols + c] - g_mean) / sigma;
                        }
                    }
                    grads[input_idx] = grads[input_idx].add(&grad_x);
                    grads[gamma_idx] = grads[gamma_idx].add(&g.sum_axis(0));
                    grads[beta_idx] = grads[beta_idx].add(&g.sum_axis(0));
                },
                Op::LayerNorm {
                    input_idx,
                    gamma_idx,
                    beta_idx,
                    eps: _eps,
                } =>
                {
                    // Analytic backward for LayerNorm using cached normalized input:
                    // y = gamma * x_norm + beta,  where x_norm = (x - mu)/sigma
                    // dL/dbeta = sum(dL/dy, axis=0)
                    // dL/dgamma = sum(dL/dy * x_norm, axis=0)
                    // dL/dx = (gamma / sigma) * (dL/dy - mean(dL/dy, axis=1) - x_norm * mean(dL/dy * x_norm, axis=1))
                    let x_norm = match &nodes[i].saved
                    {
                        SavedData::LayerNormNormed(t) => Some(t),
                        _ => None,
                    };
                    let input = &values[input_idx].as_cpu();
                    let (rows, cols) = input.shape();
                    let mut grad_x = Tensor::zeros(rows, cols);
                    let g_v = &values[gamma_idx].as_cpu();
                    let n = cols as f32;

                    if let Some(norm) = x_norm
                    {
                        // Full analytic backward with cached x_norm
                        for r in 0..rows
                        {
                            let mut g_mean = 0.0f32;
                            let mut gxnorm_mean = 0.0f32;
                            for c in 0..cols
                            {
                                g_mean += g.data[r * cols + c];
                                gxnorm_mean += g.data[r * cols + c] * norm.data[r * cols + c];
                            }
                            g_mean /= n;
                            gxnorm_mean /= n;

                            // Recompute sigma
                            let mut mean = 0.0f32;
                            let mut var = 0.0f32;
                            for c in 0..cols
                            {
                                mean += input.data[r * cols + c];
                            }
                            mean /= n;
                            for c in 0..cols
                            {
                                let d = input.data[r * cols + c] - mean;
                                var += d * d;
                            }
                            var /= n;

                            // Use safe epsilon for numerical stability
                            let eps_val = if var < 1e-12 { 1e-6 } else { 0.0 };
                            let sigma = (var + eps_val).sqrt();

                            for c in 0..cols
                            {
                                grad_x.data[r * cols + c] = (g_v.data[c] / sigma)
                                    * (g.data[r * cols + c]
                                        - g_mean
                                        - norm.data[r * cols + c] * gxnorm_mean);
                            }
                        }
                    }
                    else
                    {
                        // Fallback: approximate backward (no cached normed)
                        for r in 0..rows
                        {
                            let mut mean = 0.0f32;
                            let mut var = 0.0f32;
                            for c in 0..cols
                            {
                                mean += input.data[r * cols + c];
                            }
                            mean /= n;
                            for c in 0..cols
                            {
                                let d = input.data[r * cols + c] - mean;
                                var += d * d;
                            }
                            var /= n;
                            let eps_val = if var < 1e-12 { 1e-6 } else { 0.0 };
                            let sigma = (var + eps_val).sqrt();
                            let mut g_mean = 0.0f32;
                            for c in 0..cols
                            {
                                g_mean += g.data[r * cols + c];
                            }
                            g_mean /= n;
                            for c in 0..cols
                            {
                                grad_x.data[r * cols + c] =
                                    g_v.data[c] * (g.data[r * cols + c] - g_mean) / sigma;
                            }
                        }
                    }
                    grads[input_idx] = grads[input_idx].add(&grad_x);
                    grads[gamma_idx] = grads[gamma_idx].add(&g.sum_axis(0));
                    grads[beta_idx] = grads[beta_idx].add(&g.sum_axis(0));
                },
                Op::Conv2dForward {
                    input,
                    weight,
                    bias,
                    batch,
                    in_c,
                    h,
                    w,
                    out_c,
                    kernel,
                    stride,
                    pad,
                } =>
                {
                    let input_t = values[input].as_cpu().clone();
                    let weight_t = values[weight].as_cpu().clone();
                    let h_out = (h + 2 * pad - kernel) / stride + 1;
                    let w_out = (w + 2 * pad - kernel) / stride + 1;
                    let hw = h_out * w_out;
                    let n = batch * hw;

                    // Réorganise g (batch, out_c*hw) -> dout (out_c, N)
                    let mut dout = Tensor::zeros(out_c, n);
                    for bi in 0..batch
                    {
                        for oc in 0..out_c
                        {
                            let src = bi * out_c * hw + oc * hw;
                            let dst = oc * n + bi * hw;
                            for p in 0..hw
                            {
                                dout.data[dst + p] = g.data[src + p];
                            }
                        }
                    }

                    // db[oc] : somme sur (bi,oh,ow) -> bit-exact
                    if let Some(b_idx) = bias
                    {
                        let mut db = Tensor::zeros(1, out_c);
                        for oc in 0..out_c
                        {
                            let mut acc = 0.0f32;
                            for nn in 0..n
                            {
                                acc += dout.data[oc * n + nn];
                            }
                            db.data[oc] = acc;
                        }
                        grads[b_idx] = grads[b_idx].add(&db);
                    }

                    // col = im2col(input) : (in_c*k*k, N)
                    let col = crate::nn::conv_utils::im2col_raw(
                        &input_t, batch, in_c, h, w, kernel, stride, pad,
                    );

                    // dW = dout @ col^T  (matmul parallèle) -> bit-exact
                    let dw = dout.matmul(&col.transpose());
                    grads[weight] = grads[weight].add(&dw);

                    // dcol = W^T @ dout ; dx = col2im(dcol)
                    let dcol = weight_t.transpose().matmul(&dout);
                    let dx = crate::nn::conv_utils::col2im_raw(
                        &dcol, batch, in_c, h, w, kernel, stride, pad,
                    );
                    grads[input] = grads[input].add(&dx);
                },
                Op::Conv2dTransposeForward {
                    input,
                    weight,
                    bias,
                    batch,
                    in_c,
                    h: h_in,
                    w: w_in,
                    out_c,
                    kernel,
                    stride,
                    pad,
                    output_padding,
                } =>
                {
                    let input_t = &values[input].as_cpu();
                    let weight_t = &values[weight].as_cpu();
                    let h_out = (h_in - 1) * stride + kernel - 2 * pad + output_padding;
                    let w_out = (w_in - 1) * stride + kernel - 2 * pad + output_padding;

                    // dL/db
                    if let Some(b_idx) = bias
                    {
                        let mut db = Tensor::zeros(1, out_c);
                        for b_i in 0..batch
                        {
                            for co in 0..out_c
                            {
                                for oh in 0..h_out
                                {
                                    for ow in 0..w_out
                                    {
                                        let out_idx = b_i * out_c * h_out * w_out
                                            + co * h_out * w_out
                                            + oh * w_out
                                            + ow;
                                        db.data[co] += g.data[out_idx];
                                    }
                                }
                            }
                        }
                        grads[b_idx] = grads[b_idx].add(&db);
                    }

                    // dL/dX: standard conv2d on grad_out with weight W (not transposed)
                    // dX[b,ci,ih,iw] = sum_co sum_kh sum_kw dY[b,co,oh,ow] * W[ci,co,kh,kw]
                    // oh = ih*S - P + kh,  ow = iw*S - P + kw
                    let mut dx = Tensor::zeros(input_t.rows, input_t.cols);
                    for b_i in 0..batch
                    {
                        for co in 0..out_c
                        {
                            for oh in 0..h_out
                            {
                                for ow in 0..w_out
                                {
                                    let out_idx = b_i * out_c * h_out * w_out
                                        + co * h_out * w_out
                                        + oh * w_out
                                        + ow;
                                    let grad_out = g.data[out_idx];
                                    for ci in 0..in_c
                                    {
                                        for kh in 0..kernel
                                        {
                                            for kw in 0..kernel
                                            {
                                                let ih_signed =
                                                    oh as isize + pad as isize - kh as isize;
                                                let iw_signed =
                                                    ow as isize + pad as isize - kw as isize;
                                                if ih_signed >= 0
                                                    && ih_signed < h_in as isize
                                                    && iw_signed >= 0
                                                    && iw_signed < w_in as isize
                                                    && (ih_signed as usize - (oh - kh + pad))
                                                        % stride
                                                        == 0
                                                    && (iw_signed as usize - (ow - kw + pad))
                                                        % stride
                                                        == 0
                                                {
                                                    let ih = ih_signed as usize;
                                                    let iw = iw_signed as usize;
                                                    if (oh + pad as usize) >= kh
                                                        && (oh + pad as usize - kh) % stride == 0
                                                        && (ow + pad as usize - kw) % stride == 0
                                                    {
                                                        let w_idx = ci * out_c * kernel * kernel
                                                            + co * kernel * kernel
                                                            + kh * kernel
                                                            + kw;
                                                        let in_idx = b_i * in_c * h_in * w_in
                                                            + ci * h_in * w_in
                                                            + ih * w_in
                                                            + iw;
                                                        dx.data[in_idx] +=
                                                            grad_out * weight_t.data[w_idx];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    grads[input] = grads[input].add(&dx);

                    // dL/dW: im2col on input, matmul with grad_out
                    // Use ConvConfig with stride=1 for im2col on the "transposed space"
                    // Actually: dW[ci,co,kh,kw] = sum_b sum_ih sum_iw dY[b,co,oh,ow] * X[b,ci,ih,iw]
                    // where oh = ih*S - P + kh
                    let mut dw = Tensor::zeros(weight_t.rows, weight_t.cols);
                    for b_i in 0..batch
                    {
                        for ci in 0..in_c
                        {
                            for ih in 0..h_in
                            {
                                for iw in 0..w_in
                                {
                                    let in_val = input_t.data[b_i * in_c * h_in * w_in
                                        + ci * h_in * w_in
                                        + ih * w_in
                                        + iw];
                                    for co in 0..out_c
                                    {
                                        for kh in 0..kernel
                                        {
                                            for kw in 0..kernel
                                            {
                                                let oh_signed = ih as isize * stride as isize
                                                    + kh as isize
                                                    - pad as isize;
                                                let ow_signed = iw as isize * stride as isize
                                                    + kw as isize
                                                    - pad as isize;
                                                if oh_signed >= 0
                                                    && oh_signed < h_out as isize
                                                    && ow_signed >= 0
                                                    && ow_signed < w_out as isize
                                                {
                                                    let oh = oh_signed as usize;
                                                    let ow = ow_signed as usize;
                                                    let out_idx = b_i * out_c * h_out * w_out
                                                        + co * h_out * w_out
                                                        + oh * w_out
                                                        + ow;
                                                    let w_idx = ci * out_c * kernel * kernel
                                                        + co * kernel * kernel
                                                        + kh * kernel
                                                        + kw;
                                                    dw.data[w_idx] += g.data[out_idx] * in_val;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    grads[weight] = grads[weight].add(&dw);
                },
                Op::Reshape(input, old_rows, old_cols) =>
                {
                    grads[input] = grads[input].add(&g.reshape(old_rows, old_cols));
                },
                Op::FakeQuantize { input, .. } =>
                {
                    // Straight-Through Estimator (STE): pass gradients through unmodified
                    grads[input].add_assign(&g);
                },
                Op::TtContract { input_idx, core_indices, bias_idx, in_dims, out_dims, ranks, d, .. } =>
                {
                    let (x, w) = match &nodes[i].saved {
                        SavedData::TtContractState { input, weight } => (input.clone(), weight.clone()),
                        _ => unreachable!(),
                    };

                    // dL/dinput = g @ W^T: m=batch, k=out, n=in
                    let dx = &mut grads[input_idx];
                    unsafe {
                        sgemm(
                            g.rows,
                            g.cols,
                            dx.cols,
                            1.0,
                            g.data.as_ptr(),
                            g.cols as isize,
                            1,
                            w.data.as_ptr(),
                            1,
                            w.cols as isize,
                            1.0,
                            dx.data.as_mut_ptr(),
                            dx.cols as isize,
                            1,
                        );
                    }

                    // dL/dW = x^T @ g: m=in, k=batch, n=out
                    let mut dw_tensor = Tensor::zeros(x.cols, g.cols);
                    unsafe {
                        sgemm(
                            x.cols,
                            x.rows,
                            g.cols,
                            1.0,
                            x.data.as_ptr(),
                            1,
                            x.cols as isize,
                            g.data.as_ptr(),
                            g.cols as isize,
                            1,
                            0.0,
                            dw_tensor.data.as_mut_ptr(),
                            dw_tensor.cols as isize,
                            1,
                        );
                    }

                    let dd = d as usize;
                    let in_dims_slice = &in_dims[..dd];
                    let out_dims_slice = &out_dims[..dd];
                    let interleaved_tnd = interleave_weight(&dw_tensor.data, in_dims_slice, out_dims_slice);

                    let dims_2d: Vec<usize> = (0..dd).map(|i| in_dims[i] * out_dims[i]).collect();

                    for k in 0..dd {
                        let core_idx = core_indices[k];
                        let r_k = ranks[k];
                        let n_k = in_dims[k] * out_dims[k];
                        let r_next = ranks[k + 1];

                        let mut d_core_data = vec![0.0; r_k * n_k * r_next];

                        match dd {
                            2 => {
                                if k == 0 {
                                    // d_core_0 = d_interleaved @ core_1^T
                                    // m=n_0, k=n_1, n=r_1
                                    let core_1 = &values[core_indices[1]].as_cpu();
                                    let m = dims_2d[0];
                                    let kk = dims_2d[1];
                                    let nn = r_next;
                                    unsafe {
                                        sgemm(
                                            m, kk, nn,
                                            1.0,
                                            interleaved_tnd.data.as_ptr(),
                                            dims_2d[1] as isize, 1,
                                            core_1.data.as_ptr(),
                                            1, core_1.cols as isize,
                                            0.0,
                                            d_core_data.as_mut_ptr(),
                                            nn as isize, 1,
                                        );
                                    }
                                } else {
                                    // d_core_1 = core_0^T @ d_interleaved
                                    // m=r_1, k=n_0, n=n_1
                                    let core_0 = &values[core_indices[0]].as_cpu();
                                    let m = r_k;
                                    let kk = dims_2d[0];
                                    let nn = dims_2d[1];
                                    unsafe {
                                        sgemm(
                                            m, kk, nn,
                                            1.0,
                                            core_0.data.as_ptr(),
                                            1, core_0.cols as isize,
                                            interleaved_tnd.data.as_ptr(),
                                            dims_2d[1] as isize, 1,
                                            0.0,
                                            d_core_data.as_mut_ptr(),
                                            nn as isize, 1,
                                        );
                                    }
                                }
                            },
                            _ => {
                                // d>2: gradient set to zero (not yet implemented)
                            }
                        }

                        let d_core_tensor = Tensor { rows: r_k * n_k, cols: r_next, data: d_core_data };
                        grads[core_idx] = grads[core_idx].add(&d_core_tensor);
                    }

                    if let Some(b_idx) = bias_idx {
                        let mut db = vec![0.0; g.cols];
                        for j in 0..g.cols {
                            for i in 0..g.rows {
                                db[j] += g.data[i * g.cols + j];
                            }
                        }
                        let db_tensor = Tensor { rows: 1, cols: g.cols, data: db };
                        grads[b_idx] = grads[b_idx].add(&db_tensor);
                    }
                },
                Op::FlashAttention {
                    q,
                    k,
                    v,
                    mask: _mask,
                    batch,
                    n_heads,
                    seq_len,
                    d_head,
                    scale,
                    block_size,
                } =>
                {
                    // Restore saved m, l
                    let (saved_m, saved_l) = match &nodes[i].saved
                    {
                        SavedData::FlashAttentionState { m, l } => (m, l),
                        _ => panic!("FlashAttention backward: missing saved state"),
                    };

                    let q_t = &values[q].as_cpu();
                    let k_t = &values[k].as_cpu();
                    let v_t = &values[v].as_cpu();
                    let o_t = &values[i].as_cpu();
                    let dv = o_t.cols;
                    let total_heads = batch * n_heads;
                    let s_len = seq_len;

                    let mut dq = vec![0.0f32; q_t.data.len()];
                    let mut dk = vec![0.0f32; k_t.data.len()];
                    let mut dv_ = vec![0.0f32; v_t.data.len()];

                    for h in 0..total_heads
                    {
                        let q_base = h * seq_len * d_head;
                        let k_base = h * s_len * d_head;
                        let v_base = h * s_len * dv;
                        let o_base = h * seq_len * dv;
                        let m_base = h * seq_len;

                        for qi in (0..seq_len).step_by(block_size)
                        {
                            let br = (seq_len - qi).min(block_size);

                            // Replay the inner loop to recompute P_ij
                            let mut m_i = vec![-f32::INFINITY; br];
                            let mut l_i = vec![0.0f32; br];

                            for kj in (0..s_len).step_by(block_size)
                            {
                                let bc = (s_len - kj).min(block_size);

                                // S_ij = Q_i @ K_j^T
                                let mut s_ij = vec![0.0f32; br * bc];
                                for r in 0..br
                                {
                                    for c in 0..bc
                                    {
                                        let mut sum = 0.0f32;
                                        for d in 0..d_head
                                        {
                                            sum += q_t.data[q_base + (qi + r) * d_head + d]
                                                * k_t.data[k_base + (kj + c) * d_head + d];
                                        }
                                        s_ij[r * bc + c] = sum * scale;
                                    }
                                }

                                // Recompute online softmax state
                                let mut row_max = vec![-f32::INFINITY; br];
                                for r in 0..br
                                {
                                    for c in 0..bc
                                    {
                                        row_max[r] = row_max[r].max(s_ij[r * bc + c]);
                                    }
                                }

                                let mut m_new = vec![-f32::INFINITY; br];
                                for r in 0..br
                                {
                                    m_new[r] = m_i[r].max(row_max[r]);
                                }

                                let mut p_ij = vec![0.0f32; br * bc];
                                let mut row_sum_p = vec![0.0f32; br];
                                for r in 0..br
                                {
                                    for c in 0..bc
                                    {
                                        p_ij[r * bc + c] = (s_ij[r * bc + c] - m_new[r]).exp();
                                        row_sum_p[r] += p_ij[r * bc + c];
                                    }
                                }

                                let mut rescale = vec![0.0f32; br];
                                for r in 0..br
                                {
                                    rescale[r] = (m_i[r] - m_new[r]).exp();
                                }

                                // Rescale l_i
                                for r in 0..br
                                {
                                    l_i[r] = rescale[r] * l_i[r] + row_sum_p[r];
                                }

                                for r in 0..br
                                {
                                    m_i[r] = m_new[r];
                                }
                            }

                            // Now compute gradients for this Q-block
                            for kj in (0..s_len).step_by(block_size)
                            {
                                let bc = (s_len - kj).min(block_size);

                                // Recompute S_ij
                                let mut s_ij = vec![0.0f32; br * bc];
                                for r in 0..br
                                {
                                    for c in 0..bc
                                    {
                                        let mut sum = 0.0f32;
                                        for d in 0..d_head
                                        {
                                            sum += q_t.data[q_base + (qi + r) * d_head + d]
                                                * k_t.data[k_base + (kj + c) * d_head + d];
                                        }
                                        s_ij[r * bc + c] = sum * scale;
                                    }
                                }

                                // Recompute row_max
                                let mut row_max = vec![-f32::INFINITY; br];
                                for r in 0..br
                                {
                                    for c in 0..bc
                                    {
                                        row_max[r] = row_max[r].max(s_ij[r * bc + c]);
                                    }
                                }
                                let mut m_new = vec![-f32::INFINITY; br];
                                for r in 0..br
                                {
                                    m_new[r] = m_i[r].max(row_max[r]);
                                }

                                // P_ij
                                let mut p_ij_unscaled = vec![0.0f32; br * bc];
                                for r in 0..br
                                {
                                    for c in 0..bc
                                    {
                                        p_ij_unscaled[r * bc + c] =
                                            (s_ij[r * bc + c] - m_new[r]).exp();
                                    }
                                }

                                // Final normalized P_ij
                                let mut p_ij = vec![0.0f32; br * bc];
                                for r in 0..br
                                {
                                    let row_idx = m_base + qi + r;
                                    let m_final = saved_m.data[row_idx];
                                    let l_final = saved_l.data[row_idx];

                                    // P_ij_normalized = P_ij_unscaled * exp(m_new - m_final) / l_final
                                    let factor = (m_new[r] - m_final).exp() / l_final;

                                    for c in 0..bc
                                    {
                                        p_ij[r * bc + c] = p_ij_unscaled[r * bc + c] * factor;
                                    }
                                }

                                // dP = dO @ V_j^T: upstream gradient times V
                                let mut dp = vec![0.0f32; br * bc];
                                for r in 0..br
                                {
                                    for d in 0..dv
                                    {
                                        let go = g.data[o_base + (qi + r) * dv + d];
                                        if go == 0.0
                                        {
                                            continue;
                                        }
                                        for c in 0..bc
                                        {
                                            dp[r * bc + c] +=
                                                go * v_t.data[v_base + (kj + c) * dv + d];
                                        }
                                    }
                                }

                                // Softmax gradient correction:
                                // dL/dS = P .* (dP - sum_c(P .* dP))
                                let mut ds = vec![0.0f32; br * bc];
                                for r in 0..br
                                {
                                    let mut sum_p_dp = 0.0f32;
                                    for c in 0..bc
                                    {
                                        sum_p_dp += p_ij[r * bc + c] * dp[r * bc + c];
                                    }
                                    for c in 0..bc
                                    {
                                        ds[r * bc + c] =
                                            p_ij[r * bc + c] * (dp[r * bc + c] - sum_p_dp);
                                    }
                                }

                                // dQ contribution
                                for r in 0..br
                                {
                                    for d in 0..d_head
                                    {
                                        let mut sum = 0.0f32;
                                        for c in 0..bc
                                        {
                                            sum += ds[r * bc + c]
                                                * k_t.data[k_base + (kj + c) * d_head + d];
                                        }
                                        dq[q_base + (qi + r) * d_head + d] += sum * scale;
                                    }
                                }

                                // dK contribution
                                for c in 0..bc
                                {
                                    for d in 0..d_head
                                    {
                                        let mut sum = 0.0f32;
                                        for r in 0..br
                                        {
                                            sum += ds[r * bc + c]
                                                * q_t.data[q_base + (qi + r) * d_head + d];
                                        }
                                        dk[k_base + (kj + c) * d_head + d] += sum * scale;
                                    }
                                }

                                // dV contribution
                                for c in 0..bc
                                {
                                    for d in 0..dv
                                    {
                                        let mut sum = 0.0f32;
                                        for r in 0..br
                                        {
                                            sum += p_ij[r * bc + c]
                                                * g.data[o_base + (qi + r) * dv + d];
                                        }
                                        dv_[v_base + (kj + c) * dv + d] += sum;
                                    }
                                }
                            }
                        }
                    }

                    let dq_t = Tensor::from_vec(dq, q_t.rows, q_t.cols);
                    let dk_t = Tensor::from_vec(dk, k_t.rows, k_t.cols);
                    let dv_t = Tensor::from_vec(dv_, v_t.rows, v_t.cols);

                    grads[q] = grads[q].add(&dq_t);
                    grads[k] = grads[k].add(&dk_t);
                    grads[v] = grads[v].add(&dv_t);
                },
            }
        }
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================== //
//  Var                                                               //
// ================================================================== //

#[derive(Debug, Clone, Copy)]
pub struct Var<'t> {
    pub tape: &'t Tape,
    pub idx: usize,
}

impl<'t> Var<'t> {
    pub fn new(tape: &'t Tape, idx: usize) -> Self {
        Self { tape, idx }
    }
    #[inline]
    pub fn idx(&self) -> usize {
        self.idx
    }
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.tape.values.borrow()[self.idx].shape()
    }
    #[inline]
    pub fn tape(&self) -> &'t Tape {
        self.tape
    }

    pub fn backward(self) {
        self.tape.backward(self.idx);
    }

    pub fn detach(self) -> Var<'t> {
        let val = self.tape.value(self.idx);
        self.tape.input(val)
    }

    pub fn try_add(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        crate::error::check_shape("add", a.shape(), b.shape())?;
        let out = a.add(&b);
        let new_idx = self.tape.push_with_saved(
            Op::Add(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Var<'t>) -> Var<'t> {
        self.try_add(other).unwrap()
    }

    pub fn fake_quantize_ste(self, scale: f32, zero_point: i32) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let mut out_data = vec![0.0f32; a.data.len()];
        for (i, &x) in a.data.iter().enumerate()
        {
            let q = (x / scale).round() + zero_point as f32;
            let q_clamped = q.clamp(-128.0, 127.0);
            out_data[i] = (q_clamped - zero_point as f32) * scale;
        }
        let out = Tensor::from_vec(out_data, a.rows, a.cols);
        let new_idx = self.tape.push_with_saved(
            Op::FakeQuantize {
                input: self.idx,
                scale,
                zero_point,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Var<'t>) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        assert_eq!(a.shape(), b.shape(), "sub: shape mismatch");
        let out = a.sub(&b);
        let new_idx = self.tape.push_with_saved(
            Op::Sub(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Var<'t>) -> Var<'t> {
        self.hadamard(other)
    }

    pub fn try_sub(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        crate::error::check_shape("sub", a.shape(), b.shape())?;
        let out = a.sub(&b);
        let new_idx = self.tape.push_with_saved(
            Op::Sub(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }

    pub fn try_div(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        crate::error::check_shape("div", a.shape(), b.shape())?;
        let out = a.div(&b);
        let new_idx = self.tape.push_with_saved(
            Op::Div(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }

    pub fn try_matmul(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        crate::error::check_inner_dim("matmul", a.cols, b.rows)?;
        let out = a.matmul(&b);
        let new_idx = self.tape.push_with_saved(
            Op::MatMul(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }

    /// MatMul GPU-acceléré.
    pub fn try_matmul_gpu(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        crate::error::check_inner_dim("matmul_gpu", a.cols, b.rows)?;
        let out = a.matmul(&b);
        let new_idx = self.tape.push_with_saved(
            Op::MatMulGpu(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }

    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Var<'t>) -> Var<'t> {
        self.try_div(other).unwrap()
    }

    pub fn matmul(self, other: Var<'t>) -> Var<'t> {
        self.try_matmul(other).unwrap()
    }

    pub fn matmul_gpu(self, other: Var<'t>) -> Var<'t> {
        self.try_matmul_gpu(other).unwrap()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.neg();
        let new_idx =
            self.tape
                .push_with_saved(Op::Neg(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn relu(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let mut out = a.clone();
        for x in &mut out.data
        {
            *x = x.max(0.0);
        }
        let mut mask = Tensor::zeros(a.rows, a.cols);
        for (m, val) in mask.data.iter_mut().zip(&a.data)
        {
            *m = if *val > 0.0 { 1.0 } else { 0.0 };
        }
        let new_idx = self.tape.push_with_saved(
            Op::ReLU(self.idx),
            DeviceTensor::cpu(out),
            SavedData::Mask(mask),
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn sigmoid(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.sigmoid();
        let new_idx = self.tape.push_with_saved(
            Op::Sigmoid(self.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn tanh(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.tanh();
        let new_idx =
            self.tape
                .push_with_saved(Op::Tanh(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn sin(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.sin();
        let new_idx =
            self.tape
                .push_with_saved(Op::Sin(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn cos(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.cos();
        let new_idx =
            self.tape
                .push_with_saved(Op::Cos(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn tan(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.tan();
        let new_idx =
            self.tape
                .push_with_saved(Op::Tan(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn sinh(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.sinh();
        let new_idx =
            self.tape
                .push_with_saved(Op::Sinh(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn cosh(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.cosh();
        let new_idx =
            self.tape
                .push_with_saved(Op::Cosh(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn log10(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.log10();
        let new_idx =
            self.tape
                .push_with_saved(Op::Log10(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn asin(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.asin();
        let new_idx =
            self.tape
                .push_with_saved(Op::Asin(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn acos(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.acos();
        let new_idx =
            self.tape
                .push_with_saved(Op::Acos(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn atan(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.atan();
        let new_idx =
            self.tape
                .push_with_saved(Op::Atan(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn atan2(self, x: Var<'t>) -> Var<'t> {
        let y_val = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let x_val = self.tape.values.borrow()[x.idx].as_cpu().clone();
        let out = y_val.atan2(&x_val);
        let new_idx = self.tape.push_with_saved(
            Op::Atan2(self.idx, x.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn exp(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.exp();
        let new_idx =
            self.tape
                .push_with_saved(Op::Exp(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn log(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.log();
        let new_idx =
            self.tape
                .push_with_saved(Op::Log(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn sqrt(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.sqrt();
        let new_idx =
            self.tape
                .push_with_saved(Op::Sqrt(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn reciprocal(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.reciprocal();
        let new_idx = self.tape.push_with_saved(
            Op::Reciprocal(self.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn pow(self, exp: f32) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.pow(exp);
        let new_idx = self.tape.push_with_saved(
            Op::Pow {
                base: self.idx,
                exp,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn scale(self, s: f32) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.scale(s);
        let new_idx = self.tape.push_with_saved(
            Op::Scale {
                input: self.idx,
                scalar: s,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn sum(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let mut out = Tensor::zeros(1, 1);
        out.data[0] = a.sum();
        let new_idx =
            self.tape
                .push_with_saved(Op::Sum(self.idx), DeviceTensor::cpu(out), SavedData::None);
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn sum_axis(self, axis: u8) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.sum_axis(axis);
        let new_idx = self.tape.push_with_saved(
            Op::SumAxis(self.idx, axis),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    /// Broadcaste cette Var vers une nouvelle shape (rows, cols).
    /// Le backward propage la somme selon les axes élargis.
    pub fn broadcast(self, rows: usize, cols: usize) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.broadcast_to(rows, cols);
        let new_idx = self.tape.push_with_saved(
            Op::Broadcast {
                input: self.idx,
                rows,
                cols,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn mean_axis(self, axis: u8) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.mean_axis(axis);
        let new_idx = self.tape.push_with_saved(
            Op::MeanAxis(self.idx, axis),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn var_axis(self, axis: u8) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.var_axis(axis);
        let new_idx = self.tape.push_with_saved(
            Op::VarAxis(self.idx, axis),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn max_axis(self, axis: u8) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.max_axis(axis);
        let new_idx = self.tape.push_with_saved(
            Op::MaxAxis(self.idx, axis),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn try_softmax(self, axis: u8) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        if axis > 1
        {
            return Err(crate::error::SciRustError::InvalidConfig(format!(
                "softmax: axis {axis} out of range [0, 1]"
            )));
        }
        let out = a.softmax(axis);
        let new_idx = self.tape.push_with_saved(
            Op::Softmax {
                input: self.idx,
                axis,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn softmax(self, axis: u8) -> Var<'t> {
        self.try_softmax(axis).unwrap()
    }

    pub fn try_log_softmax(self, axis: u8) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        if axis > 1
        {
            return Err(crate::error::SciRustError::InvalidConfig(format!(
                "log_softmax: axis {axis} out of range [0, 1]"
            )));
        }
        let sm = a.softmax(axis);
        let out = sm.log();
        let new_idx = self.tape.push_with_saved(
            Op::LogSoftmax {
                input: self.idx,
                axis,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn log_softmax(self, axis: u8) -> Var<'t> {
        self.try_log_softmax(axis).unwrap()
    }

    pub fn transpose(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let out = a.transpose();
        let new_idx = self.tape.push_with_saved(
            Op::Transpose2d(self.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn transpose_2d(self) -> Var<'t> {
        self.transpose()
    }

    pub fn reshape(self, shape: &[usize]) -> Var<'t> {
        assert_eq!(shape.len(), 2, "reshape: shape must have 2 elements");
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let old_shape = a.shape();
        let out = a.reshape(shape[0], shape[1]);
        let new_idx = self.tape.push_with_saved(
            Op::Reshape(self.idx, old_shape.0, old_shape.1),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn try_add_broadcast(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        if b.rows != 1 && b.cols != a.cols
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "add_broadcast",
                expected: (1, a.cols),
                got: (b.rows, b.cols),
            });
        }
        let out = a.add(&b.broadcast_to(a.rows, a.cols));
        let new_idx = self.tape.push_with_saved(
            Op::AddBroadcast(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn add_broadcast(self, other: Var<'t>) -> Var<'t> {
        self.try_add_broadcast(other).unwrap()
    }
    pub fn add_bias(self, bias: Var<'t>) -> Var<'t> {
        self.add_broadcast(bias)
    }
    pub fn try_add_bias(self, bias: Var<'t>) -> crate::error::Result<Var<'t>> {
        self.try_add_broadcast(bias)
    }

    pub fn sub_broadcast(self, other: Var<'t>) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        let out = a.sub(&b.broadcast_to(a.rows, a.cols));
        let new_idx = self.tape.push_with_saved(
            Op::SubBroadcast(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn try_mul_broadcast(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        if b.rows != 1 && b.cols != a.cols
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "mul_broadcast",
                expected: (1, a.cols),
                got: (b.rows, b.cols),
            });
        }
        let out = a.hadamard(&b.broadcast_to(a.rows, a.cols));
        let new_idx = self.tape.push_with_saved(
            Op::MulBroadcast(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn mul_broadcast(self, other: Var<'t>) -> Var<'t> {
        self.try_mul_broadcast(other).unwrap()
    }

    pub fn div_broadcast(self, other: Var<'t>) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let b = self.tape.values.borrow()[other.idx].as_cpu().clone();
        let out = a.div(&b.broadcast_to(a.rows, a.cols));
        let new_idx = self.tape.push_with_saved(
            Op::DivBroadcast(self.idx, other.idx),
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn try_hadamard(self, other: Var<'t>) -> crate::error::Result<Var<'t>> {
        self.try_mul_broadcast(other)
    }
    pub fn hadamard(self, other: Var<'t>) -> Var<'t> {
        self.try_hadamard(other).unwrap()
    }

    pub fn try_slice_rows(self, start: usize, len: usize) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        if start + len > a.rows
        {
            return Err(crate::error::SciRustError::InvalidConfig(format!(
                "slice_rows: start {start} + len {len} > rows {}",
                a.rows
            )));
        }
        let mut out = Tensor::zeros(len, a.cols);
        for r in 0..len
        {
            for c in 0..a.cols
            {
                out.data[r * a.cols + c] = a.data[(start + r) * a.cols + c];
            }
        }
        let new_idx = self.tape.push_with_saved(
            Op::Slice {
                input_idx: self.idx,
                start,
                len,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn slice_rows(self, start: usize, len: usize) -> Var<'t> {
        self.try_slice_rows(start, len).unwrap()
    }

    pub fn try_slice_cols(self, start: usize, len: usize) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        if start + len > a.cols
        {
            return Err(crate::error::SciRustError::InvalidConfig(format!(
                "slice_cols: start {start} + len {len} > cols {}",
                a.cols
            )));
        }
        let mut out = Tensor::zeros(a.rows, len);
        for r in 0..a.rows
        {
            for c in 0..len
            {
                out.data[r * len + c] = a.data[r * a.cols + (start + c)];
            }
        }
        let new_idx = self.tape.push_with_saved(
            Op::SliceCols {
                input_idx: self.idx,
                start,
                len,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn slice_cols(self, start: usize, len: usize) -> Var<'t> {
        self.try_slice_cols(start, len).unwrap()
    }

    pub fn try_embedding(self, indices: Vec<u32>) -> crate::error::Result<Var<'t>> {
        let table = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let vocab = table.rows;
        let d = table.cols;
        let n = indices.len();
        for &idx_u in &indices
        {
            let i_u = idx_u as usize;
            if i_u >= vocab
            {
                return Err(crate::error::SciRustError::InvalidConfig(format!(
                    "embedding: index {i_u} >= vocab {vocab}"
                )));
            }
        }
        let mut out = Tensor::zeros(n, d);
        for (i, &idx_u) in indices.iter().enumerate()
        {
            let i_u = idx_u as usize;
            for j in 0..d
            {
                out.data[i * d + j] = table.data[i_u * d + j];
            }
        }
        let new_idx = self.tape.push_with_saved(
            Op::Embedding {
                table_idx: self.idx,
                n_tokens: n,
            },
            DeviceTensor::cpu(out),
            SavedData::Indices(indices),
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn embedding(self, indices: Vec<u32>) -> Var<'t> {
        self.try_embedding(indices).unwrap()
    }

    pub fn try_linear(self, w: Var<'t>, b: Option<Var<'t>>) -> crate::error::Result<Var<'t>> {
        let mut out = self.try_matmul(w)?;
        if let Some(bias) = b
        {
            out = out.try_add_broadcast(bias)?;
        }
        Ok(out)
    }
    pub fn linear(self, w: Var<'t>, b: Option<Var<'t>>) -> Var<'t> {
        self.try_linear(w, b).unwrap()
    }

    pub fn try_tt_contract(
        self,
        cores: Vec<Var<'t>>,
        bias: Option<Var<'t>>,
        in_dims: Vec<usize>,
        out_dims: Vec<usize>,
        ranks: Vec<usize>,
    ) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();

        let d = in_dims.len();
        assert!(d <= 8, "TT d must be <= 8");
        let mut core_indices_arr = [0usize; 8];
        let mut in_dims_arr = [0usize; 8];
        let mut out_dims_arr = [0usize; 8];
        let mut ranks_arr = [0usize; 9];
        for (k, idx) in cores.iter().map(|c| c.idx).enumerate() {
            core_indices_arr[k] = idx;
        }
        for (k, &d_k) in in_dims.iter().enumerate() {
            in_dims_arr[k] = d_k;
        }
        for (k, &d_k) in out_dims.iter().enumerate() {
            out_dims_arr[k] = d_k;
        }
        for (k, &r_k) in ranks.iter().enumerate() {
            ranks_arr[k] = r_k;
        }

        let core_tnd: Vec<TensorND> = cores.iter().enumerate().map(|(k, c)| {
            let cv = self.tape.values.borrow()[c.idx].as_cpu().clone();
            let r_k = ranks_arr[k];
            let n_k = in_dims_arr[k] * out_dims_arr[k];
            let r_next = ranks_arr[k + 1];
            assert_eq!(cv.data.len(), r_k * n_k * r_next, "core {k} data len mismatch");
            TensorND::new(cv.data, vec![r_k, n_k, r_next])
        }).collect();

        let mode_dims: Vec<usize> = (0..d)
            .map(|k| in_dims[k] * out_dims[k]).collect();
        let tt = TTCores { cores: core_tnd, ranks, mode_dims };

        let w_data = reconstruct_matrix(&tt, &in_dims, &out_dims);
        let in_features: usize = in_dims.iter().product();
        let out_features: usize = out_dims.iter().product();
        let w_tensor = Tensor { rows: in_features, cols: out_features, data: w_data };

        let mut out_data = vec![0.0; a.rows * out_features];
        unsafe {
            sgemm(
                a.rows, a.cols, out_features,
                1.0,
                a.data.as_ptr(), a.cols as isize, 1,
                w_tensor.data.as_ptr(), w_tensor.cols as isize, 1,
                0.0,
                out_data.as_mut_ptr(), out_features as isize, 1,
            );
        }
        let mut out_tensor = Tensor { rows: a.rows, cols: out_features, data: out_data };

        let bias_idx = bias.as_ref().map(|b| b.idx);
        if let Some(ref b) = bias {
            let bv = self.tape.values.borrow()[b.idx].as_cpu().clone();
            for j in 0..out_features {
                for i in 0..out_tensor.rows {
                    out_tensor.data[i * out_features + j] += bv.data[j % bv.cols];
                }
            }
        }

        let saved = SavedData::TtContractState {
            input: a,
            weight: w_tensor,
        };

        let new_idx = self.tape.push_with_saved(
            Op::TtContract {
                input_idx: self.idx,
                core_indices: core_indices_arr,
                num_cores: d,
                bias_idx,
                in_dims: in_dims_arr,
                out_dims: out_dims_arr,
                ranks: ranks_arr,
                d,
            },
            DeviceTensor::cpu(out_tensor),
            saved,
        );

        Ok(Var { tape: self.tape, idx: new_idx })
    }
    pub fn tt_contract(
        self,
        cores: Vec<Var<'t>>,
        bias: Option<Var<'t>>,
        in_dims: Vec<usize>,
        out_dims: Vec<usize>,
        ranks: Vec<usize>,
    ) -> Var<'t> {
        self.try_tt_contract(cores, bias, in_dims, out_dims, ranks).unwrap()
    }

    pub fn causal_mask(self, seq_len: usize) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let mut out = a.clone();
        for (r, row) in out.data.chunks_exact_mut(a.cols).enumerate()
        {
            let row_in_seq = r % seq_len;
            for (c, val) in row.iter_mut().enumerate()
            {
                let col_in_seq = c % seq_len;
                if col_in_seq > row_in_seq
                {
                    *val = -1e9;
                }
            }
        }
        let new_idx = self.tape.push_with_saved(
            Op::CausalMask {
                input_idx: self.idx,
                seq_len,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn dropout(self, p: f32) -> Var<'t> {
        if p == 0.0
        {
            return self;
        }
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let scale = 1.0 / (1.0 - p);
        let mut mask_data = vec![0.0f32; a.rows * a.cols];
        let mut rng = PcgEngine::new(42);
        for item in mask_data.iter_mut()
        {
            *item = if rng.float() < p { 0.0 } else { scale };
        }
        let mask_t = Tensor::from_vec(mask_data, a.rows, a.cols);
        let mask_v = self.tape.input(mask_t);
        let out = a.hadamard(&self.tape.values.borrow()[mask_v.idx].as_cpu().clone());
        let new_idx = self.tape.push_with_saved(
            Op::Dropout {
                input_idx: self.idx,
                mask_idx: mask_v.idx,
                p,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    pub fn try_layer_norm(
        self,
        gamma: Var<'t>,
        beta: Var<'t>,
        eps: f32,
    ) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let (rows, cols) = a.shape();
        let gv = self.tape.values.borrow()[gamma.idx].as_cpu().clone();
        let bv = self.tape.values.borrow()[beta.idx].as_cpu().clone();
        if gv.shape() != (1, cols)
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "layer_norm",
                expected: (1, cols),
                got: gv.shape(),
            });
        }
        if bv.shape() != (1, cols)
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "layer_norm",
                expected: (1, cols),
                got: bv.shape(),
            });
        }
        let mut out = Tensor::zeros(rows, cols);
        let mut normed = Tensor::zeros(rows, cols);
        for r in 0..rows
        {
            let mut mean = 0.0f32;
            for c in 0..cols
            {
                mean += a.data[r * cols + c];
            }
            mean /= cols as f32;
            let mut var = 0.0f32;
            for c in 0..cols
            {
                let d = a.data[r * cols + c] - mean;
                var += d * d;
            }
            var /= cols as f32;
            let std = (var + eps).sqrt();
            for c in 0..cols
            {
                let norm_val = (a.data[r * cols + c] - mean) / std;
                out.data[r * cols + c] = norm_val * gv.data[c] + bv.data[c];
                normed.data[r * cols + c] = norm_val;
            }
        }
        let new_idx = self.tape.push_with_saved(
            Op::LayerNorm {
                input_idx: self.idx,
                gamma_idx: gamma.idx,
                beta_idx: beta.idx,
                eps,
            },
            DeviceTensor::cpu(out),
            SavedData::LayerNormNormed(normed),
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }
    pub fn layer_norm(self, gamma: Var<'t>, beta: Var<'t>, eps: f32) -> Var<'t> {
        self.try_layer_norm(gamma, beta, eps).unwrap()
    }

    pub fn max_pool2d(self, c: usize, h: usize, w: usize, kernel: usize, stride: usize) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let h_out = (h - kernel) / stride + 1;
        let w_out = (w - kernel) / stride + 1;
        let out_rows = a.rows;
        let out_cols = c * h_out * w_out;
        let mut out = Tensor::zeros(out_rows, out_cols);
        for b in 0..a.rows
        {
            for ch in 0..c
            {
                for oh in 0..h_out
                {
                    for ow in 0..w_out
                    {
                        let mut m = -f32::INFINITY;
                        for kh in 0..kernel
                        {
                            for kw in 0..kernel
                            {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                let idx = b * c * h * w + ch * h * w + ih * w + iw;
                                m = m.max(a.data[idx]);
                            }
                        }
                        let out_idx = b * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        out.data[out_idx] = m;
                    }
                }
            }
        }
        let new_idx = self.tape.push_with_saved(
            Op::MaxPool2d {
                input_idx: self.idx,
                c,
                h,
                w,
                kernel,
                stride,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Var {
            tape: self.tape,
            idx: new_idx,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_conv2d_forward(
        self,
        weight: Var<'t>,
        bias: Option<Var<'t>>,
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
    ) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let wv = self.tape.values.borrow()[weight.idx].as_cpu().clone();
        let expected_input_cols = in_c * h * w;
        if a.cols != expected_input_cols || a.rows != batch
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "conv2d_forward",
                expected: (batch, expected_input_cols),
                got: a.shape(),
            });
        }
        let expected_w_rows = out_c;
        let expected_w_cols = in_c * kernel * kernel;
        if wv.shape() != (expected_w_rows, expected_w_cols)
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "conv2d_forward",
                expected: (expected_w_rows, expected_w_cols),
                got: wv.shape(),
            });
        }
        let h_out = (h + 2 * pad - kernel) / stride + 1;
        let w_out = (w + 2 * pad - kernel) / stride + 1;
        let hw = h_out * w_out;

        let col = im2col_raw(&a, batch, in_c, h, w, kernel, stride, pad);

        let mut out_2d = wv.matmul(&col);

        if let Some(b_v) = bias
        {
            let bv = self.tape.values.borrow()[b_v.idx].as_cpu().clone();
            for oc in 0..out_c
            {
                let b_val = bv.data[oc];
                for i in 0..(batch * hw)
                {
                    out_2d.data[oc * (batch * hw) + i] += b_val;
                }
            }
        }

        let mut out = Tensor::zeros(batch, out_c * hw);
        for bi in 0..batch
        {
            for oc in 0..out_c
            {
                let src_off = oc * (batch * hw) + bi * hw;
                let dst_off = bi * (out_c * hw) + oc * hw;
                out.data[dst_off..dst_off + hw]
                    .copy_from_slice(&out_2d.data[src_off..src_off + hw]);
            }
        }

        let b_idx = bias.map(|v| v.idx);
        let new_idx = self.tape.push_with_saved(
            Op::Conv2dForward {
                input: self.idx,
                weight: weight.idx,
                bias: b_idx,
                batch,
                in_c,
                h,
                w,
                out_c,
                kernel,
                stride,
                pad,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_forward(
        self,
        weight: Var<'t>,
        bias: Option<Var<'t>>,
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
    ) -> Var<'t> {
        self.try_conv2d_forward(weight, bias, batch, in_c, h, w, out_c, kernel, stride, pad)
            .unwrap()
    }
    pub fn try_conv2d_transpose_forward(
        self,
        weight: Var<'t>,
        bias: Option<Var<'t>>,
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        output_padding: usize,
    ) -> crate::error::Result<Var<'t>> {
        let a = self.tape.values.borrow()[self.idx].as_cpu().clone();
        let wv = self.tape.values.borrow()[weight.idx].as_cpu().clone();
        let expected_input_cols = in_c * h * w;
        if a.cols != expected_input_cols || a.rows != batch
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "conv2d_transpose_forward",
                expected: (batch, expected_input_cols),
                got: a.shape(),
            });
        }
        let expected_w_rows = in_c;
        let expected_w_cols = out_c * kernel * kernel;
        if wv.shape() != (expected_w_rows, expected_w_cols)
        {
            return Err(crate::error::SciRustError::ShapeMismatch {
                op: "conv2d_transpose_forward",
                expected: (expected_w_rows, expected_w_cols),
                got: wv.shape(),
            });
        }
        let h_out = (h - 1) * stride + kernel - 2 * pad + output_padding;
        let w_out = (w - 1) * stride + kernel - 2 * pad + output_padding;
        let out_rows = batch;
        let out_cols = out_c * h_out * w_out;
        let mut out = Tensor::zeros(out_rows, out_cols);
        for b_i in 0..batch
        {
            for co in 0..out_c
            {
                for ci in 0..in_c
                {
                    for kh in 0..kernel
                    {
                        for kw in 0..kernel
                        {
                            for ih in 0..h
                            {
                                for iw in 0..w
                                {
                                    let oh = (ih * stride) as isize + kh as isize - pad as isize;
                                    let ow = (iw * stride) as isize + kw as isize - pad as isize;
                                    if oh >= 0
                                        && ow >= 0
                                        && (oh as usize) < h_out
                                        && (ow as usize) < w_out
                                    {
                                        let oh = oh as usize;
                                        let ow = ow as usize;
                                        let w_idx = ci * out_c * kernel * kernel
                                            + co * kernel * kernel
                                            + kh * kernel
                                            + kw;
                                        let in_idx =
                                            b_i * in_c * h * w + ci * h * w + ih * w + iw;
                                        let out_idx = b_i * out_c * h_out * w_out
                                            + co * h_out * w_out
                                            + oh * w_out
                                            + ow;
                                        out.data[out_idx] += a.data[in_idx] * wv.data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(ref b_v) = bias
        {
            let b_data = self.tape.values.borrow()[b_v.idx].as_cpu().clone();
            for b_i in 0..batch
            {
                for co in 0..out_c
                {
                    let b_val = b_data.data[co];
                    for oh in 0..h_out
                    {
                        for ow in 0..w_out
                        {
                            let out_idx = b_i * out_c * h_out * w_out + co * h_out * w_out
                                + oh * w_out
                                + ow;
                            out.data[out_idx] += b_val;
                        }
                    }
                }
            }
        }
        let b_idx = bias.map(|v| v.idx);
        let new_idx = self.tape.push_with_saved(
            Op::Conv2dTransposeForward {
                input: self.idx,
                weight: weight.idx,
                bias: b_idx,
                batch,
                in_c,
                h,
                w,
                out_c,
                kernel,
                stride,
                pad,
                output_padding,
            },
            DeviceTensor::cpu(out),
            SavedData::None,
        );
        Ok(Var {
            tape: self.tape,
            idx: new_idx,
        })
    }

    pub fn conv2d_transpose_forward(
        self,
        weight: Var<'t>,
        bias: Option<Var<'t>>,
        batch: usize,
        in_c: usize,
        h: usize,
        w: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        output_padding: usize,
    ) -> Var<'t> {
        self.try_conv2d_transpose_forward(
            weight, bias, batch, in_c, h, w, out_c, kernel, stride, pad, output_padding,
        )
        .unwrap()
    }
}

// ================================================================== //
//  concat_rows                                                       //
// ================================================================== //

pub fn concat_rows<'t>(tape: &'t Tape, rows: &[Var<'t>]) -> Var<'t> {
    if rows.is_empty()
    {
        panic!("concat_rows: empty slice");
    }
    // Recursive concat for N > 3 by grouping in chunks of 3
    if rows.len() > 3
    {
        let mut chunks: Vec<Var<'t>> = Vec::new();
        for chunk in rows.chunks(3)
        {
            chunks.push(concat_rows(tape, chunk));
        }
        return concat_rows(tape, &chunks);
    }
    let cols = rows[0].tape.values.borrow()[rows[0].idx].shape().1;
    let mut indices = [0usize; 3];
    let mut counts = [0usize; 3];
    for (i, r) in rows.iter().enumerate().take(3)
    {
        indices[i] = r.idx;
        counts[i] = r.tape.values.borrow()[r.idx].shape().0;
    }
    let total_rows: usize = counts.iter().sum();
    let mut out = Tensor::zeros(total_rows, cols);
    let mut off = 0;
    for (_i, r) in rows.iter().enumerate().take(3)
    {
        let a = r.tape.values.borrow()[r.idx].as_cpu().clone();
        let (n, _) = a.shape();
        for rr in 0..n
        {
            for c in 0..cols
            {
                out.data[(off + rr) * cols + c] = a.data[rr * cols + c];
            }
        }
        off += n;
    }
    let new_idx = tape.push_with_saved(
        Op::Concat {
            input_indices: indices,
            row_counts: counts,
        },
        DeviceTensor::cpu(out),
        SavedData::None,
    );
    Var { tape, idx: new_idx }
}

// ================================================================== //
//  Tests                                                             //
// ================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_in_place_ops() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Tensor::from_vec(vec![0.5, 1.5, 2.5, 3.5], 2, 2);

        a.add_assign(&b);
        assert_eq!(a.data, vec![1.5, 3.5, 5.5, 7.5]);

        a.sub_assign(&b);
        assert_eq!(a.data, vec![1.0, 2.0, 3.0, 4.0]);

        a.hadamard_assign(&b);
        assert_eq!(a.data, vec![0.5, 3.0, 7.5, 14.0]);
    }

    #[test]
    fn test_exp_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.0, 1.0, 2.0], 1, 3));
        let x_idx = x.idx();
        let y = x.exp();
        let loss = y.sum();
        loss.backward();
        let grad = tape.grad(x_idx);
        assert!((grad.data[0] - 1.0).abs() < 1e-5);
        assert!((grad.data[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let x_idx = x.idx();
        let y = x.sigmoid();
        let loss = y.sum();
        loss.backward();
        let grad = tape.grad(x_idx);
        assert!((grad.data[0] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let x_idx = x.idx();
        let y = x.softmax(1);
        let loss = y.sum();
        loss.backward();
        let grad = tape.grad(x_idx);
        let sum_grad: f32 = grad.data.iter().sum();
        assert!(sum_grad.abs() < 1e-5);
    }

    #[test]
    fn test_matmul_gradient() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let b = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2));
        let a_idx = a.idx();
        let y = a.matmul(b);
        let loss = y.sum();
        loss.backward();
        let grad_a = tape.grad(a_idx);
        assert_eq!(grad_a.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sub_gradient() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![3.0, 4.0], 1, 2));
        let b = tape.input(Tensor::from_vec(vec![1.0, 1.0], 1, 2));
        let a_idx = a.idx();
        let b_idx = b.idx();
        let y = a.sub(b);
        let loss = y.sum();
        loss.backward();
        assert_eq!(tape.grad(a_idx).data, vec![1.0, 1.0]);
        assert_eq!(tape.grad(b_idx).data, vec![-1.0, -1.0]);
    }

    #[test]
    fn test_div_gradient() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![4.0, 6.0], 1, 2));
        let b = tape.input(Tensor::from_vec(vec![2.0, 3.0], 1, 2));
        let a_idx = a.idx();
        let b_idx = b.idx();
        let y = a.div(b);
        let loss = y.sum();
        loss.backward();
        let ga = tape.grad(a_idx);
        let gb = tape.grad(b_idx);
        assert!((ga.data[0] - 0.5).abs() < 1e-5);
        assert!((ga.data[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((gb.data[0] - (-4.0 / 4.0)).abs() < 1e-5);
        assert!((gb.data[1] - (-6.0 / 9.0)).abs() < 1e-5);
    }

    #[test]
    fn test_relu_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], 2, 2));
        let x_idx = x.idx();
        let y = x.relu();
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert_eq!(g.data, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_tanh_at_zero() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let y = x.tanh();
        let y_idx = y.idx();
        let val = tape.value(y_idx).data[0];
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_at_zero() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let y = x.sigmoid();
        let y_idx = y.idx();
        let val = tape.value(y_idx).data[0];
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_exp_log_composition() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let x_idx = x.idx();
        let y = x.exp().log();
        let y_idx = y.idx();
        let val = tape.value(y_idx);
        let x_val = tape.value(x_idx);
        for i in 0..3
        {
            assert!((val.data[i] - x_val.data[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_jacobian_matches_formula_1d() {
        // Formule analytique : ∂s_i/∂x_j = s_i · (δ_ij - s_j)
        // On vérifie que l'autograd produit exactement ces valeurs.
        let logits = vec![1.0f32, 2.0, 3.0];
        let n = logits.len();

        // softmax analytique
        let exp: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
        let z: f32 = exp.iter().sum();
        let s: Vec<f32> = exp.iter().map(|e| e / z).collect();

        for j in 0..n
        {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(logits.clone(), 1, n));
            let x_idx = x.idx();
            let y = x.softmax(1);

            // backprop sur y_j seul
            let mut upstream = vec![0.0f32; n];
            upstream[j] = 1.0;
            let g_var = tape.input(Tensor::from_vec(upstream, 1, n));
            let loss = y.hadamard(g_var).sum();
            loss.backward();

            let grad = tape.grad(x_idx);
            for i in 0..n
            {
                let expected = s[i] * ((i == j) as i32 as f32 - s[j]);
                assert!(
                    (grad.data[i] - expected).abs() < 1e-4,
                    "J[{},{}] = {}, expected {} (s_i={}, s_j={})",
                    i,
                    j,
                    grad.data[i],
                    expected,
                    s[i],
                    s[j]
                );
            }
        }
    }

    #[test]
    fn test_softmax_rows_sum_to_one() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, -1.0, 2.0, 3.0],
            3,
            4,
        ));
        let y = x.softmax(1);
        let y_idx = y.idx();
        let v = tape.value(y_idx);
        for i in 0..3
        {
            let s: f32 = v.data[i * 4..(i + 1) * 4].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {} sum = {}", i, s);
        }
    }

    #[test]
    fn test_transpose2d_gradient() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let a_idx = a.idx();
        let y = a.transpose_2d();
        let loss = y.sum();
        loss.backward();
        let ga = tape.grad(a_idx);
        assert_eq!(ga.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_axis_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let x_idx = x.idx();
        let y = x.mean_axis(0);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert!((g.data[0] - 0.5).abs() < 1e-6);
        assert!((g.data[1] - 0.5).abs() < 1e-6);
        assert!((g.data[2] - 0.5).abs() < 1e-6);
        assert!((g.data[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sum_axis_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let x_idx = x.idx();
        // sum_axis(0) -> shape (1, 2); each output element is sum of a column
        let y = x.sum_axis(0);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        // gradient of sum over all elements wrt each input is 1.0
        assert!((g.data[0] - 1.0).abs() < 1e-6);
        assert!((g.data[1] - 1.0).abs() < 1e-6);
        assert!((g.data[2] - 1.0).abs() < 1e-6);
        assert!((g.data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn broadcast_identity_is_passthrough() {
        // broadcast_to même shape = passthrough (pas de copie, pas d'Op supplémentaire)
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let x_idx = x.idx();
        let y = x.broadcast(2, 2);
        let y_idx = y.idx();

        // Valeur identique
        let v = tape.value(y_idx);
        assert_eq!(v.data, vec![1.0, 2.0, 3.0, 4.0]);

        // Gradient identique (pas de sum implicite)
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert_eq!(g.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_broadcast_gradient_rows() {
        let tape = Tape::new();
        // (1, 2) broadcast to (3, 2)
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let x_idx = x.idx();
        let y = x.broadcast(3, 2);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        // gradient sums over the broadcasted rows -> each col gets 3.0
        assert!((g.data[0] - 3.0).abs() < 1e-6);
        assert!((g.data[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_broadcast_gradient_cols() {
        let tape = Tape::new();
        // (3, 1) broadcast to (3, 2)
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 3, 1));
        let x_idx = x.idx();
        let y = x.broadcast(3, 2);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        // gradient sums over the broadcasted cols -> each row gets 2.0
        assert!((g.data[0] - 2.0).abs() < 1e-6);
        assert!((g.data[1] - 2.0).abs() < 1e-6);
        assert!((g.data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_broadcast_gradient_scalar() {
        let tape = Tape::new();
        // (1, 1) broadcast to (2, 3)
        let x = tape.input(Tensor::from_vec(vec![5.0], 1, 1));
        let x_idx = x.idx();
        let y = x.broadcast(2, 3);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        // gradient sums over all broadcasted elements -> 6.0
        assert!((g.data[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let x_idx = x.idx();
        let y = x.scale(3.0);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert_eq!(g.data, vec![3.0, 3.0]);
    }

    #[test]
    fn test_pow_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![2.0, 3.0], 1, 2));
        let x_idx = x.idx();
        let y = x.pow(2.0);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert!((g.data[0] - 4.0).abs() < 1e-5);
        assert!((g.data[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_sqrt_gradient() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![4.0], 1, 1));
        let x_idx = x.idx();
        let y = x.sqrt();
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert!((g.data[0] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_hadamard_gradient() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![2.0, 3.0], 1, 2));
        let b = tape.input(Tensor::from_vec(vec![4.0, 5.0], 1, 2));
        let a_idx = a.idx();
        let b_idx = b.idx();
        let y = a.hadamard(b);
        let loss = y.sum();
        loss.backward();
        assert_eq!(tape.grad(a_idx).data, vec![4.0, 5.0]);
        assert_eq!(tape.grad(b_idx).data, vec![2.0, 3.0]);
    }

    #[test]
    fn test_reshape_forward_backward() {
        let tape = Tape::new();
        // 2x3 -> 3x2
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3));
        let x_idx = x.idx();
        let y = x.reshape(&[3, 2]);
        let y_idx = y.idx();

        // Verify forward shape
        assert_eq!(tape.value(y_idx).shape(), (3, 2));
        assert_eq!(tape.value(y_idx).data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Verify backward: gradient gets reshaped back to original shape
        let loss = y.sum();
        loss.backward();
        let gx = tape.grad(x_idx);
        assert_eq!(gx.shape(), (2, 3));
        assert_eq!(gx.data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_reciprocal_gradient_at_zero() {
        let tape = Tape::new();
        // x=0 produces gradient via -1/x^2; with epsilon guard, no NaN
        let x = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let x_idx = x.idx();
        let y = x.reciprocal();
        // loss = y, gradient of loss w.r.t. y is 1, so
        // d(loss)/dx = -1/x^2 which at x=0 would be -inf without epsilon guard
        y.backward();
        let g = tape.grad(x_idx);
        assert!(!g.data[0].is_nan(), "gradient should not be NaN");
        assert!(g.data[0].is_finite(), "gradient should be finite");
    }

    #[test]
    fn test_reciprocal_gradient_zero_g_times_inf() {
        let tape = Tape::new();
        // When upstream gradient g = 0 and x = 0, we get 0 * (-inf) = NaN without epsilon guard
        let x = tape.input(Tensor::from_vec(vec![0.0, 2.0], 1, 2));
        let x_idx = x.idx();
        // Create a situation where gradient of y w.r.t. x has 0 upstream:
        // y = reciprocal(x), then z = y * 0 (scale by 0) so upstream grad is 0
        let y = x.reciprocal();
        let zero_grad = y.scale(0.0);
        let loss = zero_grad.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert!(
            !g.data[0].is_nan(),
            "gradient should not be NaN when g=0 and x=0"
        );
        assert!(!g.data[1].is_nan(), "gradient should not be NaN");
    }

    #[test]
    fn detach_cuts_graph() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let x_idx = x.idx();

        let y = x.scale(2.0); // y = [2, 4]
        let y_detached = y.detach(); // detached : nouveau Input sans parents
        let z = y_detached.scale(3.0); // z = [6, 12]
        let loss = z.sum();
        loss.backward();

        // Gradient sur z est 1, mais z n'a pas de lien avec y
        // y_detached est un Input -> backward s'arrete la
        let g_y = tape.grad(y.idx());
        assert!(
            g_y.data.iter().all(|&v| v == 0.0),
            "grad on y should be zero (detached)"
        );

        // x non plus ne devrait pas avoir de gradient
        let g_x = tape.grad(x_idx);
        assert!(
            g_x.data.iter().all(|&v| v == 0.0),
            "grad on x should be zero (detached chain)"
        );
    }

    #[test]
    fn no_grad_forward_works() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));

        let y = tape.no_grad(|| x.scale(3.0));

        // Le forward a quand meme calcule la valeur
        let v = tape.value(y.idx());
        assert_eq!(v.data, vec![3.0, 6.0]);
    }

    #[test]
    fn no_grad_backward_does_not_propagate() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let x_idx = x.idx();

        let y = tape.no_grad(|| x.scale(3.0));
        let loss = y.sum();
        loss.backward();

        // y est un Input (pas de parents), donc grad sur x = 0
        let g_x = tape.grad(x_idx);
        assert!(
            g_x.data.iter().all(|&v| v == 0.0),
            "grad on x should be zero in no_grad"
        );
    }

    #[test]
    fn no_grad_scope_restores_grad() {
        let tape = Tape::new();
        assert!(tape.is_grad_enabled());

        tape.no_grad(|| {
            assert!(!tape.is_grad_enabled());
        });

        assert!(tape.is_grad_enabled());
    }

    #[test]
    fn no_grad_nested() {
        let tape = Tape::new();
        assert!(tape.is_grad_enabled());

        tape.no_grad(|| {
            assert!(!tape.is_grad_enabled());
            tape.no_grad(|| {
                assert!(!tape.is_grad_enabled());
            });
            assert!(!tape.is_grad_enabled()); // toujours false
        });

        assert!(tape.is_grad_enabled()); // restaure
    }

    #[test]
    fn tensor_index_read() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 2)], 3.0);
        assert_eq!(t[(1, 0)], 4.0);
        assert_eq!(t[(1, 2)], 6.0);
    }

    #[test]
    fn tensor_index_mut_write() {
        let mut t = Tensor::zeros(2, 3);
        t[(1, 2)] = 7.0;
        assert_eq!(t[(1, 2)], 7.0);
        assert_eq!(t[(0, 0)], 0.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn tensor_index_panics_oob_row() {
        let t = Tensor::zeros(2, 3);
        let _ = t[(5, 0)];
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn tensor_index_panics_oob_col() {
        let t = Tensor::zeros(2, 3);
        let _ = t[(0, 5)];
    }

    #[test]
    #[should_panic(expected = "size mismatch")]
    fn tensor_from_vec_panics_on_size_mismatch() {
        let _ = Tensor::from_vec(vec![1.0, 2.0, 3.0], 2, 2);
    }

    #[test]
    fn tensor_from_vec_accepts_exact_size() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(1, 1)], 4.0);
    }

    #[test]
    fn matmul_identity() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let i = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2));
        let y = a.matmul(i);
        assert_eq!(tape.value(y.idx()).data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "matmul")]
    fn matmul_panics_on_incompatible_shapes() {
        let tape = Tape::new();
        let a = tape.input(Tensor::zeros(2, 3));
        let b = tape.input(Tensor::zeros(4, 2));
        let _ = a.matmul(b);
    }

    #[test]
    fn softmax_single_element_is_one() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![5.0], 1, 1));
        let y = x.softmax(1);
        let v = tape.value(y.idx());
        assert!((v.data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_numerical_stability_large_values() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], 1, 3));
        let y = x.softmax(1);
        let v = tape.value(y.idx());
        let sum: f32 = v.data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
        // Check no NaN/Inf
        assert!(v.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn causal_mask_blocks_future_tokens() {
        let tape = Tape::new();
        // 2 sequences of length 3 each: shape (2, 3)
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3));
        let y = x.causal_mask(3);
        let v = tape.value(y.idx());
        // Row 0: positions [0,1,2] — future positions 1,2 should be -1e9
        assert!((v.data[0] - 1.0).abs() < 1e-6);
        assert!((v.data[1] - (-1e9)).abs() < 1e-3);
        assert!((v.data[2] - (-1e9)).abs() < 1e-3);
        // Row 1: positions [0,1,2] — only position 2 is future
        assert!((v.data[3] - 4.0).abs() < 1e-6);
        assert!((v.data[4] - 5.0).abs() < 1e-6);
        assert!((v.data[5] - (-1e9)).abs() < 1e-3);
    }

    #[test]
    fn causal_mask_gradient_flows() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let x_idx = x.idx();
        let y = x.causal_mask(3);
        let loss = y.sum();
        loss.backward();
        let g = tape.grad(x_idx);
        // Only first element is unmasked, so only its grad is 1.0
        assert!((g.data[0] - 1.0).abs() < 1e-6);
        assert!((g.data[1] - 0.0).abs() < 1e-6);
        assert!((g.data[2] - 0.0).abs() < 1e-6);
    }
}
