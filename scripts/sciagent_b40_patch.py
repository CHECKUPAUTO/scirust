#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"


def must_replace(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        raise SystemExit(f"expected {count}, found {n}: {old[:160]!r}")
    return text.replace(old, new, count)


def replace_fn(text: str, marker: str, new_src: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing {marker!r}")
    brace = text.find("{", start)
    depth = 0
    in_str = False
    escaped = False
    i = brace
    while i < len(text):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[:start] + new_src.rstrip() + text[i + 1 :]
        i += 1
    raise SystemExit(f"unterminated {marker!r}")


ROPE_KERNEL_OLD = r'''// RoPE: interleaved-pair rotation. pos = (row mod seq_len) + offset,
// freq_p = theta^(-2p/dim), angle = pos*freq_p; one thread per (row, pair).
extern "C" __global__ void rope_kernel(
    unsigned short* out, const unsigned short* x, const size_t rows, const size_t dim,
    const size_t seq_len, const size_t offset, const float theta)
{
    size_t pairs = dim / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * pairs) {
        size_t r = idx / pairs, p = idx % pairs;
        float pos = (float)((r % seq_len) + offset);
        float freq = powf(theta, -2.0f * (float)p / (float)dim);
        float ang = pos * freq, c = cosf(ang), s = sinf(ang);
        float x0 = b2f(x[r*dim + 2*p]);
        float x1 = b2f(x[r*dim + 2*p + 1]);
        out[r*dim + 2*p]     = f2b(x0 * c - x1 * s);
        out[r*dim + 2*p + 1] = f2b(x0 * s + x1 * c);
    }
}
'''

ROPE_KERNEL_NEW = r'''// RoPE: interleaved-pair rotation. `head_dim` controls the rotary frequency
// period. Passing head_dim=dim preserves plain full-width RoPE; passing d_head makes
// the frequency index restart inside every logical attention head without slicing
// the matrix into one CUDA launch per head.
extern "C" __global__ void rope_kernel(
    unsigned short* out, const unsigned short* x, const size_t rows, const size_t dim,
    const size_t head_dim, const size_t seq_len, const size_t offset, const float theta)
{
    const size_t pairs = dim / 2;
    const size_t head_pairs = head_dim / 2;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * pairs) {
        const size_t r = idx / pairs;
        const size_t p = idx % pairs;
        const size_t local_p = p % head_pairs;
        const float pos = (float)((r % seq_len) + offset);
        const float freq = powf(theta, -2.0f * (float)local_p / (float)head_dim);
        const float ang = pos * freq, c = cosf(ang), s = sinf(ang);
        const float x0 = b2f(x[r*dim + 2*p]);
        const float x1 = b2f(x[r*dim + 2*p + 1]);
        out[r*dim + 2*p]     = f2b(x0 * c - x1 * s);
        out[r*dim + 2*p + 1] = f2b(x0 * s + x1 * c);
    }
}
'''

ROPE_BWD_OLD = r'''// RoPE backward — the adjoint (transpose) rotation, same pos/freq as rope_kernel:
// dx[2p] = c·dy[2p] + s·dy[2p+1], dx[2p+1] = −s·dy[2p] + c·dy[2p+1].
extern "C" __global__ void rope_bwd_kernel(
    unsigned short* dx, const unsigned short* dy, const size_t rows, const size_t dim,
    const size_t seq_len, const size_t offset, const float theta)
{
    size_t pairs = dim / 2;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * pairs) {
        size_t r = idx / pairs, p = idx % pairs;
        float pos = (float)((r % seq_len) + offset);
        float freq = powf(theta, -2.0f * (float)p / (float)dim);
        float ang = pos * freq, c = cosf(ang), s = sinf(ang);
        float ge = b2f(dy[r*dim + 2*p]);
        float go = b2f(dy[r*dim + 2*p + 1]);
        dx[r*dim + 2*p]     = f2b(c * ge + s * go);
        dx[r*dim + 2*p + 1] = f2b(-s * ge + c * go);
    }
}
'''

ROPE_BWD_NEW = r'''// RoPE backward — adjoint of rope_kernel with the same head-local frequency period.
extern "C" __global__ void rope_bwd_kernel(
    unsigned short* dx, const unsigned short* dy, const size_t rows, const size_t dim,
    const size_t head_dim, const size_t seq_len, const size_t offset, const float theta)
{
    const size_t pairs = dim / 2;
    const size_t head_pairs = head_dim / 2;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * pairs) {
        const size_t r = idx / pairs;
        const size_t p = idx % pairs;
        const size_t local_p = p % head_pairs;
        const float pos = (float)((r % seq_len) + offset);
        const float freq = powf(theta, -2.0f * (float)local_p / (float)head_dim);
        const float ang = pos * freq, c = cosf(ang), s = sinf(ang);
        const float ge = b2f(dy[r*dim + 2*p]);
        const float go = b2f(dy[r*dim + 2*p + 1]);
        dx[r*dim + 2*p]     = f2b(c * ge + s * go);
        dx[r*dim + 2*p + 1] = f2b(-s * ge + c * go);
    }
}
'''

ROPE_METHOD_NEW = r'''    /// RoPE over the full matrix width (frequency denominator = width). Kept for
    /// generic callers; GQA uses [`Self::rope_head_local`] instead.
    pub fn rope(&self, x: &CudaMatrix, seq_len: usize, offset: usize, theta: f32) -> CudaMatrix {
        self.rope_with_head_dim(x, x.cols, seq_len, offset, theta)
    }

    /// Head-local RoPE in one CUDA launch. `head_dim` must evenly tile the matrix
    /// width and is the rotary-frequency denominator for every logical head.
    pub fn rope_head_local(
        &self,
        x: &CudaMatrix,
        head_dim: usize,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> CudaMatrix {
        self.rope_with_head_dim(x, head_dim, seq_len, offset, theta)
    }

    fn rope_with_head_dim(
        &self,
        x: &CudaMatrix,
        head_dim: usize,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> CudaMatrix {
        assert_eq!(x.cols % 2, 0, "rope: dim must be even, got {}", x.cols);
        assert!(head_dim > 0 && head_dim.is_multiple_of(2), "rope: head_dim must be positive/even");
        assert!(x.cols.is_multiple_of(head_dim), "rope: width must be divisible by head_dim");
        let (rows, dim) = (x.rows, x.cols);
        let total = rows * (dim / 2);
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * dim)
            .expect("cuda alloc");
        let (rows_a, dim_a, head_a, seq_a, off_a, theta_a) =
            (rows, dim, head_dim, seq_len, offset, theta);
        let mut builder = self.stream.launch_builder(&self.kernels().rope);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&rows_a);
        builder.arg(&dim_a);
        builder.arg(&head_a);
        builder.arg(&seq_a);
        builder.arg(&off_a);
        builder.arg(&theta_a);
        // SAFETY: kernel covers one adjacent pair and head_dim partitions dim.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch rope_kernel");
        }
        CudaMatrix { buf: out, rows, cols: dim }
    }'''

ROPE_BWD_METHOD_NEW = r'''    /// Full-width RoPE adjoint. GQA uses [`Self::rope_head_local_backward`].
    pub fn rope_backward(
        &self,
        dy: &CudaMatrix,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> CudaMatrix {
        self.rope_backward_with_head_dim(dy, dy.cols, seq_len, offset, theta)
    }

    pub fn rope_head_local_backward(
        &self,
        dy: &CudaMatrix,
        head_dim: usize,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> CudaMatrix {
        self.rope_backward_with_head_dim(dy, head_dim, seq_len, offset, theta)
    }

    fn rope_backward_with_head_dim(
        &self,
        dy: &CudaMatrix,
        head_dim: usize,
        seq_len: usize,
        offset: usize,
        theta: f32,
    ) -> CudaMatrix {
        assert_eq!(dy.cols % 2, 0, "rope_backward: dim must be even, got {}", dy.cols);
        assert!(head_dim > 0 && head_dim.is_multiple_of(2), "rope_backward: invalid head_dim");
        assert!(dy.cols.is_multiple_of(head_dim), "rope_backward: width/head_dim mismatch");
        let (rows, dim) = (dy.rows, dy.cols);
        let total = rows * (dim / 2);
        let mut dx = self
            .stream
            .alloc_zeros::<bf16>(rows * dim)
            .expect("cuda alloc");
        let (rows_a, dim_a, head_a, seq_a, off_a, theta_a) =
            (rows, dim, head_dim, seq_len, offset, theta);
        let mut builder = self.stream.launch_builder(&self.kernels().rope_bwd);
        builder.arg(&mut dx);
        builder.arg(&dy.buf);
        builder.arg(&rows_a);
        builder.arg(&dim_a);
        builder.arg(&head_a);
        builder.arg(&seq_a);
        builder.arg(&off_a);
        builder.arg(&theta_a);
        // SAFETY: kernel covers one adjacent pair and head_dim partitions dim.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(total as u32))
                .expect("launch rope_bwd_kernel");
        }
        CudaMatrix { buf: dx, rows, cols: dim }
    }'''

CUDA_MODEL_ROPE = r'''    /// GQA-correct RoPE without per-head slicing: one kernel covers the full
    /// projection while the frequency index restarts every `d_head` columns.
    fn rope_heads(
        &self,
        x: &CudaMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> CudaMatrix {
        assert!(n_heads > 0 && x.cols().is_multiple_of(n_heads));
        let dh = x.cols() / n_heads;
        self.chain
            .rope_head_local(x, dh, seq_len, offset, self.theta)
    }

    fn rope_heads_backward(
        &self,
        dy: &CudaMatrix,
        n_heads: usize,
        seq_len: usize,
        offset: usize,
    ) -> CudaMatrix {
        assert!(n_heads > 0 && dy.cols().is_multiple_of(n_heads));
        let dh = dy.cols() / n_heads;
        self.chain
            .rope_head_local_backward(dy, dh, seq_len, offset, self.theta)
    }'''


def patch_chain(text: str) -> str:
    if "rope_head_local(" in text:
        raise SystemExit("chain already B40 patched")
    text = must_replace(text, ROPE_KERNEL_OLD, ROPE_KERNEL_NEW)
    text = must_replace(text, ROPE_BWD_OLD, ROPE_BWD_NEW)
    text = replace_fn(text, "    pub fn rope(&self, x: &CudaMatrix", ROPE_METHOD_NEW)
    text = replace_fn(text, "    pub fn rope_backward(\n", ROPE_BWD_METHOD_NEW)
    return text


def patch_model(text: str) -> str:
    if "GQA-correct RoPE without per-head slicing" in text:
        raise SystemExit("model already B40 patched")
    text = replace_fn(text, "    fn rope_heads(\n", CUDA_MODEL_ROPE)

    old = '''        let (mut order, val_windows) = distributed_window_split(tokens.len(), s, cfg.val_frac);\n        let n_windows = order.len();\n'''
    new = '''        let (base_order, val_windows) = distributed_window_split(tokens.len(), s, cfg.val_frac);\n        let n_windows = base_order.len();\n'''
    text = must_replace(text, old, new)
    old2 = '''        let mut epoch: u64 = (consumed_windows / n_windows) as u64;\n        let mut wi = consumed_windows % n_windows;\n        let reshuffle = |order: &mut [usize], epoch: u64| {\n            shuffle_windows(order, 0x5343_4941_4745_4E54u64 ^ epoch);\n        };\n        if cfg.shuffle\n        {\n            reshuffle(&mut order, epoch);\n        }\n'''
    new2 = '''        let mut epoch: u64 = (consumed_windows / n_windows) as u64;\n        let mut wi = consumed_windows % n_windows;\n        let mut order = base_order.clone();\n        let set_epoch_order = |order: &mut Vec<usize>, epoch: u64| {\n            // Every epoch permutation is a pure function of (base_order, epoch).\n            // Do not shuffle the previous epoch's permutation: a resumed process\n            // must reconstruct epoch N without replaying epochs 0..N-1.\n            order.clone_from(&base_order);\n            if cfg.shuffle\n            {\n                shuffle_windows(order, 0x5343_4941_4745_4E54u64 ^ epoch);\n            }\n        };\n        set_epoch_order(&mut order, epoch);\n'''
    text = must_replace(text, old2, new2)
    old3 = '''                    epoch += 1;\n                    if cfg.shuffle\n                    {\n                        reshuffle(&mut order, epoch);\n                    }\n                    wi = 0;\n'''
    new3 = '''                    epoch += 1;\n                    set_epoch_order(&mut order, epoch);\n                    wi = 0;\n'''
    text = must_replace(text, old3, new3)
    return text


CHAIN.write_text(patch_chain(CHAIN.read_text()))
MODEL.write_text(patch_model(MODEL.read_text()))
print("B40 patched: pure epoch shuffle + single-launch head-local CUDA RoPE")
