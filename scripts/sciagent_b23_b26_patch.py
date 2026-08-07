#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scirust-cuda/src/chain.rs"
MODEL = ROOT / "scirust-sciagent/src/cuda_model.rs"


def must_replace(text: str, old: str, new: str, *, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise SystemExit(f"expected {count} occurrence(s), found {actual}: {old[:100]!r}")
    return text.replace(old, new, count)


def replace_rust_pub_fn(text: str, name: str, new_src: str) -> str:
    marker = f"    pub fn {name}("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing Rust function {name}")
    end_marker = "\n    }\n"
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"cannot find end of Rust function {name}")
    end += len(end_marker)
    return text[:start] + new_src.rstrip() + "\n" + text[end:]


def remove_top_level_fn(text: str, name: str) -> str:
    marker = f"fn {name}("
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing top-level function {name}")
    end_marker = "\n}\n"
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"cannot find end of top-level function {name}")
    end += len(end_marker)
    return text[:start] + text[end:]


FAST_KERNELS = r'''
// ---- B23-B26 production kernels -------------------------------------------------
// The original Route-B kernels above were deliberately simple correctness oracles.
// These kernels keep exactly the same bf16/fp32 contract while mapping row reductions
// onto a full CUDA block.  A 350M SciAgent step otherwise spends most of its time in
// scalar-per-row reductions rather than Tensor-core GEMMs.

// Block-parallel RMSNorm: one block per row, 256-way fp32 sum-of-squares reduction.
extern "C" __global__ void rmsnorm_fast_kernel(
    unsigned short* out, const unsigned short* x, const unsigned short* w,
    const size_t rows, const size_t cols, const float eps)
{
    __shared__ float red[256];
    const size_t r = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (r >= rows) return;
    float ss = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x) {
        float v = b2f(x[r*cols+j]);
        ss += v*v;
    }
    red[tid] = ss;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid+s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)cols + eps);
    for (size_t j = tid; j < cols; j += blockDim.x)
        out[r*cols+j] = f2b(b2f(x[r*cols+j]) * inv * b2f(w[j]));
}

// Block-parallel row softmax: one block per row instead of one thread per row.
extern "C" __global__ void softmax_fast_kernel(
    unsigned short* out, const unsigned short* x, const size_t rows, const size_t cols)
{
    __shared__ float red[256];
    const size_t r = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (r >= rows) return;
    float mx = -3.0e38f;
    for (size_t j = tid; j < cols; j += blockDim.x)
        mx = fmaxf(mx, b2f(x[r*cols+j]));
    red[tid] = mx;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid+s]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x)
        sum += __expf(b2f(x[r*cols+j]) - mx);
    red[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid+s];
        __syncthreads();
    }
    sum = red[0];
    for (size_t j = tid; j < cols; j += blockDim.x)
        out[r*cols+j] = f2b(__expf(b2f(x[r*cols+j]) - mx) / sum);
}

extern "C" __global__ void softmax_bwd_fast_kernel(
    unsigned short* dx, const unsigned short* y, const unsigned short* dy,
    const size_t rows, const size_t cols)
{
    __shared__ float red[256];
    const size_t r = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (r >= rows) return;
    float dot = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x)
        dot += b2f(dy[r*cols+j]) * b2f(y[r*cols+j]);
    red[tid] = dot;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid+s];
        __syncthreads();
    }
    dot = red[0];
    for (size_t j = tid; j < cols; j += blockDim.x) {
        const float yv = b2f(y[r*cols+j]);
        dx[r*cols+j] = f2b(yv * (b2f(dy[r*cols+j]) - dot));
    }
}

// Input backward: sum-of-squares and dy*w*x are reduced cooperatively once per row.
extern "C" __global__ void rmsnorm_bwd_fast_kernel(
    unsigned short* dx, const unsigned short* x, const unsigned short* w,
    const unsigned short* dy, const size_t rows, const size_t cols, const float eps)
{
    __shared__ float ssred[256];
    __shared__ float dotred[256];
    const size_t r = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (r >= rows) return;
    float ss = 0.0f;
    float dot = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x) {
        const float xv = b2f(x[r*cols+j]);
        ss += xv*xv;
        dot += b2f(dy[r*cols+j]) * b2f(w[j]) * xv;
    }
    ssred[tid] = ss;
    dotred[tid] = dot;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            ssred[tid] += ssred[tid+s];
            dotred[tid] += dotred[tid+s];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(ssred[0] / (float)cols + eps);
    const float coef = dotred[0] * inv * inv * inv / (float)cols;
    for (size_t j = tid; j < cols; j += blockDim.x) {
        const float xv = b2f(x[r*cols+j]);
        dx[r*cols+j] = f2b(b2f(dy[r*cols+j]) * b2f(w[j]) * inv - xv * coef);
    }
}

// Compute inv_rms once per row. The old gain-backward recomputed this O(cols) work
// once for every output column, making it O(rows*cols^2).
extern "C" __global__ void rmsnorm_inv_kernel(
    float* inv_rms, const unsigned short* x,
    const size_t rows, const size_t cols, const float eps)
{
    __shared__ float red[256];
    const size_t r = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (r >= rows) return;
    float ss = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x) {
        const float v = b2f(x[r*cols+j]);
        ss += v*v;
    }
    red[tid] = ss;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid+s];
        __syncthreads();
    }
    if (tid == 0) inv_rms[r] = rsqrtf(red[0] / (float)cols + eps);
}

extern "C" __global__ void rmsnorm_gain_bwd_fast_kernel(
    unsigned short* dw, const unsigned short* x, const unsigned short* dy,
    const float* inv_rms, const size_t rows, const size_t cols)
{
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float acc = 0.0f;
        for (size_t r = 0; r < rows; r++)
            acc += b2f(dy[r*cols+j]) * b2f(x[r*cols+j]) * inv_rms[r];
        dw[j] = f2b(acc);
    }
}

// Sparse deterministic embedding-gather adjoint. Host-side preprocessing supplies
// sorted unique token ids plus their source row positions. Work is O(vocab*d) only
// for the required zero-fill plus O(tokens*d) arithmetic, instead of the old
// O(vocab*d*tokens) scan (17.2B comparisons at 32768*1024*512).
extern "C" __global__ void embed_bwd_sparse_kernel(
    unsigned short* dtable, const unsigned int* unique_tokens,
    const unsigned int* offsets, const unsigned int* positions,
    const unsigned short* dout, const size_t n_unique,
    const size_t d, const size_t vocab)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_unique * d) {
        const size_t u = idx / d;
        const size_t c = idx % d;
        const size_t v = (size_t)unique_tokens[u];
        if (v >= vocab) return;
        float acc = 0.0f;
        for (unsigned int p = offsets[u]; p < offsets[u+1]; p++) {
            const size_t r = (size_t)positions[p];
            acc += b2f(dout[r*d+c]);
        }
        dtable[v*d+c] = f2b(acc);
    }
}

// Device-side CE evaluation: one block per logit row, only one fp32 scalar per row
// is copied to the host. This removes the 32 MiB bf16 logits D2H transfer per
// seq=512 training step (and the host-side exp loop over 16.8M logits).
extern "C" __global__ void ce_loss_kernel(
    float* loss_rows, const unsigned short* logits, const unsigned int* targets,
    const size_t rows, const size_t cols)
{
    __shared__ float red[256];
    const size_t r = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (r >= rows) return;
    float mx = -3.0e38f;
    for (size_t j = tid; j < cols; j += blockDim.x)
        mx = fmaxf(mx, b2f(logits[r*cols+j]));
    red[tid] = mx;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid+s]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x)
        sum += __expf(b2f(logits[r*cols+j]) - mx);
    red[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid+s];
        __syncthreads();
    }
    if (tid == 0) {
        size_t tgt = (size_t)targets[r];
        if (tgt >= cols) tgt = cols - 1;
        loss_rows[r] = -(b2f(logits[r*cols+tgt]) - mx) + logf(red[0]);
    }
}

extern "C" __global__ void ce_loss_grad_kernel(
    float* loss_rows, unsigned short* d,
    const unsigned short* logits, const unsigned int* targets,
    const size_t rows, const size_t cols)
{
    __shared__ float red[256];
    const size_t r = (size_t)blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (r >= rows) return;
    float mx = -3.0e38f;
    for (size_t j = tid; j < cols; j += blockDim.x)
        mx = fmaxf(mx, b2f(logits[r*cols+j]));
    red[tid] = mx;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid+s]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.0f;
    for (size_t j = tid; j < cols; j += blockDim.x)
        sum += __expf(b2f(logits[r*cols+j]) - mx);
    red[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid+s];
        __syncthreads();
    }
    sum = red[0];
    size_t tgt = (size_t)targets[r];
    if (tgt >= cols) tgt = cols - 1;
    if (tid == 0)
        loss_rows[r] = -(b2f(logits[r*cols+tgt]) - mx) + logf(sum);
    const float inv_rows = 1.0f / (float)rows;
    for (size_t j = tid; j < cols; j += blockDim.x) {
        float p = __expf(b2f(logits[r*cols+j]) - mx) / sum;
        if (j == tgt) p -= 1.0f;
        d[r*cols+j] = f2b(p * inv_rows);
    }
}
'''


def patch_chain(text: str) -> str:
    if "B23-B26 production kernels" in text:
        raise SystemExit("chain.rs already patched")
    text = must_replace(
        text,
        "// AdamW step (mixed precision): fp32 master `param`, fp32 moments `m`/`v`, bf16\n",
        FAST_KERNELS + "\n// AdamW step (mixed precision): fp32 master `param`, fp32 moments `m`/`v`, bf16\n",
    )

    # Kernel handles: preserve the public operation surface, but point it at the
    # production kernels. Two extra handles support cached RMS statistics and CE loss.
    text = must_replace(text, "    ce_grad: CudaFunction,\n", "    rmsnorm_inv: CudaFunction,\n    ce_loss: CudaFunction,\n    ce_loss_grad: CudaFunction,\n")
    text = must_replace(text, '            rmsnorm: f("rmsnorm_kernel"),\n', '            rmsnorm: f("rmsnorm_fast_kernel"),\n')
    text = must_replace(text, '            softmax: f("softmax_kernel"),\n', '            softmax: f("softmax_fast_kernel"),\n')
    text = must_replace(text, '            softmax_bwd: f("softmax_bwd_kernel"),\n', '            softmax_bwd: f("softmax_bwd_fast_kernel"),\n')
    text = must_replace(text, '            rmsnorm_bwd: f("rmsnorm_bwd_kernel"),\n', '            rmsnorm_bwd: f("rmsnorm_bwd_fast_kernel"),\n')
    text = must_replace(text, '            rmsnorm_gain_bwd: f("rmsnorm_gain_bwd_kernel"),\n', '            rmsnorm_gain_bwd: f("rmsnorm_gain_bwd_fast_kernel"),\n')
    text = must_replace(text, '            embed_bwd: f("embed_bwd_kernel"),\n', '            embed_bwd: f("embed_bwd_sparse_kernel"),\n')
    text = must_replace(
        text,
        '            ce_grad: f("ce_grad_kernel"),\n',
        '            rmsnorm_inv: f("rmsnorm_inv_kernel"),\n            ce_loss: f("ce_loss_kernel"),\n            ce_loss_grad: f("ce_loss_grad_kernel"),\n',
    )

    text = replace_rust_pub_fn(text, "rms_norm", r'''    pub fn rms_norm(&self, x: &CudaMatrix, weight: &CudaMatrix, eps: f32) -> CudaMatrix {
        assert_eq!(
            weight.rows * weight.cols,
            x.cols,
            "rms_norm: weight has {} elems, expected cols = {}",
            weight.rows * weight.cols,
            x.cols
        );
        let (rows, cols) = (x.rows, x.cols);
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a, eps_a) = (rows, cols, eps);
        let mut builder = self.stream.launch_builder(&self.kernels().rmsnorm);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&weight.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        builder.arg(&eps_a);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one 256-thread block owns each row and performs the fp32 reduction.
        unsafe { builder.launch(cfg).expect("launch rmsnorm_fast_kernel") };
        CudaMatrix { buf: out, rows, cols }
    }
''')

    text = replace_rust_pub_fn(text, "softmax", r'''    pub fn softmax(&self, x: &CudaMatrix) -> CudaMatrix {
        let (rows, cols) = (x.rows, x.cols);
        let mut out = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().softmax);
        builder.arg(&mut out);
        builder.arg(&x.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one 256-thread block owns each row.
        unsafe { builder.launch(cfg).expect("launch softmax_fast_kernel") };
        CudaMatrix { buf: out, rows, cols }
    }
''')

    text = replace_rust_pub_fn(text, "softmax_backward", r'''    pub fn softmax_backward(&self, y: &CudaMatrix, dy: &CudaMatrix) -> CudaMatrix {
        assert_eq!(
            (y.rows, y.cols),
            (dy.rows, dy.cols),
            "softmax_backward: y {}x{} vs dy {}x{}",
            y.rows,
            y.cols,
            dy.rows,
            dy.cols
        );
        let (rows, cols) = (y.rows, y.cols);
        let mut dx = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().softmax_bwd);
        builder.arg(&mut dx);
        builder.arg(&y.buf);
        builder.arg(&dy.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one 256-thread block owns each row.
        unsafe { builder.launch(cfg).expect("launch softmax_bwd_fast_kernel") };
        CudaMatrix { buf: dx, rows, cols }
    }
''')

    text = replace_rust_pub_fn(text, "rms_norm_backward", r'''    pub fn rms_norm_backward(
        &self,
        x: &CudaMatrix,
        weight: &CudaMatrix,
        dy: &CudaMatrix,
        eps: f32,
    ) -> CudaMatrix {
        assert_eq!((x.rows, x.cols), (dy.rows, dy.cols), "rms_norm_backward: x/dy shape");
        assert_eq!(
            weight.rows * weight.cols,
            x.cols,
            "rms_norm_backward: weight {} elems, expected {}",
            weight.rows * weight.cols,
            x.cols
        );
        let (rows, cols) = (x.rows, x.cols);
        let mut dx = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc");
        let (rows_a, cols_a, eps_a) = (rows, cols, eps);
        let mut builder = self.stream.launch_builder(&self.kernels().rmsnorm_bwd);
        builder.arg(&mut dx);
        builder.arg(&x.buf);
        builder.arg(&weight.buf);
        builder.arg(&dy.buf);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        builder.arg(&eps_a);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one block owns each row; both fp32 reductions are block-local.
        unsafe { builder.launch(cfg).expect("launch rmsnorm_bwd_fast_kernel") };
        CudaMatrix { buf: dx, rows, cols }
    }
''')

    text = replace_rust_pub_fn(text, "rms_norm_gain_backward", r'''    pub fn rms_norm_gain_backward(&self, x: &CudaMatrix, dy: &CudaMatrix, eps: f32) -> CudaMatrix {
        assert_eq!(
            (x.rows, x.cols),
            (dy.rows, dy.cols),
            "rms_norm_gain_backward: x/dy shape"
        );
        let (rows, cols) = (x.rows, x.cols);
        let mut inv_rms = self.stream.alloc_zeros::<f32>(rows).expect("cuda alloc inv_rms");
        let (rows_a, cols_a, eps_a) = (rows, cols, eps);
        let mut stats = self.stream.launch_builder(&self.kernels().rmsnorm_inv);
        stats.arg(&mut inv_rms);
        stats.arg(&x.buf);
        stats.arg(&rows_a);
        stats.arg(&cols_a);
        stats.arg(&eps_a);
        let stats_cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one block owns each row and writes exactly inv_rms[row].
        unsafe { stats.launch(stats_cfg).expect("launch rmsnorm_inv_kernel") };

        let mut dw = self.stream.alloc_zeros::<bf16>(cols).expect("cuda alloc");
        let mut builder = self.stream.launch_builder(&self.kernels().rmsnorm_gain_bwd);
        builder.arg(&mut dw);
        builder.arg(&x.buf);
        builder.arg(&dy.buf);
        builder.arg(&inv_rms);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        // SAFETY: grid covers one independent reduction per output column.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(cols as u32))
                .expect("launch rmsnorm_gain_bwd_fast_kernel");
        }
        CudaMatrix { buf: dw, rows: 1, cols }
    }
''')

    text = replace_rust_pub_fn(text, "embed_backward", r'''    pub fn embed_backward(&self, tokens: &[u32], dout: &CudaMatrix, vocab: usize) -> CudaMatrix {
        let (n, d) = (tokens.len(), dout.cols);
        assert_eq!(dout.rows, n, "embed_backward: dout rows {} != tokens {}", dout.rows, n);
        assert!(vocab > 0, "embed_backward: vocab must be non-zero");

        // Sort (token,row) pairs once on the CPU. Every sparse output row is then
        // owned by exactly d GPU threads, so repeated token ids remain deterministic
        // without atomics while absent vocab rows stay zero from alloc_zeros.
        let vmax = (vocab - 1) as u32;
        let mut pairs: Vec<(u32, u32)> = tokens
            .iter()
            .enumerate()
            .map(|(row, &tok)| (tok.min(vmax), row as u32))
            .collect();
        pairs.sort_unstable();
        let mut unique = Vec::<u32>::new();
        let mut offsets = Vec::<u32>::new();
        let mut positions = Vec::<u32>::with_capacity(n);
        for (tok, row) in pairs {
            if unique.last().copied() != Some(tok) {
                unique.push(tok);
                offsets.push(positions.len() as u32);
            }
            positions.push(row);
        }
        offsets.push(positions.len() as u32);

        let mut dtable = self
            .stream
            .alloc_zeros::<bf16>(vocab * d)
            .expect("cuda alloc embed grad");
        if unique.is_empty() {
            return CudaMatrix { buf: dtable, rows: vocab, cols: d };
        }
        let unique_dev = self.stream.clone_htod(&unique).expect("cuda htod unique tokens");
        let offsets_dev = self.stream.clone_htod(&offsets).expect("cuda htod offsets");
        let positions_dev = self.stream.clone_htod(&positions).expect("cuda htod positions");
        let (n_unique_a, d_a, vocab_a) = (unique.len(), d, vocab);
        let mut builder = self.stream.launch_builder(&self.kernels().embed_bwd);
        builder.arg(&mut dtable);
        builder.arg(&unique_dev);
        builder.arg(&offsets_dev);
        builder.arg(&positions_dev);
        builder.arg(&dout.buf);
        builder.arg(&n_unique_a);
        builder.arg(&d_a);
        builder.arg(&vocab_a);
        let work = unique.len() * d;
        // SAFETY: offsets has n_unique+1 entries and positions contains valid source rows.
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(work as u32))
                .expect("launch embed_bwd_sparse_kernel");
        }
        CudaMatrix { buf: dtable, rows: vocab, cols: d }
    }
''')

    text = replace_rust_pub_fn(text, "cross_entropy_grad", r'''    pub fn cross_entropy_grad(&self, logits: &CudaMatrix, targets: &[u32]) -> CudaMatrix {
        self.cross_entropy_loss_grad(logits, targets).1
    }
''')

    # Add CE loss APIs immediately before cross_entropy_grad's doc comment.
    marker = "    /// Cross-entropy gradient w.r.t. the logits:"
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit("missing cross_entropy_grad docs insertion point")
    ce_methods = r'''    /// Mean next-token cross-entropy computed on the device. Only `rows` fp32 loss
    /// scalars are copied back, instead of downloading the full `rows × vocab` logits.
    pub fn cross_entropy_loss(&self, logits: &CudaMatrix, targets: &[u32]) -> f32 {
        let (rows, cols) = (logits.rows, logits.cols);
        assert!(rows > 0 && cols > 0, "cross_entropy_loss: empty logits");
        assert_eq!(targets.len(), rows, "cross_entropy_loss: target count");
        let tgt = self.stream.clone_htod(targets).expect("cuda htod targets");
        let mut loss_rows = self.stream.alloc_zeros::<f32>(rows).expect("cuda alloc CE loss");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().ce_loss);
        builder.arg(&mut loss_rows);
        builder.arg(&logits.buf);
        builder.arg(&tgt);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one block owns each row and writes one loss scalar.
        unsafe { builder.launch(cfg).expect("launch ce_loss_kernel") };
        let host = self.stream.clone_dtoh(&loss_rows).expect("cuda dtoh CE loss");
        (host.iter().map(|&v| v as f64).sum::<f64>() / rows as f64) as f32
    }

    /// Mean cross-entropy plus its resident bf16 logit gradient in one pass.
    /// The only D2H transfer is `rows` fp32 scalars used for deterministic logging.
    pub fn cross_entropy_loss_grad(
        &self,
        logits: &CudaMatrix,
        targets: &[u32],
    ) -> (f32, CudaMatrix) {
        let (rows, cols) = (logits.rows, logits.cols);
        assert!(rows > 0 && cols > 0, "cross_entropy_loss_grad: empty logits");
        assert_eq!(targets.len(), rows, "cross_entropy_loss_grad: target count");
        let tgt = self.stream.clone_htod(targets).expect("cuda htod targets");
        let mut loss_rows = self.stream.alloc_zeros::<f32>(rows).expect("cuda alloc CE loss");
        let mut d = self
            .stream
            .alloc_zeros::<bf16>(rows * cols)
            .expect("cuda alloc CE grad");
        let (rows_a, cols_a) = (rows, cols);
        let mut builder = self.stream.launch_builder(&self.kernels().ce_loss_grad);
        builder.arg(&mut loss_rows);
        builder.arg(&mut d);
        builder.arg(&logits.buf);
        builder.arg(&tgt);
        builder.arg(&rows_a);
        builder.arg(&cols_a);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one block owns each row and writes disjoint gradient elements.
        unsafe { builder.launch(cfg).expect("launch ce_loss_grad_kernel") };
        let host = self.stream.clone_dtoh(&loss_rows).expect("cuda dtoh CE loss");
        let loss = (host.iter().map(|&v| v as f64).sum::<f64>() / rows as f64) as f32;
        (loss, CudaMatrix { buf: d, rows, cols })
    }

'''
    text = text[:idx] + ce_methods + text[idx:]
    return text


def patch_model(text: str) -> str:
    text = must_replace(
        text,
        "        // Forward (resident) → host loss → cross-entropy grad → backward.\n        let logits = self.model.forward_resident(tokens);\n        let host = self.model.chain.download(&logits);\n        let loss = host_cross_entropy(&host, targets, rows, vocab);\n        let dlogits = self.model.chain.cross_entropy_grad(&logits, targets);\n",
        "        // Forward and CE stay resident. Only one fp32 loss scalar per token row\n        // crosses to the host; the full rows×vocab logit matrix never does.\n        let logits = self.model.forward_resident(tokens);\n        let (loss, dlogits) = self.model.chain.cross_entropy_loss_grad(&logits, targets);\n",
    )
    # rows/vocab were used only by the removed host CE path in train_step.
    text = must_replace(text, "        let rows = tokens.len();\n        let vocab = self.model.vocab;\n\n        // Forward and CE stay resident.", "        // Forward and CE stay resident.")

    text = must_replace(
        text,
        "            let logits = self.forward_resident(inputs);\n            let host = self.chain.download(&logits);\n            total += host_cross_entropy(&host, targets, s, self.vocab) as f64;\n",
        "            let logits = self.forward_resident(inputs);\n            total += self.chain.cross_entropy_loss(&logits, targets) as f64;\n",
    )
    text = must_replace(
        text,
        "        let vocab = self.model.vocab;\n        let mut total = 0.0f64;\n",
        "        let mut total = 0.0f64;\n",
    )
    text = must_replace(
        text,
        "            let logits = self.model.forward_resident(inputs);\n            let host = self.model.chain.download(&logits);\n            total += host_cross_entropy(&host, targets, s, vocab) as f64;\n",
        "            let logits = self.model.forward_resident(inputs);\n            total += self.model.chain.cross_entropy_loss(&logits, targets) as f64;\n",
    )
    text = remove_top_level_fn(text, "host_cross_entropy")
    if "host_cross_entropy" in text:
        raise SystemExit("host_cross_entropy references remain")
    return text


chain = CHAIN.read_text()
model = MODEL.read_text()
CHAIN.write_text(patch_chain(chain))
MODEL.write_text(patch_model(model))
print("patched", CHAIN.relative_to(ROOT), MODEL.relative_to(ROOT))
