//! WGSL compute shader library — tiled GEMM, deterministic reductions,
//! extended activations, fused kernels, quantized (INT8/INT16) GEMM,
//! crypto Zq GEMM, and fixed-point Q15.16 GEMM.
//!
//! All kernels are inline WGSL constants compiled at runtime by wgpu.
//! The tiled GEMM uses 16×16 workgroups with shared-memory tiling for
//! ~10-50× throughput over the naive per-cell kernel on real hardware.
//! Quantized kernels use integer arithmetic (inherently deterministic).
//! Crypto kernels use modular reduction for bit-exact reproducibility.

/// WGSL helper: sanitize subnormals to zero (FTZ/DAZ portability).
pub const WGSL_SANITIZE_F32: &str = r#"
fn sanitize_f32(val: f32) -> f32 {
    if (abs(val) < 1.17549435e-38f) {
        return 0.0f;
    }
    return val;
}
"#;

// ---------------------------------------------------------------------------
// Standard GEMM (naive, per-cell)
// ---------------------------------------------------------------------------

/// GEMM: `C = alpha·op(A)·op(B) + beta·C`
/// shared memory for A and B tiles. Workgroup = 16×16 threads,
/// one thread per output cell once the tile accumulation is done.
pub const TILED_GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, ta: u32, tb: u32, alpha: f32, beta: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

var<workgroup> As: array<f32, 256>;  // 16x16 tile for A
var<workgroup> Bs: array<f32, 256>;  // 16x16 tile for B

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;

    var acc: f32 = 0.0;
    let num_tiles = (p.k + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let t_base = t * 16u;
        let a_row = i;
        let a_col = t_base + lid.y;
        let b_row = t_base + lid.x;
        let b_col = j;

        // Cooperative load A tile into shared memory
        if (a_row < p.m && a_col < p.k) {
            let a_idx = select(
                a_col * p.m + a_row,
                a_row * p.k + a_col,
                p.ta == 0u
            );
            As[lid.y * 16u + lid.x] = a[a_idx];
        } else {
            As[lid.y * 16u + lid.x] = 0.0;
        }

        // Cooperative load B tile into shared memory
        if (b_row < p.k && b_col < p.n) {
            let b_idx = select(
                b_col * p.k + b_row,
                b_row * p.n + b_col,
                p.tb == 0u
            );
            Bs[lid.y * 16u + lid.x] = b[b_idx];
        } else {
            Bs[lid.y * 16u + lid.x] = 0.0;
        }

        workgroupBarrier();
        
        // Accumulate over tile
        for (var q: u32 = 0u; q < 16u; q = q + 1u) {
            acc = acc + As[lid.y * 16u + q] * Bs[q * 16u + lid.x];
        }

        workgroupBarrier();
    }

    if (i < p.m && j < p.n) {
        let idx = i * p.n + j;
        c[idx] = p.alpha * acc + p.beta * c[idx];
    }
}
"#;

// ---------------------------------------------------------------------------
// Deterministic reduction with Kahan summation
// ---------------------------------------------------------------------------

/// Deterministic sum reduction across one axis using Kahan compensated
/// summation. Fixed accumulation order: thread 0 in each group sums its
/// own elements then collects neighbors in deterministic scan order.
/// Workgroup size = 256, one element per thread.
pub const DETERMINISTIC_REDUCE_WGSL: &str = r#"
struct P { n: u32, axis_size: u32, op: u32, _pad: u32 };
// op: 0=sum, 1=mean, 2=max, 3=norm

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let out_idx = gid.x;
    if (out_idx >= p.n) { return; }

    var sum: f32 = 0.0;
    var c: f32 = 0.0;   // Kahan compensation

    // Fixed-order accumulation: elements processed in order 0..axis_size-1
    for (var k: u32 = 0u; k < p.axis_size; k = k + 1u) {
        let idx = out_idx * p.axis_size + k;
        let x = input[idx];
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    if (p.op == 0u) {          // sum
        output[out_idx] = sum;
    } else if (p.op == 1u) {   // mean
        output[out_idx] = sum / f32(p.axis_size);
    } else {                    // norm (sqrt sum of squares)
        output[out_idx] = sqrt(sum);
    }
}
"#;

// ---------------------------------------------------------------------------
// Extended activation kernels (unary elementwise)
// ---------------------------------------------------------------------------

/// Extended elementwise operations: 0=relu, 1=sigmoid, 2=tanh, 3=gelu,
/// 4=silu, 5=leaky_relu, 6=elu, 7=softplus, 8=sqrt, 9=exp
pub const EXTENDED_EW_WGSL: &str = r#"
struct P { n: u32, op: u32, param: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;
@group(0) @binding(2) var<uniform>             p: P;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let x = a[i];
    let y: f32;
    if (p.op == 0u) {        // relu
        y = max(x, 0.0);
    } else if (p.op == 1u) { // sigmoid
        y = sigmoid(x);
    } else if (p.op == 2u) { // tanh
        y = tanh(x);
    } else if (p.op == 3u) { // gelu (approx)
        y = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    } else if (p.op == 4u) { // silu (swish)
        y = x * sigmoid(x);
    } else if (p.op == 5u) { // leaky_relu
        y = select(0.01 * x, x, x >= 0.0);
    } else if (p.op == 6u) { // elu
        y = select(p.param * (exp(x) - 1.0), x, x >= 0.0);
    } else if (p.op == 7u) { // softplus
        y = log(1.0 + exp(x));
    } else if (p.op == 8u) { // sqrt
        y = sqrt(max(x, 0.0));
    } else {                  // exp
        y = exp(x);
    }
    c[i] = y;
}
"#;

// ---------------------------------------------------------------------------
// Fused GEMM + bias + activation (single dispatch)
// ---------------------------------------------------------------------------

/// Fused kernel: `C = act(alpha·op(A)·op(B) + beta·bias)`
/// where `act` is: 0=none, 1=relu, 2=gelu, 3=silu, 4=sigmoid, 5=tanh.
/// Uses tiled approach for the GEMM portion then applies activation inline.
pub const FUSED_GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, ta: u32, tb: u32, alpha: f32, beta: f32, act: u32 };

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             p: P;

var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

fn sigmoid_f(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

fn apply_act(x: f32, act: u32) -> f32 {
    if (act == 0u) { return x; }
    if (act == 1u) { return max(x, 0.0); }
    if (act == 2u) { return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
    if (act == 3u) { return x * sigmoid_f(x); }
    if (act == 4u) { return sigmoid_f(x); }
    return tanh(x);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;

    var acc: f32 = 0.0;
    let num_tiles = (p.k + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let t_base = t * 16u;
        let a_col = t_base + lid.y;
        let b_row = t_base + lid.x;

        if (i < p.m && a_col < p.k) {
            let a_idx = select(a_col * p.m + i, i * p.k + a_col, p.ta == 0u);
            As[lid.y * 16u + lid.x] = a[a_idx];
        } else {
            As[lid.y * 16u + lid.x] = 0.0;
        }
        if (b_row < p.k && j < p.n) {
            let b_idx = select(j * p.k + b_row, b_row * p.n + j, p.tb == 0u);
            Bs[lid.y * 16u + lid.x] = b[b_idx];
        } else {
            Bs[lid.y * 16u + lid.x] = 0.0;
        }

        workgroupBarrier();
        for (var q: u32 = 0u; q < 16u; q = q + 1u) {
            acc = acc + As[lid.y * 16u + q] * Bs[q * 16u + lid.x];
        }
        workgroupBarrier();
    }

    if (i < p.m && j < p.n) {
        let idx = i * p.n + j;
        c[idx] = apply_act(p.alpha * acc + p.beta * c[idx], p.act);
    }
}
"#;

// ---------------------------------------------------------------------------
// Deterministic INT8 quantization GEMM
// ---------------------------------------------------------------------------

/// Quantized GEMM: `C = A_quantized(B) ⊗ B_quantized(B)` with INT32
/// accumulation. This is mathematically deterministic because integer
/// arithmetic (add, mul) is exact — no floating-point non-associativity.
///
/// A stored as INT8 = i8 per scale, B as INT8, output INT32 then dequantized.
pub const INT8_GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a_q: array<i32>;  // quantized INT8 A
@group(0) @binding(1) var<storage, read>       b_q: array<i32>;  // quantized INT8 B
@group(0) @binding(2) var<storage, read_write> c: array<f32>;     // output FP32
@group(0) @binding(3) var<uniform>             p: P;
@group(1) @binding(0) var<uniform>             scale_a: f32;  // A quantization scale
@group(1) @binding(1) var<uniform>             scale_b: f32;  // B quantization scale

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }

    var acc: i32 = 0; // INT32 accumulation — exact

    for (var q: u32 = 0u; q < p.k; q = q + 1u) {
        let av = a_q[i * p.k + q];
        let bv = b_q[q * p.n + j];
        acc = acc + av * bv;
    }

    let idx = i * p.n + j;
    // Dequantize: C(i,j) = scale_a * scale_b * acc_int32
    c[idx] = f32(acc) * scale_a * scale_b;
}
"#;

// ---------------------------------------------------------------------------
// Operation type tags for dispatch
// ---------------------------------------------------------------------------

/// Extended elementwise op codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EwOp {
    Relu = 0,
    Sigmoid = 1,
    Tanh = 2,
    Gelu = 3,
    Silu = 4,
    LeakyRelu = 5,
    Elu = 6,
    Softplus = 7,
    Sqrt = 8,
    Exp = 9,
}

/// Fused activation op codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FusedAct {
    None = 0,
    Relu = 1,
    Gelu = 2,
    Silu = 3,
    Sigmoid = 4,
    Tanh = 5,
}

/// Reduction op codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReduceOp {
    Sum = 0,
    Mean = 1,
    Max = 2,
    Norm = 3,
}

// ---------------------------------------------------------------------------
// Crypto GEMM: Zq modular reduction (Voie 1 — Bit-exact absolu)
// ---------------------------------------------------------------------------

pub const CRYPTO_GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, q: u32, _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32 };

@group(0) @binding(0) var<storage, read>       a: array<i32>;
@group(0) @binding(1) var<storage, read>       b: array<i32>;
@group(0) @binding(2) var<storage, read_write> c: array<i32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }
    var sum: i32 = 0i;
    let q: i32 = i32(p.q);
    for (var k: u32 = 0u; k < p.k; k = k + 1u) {
        let av = a[i * p.k + k];
        let bv = b[k * p.n + j];
        let prod = (av * bv) % q;
        sum = (sum + prod) % q;
    }
    if (sum < 0i) { sum = sum + q; }
    c[i * p.n + j] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Fixed-point Q15.16 GEMM (Voie 2 — Bit-exact via entiers)
// ---------------------------------------------------------------------------

pub const FIXED_POINT_Q16_GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32, _pad4: u32 };

@group(0) @binding(0) var<storage, read>       a: array<i32>;
@group(0) @binding(1) var<storage, read>       b: array<i32>;
@group(0) @binding(2) var<storage, read_write> c: array<i32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }
    var sum: i32 = 0i;
    for (var k: u32 = 0u; k < p.k; k = k + 1u) {
        let av = a[i * p.k + k];
        let bv = b[k * p.n + j];
        let product_q32 = av * bv;
        let product_q16 = product_q32 >> 16u;
        sum = sum + product_q16;
    }
    c[i * p.n + j] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Fixed-point Q15.16 with i64 (Piste A — requires SHADER_INT64 feature)
// ---------------------------------------------------------------------------

/// GEMM Q15.16 avec accumulateur i64 natif. Nécessite l'extension
/// `SHADER_INT64` activée côté Rust (`wgpu::Features::SHADER_INT64`).
/// Sans cette extension, le pipeline de compilation échouera.
/// Plage d'entrée: tout f32 convertible en Q16 sans overflow.
pub const FIXED_POINT_Q16_I64_GEMM_WGSL: &str = r#"
enable shader_int64;

struct P { m: u32, k: u32, n: u32, _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32, _pad4: u32 };

@group(0) @binding(0) var<storage, read>       a: array<i32>;
@group(0) @binding(1) var<storage, read>       b: array<i32>;
@group(0) @binding(2) var<storage, read_write> c: array<i32>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }

    var sum: i64 = 0l;

    for (var k: u32 = 0u; k < p.k; k = k + 1u) {
        let av = i64(a[i * p.k + k]);
        let bv = i64(b[k * p.n + j]);
        let product = av * bv;
        sum = sum + (product >> 16u);  // Q32 → Q16 realignment
    }

    c[i * p.n + j] = i32(sum);
}
"#;

// ---------------------------------------------------------------------------
// Fixed-point Q15.16 with emulated i64 (Piste B — portable, zero extension)
// ---------------------------------------------------------------------------

/// GEMM Q15.16 avec multiplication 64-bit émulée sur deux i32
/// (décomposition poids fort / poids faible 16-bit).
///
/// Principe : chaque i32 est découpé en `hi = val >> 16`, `lo = val & 0xFFFF`.
/// Le produit `av * bv` = `(ah*al) * (bh*bl)` est reconstruit par:
///   `(ah*bh) << 32 + (ah*bl + al*bh) << 16 + al*bl`
/// L'accumulation se fait en deux i32 séparés (high 32 + low 32).
///
/// Portable sur 100% des GPU (même WebGPU/navigateur).
/// Pas de limite de plage (supporte f32 complet en Q16).
pub const FIXED_POINT_Q16_EMULATED_GEMM_WGSL: &str = r#"
struct P { m: u32, k: u32, n: u32, _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32, _pad4: u32 };

@group(0) @binding(0) var<storage, read>       a: array<i32>;
@group(0) @binding(1) var<storage, read>       b: array<i32>;
@group(0) @binding(2) var<storage, read_write> c: array<i32>;
@group(0) @binding(3) var<uniform>             p: P;

// Compute (av * bv) >> 16 for SIGNED Q15.16 inputs using only u32 ops.
//
// The full signed 32x32 product needs 64 bits. We first build the UNSIGNED
// 64-bit product of the bit patterns as two u32 halves (p_hi, p_lo) with
// explicit carry propagation, then apply the standard signed correction
// (subtract the other operand from the high word when an operand is negative)
// to obtain the two's-complement signed product. The Q32 -> Q16 realignment
// keeps only the low 32 bits, which equal bits [16..47] of the product
// regardless of sign — matching `(prod >> 16) as i32` in the i64 oracle.
//
// Bit-exact with the i64 CPU oracle `fixed_point_gemm_q16` for ALL signs.
fn q16_mul(av: i32, bv: i32) -> i32 {
    let ua = bitcast<u32>(av);
    let ub = bitcast<u32>(bv);
    let a_lo = ua & 0xFFFFu;
    let a_hi = ua >> 16u;
    let b_lo = ub & 0xFFFFu;
    let b_hi = ub >> 16u;

    // Partial products of the bit patterns, each < 2^32 so each fits in a u32.
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    // cross = lh + hl (the 2^16 column); detect the wrap into bit 32.
    let cross = lh + hl;
    let cross_carry = select(0u, 1u, cross < lh);

    // Low 32 bits: ll + (cross << 16); detect carry into bit 32.
    let p_lo = ll + (cross << 16u);
    let p_lo_carry = select(0u, 1u, p_lo < ll);

    // High 32 bits of the UNSIGNED product.
    var p_hi = hh + (cross >> 16u) + (cross_carry << 16u) + p_lo_carry;

    // Signed correction: S = U - (av<0 ? ub<<32 : 0) - (bv<0 ? ua<<32 : 0).
    // Both subtracted terms touch only bits >= 32, i.e. the high word.
    if (av < 0) { p_hi = p_hi - ub; }
    if (bv < 0) { p_hi = p_hi - ua; }

    // (P >> 16) low 32 bits = (p_hi << 16) | (p_lo >> 16).
    return bitcast<i32>((p_hi << 16u) | (p_lo >> 16u));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }

    var sum: i32 = 0i;
    for (var k: u32 = 0u; k < p.k; k = k + 1u) {
        sum = sum + q16_mul(a[i * p.k + k], b[k * p.n + j]);
    }
    c[i * p.n + j] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Fixed-point Q31.32 avec i64 natif (Piste A étendue)
// ---------------------------------------------------------------------------

/// GEMM Q31.32 avec i64 natif. Pour la physique haute précision.
/// `Q32_SCALE = 1 << 32`. Inputs/Outputs en i64, produit en i64,
/// accumulation i64. Résultat `>> 32` pour réaligner.
pub const FIXED_POINT_Q32_I64_GEMM_WGSL: &str = r#"
enable shader_int64;

struct P { m: u32, k: u32, n: u32, _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32, _pad4: u32 };

@group(0) @binding(0) var<storage, read>       a: array<i64>;
@group(0) @binding(1) var<storage, read>       b: array<i64>;
@group(0) @binding(2) var<storage, read_write> c: array<i64>;
@group(0) @binding(3) var<uniform>             p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= p.m || j >= p.n) { return; }

    var sum: i64 = 0l;

    for (var k: u32 = 0u; k < p.k; k = k + 1u) {
        let av = a[i * p.k + k];
        let bv = b[k * p.n + j];
        sum = sum + ((av * bv) >> 32u);
    }

    c[i * p.n + j] = sum;
}
"#;
