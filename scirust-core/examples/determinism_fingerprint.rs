//! Determinism fingerprint — rayon vs no-rayon **bit-identity** differential.
//!
//! The crate documents that the rayon-parallel compute paths are bit-identical
//! to their serial fallbacks: `par_sgemm` only splits the output rows of `C`
//! (never reordering any k-accumulation), `batched_gemm` fans out whole
//! per-batch `sgemm`s (batch-parallel regime) or falls back to row-parallel
//! blocks (large-output regime), and the parallel im2col/col2im in
//! `nn::conv_utils` write disjoint chunks. This example turns that promise
//! into a checkable contract between the two *builds*.
//!
//! It runs a fixed, seeded workload chosen to force every parallel-sensitive
//! branch:
//!
//! * `matmul_512` — a 512×512×512 GEMM (2^27 fused ops, above `par_sgemm`'s
//!   `1 << 24` parallel gate) plus its two transposed backward GEMMs;
//! * `bmm2d_batchpar` — batch=4, 256×128×256, `transpose_b = true`: per-block
//!   output 256·256 ≤ 2^18 and work 2^23 ≥ 2^22, the batch fan-out branch;
//! * `bmm2d_rowpar` — batch=2, 1024×64×1024, `transpose_b = false`: per-block
//!   output 2^20 > 2^18, the row-parallel branch (each block 2^26 ops, so the
//!   inner `par_sgemm` splits rows too);
//! * `attention` — a causal 2-head MultiHeadAttention forward + backward
//!   (batch=2, seq=256, d_head=64: per-head score blocks sit exactly at the
//!   `1 << 22` batch fan-out floor);
//! * `elementwise` — a 512×512 chain of scale/tanh/sigmoid/relu/sin with
//!   shared operands;
//! * `conv2d` — batch=4, 8→16 channels, 32×32, k=3: the rayon im2col (forward)
//!   and col2im (input-gradient) paths.
//!
//! Every forward output **and** every gradient is folded into a per-stage
//! FNV-1a-64 fingerprint over the raw f32 bit patterns, then all stages into
//! one total. Any accumulation-order difference between the parallel and
//! serial paths changes at least one low-order mantissa bit and therefore the
//! printed hex.
//!
//! # CI usage
//!
//! Run the same example under both builds and diff the *entire* stdout:
//!
//! ```sh
//! cargo run -p scirust-core --release --example determinism_fingerprint \
//!     > fp_rayon.txt
//! cargo run -p scirust-core --release --no-default-features \
//!     --example determinism_fingerprint > fp_serial.txt
//! diff fp_rayon.txt fp_serial.txt   # must be empty — builds are bit-identical
//! ```
//!
//! The output is intentionally free of anything build- or machine-dependent
//! (no timings, no thread counts); if the diff is non-empty, the first
//! differing stage line localizes the divergent kernel.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::MultiHeadAttention;
use scirust_core::tensor::tensor3d::Var3D;

// ------------------------------------------------------------------ //
//  FNV-1a-64 over f32 bit patterns (same discipline as portable_f32) //
// ------------------------------------------------------------------ //

const fn fnv1a_init() -> u64 {
    0xcbf2_9ce4_8422_2325
}

const fn fnv1a_fold_bits(fp: u64, bits: u32) -> u64 {
    (fp ^ bits as u64).wrapping_mul(0x0000_0100_0000_01b3)
}

fn fold_slice(mut fp: u64, xs: &[f32]) -> u64 {
    for &x in xs
    {
        fp = fnv1a_fold_bits(fp, x.to_bits());
    }
    fp
}

fn rand_tensor(rng: &mut PcgEngine, rows: usize, cols: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.float() * 2.0 - 1.0).collect();
    Tensor::from_vec(data, rows, cols)
}

// ------------------------------------------------------------------ //
//  Stages                                                            //
// ------------------------------------------------------------------ //

/// 512×512×512 matmul: forward + the two transposed backward GEMMs, all above
/// the `1 << 24` ops gate, so the rayon build takes the row-parallel path.
fn stage_matmul_512() -> u64 {
    let mut rng = PcgEngine::new(0xC0FE);
    let tape = Tape::new();
    let a = tape.input(rand_tensor(&mut rng, 512, 512));
    let b = tape.input(rand_tensor(&mut rng, 512, 512));
    let c = a.matmul(b);
    // Non-uniform output weighting so the backward GEMMs see a non-trivial g.
    let w = tape.input(rand_tensor(&mut rng, 512, 512));
    let loss = c.hadamard(w).sum();
    loss.backward();
    let mut fp = fnv1a_init();
    fp = fold_slice(fp, &tape.value(c.idx()).data);
    fp = fold_slice(fp, &tape.grad(a.idx()).data);
    fp = fold_slice(fp, &tape.grad(b.idx()).data);
    fp
}

/// Shared body for the two `bmm2d` regimes: forward + backward, hashing the
/// output and both input gradients.
fn bmm2d_stage(seed: u64, batch: usize, m: usize, k: usize, n: usize, transpose_b: bool) -> u64 {
    let mut rng = PcgEngine::new(seed);
    let tape = Tape::new();
    let a = tape.input(rand_tensor(&mut rng, batch * m, k));
    let (br, bc) = if transpose_b
    {
        (batch * n, k)
    }
    else
    {
        (batch * k, n)
    };
    let b = tape.input(rand_tensor(&mut rng, br, bc));
    let out = a.bmm2d(b, batch, transpose_b);
    let w = tape.input(rand_tensor(&mut rng, batch * m, n));
    let loss = out.hadamard(w).sum();
    loss.backward();
    let mut fp = fnv1a_init();
    fp = fold_slice(fp, &tape.value(out.idx()).data);
    fp = fold_slice(fp, &tape.grad(a.idx()).data);
    fp = fold_slice(fp, &tape.grad(b.idx()).data);
    fp
}

/// Causal MultiHeadAttention forward + backward: slices, `bmm2d` (both in the
/// batch fan-out regime), causal mask, softmax, concat/transpose, and the
/// four Linear layers — hashing the output, the input gradient, and every
/// parameter gradient.
fn stage_attention() -> u64 {
    let (d_model, n_heads, batch, seq) = (128usize, 2usize, 2usize, 256usize);
    let mut rng = PcgEngine::new(0xA77E);
    let mut mha = MultiHeadAttention::new(d_model, n_heads, true, &KaimingNormal, &Zeros, &mut rng);

    let tape = Tape::new();
    let x = tape.input(rand_tensor(&mut rng, batch * seq, d_model));
    let out3 = mha.forward_3d(&tape, Var3D::from_var(x, batch, seq, d_model));
    let out = out3.as_var();
    let w = tape.input(rand_tensor(&mut rng, batch * seq, d_model));
    let loss = out.hadamard(w).sum();
    loss.backward();

    let mut fp = fnv1a_init();
    fp = fold_slice(fp, &tape.value(out.idx()).data);
    fp = fold_slice(fp, &tape.grad(x.idx()).data);
    for idx in mha.parameter_indices()
    {
        fp = fold_slice(fp, &tape.grad(idx).data);
    }
    fp
}

/// Elementwise chain with shared operands on a 512×512 tensor: no GEMM at
/// all — guards the pointwise kernels and the Add/Mul gradient accumulation.
fn stage_elementwise() -> u64 {
    let mut rng = PcgEngine::new(0xE1E3);
    let tape = Tape::new();
    let x = tape.input(rand_tensor(&mut rng, 512, 512));
    let y = x
        .scale(1.3)
        .tanh()
        .hadamard(x.sigmoid())
        .add(x.relu().scale(0.5))
        .sin();
    let w = tape.input(rand_tensor(&mut rng, 512, 512));
    let loss = y.hadamard(w).sum();
    loss.backward();
    let mut fp = fnv1a_init();
    fp = fold_slice(fp, &tape.value(y.idx()).data);
    fp = fold_slice(fp, &tape.grad(x.idx()).data);
    fp
}

/// Conv2d forward + backward (batch=4, 8→16 channels, 32×32, k=3, pad=1):
/// exercises the rayon-parallel im2col (forward) and col2im (input gradient)
/// in `nn::conv_utils`, plus the weight/bias gradients.
fn stage_conv2d() -> u64 {
    let (batch, in_c, h, w_dim, out_c, k) = (4usize, 8usize, 32usize, 32usize, 16usize, 3usize);
    let mut rng = PcgEngine::new(0xC04D);
    let tape = Tape::new();
    let x = tape.input(rand_tensor(&mut rng, batch, in_c * h * w_dim));
    let wt = tape.input(rand_tensor(&mut rng, out_c, in_c * k * k));
    let bias = tape.input(rand_tensor(&mut rng, 1, out_c));
    let out = x.conv2d_forward(wt, Some(bias), batch, in_c, h, w_dim, out_c, k, 1, 1);
    let w = tape.input(rand_tensor(&mut rng, batch, out_c * h * w_dim));
    let loss = out.hadamard(w).sum();
    loss.backward();
    let mut fp = fnv1a_init();
    fp = fold_slice(fp, &tape.value(out.idx()).data);
    fp = fold_slice(fp, &tape.grad(x.idx()).data);
    fp = fold_slice(fp, &tape.grad(wt.idx()).data);
    fp = fold_slice(fp, &tape.grad(bias.idx()).data);
    fp
}

/// A named stage: label plus the fingerprint function to run.
type Stage = (&'static str, fn() -> u64);

fn main() {
    let stages: [Stage; 6] = [
        ("matmul_512", stage_matmul_512),
        ("bmm2d_batchpar", || {
            bmm2d_stage(0xB4C4, 4, 256, 128, 256, true)
        }),
        ("bmm2d_rowpar", || {
            bmm2d_stage(0x2074, 2, 1024, 64, 1024, false)
        }),
        ("attention", stage_attention),
        ("elementwise", stage_elementwise),
        ("conv2d", stage_conv2d),
    ];

    let mut total = fnv1a_init();
    for (name, stage) in stages
    {
        let fp = stage();
        println!("stage {name:<16} 0x{fp:016x}");
        total = fnv1a_fold_bits(total, (fp >> 32) as u32);
        total = fnv1a_fold_bits(total, fp as u32);
    }
    println!("TOTAL {:<16} 0x{total:016x}", "");
}
