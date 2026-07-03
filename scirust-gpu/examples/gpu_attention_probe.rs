//! **Validate the GPU attention primitives on the real device.** The parity
//! tests in `scirust-gpu` *skip* their GPU assertions where no Vulkan adapter is
//! present (CI dev containers, this repo's sandbox). On a machine with a real
//! adapter — a Jetson Thor's Blackwell GPU, a desktop dGPU, or even Mesa
//! *lavapipe* — this example runs the kernels for real and checks each against
//! the deterministic CPU oracle, printing the adapter name so you can confirm
//! *which* device answered.
//!
//! It exercises the two attention-scoring bricks and then composes them:
//!   1. `softmax_rows`        vs `ops::cpu_softmax`
//!   2. `scale_causal_mask`   vs `ops::cpu_scale_causal_mask`
//!   3. the full masked-attention score path on the GPU — `S = Q·Kᵀ` (GEMM),
//!      then scale + causal mask, then row softmax — vs the same chain computed
//!      on the CPU oracle.
//!
//! Run on the device:
//!   cargo run -p scirust-gpu --features wgpu --release --example gpu_attention_probe
//!
//! Exit status is non-zero if any parity check fails, so it doubles as a
//! device smoke test in a script.

use scirust_gpu::ops::{MASK_NEG, cpu_rms_norm, cpu_scale_causal_mask, cpu_softmax, rel_err};
use scirust_gpu::{BlockWeights, CpuBackend, GpuChain, RawComputeBackend, WgpuContext};

/// Parity tolerance: GPU accumulation order is not bit-identical to the scalar
/// CPU oracle, so we assert a relative Frobenius bound rather than equality.
const TOL: f32 = 1e-4;

fn check(name: &str, err: f32, failures: &mut usize) {
    let ok = err < TOL;
    println!(
        "  {:<34} rel_err = {:>10.3e}   {}",
        name,
        err,
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok
    {
        *failures += 1;
    }
}

/// CPU reference for `Q·Kᵀ`: transpose K (`t×d` → `d×t`) then GEMM `Q·Kᵀ`.
fn cpu_scores(q: &[f32], k: &[f32], t: usize, d: usize) -> Vec<f32> {
    let mut kt = vec![0.0f32; d * t];
    for r in 0..t
    {
        for c in 0..d
        {
            kt[c * t + r] = k[r * d + c];
        }
    }
    CpuBackend.gemm_f32(q, &kt, t, d, t).unwrap()
}

fn main() {
    let ctx = match WgpuContext::new()
    {
        Ok(c) => c,
        Err(e) =>
        {
            eprintln!("no GPU adapter available ({e}). On a headless box, install a");
            eprintln!("Vulkan ICD (Mesa lavapipe works) or run on the Jetson Thor.");
            std::process::exit(2);
        },
    };
    println!("GPU adapter: {}\n", ctx.adapter_name());

    let mut failures = 0usize;

    // 1. Row-wise softmax on a 4×7 matrix with a wide dynamic range.
    let (rows, cols) = (4usize, 7usize);
    let sm_in: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.37 - 3.0).sin() * 4.0)
        .collect();
    let sm_gpu = ctx.softmax_rows(&sm_in, rows, cols).unwrap();
    let sm_cpu = cpu_softmax(&sm_in, rows, cols);
    check("softmax_rows", rel_err(&sm_gpu, &sm_cpu), &mut failures);
    let worst_row_sum = (0..rows)
        .map(|r| (sm_gpu[r * cols..r * cols + cols].iter().sum::<f32>() - 1.0).abs())
        .fold(0.0f32, f32::max);
    println!("    (worst |row_sum − 1| = {worst_row_sum:.3e})");

    // 2. Scale + causal mask on a 6×6 score matrix (scale = 1/√64).
    let n = 6usize;
    let scale = 1.0 / 64.0_f32.sqrt();
    let raw: Vec<f32> = (0..n * n)
        .map(|i| (i as f32 * 0.19 - 2.5).cos() * 3.0)
        .collect();
    let mask_gpu = ctx.scale_causal_mask(&raw, n, n, scale, true).unwrap();
    let mask_cpu = cpu_scale_causal_mask(&raw, n, n, scale, true);
    check(
        "scale_causal_mask",
        rel_err(&mask_gpu, &mask_cpu),
        &mut failures,
    );
    let sentinel_ok = (0..n).all(|i| (i + 1..n).all(|j| mask_gpu[i * n + j] == MASK_NEG));
    println!(
        "    (above-diagonal sentinel exact on GPU: {})",
        if sentinel_ok { "yes" } else { "NO" }
    );
    if !sentinel_ok
    {
        failures += 1;
    }

    // 3. Full masked-attention score path on the GPU, composed from the bricks:
    //    S = Q·Kᵀ  →  scale + causal mask  →  row softmax.
    let (t, d) = (8usize, 16usize);
    let q: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.11 - 1.0).sin()).collect();
    let k: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.07 + 0.5).cos()).collect();
    let att_scale = 1.0 / (d as f32).sqrt();

    // GPU chain.
    let mut s_gpu = vec![0.0f32; t * t];
    ctx.gemm(1.0, &q, &k, 0.0, &mut s_gpu, t, d, t, false, true)
        .unwrap(); // op(B)=Kᵀ (tb = true) ⇒ S = Q·Kᵀ
    let s_gpu = ctx
        .scale_causal_mask(&s_gpu, t, t, att_scale, true)
        .unwrap();
    let att_gpu = ctx.softmax_rows(&s_gpu, t, t).unwrap();

    // CPU oracle for the same chain.
    let s_cpu = cpu_scores(&q, &k, t, d);
    let s_cpu = cpu_scale_causal_mask(&s_cpu, t, t, att_scale, true);
    let att_cpu = cpu_softmax(&s_cpu, t, t);
    check(
        "attention scores (Q·Kᵀ→mask→softmax)",
        rel_err(&att_gpu, &att_cpu),
        &mut failures,
    );
    // Causality: query i must place zero weight on every future key j > i.
    let leak = (0..t)
        .map(|i| {
            (i + 1..t)
                .map(|j| att_gpu[i * t + j])
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max);
    println!("    (max attention weight leaked to a future key = {leak:.3e})");

    // 4. The SAME attention, but as a FULLY RESIDENT block: with GpuChain the
    //    t×t scores never leave VRAM between the score GEMM, scale/mask, softmax
    //    and value GEMM — only the final t×dv context is downloaded. This is the
    //    on-device attention block the SML forward will call.
    let dv = 5usize;
    let vv: Vec<f32> = (0..t * dv).map(|i| (i as f32 * 0.17 - 0.6).sin()).collect();
    let chain = GpuChain::new().expect("adapter was available a moment ago");
    let gq = chain.upload(&q, t, d);
    let gk = chain.upload(&k, t, d);
    let gv = chain.upload(&vv, t, dv);
    let ctx_gpu = chain
        .download(&chain.attention(&gq, &gk, &gv, true).unwrap())
        .unwrap();
    // CPU oracle for the context: (attention weights) · V.
    let ctx_cpu = CpuBackend.gemm_f32(&att_cpu, &vv, t, t, dv).unwrap();
    check(
        "resident attention block (·V, scores stay in VRAM)",
        rel_err(&ctx_gpu, &ctx_cpu),
        &mut failures,
    );

    // 5. Resident RMSNorm (the pre-norm of a transformer block).
    let eps = 1e-5f32;
    let nx: Vec<f32> = (0..t * d)
        .map(|i| (i as f32 * 0.13 - 0.8).cos() * 2.0)
        .collect();
    let nw: Vec<f32> = (0..d).map(|i| 0.6 + 0.05 * i as f32).collect();
    let gnx = chain.upload(&nx, t, d);
    let gnw = chain.upload(&nw, 1, d);
    let rms_gpu = chain
        .download(&chain.rms_norm(&gnx, &gnw, eps).unwrap())
        .unwrap();
    check(
        "resident RMSNorm",
        rel_err(&rms_gpu, &cpu_rms_norm(&nx, &nw, eps, t, d)),
        &mut failures,
    );

    // 6. The SwiGLU MLP block, fully resident: (silu(x·Wg) ⊙ (x·Wu))·Wd, every
    //    t×h intermediate kept in VRAM — the MLP half of a transformer layer.
    let hh = 12usize;
    let wg: Vec<f32> = (0..d * hh)
        .map(|i| (i as f32 * 0.07 + 0.2).cos() * 0.5)
        .collect();
    let wu: Vec<f32> = (0..d * hh)
        .map(|i| (i as f32 * 0.05 - 0.4).sin() * 0.5)
        .collect();
    let wd: Vec<f32> = (0..hh * d)
        .map(|i| (i as f32 * 0.09 + 0.1).cos() * 0.5)
        .collect();
    let gwg = chain.upload(&wg, d, hh);
    let gwu = chain.upload(&wu, d, hh);
    let gwd = chain.upload(&wd, hh, d);
    let mlp_gpu = chain
        .download(&chain.swiglu_mlp(&gnx, &gwg, &gwu, &gwd).unwrap())
        .unwrap();
    // CPU oracle: gate = x·Wg, up = x·Wu, act = silu(gate)⊙up, out = act·Wd.
    let gate = CpuBackend.gemm_f32(&nx, &wg, t, d, hh).unwrap();
    let up = CpuBackend.gemm_f32(&nx, &wu, t, d, hh).unwrap();
    let act: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let mlp_cpu = CpuBackend.gemm_f32(&act, &wd, t, hh, d).unwrap();
    check(
        "resident SwiGLU MLP (silu(x·Wg)⊙(x·Wu)·Wd)",
        rel_err(&mlp_gpu, &mlp_cpu),
        &mut failures,
    );

    // 7. A COMPLETE residual transformer block in ONE resident call — the whole
    //    350M layer forward: h = x + attn(rms_norm(x)·Wq,·Wk,·Wv)·Wo ; then
    //    out = h + swiglu_mlp(rms_norm(h)). Reuses the MLP weights from check 6.
    let mk = |phase: f32| -> Vec<f32> {
        (0..d * d)
            .map(|i| (i as f32 * 0.03 + phase).sin() * 0.3)
            .collect()
    };
    let (bq, bk, bv, bo) = (mk(0.5), mk(1.1), mk(1.7), mk(2.3));
    let n2: Vec<f32> = (0..d).map(|i| 0.9 - 0.01 * i as f32).collect();
    let (gbq, gbk, gbv, gbo) = (
        chain.upload(&bq, d, d),
        chain.upload(&bk, d, d),
        chain.upload(&bv, d, d),
        chain.upload(&bo, d, d),
    );
    let gn2 = chain.upload(&n2, 1, d);
    let bw = BlockWeights {
        norm1: &gnw,
        wq: &gbq,
        wk: &gbk,
        wv: &gbv,
        wo: &gbo,
        norm2: &gn2,
        wg: &gwg,
        wu: &gwu,
        wd: &gwd,
    };
    let blk_gpu = chain
        .download(&chain.transformer_block(&gnx, &bw, eps, true).unwrap())
        .unwrap();
    // CPU oracle for the full block.
    let gemm = |a: &[f32], b: &[f32], m, k, n| CpuBackend.gemm_f32(a, b, m, k, n).unwrap();
    let transpose = |a: &[f32], r: usize, c: usize| {
        let mut o = vec![0.0f32; r * c];
        for i in 0..r
        {
            for j in 0..c
            {
                o[j * r + i] = a[i * c + j];
            }
        }
        o
    };
    let xn = cpu_rms_norm(&nx, &nw, eps, t, d);
    let q = gemm(&xn, &bq, t, d, d);
    let kk = gemm(&xn, &bk, t, d, d);
    let vv2 = gemm(&xn, &bv, t, d, d);
    let s = cpu_scale_causal_mask(
        &gemm(&q, &transpose(&kk, t, d), t, d, t),
        t,
        t,
        1.0 / (d as f32).sqrt(),
        true,
    );
    let a = gemm(&cpu_softmax(&s, t, t), &vv2, t, t, d);
    let ao = gemm(&a, &bo, t, d, d);
    let hblk: Vec<f32> = nx.iter().zip(&ao).map(|(a, b)| a + b).collect();
    let hn = cpu_rms_norm(&hblk, &n2, eps, t, d);
    let g2 = gemm(&hn, &wg, t, d, hh);
    let u2 = gemm(&hn, &wu, t, d, hh);
    let act2: Vec<f32> = g2
        .iter()
        .zip(&u2)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let m2 = gemm(&act2, &wd, t, hh, d);
    let blk_cpu: Vec<f32> = hblk.iter().zip(&m2).map(|(a, b)| a + b).collect();
    check(
        "resident transformer block (attn+MLP+residuals)",
        rel_err(&blk_gpu, &blk_cpu),
        &mut failures,
    );

    println!();
    if failures == 0
    {
        println!("All GPU attention primitives match the CPU oracle on this device.");
    }
    else
    {
        eprintln!("{failures} parity check(s) FAILED on this device.");
        std::process::exit(1);
    }
}
