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

use scirust_gpu::ops::{
    MASK_NEG, cpu_cross_entropy, cpu_cross_entropy_grad, cpu_embed, cpu_embed_backward,
    cpu_rms_norm, cpu_rms_norm_backward, cpu_scale_causal_mask, cpu_scale_causal_mask_backward,
    cpu_softmax, cpu_softmax_backward, cpu_swiglu_backward, rel_err,
};
use scirust_gpu::{
    BlockWeights, CpuBackend, GpuChain, ModelWeights, RawComputeBackend, WgpuContext,
};

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
    // One CPU transformer block (reused for the stack check below).
    let cpu_block_once = |input: &[f32]| -> Vec<f32> {
        let xn = cpu_rms_norm(input, &nw, eps, t, d);
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
        let hblk: Vec<f32> = input.iter().zip(&ao).map(|(a, b)| a + b).collect();
        let hn = cpu_rms_norm(&hblk, &n2, eps, t, d);
        let g2 = gemm(&hn, &wg, t, d, hh);
        let u2 = gemm(&hn, &wu, t, d, hh);
        let act2: Vec<f32> = g2
            .iter()
            .zip(&u2)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let m2 = gemm(&act2, &wd, t, hh, d);
        hblk.iter().zip(&m2).map(|(a, b)| a + b).collect()
    };
    let blk_cpu = cpu_block_once(&nx);
    check(
        "resident transformer block (attn+MLP+residuals)",
        rel_err(&blk_gpu, &blk_cpu),
        &mut failures,
    );

    // 8. A 2-block STACK run resident — block 1's output feeds block 2 in VRAM
    //    (same weights both layers). The resident trunk of the 350M forward.
    let stack = [
        BlockWeights {
            norm1: &gnw,
            wq: &gbq,
            wk: &gbk,
            wv: &gbv,
            wo: &gbo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
        },
        BlockWeights {
            norm1: &gnw,
            wq: &gbq,
            wk: &gbk,
            wv: &gbv,
            wo: &gbo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
        },
    ];
    let stack_gpu = chain
        .download(&chain.transformer_stack(&gnx, &stack, eps, true).unwrap())
        .unwrap();
    let stack_cpu = cpu_block_once(&cpu_block_once(&nx));
    check(
        "resident 2-block stack (trunk of the 350M)",
        rel_err(&stack_gpu, &stack_cpu),
        &mut failures,
    );

    // 9. The FULL model forward, tokens → logits, fully resident: embed gather →
    //    2 transformer blocks → final RMSNorm → tied LM head (h·Eᵀ). A whole
    //    350M forward pass in miniature, on the GPU from token ids to logits.
    let vocab = 20usize;
    let etab: Vec<f32> = (0..vocab * d)
        .map(|i| (i as f32 * 0.05 - 1.0).sin())
        .collect();
    let fnorm: Vec<f32> = (0..d).map(|i| 0.8 + 0.01 * i as f32).collect();
    let toks: Vec<u32> = (0..t as u32).map(|i| (i * 7 + 3) % vocab as u32).collect();
    let getab = chain.upload(&etab, vocab, d);
    let gfn = chain.upload(&fnorm, 1, d);
    let mstack = [
        BlockWeights {
            norm1: &gnw,
            wq: &gbq,
            wk: &gbk,
            wv: &gbv,
            wo: &gbo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
        },
        BlockWeights {
            norm1: &gnw,
            wq: &gbq,
            wk: &gbk,
            wv: &gbv,
            wo: &gbo,
            norm2: &gn2,
            wg: &gwg,
            wu: &gwu,
            wd: &gwd,
        },
    ];
    let mw = ModelWeights {
        embedding: &getab,
        blocks: &mstack,
        final_norm: &gfn,
    };
    let logits_gpu = chain
        .download(&chain.model_forward_tied(&toks, &mw, eps, true).unwrap())
        .unwrap();
    // CPU oracle: embed → 2 blocks → final norm → logits = h·Eᵀ.
    let emb = cpu_embed(&toks, &etab, d, vocab);
    let trunk = cpu_block_once(&cpu_block_once(&emb));
    let normed = cpu_rms_norm(&trunk, &fnorm, eps, t, d);
    let mut et = vec![0.0f32; d * vocab];
    for r in 0..vocab
    {
        for c in 0..d
        {
            et[c * vocab + r] = etab[r * d + c];
        }
    }
    let logits_cpu = CpuBackend.gemm_f32(&normed, &et, t, d, vocab).unwrap();
    check(
        "resident model forward (tokens→logits, tied)",
        rel_err(&logits_gpu, &logits_cpu),
        &mut failures,
    );

    // 10. BACKWARD: the GEMM vjp on-device — for C = A·B, grad_a = grad_c·Bᵀ and
    //     grad_b = Aᵀ·grad_c. The adjoint the whole backward pass builds on.
    let (bm, bk, bn) = (3usize, 5usize, 4usize);
    let ba: Vec<f32> = (0..bm * bk)
        .map(|i| (i as f32 * 0.21 - 0.5).sin())
        .collect();
    let bb: Vec<f32> = (0..bk * bn)
        .map(|i| (i as f32 * 0.17 + 0.3).cos())
        .collect();
    let bgc: Vec<f32> = (0..bm * bn)
        .map(|i| (i as f32 * 0.31 - 1.0).sin())
        .collect(); // dL/dC
    let (gda, gdb) = chain
        .matmul_backward(
            &chain.upload(&ba, bm, bk),
            &chain.upload(&bb, bk, bn),
            &chain.upload(&bgc, bm, bn),
        )
        .unwrap();
    let grad_a_gpu = chain.download(&gda).unwrap();
    let grad_b_gpu = chain.download(&gdb).unwrap();
    // CPU analytic: grad_a = grad_c · Bᵀ, grad_b = Aᵀ · grad_c.
    let mut bbt = vec![0.0f32; bn * bk]; // Bᵀ (n×k)
    for r in 0..bk
    {
        for c in 0..bn
        {
            bbt[c * bk + r] = bb[r * bn + c];
        }
    }
    let grad_a_cpu = CpuBackend.gemm_f32(&bgc, &bbt, bm, bn, bk).unwrap();
    let mut bat = vec![0.0f32; bk * bm]; // Aᵀ (k×m)
    for r in 0..bm
    {
        for c in 0..bk
        {
            bat[c * bm + r] = ba[r * bk + c];
        }
    }
    let grad_b_cpu = CpuBackend.gemm_f32(&bat, &bgc, bk, bm, bn).unwrap();
    let back_err = rel_err(&grad_a_gpu, &grad_a_cpu).max(rel_err(&grad_b_gpu, &grad_b_cpu));
    check(
        "backward: matmul vjp (grad_a=G·Bᵀ, grad_b=Aᵀ·G)",
        back_err,
        &mut failures,
    );

    // 11. BACKWARD: softmax adjoint dx = y ⊙ (dy − Σ dy·y), on-device.
    let (sr, sc) = (4usize, 6usize);
    let sx: Vec<f32> = (0..sr * sc)
        .map(|i| (i as f32 * 0.3 - 1.0).sin() * 2.0)
        .collect();
    let sy = cpu_softmax(&sx, sr, sc);
    let sdy: Vec<f32> = (0..sr * sc).map(|i| (i as f32 * 0.5 + 0.2).cos()).collect();
    let sdx_gpu = chain
        .download(
            &chain
                .softmax_backward(&chain.upload(&sy, sr, sc), &chain.upload(&sdy, sr, sc))
                .unwrap(),
        )
        .unwrap();
    check(
        "backward: softmax vjp (y⊙(dy−Σdy·y))",
        rel_err(&sdx_gpu, &cpu_softmax_backward(&sy, &sdy, sr, sc)),
        &mut failures,
    );

    // 12. BACKWARD: SwiGLU adjoint — da = dc·silu'(a)·b, db = dc·silu(a).
    let sn = 14usize;
    let sa: Vec<f32> = (0..sn)
        .map(|i| (i as f32 * 0.4 - 1.5).sin() * 2.0)
        .collect();
    let sb: Vec<f32> = (0..sn).map(|i| (i as f32 * 0.3 + 0.5).cos()).collect();
    let sdc: Vec<f32> = (0..sn).map(|i| (i as f32 * 0.6 - 0.3).sin()).collect();
    let (gda2, gdb2) = chain
        .swiglu_backward(
            &chain.upload(&sa, 1, sn),
            &chain.upload(&sb, 1, sn),
            &chain.upload(&sdc, 1, sn),
        )
        .unwrap();
    let (da_cpu, db_cpu) = cpu_swiglu_backward(&sa, &sb, &sdc);
    let sg_err = rel_err(&chain.download(&gda2).unwrap(), &da_cpu)
        .max(rel_err(&chain.download(&gdb2).unwrap(), &db_cpu));
    check("backward: swiglu vjp (da, db)", sg_err, &mut failures);

    // 13. BACKWARD: RMSNorm input gradient (the mean-coupling jacobian).
    let (rr, rc) = (4usize, 6usize);
    let rx: Vec<f32> = (0..rr * rc)
        .map(|i| (i as f32 * 0.3 - 1.0).sin() * 2.0)
        .collect();
    let rwt: Vec<f32> = (0..rc).map(|i| 0.6 + 0.1 * i as f32).collect();
    let rdy: Vec<f32> = (0..rr * rc).map(|i| (i as f32 * 0.5 + 0.2).cos()).collect();
    let rdx_gpu = chain
        .download(
            &chain
                .rms_norm_backward(
                    &chain.upload(&rx, rr, rc),
                    &chain.upload(&rwt, 1, rc),
                    &chain.upload(&rdy, rr, rc),
                    eps,
                )
                .unwrap(),
        )
        .unwrap();
    check(
        "backward: rmsnorm vjp (dx, mean-coupled)",
        rel_err(
            &rdx_gpu,
            &cpu_rms_norm_backward(&rx, &rwt, &rdy, eps, rr, rc),
        ),
        &mut failures,
    );

    // 14. BACKWARD: scale + causal mask — scale below/on diagonal, 0 above.
    let mn = 6usize;
    let mdout: Vec<f32> = (0..mn * mn).map(|i| (i as f32 * 0.2 - 1.0).sin()).collect();
    let mdin_gpu = chain
        .download(
            &chain
                .scale_causal_mask_backward(&chain.upload(&mdout, mn, mn), 0.125, true)
                .unwrap(),
        )
        .unwrap();
    check(
        "backward: scale+mask vjp (0 above diagonal)",
        rel_err(
            &mdin_gpu,
            &cpu_scale_causal_mask_backward(&mdout, mn, mn, 0.125, true),
        ),
        &mut failures,
    );

    // 15. BACKWARD: embedding scatter-sum — dE[v] = Σ over tokens==v of dOut.
    let evocab = 9usize;
    let etoks: Vec<u32> = vec![0, 4, 8, 4, 1, 4]; // token 4 repeats → accumulation
    let et_len = etoks.len();
    let edout: Vec<f32> = (0..et_len * d)
        .map(|i| (i as f32 * 0.3 - 0.5).sin())
        .collect();
    let edt_gpu = chain
        .download(
            &chain
                .embed_backward(&etoks, &chain.upload(&edout, et_len, d), evocab)
                .unwrap(),
        )
        .unwrap();
    check(
        "backward: embedding scatter-sum (dE[v])",
        rel_err(&edt_gpu, &cpu_embed_backward(&etoks, &edout, d, evocab)),
        &mut failures,
    );

    // 16. BACKWARD (integration): dx through the WHOLE transformer block, checked
    //     against central finite differences of L = Σ block(x)⊙G (via the CPU
    //     block closure). Absolute tolerance since finite differences carry ~1e-3
    //     truncation error. This proves every adjoint composes correctly.
    let bwd_bw = BlockWeights {
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
    let bg: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.23 - 0.4).sin()).collect(); // dL/dout
    let (_out_c, bcache) = chain
        .transformer_block_forward_cached(&gnx, &bwd_bw, eps, true)
        .unwrap();
    let bdx_gpu = chain
        .download(
            &chain
                .transformer_block_backward(
                    &gnx,
                    &bwd_bw,
                    &bcache,
                    &chain.upload(&bg, t, d),
                    eps,
                    true,
                )
                .unwrap(),
        )
        .unwrap();
    let bloss =
        |xx: &[f32]| -> f32 { cpu_block_once(xx).iter().zip(&bg).map(|(a, b)| a * b).sum() };
    let hstep = 1e-3f32;
    let mut max_fd_err = 0.0f32;
    for idx in 0..t * d
    {
        let (mut xp, mut xm) = (nx.clone(), nx.clone());
        xp[idx] += hstep;
        xm[idx] -= hstep;
        let fd = (bloss(&xp) - bloss(&xm)) / (2.0 * hstep);
        max_fd_err = max_fd_err.max((fd - bdx_gpu[idx]).abs());
    }
    let ok = max_fd_err < 3e-2;
    println!(
        "  {:<34} max|fd−gpu| = {:>10.3e}   {}",
        "backward: full block dx (finite-diff)",
        max_fd_err,
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok
    {
        failures += 1;
    }

    // 17. LOSS: cross-entropy gradient dlogits = (softmax − onehot)/t — the seed
    //     of the whole training backward.
    let (lrows, lvocab) = (5usize, 11usize);
    let logits: Vec<f32> = (0..lrows * lvocab)
        .map(|i| (i as f32 * 0.3 - 1.0).sin() * 2.0)
        .collect();
    let ltargets: Vec<u32> = (0..lrows as u32)
        .map(|i| (i * 3 + 2) % lvocab as u32)
        .collect();
    let dl_gpu = chain
        .download(
            &chain
                .cross_entropy_grad(&chain.upload(&logits, lrows, lvocab), &ltargets)
                .unwrap(),
        )
        .unwrap();
    check(
        "loss: cross-entropy grad (softmax−onehot)/t",
        rel_err(
            &dl_gpu,
            &cpu_cross_entropy_grad(&logits, &ltargets, lrows, lvocab),
        ),
        &mut failures,
    );

    // 18. CAPSTONE — a real on-device training loop reduces the loss. Linear
    //     model logits = x·W with cross-entropy targets; each step runs
    //     xent_grad → matmul_backward (dW) → sgd_step(W), entirely on the GPU.
    let (tt, td, tv) = (6usize, 5usize, 8usize);
    let tx: Vec<f32> = (0..tt * td)
        .map(|i| (i as f32 * 0.21 - 0.7).sin())
        .collect();
    let tw0: Vec<f32> = (0..td * tv)
        .map(|i| (i as f32 * 0.13 + 0.2).cos() * 0.3)
        .collect();
    let ttargets: Vec<u32> = (0..tt as u32).map(|i| (i * 5 + 1) % tv as u32).collect();
    let tgx = chain.upload(&tx, tt, td);
    let mut tgw = chain.upload(&tw0, td, tv);
    let (mut loss0, mut lossf) = (0.0f32, 0.0f32);
    for step in 0..12
    {
        let logits = chain.matmul(&tgx, &tgw).unwrap();
        let l = cpu_cross_entropy(&chain.download(&logits).unwrap(), &ttargets, tt, tv);
        if step == 0
        {
            loss0 = l;
        }
        lossf = l;
        let dl = chain.cross_entropy_grad(&logits, &ttargets).unwrap();
        let (_dx, dw) = chain.matmul_backward(&tgx, &tgw, &dl).unwrap();
        tgw = chain.sgd_step(&tgw, &dw, 0.5).unwrap();
    }
    let trained = lossf < loss0 * 0.7;
    println!(
        "  {:<34} loss {:.4} → {:.4}   {}",
        "TRAIN LOOP (xent→grad→sgd on GPU)",
        loss0,
        lossf,
        if trained { "PASS" } else { "FAIL" }
    );
    if !trained
    {
        failures += 1;
    }

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
