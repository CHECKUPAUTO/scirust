//! Benchmark d'un **bloc décodeur quantifié int8 (W8A8/AMX)** vs `f32`.
//!
//! ```text
//! cargo +nightly-2026-07-02 run -p scirust-simd --release \
//!   --features nightly-simd --example qtransformer_bench
//! ```
//!
//! Compare, pour un bloc décodeur `s×d` (attention multi-tête causale + FFN) :
//! * **débit** : `f32` (GEMM tuilé AVX-512) vs **int8** (projections AMX) ;
//! * **précision** : erreur relative RMS de la sortie quantifiée vs `f32` ;
//! * **mémoire** : octets des poids (int8 = ¼ de `f32`).
//!
//! Les non-linéarités (RMSNorm, softmax, SiLU) restent `f32` dans les deux cas.

#[cfg(target_arch = "x86_64")]
fn main() {
    use std::time::Instant;

    use scirust_simd::amx::amx_int8_usable;
    use scirust_simd::qtransformer::QuantizedTransformerBlock;
    use scirust_simd::transformer::TransformerBlock;

    let (s, d, h, dff) = (128usize, 1024usize, 16usize, 4096usize);
    println!("AMX int8 utilisable : {}", amx_int8_usable());
    println!("Bloc décodeur : s={s} d={d} h={h} d_ff={dff}\n");

    // Données décorrélées (LCG) — représentatives d'activations/poids réels,
    // contrairement à des `sin`/`cos` quasi-orthogonaux (produit scalaire ≈ 0,
    // qui gonfle artificiellement l'erreur relative de quantification).
    let mk = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.5
            })
            .collect()
    };
    let wq = mk(d * d, 1);
    let wk = mk(d * d, 2);
    let wv = mk(d * d, 3);
    let wo = mk(d * d, 4);
    let w1 = mk(d * dff, 5);
    let b1 = mk(dff, 6);
    let w2 = mk(dff * d, 7);
    let norm1: Vec<f32> = (0..d).map(|i| 1.0 + i as f32 * 1e-4).collect();
    let norm2: Vec<f32> = (0..d).map(|i| 0.9 + i as f32 * 1e-4).collect();
    let x0: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.001).cos()).collect();

    let block = TransformerBlock {
        d_model: d,
        n_heads: h,
        d_ff: dff,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wo: &wo,
        w1: &w1,
        b1: &b1,
        w2: &w2,
        norm1: &norm1,
        norm2: &norm2,
        eps: 1e-5,
        rope_base: 10000.0,
        causal: true,
    };
    let qblock = QuantizedTransformerBlock::from_f32(
        d, h, dff, &wq, &wk, &wv, &wo, &w1, &b1, &w2, &norm1, &norm2, 1e-5, 10000.0, true,
    );

    // Chauffe + mesure (moyenne sur quelques itérations).
    let iters = 20;
    let mut f32_out = x0.clone();
    let t = Instant::now();
    for _ in 0..iters
    {
        f32_out.copy_from_slice(&x0);
        block.forward(&mut f32_out, s);
    }
    let dt_f32 = t.elapsed().as_secs_f64() / iters as f64;

    let mut q_out = x0.clone();
    let t = Instant::now();
    for _ in 0..iters
    {
        q_out.copy_from_slice(&x0);
        qblock.forward(&mut q_out, s);
    }
    let dt_q = t.elapsed().as_secs_f64() / iters as f64;

    // Précision (RMS relatif).
    let mut num = 0f64;
    let mut den = 0f64;
    for i in 0..s * d
    {
        num += (q_out[i] - f32_out[i]).powi(2) as f64;
        den += (f32_out[i] as f64).powi(2);
    }
    let rel = (num / den).sqrt();

    // Mémoire des poids (6 matrices).
    let w_elems = 4 * d * d + 2 * d * dff;
    let bytes_f32 = w_elems * 4;
    let bytes_i8 = w_elems; // int8 = 1 octet

    println!("== Débit (temps par forward) ==");
    println!("  f32  (GEMM tuilé) : {:8.2} ms", dt_f32 * 1e3);
    println!(
        "  int8 (proj AMX)   : {:8.2} ms   ×{:.2}",
        dt_q * 1e3,
        dt_f32 / dt_q
    );
    println!("\n== Précision ==");
    println!("  erreur relative RMS : {:.4} ({:.2} %)", rel, rel * 100.0);
    println!("\n== Mémoire des poids ==");
    println!(
        "  f32 : {:.1} Mio   int8 : {:.1} Mio   (÷{:.1})",
        bytes_f32 as f64 / (1024.0 * 1024.0),
        bytes_i8 as f64 / (1024.0 * 1024.0),
        bytes_f32 as f64 / bytes_i8 as f64
    );
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("qtransformer_bench : cible non-x86_64 — AMX indisponible.");
}
