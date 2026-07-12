//! Cache KV **int8** vs `f32` : mémoire et fidélité sur un long contexte.
//!
//! ```text
//! cargo run -p scirust-simd --release --example qkv_bench
//! ```
//!
//! Décode une séquence entière token par token via les deux caches (le `f32`
//! [`KvCache`] et le quantifié [`QuantizedKvCache`]) et compare : mémoire `K`+`V`
//! occupée, et erreur relative RMS de la sortie int8 vs `f32`.

use scirust_simd::kv_cache::KvCache;
use scirust_simd::qkv_cache::QuantizedKvCache;

fn main() {
    let (s, h, dh) = (512usize, 16usize, 64usize);
    let dm = h * dh;
    let scale = 1.0 / (dh as f32).sqrt();
    println!("Contexte : s={s} tokens, d_model={dm} ({h} têtes × {dh})\n");

    // Données décorrélées (LCG).
    let mut seed = 0xC0FFEEu64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    };
    let q: Vec<f32> = (0..s * dm).map(|_| rnd()).collect();
    let k: Vec<f32> = (0..s * dm).map(|_| rnd()).collect();
    let v: Vec<f32> = (0..s * dm).map(|_| rnd()).collect();

    let mut f32_cache = KvCache::new(s, dm);
    let mut q8_cache = QuantizedKvCache::new(s, dm);
    let mut out_f32 = vec![0.0f32; s * dm];
    let mut out_q8 = vec![0.0f32; s * dm];

    for i in 0..s
    {
        let kr = &k[i * dm..i * dm + dm];
        let vr = &v[i * dm..i * dm + dm];
        let qr = &q[i * dm..i * dm + dm];
        f32_cache.append(kr, vr);
        q8_cache.append(kr, vr);
        let mut of = vec![0.0f32; dm];
        let mut oq = vec![0.0f32; dm];
        f32_cache.decode_step(qr, h, dh, scale, &mut of);
        q8_cache.decode_step(qr, h, dh, scale, &mut oq);
        out_f32[i * dm..i * dm + dm].copy_from_slice(&of);
        out_q8[i * dm..i * dm + dm].copy_from_slice(&oq);
    }

    let mut num = 0f64;
    let mut den = 0f64;
    for i in 0..s * dm
    {
        num += (out_q8[i] - out_f32[i]).powi(2) as f64;
        den += (out_f32[i] as f64).powi(2);
    }
    let rel = (num / den).sqrt();

    let f32_bytes = 2 * s * dm * 4;
    println!("== Mémoire K+V ==");
    println!(
        "  f32  : {:.2} Mio   int8 : {:.2} Mio   (÷{:.1})",
        f32_bytes as f64 / (1024.0 * 1024.0),
        q8_cache.kv_bytes() as f64 / (1024.0 * 1024.0),
        f32_bytes as f64 / q8_cache.kv_bytes() as f64
    );
    println!("\n== Fidélité (int8 vs f32) ==");
    println!("  erreur relative RMS : {:.4} ({:.2} %)", rel, rel * 100.0);
}
