//! Benchmark des accélérateurs matriciels **Intel AMX** sur le silicium courant.
//!
//! Exécuter en release :
//! ```text
//! cargo +nightly-2026-07-02 run -p scirust-simd --release \
//!   --features nightly-simd --example amx_bench
//! ```
//!
//! Compare, pour un GEMM carré `n³` :
//! * **int8** (`A·B → i32`) : AMX `_tile_dpbssd` vs référence scalaire ;
//! * **bf16** (`A·B → f32`) : AMX `_tile_dpbf16ps` vs référence scalaire.
//!
//! Affiche temps, débit (GOP/s int8 = 2·n³ MAC, GFLOP/s bf16) et accélération.
//! Aucun `assert` de perf (dépend du CPU) — les résultats sont vérifiés
//! numériquement contre la référence scalaire. Sur une puce **sans** AMX, les
//! deux chemins retombent sur le scalaire (accélération ≈ 1) : le binaire reste
//! correct et unique.

#[cfg(target_arch = "x86_64")]
fn main() {
    use std::time::Instant;

    use scirust_simd::amx::{
        amx_bf16_usable, amx_int8_usable, amx_matmul_bf16, amx_matmul_i8, matmul_bf16_scalar,
        matmul_i8_scalar,
    };
    use scirust_simd::quant::f32_to_bf16;

    println!("AMX int8 utilisable : {}", amx_int8_usable());
    println!("AMX bf16 utilisable : {}\n", amx_bf16_usable());

    let n = 512usize;
    let ops = 2.0 * (n as f64).powi(3); // 2·n³ MAC (mul+add)

    // ---- int8 ----
    println!("== GEMM int8 {n}³ ==");
    let a8: Vec<i8> = (0..n * n)
        .map(|t| ((t as i32 * 7 - 61) % 128) as i8)
        .collect();
    let b8: Vec<i8> = (0..n * n)
        .map(|t| ((t as i32 * -5 + 23) % 128) as i8)
        .collect();

    let t = Instant::now();
    let want8 = matmul_i8_scalar(&a8, &b8, n, n, n);
    let dt_s8 = t.elapsed().as_secs_f64();
    report("scalaire", dt_s8, ops, dt_s8);

    let t = Instant::now();
    let got8 = amx_matmul_i8(&a8, &b8, n, n, n);
    let dt_a8 = t.elapsed().as_secs_f64();
    report("AMX     ", dt_a8, ops, dt_s8);
    println!(
        "  cohérence : {}\n",
        if got8 == want8 { "OK" } else { "ÉCART" }
    );

    // ---- bf16 ----
    println!("== GEMM bf16 {n}³ ==");
    let af: Vec<u16> = (0..n * n)
        .map(|t| f32_to_bf16((t as f32 * 0.017).sin() * 0.5))
        .collect();
    let bf: Vec<u16> = (0..n * n)
        .map(|t| f32_to_bf16((t as f32 * 0.013).cos() * 0.5))
        .collect();

    let t = Instant::now();
    let wantf = matmul_bf16_scalar(&af, &bf, n, n, n);
    let dt_sf = t.elapsed().as_secs_f64();
    report("scalaire", dt_sf, ops, dt_sf);

    let t = Instant::now();
    let gotf = amx_matmul_bf16(&af, &bf, n, n, n);
    let dt_af = t.elapsed().as_secs_f64();
    report("AMX     ", dt_af, ops, dt_sf);

    let mut ok = true;
    for t in (0..n * n).step_by(1013)
    {
        if (gotf[t] - wantf[t]).abs() > 1e-2 * (1.0 + wantf[t].abs())
        {
            ok = false;
            break;
        }
    }
    println!("  cohérence : {}", if ok { "OK" } else { "ÉCART" });
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("amx_bench : cible non-x86_64 — AMX indisponible.");
}

#[allow(dead_code)]
fn report(label: &str, dt: f64, ops: f64, baseline: f64) {
    println!(
        "  {label} : {:8.1} ms   {:7.2} GOP/s   ×{:.1}",
        dt * 1e3,
        ops / dt / 1e9,
        baseline / dt
    );
}
