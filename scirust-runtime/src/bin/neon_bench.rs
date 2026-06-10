// BENCH matmul int8 : scalaire vs NEON (aarch64), avec controle bit-exact.

#[cfg(target_arch = "aarch64")]
fn main() {
    use scirust_core::quantization::{matmul_int8, matmul_int8_neon};
    use std::time::Instant;

    let (m, k, n) = (64usize, 784usize, 256usize);
    let mut s: u64 = 0xC0FFEE;
    let mut nxt = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 56) as i64 - 128) as i8
    };
    let a: Vec<i8> = (0..m * k).map(|_| nxt()).collect();
    let b: Vec<i8> = (0..k * n).map(|_| nxt()).collect();

    let exact = matmul_int8(&a, &b, m, k, n) == matmul_int8_neon(&a, &b, m, k, n);

    let bench = |f: &dyn Fn() -> Vec<i32>, iters: usize| -> f64 {
        for _ in 0..5
        {
            let _ = f();
        }
        let mut lat = Vec::with_capacity(iters);
        for _ in 0..iters
        {
            let t = Instant::now();
            let _ = f();
            lat.push(t.elapsed().as_nanos() as u64);
        }
        lat.sort_unstable();
        lat[lat.len() / 2] as f64 / 1000.0
    };
    let iters = 100;
    let p_scal = bench(&|| matmul_int8(&a, &b, m, k, n), iters);
    let p_neon = bench(&|| matmul_int8_neon(&a, &b, m, k, n), iters);

    println!();
    println!(
        "=== BENCH matmul_int8 scalaire vs NEON ({}x{}x{}) ===",
        m, k, n
    );
    println!(
        "bit-exact (scal == neon) : {}",
        if exact { "OUI" } else { "NON" }
    );
    println!("scalaire p50 : {:.1} us", p_scal);
    println!("NEON     p50 : {:.1} us  (transpose de b incluse)", p_neon);
    println!("speedup      : {:.2}x", p_scal / p_neon);
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("matmul_int8_neon requiert aarch64");
}
