//! Measured RLS benchmark — ns per `update()` for the three variants.
//!
//! Run with `cargo run -p scirust-estimation --bin bench_rls --release`.
//! Prints a table; numbers depend on the host, so any figure quoted in docs
//! must name the machine it was measured on. This exists so that performance
//! claims about the RLS stack are *measured*, never asserted.

use scirust_estimation::{
    BlockQrdRls, QrRls, RlsFilter, RlsFilterConst, SquaredGivensRls, VectorRls,
};
use std::hint::black_box;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn bench<F: FnMut(&[f64], f64)>(label: &str, n: usize, iters: usize, mut f: F) {
    let mut rng = Lcg(42);
    let inputs: Vec<Vec<f64>> = (0..256)
        .map(|_| (0..n).map(|_| rng.next()).collect())
        .collect();
    let targets: Vec<f64> = (0..256).map(|_| rng.next()).collect();
    // Warmup.
    for i in 0..iters / 10
    {
        f(&inputs[i % 256], targets[i % 256]);
    }
    let t0 = Instant::now();
    for i in 0..iters
    {
        f(&inputs[i % 256], targets[i % 256]);
    }
    let dt = t0.elapsed();
    println!(
        "{label:>28}  n={n:>3}  {:>9.1} ns/update  ({iters} iters)",
        dt.as_nanos() as f64 / iters as f64
    );
}

/// Bench `BlockQrdRls::update_block` at a fixed block size, reporting
/// ns/*sample* (total time / (iters·block_size)) so it is directly
/// comparable to the per-sample figures above.
fn bench_block(n: usize, block_size: usize, blocks: usize) {
    let mut rng = Lcg(42);
    let inputs: Vec<Vec<f64>> = (0..256)
        .map(|_| (0..(n * block_size)).map(|_| rng.next()).collect())
        .collect();
    let targets: Vec<Vec<f64>> = (0..256)
        .map(|_| (0..block_size).map(|_| rng.next()).collect())
        .collect();
    let mut b = BlockQrdRls::new(n, 1, 0.98, 100.0);
    for i in 0..blocks / 10
    {
        b.update_block(&inputs[i % 256], &targets[i % 256], block_size);
    }
    let t0 = Instant::now();
    for i in 0..blocks
    {
        b.update_block(&inputs[i % 256], &targets[i % 256], block_size);
        black_box(b.factor());
    }
    let dt = t0.elapsed();
    let samples = blocks * block_size;
    println!(
        "{:>28}  n={n:>3}  {:>9.1} ns/update  ({blocks} blocks of {block_size}, {samples} samples)",
        format!("BlockQrdRls (B={block_size})"),
        dt.as_nanos() as f64 / samples as f64
    );
}

fn main() {
    println!("RLS update() benchmark — deterministic inputs, release build\n");
    let iters = 2_000_000;

    for &n in &[4usize, 16, 64]
    {
        let mut v = VectorRls::new(n, 0.98, 100.0);
        bench("VectorRls (zero-alloc)", n, iters / n.max(1), |u, d| {
            black_box(v.update(u, d));
        });

        let mut m = RlsFilter::new(n, 1, 0.98, 100.0);
        bench(
            "RlsFilter 1-out (zero-alloc)",
            n,
            iters / n.max(1),
            |u, d| {
                black_box(m.update(u, &[d]));
            },
        );

        let mut q = QrRls::new(n, 0.98, 100.0);
        bench("QrRls (square-root)", n, iters / n.max(1), |u, d| {
            black_box(q.update(u, d));
        });

        let mut sg = SquaredGivensRls::new(n, 1, 0.98, 100.0);
        bench("SquaredGivensRls (no sqrt)", n, iters / n.max(1), |u, d| {
            sg.update(u, &[d]);
            black_box(sg.weights_diag());
        });

        // Block/batch QRD-RLS (axis 3): does grouping samples into one
        // Householder-QR reduction beat sequential per-sample updates? Not a
        // foregone conclusion — Householder reintroduces the √ and ÷ that
        // SquaredGivensRls eliminated. Measure, don't assume.
        let block_iters = (iters / n.max(1)) / 64;
        for &b in &[1usize, 4, 16, 64]
        {
            bench_block(n, b, block_iters.max(1));
        }
    }

    // Const-generic variants need compile-time sizes: measure the same three
    // sizes explicitly.
    {
        let mut c: RlsFilterConst<4, 1> = RlsFilterConst::new(0.98, 100.0);
        let mut rng = Lcg(42);
        let iters_c = 2_000_000 / 4;
        let t0 = Instant::now();
        for _ in 0..iters_c
        {
            let u = [rng.next(), rng.next(), rng.next(), rng.next()];
            black_box(c.update(&u, &[0.5]));
        }
        println!(
            "{:>28}  n=  4  {:>9.1} ns/update  ({iters_c} iters, incl. input gen)",
            "RlsFilterConst<4,1> (stack)",
            t0.elapsed().as_nanos() as f64 / iters_c as f64
        );
    }
    {
        let mut c: RlsFilterConst<16, 1> = RlsFilterConst::new(0.98, 100.0);
        let mut rng = Lcg(42);
        let mut u = [0.0; 16];
        let iters_c = 2_000_000 / 16;
        let t0 = Instant::now();
        for _ in 0..iters_c
        {
            for x in u.iter_mut()
            {
                *x = rng.next();
            }
            black_box(c.update(&u, &[0.5]));
        }
        println!(
            "{:>28}  n= 16  {:>9.1} ns/update  ({iters_c} iters, incl. input gen)",
            "RlsFilterConst<16,1> (stack)",
            t0.elapsed().as_nanos() as f64 / iters_c as f64
        );
    }
    {
        let mut c: RlsFilterConst<64, 1> = RlsFilterConst::new(0.98, 100.0);
        let mut rng = Lcg(42);
        let mut u = [0.0; 64];
        let iters_c = 2_000_000 / 64;
        let t0 = Instant::now();
        for _ in 0..iters_c
        {
            for x in u.iter_mut()
            {
                *x = rng.next();
            }
            black_box(c.update(&u, &[0.5]));
        }
        println!(
            "{:>28}  n= 64  {:>9.1} ns/update  ({iters_c} iters, incl. input gen)",
            "RlsFilterConst<64,1> (stack)",
            t0.elapsed().as_nanos() as f64 / iters_c as f64
        );
    }
}
