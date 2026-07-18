#![forbid(unsafe_code)]
//! Self-contained micro-benchmarks for `scirust-modalg`, plus an honest
//! positioning note. This is a manual tool, **not** a CI test — the timing loops
//! run only from `main`, which the test harness never invokes.
//!
//! ```text
//! cargo run --release --bin modalg-bench
//! ```
//!
//! The headline measurement is the crossover between the schoolbook
//! [`BigInt::mul`] and the NTT-based [`BigInt::mul_ntt`]: the two are exactly
//! equal in value, so this quantifies *only* the speed difference. The other
//! rows give context. The **positioning** section states, honestly, where these
//! exact reference implementations sit relative to optimized/SIMD libraries.

use scirust_modalg::bigint::BigInt;
use scirust_modalg::crc::Crc;
use scirust_modalg::ntt::Ntt;
use std::time::{Duration, Instant};

/// A deterministic pseudo-random `BigInt` with roughly `limbs` 32-bit limbs.
fn make_bigint(limbs: usize) -> BigInt {
    let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ (limbs as u64).wrapping_mul(0x1234_5679);
    let base = BigInt::from_i128(1i128 << 32);
    let mut x = BigInt::zero();
    for _ in 0..limbs
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        x = x
            .mul(&base)
            .add(&BigInt::from_i128((s & 0xFFFF_FFFF) as i128));
    }
    x
}

/// Average wall-clock time of a single call to `f`, over `iters` repetitions.
fn per_op(iters: u32, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iters
    {
        f();
    }
    start.elapsed() / iters.max(1)
}

fn bench_bigint_mul() {
    println!("BigInt multiplication — schoolbook `mul` vs `mul_ntt` (avg per op):");
    println!(
        "  {:>6}  {:>12}  {:>12}  {:>8}",
        "limbs", "schoolbook", "mul_ntt", "speedup"
    );
    for &limbs in &[512usize, 2048, 8192, 32768]
    {
        let a = make_bigint(limbs);
        let b = make_bigint(limbs + 1);
        // fewer iterations for larger inputs so total time stays bounded
        let iters = (100_000u32 / (limbs as u32 + 1)).clamp(1, 50);
        let ts = per_op(iters, || {
            let _ = a.mul(&b);
        });
        let tn = per_op(iters, || {
            let _ = a.mul_ntt(&b);
        });
        let speedup = ts.as_secs_f64() / tn.as_secs_f64().max(1e-12);
        println!("  {limbs:>6}  {ts:>12.2?}  {tn:>12.2?}  {speedup:>7.2}x");
    }
    println!(
        "  schoolbook is O(n²), mul_ntt is O(n log n). mul_ntt carries a large\n  \
         constant (three NTT primes, u128 mulmod, per-coefficient CRT), so it only\n  \
         wins for very large operands — the crossover is around ~16k–32k limbs\n  \
         (~10^6 bits); below that, schoolbook is faster.\n"
    );
}

fn bench_ntt_convolution() {
    println!("NTT convolution over Z/p (avg per op):");
    let ntt = Ntt::new_default();
    for &len in &[256usize, 4096, 65536]
    {
        let a: Vec<u64> = (0..len)
            .map(|i| (i as u64 * 2654435761) % ntt.prime())
            .collect();
        let b: Vec<u64> = (0..len)
            .map(|i| (i as u64 * 40503 + 7) % ntt.prime())
            .collect();
        let iters = (500_000u32 / (len as u32 + 1)).clamp(2, 200);
        let t = per_op(iters, || {
            let _ = ntt.convolve(&a, &b);
        });
        println!("  {len:>6} points: {t:>12.2?}");
    }
    println!();
}

fn bench_crc() {
    println!("CRC-32 throughput (bit-by-bit reference):");
    let crc = Crc::crc32_iso_hdlc();
    let data: Vec<u8> = (0..1_000_000u32).map(|i| (i ^ (i >> 8)) as u8).collect();
    let iters = 5u32;
    let t = per_op(iters, || {
        let _ = crc.checksum(&data);
    });
    let mib_per_s = (data.len() as f64 / (1 << 20) as f64) / t.as_secs_f64();
    println!(
        "  {} bytes: {t:>12.2?}  (~{mib_per_s:.1} MiB/s)\n",
        data.len()
    );
}

fn positioning() {
    println!("Positioning (honest):");
    println!("  These are single-threaded, scalar (u32/u128) *reference* implementations.");
    println!("  Their value is being exact, deterministic, auditable and dependency-free —");
    println!("  NOT raw throughput. Optimized libraries are typically 1–2 orders of");
    println!("  magnitude faster; indicative figures from the literature (NOT measured here):");
    println!("    - bignum multiply : GMP (FFT/Karatsuba + hand-written asm)  ~10–100x");
    println!("    - CRC             : hardware CRC32 / PCLMULQDQ               ~10–50x");
    println!("    - NTT / Reed–Solomon : SIMD (AVX2/AVX-512), multi-thread     ~5–20x");
    println!("  Choose scirust-modalg when bit-exact reproducibility and zero unsafe/deps");
    println!("  matter more than speed; choose an optimized library otherwise.");
}

fn main() {
    println!("=== scirust-modalg micro-benchmarks (reference, single-threaded) ===\n");
    bench_bigint_mul();
    bench_ntt_convolution();
    bench_crc();
    positioning();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_bigint_is_deterministic_and_nonzero() {
        // reproducible: same seed size → same value
        assert_eq!(make_bigint(10), make_bigint(10));
        assert!(!make_bigint(10).is_zero());
    }

    #[test]
    fn benchmarked_paths_are_correct() {
        // the bench compares mul vs mul_ntt; they must agree in value
        let a = make_bigint(20);
        let b = make_bigint(21);
        assert_eq!(a.mul(&b), a.mul_ntt(&b));
    }
}
