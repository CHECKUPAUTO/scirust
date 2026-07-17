//! Deterministic exact integer number theory over the machine integers.
//!
//! Everything here is integer-only, allocation-light, and **fully
//! deterministic**: no floating point, no OS entropy, no probabilistic
//! acceptance. In particular [`is_prime`] is a *deterministic* Miller–Rabin
//! test (a fixed witness set that is provably exact for every `u64`), and
//! [`factor`] uses a deterministic Pollard–Brent rho with a fixed constant
//! schedule, so the same input always yields the same factorization on every
//! platform.
//!
//! These are the classic reusable building blocks — modular inverse, modular
//! exponentiation, CRT, primality, factorization, Euler's totient, the Jacobi
//! symbol — that SciRust crates repeatedly need in exact, portable form.

/// Greatest common divisor of two unsigned integers (`gcd(0, 0) = 0`).
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0
    {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Least common multiple (`lcm(a, 0) = lcm(0, b) = 0`). The mathematical result
/// must fit in a `u64`.
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0
    {
        0
    }
    else
    {
        a / gcd(a, b) * b
    }
}

/// Extended Euclid over `i128`: returns `(g, x, y)` with `a·x + b·y = g` and
/// `g = gcd(|a|, |b|) ≥ 0`.
pub fn egcd(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i128, 0i128);
    let (mut old_t, mut t) = (0i128, 1i128);
    while r != 0
    {
        let q = old_r.div_euclid(r);
        let nr = old_r - q * r;
        old_r = r;
        r = nr;
        let ns = old_s - q * s;
        old_s = s;
        s = ns;
        let nt = old_t - q * t;
        old_t = t;
        t = nt;
    }
    if old_r < 0
    {
        (-old_r, -old_s, -old_t)
    }
    else
    {
        (old_r, old_s, old_t)
    }
}

/// Multiplicative inverse of `a` modulo `m` (`m ≥ 2`), or `None` when
/// `gcd(a, m) ≠ 1`. The result is the canonical representative in `[0, m)`.
pub fn inv_mod(a: u64, m: u64) -> Option<u64> {
    if m < 2
    {
        return None;
    }
    let (g, x, _) = egcd((a % m) as i128, m as i128);
    if g != 1
    {
        return None;
    }
    let mm = m as i128;
    Some((((x % mm) + mm) % mm) as u64)
}

/// `a · b mod m`, overflow-free via a 128-bit intermediate (`m ≥ 1`).
pub fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// `base^exp mod m` by square-and-multiply (`m ≥ 1`; `pow_mod(_, _, 1) = 0`).
pub fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1
    {
        return 0;
    }
    let mut acc = 1u64;
    base %= m;
    while exp > 0
    {
        if exp & 1 == 1
        {
            acc = mulmod(acc, base, m);
        }
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// Integer square root of `n` (`⌊√n⌋`), exact, for a 128-bit argument.
pub fn isqrt(n: u128) -> u128 {
    if n < 2
    {
        return n;
    }
    // Newton's method from a power-of-two upper bound; converges monotonically.
    let mut x = 1u128 << ((128 - n.leading_zeros()).div_ceil(2));
    loop
    {
        let y = (x + n / x) / 2;
        if y >= x
        {
            return x;
        }
        x = y;
    }
}

/// `true` iff `n` is a perfect square.
pub fn is_perfect_square(n: u128) -> bool {
    let r = isqrt(n);
    r * r == n
}

/// Deterministic Miller–Rabin primality test, exact for **every** `u64`.
///
/// The witness set `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` is a proven
/// deterministic set for all `n < 3.3·10^24`, which covers the entire `u64`
/// range. There is no probabilistic error.
pub fn is_prime(n: u64) -> bool {
    if n < 2
    {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    {
        if n == p
        {
            return true;
        }
        if n % p == 0
        {
            return false;
        }
    }
    // n-1 = d · 2^s with d odd
    let mut d = n - 1;
    let mut s = 0u32;
    while d & 1 == 0
    {
        d >>= 1;
        s += 1;
    }
    'witness: for &a in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1
        {
            continue;
        }
        for _ in 0..s - 1
        {
            x = mulmod(x, x, n);
            if x == n - 1
            {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// A single non-trivial factor of the composite, odd `n` via Pollard–Brent rho
/// with a deterministic constant schedule. Precondition: `n` composite and odd.
fn pollard_rho(n: u64) -> u64 {
    for c in 1u64..
    {
        let f = |x: u64| (mulmod(x, x, n) + c) % n;
        let mut x = 2u64;
        let mut y = 2u64;
        let mut d = 1u64;
        while d == 1
        {
            x = f(x);
            y = f(f(y));
            let diff = x.abs_diff(y);
            d = gcd(diff, n);
        }
        if d != n
        {
            return d;
        }
        // d == n: the constant failed, advance the schedule and retry.
    }
    unreachable!("Pollard–Brent always finds a factor of a composite")
}

/// Prime factorization of `n` as `(prime, exponent)` pairs sorted by prime.
/// `factor(0)` and `factor(1)` return an empty vector.
pub fn factor(n: u64) -> Vec<(u64, u32)> {
    let mut primes: Vec<u64> = Vec::new();
    fn rec(n: u64, out: &mut Vec<u64>) {
        if n == 1
        {
            return;
        }
        if is_prime(n)
        {
            out.push(n);
            return;
        }
        let d = if n & 1 == 0 { 2 } else { pollard_rho(n) };
        rec(d, out);
        rec(n / d, out);
    }
    if n >= 2
    {
        rec(n, &mut primes);
    }
    primes.sort_unstable();
    let mut out: Vec<(u64, u32)> = Vec::new();
    for p in primes
    {
        match out.last_mut()
        {
            Some((q, e)) if *q == p => *e += 1,
            _ => out.push((p, 1)),
        }
    }
    out
}

/// All positive divisors of `n`, sorted ascending (`divisors(0)` is empty,
/// `divisors(1) = [1]`).
pub fn divisors(n: u64) -> Vec<u64> {
    if n == 0
    {
        return Vec::new();
    }
    let mut divs = vec![1u64];
    for (p, e) in factor(n)
    {
        let base = divs.clone();
        let mut pk = 1u64;
        for _ in 0..e
        {
            pk *= p;
            for &d in &base
            {
                divs.push(d * pk);
            }
        }
    }
    divs.sort_unstable();
    divs
}

/// Euler's totient `φ(n)` (`φ(0) = 0`, `φ(1) = 1`).
pub fn euler_phi(n: u64) -> u64 {
    if n == 0
    {
        return 0;
    }
    let mut phi = 1u64;
    for (p, e) in factor(n)
    {
        // p^(e-1) · (p-1)
        let mut pk = 1u64;
        for _ in 0..e - 1
        {
            pk *= p;
        }
        phi *= pk * (p - 1);
    }
    phi
}

/// The Jacobi symbol `(a / n)` for odd `n ≥ 1`, returning `-1`, `0`, or `1`.
/// (For prime `n` this is the Legendre symbol.) Panics if `n` is even.
pub fn jacobi(mut a: i64, mut n: u64) -> i32 {
    assert!(n & 1 == 1, "Jacobi symbol requires an odd modulus");
    // reduce a into [0, n)
    let nn = n as i64;
    a = ((a % nn) + nn) % nn;
    let mut result = 1i32;
    let mut a = a as u64;
    while a != 0
    {
        while a & 1 == 0
        {
            a >>= 1;
            let r = n & 7;
            if r == 3 || r == 5
            {
                result = -result;
            }
        }
        core::mem::swap(&mut a, &mut n);
        if a & 3 == 3 && n & 3 == 3
        {
            result = -result;
        }
        a %= n;
    }
    if n == 1 { result } else { 0 }
}

/// Combine two congruences `x ≡ r1 (mod m1)`, `x ≡ r2 (mod m2)` with coprime
/// moduli into the unique `(r, m1·m2)` with `0 ≤ r < m1·m2`. Returns `None`
/// when the moduli are not coprime or their product would overflow `u64`.
pub fn crt_pair(r1: u64, m1: u64, r2: u64, m2: u64) -> Option<(u64, u64)> {
    let (g, p, _) = egcd(m1 as i128, m2 as i128);
    if g != 1
    {
        return None;
    }
    let m = m1.checked_mul(m2)?;
    let m2i = m2 as i128;
    let inv = (((p % m2i) + m2i) % m2i) as u64; // (m1)^{-1} mod m2
    let diff = (r2 % m2 + m2 - r1 % m2) % m2;
    let t = mulmod(diff, inv, m2);
    Some((r1 % m1 + m1 * t, m))
}

/// Chinese Remainder Theorem over a list of `(residue, modulus)` pairs with
/// pairwise-coprime moduli. Returns `(r, M)` with `M = ∏ modulus` and
/// `0 ≤ r < M`, or `None` on a non-coprime pair or `u64` overflow of `M`.
/// An empty list yields the trivial `(0, 1)`.
pub fn crt(congruences: &[(u64, u64)]) -> Option<(u64, u64)> {
    let mut acc = (0u64, 1u64);
    for &(r, m) in congruences
    {
        acc = crt_pair(acc.0, acc.1, r % m, m)?;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic xorshift for test inputs (no OS entropy)
    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn gcd_lcm_basics() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(17, 5), 1);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(0, 5), 0);
    }

    #[test]
    fn egcd_bezout_identity() {
        let mut s = 0x1234_5678u64;
        for _ in 0..2000
        {
            let a = (xorshift(&mut s) % 100_000) as i128;
            let b = (xorshift(&mut s) % 100_000) as i128;
            let (g, x, y) = egcd(a, b);
            assert_eq!(a * x + b * y, g, "Bezout failed for a={a} b={b}");
            assert_eq!(g, gcd(a as u64, b as u64) as i128);
        }
    }

    #[test]
    fn inv_mod_round_trips() {
        let mut s = 0xdead_beefu64;
        let m = 1_000_003u64; // prime
        for _ in 0..2000
        {
            let a = xorshift(&mut s) % m;
            if a == 0
            {
                continue;
            }
            let inv = inv_mod(a, m).expect("prime modulus, nonzero a");
            assert_eq!(mulmod(a, inv, m), 1);
        }
        // non-coprime has no inverse
        assert_eq!(inv_mod(4, 8), None);
        assert_eq!(inv_mod(6, 9), None);
    }

    #[test]
    fn pow_mod_matches_naive() {
        let mut s = 0x99u64;
        for _ in 0..500
        {
            let base = xorshift(&mut s) % 1000;
            let exp = xorshift(&mut s) % 64;
            let m = 1 + xorshift(&mut s) % 9973;
            let mut naive = 1u64 % m;
            for _ in 0..exp
            {
                naive = mulmod(naive, base % m, m);
            }
            assert_eq!(pow_mod(base, exp, m), naive);
        }
    }

    #[test]
    fn is_prime_matches_sieve() {
        // sieve of Eratosthenes up to N
        const N: usize = 100_000;
        let mut sieve = vec![true; N + 1];
        sieve[0] = false;
        sieve[1] = false;
        for i in 2..=N
        {
            if sieve[i]
            {
                let mut j = i * i;
                while j <= N
                {
                    sieve[j] = false;
                    j += i;
                }
            }
        }
        for (n, &p) in sieve.iter().enumerate()
        {
            assert_eq!(is_prime(n as u64), p, "primality mismatch at n={n}");
        }
    }

    #[test]
    fn is_prime_large_and_carmichael() {
        // large primes
        assert!(is_prime((1u64 << 61) - 1)); // Mersenne prime M61
        assert!(is_prime(18_446_744_073_709_551_557)); // largest prime < 2^64
        assert!(is_prime(2_147_483_647)); // M31
        // Carmichael numbers are composite (fool Fermat, not Miller–Rabin)
        for c in [561u64, 1105, 1729, 2465, 2821, 6601, 62745, 162401]
        {
            assert!(!is_prime(c), "Carmichael {c} reported prime");
        }
        // large composite = product of two primes
        assert!(!is_prime(1_000_003u64 * 1_000_033u64));
    }

    #[test]
    fn factor_reconstructs_and_is_prime() {
        let mut s = 0x5151_5151u64;
        for _ in 0..300
        {
            let n = 1 + xorshift(&mut s);
            let f = factor(n);
            let mut prod = 1u128;
            let mut prev = 0u64;
            for (p, e) in &f
            {
                assert!(is_prime(*p), "non-prime factor {p} of {n}");
                assert!(*p > prev, "factors not strictly sorted");
                prev = *p;
                for _ in 0..*e
                {
                    prod *= *p as u128;
                }
            }
            assert_eq!(prod, n as u128, "product mismatch for {n}");
        }
    }

    #[test]
    fn factor_hard_semiprimes_and_powers() {
        // product of two large primes (rho must crack it); both ~2^30 so the
        // product fits in u64
        let p = 1_000_000_007u64; // prime
        let q = 1_000_000_009u64; // prime
        assert_eq!(factor(p * q), vec![(p, 1), (q, 1)]);
        // prime power
        assert_eq!(factor(1u64 << 20), vec![(2, 20)]);
        // 3^10
        assert_eq!(factor(59049), vec![(3, 10)]);
    }

    #[test]
    fn divisors_and_phi() {
        assert_eq!(divisors(1), vec![1]);
        assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(divisors(28), vec![1, 2, 4, 7, 14, 28]); // perfect number
        // euler_phi against a brute-force coprime count
        for n in 1u64..=3000
        {
            let brute = (1..=n).filter(|&k| gcd(k, n) == 1).count() as u64;
            assert_eq!(euler_phi(n), brute, "phi mismatch at n={n}");
        }
        // sum of divisors of a perfect number equals 2n
        assert_eq!(divisors(28).iter().sum::<u64>(), 56);
    }

    #[test]
    fn jacobi_matches_euler_criterion() {
        for &p in &[3u64, 5, 7, 11, 13, 1009]
        {
            for a in 0..p
            {
                let j = jacobi(a as i64, p);
                let euler = pow_mod(a, (p - 1) / 2, p);
                let expect = if a % p == 0
                {
                    0
                }
                else if euler == 1
                {
                    1
                }
                else
                {
                    -1
                };
                assert_eq!(j, expect, "Jacobi/Legendre mismatch a={a} p={p}");
            }
        }
        // Jacobi is multiplicative in the top argument (composite modulus)
        let n = 45u64; // 9 · 5, odd
        for a in 0..n
        {
            for b in 0..n
            {
                assert_eq!(
                    jacobi((a * b) as i64, n),
                    jacobi(a as i64, n) * jacobi(b as i64, n)
                );
            }
        }
    }

    #[test]
    fn isqrt_bounds() {
        let mut s = 0x2468u64;
        for _ in 0..5000
        {
            let n = xorshift(&mut s) as u128;
            let r = isqrt(n);
            assert!(r * r <= n);
            assert!((r + 1) * (r + 1) > n);
        }
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(
            isqrt((u64::MAX as u128) * (u64::MAX as u128)),
            u64::MAX as u128
        );
        assert!(is_perfect_square(144));
        assert!(!is_perfect_square(145));
    }

    #[test]
    fn crt_reconstructs_residues() {
        // classic (2 mod 3, 3 mod 5, 2 mod 7) -> 23 mod 105
        let (r, m) = crt(&[(2, 3), (3, 5), (2, 7)]).unwrap();
        assert_eq!((r, m), (23, 105));

        let mut s = 0x777u64;
        for _ in 0..500
        {
            let m1 = 3 + xorshift(&mut s) % 1000;
            let m2 = 3 + xorshift(&mut s) % 1000;
            if gcd(m1, m2) != 1
            {
                continue;
            }
            let x = xorshift(&mut s) % (m1 * m2);
            let (r, m) = crt(&[(x % m1, m1), (x % m2, m2)]).unwrap();
            assert_eq!(m, m1 * m2);
            assert_eq!(r % m1, x % m1);
            assert_eq!(r % m2, x % m2);
            assert!(r < m);
        }
        // non-coprime moduli are rejected
        assert_eq!(crt(&[(1, 4), (1, 6)]), None);
        // empty list is the trivial congruence
        assert_eq!(crt(&[]), Some((0, 1)));
    }
}
