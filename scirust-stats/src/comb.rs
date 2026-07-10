//! Exact and logarithmic combinatorics.
//!
//! Exact counts are computed in `u128` and return `None` on overflow rather
//! than panicking or silently saturating; the `ln_*` forms never overflow and
//! are the numeric backbone of the discrete pmfs in [`crate::discrete`] and of
//! the game-odds calculations in [`crate::lottery`].
//!
//! `binomial` uses the multiplicative recurrence `C(m, i) = C(m−1, i−1)·m/i`
//! (exact integer division at every step) with gcd pre-reduction, so it stays
//! exact up to genuinely astronomical counts (e.g. `C(130, 65) ≈ 9.5 × 10³⁷`)
//! instead of failing at the first intermediate overflow.

use scirust_special::ln_gamma;

/// Greatest common divisor (Euclid), used to pre-reduce factors in `binomial`.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0
    {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Exact factorial `n!` as `u128`; `None` for `n > 34` (`35!` overflows u128).
pub fn factorial(n: u64) -> Option<u128> {
    if n > 34
    {
        return None;
    }
    let mut acc: u128 = 1;
    for i in 2..=u128::from(n)
    {
        acc *= i;
    }
    Some(acc)
}

/// `ln(n!)` via `ln_gamma(n + 1)`; defined for every `u64`, never overflows.
pub fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

/// Exact binomial coefficient `C(n, k)` as `u128`.
///
/// Returns `Some(0)` when `k > n` (the standard convention) and `None` only
/// when the exact value cannot be represented even after gcd reduction of the
/// intermediate products.
pub fn binomial(n: u64, k: u64) -> Option<u128> {
    if k > n
    {
        return Some(0);
    }
    let k = k.min(n - k);
    // Invariant after step i: acc = C(n − k + i, i), an exact integer.
    let mut acc: u128 = 1;
    for i in 1..=k
    {
        let f = u128::from(n - k + i);
        let mut den = u128::from(i);
        // acc·f is divisible by den; strip shared factors before multiplying
        // so the intermediate stays as small as possible.
        let g1 = gcd(acc, den);
        let acc_r = acc / g1;
        den /= g1;
        let g2 = gcd(f, den);
        let f_r = f / g2;
        den /= g2;
        acc = acc_r.checked_mul(f_r)? / den;
    }
    Some(acc)
}

/// `ln C(n, k)`; `−∞` when `k > n` (an impossible selection has zero count).
pub fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n
    {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// Exact number of ordered `k`-arrangements `P(n, k) = n!/(n−k)!` as `u128`;
/// `Some(0)` when `k > n`, `None` on overflow.
pub fn permutations(n: u64, k: u64) -> Option<u128> {
    if k > n
    {
        return Some(0);
    }
    let mut acc: u128 = 1;
    for i in (n - k + 1)..=n
    {
        acc = acc.checked_mul(u128::from(i))?;
    }
    Some(acc)
}

/// Combinations with repetition ("multichoose"): `C(n + k − 1, k)`.
///
/// The number of size-`k` multisets drawn from `n` distinct items; `Some(1)`
/// for `k = 0`, `Some(0)` for `n = 0, k > 0`, `None` on overflow.
pub fn multichoose(n: u64, k: u64) -> Option<u128> {
    if n == 0
    {
        return Some(u128::from(k == 0));
    }
    binomial(n + k - 1, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorial_exact_and_bounds() {
        assert_eq!(factorial(0), Some(1));
        assert_eq!(factorial(1), Some(1));
        assert_eq!(factorial(12), Some(479_001_600));
        // 34! is the largest factorial representable in u128.
        assert_eq!(
            factorial(34),
            Some(295_232_799_039_604_140_847_618_609_643_520_000_000)
        );
        assert_eq!(factorial(35), None);
    }

    #[test]
    fn binomial_exact_reference_values() {
        assert_eq!(binomial(0, 0), Some(1));
        assert_eq!(binomial(5, 7), Some(0));
        assert_eq!(binomial(49, 6), Some(13_983_816)); // classic 6/49
        assert_eq!(binomial(50, 5), Some(2_118_760)); // EuroMillions main
        assert_eq!(binomial(69, 5), Some(11_238_513)); // Powerball main
        assert_eq!(binomial(52, 5), Some(2_598_960)); // poker hands
        // Symmetry.
        assert_eq!(binomial(49, 6), binomial(49, 43));
        // Survives intermediate magnitudes near the u128 ceiling.
        assert_eq!(
            binomial(130, 65),
            Some(95_067_625_827_960_698_145_584_333_020_095_113_100)
        );
    }

    #[test]
    fn ln_forms_match_exact() {
        let exact = binomial(49, 6).unwrap() as f64;
        assert!((ln_binomial(49, 6) - exact.ln()).abs() < 1e-12);
        assert!(ln_binomial(5, 7) == f64::NEG_INFINITY);
        assert!((ln_factorial(12) - (479_001_600.0_f64).ln()).abs() < 1e-10);
        // Stirling territory: ln(1000!) = 5912.128178... (published value).
        assert!((ln_factorial(1000) - 5_912.128_178_488_163).abs() < 1e-8);
    }

    #[test]
    fn permutations_and_multichoose() {
        assert_eq!(permutations(49, 6), Some(10_068_347_520));
        assert_eq!(permutations(5, 0), Some(1));
        assert_eq!(permutations(3, 5), Some(0));
        assert_eq!(multichoose(5, 3), Some(35)); // C(7,3)
        assert_eq!(multichoose(0, 0), Some(1));
        assert_eq!(multichoose(0, 2), Some(0));
    }
}
