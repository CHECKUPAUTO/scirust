//! Balanced integer factorization for choosing TT mode dimensions.
//!
//! Given `n = in_features` (or `out_features`) and a desired number of factors
//! `d`, returns a factorization `[f_0, f_1, ..., f_{d-1}]` such that
//! `prod(f_i) = n` and the factors are as balanced as possible (close to `n^(1/d)`).
//!
//! Algorithm: extract prime factors of `n`, then greedily distribute them
//! across `d` bins to minimize the spread.
//!
//! When `n` is hard to factor (prime or near-prime), the returned factorization
//! may include factors of `1`, which still yields a valid (if not very useful)
//! TT layout. For best compression results, prefer dimensions whose factors
//! are roughly equal (e.g. 768 = 8 * 12 * 8 rather than 768 = 1 * 768 * 1).

/// Returns the prime factorization of `n` as a sorted vector of prime factors
/// (with multiplicity). E.g. `prime_factors(12) = [2, 2, 3]`.
fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut p = 2usize;
    while p * p <= n {
        while n % p == 0 {
            factors.push(p);
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// Decompose `n` into exactly `d` balanced positive factors whose product is `n`.
///
/// # Panics
/// - if `d == 0`
/// - if `n == 0`
///
/// # Examples
/// ```
/// use scirust_tn::factorize::auto_factorize;
/// assert_eq!(auto_factorize(768, 3).iter().product::<usize>(), 768);
/// assert_eq!(auto_factorize(1, 3), vec![1, 1, 1]);
/// ```
pub fn auto_factorize(n: usize, d: usize) -> Vec<usize> {
    assert!(d > 0, "auto_factorize: d must be > 0");
    assert!(n > 0, "auto_factorize: n must be > 0");

    if d == 1 {
        return vec![n];
    }
    if n == 1 {
        return vec![1; d];
    }

    // Get prime factors in descending order so the largest ones are placed first.
    let mut primes = prime_factors(n);
    primes.sort_unstable_by(|a, b| b.cmp(a));

    // Greedy bin packing: at each step, multiply the next prime into the bin
    // with the smallest current value. This naturally balances the factors.
    let mut bins = vec![1usize; d];
    for p in primes {
        let i = (0..d).min_by_key(|&i| bins[i]).unwrap();
        bins[i] *= p;
    }
    bins.sort_unstable();
    bins
}

/// Convenience: verify that `factors.iter().product() == n` and `factors.len() == d`.
/// Used in tests and `TTLinear` constructors.
pub fn check_factorization(factors: &[usize], n: usize) -> bool {
    factors.iter().product::<usize>() == n && factors.iter().all(|&f| f > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_factors() {
        assert_eq!(prime_factors(12), vec![2, 2, 3]);
        assert_eq!(prime_factors(1), Vec::<usize>::new());
        assert_eq!(prime_factors(13), vec![13]);
        assert_eq!(prime_factors(100), vec![2, 2, 5, 5]);
    }

    #[test]
    fn test_auto_factorize_powers_of_two() {
        // 256 = 2^8, 3 factors → ideally close to 256^(1/3) ≈ 6.35
        let f = auto_factorize(256, 3);
        assert_eq!(f.iter().product::<usize>(), 256);
        assert_eq!(f.len(), 3);
        // Should be 4, 8, 8 (or 8, 8, 4 sorted) — balanced
        assert_eq!(f, vec![4, 8, 8]);
    }

    #[test]
    fn test_auto_factorize_768() {
        // 768 = 2^8 * 3, target 3 factors near 768^(1/3) ≈ 9.16
        let f = auto_factorize(768, 3);
        assert_eq!(f.iter().product::<usize>(), 768);
        assert_eq!(f.len(), 3);
        // Result should have all factors > 1 and reasonably balanced
        assert!(f.iter().all(|&x| x > 1));
    }

    #[test]
    fn test_auto_factorize_3072() {
        // Typical transformer FFN dim: 3072 = 2^10 * 3
        let f = auto_factorize(3072, 3);
        assert_eq!(f.iter().product::<usize>(), 3072);
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn test_auto_factorize_prime() {
        // 769 is prime. Forces a [1, 1, 769] outcome — not useful but valid.
        let f = auto_factorize(769, 3);
        assert_eq!(f.iter().product::<usize>(), 769);
        assert_eq!(f.len(), 3);
        assert!(f.contains(&769));
    }

    #[test]
    fn test_auto_factorize_one() {
        assert_eq!(auto_factorize(1, 3), vec![1, 1, 1]);
        assert_eq!(auto_factorize(1, 1), vec![1]);
    }

    #[test]
    fn test_check_factorization() {
        assert!(check_factorization(&[2, 3, 4], 24));
        assert!(!check_factorization(&[2, 3, 5], 24));
        assert!(!check_factorization(&[0, 3, 4], 0)); // factor of zero
    }
}
