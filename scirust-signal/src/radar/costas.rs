//! Costas frequency-hopping arrays.
//!
//! A Costas array is a permutation of `0..n` whose set of displacement vectors
//! `(row-shift, column-difference)` are all distinct: no two pairs of dots share
//! the same offset. That single combinatorial property gives the frequency-hopped
//! waveform an ideal *thumbtack* ambiguity surface — a sharp central peak with
//! every off-origin auto-ambiguity sidelobe held to at most one coincidence, the
//! best sidelobe floor a hopping pattern can achieve. This module builds them by
//! the **Welch** construction (from a primitive root of a prime `p`, a Costas
//! array of order `p-1`), tests an arbitrary permutation for the Costas property
//! with the difference-triangle test ([`is_costas`]), and measures its worst
//! sidelobe as the peak off-origin coincidence count ([`max_coincidence`]).
//! Dependency-free.

/// Whether `n` is prime by trial division up to `√n`.
fn is_prime(n: usize) -> bool {
    if n < 2
    {
        return false;
    }
    if n < 4
    {
        return true; // 2 and 3
    }
    if n.is_multiple_of(2)
    {
        return false;
    }
    let mut d = 3;
    while d * d <= n
    {
        if n.is_multiple_of(d)
        {
            return false;
        }
        d += 2;
    }
    true
}

/// The multiplicative order of `g` modulo `p`: the smallest `k ≥ 1` with
/// `gᵏ ≡ 1 (mod p)`. Assumes `1 ≤ g < p` and `p` prime (so `g` is a unit).
fn multiplicative_order(g: usize, p: usize) -> usize {
    let mut val = g % p;
    let mut k = 1;
    while val != 1
    {
        val = (val * g) % p;
        k += 1;
    }
    k
}

/// The smallest **primitive root** modulo a prime `p` — the least `g` whose powers
/// `g¹, g², …, g^{p-1}` run through every non-zero residue (multiplicative order
/// exactly `p-1`). Found by trial search. Returns `None` when `p` is not prime.
///
/// For `p = 2` the only unit is `1`, which has order `1 = p-1`, so `Some(1)`.
pub fn primitive_root(p: usize) -> Option<usize> {
    if !is_prime(p)
    {
        return None;
    }
    (1..p).find(|&g| multiplicative_order(g, p) == p - 1)
}

/// The **Welch construction** of a Costas array of order `p-1`: with the smallest
/// primitive root `g` of the prime `p`, the length-`(p-1)` sequence
/// `a[i] = g^{i+1} mod p` (1-based exponent) mapped down to 0-based hop indices by
/// `a[i] - 1`. The result is a permutation of `0..(p-1)` and a Costas array.
///
/// Returns an empty vector when `p` is not prime.
pub fn welch_costas(p: usize) -> Vec<usize> {
    let g = match primitive_root(p)
    {
        Some(g) => g,
        None => return Vec::new(),
    };
    let mut out = Vec::with_capacity(p - 1);
    let mut val = 1usize; // g⁰
    for _ in 0..p - 1
    {
        val = (val * g) % p; // advance to g^{i+1}
        out.push(val - 1); // 1-based residue → 0-based hop index
    }
    out
}

/// Whether `perm` is a **Costas array**: a permutation of `0..len` in which, for
/// every row-shift `s`, the column-differences `perm[i+s] - perm[i]` are pairwise
/// distinct (the difference-triangle test). Equivalent to every displacement
/// vector occurring at most once, i.e. [`max_coincidence`] `≤ 1`. An empty slice
/// is vacuously Costas.
pub fn is_costas(perm: &[usize]) -> bool {
    let n = perm.len();
    // Must be a permutation of 0..n.
    let mut seen = vec![false; n];
    for &v in perm
    {
        if v >= n || seen[v]
        {
            return false;
        }
        seen[v] = true;
    }
    // Each row of the difference triangle must hold distinct differences.
    for s in 1..n
    {
        let mut diffs: Vec<i64> = Vec::with_capacity(n - s);
        for i in 0..n - s
        {
            let d = perm[i + s] as i64 - perm[i] as i64;
            if diffs.contains(&d)
            {
                return false;
            }
            diffs.push(d);
        }
    }
    true
}

/// The **peak off-origin coincidence count**: the largest number of pairs sharing
/// one displacement vector `(s, d)` over all non-zero row-shifts `s`, i.e. the
/// highest auto-ambiguity sidelobe away from the origin. A Costas array scores
/// `≤ 1`; the identity permutation, where a whole diagonal of differences repeats,
/// scores much higher. Zero for a slice shorter than two elements.
pub fn max_coincidence(perm: &[usize]) -> usize {
    let n = perm.len();
    let mut worst = 0;
    for s in 1..n
    {
        let diffs: Vec<i64> = (0..n - s)
            .map(|i| perm[i + s] as i64 - perm[i] as i64)
            .collect();
        for &d in &diffs
        {
            let count = diffs.iter().filter(|&&x| x == d).count();
            if count > worst
            {
                worst = count;
            }
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welch_construction_is_costas_for_small_primes() {
        // The Welch array of order p-1 is a genuine Costas array with an ideal
        // (≤ 1) off-origin ambiguity floor, for every prime tested.
        for &p in &[5usize, 7, 11, 13]
        {
            let arr = welch_costas(p);
            assert_eq!(arr.len(), p - 1, "order should be p-1 for p = {p}");
            assert!(is_costas(&arr), "welch_costas({p}) must be Costas: {arr:?}");
            assert!(max_coincidence(&arr) <= 1, "sidelobe > 1 for p = {p}");
        }
    }

    #[test]
    fn welch_matches_hand_built_p5() {
        // Primitive root g = 2 of 5 gives residues 2,4,3,1 → hops 1,3,2,0.
        assert_eq!(welch_costas(5), vec![1, 3, 2, 0]);
    }

    #[test]
    fn hand_built_costas_passes() {
        // [1,3,2,0] has a distinct-difference triangle, so it is Costas with a
        // single coincidence at each shift.
        let arr = [1usize, 3, 2, 0];
        assert!(is_costas(&arr));
        assert_eq!(max_coincidence(&arr), 1);
    }

    #[test]
    fn linear_permutation_is_not_costas() {
        // The identity repeats the displacement (shift 1, +1) three times, so it
        // fails the difference-triangle test and its sidelobe is 3, not 1.
        let arr = [0usize, 1, 2, 3];
        assert!(!is_costas(&arr));
        assert_eq!(max_coincidence(&arr), 3);
    }

    #[test]
    fn primitive_root_has_full_order() {
        // The returned root's multiplicative order equals p-1 exactly.
        for &p in &[5usize, 7, 11, 13]
        {
            let g = primitive_root(p).expect("prime has a primitive root");
            // g^(p-1) ≡ 1 and no smaller positive power does.
            let mut val = 1usize;
            let mut order = 0;
            for k in 1..p
            {
                val = (val * g) % p;
                if val == 1
                {
                    order = k;
                    break;
                }
            }
            assert_eq!(order, p - 1, "root {g} of {p} is not primitive");
        }
        // Smallest roots are the textbook values.
        assert_eq!(primitive_root(5), Some(2));
        assert_eq!(primitive_root(7), Some(3));
    }

    #[test]
    fn non_prime_inputs_are_guarded() {
        // Non-primes have no primitive root and produce an empty Welch array.
        assert_eq!(primitive_root(1), None);
        assert_eq!(primitive_root(4), None);
        assert_eq!(primitive_root(9), None);
        assert!(welch_costas(1).is_empty());
        assert!(welch_costas(4).is_empty());
        assert!(welch_costas(0).is_empty());
    }

    #[test]
    fn is_costas_rejects_non_permutations() {
        // Repeated value and out-of-range value both fail the permutation gate.
        assert!(!is_costas(&[0usize, 0, 1]));
        assert!(!is_costas(&[1usize, 2, 3])); // missing 0, contains out-of-range 3
        // A single dot is vacuously Costas with no off-origin sidelobe.
        assert!(is_costas(&[0usize]));
        assert_eq!(max_coincidence(&[0usize]), 0);
    }
}
