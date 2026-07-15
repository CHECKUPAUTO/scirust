//! Multi-PRF range disambiguation by the Chinese Remainder Theorem.
//!
//! A single-PRF radar measures range only modulo its unambiguous range: an echo
//! from beyond `R_ua = c/(2·PRF)` folds back and is reported as a nearer target.
//! A staggered- (or multiple-) PRF radar breaks that ambiguity by dwelling at
//! several PRFs whose unambiguous ranges, expressed in integer range-bin units,
//! are pairwise **coprime**. Each dwell yields a residue — the true range bin
//! *modulo* that PRF's span — and the **Chinese Remainder Theorem** stitches the
//! residues back into a single bin that is unambiguous out to the least common
//! multiple of the spans. This module is pure integer arithmetic on `i64`:
//! extended Euclid ([`egcd`]) and the modular inverse ([`mod_inverse`]) build a
//! general two-modulus solver ([`crt_pair`]) that also handles non-coprime
//! moduli through a gcd consistency test; folding that solver across every dwell
//! ([`resolve_range`]) recovers the bin, and [`combined_ambiguity`] reports the
//! combined unambiguous span. Depends only on `std`.

/// Extended Euclidean algorithm. Returns `(g, x, y)` where `g = gcd(a, b) ≥ 0`
/// and the Bézout identity `a·x + b·y = g` holds. For `a = b = 0` it returns
/// `(0, 1, 0)`.
pub fn egcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1_i64, 0_i64);
    let (mut old_t, mut t) = (0_i64, 1_i64);
    while r != 0
    {
        let q = old_r / r;
        let next_r = old_r - q * r;
        old_r = r;
        r = next_r;
        let next_s = old_s - q * s;
        old_s = s;
        s = next_s;
        let next_t = old_t - q * t;
        old_t = t;
        t = next_t;
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

/// The modular inverse of `a` modulo `m`: the unique `x` in `[0, |m|)` with
/// `a·x ≡ 1 (mod m)`. Returns `None` when `m = 0` or `gcd(a, m) ≠ 1` (no inverse
/// exists).
pub fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    if m == 0
    {
        return None;
    }
    let modulus = m.abs();
    let (g, x, _) = egcd(a.rem_euclid(modulus), modulus);
    if g != 1
    {
        return None;
    }
    Some(x.rem_euclid(modulus))
}

/// Solve the pair of congruences `x ≡ r1 (mod m1)` and `x ≡ r2 (mod m2)`.
///
/// Returns `(x, lcm)` with `0 ≤ x < lcm` where `lcm = lcm(m1, m2)` is the modulus
/// of the combined solution. This is the *general* CRT: the moduli need not be
/// coprime — a solution exists iff `r1 ≡ r2 (mod gcd(m1, m2))`, and `None` is
/// returned when that consistency test fails. `None` is also returned for a
/// non-positive modulus.
pub fn crt_pair(r1: i64, m1: i64, r2: i64, m2: i64) -> Option<(i64, i64)> {
    if m1 <= 0 || m2 <= 0
    {
        return None;
    }
    let (g, p, _) = egcd(m1, m2);
    let diff = r2 - r1;
    if diff % g != 0
    {
        return None;
    }
    let lcm = m1 / g * m2;
    // p is the inverse of (m1/g) modulo (m2/g); step lands x on both residues.
    let m2g = m2 / g;
    let step = (((diff / g) % m2g) * (p % m2g)).rem_euclid(m2g);
    let x = (r1 + m1 * step).rem_euclid(lcm);
    Some((x, lcm))
}

/// Recover the true range bin from its `residues` under the matching `moduli`
/// (each PRF's unambiguous span in range-bin units) by folding [`crt_pair`]
/// across every dwell. Returns the least non-negative solution, or `None` when
/// the slices are empty, differ in length, or are mutually inconsistent.
pub fn resolve_range(residues: &[i64], moduli: &[i64]) -> Option<i64> {
    if residues.is_empty() || residues.len() != moduli.len()
    {
        return None;
    }
    if moduli[0] <= 0
    {
        return None;
    }
    let mut acc_r = residues[0].rem_euclid(moduli[0]);
    let mut acc_m = moduli[0];
    for (&r, &m) in residues.iter().zip(moduli.iter()).skip(1)
    {
        let (x, lcm) = crt_pair(acc_r, acc_m, r, m)?;
        acc_r = x;
        acc_m = lcm;
    }
    Some(acc_r)
}

/// The combined unambiguous span: the least common multiple of the `moduli`, the
/// range over which [`resolve_range`] recovers a bin without ambiguity. Empty
/// input yields `1` (the empty-lcm identity); a non-positive modulus yields `0`.
pub fn combined_ambiguity(moduli: &[i64]) -> i64 {
    let mut lcm = 1_i64;
    for &m in moduli
    {
        if m <= 0
        {
            return 0;
        }
        let (g, _, _) = egcd(lcm, m);
        lcm = lcm / g * m;
    }
    lcm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn egcd_satisfies_bezout_identity() {
        // g = gcd and a·x + b·y = g for every pair, including a negative input.
        for &(a, b, g) in &[(7_i64, 11_i64, 1_i64), (12, 18, 6), (-14, 21, 7), (0, 5, 5)]
        {
            let (gg, x, y) = egcd(a, b);
            assert_eq!(gg, g, "gcd({a}, {b})");
            assert_eq!(a * x + b * y, gg, "bezout({a}, {b})");
        }
    }

    #[test]
    fn mod_inverse_round_trips_and_rejects_non_coprime() {
        // a·inv ≡ 1 (mod m) exactly when gcd(a, m) = 1.
        for &(a, m) in &[(3_i64, 11_i64), (10, 1001), (5, 13), (8, 173)]
        {
            let inv = mod_inverse(a, m).expect("coprime inverse exists");
            assert_eq!((a * inv).rem_euclid(m), 1, "inv({a}, {m})");
        }
        // gcd(6, 9) = 3 ≠ 1, so no inverse.
        assert_eq!(mod_inverse(6, 9), None);
    }

    #[test]
    fn crt_pair_solves_coprime_congruences() {
        // x ≡ 5 (mod 7), x ≡ 8 (mod 11) → 173 mod 77 = 19.
        let (x, lcm) = crt_pair(5, 7, 8, 11).expect("coprime pair solves");
        assert_eq!(lcm, 77);
        assert_eq!(x.rem_euclid(7), 5);
        assert_eq!(x.rem_euclid(11), 8);
        assert_eq!(x, 19);
    }

    #[test]
    fn resolve_range_reconstructs_known_bin() {
        // A target at bin 173 seen through three coprime PRF spans (7, 11, 13).
        let truth = 173_i64;
        let moduli = [7_i64, 11, 13];
        let residues: Vec<i64> = moduli.iter().map(|&m| truth.rem_euclid(m)).collect();
        assert_eq!(residues, vec![5, 8, 4]);
        assert_eq!(resolve_range(&residues, &moduli), Some(truth));
        // The combined span is the product of the coprime moduli.
        assert_eq!(combined_ambiguity(&moduli), 1001);
    }

    #[test]
    fn inconsistent_residues_return_none() {
        // x ≡ 0 (mod 4) and x ≡ 1 (mod 6) disagree modulo gcd(4, 6) = 2.
        assert_eq!(crt_pair(0, 4, 1, 6), None);
        assert_eq!(resolve_range(&[0, 1], &[4, 6]), None);
    }

    #[test]
    fn non_coprime_but_consistent_pair_resolves() {
        // x ≡ 1 (mod 4), x ≡ 3 (mod 6): consistent mod 2, lcm 12, solution 9.
        let (x, lcm) = crt_pair(1, 4, 3, 6).expect("consistent non-coprime pair");
        assert_eq!(lcm, 12);
        assert_eq!(x, 9);
        assert_eq!(x.rem_euclid(4), 1);
        assert_eq!(x.rem_euclid(6), 3);
        // combined_ambiguity is the lcm, not the product, for non-coprime moduli.
        assert_eq!(combined_ambiguity(&[4, 6]), 12);
    }

    #[test]
    fn degenerate_inputs_are_guarded() {
        assert_eq!(resolve_range(&[], &[]), None); // empty
        assert_eq!(resolve_range(&[5], &[7, 11]), None); // length mismatch
        assert_eq!(crt_pair(1, 0, 2, 5), None); // non-positive modulus
        assert_eq!(mod_inverse(3, 0), None); // zero modulus
        assert_eq!(combined_ambiguity(&[]), 1); // empty-lcm identity
    }
}
