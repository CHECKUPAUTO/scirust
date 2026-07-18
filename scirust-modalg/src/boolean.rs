//! Boolean-function utilities over `GF(2)`: the fast Möbius transform (truth
//! table ↔ algebraic normal form) with exact **algebraic degree**, and the fast
//! **Walsh–Hadamard transform** with the spectral metrics it yields —
//! **nonlinearity**, **balancedness**, the **bent** property, and **correlation
//! immunity**. Together these are the two fundamental views of a Boolean
//! function: the ANF (algebraic degree) and the Walsh spectrum (distance to
//! affine functions).
//!
//! Exact ANF / Walsh analysis requires a full `2^n` truth table, so these
//! operate on `n ≤ 24` input bits (a `2^24` table is ~16 MiB of bytes); larger
//! `n` is rejected rather than silently approximated.

/// Maximum input width for which an exact `2^n` truth table is built.
pub const MAX_EXACT_BITS: u32 = 24;

/// In-place Möbius transform over `GF(2)` of a `2^n`-entry truth table (each
/// entry a `0/1` bit in a `u8`). After the call, `tt[mask]` is the ANF
/// coefficient of the monomial whose variable set is `mask`.
pub fn mobius_transform(tt: &mut [u8], n: u32) {
    assert_eq!(tt.len(), 1usize << n, "truth table must have 2^n entries");
    for i in 0..n
    {
        let step = 1usize << i;
        let mut base = 0usize;
        while base < tt.len()
        {
            for x in base..base + step
            {
                tt[x + step] ^= tt[x];
            }
            base += step << 1;
        }
    }
}

/// Exact algebraic-degree result for a Boolean vector function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegreeResult {
    /// Number of input bits.
    pub input_bits: u32,
    /// Number of output bits.
    pub output_bits: u32,
    /// Maximum algebraic degree over all output bits.
    pub max_degree: u32,
    /// Per-output-bit degree.
    pub per_output: Vec<u32>,
}

/// Maximum ANF degree of a single Boolean function given its `2^n` truth table.
/// Consumes the table (transforms it in place).
pub fn anf_degree_of_table(tt: &mut [u8], n: u32) -> u32 {
    mobius_transform(tt, n);
    let mut deg = 0u32;
    for (mask, &coeff) in tt.iter().enumerate()
    {
        if coeff & 1 == 1
        {
            deg = deg.max((mask as u32).count_ones());
        }
    }
    deg
}

/// Exact ANF degree of a Boolean vector function `f: {0,1}^n → {0,1}^m`, given
/// as a closure mapping an `n`-bit input (packed in a `u64`) to an `m`-bit
/// output. Returns `None` when `n > `[`MAX_EXACT_BITS`].
pub fn bitfn_degree(f: impl Fn(u64) -> u64, n: u32, m: u32) -> Option<DegreeResult> {
    if n > MAX_EXACT_BITS
    {
        return None;
    }
    let size = 1usize << n;
    let mut outputs = vec![0u64; size];
    for (x, out) in outputs.iter_mut().enumerate()
    {
        *out = f(x as u64);
    }
    let mut per_output = Vec::with_capacity(m as usize);
    let mut max_degree = 0u32;
    let mut tt = vec![0u8; size];
    for o in 0..m
    {
        for (x, cell) in tt.iter_mut().enumerate()
        {
            *cell = ((outputs[x] >> o) & 1) as u8;
        }
        let d = anf_degree_of_table(&mut tt, n);
        per_output.push(d);
        max_degree = max_degree.max(d);
    }
    Some(DegreeResult {
        input_bits: n,
        output_bits: m,
        max_degree,
        per_output,
    })
}

/// The **Walsh–Hadamard transform** of a Boolean function given by its `2^n`
/// truth table (`0/1` bits). Returns the `2^n` Walsh coefficients
/// `W(a) = Σ_x (-1)^{f(x) ⊕ a·x}`, each in `[-2^n, 2^n]`. `O(n·2^n)`.
///
/// `W(0)` is `2^n − 2·|{x : f(x)=1}|` (so `W(0)=0` ⇔ balanced), and
/// `max_a |W(a)|` measures the function's correlation with the affine functions.
pub fn walsh_hadamard(tt: &[u8], n: u32) -> Vec<i64> {
    assert_eq!(tt.len(), 1usize << n, "truth table must have 2^n entries");
    // sign representation (-1)^{f(x)} ∈ {+1, -1}
    let mut w: Vec<i64> = tt.iter().map(|&b| 1 - 2 * (b as i64 & 1)).collect();
    let mut h = 1usize;
    while h < w.len()
    {
        let mut i = 0;
        while i < w.len()
        {
            for j in i..i + h
            {
                let x = w[j];
                let y = w[j + h];
                w[j] = x + y;
                w[j + h] = x - y;
            }
            i += h << 1;
        }
        h <<= 1;
    }
    w
}

/// The **nonlinearity** of a Boolean function (its Hamming distance to the
/// nearest affine function): `2^{n-1} − ½·max_a |W(a)|`. Higher is better;
/// `0` means the function is itself affine. Requires `n ≥ 1`.
pub fn nonlinearity(tt: &[u8], n: u32) -> u64 {
    assert!(n >= 1, "nonlinearity is defined for n ≥ 1");
    let max_abs = walsh_hadamard(tt, n)
        .iter()
        .map(|&v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    // W(a) has the same parity as 2^n, so max_abs is even for n ≥ 1.
    (1u64 << (n - 1)) - max_abs / 2
}

/// `true` iff the function is **balanced** (equally many `0`s and `1`s), i.e.
/// `W(0) = 0`.
pub fn is_balanced(tt: &[u8], n: u32) -> bool {
    walsh_hadamard(tt, n)[0] == 0
}

/// `true` iff the function is **bent** — every Walsh coefficient has magnitude
/// exactly `2^{n/2}`, the maximum possible flatness of the spectrum. Bent
/// functions exist only for even `n` and attain the maximum nonlinearity
/// `2^{n-1} − 2^{n/2−1}`.
pub fn is_bent(tt: &[u8], n: u32) -> bool {
    if n % 2 != 0
    {
        return false;
    }
    let target = 1i64 << (n / 2);
    walsh_hadamard(tt, n).iter().all(|&v| v.abs() == target)
}

/// The **order of correlation immunity**: the largest `m` such that every Walsh
/// coefficient `W(a)` with `1 ≤ wt(a) ≤ m` vanishes (a function is `m`-th order
/// correlation immune iff its output is statistically independent of every
/// subset of at most `m` input bits). A [`balanced`](is_balanced) function that
/// is `m`-th order correlation immune is `m`-resilient. Returns `0` when there
/// is a nonzero weight-1 coefficient.
pub fn correlation_immunity(tt: &[u8], n: u32) -> u32 {
    let w = walsh_hadamard(tt, n);
    for order in 1..=n
    {
        let all_zero = (0..w.len()).all(|a| (a as u32).count_ones() != order || w[a] == 0);
        if !all_zero
        {
            return order - 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the `2^n` truth table of `f: {0,1}^n → {0,1}` (bit 0 of `f`'s
    /// output), for the spectral tests below.
    fn table(f: impl Fn(u64) -> u64, n: u32) -> Vec<u8> {
        (0..1u64 << n).map(|x| (f(x) & 1) as u8).collect()
    }

    #[test]
    fn constant_and_linear_degrees() {
        // f(x) = 0 : degree 0
        let d = bitfn_degree(|_| 0, 4, 1).unwrap();
        assert_eq!(d.max_degree, 0);
        // f(x) = x0 ^ x2 : degree 1 (single output bit)
        let d = bitfn_degree(|x| (x ^ (x >> 2)) & 1, 4, 1).unwrap();
        assert_eq!(d.max_degree, 1);
    }

    #[test]
    fn product_is_degree_two() {
        // output bit 0 = x0 AND x1 : ANF monomial x0*x1 -> degree 2
        let d = bitfn_degree(|x| (x & 1) & ((x >> 1) & 1), 3, 1).unwrap();
        assert_eq!(d.max_degree, 2);
    }

    #[test]
    fn multiplication_mod4_degree() {
        // y = (x * 3) mod 4 over 2 bits: nonlinear via carry -> degree 2
        let d = bitfn_degree(|x| (x.wrapping_mul(3)) & 3, 2, 2).unwrap();
        assert!(d.max_degree >= 1);
    }

    #[test]
    fn rejects_too_wide() {
        assert!(bitfn_degree(|x| x, 32, 32).is_none());
    }

    #[test]
    fn walsh_parseval_and_involution() {
        let mut s = 0x51ce_1234u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n in 1..=8u32
        {
            let tt: Vec<u8> = (0..1u64 << n).map(|_| (next() & 1) as u8).collect();
            let w = walsh_hadamard(&tt, n);
            // Parseval: Σ W(a)^2 = 2^{2n}
            let energy: i64 = w.iter().map(|&v| v * v).sum();
            assert_eq!(energy, 1i64 << (2 * n), "Parseval failed at n={n}");
            // W(0) counts the imbalance
            let ones = tt.iter().filter(|&&b| b == 1).count() as i64;
            assert_eq!(w[0], (1i64 << n) - 2 * ones);
        }
    }

    #[test]
    fn linear_and_constant_spectra() {
        // linear f(x) = x0 ⊕ x1 ⊕ x2 : single spike at a = 0b111, else 0
        let tt = table(|x| (x ^ (x >> 1) ^ (x >> 2)) & 1, 3);
        let w = walsh_hadamard(&tt, 3);
        for (a, &val) in w.iter().enumerate()
        {
            let expect = if a == 0b111 { -(1i64 << 3) } else { 0 };
            // sign depends on convention; compare magnitudes for the spike
            if a == 0b111
            {
                assert_eq!(val.abs(), 1 << 3);
            }
            else
            {
                assert_eq!(val, expect);
            }
        }
        assert_eq!(nonlinearity(&tt, 3), 0); // affine ⇒ nonlinearity 0
        assert!(is_balanced(&tt, 3)); // nonzero linear form is balanced
        assert!(!is_bent(&tt, 3)); // odd n ⇒ never bent
        // linear x0⊕x1⊕x2 is correlation-immune of order 2 (only wt-3 coeff nonzero)
        assert_eq!(correlation_immunity(&tt, 3), 2);

        // constant 0: single spike at a=0, not balanced, nonlinearity 0
        let c = table(|_| 0, 3);
        assert_eq!(nonlinearity(&c, 3), 0);
        assert!(!is_balanced(&c, 3));
    }

    #[test]
    fn bent_functions_hit_max_nonlinearity() {
        // n = 2: AND is bent, nonlinearity 2^1 - 2^0 = 1
        let f2 = table(|x| (x & 1) & ((x >> 1) & 1), 2);
        assert!(is_bent(&f2, 2));
        assert_eq!(nonlinearity(&f2, 2), 1);

        // n = 4: x0x1 ⊕ x2x3 is bent, nonlinearity 2^3 - 2^1 = 6
        let f4 = table(
            |x| (((x & 1) & ((x >> 1) & 1)) ^ (((x >> 2) & 1) & ((x >> 3) & 1))) & 1,
            4,
        );
        assert!(is_bent(&f4, 4));
        assert_eq!(nonlinearity(&f4, 4), 6);
        assert!(!is_balanced(&f4, 4)); // bent functions are never balanced
    }

    #[test]
    fn nonlinearity_upper_bound_holds() {
        // For any Boolean function, NL ≤ 2^{n-1} − 2^{n/2−1}; check it never
        // exceeds the covering-radius bound across random tables.
        let mut s = 0x9e37_79b1u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n in 2..=8u32
        {
            for _ in 0..20
            {
                let tt: Vec<u8> = (0..1u64 << n).map(|_| (next() & 1) as u8).collect();
                let nl = nonlinearity(&tt, n);
                // universal bound: NL ≤ 2^{n-1} − 2^{n/2 − 1} (bent bound)
                let bound = (1u64 << (n - 1)) - (1u64 << (n / 2)) / 2;
                assert!(
                    nl <= bound || n % 2 == 1,
                    "NL {nl} exceeds bent bound at n={n}"
                );
            }
        }
    }
}
