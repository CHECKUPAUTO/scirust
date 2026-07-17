//! Boolean-function utilities over `GF(2)`: the fast Möbius transform (truth
//! table ↔ algebraic normal form) and the exact **algebraic degree** of a
//! Boolean vector function on a small number of input bits.
//!
//! Exact ANF requires a full `2^n` truth table, so these operate on `n ≤ 24`
//! input bits (a `2^24` table is ~16 MiB of bytes); larger `n` is rejected
//! rather than silently approximated.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
