//! Exact analysis of substitution boxes (S-boxes) — the standard cryptographic
//! quality metrics, computed exactly and composing [`crate::boolean`].
//!
//! An [`Sbox`] wraps a lookup table `S: {0,1}^n → {0,1}^m`. From it this module
//! computes, all exactly:
//!
//! - the **difference distribution table** (DDT) and the **differential
//!   uniformity** (resistance to differential cryptanalysis);
//! - the **linear approximation table** (LAT, via the Walsh–Hadamard transform)
//!   and the **linearity** / **nonlinearity** (resistance to linear
//!   cryptanalysis);
//! - the **algebraic degree** (max ANF degree over the nonzero component
//!   functions);
//! - bijectivity, fixed points, and the strict-avalanche dependence matrix.
//!
//! Everything is integer-only and deterministic. The full tables are `2^n × 2^m`,
//! so table-producing methods are limited to `n, m ≤ 12`; the scalar summaries
//! ([`Sbox::report`]) apply the same bound.

use crate::boolean;

/// The GF(2) inner product `⟨a, b⟩ = parity(a & b)`.
#[inline]
fn dot(a: u64, b: u64) -> u8 {
    ((a & b).count_ones() & 1) as u8
}

/// A substitution box `S: {0,1}^n → {0,1}^m` given by its lookup table.
#[derive(Clone, Debug)]
pub struct Sbox {
    n: u32,
    m: u32,
    table: Vec<u64>,
}

/// A structured, machine-readable summary of an S-box's cryptographic metrics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SboxReport {
    /// Input bits.
    pub input_bits: u32,
    /// Output bits.
    pub output_bits: u32,
    /// Whether the S-box is a bijection (`n == m` and a permutation).
    pub bijective: bool,
    /// Differential uniformity — `max_{a≠0, b} DDT[a][b]`. Lower is better.
    pub differential_uniformity: u32,
    /// Linearity — `max_{a, b≠0} |2·LAT[a][b]|`. Lower is better.
    pub linearity: u32,
    /// Nonlinearity — `2^{n-1} − linearity/2`. Higher is better.
    pub nonlinearity: u64,
    /// Algebraic degree — max ANF degree over nonzero output masks.
    pub algebraic_degree: u32,
    /// Number of fixed points `#{x : S(x) = x}` (`0` if `n ≠ m`).
    pub fixed_points: u32,
}

impl Sbox {
    /// Build an S-box from a lookup table of length `2^n`, each entry `< 2^m`.
    pub fn new(n: u32, m: u32, table: Vec<u64>) -> Self {
        assert_eq!(table.len(), 1usize << n, "table must have 2^n entries");
        assert!(m <= 64, "output width too large");
        assert!(
            table.iter().all(|&y| m == 64 || y < (1u64 << m)),
            "table entry out of output range"
        );
        Sbox { n, m, table }
    }

    /// Build an S-box from a closure `S(x)` for `x` in `0 … 2^n − 1`.
    pub fn from_fn(n: u32, m: u32, f: impl Fn(u64) -> u64) -> Self {
        Self::new(n, m, (0..1u64 << n).map(f).collect())
    }

    /// Input bit width `n`.
    pub fn input_bits(&self) -> u32 {
        self.n
    }
    /// Output bit width `m`.
    pub fn output_bits(&self) -> u32 {
        self.m
    }
    /// The output `S(x)`.
    pub fn apply(&self, x: u64) -> u64 {
        self.table[x as usize]
    }

    /// `true` iff `n == m` and the table is a permutation of `0 … 2^n − 1`.
    pub fn is_bijection(&self) -> bool {
        if self.n != self.m
        {
            return false;
        }
        let mut seen = vec![false; 1usize << self.n];
        for &y in &self.table
        {
            let y = y as usize;
            if seen[y]
            {
                return false;
            }
            seen[y] = true;
        }
        true
    }

    /// Number of fixed points `#{x : S(x) = x}`.
    pub fn fixed_points(&self) -> u32 {
        self.table
            .iter()
            .enumerate()
            .filter(|&(x, &y)| x as u64 == y)
            .count() as u32
    }

    /// The **difference distribution table**: `DDT[a][b] = #{x : S(x⊕a) ⊕ S(x) =
    /// b}`, a `2^n × 2^m` table. Requires `n, m ≤ 12`.
    pub fn ddt(&self) -> Vec<Vec<u32>> {
        assert!(self.n <= 12 && self.m <= 12, "DDT limited to n, m ≤ 12");
        let sz = 1usize << self.n;
        let msz = 1usize << self.m;
        let mut d = vec![vec![0u32; msz]; sz];
        for a in 0..sz
        {
            for x in 0..sz
            {
                let b = (self.table[x] ^ self.table[x ^ a]) as usize;
                d[a][b] += 1;
            }
        }
        d
    }

    /// Differential uniformity: `max_{a≠0, b} DDT[a][b]`. The `a = 0` row is
    /// excluded (it is trivially `2^n` at `b = 0`).
    pub fn differential_uniformity(&self) -> u32 {
        assert!(self.n <= 12 && self.m <= 12, "limited to n, m ≤ 12");
        let sz = 1usize << self.n;
        let mut counts = vec![0u32; 1usize << self.m];
        let mut du = 0u32;
        for a in 1..sz
        {
            counts.iter_mut().for_each(|c| *c = 0);
            for x in 0..sz
            {
                let b = (self.table[x] ^ self.table[x ^ a]) as usize;
                counts[b] += 1;
            }
            du = du.max(*counts.iter().max().unwrap());
        }
        du
    }

    /// The **linear approximation table**: `LAT[a][b] = #{x : ⟨a,x⟩ = ⟨b,S(x)⟩}
    /// − 2^{n-1}`, computed via the Walsh–Hadamard transform of each component
    /// `⟨b, S(·)⟩`. A `2^n × 2^m` table. Requires `n, m ≤ 12`.
    pub fn lat(&self) -> Vec<Vec<i32>> {
        assert!(self.n <= 12 && self.m <= 12, "LAT limited to n, m ≤ 12");
        let sz = 1usize << self.n;
        let msz = 1usize << self.m;
        let mut lat = vec![vec![0i32; msz]; sz];
        let mut tt = vec![0u8; sz];
        for b in 0..msz
        {
            for (x, cell) in tt.iter_mut().enumerate()
            {
                *cell = dot(b as u64, self.table[x]);
            }
            let w = boolean::walsh_hadamard(&tt, self.n);
            // W_b(a) = Σ_x (-1)^{⟨b,S(x)⟩ ⊕ ⟨a,x⟩} = 2·LAT[a][b]
            for (a, row) in lat.iter_mut().enumerate()
            {
                row[b] = (w[a] / 2) as i32;
            }
        }
        lat
    }

    /// Linearity: `max_{a, b≠0} |2·LAT[a][b]|` (the largest Walsh magnitude over
    /// nonzero output masks).
    pub fn linearity(&self) -> u32 {
        let sz = 1usize << self.n;
        let mut tt = vec![0u8; sz];
        let mut lin = 0u32;
        for b in 1..1u64 << self.m
        {
            for (x, cell) in tt.iter_mut().enumerate()
            {
                *cell = dot(b, self.table[x]);
            }
            let w = boolean::walsh_hadamard(&tt, self.n);
            let peak = w.iter().map(|&v| v.unsigned_abs()).max().unwrap_or(0);
            lin = lin.max(peak as u32);
        }
        lin
    }

    /// Nonlinearity: `2^{n-1} − linearity/2` — the minimum Hamming distance from
    /// any nonzero component function to the affine functions.
    pub fn nonlinearity(&self) -> u64 {
        (1u64 << (self.n - 1)) - (self.linearity() as u64) / 2
    }

    /// Algebraic degree: the maximum ANF degree over all nonzero output masks
    /// `b` of the component function `⟨b, S(·)⟩`.
    pub fn algebraic_degree(&self) -> u32 {
        let mut deg = 0u32;
        for b in 1..1u64 << self.m
        {
            let d = boolean::bitfn_degree(|x| dot(b, self.table[x as usize]) as u64, self.n, 1)
                .expect("n within exact ANF bound")
                .max_degree;
            deg = deg.max(d);
        }
        deg
    }

    /// The strict-avalanche dependence matrix: `sac[i][j] = #{x : bit j of
    /// (S(x) ⊕ S(x ⊕ e_i)) is 1}`. The strict avalanche criterion holds iff every
    /// entry equals `2^{n-1}`.
    pub fn sac_matrix(&self) -> Vec<Vec<u32>> {
        let sz = 1usize << self.n;
        let mut sac = vec![vec![0u32; self.m as usize]; self.n as usize];
        for i in 0..self.n as usize
        {
            let ei = 1u64 << i;
            for x in 0..sz
            {
                let diff = self.table[x] ^ self.table[x ^ ei as usize];
                for (j, row_cell) in sac[i].iter_mut().enumerate()
                {
                    *row_cell += ((diff >> j) & 1) as u32;
                }
            }
        }
        sac
    }

    /// `true` iff the strict avalanche criterion holds exactly (every
    /// [`sac_matrix`](Self::sac_matrix) entry is `2^{n-1}`).
    pub fn satisfies_sac(&self) -> bool {
        let half = 1u32 << (self.n - 1);
        self.sac_matrix()
            .iter()
            .all(|row| row.iter().all(|&c| c == half))
    }

    /// A structured report of the standard metrics. Requires `n, m ≤ 12`.
    pub fn report(&self) -> SboxReport {
        SboxReport {
            input_bits: self.n,
            output_bits: self.m,
            bijective: self.is_bijection(),
            differential_uniformity: self.differential_uniformity(),
            linearity: self.linearity(),
            nonlinearity: self.nonlinearity(),
            algebraic_degree: self.algebraic_degree(),
            fixed_points: self.fixed_points(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// A random permutation of 0..2^n (Fisher–Yates with a deterministic PRNG).
    fn random_permutation(n: u32, s: &mut u64) -> Vec<u64> {
        let sz = 1usize << n;
        let mut t: Vec<u64> = (0..sz as u64).collect();
        for i in (1..sz).rev()
        {
            let j = (xorshift(s) as usize) % (i + 1);
            t.swap(i, j);
        }
        t
    }

    /// The AES / Rijndael S-box (FIPS-197), for the canonical property checks.
    const AES_SBOX: [u8; 256] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
        0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4,
        0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71,
        0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
        0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6,
        0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
        0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
        0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
        0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44,
        0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a,
        0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49,
        0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
        0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
        0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
        0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1,
        0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
        0x16,
    ];

    #[test]
    fn ddt_row_sums_and_zero_row() {
        let mut s = 0x5b0d_1234u64;
        let sbox = Sbox::new(6, 6, random_permutation(6, &mut s));
        let d = sbox.ddt();
        let sz = 1usize << 6;
        // every row sums to 2^n
        for row in &d
        {
            assert_eq!(row.iter().sum::<u32>(), sz as u32);
        }
        // a = 0 row: all mass at b = 0
        assert_eq!(d[0][0], sz as u32);
        assert!(d[0][1..].iter().all(|&c| c == 0));
        // permutation ⇒ DDT entries are even
        assert!(d.iter().all(|row| row.iter().all(|&c| c % 2 == 0)));
    }

    #[test]
    fn lat_matches_brute_force() {
        let mut s = 0xa11ce_u64;
        for _ in 0..5
        {
            let n = 4u32;
            let sbox = Sbox::new(n, n, random_permutation(n, &mut s));
            let fast = sbox.lat();
            // brute-force LAT
            let sz = 1usize << n;
            let half = (1i32 << n) / 2;
            for a in 0..sz
            {
                for b in 0..sz
                {
                    let mut agree = 0i32;
                    for x in 0..sz
                    {
                        if dot(a as u64, x as u64) == dot(b as u64, sbox.table[x])
                        {
                            agree += 1;
                        }
                    }
                    assert_eq!(fast[a][b], agree - half, "LAT mismatch a={a} b={b}");
                }
            }
        }
    }

    #[test]
    fn aes_sbox_canonical_properties() {
        let table: Vec<u64> = AES_SBOX.iter().map(|&b| b as u64).collect();
        let sbox = Sbox::new(8, 8, table);
        let r = sbox.report();
        assert!(r.bijective, "AES S-box is a bijection");
        assert_eq!(r.differential_uniformity, 4, "AES DU is 4");
        assert_eq!(r.linearity, 32, "AES linearity is 32");
        assert_eq!(r.nonlinearity, 112, "AES nonlinearity is 112");
        assert_eq!(r.algebraic_degree, 7, "AES algebraic degree is 7");
        assert_eq!(r.fixed_points, 0, "AES S-box has no fixed points");
    }

    #[test]
    fn linear_sbox_is_weak() {
        // S(x) = x (identity) is affine: nonlinearity 0, degree 1, and every
        // difference propagates deterministically (DU = 2^n).
        let sbox = Sbox::from_fn(6, 6, |x| x);
        assert_eq!(sbox.nonlinearity(), 0);
        assert_eq!(sbox.algebraic_degree(), 1);
        assert_eq!(sbox.differential_uniformity(), 1 << 6);
        assert!(sbox.is_bijection());
        assert_eq!(sbox.fixed_points(), 1 << 6);
    }

    #[test]
    fn nonlinearity_agrees_with_boolean_components() {
        // Sbox nonlinearity == min over nonzero output masks of the component's
        // Boolean nonlinearity.
        let mut s = 0x7e57u64;
        let n = 5u32;
        let sbox = Sbox::new(n, n, random_permutation(n, &mut s));
        let sz = 1usize << n;
        let mut min_nl = u64::MAX;
        for b in 1..1u64 << n
        {
            let tt: Vec<u8> = (0..sz).map(|x| dot(b, sbox.table[x])).collect();
            min_nl = min_nl.min(boolean::nonlinearity(&tt, n));
        }
        assert_eq!(sbox.nonlinearity(), min_nl);
    }

    #[test]
    fn sac_matrix_shape_and_bounds() {
        let mut s = 0x5ac0u64;
        let sbox = Sbox::new(6, 6, random_permutation(6, &mut s));
        let sac = sbox.sac_matrix();
        assert_eq!(sac.len(), 6);
        let cap = 1u32 << 6;
        for row in &sac
        {
            assert_eq!(row.len(), 6);
            assert!(row.iter().all(|&c| c <= cap));
        }
    }
}
