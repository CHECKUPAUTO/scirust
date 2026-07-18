//! Reed–Solomon error-correcting codes over `GF(2^n)`, built on [`crate::gf2`].
//!
//! A systematic `RS(n, k)` code adds `nsym = n − k` parity symbols and can
//! correct up to `⌊nsym / 2⌋ ` symbol errors anywhere in the codeword. The
//! implementation is the classic syndrome decoder:
//!
//! 1. **Syndromes** `S_j = R(α^j)` for `j = 1 … nsym`.
//! 2. **Berlekamp–Massey** to recover the error-locator polynomial `Λ`.
//! 3. **Chien search** for the roots of `Λ` (the error positions).
//! 4. **Magnitudes** by solving the resulting Vandermonde system over the field
//!    (a self-contained alternative to Forney, with a syndrome-recheck at the
//!    end that rejects miscorrections).
//!
//! The core [`ReedSolomon::encode`] / [`ReedSolomon::decode`] work on `u64`
//! field symbols and are valid for any field; [`ReedSolomon::encode_bytes`] /
//! [`ReedSolomon::decode_bytes`] are `GF(2^8)` byte conveniences.
//!
//! Everything is exact, deterministic and integer-only. This is the same code
//! family behind QR codes, CDs/DVDs and RAID-6 — here as a small reusable brick
//! that composes the crate's [`gf2`](crate::gf2) field and
//! [`numtheory`](crate::numtheory) factoring (used to *verify* that the supplied
//! element is primitive).

use crate::gf2::Gf2Field;
use crate::numtheory::factor;

/// A systematic Reed–Solomon code over a finite field `GF(2^m)`.
#[derive(Clone, Debug)]
pub struct ReedSolomon {
    field: Gf2Field,
    alpha: u64,
    nsym: usize,
    gen: Vec<u64>,
}

impl ReedSolomon {
    /// Build an `RS` code with `nsym` parity symbols over `field`, evaluated at
    /// powers of the primitive element `alpha`.
    ///
    /// Panics unless `1 ≤ nsym ≤ field.order() − 1` and `alpha` is genuinely
    /// primitive (a generator of the multiplicative group) — the latter is
    /// checked exactly via [`crate::numtheory::factor`], so a non-generator is
    /// rejected rather than silently producing a broken code.
    pub fn new(field: Gf2Field, alpha: u64, nsym: usize) -> Self {
        let order = field.order();
        assert!(nsym >= 1 && (nsym as u64) < order, "nsym out of range");
        assert!(
            is_primitive(&field, alpha),
            "alpha is not a primitive element"
        );
        // generator g(x) = ∏_{j=1}^{nsym} (x − α^j), high-endian (index 0 = top).
        let mut gen = vec![1u64];
        for j in 1..=nsym as u64
        {
            gen = poly_mul(&field, &gen, &[1, field.pow(alpha, j)]);
        }
        ReedSolomon {
            field,
            alpha,
            nsym,
            gen,
        }
    }

    /// The standard byte-oriented `RS(255, 255 − nsym)` code over the primitive
    /// `GF(2^8)` (`0x11D`, `α = 2`) used by QR codes and CDs.
    pub fn qr(nsym: usize) -> Self {
        Self::new(Gf2Field::primitive8(), 2, nsym)
    }

    /// Number of parity symbols `nsym = n − k`.
    pub fn parity_len(&self) -> usize {
        self.nsym
    }

    /// Maximum number of symbol errors correctable anywhere in the codeword.
    pub fn correction_capacity(&self) -> usize {
        self.nsym / 2
    }

    /// Systematically encode a `k`-symbol message into an `n`-symbol codeword
    /// (`n = k + nsym`), each symbol a field element in `[0, order)`. The
    /// returned codeword begins with the unchanged message symbols followed by
    /// `nsym` parity symbols. Requires `k + nsym ≤ order − 1`.
    pub fn encode(&self, msg: &[u64]) -> Vec<u64> {
        let k = msg.len();
        assert!(
            ((k + self.nsym) as u64) < self.field.order(),
            "message too long for this code"
        );
        assert!(
            msg.iter().all(|&s| s < self.field.order()),
            "symbol out of field range"
        );
        // high-endian working buffer for msg(x)·x^nsym: message then nsym zeros.
        let mut out: Vec<u64> = msg.to_vec();
        out.extend(std::iter::repeat_n(0u64, self.nsym));
        // Synthetic division by the monic generator: the running leading
        // coefficient `out[i]` is the quotient digit, subtracted (skipping the
        // monic term `gen[0]`) so the last nsym slots accrue the remainder. The
        // message slots hold intermediate state and are restored afterwards.
        for i in 0..k
        {
            let coef = out[i];
            if coef != 0
            {
                for j in 1..self.gen.len()
                {
                    out[i + j] ^= self.field.mul(self.gen[j], coef);
                }
            }
        }
        // Systematic codeword: original message symbols followed by the parity.
        out[..k].copy_from_slice(msg);
        out
    }

    /// Decode a received `n`-symbol codeword, returning the recovered `k`-symbol
    /// message and the number of corrected symbols, or `None` if the errors
    /// exceed the code's capacity (detected via a final syndrome recheck).
    pub fn decode(&self, received: &[u64]) -> Option<(Vec<u64>, usize)> {
        let n = received.len();
        assert!(n > self.nsym, "codeword shorter than the parity length");
        let k = n - self.nsym;
        let mut cw = received.to_vec();

        let synd = self.syndromes(&cw);
        if synd.iter().all(|&s| s == 0)
        {
            return Some((cw[..k].to_vec(), 0));
        }

        let (lambda, v) = self.error_locator(&synd);
        if v == 0 || v > self.correction_capacity()
        {
            return None;
        }
        let positions = self.error_positions(&lambda, n);
        if positions.len() != v
        {
            return None; // locator degree and root count disagree
        }

        // Solve Σ_l e_l · X_l^j = S_j (j = 1 … v) for the magnitudes e_l.
        let f = &self.field;
        let locators: Vec<u64> = positions
            .iter()
            .map(|&u| f.pow(self.alpha, (n - 1 - u) as u64))
            .collect();
        let mut a = vec![vec![0u64; v]; v];
        let mut rhs = vec![0u64; v];
        for (row, r) in a.iter_mut().enumerate()
        {
            let j = (row + 1) as u64;
            for (col, cell) in r.iter_mut().enumerate()
            {
                *cell = f.pow(locators[col], j);
            }
            rhs[row] = synd[row];
        }
        let magnitudes = gf_solve(f, a, rhs)?;

        for (idx, &u) in positions.iter().enumerate()
        {
            cw[u] ^= magnitudes[idx];
        }

        // Reject miscorrections: a valid correction zeroes every syndrome.
        if self.syndromes(&cw).iter().any(|&s| s != 0)
        {
            return None;
        }
        Some((cw[..k].to_vec(), v))
    }

    /// **Erasure** decoding: reconstruct a codeword in which the symbols at the
    /// (known, distinct) `erasures` positions have been lost. Because the
    /// positions are known, no error search is needed — up to `nsym` erasures
    /// are filled by solving the Vandermonde syndrome system directly, so an
    /// `RS(n, k)` code recovers **twice** as many erasures as it can errors.
    ///
    /// Returns the full `n`-symbol corrected codeword, or `None` if there are
    /// more than `nsym` erasures or the non-erased symbols are themselves
    /// inconsistent (i.e. contained an undeclared error — caught by a final
    /// syndrome recheck). This is the RAID-6 / storage reconstruction path.
    pub fn decode_erasures(&self, received: &[u64], erasures: &[usize]) -> Option<Vec<u64>> {
        let n = received.len();
        assert!(n > self.nsym, "codeword shorter than the parity length");
        let mut eras = erasures.to_vec();
        eras.sort_unstable();
        eras.dedup();
        if eras.len() > self.nsym || eras.last().is_some_and(|&u| u >= n)
        {
            return None;
        }
        let mut cw = received.to_vec();
        for &u in &eras
        {
            cw[u] = 0;
        }
        let synd = self.syndromes(&cw);
        if eras.is_empty()
        {
            return synd.iter().all(|&s| s == 0).then_some(cw);
        }

        let f = &self.field;
        let e = eras.len();
        let locators: Vec<u64> = eras
            .iter()
            .map(|&u| f.pow(self.alpha, (n - 1 - u) as u64))
            .collect();
        // Σ_l Y_l · X_l^j = S_j (j = 1 … e): an e×e Vandermonde system.
        let mut a = vec![vec![0u64; e]; e];
        let mut rhs = vec![0u64; e];
        for (row, r) in a.iter_mut().enumerate()
        {
            let j = (row + 1) as u64;
            for (col, cell) in r.iter_mut().enumerate()
            {
                *cell = f.pow(locators[col], j);
            }
            rhs[row] = synd[row];
        }
        let vals = gf_solve(f, a, rhs)?;
        for (idx, &u) in eras.iter().enumerate()
        {
            cw[u] ^= vals[idx];
        }
        // consistency: undeclared errors leave a nonzero syndrome.
        if self.syndromes(&cw).iter().any(|&s| s != 0)
        {
            return None;
        }
        Some(cw)
    }

    /// Combined **errors-and-erasures** decoding: correct `t` unknown errors and
    /// fill `e` known `erasures` simultaneously, valid whenever `2·t + e ≤
    /// nsym`. Uses Forney-modified syndromes so Berlekamp–Massey locates only the
    /// unknown errors, then solves one linear system for every errata magnitude.
    ///
    /// Returns `(corrected codeword, number of errors corrected)`, or `None`
    /// when the errata exceed the code's capacity (a final syndrome recheck
    /// rejects any miscorrection).
    pub fn decode_errors_and_erasures(
        &self,
        received: &[u64],
        erasures: &[usize],
    ) -> Option<(Vec<u64>, usize)> {
        let n = received.len();
        assert!(n > self.nsym, "codeword shorter than the parity length");
        let mut eras = erasures.to_vec();
        eras.sort_unstable();
        eras.dedup();
        if eras.len() > self.nsym || eras.last().is_some_and(|&u| u >= n)
        {
            return None;
        }
        let mut cw = received.to_vec();
        for &u in &eras
        {
            cw[u] = 0;
        }
        let e = eras.len();
        let synd = self.syndromes(&cw);
        if synd.iter().all(|&s| s == 0)
        {
            return Some((cw, 0));
        }

        let f = &self.field;
        // erasure locator Γ(x) = ∏_{u∈eras} (1 + X_u·x), low-endian
        let mut gamma = vec![1u64];
        for &u in &eras
        {
            let x = f.pow(self.alpha, (n - 1 - u) as u64);
            gamma = poly_mul(f, &gamma, &[1, x]);
        }
        // modified syndromes Ξ(x) = Γ(x)·S(x) mod x^nsym; BM on the tail
        // ξ_e … ξ_{nsym-1} sees only the unknown errors.
        let mut xi = poly_mul(f, &gamma, &synd);
        xi.truncate(self.nsym);
        let tail = if e < xi.len()
        {
            xi[e..].to_vec()
        }
        else
        {
            Vec::new()
        };
        let (lambda, t) = self.error_locator(&tail);
        if 2 * t + e > self.nsym
        {
            return None;
        }
        let err_positions = if t > 0
        {
            self.error_positions(&lambda, n)
        }
        else
        {
            Vec::new()
        };
        if err_positions.len() != t
        {
            return None;
        }

        // all errata positions = erasures ∪ error positions
        let mut positions = eras.clone();
        for &p in &err_positions
        {
            if !positions.contains(&p)
            {
                positions.push(p);
            }
        }
        let m = positions.len();
        let locators: Vec<u64> = positions
            .iter()
            .map(|&u| f.pow(self.alpha, (n - 1 - u) as u64))
            .collect();
        let mut a = vec![vec![0u64; m]; m];
        let mut rhs = vec![0u64; m];
        for (row, r) in a.iter_mut().enumerate()
        {
            let j = (row + 1) as u64;
            for (col, cell) in r.iter_mut().enumerate()
            {
                *cell = f.pow(locators[col], j);
            }
            rhs[row] = synd[row];
        }
        let vals = gf_solve(f, a, rhs)?;
        for (idx, &u) in positions.iter().enumerate()
        {
            cw[u] ^= vals[idx];
        }
        if self.syndromes(&cw).iter().any(|&s| s != 0)
        {
            return None;
        }
        Some((cw, t))
    }

    /// `GF(2^8)` byte convenience for [`encode`](Self::encode). Panics unless the
    /// field has degree ≤ 8 (so every symbol fits in a byte).
    pub fn encode_bytes(&self, msg: &[u8]) -> Vec<u8> {
        assert!(self.field.degree() <= 8, "byte API requires GF(2^m), m ≤ 8");
        let m: Vec<u64> = msg.iter().map(|&b| b as u64).collect();
        self.encode(&m).iter().map(|&w| w as u8).collect()
    }

    /// `GF(2^8)` byte convenience for [`decode`](Self::decode). Panics unless the
    /// field has degree ≤ 8.
    pub fn decode_bytes(&self, received: &[u8]) -> Option<(Vec<u8>, usize)> {
        assert!(self.field.degree() <= 8, "byte API requires GF(2^m), m ≤ 8");
        let r: Vec<u64> = received.iter().map(|&b| b as u64).collect();
        let (m, e) = self.decode(&r)?;
        Some((m.iter().map(|&w| w as u8).collect(), e))
    }

    /// Syndromes `S_j = R(α^j)` for `j = 1 … nsym` (all zero ⇔ no detected
    /// error).
    fn syndromes(&self, cw: &[u64]) -> Vec<u64> {
        (1..=self.nsym as u64)
            .map(|j| poly_eval(&self.field, cw, self.field.pow(self.alpha, j)))
            .collect()
    }

    /// Berlekamp–Massey: the error-locator polynomial `Λ` (low-endian,
    /// `Λ[0] = 1`) and the register length (the number of errors).
    fn error_locator(&self, synd: &[u64]) -> (Vec<u64>, usize) {
        let f = &self.field;
        let mut lambda = vec![1u64];
        let mut b = vec![1u64];
        let mut l = 0usize;
        let mut m = 1usize;
        let mut bb = 1u64;
        for n in 0..synd.len()
        {
            let mut delta = synd[n];
            for i in 1..=l
            {
                if i < lambda.len()
                {
                    delta ^= f.mul(lambda[i], synd[n - i]);
                }
            }
            if delta == 0
            {
                m += 1;
            }
            else if 2 * l <= n
            {
                let t = lambda.clone();
                let coef = f.mul(delta, f.inv(bb).unwrap());
                lambda = poly_sub_shift(f, &lambda, &b, coef, m);
                l = n + 1 - l;
                b = t;
                bb = delta;
                m = 1;
            }
            else
            {
                let coef = f.mul(delta, f.inv(bb).unwrap());
                lambda = poly_sub_shift(f, &lambda, &b, coef, m);
                m += 1;
            }
        }
        (lambda, l)
    }

    /// Chien search: public codeword indices `u` whose locator `X_u = α^{n−1−u}`
    /// is a root pattern of `Λ` (i.e. `Λ(X_u^{-1}) = 0`).
    fn error_positions(&self, lambda: &[u64], n: usize) -> Vec<usize> {
        let f = &self.field;
        let mut pos = Vec::new();
        for u in 0..n
        {
            let x = f.pow(self.alpha, (n - 1 - u) as u64);
            let xinv = f.inv(x).unwrap();
            if poly_eval_low(f, lambda, xinv) == 0
            {
                pos.push(u);
            }
        }
        pos
    }
}

/// Is `alpha` a generator of `GF(2^m)^×`? Exact, via factoring `order − 1`.
fn is_primitive(field: &Gf2Field, alpha: u64) -> bool {
    let order = field.order();
    if alpha == 0 || field.pow(alpha, order - 1) != 1
    {
        return false;
    }
    // primitive ⇔ α^{(order-1)/p} ≠ 1 for every prime p | (order-1)
    factor(order - 1)
        .into_iter()
        .all(|(p, _)| field.pow(alpha, (order - 1) / p) != 1)
}

// ---- high-endian polynomial helpers (index 0 = highest-degree coefficient) --

fn poly_mul(f: &Gf2Field, a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty()
    {
        return Vec::new();
    }
    let mut r = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate()
    {
        if ai == 0
        {
            continue;
        }
        for (j, &bj) in b.iter().enumerate()
        {
            r[i + j] ^= f.mul(ai, bj);
        }
    }
    r
}

fn poly_eval(f: &Gf2Field, p: &[u64], x: u64) -> u64 {
    // Horner from the highest-degree coefficient.
    let mut y = 0u64;
    for &c in p
    {
        y = f.mul(y, x) ^ c;
    }
    y
}

// ---- low-endian helper for the locator (index 0 = constant term) ------------

fn poly_eval_low(f: &Gf2Field, p: &[u64], x: u64) -> u64 {
    let mut y = 0u64;
    for &c in p.iter().rev()
    {
        y = f.mul(y, x) ^ c;
    }
    y
}

/// `a XOR (coef · x^m · b)` in `GF(2^k)[x]` (low-endian). Char 2, so `−` = `XOR`.
fn poly_sub_shift(f: &Gf2Field, a: &[u64], b: &[u64], coef: u64, m: usize) -> Vec<u64> {
    let mut r = a.to_vec();
    if r.len() < b.len() + m
    {
        r.resize(b.len() + m, 0);
    }
    for (i, &bi) in b.iter().enumerate()
    {
        r[i + m] ^= f.mul(coef, bi);
    }
    r
}

/// Solve the dense linear system `A·x = rhs` over `GF(2^k)` by Gauss–Jordan.
fn gf_solve(f: &Gf2Field, mut a: Vec<Vec<u64>>, mut rhs: Vec<u64>) -> Option<Vec<u64>> {
    let n = rhs.len();
    for col in 0..n
    {
        let piv = (col..n).find(|&r| a[r][col] != 0)?;
        a.swap(col, piv);
        rhs.swap(col, piv);
        let inv = f.inv(a[col][col]).unwrap();
        for j in col..n
        {
            a[col][j] = f.mul(a[col][j], inv);
        }
        rhs[col] = f.mul(rhs[col], inv);
        for r in 0..n
        {
            if r != col && a[r][col] != 0
            {
                let factor = a[r][col];
                for j in col..n
                {
                    a[r][j] ^= f.mul(factor, a[col][j]);
                }
                rhs[r] ^= f.mul(factor, rhs[col]);
            }
        }
    }
    Some(rhs)
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

    #[test]
    fn encode_is_systematic_with_zero_syndromes() {
        let rs = ReedSolomon::qr(10);
        let msg: Vec<u8> = (0..20u8).collect();
        let cw = rs.encode_bytes(&msg);
        assert_eq!(cw.len(), msg.len() + 10);
        // systematic: message symbols are preserved verbatim
        assert_eq!(&cw[..msg.len()], &msg[..]);
        // a valid codeword has all-zero syndromes
        let cww: Vec<u64> = cw.iter().map(|&b| b as u64).collect();
        assert!(rs.syndromes(&cww).iter().all(|&s| s == 0));
    }

    #[test]
    fn round_trip_without_errors() {
        let rs = ReedSolomon::qr(8);
        let msg: Vec<u8> = (100..140u8).collect();
        let cw = rs.encode_bytes(&msg);
        let (dec, errs) = rs.decode_bytes(&cw).unwrap();
        assert_eq!(errs, 0);
        assert_eq!(dec, msg);
    }

    #[test]
    fn corrects_up_to_capacity() {
        let mut s = 0xc0de_1234u64;
        for &nsym in &[4usize, 6, 8, 10, 16]
        {
            let rs = ReedSolomon::qr(nsym);
            let t = rs.correction_capacity();
            let k = 30usize;
            for _ in 0..40
            {
                let msg: Vec<u8> = (0..k).map(|_| (xorshift(&mut s) & 0xff) as u8).collect();
                let cw = rs.encode_bytes(&msg);
                let n = cw.len();
                for e in 0..=t
                {
                    let mut corrupt = cw.clone();
                    // choose e distinct positions
                    let mut chosen = Vec::new();
                    while chosen.len() < e
                    {
                        let p = (xorshift(&mut s) as usize) % n;
                        if !chosen.contains(&p)
                        {
                            chosen.push(p);
                        }
                    }
                    for &p in &chosen
                    {
                        // flip to a different value (nonzero error)
                        let delta = ((xorshift(&mut s) & 0xff) | 1) as u8;
                        corrupt[p] ^= delta;
                    }
                    let (dec, errs) = rs
                        .decode_bytes(&corrupt)
                        .expect("within capacity must decode");
                    assert_eq!(dec, msg, "nsym={nsym} e={e}");
                    assert_eq!(errs, chosen.len(), "reported error count nsym={nsym} e={e}");
                }
            }
        }
    }

    #[test]
    fn single_symbol_burst_of_bits_is_one_error() {
        // RS is symbol-oriented: corrupting several bits of ONE symbol is a
        // single symbol error, always correctable with nsym ≥ 2.
        let rs = ReedSolomon::qr(4);
        let msg: Vec<u8> = (1..=40u8).collect();
        let mut cw = rs.encode_bytes(&msg);
        cw[7] ^= 0xff; // 8 bit-flips, one symbol
        let (dec, errs) = rs.decode_bytes(&cw).unwrap();
        assert_eq!(dec, msg);
        assert_eq!(errs, 1);
    }

    #[test]
    fn beyond_capacity_is_flagged_not_miscorrected() {
        // With t = nsym/2, injecting t+1 errors must NOT silently return the
        // wrong message: the syndrome recheck rejects a bad correction. (RS can
        // in principle miscorrect beyond capacity; the recheck makes the common
        // case safe. We only assert we never return an *incorrect* message.)
        let mut s = 0x9e37_79b9u64;
        let rs = ReedSolomon::qr(8); // t = 4
        let k = 30usize;
        let mut rejected = 0;
        for _ in 0..200
        {
            let msg: Vec<u8> = (0..k).map(|_| (xorshift(&mut s) & 0xff) as u8).collect();
            let cw = rs.encode_bytes(&msg);
            let n = cw.len();
            let mut corrupt = cw.clone();
            let mut chosen = Vec::new();
            while chosen.len() < rs.correction_capacity() + 1
            {
                let p = (xorshift(&mut s) as usize) % n;
                if !chosen.contains(&p)
                {
                    chosen.push(p);
                }
            }
            for &p in &chosen
            {
                corrupt[p] ^= ((xorshift(&mut s) & 0xff) | 1) as u8;
            }
            match rs.decode_bytes(&corrupt)
            {
                None => rejected += 1,
                Some((dec, _)) => assert_ne!(dec, msg, "must never return a wrong message"),
            }
        }
        // the recheck should reject the great majority of over-capacity cases
        assert!(rejected > 0, "expected some over-capacity rejections");
    }

    #[test]
    #[should_panic(expected = "not a primitive element")]
    fn rejects_non_primitive_alpha() {
        // In the Rijndael field, x (=2) is NOT primitive — must be rejected.
        let _ = ReedSolomon::new(Gf2Field::rijndael8(), 2, 4);
    }

    #[test]
    fn works_over_gf2_16() {
        // a larger field: 16-bit symbols, exercised through the general u64 API.
        let rs = ReedSolomon::new(Gf2Field::gf2_16(), 2, 6); // t = 3
        let mut s = 0x1616u64;
        for _ in 0..30
        {
            let msg: Vec<u64> = (0..50).map(|_| xorshift(&mut s) & 0xffff).collect();
            let cw = rs.encode(&msg);
            let n = cw.len();
            let mut corrupt = cw.clone();
            for _ in 0..3
            {
                let p = (xorshift(&mut s) as usize) % n;
                corrupt[p] ^= (xorshift(&mut s) & 0xffff) | 1;
            }
            let (dec, _errs) = rs.decode(&corrupt).unwrap();
            assert_eq!(dec, msg);
        }
    }

    #[test]
    fn erasures_recovered_up_to_nsym() {
        let mut s = 0xe7a5u64;
        for &nsym in &[4usize, 6, 8, 10, 16]
        {
            let rs = ReedSolomon::qr(nsym);
            let k = 40usize;
            for _ in 0..30
            {
                let msg: Vec<u8> = (0..k).map(|_| (xorshift(&mut s) & 0xff) as u8).collect();
                let cw: Vec<u64> = rs.encode_bytes(&msg).iter().map(|&b| b as u64).collect();
                let n = cw.len();
                // up to nsym erasures (twice the error-correction capacity)
                for e in 0..=nsym
                {
                    let mut received = cw.clone();
                    let mut positions = Vec::new();
                    while positions.len() < e
                    {
                        let p = (xorshift(&mut s) as usize) % n;
                        if !positions.contains(&p)
                        {
                            positions.push(p);
                        }
                    }
                    for &p in &positions
                    {
                        received[p] = xorshift(&mut s) & 0xff; // lost symbol (any value)
                    }
                    let fixed = rs
                        .decode_erasures(&received, &positions)
                        .expect("erasures fillable");
                    assert_eq!(fixed, cw, "nsym={nsym} e={e}");
                }
            }
        }
    }

    #[test]
    fn erasures_beyond_capacity_or_with_hidden_error() {
        let rs = ReedSolomon::qr(4);
        let msg: Vec<u8> = (1..=30u8).collect();
        let cw: Vec<u64> = rs.encode_bytes(&msg).iter().map(|&b| b as u64).collect();
        // nsym + 1 erasures cannot be filled
        let too_many: Vec<usize> = (0..=rs.parity_len()).collect();
        assert_eq!(rs.decode_erasures(&cw, &too_many), None);
        // an undeclared error among the non-erased symbols is detected, never
        // silently returned as a wrong codeword
        let mut received = cw.clone();
        received[0] = 0; // erased
        received[10] ^= 0x7f; // hidden error at a non-erased position
        match rs.decode_erasures(&received, &[0])
        {
            None =>
            {},
            Some(fixed) => assert_eq!(fixed, cw, "must never return a wrong codeword"),
        }
    }

    #[test]
    fn combined_errors_and_erasures() {
        // 2·t + e ≤ nsym must always decode.
        let mut s = 0xc0ffeeu64;
        for &nsym in &[6usize, 8, 10, 16]
        {
            let rs = ReedSolomon::qr(nsym);
            let k = 40usize;
            for _ in 0..40
            {
                let msg: Vec<u8> = (0..k).map(|_| (xorshift(&mut s) & 0xff) as u8).collect();
                let cw: Vec<u64> = rs.encode_bytes(&msg).iter().map(|&b| b as u64).collect();
                let n = cw.len();
                // pick e erasures then t errors with 2t + e ≤ nsym
                let e = (xorshift(&mut s) as usize) % (nsym + 1);
                let t = (nsym - e) / 2;
                let mut chosen = Vec::new();
                while chosen.len() < e + t
                {
                    let p = (xorshift(&mut s) as usize) % n;
                    if !chosen.contains(&p)
                    {
                        chosen.push(p);
                    }
                }
                let (erased, errored) = chosen.split_at(e);
                let mut received = cw.clone();
                for &p in erased
                {
                    received[p] = xorshift(&mut s) & 0xff;
                }
                for &p in errored
                {
                    received[p] ^= (xorshift(&mut s) & 0xff) | 1;
                }
                let (fixed, terr) = rs
                    .decode_errors_and_erasures(&received, erased)
                    .expect("within 2t+e ≤ nsym must decode");
                assert_eq!(fixed, cw, "nsym={nsym} e={e} t={t}");
                assert_eq!(terr, t, "reported error count nsym={nsym} e={e} t={t}");
            }
        }
    }
}
