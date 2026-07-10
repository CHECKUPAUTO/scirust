//! Accumulateur **exact** de Kulisch pour produits de f32 : la réponse au
//! trou « GEMM reproductible ET parallélisable » de la cartographie
//! (volet 111, classe ReproBLAS).
//!
//! Principe : le produit de deux f32 est **exact en f64** (24 × 24 ≤ 53 bits).
//! On l'ajoute à un accumulateur **à virgule fixe** assez large pour couvrir
//! tout l'intervalle des produits (de (2⁻¹⁴⁹)² = 2⁻²⁹⁸ à ~2²⁷⁷) : chaque
//! addition est alors **exacte** — la somme finale est LE nombre réel
//! Σ aᵢ·bᵢ, arrondi une seule fois à la fin. Conséquences :
//!
//! - **indépendant de l'ordre** (l'addition exacte est associative et
//!   commutative) — plus fort que l'ordre figé de RepDL ;
//! - **correctement arrondi** (une seule opération d'arrondi, contrat plus
//!   fort que « fidèlement arrondi ») ;
//! - **fusion associative** ([`ExactAcc::merge`]) : des threads accumulent
//!   des tranches indépendantes puis fusionnent — résultat bit-identique
//!   quel que soit le nombre de threads ou le découpage ;
//! - arithmétique entière pure ⇒ **portable par construction**.
//!
//! Coût : ~2 additions 64 bits + un décalage par produit (pas de tri, pas
//! d'expansion dynamique) — l'ordre de grandeur au-dessus du GEMM naïf,
//! très loin devant `reproducible_dot` (tri O(n log n)).
//!
//! Représentation : deux sommes **positives** séparées (positifs, négatifs)
//! en virgule fixe 704 bits (11 mots u64, bit 0 de poids 2⁻³⁵² : le bit le
//! plus bas de la mantisse du plus petit produit ((2⁻¹⁴⁹)² = 2⁻²⁹⁸, mantisse
//! f64 de 53 bits) pèse 2⁻³⁵⁰ ; en tête, le plus grand produit (~2²⁷⁷) plus
//! ~2⁶⁰ de retenues tient sous 2³⁵¹ — déborder demanderait plus de 2⁶⁰
//! termes). La soustraction finale (une seule) et l'arrondi au plus près
//! pair produisent le f32.

/// Poids du bit 0 de l'accumulateur : 2^ACC_LSB_EXP.
const ACC_LSB_EXP: i32 = -352;
/// Nombre de mots de 64 bits (704 bits : couvre 2⁻³⁵² … 2³⁵¹).
const WORDS: usize = 11;

/// Accumulateur exact (sommes positives et négatives séparées).
#[derive(Debug, Clone)]
pub struct ExactAcc {
    pos: [u64; WORDS],
    neg: [u64; WORDS],
}

impl Default for ExactAcc {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactAcc {
    pub fn new() -> Self {
        Self {
            pos: [0; WORDS],
            neg: [0; WORDS],
        }
    }

    /// Ajoute une valeur f64 dont la mantisse est exacte (produit de f32,
    /// somme partielle exacte…). Les non-finis sont rejetés par assertion :
    /// l'accumulateur exact n'a pas de représentation pour ±∞/NaN.
    pub fn add_f64(&mut self, v: f64) {
        if v == 0.0
        {
            return;
        }
        assert!(v.is_finite(), "ExactAcc: valeur non finie");
        let bits = v.to_bits();
        let raw_exp = ((bits >> 52) & 0x7ff) as i32;
        let (mant, exp) = if raw_exp == 0
        {
            (bits & 0x000f_ffff_ffff_ffff, -1074)
        }
        else
        {
            (
                (bits & 0x000f_ffff_ffff_ffff) | (1u64 << 52),
                raw_exp - 1075,
            )
        };
        // mant·2^exp → décale mant à la position (exp − ACC_LSB_EXP)
        let shift = exp - ACC_LSB_EXP;
        assert!(
            shift >= 0,
            "ExactAcc: |v| sous la résolution 2^{ACC_LSB_EXP} (mantisse f64 comprise)"
        );
        let word = (shift / 64) as usize;
        let off = (shift % 64) as u32;
        let target = if bits >> 63 == 1
        {
            &mut self.neg
        }
        else
        {
            &mut self.pos
        };
        // mant (53 bits) décalée de off : s'étale sur ≤ 2 mots
        let lo = mant << off;
        let hi = if off == 0 { 0 } else { mant >> (64 - off) };
        let (w0, c0) = target[word].overflowing_add(lo);
        target[word] = w0;
        let mut carry = c0 as u64;
        let mut idx = word + 1;
        let mut add = hi;
        while (add != 0 || carry != 0) && idx < WORDS
        {
            let (w1, c1) = target[idx].overflowing_add(add);
            let (w2, c2) = w1.overflowing_add(carry);
            target[idx] = w2;
            add = 0;
            carry = (c1 as u64) + (c2 as u64);
            idx += 1;
        }
        debug_assert!(carry == 0, "ExactAcc: débordement (≫ 2⁶⁰ termes ?)");
    }

    /// Ajoute le produit **exact** `a·b` (promotion f64 : 24 × 24 ≤ 53 bits).
    #[inline]
    pub fn add_product(&mut self, a: f32, b: f32) {
        self.add_f64(a as f64 * b as f64);
    }

    /// Fusionne `other` dans `self` (addition mot à mot avec retenues) —
    /// associative et commutative : le résultat ne dépend ni du découpage
    /// ni de l'ordre des fusions.
    pub fn merge(&mut self, other: &ExactAcc) {
        for (dst, src) in [(&mut self.pos, &other.pos), (&mut self.neg, &other.neg)]
        {
            let mut carry = 0u64;
            for i in 0..WORDS
            {
                let (w1, c1) = dst[i].overflowing_add(src[i]);
                let (w2, c2) = w1.overflowing_add(carry);
                dst[i] = w2;
                carry = (c1 as u64) + (c2 as u64);
            }
            debug_assert!(carry == 0, "ExactAcc::merge: débordement");
        }
    }

    /// La somme exacte, arrondie **une seule fois** au f32 le plus proche
    /// (au plus près, pair en cas d'égalité) — correctement arrondie.
    pub fn round_f32(&self) -> f32 {
        // d = pos − neg (grand entier signé)
        let (mag, negative) = match cmp_words(&self.pos, &self.neg)
        {
            core::cmp::Ordering::Equal => return 0.0,
            core::cmp::Ordering::Greater => (sub_words(&self.pos, &self.neg), false),
            core::cmp::Ordering::Less => (sub_words(&self.neg, &self.pos), true),
        };
        // bit de poids fort
        let mut top = None;
        for i in (0..WORDS).rev()
        {
            if mag[i] != 0
            {
                top = Some(i * 64 + 63 - mag[i].leading_zeros() as usize);
                break;
            }
        }
        let top = top.expect("non nul par construction");
        // valeur = mag · 2^ACC_LSB_EXP ; exposant du bit de tête :
        let e_top = top as i32 + ACC_LSB_EXP;
        // extrait 25 bits de tête (24 de mantisse f32 + bit de garde) + sticky
        let take = 25usize;
        let mut m: u64 = 0;
        for k in 0..take
        {
            let pos_bit = top as i64 - k as i64;
            let b = if pos_bit >= 0
            {
                get_bit(&mag, pos_bit as usize)
            }
            else
            {
                0
            };
            m = (m << 1) | b;
        }
        let mut sticky = 0u64;
        if top as i64 - take as i64 >= 0
        {
            let limit = (top + 1).saturating_sub(take);
            for (i, &w) in mag.iter().enumerate()
            {
                if w != 0 && i * 64 < limit
                {
                    // mot contenant potentiellement des bits sous la garde
                    let upto = limit - i * 64;
                    if upto >= 64
                    {
                        sticky |= w;
                    }
                    else
                    {
                        sticky |= w & ((1u64 << upto) - 1);
                    }
                }
            }
        }
        // arrondi au plus près pair sur 24 bits
        let guard = m & 1;
        let mut kept = m >> 1; // 24 bits
        if guard == 1 && (sticky != 0 || kept & 1 == 1)
        {
            kept += 1;
        }
        let mut e = e_top;
        if kept == 1 << 24
        {
            kept >>= 1;
            e += 1;
        }
        // kept ∈ [2^23, 2^24) : valeur = kept · 2^(e − 23)
        let value = (kept as f64) * exp2i(e - 23);
        let signed = if negative { -value } else { value };
        signed as f32 // gère débordement (→ ±∞) et sous-normaux via le cast
    }
}

/// 2^e en f64 pour e ∈ [−1074, 1023] (sous-normaux compris, sans libm).
fn exp2i(e: i32) -> f64 {
    if e >= -1022
    {
        f64::from_bits(((e + 1023) as u64) << 52)
    }
    else
    {
        // sous-normal : 2^e = 2^(e+1074) · 2^-1074
        f64::from_bits(1u64 << (e + 1074)) // ulp scaling exact
    }
}

fn cmp_words(a: &[u64; WORDS], b: &[u64; WORDS]) -> core::cmp::Ordering {
    for i in (0..WORDS).rev()
    {
        match a[i].cmp(&b[i])
        {
            core::cmp::Ordering::Equal =>
            {},
            other => return other,
        }
    }
    core::cmp::Ordering::Equal
}

fn sub_words(a: &[u64; WORDS], b: &[u64; WORDS]) -> [u64; WORDS] {
    let mut out = [0u64; WORDS];
    let mut borrow = 0u64;
    for i in 0..WORDS
    {
        let (w1, b1) = a[i].overflowing_sub(b[i]);
        let (w2, b2) = w1.overflowing_sub(borrow);
        out[i] = w2;
        borrow = (b1 as u64) + (b2 as u64);
    }
    debug_assert_eq!(borrow, 0, "sub_words: a < b");
    out
}

fn get_bit(w: &[u64; WORDS], pos: usize) -> u64 {
    (w[pos / 64] >> (pos % 64)) & 1
}

/// Produit scalaire **exact** (somme réelle arrondie une fois) —
/// indépendant de l'ordre, correctement arrondi, portable.
pub fn dot_exact(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_exact: length mismatch");
    let mut acc = ExactAcc::new();
    for i in 0..a.len()
    {
        acc.add_product(a[i], b[i]);
    }
    acc.round_f32()
}

/// GEMM **exact** `C = A·B` (row-major) : chaque coefficient est un
/// [`dot_exact`]. Multi-thread trivialement bit-exact (chaque cellule est
/// indépendante) ; pour un dot massif découpé en tranches, utiliser
/// [`ExactAcc::merge`].
pub fn gemm_exact(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "gemm_exact: A doit être m×k");
    assert_eq!(b.len(), k * n, "gemm_exact: B doit être k×n");
    let mut c = vec![0.0f32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = ExactAcc::new();
            for l in 0..k
            {
                acc.add_product(a[i * k + l], b[l * n + j]);
            }
            c[i * n + j] = acc.round_f32();
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// La somme exacte égale bit à bit la référence correctement arrondie
    /// de `reproducible_dot` (Shewchuk) — deux constructions indépendantes
    /// du même réel arrondi.
    #[test]
    fn dot_exact_matches_shewchuk_reference_bitwise() {
        let mut rng = PcgEngine::new(31);
        for len in [1usize, 2, 7, 64, 256, 1000]
        {
            let a: Vec<f32> = (0..len).map(|_| rng.float() * 2.0 - 1.0).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.float() * 2.0 - 1.0).collect();
            let exact = dot_exact(&a, &b);
            let shewchuk = crate::reproducible::reproducible_dot(&a, &b);
            assert_eq!(
                exact.to_bits(),
                shewchuk.to_bits(),
                "len={len}: exact={exact}, shewchuk={shewchuk}"
            );
        }
    }

    /// Indépendance à l'ordre : bit-identique sous permutation.
    #[test]
    fn dot_exact_is_order_independent() {
        let mut rng = PcgEngine::new(77);
        let a: Vec<f32> = (0..512)
            .map(|i| (rng.float() * 2.0 - 1.0) * 10f32.powi(i % 13 - 6))
            .collect();
        let b: Vec<f32> = (0..512).map(|_| rng.float() * 2.0 - 1.0).collect();
        let reference = dot_exact(&a, &b);

        let mut idx: Vec<usize> = (0..a.len()).collect();
        for _ in 0..10
        {
            for i in (1..idx.len()).rev()
            {
                let j = ((rng.float() * (i as f32 + 1.0)) as usize).min(i);
                idx.swap(i, j);
            }
            let ap: Vec<f32> = idx.iter().map(|&i| a[i]).collect();
            let bp: Vec<f32> = idx.iter().map(|&i| b[i]).collect();
            assert_eq!(dot_exact(&ap, &bp).to_bits(), reference.to_bits());
        }
    }

    /// Cancellation catastrophique : l'accumulateur exact la traverse.
    #[test]
    fn dot_exact_survives_cancellation() {
        // 1e18 + 1 − 1e18 = 1 (produits via b = 1)
        let a = [1e18f32, 1.0, -1e18];
        let b = [1.0f32, 1.0, 1.0];
        assert_eq!(dot_exact(&a, &b), 1.0);
        // sous-normaux : (2^-100)·(2^-100) sommé 4 fois = 2^-198
        let tiny = f32::from_bits(0x0d80_0000); // 2^-100
        let a = [tiny; 4];
        let b = [tiny; 4];
        let expected = (2f64.powi(-198) as f32) * 4.0;
        assert_eq!(dot_exact(&a, &b), expected);
    }

    /// Fusion : threads sur tranches disjointes ⇒ mêmes bits que séquentiel,
    /// quel que soit le découpage.
    #[test]
    fn merge_makes_threading_bitexact() {
        let mut rng = PcgEngine::new(13);
        let a: Vec<f32> = (0..4096).map(|_| rng.float() * 4.0 - 2.0).collect();
        let b: Vec<f32> = (0..4096).map(|_| rng.float() * 4.0 - 2.0).collect();
        let sequential = dot_exact(&a, &b);

        for n_threads in [2usize, 3, 8]
        {
            let chunk = a.len().div_ceil(n_threads);
            let accs: Vec<ExactAcc> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..n_threads)
                    .map(|t| {
                        let (a, b) = (&a, &b);
                        scope.spawn(move || {
                            let mut acc = ExactAcc::new();
                            let s = t * chunk;
                            let e = ((t + 1) * chunk).min(a.len());
                            for i in s..e
                            {
                                acc.add_product(a[i], b[i]);
                            }
                            acc
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            // fusion dans un ordre arbitraire (droite → gauche)
            let mut total = ExactAcc::new();
            for acc in accs.iter().rev()
            {
                total.merge(acc);
            }
            assert_eq!(
                total.round_f32().to_bits(),
                sequential.to_bits(),
                "{n_threads} threads"
            );
        }
    }

    /// GEMM exact : petits entiers (résultat mathématique exact) et
    /// cohérence bit à bit avec dot_exact.
    #[test]
    fn gemm_exact_small_integers_and_consistency() {
        let c = gemm_exact(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);

        let mut rng = PcgEngine::new(1113);
        let a: Vec<f32> = (0..6 * 5).map(|_| rng.float() * 4.0 - 2.0).collect();
        let b: Vec<f32> = (0..5 * 4).map(|_| rng.float() * 4.0 - 2.0).collect();
        let c = gemm_exact(&a, &b, 6, 5, 4);
        for i in 0..6
        {
            for j in 0..4
            {
                let row: Vec<f32> = (0..5).map(|l| a[i * 5 + l]).collect();
                let col: Vec<f32> = (0..5).map(|l| b[l * 4 + j]).collect();
                assert_eq!(c[i * 4 + j].to_bits(), dot_exact(&row, &col).to_bits());
            }
        }
    }

    /// Extrêmes : produits près du min/max f32 sans débordement interne.
    #[test]
    fn extreme_magnitudes() {
        let big = 3.0e38f32;
        assert_eq!(dot_exact(&[big], &[1.0]), big);
        assert_eq!(dot_exact(&[big, big], &[1.0, 1.0]), f32::INFINITY);
        assert_eq!(dot_exact(&[big, big], &[1.0, -1.0]), 0.0);
        let min_sub = f32::from_bits(1);
        assert_eq!(dot_exact(&[min_sub], &[min_sub]), 0.0); // 2^-298 < min f32
        // produits minuscules mais résultat représentable : 4·(2^-60)² = 2^-118
        let tiny = f32::from_bits(((127 - 60) as u32) << 23); // 2^-60
        let expected = f32::from_bits(((127 - 118) as u32) << 23); // 2^-118
        assert_eq!(dot_exact(&[tiny; 4], &[tiny; 4]), expected);
    }
}
