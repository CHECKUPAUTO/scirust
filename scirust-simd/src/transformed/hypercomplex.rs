// scirust-simd/src/transformed/hypercomplex.rs
//
// # Algèbre de Cayley–Dickson générique `Hypercomplex<S, N>`
//
// Un élément de dimension `N` (puissance de 2, `N ≤ 16`) sur le scalaire `S`,
// stocké à plat `[S; N]` (composante 0 = partie réelle). La multiplication suit
// récursivement le doublement de Cayley–Dickson
//
// ```text
//   (a, b) · (c, d) = (a·c − d̄·b,  d·a + b·c̄),   conj(a, b) = (ā, −b)
// ```
//
// **strictement identique** à la convention des noyaux SIMD de
// [`crate::hypercomplex`], mais ici **générique sur le scalaire** et sans SIMD
// (le cadre TSHA cible d'abord des scalaires ; la vectorisation viendra plus
// tard). Aucune allocation : la récursion utilise des tampons de pile bornés à
// 16 éléments. Aucun `unsafe`.
//
// Alias : [`Complex`] (2), [`Quaternion`] (4), [`Octonion`] (8), [`Sedenion`]
// (16). De futures algèbres (32, 64, …) s'obtiendraient en élargissant les
// tampons — l'algorithme, lui, est déjà générique.

use core::ops::{Add, Mul, Neg, Sub};

use crate::fixed::NumericScalar;

use super::scalar::TransformedScalar;
use super::transform::{DomainError, ScalarTransform};

/// Capacité maximale des tampons de pile (⇒ `N ≤ MAX_DIM`).
const MAX_DIM: usize = 16;

/// Élément d'algèbre de Cayley–Dickson de dimension `N` sur le scalaire `S`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hypercomplex<S, const N: usize>(pub [S; N]);

/// Nombre complexe `a + b·i` (dimension 2).
pub type Complex<S> = Hypercomplex<S, 2>;
/// Quaternion (dimension 4).
pub type Quaternion<S> = Hypercomplex<S, 4>;
/// Octonion (dimension 8).
pub type Octonion<S> = Hypercomplex<S, 8>;
/// Sédénion (dimension 16).
pub type Sedenion<S> = Hypercomplex<S, 16>;

/// Conjugué de Cayley–Dickson à plat : composante réelle inchangée, imaginaires
/// niés (récursivement équivalent à `conj(a, b) = (ā, −b)`).
fn cd_conj_into<S: NumericScalar>(a: &[S], out: &mut [S]) {
    out[0] = a[0];
    for i in 1..a.len()
    {
        out[i] = -a[i];
    }
}

/// Produit de Cayley–Dickson à plat `out = a · b` (récursif, tampons de pile).
fn cd_mul_into<S: NumericScalar>(a: &[S], b: &[S], out: &mut [S]) {
    let n = a.len();
    if n == 1
    {
        out[0] = a[0] * b[0];
        return;
    }
    let h = n / 2;
    let (a1, a2) = a.split_at(h);
    let (b1, b2) = b.split_at(h);

    let mut cb1 = [S::zero(); MAX_DIM];
    let mut cb2 = [S::zero(); MAX_DIM];
    cd_conj_into(b1, &mut cb1[..h]);
    cd_conj_into(b2, &mut cb2[..h]);

    let mut t1 = [S::zero(); MAX_DIM];
    let mut t2 = [S::zero(); MAX_DIM];
    let (o1, o2) = out.split_at_mut(h);

    // o1 = a1·b1 − b2̄·a2
    cd_mul_into(a1, b1, &mut t1[..h]);
    cd_mul_into(&cb2[..h], a2, &mut t2[..h]);
    for i in 0..h
    {
        o1[i] = t1[i] - t2[i];
    }
    // o2 = b2·a1 + a2·b1̄
    cd_mul_into(b2, a1, &mut t1[..h]);
    cd_mul_into(a2, &cb1[..h], &mut t2[..h]);
    for i in 0..h
    {
        o2[i] = t1[i] + t2[i];
    }
}

impl<S: NumericScalar, const N: usize> Hypercomplex<S, N> {
    /// Construit depuis les composantes. `N` doit être une puissance de 2 ≤ 16.
    #[inline]
    pub fn new(components: [S; N]) -> Self {
        const {
            assert!(
                N.is_power_of_two() && N <= MAX_DIM,
                "Hypercomplex: N doit être une puissance de 2 ≤ 16"
            );
        }
        Self(components)
    }

    /// Zéro (toutes composantes nulles).
    #[inline]
    pub fn zero() -> Self {
        Self::new([S::zero(); N])
    }

    /// Scalaire réel pur `r` (composante 0 = `r`, reste nul).
    #[inline]
    pub fn real(r: S) -> Self {
        let mut c = [S::zero(); N];
        c[0] = r;
        Self::new(c)
    }

    /// Élément neutre multiplicatif `1`.
    #[inline]
    pub fn one() -> Self {
        Self::real(S::one())
    }

    /// `i`-ème unité de base `e_i` (composante `i` = 1).
    #[inline]
    pub fn basis(i: usize) -> Self {
        let mut c = [S::zero(); N];
        c[i] = S::one();
        Self::new(c)
    }

    /// Accès en lecture aux composantes.
    #[inline]
    pub fn components(&self) -> &[S; N] {
        &self.0
    }

    /// Conjugué de Cayley–Dickson.
    #[inline]
    pub fn conj(self) -> Self {
        let mut o = [S::zero(); N];
        cd_conj_into(&self.0, &mut o);
        Self(o)
    }

    /// Multiplication par un scalaire.
    #[inline]
    pub fn scale(self, s: S) -> Self {
        let mut o = self.0;
        for c in &mut o
        {
            *c = *c * s;
        }
        Self(o)
    }

    /// Norme au carré `Σ cᵢ²` (exacte dans l'anneau, sans racine).
    #[inline]
    pub fn norm_sqr(self) -> S {
        let mut acc = S::zero();
        for &c in &self.0
        {
            acc = acc + c * c;
        }
        acc
    }

    /// Commutateur `self·rhs − rhs·self` (nul ssi les deux commutent).
    #[inline]
    pub fn commutator(self, rhs: Self) -> Self {
        self * rhs - rhs * self
    }

    /// Associateur `(self·b)·c − self·(b·c)` (nul ssi associatif sur ce triplet).
    #[inline]
    pub fn associator(self, b: Self, c: Self) -> Self {
        (self * b) * c - self * (b * c)
    }

    /// Transporte les composantes vers un autre scalaire via `f`.
    #[inline]
    pub fn map<U: NumericScalar>(self, f: impl Fn(S) -> U) -> Hypercomplex<U, N> {
        Hypercomplex(self.0.map(f))
    }
}

impl<S: NumericScalar, const N: usize> Add for Hypercomplex<S, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut o = self.0;
        for (o_i, &r_i) in o.iter_mut().zip(rhs.0.iter())
        {
            *o_i = *o_i + r_i;
        }
        Self(o)
    }
}
impl<S: NumericScalar, const N: usize> Sub for Hypercomplex<S, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut o = self.0;
        for (o_i, &r_i) in o.iter_mut().zip(rhs.0.iter())
        {
            *o_i = *o_i - r_i;
        }
        Self(o)
    }
}
impl<S: NumericScalar, const N: usize> Neg for Hypercomplex<S, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut o = self.0;
        for s in &mut o
        {
            *s = -*s;
        }
        Self(o)
    }
}
/// Produit de Cayley–Dickson (non commutatif dès `N ≥ 4`).
impl<S: NumericScalar, const N: usize> Mul for Hypercomplex<S, N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut o = [S::zero(); N];
        cd_mul_into(&self.0, &rhs.0, &mut o);
        Self(o)
    }
}

// ------------------------------------------------------------------ //
//  Les deux modèles d'exécution transformés                           //
// ------------------------------------------------------------------ //

/// Encode chaque composante latente d'un élément (helper des deux modèles).
fn encode_element<F, const N: usize>(
    a: Hypercomplex<TransformedScalar<f64, F>, N>,
) -> Result<Hypercomplex<f64, N>, DomainError>
where
    F: ScalarTransform<f64>,
{
    let mut out = [0.0f64; N];
    for (o, comp) in out.iter_mut().zip(a.0.iter())
    {
        *o = F::encode(comp.latent())?;
    }
    Ok(Hypercomplex(out))
}

/// **Modèle A — transport en espace latent** : `φ(A ⋆ B)`.
///
/// Le produit hypercomplexe est calculé en coordonnées **latentes** (l'algèbre
/// d'origine, inchangée), puis chaque composante du résultat est encodée. Ce
/// modèle **préserve les lois algébriques** de l'algèbre latente (elles sont
/// simplement transportées par φ).
pub fn model_a_product<F, const N: usize>(
    a: Hypercomplex<TransformedScalar<f64, F>, N>,
    b: Hypercomplex<TransformedScalar<f64, F>, N>,
) -> Result<Hypercomplex<f64, N>, DomainError>
where
    F: ScalarTransform<f64>,
{
    let prod = a * b; // produit latent
    let mut out = [0.0f64; N];
    for (o, comp) in out.iter_mut().zip(prod.0.iter())
    {
        *o = F::encode(comp.latent())?;
    }
    Ok(Hypercomplex(out))
}

/// **Modèle B — algèbre transformée directe** : `φ(A) ⋆ φ(B)`.
///
/// Les composantes sont d'abord encodées, puis l'algèbre est appliquée
/// **directement sur les valeurs encodées**. En général **non équivalent** au
/// Modèle A : leur écart est le « défaut de transformation » mesuré par
/// [`super::metrics`].
pub fn model_b_product<F, const N: usize>(
    a: Hypercomplex<TransformedScalar<f64, F>, N>,
    b: Hypercomplex<TransformedScalar<f64, F>, N>,
) -> Result<Hypercomplex<f64, N>, DomainError>
where
    F: ScalarTransform<f64>,
{
    let ae = encode_element::<F, N>(a)?;
    let be = encode_element::<F, N>(b)?;
    Ok(ae * be)
}
