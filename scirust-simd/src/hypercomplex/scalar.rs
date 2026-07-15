// scirust-simd/src/hypercomplex/scalar.rs
//
// Implémentations scalaires de référence des produits hypercomplexes.
//
// Deux variantes, volontairement SANS SIMD :
//
// 1. **Récursive** (`quat_mul`/`oct_mul`/`sed_mul`) : la construction de
//    Cayley-Dickson déroulée sur des tableaux `[f32; N]`, coefficient par
//    coefficient. C'est l'oracle de correction des kernels SIMD (mêmes
//    formules, mêmes conventions, arithmétique f32 identique).
//
// 2. **Par table** (`MulTable`) : produit « boucle par boucle » via les
//    constantes de structure eᵢ·eⱼ = γᵢⱼ·e_{τ(i,j)} précalculées. C'est la
//    baseline scalaire des benchmarks : le style d'implémentation naïf
//    (double boucle + table indexée) auquel on compare le chemin
//    shuffle/FMA en registres.
//
// Ces fonctions restent sans allocation (tableaux à taille fixe sur la
// pile) mais le compilateur les traite en code scalaire ordinaire.

/// Conjugaison de Cayley-Dickson sur un tableau : x̄₀ = x₀, x̄ᵢ = −xᵢ (i ≥ 1).
#[inline]
#[must_use]
pub fn conj<const N: usize>(x: [f32; N]) -> [f32; N] {
    let mut out = x;
    for c in out.iter_mut().skip(1)
    {
        *c = -*c;
    }
    out
}

/// Produit de Hamilton scalaire (quaternions, cas de base de la récursion).
#[inline]
#[must_use]
pub fn quat_mul(p: [f32; 4], q: [f32; 4]) -> [f32; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}

/// Produit d'octonions scalaire par Cayley-Dickson :
/// (a, b)(c, d) = (a·c − d̄·b, d·a + b·c̄) avec a, b, c, d ∈ ℍ.
#[inline]
#[must_use]
pub fn oct_mul(x: [f32; 8], y: [f32; 8]) -> [f32; 8] {
    let (a, b) = split::<8, 4>(x);
    let (c, d) = split::<8, 4>(y);

    let ac = quat_mul(a, c);
    let db_conj = quat_mul(conj(d), b);
    let da = quat_mul(d, a);
    let bc_conj = quat_mul(b, conj(c));

    let mut out = [0.0f32; 8];
    for i in 0..4
    {
        out[i] = ac[i] - db_conj[i];
        out[i + 4] = da[i] + bc_conj[i];
    }
    out
}

/// Produit de sédénions scalaire par Cayley-Dickson sur les octonions.
#[inline]
#[must_use]
pub fn sed_mul(x: [f32; 16], y: [f32; 16]) -> [f32; 16] {
    let (a, b) = split::<16, 8>(x);
    let (c, d) = split::<16, 8>(y);

    let ac = oct_mul(a, c);
    let db_conj = oct_mul(conj(d), b);
    let da = oct_mul(d, a);
    let bc_conj = oct_mul(b, conj(c));

    let mut out = [0.0f32; 16];
    for i in 0..8
    {
        out[i] = ac[i] - db_conj[i];
        out[i + 8] = da[i] + bc_conj[i];
    }
    out
}

/// Coupe un tableau `[f32; N]` en deux moitiés `[f32; H]` (H = N/2).
#[inline]
fn split<const N: usize, const H: usize>(x: [f32; N]) -> ([f32; H], [f32; H]) {
    debug_assert_eq!(2 * H, N);
    let mut lo = [0.0f32; H];
    let mut hi = [0.0f32; H];
    lo.copy_from_slice(&x[..H]);
    hi.copy_from_slice(&x[H..]);
    (lo, hi)
}

/// Table des constantes de structure d'une algèbre de Cayley-Dickson de
/// dimension N : eᵢ·eⱼ = `sign[i][j]` · e_{`target[i][j]`}.
///
/// Sert de baseline « boucle par boucle » dans les benchmarks : le produit
/// se calcule par double boucle indexée, représentatif d'une
/// implémentation scalaire naïve à base de table.
#[derive(Clone, Debug)]
pub struct MulTable<const N: usize> {
    /// Indice de la base cible : eᵢ·eⱼ ∈ {±e_k} ⇒ target[i][j] = k.
    pub target: [[usize; N]; N],
    /// Signe associé : ±1.
    pub sign: [[f32; N]; N],
}

impl<const N: usize> MulTable<N> {
    /// Produit x·y par double boucle sur la table :
    /// r[τ(i,j)] += γᵢⱼ · xᵢ · yⱼ. Style volontairement naïf (baseline).
    #[inline]
    #[must_use]
    pub fn mul(&self, x: &[f32; N], y: &[f32; N]) -> [f32; N] {
        let mut out = [0.0f32; N];
        for i in 0..N
        {
            let xi = x[i];
            if xi == 0.0
            {
                continue;
            }
            for j in 0..N
            {
                out[self.target[i][j]] += self.sign[i][j] * xi * y[j];
            }
        }
        out
    }
}

/// Construit la table 8×8 des octonions en multipliant les éléments de
/// base via l'oracle récursif. Chaque produit eᵢ·eⱼ a exactement une
/// coordonnée non nulle (±1) — vérifié par assertion.
#[must_use]
pub fn oct_table() -> MulTable<8> {
    build_table::<8>(oct_mul)
}

/// Construit la table 16×16 des sédénions (même principe).
#[must_use]
pub fn sed_table() -> MulTable<16> {
    build_table::<16>(sed_mul)
}

fn build_table<const N: usize>(mul: fn([f32; N], [f32; N]) -> [f32; N]) -> MulTable<N> {
    let mut target = [[0usize; N]; N];
    let mut sign = [[0.0f32; N]; N];
    for i in 0..N
    {
        for j in 0..N
        {
            let mut ei = [0.0f32; N];
            let mut ej = [0.0f32; N];
            ei[i] = 1.0;
            ej[j] = 1.0;
            let p = mul(ei, ej);
            let mut nonzero = 0;
            for (k, &c) in p.iter().enumerate()
            {
                if c != 0.0
                {
                    assert!(c == 1.0 || c == -1.0, "constante de structure non unitaire");
                    target[i][j] = k;
                    sign[i][j] = c;
                    nonzero += 1;
                }
            }
            assert_eq!(nonzero, 1, "produit de base non monomial");
        }
    }
    MulTable { target, sign }
}
