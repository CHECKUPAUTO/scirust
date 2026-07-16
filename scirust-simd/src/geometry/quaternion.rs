// scirust-simd/src/geometry/quaternion.rs
//
// # Quaternion générique `Quaternion<T>`
//
// Un quaternion `w + x·i + y·j + z·k` **générique sur le scalaire** : la même
// implémentation sert `Quaternion<f32>`, `Quaternion<f64>` **et**
// `Quaternion<FixedI32<16>>` (virgule fixe déterministe). Seule l'algèbre
// d'anneau ([`NumericScalar`]) est requise pour le produit de Hamilton, la
// conjugaison et la rotation de vecteur ; les opérations qui prennent une
// racine ou une transcendante (norme, normalisation, angle-axe) demandent
// [`RealScalar`].
//
// C'est la concrétisation directe de la brique `RealScalar` : une bibliothèque
// d'orientation 3D qui fonctionne à l'identique en flottant et en virgule fixe,
// sans une seule ligne dupliquée.
//
// ## Ce module vs `hypercomplex`
//
// [`crate::hypercomplex`] fournit des quaternions **SIMD `f32` register-résidents**
// (le cas de base de la récursion de Cayley-Dickson vers octonions/sédénions),
// optimisés pour le débit brut. Ce module-ci fournit un quaternion **scalaire
// générique** orienté **géométrie** (rotations, orientation) : priorité à la
// généricité et au déterminisme, pas à la vectorisation.
//
// ## Non couvert (lot ultérieur)
//
// `slerp` et `to_axis_angle` exigent une trigonométrie **inverse**
// (`acos`/`atan2`) absente de [`RealScalar`]. Ils relèvent d'un lot dédié
// « trigonométrie inverse ». En attendant, [`Quaternion::nlerp`] (interpolation
// linéaire normalisée) couvre le besoin d'interpolation courant sans `acos`.

use core::ops::{Add, Mul, Neg, Sub};

use crate::fixed::{NumericScalar, RealScalar};

/// Quaternion `w + x·i + y·j + z·k`, générique sur le scalaire.
///
/// Convention de Hamilton : `i² = j² = k² = ijk = −1`, `ij = k`, `jk = i`,
/// `ki = j`. Le produit n'est **pas** commutatif.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion<T> {
    /// Partie réelle (scalaire).
    pub w: T,
    /// Coefficient de `i`.
    pub x: T,
    /// Coefficient de `j`.
    pub y: T,
    /// Coefficient de `k`.
    pub z: T,
}

impl<T: NumericScalar> Quaternion<T> {
    /// Construit `w + x·i + y·j + z·k`.
    #[inline]
    pub fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    /// Quaternion identité `1` (rotation nulle).
    #[inline]
    pub fn identity() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero())
    }

    /// Quaternion nul `0`.
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::zero())
    }

    /// Scalaire pur `w + 0i + 0j + 0k`.
    #[inline]
    pub fn from_scalar(w: T) -> Self {
        Self::new(w, T::zero(), T::zero(), T::zero())
    }

    /// Quaternion pur (imaginaire) `0 + x·i + y·j + z·k` depuis un vecteur 3D.
    #[inline]
    pub fn from_vector(v: [T; 3]) -> Self {
        Self::new(T::zero(), v[0], v[1], v[2])
    }

    /// Partie vectorielle `(x, y, z)`.
    #[inline]
    pub fn vector(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }

    /// Conjugué `w − x·i − y·j − z·k`.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Produit scalaire euclidien des 4 composantes (`⟨p, q⟩`).
    #[inline]
    pub fn dot(self, r: Self) -> T {
        self.w * r.w + self.x * r.x + self.y * r.y + self.z * r.z
    }

    /// Carré de la norme `|q|² = w² + x² + y² + z²` (exact, sans racine).
    #[inline]
    pub fn norm_sqr(self) -> T {
        self.dot(self)
    }

    /// Multiplie chaque composante par le scalaire `s`.
    #[inline]
    pub fn scale(self, s: T) -> Self {
        Self::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }

    /// Produit de Hamilton `self ⊗ r` (composition de rotations, non commutatif).
    ///
    /// `rotate_vector(a ⊗ b, v) == rotate_vector(a, rotate_vector(b, v))`.
    #[inline]
    pub fn mul_quat(self, r: Self) -> Self {
        Self {
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        }
    }

    /// Fait tourner le vecteur `v` par ce quaternion (**supposé unitaire**).
    ///
    /// Utilise la forme optimisée `v' = v + w·t + u×t`, `t = 2·(u×v)`,
    /// `u = (x, y, z)` : **uniquement des opérations d'anneau** (aucune racine
    /// ni transcendante), donc exacte et disponible dès [`NumericScalar`].
    /// Pour un quaternion non unitaire, normaliser d'abord ([`Self::normalize`]).
    #[inline]
    pub fn rotate_vector(self, v: [T; 3]) -> [T; 3] {
        let u = [self.x, self.y, self.z];
        let two = T::from_i32(2);
        // t = 2 · (u × v)
        let uxv = cross(u, v);
        let t = [uxv[0] * two, uxv[1] * two, uxv[2] * two];
        // v' = v + w·t + u × t
        let uxt = cross(u, t);
        [
            v[0] + self.w * t[0] + uxt[0],
            v[1] + self.w * t[1] + uxt[1],
            v[2] + self.w * t[2] + uxt[2],
        ]
    }

    /// Matrice de rotation 3×3 (lignes) correspondant à ce quaternion unitaire.
    ///
    /// Formule standard en opérations d'anneau (exacte, sans racine).
    #[inline]
    pub fn to_rotation_matrix(self) -> [[T; 3]; 3] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let two = T::from_i32(2);
        let one = T::one();
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        [
            [one - two * (yy + zz), two * (xy - wz), two * (xz + wy)],
            [two * (xy + wz), one - two * (xx + zz), two * (yz - wx)],
            [two * (xz - wy), two * (yz + wx), one - two * (xx + yy)],
        ]
    }
}

impl<T: RealScalar> Quaternion<T> {
    /// Norme euclidienne `|q| = √(w² + x² + y² + z²)`.
    #[inline]
    pub fn norm(self) -> T {
        self.norm_sqr().sqrt()
    }

    /// Renvoie le quaternion unitaire de même direction `q / |q|`.
    ///
    /// Indéfini pour `q = 0` (le flottant produit `inf`, la virgule fixe
    /// sature) — comportement cohérent avec la division par zéro du scalaire.
    #[inline]
    pub fn normalize(self) -> Self {
        self.scale(self.norm().recip())
    }

    /// Inverse `q⁻¹ = conj(q) / |q|²`. Pour un quaternion **unitaire**,
    /// l'inverse égale le conjugué (moins cher).
    #[inline]
    pub fn inverse(self) -> Self {
        self.conjugate().scale(self.norm_sqr().recip())
    }

    /// Quaternion unitaire d'une rotation d'angle `angle` (radians) autour de
    /// l'axe `axis` (normalisé en interne) : `q = cos(θ/2) + sin(θ/2)·û`.
    #[inline]
    pub fn from_axis_angle(axis: [T; 3], angle: T) -> Self {
        // Normalise l'axe.
        let n2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        let inv = n2.sqrt().recip();
        let u = [axis[0] * inv, axis[1] * inv, axis[2] * inv];
        // Demi-angle.
        let half = angle * T::from_i32(2).recip();
        let (s, c) = (half.sin(), half.cos());
        Self::new(c, u[0] * s, u[1] * s, u[2] * s)
    }

    /// Interpolation linéaire normalisée (**nlerp**) entre `a` et `b`, `t ∈ [0,1]`.
    ///
    /// `normalize((1−t)·a + t·b)`. Contrairement à `slerp`, ne demande pas de
    /// trigonométrie inverse : suit le plus court arc, vitesse angulaire non
    /// constante mais coût faible et résultat toujours unitaire. Aligne le signe
    /// de `b` sur `a` (`dot ≥ 0`) pour interpoler par le plus court chemin.
    #[inline]
    pub fn nlerp(a: Self, b: Self, t: T) -> Self {
        let b = if a.dot(b) < T::zero() { -b } else { b };
        let one_minus_t = T::one() - t;
        let blended = a.scale(one_minus_t) + b.scale(t);
        blended.normalize()
    }
}

/// Produit vectoriel `a × b` (fonction libre, opérations d'anneau).
#[inline]
fn cross<T: NumericScalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ------------------------------------------------------------------ //
//  Surcharge d'opérateurs                                             //
// ------------------------------------------------------------------ //

impl<T: NumericScalar> Add for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Self::new(self.w + r.w, self.x + r.x, self.y + r.y, self.z + r.z)
    }
}

impl<T: NumericScalar> Sub for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Self::new(self.w - r.w, self.x - r.x, self.y - r.y, self.z - r.z)
    }
}

impl<T: NumericScalar> Neg for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }
}

impl<T: NumericScalar> Mul for Quaternion<T> {
    type Output = Self;
    /// Produit de Hamilton (cf. [`Quaternion::mul_quat`]).
    #[inline]
    fn mul(self, r: Self) -> Self {
        self.mul_quat(r)
    }
}
