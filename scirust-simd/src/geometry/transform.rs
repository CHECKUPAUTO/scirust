// scirust-simd/src/geometry/transform.rs
//
// # Transform rigide `Transform<T>` (groupe `SE(3)`)
//
// Déplacement rigide 3D — rotation ([`Quaternion<T>`]) puis translation —
// **générique sur le scalaire**, comme [`Quaternion`] : la même implémentation
// sert `Transform<f32>`, `Transform<f64>` et `Transform<FixedI32<16>>`. Un
// `Transform<FixedI32<16>>` compose des poses de façon **reproductible
// bit-à-bit** sur toute architecture — l'usage typique en robotique
// déterministe ou en rejeu de simulation, où deux plateformes doivent
// calculer exactement la même trajectoire.
//
// ## `SE(3)` : rotation + translation, pas une affinité générale
//
// `Transform` représente un déplacement **rigide** : rotation puis
// translation, aucune mise à l'échelle ni cisaillement. C'est un sous-groupe
// strict des matrices affines 4×4 générales (que ce module ne cherche pas à
// couvrir) — la restriction est ce qui permet [`Transform::inverse`] en
// `O(1)` (pas d'inversion de matrice générale) via l'inverse du quaternion.
//
// ## Composition et convention
//
// [`Transform::compose`] (et l'opérateur `*`) suit la même convention que le
// produit de Hamilton des quaternions : `a.compose(&b)` applique `b`
// **d'abord**, puis `a` — `a.compose(&b).transform_point(p) ==
// a.transform_point(b.transform_point(p))`. C'est l'opération de groupe de
// `SE(3)` ; [`Transform::identity`] en est l'élément neutre et
// [`Transform::inverse`] l'inverse (à deux côtés).
//
// ## Représentation matricielle
//
// [`Transform::to_matrix`]/[`Transform::from_matrix`] convertissent vers/depuis
// la matrice homogène 4×4 usuelle (bloc rotation 3×3 + colonne de translation
// + dernière ligne `[0,0,0,1]`), réutilisant
// [`Quaternion::to_rotation_matrix`]/[`Quaternion::from_rotation_matrix`].
// `from_matrix` hérite donc de la même contrainte `T: RealScalar +
// Div<Output = T>` (méthode de Shepperd, division réelle non puissance de
// deux — cf. [`super::quaternion`]).

use core::ops::{Div, Mul};

use crate::fixed::{NumericScalar, RealScalar};

use super::quaternion::Quaternion;

/// Déplacement rigide 3D (`SE(3)`) : rotation puis translation.
///
/// `rotation` est supposé **unitaire** (comme pour [`Quaternion::rotate_vector`]) ;
/// normaliser au préalable si nécessaire ([`Quaternion::normalize`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform<T> {
    /// Rotation (quaternion unitaire).
    pub rotation: Quaternion<T>,
    /// Translation.
    pub translation: [T; 3],
}

impl<T: NumericScalar> Transform<T> {
    /// Construit depuis une rotation et une translation.
    #[inline]
    pub fn new(rotation: Quaternion<T>, translation: [T; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Transform identité (rotation nulle, translation nulle) : élément
    /// neutre de la composition.
    #[inline]
    pub fn identity() -> Self {
        Self::new(Quaternion::identity(), [T::zero(); 3])
    }

    /// Translation pure (rotation identité).
    #[inline]
    pub fn from_translation(translation: [T; 3]) -> Self {
        Self::new(Quaternion::identity(), translation)
    }

    /// Rotation pure (translation nulle).
    #[inline]
    pub fn from_rotation(rotation: Quaternion<T>) -> Self {
        Self::new(rotation, [T::zero(); 3])
    }

    /// Transforme le point `p` : rotation puis translation.
    #[inline]
    pub fn transform_point(&self, p: [T; 3]) -> [T; 3] {
        let r = self.rotation.rotate_vector(p);
        [
            r[0] + self.translation[0],
            r[1] + self.translation[1],
            r[2] + self.translation[2],
        ]
    }

    /// Transforme la direction `v` (rotation seule) : contrairement à un
    /// point, une direction n'est **pas** affectée par la translation (ex. un
    /// vecteur normal de surface, une vitesse).
    #[inline]
    pub fn transform_vector(&self, v: [T; 3]) -> [T; 3] {
        self.rotation.rotate_vector(v)
    }

    /// Compose deux transforms : `self ∘ other` applique `other` d'abord,
    /// puis `self`. C'est l'opération de groupe de `SE(3)` (cf. en-tête de
    /// module) ; correspond à l'opérateur [`Mul`].
    #[inline]
    pub fn compose(&self, other: &Self) -> Self {
        let t = self.rotation.rotate_vector(other.translation);
        Self::new(
            self.rotation.mul_quat(other.rotation),
            [
                t[0] + self.translation[0],
                t[1] + self.translation[1],
                t[2] + self.translation[2],
            ],
        )
    }

    /// Matrice homogène 4×4 (lignes) équivalente : bloc rotation 3×3 en haut
    /// à gauche, translation en dernière colonne, dernière ligne `[0,0,0,1]`.
    #[must_use]
    pub fn to_matrix(&self) -> [[T; 4]; 4] {
        let r = self.rotation.to_rotation_matrix();
        let t = self.translation;
        let (zero, one) = (T::zero(), T::one());
        [
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [zero, zero, zero, one],
        ]
    }
}

impl<T: RealScalar> Transform<T> {
    /// Inverse `(R, t)⁻¹ = (R⁻¹, −R⁻¹·t)`, deux côtés : `self.compose(&self.inverse())`
    /// et `self.inverse().compose(self)` valent [`Transform::identity`] (à la
    /// résolution de `T` près).
    #[must_use]
    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.inverse();
        let t_inv = r_inv.rotate_vector(self.translation);
        Self::new(r_inv, [-t_inv[0], -t_inv[1], -t_inv[2]])
    }
}

impl<T: RealScalar + Div<Output = T>> Transform<T> {
    /// Reconstruit depuis une matrice homogène 4×4 (lignes), réciproque de
    /// [`Self::to_matrix`]. La sous-matrice de rotation 3×3 est décodée par
    /// [`Quaternion::from_rotation_matrix`] (méthode de Shepperd) ; la
    /// dernière ligne `[0,0,0,1]` n'est pas relue (supposée, comme pour toute
    /// matrice homogène `SE(3)`).
    #[must_use]
    pub fn from_matrix(m: [[T; 4]; 4]) -> Self {
        let r3 = [
            [m[0][0], m[0][1], m[0][2]],
            [m[1][0], m[1][1], m[1][2]],
            [m[2][0], m[2][1], m[2][2]],
        ];
        Self::new(
            Quaternion::from_rotation_matrix(r3),
            [m[0][3], m[1][3], m[2][3]],
        )
    }
}

impl<T: NumericScalar> Mul for Transform<T> {
    type Output = Self;
    /// Composition (cf. [`Transform::compose`]).
    #[inline]
    fn mul(self, r: Self) -> Self {
        self.compose(&r)
    }
}
