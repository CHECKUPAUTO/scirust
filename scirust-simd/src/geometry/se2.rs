// scirust-simd/src/geometry/se2.rs
//
// # Transform rigide 2D `Se2<T>` (groupe `SE(2)`)
//
// Déplacement rigide **plan** — rotation `SO(2)` puis translation —
// **générique sur le scalaire**, pendant 2D du [`super::Transform`] `SE(3)`.
// La même implémentation sert `Se2<f32>`, `Se2<f64>` et `Se2<FixedI32<16>>` ;
// ce dernier compose des poses de façon **reproductible bit-à-bit** sur toute
// architecture — l'usage typique en robotique mobile déterministe (odométrie,
// SLAM 2D) ou en rejeu de simulation.
//
// ## Représentation
//
// Un `Se2` est stocké comme un **angle** `θ` (radians) et une translation
// `(x, y)` — la carte minimale de `SE(2)` (3 degrés de liberté). La rotation
// n'est matérialisée (`cos θ`, `sin θ`) qu'au moment de transformer un point.
//
// ## Composition et convention
//
// [`Se2::compose`] (et l'opérateur `*`) suit la même convention que
// [`super::Transform`] : `a.compose(&b)` applique `b` **d'abord**, puis `a` —
// `a.compose(&b).transform_point(p) == a.transform_point(b.transform_point(p))`.
// C'est l'opération de groupe de `SE(2)` ; [`Se2::identity`] en est le neutre
// et [`Se2::inverse`] l'inverse (à deux côtés). Comme en `SE(3)`, l'inverse est
// en `O(1)` (rotation transposée, aucun système à résoudre).

use core::ops::Mul;

use crate::fixed::{NumericScalar, RealScalar};

/// Déplacement rigide plan (`SE(2)`) : rotation d'angle `angle` (radians) puis
/// translation `translation = [x, y]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se2<T> {
    /// Angle de rotation (radians, sens direct).
    pub angle: T,
    /// Translation `[x, y]`.
    pub translation: [T; 2],
}

impl<T: NumericScalar> Se2<T> {
    /// Construit depuis un angle (radians) et une translation `(x, y)`.
    #[inline]
    pub fn new(angle: T, x: T, y: T) -> Self {
        Self {
            angle,
            translation: [x, y],
        }
    }

    /// Translation pure (rotation nulle).
    #[inline]
    pub fn from_translation(x: T, y: T) -> Self {
        Self::new(T::zero(), x, y)
    }

    /// Rotation pure (translation nulle).
    #[inline]
    pub fn from_angle(angle: T) -> Self {
        Self::new(angle, T::zero(), T::zero())
    }

    /// Identité (`SE(2)` neutre) : aucun déplacement.
    #[inline]
    pub fn identity() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }
}

impl<T: RealScalar> Se2<T> {
    /// Matrice de rotation `2×2` `[[cos, −sin], [sin, cos]]` de l'angle courant.
    #[inline]
    #[must_use]
    pub fn rotation_matrix(&self) -> [[T; 2]; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        [[c, -s], [s, c]]
    }

    /// Applique la rotation seule (sans translation) à un vecteur `[x, y]`.
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, v: [T; 2]) -> [T; 2] {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        [c * v[0] - s * v[1], s * v[0] + c * v[1]]
    }

    /// Transforme un point : `R·p + t` (rotation puis translation).
    #[inline]
    #[must_use]
    pub fn transform_point(&self, p: [T; 2]) -> [T; 2] {
        let r = self.transform_vector(p);
        [r[0] + self.translation[0], r[1] + self.translation[1]]
    }

    /// Composition `self ∘ other` : applique `other` **d'abord**, puis `self`
    /// (cf. en-tête de module). L'angle résultant est la somme, la translation
    /// `self.transform_point(other.translation)`.
    #[inline]
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let t = self.transform_point(other.translation);
        Self {
            angle: self.angle + other.angle,
            translation: t,
        }
    }

    /// Inverse (à deux côtés) : `self.compose(&self.inverse()) == identity`
    /// (aux arrondis près). Angle opposé, translation `−R(−θ)·t`.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv_angle = -self.angle;
        let (c, s) = (inv_angle.cos(), inv_angle.sin());
        // −R(−θ)·t.
        let tx = -(c * self.translation[0] - s * self.translation[1]);
        let ty = -(s * self.translation[0] + c * self.translation[1]);
        Self {
            angle: inv_angle,
            translation: [tx, ty],
        }
    }
}

/// L'opérateur `*` est la composition [`Se2::compose`] (`a * b` applique `b`
/// puis `a`).
impl<T: RealScalar> Mul for Se2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.compose(&rhs)
    }
}
