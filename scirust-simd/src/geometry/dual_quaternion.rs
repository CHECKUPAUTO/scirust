// scirust-simd/src/geometry/dual_quaternion.rs
//
// # Quaternion dual `DualQuaternion<T>` — déplacement rigide `SE(3)` unifié
//
// Un déplacement rigide (rotation + translation) encodé en **un seul** objet
// algébrique `q̂ = qᵣ + ε·q_d` (`qᵣ`, `q_d` deux [`Quaternion<T>`], `ε² = 0`),
// **générique sur le scalaire** comme [`Quaternion`]/[`Transform`] : la même
// implémentation sert `DualQuaternion<f32>`, `DualQuaternion<f64>` et
// `DualQuaternion<FixedI32<16>>`.
//
// ## Pourquoi, en plus de `Transform` ?
//
// [`Transform`] représente déjà `SE(3)` comme une **paire** `(rotation,
// translation)` ; composer et transformer un point y sont déjà exacts.
// `DualQuaternion` n'ajoute rien à ces deux opérations (cf. [`Self::mul_dual`],
// [`Self::transform_point`], équivalents à `Transform::compose`/
// `Transform::transform_point`) — son intérêt propre est **l'interpolation**.
//
// Interpoler *séparément* la rotation (`slerp`) et la translation (`lerp`) de
// deux `Transform` ne correspond **pas**, en général, à un mouvement
// physiquement cohérent : si l'axe de rotation ne passe pas par l'origine, le
// chemin obtenu s'écarte de l'arc de cercle réel que suivrait un point rigide
// (cf. le test `sclerp_matches_screw_motion_circular_arc`, où l'interpolation
// naïve donne un point visiblement hors de la trajectoire correcte). Le
// quaternion dual permet [`Self::sclerp`] (*screw linear interpolation*) :
// interpolation à **vissage constant** (rotation autour d'un axe fixe +
// translation le long de cet axe, à vitesse angulaire et linéaire constantes)
// — la généralisation exacte de `slerp` à `SE(3)` entier, utilisée en
// robotique et en animation (« dual quaternion skinning »).
//
// ## Convention et hypothèses
//
// `qᵣ` est supposé **unitaire** (comme `Transform::rotation`) et `q_d`
// orthogonal à `qᵣ` (`⟨qᵣ, q_d⟩ = 0`, condition de « quaternion dual unitaire »
// satisfaite automatiquement par [`Self::from_rotation_translation`]). Sous
// cette hypothèse, [`Self::conjugate`] (conjugaison de quaternion appliquée
// aux deux parties) est l'**inverse** à deux côtés — comme pour un quaternion
// de rotation unitaire, dont l'inverse égale le conjugué.
//
// ## `pow` — puissance d'un vissage
//
// [`Self::pow`] élève un quaternion dual unitaire à la puissance réelle `t` :
// pour un vissage d'angle `θ` et de pas (translation le long de l'axe) `d`,
// `q̂ᵗ` est le vissage d'angle `t·θ` et de pas `t·d`, **même axe et même
// moment** (décalage perpendiculaire de l'axe par rapport à l'origine).
// [`Self::sclerp`] n'est qu'une application de `pow` au mouvement *relatif*
// entre deux poses : `sclerp(a, b, t) = a · (a⁻¹·b)ᵗ`, exactement l'analogue
// de `slerp(a, b, t) = a · (a⁻¹·b)ᵗ` restreint aux rotations pures.
//
// Les divisions par un dénominateur non puissance de deux (`sin(θ/2)`)
// utilisent l'opérateur `/` (division réelle), pas `recip()` — même précaution
// que [`Quaternion::from_rotation_matrix`] : `recip()` perdrait de la
// précision avant même la multiplication (cf. [`crate::dsp::mel`]). Diviser
// par `2` (puissance de deux) reste en revanche `recip()`, exact.

use core::ops::{Div, Mul};

use crate::fixed::{NumericScalar, RealScalar};

use super::quaternion::Quaternion;
use super::transform::Transform;

/// Quaternion dual `qᵣ + ε·q_d` représentant un déplacement rigide `SE(3)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualQuaternion<T> {
    /// Partie réelle (rotation, quaternion unitaire).
    pub real: Quaternion<T>,
    /// Partie duale (encodage de la translation, cf. [`Self::from_rotation_translation`]).
    pub dual: Quaternion<T>,
}

impl<T: NumericScalar> DualQuaternion<T> {
    /// Construit `qᵣ + ε·q_d` directement depuis ses deux parties.
    #[inline]
    pub fn new(real: Quaternion<T>, dual: Quaternion<T>) -> Self {
        Self { real, dual }
    }

    /// Déplacement identité (rotation nulle, translation nulle).
    #[inline]
    pub fn identity() -> Self {
        Self::new(Quaternion::identity(), Quaternion::zero())
    }

    /// Conjugué (conjugaison de quaternion sur les deux parties) : **inverse**
    /// à deux côtés pour un quaternion dual unitaire (cf. en-tête de module).
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self::new(self.real.conjugate(), self.dual.conjugate())
    }

    /// Produit de deux quaternions duaux (composition des déplacements) :
    /// `(r₁+ε·d₁)(r₂+ε·d₂) = r₁r₂ + ε·(r₁d₂ + d₁r₂)` (`ε² = 0`).
    ///
    /// `a.mul_dual(b).transform_point(p) == a.transform_point(b.transform_point(p))`
    /// — même convention que [`Transform::compose`].
    #[must_use]
    pub fn mul_dual(self, r: Self) -> Self {
        Self::new(
            self.real.mul_quat(r.real),
            self.real.mul_quat(r.dual) + self.dual.mul_quat(r.real),
        )
    }
}

impl<T: RealScalar> DualQuaternion<T> {
    /// Construit depuis une rotation (quaternion unitaire) et une translation
    /// : `q_d = ½ · t̂ · qᵣ`, `t̂` le quaternion pur de `translation`. C'est
    /// l'encodage standard qui rend [`Self::conjugate`] inverse à deux côtés
    /// et [`Self::to_rotation_translation`] exact.
    #[must_use]
    pub fn from_rotation_translation(rotation: Quaternion<T>, translation: [T; 3]) -> Self {
        let half = T::from_i32(2).recip(); // puissance de 2 : recip() exact.
        let dual = Quaternion::from_vector(translation)
            .mul_quat(rotation)
            .scale(half);
        Self::new(rotation, dual)
    }

    /// Décompose en `(rotation, translation)`, réciproque de
    /// [`Self::from_rotation_translation`] : `translation = 2 · (q_d · qᵣ*)`
    /// (partie vectorielle). Suppose `qᵣ` unitaire.
    #[must_use]
    pub fn to_rotation_translation(self) -> (Quaternion<T>, [T; 3]) {
        let two = T::from_i32(2);
        let t = self.dual.mul_quat(self.real.conjugate()).scale(two);
        (self.real, t.vector())
    }

    /// Transforme le point `p` par ce déplacement rigide : équivalent à
    /// [`Transform::transform_point`] via [`Self::to_rotation_translation`].
    #[must_use]
    pub fn transform_point(self, p: [T; 3]) -> [T; 3] {
        let (r, t) = self.to_rotation_translation();
        let rp = r.rotate_vector(p);
        [rp[0] + t[0], rp[1] + t[1], rp[2] + t[2]]
    }

    /// Renormalise la partie réelle à la norme unité (`qᵣ / |qᵣ|`, `q_d`
    /// mis à l'échelle identiquement) — corrige la dérive de norme après
    /// plusieurs [`Self::mul_dual`] successifs, comme
    /// [`Quaternion::normalize`] pour une rotation seule.
    #[must_use]
    pub fn normalize(self) -> Self {
        let inv = self.real.norm().recip();
        Self::new(self.real.scale(inv), self.dual.scale(inv))
    }
}

impl<T: RealScalar + Div<Output = T>> DualQuaternion<T> {
    /// Puissance réelle `t` d'un vissage unitaire (cf. en-tête de module) :
    /// un vissage d'angle `θ` et de pas `d` devient un vissage d'angle `t·θ`
    /// et de pas `t·d`, même axe et même moment.
    ///
    /// Cas particulier `θ ≈ 0` (translation pure, axe indéterminé) : la
    /// puissance est simplement la translation mise à l'échelle par `t` —
    /// même seuil que [`Quaternion::to_axis_angle`] pour la rotation quasi
    /// nulle.
    #[must_use]
    pub fn pow(self, t: T) -> Self {
        let one = T::one();
        let two = T::from_i32(2);
        let half = two.recip(); // puissance de 2 : recip() exact.
        let w = if self.real.w > one
        {
            one
        }
        else if self.real.w < -one
        {
            -one
        }
        else
        {
            self.real.w
        };
        let sin_half = (one - w * w).sqrt(); // sin(θ/2)
        let tiny = T::from_i32(10000).recip(); // 1e-4, cf. Quaternion::to_axis_angle.

        if sin_half < tiny
        {
            // Rotation quasi nulle : vissage = translation pure le long
            // d'un axe indéterminé. q_d = ½·t̂ (qᵣ ≈ identité) ⇒ t̂ = 2·q_d.
            let translation = self.dual.scale(two).vector();
            let scaled = [translation[0] * t, translation[1] * t, translation[2] * t];
            return Self::from_rotation_translation(Quaternion::identity(), scaled);
        }

        let theta = two * w.acos();
        let cos_half = w;
        let axis = [
            self.real.x / sin_half,
            self.real.y / sin_half,
            self.real.z / sin_half,
        ];
        // Pas (translation le long de l'axe) et moment (décalage perpendiculaire).
        let d = (-two * self.dual.w) / sin_half;
        let dual_vec = self.dual.vector();
        let m = [
            (dual_vec[0] - (d * half) * cos_half * axis[0]) / sin_half,
            (dual_vec[1] - (d * half) * cos_half * axis[1]) / sin_half,
            (dual_vec[2] - (d * half) * cos_half * axis[2]) / sin_half,
        ];

        let theta_t = theta * t;
        let d_t = d * t;
        let half_theta_t = theta_t * half;
        let (s, c) = (half_theta_t.sin(), half_theta_t.cos());
        let real_t = Quaternion::new(c, axis[0] * s, axis[1] * s, axis[2] * s);
        let dual_t_w = -(d_t * half) * s;
        let dual_t_vec = [
            s * m[0] + (d_t * half) * c * axis[0],
            s * m[1] + (d_t * half) * c * axis[1],
            s * m[2] + (d_t * half) * c * axis[2],
        ];
        let dual_t = Quaternion::new(dual_t_w, dual_t_vec[0], dual_t_vec[1], dual_t_vec[2]);
        Self::new(real_t, dual_t)
    }

    /// **Screw linear interpolation** (ScLERP) entre deux poses `a`/`b`,
    /// `t ∈ [0, 1]` : `a · (a⁻¹·b)ᵗ` — vissage à vitesse angulaire **et**
    /// linéaire constantes le long de l'axe fixe reliant `a` à `b` (cf.
    /// en-tête de module). `sclerp(a, b, 0) == a`, `sclerp(a, b, 1) == b`.
    ///
    /// Généralise [`Quaternion::slerp`] à `SE(3)` entier : quand `a`/`b` ont
    /// la même translation, `sclerp` coïncide avec `slerp` appliqué à la
    /// rotation seule.
    #[must_use]
    pub fn sclerp(a: Self, b: Self, t: T) -> Self {
        let diff = a.conjugate().mul_dual(b);
        a.mul_dual(diff.pow(t))
    }
}

impl<T: NumericScalar> Mul for DualQuaternion<T> {
    type Output = Self;
    /// Composition des déplacements (cf. [`Self::mul_dual`]).
    #[inline]
    fn mul(self, r: Self) -> Self {
        self.mul_dual(r)
    }
}

impl<T: RealScalar> From<Transform<T>> for DualQuaternion<T> {
    #[inline]
    fn from(t: Transform<T>) -> Self {
        Self::from_rotation_translation(t.rotation, t.translation)
    }
}

impl<T: RealScalar> From<DualQuaternion<T>> for Transform<T> {
    #[inline]
    fn from(q: DualQuaternion<T>) -> Self {
        let (rotation, translation) = q.to_rotation_translation();
        Transform::new(rotation, translation)
    }
}
