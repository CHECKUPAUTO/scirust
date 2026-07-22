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
// ## Interpolation
//
// [`Quaternion::slerp`] (interpolation sphérique à vitesse angulaire constante,
// via `acos`) et [`Quaternion::nlerp`] (linéaire normalisée, plus rapide) sont
// toutes deux disponibles ; [`Quaternion::to_axis_angle`] réciproque
// [`Quaternion::from_axis_angle`]. Ces trois-là reposent sur la trigonométrie
// inverse déterministe de [`RealScalar`] (`acos`), elle-même à bornes ULP
// prouvées en virgule fixe.
//
// [`Quaternion::squad`] généralise `slerp` à une **séquence** de mots-clés
// (animation, trajectoire d'orientation multi-points) : interpolation
// cubique de Shoemake, à vitesse angulaire continue aux jonctions, contre la
// discontinuité d'une suite de `slerp` indépendants. [`Quaternion::squad_setup`]
// calcule le quaternion auxiliaire de chaque mot-clé (à partir de ses deux
// voisins) requis par `squad` — bâti entièrement sur `slerp`/`to_axis_angle`/
// `from_axis_angle` déjà existants, sans nouvelle primitive numérique.
//
// ## Autres représentations d'orientation
//
// [`Quaternion::from_rotation_matrix`] réciproque [`Quaternion::to_rotation_matrix`]
// (méthode de Shepperd, stable numériquement) ; [`Quaternion::from_euler`] /
// [`Quaternion::to_euler`] convertissent vers/depuis les angles d'Euler
// (convention Tait-Bryan aéronautique Z-Y-X : roulis/tangage/lacet). Ces
// quatre méthodes demandent `T: RealScalar + Div<Output = T>` : leurs
// dénominateurs ne sont pas des puissances de deux, donc `recip()` perdrait
// trop de précision en virgule fixe (voir [`crate::dsp::mel`]).

use core::ops::{Add, Div, Mul, Neg, Sub};

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

    /// Interpolation **sphérique** (slerp) entre deux quaternions unitaires,
    /// `t ∈ [0,1]`, à **vitesse angulaire constante** le long du plus court arc.
    ///
    /// `slerp = (sin((1−t)Ω)·a + sin(tΩ)·b) / sin Ω`, `Ω = acos(a·b)`. Bascule
    /// automatiquement sur [`Self::nlerp`] quand `a` et `b` sont quasi colinéaires
    /// (`a·b > 0.9995`), où `sin Ω → 0` rendrait la division instable.
    #[inline]
    pub fn slerp(a: Self, b: Self, t: T) -> Self {
        let mut d = a.dot(b);
        // Plus court arc : aligne le signe de b sur a.
        let mut b = b;
        if d < T::zero()
        {
            b = -b;
            d = -d;
        }
        // Quasi colinéaires → nlerp (évite la division par sin Ω ≈ 0).
        let threshold = T::from_i32(9995) * T::from_i32(10000).recip();
        if d > threshold
        {
            return Self::nlerp(a, b, t);
        }
        let omega = d.acos();
        let inv_sin = omega.sin().recip();
        let w1 = ((T::one() - t) * omega).sin() * inv_sin;
        let w2 = (t * omega).sin() * inv_sin;
        a.scale(w1) + b.scale(w2)
    }

    /// Décompose un quaternion **unitaire** en `(axe unitaire, angle)` (radians),
    /// réciproque de [`Self::from_axis_angle`]. `angle = 2·acos(w) ∈ [0, 2π]`.
    ///
    /// Pour une rotation quasi nulle (`sin(angle/2) ≈ 0`, axe indéterminé),
    /// renvoie l'axe `+x` par convention.
    #[inline]
    pub fn to_axis_angle(self) -> ([T; 3], T) {
        // Borne w à [-1, 1] (robustesse aux quaternions légèrement non unitaires).
        let one = T::one();
        let w = if self.w > one
        {
            one
        }
        else if self.w < -one
        {
            -one
        }
        else
        {
            self.w
        };
        let angle = T::from_i32(2) * w.acos();
        let s = (one - w * w).sqrt(); // sin(angle/2)
        let tiny = T::from_i32(10000).recip(); // 1e-4
        if s < tiny
        {
            ([one, T::zero(), T::zero()], angle)
        }
        else
        {
            let inv = s.recip();
            ([self.x * inv, self.y * inv, self.z * inv], angle)
        }
    }

    /// Logarithme d'un quaternion **unitaire** : pour `q = cos θ + sin θ·v̂`
    /// (`θ = angle/2` de la représentation axe-angle), `ln(q) = θ·v̂`
    /// (quaternion **pur**, partie réelle nulle). Utilisé par
    /// [`Self::squad_setup`] — pas destiné à un usage général (le logarithme
    /// quaternionique complet, valable pour tout `q` non nul, existe pour
    /// [`crate::hypercomplex::OctonionSimd`] mais n'est pas nécessaire ici).
    #[inline]
    fn ln_unit(self) -> Self {
        let (axis, angle) = self.to_axis_angle();
        let half = angle * T::from_i32(2).recip();
        Self::from_vector([axis[0] * half, axis[1] * half, axis[2] * half])
    }

    /// Exponentielle d'un quaternion **pur** `v = θ·v̂` (réciproque de
    /// [`Self::ln_unit`] sur son image) : `exp(v) = cos θ + sin θ·v̂`,
    /// `θ = ‖v‖`. Renvoie l'identité pour `v ≈ 0` (comme [`Self::to_axis_angle`]
    /// pour une rotation quasi nulle — même seuil).
    #[inline]
    fn exp_pure(self) -> Self {
        let v = self.vector();
        let phi = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let tiny = T::from_i32(10000).recip(); // 1e-4, même seuil que to_axis_angle
        if phi < tiny
        {
            Self::identity()
        }
        else
        {
            let inv = phi.recip();
            let axis = [v[0] * inv, v[1] * inv, v[2] * inv];
            Self::from_axis_angle(axis, phi * T::from_i32(2))
        }
    }

    /// Quaternion auxiliaire ("tangent") de Shoemake au mot-clé `q`, à partir
    /// de ses voisins `q_prev`/`q_next` dans la séquence — nécessaire pour
    /// interpoler *entre* deux mots-clés consécutifs avec [`Self::squad`] à
    /// vitesse angulaire continue (analogue quaternionique d'une tangente de
    /// Catmull-Rom).
    ///
    /// `s = q · exp(−(ln(q⁻¹·q_prev) + ln(q⁻¹·q_next)) / 4)` (Shoemake, 1987).
    #[must_use]
    pub fn squad_setup(q_prev: Self, q: Self, q_next: Self) -> Self {
        let inv_q = q.inverse();
        let log_prev = inv_q.mul_quat(q_prev).ln_unit();
        let log_next = inv_q.mul_quat(q_next).ln_unit();
        let quarter = T::from_i32(4).recip(); // puissance de 2 : recip() exact
        let exponent = -((log_prev + log_next).scale(quarter));
        q.mul_quat(exponent.exp_pure())
    }

    /// Interpolation cubique sphérique de Shoemake (**SQUAD**) entre deux
    /// mots-clés unitaires consécutifs `q0`/`q1`, `t ∈ [0,1]`, avec leurs
    /// quaternions auxiliaires `s0`/`s1` ([`Self::squad_setup`]) : double
    /// [`Self::slerp`] imbriqué. Contrairement à une suite de `slerp`
    /// indépendants (vitesse angulaire discontinue à chaque jonction),
    /// produit une trajectoire à vitesse angulaire **continue** aux mots-clés
    /// — l'équivalent quaternionique d'une spline cubique C¹.
    ///
    /// `squad(t) = slerp(slerp(q0, q1, t), slerp(s0, s1, t), 2t(1−t))`.
    #[must_use]
    pub fn squad(q0: Self, q1: Self, s0: Self, s1: Self, t: T) -> Self {
        let two = T::from_i32(2);
        let inner = Self::slerp(q0, q1, t);
        let outer = Self::slerp(s0, s1, t);
        let blend = two * t * (T::one() - t);
        Self::slerp(inner, outer, blend)
    }
}

impl<T: RealScalar + Div<Output = T>> Quaternion<T> {
    /// Reconstruit le quaternion unitaire correspondant à une matrice de
    /// rotation 3×3 (lignes), réciproque de [`Self::to_rotation_matrix`].
    ///
    /// Méthode de Shepperd (branche sur le plus grand parmi la trace et les
    /// éléments diagonaux) : évite de diviser par une racine proche de zéro,
    /// contrairement à la formule naïve à partir de la seule trace. Utilise la
    /// division réelle (`/`), pas `recip()` : `s` n'est pas une puissance de
    /// deux, et `x * y.recip()` perdrait trop de précision en virgule fixe
    /// (même leçon que [`crate::dsp::mel`], dont les dénominateurs — `700`,
    /// `2595` — ne sont pas non plus des puissances de deux).
    #[must_use]
    pub fn from_rotation_matrix(m: [[T; 3]; 3]) -> Self {
        let trace = m[0][0] + m[1][1] + m[2][2];
        let two = T::from_i32(2);
        let four = T::from_i32(4);
        if trace > T::zero()
        {
            let s = (trace + T::one()).sqrt() * two; // s = 4w
            Self::new(
                s / four,
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
            )
        }
        else if m[0][0] > m[1][1] && m[0][0] > m[2][2]
        {
            let s = (T::one() + m[0][0] - m[1][1] - m[2][2]).sqrt() * two; // s = 4x
            Self::new(
                (m[2][1] - m[1][2]) / s,
                s / four,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
            )
        }
        else if m[1][1] > m[2][2]
        {
            let s = (T::one() + m[1][1] - m[0][0] - m[2][2]).sqrt() * two; // s = 4y
            Self::new(
                (m[0][2] - m[2][0]) / s,
                (m[0][1] + m[1][0]) / s,
                s / four,
                (m[1][2] + m[2][1]) / s,
            )
        }
        else
        {
            let s = (T::one() + m[2][2] - m[0][0] - m[1][1]).sqrt() * two; // s = 4z
            Self::new(
                (m[1][0] - m[0][1]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                s / four,
            )
        }
    }

    /// Quaternion unitaire depuis des angles d'Euler (radians), convention
    /// Tait-Bryan intrinsèque Z-Y-X (aéronautique : roulis `roll` autour de
    /// `x`, tangage `pitch` autour de `y`, lacet `yaw` autour de `z`,
    /// appliqués dans l'ordre yaw puis pitch puis roll).
    #[must_use]
    pub fn from_euler(roll: T, pitch: T, yaw: T) -> Self {
        let half = T::from_i32(2).recip(); // puissance de 2 : recip() exact
        let (sr, cr) = ((roll * half).sin(), (roll * half).cos());
        let (sp, cp) = ((pitch * half).sin(), (pitch * half).cos());
        let (sy, cy) = ((yaw * half).sin(), (yaw * half).cos());
        Self::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    }

    /// Décompose un quaternion **unitaire** en angles d'Euler (radians),
    /// réciproque de [`Self::from_euler`] (convention Tait-Bryan Z-Y-X).
    ///
    /// Au gimbal lock (`|2·(wy − zx)|` à moins de `0.001` de `1` : tangage
    /// `±π/2`), roulis et lacet ne sont **pas** individuellement déterminés —
    /// seule leur différence (pôle nord) ou leur somme (pôle sud) l'est.
    /// Convention : `roll = 0`, et `yaw` porte toute l'information restante
    /// (`from_euler(0, ±π/2, yaw)` reproduit exactement la même rotation).
    /// Loin du gimbal lock, le seuil `0.001` protège aussi `asin` : au
    /// voisinage de `±1`, sa dérivée diverge, donc la moindre imprécision
    /// d'arrondi sur `sin_pitch` (flottant ou virgule fixe) y serait amplifiée
    /// démesurément.
    #[must_use]
    pub fn to_euler(self) -> (T, T, T) {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let two = T::from_i32(2);
        let one = T::one();

        let sin_pitch = two * (w * y - z * x);
        let gimbal = T::from_i32(999) / T::from_i32(1000);
        if sin_pitch >= gimbal
        {
            return (T::zero(), T::pi() / two, -(two * x.atan2(w)));
        }
        if sin_pitch <= -gimbal
        {
            return (T::zero(), -(T::pi() / two), two * x.atan2(w));
        }

        let roll = (two * (w * x + y * z)).atan2(one - two * (x * x + y * y));
        let pitch = sin_pitch.asin();
        let yaw = (two * (w * z + x * y)).atan2(one - two * (y * y + z * z));
        (roll, pitch, yaw)
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
