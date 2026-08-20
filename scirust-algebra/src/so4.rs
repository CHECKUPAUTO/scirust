//! Deterministic primitives for rotations in four Euclidean dimensions.
//!
//! The module is deliberately application-neutral.  It exposes the two forms
//! that are useful when studying `SO(4)` numerically:
//!
//! - a canonical pair of orthogonal `SO(2)` plane rotations, which makes the
//!   spectral decomposition of one-parameter `SO(4)` subgroups explicit; and
//! - the `Spin(4) ≅ SU(2) × SU(2)` action on `R^4 ≅ H`, represented by a pair
//!   of unit quaternions.
//!
//! No attention-, RoPE-, or model-specific policy lives here.  Downstream
//! projects can therefore use these primitives as a mathematical oracle
//! without coupling SciRust to a particular architecture.

use crate::lie::Su2;

/// Four-dimensional Euclidean vector, identified with the quaternion
/// `w + x i + y j + z k` when used with [`Spin4Rotor`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    /// Scalar / first coordinate.
    pub w: f64,
    /// Second coordinate.
    pub x: f64,
    /// Third coordinate.
    pub y: f64,
    /// Fourth coordinate.
    pub z: f64,
}

impl Vec4 {
    /// Construct a vector.
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Euclidean dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Squared Euclidean norm.
    #[inline]
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean norm.
    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }
}

/// Canonical double rotation in two orthogonal planes of `R^4`.
///
/// The first angle rotates `(w, x)` and the second rotates `(y, z)`.  Every
/// one-parameter subgroup of `SO(4)` is orthogonally conjugate to this form,
/// with angles linear in the subgroup parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct So4DoubleRotation {
    /// Angle in the `(w, x)` plane, in radians.
    pub angle_01: f64,
    /// Angle in the `(y, z)` plane, in radians.
    pub angle_23: f64,
}

impl So4DoubleRotation {
    /// Construct a canonical double rotation.
    pub const fn new(angle_01: f64, angle_23: f64) -> Self {
        Self { angle_01, angle_23 }
    }

    /// Identity rotation.
    pub const fn identity() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Construct the element reached at scalar parameter `t` by a
    /// one-parameter subgroup with angular frequencies `omega_01` and
    /// `omega_23`.
    #[inline]
    pub fn one_parameter(t: f64, omega_01: f64, omega_23: f64) -> Self {
        Self::new(t * omega_01, t * omega_23)
    }

    /// Compose two rotations in the same canonical planes.
    #[inline]
    pub fn compose(self, rhs: Self) -> Self {
        Self::new(self.angle_01 + rhs.angle_01, self.angle_23 + rhs.angle_23)
    }

    /// Inverse rotation.
    #[inline]
    pub fn inverse(self) -> Self {
        Self::new(-self.angle_01, -self.angle_23)
    }

    /// Apply the rotation to a vector.
    pub fn apply(self, v: Vec4) -> Vec4 {
        let (s01, c01) = self.angle_01.sin_cos();
        let (s23, c23) = self.angle_23.sin_cos();
        Vec4::new(
            v.w * c01 - v.x * s01,
            v.w * s01 + v.x * c01,
            v.y * c23 - v.z * s23,
            v.y * s23 + v.z * c23,
        )
    }

    /// Return whether this is an isoclinic canonical rotation up to `tol`.
    ///
    /// Equal plane angles describe the canonical left-isoclinic case; opposite
    /// angles describe the corresponding right-isoclinic convention.
    pub fn is_isoclinic(self, tol: f64) -> bool {
        (self.angle_01 - self.angle_23).abs() <= tol
            || (self.angle_01 + self.angle_23).abs() <= tol
    }
}

/// A unit-quaternion pair implementing the standard `Spin(4)` action
///
/// `v ↦ left · v · conjugate(right)`.
///
/// The pair `(left, right)` and `(-left, -right)` induce the same element of
/// `SO(4)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spin4Rotor {
    /// Left `SU(2)` factor.
    pub left: Su2,
    /// Right `SU(2)` factor.
    pub right: Su2,
}

impl Spin4Rotor {
    /// Construct a rotor from already-normalized `SU(2)` factors.
    pub const fn new(left: Su2, right: Su2) -> Self {
        Self { left, right }
    }

    /// Identity rotor.
    pub const fn identity() -> Self {
        Self {
            left: Su2 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
            right: Su2 { w: 1.0, x: 0.0, y: 0.0, z: 0.0 },
        }
    }

    /// Construct a left-isoclinic one-parameter rotor from a fixed unit
    /// imaginary axis and angle `angle`.
    ///
    /// `axis` is interpreted as the imaginary quaternion `(x, y, z)` and must
    /// be non-zero.  The returned rotor uses the full quaternion phase
    /// `cos(angle) + axis * sin(angle)`; callers that use the half-angle
    /// convention can pass `angle / 2` explicitly.
    pub fn left_isoclinic(axis: [f64; 3], angle: f64) -> Option<Self> {
        let left = axis_phase(axis, angle)?;
        Some(Self::new(left, Self::identity().right))
    }

    /// Construct a right-isoclinic one-parameter rotor from a fixed unit
    /// imaginary axis and angle `angle`.
    pub fn right_isoclinic(axis: [f64; 3], angle: f64) -> Option<Self> {
        let right = axis_phase(axis, angle)?;
        Some(Self::new(Self::identity().left, right))
    }

    /// Compose two `Spin(4)` rotors so that the returned rotor applies `rhs`
    /// first and `self` second.
    pub fn compose(self, rhs: Self) -> Self {
        // L1(L2 v R2*)R1* = (L1 L2) v (R1 R2)*.
        Self::new(self.left.compose(rhs.left), self.right.compose(rhs.right))
    }

    /// Inverse rotor.
    pub fn inverse(self) -> Self {
        Self::new(conjugate(self.left), conjugate(self.right))
    }

    /// Apply the rotor to a four-vector.
    pub fn apply(self, v: Vec4) -> Vec4 {
        let q = Su2 { w: v.w, x: v.x, y: v.y, z: v.z };
        let out = self.left.compose(q).compose(conjugate(self.right));
        Vec4::new(out.w, out.x, out.y, out.z)
    }
}

fn conjugate(q: Su2) -> Su2 {
    Su2 { w: q.w, x: -q.x, y: -q.y, z: -q.z }
}

fn axis_phase(axis: [f64; 3], angle: f64) -> Option<Su2> {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if norm == 0.0 || !norm.is_finite() || !angle.is_finite() {
        return None;
    }
    let (s, c) = angle.sin_cos();
    Su2::normalized(c, s * axis[0] / norm, s * axis[1] / norm, s * axis[2] / norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    fn close_vec(a: Vec4, b: Vec4) {
        close(a.w, b.w);
        close(a.x, b.x);
        close(a.y, b.y);
        close(a.z, b.z);
    }

    #[test]
    fn double_rotation_preserves_norm() {
        let v = Vec4::new(0.25, -0.75, 1.5, 2.0);
        let r = So4DoubleRotation::new(0.37, -1.1);
        close(r.apply(v).norm_squared(), v.norm_squared());
    }

    #[test]
    fn one_parameter_relative_law_is_exact_up_to_roundoff() {
        let v = Vec4::new(0.3, -0.2, 0.9, 1.1);
        let wm = So4DoubleRotation::one_parameter(17.0, 0.07, 0.013);
        let wn = So4DoubleRotation::one_parameter(5.0, 0.07, 0.013);
        let relative = wn.inverse().compose(wm);
        let expected = So4DoubleRotation::one_parameter(12.0, 0.07, 0.013);
        close_vec(relative.apply(v), expected.apply(v));
    }

    #[test]
    fn canonical_left_isoclinic_matches_two_equal_plane_rotations() {
        let v = Vec4::new(0.2, -0.4, 0.7, 1.3);
        let angle = 0.41;
        let spin = Spin4Rotor::left_isoclinic([1.0, 0.0, 0.0], angle).unwrap();
        let canonical = So4DoubleRotation::new(angle, angle);
        close_vec(spin.apply(v), canonical.apply(v));
    }

    #[test]
    fn general_spin4_rotor_preserves_norm() {
        let left = axis_phase([1.0, 2.0, -1.0], 0.23).unwrap();
        let right = axis_phase([-0.5, 1.0, 3.0], -0.61).unwrap();
        let r = Spin4Rotor::new(left, right);
        let v = Vec4::new(-0.2, 0.9, 1.1, -2.4);
        close(r.apply(v).norm_squared(), v.norm_squared());
    }
}
