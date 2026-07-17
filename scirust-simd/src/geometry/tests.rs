// scirust-simd/src/geometry/tests.rs
//
// Validation de `Quaternion<T>`. Le cœur des tests est **générique** : les mêmes
// assertions s'exécutent sur `f32`, `f64` et `Q16_16` (virgule fixe), prouvant
// que l'implémentation unique est correcte pour tous les scalaires. On combine :
//  * identités **exactes** (table de Hamilton, i² = −1) — valables au bit près ;
//  * propriétés géométriques à tolérance (rotation ↦ conserve la longueur,
//    composition, angle-axe) mesurées contre une référence `f64` ;
//  * déterminisme bit-à-bit du chemin virgule fixe.

use super::{Quaternion, Transform};
use crate::fixed::{NumericScalar, Q16_16, RealScalar};

// ------------------------------------------------------------------ //
//  Petits ponts de conversion (scalaire ↔ f64) pour tests génériques  //
// ------------------------------------------------------------------ //

trait Scalar: RealScalar {
    /// Vers `f64` pour comparaison à une référence.
    fn to_f64(self) -> f64;
    /// Depuis `f64` (construction des cas de test).
    fn of(v: f64) -> Self;
    /// Tolérance absolue (en `f64`) admise pour ce type.
    const TOL: f64;
}

impl Scalar for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn of(v: f64) -> Self {
        v as f32
    }
    const TOL: f64 = 1e-5;
}
impl Scalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn of(v: f64) -> Self {
        v
    }
    const TOL: f64 = 1e-9;
}
impl Scalar for Q16_16 {
    fn to_f64(self) -> f64 {
        Q16_16::to_f64(self)
    }
    fn of(v: f64) -> Self {
        Q16_16::try_from(v).unwrap()
    }
    // sin/cos ≤ 1 ULP + accumulation des produits Q16.16 ⇒ ~2e-3.
    const TOL: f64 = 2e-3;
}

fn qof<T: Scalar>(w: f64, x: f64, y: f64, z: f64) -> Quaternion<T> {
    Quaternion::new(T::of(w), T::of(x), T::of(y), T::of(z))
}

/// Vecteur unité canonique `i` du quaternion.
fn unit<T: NumericScalar>(i: usize) -> Quaternion<T> {
    let (o, z) = (T::one(), T::zero());
    match i
    {
        0 => Quaternion::new(o, z, z, z),
        1 => Quaternion::new(z, o, z, z),
        2 => Quaternion::new(z, z, o, z),
        _ => Quaternion::new(z, z, z, o),
    }
}

// ------------------------------------------------------------------ //
//  Identités exactes (table de Hamilton)                              //
// ------------------------------------------------------------------ //

fn check_hamilton<T: NumericScalar + core::fmt::Debug>() {
    let (i, j, k) = (unit::<T>(1), unit::<T>(2), unit::<T>(3));
    // ij = k, jk = i, ki = j — EXACT (produits de 0 et 1).
    assert_eq!(i * j, k);
    assert_eq!(j * k, i);
    assert_eq!(k * i, j);
    // i² = j² = k² = −1.
    let neg_one = Quaternion::new(-T::one(), T::zero(), T::zero(), T::zero());
    assert_eq!(i * i, neg_one);
    assert_eq!(j * j, neg_one);
    assert_eq!(k * k, neg_one);
    // Non-commutativité : ji = −k = −(ij).
    assert_eq!(j * i, -k);
    // Identité neutre.
    assert_eq!(Quaternion::<T>::identity() * i, i);
    assert_eq!(i * Quaternion::<T>::identity(), i);
}

#[test]
fn hamilton_table_exact_all_scalars() {
    check_hamilton::<f32>();
    check_hamilton::<f64>();
    check_hamilton::<Q16_16>(); // exact aussi en virgule fixe
}

// ------------------------------------------------------------------ //
//  Rotations (générique sur le scalaire)                              //
// ------------------------------------------------------------------ //

fn approx_vec<T: Scalar>(got: [T; 3], want: [f64; 3], ctx: &str) {
    for k in 0..3
    {
        let d = (got[k].to_f64() - want[k]).abs();
        assert!(
            d <= T::TOL,
            "{ctx}: composante {k} écart {d:.2e} > {:.0e}",
            T::TOL
        );
    }
}

fn check_axis_angle<T: Scalar>() {
    let pi = std::f64::consts::PI;
    // 90° autour de +z : x → y, y → −x, z → z.
    let q = Quaternion::<T>::from_axis_angle([T::of(0.0), T::of(0.0), T::of(1.0)], T::of(pi / 2.0));
    approx_vec(
        q.rotate_vector([T::of(1.0), T::of(0.0), T::of(0.0)]),
        [0.0, 1.0, 0.0],
        "Rz90·x",
    );
    approx_vec(
        q.rotate_vector([T::of(0.0), T::of(1.0), T::of(0.0)]),
        [-1.0, 0.0, 0.0],
        "Rz90·y",
    );
    approx_vec(
        q.rotate_vector([T::of(0.0), T::of(0.0), T::of(1.0)]),
        [0.0, 0.0, 1.0],
        "Rz90·z",
    );

    // 180° autour de +x : y → −y, z → −z.
    let q = Quaternion::<T>::from_axis_angle([T::of(1.0), T::of(0.0), T::of(0.0)], T::of(pi));
    approx_vec(
        q.rotate_vector([T::of(0.0), T::of(1.0), T::of(0.0)]),
        [0.0, -1.0, 0.0],
        "Rx180·y",
    );
    approx_vec(
        q.rotate_vector([T::of(0.0), T::of(0.0), T::of(1.0)]),
        [0.0, 0.0, -1.0],
        "Rx180·z",
    );
}

#[test]
fn axis_angle_rotations_all_scalars() {
    check_axis_angle::<f32>();
    check_axis_angle::<f64>();
    check_axis_angle::<Q16_16>();
}

fn check_rotate_matches_matrix<T: Scalar>() {
    // Axe quelconque, angle quelconque.
    let axis = [T::of(0.3), T::of(-0.6), T::of(0.75)];
    let q = Quaternion::<T>::from_axis_angle(axis, T::of(0.9));
    let m = q.to_rotation_matrix();
    let v = [T::of(0.5), T::of(-0.25), T::of(0.8)];
    let rv = q.rotate_vector(v);
    // m·v
    let mv = [
        (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2]).to_f64(),
        (m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2]).to_f64(),
        (m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]).to_f64(),
    ];
    approx_vec(rv, mv, "rotate_vector vs matrice");
}

#[test]
fn rotate_vector_matches_rotation_matrix() {
    check_rotate_matches_matrix::<f32>();
    check_rotate_matches_matrix::<f64>();
    check_rotate_matches_matrix::<Q16_16>();
}

fn check_composition<T: Scalar>() {
    let q1 = Quaternion::<T>::from_axis_angle([T::of(0.0), T::of(0.0), T::of(1.0)], T::of(0.5));
    let q2 = Quaternion::<T>::from_axis_angle([T::of(1.0), T::of(0.0), T::of(0.0)], T::of(0.7));
    let v = [T::of(0.2), T::of(0.9), T::of(-0.3)];
    // rotate(q1 ⊗ q2, v) == rotate(q1, rotate(q2, v)).
    let composed = (q1 * q2).rotate_vector(v);
    let sequential = q1.rotate_vector(q2.rotate_vector(v));
    let seq_f = [
        sequential[0].to_f64(),
        sequential[1].to_f64(),
        sequential[2].to_f64(),
    ];
    approx_vec(composed, seq_f, "composition Hamilton");
}

#[test]
fn hamilton_product_composes_rotations() {
    check_composition::<f32>();
    check_composition::<f64>();
    check_composition::<Q16_16>();
}

fn check_length_preserved<T: Scalar>() {
    let q = Quaternion::<T>::from_axis_angle([T::of(0.2), T::of(0.5), T::of(-0.9)], T::of(1.3));
    let v = [T::of(0.7), T::of(-0.4), T::of(0.55)];
    let len_before = (0.7f64 * 0.7 + 0.4 * 0.4 + 0.55 * 0.55).sqrt();
    let rv = q.rotate_vector(v);
    let len_after =
        (rv[0].to_f64().powi(2) + rv[1].to_f64().powi(2) + rv[2].to_f64().powi(2)).sqrt();
    assert!(
        (len_before - len_after).abs() <= T::TOL * 4.0,
        "longueur {len_before} → {len_after}"
    );
}

#[test]
fn rotation_preserves_length() {
    check_length_preserved::<f32>();
    check_length_preserved::<f64>();
    check_length_preserved::<Q16_16>();
}

// ------------------------------------------------------------------ //
//  Norme / normalisation / inverse                                    //
// ------------------------------------------------------------------ //

fn check_normalize_inverse<T: Scalar>() {
    let q = qof::<T>(0.5, -0.25, 0.75, -0.125);
    // normalize ⇒ norme ≈ 1.
    let n = q.normalize().norm().to_f64();
    assert!((n - 1.0).abs() <= T::TOL * 4.0, "‖normalize(q)‖ = {n}");
    // q · q⁻¹ ≈ identité.
    let prod = q * q.inverse();
    approx_vec([prod.x, prod.y, prod.z], [0.0, 0.0, 0.0], "q·q⁻¹ vecteur");
    assert!(
        (prod.w.to_f64() - 1.0).abs() <= T::TOL * 4.0,
        "q·q⁻¹ scalaire"
    );
    // Pour un quaternion unitaire, inverse == conjugué.
    let u = q.normalize();
    let inv = u.inverse();
    let conj = u.conjugate();
    approx_vec(
        [inv.w, inv.x, inv.y],
        [conj.w.to_f64(), conj.x.to_f64(), conj.y.to_f64()],
        "unit inv=conj",
    );
}

#[test]
fn normalize_and_inverse() {
    check_normalize_inverse::<f32>();
    check_normalize_inverse::<f64>();
    check_normalize_inverse::<Q16_16>();
}

// ------------------------------------------------------------------ //
//  nlerp                                                              //
// ------------------------------------------------------------------ //

fn check_nlerp<T: Scalar>() {
    let a = Quaternion::<T>::from_axis_angle([T::of(0.0), T::of(0.0), T::of(1.0)], T::of(0.0));
    let b = Quaternion::<T>::from_axis_angle([T::of(0.0), T::of(0.0), T::of(1.0)], T::of(1.0));
    // Extrémités : nlerp(a,b,0) = a, nlerp(a,b,1) = b (déjà unitaires).
    let at0 = Quaternion::nlerp(a, b, T::of(0.0));
    let at1 = Quaternion::nlerp(a, b, T::of(1.0));
    approx_vec(
        [at0.w, at0.x, at0.z],
        [a.w.to_f64(), a.x.to_f64(), a.z.to_f64()],
        "nlerp t=0",
    );
    approx_vec(
        [at1.w, at1.x, at1.z],
        [b.w.to_f64(), b.x.to_f64(), b.z.to_f64()],
        "nlerp t=1",
    );
    // Milieu : unitaire et « entre » a et b.
    let mid = Quaternion::nlerp(a, b, T::of(0.5));
    assert!(
        (mid.norm().to_f64() - 1.0).abs() <= T::TOL * 4.0,
        "nlerp milieu unitaire"
    );
}

#[test]
fn nlerp_interpolates() {
    check_nlerp::<f32>();
    check_nlerp::<f64>();
    check_nlerp::<Q16_16>();
}

// ------------------------------------------------------------------ //
//  Spécifique virgule fixe : accord flottant + déterminisme           //
// ------------------------------------------------------------------ //

#[test]
fn fixed_matches_float_to_resolution() {
    // La MÊME rotation, calculée en f64 et en Q16.16, coïncide à la résolution.
    let axis_f = [0.36f64, 0.48, 0.8]; // unitaire
    let angle = 1.1f64;
    let v = [0.6f64, -0.2, 0.9];

    let qf = Quaternion::<f64>::from_axis_angle(axis_f, angle);
    let rvf = qf.rotate_vector(v);

    let qx = Quaternion::<Q16_16>::from_axis_angle(
        [
            Q16_16::of(axis_f[0]),
            Q16_16::of(axis_f[1]),
            Q16_16::of(axis_f[2]),
        ],
        Q16_16::of(angle),
    );
    let rvx = qx.rotate_vector([Q16_16::of(v[0]), Q16_16::of(v[1]), Q16_16::of(v[2])]);

    for k in 0..3
    {
        let d = (rvf[k] - rvx[k].to_f64()).abs();
        assert!(d < 2e-3, "composante {k}: |Δ| = {d:.2e}");
    }
}

#[test]
fn fixed_rotation_is_bit_deterministic() {
    // Deux évaluations indépendantes ⇒ bits identiques (aucun état caché, aucune
    // dépendance au matériel flottant). C'est la garantie centrale.
    let make = || {
        let q = Quaternion::<Q16_16>::from_axis_angle(
            [Q16_16::of(0.3), Q16_16::of(-0.6), Q16_16::of(0.75)],
            Q16_16::of(0.9),
        );
        q.rotate_vector([Q16_16::of(0.5), Q16_16::of(-0.25), Q16_16::of(0.8)])
    };
    let a = make();
    let b = make();
    for k in 0..3
    {
        assert_eq!(
            a[k].to_raw(),
            b[k].to_raw(),
            "composante {k} non déterministe"
        );
    }
}

// ------------------------------------------------------------------ //
//  slerp / to_axis_angle (trigonométrie inverse)                      //
// ------------------------------------------------------------------ //

fn approx_quat<T: Scalar>(got: Quaternion<T>, want: [f64; 4], ctx: &str) {
    let g = [
        got.w.to_f64(),
        got.x.to_f64(),
        got.y.to_f64(),
        got.z.to_f64(),
    ];
    for k in 0..4
    {
        assert!(
            (g[k] - want[k]).abs() <= T::TOL * 4.0,
            "{ctx}: composante {k} {} vs {}",
            g[k],
            want[k]
        );
    }
}

fn check_slerp<T: Scalar>() {
    // a = identité, b = rotation de 1.4 rad autour de +z.
    let a = Quaternion::<T>::from_axis_angle([T::of(0.0), T::of(0.0), T::of(1.0)], T::of(0.0));
    let b = Quaternion::<T>::from_axis_angle([T::of(0.0), T::of(0.0), T::of(1.0)], T::of(1.4));

    // Extrémités.
    approx_quat(
        Quaternion::slerp(a, b, T::of(0.0)),
        [a.w.to_f64(), a.x.to_f64(), a.y.to_f64(), a.z.to_f64()],
        "slerp t=0",
    );
    approx_quat(
        Quaternion::slerp(a, b, T::of(1.0)),
        [b.w.to_f64(), b.x.to_f64(), b.y.to_f64(), b.z.to_f64()],
        "slerp t=1",
    );

    // À vitesse angulaire constante : slerp(a, b, ½) = rotation de 0.7 rad
    // autour de +z, dont le quaternion est (cos(0.35), 0, 0, sin(0.35)).
    let mid = Quaternion::slerp(a, b, T::of(0.5));
    approx_quat(
        mid,
        [(0.35f64).cos(), 0.0, 0.0, (0.35f64).sin()],
        "slerp milieu",
    );

    // Quasi colinéaires → repli nlerp, toujours unitaire (pas de division 0/0).
    let close =
        Quaternion::<T>::from_axis_angle([T::of(0.0), T::of(0.0), T::of(1.0)], T::of(0.001));
    let s = Quaternion::slerp(a, close, T::of(0.5));
    assert!(
        (s.norm().to_f64() - 1.0).abs() <= T::TOL * 4.0,
        "slerp quasi colinéaire unitaire"
    );
}

#[test]
fn slerp_constant_velocity_all_scalars() {
    check_slerp::<f32>();
    check_slerp::<f64>();
    check_slerp::<Q16_16>();
}

fn check_axis_angle_roundtrip<T: Scalar>() {
    // Axe unitaire (f64) + angle dans (0, π).
    let n = (0.3f64 * 0.3 + 0.6 * 0.6 + 0.75 * 0.75).sqrt();
    let ax = [0.3 / n, -0.6 / n, 0.75 / n];
    let angle_in = 1.2f64;
    let q = Quaternion::<T>::from_axis_angle(
        [T::of(ax[0]), T::of(ax[1]), T::of(ax[2])],
        T::of(angle_in),
    );
    let (axis, angle) = q.to_axis_angle();
    assert!(
        (angle.to_f64() - angle_in).abs() <= T::TOL * 8.0,
        "angle {} vs {angle_in}",
        angle.to_f64()
    );
    for k in 0..3
    {
        assert!(
            (axis[k].to_f64() - ax[k]).abs() <= T::TOL * 8.0,
            "axe {k}: {} vs {}",
            axis[k].to_f64(),
            ax[k]
        );
    }
    // Rotation quasi nulle : axe conventionnel +x, pas de panique.
    let idn = Quaternion::<T>::identity();
    let (axis0, angle0) = idn.to_axis_angle();
    assert!(angle0.to_f64().abs() <= T::TOL * 8.0);
    assert!(axis0[0] == T::one(), "axe par défaut != +x");
}

#[test]
fn to_axis_angle_inverts_from_axis_angle() {
    check_axis_angle_roundtrip::<f32>();
    check_axis_angle_roundtrip::<f64>();
    check_axis_angle_roundtrip::<Q16_16>();
}

// ------------------------------------------------------------------ //
//  from_rotation_matrix (réciproque de to_rotation_matrix)             //
// ------------------------------------------------------------------ //

fn check_rotation_matrix_roundtrip<T: Scalar + core::ops::Div<Output = T>>() {
    // Axe fixe à composantes positives (évite toute ambiguïté de signe pour
    // les petits angles) ; plusieurs angles pour couvrir chaque branche de
    // Shepperd (trace > 0, puis chacune des trois branches diagonales).
    let axis_raw = [0.267f64, 0.535, 0.802];
    let n =
        (axis_raw[0] * axis_raw[0] + axis_raw[1] * axis_raw[1] + axis_raw[2] * axis_raw[2]).sqrt();
    let ax = [axis_raw[0] / n, axis_raw[1] / n, axis_raw[2] / n];

    for &angle in &[0.3f64, 1.5, 2.2, 2.8]
    {
        let q = Quaternion::<T>::from_axis_angle(
            [T::of(ax[0]), T::of(ax[1]), T::of(ax[2])],
            T::of(angle),
        );
        let m = q.to_rotation_matrix();
        let q2 = Quaternion::<T>::from_rotation_matrix(m);
        // R ne détermine q qu'au signe global près : aligne q2 sur q avant
        // de comparer.
        let dot = q.w.to_f64() * q2.w.to_f64()
            + q.x.to_f64() * q2.x.to_f64()
            + q.y.to_f64() * q2.y.to_f64()
            + q.z.to_f64() * q2.z.to_f64();
        let q2 = if dot < 0.0 { -q2 } else { q2 };
        approx_quat(
            q2,
            [q.w.to_f64(), q.x.to_f64(), q.y.to_f64(), q.z.to_f64()],
            &format!("from_rotation_matrix angle={angle}"),
        );
    }
}

#[test]
fn rotation_matrix_roundtrip_all_scalars() {
    check_rotation_matrix_roundtrip::<f32>();
    check_rotation_matrix_roundtrip::<f64>();
    check_rotation_matrix_roundtrip::<Q16_16>();
}

// ------------------------------------------------------------------ //
//  from_euler / to_euler (Tait-Bryan Z-Y-X)                            //
// ------------------------------------------------------------------ //

fn check_euler_roundtrip<T: Scalar + core::ops::Div<Output = T>>() {
    for &(roll, pitch, yaw) in &[
        (0.3f64, 0.2, 0.5),
        (-0.7, 0.4, 1.1),
        (0.0, 0.0, 0.0),
        (1.0, -0.3, -0.8),
    ]
    {
        let q = Quaternion::<T>::from_euler(T::of(roll), T::of(pitch), T::of(yaw));
        let (r2, p2, y2) = q.to_euler();
        assert!(
            (r2.to_f64() - roll).abs() <= T::TOL * 8.0,
            "roll {} vs {roll}",
            r2.to_f64()
        );
        assert!(
            (p2.to_f64() - pitch).abs() <= T::TOL * 8.0,
            "pitch {} vs {pitch}",
            p2.to_f64()
        );
        assert!(
            (y2.to_f64() - yaw).abs() <= T::TOL * 8.0,
            "yaw {} vs {yaw}",
            y2.to_f64()
        );
    }
}

#[test]
fn euler_roundtrip_all_scalars() {
    check_euler_roundtrip::<f32>();
    check_euler_roundtrip::<f64>();
    check_euler_roundtrip::<Q16_16>();
}

fn check_euler_gimbal_lock<T: Scalar + core::ops::Div<Output = T>>() {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let q = Quaternion::<T>::from_euler(T::of(0.4), T::of(half_pi), T::of(0.9));
    let (roll2, pitch2, yaw2) = q.to_euler();
    assert!(
        (pitch2.to_f64() - half_pi).abs() <= T::TOL * 8.0,
        "gimbal lock : tangage {} vs {half_pi}",
        pitch2.to_f64()
    );
    // Roulis et lacet individuels ne sont pas uniques au gimbal lock, mais la
    // rotation reconstruite doit rester la même.
    let q2 = Quaternion::<T>::from_euler(roll2, pitch2, yaw2);
    let v = [T::of(0.3), T::of(-0.6), T::of(0.9)];
    let want = q.rotate_vector(v);
    approx_vec(
        q2.rotate_vector(v),
        [want[0].to_f64(), want[1].to_f64(), want[2].to_f64()],
        "gimbal lock : rotation préservée",
    );
}

#[test]
fn euler_gimbal_lock_all_scalars() {
    check_euler_gimbal_lock::<f32>();
    check_euler_gimbal_lock::<f64>();
    check_euler_gimbal_lock::<Q16_16>();
}

// ------------------------------------------------------------------ //
//  Transform (SE(3)) : composition, inverse, matrice homogène          //
// ------------------------------------------------------------------ //

fn tof<T: Scalar>(axis: [f64; 3], angle: f64, translation: [f64; 3]) -> Transform<T> {
    let rotation = Quaternion::<T>::from_axis_angle(
        [T::of(axis[0]), T::of(axis[1]), T::of(axis[2])],
        T::of(angle),
    );
    Transform::new(
        rotation,
        [
            T::of(translation[0]),
            T::of(translation[1]),
            T::of(translation[2]),
        ],
    )
}

fn check_transform_identity<T: Scalar>() {
    let t = tof::<T>([0.267, 0.535, 0.802], 1.1, [0.3, -0.5, 0.9]);
    let id = Transform::<T>::identity();
    let p = [T::of(0.4), T::of(-0.2), T::of(0.6)];

    approx_vec(
        id.transform_point(p),
        [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()],
        "identité·p",
    );

    let want = t.transform_point(p);
    let want_f64 = [want[0].to_f64(), want[1].to_f64(), want[2].to_f64()];
    approx_vec(t.compose(&id).transform_point(p), want_f64, "t∘id");
    approx_vec(id.compose(&t).transform_point(p), want_f64, "id∘t");
}

#[test]
fn transform_identity_all_scalars() {
    check_transform_identity::<f32>();
    check_transform_identity::<f64>();
    check_transform_identity::<Q16_16>();
}

fn check_transform_compose_matches_sequential<T: Scalar>() {
    let a = tof::<T>([0.267, 0.535, 0.802], 0.9, [0.2, -0.4, 0.6]);
    let b = tof::<T>([0.408, 0.408, 0.816], 1.6, [-0.5, 0.3, 0.1]);
    let p = [T::of(0.35), T::of(-0.75), T::of(0.2)];

    let sequential = a.transform_point(b.transform_point(p));
    let sequential_f64 = [
        sequential[0].to_f64(),
        sequential[1].to_f64(),
        sequential[2].to_f64(),
    ];

    let composed = a.compose(&b).transform_point(p);
    approx_vec(
        composed,
        sequential_f64,
        "compose vs application séquentielle",
    );

    // Idem via l'opérateur Mul.
    let composed_op = (a * b).transform_point(p);
    approx_vec(
        composed_op,
        sequential_f64,
        "Mul vs application séquentielle",
    );
}

#[test]
fn transform_compose_matches_sequential_application_all_scalars() {
    check_transform_compose_matches_sequential::<f32>();
    check_transform_compose_matches_sequential::<f64>();
    check_transform_compose_matches_sequential::<Q16_16>();
}

fn check_transform_compose_associative<T: Scalar>() {
    let a = tof::<T>([0.267, 0.535, 0.802], 0.5, [0.1, 0.2, 0.3]);
    let b = tof::<T>([0.408, 0.408, 0.816], 1.1, [-0.2, 0.1, -0.3]);
    let c = tof::<T>([0.0, 0.6, 0.8], 2.0, [0.4, -0.1, 0.2]);
    let p = [T::of(0.2), T::of(0.5), T::of(-0.3)];

    let left = a.compose(&b).compose(&c).transform_point(p);
    let right = a.compose(&b.compose(&c)).transform_point(p);
    approx_vec(
        left,
        [right[0].to_f64(), right[1].to_f64(), right[2].to_f64()],
        "(a∘b)∘c vs a∘(b∘c)",
    );
}

#[test]
fn transform_compose_associative_all_scalars() {
    check_transform_compose_associative::<f32>();
    check_transform_compose_associative::<f64>();
    check_transform_compose_associative::<Q16_16>();
}

fn check_transform_inverse_two_sided<T: Scalar>() {
    let t = tof::<T>([0.267, 0.535, 0.802], 1.3, [0.4, -0.6, 0.2]);
    let inv = t.inverse();
    let p = [T::of(0.1), T::of(-0.4), T::of(0.7)];
    let want = [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()];

    approx_vec(t.compose(&inv).transform_point(p), want, "t∘t⁻¹ = id");
    approx_vec(inv.compose(&t).transform_point(p), want, "t⁻¹∘t = id");
}

#[test]
fn transform_inverse_is_two_sided_all_scalars() {
    check_transform_inverse_two_sided::<f32>();
    check_transform_inverse_two_sided::<f64>();
    check_transform_inverse_two_sided::<Q16_16>();
}

fn check_transform_matrix_roundtrip<T: Scalar + core::ops::Div<Output = T>>() {
    let t = tof::<T>([0.267, 0.535, 0.802], 1.7, [0.5, -0.3, 0.8]);
    let m = t.to_matrix();
    let t2 = Transform::<T>::from_matrix(m);

    // R ne détermine q qu'au signe global près : aligne avant de comparer
    // (même précaution que `check_rotation_matrix_roundtrip`).
    let dot = t.rotation.w.to_f64() * t2.rotation.w.to_f64()
        + t.rotation.x.to_f64() * t2.rotation.x.to_f64()
        + t.rotation.y.to_f64() * t2.rotation.y.to_f64()
        + t.rotation.z.to_f64() * t2.rotation.z.to_f64();
    let t2_rotation = if dot < 0.0 { -t2.rotation } else { t2.rotation };
    approx_quat(
        t2_rotation,
        [
            t.rotation.w.to_f64(),
            t.rotation.x.to_f64(),
            t.rotation.y.to_f64(),
            t.rotation.z.to_f64(),
        ],
        "from_matrix : rotation",
    );
    approx_vec(
        t2.translation,
        [
            t.translation[0].to_f64(),
            t.translation[1].to_f64(),
            t.translation[2].to_f64(),
        ],
        "from_matrix : translation",
    );
}

#[test]
fn transform_matrix_roundtrip_all_scalars() {
    check_transform_matrix_roundtrip::<f32>();
    check_transform_matrix_roundtrip::<f64>();
    check_transform_matrix_roundtrip::<Q16_16>();
}

fn check_transform_matrix_matches_transform_point<T: Scalar>() {
    let t = tof::<T>([0.267, 0.535, 0.802], 2.1, [0.6, -0.2, 0.4]);
    let p = [T::of(0.3), T::of(0.5), T::of(-0.7)];
    let m = t.to_matrix();

    // Multiplication homogène manuelle : m · [p; 1].
    let ph = [p[0], p[1], p[2], T::one()];
    let mut want = [T::zero(); 3];
    for (i, row) in m.iter().take(3).enumerate()
    {
        let mut acc = T::zero();
        for (j, &mij) in row.iter().enumerate()
        {
            acc = acc + mij * ph[j];
        }
        want[i] = acc;
    }

    approx_vec(
        t.transform_point(p),
        [want[0].to_f64(), want[1].to_f64(), want[2].to_f64()],
        "to_matrix vs transform_point",
    );
}

#[test]
fn transform_matrix_matches_transform_point_all_scalars() {
    check_transform_matrix_matches_transform_point::<f32>();
    check_transform_matrix_matches_transform_point::<f64>();
    check_transform_matrix_matches_transform_point::<Q16_16>();
}

fn check_transform_vector_ignores_translation<T: Scalar>() {
    let t = tof::<T>(
        [0.0, 0.0, 1.0],
        std::f64::consts::FRAC_PI_2,
        [5.0, -3.0, 2.0],
    );
    let v = [T::of(1.0), T::of(0.0), T::of(0.0)];
    // Rotation seule (90° autour de +z : x → y) ; la translation ne doit pas
    // intervenir, contrairement à `transform_point`.
    approx_vec(
        t.transform_vector(v),
        [0.0, 1.0, 0.0],
        "transform_vector ignore la translation",
    );
}

#[test]
fn transform_vector_ignores_translation_all_scalars() {
    check_transform_vector_ignores_translation::<f32>();
    check_transform_vector_ignores_translation::<f64>();
    check_transform_vector_ignores_translation::<Q16_16>();
}
