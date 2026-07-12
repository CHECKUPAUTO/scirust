//! Résistance des matériaux — flexion des poutres : moments d'inertie de
//! section, contrainte de flexion et flèches des cas de charge usuels.
//!
//! ```text
//! contrainte de flexion  σ = M·c/I = M/W
//! flèche (exemples)      voir fonctions dédiées
//! ```
//!
//! `M` moment fléchissant (N·m), `I` moment quadratique de section (m⁴ ou mm⁴
//! selon l'appelant), `c` distance à la fibre neutre, `W = I/c` module de
//! flexion, `E` module de Young, `L` portée.
//!
//! **Convention** : unités cohérentes de l'appelant (SI : N, m, Pa, ou N, mm,
//! MPa). **Limite honnête** : théorie d'Euler-Bernoulli élastique linéaire,
//! petites déformations, section constante, matériau homogène isotrope ; pas de
//! cisaillement transverse (Timoshenko), ni de grands déplacements.

use core::f64::consts::PI;

/// Moment quadratique d'une section **rectangulaire** `I = b·h³/12`
/// (axe neutre horizontal), largeur `b`, hauteur `h`.
pub fn second_moment_rectangle(b: f64, h: f64) -> f64 {
    b * h.powi(3) / 12.0
}

/// Moment quadratique d'une section **circulaire pleine** `I = π·d⁴/64`.
pub fn second_moment_circle(diameter: f64) -> f64 {
    PI * diameter.powi(4) / 64.0
}

/// Contrainte de flexion `σ = M·c/I`, moment `m`, distance fibre extrême `c`,
/// moment quadratique `i`.
///
/// Panique si `i <= 0`.
pub fn bending_stress(m: f64, c: f64, i: f64) -> f64 {
    assert!(
        i > 0.0,
        "le moment quadratique doit être strictement positif"
    );
    m * c / i
}

/// Flèche maximale d'une poutre **sur deux appuis, charge ponctuelle centrale**
/// `P` : `δ = P·L³/(48·E·I)` (à mi-portée).
///
/// Panique si `e*i <= 0`.
pub fn deflection_simply_supported_center_load(p: f64, l: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    p * l.powi(3) / (48.0 * e * i)
}

/// Moment fléchissant maximal — deux appuis, charge centrale `P` : `M = P·L/4`.
pub fn moment_simply_supported_center_load(p: f64, l: f64) -> f64 {
    p * l / 4.0
}

/// Flèche maximale d'une poutre **sur deux appuis, charge répartie** `w`
/// (par unité de longueur) : `δ = 5·w·L⁴/(384·E·I)`.
pub fn deflection_simply_supported_udl(w: f64, l: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    5.0 * w * l.powi(4) / (384.0 * e * i)
}

/// Moment fléchissant maximal — deux appuis, charge répartie `w` : `M = w·L²/8`.
pub fn moment_simply_supported_udl(w: f64, l: f64) -> f64 {
    w * l * l / 8.0
}

/// Flèche maximale d'une **poutre encastrée (console), charge en bout** `P` :
/// `δ = P·L³/(3·E·I)`.
pub fn deflection_cantilever_end_load(p: f64, l: f64, e: f64, i: f64) -> f64 {
    assert!(
        e * i > 0.0,
        "la rigidité E·I doit être strictement positive"
    );
    p * l.powi(3) / (3.0 * e * i)
}

/// Moment d'encastrement maximal — console, charge en bout `P` : `M = P·L`.
pub fn moment_cantilever_end_load(p: f64, l: f64) -> f64 {
    p * l
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn section_second_moments() {
        // rectangle 20×30 mm : I = 20·27000/12 = 45000 mm⁴.
        assert_relative_eq!(
            second_moment_rectangle(20.0, 30.0),
            45_000.0,
            epsilon = 1e-6
        );
        // cercle Ø20 : I = π·160000/64 = 2500π ≈ 7853,98 mm⁴.
        assert_relative_eq!(second_moment_circle(20.0), 2500.0 * PI, epsilon = 1e-6);
    }

    #[test]
    fn bending_stress_is_moment_over_modulus() {
        // M=1e6 N·mm, section rect 20×30 (I=45000, c=15) → σ = 1e6·15/45000 ≈ 333,3 MPa.
        let i = second_moment_rectangle(20.0, 30.0);
        assert_relative_eq!(
            bending_stress(1.0e6, 15.0, i),
            1.0e6 * 15.0 / i,
            epsilon = 1e-6
        );
    }

    #[test]
    fn simply_supported_center_load_cases() {
        // M = PL/4 ; δ = PL³/(48EI).
        assert_relative_eq!(
            moment_simply_supported_center_load(1000.0, 2.0),
            500.0,
            epsilon = 1e-9
        );
        let d = deflection_simply_supported_center_load(1000.0, 2.0, 2.0e11, 1.0e-6);
        assert_relative_eq!(d, 1000.0 * 8.0 / (48.0 * 2.0e11 * 1.0e-6), epsilon = 1e-15);
    }

    #[test]
    fn udl_and_cantilever_moments() {
        // deux appuis UDL : M = wL²/8 ; console : M = PL.
        assert_relative_eq!(
            moment_simply_supported_udl(500.0, 4.0),
            500.0 * 16.0 / 8.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            moment_cantilever_end_load(300.0, 1.5),
            450.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn cantilever_deflects_more_than_simply_supported_center() {
        // À charge/portée/EI égaux, la console fléchit bien plus (PL³/3 vs PL³/48).
        let (p, l, e, i) = (1000.0, 2.0, 2.0e11, 1.0e-6);
        let cant = deflection_cantilever_end_load(p, l, e, i);
        let ss = deflection_simply_supported_center_load(p, l, e, i);
        assert!(cant > ss);
        assert_relative_eq!(cant / ss, 16.0, epsilon = 1e-9); // (1/3)/(1/48) = 16
    }

    #[test]
    #[should_panic(expected = "moment quadratique")]
    fn zero_inertia_panics() {
        bending_stress(1000.0, 10.0, 0.0);
    }
}
