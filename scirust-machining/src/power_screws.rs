//! Vis de transmission (systèmes vis-écrou) — couple de manœuvre en montée et
//! en descente, rendement mécanique et condition d'irréversibilité, pour
//! filets carrés et trapézoïdaux (ACME / métrique Tr).
//!
//! Une vis de diamètre moyen `dm` (mm) et de pas d'hélice `L` (mm, = pas × nombre
//! de filets) présente un angle d'hélice `λ` tel que :
//!
//! ```text
//! tan λ = L / (π·dm)
//! ```
//!
//! Pour lever une charge axiale `F` (N), en tenant compte du demi-angle de
//! filet `α` (0° pour un filet carré, 15° pour un Tr métrique) via le facteur
//! `sec α = 1/cos α` :
//!
//! ```text
//! T_montée   = (F·dm/2) · (L + π·μ·dm·secα) / (π·dm − μ·L·secα)
//! T_descente = (F·dm/2) · (π·μ·dm·secα − L) / (π·dm + μ·L·secα)
//! rendement  η = F·L / (2π·T_montée)
//! ```
//!
//! La vis est **irréversible** (elle tient la charge sans se dévisser seule)
//! quand `μ·secα > tan λ`. `T_descente > 0` traduit alors un couple nécessaire
//! pour descendre ; négatif, la charge entraîne la vis.
//!
//! **Convention d'unités** : charge `F` en N, `dm` et `L` en mm, couples
//! retournés en **N·m**, angles en degrés.
//!
//! **Limite honnête** : ce module traite le couple au **filet** seul. Le couple
//! de frottement au collet/butée (`T_c = μ_c·F·d_c/2`) s'ajoute et dépend de la
//! liaison ; il est laissé à l'appelant, de même que `μ` et `μ_c`, données du
//! couple matériaux/lubrification.

use core::f64::consts::PI;

/// Angle d'hélice `λ` (degrés) d'une vis : `tan λ = L / (π·dm)`, pas d'hélice
/// `lead` (mm) et diamètre moyen `mean_diameter` (mm).
///
/// Panique si `mean_diameter <= 0`.
pub fn lead_angle_deg(lead_mm: f64, mean_diameter_mm: f64) -> f64 {
    assert!(
        mean_diameter_mm > 0.0,
        "le diamètre moyen doit être strictement positif"
    );
    (lead_mm / (PI * mean_diameter_mm)).atan().to_degrees()
}

fn sec(half_angle_deg: f64) -> f64 {
    assert!(
        (0.0..90.0).contains(&half_angle_deg),
        "le demi-angle de filet doit être dans [0°, 90°["
    );
    1.0 / half_angle_deg.to_radians().cos()
}

/// Couple au filet pour **lever** une charge `load` (N), en N·m :
/// `T = (F·dm/2)·(L + π·μ·dm·secα)/(π·dm − μ·L·secα)`.
///
/// `mean_diameter`, `lead` en mm ; `mu` frottement filet ; `thread_half_angle`
/// demi-angle de filet en degrés (0 = filet carré). Panique si le dénominateur
/// s'annule (géométrie/frottement dégénérés).
pub fn raising_torque_nm(
    load_n: f64,
    mean_diameter_mm: f64,
    lead_mm: f64,
    mu: f64,
    thread_half_angle_deg: f64,
) -> f64 {
    let s = sec(thread_half_angle_deg);
    let dm = mean_diameter_mm;
    let num = lead_mm + PI * mu * dm * s;
    let den = PI * dm - mu * lead_mm * s;
    assert!(den.abs() > 0.0, "dénominateur nul : paramètres dégénérés");
    load_n * dm / 2.0 * num / den / 1000.0
}

/// Couple au filet pour **descendre** une charge `load` (N), en N·m :
/// `T = (F·dm/2)·(π·μ·dm·secα − L)/(π·dm + μ·L·secα)`.
///
/// Positif si la vis est irréversible (couple requis pour descendre), négatif
/// si la charge entraîne la vis.
pub fn lowering_torque_nm(
    load_n: f64,
    mean_diameter_mm: f64,
    lead_mm: f64,
    mu: f64,
    thread_half_angle_deg: f64,
) -> f64 {
    let s = sec(thread_half_angle_deg);
    let dm = mean_diameter_mm;
    let num = PI * mu * dm * s - lead_mm;
    let den = PI * dm + mu * lead_mm * s;
    load_n * dm / 2.0 * num / den / 1000.0
}

/// Rendement de la vis en montée `η = F·L / (2π·T_montée)` (sans dimension),
/// charge `load` (N), pas d'hélice `lead` (mm) et couple de montée
/// `raising_torque` (N·m).
///
/// Panique si `raising_torque <= 0`.
pub fn efficiency(load_n: f64, lead_mm: f64, raising_torque_nm: f64) -> f64 {
    assert!(
        raising_torque_nm > 0.0,
        "le couple de montée doit être strictement positif"
    );
    // T en N·m → N·mm pour rester cohérent avec load (N) et lead (mm).
    load_n * lead_mm / (2.0 * PI * raising_torque_nm * 1000.0)
}

/// Irréversibilité : `true` si la vis tient la charge sans se dévisser seule,
/// c'est-à-dire `μ·secα > tan λ`.
pub fn is_self_locking(
    mu: f64,
    lead_mm: f64,
    mean_diameter_mm: f64,
    thread_half_angle_deg: f64,
) -> bool {
    let tan_lambda = lead_mm / (PI * mean_diameter_mm);
    mu * sec(thread_half_angle_deg) > tan_lambda
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Vis à filet carré : dm=25 mm, L=5 mm, μ=0,15, F=6400 N (exemple Shigley).
    #[test]
    fn lead_angle_of_a_square_thread() {
        // tan λ = 5/(π·25)=0,06366 → λ ≈ 3,643°.
        assert_relative_eq!(lead_angle_deg(5.0, 25.0), 3.643, epsilon = 1e-2);
    }

    #[test]
    fn raising_torque_matches_the_formula() {
        // ≈ 17,26 N·m.
        assert_relative_eq!(
            raising_torque_nm(6400.0, 25.0, 5.0, 0.15, 0.0),
            17.258,
            epsilon = 1e-2
        );
    }

    #[test]
    fn lowering_torque_is_positive_when_self_locking() {
        // ≈ 6,84 N·m (>0 : il faut un couple pour descendre).
        let t = lowering_torque_nm(6400.0, 25.0, 5.0, 0.15, 0.0);
        assert!(t > 0.0);
        assert_relative_eq!(t, 6.842, epsilon = 1e-2);
    }

    #[test]
    fn efficiency_is_between_zero_and_one() {
        let tr = raising_torque_nm(6400.0, 25.0, 5.0, 0.15, 0.0);
        let eta = efficiency(6400.0, 5.0, tr);
        assert!(eta > 0.0 && eta < 1.0);
        assert_relative_eq!(eta, 0.2951, epsilon = 1e-3);
    }

    #[test]
    fn self_locking_when_friction_exceeds_lead_angle_tangent() {
        // μ=0,15 > tan λ=0,0637 → irréversible.
        assert!(is_self_locking(0.15, 5.0, 25.0, 0.0));
        // Un pas d'hélice très grand (filet raide) redevient réversible.
        assert!(!is_self_locking(0.15, 20.0, 25.0, 0.0));
    }

    #[test]
    fn trapezoidal_thread_needs_more_torque_than_square() {
        // Le demi-angle (secα>1) augmente le frottement effectif, donc le couple.
        let square = raising_torque_nm(6400.0, 25.0, 5.0, 0.15, 0.0);
        let trap = raising_torque_nm(6400.0, 25.0, 5.0, 0.15, 15.0);
        assert!(trap > square);
    }

    #[test]
    #[should_panic(expected = "diamètre moyen")]
    fn zero_mean_diameter_panics() {
        lead_angle_deg(5.0, 0.0);
    }
}
