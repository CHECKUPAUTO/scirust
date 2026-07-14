//! Découpage / poinçonnage de tôle — **effort de découpage** (blanking /
//! poinçonnage) : effort de cisaillement maximal, effort du dévêtisseur et
//! travail de découpage.
//!
//! ```text
//! effort de découpage   F = L·t·τ
//! effort dévêtisseur    Fs = k·F
//! travail de découpage  W = F·t·pf
//! ```
//!
//! `L` périmètre de coupe (m), `t` épaisseur de la tôle (m), `τ` résistance au
//! cisaillement du matériau (Pa), `k` facteur de dévêtissage (adimensionnel,
//! typiquement ≈ 0,02–0,20 selon le jeu et le collant), `pf` fraction de
//! pénétration du poinçon au moment de la rupture (adimensionnelle, ∈ ]0, 1]).
//! L'effort de découpage est l'aire cisaillée `L·t` multipliée par la résistance
//! au cisaillement ; l'effort du dévêtisseur retient la bande lors du retrait du
//! poinçon ; le travail est approché par le produit force × course pénétrée.
//!
//! **Convention** : SI cohérent (N, m, Pa, J). **Limite honnête** : cisaillement
//! pur sans jeu de coupe (poinçon à arêtes vives, sans affûtage biseauté qui
//! réduirait l'effort de crête), effort **maximal** (crête, pas moyen) ; `τ`, le
//! facteur de dévêtissage `k` et la fraction de pénétration `pf` sont **fournis
//! par l'appelant** — aucune valeur matériau n'est inventée ici. Pas de prise en
//! compte de l'écrouissage, de la vitesse de coupe ni de l'échauffement.

/// Effort de découpage maximal `F = L·t·τ` (N).
///
/// `L·t` est l'aire de la surface cisaillée, `τ` la résistance au cisaillement.
///
/// Panique si `cut_perimeter < 0`, `sheet_thickness < 0` ou `shear_strength < 0`.
pub fn blanking_force(cut_perimeter: f64, sheet_thickness: f64, shear_strength: f64) -> f64 {
    assert!(
        cut_perimeter >= 0.0,
        "le périmètre de coupe doit être positif"
    );
    assert!(
        sheet_thickness >= 0.0,
        "l'épaisseur de tôle doit être positive"
    );
    assert!(
        shear_strength >= 0.0,
        "la résistance au cisaillement doit être positive"
    );
    cut_perimeter * sheet_thickness * shear_strength
}

/// Effort du dévêtisseur `Fs = k·F` (N).
///
/// Fraction `k` de l'effort de découpage nécessaire pour retirer le poinçon de
/// la bande.
///
/// Panique si `blanking_force < 0` ou `stripping_factor < 0`.
pub fn stripping_force(blanking_force: f64, stripping_factor: f64) -> f64 {
    assert!(
        blanking_force >= 0.0,
        "l'effort de découpage doit être positif"
    );
    assert!(
        stripping_factor >= 0.0,
        "le facteur de dévêtissage doit être positif"
    );
    stripping_factor * blanking_force
}

/// Travail de découpage `W = F·t·pf` (J).
///
/// Approche l'aire sous la courbe effort-course par le produit de l'effort de
/// crête, de l'épaisseur et de la fraction de pénétration `pf ∈ ]0, 1]`.
///
/// Panique si `blanking_force < 0`, `sheet_thickness < 0` ou si
/// `penetration_fraction ∉ ]0, 1]`.
pub fn blanking_work(blanking_force: f64, sheet_thickness: f64, penetration_fraction: f64) -> f64 {
    assert!(
        blanking_force >= 0.0,
        "l'effort de découpage doit être positif"
    );
    assert!(
        sheet_thickness >= 0.0,
        "l'épaisseur de tôle doit être positive"
    );
    assert!(
        penetration_fraction > 0.0 && penetration_fraction <= 1.0,
        "la fraction de pénétration doit être dans ]0, 1]"
    );
    blanking_force * sheet_thickness * penetration_fraction
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn blanking_force_value_for_a_round_hole() {
        // Trou Ø20 mm dans une tôle de 2 mm, τ = 350 MPa.
        // L = π·d, F = π·0,02·0,002·350e6 ≈ 43,98 kN.
        use core::f64::consts::PI;
        let perimeter = PI * 0.020;
        let f = blanking_force(perimeter, 0.002, 350e6);
        assert_relative_eq!(f, perimeter * 0.002 * 350e6, epsilon = 1e-6);
        assert_relative_eq!(f, 43_982.297_150, epsilon = 1.0);
    }

    #[test]
    fn blanking_force_scales_linearly_with_perimeter() {
        // F ∝ L : doubler le périmètre double l'effort.
        let f1 = blanking_force(0.1, 0.002, 350e6);
        let f2 = blanking_force(0.2, 0.002, 350e6);
        assert_relative_eq!(f2 / f1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn stripping_force_is_a_fraction_of_blanking() {
        // Fs = k·F : avec k < 1, l'effort de dévêtissage reste inférieur à F.
        let f = blanking_force(0.3, 0.003, 300e6);
        let fs = stripping_force(f, 0.08);
        assert_relative_eq!(fs, 0.08 * f, epsilon = 1e-9);
        assert!(fs < f);
    }

    #[test]
    fn work_equals_force_times_penetrated_stroke() {
        // W = F·t·pf : pénétration complète (pf=1) → W = F·t.
        let f = blanking_force(0.25, 0.0025, 320e6);
        let full = blanking_work(f, 0.0025, 1.0);
        assert_relative_eq!(full, f * 0.0025, epsilon = 1e-9);
    }

    #[test]
    fn deeper_penetration_costs_more_work() {
        // W ∝ pf : une pénétration plus profonde consomme plus de travail.
        let f = blanking_force(0.25, 0.0025, 320e6);
        let shallow = blanking_work(f, 0.0025, 0.3);
        let deep = blanking_work(f, 0.0025, 0.6);
        assert_relative_eq!(deep / shallow, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "fraction de pénétration")]
    fn out_of_range_penetration_panics() {
        blanking_work(1000.0, 0.002, 1.5);
    }
}
