//! Mise en forme — **repassage** (ironing) : réduction contrôlée de l'épaisseur
//! de paroi d'un godet, déformation vraie associée et effort de repassage.
//!
//! ```text
//! réduction d'épaisseur r  = (t0 − t1)/t0
//! déformation vraie     ε  = ln(t0/t1) = −ln(1 − r)
//! effort de repassage   F  = μf·(π·d)·t1·Ȳ·ln(t0/t1)
//! ```
//!
//! `t0`/`t1` épaisseurs de paroi entrée/sortie (m), `d` diamètre moyen de paroi
//! (m), `π·d` périmètre de la paroi cisaillée (m), `Ȳ` contrainte d'écoulement
//! **moyenne** de la paroi (Pa), `μf` facteur de procédé/frottement (sans
//! dimension), `F` effort axial du poinçon (N). Le repassage n'agit que sur
//! l'épaisseur de paroi : le diamètre intérieur est fixé par le poinçon, la
//! paroi est laminée entre poinçon et matrice.
//!
//! **Convention** : SI cohérent, angles en radians le cas échéant. **Limite
//! honnête** : réduction d'épaisseur **pure** (le diamètre du godet ne change
//! pas), déformation homogène, paroi isotrope. La contrainte d'écoulement
//! moyenne `Ȳ` (courbe d'écrouissage) et le facteur de procédé/frottement `μf`
//! — qui englobe frottement de matrice, angle de filière et déformation
//! redondante — sont **fournis par l'appelant** ; aucune valeur « par défaut »
//! matériau/procédé n'est inventée ici.

use core::f64::consts::PI;

/// Périmètre de paroi `π·d` (m) à partir du diamètre moyen de paroi.
///
/// Utilitaire pour former l'argument `wall_perimeter` de [`ironing_force`].
///
/// Panique si `mean_wall_diameter <= 0`.
pub fn ironing_wall_perimeter(mean_wall_diameter: f64) -> f64 {
    assert!(
        mean_wall_diameter > 0.0,
        "le diamètre moyen de paroi doit être strictement positif"
    );
    PI * mean_wall_diameter
}

/// Réduction d'épaisseur `r = (t0 − t1)/t0` (sans dimension).
///
/// Panique si `initial_thickness <= 0` ou si `final_thickness` n'est pas dans
/// `]0, initial_thickness]` (le repassage ne peut qu'amincir la paroi).
pub fn ironing_reduction(initial_thickness: f64, final_thickness: f64) -> f64 {
    assert!(
        initial_thickness > 0.0,
        "l'épaisseur initiale doit être strictement positive"
    );
    assert!(
        final_thickness > 0.0 && final_thickness <= initial_thickness,
        "l'épaisseur finale doit vérifier 0 < t1 <= t0 (amincissement)"
    );
    (initial_thickness - final_thickness) / initial_thickness
}

/// Déformation vraie de repassage `ε = ln(t0/t1)` (sans dimension).
///
/// Panique si `initial_thickness <= 0` ou `final_thickness <= 0`.
pub fn ironing_true_strain(initial_thickness: f64, final_thickness: f64) -> f64 {
    assert!(
        initial_thickness > 0.0 && final_thickness > 0.0,
        "t0 > 0 et t1 > 0 requis"
    );
    (initial_thickness / final_thickness).ln()
}

/// Déformation vraie à partir de la réduction `ε = −ln(1 − r)` (sans dimension).
///
/// Cohérente avec [`ironing_true_strain`] via `t0/t1 = 1/(1 − r)`.
///
/// Panique si `reduction` n'est pas dans `[0, 1[`.
pub fn ironing_true_strain_from_reduction(reduction: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&reduction),
        "la réduction doit vérifier 0 <= r < 1"
    );
    -(1.0 - reduction).ln()
}

/// Effort de repassage `F = μf·(π·d)·t1·Ȳ·ln(t0/t1)` avec
/// `ln(t0/t1) = −ln(1 − r)` (N).
///
/// `wall_perimeter` = `π·d` (m, cf. [`ironing_wall_perimeter`]),
/// `final_thickness` = `t1` (m), `avg_flow_stress` = `Ȳ` (Pa),
/// `reduction` = `r` (sans dimension), `process_factor` = `μf` (sans dimension,
/// frottement + déformation redondante, fourni par l'appelant).
///
/// Panique si `wall_perimeter <= 0`, `final_thickness <= 0`,
/// `avg_flow_stress < 0`, `process_factor < 0` ou si `reduction ∉ [0, 1[`.
pub fn ironing_force(
    wall_perimeter: f64,
    final_thickness: f64,
    avg_flow_stress: f64,
    reduction: f64,
    process_factor: f64,
) -> f64 {
    assert!(
        wall_perimeter > 0.0,
        "le périmètre de paroi doit être strictement positif"
    );
    assert!(
        final_thickness > 0.0,
        "l'épaisseur finale doit être strictement positive"
    );
    assert!(
        avg_flow_stress >= 0.0,
        "la contrainte d'écoulement moyenne ne peut pas être négative"
    );
    assert!(
        process_factor >= 0.0,
        "le facteur de procédé ne peut pas être négatif"
    );
    let strain = ironing_true_strain_from_reduction(reduction);
    process_factor * wall_perimeter * final_thickness * avg_flow_stress * strain
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reduction_and_strain_are_consistent() {
        // t0=1,0 mm, t1=0,7 mm → r=0,3 ; ε = ln(1/0,7).
        let r = ironing_reduction(1.0e-3, 0.7e-3);
        assert_relative_eq!(r, 0.3, epsilon = 1e-12);
        assert_relative_eq!(
            ironing_true_strain(1.0e-3, 0.7e-3),
            (1.0_f64 / 0.7).ln(),
            epsilon = 1e-12
        );
        // Les deux voies vers ε coïncident : ln(t0/t1) = −ln(1 − r).
        assert_relative_eq!(
            ironing_true_strain(1.0e-3, 0.7e-3),
            ironing_true_strain_from_reduction(r),
            epsilon = 1e-12
        );
    }

    #[test]
    fn zero_reduction_gives_zero_strain_and_force() {
        // Cas limite t1 = t0 : aucune déformation, effort nul.
        assert_relative_eq!(ironing_reduction(1.0e-3, 1.0e-3), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            ironing_true_strain_from_reduction(0.0),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            ironing_force(0.1, 1.0e-3, 500e6, 0.0, 1.0),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn wall_perimeter_is_pi_d() {
        // d=40 mm → π·d ; effort nul si Ȳ nul (proportionnalité).
        assert_relative_eq!(ironing_wall_perimeter(0.040), PI * 0.040, epsilon = 1e-12);
    }

    #[test]
    fn force_scales_linearly_with_process_factor() {
        // Doubler μf double l'effort à géométrie et réduction fixées.
        let f1 = ironing_force(PI * 0.040, 0.7e-3, 500e6, 0.3, 1.0);
        let f2 = ironing_force(PI * 0.040, 0.7e-3, 500e6, 0.3, 2.0);
        assert_relative_eq!(f2, 2.0 * f1, epsilon = 1e-6);
    }

    #[test]
    fn ironing_force_realistic_case() {
        // d=40 mm, t1=0,7 mm, Ȳ=500 MPa, r=0,3, μf=1,0 :
        // F = π·0,04·0,0007·500e6·ln(1/0,7) ≈ 15,7 kN.
        let peri = ironing_wall_perimeter(0.040);
        let f = ironing_force(peri, 0.7e-3, 500e6, 0.3, 1.0);
        let expected = 1.0 * peri * 0.7e-3 * 500e6 * (1.0_f64 / 0.7).ln();
        assert_relative_eq!(f, expected, epsilon = 1e-6);
        assert!((15_000.0..16_500.0).contains(&f));
    }

    #[test]
    #[should_panic(expected = "amincissement")]
    fn thickening_reduction_panics() {
        // t1 > t0 est physiquement impossible en repassage.
        ironing_reduction(0.7e-3, 1.0e-3);
    }
}
