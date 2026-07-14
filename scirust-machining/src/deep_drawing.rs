//! Mise en forme — **emboutissage profond** d'un godet cylindrique : rapport
//! d'emboutissage limite, effort de poinçon et effort de serre-flan.
//!
//! ```text
//! rapport d'emboutissage   β = Db/Dp
//! effort de poinçon        F = π·Dp·t·UTS·(Db/Dp − c)
//! effort de serre-flan     Fh = Ah·ph
//! ```
//!
//! `Db` diamètre du flan (m), `Dp` diamètre du poinçon (m), `t` épaisseur de
//! tôle (m), `UTS` résistance à la traction (Pa), `c` facteur empirique de
//! correction du rendement d'emboutissage (sans dimension), `Ah` aire portante
//! du serre-flan (m²), `ph` pression de serre-flan (Pa). L'effort de poinçon
//! est proportionnel à la circonférence de la paroi `π·Dp·t` cisaillée en
//! traction.
//!
//! **Convention** : SI cohérent. **Limite honnête** : godet cylindrique à fond
//! plat, tôle isotrope, un seul passage. Le rapport d'emboutissage limite (LDR)
//! pratique vaut ~1,8–2,2 mais dépend du matériau, du rayon de matrice et de la
//! lubrification ; le facteur empirique `c`, la pression de serre-flan et le
//! LDR admissible sont **fournis par l'appelant** — aucune valeur « par défaut »
//! n'est inventée ici.

use core::f64::consts::PI;

/// Rapport d'emboutissage `β = Db/Dp` (sans dimension).
///
/// Le rapport limite (LDR) au-delà duquel la paroi se déchire est fourni par
/// l'appelant (~1,8–2,2 en pratique) ; cette fonction ne calcule que le ratio.
///
/// Panique si `blank_diameter <= 0` ou `punch_diameter <= 0`.
pub fn limiting_draw_ratio(blank_diameter: f64, punch_diameter: f64) -> f64 {
    assert!(
        blank_diameter > 0.0 && punch_diameter > 0.0,
        "les diamètres du flan et du poinçon doivent être strictement positifs"
    );
    blank_diameter / punch_diameter
}

/// Effort de poinçon `F = π·Dp·t·UTS·(Db/Dp − c)` (N).
///
/// `draw_ratio_factor` = `Db/Dp − c`, différence entre le rapport
/// d'emboutissage et le facteur empirique de correction `c`, fournie par
/// l'appelant.
///
/// Panique si `punch_diameter <= 0`, `sheet_thickness <= 0` ou
/// `tensile_strength < 0`.
pub fn drawing_force(
    punch_diameter: f64,
    sheet_thickness: f64,
    tensile_strength: f64,
    draw_ratio_factor: f64,
) -> f64 {
    assert!(
        punch_diameter > 0.0,
        "le diamètre du poinçon doit être strictement positif"
    );
    assert!(
        sheet_thickness > 0.0,
        "l'épaisseur de tôle doit être strictement positive"
    );
    assert!(
        tensile_strength >= 0.0,
        "la résistance à la traction ne peut pas être négative"
    );
    PI * punch_diameter * sheet_thickness * tensile_strength * draw_ratio_factor
}

/// Effort de serre-flan `Fh = Ah·ph` (N).
///
/// Panique si `blank_holder_area < 0` ou `holder_pressure < 0`.
pub fn blank_holder_force(blank_holder_area: f64, holder_pressure: f64) -> f64 {
    assert!(
        blank_holder_area >= 0.0,
        "l'aire du serre-flan ne peut pas être négative"
    );
    assert!(
        holder_pressure >= 0.0,
        "la pression de serre-flan ne peut pas être négative"
    );
    blank_holder_area * holder_pressure
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn draw_ratio_is_the_diameter_quotient() {
        // Flan Ø100 mm, poinçon Ø50 mm → β = 2,0 (limite haute typique).
        assert_relative_eq!(limiting_draw_ratio(0.100, 0.050), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn draw_ratio_reciprocity() {
        // Échanger flan et poinçon inverse le rapport : β(a,b)·β(b,a) = 1.
        let forward = limiting_draw_ratio(0.100, 0.050);
        let backward = limiting_draw_ratio(0.050, 0.100);
        assert_relative_eq!(forward * backward, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn drawing_force_scales_with_thickness() {
        // À géométrie fixée, doubler l'épaisseur double l'effort (linéarité).
        let thin = drawing_force(0.050, 0.001, 350e6, 0.9);
        let thick = drawing_force(0.050, 0.002, 350e6, 0.9);
        assert_relative_eq!(thick, 2.0 * thin, epsilon = 1e-3);
    }

    #[test]
    fn drawing_force_realistic_case() {
        // Dp=50 mm, t=1 mm, UTS=350 MPa, (β−c)=0,9 :
        // F = π·0,05·0,001·350e6·0,9 ≈ 49,5 kN.
        let f = drawing_force(0.050, 0.001, 350e6, 0.9);
        assert_relative_eq!(f, PI * 0.050 * 0.001 * 350e6 * 0.9, epsilon = 1e-6);
        assert!((49_000.0..50_000.0).contains(&f));
    }

    #[test]
    fn holder_force_is_pressure_times_area() {
        // Ah=8e-3 m², ph=2 MPa → Fh = 16 kN ; effort nul si pression nulle.
        assert_relative_eq!(blank_holder_force(8e-3, 2e6), 16_000.0, epsilon = 1e-6);
        assert_relative_eq!(blank_holder_force(8e-3, 0.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "strictement positifs")]
    fn zero_punch_diameter_ratio_panics() {
        limiting_draw_ratio(0.100, 0.0);
    }
}
