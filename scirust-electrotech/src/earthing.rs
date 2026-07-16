//! **Mise à la terre** — résistance des prises de terre (piquet vertical de
//! Dwight, plaque enterrée, groupement de piquets en parallèle) et élévation de
//! potentiel de terre en cas de défaut, à partir de la résistivité du sol et de
//! la géométrie d'électrode fournies.
//!
//! ```text
//! piquet vertical (Dwight)  R  = (ρ / (2·π·L)) · (ln(4·L / d) − 1)
//! plaque enterrée           R  = (ρ / 4) · √(π / A)
//! piquets en parallèle      R_eq = R_1 / (n · η)
//! élévation de potentiel    U  = R · I
//! ```
//!
//! `ρ` résistivité du sol (Ω·m), `L` longueur enfouie du piquet (m), `d`
//! diamètre du piquet (m), `A` surface d'une face de la plaque (m²), `R`
//! résistance d'une prise de terre (Ω), `R_1` résistance d'un piquet isolé (Ω),
//! `n` nombre de piquets (sans dimension, `≥ 1`), `η` coefficient de foisonnement
//! (sans dimension, `∈ ]0, 1]`), `R_eq` résistance équivalente du groupement (Ω),
//! `I` courant de défaut à la terre (A), `U` élévation de potentiel de terre (V).
//!
//! **Convention** : SI ; résistivités en Ω·m, longueurs et diamètres en m,
//! surfaces en m², résistances en Ω, courants en A, tensions en V. Les
//! logarithmes sont **népériens** (base e). **Limite honnête** : sol supposé
//! **homogène** et **isotrope**, de résistivité `ρ` **fournie par l'appelant**
//! (très variable selon la nature du terrain et l'humidité) ; la **géométrie**
//! d'électrode (`L`, `d`, `A`) est **fournie par l'appelant** ; le **coefficient
//! de foisonnement** `η` des piquets en parallèle (interaction des zones
//! d'influence, fonction de l'espacement et du nombre de piquets) est
//! **fourni par l'appelant** — aucune valeur « par défaut » n'est inventée. Les
//! formules sont les expressions **classiques** (piquet de Dwight, plaque
//! enterrée) valables pour une électrode enfouie loin de la surface.

/// Résistance d'un piquet vertical enfoui (formule de **Dwight**)
/// `R = (ρ / (2·π·L)) · (ln(4·L / d) − 1)` (Ω), à partir de la résistivité du
/// sol, de la longueur enfouie et du diamètre du piquet.
///
/// La formule suppose un piquet cylindrique long devant son diamètre
/// (`4·L / d` grand, de sorte que `ln(4·L / d) − 1 > 0`).
///
/// Panique si `soil_resistivity < 0`, si `rod_length <= 0` (division par zéro)
/// ou si `rod_diameter <= 0` (division par zéro / logarithme non défini).
pub fn earth_rod_resistance(soil_resistivity: f64, rod_length: f64, rod_diameter: f64) -> f64 {
    use core::f64::consts::PI;
    assert!(
        soil_resistivity >= 0.0,
        "la résistivité du sol ρ doit être ≥ 0"
    );
    assert!(
        rod_length > 0.0,
        "la longueur enfouie L doit être strictement positive"
    );
    assert!(
        rod_diameter > 0.0,
        "le diamètre du piquet d doit être strictement positif"
    );
    (soil_resistivity / (2.0 * PI * rod_length)) * ((4.0 * rod_length / rod_diameter).ln() - 1.0)
}

/// Résistance d'une plaque enterrée `R = (ρ / 4) · √(π / A)` (Ω), à partir de
/// la résistivité du sol et de la surface d'une face de la plaque.
///
/// Panique si `soil_resistivity < 0` ou si `plate_area <= 0` (division par zéro
/// sous la racine).
pub fn earth_plate_resistance(soil_resistivity: f64, plate_area: f64) -> f64 {
    use core::f64::consts::PI;
    assert!(
        soil_resistivity >= 0.0,
        "la résistivité du sol ρ doit être ≥ 0"
    );
    assert!(
        plate_area > 0.0,
        "la surface de la plaque A doit être strictement positive"
    );
    (soil_resistivity / 4.0) * (PI / plate_area).sqrt()
}

/// Résistance équivalente de `n` piquets identiques en **parallèle**
/// `R_eq = R_1 / (n · η)` (Ω), le coefficient de foisonnement `η` traduisant
/// l'interaction des zones d'influence.
///
/// Avec `η = 1` (piquets suffisamment éloignés, sans interaction) on retrouve la
/// mise en parallèle idéale `R_1 / n`.
///
/// Panique si `single_electrode_resistance < 0`, si `electrode_count < 1`
/// (au moins un piquet) ou si `layout_efficiency` n'est pas dans `]0, 1]`.
pub fn earth_parallel_electrodes(
    single_electrode_resistance: f64,
    electrode_count: f64,
    layout_efficiency: f64,
) -> f64 {
    assert!(
        single_electrode_resistance >= 0.0,
        "la résistance d'un piquet R_1 doit être ≥ 0"
    );
    assert!(
        electrode_count >= 1.0,
        "le nombre de piquets n doit être ≥ 1"
    );
    assert!(
        layout_efficiency > 0.0 && layout_efficiency <= 1.0,
        "le coefficient de foisonnement η doit être dans ]0, 1]"
    );
    single_electrode_resistance / (electrode_count * layout_efficiency)
}

/// Élévation de potentiel de terre `U = R · I` (V), produit de la résistance de
/// la prise de terre par le courant de défaut qui s'y écoule.
///
/// Panique si `earth_resistance < 0` ou si `fault_current < 0`.
pub fn earth_fault_voltage(earth_resistance: f64, fault_current: f64) -> f64 {
    assert!(
        earth_resistance >= 0.0,
        "la résistance de terre R doit être ≥ 0"
    );
    assert!(fault_current >= 0.0, "le courant de défaut I doit être ≥ 0");
    earth_resistance * fault_current
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rod_resistance_scales_with_resistivity() {
        // Proportionnalité : la résistance du piquet est linéaire en ρ, à
        // géométrie fixée ; doubler ρ double R.
        let r1 = earth_rod_resistance(100.0, 3.0, 0.016);
        let r2 = earth_rod_resistance(200.0, 3.0, 0.016);
        assert_relative_eq!(r2 / r1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn plate_resistance_scales_as_inverse_sqrt_area() {
        // Loi en 1/√A : quadrupler la surface divise la résistance par 2,
        // car R ∝ √(1 / A).
        let r_a = earth_plate_resistance(100.0, 1.0);
        let r_4a = earth_plate_resistance(100.0, 4.0);
        assert_relative_eq!(r_a / r_4a, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn parallel_reduces_to_ideal_when_efficiency_unity() {
        // Cas limite η = 1 : mise en parallèle idéale R_1 / n. Avec R_1 = 30 Ω
        // et n = 4 : R_eq = 30 / 4 = 7,5 Ω.
        let r_eq = earth_parallel_electrodes(30.0, 4.0, 1.0);
        assert_relative_eq!(r_eq, 7.5, epsilon = 1e-12);
    }

    #[test]
    fn fault_voltage_recovers_resistance() {
        // Réciprocité : U / I restitue la résistance de terre. R = 10 Ω,
        // I = 5000 A → U = 50 000 V, et U / I = 10 Ω.
        let r = 10.0_f64;
        let i = 5000.0_f64;
        let u = earth_fault_voltage(r, i);
        assert_relative_eq!(u, 50_000.0, epsilon = 1e-9);
        assert_relative_eq!(u / i, r, epsilon = 1e-12);
    }

    #[test]
    fn realistic_earthing_case() {
        // Cas chiffré réaliste.
        //   Piquet (Dwight), ρ = 100 Ω·m, L = 3 m, d = 0,016 m :
        //     4·L/d = 12 / 0,016 = 750
        //     ln(750) = ln2 + ln3 + 3·ln5
        //             = 0,6931472 + 1,0986123 + 4,8283137 = 6,6200732
        //     ln(750) − 1 = 5,6200732
        //     ρ/(2π·L) = 100 / (2π·3) = 100 / 18,8495559 = 5,3051648
        //     R = 5,3051648 · 5,6200732 = 29,81541 Ω
        let r_rod = earth_rod_resistance(100.0, 3.0, 0.016);
        assert_relative_eq!(r_rod, 29.81541, epsilon = 1e-3);

        //   Plaque enterrée, ρ = 100 Ω·m, A = 1 m² :
        //     R = (100/4)·√(π/1) = 25·√π = 25·1,7724539 = 44,311346 Ω
        let r_plate = earth_plate_resistance(100.0, 1.0);
        assert_relative_eq!(r_plate, 44.311346, epsilon = 1e-3);

        //   Groupement de n = 4 piquets, η = 0,75, R_1 = r_rod :
        //     R_eq = 29,81541 / (4·0,75) = 29,81541 / 3 = 9,93847 Ω
        let r_eq = earth_parallel_electrodes(r_rod, 4.0, 0.75);
        assert_relative_eq!(r_eq, r_rod / 3.0, epsilon = 1e-12);

        //   Élévation de potentiel sous I = 400 A : U = 9,93847 · 400 = 3975,4 V
        let u = earth_fault_voltage(r_eq, 400.0);
        assert_relative_eq!(u, r_eq * 400.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le coefficient de foisonnement η doit être dans ]0, 1]")]
    fn parallel_rejects_efficiency_above_one() {
        earth_parallel_electrodes(30.0, 4.0, 1.5);
    }
}
