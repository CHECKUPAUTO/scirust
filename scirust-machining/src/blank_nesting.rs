//! Mise en bande et imbrication de flans en découpage — utilisation de la
//! matière d'une bande alimentée en presse, à partir de l'aire de la pièce, du
//! pas d'avance et de la largeur de bande.
//!
//! ```text
//! pitch = L + clearance                     (pas d'avance de la bande, mm)
//! U     = part_area / (pitch · width)       (taux d'utilisation matière, -)
//! n     = ⌊strip_length / pitch⌋            (nb de pièces par bande, -)
//! scrap = 1 − U                             (fraction de chute, -)
//! ```
//!
//! Légende des variables et unités :
//! - `part_area` : aire découpée d'une pièce, mm².
//! - `L` (`part_length`) : longueur de la pièce dans le sens d'avance, mm.
//! - `clearance` (`feed_clearance`) : pont/entretoise entre deux flans, mm.
//! - `pitch` (`strip_pitch`) : pas d'avance de la bande, mm.
//! - `width` (`strip_width`) : largeur de la bande, mm.
//! - `strip_length` : longueur utile de bande disponible, mm.
//! - `U`, `scrap` : nombres sans dimension dans `[0, 1]`.
//!
//! **Limite honnête** : ce module suppose une disposition **à une seule rangée**
//! (single-row) et prend l'aire de pièce, le pas d'avance, la largeur et les
//! ponts comme **fournis par l'appelant** (règles procédé, nuance, épaisseur,
//! outil). Il ne calcule aucun pont « par défaut », n'optimise pas l'orientation
//! ni l'imbrication **multi-rangées**, et ignore les chutes de rive et de bout
//! de bande. Les valeurs de chevauchement optimal relèvent d'un calcul dédié.

/// Pas d'avance de la bande `pitch = part_length + feed_clearance` (mm),
/// distance dont la bande avance entre deux coups de presse.
///
/// Panique si `part_length <= 0` ou `feed_clearance < 0`.
pub fn nesting_strip_pitch(part_length_mm: f64, feed_clearance_mm: f64) -> f64 {
    assert!(
        part_length_mm > 0.0,
        "la longueur de pièce doit être strictement positive"
    );
    assert!(
        feed_clearance_mm >= 0.0,
        "le pont d'avance doit être positif ou nul"
    );
    part_length_mm + feed_clearance_mm
}

/// Taux d'utilisation matière `U = part_area / (strip_pitch · strip_width)`
/// (sans dimension), rapport de l'aire découpée à l'aire de bande consommée par
/// pas, pour une disposition à une rangée.
///
/// Panique si `part_area < 0`, `strip_pitch <= 0` ou `strip_width <= 0`.
pub fn nesting_material_utilization(
    part_area_mm2: f64,
    strip_pitch_mm: f64,
    strip_width_mm: f64,
) -> f64 {
    assert!(
        part_area_mm2 >= 0.0,
        "l'aire de pièce doit être positive ou nulle"
    );
    assert!(
        strip_pitch_mm > 0.0,
        "le pas d'avance doit être strictement positif"
    );
    assert!(
        strip_width_mm > 0.0,
        "la largeur de bande doit être strictement positive"
    );
    part_area_mm2 / (strip_pitch_mm * strip_width_mm)
}

/// Nombre de pièces obtenues sur une bande : `n = ⌊strip_length / pitch⌋`.
/// Disposition à une rangée ; le reste de bande est perdu (chute de bout).
///
/// Panique si `strip_length < 0` ou `strip_pitch <= 0`.
pub fn nesting_parts_per_strip(strip_length_mm: f64, strip_pitch_mm: f64) -> u32 {
    assert!(
        strip_length_mm >= 0.0,
        "la longueur de bande doit être positive ou nulle"
    );
    assert!(
        strip_pitch_mm > 0.0,
        "le pas d'avance doit être strictement positif"
    );
    (strip_length_mm / strip_pitch_mm).floor() as u32
}

/// Fraction de chute `scrap = 1 − U` (sans dimension) complémentaire du taux
/// d'utilisation matière `utilization`.
///
/// Panique si `utilization` n'est pas dans `[0, 1]`.
pub fn nesting_scrap_fraction(utilization: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&utilization),
        "le taux d'utilisation doit être dans [0, 1]"
    );
    1.0 - utilization
}

/// Aire de pièce requise pour atteindre un taux d'utilisation `utilization`
/// donné, en inversant [`nesting_material_utilization`] :
/// `part_area = U · strip_pitch · strip_width` (mm²).
///
/// Panique si `utilization < 0`, `strip_pitch <= 0` ou `strip_width <= 0`.
pub fn nesting_part_area_for_utilization(
    utilization: f64,
    strip_pitch_mm: f64,
    strip_width_mm: f64,
) -> f64 {
    assert!(
        utilization >= 0.0,
        "le taux d'utilisation doit être positif ou nul"
    );
    assert!(
        strip_pitch_mm > 0.0,
        "le pas d'avance doit être strictement positif"
    );
    assert!(
        strip_width_mm > 0.0,
        "la largeur de bande doit être strictement positive"
    );
    utilization * strip_pitch_mm * strip_width_mm
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn strip_pitch_adds_clearance_to_length() {
        // L = 40 mm, pont = 2 mm → pas = 42 mm.
        assert_relative_eq!(nesting_strip_pitch(40.0, 2.0), 42.0, epsilon = 1e-12);
    }

    #[test]
    fn utilization_and_scrap_sum_to_one() {
        // Identité de complémentarité U + chute = 1.
        let u = nesting_material_utilization(1000.0, 42.0, 60.0);
        assert_relative_eq!(u + nesting_scrap_fraction(u), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn part_area_inverts_utilization() {
        // Aller-retour U → aire → U neutre (réciprocité).
        let pitch = nesting_strip_pitch(40.0, 2.0);
        let area = nesting_part_area_for_utilization(0.65, pitch, 60.0);
        assert_relative_eq!(
            nesting_material_utilization(area, pitch, 60.0),
            0.65,
            epsilon = 1e-12
        );
    }

    #[test]
    fn utilization_is_proportional_to_part_area() {
        // Doubler l'aire de pièce double le taux d'utilisation.
        let u1 = nesting_material_utilization(500.0, 42.0, 60.0);
        let u2 = nesting_material_utilization(1000.0, 42.0, 60.0);
        assert_relative_eq!(u2, 2.0 * u1, epsilon = 1e-12);
    }

    #[test]
    fn parts_per_strip_floors_the_remainder() {
        // Bande de 1000 mm, pas 42 mm → ⌊23,809…⌋ = 23 pièces.
        assert_eq!(nesting_parts_per_strip(1000.0, 42.0), 23);
    }

    #[test]
    fn realistic_case_full_chain() {
        // Flan L=40 mm, pont 2 mm, bande large 60 mm, aire pièce 1500 mm².
        let pitch = nesting_strip_pitch(40.0, 2.0); // 42 mm
        let u = nesting_material_utilization(1500.0, pitch, 60.0); // 1500/2520
        assert_relative_eq!(u, 1500.0 / 2520.0, epsilon = 1e-12);
        assert_relative_eq!(nesting_scrap_fraction(u), 1020.0 / 2520.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "pas d'avance")]
    fn zero_pitch_panics() {
        nesting_material_utilization(1000.0, 0.0, 60.0);
    }
}
