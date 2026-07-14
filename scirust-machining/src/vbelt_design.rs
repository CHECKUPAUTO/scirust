//! Sélection de **courroies trapézoïdales** (V-belts) — puissance de
//! dimensionnement, puissance corrigée par courroie et nombre de brins requis.
//!
//! ```text
//! puissance de calcul     Pd = P · Ks                    (Ks = facteur de service)
//! puiss. corrigée/brin    Pc = P1 · Cθ · CL              (Cθ = f. d'angle, CL = f. de longueur)
//! nombre de courroies     N  = ⌈ Pd / Pc ⌉               (arrondi supérieur, N ≥ 1)
//! ```
//!
//! `P` puissance nominale du récepteur (W), `Ks` facteur de service (sans unité),
//! `Pd` puissance de calcul (W), `P1` puissance de base par courroie tirée du
//! catalogue (W), `Cθ` facteur correctif d'angle d'enroulement (sans unité),
//! `CL` facteur correctif de longueur de courroie (sans unité), `Pc` puissance
//! admissible corrigée par courroie (W), `N` nombre entier de courroies.
//!
//! **Convention** : SI cohérent (puissances en W) ; les puissances doivent
//! partager la même unité. Les facteurs sont adimensionnels.
//! **Limite honnête** : modèle de **catalogue** (type Rubber Manufacturers
//! Association / normes ISO 4184). Les facteurs de service `Ks`, correctifs
//! `Cθ`, `CL` et la puissance de base par courroie `P1` sont **tabulés et
//! fournis par l'appelant** d'après le catalogue du fabricant ; aucune valeur
//! « par défaut » n'est inventée ici. Le modèle ne couvre pas la vérification
//! de durée de vie, la vitesse linéaire admissible, ni l'échauffement.

/// Puissance de **dimensionnement** `Pd = P · Ks` (W).
///
/// Applique le facteur de service au récepteur pour majorer la puissance nominale.
///
/// Panique si `rated_power < 0` ou `service_factor <= 0`.
pub fn vbelt_design_power(rated_power: f64, service_factor: f64) -> f64 {
    assert!(
        rated_power >= 0.0,
        "la puissance nominale P ne peut être négative"
    );
    assert!(
        service_factor > 0.0,
        "le facteur de service Ks doit être strictement positif"
    );
    rated_power * service_factor
}

/// Puissance **corrigée par courroie** `Pc = P1 · Cθ · CL` (W).
///
/// Corrige la puissance de base catalogue par les facteurs d'angle d'enroulement
/// et de longueur de courroie.
///
/// Panique si `base_power_per_belt <= 0`, `wrap_angle_factor <= 0` ou
/// `belt_length_factor <= 0`.
pub fn corrected_power_per_belt(
    base_power_per_belt: f64,
    wrap_angle_factor: f64,
    belt_length_factor: f64,
) -> f64 {
    assert!(
        base_power_per_belt > 0.0,
        "la puissance de base par courroie P1 doit être strictement positive"
    );
    assert!(
        wrap_angle_factor > 0.0,
        "le facteur d'angle d'enroulement Cθ doit être strictement positif"
    );
    assert!(
        belt_length_factor > 0.0,
        "le facteur de longueur de courroie CL doit être strictement positif"
    );
    base_power_per_belt * wrap_angle_factor * belt_length_factor
}

/// **Nombre de courroies** requis `N = ⌈ Pd / Pc ⌉` (arrondi supérieur, `N ≥ 1`).
///
/// Divise la puissance de calcul par la puissance admissible corrigée d'une
/// courroie et arrondit à l'entier supérieur ; renvoie au moins 1.
///
/// Panique si `design_power < 0` ou `corrected_power_per_belt <= 0`.
pub fn number_of_belts(design_power: f64, corrected_power_per_belt: f64) -> u32 {
    assert!(
        design_power >= 0.0,
        "la puissance de calcul Pd ne peut être négative"
    );
    assert!(
        corrected_power_per_belt > 0.0,
        "la puissance corrigée par courroie Pc doit être strictement positive"
    );
    let n = (design_power / corrected_power_per_belt).ceil();
    (n.max(1.0)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn design_power_neutral_at_unit_service_factor() {
        // Ks = 1 : la puissance de calcul est exactement la puissance nominale.
        assert_relative_eq!(
            vbelt_design_power(7500.0, 1.0),
            7500.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn design_power_proportional_to_service_factor() {
        // Pd est linéaire en Ks à puissance fixée : doubler Ks double Pd.
        let p1 = vbelt_design_power(7500.0, 1.2);
        let p2 = vbelt_design_power(7500.0, 2.4);
        assert_relative_eq!(p2, 2.0 * p1, max_relative = 1e-12);
    }

    #[test]
    fn corrected_power_neutral_at_unit_factors() {
        // Cθ = CL = 1 : la puissance corrigée est la puissance de base catalogue.
        assert_relative_eq!(
            corrected_power_per_belt(3000.0, 1.0, 1.0),
            3000.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn corrected_power_is_product_of_factors() {
        // Identité : Pc = P1·Cθ·CL, ordre des facteurs indifférent.
        let a = corrected_power_per_belt(3000.0, 0.95, 1.05);
        let b = corrected_power_per_belt(3000.0, 1.05, 0.95);
        assert_relative_eq!(a, b, max_relative = 1e-12);
        assert_relative_eq!(a, 3000.0 * 0.95 * 1.05, max_relative = 1e-12);
    }

    #[test]
    fn belt_count_rounds_up() {
        // 3,1 courroies de capacité → il en faut 4 (arrondi supérieur).
        assert_eq!(number_of_belts(3100.0, 1000.0), 4);
        // Division entière exacte : pas d'arrondi parasite.
        assert_eq!(number_of_belts(4000.0, 1000.0), 4);
    }

    #[test]
    fn belt_count_realistic_case() {
        // Récepteur 11 kW, Ks = 1,4 → Pd = 15,4 kW ; base 4,2 kW, Cθ = 0,98,
        // CL = 1,02 → Pc ≈ 4,199 kW ; N = ⌈15400/4199⌉ = 4 courroies.
        let pd = vbelt_design_power(11000.0, 1.4);
        let pc = corrected_power_per_belt(4200.0, 0.98, 1.02);
        assert_relative_eq!(pd, 15400.0, max_relative = 1e-12);
        assert_eq!(number_of_belts(pd, pc), 4);
    }

    #[test]
    fn belt_count_at_least_one() {
        // Puissance nulle : on installe tout de même une courroie (N ≥ 1).
        assert_eq!(number_of_belts(0.0, 1000.0), 1);
    }

    #[test]
    #[should_panic(expected = "Pc doit être strictement positive")]
    fn zero_corrected_power_panics() {
        number_of_belts(5000.0, 0.0);
    }
}
