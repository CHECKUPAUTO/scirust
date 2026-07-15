//! Découpe laser — **bilan de puissance** : vitesse de coupe, puissance
//! requise et densité de puissance au point focal, à partir d'un bilan
//! énergétique idéalisé.
//!
//! ```text
//! vitesse de coupe    v = P / (t·w·e_s)
//! puissance requise   P = v·t·w·e_s
//! densité de puiss.   q = P / A
//! ```
//!
//! `P` puissance laser absorbée à la coupe (W), `t` épaisseur de la tôle (m),
//! `w` largeur de saignée / kerf (m), `e_s` énergie spécifique de coupe, soit
//! l'énergie nécessaire par unité de volume de matière évacuée (J/m³), `v`
//! vitesse d'avance de la coupe (m/s), `A` aire de la tache focale (m²), `q`
//! densité de puissance (W/m²). Le produit `t·w·v` est le débit volumique de
//! matière évacuée (m³/s) ; multiplié par `e_s`, il donne la puissance à
//! fournir. Les fonctions `laser_cutting_speed` et `laser_required_power` sont
//! réciproques.
//!
//! **Convention** : SI cohérent (W, m, J/m³, m/s, W/m²). **Limite honnête** :
//! bilan énergétique idéalisé — toute la puissance passée est supposée servir à
//! évacuer un volume `t·w·v` de matière (aucune perte par conduction, réflexion
//! ou plasma modélisée). L'énergie spécifique de coupe `e_s`, la largeur de
//! saignée `w` et la puissance absorbée sont **fournies par l'appelant** ;
//! aucune valeur matériau, procédé ou rendement optique n'est inventée ici.

/// Vitesse de coupe `v = P / (t·w·e_s)` (m/s).
///
/// Vitesse d'avance permise par le bilan énergétique : la puissance disponible
/// divisée par l'énergie par unité de longueur `t·w·e_s`.
///
/// Panique si `laser_power < 0`, `sheet_thickness <= 0`, `kerf_width <= 0` ou
/// `specific_energy <= 0`.
pub fn laser_cutting_speed(
    laser_power: f64,
    sheet_thickness: f64,
    kerf_width: f64,
    specific_energy: f64,
) -> f64 {
    assert!(laser_power >= 0.0, "la puissance laser doit être positive");
    assert!(
        sheet_thickness > 0.0,
        "l'épaisseur de tôle doit être strictement positive"
    );
    assert!(
        kerf_width > 0.0,
        "la largeur de saignée doit être strictement positive"
    );
    assert!(
        specific_energy > 0.0,
        "l'énergie spécifique de coupe doit être strictement positive"
    );
    laser_power / (sheet_thickness * kerf_width * specific_energy)
}

/// Puissance requise `P = v·t·w·e_s` (W).
///
/// Réciproque de [`laser_cutting_speed`] : puissance à fournir pour couper à la
/// vitesse `v` une saignée de largeur `w` dans une tôle d'épaisseur `t`, avec
/// une énergie spécifique `e_s`.
///
/// Panique si `cutting_speed < 0`, `sheet_thickness < 0`, `kerf_width < 0` ou
/// `specific_energy < 0`.
pub fn laser_required_power(
    cutting_speed: f64,
    sheet_thickness: f64,
    kerf_width: f64,
    specific_energy: f64,
) -> f64 {
    assert!(
        cutting_speed >= 0.0,
        "la vitesse de coupe doit être positive"
    );
    assert!(
        sheet_thickness >= 0.0,
        "l'épaisseur de tôle doit être positive"
    );
    assert!(
        kerf_width >= 0.0,
        "la largeur de saignée doit être positive"
    );
    assert!(
        specific_energy >= 0.0,
        "l'énergie spécifique de coupe doit être positive"
    );
    cutting_speed * sheet_thickness * kerf_width * specific_energy
}

/// Densité de puissance au point focal `q = P / A` (W/m²).
///
/// Puissance laser rapportée à l'aire de la tache focale ; caractérise
/// l'intensité disponible pour amorcer et entretenir la coupe.
///
/// Panique si `laser_power < 0` ou `spot_area <= 0`.
pub fn laser_power_density(laser_power: f64, spot_area: f64) -> f64 {
    assert!(laser_power >= 0.0, "la puissance laser doit être positive");
    assert!(
        spot_area > 0.0,
        "l'aire de la tache focale doit être strictement positive"
    );
    laser_power / spot_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn speed_and_power_are_reciprocal() {
        // v = P/(t·w·e_s) puis P = v·t·w·e_s doit redonner P.
        let (t, w, es) = (0.003, 3e-4, 5e9);
        let p = 2000.0;
        let v = laser_cutting_speed(p, t, w, es);
        let p_back = laser_required_power(v, t, w, es);
        assert_relative_eq!(p_back, p, epsilon = 1e-6);
    }

    #[test]
    fn cutting_speed_realistic_value() {
        // Tôle 2 mm, kerf 0,2 mm, e_s = 4·10⁹ J/m³, P = 1500 W.
        // v = 1500 / (0,002·2e-4·4e9) = 1500 / 1600 = 0,9375 m/s.
        let v = laser_cutting_speed(1500.0, 0.002, 2e-4, 4e9);
        assert_relative_eq!(v, 0.937_5, epsilon = 1e-9);
    }

    #[test]
    fn speed_scales_inversely_with_thickness() {
        // v ∝ 1/t : doubler l'épaisseur divise la vitesse par deux.
        let v1 = laser_cutting_speed(1500.0, 0.002, 2e-4, 4e9);
        let v2 = laser_cutting_speed(1500.0, 0.004, 2e-4, 4e9);
        assert_relative_eq!(v1 / v2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn required_power_scales_linearly_with_speed() {
        // P ∝ v : doubler la vitesse double la puissance nécessaire.
        let p1 = laser_required_power(0.5, 0.002, 2e-4, 4e9);
        let p2 = laser_required_power(1.0, 0.002, 2e-4, 4e9);
        assert_relative_eq!(p2 / p1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn power_density_realistic_value() {
        // P = 1000 W sur une tache de rayon 50 µm : A = π·r².
        // q = 1000 / (π·(5e-5)²) ≈ 1,273·10¹¹ W/m².
        use core::f64::consts::PI;
        let radius = 5e-5_f64;
        let area = PI * radius.powi(2);
        let q = laser_power_density(1000.0, area);
        assert_relative_eq!(q, 1000.0 / area, epsilon = 1e-3);
        assert_relative_eq!(q, 1.273_239_544_735e11, epsilon = 1e2);
    }

    #[test]
    fn power_density_scales_inversely_with_area() {
        // q ∝ 1/A : une tache deux fois plus grande halve la densité.
        let q1 = laser_power_density(1000.0, 1e-8);
        let q2 = laser_power_density(1000.0, 2e-8);
        assert_relative_eq!(q1 / q2, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "énergie spécifique de coupe")]
    fn zero_specific_energy_panics() {
        laser_cutting_speed(1500.0, 0.002, 2e-4, 0.0);
    }
}
