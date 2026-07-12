//! Concentration de contrainte — facteur théorique `Kt`, contrainte de pointe
//! sur section nette, et passage à la fatigue via la sensibilité à l'entaille.
//!
//! ```text
//! contrainte de pointe   σmax = Kt·σnom
//! section nette (plat percé, traction)  σnom = F/((w − d)·t)
//! facteur de fatigue     Kf = 1 + q·(Kt − 1)      0 ≤ q ≤ 1
//! ```
//!
//! `Kt` facteur de concentration **théorique** (élastique) donné par l'appelant
//! d'après un abaque (Peterson) ou une formule ; `σnom` contrainte nominale sur
//! la section nette ; `q` sensibilité à l'entaille du matériau (`q → 0` insensible,
//! `q → 1` pleinement sensible) ; `Kf` facteur effectif en fatigue.
//!
//! **Convention** : unités cohérentes de l'appelant. **Limite honnête** : `Kt`
//! est **élastique** et n'est pas calculé ici (il dépend de la géométrie exacte
//! de l'entaille — abaques de Peterson/Roark) ; cette crate en tire les
//! conséquences. La sensibilité `q` (formules de Neuber/Peterson) est fournie par
//! l'appelant.

/// Contrainte de pointe `σmax = Kt·σnom`.
///
/// Panique si `kt < 1` (un facteur de concentration est toujours ≥ 1).
pub fn peak_stress(kt: f64, nominal_stress: f64) -> f64 {
    assert!(kt >= 1.0, "le facteur de concentration Kt doit être ≥ 1");
    kt * nominal_stress
}

/// Contrainte nominale de traction sur la **section nette** d'un plat percé
/// `σnom = F/((w − d)·t)`, largeur `w`, diamètre du trou `d`, épaisseur `t`.
///
/// Panique si la section nette `(w − d)·t <= 0`.
pub fn nominal_stress_plate_with_hole(
    force: f64,
    width: f64,
    hole_diameter: f64,
    thickness: f64,
) -> f64 {
    let net = (width - hole_diameter) * thickness;
    assert!(
        net > 0.0,
        "section nette nulle ou négative (trou plus large que le plat ?)"
    );
    force / net
}

/// Facteur de concentration **effectif en fatigue** `Kf = 1 + q·(Kt − 1)`.
///
/// Panique si `kt < 1` ou si `q` sort de `[0, 1]`.
pub fn fatigue_stress_concentration(kt: f64, notch_sensitivity: f64) -> f64 {
    assert!(kt >= 1.0, "le facteur de concentration Kt doit être ≥ 1");
    assert!(
        (0.0..=1.0).contains(&notch_sensitivity),
        "la sensibilité à l'entaille q doit être dans [0, 1]"
    );
    1.0 + notch_sensitivity * (kt - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn peak_stress_scales_nominal() {
        // Kt=3 (trou circulaire dans un plat large) → σmax = 3·σnom.
        assert_relative_eq!(peak_stress(3.0, 50.0), 150.0, epsilon = 1e-12);
    }

    #[test]
    fn net_section_of_a_plate_with_hole() {
        // F=10 kN, w=40 mm, d=10 mm, t=5 mm → net=(30·5)=150 mm² → σnom≈66,7 MPa.
        let s = nominal_stress_plate_with_hole(10_000.0, 0.040, 0.010, 0.005);
        assert_relative_eq!(s, 10_000.0 / ((0.040 - 0.010) * 0.005), epsilon = 1e-3);
    }

    #[test]
    fn fatigue_factor_between_one_and_kt() {
        // q=0 → Kf=1 (insensible) ; q=1 → Kf=Kt ; q=0,8, Kt=2,5 → Kf=2,2.
        assert_relative_eq!(fatigue_stress_concentration(2.5, 0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(fatigue_stress_concentration(2.5, 1.0), 2.5, epsilon = 1e-12);
        assert_relative_eq!(fatigue_stress_concentration(2.5, 0.8), 2.2, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "Kt doit être ≥ 1")]
    fn kt_below_one_panics() {
        peak_stress(0.5, 100.0);
    }

    #[test]
    #[should_panic(expected = "section nette")]
    fn hole_wider_than_plate_panics() {
        nominal_stress_plate_with_hole(10_000.0, 0.010, 0.020, 0.005);
    }
}
