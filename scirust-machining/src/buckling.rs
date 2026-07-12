//! Flambage des poutres comprimées — charge critique d'**Euler**, longueur de
//! flambement selon les liaisons d'extrémité, élancement et domaine de validité.
//!
//! ```text
//! charge critique     Pcr = π²·E·I / Le²          Le = K·L
//! rayon de giration   r = √(I/A)
//! élancement          λ = Le/r
//! contrainte critique σcr = π²·E / λ²
//! élancement limite   λlim = π·√(E/σe)   (borne du domaine élastique d'Euler)
//! ```
//!
//! `E` module de Young (Pa), `I` moment quadratique minimal (m⁴), `A` aire de
//! section (m²), `L` longueur réelle, `Le` longueur de flambement, `σe` limite
//! élastique (Pa). Le facteur `K` traduit les liaisons d'extrémité (valeurs
//! théoriques d'Euler).
//!
//! **Convention** : SI cohérent. **Limite honnête** : flambement **élastique**
//! d'Euler (poutre droite, parfaitement centrée, matériau élastique linéaire).
//! Au-delà de `λlim` (colonnes élancées) le modèle s'applique ; en deçà (colonnes
//! trapues) la ruine est plastique et il faut une formule empirique
//! (Rankine, Johnson) — voir [`is_euler_valid`].

use core::f64::consts::PI;

/// Liaisons d'extrémité d'une colonne, avec le facteur de longueur de
/// flambement théorique `K` associé (`Le = K·L`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndCondition {
    /// Articulée–articulée (rotule aux deux bouts) : `K = 1`.
    PinnedPinned,
    /// Encastrée–libre (poteau en console) : `K = 2`.
    FixedFree,
    /// Encastrée–encastrée : `K = 0,5`.
    FixedFixed,
    /// Encastrée–articulée : `K = 0,7` (valeur théorique ≈ 0,699).
    FixedPinned,
}

impl EndCondition {
    /// Facteur de longueur de flambement théorique `K`.
    pub fn effective_length_factor(self) -> f64 {
        match self
        {
            EndCondition::PinnedPinned => 1.0,
            EndCondition::FixedFree => 2.0,
            EndCondition::FixedFixed => 0.5,
            EndCondition::FixedPinned => 0.7,
        }
    }
}

/// Longueur de flambement `Le = K·L`.
pub fn effective_length(end: EndCondition, length: f64) -> f64 {
    end.effective_length_factor() * length
}

/// Charge critique d'Euler `Pcr = π²·E·I / Le²` (N).
///
/// Panique si `effective_length <= 0`.
pub fn critical_load(e_pa: f64, i_m4: f64, effective_length_m: f64) -> f64 {
    assert!(
        effective_length_m > 0.0,
        "la longueur de flambement doit être strictement positive"
    );
    PI * PI * e_pa * i_m4 / (effective_length_m * effective_length_m)
}

/// Rayon de giration `r = √(I/A)` (m).
///
/// Panique si `area <= 0` ou `i < 0`.
pub fn radius_of_gyration(i_m4: f64, area_m2: f64) -> f64 {
    assert!(area_m2 > 0.0 && i_m4 >= 0.0, "A > 0 et I ≥ 0 requis");
    (i_m4 / area_m2).sqrt()
}

/// Élancement `λ = Le/r` (sans dimension).
///
/// Panique si `radius_of_gyration <= 0`.
pub fn slenderness_ratio(effective_length_m: f64, radius_of_gyration_m: f64) -> f64 {
    assert!(
        radius_of_gyration_m > 0.0,
        "le rayon de giration doit être strictement positif"
    );
    effective_length_m / radius_of_gyration_m
}

/// Contrainte critique d'Euler `σcr = π²·E / λ²` (Pa).
///
/// Panique si `slenderness <= 0`.
pub fn critical_stress(e_pa: f64, slenderness: f64) -> f64 {
    assert!(
        slenderness > 0.0,
        "l'élancement doit être strictement positif"
    );
    PI * PI * e_pa / (slenderness * slenderness)
}

/// Élancement limite `λlim = π·√(E/σe)` séparant le domaine d'Euler (au-dessus)
/// du domaine de ruine plastique (en dessous).
///
/// Panique si `yield_stress <= 0`.
pub fn limiting_slenderness(e_pa: f64, yield_stress_pa: f64) -> f64 {
    assert!(
        yield_stress_pa > 0.0,
        "la limite élastique doit être strictement positive"
    );
    PI * (e_pa / yield_stress_pa).sqrt()
}

/// Vrai si la colonne est assez élancée pour que le modèle d'Euler s'applique
/// (`λ ≥ λlim`).
pub fn is_euler_valid(slenderness: f64, limiting_slenderness: f64) -> bool {
    slenderness >= limiting_slenderness
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn effective_length_factors_follow_end_conditions() {
        assert_relative_eq!(
            effective_length(EndCondition::PinnedPinned, 3.0),
            3.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            effective_length(EndCondition::FixedFree, 3.0),
            6.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            effective_length(EndCondition::FixedFixed, 3.0),
            1.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn euler_critical_load_of_a_pinned_column() {
        // E=210 GPa, I=1e-8 m⁴ (r≈..), L=2 m articulée → Le=2 m.
        // Pcr = π²·210e9·1e-8/4 ≈ 5181,9 N.
        let le = effective_length(EndCondition::PinnedPinned, 2.0);
        let pcr = critical_load(210e9, 1e-8, le);
        assert_relative_eq!(pcr, PI * PI * 210e9 * 1e-8 / 4.0, epsilon = 1e-6);
    }

    #[test]
    fn critical_stress_matches_load_over_area() {
        // Pour une section donnée, σcr = Pcr/A doit égaler π²E/λ².
        let (e, i, a, le) = (210e9, 1e-8, 1e-3, 2.0);
        let r = radius_of_gyration(i, a);
        let lambda = slenderness_ratio(le, r);
        let sigma_from_load = critical_load(e, i, le) / a;
        assert_relative_eq!(critical_stress(e, lambda), sigma_from_load, epsilon = 1e-3);
    }

    #[test]
    fn slender_columns_are_in_euler_domain() {
        // Acier E=210 GPa, σe=235 MPa → λlim ≈ π√(210e9/235e6) ≈ 93,9.
        let lim = limiting_slenderness(210e9, 235e6);
        assert_relative_eq!(lim, PI * (210e9f64 / 235e6).sqrt(), epsilon = 1e-9);
        assert!(is_euler_valid(150.0, lim)); // colonne élancée
        assert!(!is_euler_valid(50.0, lim)); // colonne trapue → plastique
    }

    #[test]
    #[should_panic(expected = "longueur de flambement")]
    fn zero_length_panics() {
        critical_load(210e9, 1e-8, 0.0);
    }
}
