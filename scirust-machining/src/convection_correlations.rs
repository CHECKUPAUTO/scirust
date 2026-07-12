//! Convection — nombres sans dimension et corrélations usuelles : **Prandtl**,
//! **Nusselt** → coefficient d'échange, **Dittus-Boelter** (interne turbulent) et
//! **Rayleigh** (convection naturelle).
//!
//! ```text
//! Prandtl         Pr = µ·cp/k = ν/α
//! h depuis Nusselt  h = Nu·k/Lc
//! Dittus-Boelter  Nu = 0,023·Re^0,8·Pr^n   (n = 0,4 chauffage, 0,3 refroidissement)
//! Rayleigh        Ra = g·β·ΔT·L³/(ν·α) = Gr·Pr
//! ```
//!
//! `µ` viscosité dynamique (Pa·s), `cp` chaleur massique (J/(kg·K)), `k`
//! conductivité (W/(m·K)), `Lc` longueur caractéristique (m), `Nu`/`Re`/`Pr`/`Ra`
//! nombres sans dimension, `β` dilatation du fluide (1/K), `ν` viscosité
//! cinématique (m²/s), `α` diffusivité thermique (m²/s).
//!
//! **Convention** : SI cohérent. **Limite honnête** : Dittus-Boelter vaut pour un
//! écoulement **interne pleinement turbulent** (`Re ≳ 10⁴`, `0,7 ≤ Pr ≤ 160`,
//! tube long) ; hors domaine, d'autres corrélations s'imposent. Les propriétés du
//! fluide (`µ`, `cp`, `k`, `β`, `ν`, `α`) sont fournies par l'appelant. Voir
//! [`crate::bernoulli::reynolds_number`] pour le Reynolds.

/// Nombre de Prandtl `Pr = µ·cp/k`.
///
/// Panique si `k <= 0`.
pub fn prandtl_number(mu: f64, cp: f64, k: f64) -> f64 {
    assert!(k > 0.0, "la conductivité doit être strictement positive");
    mu * cp / k
}

/// Coefficient de convection depuis le Nusselt `h = Nu·k/Lc` (W/(m²·K)).
///
/// Panique si `characteristic_length <= 0`.
pub fn convection_coefficient(nusselt: f64, k: f64, characteristic_length: f64) -> f64 {
    assert!(
        characteristic_length > 0.0,
        "la longueur caractéristique doit être strictement positive"
    );
    nusselt * k / characteristic_length
}

/// Nusselt par la corrélation de **Dittus-Boelter** `Nu = 0,023·Re^0,8·Pr^n`,
/// `n = 0,4` si le fluide est **chauffé**, `0,3` s'il est **refroidi**.
///
/// Panique si `re < 0` ou `pr < 0`.
pub fn dittus_boelter(re: f64, pr: f64, heating: bool) -> f64 {
    assert!(re >= 0.0 && pr >= 0.0, "Re ≥ 0 et Pr ≥ 0 requis");
    let n = if heating { 0.4 } else { 0.3 };
    0.023 * re.powf(0.8) * pr.powf(n)
}

/// Nombre de Rayleigh `Ra = g·β·ΔT·L³/(ν·α)`.
///
/// Panique si `ν·α <= 0`.
pub fn rayleigh_number(g: f64, beta: f64, delta_t: f64, length: f64, nu: f64, alpha: f64) -> f64 {
    assert!(nu * alpha > 0.0, "ν·α doit être strictement positif");
    g * beta * delta_t * length.powi(3) / (nu * alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn prandtl_of_water_like_fluid() {
        // µ=1e-3, cp=4180, k=0,6 → Pr ≈ 6,97.
        assert_relative_eq!(
            prandtl_number(1e-3, 4180.0, 0.6),
            1e-3 * 4180.0 / 0.6,
            epsilon = 1e-9
        );
    }

    #[test]
    fn dittus_boelter_heating_above_cooling() {
        // À Re, Pr donnés, l'exposant 0,4 (chauffage) donne un Nu ≥ 0,3 (refroidissement)
        // dès que Pr > 1.
        let re = 1e4;
        let pr = 7.0;
        let nu_h = dittus_boelter(re, pr, true);
        let nu_c = dittus_boelter(re, pr, false);
        assert!(nu_h > nu_c);
        // valeur : 0,023·10000^0,8·7^0,4.
        assert_relative_eq!(
            nu_h,
            0.023 * re.powf(0.8) * 7.0f64.powf(0.4),
            epsilon = 1e-9
        );
    }

    #[test]
    fn nusselt_to_h_conversion() {
        // Nu=100, k=0,6, Lc=0,02 → h = 3000 W/(m²·K).
        assert_relative_eq!(
            convection_coefficient(100.0, 0.6, 0.02),
            3000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn rayleigh_equals_grashof_times_prandtl() {
        // Ra = Gr·Pr ; on vérifie la forme directe.
        let ra = rayleigh_number(9.81, 3.4e-3, 20.0, 0.1, 1.5e-5, 2.1e-5);
        assert_relative_eq!(
            ra,
            9.81 * 3.4e-3 * 20.0 * 0.1f64.powi(3) / (1.5e-5 * 2.1e-5),
            epsilon = 1e-3
        );
        assert!(ra > 0.0);
    }

    #[test]
    #[should_panic(expected = "conductivité")]
    fn zero_conductivity_prandtl_panics() {
        prandtl_number(1e-3, 4180.0, 0.0);
    }
}
