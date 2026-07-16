//! Adsorption à l'équilibre — isothermes de **Langmuir** et de **Freundlich**,
//! bilan de matière donnant la masse d'adsorbant requise et facteur de
//! séparation `RL` de Langmuir.
//!
//! ```text
//! Langmuir          q   = qm·K·C / (1 + K·C)                    [mol·kg⁻¹]
//! Freundlich        q   = Kf·C^(1/n)                            [mol·kg⁻¹]
//! bilan (1 étage)   m   = n_solute / q*                         [kg]
//! facteur RL        RL  = 1 / (1 + K·C0)                        [-]
//! ```
//!
//! `q` charge à l'équilibre (soluté adsorbé par masse d'adsorbant) [mol·kg⁻¹],
//! `qm` charge maximale de la monocouche [mol·kg⁻¹], `K` constante d'affinité de
//! **Langmuir** [m³·mol⁻¹], `C` concentration du soluté en phase fluide à
//! l'équilibre [mol·m⁻³], `Kf` constante de **Freundlich** [mol·kg⁻¹·(m³·mol⁻¹)^(1/n)],
//! `1/n` = `exponent` exposant d'hétérogénéité [sans dimension], `n_solute`
//! quantité de soluté à retirer [mol], `m` masse d'adsorbant [kg], `C0`
//! concentration d'alimentation [mol·m⁻³], `RL` facteur de séparation [sans
//! dimension, `0 < RL < 1` favorable].
//!
//! **Limite honnête** : ces relations décrivent l'**adsorption à l'équilibre**.
//! L'isotherme — paramètres de **Langmuir** (`qm`, `K`) ou de **Freundlich**
//! (`Kf`, `n`) — est **FOURNIE par des essais** (jamais inventée) ; **Langmuir**
//! suppose une **monocouche** sur une surface **homogène** à sites équivalents,
//! **Freundlich** une surface **hétérogène** (relation empirique). La masse
//! requise résulte d'un **bilan à une seule étape** supposant l'**équilibre
//! atteint** (`q*` = charge d'équilibre effectivement fournie). Ce module ne
//! modélise **ni la cinétique d'adsorption**, **ni la diffusion intraparticulaire**,
//! **ni la percée (courbe de rupture) d'une colonne à lit fixe** ; les unités
//! doivent être **cohérentes** entre `C`, `K`, `Kf` et `q`.

/// Isotherme de **Langmuir** `q = qm·K·C / (1 + K·C)` (mol·kg⁻¹).
///
/// `max_loading` (qm) charge maximale de la monocouche [mol·kg⁻¹] ;
/// `langmuir_constant` (K) constante d'affinité [m³·mol⁻¹] ;
/// `equilibrium_concentration` (C) concentration en phase fluide [mol·m⁻³].
///
/// Panique si `max_loading < 0`, `langmuir_constant < 0` ou
/// `equilibrium_concentration < 0`.
pub fn adsorb_langmuir(
    max_loading: f64,
    langmuir_constant: f64,
    equilibrium_concentration: f64,
) -> f64 {
    assert!(
        max_loading >= 0.0 && langmuir_constant >= 0.0 && equilibrium_concentration >= 0.0,
        "qm ≥ 0, K ≥ 0 et C ≥ 0 requis"
    );
    max_loading * langmuir_constant * equilibrium_concentration
        / (1.0 + langmuir_constant * equilibrium_concentration)
}

/// Isotherme de **Freundlich** `q = Kf·C^(1/n)` (mol·kg⁻¹), avec
/// `exponent = 1/n` l'exposant d'hétérogénéité.
///
/// `freundlich_constant` (Kf) constante de Freundlich [mol·kg⁻¹·(m³·mol⁻¹)^(1/n)] ;
/// `exponent` (1/n) exposant [sans dimension, généralement `0 < 1/n ≤ 1`] ;
/// `equilibrium_concentration` (C) concentration en phase fluide [mol·m⁻³].
///
/// Panique si `freundlich_constant < 0`, `exponent <= 0` ou
/// `equilibrium_concentration < 0`.
pub fn adsorb_freundlich(
    freundlich_constant: f64,
    exponent: f64,
    equilibrium_concentration: f64,
) -> f64 {
    assert!(
        freundlich_constant >= 0.0 && equilibrium_concentration >= 0.0,
        "Kf ≥ 0 et C ≥ 0 requis"
    );
    assert!(exponent > 0.0, "exposant 1/n > 0 requis");
    freundlich_constant * equilibrium_concentration.powf(exponent)
}

/// Charge d'équilibre de **Langmuir** `q = qm·K·C / (1 + K·C)` (mol·kg⁻¹),
/// forme dédiée au dimensionnement à partir des paramètres linéarisés `qm` et
/// `K` (identique à [`adsorb_langmuir`], exposée pour la lisibilité des calculs
/// de charge).
///
/// `max_loading` (qm) [mol·kg⁻¹] ; `langmuir_constant` (K) [m³·mol⁻¹] ;
/// `equilibrium_concentration` (C) [mol·m⁻³].
///
/// Panique si `max_loading < 0`, `langmuir_constant < 0` ou
/// `equilibrium_concentration < 0`.
pub fn adsorb_langmuir_linearized_loading(
    max_loading: f64,
    langmuir_constant: f64,
    equilibrium_concentration: f64,
) -> f64 {
    adsorb_langmuir(max_loading, langmuir_constant, equilibrium_concentration)
}

/// Masse d'adsorbant requise par **bilan à une étage** `m = n_solute / q*` (kg).
///
/// `solute_to_remove` (n_solute) quantité de soluté à retirer [mol] ;
/// `equilibrium_loading` (q*) charge d'équilibre effectivement atteinte
/// [mol·kg⁻¹].
///
/// Panique si `solute_to_remove < 0` ou `equilibrium_loading <= 0`.
pub fn adsorb_mass_of_adsorbent(solute_to_remove: f64, equilibrium_loading: f64) -> f64 {
    assert!(solute_to_remove >= 0.0, "n_solute ≥ 0 requis");
    assert!(
        equilibrium_loading > 0.0,
        "q* > 0 requis (charge d'équilibre strictement positive)"
    );
    solute_to_remove / equilibrium_loading
}

/// Facteur de séparation de **Langmuir** `RL = 1 / (1 + K·C0)` (sans dimension),
/// avec `0 < RL < 1` = adsorption favorable, `RL = 1` linéaire, `RL > 1`
/// défavorable.
///
/// `langmuir_constant` (K) constante d'affinité [m³·mol⁻¹] ;
/// `feed_concentration` (C0) concentration d'alimentation [mol·m⁻³].
///
/// Panique si `langmuir_constant < 0` ou `feed_concentration < 0`.
pub fn adsorb_separation_factor(langmuir_constant: f64, feed_concentration: f64) -> f64 {
    assert!(
        langmuir_constant >= 0.0 && feed_concentration >= 0.0,
        "K ≥ 0 et C0 ≥ 0 requis"
    );
    1.0 / (1.0 + langmuir_constant * feed_concentration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn langmuir_realistic_case() {
        // qm = 2, K = 0.5, C = 4 : q = 2·0.5·4/(1 + 0.5·4) = 4/(1 + 2) = 4/3.
        assert_relative_eq!(
            adsorb_langmuir(2.0_f64, 0.5_f64, 4.0_f64),
            4.0_f64 / 3.0_f64,
            max_relative = 1e-3
        );
    }

    #[test]
    fn langmuir_low_concentration_is_linear() {
        // Quand K·C ≪ 1, q ≈ qm·K·C (régime de Henry).
        let qm = 3.0_f64;
        let k = 0.4_f64;
        let c = 1.0e-4_f64;
        assert_relative_eq!(adsorb_langmuir(qm, k, c), qm * k * c, max_relative = 1e-3);
    }

    #[test]
    fn langmuir_saturates_to_max_loading() {
        // Quand K·C ≫ 1, q → qm (monocouche saturée).
        let qm = 5.0_f64;
        assert_relative_eq!(
            adsorb_langmuir(qm, 10.0_f64, 1.0e6_f64),
            qm,
            max_relative = 1e-3
        );
    }

    #[test]
    fn linearized_matches_langmuir() {
        // La forme linéarisée renvoie exactement la même charge.
        assert_relative_eq!(
            adsorb_langmuir_linearized_loading(2.0_f64, 0.5_f64, 4.0_f64),
            adsorb_langmuir(2.0_f64, 0.5_f64, 4.0_f64),
            max_relative = 1e-12
        );
    }

    #[test]
    fn freundlich_realistic_case() {
        // Kf = 2, 1/n = 0.5, C = 16 : q = 2·16^0.5 = 2·4 = 8.
        assert_relative_eq!(
            adsorb_freundlich(2.0_f64, 0.5_f64, 16.0_f64),
            8.0,
            max_relative = 1e-3
        );
    }

    #[test]
    fn mass_balance_and_separation_factor() {
        // Bilan : retirer 12 mol avec q* = 4/3 mol·kg⁻¹ ⇒ m = 12/(4/3) = 9 kg.
        assert_relative_eq!(
            adsorb_mass_of_adsorbent(12.0_f64, 4.0_f64 / 3.0_f64),
            9.0,
            max_relative = 1e-3
        );
        // RL : K = 0.5, C0 = 4 ⇒ RL = 1/(1 + 2) = 1/3 (favorable, 0 < RL < 1).
        let rl = adsorb_separation_factor(0.5_f64, 4.0_f64);
        assert_relative_eq!(rl, 1.0_f64 / 3.0_f64, max_relative = 1e-3);
        assert!(rl > 0.0 && rl < 1.0, "adsorption favorable attendue");
    }

    #[test]
    #[should_panic(expected = "q* > 0 requis")]
    fn mass_panics_on_zero_loading() {
        // Charge d'équilibre nulle ⇒ masse infinie ⇒ panique.
        let _ = adsorb_mass_of_adsorbent(10.0_f64, 0.0_f64);
    }
}
