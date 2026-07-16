//! Taux de reflux d'une **colonne à distiller** binaire — reflux minimal
//! (Underwood simplifié à alimentation liquide bouillante), reflux opératoire,
//! pente de la droite de rectification et charges thermiques
//! condenseur/rebouilleur par bilan enthalpique global.
//!
//! ```text
//! reflux minimal (q=1)  Rₘ = 1/(α−1)·(x_D/x_F − α·(1−x_D)/(1−x_F))       [-]
//! reflux opératoire      R  = f·Rₘ           (f ≈ 1,2–1,5)               [-]
//! pente rectification    s  = R / (R + 1)                                [-]
//! charge condenseur      Q_c = (R + 1)·D·λ    (condenseur total)         [W]
//! charge rebouilleur     Q_b = Q_c + ΔH_F     (bilan enthalpique global) [W]
//! ```
//!
//! `α` volatilité relative du constituant léger sur le lourd [sans dimension],
//! `x_D` fraction molaire du léger au distillat [sans dimension, 0 < x_D < 1],
//! `x_F` fraction molaire du léger à l'alimentation [sans dimension, 0 < x_F < 1],
//! `q` fraction liquide de l'alimentation (rapport de la variation d'enthalpie de
//! vaporisation) [sans dimension ; `q = 1` pour un liquide à sa température de
//! bulle], `Rₘ` taux de reflux minimal `L/D` [sans dimension], `R` taux de reflux
//! opératoire `L/D` [sans dimension], `f` facteur d'excès sur le reflux minimal
//! [sans dimension, > 1], `s` pente de la droite opératoire de rectification
//! [sans dimension], `Q_c` charge thermique du condenseur [W], `D` débit molaire
//! de distillat [mol·s⁻¹], `λ` chaleur latente de vaporisation du distillat
//! [J·mol⁻¹], `Q_b` charge thermique du rebouilleur [W], `ΔH_F` déficit
//! enthalpique global (chaleur nette apportée hors condenseur, alimentation et
//! soutirages) refermant le bilan `Q_b + Q_c,retiré = ΔH_flux` [W].
//!
//! **Limite honnête** : ces relations valent pour une **colonne binaire** à
//! **volatilité relative `α` constante FOURNIE par l'appelant** (jamais inventée ;
//! issue de tables, de corrélations ou du rapport des coefficients de partage) et
//! le long de la colonne. Le **reflux minimal** ci-dessous est la **forme
//! d'Underwood simplifiée** valable pour une **alimentation liquide à sa
//! température de bulle** (`q = 1`, droite d'alimentation verticale) ; les cas
//! `q ≠ 1` (vapeur, mélange diphasique) exigent la résolution des équations
//! d'Underwood complètes et ne sont **pas** couverts ici. Les **compositions**
//! (`x_D`, `x_F`), la **chaleur latente** `λ` et le **terme enthalpique** `ΔH_F`
//! du bilan global sont **fournis par l'appelant** d'après un modèle
//! thermodynamique ou des essais ; ce module ne calcule ni équilibres, ni
//! enthalpies, ni profils de température. Il **complète**
//! [`crate::distillation_mccabe`] et [`crate::distillation_efficiency`] sans les
//! dupliquer (pas d'étages, pas de rendement d'étage ici).

/// Taux de reflux **minimal** `Rₘ` d'un binaire à alimentation liquide bouillante
/// (`q = 1`), forme d'Underwood simplifiée :
/// `Rₘ = 1/(α − 1)·(x_D/x_F − α·(1 − x_D)/(1 − x_F))` [sans dimension].
///
/// `relative_volatility` `α` volatilité relative du léger sur le lourd
/// [sans dimension, > 1], `distillate_composition` `x_D` fraction molaire du léger
/// au distillat [sans dimension, 0 < x_D < 1], `feed_composition` `x_F` fraction
/// molaire du léger à l'alimentation [sans dimension, 0 < x_F < 1],
/// `liquid_fraction_feed` `q` fraction liquide de l'alimentation [sans dimension ;
/// cette forme n'est valable que pour `q = 1`, liquide à sa température de bulle].
///
/// Panique si `relative_volatility` n'est pas fini ou n'est pas strictement
/// supérieur à 1, si `distillate_composition` ou `feed_composition` n'est pas dans
/// l'intervalle ouvert `]0, 1[`, ou si `liquid_fraction_feed` n'est pas égal à 1
/// (hypothèse `q = 1` de la forme simplifiée).
pub fn reflux_minimum_underwood_binary(
    relative_volatility: f64,
    distillate_composition: f64,
    feed_composition: f64,
    liquid_fraction_feed: f64,
) -> f64 {
    assert!(
        relative_volatility.is_finite() && relative_volatility > 1.0,
        "la volatilité relative doit être finie et strictement supérieure à 1"
    );
    assert!(
        distillate_composition.is_finite()
            && distillate_composition > 0.0
            && distillate_composition < 1.0,
        "la composition du distillat doit être dans l'intervalle ouvert ]0, 1["
    );
    assert!(
        feed_composition.is_finite() && feed_composition > 0.0 && feed_composition < 1.0,
        "la composition de l'alimentation doit être dans l'intervalle ouvert ]0, 1["
    );
    assert!(
        liquid_fraction_feed.is_finite() && (liquid_fraction_feed - 1.0).abs() < 1.0e-9,
        "la forme simplifiée exige une alimentation liquide bouillante (q = 1)"
    );
    (distillate_composition / feed_composition
        - relative_volatility * (1.0 - distillate_composition) / (1.0 - feed_composition))
        / (relative_volatility - 1.0)
}

/// Taux de reflux **opératoire** `R = f·Rₘ` obtenu en majorant le reflux minimal
/// par un facteur d'excès [sans dimension].
///
/// `minimum_reflux` `Rₘ` taux de reflux minimal [sans dimension, ≥ 0],
/// `factor` `f` facteur d'excès sur le reflux minimal [sans dimension, > 1 ;
/// usuellement 1,2 à 1,5].
///
/// Panique si `minimum_reflux` n'est pas fini ou est négatif, ou si `factor`
/// n'est pas fini ou n'est pas strictement supérieur à 1.
pub fn reflux_operating_from_ratio(minimum_reflux: f64, factor: f64) -> f64 {
    assert!(
        minimum_reflux.is_finite() && minimum_reflux >= 0.0,
        "le taux de reflux minimal doit être fini et positif ou nul"
    );
    assert!(
        factor.is_finite() && factor > 1.0,
        "le facteur d'excès doit être fini et strictement supérieur à 1"
    );
    factor * minimum_reflux
}

/// Pente `s = R / (R + 1)` de la **droite opératoire de rectification**
/// [sans dimension].
///
/// `reflux_ratio` `R` taux de reflux `L/D` [sans dimension, ≥ 0].
///
/// Panique si `reflux_ratio` n'est pas fini ou est négatif.
pub fn reflux_rectifying_slope(reflux_ratio: f64) -> f64 {
    assert!(
        reflux_ratio.is_finite() && reflux_ratio >= 0.0,
        "le taux de reflux doit être fini et positif ou nul"
    );
    reflux_ratio / (reflux_ratio + 1.0)
}

/// Charge thermique du **condenseur total** `Q_c = (R + 1)·D·λ` [W] : toute la
/// vapeur de tête `V = (R + 1)·D` est condensée.
///
/// `reflux_ratio` `R` taux de reflux `L/D` [sans dimension, ≥ 0],
/// `distillate_flow` `D` débit molaire de distillat [mol·s⁻¹, ≥ 0],
/// `latent_heat` `λ` chaleur latente de vaporisation du distillat [J·mol⁻¹, ≥ 0].
///
/// Panique si l'un des arguments n'est pas fini ou est négatif.
pub fn reflux_condenser_duty(reflux_ratio: f64, distillate_flow: f64, latent_heat: f64) -> f64 {
    assert!(
        reflux_ratio.is_finite() && reflux_ratio >= 0.0,
        "le taux de reflux doit être fini et positif ou nul"
    );
    assert!(
        distillate_flow.is_finite() && distillate_flow >= 0.0,
        "le débit de distillat doit être fini et positif ou nul"
    );
    assert!(
        latent_heat.is_finite() && latent_heat >= 0.0,
        "la chaleur latente doit être finie et positive ou nulle"
    );
    (reflux_ratio + 1.0) * distillate_flow * latent_heat
}

/// Charge thermique du **rebouilleur** `Q_b = Q_c + ΔH_F` par bilan enthalpique
/// global sur la colonne [W].
///
/// Convention de signe : `ΔH_F` regroupe l'ensemble des flux enthalpiques nets
/// **hors condenseur** (enthalpies d'alimentation et de soutirages) refermant le
/// bilan `Q_b = Q_c + ΔH_F` ; il est **positif** lorsque ces flux réclament un
/// apport supplémentaire au rebouilleur et **négatif** dans le cas contraire.
///
/// `condenser_duty` `Q_c` charge du condenseur [W ; ≥ 0, prise en valeur retirée],
/// `feed_enthalpy_difference` `ΔH_F` terme enthalpique net du bilan global [W ;
/// signé].
///
/// Panique si `condenser_duty` n'est pas fini ou est négatif, ou si
/// `feed_enthalpy_difference` n'est pas fini.
pub fn reflux_reboiler_duty(condenser_duty: f64, feed_enthalpy_difference: f64) -> f64 {
    assert!(
        condenser_duty.is_finite() && condenser_duty >= 0.0,
        "la charge du condenseur doit être finie et positive ou nulle"
    );
    assert!(
        feed_enthalpy_difference.is_finite(),
        "le terme enthalpique du bilan global doit être fini"
    );
    condenser_duty + feed_enthalpy_difference
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn underwood_cas_chiffre() {
        // α = 2,5 ; x_D = 0,95 ; x_F = 0,50 ; q = 1.
        // x_D/x_F = 0,95/0,50 = 1,90 ;
        // α·(1−x_D)/(1−x_F) = 2,5·0,05/0,50 = 0,25 ;
        // Rₘ = (1,90 − 0,25)/(2,5 − 1) = 1,65/1,5 = 1,10.
        let rm = reflux_minimum_underwood_binary(2.5, 0.95, 0.5, 1.0);
        assert_relative_eq!(rm, 1.1, epsilon = 1.0e-3);
    }

    #[test]
    fn operatoire_proportionnel_au_minimum() {
        // R = f·Rₘ : linéaire et homogène en Rₘ.
        let rm = 2.0;
        assert_relative_eq!(reflux_operating_from_ratio(rm, 1.5), 3.0, epsilon = 1.0e-3);
        assert_relative_eq!(
            reflux_operating_from_ratio(2.0 * rm, 1.5),
            2.0 * reflux_operating_from_ratio(rm, 1.5),
            epsilon = 1.0e-3
        );
    }

    #[test]
    fn pente_rectification_limites() {
        // s(0) = 0 (reflux nul) ; s → 1 quand R → ∞ (reflux total).
        assert_relative_eq!(reflux_rectifying_slope(0.0), 0.0, epsilon = 1.0e-3);
        assert_relative_eq!(reflux_rectifying_slope(1.0), 0.5, epsilon = 1.0e-3);
        assert_relative_eq!(reflux_rectifying_slope(3.0), 0.75, epsilon = 1.0e-3);
        assert!(reflux_rectifying_slope(1.0e6) < 1.0);
    }

    #[test]
    fn condenseur_charge_calculee() {
        // R = 2 ; D = 10 mol/s ; λ = 30 000 J/mol.
        // Q_c = (2 + 1)·10·30 000 = 3·300 000 = 900 000 W.
        let qc = reflux_condenser_duty(2.0, 10.0, 30_000.0);
        assert_relative_eq!(qc, 900_000.0, epsilon = 1.0e-3);
        // Proportionnalité au débit de distillat.
        assert_relative_eq!(
            reflux_condenser_duty(2.0, 20.0, 30_000.0),
            2.0 * qc,
            epsilon = 1.0e-3
        );
    }

    #[test]
    fn rebouilleur_bilan_signe() {
        // Q_b = Q_c + ΔH_F ; un ΔH_F nul redonne Q_c, un ΔH_F négatif le réduit.
        let qc = 900_000.0;
        assert_relative_eq!(reflux_reboiler_duty(qc, 0.0), qc, epsilon = 1.0e-3);
        assert_relative_eq!(
            reflux_reboiler_duty(qc, 50_000.0),
            950_000.0,
            epsilon = 1.0e-3
        );
        assert_relative_eq!(
            reflux_reboiler_duty(qc, -100_000.0),
            800_000.0,
            epsilon = 1.0e-3
        );
    }

    #[test]
    #[should_panic(expected = "volatilité relative")]
    fn underwood_refuse_alpha_unitaire() {
        // α = 1 : séparation impossible, division par (α − 1) = 0.
        let _ = reflux_minimum_underwood_binary(1.0, 0.95, 0.5, 1.0);
    }
}
