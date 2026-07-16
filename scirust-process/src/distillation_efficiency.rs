//! Distillation — **rendement d'étage** : efficacité de Murphree (plateau),
//! conversion Murphree→global par le facteur d'entraînement (relation de Lewis),
//! nombre de plateaux réels et hauteur de la colonne à plateaux.
//!
//! ```text
//! Murphree (vapeur)   E_M = (yₙ − yₙ₊₁) / (yₙ* − yₙ₊₁)                 [-]
//! plateaux réels      N_r = N_th / E_o                                 [plateaux]
//! Lewis (global)      E_o = ln[1 + E_M·(λ − 1)] / ln(λ)               [-]
//! hauteur colonne     H   = N_r · h_t                                  [m]
//! ```
//!
//! `E_M` rendement de Murphree en phase vapeur d'un plateau [sans dimension],
//! `yₙ` fraction molaire du constituant léger dans la vapeur quittant le plateau
//! `n` [sans dimension], `yₙ₊₁` fraction molaire dans la vapeur entrant sous le
//! plateau [sans dimension], `yₙ*` fraction molaire vapeur qui serait à l'équilibre
//! avec le liquide du plateau [sans dimension], `E_o` rendement global de la colonne
//! [sans dimension, dans `]0, 1]` en pratique], `N_th` nombre de plateaux
//! **théoriques** (étages d'équilibre) [plateaux], `N_r` nombre de plateaux
//! **réels** [plateaux], `λ` facteur d'entraînement (stripping factor) `m·V̇/L̇`
//! [sans dimension, strictement positif], `H` hauteur de la partie à plateaux de la
//! colonne [m], `h_t` espacement entre plateaux (tray spacing) [m].
//!
//! **Limite honnête** : le rendement de Murphree (**plateau**) et le rendement
//! **global** (colonne) sont reliés par le **facteur d'entraînement** `λ`
//! **fourni par l'appelant** (il dépend de la pente d'équilibre `m` et du rapport
//! des débits vapeur/liquide, jamais inventé ici). Le nombre de plateaux
//! **théoriques** `N_th` provient d'un calcul étage par étage **fourni**
//! (McCabe-Thiele, Fenske…). La conversion Murphree→global (relation de **Lewis**)
//! suppose des **droites opératoire et d'équilibre linéaires** et un facteur `λ`
//! constant sur la zone considérée ; hors de ce cadre, `E_o` doit être estimé
//! autrement. L'**espacement des plateaux** `h_t` est **fourni** (choix de
//! conception hydraulique) ; cette hauteur ignore les fonds, le ciel et les
//! internes. Ces fonctions n'évaluent ni les propriétés d'équilibre, ni
//! l'hydraulique des plateaux, ni les rendements ponctuels.

/// Rendement de Murphree en phase vapeur d'un plateau
/// `E_M = (yₙ − yₙ₊₁) / (yₙ* − yₙ₊₁)` [sans dimension].
///
/// C'est le rapport entre l'**enrichissement réel** de la vapeur sur le plateau et
/// l'enrichissement qu'il faudrait pour atteindre l'équilibre avec le liquide du
/// plateau. Passer directement `actual_vapor_change = yₙ − yₙ₊₁` et
/// `equilibrium_vapor_change = yₙ* − yₙ₊₁`.
///
/// `actual_vapor_change` variation réelle de fraction molaire vapeur sur le plateau
/// [sans dimension, fini], `equilibrium_vapor_change` variation à l'équilibre
/// [sans dimension, fini, non nul].
///
/// Panique si l'un des arguments n'est pas fini ou si `equilibrium_vapor_change`
/// est nul.
pub fn deff_murphree_vapor(actual_vapor_change: f64, equilibrium_vapor_change: f64) -> f64 {
    assert!(
        actual_vapor_change.is_finite(),
        "la variation réelle de fraction vapeur doit être finie (sans dimension)"
    );
    assert!(
        equilibrium_vapor_change.is_finite() && equilibrium_vapor_change != 0.0,
        "la variation d'équilibre de fraction vapeur doit être finie et non nulle (sans dimension)"
    );
    actual_vapor_change / equilibrium_vapor_change
}

/// Nombre de plateaux **réels** `N_r = N_th / E_o` [plateaux].
///
/// Convertit le nombre de plateaux théoriques (étages d'équilibre) en plateaux
/// réels via le **rendement global** de la colonne.
///
/// `theoretical_stages` `N_th` nombre de plateaux théoriques **fourni**
/// [plateaux, positif ou nul, fini], `overall_efficiency` `E_o` rendement global
/// [sans dimension, dans `]0, 1]`].
///
/// Panique si `theoretical_stages` est négatif ou non fini, ou si
/// `overall_efficiency` sort de `]0, 1]`.
pub fn deff_actual_stages(theoretical_stages: f64, overall_efficiency: f64) -> f64 {
    assert!(
        theoretical_stages.is_finite() && theoretical_stages >= 0.0,
        "le nombre de plateaux théoriques doit être fini et positif ou nul (plateaux)"
    );
    assert!(
        overall_efficiency.is_finite() && overall_efficiency > 0.0 && overall_efficiency <= 1.0,
        "le rendement global doit être fini et compris dans ]0, 1] (sans dimension)"
    );
    theoretical_stages / overall_efficiency
}

/// Rendement **global** de la colonne à partir du rendement de Murphree et du
/// facteur d'entraînement `E_o = ln[1 + E_M·(λ − 1)] / ln(λ)` (relation de Lewis)
/// [sans dimension].
///
/// L'argument du logarithme `1 + E_M·(λ − 1)` reste strictement positif pour
/// `E_M ∈ [0, 1]` et `λ > 0` (il vaut au minimum `λ` lorsque `E_M = 1`). Pour
/// `E_M = 1` on retrouve `E_o = 1` quel que soit `λ`.
///
/// `murphree_efficiency` `E_M` rendement de Murphree du plateau
/// [sans dimension, dans `[0, 1]`], `stripping_factor` `λ` facteur d'entraînement
/// **fourni** [sans dimension, strictement positif et différent de 1].
///
/// Panique si `murphree_efficiency` sort de `[0, 1]`, ou si `stripping_factor`
/// n'est pas fini, n'est pas strictement positif, ou vaut exactement 1 (le rapport
/// de logarithmes est alors indéterminé ; la limite continue est `E_o = E_M`).
pub fn deff_overall_from_murphree(murphree_efficiency: f64, stripping_factor: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&murphree_efficiency),
        "le rendement de Murphree doit être compris dans [0, 1] (sans dimension)"
    );
    assert!(
        stripping_factor.is_finite() && stripping_factor > 0.0 && stripping_factor != 1.0,
        "le facteur d'entraînement λ doit être fini, strictement positif et différent de 1 (sans dimension)"
    );
    (1.0 + murphree_efficiency * (stripping_factor - 1.0)).ln() / stripping_factor.ln()
}

/// Hauteur de la partie à plateaux de la colonne `H = N_r · h_t` [m].
///
/// `actual_stages` `N_r` nombre de plateaux réels [plateaux, positif ou nul, fini],
/// `tray_spacing` `h_t` espacement entre plateaux **fourni** [m, strictement
/// positif, fini].
///
/// Panique si `actual_stages` est négatif ou non fini, ou si `tray_spacing` n'est
/// pas fini et strictement positif.
pub fn deff_column_height(actual_stages: f64, tray_spacing: f64) -> f64 {
    assert!(
        actual_stages.is_finite() && actual_stages >= 0.0,
        "le nombre de plateaux réels doit être fini et positif ou nul (plateaux)"
    );
    assert!(
        tray_spacing.is_finite() && tray_spacing > 0.0,
        "l'espacement des plateaux doit être fini et strictement positif (m)"
    );
    actual_stages * tray_spacing
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn murphree_is_ratio_of_changes() {
        // E_M = (yₙ − yₙ₊₁)/(yₙ* − yₙ₊₁) = 0,4/0,5 = 0,8.
        let e_m = deff_murphree_vapor(0.4, 0.5);
        assert_relative_eq!(e_m, 0.8, epsilon = 1e-12);
        // Proportionnalité : doubler l'enrichissement réel double E_M (dénominateur fixe).
        let e_m2 = deff_murphree_vapor(0.8, 0.5);
        assert_relative_eq!(e_m2, 2.0 * e_m, epsilon = 1e-12);
    }

    #[test]
    fn perfect_plate_gives_unit_overall_efficiency() {
        // Identité de Lewis : E_M = 1 ⇒ E_o = ln(λ)/ln(λ) = 1, pour tout λ.
        assert_relative_eq!(deff_overall_from_murphree(1.0, 0.5), 1.0, epsilon = 1e-12);
        assert_relative_eq!(deff_overall_from_murphree(1.0, 1.7), 1.0, epsilon = 1e-12);
        assert_relative_eq!(deff_overall_from_murphree(1.0, 2.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(deff_overall_from_murphree(1.0, 3.5), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn overall_from_murphree_reference_value() {
        // Cas chiffré : E_M = 0,5 ; λ = 2,0.
        // E_o = ln(1 + 0,5·1)/ln(2) = ln(1,5)/ln(2)
        //     = 0,405465108108164 / 0,693147180559945 = 0,584962500721156.
        let e_o = deff_overall_from_murphree(0.5, 2.0);
        assert_relative_eq!(e_o, 0.584_962_500_721_156, epsilon = 1e-3);
    }

    #[test]
    fn actual_stages_reciprocal_of_efficiency() {
        // N_r = N_th / E_o ; réciproquement N_r·E_o = N_th.
        let n_th = 10.0;
        let e_o = 0.6;
        let n_r = deff_actual_stages(n_th, e_o);
        assert_relative_eq!(n_r, 10.0 / 0.6, epsilon = 1e-12);
        assert_relative_eq!(n_r * e_o, n_th, epsilon = 1e-12);
        // Rendement parfait ⇒ plateaux réels = plateaux théoriques.
        assert_relative_eq!(deff_actual_stages(n_th, 1.0), n_th, epsilon = 1e-12);
    }

    #[test]
    fn column_height_is_linear_in_stage_count() {
        // H = N_r·h_t : 17 plateaux à 0,50 m ⇒ 8,5 m.
        let h = deff_column_height(17.0, 0.5);
        assert_relative_eq!(h, 8.5, epsilon = 1e-12);
        // Doubler l'espacement double la hauteur.
        assert_relative_eq!(deff_column_height(17.0, 1.0), 2.0 * h, epsilon = 1e-12);
    }

    #[test]
    fn realistic_chain_murphree_to_column() {
        // Chaîne réaliste : E_M = 0,70 ; λ = 1,5 ; N_th = 12 ; h_t = 0,45 m.
        // E_o = ln(1 + 0,70·0,5)/ln(1,5) = ln(1,35)/ln(1,5)
        //     = 0,300104592450338 / 0,405465108108164 = 0,740148…
        let e_o = deff_overall_from_murphree(0.70, 1.5);
        assert_relative_eq!(e_o, 0.740_148_401_5, epsilon = 1e-3);
        // N_r = 12 / 0,740148 = 16,2131… plateaux.
        let n_r = deff_actual_stages(12.0, e_o);
        assert_relative_eq!(n_r, 12.0 / e_o, epsilon = 1e-9);
        assert_relative_eq!(n_r, 16.213_15, epsilon = 1e-3);
        // H = N_r · 0,45 ≈ 7,2959 m.
        let h = deff_column_height(n_r, 0.45);
        assert_relative_eq!(h, n_r * 0.45, epsilon = 1e-12);
        assert_relative_eq!(h, 7.295_92, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "différent de 1")]
    fn unit_stripping_factor_panics() {
        // λ = 1 rend le rapport de logarithmes indéterminé : rejet.
        deff_overall_from_murphree(0.5, 1.0);
    }
}
