//! Cristallisation — sursaturation (moteur de la nucléation et de la croissance)
//! et rendement en cristaux (bilan matière), pour des cristaux anhydres ou
//! hydratés dont l'eau de cristallisation est retirée du solvant.
//!
//! ```text
//! rapport de sursaturation        S  = c / c*                              [-]
//! sursaturation absolue           Δc = c − c*                              [kg·kg⁻¹]
//! rendement cristaux anhydres     Y  = m_solv · (c₁ − c*)                  [kg]
//! rapport molaire d'hydrate       R  = M_hydrate / M_anhydre               [-]
//! rendement cristaux hydratés     Y_h = R · m_solv · (c₁ − c*)
//!                                        / [1 − c* · (R − 1)]              [kg]
//! ```
//!
//! `c` concentration **réelle** du soluté, `c*` (c\*) concentration à
//! **saturation** (solubilité) à la température de travail, toutes deux exprimées
//! par **masse de solvant** [kg de soluté anhydre · kg de solvant⁻¹] ; `S` rapport
//! de sursaturation [sans dimension] (`S > 1` solution sursaturée, `S = 1`
//! saturée, `S < 1` sous-saturée), `Δc` sursaturation absolue [même unité que
//! `c`] ; `m_solv` masse de solvant [kg], `c₁` concentration **initiale** et
//! `c*` concentration finale à **saturation** de la liqueur mère [kg·kg⁻¹] ;
//! `M_hydrate`/`M_anhydre` masses molaires du cristal **hydraté**/du sel
//! **anhydre** [kg·mol⁻¹ ou g·mol⁻¹, même unité], `R` rapport molaire
//! d'hydratation [sans dimension, `R ≥ 1`] ; `Y`/`Y_h` masses de cristaux
//! anhydres/hydratés déposées [kg].
//!
//! **Limite honnête** : ces relations décrivent une cristallisation **à
//! l'équilibre**, la liqueur mère étant supposée **saturée en fin d'opération**
//! (équilibre atteint) et les **pertes négligées** (pas d'entraînement, pas de
//! cristaux redissous). Les **solubilités** `c*` aux températures de travail, le
//! **rapport molaire** `R = M_hydrate / M_anhydre` et les masses molaires sont
//! **FOURNIS** par l'appelant ; pour un hydrate, l'**eau de cristallisation** est
//! **retirée du solvant** (le solvant disponible diminue, d'où le dénominateur
//! `1 − c*·(R − 1)`). Aucune propriété physique (enthalpies, volatilités,
//! coefficients de partage, constantes cinétiques, diffusivités, solubilités…)
//! n'est **jamais** supposée « par défaut » : elles proviennent de tables,
//! d'essais ou de l'appelant. Le modèle **néglige l'évaporation** du solvant
//! (cristallisation par refroidissement seul) ; un flash évaporatoire exigerait
//! un terme supplémentaire non couvert ici.

/// Rapport de sursaturation `S = c / c*` (sans dimension). `S > 1` indique une
/// solution **sursaturée** (moteur de la cristallisation), `S = 1` une solution
/// **saturée** et `S < 1` une solution **sous-saturée**.
///
/// `actual_concentration` (c) concentration réelle du soluté et
/// `saturation_concentration` (c*) solubilité à la température de travail,
/// exprimées dans la **même unité** par masse de solvant [kg·kg⁻¹].
///
/// Panique si `actual_concentration < 0` ou si `saturation_concentration <= 0`.
pub fn cryst_supersaturation_ratio(
    actual_concentration: f64,
    saturation_concentration: f64,
) -> f64 {
    assert!(
        actual_concentration >= 0.0,
        "c ≥ 0 requis (concentration réelle)"
    );
    assert!(
        saturation_concentration > 0.0,
        "c* > 0 requis (concentration à saturation)"
    );
    actual_concentration / saturation_concentration
}

/// Sursaturation absolue `Δc = c − c*` (même unité que les concentrations). Une
/// valeur positive traduit une solution **sursaturée** ; elle est liée au rapport
/// de sursaturation par `Δc = c*·(S − 1)`.
///
/// `actual_concentration` (c) concentration réelle et `saturation_concentration`
/// (c*) solubilité à la température de travail, exprimées dans la **même unité**
/// par masse de solvant [kg·kg⁻¹].
///
/// Panique si `actual_concentration < 0` ou `saturation_concentration < 0`.
pub fn cryst_supersaturation_difference(
    actual_concentration: f64,
    saturation_concentration: f64,
) -> f64 {
    assert!(
        actual_concentration >= 0.0,
        "c ≥ 0 requis (concentration réelle)"
    );
    assert!(
        saturation_concentration >= 0.0,
        "c* ≥ 0 requis (concentration à saturation)"
    );
    actual_concentration - saturation_concentration
}

/// Rendement en cristaux **anhydres** `Y = m_solv · (c₁ − c*)` (kg), par bilan
/// matière sur le soluté avec liqueur mère saturée en fin d'opération. Les
/// concentrations sont exprimées **par masse de solvant**, supposée constante
/// (pas d'eau de cristallisation retirée, pas d'évaporation).
///
/// `solvent_mass` (m_solv) masse de solvant [kg] ; `initial_concentration` (c₁)
/// concentration initiale du soluté et `final_saturation_concentration` (c*)
/// solubilité finale, exprimées par masse de solvant [kg·kg⁻¹].
///
/// Panique si `solvent_mass < 0`, si une concentration est négative, ou si
/// `initial_concentration < final_saturation_concentration` (rendement négatif
/// non physique : la solution n'était pas sursaturée au départ).
pub fn cryst_yield_anhydrous(
    solvent_mass: f64,
    initial_concentration: f64,
    final_saturation_concentration: f64,
) -> f64 {
    assert!(solvent_mass >= 0.0, "m_solv ≥ 0 requis (masse de solvant)");
    assert!(
        initial_concentration >= 0.0,
        "c₁ ≥ 0 requis (concentration initiale)"
    );
    assert!(
        final_saturation_concentration >= 0.0,
        "c* ≥ 0 requis (concentration à saturation finale)"
    );
    assert!(
        initial_concentration >= final_saturation_concentration,
        "c₁ ≥ c* requis (rendement non négatif : solution sursaturée)"
    );
    solvent_mass * (initial_concentration - final_saturation_concentration)
}

/// Rendement en cristaux **hydratés**
/// `Y_h = R · m_solv · (c₁ − c*) / [1 − c*·(R − 1)]` (kg), avec
/// `R = M_hydrate / M_anhydre`. Le bilan matière tient compte de l'**eau de
/// cristallisation** retirée du solvant : la fraction anhydre d'un cristal
/// hydraté vaut `1/R`, si bien qu'une masse `Y_h` de cristaux emprisonne
/// `Y_h·(R − 1)/R` de solvant, ce qui réduit la liqueur mère (d'où le
/// dénominateur `1 − c*·(R − 1)`). Pour `R = 1` (sel anhydre), la formule se
/// réduit à [`cryst_yield_anhydrous`].
///
/// `solvent_mass` (m_solv) masse de solvant initiale [kg] ;
/// `initial_concentration` (c₁) et `final_saturation_concentration` (c*)
/// concentrations en **soluté anhydre par masse de solvant** [kg·kg⁻¹] ;
/// `hydrate_molar_mass` (M_hydrate) et `anhydrous_molar_mass` (M_anhydre) masses
/// molaires du cristal hydraté et du sel anhydre [même unité, kg·mol⁻¹ ou
/// g·mol⁻¹].
///
/// Panique si `solvent_mass < 0`, si une concentration est négative, si
/// `anhydrous_molar_mass <= 0`, si `hydrate_molar_mass < anhydrous_molar_mass`
/// (donnerait `R < 1`), si `initial_concentration < final_saturation_concentration`,
/// ou si le dénominateur `1 − c*·(R − 1)` n'est pas strictement positif (solvant
/// épuisé par l'eau de cristallisation).
pub fn cryst_yield_hydrate(
    solvent_mass: f64,
    initial_concentration: f64,
    final_saturation_concentration: f64,
    hydrate_molar_mass: f64,
    anhydrous_molar_mass: f64,
) -> f64 {
    assert!(solvent_mass >= 0.0, "m_solv ≥ 0 requis (masse de solvant)");
    assert!(
        initial_concentration >= 0.0,
        "c₁ ≥ 0 requis (concentration initiale)"
    );
    assert!(
        final_saturation_concentration >= 0.0,
        "c* ≥ 0 requis (concentration à saturation finale)"
    );
    assert!(
        anhydrous_molar_mass > 0.0,
        "M_anhydre > 0 requis (masse molaire du sel anhydre)"
    );
    assert!(
        hydrate_molar_mass >= anhydrous_molar_mass,
        "M_hydrate ≥ M_anhydre requis (R ≥ 1)"
    );
    assert!(
        initial_concentration >= final_saturation_concentration,
        "c₁ ≥ c* requis (rendement non négatif : solution sursaturée)"
    );
    let r = hydrate_molar_mass / anhydrous_molar_mass;
    let denominator = 1.0 - final_saturation_concentration * (r - 1.0);
    assert!(
        denominator > 0.0,
        "1 − c*·(R − 1) > 0 requis (solvant non épuisé par l'eau de cristallisation)"
    );
    r * solvent_mass * (initial_concentration - final_saturation_concentration) / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ratio_and_difference_are_consistent() {
        // c = 0.45, c* = 0.30 ⇒ S = 1.5 et Δc = 0.15.
        let s = cryst_supersaturation_ratio(0.45_f64, 0.30_f64);
        let dc = cryst_supersaturation_difference(0.45_f64, 0.30_f64);
        assert_relative_eq!(s, 1.5, max_relative = 1e-12);
        assert_relative_eq!(dc, 0.15, max_relative = 1e-12);
        // Identité : Δc = c*·(S − 1).
        assert_relative_eq!(dc, 0.30_f64 * (s - 1.0), max_relative = 1e-12);
    }

    #[test]
    fn saturated_solution_gives_unit_ratio_and_zero_difference() {
        // À saturation, c = c* ⇒ S = 1 et Δc = 0.
        assert_relative_eq!(
            cryst_supersaturation_ratio(0.30_f64, 0.30_f64),
            1.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            cryst_supersaturation_difference(0.30_f64, 0.30_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn anhydrous_yield_realistic_and_proportional() {
        // m_solv = 100 kg, c₁ = 0.50, c* = 0.20 ⇒ Y = 100·0.30 = 30 kg.
        let y = cryst_yield_anhydrous(100.0_f64, 0.50_f64, 0.20_f64);
        assert_relative_eq!(y, 30.0, max_relative = 1e-12);
        // Proportionnalité : doubler le solvant double le rendement.
        let y2 = cryst_yield_anhydrous(200.0_f64, 0.50_f64, 0.20_f64);
        assert_relative_eq!(y2, 2.0 * y, max_relative = 1e-12);
    }

    #[test]
    fn hydrate_yield_realistic_case() {
        // R = 2, m_solv = 100, c₁ = 0.50, c* = 0.20 :
        // Y_h = 2·100·(0.50 − 0.20) / (1 − 0.20·(2 − 1))
        //     = 60 / 0.80 = 75 kg.
        let y_h = cryst_yield_hydrate(100.0_f64, 0.50_f64, 0.20_f64, 2.0_f64, 1.0_f64);
        assert_relative_eq!(y_h, 75.0, max_relative = 1e-9);
    }

    #[test]
    fn hydrate_reduces_to_anhydrous_when_r_is_one() {
        // R = 1 (M_hydrate = M_anhydre) ⇒ Y_h = Y anhydre.
        let y_h = cryst_yield_hydrate(100.0_f64, 0.50_f64, 0.20_f64, 1.0_f64, 1.0_f64);
        let y = cryst_yield_anhydrous(100.0_f64, 0.50_f64, 0.20_f64);
        assert_relative_eq!(y_h, y, max_relative = 1e-12);
    }

    #[test]
    fn hydrate_yield_closes_the_mass_balance() {
        // Vérification physique : avec Y_h = 75 kg (R = 2), la fraction anhydre
        // des cristaux est Y_h/R = 37.5, l'eau de cristallisation Y_h·(R−1)/R
        // = 37.5, le solvant restant 100 − 37.5 = 62.5, le sel en solution
        // 62.5·c* = 62.5·0.20 = 12.5. Bilan : 37.5 + 12.5 = 50 = m_solv·c₁.
        let r = 2.0_f64;
        let y_h = cryst_yield_hydrate(100.0_f64, 0.50_f64, 0.20_f64, 2.0_f64, 1.0_f64);
        let anhydrous_in_crystals = y_h / r;
        let water_of_crystallization = y_h * (r - 1.0) / r;
        let solvent_left = 100.0_f64 - water_of_crystallization;
        let salt_in_solution = solvent_left * 0.20_f64;
        assert_relative_eq!(
            anhydrous_in_crystals + salt_in_solution,
            100.0_f64 * 0.50_f64,
            max_relative = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "c* > 0 requis")]
    fn ratio_panics_on_zero_saturation() {
        // Solubilité nulle ⇒ rapport indéfini (division par zéro) ⇒ panique.
        let _ = cryst_supersaturation_ratio(0.30_f64, 0.0_f64);
    }
}
