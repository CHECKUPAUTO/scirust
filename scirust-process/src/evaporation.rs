//! Évaporation (concentration par ébullition) — eau évaporée par bilan matière
//! sur les solides non volatils, économie de vapeur d'un simple ou d'un multiple
//! effet, et élévation ébullioscopique par la règle de Dühring.
//!
//! ```text
//! eau évaporée (bilan solides)    V   = F · (1 − x_F / x_P)          [kg·s⁻¹]
//! économie de vapeur              E   = V / S                        [-]
//! élévation ébullioscopique       ΔT  = k · (T_b − T_ref)            [K]
//!   (règle de Dühring)
//! économie multiple effet         E_N = N · E_1                      [-]
//! ```
//!
//! `F` débit d'alimentation [kg·s⁻¹], `x_F`/`x_P` fractions massiques de solides
//! **non volatils** dans l'alimentation / le produit concentré [sans dimension,
//! 0 ≤ x ≤ 1], `V` débit d'eau (solvant) évaporée [kg·s⁻¹] ; `S` débit de vapeur
//! vive consommée [kg·s⁻¹], `E` économie de vapeur [sans dimension, kg évaporé
//! par kg de vapeur vive] ; `k` pente de la droite de Dühring [sans dimension],
//! `T_b` température d'ébullition du **solvant pur** à la pression de l'effet
//! [K], `T_ref` température d'ébullition de référence [K], `ΔT` élévation du point
//! d'ébullition de la solution au-dessus du solvant pur [K] ; `N` nombre d'effets
//! [effets], `E_1` économie moyenne par effet [sans dimension], `E_N` économie
//! approchée de la batterie [sans dimension].
//!
//! **Limite honnête** : le bilan matière porte sur les **solides non volatils**
//! (supposés entièrement retenus dans le produit, aucun entraînement) et suppose
//! que seul le **solvant** s'évapore. L'**économie de vapeur** (`E`, ~0,8 par
//! effet en simple effet), l'**économie par effet** (`E_1`) et la **pente de
//! Dühring** (`k`) sont **FOURNIES** par l'appelant : elles dépendent du produit,
//! des enthalpies, de la pression et du nombre d'effets, et ne sont **jamais**
//! supposées « par défaut ». L'élévation ébullioscopique `ΔT` **réduit la force
//! motrice thermique** disponible mais n'est pas retranchée ici automatiquement.
//! Aucune propriété physique (enthalpies, chaleurs latentes, volatilités,
//! coefficients de partage, constantes cinétiques, diffusivités, coefficient
//! global d'échange…) n'est calculée : la **surface d'échange** détaillée relève
//! du coefficient global **FOURNI** par l'appelant et n'est pas traitée ici.

/// Eau (solvant) évaporée par bilan matière sur les solides non volatils
/// `V = F · (1 − x_F / x_P)` (kg·s⁻¹), où le débit de produit concentré vaut
/// `P = F · x_F / x_P` par conservation des solides `F · x_F = P · x_P`.
///
/// `feed_flow` (F) débit d'alimentation [kg·s⁻¹], `feed_solids_fraction` (x_F)
/// et `product_solids_fraction` (x_P) fractions massiques de solides non volatils
/// [sans dimension]. La concentration augmentant, on exige `x_P ≥ x_F`.
///
/// Panique si `feed_flow < 0`, si `x_F` hors de `[0, 1]`, si `x_P` hors de
/// `]0, 1]`, ou si `x_P < x_F` (le produit ne peut être moins concentré).
pub fn evap_water_evaporated(
    feed_flow: f64,
    feed_solids_fraction: f64,
    product_solids_fraction: f64,
) -> f64 {
    assert!(feed_flow >= 0.0, "F ≥ 0 requis (débit d'alimentation)");
    assert!(
        (0.0..=1.0).contains(&feed_solids_fraction),
        "0 ≤ x_F ≤ 1 requis (fraction de solides à l'alimentation)"
    );
    assert!(
        product_solids_fraction > 0.0 && product_solids_fraction <= 1.0,
        "0 < x_P ≤ 1 requis (fraction de solides au produit)"
    );
    assert!(
        product_solids_fraction >= feed_solids_fraction,
        "x_P ≥ x_F requis (le produit doit être au moins aussi concentré)"
    );
    feed_flow * (1.0 - feed_solids_fraction / product_solids_fraction)
}

/// Économie de vapeur `E = V / S` (sans dimension), masse de solvant évaporée par
/// masse de vapeur vive consommée (~0,8 par effet en simple effet).
///
/// `water_evaporated` (V) débit d'eau évaporée [kg·s⁻¹], `steam_consumed` (S)
/// débit de vapeur vive [kg·s⁻¹], exprimés dans la **même unité cohérente**.
///
/// Panique si `water_evaporated < 0` ou si `steam_consumed <= 0`.
pub fn evap_steam_economy(water_evaporated: f64, steam_consumed: f64) -> f64 {
    assert!(water_evaporated >= 0.0, "V ≥ 0 requis (eau évaporée)");
    assert!(steam_consumed > 0.0, "S > 0 requis (vapeur vive consommée)");
    water_evaporated / steam_consumed
}

/// Élévation ébullioscopique par la règle de Dühring
/// `ΔT = k · (T_b − T_ref)` (K), forme linéaire où la pente `k` traduit la droite
/// de Dühring propre au produit et à sa concentration.
///
/// `duhring_slope` (k) pente de Dühring [sans dimension], `solvent_boiling_point`
/// (T_b) température d'ébullition du solvant pur à la pression de l'effet [K],
/// `reference_boiling_point` (T_ref) température d'ébullition de référence [K].
///
/// Panique si `duhring_slope < 0`, si `T_b <= 0`, si `T_ref <= 0`, ou si
/// `T_b < T_ref` (l'élévation ébullioscopique serait négative).
pub fn evap_boiling_point_elevation_duhring(
    duhring_slope: f64,
    solvent_boiling_point: f64,
    reference_boiling_point: f64,
) -> f64 {
    assert!(duhring_slope >= 0.0, "k ≥ 0 requis (pente de Dühring)");
    assert!(
        solvent_boiling_point > 0.0,
        "T_b > 0 K requis (ébullition du solvant pur)"
    );
    assert!(
        reference_boiling_point > 0.0,
        "T_ref > 0 K requis (ébullition de référence)"
    );
    assert!(
        solvent_boiling_point >= reference_boiling_point,
        "T_b ≥ T_ref requis (élévation ébullioscopique non négative)"
    );
    duhring_slope * (solvent_boiling_point - reference_boiling_point)
}

/// Économie de vapeur approchée d'un multiple effet
/// `E_N = N · E_1` (sans dimension), somme des contributions de `N` effets
/// d'économie moyenne `E_1` chacun.
///
/// `number_of_effects` (N) nombre d'effets [entier ≥ 1], `economy_per_effect`
/// (E_1) économie moyenne par effet [sans dimension].
///
/// Panique si `number_of_effects == 0` ou si `economy_per_effect < 0`.
pub fn evap_multiple_effect_economy(number_of_effects: u32, economy_per_effect: f64) -> f64 {
    assert!(number_of_effects >= 1, "N ≥ 1 requis (nombre d'effets)");
    assert!(
        economy_per_effect >= 0.0,
        "E_1 ≥ 0 requis (économie par effet)"
    );
    f64::from(number_of_effects) * economy_per_effect
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn water_evaporated_closes_solids_balance() {
        // F = 10 kg/s, x_F = 0.05, x_P = 0.20 ⇒ V = 10·(1 − 0.05/0.20)
        //   = 10·(1 − 0.25) = 7.5 kg/s.
        let feed = 10.0_f64;
        let (x_f, x_p) = (0.05_f64, 0.20_f64);
        let v = evap_water_evaporated(feed, x_f, x_p);
        assert_relative_eq!(v, 7.5, max_relative = 1e-12);
        // Bilan solides : F·x_F = P·x_P avec P = F − V.
        let product = feed - v;
        assert_relative_eq!(feed * x_f, product * x_p, max_relative = 1e-12);
    }

    #[test]
    fn water_evaporated_zero_when_no_concentration() {
        // x_P = x_F ⇒ aucune concentration ⇒ aucune eau évaporée.
        assert_relative_eq!(
            evap_water_evaporated(10.0_f64, 0.10_f64, 0.10_f64),
            0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn steam_economy_is_inverse_of_specific_consumption() {
        // V = 8, S = 10 ⇒ E = 0.8 ; réciproquement E·S = V.
        let (v, s) = (8.0_f64, 10.0_f64);
        let e = evap_steam_economy(v, s);
        assert_relative_eq!(e, 0.8, max_relative = 1e-12);
        assert_relative_eq!(e * s, v, max_relative = 1e-12);
    }

    #[test]
    fn duhring_elevation_realistic_and_zero_limit() {
        // k = 0.5, T_b = 383 K, T_ref = 373 K ⇒ ΔT = 0.5·10 = 5 K.
        assert_relative_eq!(
            evap_boiling_point_elevation_duhring(0.5_f64, 383.0_f64, 373.0_f64),
            5.0,
            max_relative = 1e-12
        );
        // T_b = T_ref ⇒ ΔT = 0 (pas d'élévation).
        assert_relative_eq!(
            evap_boiling_point_elevation_duhring(0.5_f64, 373.0_f64, 373.0_f64),
            0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn multiple_effect_scales_with_number_of_effects() {
        // N = 3, E_1 = 0.8 ⇒ E_N = 2.4 ; identité E_N = N · E_1 face au 1 effet.
        let e1 = 0.8_f64;
        let single = evap_multiple_effect_economy(1, e1);
        let triple = evap_multiple_effect_economy(3, e1);
        assert_relative_eq!(single, 0.8, max_relative = 1e-12);
        assert_relative_eq!(triple, 2.4, max_relative = 1e-12);
        assert_relative_eq!(triple, 3.0 * single, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "x_P ≥ x_F requis")]
    fn water_evaporated_panics_when_product_less_concentrated() {
        // Produit moins concentré que l'alimentation ⇒ impossible ⇒ panique.
        let _ = evap_water_evaporated(10.0_f64, 0.20_f64, 0.05_f64);
    }
}
