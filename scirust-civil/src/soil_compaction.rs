//! **Géotechnique — compactage des sols (essai Proctor)** : masse volumique sèche
//! à partir de la densité humide et de la teneur en eau, compacité relative par
//! rapport à l'optimum Proctor, courbe de saturation (« sans vide d'air ») et
//! degré de saturation d'un sol compacté.
//!
//! ```text
//! masse volumique sèche          ρd = ρ / (1 + w)
//! compacité relative             RC = ρd_field / ρd_max
//! densité sans vide d'air        ρd_zav = Gs·ρw / (1 + w·Gs)
//! degré de saturation            Sr = w·Gs / e
//! ```
//!
//! `ρd` = masse volumique **sèche** du sol (kg/m³), `ρ` = `bulk_density` masse
//! volumique **humide** (totale) du sol (kg/m³), `w` = `water_content` teneur en
//! eau **en base sèche** (masse d'eau / masse des grains, sans dimension), `RC` =
//! compacité relative (sans dimension), `ρd_field` = `field_dry_density` masse
//! volumique sèche mesurée en place (kg/m³), `ρd_max` = `maximum_dry_density` masse
//! volumique sèche **maximale** de l'optimum Proctor (kg/m³), `ρd_zav` = masse
//! volumique sèche théorique **à saturation** (courbe sans vide d'air, kg/m³),
//! `Gs` = `specific_gravity` densité des grains solides (sans dimension), `ρw` =
//! `water_density` masse volumique de l'eau (kg/m³), `Sr` = degré de saturation
//! (sans dimension ; `Sr = 1` correspond au sol saturé), `e` = `void_ratio` indice
//! des vides (sans dimension).
//!
//! **Convention** : SI strict — **kg, m, s**, masses volumiques en **kg/m³** ; la
//! teneur en eau `w`, la compacité `RC`, la densité des grains `Gs`, le degré de
//! saturation `Sr` et l'indice des vides `e` sont **sans dimension**. La teneur en
//! eau est exprimée **en base sèche** (rapporté à la masse des grains), comme dans
//! l'essai Proctor. Si l'appelant travaille en unités de poids volumique (N/m³),
//! il suffit de remplacer `ρw` par le poids volumique de l'eau et d'interpréter
//! toutes les masses volumiques comme des poids volumiques : les formules,
//! homogènes, restent valables.
//!
//! **Limite honnête** : les grandeurs d'entrée — masse volumique humide `ρ` et
//! teneur en eau `w` de l'échantillon, masse volumique sèche **maximale** `ρd_max`
//! et teneur en eau optimale de l'**essai Proctor** (normal ou modifié), densité
//! des grains `Gs` — sont **fournies par l'essai** (Proctor, pycnomètre) ; aucune
//! valeur « par défaut » n'est inventée. La compacité relative `RC` se rapporte à
//! un **essai Proctor de référence fourni** (normal / modifié) : le module ne
//! choisit pas la référence. La **courbe sans vide d'air** (`ρd_zav`) borne
//! supérieurement la masse volumique sèche atteignable à une teneur en eau donnée
//! (`ρd ≤ ρd_zav`) mais **n'est pas une cible de chantier** (le sol n'atteint pas
//! la saturation totale au compactage). Les éventuelles résistances
//! caractéristiques du sol **et** les coefficients partiels de sécurité relèvent de
//! l'appelant selon l'Eurocode 7 et son Annexe Nationale. Ce module ne couvre
//! **ni** l'énergie de compactage, **ni** la portance (`CBR`), **ni** la
//! perméabilité du sol compacté.

/// Masse volumique **sèche** du sol `ρd = ρ / (1 + w)` (kg/m³), à partir de la
/// masse volumique humide `ρ` (kg/m³) et de la teneur en eau `w` **en base sèche**
/// (sans dimension).
///
/// Panique si `bulk_density <= 0` ou si `water_content < 0`.
pub fn compact_dry_density(bulk_density: f64, water_content: f64) -> f64 {
    assert!(
        bulk_density > 0.0,
        "la masse volumique humide ρ doit être strictement positive"
    );
    assert!(water_content >= 0.0, "la teneur en eau w doit être ≥ 0");
    bulk_density / (1.0 + water_content)
}

/// Compacité relative `RC = ρd_field / ρd_max` (sans dimension), rapport de la
/// masse volumique sèche mesurée en place à la masse volumique sèche **maximale**
/// de l'optimum Proctor.
///
/// Panique si `field_dry_density < 0` ou si `maximum_dry_density <= 0`
/// (division par zéro).
pub fn compact_relative_compaction(field_dry_density: f64, maximum_dry_density: f64) -> f64 {
    assert!(
        field_dry_density >= 0.0,
        "la masse volumique sèche en place ρd_field doit être ≥ 0"
    );
    assert!(
        maximum_dry_density > 0.0,
        "la masse volumique sèche maximale ρd_max doit être strictement positive"
    );
    field_dry_density / maximum_dry_density
}

/// Masse volumique sèche théorique **à saturation** (courbe sans vide d'air)
/// `ρd_zav = Gs·ρw / (1 + w·Gs)` (kg/m³), avec `Gs` densité des grains (sans
/// dimension), `ρw` masse volumique de l'eau (kg/m³) et `w` teneur en eau (sans
/// dimension).
///
/// Panique si `water_content < 0`, si `specific_gravity <= 0` ou si
/// `water_density <= 0`.
pub fn compact_zero_air_voids_density(
    water_content: f64,
    specific_gravity: f64,
    water_density: f64,
) -> f64 {
    assert!(water_content >= 0.0, "la teneur en eau w doit être ≥ 0");
    assert!(
        specific_gravity > 0.0,
        "la densité des grains Gs doit être strictement positive"
    );
    assert!(
        water_density > 0.0,
        "la masse volumique de l'eau ρw doit être strictement positive"
    );
    specific_gravity * water_density / (1.0 + water_content * specific_gravity)
}

/// Degré de saturation `Sr = w·Gs / e` (sans dimension), avec `w` teneur en eau,
/// `Gs` densité des grains et `e` indice des vides (tous sans dimension) ;
/// `Sr = 1` correspond au sol saturé.
///
/// Panique si `water_content < 0`, si `specific_gravity <= 0` ou si
/// `void_ratio <= 0` (division par zéro).
pub fn compact_degree_of_saturation(
    water_content: f64,
    specific_gravity: f64,
    void_ratio: f64,
) -> f64 {
    assert!(water_content >= 0.0, "la teneur en eau w doit être ≥ 0");
    assert!(
        specific_gravity > 0.0,
        "la densité des grains Gs doit être strictement positive"
    );
    assert!(
        void_ratio > 0.0,
        "l'indice des vides e doit être strictement positif"
    );
    water_content * specific_gravity / void_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dry_density_matches_hand_computation() {
        // Cas chiffré : ρ = 2000 kg/m³, w = 0,25 (25 %).
        //   ρd = 2000 / (1 + 0,25) = 2000 / 1,25 = 1600 kg/m³.
        let rho_d = compact_dry_density(2000.0, 0.25);
        assert_relative_eq!(rho_d, 1600.0, max_relative = 1e-12);
    }

    #[test]
    fn dry_density_equals_bulk_when_dry() {
        // Cas limite : w = 0 ⇒ le sol est sec, ρd = ρ.
        let rho_d = compact_dry_density(1850.0, 0.0);
        assert_relative_eq!(rho_d, 1850.0, max_relative = 1e-12);
    }

    #[test]
    fn relative_compaction_is_one_at_optimum() {
        // À l'optimum, ρd_field = ρd_max ⇒ RC = 1 (100 %).
        let rc = compact_relative_compaction(1800.0, 1800.0);
        assert_relative_eq!(rc, 1.0, max_relative = 1e-12);
        // Cas chiffré : 1600 / 1800 = 0,888888… (≈ 88,9 %).
        let rc2 = compact_relative_compaction(1600.0, 1800.0);
        assert_relative_eq!(rc2, 0.888_888_888_888_889, max_relative = 1e-12);
    }

    #[test]
    fn zero_air_voids_matches_hand_computation() {
        // Cas chiffré : w = 0,15, Gs = 2,70, ρw = 1000 kg/m³.
        //   ρd_zav = 2,70·1000 / (1 + 0,15·2,70)
        //          = 2700 / (1 + 0,405) = 2700 / 1,405
        //          ≈ 1921,708185053381 kg/m³.
        let rho_zav = compact_zero_air_voids_density(0.15, 2.70, 1000.0);
        assert_relative_eq!(rho_zav, 1_921.708_185_053_381, max_relative = 1e-9);
    }

    #[test]
    fn zero_air_voids_equals_solids_density_when_dry() {
        // Cas limite : w = 0 ⇒ ρd_zav = Gs·ρw (masse volumique des grains).
        let rho_zav = compact_zero_air_voids_density(0.0, 2.65, 1000.0);
        assert_relative_eq!(rho_zav, 2650.0, max_relative = 1e-12);
    }

    #[test]
    fn saturation_and_zero_air_voids_are_consistent() {
        // Identité : sur la courbe sans vide d'air, le sol est saturé (Sr = 1).
        // À saturation, e = w·Gs, donc ρd = Gs·ρw / (1 + e) = ρd_zav, et
        // Sr(w, Gs, e = w·Gs) = w·Gs / (w·Gs) = 1.
        let (w, gs, rho_w) = (0.20_f64, 2.65_f64, 1000.0_f64);
        let e_sat = w * gs; // indice des vides à saturation
        let sr = compact_degree_of_saturation(w, gs, e_sat);
        assert_relative_eq!(sr, 1.0, max_relative = 1e-12);
        // La densité sèche à saturation vaut alors ρd_zav.
        let rho_sat = gs * rho_w / (1.0 + e_sat);
        let rho_zav = compact_zero_air_voids_density(w, gs, rho_w);
        assert_relative_eq!(rho_sat, rho_zav, max_relative = 1e-12);
    }

    #[test]
    fn degree_of_saturation_matches_hand_computation() {
        // Cas chiffré : w = 0,10, Gs = 2,70, e = 0,60.
        //   Sr = 0,10·2,70 / 0,60 = 0,27 / 0,60 = 0,45 (45 %).
        let sr = compact_degree_of_saturation(0.10, 2.70, 0.60);
        assert_relative_eq!(sr, 0.45, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(
        expected = "la masse volumique sèche maximale ρd_max doit être strictement positive"
    )]
    fn relative_compaction_rejects_zero_maximum() {
        // ρd_max = 0 interdit : division par zéro.
        compact_relative_compaction(1600.0, 0.0);
    }
}
