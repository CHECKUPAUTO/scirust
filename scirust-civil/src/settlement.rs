//! **Géotechnique — tassement de consolidation œdométrique (Terzaghi)** :
//! tassement final d'une couche d'argile par consolidation **unidimensionnelle**,
//! selon qu'elle est normalement consolidée (indice de compression `Cc`) ou
//! rechargée sous la pression de préconsolidation (indice de recompression `Cr`),
//! plus le facteur temps `Tv` de la théorie de Terzaghi et le tassement déduit
//! d'une déformation verticale mesurée.
//!
//! ```text
//! argile normalement consolidée   s = (Cc·H / (1 + e0)) · log10((σ0 + Δσ) / σ0)
//! recompression (surconsolidée)   s = (Cr·H / (1 + e0)) · log10(σf / σ0)
//! facteur temps                   Tv = cv · t / Hdr²
//! tassement depuis déformation    s = εv · H
//! ```
//!
//! `s` tassement de la couche (m), `Cc` = `compression_index` indice de
//! compression (sans dimension), `Cr` = `recompression_index` indice de
//! recompression (sans dimension), `H` = `layer_thickness` épaisseur de la couche
//! compressible (m), `e0` = `initial_void_ratio` indice des vides initial (sans
//! dimension), `σ0` = `initial_stress` contrainte effective verticale initiale
//! (Pa), `Δσ` = `stress_increment` incrément de contrainte effective (Pa), `σf` =
//! `final_stress` contrainte effective finale (Pa), `cv` =
//! `coefficient_consolidation` coefficient de consolidation (m²/s), `t` = `time`
//! temps écoulé (s), `Hdr` = `drainage_path` longueur du chemin de drainage (m),
//! `Tv` facteur temps (sans dimension), `εv` = `vertical_strain` déformation
//! verticale (sans dimension).
//!
//! **Convention** : SI strict — **m, s, Pa** (avec `1 Pa = 1 N/m²`). Les
//! tassements et longueurs ressortent en **mètres**, les contraintes en
//! **pascals**, le temps en **secondes**, le coefficient de consolidation en
//! **mètres carrés par seconde** ; les indices (`Cc`, `Cr`, `e0`, `εv`) et le
//! facteur temps `Tv` sont **sans dimension**. Le logarithme est **décimal**
//! (base 10), conformément à la définition usuelle de `Cc` et `Cr`.
//!
//! **Limite honnête** : consolidation **unidimensionnelle** (essai œdométrique,
//! théorie de Terzaghi), sol **saturé** et **homogène**, drainage vertical. Les
//! indices de compression et de recompression `Cc`/`Cr`, l'indice des vides
//! initial `e0`, le coefficient de consolidation `cv` et les contraintes
//! effectives (`σ0`, `Δσ`, `σf`) sont **fournis par l'appelant** d'après les
//! **essais** (œdomètre) ; aucune valeur « par défaut » n'est inventée. Les
//! éventuelles résistances caractéristiques du sol **et** les coefficients
//! partiels de sécurité (`γc`, `γs`, `γM`…) relèvent de l'appelant selon
//! l'Eurocode 7 et son Annexe Nationale. Ce module **ne calcule ni** le tassement
//! immédiat (élastique), **ni** le fluage (consolidation secondaire), **ni** le
//! degré de consolidation `U(Tv)` (série de Terzaghi non couverte ici).

/// Tassement de consolidation d'une argile **normalement consolidée**
/// `s = (Cc·H / (1 + e0)) · log10((σ0 + Δσ) / σ0)` (m), avec `H` en m et `σ0`,
/// `Δσ` en Pa ; le logarithme est **décimal**.
///
/// Panique si `compression_index < 0`, si `layer_thickness <= 0`, si
/// `initial_void_ratio <= 0`, si `initial_stress <= 0` ou si
/// `stress_increment < 0`.
pub fn settle_normally_consolidated(
    compression_index: f64,
    layer_thickness: f64,
    initial_void_ratio: f64,
    initial_stress: f64,
    stress_increment: f64,
) -> f64 {
    assert!(
        compression_index >= 0.0,
        "l'indice de compression Cc doit être ≥ 0"
    );
    assert!(
        layer_thickness > 0.0,
        "l'épaisseur de couche H doit être strictement positive"
    );
    assert!(
        initial_void_ratio > 0.0,
        "l'indice des vides initial e0 doit être strictement positif"
    );
    assert!(
        initial_stress > 0.0,
        "la contrainte initiale σ0 doit être strictement positive"
    );
    assert!(
        stress_increment >= 0.0,
        "l'incrément de contrainte Δσ doit être ≥ 0"
    );
    (compression_index * layer_thickness / (1.0 + initial_void_ratio))
        * ((initial_stress + stress_increment) / initial_stress).log10()
}

/// Tassement de **recompression** d'une argile surconsolidée sous la pression de
/// préconsolidation `s = (Cr·H / (1 + e0)) · log10(σf / σ0)` (m), avec `H` en m
/// et `σ0`, `σf` en Pa ; le logarithme est **décimal**.
///
/// Panique si `recompression_index < 0`, si `layer_thickness <= 0`, si
/// `initial_void_ratio <= 0`, si `initial_stress <= 0` ou si `final_stress <= 0`.
pub fn settle_overconsolidated_recompression(
    recompression_index: f64,
    layer_thickness: f64,
    initial_void_ratio: f64,
    initial_stress: f64,
    final_stress: f64,
) -> f64 {
    assert!(
        recompression_index >= 0.0,
        "l'indice de recompression Cr doit être ≥ 0"
    );
    assert!(
        layer_thickness > 0.0,
        "l'épaisseur de couche H doit être strictement positive"
    );
    assert!(
        initial_void_ratio > 0.0,
        "l'indice des vides initial e0 doit être strictement positif"
    );
    assert!(
        initial_stress > 0.0,
        "la contrainte initiale σ0 doit être strictement positive"
    );
    assert!(
        final_stress > 0.0,
        "la contrainte finale σf doit être strictement positive"
    );
    (recompression_index * layer_thickness / (1.0 + initial_void_ratio))
        * (final_stress / initial_stress).log10()
}

/// Facteur temps de la théorie de Terzaghi `Tv = cv · t / Hdr²` (sans dimension),
/// avec `cv` en m²/s, `t` en s et `Hdr` en m.
///
/// Panique si `coefficient_consolidation < 0`, si `time < 0` ou si
/// `drainage_path <= 0` (division par zéro).
pub fn settle_time_factor(coefficient_consolidation: f64, time: f64, drainage_path: f64) -> f64 {
    assert!(
        coefficient_consolidation >= 0.0,
        "le coefficient de consolidation cv doit être ≥ 0"
    );
    assert!(time >= 0.0, "le temps t doit être ≥ 0");
    assert!(
        drainage_path > 0.0,
        "le chemin de drainage Hdr doit être strictement positif"
    );
    coefficient_consolidation * time / (drainage_path * drainage_path)
}

/// Tassement déduit d'une déformation verticale mesurée `s = εv · H` (m), avec
/// `H` en m et `εv` sans dimension.
///
/// Panique si `vertical_strain < 0` ou si `layer_thickness <= 0`.
pub fn settle_final_from_strain(vertical_strain: f64, layer_thickness: f64) -> f64 {
    assert!(
        vertical_strain >= 0.0,
        "la déformation verticale εv doit être ≥ 0"
    );
    assert!(
        layer_thickness > 0.0,
        "l'épaisseur de couche H doit être strictement positive"
    );
    vertical_strain * layer_thickness
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn normally_consolidated_matches_hand_computation() {
        // Cas chiffré : Cc = 0,30, H = 4 m, e0 = 0,90, σ0 = 100 kPa,
        // Δσ = 100 kPa (donc σ0 + Δσ = 200 kPa, rapport = 2).
        //   coefficient = 0,30·4 / 1,90 = 1,2 / 1,9 = 0,631578947368
        //   s = 0,631578947368 · log10(2) = 0,631578947368 · 0,301029995664
        //     ≈ 0,190124208 m
        let s = settle_normally_consolidated(0.30, 4.0, 0.90, 100_000.0, 100_000.0);
        assert_relative_eq!(s, 0.190_124_208, max_relative = 1e-6);
    }

    #[test]
    fn nc_settlement_is_proportional_to_thickness() {
        // Le tassement est linéaire en H : doubler l'épaisseur double le tassement.
        let base = settle_normally_consolidated(0.25, 3.0, 0.80, 80_000.0, 60_000.0);
        let double = settle_normally_consolidated(0.25, 6.0, 0.80, 80_000.0, 60_000.0);
        assert_relative_eq!(double, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn recompression_equals_nc_formula_with_same_index() {
        // Identité : les deux formules ne diffèrent que par l'argument du log.
        // Avec Cr = Cc et σf = σ0 + Δσ, la recompression égale le cas NC.
        let (index, h, e0, sigma0, dsigma) =
            (0.06_f64, 4.0_f64, 0.90_f64, 120_000.0_f64, 90_000.0_f64);
        let nc = settle_normally_consolidated(index, h, e0, sigma0, dsigma);
        let oc = settle_overconsolidated_recompression(index, h, e0, sigma0, sigma0 + dsigma);
        assert_relative_eq!(oc, nc, max_relative = 1e-12);
    }

    #[test]
    fn no_stress_increase_gives_zero_settlement() {
        // Cas limite : sans surcharge (Δσ = 0), log10(1) = 0 ⇒ tassement nul.
        // De même σf = σ0 annule la recompression.
        let nc = settle_normally_consolidated(0.30, 4.0, 0.90, 150_000.0, 0.0);
        let oc = settle_overconsolidated_recompression(0.05, 4.0, 0.90, 150_000.0, 150_000.0);
        assert_relative_eq!(nc, 0.0, epsilon = 1e-12);
        assert_relative_eq!(oc, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn time_factor_scales_with_time_and_inverse_square_of_drainage() {
        // Tv est linéaire en t et varie en 1/Hdr² : cv = 3e-8 m²/s, t = 1 an
        // (31 536 000 s), Hdr = 2 m (double drainage d'une couche de 4 m).
        //   Tv = 3e-8 · 31 536 000 / 2² = 0,94608 / 4 = 0,23652
        let cv = 3.0e-8_f64;
        let one_year = 31_536_000.0_f64;
        let tv = settle_time_factor(cv, one_year, 2.0);
        assert_relative_eq!(tv, 0.236_52, max_relative = 1e-9);
        // Deux fois le temps ⇒ deux fois Tv.
        let tv2 = settle_time_factor(cv, 2.0 * one_year, 2.0);
        assert_relative_eq!(tv2, 2.0 * tv, max_relative = 1e-12);
        // Chemin de drainage divisé par 2 ⇒ Tv multiplié par 4.
        let tv_half = settle_time_factor(cv, one_year, 1.0);
        assert_relative_eq!(tv_half, 4.0 * tv, max_relative = 1e-12);
    }

    #[test]
    fn strain_to_settlement_is_linear() {
        // s = εv·H : produit direct, vérifié sur un cas simple.
        let s = settle_final_from_strain(0.05, 4.0);
        assert_relative_eq!(s, 0.20, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la contrainte initiale σ0 doit être strictement positive")]
    fn nc_rejects_zero_initial_stress() {
        // σ0 = 0 interdit : division par zéro et log non défini.
        settle_normally_consolidated(0.30, 4.0, 0.90, 0.0, 100_000.0);
    }
}
