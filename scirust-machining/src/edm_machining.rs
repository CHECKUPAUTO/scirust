//! Électroérosion (EDM) — **enlèvement de matière** : débit d'enlèvement
//! proportionnel au courant d'usinage, rapport d'usure électrode/pièce et
//! surdimensionnement (surcoupe) du trou dû à l'entrefer d'étincelage.
//!
//! ```text
//! débit d'enlèvement   MRR = k · I           (loi empirique linéaire)
//! rapport d'usure      w   = Vt / Vw
//! surcoupe             oc  = 2 · gap
//! ```
//!
//! `I` courant moyen d'usinage (A), `k` coefficient d'enlèvement du couple
//! électrode/diélectrique (cm³/(min·A)), `MRR` débit volumique d'enlèvement de
//! matière (cm³/min), `Vt` volume usé de l'électrode-outil (m³ ou cm³, unité au
//! choix mais commune avec `Vw`), `Vw` volume enlevé sur la pièce (même unité
//! que `Vt`), `w` rapport d'usure (sans dimension), `gap` entrefer d'étincelage
//! de part et d'autre de l'électrode (m), `oc` surcoupe diamétrale, c'est-à-dire
//! l'excès de diamètre du trou par rapport au diamètre de l'électrode (m).
//!
//! **Convention** : SI cohérent, sauf `MRR`/`k` en unités atelier usuelles
//! (cm³/min, cm³/(min·A)) comme dans le brief. **Limite honnête** : le débit
//! d'enlèvement est modélisé **linéaire en courant**, une corrélation empirique
//! dont le coefficient `k` dépend du couple électrode/diélectrique et est
//! **fourni par l'appelant** ; de même l'entrefer `gap` résulte des réglages du
//! procédé (tension, diélectrique) et n'est jamais inventé ici. Aucune valeur de
//! matériau, de procédé ou de coefficient n'a de « valeur par défaut ».

/// Débit d'enlèvement de matière `MRR = k · I` (cm³/min).
///
/// Corrélation empirique linéaire : le volume enlevé par unité de temps croît
/// proportionnellement au courant moyen d'usinage, le coefficient `k` étant
/// caractéristique du couple électrode/diélectrique.
///
/// Panique si `current < 0` ou `mrr_coefficient < 0`.
pub fn edm_material_removal_rate(current: f64, mrr_coefficient: f64) -> f64 {
    assert!(current >= 0.0, "le courant d'usinage doit être positif");
    assert!(
        mrr_coefficient >= 0.0,
        "le coefficient d'enlèvement doit être positif"
    );
    mrr_coefficient * current
}

/// Rapport d'usure électrode/pièce `w = Vt / Vw` (sans dimension).
///
/// Fraction du volume enlevé sur la pièce qui se retrouve consommée sous forme
/// d'usure de l'électrode-outil ; un rapport faible traduit une électrode peu
/// érodable.
///
/// Panique si `tool_wear_volume < 0` ou `workpiece_removed_volume <= 0`.
pub fn edm_electrode_wear_ratio(tool_wear_volume: f64, workpiece_removed_volume: f64) -> f64 {
    assert!(
        tool_wear_volume >= 0.0,
        "le volume usé de l'électrode doit être positif"
    );
    assert!(
        workpiece_removed_volume > 0.0,
        "le volume enlevé sur la pièce doit être strictement positif"
    );
    tool_wear_volume / workpiece_removed_volume
}

/// Surcoupe diamétrale `oc = 2 · gap` (m).
///
/// Excès de diamètre du trou par rapport au diamètre de l'électrode : l'étincelle
/// franchit l'entrefer des deux côtés, d'où un facteur deux.
///
/// Panique si `spark_gap < 0`.
pub fn edm_overcut(spark_gap: f64) -> f64 {
    assert!(
        spark_gap >= 0.0,
        "l'entrefer d'étincelage doit être positif"
    );
    2.0_f64 * spark_gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mrr_realistic_value() {
        // Ébauche acier, k = 0,04 cm³/(min·A), I = 20 A :
        // MRR = 0,04 · 20 = 0,8 cm³/min.
        let mrr = edm_material_removal_rate(20.0, 0.04);
        assert_relative_eq!(mrr, 0.8, epsilon = 1e-12);
    }

    #[test]
    fn mrr_is_linear_in_current() {
        // MRR ∝ I : doubler le courant double le débit.
        let base = edm_material_removal_rate(15.0, 0.04);
        let double_i = edm_material_removal_rate(30.0, 0.04);
        assert_relative_eq!(double_i / base, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn mrr_zero_current_gives_zero() {
        // Cas limite : sans courant, aucun enlèvement.
        assert_relative_eq!(edm_material_removal_rate(0.0, 0.04), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn wear_ratio_realistic_value() {
        // Électrode cuivre, Vt = 3 cm³ usés pour Vw = 50 cm³ enlevés :
        // w = 3/50 = 0,06 (soit 6 % d'usure relative).
        let w = edm_electrode_wear_ratio(3.0, 50.0);
        assert_relative_eq!(w, 0.06, epsilon = 1e-12);
    }

    #[test]
    fn wear_ratio_inverts_volume_definition() {
        // Réciprocité : w · Vw = Vt reconstruit le volume d'usure.
        let vt = 2.5_f64;
        let vw = 40.0_f64;
        let w = edm_electrode_wear_ratio(vt, vw);
        assert_relative_eq!(w * vw, vt, epsilon = 1e-12);
    }

    #[test]
    fn overcut_is_twice_the_gap() {
        // oc = 2·gap : un entrefer de 25 µm donne 50 µm de surcoupe.
        let gap = 25e-6_f64;
        let oc = edm_overcut(gap);
        assert_relative_eq!(oc, 50e-6, epsilon = 1e-18);
        assert_relative_eq!(oc / gap, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "volume enlevé sur la pièce")]
    fn zero_workpiece_volume_panics() {
        edm_electrode_wear_ratio(3.0, 0.0);
    }
}
