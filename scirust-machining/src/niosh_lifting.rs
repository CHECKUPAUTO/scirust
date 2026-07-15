//! **Équation NIOSH révisée (1991)** — limite de poids recommandée et indice
//! de levage pour la manutention manuelle de charges.
//!
//! ```text
//! poids recommandé   RWL = LC·HM·VM·DM·AM·FM·CM
//! indice de levage   LI  = charge/RWL              (LI > 1 ⇒ risque)
//! multiplicateurs
//!   horizontal       HM = 25/H                     (H ≥ 25 cm ; 25/H sinon plafonné)
//!   vertical         VM = 1 − 0,003·|V − 75|
//!   distance         DM = 0,82 + 4,5/D
//! ```
//!
//! `LC` constante de charge (23 kg, [`LOAD_CONSTANT`]), `RWL` limite de poids
//! recommandée (kg), `LI` indice de levage (sans dimension), `H` distance
//! horizontale main–chevilles (cm), `V` hauteur verticale de prise au sol (cm),
//! `D` distance verticale de déplacement de la charge (cm), `HM`/`VM`/`DM`/`AM`/
//! `FM`/`CM` multiplicateurs (sans dimension, ∈ [0, 1]).
//!
//! **Convention** : distances en cm, poids en kg (unités d'origine de l'équation
//! NIOSH). **Limite honnête** : le multiplicateur d'asymétrie `AM = 1 − 0,0032·A`,
//! le multiplicateur de fréquence `FM` et le multiplicateur de prise `CM` (qualité
//! de la préhension) proviennent des **tables NIOSH fournies par l'appelant**
//! (aucune valeur inventée ici) ; seules les corrélations continues `HM`, `VM`,
//! `DM` et `AM` sont calculées. Modèle empirique valable dans son domaine de
//! validité (25 ≤ H, 0 ≤ V ≤ 175 cm, 25 ≤ D ≤ 175 cm).

/// Constante de charge de l'équation NIOSH révisée : `LC = 23 kg`.
pub const LOAD_CONSTANT: f64 = 23.0;

/// Multiplicateur horizontal `HM = 25/H` (`H` en cm, plafonné à 1 pour `H ≤ 25`).
///
/// Panique si `h_cm <= 0`.
pub fn niosh_horizontal_multiplier(h_cm: f64) -> f64 {
    assert!(h_cm > 0.0, "H > 0 cm requis");
    if h_cm <= 25.0 { 1.0 } else { 25.0 / h_cm }
}

/// Multiplicateur vertical `VM = 1 − 0,003·|V − 75|` (`V` en cm, optimum à 75 cm).
///
/// Panique si `v_cm < 0`.
pub fn niosh_vertical_multiplier(v_cm: f64) -> f64 {
    assert!(v_cm >= 0.0, "V ≥ 0 cm requis");
    1.0 - 0.003 * (v_cm - 75.0).abs()
}

/// Multiplicateur de distance verticale `DM = 0,82 + 4,5/D` (`D` en cm).
///
/// Panique si `d_cm <= 0`.
pub fn niosh_distance_multiplier(d_cm: f64) -> f64 {
    assert!(d_cm > 0.0, "D > 0 cm requis");
    0.82 + 4.5 / d_cm
}

/// Multiplicateur d'asymétrie `AM = 1 − 0,0032·A` (`A` angle de torsion en degrés).
///
/// Panique si `asymmetry_deg < 0`.
pub fn niosh_asymmetry_multiplier(asymmetry_deg: f64) -> f64 {
    assert!(asymmetry_deg >= 0.0, "A ≥ 0° requis");
    1.0 - 0.0032 * asymmetry_deg
}

/// Limite de poids recommandée `RWL = LC·HM·VM·DM·AM·FM·CM` (kg).
///
/// `fm` (fréquence) et `cm` (prise) sont issus des tables NIOSH fournies par
/// l'appelant ; les autres multiplicateurs proviennent des corrélations continues.
///
/// Panique si un multiplicateur n'est pas dans `[0, 1]`.
pub fn lifting_recommended_weight_limit(
    hm: f64,
    vm: f64,
    dm: f64,
    am: f64,
    fm: f64,
    cm: f64,
) -> f64 {
    for (name, m) in [
        ("HM", hm),
        ("VM", vm),
        ("DM", dm),
        ("AM", am),
        ("FM", fm),
        ("CM", cm),
    ]
    {
        assert!(
            (0.0..=1.0).contains(&m),
            "multiplicateur {name} ∈ [0, 1] requis"
        );
    }
    LOAD_CONSTANT * hm * vm * dm * am * fm * cm
}

/// Indice de levage `LI = charge/RWL` (`LI > 1` signale un risque lombaire).
///
/// Panique si `load < 0` ou `rwl <= 0`.
pub fn lifting_index(load: f64, rwl: f64) -> f64 {
    assert!(load >= 0.0, "charge ≥ 0 kg requise");
    assert!(rwl > 0.0, "RWL > 0 kg requis");
    load / rwl
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn horizontal_multiplier_reciprocity() {
        // À H=25 cm, HM=25/25=1 (limite haute) ; à H=50, HM=0,5.
        assert_relative_eq!(niosh_horizontal_multiplier(25.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(niosh_horizontal_multiplier(50.0), 0.5, epsilon = 1e-12);
        // Proportionnalité inverse : doubler H halve HM.
        assert_relative_eq!(
            niosh_horizontal_multiplier(80.0),
            0.5 * niosh_horizontal_multiplier(40.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn vertical_multiplier_optimum_at_75() {
        // Optimum : VM=1 exactement à V=75 cm, symétrique autour.
        assert_relative_eq!(niosh_vertical_multiplier(75.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(
            niosh_vertical_multiplier(75.0 - 20.0),
            niosh_vertical_multiplier(75.0 + 20.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn distance_multiplier_known_value() {
        // D=25 cm : DM = 0,82 + 4,5/25 = 0,82 + 0,18 = 1,0.
        assert_relative_eq!(niosh_distance_multiplier(25.0), 1.0, epsilon = 1e-12);
        // Décroît quand D augmente.
        assert!(niosh_distance_multiplier(175.0) < niosh_distance_multiplier(25.0));
    }

    #[test]
    fn ideal_conditions_give_load_constant() {
        // Tous les multiplicateurs = 1 ⇒ RWL = LC = 23 kg.
        let rwl = lifting_recommended_weight_limit(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_relative_eq!(rwl, LOAD_CONSTANT, epsilon = 1e-12);
    }

    #[test]
    fn realistic_case_and_index() {
        // Cas chiffré : H=50 (HM=0,5), V=75 (VM=1), D=25 (DM=1),
        // A=0° (AM=1), FM=0,88, CM=0,95 (tables appelant).
        let hm = niosh_horizontal_multiplier(50.0);
        let vm = niosh_vertical_multiplier(75.0);
        let dm = niosh_distance_multiplier(25.0);
        let am = niosh_asymmetry_multiplier(0.0);
        let rwl = lifting_recommended_weight_limit(hm, vm, dm, am, 0.88, 0.95);
        // RWL = 23·0,5·1·1·1·0,88·0,95 = 9,614 kg.
        assert_relative_eq!(rwl, 23.0 * 0.5 * 0.88 * 0.95, epsilon = 1e-9);
        // Une charge égale à RWL donne LI = 1 (seuil de risque).
        assert_relative_eq!(lifting_index(rwl, rwl), 1.0, epsilon = 1e-12);
        // Charge double ⇒ LI = 2 (proportionnalité).
        assert_relative_eq!(lifting_index(2.0 * rwl, rwl), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn asymmetry_reduces_limit() {
        // AM strictement décroissant avec l'angle de torsion.
        assert_relative_eq!(niosh_asymmetry_multiplier(0.0), 1.0, epsilon = 1e-12);
        assert!(niosh_asymmetry_multiplier(90.0) < niosh_asymmetry_multiplier(0.0));
    }

    #[test]
    #[should_panic(expected = "RWL > 0 kg requis")]
    fn zero_rwl_panics() {
        lifting_index(10.0, 0.0);
    }
}
