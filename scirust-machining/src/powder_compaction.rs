//! Métallurgie des poudres — compaction et frittage : densité relative du
//! comprimé cru, rapport de pression de compaction, retrait linéaire et
//! volumique au frittage (retrait isotrope), dimension finale frittée.
//!
//! Un comprimé « cru » (green compact) de masse volumique `ρ_green` est comparé
//! à la masse volumique théorique du plein `ρ_th` du matériau. Au frittage, la
//! densification fait passer la masse volumique de `ρ_green` à `ρ_sint` ; à masse
//! constante, la contraction volumique se répartit de façon **isotrope**, d'où
//! un retrait linéaire identique dans les trois directions :
//!
//! ```text
//! D_green = ρ_green / ρ_th                          (densité relative crue, [-])
//! P_r     = P_app / P_ref                           (rapport de pression, [-])
//! s_lin   = 1 − (ρ_green / ρ_sint)^(1/3)            (retrait linéaire, [-])
//! s_vol   = 1 − (1 − s_lin)^3                       (retrait volumique, [-])
//! L_sint  = L_green · (1 − s_lin)                   (dimension frittée, m)
//! ϕ_green = 1 − D_green                             (porosité crue, [-])
//! ```
//!
//! Légende (unités SI cohérentes) :
//! - `ρ_green`, `ρ_sint`, `ρ_th` : masses volumiques crue, frittée, théorique
//!   (kg/m³), avec `0 < ρ_green ≤ ρ_sint ≤ ρ_th`.
//! - `D_green` : densité relative du comprimé cru (sans dimension, dans `]0, 1]`).
//! - `ϕ_green` : porosité fractionnaire crue (sans dimension, dans `[0, 1[`).
//! - `P_app`, `P_ref` : pression de compaction appliquée et pression de
//!   référence (Pa), `P_ref > 0`.
//! - `s_lin`, `s_vol` : retraits linéaire et volumique au frittage (fractions,
//!   dans `[0, 1[`).
//! - `L_green`, `L_sint` : dimension d'une cote avant/après frittage (m).
//!
//! **Limite honnête** : le retrait au frittage est supposé **isotrope**
//! (contraction identique dans les trois directions, sans gradient de densité ni
//! gauchissement). La relation empirique pression → densité (Heckel, Kawakita,
//! etc.) n'est **pas** modélisée : seul un rapport de pressions adimensionnel
//! est fourni. Toutes les masses volumiques (crue, frittée, théorique) et les
//! pressions sont **fournies par l'appelant** ; ce module n'invente aucune
//! constante matériau « par défaut ».

/// Densité relative du comprimé cru `D_green = ρ_green / ρ_th` (sans dimension),
/// à partir de la masse volumique crue `green_density` et de la masse volumique
/// théorique du plein `theoretical_density` (mêmes unités, kg/m³).
///
/// Panique si `theoretical_density <= 0`, si `green_density <= 0`, ou si
/// `green_density > theoretical_density`.
pub fn pm_green_density_ratio(green_density: f64, theoretical_density: f64) -> f64 {
    assert!(
        green_density > 0.0,
        "la masse volumique crue doit être strictement positive"
    );
    assert!(
        theoretical_density > 0.0,
        "la masse volumique théorique doit être strictement positive"
    );
    assert!(
        green_density <= theoretical_density,
        "la masse volumique crue ne peut pas dépasser la masse volumique théorique"
    );
    green_density / theoretical_density
}

/// Porosité fractionnaire du comprimé cru `ϕ_green = 1 − D_green` (sans
/// dimension), complément de la densité relative `green_density_ratio`.
///
/// Panique si `green_density_ratio` n'est pas dans `]0, 1]`.
pub fn pm_green_porosity(green_density_ratio: f64) -> f64 {
    assert!(
        green_density_ratio > 0.0 && green_density_ratio <= 1.0,
        "la densité relative crue doit être dans ]0, 1]"
    );
    1.0 - green_density_ratio
}

/// Rapport de pression de compaction `P_r = P_app / P_ref` (sans dimension),
/// forme simplifiée normalisant la pression appliquée `applied_pressure` par une
/// pression de référence `reference_pressure` (mêmes unités, Pa).
///
/// Panique si `reference_pressure <= 0` ou si `applied_pressure < 0`.
pub fn pm_compaction_pressure_ratio(applied_pressure: f64, reference_pressure: f64) -> f64 {
    assert!(
        applied_pressure >= 0.0,
        "la pression appliquée doit être positive ou nulle"
    );
    assert!(
        reference_pressure > 0.0,
        "la pression de référence doit être strictement positive"
    );
    applied_pressure / reference_pressure
}

/// Retrait linéaire isotrope au frittage `s_lin = 1 − (ρ_green / ρ_sint)^(1/3)`
/// (fraction sans dimension), calculé à partir des densités relatives crue
/// `green_density_ratio` et frittée `sintered_density_ratio` (toutes deux
/// rapportées à la même masse volumique théorique, donc leur quotient vaut
/// `ρ_green / ρ_sint`).
///
/// Panique si l'une des densités relatives n'est pas dans `]0, 1]`, ou si
/// `sintered_density_ratio < green_density_ratio` (le frittage densifie).
pub fn pm_sintering_linear_shrinkage(green_density_ratio: f64, sintered_density_ratio: f64) -> f64 {
    assert!(
        green_density_ratio > 0.0 && green_density_ratio <= 1.0,
        "la densité relative crue doit être dans ]0, 1]"
    );
    assert!(
        sintered_density_ratio > 0.0 && sintered_density_ratio <= 1.0,
        "la densité relative frittée doit être dans ]0, 1]"
    );
    assert!(
        sintered_density_ratio >= green_density_ratio,
        "la densité frittée doit être supérieure ou égale à la densité crue"
    );
    1.0 - (green_density_ratio / sintered_density_ratio).cbrt()
}

/// Retrait volumique au frittage `s_vol = 1 − (1 − s_lin)^3` (fraction sans
/// dimension) déduit du retrait linéaire isotrope `linear_shrinkage`.
///
/// Panique si `linear_shrinkage` n'est pas dans `[0, 1[`.
pub fn pm_sintering_volume_shrinkage(linear_shrinkage: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&linear_shrinkage),
        "le retrait linéaire doit être dans [0, 1["
    );
    1.0 - (1.0 - linear_shrinkage).powi(3)
}

/// Dimension frittée `L_sint = L_green · (1 − s_lin)` (m) d'une cote crue
/// `green_dimension` soumise au retrait linéaire isotrope `linear_shrinkage`.
///
/// Panique si `green_dimension < 0` ou si `linear_shrinkage` n'est pas dans
/// `[0, 1[`.
pub fn pm_sintered_dimension(green_dimension: f64, linear_shrinkage: f64) -> f64 {
    assert!(
        green_dimension >= 0.0,
        "la dimension crue doit être positive ou nulle"
    );
    assert!(
        (0.0..1.0).contains(&linear_shrinkage),
        "le retrait linéaire doit être dans [0, 1["
    );
    green_dimension * (1.0 - linear_shrinkage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn green_density_ratio_matches_definition() {
        // Fer fritté typique : ρ_green = 6800 kg/m³, ρ_th = 7800 kg/m³.
        assert_relative_eq!(
            pm_green_density_ratio(6800.0, 7800.0),
            6800.0 / 7800.0,
            epsilon = 1e-12
        );
        // Densité relative et porosité sont complémentaires (somme = 1).
        let d = pm_green_density_ratio(6800.0, 7800.0);
        assert_relative_eq!(d + pm_green_porosity(d), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn pressure_ratio_is_a_plain_quotient() {
        // P_app = 600 MPa, P_ref = 300 MPa → rapport = 2.
        assert_relative_eq!(
            pm_compaction_pressure_ratio(600.0e6, 300.0e6),
            2.0,
            epsilon = 1e-12
        );
        // À la pression de référence, le rapport vaut exactement 1.
        assert_relative_eq!(
            pm_compaction_pressure_ratio(300.0e6, 300.0e6),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn linear_shrinkage_perfect_cube_case() {
        // Cas chiffré choisi comme cube parfait : ρ_green/ρ_sint = 0,729 = 0,9³.
        // ⇒ (0,729)^(1/3) = 0,9 exactement ⇒ s_lin = 1 − 0,9 = 0,1.
        let s = pm_sintering_linear_shrinkage(0.729, 1.0);
        assert_relative_eq!(s, 0.1, epsilon = 1e-12);
    }

    #[test]
    fn no_densification_gives_zero_shrinkage() {
        // ρ_green = ρ_sint ⇒ quotient = 1 ⇒ retrait linéaire et volumique nuls.
        let s = pm_sintering_linear_shrinkage(0.85, 0.85);
        assert_relative_eq!(s, 0.0, epsilon = 1e-12);
        assert_relative_eq!(pm_sintering_volume_shrinkage(s), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn volume_shrinkage_equals_density_deficit() {
        // Identité : s_vol = 1 − (1 − s_lin)³ = 1 − ρ_green/ρ_sint.
        // Avec ρ_green/ρ_sint = 0,729, s_vol doit valoir 1 − 0,729 = 0,271.
        let s_lin = pm_sintering_linear_shrinkage(0.729, 1.0);
        let s_vol = pm_sintering_volume_shrinkage(s_lin);
        assert_relative_eq!(s_vol, 1.0 - 0.729, epsilon = 1e-12);
        assert_relative_eq!(s_vol, 0.271, epsilon = 1e-12);
    }

    #[test]
    fn sintered_dimension_applies_isotropic_shrinkage() {
        // Cote crue 100 mm, retrait linéaire 0,1 ⇒ cote frittée 90 mm.
        assert_relative_eq!(pm_sintered_dimension(0.100, 0.1), 0.090, epsilon = 1e-12);
        // Cohérence volumique : (L_sint/L_green)³ = 1 − s_vol = ρ_green/ρ_sint.
        let ratio = pm_sintered_dimension(0.100, 0.1) / 0.100;
        assert_relative_eq!(ratio.powi(3), 0.729, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "masse volumique théorique")]
    fn zero_theoretical_density_panics() {
        pm_green_density_ratio(6800.0, 0.0);
    }
}
