//! **Géotechnique — écoulement dans les sols (loi de Darcy, réseau d'écoulement)** :
//! vitesse de Darcy en régime **permanent** saturé, débit de fuite estimé par un
//! **réseau d'écoulement** (rapport nombre de canaux `Nf` / nombre de pertes de
//! potentiel `Nd`), gradient hydraulique **critique** de boulance et facteur de
//! sécurité au **renard** (soulèvement hydraulique).
//!
//! ```text
//! vitesse de Darcy       v = k · i
//! débit (réseau)         Q = k · H · (Nf / Nd) · largeur
//! gradient critique      icr = γ' / γw = (γsat − γw) / γw
//! sécurité au renard     Fs = icr / iexit
//! ```
//!
//! `v` vitesse de Darcy (m/s), `k` = `hydraulic_conductivity` conductivité
//! hydraulique (m/s), `i` = `hydraulic_gradient` gradient hydraulique (sans
//! dimension), `Q` débit de fuite (m³/s), `H` = `head_loss` perte de charge totale
//! sous l'ouvrage (m), `Nf` = `flow_channels` nombre de canaux d'écoulement du
//! tracé (sans dimension), `Nd` = `potential_drops` nombre de pertes de potentiel
//! du tracé (sans dimension), `largeur` = `width` largeur de l'ouvrage
//! perpendiculaire à l'écoulement (m), `icr` gradient hydraulique critique (sans
//! dimension), `γ'` poids volumique déjaugé (N/m³), `γsat` =
//! `saturated_unit_weight` poids volumique saturé du sol (N/m³), `γw` =
//! `water_unit_weight` poids volumique de l'eau (N/m³), `iexit` = `exit_gradient`
//! gradient hydraulique de sortie (sans dimension), `Fs` facteur de sécurité (sans
//! dimension).
//!
//! **Convention** : SI strict — **m, s, N** (avec `1 Pa = 1 N/m²`,
//! `1 N/m³` pour les poids volumiques). Les vitesses ressortent en **m/s**, les
//! débits en **m³/s**, les longueurs (perte de charge, largeur) en **mètres**, les
//! poids volumiques en **N/m³** ; les gradients (`i`, `icr`, `iexit`), les nombres
//! de canaux/pertes (`Nf`, `Nd`) et le facteur de sécurité `Fs` sont **sans
//! dimension**. La conductivité hydraulique `k` s'exprime en **m/s**.
//!
//! **Limite honnête** : écoulement **permanent** (stationnaire) en milieu
//! **saturé**, sol **homogène** et **isotrope**, loi de Darcy **linéaire**
//! (régime laminaire). La conductivité hydraulique `k` est **fournie par
//! l'appelant** d'après les essais (perméamètre, essais de pompage) ; le réseau
//! d'écoulement — nombre de canaux `Nf` et nombre de pertes de potentiel `Nd` —
//! est **fourni par le tracé** graphique ou numérique, ce module ne le construit
//! pas. Le gradient critique `icr` marque le **début** de la boulance (renard
//! hydraulique) : il n'inclut aucun coefficient de sécurité. Les résistances
//! caractéristiques du sol **et** les coefficients partiels de sécurité (`γM`,
//! valeur cible de `Fs`…) relèvent de l'appelant selon l'**Eurocode 7** et son
//! **Annexe Nationale** ; aucune valeur « par défaut » n'est inventée. Ce module
//! **ne traite ni** l'écoulement transitoire, **ni** l'anisotropie, **ni** les
//! sols stratifiés.

/// Vitesse de Darcy en régime permanent saturé `v = k · i` (m/s), avec `k` en m/s
/// et `i` sans dimension.
///
/// Panique si `hydraulic_conductivity < 0` ou si `hydraulic_gradient < 0`.
pub fn seep_darcy_velocity(hydraulic_conductivity: f64, hydraulic_gradient: f64) -> f64 {
    assert!(
        hydraulic_conductivity >= 0.0,
        "la conductivité hydraulique k doit être ≥ 0"
    );
    assert!(
        hydraulic_gradient >= 0.0,
        "le gradient hydraulique i doit être ≥ 0"
    );
    hydraulic_conductivity * hydraulic_gradient
}

/// Débit de fuite estimé par un réseau d'écoulement
/// `Q = k · H · (Nf / Nd) · largeur` (m³/s), avec `k` en m/s, `H` et `largeur` en
/// m, `Nf` et `Nd` sans dimension.
///
/// Panique si `hydraulic_conductivity < 0`, si `head_loss < 0`, si
/// `flow_channels < 0`, si `potential_drops <= 0` (division par zéro) ou si
/// `width <= 0`.
pub fn seep_flow_rate(
    hydraulic_conductivity: f64,
    head_loss: f64,
    flow_channels: f64,
    potential_drops: f64,
    width: f64,
) -> f64 {
    assert!(
        hydraulic_conductivity >= 0.0,
        "la conductivité hydraulique k doit être ≥ 0"
    );
    assert!(head_loss >= 0.0, "la perte de charge H doit être ≥ 0");
    assert!(
        flow_channels >= 0.0,
        "le nombre de canaux d'écoulement Nf doit être ≥ 0"
    );
    assert!(
        potential_drops > 0.0,
        "le nombre de pertes de potentiel Nd doit être strictement positif"
    );
    assert!(width > 0.0, "la largeur doit être strictement positive");
    hydraulic_conductivity * head_loss * flow_channels * width / potential_drops
}

/// Gradient hydraulique critique de boulance
/// `icr = (γsat − γw) / γw` (sans dimension), avec `γsat` et `γw` en N/m³.
///
/// Panique si `water_unit_weight <= 0` (division par zéro) ou si
/// `saturated_unit_weight < water_unit_weight` (poids déjaugé négatif, non
/// physique).
pub fn seep_critical_gradient(saturated_unit_weight: f64, water_unit_weight: f64) -> f64 {
    assert!(
        water_unit_weight > 0.0,
        "le poids volumique de l'eau γw doit être strictement positif"
    );
    assert!(
        saturated_unit_weight >= water_unit_weight,
        "le poids volumique saturé γsat doit être ≥ γw (poids déjaugé ≥ 0)"
    );
    (saturated_unit_weight - water_unit_weight) / water_unit_weight
}

/// Facteur de sécurité au renard (soulèvement hydraulique)
/// `Fs = icr / iexit` (sans dimension), avec `icr` et `iexit` sans dimension.
///
/// Panique si `critical_gradient < 0` ou si `exit_gradient <= 0` (division par
/// zéro).
pub fn seep_factor_of_safety_piping(critical_gradient: f64, exit_gradient: f64) -> f64 {
    assert!(
        critical_gradient >= 0.0,
        "le gradient critique icr doit être ≥ 0"
    );
    assert!(
        exit_gradient > 0.0,
        "le gradient de sortie iexit doit être strictement positif"
    );
    critical_gradient / exit_gradient
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn darcy_velocity_is_linear_and_reciprocal() {
        // v = k·i : cas chiffré k = 1e-6 m/s, i = 0,25 ⇒ v = 2,5e-7 m/s.
        let v = seep_darcy_velocity(1.0e-6, 0.25);
        assert_relative_eq!(v, 2.5e-7, max_relative = 1e-12);
        // Réciprocité : on retrouve i = v / k.
        assert_relative_eq!(v / 1.0e-6, 0.25, max_relative = 1e-12);
        // Linéarité : doubler i double v.
        let v2 = seep_darcy_velocity(1.0e-6, 0.50);
        assert_relative_eq!(v2, 2.0 * v, max_relative = 1e-12);
    }

    #[test]
    fn flow_rate_matches_hand_computation() {
        // Cas chiffré (barrage) : k = 2e-6 m/s, H = 6 m, Nf = 4, Nd = 12,
        // largeur = 50 m.
        //   Q = 2e-6 · 6 · (4/12) · 50
        //     = 2e-6 · 6 · 0,333333... · 50
        //     = 2e-6 · 100 = 2e-4 m³/s
        // Détail : 6·50 = 300 ; 4/12 = 1/3 ; 300/3 = 100 ; ·2e-6 = 2e-4.
        let q = seep_flow_rate(2.0e-6, 6.0, 4.0, 12.0, 50.0);
        assert_relative_eq!(q, 2.0e-4, max_relative = 1e-9);
    }

    #[test]
    fn flow_rate_scales_with_conductivity_and_width() {
        // Q est linéaire en k et en largeur : doubler l'un double Q.
        let base = seep_flow_rate(1.0e-6, 5.0, 3.0, 10.0, 20.0);
        let double_k = seep_flow_rate(2.0e-6, 5.0, 3.0, 10.0, 20.0);
        let double_w = seep_flow_rate(1.0e-6, 5.0, 3.0, 10.0, 40.0);
        assert_relative_eq!(double_k, 2.0 * base, max_relative = 1e-12);
        assert_relative_eq!(double_w, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn critical_gradient_typical_value() {
        // Sable saturé usuel : γsat = 20 kN/m³, γw = 9,81 kN/m³.
        //   icr = (20000 − 9810) / 9810 = 10190 / 9810 ≈ 1,038735
        let icr = seep_critical_gradient(20_000.0, 9_810.0);
        assert_relative_eq!(icr, 1.038_735_0, max_relative = 1e-6);
        // Cas limite : γsat = γw ⇒ poids déjaugé nul ⇒ icr = 0.
        let icr0 = seep_critical_gradient(9_810.0, 9_810.0);
        assert_relative_eq!(icr0, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn safety_factor_equals_ratio_and_unity_at_onset() {
        // Fs = icr/iexit : icr = 1,04, iexit = 0,52 ⇒ Fs = 2,0.
        let fs = seep_factor_of_safety_piping(1.04, 0.52);
        assert_relative_eq!(fs, 2.0, max_relative = 1e-12);
        // Au début de la boulance (iexit = icr), Fs = 1 exactement.
        let fs_onset = seep_factor_of_safety_piping(1.04, 1.04);
        assert_relative_eq!(fs_onset, 1.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le poids volumique de l'eau γw doit être strictement positif")]
    fn critical_gradient_rejects_zero_water_weight() {
        // γw = 0 interdit : division par zéro.
        seep_critical_gradient(20_000.0, 0.0);
    }
}
