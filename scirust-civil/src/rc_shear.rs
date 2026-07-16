//! **Béton armé — effort tranchant (Eurocode 2, ELU, méthode des bielles)** :
//! résistance de l'âme sans armatures d'effort tranchant, coefficient d'échelle,
//! résistance apportée par les étriers (modèle du treillis à angle de bielle
//! variable) et résistance limitée par l'écrasement des bielles de béton.
//!
//! ```text
//! sans armatures            VRd,c   = CRd,c · k · (100 · ρl · fck)^(1/3) · bw · d
//! coefficient d'échelle     k       = 1 + √(200 / d)   (plafonné à 2)
//! avec étriers (treillis)   VRd,s   = (Asw/s) · z · fywd · cot θ
//! écrasement des bielles     VRd,max = αcw · bw · z · ν1 · fcd / (cot θ + tan θ)
//! ```
//!
//! `VRd,c` effort tranchant résistant sans armatures (N), `CRd,c` coefficient de
//! l'EC2 (sans dimension, usuellement `γc`-dépendant), `k` coefficient d'échelle
//! (sans dimension), `ρl` ratio d'armatures longitudinales tendues (sans
//! dimension, `= Asl / (bw·d)`), `fck` résistance caractéristique en compression
//! du béton (MPa), `bw` largeur d'âme (mm), `d` hauteur utile (mm), `VRd,s`
//! effort tranchant repris par les étriers (N), `Asw/s` section d'un cours
//! d'étriers par unité de longueur (mm²/mm), `z` bras de levier des forces
//! internes (mm), `fywd` limite d'élasticité de calcul des étriers (MPa),
//! `θ` angle des bielles de béton (rad, `cot θ = 1/tan θ`), `VRd,max` effort
//! tranchant limité par l'écrasement des bielles (N), `αcw` coefficient tenant
//! compte de l'état de contrainte dans la membrure comprimée (sans dimension),
//! `ν1` coefficient de réduction de la résistance du béton fissuré à l'effort
//! tranchant (sans dimension), `fcd` résistance de calcul en compression du
//! béton (MPa).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`), donc les résistances
//! ressortent en **newtons** ; angles en **radians** pour la trigonométrie.
//! **Limite honnête** : modèle du **treillis à angle de bielle variable** de
//! l'EC2 (usuellement `1 ≤ cot θ ≤ 2,5`, soit `θ ∈ [21,8° ; 45°]`, ce que
//! l'appelant doit respecter) ; les résistances caractéristiques (`fck`, `fyk`,
//! `fy`…) **et** les coefficients partiels de sécurité (`γc`, `γs`, `γM`…) — donc
//! `fcd`, `fywd` — ainsi que les coefficients réglementaires `CRd,c`, `ν1` et
//! `αcw`, le ratio d'armatures `ρl` et la densité d'étriers `Asw/s` sont
//! **fournis par l'appelant** d'après l'Eurocode 2 et son Annexe Nationale ;
//! aucune valeur « par défaut » n'est inventée. Le coefficient d'échelle `k` est
//! **plafonné à 2,0**. La vérification finale (`VEd ≤ min(VRd,s, VRd,max)`, choix
//! de `θ`, dispositions constructives) reste à la charge de l'ingénieur.

use core::f64::consts::FRAC_PI_2;

/// Effort tranchant résistant sans armatures d'effort tranchant
/// `VRd,c = CRd,c · k · (100 · ρl · fck)^(1/3) · bw · d` (N), avec `fck` en MPa
/// et `bw`, `d` en mm.
///
/// Panique si `crd_c <= 0`, si `k_factor` n'est pas dans `[1, 2]`, si
/// `rho_l < 0`, si `fck <= 0`, si `width <= 0` ou si `effective_depth <= 0`.
pub fn rcshear_resistance_without_reinforcement(
    crd_c: f64,
    k_factor: f64,
    rho_l: f64,
    fck: f64,
    width: f64,
    effective_depth: f64,
) -> f64 {
    assert!(
        crd_c > 0.0,
        "le coefficient CRd,c doit être strictement positif"
    );
    assert!(
        (1.0..=2.0).contains(&k_factor),
        "le coefficient d'échelle k doit être dans [1, 2]"
    );
    assert!(
        rho_l >= 0.0,
        "le ratio d'armatures longitudinales ρl doit être ≥ 0"
    );
    assert!(
        fck > 0.0,
        "la résistance fck doit être strictement positive"
    );
    assert!(
        width > 0.0,
        "la largeur d'âme bw doit être strictement positive"
    );
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    crd_c * k_factor * (100.0 * rho_l * fck).cbrt() * width * effective_depth
}

/// Coefficient d'échelle `k = 1 + √(200 / d)`, **plafonné à 2,0** (sans
/// dimension), avec la hauteur utile `d` en mm.
///
/// Panique si `effective_depth <= 0` (division par zéro et racine indéfinie).
pub fn rcshear_size_factor(effective_depth: f64) -> f64 {
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    (1.0 + (200.0 / effective_depth).sqrt()).min(2.0)
}

/// Effort tranchant repris par les étriers `VRd,s = (Asw/s) · z · fywd · cot θ`
/// (N), modèle du treillis à angle de bielle `θ` (rad), avec `Asw/s` en mm²/mm,
/// `z` en mm et `fywd` en MPa.
///
/// Panique si `asw_over_s < 0`, si `lever_arm <= 0`, si `fywd < 0` ou si
/// `strut_angle_rad` n'est pas dans `]0, π/2[` (bornes où `tan θ` s'annule ou
/// diverge).
pub fn rcshear_resistance_with_stirrups(
    asw_over_s: f64,
    lever_arm: f64,
    fywd: f64,
    strut_angle_rad: f64,
) -> f64 {
    assert!(
        asw_over_s >= 0.0,
        "la densité d'étriers Asw/s doit être ≥ 0"
    );
    assert!(
        lever_arm > 0.0,
        "le bras de levier z doit être strictement positif"
    );
    assert!(fywd >= 0.0, "la limite d'élasticité fywd doit être ≥ 0");
    assert!(
        strut_angle_rad > 0.0 && strut_angle_rad < FRAC_PI_2,
        "l'angle de bielle θ doit être dans ]0, π/2["
    );
    asw_over_s * lever_arm * fywd / strut_angle_rad.tan()
}

/// Effort tranchant limité par l'écrasement des bielles
/// `VRd,max = αcw · bw · z · ν1 · fcd / (cot θ + tan θ)` (N), avec `bw`, `z` en
/// mm et `fcd` en MPa.
///
/// Panique si `alpha_cw <= 0`, si `width <= 0`, si `lever_arm <= 0`, si
/// `nu1 <= 0`, si `fcd <= 0` ou si `strut_angle_rad` n'est pas dans `]0, π/2[`.
pub fn rcshear_max_resistance(
    alpha_cw: f64,
    width: f64,
    lever_arm: f64,
    nu1: f64,
    fcd: f64,
    strut_angle_rad: f64,
) -> f64 {
    assert!(
        alpha_cw > 0.0,
        "le coefficient αcw doit être strictement positif"
    );
    assert!(
        width > 0.0,
        "la largeur d'âme bw doit être strictement positive"
    );
    assert!(
        lever_arm > 0.0,
        "le bras de levier z doit être strictement positif"
    );
    assert!(nu1 > 0.0, "le coefficient ν1 doit être strictement positif");
    assert!(
        fcd > 0.0,
        "la résistance de calcul fcd doit être strictement positive"
    );
    assert!(
        strut_angle_rad > 0.0 && strut_angle_rad < FRAC_PI_2,
        "l'angle de bielle θ doit être dans ]0, π/2["
    );
    let tan_theta = strut_angle_rad.tan();
    alpha_cw * width * lever_arm * nu1 * fcd / (tan_theta + 1.0 / tan_theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::FRAC_PI_4;

    #[test]
    fn size_factor_hits_two_and_is_capped() {
        // À d = 200 mm : k = 1 + √(200/200) = 1 + 1 = 2, valeur exacte du plafond.
        assert_relative_eq!(rcshear_size_factor(200.0), 2.0, epsilon = 1e-12);
        // À d = 100 mm : 1 + √2 ≈ 2,414 dépasse le plafond → ramené à 2,0.
        assert_relative_eq!(rcshear_size_factor(100.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn size_factor_decreases_with_depth() {
        // Décroissance : plus la hauteur utile est grande, plus k tend vers 1.
        //   d = 450 mm → 1 + √(200/450) ≈ 1,6667
        //   d = 800 mm → 1 + √(200/800) = 1,5
        let k450 = rcshear_size_factor(450.0);
        let k800 = rcshear_size_factor(800.0);
        assert!(k800 < k450);
        assert_relative_eq!(k800, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn resistance_without_reinforcement_clean_case() {
        // Cas chiffré choisi pour un cube parfait :
        //   100 · ρl · fck = 100 · 0,01 · 27 = 27  →  (27)^(1/3) = 3
        //   VRd,c = CRd,c · k · 3 · bw · d
        //         = 0,12 · 2,0 · 3 · 300 · 200 = 43 200 N
        let v = rcshear_resistance_without_reinforcement(0.12, 2.0, 0.01, 27.0, 300.0, 200.0);
        assert_relative_eq!(v, 43_200.0, epsilon = 1e-6);
    }

    #[test]
    fn resistance_without_reinforcement_scales_with_section() {
        // Proportionnalité : VRd,c est linéaire en bw et en d ; doubler la
        // largeur d'âme double la résistance.
        let v1 = rcshear_resistance_without_reinforcement(0.12, 1.8, 0.008, 30.0, 250.0, 400.0);
        let v2 = rcshear_resistance_without_reinforcement(0.12, 1.8, 0.008, 30.0, 500.0, 400.0);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn stirrups_clean_case_and_proportionality() {
        // Cas chiffré à θ = 45° (cot θ = 1) :
        //   VRd,s = (Asw/s) · z · fywd · cot θ = 0,5 · 400 · 435 · 1 = 87 000 N
        let v = rcshear_resistance_with_stirrups(0.5, 400.0, 435.0, FRAC_PI_4);
        assert_relative_eq!(v, 87_000.0, epsilon = 1e-3);
        // Proportionnalité : doubler Asw/s double VRd,s.
        let v2 = rcshear_resistance_with_stirrups(1.0, 400.0, 435.0, FRAC_PI_4);
        assert_relative_eq!(v2 / v, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn max_resistance_clean_case_and_symmetry() {
        // Cas chiffré à θ = 45° (cot θ + tan θ = 2) :
        //   VRd,max = αcw · bw · z · ν1 · fcd / 2
        //          = 1,0 · 300 · 400 · 0,6 · 20 / 2 = 720 000 N
        let v = rcshear_max_resistance(1.0, 300.0, 400.0, 0.6, 20.0, FRAC_PI_4);
        assert_relative_eq!(v, 720_000.0, epsilon = 1e-3);
        // Symétrie : le dénominateur cot θ + tan θ est invariant par θ ↔ π/2 − θ,
        // donc VRd,max(θ) = VRd,max(π/2 − θ). On compare θ = atan(0,4) (cot θ =
        // 2,5) et son complément θ' = atan(2,5) (cot θ' = 0,4).
        let theta = (0.4_f64).atan();
        let theta_c = (2.5_f64).atan();
        let a = rcshear_max_resistance(1.0, 300.0, 400.0, 0.6, 20.0, theta);
        let b = rcshear_max_resistance(1.0, 300.0, 400.0, 0.6, 20.0, theta_c);
        assert_relative_eq!(a, b, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "l'angle de bielle θ doit être dans ]0, π/2[")]
    fn stirrups_reject_null_angle() {
        // tan(0) = 0 : division par zéro interdite.
        rcshear_resistance_with_stirrups(0.5, 400.0, 435.0, 0.0);
    }
}
