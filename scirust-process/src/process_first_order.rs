//! Dynamique de procédé du **premier ordre** et **réglage de régulateur** —
//! réponse indicielle d'un système du premier ordre, modèle du premier ordre à
//! retard pur (FOPDT) et réglage empirique PI par la méthode de la réponse
//! indicielle de Ziegler-Nichols (boucle ouverte).
//!
//! ```text
//! réponse indicielle 1er ordre  y(t) = K·A·(1 − exp(−t/τ))       [unité de y]
//! temps pour une fraction       t    = −τ·ln(1 − f)               [s]
//! gain proportionnel PI (Z-N)   Kc   = 0.9·τ / (Kp·θ)             [%CO/%VP]
//! temps intégral PI (Z-N)       Ti   = 3.33·θ                     [s]
//! réponse FOPDT                 y(t) = 0                si t < θ
//!                               y(t) = K·A·(1 − exp(−(t−θ)/τ))    si t ≥ θ
//! ```
//!
//! `y(t)` écart de la sortie par rapport à son état initial [unité de la variable
//! réglée], `K` gain statique du procédé [unité de y par unité d'échelon], `A`
//! amplitude de l'échelon appliqué à l'entrée [unité de l'entrée], `t` temps
//! écoulé depuis l'application de l'échelon [s], `τ` constante de temps [s], `f`
//! fraction de la valeur finale visée [sans dimension, dans `[0, 1[`] (`f = 0.632`
//! à `t = τ`), `Kp` gain statique identifié du procédé, `θ` retard pur (temps
//! mort) [s], `Kc` gain proportionnel du régulateur, `Ti` temps intégral [s].
//!
//! **Limite honnête** : le gain statique `Kp`/`K`, la constante de temps `τ` et le
//! retard pur `θ` sont des **paramètres identifiés sur la réponse mesurée du
//! procédé** (méthode de la tangente, du point d'inflexion, moindres carrés…) :
//! ils sont **fournis par l'appelant** et jamais inventés. Le modèle est
//! **linéaire du premier ordre (+ retard)**, valable au voisinage d'un point de
//! fonctionnement ; les coefficients de Ziegler-Nichols (`0.9`, `3.33`) sont des
//! **règles empiriques** de la réponse indicielle qui donnent un **point de départ
//! de réglage à affiner** (elles visent une décroissance en quart d'amplitude,
//! souvent trop agressive). Ces fonctions ne modélisent ni les non-linéarités, ni
//! les ordres supérieurs, ni la stabilité en boucle fermée.

/// Réponse indicielle d'un système du **premier ordre**
/// `y(t) = K·A·(1 − exp(−t/τ))` [unité de la variable réglée].
///
/// `gain` `K` gain statique du procédé, `time_constant` `τ` constante de temps
/// [s], `time` `t` temps depuis l'application de l'échelon [s], `step_amplitude`
/// `A` amplitude de l'échelon d'entrée. À `t = τ`, la sortie atteint `≈ 63.2 %`
/// de sa valeur finale `K·A`.
///
/// Panique si `gain`, `time` ou `step_amplitude` n'est pas fini, si `time` est
/// négatif, ou si `time_constant` n'est pas strictement positif (division).
pub fn ford_step_response(gain: f64, time_constant: f64, time: f64, step_amplitude: f64) -> f64 {
    assert!(gain.is_finite(), "le gain statique doit être fini");
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive (s)"
    );
    assert!(
        time.is_finite() && time >= 0.0,
        "le temps doit être fini et positif ou nul (s)"
    );
    assert!(
        step_amplitude.is_finite(),
        "l'amplitude de l'échelon doit être finie"
    );
    let ratio: f64 = -time / time_constant;
    gain * step_amplitude * (1.0 - ratio.exp())
}

/// Temps nécessaire pour atteindre une **fraction** `f` de la valeur finale d'une
/// réponse du premier ordre `t = −τ·ln(1 − f)` [s] — réciproque de
/// [`ford_step_response`] au niveau de la fraction.
///
/// `time_constant` `τ` constante de temps [s], `fraction` `f` fraction de la
/// valeur finale visée [sans dimension, dans `[0, 1[`]. Cas usuels : `f = 0.632`
/// donne `t = τ`, `f = 0.95` donne `t ≈ 3·τ`, `f = 0.99` donne `t ≈ 4.6·τ`.
///
/// Panique si `time_constant` n'est pas strictement positif, ou si `fraction`
/// sort de `[0, 1[` (la borne 1 fait diverger le logarithme).
pub fn ford_time_to_fraction(time_constant: f64, fraction: f64) -> f64 {
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive (s)"
    );
    assert!(
        fraction.is_finite() && (0.0..1.0).contains(&fraction),
        "la fraction doit être comprise dans [0, 1[ (strictement inférieure à 1)"
    );
    -time_constant * (1.0 - fraction).ln()
}

/// Gain proportionnel d'un régulateur **PI** par la méthode de la réponse
/// indicielle de Ziegler-Nichols `Kc = 0.9·τ / (Kp·θ)` [gain sans dimension].
///
/// `process_gain` `Kp` gain statique identifié du procédé, `dead_time` `θ` retard
/// pur (temps mort) [s], `time_constant` `τ` constante de temps identifiée [s].
/// Le rapport `τ/θ` (contrôlabilité) gouverne l'agressivité du réglage.
///
/// Panique si `process_gain` n'est pas strictement positif, si `dead_time` n'est
/// pas strictement positif (division), ou si `time_constant` est négatif ou non
/// fini.
pub fn ford_zn_pi_gain(process_gain: f64, dead_time: f64, time_constant: f64) -> f64 {
    assert!(
        process_gain > 0.0,
        "le gain statique du procédé doit être strictement positif"
    );
    assert!(
        dead_time > 0.0,
        "le retard pur doit être strictement positif (s)"
    );
    assert!(
        time_constant.is_finite() && time_constant >= 0.0,
        "la constante de temps doit être finie et positive ou nulle (s)"
    );
    0.9 * time_constant / (process_gain * dead_time)
}

/// Temps intégral d'un régulateur **PI** par Ziegler-Nichols (réponse indicielle)
/// `Ti = 3.33·θ` [s].
///
/// `dead_time` `θ` retard pur (temps mort) identifié [s]. Le temps intégral
/// s'exprime uniquement à partir du retard pur dans cette règle empirique.
///
/// Panique si `dead_time` est négatif ou non fini.
pub fn ford_zn_pi_integral_time(dead_time: f64) -> f64 {
    assert!(
        dead_time.is_finite() && dead_time >= 0.0,
        "le retard pur doit être fini et positif ou nul (s)"
    );
    3.33 * dead_time
}

/// Réponse indicielle d'un modèle du **premier ordre à retard pur** (FOPDT)
/// `y(t) = 0` si `t < θ`, sinon `y(t) = K·A·(1 − exp(−(t − θ)/τ))`
/// [unité de la variable réglée].
///
/// `gain` `K` gain statique, `time_constant` `τ` constante de temps [s],
/// `dead_time` `θ` retard pur [s], `time` `t` temps depuis l'échelon [s],
/// `step_amplitude` `A` amplitude de l'échelon d'entrée. Tant que `t < θ`, la
/// sortie n'a pas encore réagi (retard de transport / temps mort).
///
/// Panique si `gain`, `time` ou `step_amplitude` n'est pas fini, si `time` ou
/// `dead_time` est négatif, ou si `time_constant` n'est pas strictement positif.
pub fn ford_first_order_plus_deadtime(
    gain: f64,
    time_constant: f64,
    dead_time: f64,
    time: f64,
    step_amplitude: f64,
) -> f64 {
    assert!(gain.is_finite(), "le gain statique doit être fini");
    assert!(
        time_constant > 0.0,
        "la constante de temps doit être strictement positive (s)"
    );
    assert!(
        dead_time.is_finite() && dead_time >= 0.0,
        "le retard pur doit être fini et positif ou nul (s)"
    );
    assert!(
        time.is_finite() && time >= 0.0,
        "le temps doit être fini et positif ou nul (s)"
    );
    assert!(
        step_amplitude.is_finite(),
        "l'amplitude de l'échelon doit être finie"
    );
    if time < dead_time
    {
        0.0
    }
    else
    {
        gain * step_amplitude * (1.0 - (-(time - dead_time) / time_constant).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::E;

    /// À `t = τ`, la réponse indicielle atteint exactement `1 − 1/e ≈ 63.2 %` de
    /// la valeur finale `K·A`. Cas chiffré : `K = 2`, `A = 5`, `τ = 10`, `t = 10`.
    /// Valeur finale `K·A = 10` ; `10·(1 − 1/e) = 10·0.6321205588 = 6.321205588`.
    /// (Recalcul indépendant : `1/e = 0.3678794412`, `1 − 0.3678794412 =
    /// 0.6321205588`, `× 10 = 6.321205588`.)
    #[test]
    fn step_response_atteint_63pct_a_tau() {
        let y = ford_step_response(2.0, 10.0, 10.0, 5.0);
        assert_relative_eq!(y, 6.321205588, epsilon = 1e-6);
        // Fraction exacte 1 − 1/e.
        assert_relative_eq!(y / (2.0 * 5.0), 1.0 - 1.0 / E, epsilon = 1e-9);
    }

    /// Réciprocité fraction ↔ temps : le temps rendu par [`ford_time_to_fraction`]
    /// pour une fraction `f` réinjecté dans [`ford_step_response`] redonne `f` de
    /// la valeur finale.
    #[test]
    fn reciprocite_fraction_temps() {
        let tau = 7.0;
        let f = 0.8;
        let t = ford_time_to_fraction(tau, f);
        let y = ford_step_response(1.0, tau, t, 1.0);
        assert_relative_eq!(y, f, epsilon = 1e-9);
    }

    /// `t = τ` correspond à la fraction `1 − 1/e`, et `t = 3τ` dépasse 95 %.
    #[test]
    fn time_to_fraction_points_reperes() {
        let tau = 4.0;
        assert_relative_eq!(
            ford_time_to_fraction(tau, 1.0 - 1.0 / E),
            tau,
            epsilon = 1e-9
        );
        // Repère classique : atteindre 95 % demande ≈ 3·τ (exactement −τ·ln 0,05 ≈ 2,996·τ).
        let t95 = ford_time_to_fraction(tau, 0.95);
        assert!(t95 > 2.9 * tau && t95 < 3.0 * tau);
    }

    /// Réglage PI de Ziegler-Nichols. Cas chiffré : `Kp = 2`, `θ = 3`, `τ = 30`.
    /// `Kc = 0.9·30/(2·3) = 27/6 = 4.5` ; `Ti = 3.33·3 = 9.99`.
    /// (Recalcul indépendant : `0.9·30 = 27`, `2·3 = 6`, `27/6 = 4.5` ;
    /// `3.33·3 = 9.99`.)
    #[test]
    fn reglage_zn_pi_cas_chiffre() {
        assert_relative_eq!(ford_zn_pi_gain(2.0, 3.0, 30.0), 4.5, epsilon = 1e-12);
        assert_relative_eq!(ford_zn_pi_integral_time(3.0), 9.99, epsilon = 1e-12);
    }

    /// Le gain PI est proportionnel à `τ` et inversement proportionnel à `Kp·θ` :
    /// doubler `τ` double `Kc`, doubler `θ` le divise par deux.
    #[test]
    fn zn_pi_gain_proportionnalites() {
        let base = ford_zn_pi_gain(2.0, 3.0, 30.0);
        assert_relative_eq!(ford_zn_pi_gain(2.0, 3.0, 60.0), 2.0 * base, epsilon = 1e-12);
        assert_relative_eq!(ford_zn_pi_gain(2.0, 6.0, 30.0), 0.5 * base, epsilon = 1e-12);
    }

    /// FOPDT : sortie nulle avant le retard, puis réponse identique à un premier
    /// ordre décalé de `θ`. Cas chiffré : `K = 2`, `τ = 10`, `θ = 3`, `A = 5`,
    /// `t = 13` ⇒ `t − θ = 10 = τ` ⇒ `y = 10·(1 − 1/e) = 6.321205588`.
    #[test]
    fn fopdt_retard_puis_premier_ordre() {
        assert_relative_eq!(
            ford_first_order_plus_deadtime(2.0, 10.0, 3.0, 2.9, 5.0),
            0.0,
            epsilon = 1e-12
        );
        let y = ford_first_order_plus_deadtime(2.0, 10.0, 3.0, 13.0, 5.0);
        assert_relative_eq!(y, 6.321205588, epsilon = 1e-6);
        // Équivalence au premier ordre décalé du retard.
        assert_relative_eq!(y, ford_step_response(2.0, 10.0, 10.0, 5.0), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le retard pur doit être strictement positif")]
    fn zn_pi_gain_retard_nul_panique() {
        let _ = ford_zn_pi_gain(2.0, 0.0, 30.0);
    }
}
