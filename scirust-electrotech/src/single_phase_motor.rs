//! **Moteur asynchrone monophasé** — module de la **théorie du double champ
//! tournant** : le champ pulsant de l'enroulement unique se décompose en un
//! champ **direct** et un champ **inverse** de même amplitude tournant en sens
//! opposés. On y calcule les glissements vus par chacun de ces deux champs, la
//! vitesse de synchronisme, et la capacité de l'enroulement de démarrage.
//!
//! ```text
//! glissement direct   s_f = (n_s − n_r) / n_s
//! glissement inverse  s_b = 2 − s_f
//! vitesse synchrone   n_s = 60·f / p      (tr/min)
//! capacité démarrage  C   = Q_start / (2·π·f·V²)
//! ```
//!
//! `n_s` vitesse de synchronisme (même unité que `n_r`, tr/min ou rad/s), `n_r`
//! vitesse du rotor, `s_f` glissement du champ **direct** (sans dimension),
//! `s_b` glissement du champ **inverse** (sans dimension), `f` fréquence
//! d'alimentation (Hz), `p` nombre de **paires de pôles** (sans dimension),
//! `V` tension aux bornes de l'enroulement (V), `Q_start` puissance réactive de
//! démarrage à fournir par la branche capacitive (var), `C` capacité de
//! démarrage (F).
//!
//! **Convention** : SI ; tensions en V, puissance réactive en var, fréquences
//! en Hz, capacités en F, vitesses en tr/min (ou rad/s, pourvu que `n_s` et
//! `n_r` partagent la même unité) ; les glissements et le nombre de paires de
//! pôles sont sans dimension. **Limite honnête** : moteur **monophasé** analysé
//! par la **théorie du double champ tournant** (champs direct et inverse). Sans
//! dispositif auxiliaire, les couples des deux champs s'annulent à l'arrêt : le
//! **couple de démarrage est nul** et le champ inverse produit en marche un
//! **couple de freinage**. Les vitesses `n_s`/`n_r` (d'où les glissements), la
//! tension `V`, la fréquence `f`, le nombre de paires de pôles `p` et la
//! puissance réactive de démarrage `Q_start` sont **FOURNIS par l'appelant**
//! (plaque signalétique, essais, cahier des charges). Ce module ne modélise
//! **pas** le schéma équivalent complet (impédances des champs direct/inverse,
//! réactance magnétisante) ni le régime transitoire du démarrage.

use core::f64::consts::PI;

/// Glissement du **champ direct** `s_f = (n_s − n_r) / n_s` (sans dimension).
///
/// À l'arrêt (`n_r = 0`) vaut 1 ; au synchronisme (`n_r = n_s`) vaut 0.
///
/// Panique si `synchronous_speed <= 0` (division par zéro exclue) ou si
/// `rotor_speed < 0`.
pub fn spmot_forward_slip(synchronous_speed: f64, rotor_speed: f64) -> f64 {
    assert!(
        synchronous_speed > 0.0,
        "la vitesse de synchronisme n_s doit être strictement positive"
    );
    assert!(rotor_speed >= 0.0, "la vitesse du rotor n_r doit être ≥ 0");
    (synchronous_speed - rotor_speed) / synchronous_speed
}

/// Glissement du **champ inverse** `s_b = 2 − s_f` (sans dimension).
///
/// Le champ inverse tourne en sens opposé : le rotor le « voit » avec un
/// glissement complémentaire à 2. À l'arrêt (`s_f = 1`) les deux champs voient
/// le même glissement `s_b = 1`.
///
/// Panique si `forward_slip < 0` ou `forward_slip > 2` (hors plage physique).
pub fn spmot_backward_slip(forward_slip: f64) -> f64 {
    assert!(
        (0.0..=2.0).contains(&forward_slip),
        "le glissement direct s_f doit être compris entre 0 et 2"
    );
    2.0 - forward_slip
}

/// Vitesse de synchronisme `n_s = 60·f / p` (tr/min).
///
/// Panique si `frequency <= 0` ou `pole_pairs <= 0` (division par zéro exclue).
pub fn spmot_synchronous_speed_rpm(frequency: f64, pole_pairs: f64) -> f64 {
    assert!(
        frequency > 0.0,
        "la fréquence f doit être strictement positive"
    );
    assert!(
        pole_pairs > 0.0,
        "le nombre de paires de pôles p doit être strictement positif"
    );
    60.0 * frequency / pole_pairs
}

/// Capacité de démarrage `C = Q_start / (2·π·f·V²)` (F).
///
/// Capacité en série avec l'enroulement auxiliaire qui, sous la tension `V` à
/// la fréquence `f`, absorbe la puissance réactive `Q_start` créant le
/// déphasage nécessaire au couple de démarrage.
///
/// Panique si `phase_voltage <= 0`, `starting_reactive_power < 0` ou
/// `frequency <= 0` (division par zéro exclue).
pub fn spmot_starting_capacitance(
    phase_voltage: f64,
    starting_reactive_power: f64,
    frequency: f64,
) -> f64 {
    assert!(
        phase_voltage > 0.0,
        "la tension V doit être strictement positive"
    );
    assert!(
        starting_reactive_power >= 0.0,
        "la puissance réactive de démarrage Q_start doit être ≥ 0"
    );
    assert!(
        frequency > 0.0,
        "la fréquence f doit être strictement positive"
    );
    starting_reactive_power / (2.0 * PI * frequency * phase_voltage * phase_voltage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn forward_and_backward_slip_sum_to_two() {
        // Identité de la théorie du double champ tournant : quels que soient
        // n_s et n_r, s_f + s_b = 2 (les deux champs sont symétriques).
        let s_f = spmot_forward_slip(1500.0, 1440.0);
        let s_b = spmot_backward_slip(s_f);
        assert_relative_eq!(s_f + s_b, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn slip_is_unity_at_standstill() {
        // Cas limite : rotor à l'arrêt (n_r = 0) ⇒ s_f = 1, et le champ inverse
        // voit alors le même glissement s_b = 1.
        let s_f = spmot_forward_slip(1500.0, 0.0);
        assert_relative_eq!(s_f, 1.0, epsilon = 1e-15);
        assert_relative_eq!(spmot_backward_slip(s_f), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn slip_is_zero_at_synchronism() {
        // Cas limite : au synchronisme (n_r = n_s) le glissement direct est nul.
        assert_relative_eq!(spmot_forward_slip(3000.0, 3000.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn synchronous_speed_scales_inversely_with_pole_pairs() {
        // Proportionnalité : n_s ∝ 1/p. Doubler p divise n_s par deux.
        // f = 50 Hz : p = 1 ⇒ 3000 tr/min ; p = 2 ⇒ 1500 tr/min.
        assert_relative_eq!(
            spmot_synchronous_speed_rpm(50.0, 1.0),
            3000.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            spmot_synchronous_speed_rpm(50.0, 2.0),
            1500.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            2.0 * spmot_synchronous_speed_rpm(50.0, 2.0),
            spmot_synchronous_speed_rpm(50.0, 1.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn starting_capacitance_matches_hand_calc() {
        // Cas chiffré : V = 230 V, f = 50 Hz, Q_start = 500 var.
        //   dénominateur = 2·π·f·V² = 2·π·50·230²
        //                = 314,159 265 358 979…·52 900 = 16 619 025,137… var/F
        //   C = 500 / 16 619 025,137… ≈ 3,0086 × 10⁻⁵ F ≈ 30 µF
        let c = spmot_starting_capacitance(230.0, 500.0, 50.0);
        // Littéral recalculé indépendamment (deux fois) : 500 / (2π·50·230²).
        assert_relative_eq!(
            c,
            500.0 / (2.0 * PI * 50.0 * 230.0 * 230.0),
            epsilon = 1e-15
        );
        assert_relative_eq!(c, 3.008_60e-5, epsilon = 1e-3);
    }

    #[test]
    fn capacitance_reciprocity_with_reactive_power() {
        // Réciprocité : la capacité calculée pour Q_start restitue exactement
        // Q_start via Q = 2·π·f·V²·C. Aller-retour sans perte d'information.
        let v = 400.0;
        let f = 60.0;
        let q = 1200.0;
        let c = spmot_starting_capacitance(v, q, f);
        let q_back = 2.0 * PI * f * v * v * c;
        assert_relative_eq!(q_back, q, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le nombre de paires de pôles p doit être strictement positif")]
    fn zero_pole_pairs_panics() {
        spmot_synchronous_speed_rpm(50.0, 0.0);
    }
}
