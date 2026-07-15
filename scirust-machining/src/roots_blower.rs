//! **Surpresseur à lobes (Roots)** — machine à déplacement positif à compression
//! **externe** : débit balayé théorique, débit réel après fuites de retour,
//! rendement volumétrique et puissance approchée de refoulement.
//!
//! ```text
//! débit théorique     Q_th = D·N / 60
//! débit réel          Q    = Q_th − Q_slip
//! rendement volum.    η_v  = Q / Q_th
//! puissance           P    = Δp·Q_th
//! ```
//!
//! `D` cylindrée (déplacement) par tour (m³·tour⁻¹), `N` vitesse de rotation
//! (tr·min⁻¹), `Q_th` débit balayé théorique (m³·s⁻¹), `Q_slip` fuites de retour
//! internes ramenées en débit (m³·s⁻¹), `Q` débit réel refoulé (m³·s⁻¹), `η_v`
//! rendement volumétrique (sans dimension), `Δp` élévation de pression entre
//! aspiration et refoulement (Pa), `P` puissance de compression (W = Pa·m³·s⁻¹).
//!
//! **Convention** : unités SI, débits volumiques aux conditions d'aspiration.
//!
//! **Limite honnête** : surpresseur à **lobes** à déplacement positif, compression
//! **externe** à volume balayé constant (le gaz n'est pas comprimé dans la machine,
//! il l'est par refoulement contre la pression aval). La cylindrée par tour et les
//! fuites (`slip`) sont des **données de machine/procédé fournies par l'appelant** —
//! aucune valeur « par défaut » n'est inventée ici. La puissance `Δp·Q_th` est une
//! **approximation** (la compression réelle n'est pas isentropique ; échauffement,
//! pertes mécaniques et fuites non pris en compte) et l'étage à lobes ne délivre
//! qu'un **faible taux de compression**. Complète [`crate::compressed_air`].

/// Débit balayé **théorique** d'un surpresseur à lobes `Q_th = D·N / 60`.
///
/// `displacement_per_revolution` cylindrée par tour (m³·tour⁻¹),
/// `rotational_speed_rpm` vitesse (tr·min⁻¹) ; renvoie le débit en m³·s⁻¹.
///
/// Panique si un paramètre est `<= 0`.
pub fn roots_theoretical_flow(displacement_per_revolution: f64, rotational_speed_rpm: f64) -> f64 {
    assert!(
        displacement_per_revolution > 0.0 && rotational_speed_rpm > 0.0,
        "cylindrée par tour et vitesse de rotation strictement positives requises"
    );
    displacement_per_revolution * rotational_speed_rpm / 60.0
}

/// Débit **réel** refoulé après déduction des fuites de retour internes
/// `Q = Q_th − Q_slip`.
///
/// `theoretical_flow` débit balayé théorique (m³·s⁻¹), `slip_flow` fuites de
/// retour ramenées en débit (m³·s⁻¹) ; renvoie le débit réel en m³·s⁻¹.
///
/// Panique si `theoretical_flow <= 0`, si `slip_flow < 0` ou si les fuites
/// dépassent le débit théorique (`slip_flow > theoretical_flow`).
pub fn roots_actual_flow(theoretical_flow: f64, slip_flow: f64) -> f64 {
    assert!(
        theoretical_flow > 0.0,
        "débit théorique strictement positif requis"
    );
    assert!(
        slip_flow >= 0.0 && slip_flow <= theoretical_flow,
        "fuites comprises entre 0 et le débit théorique requises"
    );
    theoretical_flow - slip_flow
}

/// Rendement **volumétrique** `η_v = Q / Q_th` (sans dimension, dans `]0 ; 1]`).
///
/// `actual_flow` débit réel refoulé (m³·s⁻¹), `theoretical_flow` débit balayé
/// théorique (m³·s⁻¹) ; renvoie le rapport sans dimension.
///
/// Panique si `theoretical_flow <= 0`, si `actual_flow < 0` ou si
/// `actual_flow > theoretical_flow`.
pub fn roots_volumetric_efficiency(actual_flow: f64, theoretical_flow: f64) -> f64 {
    assert!(
        theoretical_flow > 0.0,
        "débit théorique strictement positif requis"
    );
    assert!(
        actual_flow >= 0.0 && actual_flow <= theoretical_flow,
        "débit réel compris entre 0 et le débit théorique requis"
    );
    actual_flow / theoretical_flow
}

/// Puissance de compression **approchée** `P = Δp·Q_th` (W).
///
/// `pressure_rise` élévation de pression aspiration → refoulement (Pa),
/// `theoretical_flow` débit balayé théorique (m³·s⁻¹) ; renvoie la puissance en W.
/// Approximation à volume balayé constant, hors rendements mécaniques et thermique.
///
/// Panique si un paramètre est `<= 0`.
pub fn roots_power(pressure_rise: f64, theoretical_flow: f64) -> f64 {
    assert!(
        pressure_rise > 0.0 && theoretical_flow > 0.0,
        "élévation de pression et débit théorique strictement positifs requis"
    );
    pressure_rise * theoretical_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn theoretical_flow_hand_calc() {
        // D = 0,01 m³/tour, N = 3000 tr/min :
        // Q_th = 0,01·3000/60 = 0,5 m³/s.
        let q = roots_theoretical_flow(0.01, 3000.0);
        assert_relative_eq!(q, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn theoretical_flow_proportional_to_speed() {
        // Q_th ∝ N à cylindrée fixée : doubler la vitesse double le débit.
        let q1 = roots_theoretical_flow(0.008, 1500.0);
        let q2 = roots_theoretical_flow(0.008, 3000.0);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn slip_efficiency_round_trip() {
        // Identité : η_v = (Q_th − Q_slip)/Q_th, donc Q_slip = Q_th·(1 − η_v).
        let (q_th, slip) = (0.5_f64, 0.05);
        let q = roots_actual_flow(q_th, slip);
        let eta = roots_volumetric_efficiency(q, q_th);
        assert_relative_eq!(eta, 0.9, epsilon = 1e-12);
        assert_relative_eq!(q_th * (1.0 - eta), slip, epsilon = 1e-12);
    }

    #[test]
    fn zero_slip_gives_unit_efficiency() {
        // Sans fuites, le débit réel égale le théorique et η_v = 1.
        let q_th = roots_theoretical_flow(0.01, 3000.0);
        let q = roots_actual_flow(q_th, 0.0);
        assert_relative_eq!(q, q_th, epsilon = 1e-12);
        assert_relative_eq!(roots_volumetric_efficiency(q, q_th), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn power_hand_calc() {
        // Δp = 0,5 bar = 5e4 Pa, Q_th = 0,5 m³/s :
        // P = 5e4·0,5 = 2,5e4 W = 25 kW.
        let p = roots_power(5e4, 0.5);
        assert_relative_eq!(p, 25_000.0, epsilon = 1e-9);
    }

    #[test]
    fn power_proportional_to_pressure_rise() {
        // P ∝ Δp à débit fixé.
        let p1 = roots_power(3e4, 0.4);
        let p2 = roots_power(6e4, 0.4);
        assert_relative_eq!(p2, 2.0 * p1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "fuites comprises entre 0 et le débit théorique")]
    fn actual_flow_rejects_excessive_slip() {
        let _ = roots_actual_flow(0.5, 0.6);
    }
}
