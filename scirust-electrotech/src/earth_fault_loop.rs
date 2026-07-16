//! **Boucle de défaut à la terre** (protection des personnes en schéma TN) —
//! impédance de boucle, courant de défaut présumé, impédance maximale garantissant
//! le déclenchement du dispositif de protection et tension de contact présumée, à
//! partir des impédances de boucle et des grandeurs de réseau fournies.
//!
//! ```text
//! impédance de boucle      Zs = Z_source + Z_phase + Z_pe
//! courant de défaut présumé Ia = U0 / Zs
//! impédance de boucle maxi  Zs_max = U0 / Ia
//! tension de contact        Uc = Ia · Z_pe
//! condition de déclenchement Zs ≤ U0 / Ia   (Ia = courant de fonctionnement)
//! ```
//!
//! `Z_source` impédance amont de la source (Ω), `Z_phase` impédance du conducteur
//! de phase de la boucle (Ω), `Z_pe` impédance du conducteur de protection (PE) de
//! la boucle (Ω), `Zs` impédance totale de la boucle de défaut (Ω), `U0` tension
//! simple phase-terre du réseau (V), `Ia` courant de défaut présumé ou courant de
//! fonctionnement du dispositif (A), `Zs_max` impédance de boucle maximale
//! admissible (Ω), `Uc` tension de contact présumée (V).
//!
//! **Convention** : SI ; impédances en Ω, tensions en V, courants en A ;
//! grandeurs en **module** et en **régime établi**. **Limite honnête** : schéma
//! **TN** (masses reliées au neutre) ; les **impédances de boucle** (`Z_source`,
//! `Z_phase`, `Z_pe`) et la **tension simple** `U0` sont **fournies par
//! l'appelant** (dépendent du transformateur, de la section et de la longueur des
//! conducteurs) — aucune valeur n'est inventée. Le **déclenchement dans le temps
//! imparti** exige `Zs ≤ U0 / Ia`, où `Ia` est le **courant de fonctionnement**
//! du dispositif de protection **fourni par l'appelant** (courbe du disjoncteur ou
//! du fusible, CEI 60364). Calcul en **arithmétique réelle** (modules), sans
//! composition vectorielle des impédances.

/// Impédance totale de la boucle de défaut `Zs = Z_source + Z_phase + Z_pe` (Ω),
/// somme de l'impédance amont de la source et des impédances des conducteurs de
/// phase et de protection de la boucle.
///
/// Panique si `source_impedance < 0`, `phase_conductor_impedance < 0` ou
/// `protective_conductor_impedance < 0`.
pub fn earthloop_impedance(
    source_impedance: f64,
    phase_conductor_impedance: f64,
    protective_conductor_impedance: f64,
) -> f64 {
    assert!(
        source_impedance >= 0.0,
        "l'impédance de source Z_source doit être ≥ 0"
    );
    assert!(
        phase_conductor_impedance >= 0.0,
        "l'impédance du conducteur de phase Z_phase doit être ≥ 0"
    );
    assert!(
        protective_conductor_impedance >= 0.0,
        "l'impédance du conducteur de protection Z_pe doit être ≥ 0"
    );
    source_impedance + phase_conductor_impedance + protective_conductor_impedance
}

/// Courant de défaut présumé `Ia = U0 / Zs` (A), quotient de la tension simple
/// par l'impédance de boucle.
///
/// Panique si `phase_voltage < 0` ou si `loop_impedance <= 0` (division par zéro).
pub fn earthloop_prospective_fault_current(phase_voltage: f64, loop_impedance: f64) -> f64 {
    assert!(phase_voltage >= 0.0, "la tension simple U0 doit être ≥ 0");
    assert!(
        loop_impedance > 0.0,
        "l'impédance de boucle Zs doit être strictement positive"
    );
    phase_voltage / loop_impedance
}

/// Impédance de boucle maximale admissible `Zs_max = U0 / Ia` (Ω), garantissant
/// que le courant de défaut atteigne le courant de fonctionnement `Ia` du
/// dispositif de protection (déclenchement dans le temps imparti).
///
/// Le déclenchement est assuré tant que l'impédance réelle de la boucle vérifie
/// `Zs ≤ Zs_max`.
///
/// Panique si `phase_voltage < 0` ou si `trip_current <= 0` (division par zéro).
pub fn earthloop_max_impedance_for_disconnection(phase_voltage: f64, trip_current: f64) -> f64 {
    assert!(phase_voltage >= 0.0, "la tension simple U0 doit être ≥ 0");
    assert!(
        trip_current > 0.0,
        "le courant de fonctionnement Ia doit être strictement positif"
    );
    phase_voltage / trip_current
}

/// Tension de contact présumée `Uc = Ia · Z_pe` (V), chute de tension dans le
/// conducteur de protection sous le courant de défaut.
///
/// Panique si `fault_current < 0` ou si `protective_conductor_impedance < 0`.
pub fn earthloop_touch_voltage(fault_current: f64, protective_conductor_impedance: f64) -> f64 {
    assert!(
        fault_current >= 0.0,
        "le courant de défaut Ia doit être ≥ 0"
    );
    assert!(
        protective_conductor_impedance >= 0.0,
        "l'impédance du conducteur de protection Z_pe doit être ≥ 0"
    );
    fault_current * protective_conductor_impedance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn impedance_is_additive() {
        // La somme des trois contributions donne l'impédance de boucle.
        let zs = earthloop_impedance(0.05, 0.35, 0.25);
        assert_relative_eq!(zs, 0.65, epsilon = 1e-12);
    }

    #[test]
    fn fault_current_recovers_impedance() {
        // Réciprocité : Ia = U0 / Zs, donc U0 / Ia restitue Zs.
        let u0 = 230.0_f64;
        let zs = 0.65_f64;
        let ia = earthloop_prospective_fault_current(u0, zs);
        assert_relative_eq!(u0 / ia, zs, epsilon = 1e-12);
    }

    #[test]
    fn max_impedance_triggers_exactly_at_trip_current() {
        // À l'impédance maximale Zs_max = U0 / Ia, le courant de défaut présumé
        // vaut exactement le courant de fonctionnement Ia : la condition limite
        // du déclenchement est atteinte.
        let u0 = 230.0_f64;
        let trip = 320.0_f64;
        let zs_max = earthloop_max_impedance_for_disconnection(u0, trip);
        let ia = earthloop_prospective_fault_current(u0, zs_max);
        assert_relative_eq!(ia, trip, epsilon = 1e-9);
    }

    #[test]
    fn touch_voltage_scales_with_pe_impedance() {
        // Proportionnalité : à courant de défaut fixé, doubler Z_pe double la
        // tension de contact.
        let uc1 = earthloop_touch_voltage(300.0, 0.20);
        let uc2 = earthloop_touch_voltage(300.0, 0.40);
        assert_relative_eq!(uc2 / uc1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_tn_loop_case() {
        // Cas chiffré réaliste — schéma TN, U0 = 230 V.
        //   Impédances de boucle : Z_source = 0,05 Ω, Z_phase = 0,35 Ω,
        //   Z_pe = 0,25 Ω → Zs = 0,05 + 0,35 + 0,25 = 0,65 Ω.
        let zs = earthloop_impedance(0.05, 0.35, 0.25);
        assert_relative_eq!(zs, 0.65, epsilon = 1e-12);

        //   Courant de défaut présumé Ia = 230 / 0,65 = 23000 / 65 = 4600 / 13
        //     = 353,84615385 A.
        let ia = earthloop_prospective_fault_current(230.0, zs);
        assert_relative_eq!(ia, 353.846_153_85, epsilon = 1e-3);

        //   Tension de contact présumée Uc = Ia · Z_pe = (4600/13) · 0,25
        //     = 1150 / 13 = 88,46153846 V.
        let uc = earthloop_touch_voltage(ia, 0.25);
        assert_relative_eq!(uc, 88.461_538_46, epsilon = 1e-3);

        //   Dispositif de protection : courant de fonctionnement Ia = 320 A.
        //   Impédance de boucle maximale Zs_max = 230 / 320 = 0,71875 Ω.
        let zs_max = earthloop_max_impedance_for_disconnection(230.0, 320.0);
        assert_relative_eq!(zs_max, 0.71875, epsilon = 1e-12);

        //   Comme Zs = 0,65 Ω ≤ Zs_max = 0,71875 Ω, le déclenchement est garanti.
        assert!(zs <= zs_max);
    }

    #[test]
    #[should_panic(expected = "l'impédance de boucle Zs doit être strictement positive")]
    fn fault_current_rejects_zero_impedance() {
        earthloop_prospective_fault_current(230.0, 0.0);
    }
}
