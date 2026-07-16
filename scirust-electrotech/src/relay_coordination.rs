//! **Sélectivité chronométrique des protections (coordination)** — vérification de
//! la sélectivité (discrimination) entre deux relais à maximum de courant placés en
//! cascade sur un même départ : marge de temps entre l'amont et l'aval, critère de
//! sélectivité, rapport des réglages en courant et temps amont minimal requis.
//!
//! ```text
//! marge de sélectivité         Δt = t_amont − t_aval
//! critère de sélectivité       sélectif  ⟺  Δt ≥ Δt_min
//! rapport de réglage courant   r = I_r,amont / I_r,aval
//! temps amont minimal          t_amont,min = t_aval + Δt_min
//! ```
//!
//! `t_amont` temps de déclenchement du relais **amont** (le plus proche de la
//! source, s), `t_aval` temps de déclenchement du relais **aval** (le plus proche
//! du défaut, s), `Δt` marge de sélectivité chronométrique (s), `Δt_min` marge
//! minimale de discrimination exigée (s, typiquement 0,3 à 0,4 s), `I_r,amont` et
//! `I_r,aval` courants de réglage des deux relais (A), `r` rapport de réglage (sans
//! dimension) et `t_amont,min` temps amont minimal garantissant la sélectivité (s).
//!
//! **Convention** : SI ; temps en secondes, courants de réglage en A ; le rapport
//! de réglage est sans dimension ; arithmétique **réelle** (f64).
//! **Limite honnête** : sélectivité **chronométrique** (temps gradués) entre relais
//! en cascade. Les temps de déclenchement `t_amont` et `t_aval` sont **fournis par
//! l'appelant** (calculés en amont, par exemple via le module `overcurrent_relay`
//! pour un même courant de défaut) ; la marge minimale `Δt_min` est **fournie**
//! (elle intègre le temps de coupure du disjoncteur aval, le dépassement du relais
//! et les tolérances des courbes). Ce module suppose des **courbes coordonnées** et
//! ne modélise ni la sélectivité **ampèremétrique** (par seuils de courant) ni la
//! sélectivité **logique** (verrouillage par échange de signaux), qui sont
//! distinctes.

/// Marge de sélectivité chronométrique `Δt = t_amont − t_aval` (s).
///
/// `upstream_trip_time` est le temps de déclenchement du relais amont (`t_amont`,
/// s) et `downstream_trip_time` celui du relais aval (`t_aval`, s) ; le résultat est
/// la marge de sélectivité (s), positive quand l'amont déclenche après l'aval.
///
/// Panique si `upstream_trip_time < 0` ou si `downstream_trip_time < 0`.
pub fn relaycoord_discrimination_margin(upstream_trip_time: f64, downstream_trip_time: f64) -> f64 {
    assert!(upstream_trip_time >= 0.0, "t_amont ≥ 0 requis");
    assert!(downstream_trip_time >= 0.0, "t_aval ≥ 0 requis");
    upstream_trip_time - downstream_trip_time
}

/// Critère de sélectivité `sélectif ⟺ Δt ≥ Δt_min`.
///
/// `discrimination_margin` est la marge de sélectivité mesurée (`Δt`, s) et
/// `minimum_margin` la marge minimale exigée (`Δt_min`, s, fournie par l'appelant) ;
/// le résultat vaut `true` si la marge atteint ou dépasse la marge minimale.
///
/// Panique si `minimum_margin < 0`.
pub fn relaycoord_is_selective(discrimination_margin: f64, minimum_margin: f64) -> bool {
    assert!(minimum_margin >= 0.0, "Δt_min ≥ 0 requis");
    discrimination_margin >= minimum_margin
}

/// Rapport de réglage en courant `r = I_r,amont / I_r,aval` (sans dimension).
///
/// `upstream_setting` est le courant de réglage du relais amont (`I_r,amont`, A) et
/// `downstream_setting` celui du relais aval (`I_r,aval`, A) ; le résultat est le
/// rapport de réglage (sans dimension). Une sélectivité correcte exige `r > 1`
/// (réglage amont plus élevé), ce que l'appelant doit vérifier sur le résultat.
///
/// Panique si `upstream_setting <= 0` ou si `downstream_setting <= 0`.
pub fn relaycoord_current_setting_ratio(upstream_setting: f64, downstream_setting: f64) -> f64 {
    assert!(upstream_setting > 0.0, "I_r,amont > 0 requis");
    assert!(downstream_setting > 0.0, "I_r,aval > 0 requis");
    upstream_setting / downstream_setting
}

/// Temps amont minimal pour la sélectivité `t_amont,min = t_aval + Δt_min` (s).
///
/// `downstream_time` est le temps de déclenchement du relais aval (`t_aval`, s) et
/// `minimum_margin` la marge minimale exigée (`Δt_min`, s) ; le résultat est le
/// temps de déclenchement amont minimal garantissant la sélectivité (s).
///
/// Panique si `downstream_time < 0` ou si `minimum_margin < 0`.
pub fn relaycoord_required_upstream_time(downstream_time: f64, minimum_margin: f64) -> f64 {
    assert!(downstream_time >= 0.0, "t_aval ≥ 0 requis");
    assert!(minimum_margin >= 0.0, "Δt_min ≥ 0 requis");
    downstream_time + minimum_margin
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn margin_realistic_value() {
        // Cas chiffré : relais amont t = 0,75 s, relais aval t = 0,35 s
        // → Δt = 0,75 − 0,35 = 0,40 s.
        let margin = relaycoord_discrimination_margin(0.75, 0.35);
        assert_relative_eq!(margin, 0.40, epsilon = 1e-9);
    }

    #[test]
    fn margin_is_antisymmetric() {
        // Réciprocité : échanger amont/aval change le signe de la marge.
        let ab = relaycoord_discrimination_margin(0.9, 0.5);
        let ba = relaycoord_discrimination_margin(0.5, 0.9);
        assert_relative_eq!(ab, -ba, epsilon = 1e-12);
    }

    #[test]
    fn required_time_reconstructs_the_minimum_margin() {
        // Identité : la marge entre le temps amont minimal et l'aval vaut Δt_min.
        let downstream = 0.35_f64;
        let min_margin = 0.30_f64;
        let upstream_min = relaycoord_required_upstream_time(downstream, min_margin);
        // upstream_min = 0,35 + 0,30 = 0,65 s.
        assert_relative_eq!(upstream_min, 0.65, epsilon = 1e-9);
        assert_relative_eq!(
            relaycoord_discrimination_margin(upstream_min, downstream),
            min_margin,
            epsilon = 1e-12
        );
    }

    #[test]
    fn selectivity_holds_exactly_at_the_boundary() {
        // Au seuil Δt = Δt_min, le critère (≥) est satisfait ; juste en dessous, non.
        assert!(relaycoord_is_selective(0.30, 0.30));
        assert!(relaycoord_is_selective(0.40, 0.30));
        assert!(!relaycoord_is_selective(0.29, 0.30));
    }

    #[test]
    fn setting_ratio_is_reciprocal() {
        // Réciprocité : r(amont, aval) = 1 / r(aval, amont).
        let up = 160.0_f64;
        let down = 100.0_f64;
        let r = relaycoord_current_setting_ratio(up, down);
        // r = 160 / 100 = 1,6 (> 1, sélectivité correcte).
        assert_relative_eq!(r, 1.6, epsilon = 1e-12);
        let r_inv = relaycoord_current_setting_ratio(down, up);
        assert_relative_eq!(r, 1.0 / r_inv, epsilon = 1e-12);
    }

    #[test]
    fn required_time_scales_with_min_margin() {
        // Proportionnalité de l'accroissement : doubler Δt_min double le supplément
        // de temps amont (t_amont,min − t_aval).
        let td = 0.5_f64;
        let extra1 = relaycoord_required_upstream_time(td, 0.2) - td;
        let extra2 = relaycoord_required_upstream_time(td, 0.4) - td;
        assert_relative_eq!(extra2 / extra1, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "I_r,aval > 0 requis")]
    fn zero_downstream_setting_panics() {
        // Un courant de réglage aval nul rend le rapport indéfini → panique.
        relaycoord_current_setting_ratio(160.0, 0.0);
    }
}
