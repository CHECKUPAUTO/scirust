//! **Filtrage capacitif d'un redresseur** — estimation de l'ondulation résiduelle
//! et dimensionnement du condensateur de tête (« filtre à condensateur en tête »)
//! placé en sortie d'un pont redresseur : ondulation crête-à-crête en double et
//! simple alternance, facteur d'ondulation, capacité requise pour une ondulation
//! visée et tension continue moyenne approchée.
//!
//! ```text
//! ondulation double alternance  ΔV = I / (2 · f · C)
//! ondulation simple alternance  ΔV = I / (f · C)
//! facteur d'ondulation          r  = V_r,rms / V_dc
//! capacité requise (double)     C  = I / (2 · f · ΔV)
//! tension continue moyenne      V_dc ≈ V_pk − ΔV / 2
//! ```
//!
//! `I` courant de charge continu (A), `f` fréquence du réseau d'alimentation (Hz),
//! `C` capacité de filtrage (F), `ΔV` ondulation crête-à-crête (V), `V_r,rms`
//! valeur efficace de l'ondulation (V), `V_dc` tension continue moyenne (V),
//! `V_pk` tension de crête redressée (V), `r` facteur d'ondulation (sans unité).
//!
//! **Convention** : SI ; tensions en V, courants en A, capacités en F, fréquences
//! en Hz. Types `f64`, arithmétique réelle.
//!
//! **Limite honnête** : filtre à condensateur en tête avec **décharge linéaire
//! approchée** sur la charge (courant de charge supposé constant durant la
//! décharge). La fréquence d'ondulation vaut `2·f` en double alternance et `f` en
//! simple alternance. Le courant de charge `I` et la capacité `C` sont **fournis
//! par l'appelant** (point de fonctionnement, non inventés). L'approximation
//! `ΔV = I/(k·f·C)` n'est valable que pour une **ondulation faible** (`RC ≫`
//! période, soit une constante de temps grande devant la période du réseau) ;
//! ni la résistance série du condensateur, ni l'angle de conduction réel des
//! diodes ne sont modélisés.

/// Ondulation de tension crête-à-crête d'un redressement **double alternance**
/// avec condensateur de filtrage : `ΔV = I / (2 · f · C)` (V). La décharge
/// linéaire du condensateur se produit deux fois par période du réseau, d'où le
/// facteur 2 au dénominateur.
///
/// `load_current` en A, `capacitance` en F, `supply_frequency` en Hz, résultat
/// en V.
///
/// Panique si `load_current < 0` (courant physiquement ≥ 0), si
/// `capacitance <= 0` ou si `supply_frequency <= 0` (division par zéro).
pub fn smooth_ripple_voltage_fullwave(
    load_current: f64,
    capacitance: f64,
    supply_frequency: f64,
) -> f64 {
    assert!(load_current >= 0.0, "le courant de charge I doit être ≥ 0");
    assert!(
        capacitance > 0.0,
        "la capacité de filtrage C doit être strictement positive"
    );
    assert!(
        supply_frequency > 0.0,
        "la fréquence d'alimentation f doit être strictement positive"
    );
    load_current / (2.0 * supply_frequency * capacitance)
}

/// Ondulation de tension crête-à-crête d'un redressement **simple alternance**
/// avec condensateur de filtrage : `ΔV = I / (f · C)` (V). Le condensateur se
/// décharge une seule fois par période du réseau : à capacité et courant égaux,
/// l'ondulation est deux fois plus grande qu'en double alternance.
///
/// `load_current` en A, `capacitance` en F, `supply_frequency` en Hz, résultat
/// en V.
///
/// Panique si `load_current < 0` (courant physiquement ≥ 0), si
/// `capacitance <= 0` ou si `supply_frequency <= 0` (division par zéro).
pub fn smooth_ripple_voltage_halfwave(
    load_current: f64,
    capacitance: f64,
    supply_frequency: f64,
) -> f64 {
    assert!(load_current >= 0.0, "le courant de charge I doit être ≥ 0");
    assert!(
        capacitance > 0.0,
        "la capacité de filtrage C doit être strictement positive"
    );
    assert!(
        supply_frequency > 0.0,
        "la fréquence d'alimentation f doit être strictement positive"
    );
    load_current / (supply_frequency * capacitance)
}

/// Facteur d'ondulation : rapport de la valeur efficace de l'ondulation à la
/// tension continue moyenne, `r = V_r,rms / V_dc` (sans unité). Plus il est
/// faible, plus la sortie redressée est « lisse ».
///
/// `ripple_voltage_rms` en V, `dc_voltage` en V, résultat sans unité.
///
/// Panique si `ripple_voltage_rms < 0` (valeur efficace ≥ 0) ou si
/// `dc_voltage <= 0` (division par zéro ; tension continue strictement positive).
pub fn smooth_ripple_factor(ripple_voltage_rms: f64, dc_voltage: f64) -> f64 {
    assert!(
        ripple_voltage_rms >= 0.0,
        "la valeur efficace de l'ondulation V_r,rms doit être ≥ 0"
    );
    assert!(
        dc_voltage > 0.0,
        "la tension continue moyenne V_dc doit être strictement positive"
    );
    ripple_voltage_rms / dc_voltage
}

/// Capacité de filtrage requise pour ne pas dépasser une ondulation crête-à-crête
/// visée, en **double alternance** : `C = I / (2 · f · ΔV)` (F). C'est la relation
/// `smooth_ripple_voltage_fullwave` résolue en `C`.
///
/// `load_current` en A, `supply_frequency` en Hz, `ripple_voltage` (ondulation
/// visée) en V, résultat en F.
///
/// Panique si `load_current < 0` (courant physiquement ≥ 0), si
/// `supply_frequency <= 0` ou si `ripple_voltage <= 0` (division par zéro).
pub fn smooth_required_capacitance(
    load_current: f64,
    supply_frequency: f64,
    ripple_voltage: f64,
) -> f64 {
    assert!(load_current >= 0.0, "le courant de charge I doit être ≥ 0");
    assert!(
        supply_frequency > 0.0,
        "la fréquence d'alimentation f doit être strictement positive"
    );
    assert!(
        ripple_voltage > 0.0,
        "l'ondulation visée ΔV doit être strictement positive"
    );
    load_current / (2.0 * supply_frequency * ripple_voltage)
}

/// Tension continue moyenne approchée en sortie du filtre : la sortie oscille
/// entre `V_pk` (crête) et `V_pk − ΔV` (creux), sa moyenne vaut donc
/// `V_dc ≈ V_pk − ΔV / 2` (V).
///
/// `peak_voltage` en V, `ripple_voltage` (ondulation crête-à-crête) en V,
/// résultat en V.
///
/// Panique si `peak_voltage < 0` (tension de crête ≥ 0) ou si
/// `ripple_voltage < 0` (ondulation ≥ 0).
pub fn smooth_average_dc_voltage(peak_voltage: f64, ripple_voltage: f64) -> f64 {
    assert!(
        peak_voltage >= 0.0,
        "la tension de crête V_pk doit être ≥ 0"
    );
    assert!(
        ripple_voltage >= 0.0,
        "l'ondulation crête-à-crête ΔV doit être ≥ 0"
    );
    peak_voltage - ripple_voltage / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fullwave_half_of_halfwave() {
        // Identité : à courant, capacité et fréquence égaux, l'ondulation en
        // double alternance vaut la moitié de celle en simple alternance
        // (décharge deux fois par période).
        let i = 1.5_f64;
        let c = 470.0e-6_f64;
        let f = 50.0_f64;
        let full = smooth_ripple_voltage_fullwave(i, c, f);
        let half = smooth_ripple_voltage_halfwave(i, c, f);
        assert_relative_eq!(full, half / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn ripple_scales_inversely_with_capacitance() {
        // Proportionnalité inverse : à courant et fréquence fixés, doubler la
        // capacité divise par deux l'ondulation.
        let i = 2.0_f64;
        let f = 50.0_f64;
        let v1 = smooth_ripple_voltage_fullwave(i, 100.0e-6, f);
        let v2 = smooth_ripple_voltage_fullwave(i, 200.0e-6, f);
        assert_relative_eq!(v1 / v2, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn required_capacitance_inverts_ripple_fullwave() {
        // Réciprocité : la capacité requise pour une ondulation visée, réinjectée
        // dans le calcul d'ondulation double alternance, restitue l'ondulation.
        let i = 1.0_f64;
        let f = 50.0_f64;
        let target = 0.5_f64;
        let c = smooth_required_capacitance(i, f, target);
        let v = smooth_ripple_voltage_fullwave(i, c, f);
        assert_relative_eq!(v, target, epsilon = 1e-9);
    }

    #[test]
    fn ripple_fullwave_worked_case() {
        // Cas chiffré : I = 1 A, C = 1000 µF, f = 50 Hz.
        //   ΔV = 1 / (2 · 50 · 1000e-6) = 1 / (2 · 50 · 1e-3) = 1 / 0.1 = 10 V.
        // Recalcul indépendant : 2 · 50 = 100 ; 100 · 1.0e-3 = 0.1 ; 1 / 0.1 = 10.
        // (littéral vérifié deux fois)
        let v = smooth_ripple_voltage_fullwave(1.0, 1000.0e-6, 50.0);
        assert_relative_eq!(v, 10.0, epsilon = 1e-3);
    }

    #[test]
    fn ripple_factor_worked_case() {
        // Cas chiffré : V_r,rms = 1.2 V, V_dc = 24 V.
        //   r = 1.2 / 24 = 0.05.
        // Recalcul indépendant : 1.2 / 24 = 0.05. (vérifié deux fois)
        let r = smooth_ripple_factor(1.2, 24.0);
        assert_relative_eq!(r, 0.05, epsilon = 1e-9);
    }

    #[test]
    fn average_dc_voltage_worked_case() {
        // Cas chiffré : V_pk = 17 V, ΔV = 2 V.
        //   V_dc = 17 − 2/2 = 17 − 1 = 16 V.
        // Recalcul indépendant : 2/2 = 1 ; 17 − 1 = 16. (vérifié deux fois)
        let v_dc = smooth_average_dc_voltage(17.0, 2.0);
        assert_relative_eq!(v_dc, 16.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la capacité de filtrage C doit être strictement positive")]
    fn zero_capacitance_panics() {
        smooth_ripple_voltage_fullwave(1.0, 0.0, 50.0);
    }
}
