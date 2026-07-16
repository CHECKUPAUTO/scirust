//! Puissance de compression d'un gaz — travail massique isentropique et
//! polytropique d'un compresseur, température de refoulement, puissance
//! d'entraînement au frein et découpage d'un rapport global en étages égaux
//! (compression multi-étagée avec refroidissement intermédiaire).
//!
//! ```text
//! travail isentropique massique
//!   w_s = r · T1 · [γ/(γ−1)] · (Π^((γ−1)/γ) − 1)          [J·kg⁻¹]
//! travail polytropique massique
//!   w_p = r · T1 · [n/(n−1)] · (Π^((n−1)/n) − 1)          [J·kg⁻¹]
//! température de refoulement isentropique
//!   T2  = T1 · Π^((γ−1)/γ)                                [K]
//! puissance d'entraînement (au frein)
//!   P   = ṁ · w / η                                       [W]
//! rapport de pression par étage (N étages égaux)
//!   Π_e = Π_g^(1/N)                                        [-]
//! ```
//!
//! `r` constante spécifique du gaz `r = R/M` [J·kg⁻¹·K⁻¹], `T1` température
//! d'aspiration [K], `Π` (`pressure_ratio`) rapport de pression absolue
//! refoulement/aspiration `p2/p1` [sans dimension, ≥ 1], `γ` (`gamma`) rapport
//! des chaleurs massiques `cp/cv` [sans dimension, > 1], `n`
//! (`polytropic_exponent`) exposant polytropique [sans dimension, > 1] ; `w_s`
//! travail réversible isentropique massique et `w_p` travail polytropique
//! massique [J·kg⁻¹] ; `T2` température de refoulement isentropique [K] ; `ṁ`
//! (`mass_flow`) débit massique [kg·s⁻¹], `w` (`specific_work`) travail massique
//! réversible [J·kg⁻¹], `η` (`isentropic_efficiency`) rendement isentropique (ou
//! polytropique selon le travail fourni) [sans dimension, 0 < η ≤ 1], `P`
//! puissance mécanique à l'arbre [W] ; `Π_g` (`overall_ratio`) rapport de
//! pression global [sans dimension, ≥ 1], `N` (`stages`) nombre d'étages
//! [sans dimension, ≥ 1], `Π_e` rapport par étage [sans dimension].
//!
//! **Limite honnête** : le gaz est supposé **PARFAIT** (loi des gaz parfaits,
//! facteur de compressibilité `Z = 1`) ; les gaz réels ne sont **pas** traités.
//! La constante spécifique `r = R/M`, l'exposant isentropique `γ` ou
//! polytropique `n` et le **rendement** `η` sont **FOURNIS** par l'appelant :
//! ce sont des propriétés du gaz et de la machine (elles dépendent de la nature
//! du gaz, de la température, du taux de compression et du dessin de la roue) et
//! ne sont **jamais** inventées ni supposées « par défaut ». Le découpage
//! multi-étagé **égalise** les rapports par étage `Π_e = Π_g^(1/N)`, ce qui
//! **minimise** le travail total à condition d'un **refroidissement
//! intermédiaire** ramenant chaque étage à la même température d'aspiration
//! `T1` (hypothèse du refroidissement parfait) ; les pertes de charge des
//! réfrigérants ne sont pas comptées. Aucune propriété d'état
//! (enthalpies, entropies, `Z`, cycles thermodynamiques) n'est calculée ici —
//! cela relève de `scirust-thermo` — et aucune mécanique des fluides
//! fondamentale (pertes de charge détaillées) ne relève de ce module.

/// Travail massique **isentropique** (réversible adiabatique) d'un compresseur
/// de gaz parfait `w_s = r · T1 · [γ/(γ−1)] · (Π^((γ−1)/γ) − 1)` (J·kg⁻¹).
///
/// `gas_constant_specific` (r) constante spécifique du gaz `r = R/M`
/// [J·kg⁻¹·K⁻¹], `inlet_temperature` (T1) température d'aspiration [K],
/// `pressure_ratio` (Π) rapport de pression `p2/p1` [sans dimension],
/// `gamma` (γ) rapport des chaleurs massiques `cp/cv` [sans dimension].
///
/// Panique si `r < 0`, si `T1 ≤ 0` (température absolue), si `Π < 1` (une
/// compression relève la pression), ou si `γ ≤ 1` (`γ/(γ−1)` non défini ou
/// négatif).
pub fn cmp_isentropic_work(
    gas_constant_specific: f64,
    inlet_temperature: f64,
    pressure_ratio: f64,
    gamma: f64,
) -> f64 {
    assert!(
        gas_constant_specific >= 0.0,
        "r ≥ 0 requis (constante spécifique du gaz)"
    );
    assert!(
        inlet_temperature > 0.0,
        "T1 > 0 requis (température absolue d'aspiration)"
    );
    assert!(
        pressure_ratio >= 1.0,
        "Π ≥ 1 requis (rapport de pression d'une compression)"
    );
    assert!(gamma > 1.0, "γ > 1 requis (rapport des chaleurs massiques)");
    let exponent = (gamma - 1.0) / gamma;
    gas_constant_specific
        * inlet_temperature
        * (gamma / (gamma - 1.0))
        * (pressure_ratio.powf(exponent) - 1.0)
}

/// Travail massique **polytropique** d'un compresseur de gaz parfait
/// `w_p = r · T1 · [n/(n−1)] · (Π^((n−1)/n) − 1)` (J·kg⁻¹). Même forme que le
/// travail isentropique avec l'exposant polytropique `n` à la place de `γ`.
///
/// `gas_constant_specific` (r) constante spécifique `r = R/M` [J·kg⁻¹·K⁻¹],
/// `inlet_temperature` (T1) température d'aspiration [K], `pressure_ratio` (Π)
/// rapport de pression [sans dimension], `polytropic_exponent` (n) exposant
/// polytropique [sans dimension].
///
/// Panique si `r < 0`, si `T1 ≤ 0`, si `Π < 1`, ou si `n ≤ 1` (`n/(n−1)` non
/// défini ou négatif).
pub fn cmp_polytropic_work(
    gas_constant_specific: f64,
    inlet_temperature: f64,
    pressure_ratio: f64,
    polytropic_exponent: f64,
) -> f64 {
    assert!(
        gas_constant_specific >= 0.0,
        "r ≥ 0 requis (constante spécifique du gaz)"
    );
    assert!(
        inlet_temperature > 0.0,
        "T1 > 0 requis (température absolue d'aspiration)"
    );
    assert!(
        pressure_ratio >= 1.0,
        "Π ≥ 1 requis (rapport de pression d'une compression)"
    );
    assert!(
        polytropic_exponent > 1.0,
        "n > 1 requis (exposant polytropique)"
    );
    let exponent = (polytropic_exponent - 1.0) / polytropic_exponent;
    gas_constant_specific
        * inlet_temperature
        * (polytropic_exponent / (polytropic_exponent - 1.0))
        * (pressure_ratio.powf(exponent) - 1.0)
}

/// Température de refoulement **isentropique** d'un gaz parfait
/// `T2 = T1 · Π^((γ−1)/γ)` (K), température atteinte en fin de compression
/// réversible adiabatique.
///
/// `inlet_temperature` (T1) température d'aspiration [K], `pressure_ratio` (Π)
/// rapport de pression `p2/p1` [sans dimension], `gamma` (γ) rapport des
/// chaleurs massiques [sans dimension].
///
/// Panique si `T1 ≤ 0` (température absolue), si `Π < 1`, ou si `γ ≤ 1`.
pub fn cmp_discharge_temperature(inlet_temperature: f64, pressure_ratio: f64, gamma: f64) -> f64 {
    assert!(
        inlet_temperature > 0.0,
        "T1 > 0 requis (température absolue d'aspiration)"
    );
    assert!(
        pressure_ratio >= 1.0,
        "Π ≥ 1 requis (rapport de pression d'une compression)"
    );
    assert!(gamma > 1.0, "γ > 1 requis (rapport des chaleurs massiques)");
    let exponent = (gamma - 1.0) / gamma;
    inlet_temperature * pressure_ratio.powf(exponent)
}

/// Puissance mécanique d'entraînement (au frein) d'un compresseur
/// `P = ṁ · w / η` (W), obtenue en corrigeant le travail réversible massique
/// par le rendement de la machine.
///
/// `mass_flow` (ṁ) débit massique de gaz [kg·s⁻¹], `specific_work` (w) travail
/// réversible massique [J·kg⁻¹] (isentropique ou polytropique), `η`
/// `isentropic_efficiency` rendement de la machine [sans dimension].
///
/// Panique si `ṁ < 0`, si `w < 0`, ou si `η` hors de `]0, 1]` (un rendement
/// nul donnerait une puissance infinie ; un rendement > 1 violerait la
/// thermodynamique).
pub fn cmp_power(mass_flow: f64, specific_work: f64, isentropic_efficiency: f64) -> f64 {
    assert!(mass_flow >= 0.0, "ṁ ≥ 0 requis (débit massique)");
    assert!(specific_work >= 0.0, "w ≥ 0 requis (travail massique)");
    assert!(
        isentropic_efficiency > 0.0 && isentropic_efficiency <= 1.0,
        "0 < η ≤ 1 requis (rendement du compresseur)"
    );
    mass_flow * specific_work / isentropic_efficiency
}

/// Rapport de pression **par étage** d'une compression multi-étagée à étages
/// égaux `Π_e = Π_g^(1/N)` (sans dimension). Répartir le rapport global en
/// étages égaux minimise le travail total sous refroidissement intermédiaire
/// parfait (retour à `T1` avant chaque étage).
///
/// `overall_ratio` (Π_g) rapport de pression global `p_final/p_initial`
/// [sans dimension], `stages` (N) nombre d'étages [sans dimension].
///
/// Panique si `Π_g < 1` (une compression relève la pression) ou si `N < 1`
/// (il faut au moins un étage).
pub fn cmp_stage_pressure_ratio(overall_ratio: f64, stages: f64) -> f64 {
    assert!(
        overall_ratio >= 1.0,
        "Π_g ≥ 1 requis (rapport de pression global)"
    );
    assert!(stages >= 1.0, "N ≥ 1 requis (nombre d'étages)");
    overall_ratio.powf(1.0 / stages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn isentropic_work_zero_at_unit_ratio() {
        // Π = 1 ⇒ Π^k = 1 ⇒ travail nul : aucune compression, aucun travail.
        let w = cmp_isentropic_work(287.0_f64, 300.0_f64, 1.0_f64, 1.4_f64);
        assert_relative_eq!(w, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn discharge_temperature_unit_ratio_is_inlet() {
        // Π = 1 ⇒ T2 = T1 · 1 = T1 : pas de compression, pas d'échauffement.
        let t2 = cmp_discharge_temperature(310.0_f64, 1.0_f64, 1.4_f64);
        assert_relative_eq!(t2, 310.0, max_relative = 1e-12);
    }

    #[test]
    fn isentropic_work_matches_enthalpy_rise() {
        // Identité w_s = cp · (T2 − T1) avec cp = r · γ/(γ−1) : le travail
        // isentropique égale l'élévation d'enthalpie du gaz parfait.
        // Air : r = 287, T1 = 300 K, Π = 4, γ = 1.4.
        let r = 287.0_f64;
        let t1 = 300.0_f64;
        let pr = 4.0_f64;
        let gamma = 1.4_f64;
        let w = cmp_isentropic_work(r, t1, pr, gamma);
        let t2 = cmp_discharge_temperature(t1, pr, gamma);
        let cp = r * gamma / (gamma - 1.0);
        assert_relative_eq!(w, cp * (t2 - t1), max_relative = 1e-9);
    }

    #[test]
    fn polytropic_equals_isentropic_when_n_is_gamma() {
        // n = γ ⇒ le travail polytropique coïncide avec l'isentropique
        // (la compression polytropique dégénère en compression isentropique).
        let ws = cmp_isentropic_work(287.0_f64, 300.0_f64, 4.0_f64, 1.4_f64);
        let wp = cmp_polytropic_work(287.0_f64, 300.0_f64, 4.0_f64, 1.4_f64);
        assert_relative_eq!(ws, wp, max_relative = 1e-12);
    }

    #[test]
    fn discharge_temperature_worked_case() {
        // Air : T1 = 300 K, Π = 4, γ = 1.4 ⇒ (γ−1)/γ = 2/7 ≈ 0.2857142857.
        //   4^(2/7) = exp(0.2857142857 · ln 4) = exp(0.3960841032) = 1.4859943.
        //   T2 = 300 · 1.4859943 = 445.7983 K.
        // Littéral recalculé indépendamment : 300 · 1.4859943 = 445.79829 K.
        let t2 = cmp_discharge_temperature(300.0_f64, 4.0_f64, 1.4_f64);
        assert_relative_eq!(t2, 445.7983, max_relative = 1e-3);
    }

    #[test]
    fn stage_ratio_reconstructs_overall() {
        // Réciprocité du découpage : Π_e^N = Π_g. Avec Π_g = 16, N = 2 ⇒
        //   Π_e = 16^(1/2) = 4, et 4^2 = 16.
        let stage = cmp_stage_pressure_ratio(16.0_f64, 2.0_f64);
        assert_relative_eq!(stage, 4.0, max_relative = 1e-12);
        assert_relative_eq!(stage.powf(2.0), 16.0, max_relative = 1e-12);
    }

    #[test]
    fn power_scales_inversely_with_efficiency() {
        // P = ṁ · w / η : diviser le rendement par deux double la puissance.
        // ṁ = 2 kg/s, w = 150000 J/kg, η = 0.8 ⇒ P = 2·150000/0.8 = 375000 W.
        let p_high = cmp_power(2.0_f64, 150_000.0_f64, 0.8_f64);
        let p_low = cmp_power(2.0_f64, 150_000.0_f64, 0.4_f64);
        assert_relative_eq!(p_high, 375_000.0, max_relative = 1e-12);
        assert_relative_eq!(p_low, 2.0 * p_high, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "0 < η ≤ 1 requis")]
    fn power_panics_on_zero_efficiency() {
        // Rendement nul ⇒ puissance infinie non physique ⇒ entrée rejetée.
        let _ = cmp_power(2.0_f64, 150_000.0_f64, 0.0_f64);
    }
}
