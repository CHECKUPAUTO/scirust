//! Économie des procédés — extrapolation des coûts d'investissement (règle des
//! six dixièmes et forme générale à exposant), investissement total par le
//! facteur de Lang, temps de retour simple et annuité de récupération du capital.
//!
//! ```text
//! règle des six dixièmes   C₂ = C₁ · (Q₂ / Q₁)^0.6                    [$]
//! extrapolation générale   C₂ = C₁ · (Q₂ / Q₁)^n                      [$]
//! facteur de Lang          C_TIC = C_PEC · f_L                        [$]
//! temps de retour simple   t_pb  = I / A_cf                           [an]
//! annuité de capital       A = I · i · (1+i)^N / ((1+i)^N − 1)        [$·an⁻¹]
//! ```
//!
//! `C₁` coût de référence connu [$], `Q₁` capacité (ou taille) de référence
//! [unité cohérente], `Q₂` capacité de l'unité à estimer [même unité], `C₂` coût
//! extrapolé [$] ; `n` exposant d'échelle [sans dimension, ≈0,6 typique],
//! `C_PEC` coût des équipements achetés [$], `f_L` facteur de Lang [sans
//! dimension, ≥1], `C_TIC` investissement total immobilisé [$] ; `I`
//! investissement en capital [$], `A_cf` flux de trésorerie net annuel
//! [$·an⁻¹], `t_pb` temps de retour simple [an] ; `i` taux d'intérêt annuel
//! [sans dimension, par an], `N` durée de vie de l'installation [an], `A`
//! annuité de récupération du capital [$·an⁻¹].
//!
//! **Limite honnête** : ce sont des **estimations d'ordre de grandeur**.
//! L'**exposant d'échelle** (`n` ≈ 0,6), le **facteur de Lang** (`f_L`), le
//! **coût de référence** (`C₁`, `C_PEC`) et le **taux d'intérêt** (`i`) sont
//! **FOURNIS** par l'appelant : ils dépendent du type de procédé, du matériau,
//! du pays, de l'indice de coût (CEPCI, indice Marshall & Swift…) et de l'année
//! de référence, et ne sont **jamais** supposés « par défaut ». Le **temps de
//! retour simple** ignore l'**actualisation** (valeur temporelle de l'argent) ;
//! l'**annuité** suppose des **flux constants** et un taux constant. Aucun
//! indice d'actualisation, aucune inflation, aucune fiscalité n'est calculé ici.
//! Pour une **étude fine**, utiliser une **actualisation complète** (VAN/TRI).

/// Extrapolation de coût par la **règle des six dixièmes**
/// `C₂ = C₁ · (Q₂ / Q₁)^0.6` ($), cas particulier de l'extrapolation à exposant
/// avec l'exposant typique `n = 0,6`.
///
/// `known_cost` (C₁) coût de référence [$], `known_capacity` (Q₁) et
/// `new_capacity` (Q₂) capacités exprimées dans la **même unité cohérente**.
///
/// Panique si `known_cost < 0`, si `known_capacity <= 0` ou si
/// `new_capacity < 0`.
pub fn econ_six_tenths_rule(known_cost: f64, known_capacity: f64, new_capacity: f64) -> f64 {
    assert!(known_cost >= 0.0, "C₁ ≥ 0 requis (coût de référence)");
    assert!(
        known_capacity > 0.0,
        "Q₁ > 0 requis (capacité de référence)"
    );
    assert!(new_capacity >= 0.0, "Q₂ ≥ 0 requis (capacité à estimer)");
    known_cost * (new_capacity / known_capacity).powf(0.6)
}

/// Extrapolation de coût par la **forme générale à exposant**
/// `C₂ = C₁ · (Q₂ / Q₁)^n` ($), où l'exposant d'échelle `n` est **FOURNI** par
/// l'appelant (souvent 0,6 ≤ `n` ≤ 0,9 selon l'équipement).
///
/// `known_cost` (C₁) coût de référence [$], `known_capacity` (Q₁) et
/// `new_capacity` (Q₂) capacités [même unité cohérente], `exponent` (n) exposant
/// d'échelle [sans dimension].
///
/// Panique si `known_cost < 0`, si `known_capacity <= 0`, si `new_capacity < 0`
/// ou si `exponent` n'est pas fini.
pub fn econ_scale_cost(
    known_cost: f64,
    known_capacity: f64,
    new_capacity: f64,
    exponent: f64,
) -> f64 {
    assert!(known_cost >= 0.0, "C₁ ≥ 0 requis (coût de référence)");
    assert!(
        known_capacity > 0.0,
        "Q₁ > 0 requis (capacité de référence)"
    );
    assert!(new_capacity >= 0.0, "Q₂ ≥ 0 requis (capacité à estimer)");
    assert!(exponent.is_finite(), "n fini requis (exposant d'échelle)");
    known_cost * (new_capacity / known_capacity).powf(exponent)
}

/// Investissement total immobilisé par le **facteur de Lang**
/// `C_TIC = C_PEC · f_L` ($), le facteur multipliant le coût des équipements
/// achetés pour couvrir installation, tuyauterie, instrumentation, génie civil…
///
/// `purchased_equipment_cost` (C_PEC) coût des équipements achetés [$],
/// `lang_factor` (f_L) facteur de Lang [sans dimension], **FOURNI** par
/// l'appelant (typiquement de l'ordre de 3 à 5 selon le type de procédé).
///
/// Panique si `purchased_equipment_cost < 0` ou si `lang_factor < 1`
/// (l'investissement total ne peut être inférieur au coût des équipements).
pub fn econ_lang_factor_capital(purchased_equipment_cost: f64, lang_factor: f64) -> f64 {
    assert!(
        purchased_equipment_cost >= 0.0,
        "C_PEC ≥ 0 requis (coût des équipements achetés)"
    );
    assert!(lang_factor >= 1.0, "f_L ≥ 1 requis (facteur de Lang)");
    purchased_equipment_cost * lang_factor
}

/// Temps de retour simple `t_pb = I / A_cf` (an), durée nécessaire pour récupérer
/// l'investissement à flux de trésorerie net annuel constant, **sans**
/// actualisation.
///
/// `capital_investment` (I) investissement en capital [$], `annual_net_cash_flow`
/// (A_cf) flux de trésorerie net annuel [$·an⁻¹].
///
/// Panique si `capital_investment < 0` ou si `annual_net_cash_flow <= 0`
/// (un flux nul ou négatif ne rembourse jamais l'investissement).
pub fn econ_payback_period(capital_investment: f64, annual_net_cash_flow: f64) -> f64 {
    assert!(
        capital_investment >= 0.0,
        "I ≥ 0 requis (investissement en capital)"
    );
    assert!(
        annual_net_cash_flow > 0.0,
        "A_cf > 0 requis (flux de trésorerie net annuel)"
    );
    capital_investment / annual_net_cash_flow
}

/// Annuité de récupération du capital
/// `A = I · i · (1+i)^N / ((1+i)^N − 1)` ($·an⁻¹), versement annuel constant
/// amortissant l'investissement `I` sur `N` ans au taux `i` (facteur de
/// récupération du capital, CRF). Pour `N = 1`, `A = I · (1 + i)`.
///
/// `capital_investment` (I) investissement en capital [$], `interest_rate` (i)
/// taux d'intérêt annuel [sans dimension, par an], **FOURNI** par l'appelant,
/// `plant_life_years` (N) durée de vie [an].
///
/// Panique si `capital_investment < 0`, si `interest_rate <= 0` ou si
/// `plant_life_years <= 0`.
pub fn econ_annual_capital_charge(
    capital_investment: f64,
    interest_rate: f64,
    plant_life_years: f64,
) -> f64 {
    assert!(
        capital_investment >= 0.0,
        "I ≥ 0 requis (investissement en capital)"
    );
    assert!(interest_rate > 0.0, "i > 0 requis (taux d'intérêt annuel)");
    assert!(
        plant_life_years > 0.0,
        "N > 0 requis (durée de vie de l'installation)"
    );
    let growth = (1.0 + interest_rate).powf(plant_life_years);
    capital_investment * interest_rate * growth / (growth - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn six_tenths_matches_general_scale_with_exponent_zero_point_six() {
        // La règle des six dixièmes = extrapolation générale avec n = 0,6.
        let (c1, q1, q2) = (1.0e6_f64, 100.0_f64, 200.0_f64);
        let six = econ_six_tenths_rule(c1, q1, q2);
        let general = econ_scale_cost(c1, q1, q2, 0.6_f64);
        assert_relative_eq!(six, general, max_relative = 1e-12);
        // Doubler la capacité : C₂ = 1e6 · 2^0.6 ≈ 1 515 716,57 $.
        assert_relative_eq!(six, 1_515_716.566_510_4, max_relative = 1e-9);
    }

    #[test]
    fn scale_cost_unit_exponent_is_linear_and_same_capacity_is_identity() {
        // n = 1 ⇒ coût proportionnel à la capacité.
        let (c1, q1, q2) = (2.0e5_f64, 50.0_f64, 150.0_f64);
        let linear = econ_scale_cost(c1, q1, q2, 1.0_f64);
        assert_relative_eq!(linear, c1 * (q2 / q1), max_relative = 1e-12);
        assert_relative_eq!(linear, 6.0e5, max_relative = 1e-12);
        // Q₂ = Q₁ ⇒ C₂ = C₁ quel que soit l'exposant.
        assert_relative_eq!(
            econ_scale_cost(c1, q1, q1, 0.6_f64),
            c1,
            max_relative = 1e-12
        );
    }

    #[test]
    fn lang_factor_scales_capital_linearly() {
        // C_PEC = 1,0 M$, f_L = 4,74 ⇒ C_TIC = 4,74 M$.
        let pec = 1.0e6_f64;
        let tic = econ_lang_factor_capital(pec, 4.74_f64);
        assert_relative_eq!(tic, 4.74e6, max_relative = 1e-12);
        // Facteur unité ⇒ investissement = coût des équipements.
        assert_relative_eq!(
            econ_lang_factor_capital(pec, 1.0_f64),
            pec,
            max_relative = 1e-12
        );
    }

    #[test]
    fn payback_period_is_inverse_of_cash_flow() {
        // I = 5,0 M$, A_cf = 1,25 M$/an ⇒ t_pb = 4 ans.
        let (capital, cash) = (5.0e6_f64, 1.25e6_f64);
        let t = econ_payback_period(capital, cash);
        assert_relative_eq!(t, 4.0, max_relative = 1e-12);
        // Réciprocité : t · A_cf = I.
        assert_relative_eq!(t * cash, capital, max_relative = 1e-9);
    }

    #[test]
    fn annual_capital_charge_realistic_and_one_year_identity() {
        // I = 1,0 M$, i = 0,10, N = 10 ⇒ CRF = 0,162745... ⇒ A ≈ 162 745,39 $/an.
        let a = econ_annual_capital_charge(1.0e6_f64, 0.10_f64, 10.0_f64);
        assert_relative_eq!(a, 162_745.394_882_51, max_relative = 1e-3);
        // Cas limite N = 1 : A = I · (1 + i).
        let a1 = econ_annual_capital_charge(1.0e6_f64, 0.08_f64, 1.0_f64);
        assert_relative_eq!(a1, 1.0e6 * 1.08, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "A_cf > 0 requis")]
    fn payback_period_panics_on_nonpositive_cash_flow() {
        // Flux net nul ⇒ jamais remboursé ⇒ panique.
        let _ = econ_payback_period(1.0e6_f64, 0.0_f64);
    }
}
