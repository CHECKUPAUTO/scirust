//! Courbe de séchage convectif à deux périodes — humidité libre (base sèche),
//! durée de la période à allure constante, durée de la période à allure
//! décroissante (modèle linéaire passant par l'origine) et durée totale.
//!
//! ```text
//! humidité libre (base sèche)   X   = X_t − X*                         [-]
//! durée allure constante        t_c = M_s·(X_1 − X_c)/(A·N_c)          [s]
//! durée allure décroissante     t_f = M_s·X_c·ln(X_c/X_2)/(A·N_c)      [s]
//! durée totale de séchage       t   = t_c + t_f                        [s]
//! ```
//!
//! `X` humidité **libre** en **base sèche** [kg eau · kg solide sec⁻¹, sans
//! dimension], `X_t` teneur en eau totale (base sèche), `X*` teneur en eau
//! d'**équilibre** (base sèche) ; `M_s` masse de **solide sec** [kg], `A` aire
//! d'échange (surface de séchage exposée) [m²], `N_c` allure (flux) de séchage
//! de la période à **allure constante** [kg eau · m⁻² · s⁻¹] ; `X_1` humidité
//! libre **initiale**, `X_c` humidité libre **critique** (fin de l'allure
//! constante), `X_2` humidité libre **finale** [toutes sans dimension, base
//! sèche] ; `t_c`, `t_f`, `t` durées [s].
//!
//! **Limite honnête** : ce module décrit un séchage **convectif à deux
//! périodes** — une période à **allure constante** `N_c` suivie d'une période à
//! **allure décroissante LINÉAIRE** passant par l'**origine** (`N = N_c·X/X_c`),
//! avec les teneurs en eau exprimées en **BASE SÈCHE**. L'allure constante
//! `N_c`, l'aire d'échange `A`, la masse de solide sec `M_s` et les humidités
//! **critique** `X_c` et d'**équilibre** `X*` sont **FOURNIES** par l'appelant
//! (essais, tables). Aucune propriété physique (enthalpies, volatilités,
//! coefficients de partage, constantes cinétiques, diffusivités, coefficients de
//! transfert…) n'est **jamais** supposée « par défaut ». Ce modèle **ne décrit
//! pas** la diffusion interne détaillée ni un profil d'allure décroissante non
//! linéaire.

/// Humidité **libre** en base sèche `X = X_t − X*` (sans dimension), écart entre
/// la teneur en eau totale et la teneur en eau d'équilibre. C'est l'eau
/// susceptible d'être évaporée dans les conditions de séchage considérées.
///
/// `moisture_content` (X_t) teneur en eau totale et `equilibrium_moisture` (X*)
/// teneur en eau d'équilibre, toutes deux en **base sèche** [kg eau · kg solide
/// sec⁻¹].
///
/// Panique si `moisture_content < 0`, `equilibrium_moisture < 0` ou si
/// `moisture_content < equilibrium_moisture` (humidité libre négative).
pub fn drying_free_moisture(moisture_content: f64, equilibrium_moisture: f64) -> f64 {
    assert!(
        moisture_content >= 0.0,
        "X_t ≥ 0 requis (teneur en eau totale, base sèche)"
    );
    assert!(
        equilibrium_moisture >= 0.0,
        "X* ≥ 0 requis (teneur en eau d'équilibre, base sèche)"
    );
    assert!(
        moisture_content >= equilibrium_moisture,
        "X_t ≥ X* requis (humidité libre X = X_t − X* ≥ 0)"
    );
    moisture_content - equilibrium_moisture
}

/// Durée de la période à **allure constante**
/// `t_c = M_s·(X_1 − X_c)/(A·N_c)` (s), obtenue en évaporant l'eau libre de
/// `X_1` à `X_c` à flux constant `N_c`.
///
/// `dry_solid_mass` (M_s) masse de solide sec [kg], `drying_area` (A) aire
/// d'échange [m²], `constant_rate` (N_c) flux de séchage [kg eau · m⁻² · s⁻¹],
/// `initial_free_moisture` (X_1) et `critical_free_moisture` (X_c) humidités
/// libres initiale et critique [sans dimension, base sèche].
///
/// Panique si `dry_solid_mass <= 0`, `drying_area <= 0`, `constant_rate <= 0`,
/// `critical_free_moisture < 0` ou `initial_free_moisture < critical_free_moisture`.
pub fn drying_constant_rate_time(
    dry_solid_mass: f64,
    drying_area: f64,
    constant_rate: f64,
    initial_free_moisture: f64,
    critical_free_moisture: f64,
) -> f64 {
    assert!(dry_solid_mass > 0.0, "M_s > 0 requis (masse de solide sec)");
    assert!(drying_area > 0.0, "A > 0 requis (aire d'échange)");
    assert!(constant_rate > 0.0, "N_c > 0 requis (allure constante)");
    assert!(
        critical_free_moisture >= 0.0,
        "X_c ≥ 0 requis (humidité libre critique)"
    );
    assert!(
        initial_free_moisture >= critical_free_moisture,
        "X_1 ≥ X_c requis (l'allure constante réduit l'humidité libre)"
    );
    dry_solid_mass * (initial_free_moisture - critical_free_moisture)
        / (drying_area * constant_rate)
}

/// Durée de la période à **allure décroissante LINÉAIRE** passant par l'origine
/// `t_f = M_s·X_c·ln(X_c/X_2)/(A·N_c)` (s). L'allure y vaut `N = N_c·X/X_c` ;
/// l'intégration de `X_c` à `X_2` donne le logarithme.
///
/// `dry_solid_mass` (M_s) masse de solide sec [kg], `drying_area` (A) aire
/// d'échange [m²], `constant_rate` (N_c) allure au point critique [kg eau ·
/// m⁻² · s⁻¹], `critical_free_moisture` (X_c) et `final_free_moisture` (X_2)
/// humidités libres critique et finale [sans dimension, base sèche].
///
/// Panique si `dry_solid_mass <= 0`, `drying_area <= 0`, `constant_rate <= 0`,
/// `critical_free_moisture <= 0`, `final_free_moisture <= 0` ou
/// `critical_free_moisture < final_free_moisture`.
pub fn drying_falling_rate_time(
    dry_solid_mass: f64,
    drying_area: f64,
    constant_rate: f64,
    critical_free_moisture: f64,
    final_free_moisture: f64,
) -> f64 {
    assert!(dry_solid_mass > 0.0, "M_s > 0 requis (masse de solide sec)");
    assert!(drying_area > 0.0, "A > 0 requis (aire d'échange)");
    assert!(
        constant_rate > 0.0,
        "N_c > 0 requis (allure au point critique)"
    );
    assert!(
        critical_free_moisture > 0.0,
        "X_c > 0 requis (humidité libre critique)"
    );
    assert!(
        final_free_moisture > 0.0,
        "X_2 > 0 requis (humidité libre finale, modèle linéaire vers l'origine)"
    );
    assert!(
        critical_free_moisture >= final_free_moisture,
        "X_c ≥ X_2 requis (le séchage réduit l'humidité libre)"
    );
    dry_solid_mass * critical_free_moisture * (critical_free_moisture / final_free_moisture).ln()
        / (drying_area * constant_rate)
}

/// Durée **totale** de séchage `t = t_c + t_f` (s), somme des durées des périodes
/// à allure constante et à allure décroissante.
///
/// `constant_rate_time` (t_c) et `falling_rate_time` (t_f) durées des deux
/// périodes [s].
///
/// Panique si `constant_rate_time < 0` ou `falling_rate_time < 0`.
pub fn drying_total_time(constant_rate_time: f64, falling_rate_time: f64) -> f64 {
    assert!(
        constant_rate_time >= 0.0,
        "t_c ≥ 0 requis (durée allure constante)"
    );
    assert!(
        falling_rate_time >= 0.0,
        "t_f ≥ 0 requis (durée allure décroissante)"
    );
    constant_rate_time + falling_rate_time
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn free_moisture_definition_and_zero_limit() {
        // X_t = 0.30, X* = 0.05 ⇒ X = 0.25 (base sèche).
        assert_relative_eq!(
            drying_free_moisture(0.30_f64, 0.05_f64),
            0.25,
            max_relative = 1e-12
        );
        // Cas limite : à l'équilibre (X_t = X*), l'humidité libre est nulle.
        assert_relative_eq!(
            drying_free_moisture(0.12_f64, 0.12_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn constant_rate_time_realistic_and_mass_proportional() {
        // M_s = 100 kg, A = 10 m², N_c = 0.001 kg·m⁻²·s⁻¹, X_1 = 0.25, X_c = 0.10.
        // t_c = 100·(0.25 − 0.10)/(10·0.001) = 100·0.15/0.01 = 15/0.01 = 1500 s.
        assert_relative_eq!(
            drying_constant_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.25_f64, 0.10_f64),
            1500.0,
            max_relative = 1e-9
        );
        // Proportionnalité : doubler la masse de solide sec double la durée.
        let base = drying_constant_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.25_f64, 0.10_f64);
        let doubled = drying_constant_rate_time(200.0_f64, 10.0_f64, 0.001_f64, 0.25_f64, 0.10_f64);
        assert_relative_eq!(doubled, 2.0 * base, max_relative = 1e-12);
    }

    #[test]
    fn falling_rate_time_realistic_and_zero_at_critical() {
        // M_s = 100 kg, A = 10 m², N_c = 0.001, X_c = 0.10, X_2 = 0.02.
        // t_f = 100·0.10·ln(0.10/0.02)/(10·0.001) = 10·ln(5)/0.01
        //     = 10·1.6094379124341003/0.01 = 1609.4379124341003 s.
        assert_relative_eq!(
            drying_falling_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.10_f64, 0.02_f64),
            1609.4379124341003,
            max_relative = 1e-3
        );
        // Cas limite : si X_2 = X_c, ln(1) = 0 ⇒ aucune période décroissante.
        assert_relative_eq!(
            drying_falling_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.10_f64, 0.10_f64),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn falling_rate_time_area_inverse_proportional() {
        // t_f ∝ 1/A : doubler l'aire d'échange divise la durée par deux.
        let base = drying_falling_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.10_f64, 0.02_f64);
        let double_area =
            drying_falling_rate_time(100.0_f64, 20.0_f64, 0.001_f64, 0.10_f64, 0.02_f64);
        assert_relative_eq!(double_area, base / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn total_time_is_sum_of_periods() {
        // t = t_c + t_f = 1500 + 1609.4379124341003 = 3109.4379124341003 s.
        let t_c = drying_constant_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.25_f64, 0.10_f64);
        let t_f = drying_falling_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.10_f64, 0.02_f64);
        assert_relative_eq!(
            drying_total_time(t_c, t_f),
            3109.4379124341003,
            max_relative = 1e-3
        );
        // Identité : la durée totale vaut bien la somme des deux durées.
        assert_relative_eq!(drying_total_time(t_c, t_f), t_c + t_f, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "X_c ≥ X_2 requis")]
    fn falling_rate_time_panics_when_final_above_critical() {
        // X_2 > X_c ⇒ ln < 0 (humidité qui augmente) : entrée invalide.
        let _ = drying_falling_rate_time(100.0_f64, 10.0_f64, 0.001_f64, 0.10_f64, 0.20_f64);
    }
}
