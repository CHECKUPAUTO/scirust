//! **Effet couronne** — apparition de l'ionisation à la surface des
//! conducteurs d'une ligne aérienne et estimation de la perte associée par la
//! **formule empirique de Peek**.
//!
//! ```text
//! facteur de densité de l'air    δ    = 0.386·b / T
//! tension critique disruptive    V_c  = E₀·m·δ·r·ln(D / r)
//! perte couronne (par phase)     P    = (243/δ)·(f + 25)·√(r/D)·(V − V_c)²·10⁻⁵
//! couronne active                             V > V_c
//! ```
//!
//! `δ` facteur de densité de l'air (sans dimension), `b` pression atmosphérique
//! (cmHg), `T` température absolue (K), `E₀` rigidité diélectrique de l'air
//! (kV/cm, ≈ 21,1 kV/cm efficace ou ≈ 30 kV/cm crête), `m` facteur
//! d'irrégularité de surface (sans dimension, `≤ 1`), `r` rayon du conducteur
//! (cm), `D` distance entre conducteurs (cm), `V` tension simple de service
//! (phase-neutre, kV efficace), `V_c` tension critique disruptive (kV
//! efficace), `f` fréquence (Hz), `P` perte couronne par phase (kW/km).
//!
//! **Convention** : ces formules de Peek sont **empiriques** et s'emploient
//! dans leurs unités historiques — `E₀` en kV/cm, `r` et `D` en cm, tensions
//! en kV efficaces, `P` en kW/km, fréquence en Hz ; l'angle implicite du régime
//! est en **radians**. La formule usuelle du facteur de densité s'écrit aussi
//! `δ = 3.92·b/(273+t)` avec `b` en cmHg et `t` en °C ; on retient ici la forme
//! équivalente `δ = 0.386·b/T` avec `T` en K.
//!
//! **Limite honnête** : la **formule de Peek est empirique** (temps **sec**,
//! ligne aérienne) et ne fournit qu'une **estimation indicative**. La couronne
//! n'apparaît que si la tension de service **dépasse** la tension critique
//! disruptive `V_c`. Le facteur d'irrégularité de surface `m` (état du
//! conducteur : `1` lisse, `< 1` toronné ou rugueux), la rigidité diélectrique
//! de l'air `E₀` (≈ 21,1 kV/cm efficace / ≈ 30 kV/cm crête) et le facteur de
//! densité `δ` (ou la pression `b` et la température `T` qui le déterminent)
//! sont **fournis par l'appelant** — aucune valeur « par défaut » n'est
//! inventée. La pluie, la neige, le brouillard et la pollution abaissent
//! fortement `V_c` et ne sont **pas** modélisés.

/// Facteur de densité de l'air `δ = 0.386·b / T` (sans dimension), avec la
/// pression `b` en cmHg et la température absolue `T` en kelvins.
///
/// Panique si `pressure <= 0` ou si `temperature <= 0` (grandeurs physiques
/// strictement positives ; division par zéro).
pub fn corona_air_density_factor(pressure: f64, temperature: f64) -> f64 {
    assert!(
        pressure > 0.0,
        "la pression b doit être strictement positive (cmHg)"
    );
    assert!(
        temperature > 0.0,
        "la température T doit être strictement positive (K)"
    );
    0.386 * pressure / temperature
}

/// Tension critique disruptive `V_c = E₀·m·δ·r·ln(D/r)` (kV efficaces),
/// tension de service à partir de laquelle la couronne apparaît.
///
/// `dielectric_strength` `E₀` en kV/cm, `irregularity_factor` `m` sans
/// dimension, `air_density_factor` `δ` sans dimension, `conductor_radius` `r`
/// et `spacing` `D` en cm.
///
/// Panique si `dielectric_strength <= 0`, si `irregularity_factor <= 0`, si
/// `air_density_factor <= 0`, si `conductor_radius <= 0` ou si
/// `spacing <= conductor_radius` (le logarithme `ln(D/r)` doit être défini et
/// positif : les conducteurs sont distants d'au moins leur rayon).
pub fn corona_critical_disruptive_voltage(
    irregularity_factor: f64,
    air_density_factor: f64,
    conductor_radius: f64,
    spacing: f64,
    dielectric_strength: f64,
) -> f64 {
    assert!(
        dielectric_strength > 0.0,
        "la rigidité diélectrique E₀ doit être strictement positive (kV/cm)"
    );
    assert!(
        irregularity_factor > 0.0,
        "le facteur d'irrégularité m doit être strictement positif"
    );
    assert!(
        air_density_factor > 0.0,
        "le facteur de densité δ doit être strictement positif"
    );
    assert!(
        conductor_radius > 0.0,
        "le rayon r doit être strictement positif (cm)"
    );
    assert!(
        spacing > conductor_radius,
        "la distance D doit être strictement supérieure au rayon r (ln(D/r) > 0)"
    );
    dielectric_strength
        * irregularity_factor
        * air_density_factor
        * conductor_radius
        * (spacing / conductor_radius).ln()
}

/// Perte par effet couronne et par phase selon la **formule empirique de
/// Peek** `P = (243/δ)·(f+25)·√(r/D)·(V − V_c)²·10⁻⁵` (kW/km).
///
/// `air_density_factor` `δ` sans dimension, `frequency` `f` en Hz,
/// `conductor_radius` `r` et `spacing` `D` en cm, `phase_voltage` `V` (tension
/// simple) et `critical_voltage` `V_c` en kV efficaces.
///
/// La perte n'a de sens physique que si `V > V_c` ; sous le seuil, la valeur
/// renvoyée reste positive (terme carré) mais ne correspond à **aucune**
/// couronne réelle — vérifier au préalable avec [`corona_is_active`].
///
/// Panique si `air_density_factor <= 0`, si `frequency < 0`, si
/// `conductor_radius <= 0` ou si `spacing <= 0` (division par zéro, racine d'un
/// nombre négatif ou fréquence sans sens physique).
pub fn corona_peek_loss(
    frequency: f64,
    air_density_factor: f64,
    conductor_radius: f64,
    spacing: f64,
    phase_voltage: f64,
    critical_voltage: f64,
) -> f64 {
    assert!(
        air_density_factor > 0.0,
        "le facteur de densité δ doit être strictement positif"
    );
    assert!(frequency >= 0.0, "la fréquence f doit être ≥ 0 (Hz)");
    assert!(
        conductor_radius > 0.0,
        "le rayon r doit être strictement positif (cm)"
    );
    assert!(
        spacing > 0.0,
        "la distance D doit être strictement positive (cm)"
    );
    let delta_voltage = phase_voltage - critical_voltage;
    (243.0 / air_density_factor)
        * (frequency + 25.0)
        * (conductor_radius / spacing).sqrt()
        * delta_voltage.powi(2)
        * 1e-5
}

/// Indique si la couronne est **active** : la tension de service simple `V`
/// dépasse la tension critique disruptive `V_c` (`V > V_c`).
///
/// `phase_voltage` `V` et `critical_voltage` `V_c` en kV efficaces.
///
/// Panique si `phase_voltage < 0` ou si `critical_voltage < 0` (tensions
/// efficaces sans sens physique si négatives).
pub fn corona_is_active(phase_voltage: f64, critical_voltage: f64) -> bool {
    assert!(phase_voltage >= 0.0, "la tension V doit être ≥ 0 (kV)");
    assert!(
        critical_voltage >= 0.0,
        "la tension critique V_c doit être ≥ 0 (kV)"
    );
    phase_voltage > critical_voltage
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn air_density_factor_reference_conditions() {
        // Conditions normalisées de Peek : b = 76 cmHg, t = 25 °C soit
        // T = 298,15 K. δ = 0.386·76/298.15. On recalcule le littéral :
        // 0.386·76 = 29.336 ; 29.336/298.15 = 0.098393… ≈ 0.0984.
        let delta = corona_air_density_factor(76.0, 298.15);
        assert_relative_eq!(delta, 0.0983833_f64, epsilon = 1e-3);
    }

    #[test]
    fn air_density_factor_proportional_to_pressure() {
        // Proportionnalité : à température fixée, δ ∝ b. Doubler la pression
        // double le facteur de densité.
        let d1 = corona_air_density_factor(38.0, 300.0);
        let d2 = corona_air_density_factor(76.0, 300.0);
        assert_relative_eq!(d2, 2.0 * d1, epsilon = 1e-12);
    }

    #[test]
    fn critical_voltage_vanishes_when_log_is_unity_scaling() {
        // Cas chiffré : m = 1, δ = 1, r = 1 cm, D = r·e (ln(D/r) = 1),
        // E₀ = 21.1 kV/cm. V_c = 21.1·1·1·1·1 = 21.1 kV. On recalcule :
        // le produit des facteurs unitaires vaut 21.1, ln(e) = 1 → 21.1.
        let d = core::f64::consts::E; // D/r = e
        let vc = corona_critical_disruptive_voltage(1.0, 1.0, 1.0, d, 21.1);
        assert_relative_eq!(vc, 21.1_f64, epsilon = 1e-9);
    }

    #[test]
    fn critical_voltage_proportional_to_irregularity_factor() {
        // Proportionnalité : V_c ∝ m à géométrie et densité fixées. Un
        // conducteur toronné (m = 0.85) abaisse V_c de 15 %.
        let vc_smooth = corona_critical_disruptive_voltage(1.0, 0.95, 1.0, 200.0, 21.1);
        let vc_strand = corona_critical_disruptive_voltage(0.85, 0.95, 1.0, 200.0, 21.1);
        assert_relative_eq!(vc_strand, 0.85 * vc_smooth, epsilon = 1e-9);
    }

    #[test]
    fn peek_loss_worked_case() {
        // Cas chiffré : δ = 1, f = 50 Hz, r/D = 1 (√ = 1), V − V_c = 100 kV.
        // P = (243/1)·(50+25)·1·100²·1e-5.
        // Recalcul 1 : 243·75 = 18225 ; 100² = 10000 ; 10000·1e-5 = 0.1 ;
        //             18225·0.1 = 1822.5.
        // Recalcul 2 : (50+25) = 75 ; 243·75 = 18225 ; 18225·0.1 = 1822.5.
        let p = corona_peek_loss(50.0, 1.0, 1.0, 1.0, 150.0, 50.0);
        assert_relative_eq!(p, 1822.5_f64, epsilon = 1e-3);
    }

    #[test]
    fn peek_loss_zero_at_critical_voltage() {
        // Limite : à V = V_c le terme (V − V_c)² s'annule, donc P = 0 quelle
        // que soit la géométrie ou la fréquence.
        let p = corona_peek_loss(60.0, 0.98, 1.2, 250.0, 132.0, 132.0);
        assert_relative_eq!(p, 0.0_f64, epsilon = 1e-12);
    }

    #[test]
    fn activity_threshold_matches_critical_voltage() {
        // Cohérence du seuil : la couronne est active au-dessus de V_c et
        // inactive à/sous le seuil.
        assert!(corona_is_active(140.0, 132.0));
        assert!(!corona_is_active(120.0, 132.0));
        assert!(!corona_is_active(132.0, 132.0));
    }

    #[test]
    #[should_panic(expected = "la distance D doit être strictement supérieure au rayon r")]
    fn critical_voltage_rejects_spacing_below_radius() {
        // D ≤ r rend ln(D/r) ≤ 0 : entrée rejetée.
        let _ = corona_critical_disruptive_voltage(1.0, 1.0, 2.0, 1.5, 21.1);
    }
}
