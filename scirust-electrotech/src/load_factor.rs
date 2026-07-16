//! **Facteurs de charge** — module d'indicateurs de dimensionnement d'une
//! installation électrique caractérisant l'aplatissement de la courbe de charge,
//! la simultanéité et le foisonnement des pointes.
//!
//! ```text
//! facteur de charge        LF = P_moyenne / P_pointe
//! facteur de demande       DF = D_max / P_installée
//! facteur de diversité     FD = Σ D_i,max / D_max_coïncidente
//! facteur d'utilisation    UF = D_max / P_capacité
//! puissance moyenne        P_moyenne = E / t
//! ```
//!
//! `LF` facteur de charge (sans dimension, ∈ ]0, 1]), `P_moyenne` puissance
//! moyenne appelée (W ou VA), `P_pointe` puissance de pointe (W ou VA),
//! `DF` facteur de demande/simultanéité (sans dimension), `D_max` demande
//! maximale (W ou VA), `P_installée` puissance totale installée/raccordée
//! (W ou VA), `FD` facteur de diversité (sans dimension, ≥ 1),
//! `Σ D_i,max` somme des demandes maximales individuelles (W ou VA),
//! `D_max_coïncidente` demande maximale coïncidente au poste (W ou VA),
//! `UF` facteur d'utilisation (sans dimension), `P_capacité` capacité de
//! l'installation/du poste (W ou VA), `E` énergie consommée sur la période
//! (W·h), `t` durée de la période (h).
//!
//! **Convention** : SI ; puissances homogènes (toutes en W, ou toutes en var,
//! ou toutes en VA), énergie en W·h et durée en h de sorte que `E / t` redonne
//! une puissance en W. **Limite honnête** : ce sont des **indicateurs de
//! dimensionnement d'installation** ; les charges, puissances, énergies et
//! durées sont **fournies par l'appelant** (relevé de comptage, courbe de
//! charge mesurée, estimation prévisionnelle) — aucune valeur « typique » n'est
//! inventée. Le facteur de charge mesure l'**aplatissement de la courbe de
//! charge** (proche de 1 = charge quasi constante), le facteur de diversité
//! (≥ 1) traduit le **foisonnement** des pointes qui ne se produisent pas
//! simultanément. Les grandeurs comparées doivent être **cohérentes en
//! puissance** (même nature, même unité) ; ce module ne vérifie pas cette
//! homogénéité, il suppose des rapports de puissances de même nature.

/// Facteur de charge `LF = average_load / peak_load` (sans dimension), rapport
/// de la puissance moyenne à la puissance de pointe sur la période considérée.
///
/// Panique si `average_load < 0` ou si `peak_load <= 0` (division par zéro ou
/// puissance de pointe non physique).
pub fn loadfactor_load_factor(average_load: f64, peak_load: f64) -> f64 {
    assert!(
        average_load >= 0.0,
        "la puissance moyenne average_load doit être ≥ 0"
    );
    assert!(
        peak_load > 0.0,
        "la puissance de pointe peak_load doit être strictement positive"
    );
    average_load / peak_load
}

/// Facteur de demande (ou de simultanéité) `DF = maximum_demand /
/// total_connected_load` (sans dimension), rapport de la demande maximale à la
/// puissance totale raccordée.
///
/// Panique si `maximum_demand < 0` ou si `total_connected_load <= 0` (division
/// par zéro ou puissance raccordée non physique).
pub fn loadfactor_demand_factor(maximum_demand: f64, total_connected_load: f64) -> f64 {
    assert!(
        maximum_demand >= 0.0,
        "la demande maximale maximum_demand doit être ≥ 0"
    );
    assert!(
        total_connected_load > 0.0,
        "la puissance raccordée total_connected_load doit être strictement positive"
    );
    maximum_demand / total_connected_load
}

/// Facteur de diversité `FD = sum_of_individual_maximum_demands /
/// coincident_maximum_demand` (sans dimension, ≥ 1), traduisant le foisonnement
/// des pointes individuelles qui ne coïncident pas.
///
/// Panique si `sum_of_individual_maximum_demands < 0` ou si
/// `coincident_maximum_demand <= 0` (division par zéro ou demande coïncidente
/// non physique).
pub fn loadfactor_diversity_factor(
    sum_of_individual_maximum_demands: f64,
    coincident_maximum_demand: f64,
) -> f64 {
    assert!(
        sum_of_individual_maximum_demands >= 0.0,
        "la somme des demandes maximales sum_of_individual_maximum_demands doit être ≥ 0"
    );
    assert!(
        coincident_maximum_demand > 0.0,
        "la demande coïncidente coincident_maximum_demand doit être strictement positive"
    );
    sum_of_individual_maximum_demands / coincident_maximum_demand
}

/// Facteur d'utilisation `UF = maximum_demand / plant_capacity` (sans
/// dimension), rapport de la demande maximale à la capacité de l'installation.
///
/// Panique si `maximum_demand < 0` ou si `plant_capacity <= 0` (division par
/// zéro ou capacité non physique).
pub fn loadfactor_utilization_factor(maximum_demand: f64, plant_capacity: f64) -> f64 {
    assert!(
        maximum_demand >= 0.0,
        "la demande maximale maximum_demand doit être ≥ 0"
    );
    assert!(
        plant_capacity > 0.0,
        "la capacité plant_capacity doit être strictement positive"
    );
    maximum_demand / plant_capacity
}

/// Puissance moyenne à partir de l'énergie `P_moyenne = energy_consumed /
/// period_hours` (W si l'énergie est en W·h et la durée en h).
///
/// Panique si `energy_consumed < 0` ou si `period_hours <= 0` (division par
/// zéro ou durée non physique).
pub fn loadfactor_average_load_from_energy(energy_consumed: f64, period_hours: f64) -> f64 {
    assert!(
        energy_consumed >= 0.0,
        "l'énergie consommée energy_consumed doit être ≥ 0"
    );
    assert!(
        period_hours > 0.0,
        "la durée period_hours doit être strictement positive"
    );
    energy_consumed / period_hours
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn load_factor_bounds_and_unity() {
        // Cas limite : charge parfaitement constante (moyenne = pointe) → LF = 1.
        assert_relative_eq!(loadfactor_load_factor(500.0, 500.0), 1.0, epsilon = 1e-12);
        // Cas chiffré : moyenne 300 kW, pointe 500 kW → LF = 300/500 = 0,6.
        assert_relative_eq!(
            loadfactor_load_factor(300.0e3, 500.0e3),
            0.6,
            epsilon = 1e-12
        );
    }

    #[test]
    fn load_factor_via_average_from_energy_matches_direct() {
        // Réciprocité : calculer la moyenne depuis l'énergie puis le facteur de
        // charge doit coïncider avec l'application directe de la définition.
        //   E = 720 kW·h sur t = 24 h → P_moyenne = 720/24 = 30 kW.
        //   pointe = 50 kW → LF = 30/50 = 0,6.
        let avg = loadfactor_average_load_from_energy(720.0e3, 24.0);
        assert_relative_eq!(avg, 30.0e3, epsilon = 1e-9);
        let lf = loadfactor_load_factor(avg, 50.0e3);
        assert_relative_eq!(lf, 0.6, epsilon = 1e-12);
    }

    #[test]
    fn diversity_factor_at_least_one() {
        // Le facteur de diversité vaut 1 quand toutes les pointes coïncident,
        // et croît avec le foisonnement.
        //   Σ D_i = 100 kW, coïncidente = 100 kW → FD = 1 (aucun foisonnement).
        assert_relative_eq!(
            loadfactor_diversity_factor(100.0e3, 100.0e3),
            1.0,
            epsilon = 1e-12
        );
        //   Σ D_i = 120 kW, coïncidente = 80 kW → FD = 120/80 = 1,5 (≥ 1).
        let fd = loadfactor_diversity_factor(120.0e3, 80.0e3);
        assert_relative_eq!(fd, 1.5, epsilon = 1e-12);
        assert!(fd >= 1.0);
    }

    #[test]
    fn demand_and_utilization_factors_numeric() {
        // Cas chiffré facteur de demande : demande max 45 kW pour 60 kW
        //   raccordés → DF = 45/60 = 0,75.
        assert_relative_eq!(
            loadfactor_demand_factor(45.0e3, 60.0e3),
            0.75,
            epsilon = 1e-12
        );
        // Cas chiffré facteur d'utilisation : demande max 45 kW pour un poste de
        //   90 kW → UF = 45/90 = 0,5.
        assert_relative_eq!(
            loadfactor_utilization_factor(45.0e3, 90.0e3),
            0.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn average_load_scales_inversely_with_period() {
        // Proportionnalité : à énergie fixée, la puissance moyenne est
        // inversement proportionnelle à la durée ; doubler la période divise la
        // moyenne par deux.
        let p1 = loadfactor_average_load_from_energy(240.0e3, 12.0);
        let p2 = loadfactor_average_load_from_energy(240.0e3, 24.0);
        assert_relative_eq!(p1, 20.0e3, epsilon = 1e-9);
        assert_relative_eq!(p2, p1 / 2.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la puissance de pointe peak_load doit être strictement positive")]
    fn zero_peak_load_panics() {
        loadfactor_load_factor(100.0, 0.0);
    }
}
