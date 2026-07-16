//! Hydrologie urbaine — **méthode rationnelle** pour l'estimation du débit de
//! pointe de ruissellement d'un petit bassin versant, coefficient de
//! ruissellement pondéré, temps de concentration de Kirpich et volume ruisselé.
//!
//! ```text
//! débit de pointe        Q  = C·i·A                       (méthode rationnelle)
//! coefficient pondéré    C  = Σ(Aᵢ·Cᵢ) / Σ(Aᵢ)            (pondération par surface)
//! temps de concentration tc = 0,0195·L^0,77·S^(−0,385)    (Kirpich, minutes)
//! volume ruisselé        Vol = Q·Δt                       (approximation)
//! ```
//!
//! `Q` débit de pointe (m³/s), `C` coefficient de ruissellement (sans
//! dimension, 0 ≤ C ≤ 1), `i` intensité de pluie (m/s), `A` surface du bassin
//! versant (m²), `Aᵢ` surface de la sous-zone `i` (m²), `Cᵢ` coefficient de la
//! sous-zone `i` (sans dimension), `tc` temps de concentration (minutes), `L`
//! longueur du plus long chemin hydraulique (m), `S` pente moyenne du chemin
//! (m/m), `Vol` volume ruisselé (m³), `Δt` durée de l'événement (s).
//!
//! **Convention** : SI strict et cohérent — mètres (m), secondes (s), donc `i`
//! en m/s et `Q` en m³/s. **Exception documentée** : la corrélation empirique
//! de Kirpich impose ses propres unités — longueur `L` en mètres, pente `S` en
//! m/m, résultat `tc` en **minutes** (le facteur 0,0195 est calibré pour ces
//! unités précises). Types `f64`.
//!
//! **Limite honnête** : la **méthode rationnelle** ne vaut que pour les
//! **petits bassins versants** (typiquement < quelques km²) et suppose une
//! **intensité de pluie uniforme** et un **régime permanent** atteint au bout
//! du temps de concentration. Le coefficient de ruissellement `C` (nature des
//! surfaces, imperméabilisation) et l'intensité de pluie `i` — lue sur une
//! **courbe IDF** (Intensité-Durée-Fréquence) pour une durée égale au temps de
//! concentration et une période de retour donnée — sont **fournis par
//! l'appelant** d'après les guides et normes applicables, jamais des valeurs
//! « par défaut » inventées. Le temps de concentration de Kirpich est une
//! **corrélation empirique** (unités spécifiques ci-dessus) : ce n'est pas une
//! loi physique. Ce module n'est **pas adapté aux grands bassins**, aux
//! régimes non permanents, ni au routage hydraulique détaillé (hydrogramme).

/// Débit de pointe de ruissellement par la méthode rationnelle
/// `Q = C·i·A` (m³/s), avec `i` en m/s et `A` en m².
///
/// Panique si `runoff_coefficient` n'est pas dans `[0, 1]`, si
/// `rainfall_intensity < 0` ou si `catchment_area < 0`.
pub fn runoff_peak_flow(
    runoff_coefficient: f64,
    rainfall_intensity: f64,
    catchment_area: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&runoff_coefficient),
        "le coefficient de ruissellement C doit être compris entre 0 et 1"
    );
    assert!(
        rainfall_intensity >= 0.0,
        "l'intensité de pluie i doit être positive ou nulle"
    );
    assert!(
        catchment_area >= 0.0,
        "la surface du bassin versant A doit être positive ou nulle"
    );
    runoff_coefficient * rainfall_intensity * catchment_area
}

/// Coefficient de ruissellement composite pondéré par les surfaces
/// `C = Σ(Aᵢ·Cᵢ) / Σ(Aᵢ)` (sans dimension).
///
/// Panique si les tableaux `areas` et `coefficients` sont vides ou de
/// longueurs différentes, si une surface est négative, si un coefficient
/// n'est pas dans `[0, 1]`, ou si la somme des surfaces est nulle.
pub fn runoff_composite_coefficient(areas: &[f64], coefficients: &[f64]) -> f64 {
    assert!(
        !areas.is_empty(),
        "le tableau des surfaces ne doit pas être vide"
    );
    assert!(
        areas.len() == coefficients.len(),
        "les tableaux des surfaces et des coefficients doivent avoir la même longueur"
    );
    assert!(
        areas.iter().all(|&area| area >= 0.0),
        "chaque surface Aᵢ doit être positive ou nulle"
    );
    assert!(
        coefficients
            .iter()
            .all(|&coefficient| (0.0..=1.0).contains(&coefficient)),
        "chaque coefficient Cᵢ doit être compris entre 0 et 1"
    );
    let total_area: f64 = areas.iter().sum();
    assert!(
        total_area > 0.0,
        "la somme des surfaces doit être strictement positive"
    );
    let weighted: f64 = areas
        .iter()
        .zip(coefficients.iter())
        .map(|(&area, &coefficient)| area * coefficient)
        .sum();
    weighted / total_area
}

/// Temps de concentration de Kirpich `tc = 0,0195·L^0,77·S^(−0,385)`
/// (minutes), corrélation empirique avec `L` en mètres et `S` en m/m.
///
/// Panique si `length <= 0` ou si `slope <= 0`.
pub fn runoff_time_of_concentration_kirpich(length: f64, slope: f64) -> f64 {
    assert!(
        length > 0.0,
        "la longueur L du chemin hydraulique doit être strictement positive"
    );
    assert!(
        slope > 0.0,
        "la pente S du chemin doit être strictement positive"
    );
    0.0195_f64 * length.powf(0.77) * slope.powf(-0.385)
}

/// Volume ruisselé approché `Vol = Q·Δt` (m³), produit du débit de pointe
/// par la durée de l'événement (en secondes).
///
/// Panique si `peak_flow < 0` ou si `duration < 0`.
pub fn runoff_volume(peak_flow: f64, duration: f64) -> f64 {
    assert!(
        peak_flow >= 0.0,
        "le débit de pointe Q doit être positif ou nul"
    );
    assert!(duration >= 0.0, "la durée Δt doit être positive ou nulle");
    peak_flow * duration
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn peak_flow_is_proportional_to_each_factor() {
        // Q = C·i·A : linéarité stricte en chacun des trois facteurs.
        let q = runoff_peak_flow(0.5, 1.0e-5, 2000.0);
        // Doubler le coefficient double le débit.
        assert_relative_eq!(
            runoff_peak_flow(1.0, 1.0e-5, 2000.0),
            2.0 * q,
            max_relative = 1e-12
        );
        // Doubler la surface double le débit.
        assert_relative_eq!(
            runoff_peak_flow(0.5, 1.0e-5, 4000.0),
            2.0 * q,
            max_relative = 1e-12
        );
        // Intensité nulle : aucun ruissellement.
        assert_relative_eq!(runoff_peak_flow(0.5, 0.0, 2000.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn composite_of_uniform_coefficients_equals_that_coefficient() {
        // Si toutes les sous-zones ont le même C, le composite vaut ce C,
        // quelles que soient les surfaces (moyenne pondérée d'une constante).
        let c = runoff_composite_coefficient(&[1200.0, 300.0, 500.0], &[0.7, 0.7, 0.7]);
        assert_relative_eq!(c, 0.7, max_relative = 1e-12);
        // Une seule sous-zone : le composite est son propre coefficient.
        let single = runoff_composite_coefficient(&[850.0], &[0.35]);
        assert_relative_eq!(single, 0.35, max_relative = 1e-12);
    }

    #[test]
    fn composite_worked_case() {
        // Bassin de 2000 m² : 1000 m² imperméable (C = 0,90),
        // 500 m² semi-perméable (C = 0,30), 500 m² espace vert (C = 0,20).
        // Σ(Aᵢ·Cᵢ) = 900 + 150 + 100 = 1150 ; Σ(Aᵢ) = 2000.
        // C = 1150 / 2000 = 0,575.
        let c = runoff_composite_coefficient(&[1000.0, 500.0, 500.0], &[0.90, 0.30, 0.20]);
        assert_relative_eq!(c, 0.575, max_relative = 1e-9);
    }

    #[test]
    fn kirpich_scaling_and_worked_case() {
        // Loi de puissance : multiplier L par 2 multiplie tc par 2^0,77.
        let tc1 = runoff_time_of_concentration_kirpich(300.0, 0.01);
        let tc2 = runoff_time_of_concentration_kirpich(600.0, 0.01);
        assert_relative_eq!(tc2 / tc1, 2.0_f64.powf(0.77), max_relative = 1e-9);
        // Cas chiffré : L = 300 m, S = 0,01 m/m.
        // L^0,77 = 300^0,77 ≈ 80,795.
        // S^(−0,385) = 0,01^(−0,385) = 10^0,77 ≈ 5,8884.
        // tc = 0,0195 · 80,795 · 5,8884 ≈ 9,277 minutes.
        assert_relative_eq!(tc1, 9.277, max_relative = 1e-3);
    }

    #[test]
    fn volume_is_flow_times_duration() {
        // Vol = Q·Δt : Q = 0,016 m³/s pendant Δt = 600 s → 9,6 m³.
        assert_relative_eq!(runoff_volume(0.016, 600.0), 9.6, max_relative = 1e-12);
        // Doubler la durée double le volume.
        assert_relative_eq!(
            runoff_volume(0.016, 1200.0),
            2.0 * runoff_volume(0.016, 600.0),
            max_relative = 1e-12
        );
    }

    #[test]
    fn end_to_end_rational_method() {
        // Chaîne complète : composite → débit de pointe → volume.
        // C = 0,575 ; i = 50 mm/h = 0,05/3600 = 1,388889e-5 m/s ; A = 2000 m².
        // Q = 0,575 · 1,388889e-5 · 2000 = 0,0159722 m³/s.
        // Sur Δt = 600 s : Vol = 0,0159722 · 600 = 9,58333 m³.
        let c = runoff_composite_coefficient(&[1000.0, 500.0, 500.0], &[0.90, 0.30, 0.20]);
        let intensity = 0.05 / 3600.0;
        let q = runoff_peak_flow(c, intensity, 2000.0);
        assert_relative_eq!(q, 0.0159722, max_relative = 1e-3);
        let volume = runoff_volume(q, 600.0);
        assert_relative_eq!(volume, 9.58333, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(
        expected = "les tableaux des surfaces et des coefficients doivent avoir la même longueur"
    )]
    fn mismatched_arrays_panics() {
        let _ = runoff_composite_coefficient(&[1000.0, 500.0], &[0.9]);
    }
}
