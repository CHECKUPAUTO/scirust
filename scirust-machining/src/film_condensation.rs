//! Condensation en film laminaire selon la **théorie de Nusselt** :
//! coefficient d'échange moyen sur plaque verticale isotherme, sur tube
//! horizontal, et nombre de Reynolds du film ruisselant.
//!
//! ```text
//! Plaque verticale   h = 0,943·(ρ²·g·hfg·k³/(µ·ΔT·L))^0,25
//! Tube horizontal    h = 0,729·(ρ²·g·hfg·k³/(µ·ΔT·D))^0,25
//! Reynolds du film   Re = 4·Γ/µ
//! ```
//!
//! `ρ` masse volumique du condensat (kg/m³), `g` accélération de la pesanteur
//! (m/s²), `hfg` chaleur latente de vaporisation (J/kg), `k` conductivité du
//! condensat (W/(m·K)), `µ` viscosité dynamique du condensat (Pa·s), `ΔT`
//! écart entre température de saturation et paroi (K), `L` hauteur de plaque
//! (m), `D` diamètre du tube (m), `h` coefficient d'échange moyen (W/(m²·K)),
//! `Γ` débit-masse de condensat par unité de largeur (kg/(m·s)), `Re` nombre
//! de Reynolds du film (sans dimension).
//!
//! **Convention** : SI cohérent, types `f64`.
//!
//! **Limite honnête** : ces relations décrivent un **film laminaire de
//! Nusselt** (`Re ≲ 1800`), paroi **isotherme**, **vapeur au repos**. Elles
//! **négligent la sous-refroidissement du film** (le facteur correctif de
//! Rohsenow sur `hfg` reste à la charge de l'appelant). Toutes les propriétés
//! du condensat (`ρ`, `k`, `µ`, `hfg`), évaluées à la **température de film**,
//! ainsi que `g` et `ΔT`, sont **fournies par l'appelant** : aucune valeur
//! « par défaut » n'est inventée ici.

/// Coefficient d'échange moyen en condensation en film sur **plaque verticale**
/// isotherme : `h = 0,943·(ρ²·g·hfg·k³/(µ·ΔT·L))^0,25` (W/(m²·K)).
///
/// Panique si l'un des arguments est négatif ou nul (`µ·ΔT·L` doit être
/// strictement positif).
pub fn condensation_vertical_plate_coefficient(
    density: f64,
    gravity: f64,
    latent_heat: f64,
    conductivity: f64,
    viscosity: f64,
    temperature_difference: f64,
    plate_height: f64,
) -> f64 {
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    assert!(
        latent_heat > 0.0,
        "la chaleur latente doit être strictement positive"
    );
    assert!(
        conductivity > 0.0,
        "la conductivité doit être strictement positive"
    );
    assert!(
        viscosity > 0.0,
        "la viscosité doit être strictement positive"
    );
    assert!(
        temperature_difference > 0.0,
        "l'écart de température doit être strictement positif"
    );
    assert!(
        plate_height > 0.0,
        "la hauteur de plaque doit être strictement positive"
    );
    let group = density.powi(2) * gravity * latent_heat * conductivity.powi(3)
        / (viscosity * temperature_difference * plate_height);
    0.943_f64 * group.powf(0.25)
}

/// Coefficient d'échange moyen en condensation en film sur **tube horizontal**
/// isotherme : `h = 0,729·(ρ²·g·hfg·k³/(µ·ΔT·D))^0,25` (W/(m²·K)).
///
/// Panique si l'un des arguments est négatif ou nul (`µ·ΔT·D` doit être
/// strictement positif).
pub fn condensation_horizontal_tube_coefficient(
    density: f64,
    gravity: f64,
    latent_heat: f64,
    conductivity: f64,
    viscosity: f64,
    temperature_difference: f64,
    tube_diameter: f64,
) -> f64 {
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    assert!(
        latent_heat > 0.0,
        "la chaleur latente doit être strictement positive"
    );
    assert!(
        conductivity > 0.0,
        "la conductivité doit être strictement positive"
    );
    assert!(
        viscosity > 0.0,
        "la viscosité doit être strictement positive"
    );
    assert!(
        temperature_difference > 0.0,
        "l'écart de température doit être strictement positif"
    );
    assert!(
        tube_diameter > 0.0,
        "le diamètre du tube doit être strictement positif"
    );
    let group = density.powi(2) * gravity * latent_heat * conductivity.powi(3)
        / (viscosity * temperature_difference * tube_diameter);
    0.729_f64 * group.powf(0.25)
}

/// Nombre de **Reynolds du film** de condensat ruisselant :
/// `Re = 4·Γ/µ`, où `Γ` est le débit-masse par unité de largeur (kg/(m·s)).
///
/// Panique si `mass_flow_per_width < 0` ou si `viscosity <= 0`.
pub fn condensation_film_reynolds(mass_flow_per_width: f64, viscosity: f64) -> f64 {
    assert!(
        mass_flow_per_width >= 0.0,
        "le débit-masse par unité de largeur doit être positif"
    );
    assert!(
        viscosity > 0.0,
        "la viscosité doit être strictement positive"
    );
    4.0 * mass_flow_per_width / viscosity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Propriétés typiques d'un film d'eau condensant vers 100 °C (fournies).
    const RHO: f64 = 960.0; // kg/m³
    const G: f64 = 9.81; // m/s²
    const HFG: f64 = 2_257_000.0; // J/kg
    const K: f64 = 0.68; // W/(m·K)
    const MU: f64 = 2.8e-4; // Pa·s
    const DT: f64 = 10.0; // K

    #[test]
    fn vertical_plate_matches_direct_formula() {
        // Identité : la fonction reproduit exactement 0,943·(groupe)^0,25.
        let l = 1.0;
        let group = RHO.powi(2) * G * HFG * K.powi(3) / (MU * DT * l);
        assert_relative_eq!(
            condensation_vertical_plate_coefficient(RHO, G, HFG, K, MU, DT, l),
            0.943_f64 * group.powf(0.25),
            epsilon = 1e-9
        );
    }

    #[test]
    fn vertical_plate_realistic_value() {
        // Cas chiffré : eau à 1 m, ΔT = 10 K.
        // groupe = 960²·9,81·2,257e6·0,68³ / (2,8e-4·10·1)
        //        = 921600·9,81·2,257e6·0,314432 / 2,8e-3 ≈ 2,29146e15
        // h = 0,943·(2,29146e15)^0,25 ≈ 6524,39 W/(m²·K).
        let h = condensation_vertical_plate_coefficient(RHO, G, HFG, K, MU, DT, 1.0);
        assert_relative_eq!(h, 6524.39, epsilon = 0.1);
    }

    #[test]
    fn horizontal_over_vertical_ratio_is_constant() {
        // Pour un même argument (D = L), h_tube/h_plaque = 0,729/0,943,
        // indépendamment des propriétés.
        let dim = 0.05;
        let hv = condensation_vertical_plate_coefficient(RHO, G, HFG, K, MU, DT, dim);
        let ht = condensation_horizontal_tube_coefficient(RHO, G, HFG, K, MU, DT, dim);
        assert_relative_eq!(ht / hv, 0.729 / 0.943, epsilon = 1e-12);
    }

    #[test]
    fn vertical_plate_scales_as_height_power_minus_quarter() {
        // h ∝ L^(-1/4) : diviser la hauteur par 16 multiplie h par 2.
        let h1 = condensation_vertical_plate_coefficient(RHO, G, HFG, K, MU, DT, 1.6);
        let h2 = condensation_vertical_plate_coefficient(RHO, G, HFG, K, MU, DT, 0.1);
        assert_relative_eq!(h2 / h1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn film_reynolds_is_linear_in_mass_flow() {
        // Re = 4·Γ/µ ; cas chiffré Γ = 0,07, µ = 2,8e-4 → Re = 1000.
        assert_relative_eq!(
            condensation_film_reynolds(0.07, 2.8e-4),
            1000.0,
            epsilon = 1e-9
        );
        // Linéarité : doubler Γ double Re.
        let r1 = condensation_film_reynolds(0.03, MU);
        let r2 = condensation_film_reynolds(0.06, MU);
        assert_relative_eq!(r2 / r1, 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "écart de température")]
    fn zero_temperature_difference_panics() {
        condensation_vertical_plate_coefficient(RHO, G, HFG, K, MU, 0.0, 1.0);
    }
}
