//! Ébullition nucléée en vase — corrélation de **Rohsenow** (flux de chaleur)
//! et **flux critique** (CHF) de **Zuber**.
//!
//! ```text
//! surchauffe        ΔTe = Ts - Tsat                                              (K)
//! Rohsenow          q'' = μ·hfg·√(g·(ρ_l-ρ_v)/σ)·(cp·ΔTe/(Csf·hfg·Pr^n))³       (W·m⁻²)
//! CHF de Zuber      q''_max = 0.149·hfg·ρ_v^0.5·(σ·g·(ρ_l-ρ_v))^0.25            (W·m⁻²)
//! ```
//!
//! `Ts` température de paroi (K), `Tsat` température de saturation (K), `ΔTe`
//! surchauffe de paroi (K), `μ` viscosité dynamique du liquide (Pa·s), `hfg`
//! chaleur latente de vaporisation (J·kg⁻¹), `g` accélération de la pesanteur
//! (m·s⁻²), `ρ_l` masse volumique du liquide (kg·m⁻³), `ρ_v` masse volumique de
//! la vapeur (kg·m⁻³), `σ` tension superficielle liquide-vapeur (N·m⁻¹), `cp`
//! chaleur massique du liquide (J·kg⁻¹·K⁻¹), `Pr` nombre de Prandtl du liquide
//! (sans dimension), `Csf` coefficient empirique du couple surface-fluide (sans
//! dimension), `n` exposant de Prandtl de Rohsenow (sans dimension), `q''` densité
//! de flux thermique (W·m⁻²), `q''_max` flux critique (W·m⁻²).
//!
//! **Convention** : unités SI.
//! **Limite honnête** : corrélations **empiriques** d'ébullition **nucléée en
//! vase** (pool boiling). Le coefficient surface-fluide `Csf` et l'exposant `n`
//! de Rohsenow dépendent du **couple surface/fluide** et sont **fournis par
//! l'appelant** ; toutes les propriétés (μ, hfg, ρ_l, ρ_v, σ, cp, Pr) sont
//! évaluées à **saturation** et **fournies par l'appelant** — aucune valeur
//! « par défaut » n'est inventée. La constante `0.149` de Zuber correspond au
//! cas d'une **plaque horizontale infinie**.

/// Surchauffe de paroi `ΔTe = Ts - Tsat` (K), moteur de l'ébullition nucléée.
///
/// Panique si `surface_temp < 0`, `saturation_temp < 0` (températures absolues)
/// ou si `surface_temp < saturation_temp` (pas d'ébullition nucléée).
pub fn boiling_excess_temperature(surface_temp: f64, saturation_temp: f64) -> f64 {
    assert!(surface_temp >= 0.0, "Ts ≥ 0 (température absolue) requis");
    assert!(
        saturation_temp >= 0.0,
        "Tsat ≥ 0 (température absolue) requis"
    );
    assert!(
        surface_temp >= saturation_temp,
        "Ts ≥ Tsat requis (surchauffe positive)"
    );
    surface_temp - saturation_temp
}

/// Densité de flux thermique en ébullition nucléée (corrélation de Rohsenow)
/// `q'' = μ·hfg·√(g·(ρ_l-ρ_v)/σ)·(cp·ΔTe/(Csf·hfg·Pr^n))³` (W·m⁻²).
///
/// Panique si l'une des propriétés `μ, hfg, g, σ, cp, Csf` est `≤ 0`, si
/// `ρ_l ≤ ρ_v` (liquide plus dense que la vapeur requis), si `Pr ≤ 0` ou si la
/// surchauffe `ΔTe` est `< 0`.
#[allow(clippy::too_many_arguments)]
pub fn boiling_rohsenow_heat_flux(
    viscosity: f64,
    latent_heat: f64,
    gravity: f64,
    density_liquid: f64,
    density_vapor: f64,
    surface_tension: f64,
    specific_heat_liquid: f64,
    excess_temperature: f64,
    prandtl_liquid: f64,
    surface_fluid_coefficient: f64,
    prandtl_exponent: f64,
) -> f64 {
    assert!(viscosity > 0.0, "μ > 0 requis");
    assert!(latent_heat > 0.0, "hfg > 0 requis");
    assert!(gravity > 0.0, "g > 0 requis");
    assert!(surface_tension > 0.0, "σ > 0 requis");
    assert!(specific_heat_liquid > 0.0, "cp > 0 requis");
    assert!(surface_fluid_coefficient > 0.0, "Csf > 0 requis");
    assert!(prandtl_liquid > 0.0, "Pr > 0 requis");
    assert!(
        density_liquid > density_vapor,
        "ρ_l > ρ_v requis (liquide plus dense que la vapeur)"
    );
    assert!(excess_temperature >= 0.0, "ΔTe ≥ 0 requis");
    let buoyancy = (gravity * (density_liquid - density_vapor) / surface_tension).sqrt();
    let inner = specific_heat_liquid * excess_temperature
        / (surface_fluid_coefficient * latent_heat * prandtl_liquid.powf(prandtl_exponent));
    viscosity * latent_heat * buoyancy * inner.powi(3)
}

/// Flux critique d'ébullition (CHF) de Zuber, plaque horizontale
/// `q''_max = 0.149·hfg·ρ_v^0.5·(σ·g·(ρ_l-ρ_v))^0.25` (W·m⁻²).
///
/// Panique si `hfg ≤ 0`, `ρ_v ≤ 0`, `σ ≤ 0`, `g ≤ 0` ou si `ρ_l ≤ ρ_v`.
pub fn boiling_critical_heat_flux_zuber(
    latent_heat: f64,
    density_vapor: f64,
    surface_tension: f64,
    gravity: f64,
    density_liquid: f64,
) -> f64 {
    assert!(latent_heat > 0.0, "hfg > 0 requis");
    assert!(density_vapor > 0.0, "ρ_v > 0 requis");
    assert!(surface_tension > 0.0, "σ > 0 requis");
    assert!(gravity > 0.0, "g > 0 requis");
    assert!(
        density_liquid > density_vapor,
        "ρ_l > ρ_v requis (liquide plus dense que la vapeur)"
    );
    0.149_f64
        * latent_heat
        * density_vapor.powf(0.5)
        * (surface_tension * gravity * (density_liquid - density_vapor)).powf(0.25)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Propriétés de l'eau saturée à 100 °C / 1 atm, couple eau-cuivre poli
    // (Csf = 0.013, n = 1.0), utilisées par plusieurs tests.
    const MU: f64 = 279e-6;
    const HFG: f64 = 2257e3;
    const G: f64 = 9.81;
    const RHO_L: f64 = 957.9;
    const RHO_V: f64 = 0.5955;
    const SIGMA: f64 = 0.0589;
    const CP: f64 = 4217.0;
    const PR: f64 = 1.76;
    const CSF: f64 = 0.013;
    const N: f64 = 1.0;

    #[test]
    fn excess_temperature_is_difference() {
        // Identité : ΔTe = Ts - Tsat = 383.15 - 373.15 = 10 K.
        assert_relative_eq!(
            boiling_excess_temperature(383.15, 373.15),
            10.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn rohsenow_realistic_case() {
        // Eau à 100 °C, ΔTe = 10 K, cuivre poli : q'' ≈ 136.9 kW·m⁻².
        let q = boiling_rohsenow_heat_flux(MU, HFG, G, RHO_L, RHO_V, SIGMA, CP, 10.0, PR, CSF, N);
        assert_relative_eq!(q, 136_925.941_023_2, max_relative = 1e-9);
    }

    #[test]
    fn rohsenow_scales_as_excess_temperature_cubed() {
        // q'' ∝ ΔTe³ : doubler la surchauffe multiplie le flux par 8.
        let q1 = boiling_rohsenow_heat_flux(MU, HFG, G, RHO_L, RHO_V, SIGMA, CP, 10.0, PR, CSF, N);
        let q2 = boiling_rohsenow_heat_flux(MU, HFG, G, RHO_L, RHO_V, SIGMA, CP, 20.0, PR, CSF, N);
        assert_relative_eq!(q2 / q1, 8.0, max_relative = 1e-12);
    }

    #[test]
    fn rohsenow_zero_excess_gives_zero_flux() {
        // Cas limite : sans surchauffe (ΔTe = 0) le flux nucléé est nul.
        let q = boiling_rohsenow_heat_flux(MU, HFG, G, RHO_L, RHO_V, SIGMA, CP, 0.0, PR, CSF, N);
        assert_relative_eq!(q, 0.0, max_relative = 1e-12);
    }

    #[test]
    fn zuber_realistic_case() {
        // Eau à 100 °C : flux critique ≈ 1.26 MW·m⁻².
        let qmax = boiling_critical_heat_flux_zuber(HFG, RHO_V, SIGMA, G, RHO_L);
        assert_relative_eq!(qmax, 1_258_540.834_067_69, max_relative = 1e-9);
    }

    #[test]
    fn zuber_scales_linearly_with_latent_heat() {
        // q''_max ∝ hfg : doubler la chaleur latente double le flux critique.
        let q1 = boiling_critical_heat_flux_zuber(HFG, RHO_V, SIGMA, G, RHO_L);
        let q2 = boiling_critical_heat_flux_zuber(2.0 * HFG, RHO_V, SIGMA, G, RHO_L);
        assert_relative_eq!(q2 / q1, 2.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "ρ_l > ρ_v")]
    fn rohsenow_denser_vapor_panics() {
        boiling_rohsenow_heat_flux(MU, HFG, G, 0.5, RHO_V, SIGMA, CP, 10.0, PR, CSF, N);
    }
}
