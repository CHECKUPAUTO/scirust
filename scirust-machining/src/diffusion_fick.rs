//! Diffusion en phase solide — **lois de Fick** : flux surfacique (1re loi),
//! profil de concentration en milieu semi-infini (2e loi), dépendance en
//! température du coefficient de diffusion (Arrhenius) et longueur de diffusion.
//!
//! ```text
//! 1re loi (flux)        J   = -D · dC/dx
//! 2e loi (profil)       C(x,t) = Ci + (Cs - Ci) · erfc( x / (2·√(D·t)) )
//! Arrhenius             D   = D0 · exp( -Q / (R·T) )
//! longueur de diffusion L   = √(D·t)
//! ```
//!
//! `J` densité de flux molaire surfacique (mol·m⁻²·s⁻¹), `D` coefficient de
//! diffusion (m²·s⁻¹), `dC/dx` gradient de concentration (mol·m⁻³ par m, soit
//! mol·m⁻⁴), `C(x,t)` concentration au point `x` (m) et à l'instant `t` (s),
//! `Cs` concentration de surface maintenue constante et `Ci` concentration
//! initiale uniforme (mol·m⁻³, unité libre mais commune aux trois), `erfc` la
//! fonction erreur complémentaire, `D0` facteur pré-exponentiel (m²·s⁻¹), `Q`
//! énergie d'activation (J·mol⁻¹), `R` constante des gaz (J·mol⁻¹·K⁻¹), `T`
//! température absolue (K), `L` longueur caractéristique de pénétration (m).
//!
//! **Convention** : SI cohérent, `f64`. **Limite honnête** : diffusion **1D**
//! dans un milieu **semi-infini** à coefficient `D` **constant** (Cs imposée en
//! surface, Ci uniforme au départ). Le coefficient `D` — ou le couple `D0`/`Q`
//! d'Arrhenius — dépend du **matériau, de la température et du couple diffusant**
//! et est **fourni par l'appelant** ; aucune valeur « par défaut » n'est
//! inventée. La fonction `erfc` est évaluée par l'**approximation rationnelle
//! d'Abramowitz & Stegun 7.1.26** (erreur absolue maximale ≈ 1,5·10⁻⁷ pour un
//! argument positif). Distinct de [`crate::carburizing`], qui applique ces
//! principes au **cas industriel de la cémentation** (profondeur de couche).

/// Fonction erreur complémentaire `erfc(x)` pour `x ≥ 0`.
///
/// Approximation rationnelle d'Abramowitz & Stegun 7.1.26 : erreur absolue
/// maximale ≈ 1,5·10⁻⁷. Documentée en interne, non exposée hors du module.
fn erfc_approx(x: f64) -> f64 {
    // Coefficients d'Abramowitz & Stegun 7.1.26 (valables pour x ≥ 0).
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let t = 1.0_f64 / (1.0_f64 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    poly * (-x * x).exp()
}

/// Densité de flux molaire surfacique par la 1re loi de Fick `J = -D·dC/dx`
/// (mol·m⁻²·s⁻¹).
///
/// `diffusion_coefficient` = `D` (m²·s⁻¹), `concentration_gradient` = `dC/dx`
/// (mol·m⁻⁴). Le signe négatif traduit un flux dirigé des concentrations
/// fortes vers les faibles ; `concentration_gradient` peut être de signe
/// quelconque.
///
/// Panique si `diffusion_coefficient < 0`.
pub fn fick_first_law_flux(diffusion_coefficient: f64, concentration_gradient: f64) -> f64 {
    assert!(
        diffusion_coefficient >= 0.0,
        "le coefficient de diffusion doit être positif"
    );
    -diffusion_coefficient * concentration_gradient
}

/// Concentration en milieu semi-infini par la 2e loi de Fick
/// `C = Ci + (Cs - Ci)·erfc( x / (2·√(D·t)) )` (même unité que `Cs`/`Ci`).
///
/// `surface_concentration` = `Cs` (mol·m⁻³) imposée en surface, maintenue
/// constante, `initial_concentration` = `Ci` (mol·m⁻³) uniforme à `t = 0`,
/// `position` = `x` (m) mesurée depuis la surface, `diffusion_coefficient`
/// = `D` (m²·s⁻¹), `time` = `t` (s). `erfc` est évaluée par l'approximation
/// d'Abramowitz & Stegun 7.1.26 (erreur ≈ 1,5·10⁻⁷).
///
/// Panique si `position < 0`, `diffusion_coefficient <= 0` ou `time <= 0`.
pub fn fick_penetration_concentration(
    surface_concentration: f64,
    initial_concentration: f64,
    position: f64,
    diffusion_coefficient: f64,
    time: f64,
) -> f64 {
    assert!(position >= 0.0, "la position doit être positive ou nulle");
    assert!(
        diffusion_coefficient > 0.0,
        "le coefficient de diffusion doit être strictement positif"
    );
    assert!(time > 0.0, "la durée doit être strictement positive");
    let argument = position / (2.0_f64 * (diffusion_coefficient * time).sqrt());
    initial_concentration + (surface_concentration - initial_concentration) * erfc_approx(argument)
}

/// Coefficient de diffusion par la loi d'Arrhenius `D = D0·exp(-Q/(R·T))`
/// (m²·s⁻¹).
///
/// `pre_exponential` = `D0` (m²·s⁻¹), `activation_energy` = `Q` (J·mol⁻¹),
/// `gas_constant` = `R` (J·mol⁻¹·K⁻¹, **fournie** par l'appelant),
/// `temperature_kelvin` = `T` (K, absolue).
///
/// Panique si `pre_exponential < 0`, `activation_energy < 0`,
/// `gas_constant <= 0` ou `temperature_kelvin <= 0`.
pub fn fick_diffusion_coefficient_arrhenius(
    pre_exponential: f64,
    activation_energy: f64,
    gas_constant: f64,
    temperature_kelvin: f64,
) -> f64 {
    assert!(
        pre_exponential >= 0.0,
        "le facteur pré-exponentiel doit être positif"
    );
    assert!(
        activation_energy >= 0.0,
        "l'énergie d'activation doit être positive"
    );
    assert!(
        gas_constant > 0.0,
        "la constante des gaz doit être strictement positive"
    );
    assert!(
        temperature_kelvin > 0.0,
        "la température absolue doit être strictement positive"
    );
    pre_exponential * (-activation_energy / (gas_constant * temperature_kelvin)).exp()
}

/// Longueur caractéristique de diffusion `L = √(D·t)` (m).
///
/// `diffusion_coefficient` = `D` (m²·s⁻¹), `time` = `t` (s). Cette longueur
/// fixe l'échelle de pénétration : l'argument de `erfc` dans la 2e loi vaut
/// `x / (2·L)`.
///
/// Panique si `diffusion_coefficient < 0` ou `time < 0`.
pub fn fick_diffusion_length(diffusion_coefficient: f64, time: f64) -> f64 {
    assert!(
        diffusion_coefficient >= 0.0,
        "le coefficient de diffusion doit être positif"
    );
    assert!(time >= 0.0, "la durée doit être positive");
    (diffusion_coefficient * time).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn first_law_sign_and_proportionality() {
        // Un gradient négatif (dC/dx < 0) produit un flux positif : J = -D·dC/dx.
        let d = 1.0e-11_f64;
        let grad = -2.0_f64; // mol·m⁻⁴
        let j = fick_first_law_flux(d, grad);
        assert_relative_eq!(j, 2.0e-11_f64, epsilon = 1e-24);
        // Doubler D double le flux (proportionnalité).
        assert_relative_eq!(fick_first_law_flux(2.0 * d, grad), 2.0 * j, epsilon = 1e-24);
    }

    #[test]
    fn length_squared_equals_dt() {
        // L = √(D·t) ⇒ L² = D·t (identité).
        let d = 3.0e-12_f64;
        let t = 7200.0_f64;
        let l = fick_diffusion_length(d, t);
        assert_relative_eq!(l * l, d * t, epsilon = 1e-24);
    }

    #[test]
    fn penetration_limits_are_cs_and_ci() {
        // À x = 0 : erfc(0) = 1 ⇒ C = Cs. Loin de la surface : erfc → 0 ⇒ C → Ci.
        let cs = 1.4_f64;
        let ci = 0.2_f64;
        let d = 1.0e-11_f64;
        let t = 1.0e4_f64;
        let at_surface = fick_penetration_concentration(cs, ci, 0.0, d, t);
        assert_relative_eq!(at_surface, cs, epsilon = 1e-6);
        // Très profond (argument ≫ 1) : la concentration retombe vers Ci.
        let deep = fick_penetration_concentration(cs, ci, 5.0e-3, d, t);
        assert_relative_eq!(deep, ci, epsilon = 1e-6);
    }

    #[test]
    fn penetration_matches_known_erfc_value() {
        // Cas chiffré : on choisit D·t = x² pour que l'argument vaille exactement
        // x/(2·√(D·t)) = 0,5. Avec Cs = 1, Ci = 0 : C = erfc(0,5) = 0,479500122.
        // On vérifie d'abord l'argument via la longueur de diffusion.
        let x = 1.0e-3_f64;
        let d = 1.0e-11_f64;
        let t = 1.0e5_f64;
        let arg = x / (2.0_f64 * fick_diffusion_length(d, t));
        assert_relative_eq!(arg, 0.5_f64, epsilon = 1e-12);
        let c = fick_penetration_concentration(1.0, 0.0, x, d, t);
        assert_relative_eq!(c, 0.479_500_122_f64, epsilon = 2e-7);
    }

    #[test]
    fn arrhenius_zero_activation_and_proportionality() {
        // Q = 0 ⇒ exp(0) = 1 ⇒ D = D0.
        let d0 = 2.3e-5_f64;
        let r = 8.314_f64;
        let t = 1123.0_f64;
        assert_relative_eq!(
            fick_diffusion_coefficient_arrhenius(d0, 0.0, r, t),
            d0,
            epsilon = 1e-20
        );
        // Doubler D0 double D (facteur exponentiel identique).
        let q = 148_000.0_f64;
        let d = fick_diffusion_coefficient_arrhenius(d0, q, r, t);
        assert_relative_eq!(
            fick_diffusion_coefficient_arrhenius(2.0 * d0, q, r, t),
            2.0 * d,
            epsilon = 1e-24
        );
    }

    #[test]
    fn arrhenius_inverts_via_logarithm() {
        // D = D0·exp(-Q/(RT)) ⇒ D0 = D·exp(Q/(RT)) : on retrouve D0.
        let d0 = 1.6e-4_f64;
        let q = 250_000.0_f64;
        let r = 8.314_f64;
        let t = 900.0_f64;
        let d = fick_diffusion_coefficient_arrhenius(d0, q, r, t);
        let recovered = d * (q / (r * t)).exp();
        assert_relative_eq!(recovered, d0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le coefficient de diffusion doit être positif")]
    fn negative_diffusion_length_panics() {
        let _ = fick_diffusion_length(-1.0e-12, 100.0);
    }
}
