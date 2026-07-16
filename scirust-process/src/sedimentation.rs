//! Sédimentation et épaississement — vitesse de chute entravée (corrélation de
//! Richardson-Zaki), flux de solides, aire d'un épaississeur par la méthode du
//! flux limite et concentration du sous-verse par bilan matière.
//!
//! ```text
//! vitesse de chute entravée (Richardson-Zaki)
//!   v   = v_t · ε^n                                       [m·s⁻¹]
//! flux de solides par sédimentation
//!   G_s = C · v                                           [kg·m⁻²·s⁻¹]
//! aire d'un épaississeur (méthode du flux limite)
//!   A   = M_s / G_s                                       [m²]
//! concentration du sous-verse (bilan matière)
//!   C_u = C_f · Q_f / Q_u                                 [kg·m⁻³]
//! ```
//!
//! `v_t` vitesse terminale de chute d'une particule **isolée** [m·s⁻¹], `ε`
//! porosité (fraction volumique de liquide, ou fraction de vide) de la suspension
//! [sans dimension, 0 < ε ≤ 1], `n` indice de Richardson-Zaki [sans dimension] ;
//! `v` vitesse de chute **entravée** de la suspension [m·s⁻¹] ; `C` concentration
//! en solides [kg·m⁻³], `G_s` flux de solides (débit massique de solides par unité
//! de section) [kg·m⁻²·s⁻¹] ; `M_s` débit massique de solides à l'alimentation
//! [kg·s⁻¹], `A` aire de la section horizontale de l'épaississeur [m²] ; `C_f`
//! concentration de l'alimentation [kg·m⁻³], `Q_f` débit volumique d'alimentation
//! [m³·s⁻¹], `Q_u` débit volumique du sous-verse (underflow) [m³·s⁻¹], `C_u`
//! concentration du sous-verse [kg·m⁻³].
//!
//! **Limite honnête** : modèle de sédimentation de particules à l'échelle des
//! **opérations unitaires**, suspension supposée **homogène par zone**. La vitesse
//! terminale isolée `v_t` est **FOURNIE** par l'appelant (par exemple via la loi
//! de Stokes du module `fluidization`), jamais recalculée ici. L'indice de
//! Richardson-Zaki `n` est **FOURNI** : il dépend du nombre de Reynolds
//! particulaire et du rapport de tailles, et n'est pas supposé « par défaut ».
//! L'aire d'épaississeur suit la **méthode du flux limite** : le flux de solides
//! critique (ou limite) `G_s` est **FOURNI** ou déterminé par la courbe de flux de
//! la suspension étudiée. Aucune propriété physique (masse volumique, viscosité,
//! comportement de tassement) n'est inventée par ce module.

/// Vitesse de chute **entravée** d'une suspension par la corrélation de
/// **Richardson-Zaki** `v = v_t · ε^n` (m·s⁻¹).
///
/// `terminal_velocity` (v_t) vitesse terminale d'une particule isolée [m·s⁻¹],
/// `voidage` (ε) porosité de la suspension [sans dimension], `richardson_zaki_index`
/// (n) indice de Richardson-Zaki [sans dimension].
///
/// Panique si `v_t < 0`, si `ε` hors de `]0, 1]`, ou si `n < 0` (un indice négatif
/// donnerait une vitesse croissante avec la concentration, non physique).
pub fn sed_hindered_settling_velocity(
    terminal_velocity: f64,
    voidage: f64,
    richardson_zaki_index: f64,
) -> f64 {
    assert!(
        terminal_velocity >= 0.0,
        "v_t ≥ 0 requis (vitesse terminale)"
    );
    assert!(
        voidage > 0.0 && voidage <= 1.0,
        "0 < ε ≤ 1 requis (porosité de la suspension)"
    );
    assert!(
        richardson_zaki_index >= 0.0,
        "n ≥ 0 requis (indice de Richardson-Zaki)"
    );
    terminal_velocity * voidage.powf(richardson_zaki_index)
}

/// Flux de solides par sédimentation `G_s = C · v` (kg·m⁻²·s⁻¹), débit massique de
/// solides transporté par unité de section horizontale.
///
/// `concentration` (C) concentration en solides [kg·m⁻³], `settling_velocity` (v)
/// vitesse de chute de la suspension [m·s⁻¹].
///
/// Panique si `C < 0` ou si `v < 0`.
pub fn sed_solids_flux(concentration: f64, settling_velocity: f64) -> f64 {
    assert!(
        concentration >= 0.0,
        "C ≥ 0 requis (concentration en solides)"
    );
    assert!(
        settling_velocity >= 0.0,
        "v ≥ 0 requis (vitesse de sédimentation)"
    );
    concentration * settling_velocity
}

/// Aire de la section horizontale d'un épaississeur par la **méthode du flux
/// limite** `A = M_s / G_s` (m²).
///
/// `solids_mass_flow` (M_s) débit massique de solides à l'alimentation [kg·s⁻¹],
/// `solids_flux` (G_s) flux de solides limite (critique) [kg·m⁻²·s⁻¹].
///
/// Panique si `M_s < 0` ou si `G_s ≤ 0` (flux limite strictement positif requis
/// pour une aire finie).
pub fn sed_thickener_area(solids_mass_flow: f64, solids_flux: f64) -> f64 {
    assert!(
        solids_mass_flow >= 0.0,
        "M_s ≥ 0 requis (débit massique de solides)"
    );
    assert!(solids_flux > 0.0, "G_s > 0 requis (flux de solides limite)");
    solids_mass_flow / solids_flux
}

/// Concentration du sous-verse d'un épaississeur par **bilan matière** sur les
/// solides `C_u = C_f · Q_f / Q_u` (kg·m⁻³).
///
/// `feed_concentration` (C_f) concentration de l'alimentation [kg·m⁻³],
/// `feed_flow` (Q_f) débit volumique d'alimentation [m³·s⁻¹], `underflow_flow`
/// (Q_u) débit volumique du sous-verse [m³·s⁻¹].
///
/// Panique si `C_f < 0`, si `Q_f < 0`, ou si `Q_u ≤ 0` (débit de sous-verse
/// strictement positif requis).
pub fn sed_underflow_concentration(
    feed_concentration: f64,
    feed_flow: f64,
    underflow_flow: f64,
) -> f64 {
    assert!(
        feed_concentration >= 0.0,
        "C_f ≥ 0 requis (concentration d'alimentation)"
    );
    assert!(feed_flow >= 0.0, "Q_f ≥ 0 requis (débit d'alimentation)");
    assert!(underflow_flow > 0.0, "Q_u > 0 requis (débit de sous-verse)");
    feed_concentration * feed_flow / underflow_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hindered_velocity_index_zero_returns_terminal() {
        // n = 0 ⇒ ε^0 = 1 : aucune entrave, v = v_t quelle que soit la porosité.
        let v = sed_hindered_settling_velocity(0.0037_f64, 0.42_f64, 0.0_f64);
        assert_relative_eq!(v, 0.0037, max_relative = 1e-12);
    }

    #[test]
    fn hindered_velocity_index_one_is_linear() {
        // n = 1 ⇒ v = v_t · ε : proportionnalité directe à la porosité.
        let v = sed_hindered_settling_velocity(0.004_f64, 0.75_f64, 1.0_f64);
        assert_relative_eq!(v, 0.003, max_relative = 1e-12);
    }

    #[test]
    fn hindered_velocity_realistic_case() {
        // v_t = 2.5 mm/s, ε = 0.6, n = 4.65 (Richardson-Zaki, régime laminaire) ⇒
        //   0.6^4.65 = 0.0929829
        //   v = 0.0025 · 0.0929829 ≈ 0.00023246 m/s.
        let v = sed_hindered_settling_velocity(0.0025_f64, 0.6_f64, 4.65_f64);
        assert_relative_eq!(v, 0.00023246, max_relative = 1e-3);
    }

    #[test]
    fn solids_flux_scales_with_velocity() {
        // G_s ∝ v : doubler la vitesse de chute double le flux de solides.
        let single = sed_solids_flux(5.0_f64, 0.0002_f64);
        let double = sed_solids_flux(5.0_f64, 0.0004_f64);
        assert_relative_eq!(single, 0.001, max_relative = 1e-12);
        assert_relative_eq!(double, 2.0 * single, max_relative = 1e-12);
    }

    #[test]
    fn thickener_area_from_flux_chain() {
        // Chaîne flux → aire : C = 5 kg/m³, v = 0.0002 m/s ⇒ G_s = 0.001 ;
        // M_s = 2 kg/s ⇒ A = M_s / G_s = 2 / 0.001 = 2000 m².
        let flux = sed_solids_flux(5.0_f64, 0.0002_f64);
        let area = sed_thickener_area(2.0_f64, flux);
        assert_relative_eq!(area, 2000.0, max_relative = 1e-12);
        // Réciprocité : A · G_s = M_s.
        assert_relative_eq!(area * flux, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn underflow_concentration_conserves_solids_mass() {
        // Bilan solides : C_f = 10 kg/m³, Q_f = 0.01 m³/s, Q_u = 0.002 m³/s ⇒
        //   C_u = 10 · 0.01 / 0.002 = 50 kg/m³.
        let c_u = sed_underflow_concentration(10.0_f64, 0.01_f64, 0.002_f64);
        assert_relative_eq!(c_u, 50.0, max_relative = 1e-12);
        // Conservation de la masse de solides : C_u · Q_u = C_f · Q_f.
        assert_relative_eq!(c_u * 0.002, 10.0 * 0.01, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "G_s > 0 requis")]
    fn thickener_area_panics_on_zero_flux() {
        // Flux limite nul ⇒ aire infinie non physique ⇒ entrée rejetée.
        let _ = sed_thickener_area(2.0_f64, 0.0_f64);
    }
}
