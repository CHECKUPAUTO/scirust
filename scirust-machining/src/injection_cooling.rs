//! Injection plastique — **temps de refroidissement** d'une paroi (plaque),
//! par la solution du **premier mode** de la conduction transitoire 1D.
//!
//! ```text
//! temps de refroidissement   t = (s²/(π²·α))·ln[(4/π)·(Tmelt−Tmold)/(Teject−Tmold)]
//! température au cœur         T(t) = Tmold + (4/π)·(Tmelt−Tmold)·exp(−π²·α·t/s²)
//! épaisseur pour un temps t   s = √[ t·π²·α / ln((4/π)·(Tmelt−Tmold)/(Teject−Tmold)) ]
//! nombre de Fourier           Fo = α·t/s²
//! ```
//!
//! `s` épaisseur de paroi (m), `α` diffusivité thermique du polymère (m²/s), `t`
//! temps de refroidissement (s), `Tmelt` température d'injection (matière),
//! `Tmold` température du moule (paroi), `Teject` température d'éjection (au cœur),
//! `Fo` nombre de Fourier (sans dimension). Les températures sont en K **ou** en
//! °C, du moment que les trois sont dans la **même** unité (seules les différences
//! interviennent).
//!
//! **Convention** : SI cohérent, `s` en mètres. **Limite honnête** : modèle de
//! plaque **1D** (conduction transitoire pure, paroi mince devant les autres
//! dimensions, moule à température de paroi constante) tronqué à son **premier
//! mode** — le facteur `4/π` est celui de la température au **cœur** de la plaque
//! et n'est valable que pour `Fo` assez grand (régime établi). La diffusivité
//! thermique `α`, les températures de mise en œuvre et la géométrie sont
//! **fournies par l'appelant** : aucune valeur matériau n'est supposée ici.

use core::f64::consts::PI;

/// Vérifie les invariants communs aux relations de refroidissement et renvoie
/// l'argument `(4/π)·(Tmelt−Tmold)/(Teject−Tmold)` du logarithme.
///
/// Panique si une différence de température est nulle ou négative.
fn log_argument(melt_temp: f64, mold_temp: f64, ejection_temp: f64) -> f64 {
    assert!(
        melt_temp > mold_temp,
        "la température d'injection doit dépasser celle du moule (Tmelt > Tmold)"
    );
    assert!(
        ejection_temp > mold_temp,
        "la température d'éjection doit dépasser celle du moule (Teject > Tmold)"
    );
    assert!(
        melt_temp >= ejection_temp,
        "la matière ne peut refroidir en dessous de l'injection (Tmelt >= Teject)"
    );
    (4.0 / PI) * (melt_temp - mold_temp) / (ejection_temp - mold_temp)
}

/// Temps de refroidissement (cœur de plaque)
/// `t = (s²/(π²·α))·ln[(4/π)·(Tmelt−Tmold)/(Teject−Tmold)]` (s).
///
/// Panique si `wall_thickness <= 0`, `thermal_diffusivity <= 0`, ou si une
/// différence de température est nulle/négative.
pub fn cooling_time(
    wall_thickness: f64,
    thermal_diffusivity: f64,
    melt_temp: f64,
    mold_temp: f64,
    ejection_temp: f64,
) -> f64 {
    assert!(
        wall_thickness > 0.0,
        "l'épaisseur de paroi doit être strictement positive"
    );
    assert!(
        thermal_diffusivity > 0.0,
        "la diffusivité thermique doit être strictement positive"
    );
    let arg = log_argument(melt_temp, mold_temp, ejection_temp);
    (wall_thickness * wall_thickness) / (PI * PI * thermal_diffusivity) * arg.ln()
}

/// Température au **cœur** de la plaque après un temps `t`
/// `T(t) = Tmold + (4/π)·(Tmelt−Tmold)·exp(−π²·α·t/s²)` (même unité que les
/// entrées). Relation réciproque de [`cooling_time`].
///
/// Panique si `wall_thickness <= 0`, `thermal_diffusivity <= 0` ou `time < 0`.
pub fn center_temperature(
    wall_thickness: f64,
    thermal_diffusivity: f64,
    melt_temp: f64,
    mold_temp: f64,
    time: f64,
) -> f64 {
    assert!(
        wall_thickness > 0.0,
        "l'épaisseur de paroi doit être strictement positive"
    );
    assert!(
        thermal_diffusivity > 0.0,
        "la diffusivité thermique doit être strictement positive"
    );
    assert!(time >= 0.0, "le temps doit être positif");
    let fo = thermal_diffusivity * time / (wall_thickness * wall_thickness);
    mold_temp + (4.0 / PI) * (melt_temp - mold_temp) * (-PI * PI * fo).exp()
}

/// Épaisseur de paroi admissible pour un temps de refroidissement cible
/// `s = √[ t·π²·α / ln((4/π)·(Tmelt−Tmold)/(Teject−Tmold)) ]` (m).
/// Inverse de [`cooling_time`].
///
/// Panique si `cooling_time <= 0`, `thermal_diffusivity <= 0`, ou si une
/// différence de température est nulle/négative.
pub fn wall_thickness_for_time(
    cooling_time: f64,
    thermal_diffusivity: f64,
    melt_temp: f64,
    mold_temp: f64,
    ejection_temp: f64,
) -> f64 {
    assert!(
        cooling_time > 0.0,
        "le temps de refroidissement doit être strictement positif"
    );
    assert!(
        thermal_diffusivity > 0.0,
        "la diffusivité thermique doit être strictement positive"
    );
    let arg = log_argument(melt_temp, mold_temp, ejection_temp);
    (cooling_time * PI * PI * thermal_diffusivity / arg.ln()).sqrt()
}

/// Nombre de Fourier `Fo = α·t/s²` (sans dimension), mesure de l'avancement de
/// la diffusion thermique dans la paroi.
///
/// Panique si `wall_thickness <= 0`.
pub fn cooling_fourier_number(thermal_diffusivity: f64, time: f64, wall_thickness: f64) -> f64 {
    assert!(
        wall_thickness > 0.0,
        "l'épaisseur de paroi doit être strictement positive"
    );
    thermal_diffusivity * time / (wall_thickness * wall_thickness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn time_scales_with_thickness_squared() {
        // t ∝ s² : doubler l'épaisseur quadruple le temps de refroidissement.
        let t1 = cooling_time(2.0e-3, 1.0e-7, 230.0, 40.0, 90.0);
        let t2 = cooling_time(4.0e-3, 1.0e-7, 230.0, 40.0, 90.0);
        assert_relative_eq!(t2 / t1, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn time_is_inversely_proportional_to_diffusivity() {
        // t ∝ 1/α : une diffusivité doublée divise le temps par deux.
        let t1 = cooling_time(3.0e-3, 1.0e-7, 230.0, 40.0, 90.0);
        let t2 = cooling_time(3.0e-3, 2.0e-7, 230.0, 40.0, 90.0);
        assert_relative_eq!(t1 / t2, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn center_temperature_reaches_ejection_at_cooling_time() {
        // Réciprocité : au temps de refroidissement, le cœur est à Teject.
        let (s, alpha, tmelt, tmold, teject) = (2.5e-3_f64, 9.0e-8, 240.0, 50.0, 95.0);
        let t = cooling_time(s, alpha, tmelt, tmold, teject);
        assert_relative_eq!(
            center_temperature(s, alpha, tmelt, tmold, t),
            teject,
            epsilon = 1e-9
        );
    }

    #[test]
    fn thickness_for_time_inverts_cooling_time() {
        // Réciprocité : l'épaisseur calculée redonne exactement le temps visé.
        let (alpha, tmelt, tmold, teject) = (1.1e-7_f64, 220.0, 45.0, 100.0);
        let target = 12.0_f64;
        let s = wall_thickness_for_time(target, alpha, tmelt, tmold, teject);
        assert_relative_eq!(
            cooling_time(s, alpha, tmelt, tmold, teject),
            target,
            epsilon = 1e-9
        );
    }

    #[test]
    fn realistic_case_matches_closed_form() {
        // Cas chiffré : s=2 mm, α=1e-7 m²/s, 230/40/90 °C.
        // t = (0.002²/(π²·1e-7))·ln((4/π)·190/50).
        let t = cooling_time(2.0e-3, 1.0e-7, 230.0, 40.0, 90.0);
        let expected = (2.0e-3_f64.powi(2) / (PI * PI * 1.0e-7)) * ((4.0 / PI) * 190.0 / 50.0).ln();
        assert_relative_eq!(t, expected, epsilon = 1e-12);
        // Fourier correspondant : Fo = α·t/s².
        let fo = cooling_fourier_number(1.0e-7, t, 2.0e-3);
        assert_relative_eq!(fo, 1.0e-7 * t / 2.0e-3_f64.powi(2), epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "moule")]
    fn ejection_below_mold_panics() {
        // Teject <= Tmold rend le logarithme indéfini.
        cooling_time(2.0e-3, 1.0e-7, 230.0, 90.0, 90.0);
    }
}
