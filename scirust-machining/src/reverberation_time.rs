//! Temps de réverbération d'un local (RT60) par les formules de Sabine et
//! d'Eyring, à partir de l'aire d'absorption équivalente.
//!
//! ```text
//! aire d'absorption équiv.   A  = Σ S_i · α_i
//! absorption moyenne         ᾱ  = A / S
//! réverbération de Sabine    RT = 0.161 · V / A
//! réverbération d'Eyring     RT = 0.161 · V / ( -S · ln(1 - ᾱ) )
//! ```
//!
//! `V` volume du local (m³), `S` aire totale des parois (m²), `S_i` aire d'une
//! surface (m²), `α_i` coefficient d'absorption de cette surface (sans
//! dimension, 0..1), `A` aire d'absorption équivalente (m² sabine), `ᾱ`
//! coefficient d'absorption moyen (sans dimension), `RT` temps de réverbération
//! RT60 (s). La constante 0,161 s/m vaut `24·ln(10)/c` pour la célérité `c` de
//! l'air standard.
//!
//! **Convention** : SI cohérent, champ acoustique **diffus** supposé. **Limite
//! honnête** : les coefficients d'absorption `α_i` sont **fournis par
//! l'appelant** (par matériau et par fréquence — tables du fabricant) ; aucune
//! valeur « par défaut » n'est inventée. La constante 0,161 correspond à l'**air
//! standard** ; la formule de **Sabine** vaut pour une absorption faible, celle
//! d'**Eyring** est préférée pour une absorption moyenne élevée. L'absorption de
//! l'air (dissipation dans le volume) n'est pas modélisée ici.

/// Aire d'absorption équivalente `A = Σ S_i · α_i` (m² sabine).
///
/// Panique si les tranches sont vides ou de longueurs différentes, si une aire
/// est `<= 0`, ou si un coefficient d'absorption n'est pas dans `[0, 1]`.
pub fn reverb_total_absorption(surfaces_areas: &[f64], absorption_coefficients: &[f64]) -> f64 {
    assert!(
        !surfaces_areas.is_empty(),
        "la liste des aires de surface ne doit pas être vide"
    );
    assert!(
        surfaces_areas.len() == absorption_coefficients.len(),
        "les aires et les coefficients d'absorption doivent avoir la même longueur"
    );
    let mut total = 0.0;
    for (&area, &alpha) in surfaces_areas.iter().zip(absorption_coefficients.iter())
    {
        assert!(
            area > 0.0,
            "chaque aire de surface doit être strictement positive"
        );
        assert!(
            (0.0..=1.0).contains(&alpha),
            "chaque coefficient d'absorption doit être compris entre 0 et 1"
        );
        total += area * alpha;
    }
    total
}

/// Coefficient d'absorption moyen `ᾱ = A / S` (sans dimension).
///
/// Panique si `total_surface <= 0` ou si `total_absorption < 0`.
pub fn reverb_mean_absorption(total_absorption: f64, total_surface: f64) -> f64 {
    assert!(
        total_surface > 0.0,
        "l'aire totale des parois doit être strictement positive"
    );
    assert!(
        total_absorption >= 0.0,
        "l'aire d'absorption équivalente ne doit pas être négative"
    );
    total_absorption / total_surface
}

/// Temps de réverbération RT60 de Sabine `RT = 0.161 · V / A` (s).
///
/// Panique si `room_volume <= 0` ou `total_absorption <= 0`.
pub fn reverb_sabine_time(room_volume: f64, total_absorption: f64) -> f64 {
    assert!(
        room_volume > 0.0,
        "le volume du local doit être strictement positif"
    );
    assert!(
        total_absorption > 0.0,
        "l'aire d'absorption équivalente doit être strictement positive"
    );
    0.161 * room_volume / total_absorption
}

/// Temps de réverbération RT60 d'Eyring
/// `RT = 0.161 · V / ( -S · ln(1 - ᾱ) )` (s).
///
/// Panique si `room_volume <= 0`, `total_surface <= 0`, ou si
/// `mean_absorption_coefficient` n'est pas dans l'intervalle ouvert `]0, 1[`.
pub fn reverb_eyring_time(
    room_volume: f64,
    total_surface: f64,
    mean_absorption_coefficient: f64,
) -> f64 {
    assert!(
        room_volume > 0.0,
        "le volume du local doit être strictement positif"
    );
    assert!(
        total_surface > 0.0,
        "l'aire totale des parois doit être strictement positive"
    );
    assert!(
        mean_absorption_coefficient > 0.0 && mean_absorption_coefficient < 1.0,
        "le coefficient d'absorption moyen doit être strictement compris entre 0 et 1"
    );
    0.161 * room_volume / (-total_surface * (1.0 - mean_absorption_coefficient).ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn total_absorption_reference_case() {
        // S = [10, 20, 30] m², α = [0,1 ; 0,2 ; 0,3]
        // A = 10·0,1 + 20·0,2 + 30·0,3 = 1 + 4 + 9 = 14 m² sabine.
        let areas = [10.0_f64, 20.0, 30.0];
        let alphas = [0.1_f64, 0.2, 0.3];
        let a = reverb_total_absorption(&areas, &alphas);
        assert_relative_eq!(a, 14.0, epsilon = 1e-12);
    }

    #[test]
    fn mean_absorption_reference_case() {
        // ᾱ = A / S = 14 / 60 = 0,233333…
        let alpha = reverb_mean_absorption(14.0, 60.0);
        assert_relative_eq!(alpha, 14.0 / 60.0, epsilon = 1e-15);
    }

    #[test]
    fn sabine_reference_case() {
        // V = 100 m³, A = 20 m² sabine → RT = 0,161·100/20 = 0,805 s.
        let rt = reverb_sabine_time(100.0, 20.0);
        assert_relative_eq!(rt, 0.805, epsilon = 1e-12);
    }

    #[test]
    fn sabine_inverse_proportional_to_absorption() {
        // À volume fixé, doubler l'aire d'absorption divise RT par deux.
        let rt1 = reverb_sabine_time(250.0, 15.0);
        let rt2 = reverb_sabine_time(250.0, 30.0);
        assert_relative_eq!(rt1 / rt2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn eyring_reference_case() {
        // V = 100 m³, S = 200 m², ᾱ = 0,1.
        // -S·ln(1-ᾱ) = -200·ln(0,9) = 21,072103… → RT = 16,1/21,072103 = 0,764043 s.
        let rt = reverb_eyring_time(100.0, 200.0, 0.1);
        let expected = 0.161 * 100.0 / (-200.0 * 0.9_f64.ln());
        assert_relative_eq!(rt, expected, epsilon = 1e-15);
        assert_relative_eq!(rt, 0.764_043, epsilon = 1e-5);
    }

    #[test]
    fn eyring_converges_to_sabine_for_small_absorption() {
        // Pour ᾱ faible, -ln(1-ᾱ) ≈ ᾱ, donc Eyring ≈ Sabine avec A = ᾱ·S.
        let (volume, surface, alpha) = (100.0_f64, 200.0_f64, 1.0e-4_f64);
        let eyring = reverb_eyring_time(volume, surface, alpha);
        let sabine = reverb_sabine_time(volume, alpha * surface);
        assert_relative_eq!(eyring, sabine, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "coefficient d'absorption moyen")]
    fn eyring_full_absorption_panics() {
        reverb_eyring_time(100.0, 200.0, 1.0);
    }
}
