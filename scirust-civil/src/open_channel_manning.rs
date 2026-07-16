//! Hydraulique à surface libre (canal ouvert) — formule de **Manning-Strickler**
//! pour la vitesse moyenne et le débit en régime permanent uniforme, rayon
//! hydraulique et profondeur critique d'une section rectangulaire.
//!
//! ```text
//! rayon hydraulique     Rh = A / P
//! Manning-Strickler     V  = (1/n)·Rh^{2/3}·√S      (vitesse moyenne)
//! débit                 Q  = V·A
//! section rectangulaire A  = b·y
//! profondeur critique   yc = (Q² / (b²·g))^{1/3}    (canal rectangulaire)
//! ```
//!
//! `n` coefficient de Manning (rugosité du revêtement, s·m^{−1/3}), `Rh` rayon
//! hydraulique (m), `S` pente du radier (m/m, faible), `V` vitesse moyenne (m/s),
//! `A` section mouillée (m²), `P` périmètre mouillé (m), `Q` débit (m³/s),
//! `b` largeur du canal (m), `y` profondeur d'eau (m), `yc` profondeur critique
//! (m), `g` accélération de la pesanteur (m/s²).
//!
//! **Convention** : SI strict et cohérent — mètres (m) et secondes (s). Le
//! coefficient de Manning est exprimé en s·m^{−1/3} de sorte que la formule
//! s'écrit sans facteur d'unité additionnel. Types `f64`.
//!
//! **Limite honnête** : écoulement **permanent et uniforme** (hypothèse de
//! Manning : pente motrice = pente du radier). Le coefficient de rugosité `n`
//! est **fourni par l'appelant** d'après le revêtement du canal (jamais une
//! valeur « par défaut » inventée), la pente du radier `S` (faible) est
//! **fournie**, et le rayon hydraulique découle de la géométrie **fournie**.
//! La profondeur critique suppose une **section rectangulaire**. Ce module ne
//! traite ni les écoulements graduellement variés (courbes de remous) ni les
//! écoulements rapidement variés (ressaut hydraulique).

/// Rayon hydraulique `Rh = A / P` (m), rapport de la section mouillée sur le
/// périmètre mouillé.
///
/// Panique si `flow_area < 0` ou si `wetted_perimeter <= 0`.
pub fn channel_hydraulic_radius(flow_area: f64, wetted_perimeter: f64) -> f64 {
    assert!(
        flow_area >= 0.0,
        "la section mouillée A doit être positive ou nulle"
    );
    assert!(
        wetted_perimeter > 0.0,
        "le périmètre mouillé P doit être strictement positif"
    );
    flow_area / wetted_perimeter
}

/// Section mouillée d'un canal rectangulaire `A = b·y` (m²).
///
/// Panique si `width < 0` ou `depth < 0`.
pub fn channel_rectangular_area(width: f64, depth: f64) -> f64 {
    assert!(width >= 0.0, "la largeur b doit être positive ou nulle");
    assert!(depth >= 0.0, "la profondeur y doit être positive ou nulle");
    width * depth
}

/// Vitesse moyenne de Manning-Strickler `V = (1/n)·Rh^{2/3}·√S` (m/s).
///
/// Panique si `manning_n <= 0`, `hydraulic_radius < 0` ou `bed_slope < 0`.
pub fn channel_manning_velocity(manning_n: f64, hydraulic_radius: f64, bed_slope: f64) -> f64 {
    assert!(
        manning_n > 0.0,
        "le coefficient de Manning n doit être strictement positif"
    );
    assert!(
        hydraulic_radius >= 0.0,
        "le rayon hydraulique Rh doit être positif ou nul"
    );
    assert!(
        bed_slope >= 0.0,
        "la pente du radier S doit être positive ou nulle"
    );
    (1.0 / manning_n) * hydraulic_radius.powf(2.0 / 3.0) * bed_slope.sqrt()
}

/// Débit `Q = V·A` (m³/s), produit de la vitesse moyenne par la section mouillée.
///
/// Panique si `velocity < 0` ou `flow_area < 0`.
pub fn channel_manning_discharge(velocity: f64, flow_area: f64) -> f64 {
    assert!(velocity >= 0.0, "la vitesse V doit être positive ou nulle");
    assert!(
        flow_area >= 0.0,
        "la section mouillée A doit être positive ou nulle"
    );
    velocity * flow_area
}

/// Profondeur critique d'un canal rectangulaire `yc = (Q² / (b²·g))^{1/3}` (m).
///
/// Panique si `discharge < 0`, `width <= 0` ou `gravity <= 0`.
pub fn channel_critical_depth_rectangular(discharge: f64, width: f64, gravity: f64) -> f64 {
    assert!(discharge >= 0.0, "le débit Q doit être positif ou nul");
    assert!(width > 0.0, "la largeur b doit être strictement positive");
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    (discharge * discharge / (width * width * gravity)).cbrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hydraulic_radius_of_wide_rectangular_channel() {
        // Canal rectangulaire b = 2 m, y = 1 m : A = 2, P = b + 2y = 4, Rh = 0,5.
        let area = channel_rectangular_area(2.0, 1.0);
        assert_relative_eq!(area, 2.0, max_relative = 1e-12);
        let rh = channel_hydraulic_radius(area, 2.0 + 2.0 * 1.0);
        assert_relative_eq!(rh, 0.5, max_relative = 1e-12);
    }

    #[test]
    fn discharge_is_area_times_velocity() {
        // Identité Q = V·A : proportionnalité stricte au produit.
        let v = 1.5_f64;
        let a = 2.0_f64;
        assert_relative_eq!(channel_manning_discharge(v, a), 3.0, max_relative = 1e-12);
        // Doubler la section double le débit à vitesse fixe.
        assert_relative_eq!(
            channel_manning_discharge(v, 2.0 * a),
            2.0 * channel_manning_discharge(v, a),
            max_relative = 1e-12
        );
    }

    #[test]
    fn velocity_scales_with_inverse_manning_and_sqrt_slope() {
        // V ∝ 1/n : diviser n par deux double la vitesse (Rh, S fixés).
        let v1 = channel_manning_velocity(0.026, 0.5, 0.001);
        let v2 = channel_manning_velocity(0.013, 0.5, 0.001);
        assert_relative_eq!(v2, 2.0 * v1, max_relative = 1e-12);
        // V ∝ √S : quadrupler la pente double la vitesse (n, Rh fixés).
        let vs1 = channel_manning_velocity(0.013, 0.5, 0.001);
        let vs4 = channel_manning_velocity(0.013, 0.5, 0.004);
        assert_relative_eq!(vs4, 2.0 * vs1, max_relative = 1e-12);
    }

    #[test]
    fn zero_slope_gives_no_flow() {
        // Radier horizontal (S = 0) : pas de moteur, vitesse nulle.
        assert_relative_eq!(
            channel_manning_velocity(0.013, 0.5, 0.0),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn worked_case_concrete_channel() {
        // Canal béton b = 2 m, y = 1 m, n = 0,013, S = 0,001.
        // A = 2 m², P = 4 m, Rh = 0,5 m.
        // V = (1/0,013)·0,5^{2/3}·√0,001
        //   = 76,9231 · 0,629961 · 0,0316228 ≈ 1,5322 m/s.
        // Q = V·A ≈ 3,0645 m³/s.
        // yc = (Q²/(b²·g))^{1/3} = (9,3911/39,24)^{1/3}
        //    = 0,239324^{1/3} ≈ 0,6209 m.
        let area = channel_rectangular_area(2.0, 1.0);
        let rh = channel_hydraulic_radius(area, 4.0);
        let v = channel_manning_velocity(0.013, rh, 0.001);
        assert_relative_eq!(v, 1.5322, max_relative = 1e-3);
        let q = channel_manning_discharge(v, area);
        assert_relative_eq!(q, 3.0645, max_relative = 1e-3);
        let yc = channel_critical_depth_rectangular(q, 2.0, 9.81);
        assert_relative_eq!(yc, 0.6209, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le coefficient de Manning n doit être strictement positif")]
    fn zero_manning_panics() {
        let _ = channel_manning_velocity(0.0, 0.5, 0.001);
    }
}
