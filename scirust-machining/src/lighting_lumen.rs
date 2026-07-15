//! Éclairagisme — **méthode des lumens** : éclairement **moyen** sur le plan
//! utile, dimensionnement du nombre de luminaires, indice du local et flux total.
//!
//! ```text
//! éclairement moyen   E = N·n·Φ·UF·MF / A            (lux)
//! luminaires requis   N = E·A / (n·Φ·UF·MF)          (nombre)
//! indice du local     K = (L·W) / (h·(L+W))          (sans dimension)
//! flux total installé Φ_tot = E·A / (UF·MF)          (lumen)
//! ```
//!
//! `N` nombre de luminaires, `n` lampes par luminaire, `Φ` flux d'une lampe
//! (lm), `UF` facteur d'utilisation (sans unité), `MF` facteur de maintenance
//! (sans unité), `A` surface du plan utile (m²), `E` éclairement (lux = lm/m²),
//! `L`,`W` longueur et largeur du local (m), `h` hauteur de suspension au-dessus
//! du plan utile (m), `Φ_tot` flux total installé (lm).
//!
//! **Convention** : unités SI, surfaces en m², flux en lumen, éclairement en lux.
//! **Limite honnête** : la méthode des lumens ne fournit que l'éclairement
//! **moyen** sur le plan utile — ni l'**uniformité**, ni l'**éblouissement**
//! (UGR). Le facteur d'utilisation `UF` (fonction de l'indice du local et des
//! réflectances des parois, issu de la **photométrie** du luminaire) et le
//! facteur de maintenance `MF` sont **fournis par l'appelant** ; aucune valeur
//! « par défaut » n'est inventée ici.

/// Éclairement moyen `E = N·n·Φ·UF·MF / A` (lux).
///
/// Panique si un paramètre est négatif ou nul, ou si `UF`/`MF` sortent de `]0, 1]`.
pub fn lighting_average_illuminance(
    luminaire_count: f64,
    lamps_per_luminaire: f64,
    lumens_per_lamp: f64,
    utilization_factor: f64,
    maintenance_factor: f64,
    area: f64,
) -> f64 {
    assert!(
        luminaire_count > 0.0 && lamps_per_luminaire > 0.0 && lumens_per_lamp > 0.0,
        "N, n et Φ doivent être strictement positifs"
    );
    assert!(area > 0.0, "la surface A doit être strictement positive");
    assert!(
        utilization_factor > 0.0 && utilization_factor <= 1.0,
        "UF doit être dans ]0, 1]"
    );
    assert!(
        maintenance_factor > 0.0 && maintenance_factor <= 1.0,
        "MF doit être dans ]0, 1]"
    );
    luminaire_count
        * lamps_per_luminaire
        * lumens_per_lamp
        * utilization_factor
        * maintenance_factor
        / area
}

/// Nombre de luminaires requis `N = E·A / (n·Φ·UF·MF)` (résultat réel, non arrondi).
///
/// Panique si un paramètre est négatif ou nul, ou si `UF`/`MF` sortent de `]0, 1]`.
pub fn lighting_luminaires_required(
    target_illuminance: f64,
    area: f64,
    lamps_per_luminaire: f64,
    lumens_per_lamp: f64,
    utilization_factor: f64,
    maintenance_factor: f64,
) -> f64 {
    assert!(
        target_illuminance > 0.0 && area > 0.0,
        "E et A doivent être strictement positifs"
    );
    assert!(
        lamps_per_luminaire > 0.0 && lumens_per_lamp > 0.0,
        "n et Φ doivent être strictement positifs"
    );
    assert!(
        utilization_factor > 0.0 && utilization_factor <= 1.0,
        "UF doit être dans ]0, 1]"
    );
    assert!(
        maintenance_factor > 0.0 && maintenance_factor <= 1.0,
        "MF doit être dans ]0, 1]"
    );
    target_illuminance * area
        / (lamps_per_luminaire * lumens_per_lamp * utilization_factor * maintenance_factor)
}

/// Indice du local `K = (L·W) / (h·(L+W))` (sans dimension).
///
/// Panique si une dimension est négative ou nulle.
pub fn lighting_room_index(room_length: f64, room_width: f64, mounting_height: f64) -> f64 {
    assert!(
        room_length > 0.0 && room_width > 0.0 && mounting_height > 0.0,
        "L, W et h doivent être strictement positifs"
    );
    (room_length * room_width) / (mounting_height * (room_length + room_width))
}

/// Flux total installé `Φ_tot = E·A / (UF·MF)` (lumen).
///
/// Panique si un paramètre est négatif ou nul, ou si `UF`/`MF` sortent de `]0, 1]`.
pub fn lighting_flux_required(
    target_illuminance: f64,
    area: f64,
    utilization_factor: f64,
    maintenance_factor: f64,
) -> f64 {
    assert!(
        target_illuminance > 0.0 && area > 0.0,
        "E et A doivent être strictement positifs"
    );
    assert!(
        utilization_factor > 0.0 && utilization_factor <= 1.0,
        "UF doit être dans ]0, 1]"
    );
    assert!(
        maintenance_factor > 0.0 && maintenance_factor <= 1.0,
        "MF doit être dans ]0, 1]"
    );
    target_illuminance * area / (utilization_factor * maintenance_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn average_illuminance_realistic_case() {
        // 20 luminaires × 2 lampes × 3200 lm, UF=0,6, MF=0,8, A=100 m² :
        // E = 20·2·3200·0,6·0,8/100 = 128000·0,48/100 = 614,4 lux.
        let e = lighting_average_illuminance(20.0, 2.0, 3200.0, 0.6, 0.8, 100.0);
        assert_relative_eq!(e, 614.4, max_relative = 1e-12);
    }

    #[test]
    fn illuminance_and_count_are_reciprocal() {
        // À partir de E calculé, retrouver exactement N = 20.
        let e = lighting_average_illuminance(20.0, 2.0, 3200.0, 0.6, 0.8, 100.0);
        let n = lighting_luminaires_required(e, 100.0, 2.0, 3200.0, 0.6, 0.8);
        assert_relative_eq!(n, 20.0, max_relative = 1e-12);
    }

    #[test]
    fn room_index_symmetric_and_valued() {
        // K = (L·W)/(h·(L+W)) est symétrique en L et W.
        let k1 = lighting_room_index(10.0, 6.0, 3.0);
        let k2 = lighting_room_index(6.0, 10.0, 3.0);
        assert_relative_eq!(k1, k2, max_relative = 1e-12);
        // Cas chiffré : (10·6)/(3·16) = 60/48 = 1,25.
        assert_relative_eq!(k1, 1.25, max_relative = 1e-12);
    }

    #[test]
    fn flux_required_matches_count_times_lamp_flux() {
        // N·(n·Φ) = Φ_tot puisque N = E·A/(n·Φ·UF·MF) et Φ_tot = E·A/(UF·MF).
        let (e, area, n_lamps, phi, uf, mf) =
            (500.0_f64, 100.0_f64, 2.0_f64, 3200.0_f64, 0.6_f64, 0.8_f64);
        let flux = lighting_flux_required(e, area, uf, mf);
        let count = lighting_luminaires_required(e, area, n_lamps, phi, uf, mf);
        assert_relative_eq!(count * n_lamps * phi, flux, max_relative = 1e-12);
        // Cas chiffré : Φ_tot = 500·100/(0,6·0,8) = 50000/0,48 ≈ 104166,67 lm.
        assert_relative_eq!(flux, 104_166.667, max_relative = 1e-6);
    }

    #[test]
    fn illuminance_proportional_to_luminaire_count() {
        // E ∝ N à tous les autres paramètres fixés.
        let e1 = lighting_average_illuminance(10.0, 2.0, 3200.0, 0.6, 0.8, 100.0);
        let e2 = lighting_average_illuminance(30.0, 2.0, 3200.0, 0.6, 0.8, 100.0);
        assert_relative_eq!(e2 / e1, 3.0, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "UF doit être dans ]0, 1]")]
    fn utilization_factor_above_one_panics() {
        lighting_average_illuminance(20.0, 2.0, 3200.0, 1.2, 0.8, 100.0);
    }
}
