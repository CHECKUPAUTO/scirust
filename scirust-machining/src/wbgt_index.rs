//! Indice de température au thermomètre-globe mouillé (WBGT) — évaluation du
//! stress thermique selon la pondération normalisée de l'ISO 7243.
//!
//! ```text
//! extérieur (soleil)  WBGT = 0.7·Tnw + 0.2·Tg + 0.1·Td      (°C)
//! intérieur (sans soleil) WBGT = 0.7·Tnw + 0.3·Tg           (°C)
//! ajustement vêtement WBGT_eff = WBGT + ΔT_clo              (°C)
//! ```
//!
//! `Tnw` température au thermomètre humide naturel (°C), `Tg` température au
//! thermomètre-globe (°C), `Td` température de l'air au thermomètre sec (°C),
//! `WBGT` indice résultant (°C), `ΔT_clo` ajustement vestimentaire ajouté à
//! l'indice (°C), `WBGT_eff` indice effectif corrigé du vêtement (°C).
//!
//! **Convention** : températures en °C ; les pondérations (0.7 / 0.2 / 0.1 en
//! extérieur, 0.7 / 0.3 en intérieur) somment à 1 et sont celles de l'ISO 7243.
//! **Limite honnête** : les pondérations sont **normalisées** par l'indice WBGT
//! (ISO 7243) ; les températures mesurées (globe, humide naturel, sec) sont
//! **fournies par l'appelant**. L'ajustement vestimentaire `ΔT_clo` et les
//! limites d'exposition admissibles (fonction du métabolisme et de l'acclimatation)
//! relèvent de la **réglementation fournie** — aucune valeur n'est inventée ici.

/// Zéro absolu en degrés Celsius, borne physique inférieure des températures.
const WBGT_ABSOLUTE_ZERO_CELSIUS: f64 = -273.15;

/// Indice WBGT extérieur (ensoleillé)
/// `WBGT = 0.7·Tnw + 0.2·Tg + 0.1·Td` (°C).
///
/// Panique si l'une des températures est sous le zéro absolu (`< -273.15 °C`).
pub fn wbgt_outdoor(natural_wet_bulb: f64, globe_temperature: f64, dry_bulb: f64) -> f64 {
    assert!(
        natural_wet_bulb >= WBGT_ABSOLUTE_ZERO_CELSIUS,
        "Tnw ≥ -273.15 °C requis (au-dessus du zéro absolu)"
    );
    assert!(
        globe_temperature >= WBGT_ABSOLUTE_ZERO_CELSIUS,
        "Tg ≥ -273.15 °C requis (au-dessus du zéro absolu)"
    );
    assert!(
        dry_bulb >= WBGT_ABSOLUTE_ZERO_CELSIUS,
        "Td ≥ -273.15 °C requis (au-dessus du zéro absolu)"
    );
    0.7 * natural_wet_bulb + 0.2 * globe_temperature + 0.1 * dry_bulb
}

/// Indice WBGT intérieur ou sans ensoleillement
/// `WBGT = 0.7·Tnw + 0.3·Tg` (°C).
///
/// Panique si l'une des températures est sous le zéro absolu (`< -273.15 °C`).
pub fn wbgt_indoor(natural_wet_bulb: f64, globe_temperature: f64) -> f64 {
    assert!(
        natural_wet_bulb >= WBGT_ABSOLUTE_ZERO_CELSIUS,
        "Tnw ≥ -273.15 °C requis (au-dessus du zéro absolu)"
    );
    assert!(
        globe_temperature >= WBGT_ABSOLUTE_ZERO_CELSIUS,
        "Tg ≥ -273.15 °C requis (au-dessus du zéro absolu)"
    );
    0.7 * natural_wet_bulb + 0.3 * globe_temperature
}

/// Indice WBGT corrigé de l'ajustement vestimentaire
/// `WBGT_eff = WBGT + ΔT_clo` (°C), l'ajustement étant **fourni** par la
/// réglementation.
///
/// Panique si `wbgt < -273.15 °C` ou si `clothing_adjustment < 0` (un
/// vêtement ajoute au stress thermique, il ne le réduit pas dans ce modèle).
pub fn wbgt_with_clothing_adjustment(wbgt: f64, clothing_adjustment: f64) -> f64 {
    assert!(
        wbgt >= WBGT_ABSOLUTE_ZERO_CELSIUS,
        "WBGT ≥ -273.15 °C requis (au-dessus du zéro absolu)"
    );
    assert!(clothing_adjustment >= 0.0, "ΔT_clo ≥ 0 requis");
    wbgt + clothing_adjustment
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn outdoor_realistic_case() {
        // Chantier ensoleillé : Tnw=25, Tg=35, Td=30 °C.
        // WBGT = 0.7·25 + 0.2·35 + 0.1·30 = 17.5 + 7.0 + 3.0 = 27.5 °C.
        let w = wbgt_outdoor(25.0, 35.0, 30.0);
        assert_relative_eq!(w, 27.5, max_relative = 1e-12);
    }

    #[test]
    fn indoor_realistic_case() {
        // Atelier sans soleil : Tnw=25, Tg=30 °C.
        // WBGT = 0.7·25 + 0.3·30 = 17.5 + 9.0 = 26.5 °C.
        let w = wbgt_indoor(25.0, 30.0);
        assert_relative_eq!(w, 26.5, max_relative = 1e-12);
    }

    #[test]
    fn outdoor_reduces_to_indoor_when_globe_equals_dry() {
        // Identité : si Td = Tg, alors 0.2·Tg + 0.1·Tg = 0.3·Tg,
        // donc l'expression extérieure coïncide avec l'intérieure.
        let out = wbgt_outdoor(24.0, 31.0, 31.0);
        let ind = wbgt_indoor(24.0, 31.0);
        assert_relative_eq!(out, ind, max_relative = 1e-12);
    }

    #[test]
    fn outdoor_equals_common_temperature_when_all_equal() {
        // Les pondérations somment à 1 : si Tnw=Tg=Td=T, WBGT = T.
        let t = 28.3;
        assert_relative_eq!(wbgt_outdoor(t, t, t), t, max_relative = 1e-12);
    }

    #[test]
    fn indoor_equals_common_temperature_when_both_equal() {
        // 0.7 + 0.3 = 1 : si Tnw = Tg = T, WBGT = T.
        let t = 22.0;
        assert_relative_eq!(wbgt_indoor(t, t), t, max_relative = 1e-12);
    }

    #[test]
    fn clothing_adjustment_is_additive() {
        // WBGT_eff = WBGT + ΔT_clo : 27.5 + 3.5 = 31.0 °C.
        let base = wbgt_outdoor(25.0, 35.0, 30.0);
        assert_relative_eq!(
            wbgt_with_clothing_adjustment(base, 3.5),
            31.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "ΔT_clo ≥ 0")]
    fn negative_clothing_adjustment_panics() {
        wbgt_with_clothing_adjustment(27.5, -1.0);
    }
}
