//! Poutre sur sol élastique — modèle de **Winkler** (fondation à ressorts
//! indépendants) : paramètre de rigidité relative sol/poutre, flèche et moment
//! maximaux sous charge concentrée pour une poutre infinie, classification
//! rigide/longue et pression de contact sol-poutre.
//!
//! ```text
//! paramètre de rigidité   β    = ( k·b / (4·E·I) )^(1/4)
//! flèche max (charge P)    y_max = P·β / (2·k·b)
//! moment max (charge P)    M_max = P / (4·β)
//! classification           βL   (rigide si βL < π/4, longue si βL > π)
//! pression de contact      p    = k·y
//! ```
//!
//! `k` module de réaction du sol (N/m³, dit aussi coefficient de Winkler), `b`
//! largeur de contact de la poutre (m), `E` module d'élasticité du matériau de
//! la poutre (Pa), `I` moment quadratique de la section (m⁴), `β` paramètre de
//! rigidité relative sol/poutre (m⁻¹), `P` charge concentrée (N), `y` flèche
//! verticale (m), `y_max` flèche maximale au droit de la charge (m), `M_max`
//! moment fléchissant maximal (N·m), `L` longueur de la poutre (m), `βL` produit
//! adimensionnel (–), `p` pression de contact sol-poutre (Pa, soit N/m²).
//!
//! **Convention** : SI strict et cohérent — newtons (N), mètres (m), pascals
//! (Pa) ; le module de réaction `k` est en N/m³ et la largeur `b` en m, de sorte
//! que `k·b` a la dimension N/m² (raideur linéique de sol). Le paramètre `β` est
//! en m⁻¹. Types `f64`.
//!
//! **Limite honnête** : modèle de **Winkler** — le sol est représenté par un
//! lit de ressorts **indépendants** (aucun couplage de cisaillement entre
//! ressorts voisins), hypothèse simplificatrice qui ignore la continuité réelle
//! du massif. Le **module de réaction du sol `k`** est **fourni par l'appelant**
//! et **n'est pas une propriété intrinsèque du sol** : il dépend de la géométrie
//! de la fondation (largeur, forme), du niveau de chargement et des conditions
//! d'appui, et doit être établi par essais de plaque ou corrélations — jamais
//! inventé. Les formules de flèche et de moment données ici valent pour une
//! **poutre infiniment longue** soumise à une **charge concentrée** isolée
//! (solution de Hetényi) ; pour une poutre courte, chargée en bord ou en
//! chargement multiple, l'appelant doit recourir à la solution complète. Le
//! comportement est supposé **élastique linéaire** (matériau de la poutre et
//! réaction du sol), sans décollement ni plastification.

/// Paramètre de rigidité relative sol/poutre `β = (k·b / (4·E·I))^(1/4)` (m⁻¹).
///
/// C'est l'inverse de la longueur caractéristique de la poutre sur sol
/// élastique : plus `β` est grand, plus la poutre est souple vis-à-vis du sol et
/// plus la réponse est localisée sous la charge.
///
/// Panique si `subgrade_modulus <= 0`, `width <= 0`, `elastic_modulus <= 0` ou
/// `inertia <= 0`.
pub fn boef_beta(subgrade_modulus: f64, width: f64, elastic_modulus: f64, inertia: f64) -> f64 {
    assert!(
        subgrade_modulus > 0.0,
        "le module de réaction du sol k doit être strictement positif"
    );
    assert!(
        width > 0.0,
        "la largeur de contact b doit être strictement positive"
    );
    assert!(
        elastic_modulus > 0.0,
        "le module d'élasticité E doit être strictement positif"
    );
    assert!(
        inertia > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    ((subgrade_modulus * width) / (4.0_f64 * elastic_modulus * inertia)).powf(0.25)
}

/// Flèche maximale sous charge concentrée sur poutre infinie
/// `y_max = P·β / (2·k·b)` (m), atteinte au droit de la charge.
///
/// Panique si `point_load < 0`, `beta <= 0`, `subgrade_modulus <= 0` ou
/// `width <= 0`.
pub fn boef_max_deflection_point_load(
    point_load: f64,
    beta: f64,
    subgrade_modulus: f64,
    width: f64,
) -> f64 {
    assert!(
        point_load >= 0.0,
        "la charge concentrée P doit être positive ou nulle"
    );
    assert!(beta > 0.0, "le paramètre β doit être strictement positif");
    assert!(
        subgrade_modulus > 0.0,
        "le module de réaction du sol k doit être strictement positif"
    );
    assert!(
        width > 0.0,
        "la largeur de contact b doit être strictement positive"
    );
    point_load * beta / (2.0_f64 * subgrade_modulus * width)
}

/// Moment fléchissant maximal sous charge concentrée sur poutre infinie
/// `M_max = P / (4·β)` (N·m), atteint au droit de la charge.
///
/// Panique si `point_load < 0` ou `beta <= 0`.
pub fn boef_max_moment_point_load(point_load: f64, beta: f64) -> f64 {
    assert!(
        point_load >= 0.0,
        "la charge concentrée P doit être positive ou nulle"
    );
    assert!(beta > 0.0, "le paramètre β doit être strictement positif");
    point_load / (4.0_f64 * beta)
}

/// Produit adimensionnel `βL` servant à classer la poutre sur sol élastique.
///
/// Convention usuelle (Hetényi) : `βL < π/4` → poutre **courte/rigide** (elle se
/// déplace quasi en bloc, le sol réagit uniformément) ; `βL > π` → poutre
/// **longue/souple** (la charge d'une extrémité n'influence plus l'autre, la
/// solution de poutre infinie s'applique) ; entre les deux, poutre de longueur
/// **intermédiaire** (la solution complète est requise).
///
/// Panique si `beta <= 0` ou `length <= 0`.
pub fn boef_classification(beta: f64, length: f64) -> f64 {
    assert!(beta > 0.0, "le paramètre β doit être strictement positif");
    assert!(length > 0.0, "la longueur L doit être strictement positive");
    beta * length
}

/// Pression de contact sol-poutre `p = k·y` (Pa), produit du module de réaction
/// du sol par la flèche locale (hypothèse de Winkler).
///
/// Panique si `subgrade_modulus <= 0`.
pub fn boef_contact_pressure(deflection: f64, subgrade_modulus: f64) -> f64 {
    assert!(
        subgrade_modulus > 0.0,
        "le module de réaction du sol k doit être strictement positif"
    );
    deflection * subgrade_modulus
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn beta_scales_as_fourth_root_of_subgrade() {
        // β ∝ (k·b)^(1/4) : multiplier k par 16 double β (E, I, b fixés).
        let b1 = boef_beta(20.0e6, 0.3, 30.0e9, 6.75e-4);
        let b2 = boef_beta(16.0 * 20.0e6, 0.3, 30.0e9, 6.75e-4);
        assert_relative_eq!(b2, 2.0 * b1, max_relative = 1e-12);
        // β ∝ (E·I)^(-1/4) : diviser E·I par 16 double également β.
        let b3 = boef_beta(20.0e6, 0.3, 30.0e9 / 16.0, 6.75e-4);
        assert_relative_eq!(b3, 2.0 * b1, max_relative = 1e-12);
    }

    #[test]
    fn deflection_and_moment_are_linear_in_load() {
        // y_max et M_max sont proportionnels à P (β, k, b fixés).
        let beta = 0.5_f64;
        let y1 = boef_max_deflection_point_load(100.0e3, beta, 20.0e6, 0.3);
        let y2 = boef_max_deflection_point_load(300.0e3, beta, 20.0e6, 0.3);
        assert_relative_eq!(y2, 3.0 * y1, max_relative = 1e-12);
        let m1 = boef_max_moment_point_load(100.0e3, beta);
        let m2 = boef_max_moment_point_load(300.0e3, beta);
        assert_relative_eq!(m2, 3.0 * m1, max_relative = 1e-12);
        // Charge nulle : flèche et moment nuls.
        assert_relative_eq!(
            boef_max_deflection_point_load(0.0, beta, 20.0e6, 0.3),
            0.0,
            epsilon = 1e-15
        );
        assert_relative_eq!(boef_max_moment_point_load(0.0, beta), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn contact_pressure_recovers_subgrade_law() {
        // Identité de Winkler : p = k·y ; et p / y = k pour toute flèche.
        let k = 20.0e6_f64;
        let y = 0.004_f64;
        assert_relative_eq!(boef_contact_pressure(y, k), k * y, max_relative = 1e-12);
        // Composition avec la flèche max : p_max = k·y_max = P·β / (2·b).
        let beta = 0.5_f64;
        let p = 100.0e3_f64;
        let b = 0.3_f64;
        let y_max = boef_max_deflection_point_load(p, beta, k, b);
        assert_relative_eq!(
            boef_contact_pressure(y_max, k),
            p * beta / (2.0 * b),
            max_relative = 1e-12
        );
    }

    #[test]
    fn classification_thresholds_bracket_a_medium_beam() {
        // βL croît avec L à β fixé ; on encadre les seuils π/4 et π.
        let beta = 0.5216948600244291_f64;
        // L = 3 m → βL ≈ 1,565 : au-dessus de π/4 (rigide) mais sous π (longue).
        let medium = boef_classification(beta, 3.0);
        assert!(medium > PI / 4.0);
        assert!(medium < PI);
        // L = 8 m → βL ≈ 4,17 > π : régime de poutre longue (infinie).
        let long = boef_classification(beta, 8.0);
        assert!(long > PI);
    }

    #[test]
    fn worked_numeric_case() {
        // Poutre béton 0,30 × 0,30 m : I = b·h³/12 = 0,3·0,3³/12 = 6,75e-4 m⁴,
        // E = 30 GPa, sol k = 20 MN/m³, largeur de contact b = 0,30 m, P = 100 kN.
        //   k·b / (4·E·I) = 6,0e6 / 8,1e7 = 0,074074074…
        //   β = 0,074074074^0,25 = 0,521694860… m⁻¹
        //   y_max = P·β / (2·k·b) = 100e3·0,52169486 / (2·20e6·0,3) = 4,347457e-3 m
        //   M_max = P / (4·β) = 100e3 / (4·0,52169486) = 47920,7328 N·m
        //   p_max = k·y_max = 20e6·4,347457e-3 = 86949,1433 Pa
        let k = 20.0e6_f64;
        let b = 0.3_f64;
        let e = 30.0e9_f64;
        let i = 6.75e-4_f64;
        let p = 100.0e3_f64;
        let beta = boef_beta(k, b, e, i);
        assert_relative_eq!(beta, 0.521_694_860_024, max_relative = 1e-3);
        let y_max = boef_max_deflection_point_load(p, beta, k, b);
        assert_relative_eq!(y_max, 0.004_347_457_167, max_relative = 1e-3);
        let m_max = boef_max_moment_point_load(p, beta);
        assert_relative_eq!(m_max, 47_920.732_818, max_relative = 1e-3);
        let p_contact = boef_contact_pressure(y_max, k);
        assert_relative_eq!(p_contact, 86_949.143_337, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le module de réaction du sol k doit être strictement positif")]
    fn zero_subgrade_modulus_panics() {
        let _ = boef_beta(0.0, 0.3, 30.0e9, 6.75e-4);
    }
}
