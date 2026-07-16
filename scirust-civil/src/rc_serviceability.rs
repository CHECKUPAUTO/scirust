//! **Béton armé — états limites de service (Eurocode 2, ELS)** : limites de
//! contrainte du béton et de l'acier, espacement maximal des fissures `sr,max`,
//! ouverture de fissure `wk` et vérification simplifiée de la flèche par le
//! rapport portée/hauteur utile `L/d`.
//!
//! ```text
//! contrainte béton   σc,lim  = factor · fck
//! contrainte acier   σs,lim  = factor · fyk
//! espacement fissures sr,max = k3 · c + k1 · k2 · k4 · φ / ρp,eff
//! ouverture fissure   wk     = sr,max · (εsm − εcm)
//! maîtrise flèche            L / d ≤ (L/d)lim
//! ```
//!
//! `σc,lim` limite de contrainte de compression du béton (MPa), `fck`
//! résistance caractéristique en compression du béton (MPa), `σs,lim` limite de
//! contrainte de traction de l'acier (MPa), `fyk` limite caractéristique
//! d'élasticité de l'acier (MPa), `factor` coefficient de limitation de l'EC2
//! (sans dimension, p. ex. `0,6·fck` ou `0,8·fyk` en combinaison
//! caractéristique), `sr,max` espacement maximal des fissures (mm), `c`
//! enrobage des armatures (mm), `k1..k4` coefficients réglementaires de l'EC2
//! (sans dimension : `k1` adhérence, `k2` diagramme des déformations, `k3` et
//! `k4` valeurs recommandées/de l'Annexe Nationale), `φ` diamètre des barres
//! (mm), `ρp,eff` ratio d'armature effectif dans la zone tendue (sans
//! dimension), `wk` ouverture de fissure (mm), `εsm − εcm` différence de
//! déformation moyenne acier/béton (sans dimension), `L` portée (mm), `d`
//! hauteur utile (mm), `(L/d)lim` rapport limite (sans dimension).
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`) ; les contraintes sont
//! en **MPa**, les longueurs (enrobage, diamètre, espacement, ouverture,
//! portée, hauteur utile) en **mm**, les déformations et ratios **sans
//! dimension**.
//! **Limite honnête** : vérifications ELS **simplifiées**. Les résistances
//! caractéristiques (`fck`, `fyk`, `fy`…) **et** les coefficients partiels de
//! sécurité (`γc`, `γs`, `γM`…), les coefficients réglementaires (`k1`, `k2`,
//! `k3`, `k4`, les `factor` de limitation de contrainte, le rapport limite
//! `(L/d)lim`) ainsi que la différence de déformation `εsm − εcm` sont
//! **fournis par l'appelant** d'après l'Eurocode 2 et son Annexe Nationale ;
//! aucune valeur « par défaut » n'est inventée. La flèche n'est **pas** calculée
//! par intégration des courbures : seule la maîtrise par le rapport `L/d` est
//! proposée. Le choix des combinaisons d'actions et la conclusion réglementaire
//! restent à la charge de l'ingénieur.

/// Limite de contrainte de compression du béton `σc,lim = factor · fck` (MPa),
/// avec `fck` en MPa et `factor` le coefficient de limitation de l'EC2 (p. ex.
/// `0,6` en combinaison caractéristique).
///
/// Panique si `fck <= 0` ou si `factor <= 0`.
pub fn rcsls_concrete_stress_limit(fck: f64, factor: f64) -> f64 {
    assert!(
        fck > 0.0,
        "la résistance fck doit être strictement positive"
    );
    assert!(
        factor > 0.0,
        "le coefficient de limitation factor doit être strictement positif"
    );
    factor * fck
}

/// Limite de contrainte de traction de l'acier `σs,lim = factor · fyk` (MPa),
/// avec `fyk` en MPa et `factor` le coefficient de limitation de l'EC2 (p. ex.
/// `0,8` en combinaison caractéristique).
///
/// Panique si `fyk <= 0` ou si `factor <= 0`.
pub fn rcsls_steel_stress_limit(fyk: f64, factor: f64) -> f64 {
    assert!(fyk > 0.0, "la limite fyk doit être strictement positive");
    assert!(
        factor > 0.0,
        "le coefficient de limitation factor doit être strictement positif"
    );
    factor * fyk
}

/// Espacement maximal des fissures
/// `sr,max = k3 · c + k1 · k2 · k4 · φ / ρp,eff` (mm), avec l'enrobage `c` et le
/// diamètre `φ` en mm, les `k1..k4` et le ratio `ρp,eff` sans dimension.
///
/// Panique si `cover < 0`, si l'un des coefficients `coefficient_k1`,
/// `coefficient_k2`, `coefficient_k3`, `coefficient_k4` est `<= 0`, si
/// `bar_diameter <= 0` ou si `effective_reinforcement_ratio <= 0` (division).
pub fn rcsls_crack_spacing(
    cover: f64,
    coefficient_k3: f64,
    coefficient_k1: f64,
    coefficient_k2: f64,
    coefficient_k4: f64,
    bar_diameter: f64,
    effective_reinforcement_ratio: f64,
) -> f64 {
    assert!(cover >= 0.0, "l'enrobage c doit être ≥ 0");
    assert!(
        coefficient_k3 > 0.0,
        "le coefficient k3 doit être strictement positif"
    );
    assert!(
        coefficient_k1 > 0.0,
        "le coefficient k1 doit être strictement positif"
    );
    assert!(
        coefficient_k2 > 0.0,
        "le coefficient k2 doit être strictement positif"
    );
    assert!(
        coefficient_k4 > 0.0,
        "le coefficient k4 doit être strictement positif"
    );
    assert!(
        bar_diameter > 0.0,
        "le diamètre des barres φ doit être strictement positif"
    );
    assert!(
        effective_reinforcement_ratio > 0.0,
        "le ratio d'armature effectif ρp,eff doit être strictement positif"
    );
    coefficient_k3 * cover
        + coefficient_k1 * coefficient_k2 * coefficient_k4 * bar_diameter
            / effective_reinforcement_ratio
}

/// Ouverture de fissure `wk = sr,max · (εsm − εcm)` (mm), avec l'espacement
/// `sr,max` en mm et la différence de déformation `εsm − εcm` sans dimension.
///
/// Panique si `crack_spacing < 0` ou si `strain_difference < 0`.
pub fn rcsls_crack_width(crack_spacing: f64, strain_difference: f64) -> f64 {
    assert!(
        crack_spacing >= 0.0,
        "l'espacement des fissures sr,max doit être ≥ 0"
    );
    assert!(
        strain_difference >= 0.0,
        "la différence de déformation εsm − εcm doit être ≥ 0"
    );
    crack_spacing * strain_difference
}

/// Maîtrise simplifiée de la flèche : renvoie `true` si `L / d ≤ (L/d)lim`,
/// avec la portée `L` et la hauteur utile `d` en mm et le rapport limite sans
/// dimension.
///
/// Panique si `span <= 0`, si `effective_depth <= 0` ou si `limit_ratio <= 0`.
pub fn rcsls_span_depth_ratio_ok(span: f64, effective_depth: f64, limit_ratio: f64) -> bool {
    assert!(span > 0.0, "la portée L doit être strictement positive");
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    assert!(
        limit_ratio > 0.0,
        "le rapport limite (L/d)lim doit être strictement positif"
    );
    span / effective_depth <= limit_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn concrete_stress_limit_case_and_proportionality() {
        // Cas usuel : 0,6 · fck en combinaison caractéristique, fck = 30 MPa
        //   σc,lim = 0,6 · 30 = 18 MPa
        let sigma = rcsls_concrete_stress_limit(30.0, 0.6);
        assert_relative_eq!(sigma, 18.0, epsilon = 1e-12);
        // Proportionnalité : doubler fck double la limite.
        let sigma2 = rcsls_concrete_stress_limit(60.0, 0.6);
        assert_relative_eq!(sigma2 / sigma, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn steel_stress_limit_case() {
        // Cas usuel : 0,8 · fyk, acier B500
        //   σs,lim = 0,8 · 500 = 400 MPa
        let sigma = rcsls_steel_stress_limit(500.0, 0.8);
        assert_relative_eq!(sigma, 400.0, epsilon = 1e-12);
    }

    #[test]
    fn crack_spacing_clean_case() {
        // Cas chiffré (nombres choisis pour un résultat entier) :
        //   k3 · c            = 3,4 · 30 = 102
        //   k1 · k2 · k4      = 0,8 · 0,5 · 0,425 = 0,17
        //   0,17 · φ / ρ      = 0,17 · 16 / 0,02 = 2,72 / 0,02 = 136
        //   sr,max           = 102 + 136 = 238 mm
        let sr = rcsls_crack_spacing(30.0, 3.4, 0.8, 0.5, 0.425, 16.0, 0.02);
        assert_relative_eq!(sr, 238.0, epsilon = 1e-6);
    }

    #[test]
    fn crack_spacing_second_term_scales_with_diameter() {
        // À enrobage nul, seul le second terme subsiste et il est linéaire en φ :
        // doubler le diamètre double l'espacement.
        let sr1 = rcsls_crack_spacing(0.0, 3.4, 0.8, 0.5, 0.425, 16.0, 0.02);
        let sr2 = rcsls_crack_spacing(0.0, 3.4, 0.8, 0.5, 0.425, 32.0, 0.02);
        assert_relative_eq!(sr2 / sr1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn crack_width_composes_with_spacing() {
        // Identité de composition : wk = sr,max · (εsm − εcm).
        //   sr,max = 238 mm (cas ci-dessus), εsm − εcm = 0,001
        //   wk = 238 · 0,001 = 0,238 mm
        let sr = rcsls_crack_spacing(30.0, 3.4, 0.8, 0.5, 0.425, 16.0, 0.02);
        let wk = rcsls_crack_width(sr, 0.001);
        assert_relative_eq!(wk, 0.238, epsilon = 1e-6);
        // Un espacement nul donne une ouverture nulle.
        assert_relative_eq!(rcsls_crack_width(0.0, 0.001), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn span_depth_ratio_boundary() {
        // L = 6000 mm, d = 300 mm → L/d = 20, exactement à la limite.
        assert!(rcsls_span_depth_ratio_ok(6000.0, 300.0, 20.0));
        // Même géométrie mais limite plus sévère (18) → non vérifié.
        assert!(!rcsls_span_depth_ratio_ok(6000.0, 300.0, 18.0));
    }

    #[test]
    #[should_panic(expected = "le ratio d'armature effectif ρp,eff doit être strictement positif")]
    fn crack_spacing_rejects_zero_ratio() {
        // ρp,eff = 0 : division par zéro interdite.
        rcsls_crack_spacing(30.0, 3.4, 0.8, 0.5, 0.425, 16.0, 0.0);
    }
}
