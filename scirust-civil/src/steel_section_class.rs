//! **Charpente métallique — classification des sections comprimées (Eurocode 3,
//! tableau 5.2)** : coefficient `ε` dépendant de la limite d'élasticité,
//! élancements `c/t` d'une semelle en console et d'une âme, puis attribution de
//! la classe (1 à 4) d'une semelle comprimée et d'une âme comprimée.
//!
//! ```text
//! coefficient          ε   = √(235 / fy)
//! élancement semelle   c/t = flange_outstand / flange_thickness
//! élancement âme       c/t = web_clear_depth / web_thickness
//!
//! semelle comprimée en console (paroi console)
//!   c/t ≤  9·ε → classe 1 ; ≤ 10·ε → classe 2 ; ≤ 14·ε → classe 3 ; sinon 4
//! âme comprimée (paroi interne, compression uniforme)
//!   c/t ≤ 33·ε → classe 1 ; ≤ 38·ε → classe 2 ; ≤ 42·ε → classe 3 ; sinon 4
//! ```
//!
//! `ε` coefficient (sans dimension), `fy` limite d'élasticité de l'acier (MPa),
//! `c` = `flange_outstand`/`web_clear_depth` largeur droite comprimée de la paroi
//! (mm), `t` = `flange_thickness`/`web_thickness` épaisseur de la paroi (mm),
//! `c/t` élancement (sans dimension), classe entière `1..=4`.
//!
//! **Convention** : N, mm, MPa (avec `1 MPa = 1 N/mm²`) ; les élancements et `ε`
//! sont des grandeurs sans dimension.
//! **Limite honnête** : classification en **compression uniforme** (semelle en
//! console d'après le tableau 5.2 feuillet 2, âme d'après le feuillet 1) selon
//! l'Eurocode 3 (EN 1993-1-1, tableau 5.2) ; la classe 1 correspond à une section
//! plastique, la classe 4 au voilement local. Le coefficient `ε` dépend de la
//! limite d'élasticité caractéristique `fy` **fournie par l'appelant** d'après
//! l'Eurocode et son Annexe Nationale ; aucune nuance ni valeur « par défaut »
//! n'est inventée. La classification en **flexion** ou en **flexion composée**
//! (âme partiellement comprimée, semelle intérieure d'un caisson…) emploie
//! d'autres bornes du tableau 5.2 et reste **à la charge de l'appelant**, de même
//! que le calcul de `c` (largeur droite après déduction des rayons de raccordement
//! ou des cordons de soudure).

/// Coefficient `ε = √(235 / fy)` de l'Eurocode 3 (sans dimension), avec la limite
/// d'élasticité `fy` en MPa (la valeur de référence `235` est en MPa).
///
/// Panique si `fy <= 0` (racine d'un rapport négatif et division par zéro).
pub fn steelclass_epsilon(fy: f64) -> f64 {
    assert!(
        fy > 0.0,
        "la limite d'élasticité fy doit être strictement positive (MPa)"
    );
    (235.0 / fy).sqrt()
}

/// Élancement `c/t = flange_outstand / flange_thickness` d'une semelle comprimée
/// en console (sans dimension), avec `flange_outstand` la largeur droite `c` en
/// console (mm) et `flange_thickness` l'épaisseur `t` de la semelle (mm).
///
/// Panique si `flange_outstand < 0` ou si `flange_thickness <= 0`.
pub fn steelclass_flange_slenderness(flange_outstand: f64, flange_thickness: f64) -> f64 {
    assert!(
        flange_outstand >= 0.0,
        "la largeur en console c de la semelle doit être ≥ 0 (mm)"
    );
    assert!(
        flange_thickness > 0.0,
        "l'épaisseur t de la semelle doit être strictement positive (mm)"
    );
    flange_outstand / flange_thickness
}

/// Élancement `c/t = web_clear_depth / web_thickness` d'une âme comprimée (sans
/// dimension), avec `web_clear_depth` la hauteur droite `c` de l'âme (mm) et
/// `web_thickness` l'épaisseur `t` de l'âme (mm).
///
/// Panique si `web_clear_depth < 0` ou si `web_thickness <= 0`.
pub fn steelclass_web_slenderness(web_clear_depth: f64, web_thickness: f64) -> f64 {
    assert!(
        web_clear_depth >= 0.0,
        "la hauteur droite c de l'âme doit être ≥ 0 (mm)"
    );
    assert!(
        web_thickness > 0.0,
        "l'épaisseur t de l'âme doit être strictement positive (mm)"
    );
    web_clear_depth / web_thickness
}

/// Classe (`1..=4`) d'une **semelle comprimée en console** d'après l'élancement
/// `c/t` et le coefficient `ε` : `c/t ≤ 9·ε` → 1, `≤ 10·ε` → 2, `≤ 14·ε` → 3,
/// sinon 4 (voilement local).
///
/// Panique si `flange_slenderness < 0` ou si `epsilon <= 0`.
pub fn steelclass_flange_class(flange_slenderness: f64, epsilon: f64) -> u8 {
    assert!(
        flange_slenderness >= 0.0,
        "l'élancement c/t de la semelle doit être ≥ 0"
    );
    assert!(
        epsilon > 0.0,
        "le coefficient ε doit être strictement positif"
    );
    if flange_slenderness <= 9.0 * epsilon
    {
        1
    }
    else if flange_slenderness <= 10.0 * epsilon
    {
        2
    }
    else if flange_slenderness <= 14.0 * epsilon
    {
        3
    }
    else
    {
        4
    }
}

/// Classe (`1..=4`) d'une **âme en compression uniforme** d'après l'élancement
/// `c/t` et le coefficient `ε` : `c/t ≤ 33·ε` → 1, `≤ 38·ε` → 2, `≤ 42·ε` → 3,
/// sinon 4 (voilement local).
///
/// Panique si `web_slenderness < 0` ou si `epsilon <= 0`.
pub fn steelclass_web_class_compression(web_slenderness: f64, epsilon: f64) -> u8 {
    assert!(
        web_slenderness >= 0.0,
        "l'élancement c/t de l'âme doit être ≥ 0"
    );
    assert!(
        epsilon > 0.0,
        "le coefficient ε doit être strictement positif"
    );
    if web_slenderness <= 33.0 * epsilon
    {
        1
    }
    else if web_slenderness <= 38.0 * epsilon
    {
        2
    }
    else if web_slenderness <= 42.0 * epsilon
    {
        3
    }
    else
    {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn epsilon_reference_and_identity() {
        // À fy = 235 MPa (nuance de référence S235) : ε = √(235/235) = 1 exactement.
        assert_relative_eq!(steelclass_epsilon(235.0), 1.0, epsilon = 1e-12);
        // Identité : par définition ε² · fy = 235, quelle que soit fy fournie.
        let fy = 355.0_f64;
        let eps = steelclass_epsilon(fy);
        assert_relative_eq!(eps.powi(2) * fy, 235.0, epsilon = 1e-9);
    }

    #[test]
    fn epsilon_s355_value() {
        // Cas chiffré S355 : ε = √(235/355) = √0,6619718… = 0,813616… (arrondi
        // usuel du tableau : 0,81). Recalcul : 235/355 = 0,66197183 ; racine ≈
        // 0,8136165.
        assert_relative_eq!(steelclass_epsilon(355.0), 0.813_616_5, epsilon = 1e-6);
    }

    #[test]
    fn slenderness_ratios_are_plain_quotients() {
        // Semelle : c/t = 90 / 10 = 9 ; âme : c/t = 400 / 8 = 50.
        assert_relative_eq!(
            steelclass_flange_slenderness(90.0, 10.0),
            9.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            steelclass_web_slenderness(400.0, 8.0),
            50.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn flange_class_thresholds_at_unit_epsilon() {
        // Avec ε = 1 (S235), bornes 9 / 10 / 14.
        assert_eq!(steelclass_flange_class(9.0, 1.0), 1); // = 9·ε → classe 1
        assert_eq!(steelclass_flange_class(9.5, 1.0), 2); // ]9 ; 10] → classe 2
        assert_eq!(steelclass_flange_class(12.0, 1.0), 3); // ]10 ; 14] → classe 3
        assert_eq!(steelclass_flange_class(15.0, 1.0), 4); // > 14 → classe 4
    }

    #[test]
    fn web_class_thresholds_and_epsilon_scaling() {
        // Avec ε = 1 (S235), bornes 33 / 38 / 42.
        assert_eq!(steelclass_web_class_compression(33.0, 1.0), 1);
        assert_eq!(steelclass_web_class_compression(35.0, 1.0), 2);
        assert_eq!(steelclass_web_class_compression(40.0, 1.0), 3);
        assert_eq!(steelclass_web_class_compression(50.0, 1.0), 4);
        // Effet de ε : une âme d'élancement 40 est classe 3 en S235 (ε = 1, borne
        // 42) mais classe 4 en S355 (ε ≈ 0,8136, borne 42·ε ≈ 34,17).
        let eps355 = steelclass_epsilon(355.0);
        assert_eq!(steelclass_web_class_compression(40.0, eps355), 4);
    }

    #[test]
    #[should_panic(expected = "la limite d'élasticité fy doit être strictement positive")]
    fn epsilon_rejects_non_positive_fy() {
        // fy nulle : rapport indéfini, entrée refusée.
        steelclass_epsilon(0.0);
    }
}
