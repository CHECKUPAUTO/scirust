//! **Poutre mixte acier-béton** — vérifications de section à l'**ELU en flexion
//! simple** (Eurocode 4, EN 1994-1-1) : coefficient d'équivalence acier-béton,
//! largeur efficace de dalle, aire transformée en section homogène acier, moment
//! résistant plastique d'une section mixte à **connexion complète** (axe neutre
//! plastique dans la dalle) et nombre de connecteurs de cisaillement requis.
//!
//! ```text
//! coefficient d'équivalence   n   = Es / Ec
//! largeur efficace (simplif.) beff = min(L/4, s)
//! aire transformée (→ acier)  At  = b·hc / n + As
//! effort plastique de l'acier Fa  = As · fy
//! hauteur de béton comprimé   x   = Fa / (0,85 · fcd · b)
//! bras de levier plastique    z   = ha/2 + hc − x/2
//! moment plastique            Mpl = Fa · z
//! nombre de connecteurs       nc  = Fc / PRd
//! ```
//!
//! `Es` module d'élasticité de l'acier (MPa), `Ec` module d'élasticité du béton
//! (MPa), `n` coefficient d'équivalence (sans dimension) ; `L` portée de la
//! poutre, `s` entraxe des poutres, `beff` largeur efficace de la dalle (même
//! unité de longueur que `L` et `s`) ; `b` largeur de béton considérée, `hc`
//! épaisseur de la dalle, `As` aire de la section d'acier, `At` aire transformée
//! (unité d'aire) ; `fy` limite d'élasticité de l'acier (MPa), `fcd` résistance
//! de calcul du béton en compression (MPa), `Fa` effort normal plastique de la
//! semelle d'acier (N), `x` hauteur du bloc de béton comprimé, `ha` hauteur de la
//! section d'acier, `z` bras de levier, `Mpl` moment résistant plastique (N·mm) ;
//! `Fc` effort de compression à transmettre à l'interface (N), `PRd` résistance
//! de calcul d'un connecteur (N), `nc` nombre de connecteurs (sans dimension).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; les moments s'expriment donc en **N·mm** (1 kN·m = 10⁶
//! N·mm), les aires en **mm²**. `comp_modular_ratio` (rapport de modules) et
//! `comp_effective_width` (géométrie) sont **sans dimension / homogènes** : elles
//! acceptent n'importe quelle unité de longueur pourvu qu'elle soit cohérente
//! entre leurs arguments. Types `f64`.
//!
//! **Limite honnête** : section **mixte à connexion complète** avec **axe neutre
//! plastique situé dans la dalle** (`x ≤ hc`) — hypothèse **vérifiée par
//! l'appelant** et contrôlée par un `assert!` ; le cas de l'axe neutre dans la
//! semelle ou l'âme d'acier n'est **pas** traité. La largeur efficace `beff` est
//! l'**estimation simplifiée** `min(L/4, s)` : la formule complète de l'EN 1994-1-1
//! §5.4.1.2 (distances entre points de moment nul, largeurs `be,i`) n'est pas
//! reproduite. Seule la **flexion simple à l'ELU** est couverte : ni cisaillement
//! d'âme, ni interaction M-V, ni déversement, ni vérifications à l'ELS (flèche,
//! fissuration, fluage/retrait du béton via `n` à long terme). Toutes les
//! **résistances caractéristiques et de calcul** (`fy`, `fcd`) et **tous les
//! modules** (`Es`, `Ec`) ainsi que la **résistance des connecteurs** (`PRd`)
//! sont **fournis par l'appelant** d'après l'**Eurocode 4 (EN 1994-1-1)**,
//! l'**Eurocode 2** et leurs **Annexes Nationales** — aucune valeur « par défaut »
//! n'est inventée.

/// Coefficient d'équivalence acier-béton `n = Es / Ec` (sans dimension), rapport
/// des modules d'élasticité utilisé pour homogénéiser la section (EN 1994-1-1).
///
/// `steel_modulus` = `Es` module de l'acier (MPa), `concrete_modulus` = `Ec`
/// module sécant du béton (MPa) fourni par l'Eurocode 2 ; renvoie `n` (sans
/// dimension). Pour les actions de longue durée, l'appelant fournit un module
/// effectif du béton tenant compte du fluage.
///
/// Panique si `steel_modulus < 0` ou si `concrete_modulus <= 0` (division par
/// zéro / module non physique).
pub fn comp_modular_ratio(steel_modulus: f64, concrete_modulus: f64) -> f64 {
    assert!(
        steel_modulus >= 0.0,
        "le module de l'acier Es doit être ≥ 0"
    );
    assert!(
        concrete_modulus > 0.0,
        "le module du béton Ec doit être strictement positif"
    );
    steel_modulus / concrete_modulus
}

/// Largeur efficace simplifiée de la dalle `beff = min(L/4, s)` (même unité de
/// longueur que les arguments), estimation courante de l'EN 1994-1-1 §5.4.1.2
/// (plafonnée au quart de la portée et à l'entraxe des poutres).
///
/// `span` = `L` portée de la poutre, `beam_spacing` = `s` entraxe entre poutres
/// (même unité) ; renvoie la largeur efficace `beff`. Estimation **simplifiée**
/// (voir la limite honnête du module).
///
/// Panique si `span < 0` ou si `beam_spacing < 0`.
pub fn comp_effective_width(span: f64, beam_spacing: f64) -> f64 {
    assert!(span >= 0.0, "la portée L doit être ≥ 0");
    assert!(beam_spacing >= 0.0, "l'entraxe des poutres s doit être ≥ 0");
    (span / 4.0).min(beam_spacing)
}

/// Aire transformée en section homogène acier `At = b·hc / n + As` (unité
/// d'aire), obtenue en divisant l'aire de béton par le coefficient d'équivalence
/// `n` puis en ajoutant l'aire d'acier (EN 1994-1-1, section homogénéisée).
///
/// `concrete_width` = `b`, `concrete_thickness` = `hc` (mêmes unités de
/// longueur), `modular_ratio` = `n` (sans dimension, `n = Es/Ec`),
/// `steel_area` = `As` (unité d'aire cohérente avec `b·hc`) ; renvoie `At`.
///
/// Panique si `concrete_width < 0`, `concrete_thickness < 0`, `steel_area < 0`
/// ou si `modular_ratio <= 0` (division par zéro / coefficient non physique).
pub fn comp_transformed_area(
    concrete_width: f64,
    concrete_thickness: f64,
    modular_ratio: f64,
    steel_area: f64,
) -> f64 {
    assert!(concrete_width >= 0.0, "la largeur de béton b doit être ≥ 0");
    assert!(
        concrete_thickness >= 0.0,
        "l'épaisseur de béton hc doit être ≥ 0"
    );
    assert!(
        modular_ratio > 0.0,
        "le coefficient d'équivalence n doit être strictement positif"
    );
    assert!(steel_area >= 0.0, "l'aire d'acier As doit être ≥ 0");
    concrete_width * concrete_thickness / modular_ratio + steel_area
}

/// Moment résistant plastique d'une section mixte à **connexion complète** avec
/// **axe neutre plastique dans la dalle** (EN 1994-1-1 §6.2.1.2) :
///
/// ```text
/// Fa  = As · fy                       (effort normal plastique de l'acier)
/// x   = Fa / (0,85 · fcd · b)         (hauteur de béton comprimé)
/// z   = ha/2 + hc − x/2               (bras de levier)
/// Mpl = Fa · z
/// ```
///
/// **Hypothèses** : toute la section d'acier est plastifiée en **traction**
/// (résultante `Fa` au **centre de gravité** de l'acier, supposé à mi-hauteur
/// `ha/2`), le béton comprimé travaille selon le **diagramme rectangulaire**
/// `0,85·fcd` sur une hauteur `x` depuis la fibre supérieure de la dalle, et
/// l'axe neutre plastique tombe **dans la dalle** (`x ≤ hc`). Le bloc de béton
/// comprimé agit à `x/2` sous la fibre supérieure ; la dalle repose sur la
/// section d'acier (sommet de l'acier = sous-face de la dalle), d'où le bras de
/// levier `z = ha/2 + hc − x/2`.
///
/// `steel_area` = `As` (mm²), `yield_strength` = `fy` (MPa), `concrete_width` =
/// `b` largeur efficace de béton (mm), `concrete_thickness` = `hc` (mm),
/// `concrete_strength` = `fcd` résistance de calcul du béton (MPa),
/// `steel_depth` = `ha` hauteur de la section d'acier (mm) ; renvoie `Mpl`
/// (N·mm).
///
/// Panique si `steel_area < 0`, `yield_strength < 0`, `concrete_thickness < 0`,
/// `steel_depth < 0`, si `concrete_width <= 0` ou `concrete_strength <= 0`
/// (division par zéro), ou si l'axe neutre plastique sort de la dalle
/// (`x > concrete_thickness`, hypothèse violée).
pub fn comp_plastic_moment_full(
    steel_area: f64,
    yield_strength: f64,
    concrete_width: f64,
    concrete_thickness: f64,
    concrete_strength: f64,
    steel_depth: f64,
) -> f64 {
    assert!(steel_area >= 0.0, "l'aire d'acier As doit être ≥ 0");
    assert!(
        yield_strength >= 0.0,
        "la limite d'élasticité fy doit être ≥ 0"
    );
    assert!(
        concrete_width > 0.0,
        "la largeur de béton b doit être strictement positive"
    );
    assert!(
        concrete_thickness >= 0.0,
        "l'épaisseur de dalle hc doit être ≥ 0"
    );
    assert!(
        concrete_strength > 0.0,
        "la résistance de calcul du béton fcd doit être strictement positive"
    );
    assert!(
        steel_depth >= 0.0,
        "la hauteur de la section d'acier ha doit être ≥ 0"
    );
    let axial_force = steel_area * yield_strength;
    let compression_depth = axial_force / (0.85 * concrete_strength * concrete_width);
    assert!(
        compression_depth <= concrete_thickness,
        "l'axe neutre plastique doit tomber dans la dalle (x ≤ épaisseur de béton)"
    );
    let lever_arm = steel_depth / 2.0 + concrete_thickness - compression_depth / 2.0;
    axial_force * lever_arm
}

/// Nombre de connecteurs de cisaillement pour une **connexion complète**
/// `nc = Fc / PRd` (sans dimension), EN 1994-1-1 §6.6 : effort de compression à
/// transmettre à l'interface acier-béton divisé par la résistance d'un
/// connecteur.
///
/// `compression_force` = `Fc` effort à transmettre sur la demi-portée
/// (N), `connector_resistance` = `PRd` résistance de calcul d'un connecteur (N)
/// fournie par l'Eurocode 4 ; renvoie `nc` **réel** (l'appelant l'**arrondit à
/// l'entier supérieur** et le répartit selon les règles de l'Eurocode).
///
/// Panique si `compression_force < 0` ou si `connector_resistance <= 0`
/// (division par zéro).
pub fn comp_shear_connector_number(compression_force: f64, connector_resistance: f64) -> f64 {
    assert!(
        compression_force >= 0.0,
        "l'effort de compression Fc doit être ≥ 0"
    );
    assert!(
        connector_resistance > 0.0,
        "la résistance d'un connecteur PRd doit être strictement positive"
    );
    compression_force / connector_resistance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn modular_ratio_definition_and_proportionality() {
        // n = Es/Ec. Es = 210 000 MPa, Ec = 35 000 MPa → n = 6.
        let n = comp_modular_ratio(210_000.0, 35_000.0);
        assert_relative_eq!(n, 6.0, epsilon = 1e-9);
        // Réciprocité : diviser Ec par n redonne Es.
        assert_relative_eq!(35_000.0 * n, 210_000.0, epsilon = 1e-6);
        // Halver Ec double n.
        let n2 = comp_modular_ratio(210_000.0, 17_500.0);
        assert_relative_eq!(n2 / n, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn effective_width_both_branches() {
        // beff = min(L/4, s). L = 8 m → L/4 = 2 m ; s = 3 m → L/4 gouverne.
        let b1 = comp_effective_width(8.0, 3.0);
        assert_relative_eq!(b1, 2.0, epsilon = 1e-12);
        // s = 1,5 m < L/4 = 2 m → l'entraxe gouverne.
        let b2 = comp_effective_width(8.0, 1.5);
        assert_relative_eq!(b2, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn transformed_area_worked_case() {
        // At = b·hc/n + As. b = 2000 mm, hc = 150 mm, n = 6, As = 10 000 mm².
        //   b·hc/n = 2000·150/6 = 300 000/6 = 50 000 mm²
        //   At     = 50 000 + 10 000 = 60 000 mm²
        let at = comp_transformed_area(2000.0, 150.0, 6.0, 10_000.0);
        assert_relative_eq!(at, 60_000.0, epsilon = 1e-9);
        // Limite n → ∞ (acier infiniment plus raide) : la dalle disparaît, At → As.
        let at_stiff = comp_transformed_area(2000.0, 150.0, 1.0e12, 10_000.0);
        assert_relative_eq!(at_stiff, 10_000.0, epsilon = 1e-3);
    }

    #[test]
    fn plastic_moment_worked_case() {
        // As = 10 000 mm², fy = 300 MPa, b = 2000 mm, hc = 150 mm,
        // fcd = 20 MPa, ha = 400 mm.
        //   Fa  = 10 000 · 300 = 3 000 000 N
        //   x   = 3e6 / (0,85·20·2000) = 3e6 / 34 000 = 88,2352941 mm  (< hc → OK)
        //   z   = 400/2 + 150 − 88,2352941/2 = 200 + 150 − 44,1176471 = 305,8823529 mm
        //   Mpl = 3e6 · 305,8823529 = 917 647 058,8 N·mm
        let mpl = comp_plastic_moment_full(10_000.0, 300.0, 2000.0, 150.0, 20.0, 400.0);
        assert_relative_eq!(mpl, 917_647_058.823_529, epsilon = 1e-3);
    }

    #[test]
    fn shear_connector_number_worked_and_proportional() {
        // nc = Fc/PRd. Fc = 3 000 000 N, PRd = 100 000 N → nc = 30.
        let nc = comp_shear_connector_number(3_000_000.0, 100_000.0);
        assert_relative_eq!(nc, 30.0, epsilon = 1e-9);
        // Doubler la résistance d'un connecteur halve le nombre requis.
        let nc2 = comp_shear_connector_number(3_000_000.0, 200_000.0);
        assert_relative_eq!(nc2 / nc, 0.5, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "l'axe neutre plastique doit tomber dans la dalle")]
    fn plastic_moment_rejects_neutral_axis_outside_slab() {
        // Fa = 3e6 N, x = 88,2 mm mais dalle mince hc = 50 mm : x > hc,
        // l'axe neutre plastique sortirait de la dalle (hypothèse violée).
        comp_plastic_moment_full(10_000.0, 300.0, 2000.0, 50.0, 20.0, 400.0);
    }
}
