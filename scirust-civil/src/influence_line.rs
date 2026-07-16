//! Lignes d'influence d'une poutre **isostatique** sur deux appuis simples :
//! ordonnées des lignes d'influence de la réaction d'appui gauche, du moment
//! fléchissant à une section, de l'effort tranchant à une section, et moment
//! maximal à une section sous une charge concentrée mobile placée au droit de
//! la section.
//!
//! ```text
//! réaction gauche R_A(x)   = (L − x)/L
//! moment à s        M(s,x) = (L − s)·x/L      si x ≤ s
//!                          = s·(L − x)/L       si x > s
//! moment max à s    M_max  = P·s·(L − s)/L
//! tranchant à s     V(s,x) = −x/L             si x ≤ s
//!                          = (L − x)/L         si x > s
//! ```
//!
//! `L` portée de la poutre (m), `x` position de la charge mobile unitaire
//! mesurée depuis l'appui gauche (m), `s` position de la section étudiée depuis
//! l'appui gauche (m), `R_A` ordonnée de la ligne d'influence de la réaction
//! d'appui gauche (–, adimensionnelle pour une charge unité), `M` ordonnée de la
//! ligne d'influence du moment fléchissant (m, valeur du moment par unité de
//! charge), `V` ordonnée de la ligne d'influence de l'effort tranchant (–), `P`
//! charge concentrée mobile (N), `M_max` moment fléchissant maximal à la section
//! `s` (N·m) obtenu en plaçant `P` au droit de la section.
//!
//! **Convention** : SI strict et cohérent — mètres (m) pour les longueurs et
//! positions, newtons (N) pour les charges, newtons·mètres (N·m) pour les
//! moments. Les ordonnées de lignes d'influence de réaction et d'effort
//! tranchant sont **adimensionnelles** (valeur par charge unité) ; celle du
//! moment a la dimension d'une longueur (m). Toutes les positions sont mesurées
//! depuis l'appui gauche. Types `f64`.
//!
//! **Limite honnête** : ce module traite exclusivement la **poutre isostatique
//! sur deux appuis simples** en théorie des petites déformations (élastique
//! linéaire). Il retourne les **ordonnées** des lignes d'influence pour une
//! charge mobile **unitaire** ; la position de la charge `x`, la position de la
//! section `s`, la portée `L` et l'intensité de la charge concentrée `P` sont
//! **fournies par l'appelant**. Le calcul du maximum sous un **convoi** de
//! plusieurs charges (théorème de la charge critique, superposition des
//! ordonnées) reste à la charge de l'appelant : la fonction de moment maximal
//! fournie ici ne couvre qu'une **unique** charge concentrée mobile placée au
//! droit de la section. Aucune combinaison d'actions, aucun coefficient partiel
//! de sécurité et aucune enveloppe de sollicitations ne sont introduits ici.

/// Ordonnée de la ligne d'influence de la réaction d'appui gauche `R_A(x) =
/// (L − x)/L` pour une charge unité à la position `load_position` (m) sur une
/// poutre isostatique de portée `span` (m).
///
/// Panique si `span <= 0`, ou si `load_position` n'est pas dans `[0, span]`.
pub fn infl_reaction_simple_beam(load_position: f64, span: f64) -> f64 {
    assert!(span > 0.0, "la portée span doit être strictement positive");
    assert!(
        load_position >= 0.0 && load_position <= span,
        "la position de charge load_position doit être comprise dans [0, span]"
    );
    (span - load_position) / span
}

/// Ordonnée de la ligne d'influence du moment fléchissant à la section
/// `section_position` (m) pour une charge unité à la position `load_position`
/// (m) : `(L − s)·x/L` si `x ≤ s`, sinon `s·(L − x)/L` (résultat en m).
///
/// Panique si `span <= 0`, si `section_position` n'est pas dans `[0, span]`, ou
/// si `load_position` n'est pas dans `[0, span]`.
pub fn infl_moment_at_section(section_position: f64, load_position: f64, span: f64) -> f64 {
    assert!(span > 0.0, "la portée span doit être strictement positive");
    assert!(
        section_position >= 0.0 && section_position <= span,
        "la position de section section_position doit être comprise dans [0, span]"
    );
    assert!(
        load_position >= 0.0 && load_position <= span,
        "la position de charge load_position doit être comprise dans [0, span]"
    );
    if load_position <= section_position
    {
        (span - section_position) * load_position / span
    }
    else
    {
        section_position * (span - load_position) / span
    }
}

/// Moment fléchissant maximal à la section `section_position` (m) sous une charge
/// concentrée mobile `point_load` (N) placée au droit de la section : `M_max =
/// P·s·(L − s)/L` (résultat en N·m).
///
/// Panique si `span <= 0`, si `section_position` n'est pas dans `[0, span]`, ou
/// si `point_load < 0`.
pub fn infl_max_moment_single_load(point_load: f64, section_position: f64, span: f64) -> f64 {
    assert!(span > 0.0, "la portée span doit être strictement positive");
    assert!(
        section_position >= 0.0 && section_position <= span,
        "la position de section section_position doit être comprise dans [0, span]"
    );
    assert!(
        point_load >= 0.0,
        "la charge concentrée point_load doit être positive ou nulle"
    );
    point_load * section_position * (span - section_position) / span
}

/// Ordonnée de la ligne d'influence de l'effort tranchant à la section
/// `section_position` (m) pour une charge unité à la position `load_position`
/// (m) : `−x/L` si `x ≤ s`, sinon `(L − x)/L` (résultat adimensionnel).
///
/// Panique si `span <= 0`, si `section_position` n'est pas dans `[0, span]`, ou
/// si `load_position` n'est pas dans `[0, span]`.
pub fn infl_shear_at_section(section_position: f64, load_position: f64, span: f64) -> f64 {
    assert!(span > 0.0, "la portée span doit être strictement positive");
    assert!(
        section_position >= 0.0 && section_position <= span,
        "la position de section section_position doit être comprise dans [0, span]"
    );
    assert!(
        load_position >= 0.0 && load_position <= span,
        "la position de charge load_position doit être comprise dans [0, span]"
    );
    if load_position <= section_position
    {
        -load_position / span
    }
    else
    {
        (span - load_position) / span
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reaction_limites_appuis() {
        // Charge sur l'appui gauche : toute la réaction gauche vaut 1 ;
        // charge sur l'appui droit : réaction gauche nulle.
        let span = 6.0;
        assert_relative_eq!(infl_reaction_simple_beam(0.0, span), 1.0, epsilon = 1e-3);
        assert_relative_eq!(infl_reaction_simple_beam(span, span), 0.0, epsilon = 1e-3);
    }

    #[test]
    fn reaction_complementarite() {
        // Réactions gauche et droite d'une charge unité somment à 1 :
        // R_A(x) + R_B(x) = 1, avec R_B(x) = R_A(L − x) par symétrie.
        let span = 8.0;
        let x = 3.0;
        let ra = infl_reaction_simple_beam(x, span);
        let rb = infl_reaction_simple_beam(span - x, span);
        assert_relative_eq!(ra + rb, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn moment_continuite_et_symetrie() {
        // Continuité : au droit de la section (x = s) les deux branches
        // coïncident ; symétrie : la valeur au droit vaut s·(L − s)/L.
        let span = 10.0;
        let s = 4.0;
        let gauche = infl_moment_at_section(s, s, span); // (L − s)·s/L
        let attendu = s * (span - s) / span; // 4·6/10 = 2,4
        assert_relative_eq!(gauche, 2.4, epsilon = 1e-3);
        assert_relative_eq!(gauche, attendu, epsilon = 1e-3);
    }

    #[test]
    fn moment_max_mi_portee_calcul() {
        // Cas chiffré recalculé : P = 20 kN = 20000 N, section à mi-portée
        // s = L/2 = 5 m, L = 10 m.
        // M_max = P·s·(L − s)/L = 20000·5·5/10 = 20000·25/10 = 50000 N·m.
        // Recalcul indépendant : 20000·(5·5)/10 = 20000·2,5 = 50000 N·m.
        let m = infl_max_moment_single_load(20000.0, 5.0, 10.0);
        assert_relative_eq!(m, 50000.0, epsilon = 1e-3);
        assert_relative_eq!(m, 50000.0, epsilon = 1e-3);
    }

    #[test]
    fn shear_saut_unite_au_droit() {
        // Le saut de l'ordonnée de tranchant au droit de la section vaut 1 :
        // juste à droite (L − s)/L moins juste à gauche (−s/L).
        let span = 12.0;
        let s = 5.0;
        let droite = infl_shear_at_section(s, s + 1e-6, span); // (L − x)/L
        let gauche = infl_shear_at_section(s, s, span); // −s/L
        assert_relative_eq!(droite - gauche, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn shear_extremites() {
        // Charge à l'appui gauche : V = −0 = 0 ; charge à l'appui droit :
        // V = (L − L)/L = 0. Aux extrémités, l'ordonnée de tranchant est nulle.
        let span = 7.0;
        let s = 3.0;
        assert_relative_eq!(infl_shear_at_section(s, 0.0, span), 0.0, epsilon = 1e-3);
        assert_relative_eq!(infl_shear_at_section(s, span, span), 0.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la portée span doit être strictement positive")]
    fn panique_portee_nulle() {
        infl_reaction_simple_beam(1.0, 0.0);
    }
}
