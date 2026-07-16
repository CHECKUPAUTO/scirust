//! **Béton armé — dalle (Eurocode 2, ELU)** : moment isostatique d'une dalle
//! portant sur un sens (appuis simples, par mètre de largeur), moment d'une
//! dalle sur quatre appuis à partir d'un coefficient de table, section minimale
//! d'armatures `As,min` et portée de calcul.
//!
//! ```text
//! moment sens porteur   M_1sens = q · L² / 8
//! moment sur 4 appuis   M       = α · q · Lx²
//! armatures minimales   As,min  = max(0,26 · fctm/fyk · b · d ; 0,0013 · b · d)
//! portée de calcul      Leff    = Ln + min(d ; a)
//! ```
//!
//! `q` = `distributed_load` charge répartie de calcul par mètre de largeur
//! (N/mm, c.-à-d. N par mm de portée et par mètre de largeur), `L` = `span`
//! portée d'une dalle sur un seul sens (mm), `M_1sens` moment isostatique par
//! mètre de largeur (N·mm) ; `α` = `coefficient` coefficient de moment de table
//! (sens porteur, table Marcus/Pigeaud) fourni selon le rapport des portées
//! `Ly/Lx` (sans dimension), `Lx` = `short_span` petite portée (mm), `M` moment
//! de la dalle sur quatre appuis (N·mm) ; `fctm` = `mean_tensile_strength`
//! résistance moyenne en traction du béton (MPa), `fyk` = `yield_strength`
//! limite d'élasticité caractéristique de l'acier (MPa), `b` = `width` largeur
//! considérée (mm, usuellement 1000 mm par mètre), `d` = `effective_depth`
//! hauteur utile (mm), `As,min` section minimale d'armatures tendues (mm²) ;
//! `Ln` = `clear_span` portée libre entre nus d'appuis (mm), `a` =
//! `support_width` largeur d'appui (mm), `Leff` portée de calcul (mm).
//!
//! **Convention** : unités **N, mm, MPa** (1 MPa = 1 N/mm²), cohérentes entre
//! elles (Eurocode) ; les moments ressortent donc en **N·mm** (1 kN·m = 10⁶
//! N·mm) et les aires en **mm²**. La charge répartie `q` est une **charge
//! linéique** (N/mm) prise par mètre de largeur de dalle, si bien que le moment
//! obtenu s'entend **par mètre de largeur**. Types `f64`.
//!
//! **Limite honnête** : dalle en **flexion** seule. Les **coefficients de
//! moment** `α` (sens porteur, table Marcus/Pigeaud selon `Ly/Lx`) et **toutes
//! les résistances** (`fctm`, `fyk`) sont **fournis par l'appelant** d'après
//! l'**Eurocode 2 (EN 1992-1-1)** et son **Annexe Nationale** — aucune valeur
//! « par défaut » n'est inventée. La **section minimale** `As,min` suit
//! l'Eurocode 2 (**maximum des deux termes**). Ce module **ne vérifie ni le
//! poinçonnement** (voir `rc_punching`) **ni la flèche détaillée**.

/// Moment isostatique d'une dalle portant sur un sens `M = q · L² / 8` (N·mm par
/// mètre de largeur), pour une dalle sur **appuis simples** soumise à une charge
/// répartie uniforme `q` (N/mm) sur une portée `L` (mm).
///
/// Panique si `distributed_load < 0` ou si `span < 0`.
pub fn rcslab_one_way_moment_udl(distributed_load: f64, span: f64) -> f64 {
    assert!(
        distributed_load >= 0.0,
        "la charge répartie q doit être ≥ 0"
    );
    assert!(span >= 0.0, "la portée L doit être ≥ 0");
    distributed_load * span * span / 8.0
}

/// Moment d'une dalle sur quatre appuis `M = α · q · Lx²` (N·mm par mètre de
/// largeur), avec `α` coefficient de moment de table (sens porteur, Marcus/
/// Pigeaud), `q` charge répartie (N/mm) et `Lx` petite portée (mm).
///
/// `coefficient` est fourni par l'appelant selon le rapport des portées `Ly/Lx`.
///
/// Panique si `coefficient < 0`, si `distributed_load < 0` ou si `short_span < 0`.
pub fn rcslab_two_way_moment(coefficient: f64, distributed_load: f64, short_span: f64) -> f64 {
    assert!(
        coefficient >= 0.0,
        "le coefficient de moment α doit être ≥ 0"
    );
    assert!(
        distributed_load >= 0.0,
        "la charge répartie q doit être ≥ 0"
    );
    assert!(short_span >= 0.0, "la petite portée Lx doit être ≥ 0");
    coefficient * distributed_load * short_span * short_span
}

/// Section minimale d'armatures tendues
/// `As,min = max(0,26 · fctm/fyk · b · d ; 0,0013 · b · d)` (mm²), selon
/// l'Eurocode 2 (maximum des deux termes), avec `fctm`, `fyk` en MPa et `b`, `d`
/// en mm.
///
/// Panique si `mean_tensile_strength < 0`, si `yield_strength <= 0` (division
/// par zéro), si `width <= 0` ou si `effective_depth <= 0`.
pub fn rcslab_minimum_reinforcement(
    mean_tensile_strength: f64,
    yield_strength: f64,
    width: f64,
    effective_depth: f64,
) -> f64 {
    assert!(
        mean_tensile_strength >= 0.0,
        "la résistance en traction fctm doit être ≥ 0"
    );
    assert!(
        yield_strength > 0.0,
        "la limite d'élasticité fyk doit être strictement positive"
    );
    assert!(width > 0.0, "la largeur b doit être strictement positive");
    assert!(
        effective_depth > 0.0,
        "la hauteur utile d doit être strictement positive"
    );
    (0.26 * mean_tensile_strength / yield_strength * width * effective_depth)
        .max(0.0013 * width * effective_depth)
}

/// Portée de calcul `Leff = Ln + min(d ; a)` (mm), où `Ln` est la portée libre
/// entre nus d'appuis, `d` la hauteur utile et `a` la largeur d'appui (tous en
/// mm).
///
/// Panique si `clear_span < 0`, si `effective_depth < 0` ou si
/// `support_width < 0`.
pub fn rcslab_effective_span(clear_span: f64, effective_depth: f64, support_width: f64) -> f64 {
    assert!(clear_span >= 0.0, "la portée libre Ln doit être ≥ 0");
    assert!(effective_depth >= 0.0, "la hauteur utile d doit être ≥ 0");
    assert!(support_width >= 0.0, "la largeur d'appui a doit être ≥ 0");
    clear_span + effective_depth.min(support_width)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn one_way_moment_clean_case_and_proportionality() {
        // Cas propre : q = 10 N/mm (10 kN/m par mètre de largeur), L = 5000 mm :
        //   M = 10 · 5000² / 8 = 10 · 25 000 000 / 8 = 250 000 000 / 8
        //     = 31 250 000 N·mm = 31,25 kN·m
        let m = rcslab_one_way_moment_udl(10.0, 5000.0);
        assert_relative_eq!(m, 31_250_000.0, epsilon = 1e-3);
        // Proportionnalité : à L fixée, doubler la charge double le moment.
        let m2 = rcslab_one_way_moment_udl(20.0, 5000.0);
        assert_relative_eq!(m2 / m, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn one_way_moment_scales_with_span_squared() {
        // Le moment varie comme le carré de la portée : ×2 sur L ⇒ ×4 sur M.
        let m1 = rcslab_one_way_moment_udl(8.0, 3000.0);
        let m2 = rcslab_one_way_moment_udl(8.0, 6000.0);
        assert_relative_eq!(m2 / m1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn two_way_moment_clean_case() {
        // Cas propre : α = 0,062, q = 10 N/mm, Lx = 4000 mm :
        //   M = 0,062 · 10 · 4000² = 0,062 · 10 · 16 000 000
        //     = 0,062 · 160 000 000 = 9 920 000 N·mm
        let m = rcslab_two_way_moment(0.062, 10.0, 4000.0);
        assert_relative_eq!(m, 9_920_000.0, epsilon = 1e-3);
        // Cohérence : le coefficient 1/8 = 0,125 restitue le cas « un sens ».
        let m_one_way = rcslab_two_way_moment(0.125, 10.0, 5000.0);
        assert_relative_eq!(
            m_one_way,
            rcslab_one_way_moment_udl(10.0, 5000.0),
            epsilon = 1e-6
        );
    }

    #[test]
    fn minimum_reinforcement_takes_governing_term() {
        // C25/30 (fctm = 2,6 MPa), B500 (fyk = 500), b = 1000 mm, d = 150 mm :
        //   terme1 = 0,26 · 2,6/500 · 1000 · 150 = 0,001352 · 150 000 = 202,8 mm²
        //   terme2 = 0,0013 · 1000 · 150 = 195,0 mm²
        //   As,min = max(202,8 ; 195,0) = 202,8 mm²
        let as_min = rcslab_minimum_reinforcement(2.6, 500.0, 1000.0, 150.0);
        assert_relative_eq!(as_min, 202.8, epsilon = 1e-3);
        // Faible fctm : le plancher 0,0013·b·d gouverne (terme2 > terme1).
        //   terme1 = 0,26 · 1,0/500 · 1000 · 150 = 0,00052 · 150 000 = 78,0 mm²
        //   terme2 = 195,0 mm² ⇒ As,min = 195,0 mm²
        let as_min_floor = rcslab_minimum_reinforcement(1.0, 500.0, 1000.0, 150.0);
        assert_relative_eq!(as_min_floor, 195.0, epsilon = 1e-3);
    }

    #[test]
    fn effective_span_takes_minimum_of_depth_and_support() {
        // Cas d < a : Leff = Ln + d = 4800 + 150 = 4950 mm.
        let leff_a = rcslab_effective_span(4800.0, 150.0, 200.0);
        assert_relative_eq!(leff_a, 4950.0, epsilon = 1e-9);
        // Cas a < d : Leff = Ln + a = 4800 + 120 = 4920 mm.
        let leff_b = rcslab_effective_span(4800.0, 150.0, 120.0);
        assert_relative_eq!(leff_b, 4920.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_one_way_slab_chain() {
        // Dalle sur un sens, appuis simples, C25/30, B500 :
        //   portée libre Ln = 4800 mm, d = 160 mm, appui a = 200 mm
        //   Leff = 4800 + min(160 ; 200) = 4800 + 160 = 4960 mm
        //   charge q = 12 N/mm (12 kN/m par mètre de largeur)
        //   M = 12 · 4960² / 8 = 12 · 24 601 600 / 8 = 295 219 200 / 8
        //     = 36 902 400 N·mm ≈ 36,90 kN·m
        //   As,min = max(0,26·2,6/500·1000·160 ; 0,0013·1000·160)
        //          = max(216,32 ; 208,0) = 216,32 mm²
        let leff = rcslab_effective_span(4800.0, 160.0, 200.0);
        assert_relative_eq!(leff, 4960.0, epsilon = 1e-9);
        let m = rcslab_one_way_moment_udl(12.0, leff);
        assert_relative_eq!(m, 36_902_400.0, epsilon = 1e-1);
        let as_min = rcslab_minimum_reinforcement(2.6, 500.0, 1000.0, 160.0);
        assert_relative_eq!(as_min, 216.32, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la limite d'élasticité fyk doit être strictement positive")]
    fn minimum_reinforcement_rejects_null_yield_strength() {
        // Division par zéro interdite : fyk = 0.
        rcslab_minimum_reinforcement(2.6, 0.0, 1000.0, 150.0);
    }
}
