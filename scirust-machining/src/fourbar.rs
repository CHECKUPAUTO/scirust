//! Quadrilatère articulé (four-bar linkage) — critère de **Grashof** et
//! classification du mécanisme selon les longueurs des quatre barres.
//!
//! ```text
//! condition de Grashof   s + l ≤ p + q
//!   (s = plus courte, l = plus longue, p,q = intermédiaires)
//! si s + l < p + q : au moins une barre fait un tour complet
//!   • plus courte = bâti      → double-manivelle
//!   • plus courte = coupleur  → double-balancier (Grashof)
//!   • plus courte = côté (adjacente au bâti) → manivelle-balancier
//! si s + l = p + q : point de changement (positions de repliement)
//! si s + l > p + q : non-Grashof → triple-balancier (aucune rotation complète)
//! ```
//!
//! Les quatre arguments sont les longueurs des barres **dans l'ordre de la
//! boucle** : `ground` (bâti), `input` (entrée), `coupler` (coupleur), `output`
//! (sortie) ; le bâti et le coupleur sont opposés, l'entrée et la sortie sont
//! adjacentes au bâti.
//!
//! **Convention** : longueurs cohérentes strictement positives. **Limite
//! honnête** : critère de mobilité **géométrique** (Grashof) ; ne décrit ni les
//! positions singulières atteintes en service, ni l'angle de transmission, ni le
//! choix du moteur — seulement le type de rotation possible.

/// Type cinématique d'un quadrilatère articulé.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourBarType {
    /// Une barre latérale fait un tour complet, l'opposée oscille.
    CrankRocker,
    /// Les deux barres reliées au bâti tournent complètement.
    DoubleCrank,
    /// Grashof avec coupleur le plus court : aucune barre du bâti ne tourne.
    DoubleRocker,
    /// `s + l = p + q` : positions de changement (repliement).
    ChangePoint,
    /// Non-Grashof : les trois barres mobiles oscillent seulement.
    TripleRocker,
}

fn check_closure(links: [f64; 4]) {
    let total: f64 = links.iter().sum();
    for &li in &links
    {
        assert!(
            li > 0.0,
            "chaque barre doit avoir une longueur strictement positive"
        );
        assert!(
            li < total - li,
            "boucle impossible : une barre est plus longue que la somme des trois autres"
        );
    }
}

/// Vrai si la condition de Grashof `s + l < p + q` est satisfaite (au moins une
/// barre fait un tour complet). L'égalité (point de changement) renvoie `false`.
pub fn is_grashof(ground: f64, input: f64, coupler: f64, output: f64) -> bool {
    let links = [ground, input, coupler, output];
    check_closure(links);
    let mut sorted = links;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (s, p, q, l) = (sorted[0], sorted[1], sorted[2], sorted[3]);
    s + l < p + q - 1e-9 * l
}

/// Classe le mécanisme selon Grashof et la position de la barre la plus courte.
pub fn classify(ground: f64, input: f64, coupler: f64, output: f64) -> FourBarType {
    let links = [ground, input, coupler, output];
    check_closure(links);
    let mut sorted = links;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (s, p, q, l) = (sorted[0], sorted[1], sorted[2], sorted[3]);
    let sum_sl = s + l;
    let sum_pq = p + q;
    let tol = 1e-9 * l;

    if sum_sl > sum_pq + tol
    {
        return FourBarType::TripleRocker;
    }
    if (sum_sl - sum_pq).abs() <= tol
    {
        return FourBarType::ChangePoint;
    }
    // Grashof : la position de la barre la plus courte fixe le type.
    let shortest = links
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    match shortest
    {
        0 => FourBarType::DoubleCrank,  // bâti
        2 => FourBarType::DoubleRocker, // coupleur
        _ => FourBarType::CrankRocker,  // entrée ou sortie (côté)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crank_rocker_when_input_is_shortest() {
        // Entrée courte (2), bâti 4, coupleur 5, sortie 4 : s+l=2+5=7 < p+q=4+4=8.
        assert!(is_grashof(4.0, 2.0, 5.0, 4.0));
        assert_eq!(classify(4.0, 2.0, 5.0, 4.0), FourBarType::CrankRocker);
    }

    #[test]
    fn double_crank_when_ground_is_shortest() {
        // Bâti le plus court (2) : double-manivelle.
        assert!(is_grashof(2.0, 4.0, 5.0, 4.0));
        assert_eq!(classify(2.0, 4.0, 5.0, 4.0), FourBarType::DoubleCrank);
    }

    #[test]
    fn double_rocker_when_coupler_is_shortest() {
        // Coupleur le plus court (2) : double-balancier Grashof.
        assert_eq!(classify(4.0, 4.0, 2.0, 5.0), FourBarType::DoubleRocker);
    }

    #[test]
    fn triple_rocker_when_non_grashof() {
        // s+l > p+q : 1+5=6 > 3+3=... choisissons 1,3,5,3 → s+l=1+5=6, p+q=3+3=6 égal.
        // Prenons 2,3,6,3 : s=2,l=6,p=3,q=3 → 8 > 6 → non-Grashof.
        assert!(!is_grashof(2.0, 3.0, 6.0, 3.0));
        assert_eq!(classify(2.0, 3.0, 6.0, 3.0), FourBarType::TripleRocker);
    }

    #[test]
    fn change_point_at_equality() {
        // s+l = p+q : parallélogramme 3,5,3,5 → s=3,l=5,p=3,q=5 → 8=8.
        assert!(!is_grashof(3.0, 5.0, 3.0, 5.0));
        assert_eq!(classify(3.0, 5.0, 3.0, 5.0), FourBarType::ChangePoint);
    }

    #[test]
    #[should_panic(expected = "boucle impossible")]
    fn open_loop_panics() {
        // Une barre (10) plus longue que la somme des autres (1+1+1=3).
        classify(1.0, 1.0, 1.0, 10.0);
    }
}
