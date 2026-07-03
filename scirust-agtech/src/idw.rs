//! Interpolation spatiale par pondération inverse à la distance (IDW) —
//! avec un filtre d'aberrants explicite ([`crate::outlier_filter`]),
//! l'autre moitié du pipeline qui élimine la dépendance aux valeurs par
//! défaut propres à chaque logiciel SIG documentée comme source de
//! divergence entre cartes de rendement (voir [`crate::outlier_filter`]
//! pour la référence complète — Walczykova et al. 2018 y trouvent le
//! choix du filtre plus déterminant que celui de l'interpolation, mais
//! les deux doivent être explicites pour qu'un résultat soit
//! reproductible).
//!
//! `ẑ(x) = Σ wᵢ·zᵢ / Σ wᵢ`, `wᵢ = 1/dᵢᵖ` (`p` = puissance, typiquement 2),
//! calculé sur les `k_neighbors` points connus les plus proches du point
//! interrogé (IDW local, la variante la plus utilisée en pratique).

use crate::YieldPoint;

fn distance(query: (f64, f64), p: &YieldPoint) -> f64 {
    let (dx, dy) = (query.0 - p.x, query.1 - p.y);
    (dx * dx + dy * dy).sqrt()
}

/// Interpole la valeur de rendement au point `query` par IDW sur les
/// `k_neighbors` points connus les plus proches, avec départage
/// déterministe des ex æquo de distance par index croissant. Si `query`
/// coïncide avec un point connu (distance négligeable), renvoie sa
/// valeur directement plutôt que de diviser par un poids infini.
/// `None` si `known` est vide ou `k_neighbors == 0`.
pub fn idw_interpolate(
    known: &[YieldPoint],
    query: (f64, f64),
    power: f64,
    k_neighbors: usize,
) -> Option<f64> {
    if known.is_empty() || k_neighbors == 0
    {
        return None;
    }
    let mut dists: Vec<(f64, usize)> = known
        .iter()
        .enumerate()
        .map(|(i, p)| (distance(query, p), i))
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(&(d0, i0)) = dists.first()
    {
        if d0 < 1e-12
        {
            return Some(known[i0].yield_value);
        }
    }

    let k = k_neighbors.min(dists.len());
    let mut weight_sum = 0.0;
    let mut weighted_value_sum = 0.0;
    for &(d, i) in &dists[..k]
    {
        let w = 1.0 / d.powf(power);
        weight_sum += w;
        weighted_value_sum += w * known[i].yield_value;
    }
    Some(weighted_value_sum / weight_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn pt(x: f64, y: f64, yield_value: f64) -> YieldPoint {
        YieldPoint { x, y, yield_value }
    }

    #[test]
    fn equidistant_equal_values_return_that_value() {
        let known = vec![pt(-1.0, 0.0, 10.0), pt(1.0, 0.0, 10.0)];
        let z = idw_interpolate(&known, (0.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn equidistant_unequal_values_average_evenly() {
        let known = vec![pt(-1.0, 0.0, 8.0), pt(1.0, 0.0, 12.0)];
        let z = idw_interpolate(&known, (0.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn closer_point_dominates_the_weighted_average() {
        // known at distance 1 (yield 10) and distance 3 (yield 20), power=2:
        // w1=1, w2=1/9 -> (10*1 + 20/9) / (1 + 1/9) = (110/9)/(10/9) = 11.0 exactly.
        let known = vec![pt(0.0, 0.0, 10.0), pt(4.0, 0.0, 20.0)];
        let z = idw_interpolate(&known, (1.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 11.0, epsilon = 1e-9);
    }

    #[test]
    fn query_at_a_known_point_returns_its_exact_value() {
        let known = vec![pt(0.0, 0.0, 7.0), pt(5.0, 5.0, 20.0)];
        let z = idw_interpolate(&known, (0.0, 0.0), 2.0, 2).unwrap();
        assert_relative_eq!(z, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn empty_known_set_returns_none() {
        assert!(idw_interpolate(&[], (0.0, 0.0), 2.0, 2).is_none());
    }

    #[test]
    fn zero_k_neighbors_returns_none() {
        let known = vec![pt(0.0, 0.0, 7.0)];
        assert!(idw_interpolate(&known, (1.0, 1.0), 2.0, 0).is_none());
    }
}
