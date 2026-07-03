//! Filtrage des points aberrants sur une carte de rendement — les deux
//! filtres les plus largement cités dans la littérature de nettoyage de
//! cartes de rendement (Sudduth & Drummond, "Yield Editor: Software for
//! Removing Errors from Crop Yield Maps", Agronomy Journal 99(6), 2007 —
//! l'outil de référence USDA-ARS).
//!
//! **Pourquoi ce module existe** : Walczykova et al. (2018) montrent que
//! les mêmes données de rendement brutes, traitées par QGIS, Agro-Map ou
//! Farm Works avec leurs filtres par défaut respectifs, produisent des
//! cartes visiblement différentes — le choix du filtre d'aberrants (et
//! de son seuil) affecte le coefficient de variation résultant de 32 à
//! 46%, plus que le choix de la méthode d'interpolation. Ce module ne
//! prétend pas trancher le désaccord sur le "bon" seuil — il rend le
//! filtre *explicite et auditable* : mêmes points + mêmes paramètres ⇒
//! même résultat, byte pour byte, sur toute plateforme, plutôt qu'un
//! réglage implicite propre à l'outil.

use crate::YieldPoint;

/// Indices des points conservés après filtre global : rejette tout point
/// dont le rendement s'écarte de la moyenne globale de plus de `k_std`
/// écarts-types (Sudduth & Drummond, filtres "globaux").
pub fn global_filter(points: &[YieldPoint], k_std: f64) -> Vec<usize> {
    let n = points.len();
    if n == 0
    {
        return Vec::new();
    }
    let mean = points.iter().map(|p| p.yield_value).sum::<f64>() / n as f64;
    let variance = points
        .iter()
        .map(|p| (p.yield_value - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let std = variance.sqrt();
    if std < 1e-12
    {
        return (0..n).collect(); // no variation: nothing to filter
    }
    (0..n)
        .filter(|&i| ((points[i].yield_value - mean) / std).abs() <= k_std)
        .collect()
}

fn distance(a: &YieldPoint, b: &YieldPoint) -> f64 {
    let (dx, dy) = (a.x - b.x, a.y - b.y);
    (dx * dx + dy * dy).sqrt()
}

/// Indices des points conservés après filtre local (voisinage) :
/// rejette tout point dont le rendement s'écarte de la moyenne de ses
/// `k_neighbors` plus proches voisins de plus de `k_std` écarts-types
/// *locales* (calculées sur ce même voisinage) — attrape les anomalies
/// localisées (par ex. une bande de relevage de tête de champ) qu'un
/// filtre global au seuil généreux laisse passer, car elles restent
/// dans la plage globale tout en tranchant avec leur environnement
/// immédiat (Sudduth & Drummond, filtres "spatiaux").
///
/// Recherche de voisins par force brute (`O(n²)`), avec départage
/// déterministe des ex æquo de distance par index croissant — adapté aux
/// tailles de jeux de données de parcelle habituelles, pas à un flux à
/// très grande échelle.
pub fn local_filter(points: &[YieldPoint], k_neighbors: usize, k_std: f64) -> Vec<usize> {
    let n = points.len();
    if n == 0 || k_neighbors == 0
    {
        return (0..n).collect();
    }
    let mut kept = Vec::with_capacity(n);
    for i in 0..n
    {
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (distance(&points[i], &points[j]), j))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = k_neighbors.min(dists.len());
        let neighbor_yields: Vec<f64> = dists[..k]
            .iter()
            .map(|&(_, j)| points[j].yield_value)
            .collect();
        let mean = neighbor_yields.iter().sum::<f64>() / k as f64;
        let variance = neighbor_yields
            .iter()
            .map(|y| (y - mean).powi(2))
            .sum::<f64>()
            / k as f64;
        let std = variance.sqrt();
        let keep = if std < 1e-12
        {
            (points[i].yield_value - mean).abs() < 1e-9
        }
        else
        {
            ((points[i].yield_value - mean) / std).abs() <= k_std
        };
        if keep
        {
            kept.push(i);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64, yield_value: f64) -> YieldPoint {
        YieldPoint { x, y, yield_value }
    }

    #[test]
    fn global_filter_rejects_a_gross_outlier() {
        let points = vec![
            pt(0.0, 0.0, 8.0),
            pt(1.0, 0.0, 9.0),
            pt(2.0, 0.0, 10.0),
            pt(3.0, 0.0, 9.5),
            pt(4.0, 0.0, 8.5),
            pt(5.0, 0.0, 50.0), // combine sensor glitch
        ];
        let kept = global_filter(&points, 2.0);
        assert!(!kept.contains(&5), "the 50.0 outlier should be rejected");
        assert_eq!(kept.len(), 5);
    }

    #[test]
    fn global_filter_keeps_everything_when_uniform() {
        let points = vec![pt(0.0, 0.0, 10.0); 5];
        assert_eq!(global_filter(&points, 1.0).len(), 5);
    }

    /// Deux zones de sol légitimement différentes (6 t/ha puis 12 t/ha),
    /// avec un point anormal à x=15 qui vaut 6.0 (comme la zone A) alors
    /// que ses voisins immédiats sont tous à 12.0 (zone B). Comme ce
    /// point anormal a *exactement* la même valeur qu'un point normal de
    /// zone A, un filtre global lui donne le même score-z que ce point
    /// légitime — il ne peut structurellement pas distinguer les deux.
    /// Le filtre local, lui, ne regarde que le voisinage immédiat et
    /// détecte la rupture. Valeurs vérifiées indépendamment (numpy)
    /// avant portage.
    #[test]
    fn local_filter_catches_what_global_filter_structurally_cannot() {
        let mut points = vec![];
        for i in 0..10
        {
            points.push(pt(i as f64, 0.0, 6.0)); // zone A, indices 0..9
        }
        for i in 10..20
        {
            let yield_value = if i == 15 { 6.0 } else { 12.0 }; // zone B, anomaly at index 15
            points.push(pt(i as f64, 0.0, yield_value));
        }

        let global_kept = global_filter(&points, 2.0);
        assert_eq!(
            global_kept.len(),
            points.len(),
            "a global filter cannot single out a value it also sees as legitimate elsewhere"
        );

        let local_kept = local_filter(&points, 3, 2.0);
        assert!(
            !local_kept.contains(&15),
            "the zone-B anomaly should be rejected locally"
        );
        assert_eq!(local_kept.len(), points.len() - 1);
    }
}
