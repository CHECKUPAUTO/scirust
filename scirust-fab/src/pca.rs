//! Détection de défauts multivariée par analyse en composantes
//! principales (PCA) — la paire de statistiques T²/SPE (Q-résidu)
//! standard en FDC de fabrication de semi-conducteurs (Kourti &
//! MacGregor, "Multivariate SPC Methods for Process and Product
//! Monitoring", Journal of Quality Technology 27(2), 1995 ; Wise &
//! Gallagher, "The process chemometrics approach to process monitoring
//! and fault detection", Journal of Process Control 6(6), 1996).
//!
//! `T²` capture les excursions *dans* le sous-espace des `k` premières
//! composantes principales (variation normale mais anormalement grande) ;
//! `SPE` (Squared Prediction Error, ou Q-statistic) capture ce qui *sort*
//! de ce sous-espace (une rupture de la structure de corrélation entre
//! variables que le modèle PCA en régime nominal n'a jamais vue). Les
//! deux ensemble couvrent des modes de défaut complémentaires qu'aucun
//! des deux seuls ne détecte — voir les tests pour un exemple explicite
//! des deux cas.
//!
//! Repose sur [`scirust_solvers::linalg::svd`] (SVD de Jacobi à un côté,
//! déterministe) pour l'extraction des composantes : la PCA sur des
//! données centrées `X` équivaut à la SVD `X = UΣVᵀ`, les colonnes de `V`
//! étant les chargements (loadings) et `UΣ` les scores.
//!
//! **Limite honnête** : ni sélection automatique de `k` (variance
//! expliquée, validation croisée — laissée à l'appelant), ni seuils
//! UCL pour `T²`/`SPE` calculés ici (comme
//! `scirust_multivariate::mahalanobis_outliers`, ce module ne
//! réimplémente pas l'inverse d'une fonction de répartition — les seuils
//! se lisent dans une table ou s'estiment empiriquement sur des données
//! en régime nominal).

use scirust_solvers::SolverError;
use scirust_solvers::linalg::{Matrix, svd};

/// Modèle PCA ajusté sur des données en régime nominal.
#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    mean: Vec<f64>,
    /// Chargements (p × k) : chaque colonne une composante retenue.
    loadings: Matrix,
    /// Écart-type des scores sur chaque composante retenue (pour T²).
    score_std: Vec<f64>,
}

impl Pca {
    /// Ajuste un modèle PCA à `k` composantes sur des observations en
    /// régime nominal (chaque ligne un vecteur de longueur `p`).
    /// `k` est plafonné à `min(n, p)` ; les composantes de variance
    /// numériquement nulle sont ignorées pour éviter une division par
    /// zéro dans [`Pca::t2`].
    pub fn fit(data: &[Vec<f64>], k: usize) -> Result<Self, SolverError> {
        let n = data.len();
        if n < 2 || k == 0
        {
            return Err(SolverError::InvalidInput(
                "Pca::fit: need at least 2 rows and k >= 1".to_string(),
            ));
        }
        let p = data[0].len();
        let mut mean = vec![0.0; p];
        for row in data
        {
            for (m, &v) in mean.iter_mut().zip(row)
            {
                *m += v;
            }
        }
        for m in mean.iter_mut()
        {
            *m /= n as f64;
        }

        let centered = Matrix::from_fn(n, p, |i, j| data[i][j] - mean[j]);
        let decomposition = svd(&centered)?;
        let k = k.min(decomposition.s.len());
        let denom = ((n - 1).max(1)) as f64;
        let score_std: Vec<f64> = decomposition.s[..k]
            .iter()
            .map(|&s| s / denom.sqrt())
            .collect();
        // Drop trailing components with negligible variance (rank-deficient
        // input) rather than risk a near-zero denominator in `t2`.
        let kept = score_std
            .iter()
            .take_while(|&&sd| sd > 1e-10)
            .count()
            .max(1);
        let loadings = Matrix::from_fn(p, kept, |i, j| decomposition.v[(i, j)]);
        Ok(Self {
            mean,
            loadings,
            score_std: score_std[..kept].to_vec(),
        })
    }

    /// Scores de `x` dans le sous-espace PCA (longueur = nombre de composantes retenues).
    pub fn scores(&self, x: &[f64]) -> Vec<f64> {
        let centered: Vec<f64> = x.iter().zip(&self.mean).map(|(xi, m)| xi - m).collect();
        (0..self.loadings.cols())
            .map(|j| {
                (0..self.loadings.rows())
                    .map(|i| centered[i] * self.loadings[(i, j)])
                    .sum()
            })
            .collect()
    }

    /// `T²` de Hotelling dans le sous-espace des composantes retenues.
    pub fn t2(&self, x: &[f64]) -> f64 {
        self.scores(x)
            .iter()
            .zip(&self.score_std)
            .map(|(s, sd)| (s / sd).powi(2))
            .sum()
    }

    /// SPE (Q-résidu) : énergie non expliquée par les composantes retenues.
    pub fn spe(&self, x: &[f64]) -> f64 {
        let centered: Vec<f64> = x.iter().zip(&self.mean).map(|(xi, m)| xi - m).collect();
        let scores = self.scores(x);
        let mut residual = centered;
        for (j, &score) in scores.iter().enumerate()
        {
            for (i, r) in residual.iter_mut().enumerate()
            {
                *r -= self.loadings[(i, j)] * score;
            }
        }
        residual.iter().map(|r| r * r).sum()
    }

    pub fn n_components(&self) -> usize {
        self.loadings.cols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Données exactement de rang 1 (`x2 = 2·x1`) : la 1ère composante
    /// principale capture toute la variance, la 2e est numériquement
    /// nulle. Valeurs vérifiées indépendamment (numpy SVD) avant
    /// portage.
    fn rank_one_training_set() -> Vec<Vec<f64>> {
        (1..=10).map(|t| vec![t as f64, 2.0 * t as f64]).collect()
    }

    #[test]
    fn drops_the_numerically_null_second_component() {
        let pca = Pca::fit(&rank_one_training_set(), 2).unwrap();
        assert_eq!(pca.n_components(), 1);
    }

    #[test]
    fn in_control_point_has_near_zero_spe_and_small_t2() {
        let pca = Pca::fit(&rank_one_training_set(), 2).unwrap();
        let x = vec![5.0, 10.0]; // from the training set, on the line
        assert_relative_eq!(pca.t2(&x), 0.027_272_727_27, epsilon = 1e-6);
        assert!(pca.spe(&x) < 1e-20, "SPE {}", pca.spe(&x));
    }

    #[test]
    fn breaking_the_correlation_is_caught_by_spe_not_t2() {
        let pca = Pca::fit(&rank_one_training_set(), 2).unwrap();
        // Same order of magnitude as training data, but violates x2=2*x1.
        let x = vec![5.0, 5.0];
        assert_relative_eq!(pca.t2(&x), 0.681_818_18, epsilon = 1e-6);
        assert_relative_eq!(pca.spe(&x), 5.0, epsilon = 1e-6);
        // The point where T2/SPE roles reverse (see next test) shows why both
        // statistics are needed: T2 alone is modest here, SPE is decisive.
    }

    #[test]
    fn a_large_excursion_along_the_known_correlation_is_caught_by_t2_not_spe() {
        let pca = Pca::fit(&rank_one_training_set(), 2).unwrap();
        // Respects x2=2*x1 exactly, but far outside the training range.
        let x = vec![20.0, 40.0];
        assert_relative_eq!(pca.t2(&x), 22.936_363_64, epsilon = 1e-5);
        assert!(pca.spe(&x) < 1e-20, "SPE {}", pca.spe(&x));
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(Pca::fit(&[vec![1.0, 2.0]], 1).is_err()); // n < 2
        assert!(Pca::fit(&rank_one_training_set(), 0).is_err()); // k == 0
    }
}
