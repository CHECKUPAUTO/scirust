//! Multivariate pattern detection algorithms.
//!
//! Provides PCA, ICA, K-Means clustering, Mahalanobis distance,
//! Multi-dimensional Scaling, and Canonical Correlation Analysis.
//!
//! All algorithms operate on `f64` data and use pure-Rust linear algebra
//! with no external dependencies beyond `scirust-core` and `serde`.

use serde::{Deserialize, Serialize};
use std::fmt;

// ────────────────────────────── helpers ──────────────────────────────

/// Row-major matrix stored as `Vec<Vec<f64>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn from_slice(data: &[&[f64]]) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self {
            rows,
            cols,
            data: data.iter().map(|r| r.to_vec()).collect(),
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r][c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r][c] = v;
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows
        {
            for j in 0..self.cols
            {
                t.data[j][i] = self.data[i][j];
            }
        }
        t
    }

    /// Matrix multiply: self (m×n) × other (n×p) → (m×p).
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "dimension mismatch for multiplication"
        );
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows
        {
            for k in 0..self.cols
            {
                let aik = self.data[i][k];
                if aik == 0.0
                {
                    continue;
                }
                for j in 0..other.cols
                {
                    out.data[i][j] += aik * other.data[k][j];
                }
            }
        }
        out
    }

    /// self × v (matrix-vector).
    #[allow(clippy::needless_range_loop)]
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows
        {
            for j in 0..self.cols
            {
                out[i] += self.data[i][j] * v[j];
            }
        }
        out
    }

    /// Column-wise mean.
    pub fn col_mean(&self) -> Vec<f64> {
        let n = self.rows as f64;
        (0..self.cols)
            .map(|j| self.data.iter().map(|r| r[j]).sum::<f64>() / n)
            .collect()
    }

    /// Center each column (subtract mean).
    pub fn center(&self) -> (Self, Vec<f64>) {
        let means = self.col_mean();
        let mut c = self.clone();
        for row in &mut c.data
        {
            for (j, &m) in means.iter().enumerate()
            {
                row[j] -= m;
            }
        }
        (c, means)
    }

    /// Covariance matrix of a centered matrix (columns = variables, rows = observations).
    pub fn cov_matrix(&self) -> Matrix {
        let n = self.rows as f64;
        let p = self.cols;
        let mut cov = Matrix::zeros(p, p);
        for i in 0..p
        {
            for j in 0..p
            {
                let mut s = 0.0;
                for k in 0..self.rows
                {
                    s += self.data[k][i] * self.data[k][j];
                }
                cov.data[i][j] = s / n;
            }
        }
        cov
    }

    /// Extract a sub-matrix by row indices.
    pub fn select_rows(&self, indices: &[usize]) -> Self {
        let mut out = Matrix::zeros(indices.len(), self.cols);
        for (oi, &ri) in indices.iter().enumerate()
        {
            out.data[oi] = self.data[ri].clone();
        }
        out
    }

    /// Extract a single column as a Vec.
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i][j]).collect()
    }

    /// Sum of all elements.
    pub fn sum(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).sum()
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
    }

    /// Element-wise subtraction: self - other.
    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = self.clone();
        for i in 0..self.rows
        {
            for j in 0..self.cols
            {
                out.data[i][j] -= other.data[i][j];
            }
        }
        out
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.data.iter().enumerate()
        {
            if i > 0
            {
                writeln!(f)?;
            }
            for (j, &v) in row.iter().enumerate()
            {
                if j > 0
                {
                    write!(f, " ")?;
                }
                write!(f, "{v:12.6}")?;
            }
        }
        Ok(())
    }
}

// ────────────────────────────── PCA ──────────────────────────────

/// Result of fitting PCA to data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaResult {
    /// Eigenvalues of the covariance matrix, sorted descending.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as rows, sorted by eigenvalue descending.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Explained variance ratio per component.
    pub explained_variance_ratio: Vec<f64>,
    /// Column means used for centering.
    pub means: Vec<f64>,
}

impl PcaResult {
    /// Number of components.
    pub fn n_components(&self) -> usize {
        self.eigenvalues.len()
    }
}

/// Fit PCA on a data matrix (rows = observations, cols = variables).
/// Uses Jacobi eigenvalue algorithm on the covariance matrix.
pub fn pca_fit(data: &Matrix) -> PcaResult {
    let (centered, means) = data.center();
    let cov = centered.cov_matrix();
    let n = cov.cols;

    // Jacobi eigenvalue algorithm for symmetric matrix
    let (eigenvalues, eigenvectors) = jacobi_eigen(&cov);

    // Sort descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_evals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let sorted_evecs: Vec<Vec<f64>> = indices.iter().map(|&i| eigenvectors[i].clone()).collect();

    let total_var: f64 = sorted_evals.iter().sum();
    let explained_variance_ratio: Vec<f64> = sorted_evals.iter().map(|e| e / total_var).collect();

    PcaResult {
        eigenvalues: sorted_evals,
        eigenvectors: sorted_evecs,
        explained_variance_ratio,
        means,
    }
}

/// Project data onto principal components. Returns (n_obs × n_components).
pub fn pca_transform(data: &Matrix, pca: &PcaResult, n_components: usize) -> Matrix {
    let nc = n_components.min(pca.eigenvectors.len());
    let centered = {
        let mut c = data.clone();
        for row in &mut c.data
        {
            for (j, &m) in pca.means.iter().enumerate()
            {
                row[j] -= m;
            }
        }
        c
    };

    let mut out = Matrix::zeros(data.rows, nc);
    for i in 0..data.rows
    {
        for j in 0..nc
        {
            let mut s = 0.0;
            for k in 0..data.cols
            {
                s += centered.data[i][k] * pca.eigenvectors[j][k];
            }
            out.data[i][j] = s;
        }
    }
    out
}

/// Inverse transform from principal components back to original space.
pub fn pca_inverse_transform(projections: &Matrix, pca: &PcaResult, n_components: usize) -> Matrix {
    let nc = n_components.min(pca.eigenvectors.len());
    assert_eq!(projections.cols, nc);
    let mut out = Matrix::zeros(projections.rows, pca.means.len());

    for i in 0..projections.rows
    {
        for j in 0..out.cols
        {
            let mut s = 0.0;
            for k in 0..nc
            {
                s += projections.data[i][k] * pca.eigenvectors[k][j];
            }
            out.data[i][j] = s + pca.means[j];
        }
    }
    out
}

/// Scree plot data: returns (component_index, cumulative_variance_ratio).
pub fn pca_scree(pca: &PcaResult) -> Vec<(usize, f64)> {
    let mut cum = 0.0;
    pca.explained_variance_ratio
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            cum += v;
            (i + 1, cum)
        })
        .collect()
}

// ──────────────────────── Jacobi eigenvalue ────────────────────────

/// Compute all eigenvalues and eigenvectors of a symmetric matrix
/// using the Jacobi rotation method.
fn jacobi_eigen(a: &Matrix) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.cols;
    let mut s = a.clone();
    let mut v = Matrix::zeros(n, n);
    for i in 0..n
    {
        v.data[i][i] = 1.0;
    }

    let max_iters = 100 * n * n;
    for _ in 0..max_iters
    {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n
        {
            for j in (i + 1)..n
            {
                if s.data[i][j].abs() > max_val
                {
                    max_val = s.data[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12
        {
            break;
        }

        // Compute rotation angle
        let diff = s.data[q][q] - s.data[p][p];
        let t = if s.data[p][p].abs() < 1e-15
        {
            1.0
        }
        else
        {
            let tau = diff / (2.0 * s.data[p][q]);
            if tau >= 0.0
            {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            }
            else
            {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            }
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s_val = t * c;

        // Apply rotation to s
        let mut new_s = s.clone();
        new_s.data[p][p] = s.data[p][p] - t * s.data[p][q];
        new_s.data[q][q] = s.data[q][q] + t * s.data[p][q];
        new_s.data[p][q] = 0.0;
        new_s.data[q][p] = 0.0;

        for r in 0..n
        {
            if r != p && r != q
            {
                let srp = s.data[r][p];
                let srq = s.data[r][q];
                new_s.data[r][p] = c * srp - s_val * srq;
                new_s.data[p][r] = new_s.data[r][p];
                new_s.data[r][q] = s_val * srp + c * srq;
                new_s.data[q][r] = new_s.data[r][q];
            }
        }
        s = new_s;

        // Update eigenvectors
        for r in 0..n
        {
            let vrp = v.data[r][p];
            let vrq = v.data[r][q];
            v.data[r][p] = c * vrp - s_val * vrq;
            v.data[r][q] = s_val * vrp + c * vrq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| s.data[i][i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n).map(|i| v.col(i)).collect();

    (eigenvalues, eigenvectors)
}

// ────────────────────────────── ICA ──────────────────────────────

/// Result of FastICA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcaResult {
    /// Separation matrix (k × n_features).
    pub separation_matrix: Vec<Vec<f64>>,
    /// Mixing matrix (n_features × k).
    pub mixing_matrix: Vec<Vec<f64>>,
    /// Number of independent components.
    pub n_components: usize,
    /// Column means used for centering.
    pub means: Vec<f64>,
}

/// Run FastICA with symmetric orthogonalization.
///
/// `data`: rows = observations, cols = variables.
/// `n_components`: number of independent components to extract.
/// `max_iter`: maximum iterations.
/// `tol`: convergence tolerance.
#[allow(clippy::needless_range_loop)]
pub fn ica_fit(data: &Matrix, n_components: usize, max_iter: usize, tol: f64) -> IcaResult {
    let (centered, means) = data.center();
    let n = centered.rows;
    let p = centered.cols;
    let k = n_components.min(p);

    // Whitening: compute covariance and its eigendecomposition
    let cov = centered.cov_matrix();
    let (evals, evecs) = jacobi_eigen(&cov);

    // Sort eigenvalues descending and take top k
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap());

    // Whitening matrix: W = D^{-1/2} * E^T
    let mut whitening = Matrix::zeros(k, p);
    for i in 0..k
    {
        let ei = idx[i];
        let scale = if evals[ei] > 1e-12
        {
            1.0 / evals[ei].sqrt()
        }
        else
        {
            0.0
        };
        for j in 0..p
        {
            whitening.data[i][j] = scale * evecs[ei][j];
        }
    }

    let x_white = whitening.mul(&centered.transpose()).transpose(); // (n × k)

    // Initialize W randomly (deterministic seed for reproducibility)
    let mut w = Matrix::zeros(k, k);
    let mut seed: u64 = 42;
    for i in 0..k
    {
        for j in 0..k
        {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            w.data[i][j] = ((seed >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
        }
    }

    // Symmetric decorrelation
    symmetric_decorrelate(&mut w);

    for _iter in 0..max_iter.min(max_iters(k))
    {
        let w_old = w.clone();

        // wX = W * X^T (k × n)
        let wx = w.mul(&x_white.transpose());

        // g(wx) = tanh(wx), g'(wx) = 1 - tanh(wx)^2
        let mut g_wx = Matrix::zeros(k, n);
        let mut gp_wx = Matrix::zeros(k, n);
        for i in 0..k
        {
            for j in 0..n
            {
                let v = wx.data[i][j].tanh();
                g_wx.data[i][j] = v;
                gp_wx.data[i][j] = 1.0 - v * v;
            }
        }

        // E{g(wX) * X^T} - E{g'(wX)} * w
        let gxt = g_wx.mul(&x_white); // (k × k) — approximate expectation via sum
        let gp_mean: Vec<f64> = (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        let mut s = 0.0;
                        for t in 0..n
                        {
                            if i == j
                            {
                                s += gp_wx.data[i][t];
                            }
                        }
                        s / n as f64
                    })
                    .sum::<f64>()
            })
            .collect();

        for i in 0..k
        {
            for j in 0..k
            {
                w.data[i][j] = gxt.data[i][j] / n as f64 - gp_mean[i] * w.data[i][j];
            }
        }

        symmetric_decorrelate(&mut w);

        // Check convergence
        let diff = w.sub(&w_old).frobenius_norm();
        if diff < tol
        {
            break;
        }
    }

    // Compute separation: W_full = W * whitening
    let sep = w.mul(&whitening);

    // Mixing = pseudo-inverse of separation
    let mixing = pseudo_inverse(&sep);

    IcaResult {
        separation_matrix: sep.data,
        mixing_matrix: mixing.data,
        n_components: k,
        means,
    }
}

/// Transform data to independent components.
pub fn ica_transform(data: &Matrix, ica: &IcaResult) -> Matrix {
    let centered = {
        let mut c = data.clone();
        for row in &mut c.data
        {
            for (j, &m) in ica.means.iter().enumerate()
            {
                row[j] -= m;
            }
        }
        c
    };
    let sep = Matrix::from_slice(
        &ica.separation_matrix
            .iter()
            .map(|r| r.as_slice())
            .collect::<Vec<_>>(),
    );
    sep.mul(&centered.transpose()).transpose()
}

#[allow(clippy::needless_range_loop)]
fn symmetric_decorrelate(w: &mut Matrix) {
    let k = w.rows;
    // WW^T
    let wwt = w.mul(&w.transpose());
    let (evals, evecs) = jacobi_eigen(&wwt);
    let mut inv_sqrt_d = Matrix::zeros(k, k);
    for i in 0..k
    {
        let s = if evals[i] > 1e-12
        {
            1.0 / evals[i].sqrt()
        }
        else
        {
            0.0
        };
        inv_sqrt_d.data[i][i] = s;
    }
    // W = E * D^{-1/2} * E^T * W
    let temp = inv_sqrt_d.mul(&evecs_transpose(&evecs));
    let temp2 = temp.mul(w);
    *w = temp2;
}

#[allow(clippy::needless_range_loop)]
fn evecs_transpose(evecs: &[Vec<f64>]) -> Matrix {
    let n = evecs.len();
    let mut m = Matrix::zeros(n, n);
    for i in 0..n
    {
        for j in 0..n
        {
            m.data[i][j] = evecs[j][i];
        }
    }
    m
}

fn max_iters(k: usize) -> usize {
    1000.max(k * 100)
}

/// Moore-Penrose pseudo-inverse via SVD (simplified).
#[allow(clippy::needless_range_loop)]
fn pseudo_inverse(m: &Matrix) -> Matrix {
    let (u, s, vt) = svd(m);
    let mut inv = Matrix::zeros(m.cols, m.rows);
    for i in 0..m.cols.min(m.rows)
    {
        let si = if s[i].abs() > 1e-12 { 1.0 / s[i] } else { 0.0 };
        for j in 0..m.rows.min(u.rows)
        {
            for k in 0..vt.rows.min(m.cols)
            {
                inv.data[k][j] += vt.data[i][k] * si * u.data[j][i];
            }
        }
    }
    inv
}

/// Thin SVD via Jacobi rotations (for small matrices).
/// Returns (U, singular_values, V^T).
#[allow(clippy::needless_range_loop)]
fn svd(m: &Matrix) -> (Matrix, Vec<f64>, Matrix) {
    let n = m.rows;
    let p = m.cols;
    let k = n.min(p);

    // Compute M^T M (p × p)
    let mt = m.transpose();
    let mtm = mt.mul(m);

    let (evals, evecs) = jacobi_eigen(&mtm);

    // Singular values = sqrt of eigenvalues of M^T M
    let mut svs: Vec<(f64, Vec<f64>)> = evals
        .iter()
        .zip(evecs.iter())
        .map(|(e, v)| (e.max(0.0).sqrt(), v.clone()))
        .collect();
    svs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let singular_values: Vec<f64> = svs.iter().map(|(s, _)| *s).collect();
    let v = Matrix {
        rows: p,
        cols: k,
        data: svs.iter().map(|(_, v)| v.clone()).collect(),
    };

    // U = M * V * S^{-1}
    let mut u = Matrix::zeros(n, k);
    for j in 0..k
    {
        let s_val = if singular_values[j] > 1e-12
        {
            1.0 / singular_values[j]
        }
        else
        {
            0.0
        };
        for i in 0..n
        {
            let mut dot = 0.0;
            for l in 0..p
            {
                dot += m.data[i][l] * v.data[l][j];
            }
            u.data[i][j] = dot * s_val;
        }
    }

    (u, singular_values, v.transpose())
}

// ────────────────────────── K-Means ──────────────────────────────

/// Result of K-Means clustering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansResult {
    /// Cluster assignments for each observation.
    pub labels: Vec<usize>,
    /// Centroid coordinates (k × n_features).
    pub centroids: Vec<Vec<f64>>,
    /// Final inertia (sum of squared distances to assigned centroid).
    pub inertia: f64,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Number of clusters.
    pub k: usize,
}

/// Run K-Means++ on data (rows = observations, cols = features).
///
/// `k`: number of clusters.
/// `max_iter`: maximum iterations.
/// `tol`: convergence tolerance on centroid movement.
#[allow(clippy::needless_range_loop)]
pub fn kmeans_fit(data: &Matrix, k: usize, max_iter: usize, tol: f64) -> KMeansResult {
    let n = data.rows;
    let p = data.cols;

    // K-Means++ initialization
    let mut centroids = kmeans_pp_init(data, k);
    let mut labels = vec![0usize; n];

    let mut iter_count = 0;
    for _iter in 0..max_iter
    {
        iter_count = _iter + 1;

        // Assign each point to nearest centroid
        let mut changed = false;
        for i in 0..n
        {
            let nearest = nearest_centroid(&data.data[i], &centroids);
            if nearest != labels[i]
            {
                labels[i] = nearest;
                changed = true;
            }
        }

        // Update centroids
        let mut new_centroids = vec![vec![0.0; p]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n
        {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..p
            {
                new_centroids[c][j] += data.data[i][j];
            }
        }
        for c in 0..k
        {
            if counts[c] > 0
            {
                for j in 0..p
                {
                    new_centroids[c][j] /= counts[c] as f64;
                }
            }
        }

        // Check convergence
        let shift: f64 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| {
                old.iter()
                    .zip(new.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
            })
            .sum();

        centroids = new_centroids;

        if !changed || shift.sqrt() < tol
        {
            break;
        }
    }

    // Final assignment and inertia
    let mut inertia = 0.0;
    for i in 0..n
    {
        let c = labels[i];
        for j in 0..p
        {
            let d = data.data[i][j] - centroids[c][j];
            inertia += d * d;
        }
    }

    KMeansResult {
        labels,
        centroids,
        inertia,
        n_iter: iter_count,
        k,
    }
}

#[allow(clippy::needless_range_loop)]
fn kmeans_pp_init(data: &Matrix, k: usize) -> Vec<Vec<f64>> {
    let n = data.rows;
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut rng_state: u64 = 12345;

    // First centroid: random
    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let first = (rng_state >> 11) as usize % n;
    centroids.push(data.data[first].clone());

    // Remaining centroids: probability proportional to distance
    for _ in 1..k
    {
        let mut dists: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n
        {
            let mut min_d = f64::INFINITY;
            for c in &centroids
            {
                let d: f64 = data.data[i]
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                if d < min_d
                {
                    min_d = d;
                }
            }
            dists.push(min_d);
        }

        let total: f64 = dists.iter().sum();
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut threshold = ((rng_state >> 11) as f64) / (1u64 << 53) as f64 * total;

        let mut pushed = false;
        for i in 0..n
        {
            threshold -= dists[i];
            if threshold <= 0.0
            {
                centroids.push(data.data[i].clone());
                pushed = true;
                break;
            }
        }
        if !pushed
        {
            // fallback: pick point with max distance
            let last = dists
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            centroids.push(data.data[last].clone());
        }
    }
    centroids
}

fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (i, c) in centroids.iter().enumerate()
    {
        let d: f64 = point
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        if d < best_d
        {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Elbow method: compute inertia for k = 1..max_k.
pub fn elbow_method(data: &Matrix, max_k: usize, max_iter: usize) -> Vec<(usize, f64)> {
    (1..=max_k)
        .map(|k| {
            let result = kmeans_fit(data, k, max_iter, 1e-6);
            (k, result.inertia)
        })
        .collect()
}

/// Compute silhouette score for a clustering.
///
/// Returns average silhouette coefficient over all observations.
/// Values range from -1 (bad) to +1 (good).
#[allow(clippy::needless_range_loop)]
pub fn silhouette_score(data: &Matrix, labels: &[usize]) -> f64 {
    let n = data.rows;
    if n <= 1
    {
        return 0.0;
    }

    let k = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut scores = Vec::with_capacity(n);

    for i in 0..n
    {
        let ci = labels[i];

        // a(i): mean intra-cluster distance
        let mut a_sum = 0.0;
        let mut a_count = 0usize;
        for j in 0..n
        {
            if j != i && labels[j] == ci
            {
                a_sum += euclidean_dist(&data.data[i], &data.data[j]);
                a_count += 1;
            }
        }
        let a_i = if a_count > 0
        {
            a_sum / a_count as f64
        }
        else
        {
            0.0
        };

        // b(i): min mean inter-cluster distance
        let mut b_i = f64::INFINITY;
        for cluster in 0..k
        {
            if cluster == ci
            {
                continue;
            }
            let mut b_sum = 0.0;
            let mut b_count = 0usize;
            for j in 0..n
            {
                if labels[j] == cluster
                {
                    b_sum += euclidean_dist(&data.data[i], &data.data[j]);
                    b_count += 1;
                }
            }
            if b_count > 0
            {
                let b_mean = b_sum / b_count as f64;
                if b_mean < b_i
                {
                    b_i = b_mean;
                }
            }
        }

        let s_i = if b_i.is_infinite()
        {
            // No other clusters: silhouette is undefined, treat as 0
            0.0
        }
        else if a_i.max(b_i) > 1e-15
        {
            (b_i - a_i) / a_i.max(b_i)
        }
        else
        {
            0.0
        };
        scores.push(s_i);
    }

    scores.iter().sum::<f64>() / n as f64
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ──────────────────────── Mahalanobis ────────────────────────────

/// Compute the Mahalanobis distance from a point to a centroid,
/// given the inverse covariance matrix.
#[allow(clippy::needless_range_loop)]
pub fn mahalanobis_distance(point: &[f64], mean: &[f64], inv_cov: &Matrix) -> f64 {
    assert_eq!(point.len(), mean.len());
    assert_eq!(point.len(), inv_cov.rows);
    let n = point.len();

    // d = sqrt((x - mu)^T * S^{-1} * (x - mu))
    let mut diff = vec![0.0; n];
    for i in 0..n
    {
        diff[i] = point[i] - mean[i];
    }

    let mut temp = vec![0.0; n];
    for i in 0..n
    {
        for j in 0..n
        {
            temp[i] += inv_cov.data[i][j] * diff[j];
        }
    }

    let mut d2 = 0.0;
    for i in 0..n
    {
        d2 += diff[i] * temp[i];
    }
    d2.sqrt()
}

/// Detect outliers using Mahalanobis distance with a chi-squared threshold.
///
/// Returns indices of outlier observations.
pub fn mahalanobis_outliers(data: &Matrix, threshold: f64) -> Vec<usize> {
    let (centered, means) = data.center();
    let cov = centered.cov_matrix();
    let inv_cov = invert_matrix(&cov);

    let mut outliers = Vec::new();
    for i in 0..data.rows
    {
        let d = mahalanobis_distance(&data.data[i], &means, &inv_cov);
        if d > threshold
        {
            outliers.push(i);
        }
    }
    outliers
}

/// Compute Mahalanobis distances for all observations.
pub fn mahalanobis_distances(data: &Matrix) -> Vec<f64> {
    let (centered, means) = data.center();
    let cov = centered.cov_matrix();
    let inv_cov = invert_matrix(&cov);

    (0..data.rows)
        .map(|i| mahalanobis_distance(&data.data[i], &means, &inv_cov))
        .collect()
}

/// Invert a symmetric positive-definite matrix via Cholesky decomposition.
fn invert_matrix(m: &Matrix) -> Matrix {
    let n = m.rows;
    assert_eq!(n, m.cols);

    // Cholesky decomposition: M = L * L^T
    let l = cholesky(m);

    // Invert L (lower triangular)
    let mut l_inv = Matrix::zeros(n, n);
    for i in 0..n
    {
        l_inv.data[i][i] = 1.0 / l.data[i][i];
        for j in (i + 1)..n
        {
            let mut s = 0.0;
            for k in i..j
            {
                s += l.data[j][k] * l_inv.data[k][i];
            }
            l_inv.data[j][i] = -s / l.data[j][j];
        }
    }

    // M^{-1} = L^{-T} * L^{-1}
    let lt_inv = l_inv.transpose();
    lt_inv.mul(&l_inv)
}

/// Cholesky decomposition: returns lower triangular L such that M = L * L^T.
fn cholesky(m: &Matrix) -> Matrix {
    let n = m.rows;
    let mut l = Matrix::zeros(n, n);

    for i in 0..n
    {
        for j in 0..=i
        {
            let mut s = 0.0;
            for k in 0..j
            {
                s += l.data[i][k] * l.data[j][k];
            }
            if i == j
            {
                let val = m.data[i][i] - s;
                l.data[i][j] = if val > 0.0 { val.sqrt() } else { 1e-10 };
            }
            else
            {
                l.data[i][j] = (m.data[i][j] - s) / l.data[j][j];
            }
        }
    }
    l
}

// ────────────────────────── MDS ──────────────────────────────────

/// Result of Multi-Dimensional Scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdsResult {
    /// Embedding coordinates (n × n_components).
    pub coordinates: Vec<Vec<f64>>,
    /// Stress value (goodness of fit).
    pub stress: f64,
    /// Eigenvalues from the double centering.
    pub eigenvalues: Vec<f64>,
}

/// Classical (metric) MDS.
///
/// `dist_matrix`: symmetric distance matrix (n × n).
/// `n_components`: target dimensionality.
#[allow(clippy::needless_range_loop)]
pub fn classical_mds(dist_matrix: &Matrix, n_components: usize) -> MdsResult {
    let n = dist_matrix.rows;
    assert_eq!(n, dist_matrix.cols);

    // Double centering: B = -0.5 * H * D^2 * H
    // where H = I - (1/n)*11^T and D^2 is squared distances
    let mut b = Matrix::zeros(n, n);
    let mut row_sums = vec![0.0; n];
    let mut col_sums = vec![0.0; n];
    let mut grand_sum = 0.0;

    // Compute squared distances and row/col sums
    for i in 0..n
    {
        for j in 0..n
        {
            let d2 = dist_matrix.data[i][j].powi(2);
            b.data[i][j] = d2;
            row_sums[i] += d2;
            col_sums[j] += d2;
            grand_sum += d2;
        }
    }

    let inv_n = 1.0 / n as f64;
    let inv_n2 = inv_n * inv_n;

    // B = -0.5 * (D^2 - row_mean - col_mean + grand_mean)
    for i in 0..n
    {
        for j in 0..n
        {
            b.data[i][j] = -0.5
                * (b.data[i][j] - row_sums[i] * inv_n - col_sums[j] * inv_n + grand_sum * inv_n2);
        }
    }

    // Eigendecomposition of B
    let (evals, evecs) = jacobi_eigen(&b);

    // Sort descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap());

    let sorted_evals: Vec<f64> = indices
        .iter()
        .map(|&i| evals[i])
        .take(n_components)
        .collect();
    let sorted_evecs: Vec<Vec<f64>> = indices
        .iter()
        .map(|&i| evecs[i].clone())
        .take(n_components)
        .collect();

    // Coordinates = sqrt(eigenvalues) * eigenvectors
    let mut coords = Matrix::zeros(n, n_components);
    for j in 0..n_components
    {
        let scale = if sorted_evals[j] > 0.0
        {
            sorted_evals[j].sqrt()
        }
        else
        {
            0.0
        };
        for i in 0..n
        {
            coords.data[i][j] = sorted_evecs[j][i] * scale;
        }
    }

    let stress = compute_stress(dist_matrix, &coords);

    MdsResult {
        coordinates: coords.data,
        stress,
        eigenvalues: sorted_evals,
    }
}

/// Compute Kruskal's stress for an embedding.
pub fn compute_stress(dist_matrix: &Matrix, embedding: &Matrix) -> f64 {
    let n = dist_matrix.rows;
    let mut num = 0.0;
    let mut den = 0.0;

    for i in 0..n
    {
        for j in (i + 1)..n
        {
            let orig = dist_matrix.data[i][j];
            let emb: f64 = embedding.data[i]
                .iter()
                .zip(embedding.data[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            num += (orig - emb).powi(2);
            den += orig * orig;
        }
    }

    if den > 0.0 { (num / den).sqrt() } else { 0.0 }
}

// ────────────────────────── CCA ──────────────────────────────────

/// Result of Canonical Correlation Analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CcaResult {
    /// Canonical correlations.
    pub correlations: Vec<f64>,
    /// Canonical weights for the first view.
    pub weights_x: Vec<Vec<f64>>,
    /// Canonical weights for the second view.
    pub weights_y: Vec<Vec<f64>>,
}

/// Canonical Correlation Analysis between two sets of variables.
///
/// `x`: first data matrix (n × p1).
/// `y`: second data matrix (n × p2).
/// `n_components`: number of canonical pairs to return.
pub fn cca_fit(x: &Matrix, y: &Matrix, n_components: usize) -> CcaResult {
    assert_eq!(
        x.rows, y.rows,
        "x and y must have same number of observations"
    );
    let n = x.rows;
    let p1 = x.cols;
    let p2 = y.cols;

    let nc = n_components.min(p1).min(p2);

    // Center both views
    let (cx, _) = x.center();
    let (cy, _) = y.center();

    // Covariance matrices
    let sigma_xx = cx.cov_matrix();
    let sigma_yy = cy.cov_matrix();
    let sigma_xy = {
        let mut m = Matrix::zeros(p1, p2);
        for i in 0..p1
        {
            for j in 0..p2
            {
                let mut s = 0.0;
                for k in 0..n
                {
                    s += cx.data[k][i] * cy.data[k][j];
                }
                m.data[i][j] = s / n as f64;
            }
        }
        m
    };
    let sigma_yx = sigma_xy.transpose();

    // Invert covariance matrices
    let inv_xx = invert_matrix(&sigma_xx);
    let inv_yy = invert_matrix(&sigma_yy);

    // Solve generalized eigenproblem:
    // Sigma_xx^{-1} * Sigma_xy * Sigma_yy^{-1} * Sigma_yx * w = lambda * w
    let a = inv_xx.mul(&sigma_xy).mul(&inv_yy).mul(&sigma_yx);
    let b = inv_yy.mul(&sigma_yx).mul(&inv_xx).mul(&sigma_xy);

    let (evals_a, evecs_a) = jacobi_eigen(&a);
    let (evals_b, evecs_b) = jacobi_eigen(&b);

    // Sort by eigenvalue descending
    let mut indices_a: Vec<usize> = (0..p1).collect();
    indices_a.sort_by(|&i, &j| evals_a[j].partial_cmp(&evals_a[i]).unwrap());

    let mut correlations = Vec::with_capacity(nc);
    let mut weights_x = Vec::with_capacity(nc);
    let mut weights_y = Vec::with_capacity(nc);

    for &idx in indices_a.iter().take(nc)
    {
        let r = evals_a[idx].clamp(0.0, 1.0).sqrt();
        correlations.push(r);
        weights_x.push(evecs_a[idx].clone());
    }

    // Derive weights_y from weights_x: w_y = Sigma_yy^{-1} * Sigma_yx * w_x / lambda
    let mut indices_b: Vec<usize> = (0..p2).collect();
    indices_b.sort_by(|&i, &j| evals_b[j].partial_cmp(&evals_b[i]).unwrap());

    for (ci, &idx_a) in indices_a.iter().take(nc).enumerate()
    {
        let lambda = evals_a[idx_a];
        if lambda > 1e-12 && ci < indices_b.len()
        {
            let idx_b = indices_b[ci];
            weights_y.push(evecs_b[idx_b].clone());
        }
        else
        {
            weights_y.push(vec![0.0; p2]);
        }
    }

    CcaResult {
        correlations,
        weights_x,
        weights_y,
    }
}

/// Transform data to canonical variates.
pub fn cca_transform(x: &Matrix, y: &Matrix, cca: &CcaResult) -> (Matrix, Matrix) {
    let (cx, _) = x.center();
    let (cy, _) = y.center();

    let wx = Matrix::from_slice(
        &cca.weights_x
            .iter()
            .map(|r| r.as_slice())
            .collect::<Vec<_>>(),
    );
    let wy = Matrix::from_slice(
        &cca.weights_y
            .iter()
            .map(|r| r.as_slice())
            .collect::<Vec<_>>(),
    );

    // Canonical variates: X_can = X * W_x, Y_can = Y * W_y
    // weights_x is (nc × p1), so transpose to (p1 × nc) for right-multiply
    let x_can = cx.mul(&wx.transpose());
    let y_can = cy.mul(&wy.transpose());

    (x_can, y_can)
}

// ─────────────────────────── tests ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iris_like_data() -> Matrix {
        // Small 4-variate dataset (5 observations)
        Matrix::from_slice(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.3, 3.3, 4.6, 1.3],
        ])
    }

    // ── Matrix ──

    #[test]
    fn matrix_transpose() {
        let m = Matrix::from_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.data[0], vec![1.0, 4.0]);
        assert_eq!(t.data[1], vec![2.0, 5.0]);
        assert_eq!(t.data[2], vec![3.0, 6.0]);
    }

    #[test]
    fn matrix_mul() {
        let a = Matrix::from_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = Matrix::from_slice(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let c = a.mul(&b);
        assert_eq!(c.data[0], vec![19.0, 22.0]);
        assert_eq!(c.data[1], vec![43.0, 50.0]);
    }

    #[test]
    fn matrix_col_mean() {
        let m = Matrix::from_slice(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let mean = m.col_mean();
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_covariance() {
        let m = Matrix::from_slice(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let (centered, _means) = m.center();
        let cov = centered.cov_matrix();
        // Variance of [1,3,5] = 8/3 ≈ 2.6667
        let expected_var = 8.0 / 3.0;
        assert!((cov.data[0][0] - expected_var).abs() < 1e-10);
    }

    // ── PCA ──

    #[test]
    fn pca_basic() {
        let data = iris_like_data();
        let pca = pca_fit(&data);
        assert_eq!(pca.n_components(), 4);
        // All eigenvalues should be non-negative
        for &e in &pca.eigenvalues
        {
            assert!(e >= -1e-10, "negative eigenvalue: {e}");
        }
        // Explained variance ratios should sum to ~1
        let total: f64 = pca.explained_variance_ratio.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "total explained variance: {total}"
        );
    }

    #[test]
    fn pca_transform_roundtrip() {
        let data = iris_like_data();
        let pca = pca_fit(&data);
        let proj = pca_transform(&data, &pca, 2);
        assert_eq!(proj.rows, 5);
        assert_eq!(proj.cols, 2);

        let reconstructed = pca_inverse_transform(&proj, &pca, 2);
        assert_eq!(reconstructed.rows, 5);
        assert_eq!(reconstructed.cols, 4);

        // Reconstruction error should be small (2 components < full rank)
        for i in 0..5
        {
            for j in 0..4
            {
                let err = (data.data[i][j] - reconstructed.data[i][j]).abs();
                // With only 2 of 4 components, some error is expected
                assert!(err < 5.0, "large reconstruction error at [{i}][{j}]: {err}");
            }
        }
    }

    #[test]
    fn pca_scree_data() {
        let data = iris_like_data();
        let pca = pca_fit(&data);
        let scree = super::pca_scree(&pca);
        assert_eq!(scree.len(), 4);
        // Last cumulative value should be 1.0
        assert!((scree[3].1 - 1.0).abs() < 1e-10);
        // Monotonically increasing
        for i in 1..scree.len()
        {
            assert!(scree[i].1 >= scree[i - 1].1);
        }
    }

    // ── ICA ──

    #[test]
    fn ica_basic() {
        // Create synthetic independent sources
        let n = 100;
        let mut data = Matrix::zeros(n, 2);
        let mut seed: u64 = 99;
        for i in 0..n
        {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s1 = ((seed >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s2 = ((seed >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
            // Mix
            data.data[i] = vec![s1 + 0.5 * s2, 0.3 * s1 + s2];
        }

        let ica = ica_fit(&data, 2, 200, 1e-6);
        assert_eq!(ica.n_components, 2);
        assert_eq!(ica.separation_matrix.len(), 2);
        assert_eq!(ica.separation_matrix[0].len(), 2);
    }

    #[test]
    fn ica_transform_produces_output() {
        let data = iris_like_data();
        let ica = ica_fit(&data, 2, 200, 1e-6);
        let transformed = ica_transform(&data, &ica);
        assert_eq!(transformed.rows, 5);
        assert_eq!(transformed.cols, 2);
    }

    // ── K-Means ──

    #[test]
    fn kmeans_basic() {
        let data = Matrix::from_slice(&[
            &[0.0, 0.0],
            &[0.1, 0.1],
            &[0.0, 0.1],
            &[10.0, 10.0],
            &[10.1, 10.1],
            &[10.0, 10.1],
        ]);
        let result = kmeans_fit(&data, 2, 100, 1e-6);
        assert_eq!(result.k, 2);
        assert_eq!(result.labels.len(), 6);

        // Points in same cluster should be close
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[3], result.labels[5]);

        // Different clusters for the two groups
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn kmeans_single_cluster() {
        let data = Matrix::from_slice(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let result = kmeans_fit(&data, 1, 100, 1e-6);
        assert_eq!(result.k, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
        // Centroid should be the mean [3, 4]
        assert!((result.centroids[0][0] - 3.0).abs() < 1e-10);
        assert!((result.centroids[0][1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn elbow_method_returns_results() {
        let data = iris_like_data();
        let results = elbow_method(&data, 3, 50);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[2].0, 3);
        // Inertia should be non-increasing
        for i in 1..results.len()
        {
            assert!(results[i].1 <= results[i - 1].1 + 1e-10);
        }
    }

    #[test]
    fn silhouette_score_good_clustering() {
        let data = Matrix::from_slice(&[&[0.0, 0.0], &[0.1, 0.1], &[10.0, 10.0], &[10.1, 10.1]]);
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels);
        assert!(score > 0.8, "expected high silhouette, got {score}");
    }

    #[test]
    fn silhouette_score_poor_clustering() {
        let data = Matrix::from_slice(&[&[0.0, 0.0], &[0.1, 0.1], &[10.0, 10.0], &[10.1, 10.1]]);
        let labels = vec![0, 1, 0, 1];
        let score = silhouette_score(&data, &labels);
        assert!(score < 0.0, "expected negative silhouette, got {score}");
    }

    // ── Mahalanobis ──

    #[test]
    fn mahalanobis_distance_basic() {
        let data = Matrix::from_slice(&[&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
        let mean = vec![0.5, 0.5];
        let cov = data.cov_matrix();
        let inv_cov = invert_matrix(&cov);
        let d = mahalanobis_distance(&mean, &mean, &inv_cov);
        assert!(
            (d - 0.0).abs() < 1e-10,
            "distance from centroid to itself should be 0"
        );
    }

    #[test]
    fn mahalanobis_distances_returns_all() {
        let data = iris_like_data();
        let dists = mahalanobis_distances(&data);
        assert_eq!(dists.len(), 5);
        for d in &dists
        {
            assert!(*d >= 0.0);
        }
    }

    #[test]
    fn mahalanobis_outliers_detects() {
        // Clusters of normal points with one outlier
        let data = Matrix::from_slice(&[
            &[0.0, 0.0],
            &[0.1, 0.0],
            &[0.0, 0.1],
            &[0.1, 0.1],
            &[0.05, 0.05],
            &[0.08, 0.02],
            &[0.02, 0.07],
            &[0.0, 0.05],
            &[0.05, 0.0],
            &[0.0, 0.0],
            &[10.0, 10.0], // outlier
        ]);
        let dists = mahalanobis_distances(&data);
        let max_d = dists.iter().cloned().fold(0.0f64, f64::max);
        let outliers = mahalanobis_outliers(&data, max_d - 0.01);
        assert!(
            outliers.contains(&10),
            "should detect the outlier at index 10, dists={:?}",
            dists
        );
    }

    // ── Invert / Cholesky ──

    #[test]
    fn invert_matrix_identity() {
        let m = Matrix::from_slice(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let inv = invert_matrix(&m);
        let product = m.mul(&inv);
        for i in 0..2
        {
            for j in 0..2
            {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.data[i][j] - expected).abs() < 1e-8,
                    "inverse product mismatch at [{i}][{j}]: {}",
                    product.data[i][j]
                );
            }
        }
    }

    #[test]
    fn cholesky_basic() {
        let m = Matrix::from_slice(&[&[4.0, 2.0], &[2.0, 3.0]]);
        let l = cholesky(&m);
        // L * L^T should equal M
        let lt = l.transpose();
        let product = l.mul(&lt);
        for i in 0..2
        {
            for j in 0..2
            {
                assert!(
                    (product.data[i][j] - m.data[i][j]).abs() < 1e-8,
                    "Cholesky reconstruction mismatch"
                );
            }
        }
    }

    // ── MDS ──

    #[test]
    fn mds_basic() {
        let dist = Matrix::from_slice(&[
            &[0.0, 1.0, 2.0, 1.0],
            &[1.0, 0.0, 1.0, 2.0],
            &[2.0, 1.0, 0.0, 1.0],
            &[1.0, 2.0, 1.0, 0.0],
        ]);
        let result = classical_mds(&dist, 2);
        assert_eq!(result.coordinates.len(), 4);
        assert_eq!(result.coordinates[0].len(), 2);
        assert!(result.stress >= 0.0);
        assert!(result.stress < 0.5, "stress too high: {}", result.stress);
    }

    #[test]
    fn mds_zero_stress_for_planar() {
        // Points on a line have zero stress in 1D
        let dist = Matrix::from_slice(&[&[0.0, 1.0, 2.0], &[1.0, 0.0, 1.0], &[2.0, 1.0, 0.0]]);
        let result = classical_mds(&dist, 1);
        assert!(
            result.stress < 0.01,
            "stress for linear embedding: {}",
            result.stress
        );
    }

    #[test]
    fn compute_stress_identity() {
        let dist = Matrix::from_slice(&[&[0.0, 1.0, 2.0], &[1.0, 0.0, 1.0], &[2.0, 1.0, 0.0]]);
        let emb = Matrix::from_slice(&[&[0.0], &[1.0], &[2.0]]);
        let stress = compute_stress(&dist, &emb);
        assert!(
            (stress - 0.0).abs() < 1e-10,
            "perfect embedding should have zero stress"
        );
    }

    // ── CCA ──

    #[test]
    fn cca_basic() {
        let n = 50;
        let mut x = Matrix::zeros(n, 3);
        let mut y = Matrix::zeros(n, 2);
        let mut seed: u64 = 77;
        for i in 0..n
        {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s = ((seed >> 11) as f64) / (1u64 << 53) as f64;
            x.data[i] = vec![s, s * 0.5, s * 0.3 + 0.1];
            y.data[i] = vec![s * 0.8 + 0.2, s * 0.6];
        }

        let result = cca_fit(&x, &y, 2);
        assert_eq!(result.correlations.len(), 2);
        // Correlations should be between 0 and 1
        for &r in &result.correlations
        {
            assert!(
                (0.0..=1.0 + 1e-6).contains(&r),
                "correlation out of range: {r}"
            );
        }
        // First correlation should be high (data is linearly related)
        assert!(
            result.correlations[0] > 0.9,
            "expected high first correlation"
        );
    }

    #[test]
    fn cca_transform_produces_output() {
        let n = 30;
        let mut x = Matrix::zeros(n, 2);
        let mut y = Matrix::zeros(n, 2);
        let mut seed: u64 = 42;
        for i in 0..n
        {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s = ((seed >> 11) as f64) / (1u64 << 53) as f64;
            x.data[i] = vec![s, s * 2.0];
            y.data[i] = vec![s * 1.5, s * 0.5];
        }

        let cca = cca_fit(&x, &y, 1);
        let (x_can, y_can) = cca_transform(&x, &y, &cca);
        assert_eq!(x_can.rows, 30);
        assert_eq!(x_can.cols, 1);
        assert_eq!(y_can.rows, 30);
        assert_eq!(y_can.cols, 1);
    }

    // ── SVD ──

    #[test]
    fn svd_basic() {
        let m = Matrix::from_slice(&[&[1.0, 0.0], &[0.0, 2.0]]);
        let (u, s, vt) = svd(&m);
        // Singular values should be [2, 1] (sorted descending)
        assert!((s[0] - 2.0).abs() < 1e-8);
        assert!((s[1] - 1.0).abs() < 1e-8);
        // U * S * V^T should reconstruct M
        let mut s_diag = Matrix::zeros(2, 2);
        s_diag.data[0][0] = s[0];
        s_diag.data[1][1] = s[1];
        let reconstructed = u.mul(&s_diag).mul(&vt);
        for i in 0..2
        {
            for j in 0..2
            {
                assert!(
                    (reconstructed.data[i][j] - m.data[i][j]).abs() < 1e-8,
                    "SVD reconstruction mismatch"
                );
            }
        }
    }

    // ── Edge cases ──

    #[test]
    fn pca_single_column() {
        let data = Matrix::from_slice(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
        let pca = pca_fit(&data);
        assert_eq!(pca.n_components(), 1);
        assert!((pca.explained_variance_ratio[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kmeans_all_same_point() {
        let data = Matrix::from_slice(&[&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]]);
        let result = kmeans_fit(&data, 2, 100, 1e-6);
        assert_eq!(result.k, 2);
        assert!(result.inertia.abs() < 1e-10);
    }

    #[test]
    fn silhouette_single_cluster() {
        let data = Matrix::from_slice(&[&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
        let labels = vec![0, 0, 0];
        let score = silhouette_score(&data, &labels);
        // All in one cluster: silhouette is 0 (undefined but we return 0)
        assert!((score).abs() < 1e-10);
    }

    #[test]
    fn mds_triangle_inequality() {
        // 3 points forming a triangle
        let dist = Matrix::from_slice(&[&[0.0, 3.0, 4.0], &[3.0, 0.0, 5.0], &[4.0, 5.0, 0.0]]);
        let result = classical_mds(&dist, 2);
        assert_eq!(result.coordinates.len(), 3);
        // Stress should be reasonable
        assert!(result.stress < 0.5);
    }
}
