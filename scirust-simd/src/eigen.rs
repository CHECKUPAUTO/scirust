// scirust-simd/src/eigen.rs
//
// # Lanczos — valeurs propres extrêmes d'une matrice symétrique, sans factorisation
//
// [`lanczos_eigen_symmetric`] calcule quelques couples propres **extrêmes**
// (les plus grandes ET les plus petites valeurs propres — elles convergent en
// premier, propriété classique de la méthode) d'une matrice symétrique de
// grande taille, en `O(steps·n²)` (seulement des produits matrice-vecteur)
// plutôt que le `O(n³)` d'une diagonalisation complète
// ([`crate::fixed::linalg::jacobi_eigen`]/[`crate::fixed::linalg::svd`]) —
// utile quand `n` est grand et que seuls quelques couples propres
// intéressent (analyse spectrale, réduction de dimension, etc.).
//
// ## Pourquoi `f32`/`f64` uniquement, pas générique `RealScalar`/virgule fixe
//
// Contrairement au reste de la pile de ce crate (`dsp`, `geometry`,
// `hypercomplex`), ce module n'est **pas** générique sur
// [`crate::fixed::RealScalar`] : il est délibérément restreint à `f32`/`f64`
// natifs via [`LanczosScalar`]. La méthode de Lanczos a besoin d'une
// **réorthogonalisation complète** (Gram-Schmidt contre tous les vecteurs de
// Lanczos précédents, à chaque pas) pour rester stable numériquement — une
// question bien comprise depuis des décennies en arithmétique flottante IEEE
// (Golub & Van Loan), mais dont l'analogue en virgule fixe (l'arrondi
// enveloppant de [`crate::fixed`] compose-t-il correctement la perte
// d'orthogonalité accumulée sur de nombreuses itérations ?) reste une
// question ouverte, non résolue ici. Le support virgule fixe est donc
// **explicitement différé** (travail futur documenté), plutôt que
// silencieusement absent ou approximé sans garantie — [`crate::grad`] et
// [`crate::gemm`] sont déjà, pour d'autres raisons, des modules racine
// `f32`/`f64` concrets sans généricité virgule fixe : ce n'est pas un
// précédent isolé dans ce crate.
//
// ## Algorithme
//
// 1. **Tridiagonalisation de Lanczos** : à partir d'un vecteur de départ
//    **déterministe** `(1,…,1)/√n` (aucune source d'aléa, cf. philosophie du
//    crate), construit une base orthonormée `v₁,…,v_m` (`m = steps`) du
//    sous-espace de Krylov `{v₁, A·v₁, A²·v₁, …}` par récurrence à trois
//    termes, avec **réorthogonalisation complète** contre tous les `vᵢ`
//    précédents à chaque pas (simplicité et robustesse, plutôt qu'une
//    réorthogonalisation sélective/partielle moins coûteuse mais plus
//    délicate à valider). La projection de `A` sur cette base est la
//    matrice tridiagonale `T = tridiag(β, α, β)`.
// 2. **Diagonalisation de `T`** (`m×m`, petite et dense) par rotations de
//    Jacobi ([`jacobi_eigen_dense`], transcription directe de
//    [`crate::fixed::linalg::jacobi_eigen`] en arithmétique flottante
//    native — mêmes formules de rotation, `+`/`-`/`*`/`/`/`sqrt` au lieu de
//    `wrapping_add`/`wrapping_mul`/`checked_div`).
// 3. **Vecteurs de Ritz** : `yₗ = Σᵢ sᵢₗ·vᵢ` (combinaison des vecteurs de
//    Lanczos pondérée par les vecteurs propres `s` de `T`) approxime le
//    vecteur propre de `A` associé à la valeur propre de Ritz `θₗ`.
//
// Rupture chanceuse (`‖w‖` quasi nul avant `steps` pas, cf.
// [`lanczos_eigen_symmetric`]) : le sous-espace de Krylov s'épuise
// exactement — c'est une convergence, pas un échec. Moins de `steps`
// couples sont alors renvoyés.

/// Scalaire flottant supporté par ce module — délibérément restreint à
/// `f32`/`f64` natifs (voir en-tête de module pour la justification).
pub trait LanczosScalar:
    Copy
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    /// Zéro additif.
    fn zero() -> Self;
    /// Un multiplicatif.
    fn one() -> Self;
    /// Petit entier (taille de problème) → scalaire, pour `1/√n`.
    fn from_usize(n: usize) -> Self;
    /// Racine carrée.
    fn sqrt(self) -> Self;
    /// Valeur absolue.
    fn abs(self) -> Self;
}

impl LanczosScalar for f32 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }
}

impl LanczosScalar for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

/// Produit matrice-vecteur `A·x` (`a` : `n×n` row-major).
fn matvec<T: LanczosScalar>(a: &[T], x: &[T], n: usize) -> Vec<T> {
    let mut y = vec![T::zero(); n];
    for i in 0..n
    {
        let mut acc = T::zero();
        for j in 0..n
        {
            acc = acc + a[i * n + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Produit scalaire euclidien.
fn dot<T: LanczosScalar>(a: &[T], b: &[T]) -> T {
    let mut acc = T::zero();
    for i in 0..a.len()
    {
        acc = acc + a[i] * b[i];
    }
    acc
}

/// Norme euclidienne.
fn norm<T: LanczosScalar>(a: &[T]) -> T {
    dot(a, a).sqrt()
}

/// `y += alpha · x` (en place).
fn axpy<T: LanczosScalar>(y: &mut [T], alpha: T, x: &[T]) {
    for i in 0..y.len()
    {
        y[i] = y[i] + alpha * x[i];
    }
}

/// Diagonalise une matrice symétrique dense `m×m` (ici : la matrice
/// tridiagonale de Lanczos) par rotations de Jacobi cycliques — transcription
/// directe de [`crate::fixed::linalg::jacobi_eigen`] en arithmétique
/// flottante native (mêmes formules de rotation) : mécanique, même algorithme
/// déjà validé, faible risque. `eigenvectors` : `m×m` row-major, la colonne
/// `j` est le vecteur propre associé à `eigenvalues[j]` (non triés, comme
/// l'original — le tri est la responsabilité de l'appelant).
fn jacobi_eigen_dense<T: LanczosScalar>(
    a: &[T],
    m: usize,
    tol: T,
    max_sweeps: usize,
) -> (Vec<T>, Vec<T>) {
    let mut mat = a.to_vec();
    let mut v = vec![T::zero(); m * m];
    for i in 0..m
    {
        v[i * m + i] = T::one();
    }

    let mut sweeps = 0usize;
    while sweeps < max_sweeps
    {
        let mut max_off = T::zero();
        for p in 0..m
        {
            for q in (p + 1)..m
            {
                let apq = mat[p * m + q];
                let apq_abs = apq.abs();
                if apq_abs > max_off
                {
                    max_off = apq_abs;
                }
                if apq_abs <= tol
                {
                    continue;
                }

                let app = mat[p * m + p];
                let aqq = mat[q * m + q];
                let two_apq = apq + apq;
                let theta = (aqq - app) / two_apq;
                let sign_theta = if theta < T::zero()
                {
                    T::zero() - T::one()
                }
                else
                {
                    T::one()
                };
                let denom = theta.abs() + (T::one() + theta * theta).sqrt();
                let t = sign_theta / denom;
                let c = T::one() / (T::one() + t * t).sqrt();
                let s = t * c;

                let new_app = app - t * apq;
                let new_aqq = aqq + t * apq;

                for k in 0..m
                {
                    if k == p || k == q
                    {
                        continue;
                    }
                    let akp = mat[k * m + p];
                    let akq = mat[k * m + q];
                    let new_akp = c * akp - s * akq;
                    let new_akq = s * akp + c * akq;
                    mat[k * m + p] = new_akp;
                    mat[p * m + k] = new_akp;
                    mat[k * m + q] = new_akq;
                    mat[q * m + k] = new_akq;
                }
                mat[p * m + p] = new_app;
                mat[q * m + q] = new_aqq;
                mat[p * m + q] = T::zero();
                mat[q * m + p] = T::zero();

                for k in 0..m
                {
                    let vkp = v[k * m + p];
                    let vkq = v[k * m + q];
                    v[k * m + p] = c * vkp - s * vkq;
                    v[k * m + q] = s * vkp + c * vkq;
                }
            }
        }
        sweeps += 1;
        if max_off <= tol
        {
            break;
        }
    }

    let eigenvalues: Vec<T> = (0..m).map(|i| mat[i * m + i]).collect();
    (eigenvalues, v)
}

/// Calcule des couples propres (valeur, vecteur) d'une matrice symétrique `a`
/// (`n×n` row-major) par tridiagonalisation de Lanczos à réorthogonalisation
/// complète, sans factorisation directe — utile quand seuls les quelques
/// couples propres extrêmes intéressent, pour un `n` où une diagonalisation
/// complète serait disproportionnée (cf. en-tête de module).
///
/// `steps` pas de Lanczos (`1 ≤ steps ≤ n`) construisent une matrice
/// tridiagonale `m×m` (`m = steps`, moins en cas de rupture chanceuse —
/// cf. en-tête de module), diagonalisée par rotations de Jacobi (`tol`/
/// `max_sweeps`, mêmes paramètres que [`crate::fixed::linalg::jacobi_eigen`]).
///
/// Renvoie les couples **triés par valeur propre décroissante** : les
/// premiers/derniers éléments sont les approximations les plus fiables
/// (respectivement les plus grandes et les plus petites valeurs propres de
/// `a`, qui convergent en premier dans l'itération de Lanczos) ; les
/// éléments du milieu, si `steps` est petit devant `n`, sont typiquement peu
/// fiables et à ignorer.
///
/// Vecteur de départ déterministe `(1,…,1)/√n` (cf. en-tête de module).
///
/// Panique si `a.len() != n·n`, ou si `steps == 0` ou `steps > n`.
#[must_use]
pub fn lanczos_eigen_symmetric<T: LanczosScalar>(
    a: &[T],
    n: usize,
    steps: usize,
    tol: T,
    max_sweeps: usize,
) -> (Vec<T>, Vec<Vec<T>>) {
    assert_eq!(
        a.len(),
        n * n,
        "lanczos_eigen_symmetric : A de longueur {} ≠ {n}×{n}",
        a.len()
    );
    assert!(
        steps >= 1 && steps <= n,
        "lanczos_eigen_symmetric : steps doit être dans [1, {n}]"
    );

    let mut vecs: Vec<Vec<T>> = Vec::with_capacity(steps);
    let mut alpha: Vec<T> = Vec::with_capacity(steps);
    let mut beta: Vec<T> = Vec::with_capacity(steps.saturating_sub(1));

    let inv_sqrt_n = T::one() / T::from_usize(n).sqrt();
    vecs.push(vec![inv_sqrt_n; n]);

    for j in 0..steps
    {
        let mut w = matvec(a, &vecs[j], n);
        let a_j = dot(&w, &vecs[j]);
        alpha.push(a_j);
        axpy(&mut w, T::zero() - a_j, &vecs[j]);
        if j > 0
        {
            let b_j = beta[j - 1];
            axpy(&mut w, T::zero() - b_j, &vecs[j - 1]);
        }
        // Réorthogonalisation complète contre tous les vᵢ déjà construits.
        for vi in &vecs
        {
            let proj = dot(&w, vi);
            axpy(&mut w, T::zero() - proj, vi);
        }

        if j + 1 < steps
        {
            let beta_next = norm(&w);
            if beta_next <= tol
            {
                break; // rupture chanceuse : sous-espace de Krylov épuisé.
            }
            let inv = T::one() / beta_next;
            vecs.push(w.iter().map(|&x| x * inv).collect());
            beta.push(beta_next);
        }
    }

    let m = alpha.len();
    let mut t = vec![T::zero(); m * m];
    for i in 0..m
    {
        t[i * m + i] = alpha[i];
        if i + 1 < m
        {
            t[i * m + i + 1] = beta[i];
            t[(i + 1) * m + i] = beta[i];
        }
    }
    let (theta, s) = jacobi_eigen_dense(&t, m, tol, max_sweeps);

    // Vecteurs de Ritz yₗ = Σᵢ sᵢₗ·vᵢ (colonne l de s).
    let mut ritz_vectors: Vec<Vec<T>> = Vec::with_capacity(m);
    for l in 0..m
    {
        let mut y = vec![T::zero(); n];
        for (i, vi) in vecs.iter().enumerate()
        {
            axpy(&mut y, s[i * m + l], vi);
        }
        ritz_vectors.push(y);
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| {
        theta[j]
            .partial_cmp(&theta[i])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let eigenvalues: Vec<T> = order.iter().map(|&i| theta[i]).collect();
    let eigenvectors: Vec<Vec<T>> = order.iter().map(|&i| ritz_vectors[i].clone()).collect();
    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests;
