// scirust-simd/src/fixed/linalg.rs
//
// # Algèbre linéaire virgule fixe — GEMM et décompositions déterministes
//
// Produit matrice-matrice (`matmul`, et sa variante `matmul_bt` quand `Bᵀ`
// est déjà disponible) et matrice-vecteur (`matvec`) sur des matrices
// **denses row-major** de scalaires virgule fixe, construits sur le produit
// scalaire SIMD [`super::reductions::dot`]. C'est le socle d'une inférence
// **quantifiée** (réseaux de neurones à poids/activations entiers) : la
// virgule fixe symétrique `Fixed<I, FRAC>` est exactement une quantification
// symétrique d'échelle `2^-FRAC` et de zéro nul. [`super::layer::Linear`]
// s'appuie sur `matmul_bt` pour l'inférence **par lot**
// ([`super::layer::Linear::forward_batch`]) : la matrice de poids `out × in`
// est déjà dans la disposition `Bᵀ` attendue, aucune transposition requise.
//
// Le module fournit aussi les décompositions directes classiques pour
// résoudre `A·x = b` : [`cholesky`]/[`cholesky_solve`] (matrices symétriques
// définies positives), [`lu_decompose`]/[`lu_solve`] (matrices générales,
// pivot partiel) plus [`determinant`], et [`qr_decompose`]/[`qr_solve`]
// (moindres carrés, systèmes surdéterminés). Toutes reposent sur les mêmes
// primitives déterministes (`dot`, division réelle vérifiée) que le GEMM.
//
// [`jacobi_eigen`] complète ces décompositions directes par la **décomposition
// spectrale** d'une matrice symétrique (`A = V·diag(λ)·Vᵀ`, méthode de Jacobi
// à rotations cycliques) : contrairement aux trois précédentes, c'est un
// algorithme **itératif** (convergence à une tolérance près, pas une formule
// fermée), mais dont chaque rotation ne demande que racine carrée et division
// réelle — aucune transcendante (`atan`/`sin`/`cos`) — donc généralisable à
// **tout stockage** [`FixedReducible`], `i32` **et** `i64`.
//
// [`svd`] (et [`svd_solve`]) construit la **décomposition en valeurs
// singulières** de toute matrice `m × n` par-dessus `jacobi_eigen` (appliqué à
// `AᵀA`) : `svd_solve` complète `qr_solve` pour les systèmes de **rang
// déficient**, où `qr_solve` échoue (pivot nul) — la pseudo-inverse de
// Moore-Penrose donne la solution de norme minimale au lieu d'abandonner.
//
// [`hessenberg`] et [`eigenvalues_general`] complètent [`jacobi_eigen`] pour
// les matrices **non symétriques** : `jacobi_eigen`/`svd` supposent toutes
// deux une matrice symétrique (ou `AᵀA`, toujours symétrique semi-définie
// positive) — une matrice quelconque (jacobienne d'un système dynamique,
// matrice de transition non réversible…) peut avoir des valeurs propres
// **complexes conjuguées**, qu'aucune des deux ne peut représenter.
// `eigenvalues_general` réduit d'abord `A` à la forme de Hessenberg
// supérieure (similarité orthogonale, [`hessenberg`]), puis itère un
// algorithme QR à décalage (décalage de Wilkinson, quelques itérations
// « ad hoc » de secours toutes les 10 itérations sans progrès — technique
// classique EISPACK/Numerical Recipes) avec déflation, jusqu'à isoler
// chaque valeur propre (bloc `1×1`) ou paire complexe conjuguée (bloc `2×2`
// final, résolu analytiquement par la formule quadratique sur trace/déterminant
// — **aucune arithmétique complexe** n'est nécessaire, seul [`Eigenvalue`]
// distingue les deux cas en sortie).
//
// `qr_solve` complète plutôt qu'il ne duplique Cholesky : résoudre les
// moindres carrés via les équations normales (`cholesky_solve` sur `AᵀA`)
// **double** l'exposant de conditionnement du problème (`cond(AᵀA) =
// cond(A)²`), alors que QR opère directement sur `A` — la voie standard
// lorsque `A` est mal conditionnée.
//
// ## Reproductibilité bit-à-bit (point clé)
//
// Chaque coefficient de sortie est un produit scalaire `Σ aᵢ·bᵢ` où **chaque
// produit** est arrondi vers zéro (comme l'opérateur `*`) puis **sommé
// exactement** (accumulateur élargi, cf. [`super::reductions`]). L'addition
// virgule fixe étant exacte et associative, le résultat est **indépendant** de
// l'ordre de parcours, du nombre de lanes SIMD, de l'architecture et du nombre
// de threads. Un GEMM virgule fixe donne donc le **même bit** partout — ce que
// le GEMM flottant ne garantit jamais (somme non associative).
//
// En particulier, `matmul` coïncide **exactement** avec la triple boucle naïve
// `c += a·b` (produits et sommes enveloppants) : sommer puis rétrécir une seule
// fois équivaut à des additions enveloppantes progressives (`mod 2^BITS`).
//
// Les décompositions héritent de cette reproductibilité : `cholesky` et
// `lu_decompose` n'effectuent que des additions/multiplications exactes
// (via `dot`) et des divisions réelles **vérifiées** ([`FixedReducible::checked_div`],
// jamais `x * y.recip()` — cf. la leçon de précision du module `dsp::mel`, une
// réciproque isolée d'un dénominateur non-puissance-de-deux perd de la
// précision avant même la multiplication). Le pivotage partiel de `lu_decompose`
// compare des valeurs absolues via l'ordre entier total ([`Ord`]) — exact,
// donc **sans aucune ambiguïté de bord** contrairement à un pivotage flottant
// où deux magnitudes proches d'une erreur d'arrondi pourraient inverser l'ordre
// différemment selon la plateforme.
//
// ## Disposition mémoire
//
// Une matrice `m × n` est un `&[T]` de longueur `m·n` en **row-major** : la
// ligne `i`, colonne `j`, est à l'indice `i·n + j`. `matmul` transpose la
// matrice de droite une fois (`O(k·n)`) pour rendre chaque produit scalaire
// **contigu** (donc vectorisable), coût négligeable devant le `O(m·k·n)` global.
//
// `lu_decompose` combine `L` et `U` dans une seule matrice `n × n` (convention
// LAPACK `*getrf`) : la diagonale unité de `L` est **implicite** (jamais
// stockée, jamais lue) — la partie strictement inférieure du buffer est `L`,
// la partie triangulaire supérieure (diagonale incluse) est `U`.
//
// `qr_decompose` renvoie la forme **réduite** (« thin QR ») : `A` est `m × n`
// (`m ≥ n`), `Q` est `m × n` à colonnes orthonormées, `R` est `n × n`
// triangulaire supérieure — la partie « inutile » de la forme complète (les
// `m − n` dernières lignes, nulles par construction) n'est jamais matérialisée.
//
// ## Panique
//
// Les fonctions **paniquent** sur incohérence dimensionnelle (longueur de slice
// ≠ produit des dimensions annoncées, ou `m < n` pour QR) — c'est un bug
// d'appelant, pas une donnée. En revanche, une matrice non définie positive
// (`cholesky`) ou singulière (`lu_decompose`, `qr_solve` via un pivot nul de
// `R`) renvoie `None` : c'est une propriété des **données**, pas une erreur
// d'appel. `qr_decompose` elle-même existe pour **toute** matrice (aucune
// condition d'inversibilité) — son `None` ne survient qu'en cas de
// débordement virgule fixe pendant une réflexion.

use core::ops::Sub;

use super::reductions::{FixedReducible, dot};

/// Transpose une matrice dense row-major `rows × cols` en `cols × rows`.
///
/// `t[j·rows + i] = a[i·cols + j]`. Panique si `a.len() != rows·cols`.
#[must_use]
pub fn transpose<T: Copy>(a: &[T], rows: usize, cols: usize) -> Vec<T> {
    assert_eq!(
        a.len(),
        rows * cols,
        "transpose : slice de longueur {} incompatible avec {rows}×{cols}",
        a.len()
    );
    let mut t = Vec::with_capacity(a.len());
    for j in 0..cols
    {
        for i in 0..rows
        {
            t.push(a[i * cols + j]);
        }
    }
    t
}

/// Produit matrice-matrice `C = A · B` (déterministe, virgule fixe).
///
/// `a` est `m × k`, `b` est `k × n`, le résultat `C` est `m × n`, tous
/// row-major. Chaque `C[i, j] = Σₗ A[i, l]·B[l, j]` (produits arrondis vers
/// zéro, somme exacte). Panique si `a.len() != m·k` ou `b.len() != k·n`.
///
/// La matrice de droite est transposée une fois pour rendre chaque produit
/// scalaire contigu et **vectorisé** ([`super::reductions::dot`]) — voir
/// [`matmul_bt`] si l'appelant dispose déjà de `Bᵀ` (évite cette transposition).
#[must_use]
pub fn matmul<T: FixedReducible>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    assert_eq!(
        a.len(),
        m * k,
        "matmul : A de longueur {} ≠ {m}×{k}",
        a.len()
    );
    assert_eq!(
        b.len(),
        k * n,
        "matmul : B de longueur {} ≠ {k}×{n}",
        b.len()
    );
    // Bᵀ (n × k) : la ligne j de Bᵀ est la colonne j de B, désormais contiguë.
    let bt = transpose(b, k, n);
    matmul_bt(a, &bt, m, k, n)
}

/// Produit matrice-matrice `C = A · Bᵀ` (déterministe, virgule fixe), `Bᵀ`
/// étant fourni **déjà transposé** (`bt`, `n × k` row-major : la ligne `j` de
/// `bt` est la colonne `j` de `B`).
///
/// `matmul_bt(a, bt, m, k, n) == matmul(a, transpose(bt, n, k), m, k, n)`,
/// sans le coût `O(k·n)` de cette transposition — utile quand l'appelant
/// dispose déjà de `Bᵀ`, ce qui est le cas typique d'une matrice de poids de
/// réseau de neurones stockée `out × in` (exactement la forme `Bᵀ` attendue
/// pour calculer `X · Wᵀ`, cf. [`super::layer::Linear::forward_batch`]).
/// Panique si `a.len() != m·k` ou `bt.len() != n·k`.
#[must_use]
pub fn matmul_bt<T: FixedReducible>(a: &[T], bt: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    assert_eq!(
        a.len(),
        m * k,
        "matmul_bt : A de longueur {} ≠ {m}×{k}",
        a.len()
    );
    assert_eq!(
        bt.len(),
        n * k,
        "matmul_bt : Bᵀ de longueur {} ≠ {n}×{k}",
        bt.len()
    );
    let mut c = Vec::with_capacity(m * n);
    for i in 0..m
    {
        let a_row = &a[i * k..i * k + k];
        for j in 0..n
        {
            let bt_row = &bt[j * k..j * k + k];
            c.push(dot(a_row, bt_row));
        }
    }
    c
}

/// Produit matrice-vecteur `y = A · x` (déterministe, virgule fixe).
///
/// `a` est `m × k` row-major, `x` a `k` éléments, le résultat `y` a `m`
/// éléments. `y[i] = Σₗ A[i, l]·x[l]`. Panique si `a.len() != m·k` ou
/// `x.len() != k`.
///
/// Chaque ligne de `A` est déjà contiguë : aucune transposition, produit
/// scalaire directement vectorisé.
#[must_use]
pub fn matvec<T: FixedReducible>(a: &[T], x: &[T], m: usize, k: usize) -> Vec<T> {
    assert_eq!(
        a.len(),
        m * k,
        "matvec : A de longueur {} ≠ {m}×{k}",
        a.len()
    );
    assert_eq!(x.len(), k, "matvec : x de longueur {} ≠ {k}", x.len());
    let mut y = Vec::with_capacity(m);
    for i in 0..m
    {
        y.push(dot(&a[i * k..i * k + k], x));
    }
    y
}

// ------------------------------------------------------------------ //
//  Substitutions triangulaires                                        //
// ------------------------------------------------------------------ //

/// Résout `L·y = b` par substitution avant, `L` étant `n × n` **triangulaire
/// inférieure** row-major (diagonale incluse et lue). `None` si un pivot
/// diagonal est nul (système singulier). Panique si `l.len() != n·n` ou
/// `b.len() != n`.
///
/// Seule la partie triangulaire inférieure de `l` (diagonale incluse) est
/// lue ; la partie strictement supérieure est ignorée.
#[must_use]
pub fn forward_substitution<T>(l: &[T], b: &[T], n: usize) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(
        l.len(),
        n * n,
        "forward_substitution : L de longueur {} ≠ {n}×{n}",
        l.len()
    );
    assert_eq!(
        b.len(),
        n,
        "forward_substitution : b de longueur {} ≠ {n}",
        b.len()
    );
    let mut y = Vec::with_capacity(n);
    for i in 0..n
    {
        let s = dot(&l[i * n..i * n + i], &y[..i]);
        y.push((b[i] - s).checked_div(l[i * n + i])?);
    }
    Some(y)
}

/// Résout `U·x = y` par substitution arrière, `U` étant `n × n` **triangulaire
/// supérieure** row-major (diagonale incluse et lue). `None` si un pivot
/// diagonal est nul (système singulier). Panique si `u.len() != n·n` ou
/// `y.len() != n`.
///
/// Seule la partie triangulaire supérieure de `u` (diagonale incluse) est
/// lue ; la partie strictement inférieure est ignorée.
#[must_use]
pub fn back_substitution<T>(u: &[T], y: &[T], n: usize) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(
        u.len(),
        n * n,
        "back_substitution : U de longueur {} ≠ {n}×{n}",
        u.len()
    );
    assert_eq!(
        y.len(),
        n,
        "back_substitution : y de longueur {} ≠ {n}",
        y.len()
    );
    let mut x = vec![T::ZERO; n];
    for i in (0..n).rev()
    {
        let s = dot(&u[i * n + i + 1..i * n + n], &x[i + 1..n]);
        x[i] = (y[i] - s).checked_div(u[i * n + i])?;
    }
    Some(x)
}

// ------------------------------------------------------------------ //
//  Cholesky (matrices symétriques définies positives)                 //
// ------------------------------------------------------------------ //

/// Décomposition de Cholesky `A = L·Lᵀ` (méthode de Cholesky–Banachiewicz,
/// ligne par ligne). `a` est `n × n` row-major, symétrique définie positive ;
/// seule sa partie triangulaire **inférieure** (diagonale incluse) est lue —
/// la partie supérieure est ignorée, comme les routines LAPACK `?potrf`.
///
/// Renvoie `L` (`n × n` row-major, triangulaire inférieure, zéros au-dessus
/// de la diagonale), ou `None` si un pivot diagonal calculé est `≤ 0` (`A`
/// n'est pas définie positive, ou l'est trop faiblement pour la résolution
/// de `T`). Panique si `a.len() != n·n`.
///
/// Chaque coefficient hors diagonale utilise une division **réelle vérifiée**
/// ([`FixedReducible::checked_div`]), jamais une réciproque isolée — cf. la
/// leçon de précision du module `dsp::mel` pour les dénominateurs non
/// puissance de deux.
#[must_use]
pub fn cholesky<T>(a: &[T], n: usize) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(
        a.len(),
        n * n,
        "cholesky : A de longueur {} ≠ {n}×{n}",
        a.len()
    );
    let mut l = vec![T::ZERO; n * n];
    for i in 0..n
    {
        for j in 0..=i
        {
            let s = dot(&l[i * n..i * n + j], &l[j * n..j * n + j]);
            let d = a[i * n + j] - s;
            if i == j
            {
                if d <= T::ZERO
                {
                    return None;
                }
                l[i * n + i] = d.sqrt();
            }
            else
            {
                l[i * n + j] = d.checked_div(l[j * n + j])?;
            }
        }
    }
    Some(l)
}

/// Résout `A·x = b` pour `A` symétrique définie positive, via Cholesky puis
/// deux substitutions triangulaires (`L·y = b`, puis `Lᵀ·x = y`). `None` si
/// `A` n'est pas définie positive (cf. [`cholesky`]). Panique si
/// `a.len() != n·n` ou `b.len() != n`.
#[must_use]
pub fn cholesky_solve<T>(a: &[T], b: &[T], n: usize) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    let l = cholesky(a, n)?;
    let y = forward_substitution(&l, b, n)?;
    let lt = transpose(&l, n, n);
    back_substitution(&lt, &y, n)
}

// ------------------------------------------------------------------ //
//  LU à pivot partiel (matrices générales)                            //
// ------------------------------------------------------------------ //

/// Décomposition LU à pivot partiel `P·A = L·U` (méthode de Doolittle). `a`
/// est `n × n` row-major, quelconque (pas nécessairement symétrique).
///
/// Renvoie `(lu, perm)` : `lu` est `n × n` row-major, combinant `L` et `U`
/// (convention LAPACK `?getrf`) — sa partie strictement inférieure est `L`
/// (diagonale unité **implicite**, non stockée), sa partie triangulaire
/// supérieure (diagonale incluse) est `U`. `perm[i]` est l'indice, dans `A`,
/// de la ligne placée en position `i` après pivotage : la matrice `A` dont
/// les lignes sont réordonnées selon `perm` égale exactement `L·U`.
///
/// `None` si `A` est singulière (pivot nul rencontré, à une précision de
/// résolution de `T` près). Panique si `a.len() != n·n`.
///
/// Le pivot de chaque étape est la ligne de plus grande valeur absolue dans
/// la colonne courante (comparaison exacte, [`Ord`] entier) — stabilise
/// l'élimination sans jamais introduire d'ambiguïté d'arrondi.
#[must_use]
pub fn lu_decompose<T>(a: &[T], n: usize) -> Option<(Vec<T>, Vec<usize>)>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(
        a.len(),
        n * n,
        "lu_decompose : A de longueur {} ≠ {n}×{n}",
        a.len()
    );
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n
    {
        let mut pivot_row = k;
        let mut pivot_val = lu[k * n + k].abs();
        for i in (k + 1)..n
        {
            let v = lu[i * n + k].abs();
            if v > pivot_val
            {
                pivot_val = v;
                pivot_row = i;
            }
        }
        if pivot_val == T::ZERO
        {
            return None;
        }
        if pivot_row != k
        {
            for j in 0..n
            {
                lu.swap(k * n + j, pivot_row * n + j);
            }
            perm.swap(k, pivot_row);
        }
        for i in (k + 1)..n
        {
            let factor = lu[i * n + k].checked_div(lu[k * n + k])?;
            lu[i * n + k] = factor;
            for j in (k + 1)..n
            {
                lu[i * n + j] = lu[i * n + j] - factor.wrapping_mul(lu[k * n + j]);
            }
        }
    }
    Some((lu, perm))
}

/// Résout `A·x = b` pour `A` quelconque, via LU à pivot partiel. `None` si
/// `A` est singulière (cf. [`lu_decompose`]). Panique si `a.len() != n·n` ou
/// `b.len() != n`.
#[must_use]
pub fn lu_solve<T>(a: &[T], b: &[T], n: usize) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(b.len(), n, "lu_solve : b de longueur {} ≠ {n}", b.len());
    let (lu, perm) = lu_decompose(a, n)?;
    // L a une diagonale unité implicite : la substitution avant ne divise pas.
    let mut y = Vec::with_capacity(n);
    for i in 0..n
    {
        let s = dot(&lu[i * n..i * n + i], &y[..i]);
        y.push(b[perm[i]] - s);
    }
    back_substitution(&lu, &y, n)
}

/// Déterminant de `A` (`n × n`, `n ≥ 1`) via LU à pivot partiel :
/// `det(A) = (−1)ˢ · Πᵢ U[i, i]`, où `s` est le nombre de transpositions de
/// lignes effectuées par le pivotage. Renvoie `T::ZERO` si `A` est singulière.
/// Panique si `n == 0` ou `a.len() != n·n`.
#[must_use]
pub fn determinant<T>(a: &[T], n: usize) -> T
where
    T: FixedReducible + Sub<Output = T>,
{
    assert!(n >= 1, "determinant : dimension nulle non supportée");
    match lu_decompose(a, n)
    {
        None => T::ZERO,
        Some((lu, perm)) =>
        {
            let mut det = lu[0];
            for i in 1..n
            {
                det = det.wrapping_mul(lu[i * n + i]);
            }
            if permutation_is_odd(&perm)
            {
                T::ZERO - det
            }
            else
            {
                det
            }
        },
    }
}

/// Parité de `perm` : `true` si sa décomposition en cycles requiert un nombre
/// **impair** de transpositions (`n − nombre_de_cycles`).
fn permutation_is_odd(perm: &[usize]) -> bool {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut swaps = 0usize;
    for start in 0..n
    {
        if visited[start]
        {
            continue;
        }
        let mut j = start;
        let mut cycle_len = 0usize;
        while !visited[j]
        {
            visited[j] = true;
            j = perm[j];
            cycle_len += 1;
        }
        swaps += cycle_len - 1;
    }
    swaps % 2 == 1
}

// ------------------------------------------------------------------ //
//  QR (Householder) — moindres carrés pour systèmes surdéterminés     //
// ------------------------------------------------------------------ //

/// Décomposition QR par réflexions de Householder, forme **réduite**
/// (« thin QR ») : `A = Q·R`, `A` étant `m × n` (`m ≥ n`) row-major, `Q`
/// `m × n` à colonnes orthonormées, `R` `n × n` triangulaire supérieure.
///
/// Contrairement à Cholesky/LU, la décomposition QR existe pour **toute**
/// matrice (aucune condition de définie-positivité ou d'inversibilité) : si
/// `A` est de rang déficient, un pivot diagonal de `R` sera nul — visible via
/// [`back_substitution`]/[`qr_solve`], qui renverront alors `None`. Le `None`
/// de cette fonction ne survient donc qu'en cas de débordement virgule fixe
/// pendant une réflexion (résolution insuffisante de `T` pour l'échelle du
/// problème). Panique si `m < n` ou `a.len() != m·n`.
///
/// Pour chaque colonne `k`, le vecteur de Householder utilise la convention
/// de signe usuelle `α = −sign(x₀)·‖x‖` (`x` la sous-colonne à annuler), qui
/// évite l'annulation catastrophique dans `v = x − α·e₁` — pertinent même en
/// virgule fixe, où l'annulation dégrade la précision **relative** restante
/// pour la suite du calcul (même principe que la division réelle plutôt que
/// la réciproque isolée, cf. en-tête de module).
#[must_use]
pub fn qr_decompose<T>(a: &[T], m: usize, n: usize) -> Option<(Vec<T>, Vec<T>)>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert!(m >= n, "qr_decompose : m ({m}) doit être ≥ n ({n})");
    assert_eq!(
        a.len(),
        m * n,
        "qr_decompose : A de longueur {} ≠ {m}×{n}",
        a.len()
    );

    let mut r = a.to_vec();
    // vs[(k+i)·n + k] = v_k[i] pour i in 0..(m−k) ; nul <=> réflexion k = identité.
    let mut vs = vec![T::ZERO; m * n];

    for k in 0..n
    {
        let len = m - k;
        let x: Vec<T> = (0..len).map(|i| r[(k + i) * n + k]).collect();
        let norm_x = dot(&x, &x).sqrt();
        if norm_x == T::ZERO
        {
            continue; // sous-colonne déjà nulle : réflexion = identité.
        }
        let alpha = if x[0] < T::ZERO
        {
            norm_x
        }
        else
        {
            T::ZERO - norm_x
        };
        let mut v = x;
        v[0] = v[0] - alpha;
        let vtv = dot(&v, &v);
        if vtv == T::ZERO
        {
            continue; // x déjà aligné (bon signe) : réflexion = identité.
        }
        for j in k..n
        {
            let col: Vec<T> = (0..len).map(|i| r[(k + i) * n + j]).collect();
            let vtcol = dot(&v, &col);
            let scale = vtcol.wrapping_add(vtcol).checked_div(vtv)?;
            for i in 0..len
            {
                r[(k + i) * n + j] = col[i] - v[i].wrapping_mul(scale);
            }
        }
        // Rangs k+1..m de la colonne k : nuls par construction analytique —
        // impose l'exactitude triangulaire plutôt que de garder un résidu
        // d'arrondi flottant-fixe sans signification.
        for i in 1..len
        {
            r[(k + i) * n + k] = T::ZERO;
        }
        for (i, &vi) in v.iter().enumerate()
        {
            vs[(k + i) * n + k] = vi;
        }
    }
    r.truncate(n * n); // ne garde que les n premières lignes (les m−n dernières sont nulles).

    // Q réduit : part de [Iₙ ; 0] (m×n) et applique les réflexions dans
    // l'ordre inverse (Q = H₀·H₁·⋯·H_{n−1}, cf. en-tête de module).
    let mut q = vec![T::ZERO; m * n];
    for i in 0..n
    {
        q[i * n + i] = T::one();
    }
    for k in (0..n).rev()
    {
        let len = m - k;
        let v: Vec<T> = (0..len).map(|i| vs[(k + i) * n + k]).collect();
        let vtv = dot(&v, &v);
        if vtv == T::ZERO
        {
            continue; // réflexion k était l'identité (cf. boucle ci-dessus).
        }
        for j in 0..n
        {
            let col: Vec<T> = (0..len).map(|i| q[(k + i) * n + j]).collect();
            let vtcol = dot(&v, &col);
            let scale = vtcol.wrapping_add(vtcol).checked_div(vtv)?;
            for i in 0..len
            {
                q[(k + i) * n + j] = col[i] - v[i].wrapping_mul(scale);
            }
        }
    }

    Some((q, r))
}

/// Résout le problème des moindres carrés `min_x ‖A·x − b‖` (`A` `m × n`,
/// `m ≥ n`, `b` de longueur `m`) via QR : `x = R⁻¹·(Qᵀ·b)`. Si `m = n` et `A`
/// est inversible, c'est la solution exacte du système carré (à comparer à
/// [`lu_solve`] sur le même système).
///
/// `None` si un pivot de `R` est nul (rang déficient) ou en cas de
/// débordement (cf. [`qr_decompose`]). Panique si `m < n`, `a.len() != m·n`
/// ou `b.len() != m`.
#[must_use]
pub fn qr_solve<T>(a: &[T], b: &[T], m: usize, n: usize) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(b.len(), m, "qr_solve : b de longueur {} ≠ {m}", b.len());
    let (q, r) = qr_decompose(a, m, n)?;
    let mut qtb = Vec::with_capacity(n);
    for j in 0..n
    {
        let col: Vec<T> = (0..m).map(|i| q[i * n + j]).collect();
        qtb.push(dot(&col, b));
    }
    back_substitution(&r, &qtb, n)
}

// ------------------------------------------------------------------ //
//  Jacobi (matrices symétriques) — décomposition spectrale            //
// ------------------------------------------------------------------ //

/// Décomposition spectrale d'une matrice symétrique par la méthode de Jacobi
/// à rotations cycliques (Golub & Van Loan, §8.4) : `A = V·diag(λ)·Vᵀ`.
///
/// `a` est `n × n` row-major, symétrique — seule sa partie triangulaire
/// **inférieure** (diagonale incluse) est lue, comme [`cholesky`]. Renvoie
/// `(eigenvalues, eigenvectors, sweeps)` :
///
/// * `eigenvalues` : les `n` valeurs propres, **non triées** (ordre issu de
///   la convergence, déterministe mais sans signification particulière —
///   `T: Ord` permet à l'appelant de les trier si besoin).
/// * `eigenvectors` : `n × n` row-major, la **colonne** `j` est le vecteur
///   propre unitaire associé à `eigenvalues[j]`.
/// * `sweeps` : nombre de passes cycliques effectuées (`< max_sweeps` si
///   convergé avant la limite, `== max_sweeps` sinon).
///
/// Chaque rotation annule un coefficient hors diagonale `a[p,q]` via les
/// formules algébriques classiques (racine carrée et division réelle
/// vérifiée uniquement — **aucune transcendante**, contrairement à une
/// formulation par angle explicite `atan`/`sin`/`cos`) : fonctionne donc pour
/// **tout stockage** ([`FixedReducible`], `i32` **et** `i64`), comme
/// [`cholesky`]/[`lu_decompose`]/[`qr_decompose`].
///
/// `tol` est le seuil (valeur absolue) en dessous duquel un coefficient hors
/// diagonale est jugé négligeable — la convergence exacte à zéro n'existe pas
/// en arithmétique finie. `max_sweeps` borne le nombre de passes (garantit la
/// terminaison ; la convergence est quadratique une fois amorcée, quelques
/// passes suffisent typiquement en pratique).
///
/// `None` en cas de débordement virgule fixe pendant une rotation (résolution
/// insuffisante de `T` pour l'échelle du problème — même caveat que
/// [`qr_decompose`]). Panique si `a.len() != n·n`.
#[must_use]
pub fn jacobi_eigen<T>(
    a: &[T],
    n: usize,
    tol: T,
    max_sweeps: usize,
) -> Option<(Vec<T>, Vec<T>, usize)>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(
        a.len(),
        n * n,
        "jacobi_eigen : A de longueur {} ≠ {n}×{n}",
        a.len()
    );
    let mut m = a.to_vec();
    // Symétrise explicitement depuis la partie inférieure (comme `cholesky`,
    // la partie supérieure fournie n'est jamais lue).
    for i in 0..n
    {
        for j in 0..i
        {
            m[j * n + i] = m[i * n + j];
        }
    }
    let mut v = vec![T::ZERO; n * n];
    for i in 0..n
    {
        v[i * n + i] = T::one();
    }

    let mut sweeps = 0usize;
    while sweeps < max_sweeps
    {
        let mut max_off = T::ZERO;
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                let apq = m[p * n + q];
                let apq_abs = apq.abs();
                if apq_abs > max_off
                {
                    max_off = apq_abs;
                }
                if apq_abs <= tol
                {
                    continue;
                }

                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let two_apq = apq.wrapping_add(apq);
                let theta = (aqq - app).checked_div(two_apq)?;
                let sign_theta = if theta < T::ZERO
                {
                    T::ZERO - T::one()
                }
                else
                {
                    T::one()
                };
                let denom = theta
                    .abs()
                    .wrapping_add(T::one().wrapping_add(theta.wrapping_mul(theta)).sqrt());
                let t = sign_theta.checked_div(denom)?;
                let c = T::one().checked_div(T::one().wrapping_add(t.wrapping_mul(t)).sqrt())?;
                let s = t.wrapping_mul(c);

                let new_app = app - t.wrapping_mul(apq);
                let new_aqq = aqq.wrapping_add(t.wrapping_mul(apq));

                for k in 0..n
                {
                    if k == p || k == q
                    {
                        continue;
                    }
                    let akp = m[k * n + p];
                    let akq = m[k * n + q];
                    let new_akp = c.wrapping_mul(akp) - s.wrapping_mul(akq);
                    let new_akq = s.wrapping_mul(akp).wrapping_add(c.wrapping_mul(akq));
                    m[k * n + p] = new_akp;
                    m[p * n + k] = new_akp;
                    m[k * n + q] = new_akq;
                    m[q * n + k] = new_akq;
                }
                m[p * n + p] = new_app;
                m[q * n + q] = new_aqq;
                m[p * n + q] = T::ZERO;
                m[q * n + p] = T::ZERO;

                for k in 0..n
                {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c.wrapping_mul(vkp) - s.wrapping_mul(vkq);
                    v[k * n + q] = s.wrapping_mul(vkp).wrapping_add(c.wrapping_mul(vkq));
                }
            }
        }
        sweeps += 1;
        if max_off <= tol
        {
            break;
        }
    }

    let eigenvalues: Vec<T> = (0..n).map(|i| m[i * n + i]).collect();
    Some((eigenvalues, v, sweeps))
}

// ------------------------------------------------------------------ //
//  SVD (via Jacobi sur AᵀA) — décomposition en valeurs singulières    //
// ------------------------------------------------------------------ //

/// `(u, sigma, vt, sweeps)` renvoyé par [`svd`] — cf. sa documentation.
pub type SvdResult<T> = (Vec<T>, Vec<T>, Vec<T>, usize);

/// Décomposition en valeurs singulières **réduite** (« thin SVD ») :
/// `A = U·diag(σ)·Vᵀ`, `A` étant `m × n` (`m ≥ n`) row-major.
///
/// Calculée via [`jacobi_eigen`] sur `AᵀA` (`n × n`, symétrique semi-définie
/// positive) : ses valeurs propres sont les `σᵢ²`, ses vecteurs propres les
/// colonnes de `V`. `U = A·V·diag(σ)⁻¹` (colonne nulle pour un `σᵢ` négligeable
/// — direction non déterminée par les données, cf. [`svd_solve`], qui en fait
/// l'usage pratique). Hérite de la généricité `i32`/`i64` de `jacobi_eigen`
/// (aucune transcendante).
///
/// Renvoie `(u, sigma, vt, sweeps)` ([`SvdResult`]) :
/// * `u` : `m × n` row-major, colonnes orthonormées (sauf celles associées à
///   un `σᵢ` négligeable, laissées nulles).
/// * `sigma` : les `n` valeurs singulières, **triées par ordre décroissant**
///   (contrairement à [`jacobi_eigen`], dont les valeurs propres ne sont pas
///   triées) — convention SVD usuelle.
/// * `vt` : `Vᵀ`, `n × n` row-major.
/// * `sweeps` : nombre de passes Jacobi effectuées (cf. [`jacobi_eigen`]).
///
/// `tol`/`max_sweeps` : mêmes paramètres que [`jacobi_eigen`] (seuil de
/// négligeabilité hors diagonale, borne du nombre de passes) ; `tol` sert
/// aussi de seuil de négligeabilité pour une valeur singulière elle-même
/// (colonne de `U` laissée nulle en dessous).
///
/// `None` en cas de débordement virgule fixe (cf. [`jacobi_eigen`]). Panique
/// si `m < n` ou `a.len() != m·n`.
#[must_use]
pub fn svd<T>(a: &[T], m: usize, n: usize, tol: T, max_sweeps: usize) -> Option<SvdResult<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert!(m >= n, "svd : m ({m}) doit être ≥ n ({n})");
    assert_eq!(a.len(), m * n, "svd : A de longueur {} ≠ {m}×{n}", a.len());

    let at = transpose(a, m, n);
    let ata = matmul(&at, a, n, m, n);
    let (eigenvalues, v, sweeps) = jacobi_eigen(&ata, n, tol, max_sweeps)?;

    // Trie par valeur propre décroissante (T: Ord, cf. FixedReducible).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eigenvalues[j].cmp(&eigenvalues[i]));

    let mut v_sorted = vec![T::ZERO; n * n];
    let mut sigma = vec![T::ZERO; n];
    for (new_col, &old_col) in order.iter().enumerate()
    {
        let ev = eigenvalues[old_col];
        sigma[new_col] = if ev <= T::ZERO { T::ZERO } else { ev.sqrt() };
        for row in 0..n
        {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
    }

    let av = matmul(a, &v_sorted, m, n, n);
    let mut u = vec![T::ZERO; m * n];
    for col in 0..n
    {
        if sigma[col] <= tol
        {
            continue; // valeur singulière négligeable : colonne de U laissée nulle.
        }
        for row in 0..m
        {
            u[row * n + col] = av[row * n + col].checked_div(sigma[col])?;
        }
    }

    let vt = transpose(&v_sorted, n, n);
    Some((u, sigma, vt, sweeps))
}

/// Résout `min_x ‖A·x − b‖` (`A` `m × n`, `m ≥ n`, `b` de longueur `m`) via
/// SVD : `x = V·diag(σ⁺)·Uᵀ·b`, `σᵢ⁺ = 1/σᵢ` si `σᵢ > tol`, sinon `0`.
///
/// Contrairement à [`qr_solve`] (qui renvoie `None` dès qu'un pivot de `R`
/// est nul — rang déficient), `svd_solve` **traite explicitement** les
/// directions de rang déficient (valeurs singulières `≤ tol`) en leur
/// affectant une contribution nulle plutôt que d'échouer : le résultat est la
/// solution de **norme minimale** parmi toutes les solutions optimales des
/// moindres carrés (propriété classique de la pseudo-inverse de
/// Moore-Penrose). Pour un système bien conditionné et de rang plein,
/// coïncide avec [`qr_solve`] à la résolution de `T` près.
///
/// `None` en cas de débordement virgule fixe (cf. [`svd`]). Panique si
/// `m < n`, `a.len() != m·n` ou `b.len() != m`.
#[must_use]
pub fn svd_solve<T>(
    a: &[T],
    b: &[T],
    m: usize,
    n: usize,
    tol: T,
    max_sweeps: usize,
) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(b.len(), m, "svd_solve : b de longueur {} ≠ {m}", b.len());
    let (u, sigma, vt, _) = svd(a, m, n, tol, max_sweeps)?;

    let mut d = vec![T::ZERO; n];
    for i in 0..n
    {
        if sigma[i] <= tol
        {
            continue; // direction de rang déficient : contribution nulle (norme minimale).
        }
        let ui_col: Vec<T> = (0..m).map(|row| u[row * n + i]).collect();
        let ci = dot(&ui_col, b);
        d[i] = ci.checked_div(sigma[i])?;
    }

    let v = transpose(&vt, n, n);
    Some(matvec(&v, &d, n, n))
}

// ------------------------------------------------------------------ //
//  Hessenberg + QR décalé — valeurs propres d'une matrice quelconque  //
// ------------------------------------------------------------------ //

/// Rotation de Givens `(c, s)` telle que `c² + s² = 1` et `−s·x + c·y = 0`
/// (élimine `y`, `x` restant le pivot). Construite par **ratio**
/// (`t = min(|x|,|y|) / max(|x|,|y|)`, borné dans `[−1, 1]`), jamais par
/// `√(x² + y²)` directement : cette dernière voie **sous-déborde** en
/// virgule fixe dès que `|x|`/`|y|` sont petits devant `1` — leur carré, lui,
/// peut tomber sous la résolution du format (ex. `Q16.16`, résolution
/// `1,5·10⁻⁵` : tout `x < √(1,5·10⁻⁵) ≈ 0,004` a un carré qui s'arrondit à
/// zéro), corrompant silencieusement la rotation. Précaution nécessaire ici
/// car [`eigenvalues_general`] applique des rotations à répétition sur des
/// sous-diagonales qui **rétrécissent** à mesure que l'algorithme converge —
/// contrairement à [`hessenberg`], qui n'opère qu'une fois sur les données
/// d'entrée à leur échelle d'origine.
fn givens<T>(x: T, y: T) -> Option<(T, T)>
where
    T: FixedReducible + Sub<Output = T>,
{
    if y == T::ZERO
    {
        return Some((T::one(), T::ZERO));
    }
    if x == T::ZERO
    {
        return Some((T::ZERO, T::one()));
    }
    if y.abs() > x.abs()
    {
        let t = x.checked_div(y)?;
        let u = T::one().wrapping_add(t.wrapping_mul(t)).sqrt();
        let s = T::one().checked_div(u)?;
        let c = s.wrapping_mul(t);
        Some((c, s))
    }
    else
    {
        let t = y.checked_div(x)?;
        let u = T::one().wrapping_add(t.wrapping_mul(t)).sqrt();
        let c = T::one().checked_div(u)?;
        let s = c.wrapping_mul(t);
        Some((c, s))
    }
}

/// Réduit `A` (`n × n` row-major, **quelconque** — pas nécessairement
/// symétrique) à la forme de Hessenberg supérieure par similarité
/// orthogonale : `H = Qᵀ·A·Q`, `H[i,j] = 0` pour `j < i − 1`. `H` a les mêmes
/// valeurs propres que `A` (similarité), ce qui rend [`eigenvalues_general`]
/// beaucoup moins coûteux : chaque étape QR décalée sur une matrice de
/// Hessenberg est `O(n²)` au lieu de `O(n³)` pour une matrice dense.
///
/// Par réflexions de Householder, colonne par colonne, comme
/// [`qr_decompose`] — mais appliquées **des deux côtés** (`H := Hₖ·A·Hₖ`)
/// puisqu'il s'agit d'une similarité, pas d'une simple élimination.
///
/// `None` en cas de débordement virgule fixe pendant une réflexion (même
/// caveat que [`qr_decompose`]). Panique si `a.len() != n·n`.
#[must_use]
pub fn hessenberg<T>(a: &[T], n: usize) -> Option<Vec<T>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(
        a.len(),
        n * n,
        "hessenberg : A de longueur {} ≠ {n}×{n}",
        a.len()
    );
    let mut h = a.to_vec();
    if n < 3
    {
        return Some(h); // toute matrice 0×0/1×1/2×2 est déjà de Hessenberg.
    }

    for k in 0..n - 2
    {
        let len = n - k - 1;
        let x: Vec<T> = (0..len).map(|i| h[(k + 1 + i) * n + k]).collect();
        let norm_x = dot(&x, &x).sqrt();
        if norm_x == T::ZERO
        {
            continue; // sous-colonne déjà nulle : réflexion = identité.
        }
        let alpha = if x[0] < T::ZERO
        {
            norm_x
        }
        else
        {
            T::ZERO - norm_x
        };
        let mut v = x;
        v[0] = v[0] - alpha;
        let vtv = dot(&v, &v);
        if vtv == T::ZERO
        {
            continue; // x déjà aligné (bon signe) : réflexion = identité.
        }

        // Application à gauche : H[k+1.., ..] −= 2·v·(vᵀ·H[k+1.., ..])/vᵀv.
        for j in 0..n
        {
            let col: Vec<T> = (0..len).map(|i| h[(k + 1 + i) * n + j]).collect();
            let vtcol = dot(&v, &col);
            let scale = vtcol.wrapping_add(vtcol).checked_div(vtv)?;
            for i in 0..len
            {
                h[(k + 1 + i) * n + j] = col[i] - v[i].wrapping_mul(scale);
            }
        }
        // Rangs k+2..n de la colonne k : nuls par construction analytique
        // (mêmes précautions que qr_decompose contre le résidu d'arrondi).
        for i in 1..len
        {
            h[(k + 1 + i) * n + k] = T::ZERO;
        }

        // Application à droite : H[.., k+1..] −= 2·(H[.., k+1..]·v)·vᵀ/vᵀv.
        for i in 0..n
        {
            let row: Vec<T> = (0..len).map(|j| h[i * n + (k + 1 + j)]).collect();
            let rowv = dot(&row, &v);
            let scale = rowv.wrapping_add(rowv).checked_div(vtv)?;
            for j in 0..len
            {
                h[i * n + (k + 1 + j)] = row[j] - v[j].wrapping_mul(scale);
            }
        }
    }
    Some(h)
}

/// Valeur propre d'une matrice non symétrique : réelle, ou l'une des deux
/// composantes d'une paire complexe conjuguée (`re ± i·im`, `im` toujours
/// stocké **positif** — la conjuguée a la même partie réelle et l'opposée de
/// `im`). Renvoyée par [`eigenvalues_general`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Eigenvalue<T> {
    /// Valeur propre réelle.
    Real(T),
    /// Une composante `(partie réelle, partie imaginaire > 0)` d'une paire
    /// complexe conjuguée — la conjuguée `(re, −im)` est l'autre valeur
    /// propre renvoyée pour ce même bloc `2×2`.
    Complex(T, T),
}

/// Valeurs propres du bloc `2×2` `[[a, b], [c, d]]` par la formule quadratique
/// sur trace/déterminant (`λ² − tr·λ + det = 0`) : réelles si le discriminant
/// `(a−d)² + 4bc ≥ 0`, sinon la paire complexe conjuguée
/// `tr/2 ± i·√(−discriminant)/2`. Aucune arithmétique complexe requise.
fn eig2x2<T>(a: T, b: T, c: T, d: T) -> Option<(Eigenvalue<T>, Eigenvalue<T>)>
where
    T: FixedReducible + Sub<Output = T>,
{
    let two = T::one().wrapping_add(T::one());
    let tr = a.wrapping_add(d);
    let diff = a - d;
    let bc = b.wrapping_mul(c);
    let four_bc = bc.wrapping_add(bc).wrapping_add(bc).wrapping_add(bc);
    let disc = diff.wrapping_mul(diff).wrapping_add(four_bc);

    if disc >= T::ZERO
    {
        let sq = disc.sqrt();
        let e1 = tr.wrapping_add(sq).checked_div(two)?;
        let e2 = (tr - sq).checked_div(two)?;
        Some((Eigenvalue::Real(e1), Eigenvalue::Real(e2)))
    }
    else
    {
        let sq = (T::ZERO - disc).sqrt();
        let re = tr.checked_div(two)?;
        let im = sq.checked_div(two)?;
        Some((
            Eigenvalue::Complex(re, im),
            Eigenvalue::Complex(re, T::ZERO - im),
        ))
    }
}

/// Une étape de l'algorithme QR décalé (Givens), appliquée « en place » au
/// bloc actif `H[l..=hi, l..=hi]` d'une matrice de Hessenberg : `H_actif :=
/// R·Q + décalage·I`, où `Q·R = H_actif − décalage·I` (`Q` produit de
/// rotations de Givens [`givens`] éliminant chaque sous-diagonale de haut en
/// bas — la structure de Hessenberg garantit qu'une seule rotation par
/// colonne suffit). Préserve la forme de Hessenberg (théorème classique :
/// `R·Q` est de Hessenberg si `Q·R` l'est).
fn shifted_qr_step<T>(h: &mut [T], n: usize, l: usize, hi: usize, shift: T) -> Option<()>
where
    T: FixedReducible + Sub<Output = T>,
{
    for i in l..=hi
    {
        h[i * n + i] = h[i * n + i] - shift;
    }

    let mut rotations = Vec::with_capacity(hi - l);
    for i in l..hi
    {
        let x = h[i * n + i];
        let y = h[(i + 1) * n + i];
        let (c, s) = givens(x, y)?;
        rotations.push((c, s));
        for j in l..=hi
        {
            let t1 = h[i * n + j];
            let t2 = h[(i + 1) * n + j];
            h[i * n + j] = c.wrapping_mul(t1).wrapping_add(s.wrapping_mul(t2));
            h[(i + 1) * n + j] = (T::ZERO - s)
                .wrapping_mul(t1)
                .wrapping_add(c.wrapping_mul(t2));
        }
    }
    // Sous-diagonale de la fenêtre active : nulle par construction analytique
    // (mêmes précautions que hessenberg/qr_decompose contre le résidu d'arrondi).
    for i in l..hi
    {
        h[(i + 1) * n + i] = T::ZERO;
    }

    for (idx, i) in (l..hi).enumerate()
    {
        let (c, s) = rotations[idx];
        for j in l..=hi
        {
            let t1 = h[j * n + i];
            let t2 = h[j * n + (i + 1)];
            h[j * n + i] = c.wrapping_mul(t1).wrapping_add(s.wrapping_mul(t2));
            h[j * n + (i + 1)] = (T::ZERO - s)
                .wrapping_mul(t1)
                .wrapping_add(c.wrapping_mul(t2));
        }
    }

    for i in l..=hi
    {
        h[i * n + i] = h[i * n + i].wrapping_add(shift);
    }
    Some(())
}

/// Valeurs propres d'une matrice **quelconque** `A` (`n × n` row-major, pas
/// nécessairement symétrique) : réduction de Hessenberg ([`hessenberg`]) puis
/// algorithme QR à décalage simple (Wilkinson) avec déflation, isolant
/// chaque bloc `1×1` (valeur propre réelle) ou `2×2` final (paire complexe
/// conjuguée résolue analytiquement, cf. en-tête de module) de bas en haut.
///
/// `tol` est le seuil (valeur absolue) en dessous duquel une sous-diagonale
/// de Hessenberg est jugée négligeable — même convention que [`jacobi_eigen`].
/// `max_iter` borne le nombre total d'étapes QR (garantit la terminaison) ;
/// au-delà de 10 étapes sans déflation sur le bloc actif courant, un
/// décalage « ad hoc » (`|H[hi,hi−1]| + |H[hi−1,hi−2]|`, technique classique
/// EISPACK/Numerical Recipes) remplace le décalage de Wilkinson pour éviter
/// toute stagnation cyclique (un bloc final `3×3` ou plus dont le sous-bloc
/// `2×2` traînant a des valeurs propres complexes ne peut pas converger vers
/// un bloc `2×2` plus petit via un unique décalage réel — le décalage
/// « ad hoc » brise ce cycle en s'écartant délibérément de l'estimation de
/// Wilkinson).
///
/// Renvoie les `n` valeurs propres ([`Eigenvalue`]), **non triées** (ordre de
/// déflation, de bas en haut — comme [`jacobi_eigen`], `T: Ord` permet à
/// l'appelant de les trier si besoin, en comparant par exemple les parties
/// réelles). `None` en cas de débordement virgule fixe (cf. [`hessenberg`])
/// ou si `max_iter` est atteint sans convergence complète. Panique si
/// `a.len() != n·n`.
#[must_use]
pub fn eigenvalues_general<T>(
    a: &[T],
    n: usize,
    tol: T,
    max_iter: usize,
) -> Option<Vec<Eigenvalue<T>>>
where
    T: FixedReducible + Sub<Output = T>,
{
    assert_eq!(
        a.len(),
        n * n,
        "eigenvalues_general : A de longueur {} ≠ {n}×{n}",
        a.len()
    );
    if n == 0
    {
        return Some(Vec::new());
    }

    let mut h = hessenberg(a, n)?;
    let mut eigenvalues = vec![Eigenvalue::Real(T::ZERO); n];

    let mut nn = n;
    let mut total_iters = 0usize;
    let mut stall = 0usize;
    while nn > 0
    {
        if nn == 1
        {
            eigenvalues[0] = Eigenvalue::Real(h[0]);
            nn = 0;
            continue;
        }

        let mut l = nn - 1;
        while l > 0
        {
            if h[l * n + (l - 1)].abs() <= tol
            {
                h[l * n + (l - 1)] = T::ZERO;
                break;
            }
            l -= 1;
        }

        if l == nn - 1
        {
            eigenvalues[nn - 1] = Eigenvalue::Real(h[(nn - 1) * n + (nn - 1)]);
            nn -= 1;
            stall = 0;
        }
        else if l == nn - 2
        {
            let (a_, b_, c_, d_) = (
                h[l * n + l],
                h[l * n + (nn - 1)],
                h[(nn - 1) * n + l],
                h[(nn - 1) * n + (nn - 1)],
            );
            let (e1, e2) = eig2x2(a_, b_, c_, d_)?;
            eigenvalues[l] = e1;
            eigenvalues[nn - 1] = e2;
            nn -= 2;
            stall = 0;
        }
        else
        {
            total_iters += 1;
            if total_iters > max_iter
            {
                return None;
            }
            stall += 1;

            let hi = nn - 1;
            let shift = if stall % 11 == 10
            {
                // Décalage ad hoc de secours (cf. doc de fonction).
                h[hi * n + (hi - 1)]
                    .abs()
                    .wrapping_add(h[(hi - 1) * n + (hi - 2)].abs())
            }
            else
            {
                let (a_, b_, c_, d_) = (
                    h[(hi - 1) * n + (hi - 1)],
                    h[(hi - 1) * n + hi],
                    h[hi * n + (hi - 1)],
                    h[hi * n + hi],
                );
                match eig2x2(a_, b_, c_, d_)
                {
                    Some((Eigenvalue::Real(e1), Eigenvalue::Real(e2))) =>
                    {
                        if (e1 - d_).abs() <= (e2 - d_).abs()
                        {
                            e1
                        }
                        else
                        {
                            e2
                        }
                    },
                    _ => d_,
                }
            };
            shifted_qr_step(&mut h, n, l, hi, shift)?;
        }
    }

    Some(eigenvalues)
}
