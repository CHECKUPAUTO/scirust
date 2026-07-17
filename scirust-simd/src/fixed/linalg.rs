// scirust-simd/src/fixed/linalg.rs
//
// # Algèbre linéaire virgule fixe — GEMM et décompositions déterministes
//
// Produit matrice-matrice (`matmul`) et matrice-vecteur (`matvec`) sur des
// matrices **denses row-major** de scalaires virgule fixe, construits sur le
// produit scalaire SIMD [`super::reductions::dot`]. C'est le socle d'une
// inférence **quantifiée** (réseaux de neurones à poids/activations entiers) :
// la virgule fixe symétrique `Fixed<I, FRAC>` est exactement une quantification
// symétrique d'échelle `2^-FRAC` et de zéro nul.
//
// Le module fournit aussi les deux décompositions directes classiques pour
// résoudre `A·x = b` : [`cholesky`]/[`cholesky_solve`] (matrices symétriques
// définies positives) et [`lu_decompose`]/[`lu_solve`] (matrices générales,
// pivot partiel), plus [`determinant`]. Toutes reposent sur les mêmes
// primitives déterministes (`dot`, division réelle vérifiée) que le GEMM.
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
// ## Panique
//
// Les fonctions **paniquent** sur incohérence dimensionnelle (longueur de slice
// ≠ produit des dimensions annoncées) — c'est un bug d'appelant, pas une donnée.
// En revanche, une matrice non définie positive (`cholesky`) ou singulière
// (`lu_decompose`) renvoie `None` : c'est une propriété des **données**, pas
// une erreur d'appel.

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
/// scalaire contigu et **vectorisé** ([`super::reductions::dot`]).
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
