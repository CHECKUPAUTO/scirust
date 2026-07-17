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
