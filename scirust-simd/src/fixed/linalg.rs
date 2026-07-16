// scirust-simd/src/fixed/linalg.rs
//
// # Algèbre linéaire virgule fixe — GEMM déterministe
//
// Produit matrice-matrice (`matmul`) et matrice-vecteur (`matvec`) sur des
// matrices **denses row-major** de scalaires virgule fixe, construits sur le
// produit scalaire SIMD [`super::reductions::dot`]. C'est le socle d'une
// inférence **quantifiée** (réseaux de neurones à poids/activations entiers) :
// la virgule fixe symétrique `Fixed<I, FRAC>` est exactement une quantification
// symétrique d'échelle `2^-FRAC` et de zéro nul.
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
// ## Disposition mémoire
//
// Une matrice `m × n` est un `&[T]` de longueur `m·n` en **row-major** : la
// ligne `i`, colonne `j`, est à l'indice `i·n + j`. `matmul` transpose la
// matrice de droite une fois (`O(k·n)`) pour rendre chaque produit scalaire
// **contigu** (donc vectorisable), coût négligeable devant le `O(m·k·n)` global.
//
// ## Panique
//
// Les fonctions **paniquent** sur incohérence dimensionnelle (longueur de slice
// ≠ produit des dimensions annoncées) — c'est un bug d'appelant, pas une donnée.

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
