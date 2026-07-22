// scirust-simd/src/dsp/savgol.rs
//
// # Filtre de Savitzky–Golay — lissage & différentiation polynomiale
//
// Un filtre de Savitzky–Golay lisse un signal (ou en estime une dérivée) en
// ajustant, sur une fenêtre glissante, un **polynôme de degré `p` au sens des
// moindres carrés**, puis en évaluant ce polynôme (ou sa dérivée `d`) au centre
// de la fenêtre. Contrairement à une moyenne glissante, il **préserve les pics
// et les moments d'ordre élevé** — d'où son usage massif en traitement de
// signaux expérimentaux bruités (spectroscopie, capteurs). L'équivalent de
// `scipy.signal.savgol_filter`, **générique sur le scalaire**.
//
// ## Coefficients
//
// L'ajustement est **linéaire** en les échantillons : le résultat est une
// simple convolution FIR par des coefficients qui ne dépendent **que** de la
// géométrie `(window_len, poly_order, deriv)`, pas des données ([`savgol_coeffs`]).
// Sur la fenêtre centrée aux positions entières `z ∈ {−m, …, m}`
// (`window_len = 2m+1`), on pose la matrice de Vandermonde `A[i][j] = zᵢʲ`
// (`j = 0..=p`). Les coefficients de la dérivée `d` sont
//
// ```text
// c = d! · [ (AᵀA)⁻¹ Aᵀ ]_{d,·}
// ```
//
// c.-à-d. : on résout le petit système normal `(AᵀA)·g = e_d` (matrice
// `(p+1)×(p+1)`, symétrique définie positive — élimination de Gauss avec pivot
// partiel, [`solve`]), puis `c[i] = d!·Σⱼ gⱼ·zᵢʲ`. Pour `d = 0` (lissage) les
// coefficients **somment à 1** ; pour `d ≥ 1`, ils estiment la dérivée en
// **unités d'échantillon** (diviser par `hᵈ` si le pas physique est `h`).
//
// ## Bords
//
// [`savgol_filter`] applique la convolution en **répliquant l'échantillon de
// bord** (indices saturés à `[0, N−1]`), soit le mode `nearest` de `scipy`.
// C'est déterministe et sans allocation de contexte ; sur les `m` premiers et
// derniers points, préférer ignorer la sortie si une fidélité de bord stricte
// est requise.
//
// ## Conditionnement
//
// `AᵀA` d'une matrice de Vandermonde est notoirement **mal conditionnée** quand
// `p` grandit. Pour des `(window_len, poly_order)` usuels (`p ≤ 4`) et en
// flottant, c'est sans conséquence ; en virgule fixe, rester à des ordres bas.
// Aucun `unsafe`.

use crate::fixed::RealScalar;

/// Résout le système `a·x = b` (`n×n`, `a` supposée non singulière) par
/// élimination de Gauss avec pivot partiel. `a` et `b` sont consommés
/// (modifiés en place). Utilisé pour le petit système normal de
/// [`savgol_coeffs`] ; `n = poly_order + 1` est minuscule.
// Élimination de Gauss : les indices `r`/`c` parcourent lignes et colonnes
// tout en indexant la ligne pivot `a[col]` — plus clair qu'un itérateur.
#[allow(clippy::needless_range_loop)]
fn solve<T: RealScalar>(mut a: Vec<Vec<T>>, mut b: Vec<T>) -> Vec<T> {
    let n = b.len();
    for col in 0..n
    {
        // Pivot partiel : plus grande magnitude sur la colonne.
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n
        {
            let m = a[r][col].abs();
            if m > best
            {
                best = m;
                pivot = r;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let inv = a[col][col].recip();
        for r in (col + 1)..n
        {
            let factor = a[r][col] * inv;
            for c in col..n
            {
                a[r][c] = a[r][c] - factor * a[col][c];
            }
            b[r] = b[r] - factor * b[col];
        }
    }
    // Remontée.
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev()
    {
        let mut acc = b[i];
        for c in (i + 1)..n
        {
            acc = acc - a[i][c] * x[c];
        }
        x[i] = acc * a[i][i].recip();
    }
    x
}

/// Coefficients de convolution FIR de Savitzky–Golay pour une fenêtre de
/// `window_len` points (impair), un polynôme de degré `poly_order` et l'ordre
/// de dérivée `deriv` (cf. en-tête de module). La sortie a `window_len`
/// éléments, ordonnés du bord gauche (`z = −m`) au bord droit (`z = +m`).
///
/// Pour `deriv == 0`, les coefficients somment à `1` (lissage) ; pour
/// `deriv ≥ 1`, ils estiment la `deriv`-ième dérivée en unités d'échantillon.
///
/// Panique si `window_len` est pair ou nul, si `poly_order >= window_len`, ou
/// si `deriv > poly_order`.
#[must_use]
pub fn savgol_coeffs<T: RealScalar>(window_len: usize, poly_order: usize, deriv: usize) -> Vec<T> {
    assert!(
        window_len >= 1 && window_len % 2 == 1,
        "savgol : window_len doit être impair ≥ 1"
    );
    assert!(
        poly_order < window_len,
        "savgol : poly_order ({poly_order}) doit être < window_len ({window_len})"
    );
    assert!(
        deriv <= poly_order,
        "savgol : deriv ({deriv}) doit être ≤ poly_order ({poly_order})"
    );

    let m = (window_len / 2) as i32;
    let p1 = poly_order + 1;

    // Positions z_i = i − m, et puissances z_i^j précalculées (A[i][j]).
    let powers: Vec<Vec<T>> = (0..window_len)
        .map(|i| {
            let z = T::from_i32(i as i32 - m);
            let mut row = Vec::with_capacity(p1);
            let mut acc = T::one();
            for _ in 0..p1
            {
                row.push(acc);
                acc = acc * z;
            }
            row
        })
        .collect();

    // Système normal G = AᵀA, symétrique (p1×p1) : G[a][b] = Σ_i z_i^{a+b}.
    let mut g = vec![vec![T::zero(); p1]; p1];
    for row in &powers
    {
        for a in 0..p1
        {
            for b in 0..p1
            {
                g[a][b] = g[a][b] + row[a] * row[b];
            }
        }
    }

    // Résout G·gsol = e_deriv, puis c[i] = deriv!·Σ_j gsol_j·z_i^j.
    let mut e = vec![T::zero(); p1];
    e[deriv] = T::one();
    let gsol = solve(g, e);

    let mut fact = 1i32;
    for k in 2..=deriv
    {
        fact *= k as i32;
    }
    let fact = T::from_i32(fact);

    powers
        .iter()
        .map(|row| {
            let dot = (0..p1).fold(T::zero(), |acc, j| acc + gsol[j] * row[j]);
            fact * dot
        })
        .collect()
}

/// Applique un filtre de Savitzky–Golay à `x` : convolution par
/// [`savgol_coeffs`] avec réplication de bord (mode `nearest`, cf. en-tête de
/// module). Renvoie un signal de même longueur que `x`.
///
/// Panique selon les préconditions de [`savgol_coeffs`], ou si `x` est vide.
#[must_use]
pub fn savgol_filter<T: RealScalar>(
    x: &[T],
    window_len: usize,
    poly_order: usize,
    deriv: usize,
) -> Vec<T> {
    assert!(!x.is_empty(), "savgol_filter : signal vide");
    let coeffs = savgol_coeffs::<T>(window_len, poly_order, deriv);
    let m = (window_len / 2) as isize;
    let n = x.len() as isize;
    (0..x.len())
        .map(|i| {
            let mut acc = T::zero();
            for (k, &c) in coeffs.iter().enumerate()
            {
                // Indice saturé à [0, N−1] : réplication de bord.
                let idx = (i as isize + k as isize - m).clamp(0, n - 1) as usize;
                acc = acc + c * x[idx];
            }
            acc
        })
        .collect()
}
