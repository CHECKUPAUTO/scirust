// scirust-simd/src/fixed/attention.rs
//
// # Attention produit-scalaire mise à l'échelle, déterministe (virgule fixe)
//
// [`attention`] : `Attention(Q, K, V) = softmax(scale · Q·Kᵀ) · V`, le bloc de
// base des Transformers — le pendant **quantifié déterministe** du module
// flottant [`crate::attention`] (non déterministe, orienté débit brut avec
// dispatch AVX-512/NEON). Complète, côté séquences, ce que
// [`super::conv`]/[`super::conv2d`]/[`super::layer`] fournissent déjà côté
// convolutif : une chaîne d'inférence entièrement en virgule fixe,
// reproductible bit-à-bit sur toute architecture.
//
// Contrairement à la version flottante — qui évite de matérialiser la
// matrice de scores `s×t` via flash-attention (une optimisation **mémoire**
// pour de longues séquences sur GPU) — cette version la matérialise
// directement, comme [`super::conv`] matérialise son tampon im2col : dans un
// contexte de référence quantifiée déterministe, la simplicité et la
// clarté priment sur une optimisation mémoire qui n'a pas la même urgence
// (les séquences visées ici sont de taille modeste, embarqué/edge).
//
// ## Construction (assemblage de primitives existantes, rien de neuf)
//
// * `scores = scale · Q·Kᵀ` : [`super::linalg::matmul_bt`] — `K` est déjà
//   `t × d`, exactement la forme `Bᵀ` attendue pour `Q·Kᵀ`, donc aucune
//   transposition (même astuce que [`super::layer::Linear::forward_batch`]
//   avec la matrice de poids). Puis une multiplication ponctuelle par `scale`.
// * `softmax` par ligne : [`super::transcendental::softmax_into`], appelée
//   indépendamment sur chaque ligne. **Réservé au stockage `i32`** :
//   l'exponentielle virgule fixe l'est aussi (précision interne Q32).
// * `out = P·V` : [`super::linalg::matmul`].
//
// [`causal_attention`] restreint, pour la requête `i`, les clés à `0..=i`
// avant softmax (masquage causal, décodeur/LLM) : le travail est borné au
// triangle inférieur (divisé par ~2), sans qu'aucun score futur ne soit
// jamais calculé — pas besoin du rééchelonnement en ligne de flash-attention
// puisque chaque requête matérialise directement sa propre matrice tronquée
// (déjà petite).
//
// [`multi_head_attention`] applique [`attention`]/[`causal_attention`]
// indépendamment à chaque tête, sur des tenseurs à têtes concaténées le long
// de l'axe des features (`s × (h·d_head)`, layout usuel après projection
// QKV) — même convention que la version flottante.

use super::linalg::{matmul, matmul_bt};
use super::transcendental::softmax_into;
use super::types::Fixed;

/// Attention produit-scalaire mise à l'échelle, **une tête**, déterministe.
///
/// `q` est `s×d` (requêtes), `k` est `t×d` (clés), `v` est `t×d` (valeurs),
/// tous row-major. Retourne `s×d`. `scale` vaut typiquement `1/√d`.
///
/// Réservé au stockage `i32` (softmax, cf. en-tête de module). Panique si
/// les longueurs de slice ne correspondent pas aux dimensions annoncées.
#[must_use]
pub fn attention<const FRAC: u32>(
    q: &[Fixed<i32, FRAC>],
    s: usize,
    d: usize,
    k: &[Fixed<i32, FRAC>],
    t: usize,
    v: &[Fixed<i32, FRAC>],
    scale: Fixed<i32, FRAC>,
) -> Vec<Fixed<i32, FRAC>> {
    assert_eq!(
        q.len(),
        s * d,
        "attention : Q de longueur {} ≠ {s}×{d}",
        q.len()
    );
    assert_eq!(
        k.len(),
        t * d,
        "attention : K de longueur {} ≠ {t}×{d}",
        k.len()
    );
    assert_eq!(
        v.len(),
        t * d,
        "attention : V de longueur {} ≠ {t}×{d}",
        v.len()
    );

    // scores = scale · Q·Kᵀ (s×t) : K est déjà t×d, la forme Bᵀ de matmul_bt.
    let mut scores = matmul_bt(q, k, s, d, t);
    for sc in scores.iter_mut()
    {
        *sc *= scale;
    }

    // softmax par ligne, tampon réutilisé (une seule allocation pour tout s).
    let mut row_buf = vec![Fixed::zero(); t];
    for row in scores.chunks_exact_mut(t)
    {
        softmax_into(row, &mut row_buf);
        row.copy_from_slice(&row_buf);
    }

    // out = P·V (s×t · t×d).
    matmul(&scores, v, s, t, d)
}

/// **Attention causale** (self-attention, `t = s`), déterministe : la requête
/// `i` ne regarde que les clés `0..=i` (masquage du futur), comme en
/// inférence décodeur/LLM. `q`, `k`, `v` sont tous `s×d`.
///
/// Chaque requête matérialise directement sa propre matrice de scores
/// **tronquée** (`i+1` colonnes, pas `s`) : aucun score futur n'est jamais
/// calculé, contrairement à calculer puis masquer.
///
/// Réservé au stockage `i32`. Panique si les longueurs de slice ne
/// correspondent pas aux dimensions annoncées (`k`/`v` doivent être `s×d`).
#[must_use]
pub fn causal_attention<const FRAC: u32>(
    q: &[Fixed<i32, FRAC>],
    s: usize,
    d: usize,
    k: &[Fixed<i32, FRAC>],
    v: &[Fixed<i32, FRAC>],
    scale: Fixed<i32, FRAC>,
) -> Vec<Fixed<i32, FRAC>> {
    assert_eq!(
        q.len(),
        s * d,
        "causal_attention : Q de longueur {} ≠ {s}×{d}",
        q.len()
    );
    assert_eq!(
        k.len(),
        s * d,
        "causal_attention : K de longueur {} ≠ {s}×{d} (t doit valoir s)",
        k.len()
    );
    assert_eq!(
        v.len(),
        s * d,
        "causal_attention : V de longueur {} ≠ {s}×{d} (t doit valoir s)",
        v.len()
    );

    let mut out = vec![Fixed::zero(); s * d];
    let mut row_buf = vec![Fixed::zero(); s]; // majoré par s (t_eff ≤ s)
    for i in 0..s
    {
        let t_eff = i + 1;
        let q_row = &q[i * d..i * d + d];

        let mut scores = matmul_bt(q_row, &k[..t_eff * d], 1, d, t_eff);
        for sc in scores.iter_mut()
        {
            *sc *= scale;
        }
        softmax_into(&scores, &mut row_buf[..t_eff]);

        let row_out = matmul(&row_buf[..t_eff], &v[..t_eff * d], 1, t_eff, d);
        out[i * d..i * d + d].copy_from_slice(&row_out);
    }
    out
}

/// **Attention multi-tête**, déterministe : `h` têtes indépendantes de
/// dimension `d_head` chacune, sur des tenseurs à têtes concaténées le long
/// de l'axe des features — `q` est `s×(h·d_head)`, `k`/`v` sont
/// `t×(h·d_head)`, retourne `s×(h·d_head)`. Chaque tête attend
/// indépendamment sur ses `d_head` colonnes.
///
/// `causal = true` applique le masquage causal (exige `t == s`) via
/// [`causal_attention`] ; sinon [`attention`]. Réservé au stockage `i32`.
/// Panique si les longueurs de slice ne correspondent pas aux dimensions
/// annoncées, ou si `causal` est demandé avec `t != s`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention<const FRAC: u32>(
    q: &[Fixed<i32, FRAC>],
    s: usize,
    t: usize,
    h: usize,
    d_head: usize,
    k: &[Fixed<i32, FRAC>],
    v: &[Fixed<i32, FRAC>],
    scale: Fixed<i32, FRAC>,
    causal: bool,
) -> Vec<Fixed<i32, FRAC>> {
    let dm = h * d_head;
    assert_eq!(
        q.len(),
        s * dm,
        "multi_head_attention : Q de longueur {} ≠ {s}×{dm}",
        q.len()
    );
    assert_eq!(
        k.len(),
        t * dm,
        "multi_head_attention : K de longueur {} ≠ {t}×{dm}",
        k.len()
    );
    assert_eq!(
        v.len(),
        t * dm,
        "multi_head_attention : V de longueur {} ≠ {t}×{dm}",
        v.len()
    );
    assert!(
        !causal || t == s,
        "multi_head_attention : le masquage causal exige t == s"
    );

    let mut out = vec![Fixed::zero(); s * dm];
    let mut qh = vec![Fixed::zero(); s * d_head];
    let mut kh = vec![Fixed::zero(); t * d_head];
    let mut vh = vec![Fixed::zero(); t * d_head];

    for head in 0..h
    {
        let off = head * d_head;
        for r in 0..s
        {
            qh[r * d_head..r * d_head + d_head]
                .copy_from_slice(&q[r * dm + off..r * dm + off + d_head]);
        }
        for r in 0..t
        {
            kh[r * d_head..r * d_head + d_head]
                .copy_from_slice(&k[r * dm + off..r * dm + off + d_head]);
            vh[r * d_head..r * d_head + d_head]
                .copy_from_slice(&v[r * dm + off..r * dm + off + d_head]);
        }

        let oh = if causal
        {
            causal_attention(&qh, s, d_head, &kh, &vh, scale)
        }
        else
        {
            attention(&qh, s, d_head, &kh, t, &vh, scale)
        };

        for r in 0..s
        {
            out[r * dm + off..r * dm + off + d_head]
                .copy_from_slice(&oh[r * d_head..r * d_head + d_head]);
        }
    }
    out
}
