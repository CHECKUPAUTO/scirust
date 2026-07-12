//! # Attention produit-scalaire mise à l'échelle, à softmax vectorisé (x86_64)
//!
//! `Attention(Q, K, V) = softmax(scale · Q·Kᵀ) · V` — le cœur des Transformers.
//!
//! * `scores = scale · Q·Kᵀ` : chaque `scores[i,j]` est le produit scalaire de
//!   la ligne `i` de `Q` et la ligne `j` de `K`, calculé par le kernel `sdot`
//!   dispatché (AVX-512/AVX2/NEON/scalaire) du module [`crate::dispatch`].
//! * `softmax` par ligne : **numériquement stable** (soustraction du max) et
//!   **vectorisé** — `max` et `somme` par réduction AVX-512, `exp` par le noyau
//!   vectorisé de [`crate::activations`].
//! * `out = P·V` : produit matriciel via le GEMM tuilé/packé [`crate::gemm`].
//!
//! Repli scalaire garanti pour le softmax hors `avx512f`. Vérifié contre une
//! référence entièrement scalaire dans les tests.

use crate::dispatch::runtime_backend;
use crate::gemm::sgemm_tiled;
use crate::matrix::view::{MatrixView, MatrixViewMut};

/// Softmax **par ligne**, en place : pour chaque ligne de `x` (`rows × cols`),
/// `x[i,:] ← softmax(x[i,:])`. Stable (soustraction du max) et vectorisé
/// AVX-512, avec repli scalaire.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(x.len(), rows * cols, "softmax_rows: shape mismatch");
    if cols == 0
    {
        return;
    }
    for r in 0..rows
    {
        let row = &mut x[r * cols..r * cols + cols];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
            {
                // SAFETY: gated by the runtime detection just above.
                unsafe { softmax_row_avx512(row) };
                continue;
            }
        }
        softmax_row_scalar(row);
    }
}

fn softmax_row_scalar(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter()
    {
        m = m.max(v);
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut()
    {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut()
    {
        *v *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_row_avx512(row: &mut [f32]) {
    use crate::activations::exp_ps;
    use core::arch::x86_64::*;
    let n = row.len();

    // 1) max de la ligne.
    let mut mv = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 16 <= n
    {
        mv = _mm512_max_ps(mv, _mm512_loadu_ps(row.as_ptr().add(i)));
        i += 16;
    }
    let mut m = _mm512_reduce_max_ps(mv);
    for &v in &row[i..n]
    {
        m = m.max(v);
    }
    let mvec = _mm512_set1_ps(m);

    // 2) e = exp(x - max) en place, avec accumulation de la somme.
    let mut sv = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= n
    {
        let e = exp_ps(_mm512_sub_ps(_mm512_loadu_ps(row.as_ptr().add(i)), mvec));
        _mm512_storeu_ps(row.as_mut_ptr().add(i), e);
        sv = _mm512_add_ps(sv, e);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(sv);
    let rem = n - i;
    if rem > 0
    {
        let mask = (1u16 << rem) - 1;
        let e = exp_ps(_mm512_sub_ps(
            _mm512_maskz_loadu_ps(mask, row.as_ptr().add(i)),
            mvec,
        ));
        _mm512_mask_storeu_ps(row.as_mut_ptr().add(i), mask, e);
        // On ne somme que les voies actives (masque) de ce qu'on vient d'écrire.
        sum += _mm512_reduce_add_ps(_mm512_maskz_loadu_ps(mask, row.as_ptr().add(i)));
    }

    // 3) normalisation : x *= 1/sum.
    let inv = _mm512_set1_ps(1.0 / sum);
    let mut i = 0;
    while i + 16 <= n
    {
        _mm512_storeu_ps(
            row.as_mut_ptr().add(i),
            _mm512_mul_ps(_mm512_loadu_ps(row.as_ptr().add(i)), inv),
        );
        i += 16;
    }
    let rem = n - i;
    if rem > 0
    {
        let mask = (1u16 << rem) - 1;
        let v = _mm512_maskz_loadu_ps(mask, row.as_ptr().add(i));
        _mm512_mask_storeu_ps(row.as_mut_ptr().add(i), mask, _mm512_mul_ps(v, inv));
    }
}

/// Attention produit-scalaire mise à l'échelle, **une tête** :
/// `out = softmax(scale · Q·Kᵀ) · V`.
///
/// `q` est `s×d` (requêtes), `k` est `t×d` (clés), `v` est `t×d` (valeurs),
/// `out` est `s×d`. Tous row-major contigus. `scale` vaut typiquement
/// `1/√d`. Alloue la matrice de scores `s×t` en interne.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    s: usize,
    d: usize,
    k: &[f32],
    t: usize,
    v: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    assert_eq!(q.len(), s * d, "attention: Q shape");
    assert_eq!(k.len(), t * d, "attention: K shape");
    assert_eq!(v.len(), t * d, "attention: V shape");
    assert_eq!(out.len(), s * d, "attention: out shape");

    // scores = scale · Q·Kᵀ  (s×t).
    let backend = runtime_backend();
    let mut scores = vec![0.0f32; s * t];
    for i in 0..s
    {
        let q_row = &q[i * d..i * d + d];
        for j in 0..t
        {
            let k_row = &k[j * d..j * d + d];
            scores[i * t + j] = scale * backend.sdot_f32(q_row, k_row);
        }
    }

    // softmax par ligne (fusionné, vectorisé).
    softmax_rows(&mut scores, s, t);

    // out = P·V  (s×t · t×d).
    sgemm_tiled(
        1.0,
        MatrixView::new(&scores, s, t),
        MatrixView::new(v, t, d),
        0.0,
        MatrixViewMut::new(out, s, d),
    );
}

/// Taille de bloc de clés/valeurs pour [`flash_attention`].
const FLASH_BC: usize = 64;

/// **Flash-attention** (une tête) : même résultat que [`attention`], mais avec
/// un **softmax en ligne** qui ne matérialise **jamais** la matrice de scores
/// `s×t`. Pour chaque requête, on balaie les clés/valeurs par blocs de
/// `FLASH_BC` en maintenant un état de taille `O(d)` — le maximum courant `m`,
/// la somme courante `l` et l'accumulateur de sortie `o` — rééchelonnés à
/// chaque bloc (`×exp(m_ancien − m_nouveau)`). La mémoire de travail est donc
/// `O(d + FLASH_BC)` par requête au lieu de `O(t)`, ce qui rend les longues
/// séquences traitables (le principe de FlashAttention).
///
/// `exp` est vectorisée ([`crate::activations`]) et l'accumulation `o += p·v`
/// passe par le kernel `saxpy` dispatché. Résultat numériquement identique à
/// [`attention`] (à l'arrondi près), vérifié dans les tests.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention(
    q: &[f32],
    s: usize,
    d: usize,
    k: &[f32],
    t: usize,
    v: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    use crate::activations::exp_inplace;
    assert_eq!(q.len(), s * d, "flash_attention: Q shape");
    assert_eq!(k.len(), t * d, "flash_attention: K shape");
    assert_eq!(v.len(), t * d, "flash_attention: V shape");
    assert_eq!(out.len(), s * d, "flash_attention: out shape");

    let backend = runtime_backend();
    let mut scores = vec![0.0f32; FLASH_BC]; // buffer de bloc, réutilisé

    for i in 0..s
    {
        let q_row = &q[i * d..i * d + d];
        let o = &mut out[i * d..i * d + d];
        o.iter_mut().for_each(|x| *x = 0.0);
        let mut m = f32::NEG_INFINITY; // max courant
        let mut l = 0.0f32; // somme courante des exponentielles

        let mut j0 = 0;
        while j0 < t
        {
            let bc = FLASH_BC.min(t - j0);
            // Scores du bloc : s_j = scale · q·k_j.
            let mut block_max = f32::NEG_INFINITY;
            for (jj, sc) in scores[..bc].iter_mut().enumerate()
            {
                let k_row = &k[(j0 + jj) * d..(j0 + jj) * d + d];
                let v = scale * backend.sdot_f32(q_row, k_row);
                *sc = v;
                block_max = block_max.max(v);
            }

            let m_new = m.max(block_max);
            // Correction du bloc précédent : ×exp(m − m_new) (0 au 1er bloc).
            let corr = if m == f32::NEG_INFINITY
            {
                0.0
            }
            else
            {
                (m - m_new).exp()
            };

            // p_j = exp(s_j − m_new), vectorisé.
            for sc in scores[..bc].iter_mut()
            {
                *sc -= m_new;
            }
            exp_inplace(&mut scores[..bc]);

            // Rééchelonne l'état, puis accumule le bloc.
            l = l * corr + scores[..bc].iter().sum::<f32>();
            for x in o.iter_mut()
            {
                *x *= corr;
            }
            for (jj, &p) in scores[..bc].iter().enumerate()
            {
                let v_row = &v[(j0 + jj) * d..(j0 + jj) * d + d];
                backend.saxpy_f32(p, v_row, o); // o += p · v_j
            }

            m = m_new;
            j0 += bc;
        }

        // Normalisation finale : o /= l.
        if l != 0.0
        {
            let inv = 1.0 / l;
            for x in o.iter_mut()
            {
                *x *= inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attention_ref(
        q: &[f32],
        s: usize,
        d: usize,
        k: &[f32],
        t: usize,
        v: &[f32],
        scale: f32,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; s * t];
        for i in 0..s
        {
            for j in 0..t
            {
                let mut acc = 0.0f32;
                for e in 0..d
                {
                    acc += q[i * d + e] * k[j * d + e];
                }
                scores[i * t + j] = scale * acc;
            }
        }
        for i in 0..s
        {
            let row = &mut scores[i * t..i * t + t];
            let mut m = f32::NEG_INFINITY;
            for &vv in row.iter()
            {
                m = m.max(vv);
            }
            let mut sum = 0.0f32;
            for vv in row.iter_mut()
            {
                *vv = (*vv - m).exp();
                sum += *vv;
            }
            for vv in row.iter_mut()
            {
                *vv /= sum;
            }
        }
        let mut out = vec![0.0f32; s * d];
        for i in 0..s
        {
            for e in 0..d
            {
                let mut acc = 0.0f32;
                for j in 0..t
                {
                    acc += scores[i * t + j] * v[j * d + e];
                }
                out[i * d + e] = acc;
            }
        }
        out
    }

    #[test]
    fn softmax_rows_is_stochastic_and_stable() {
        let rows = 5;
        let cols = 17;
        let mut x: Vec<f32> = (0..rows * cols)
            .map(|t| (t as f32 * 0.13).sin() * 10.0)
            .collect();
        softmax_rows(&mut x, rows, cols);
        for r in 0..rows
        {
            let row = &x[r * cols..r * cols + cols];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() <= 1e-5, "row {r} sum {sum}");
            assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }

    #[test]
    fn attention_matches_scalar_reference() {
        for &(s, d, t) in &[(1usize, 1usize, 1usize), (3, 4, 5), (8, 16, 12), (7, 5, 33)]
        {
            let q: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.07).sin()).collect();
            let k: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.05).cos()).collect();
            let v: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.03) - 0.5).collect();
            let scale = 1.0 / (d as f32).sqrt();

            let want = attention_ref(&q, s, d, &k, t, &v, scale);
            let mut got = vec![0.0f32; s * d];
            attention(&q, s, d, &k, t, &v, scale, &mut got);

            for idx in 0..s * d
            {
                let tol = 1e-4 * (1.0 + want[idx].abs());
                assert!(
                    (got[idx] - want[idx]).abs() <= tol,
                    "s={s} d={d} t={t} idx={idx}: {} vs {}",
                    got[idx],
                    want[idx]
                );
            }
        }
    }

    #[test]
    fn flash_attention_matches_reference() {
        // t=200 > FLASH_BC(64) exerce le rééchelonnement en ligne sur 4 blocs.
        for &(s, d, t) in &[
            (1usize, 1usize, 1usize),
            (3, 4, 5),
            (8, 16, 12),
            (7, 5, 64),
            (5, 8, 200),
        ]
        {
            let q: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.07).sin()).collect();
            let k: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.05).cos() * 3.0).collect();
            let v: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.03) - 0.5).collect();
            let scale = 1.0 / (d as f32).sqrt();

            let want = attention_ref(&q, s, d, &k, t, &v, scale);
            let mut got = vec![0.0f32; s * d];
            flash_attention(&q, s, d, &k, t, &v, scale, &mut got);

            for idx in 0..s * d
            {
                let tol = 1e-4 * (1.0 + want[idx].abs());
                assert!(
                    (got[idx] - want[idx]).abs() <= tol,
                    "flash s={s} d={d} t={t} idx={idx}: {} vs {}",
                    got[idx],
                    want[idx]
                );
            }
        }
    }
}
