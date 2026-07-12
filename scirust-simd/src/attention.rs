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

/// **Flash-attention causale** (self-attention, `t = s`) : la requête `i` ne
/// regarde que les clés `0..=i` (masquage du futur), comme en inférence
/// décodeur/LLM. `q`, `k`, `v` sont tous `s×d`.
///
/// Le masquage est **gratuit** ici : au lieu de calculer puis d'annuler les
/// scores futurs, on **borne** simplement le balayage des clés à `[0, i+1)` —
/// ce qui divise aussi le travail par ~2 (triangle inférieur). Softmax en
/// ligne identique à [`flash_attention`], mémoire `O(d + FLASH_BC)` par requête.
pub fn flash_attention_causal(
    q: &[f32],
    s: usize,
    d: usize,
    k: &[f32],
    v: &[f32],
    scale: f32,
    out: &mut [f32],
) {
    use crate::activations::exp_inplace;
    assert_eq!(q.len(), s * d, "flash_attention_causal: Q shape");
    assert_eq!(
        k.len(),
        s * d,
        "flash_attention_causal: K shape (t must equal s)"
    );
    assert_eq!(
        v.len(),
        s * d,
        "flash_attention_causal: V shape (t must equal s)"
    );
    assert_eq!(out.len(), s * d, "flash_attention_causal: out shape");

    let backend = runtime_backend();
    let mut scores = vec![0.0f32; FLASH_BC];

    for i in 0..s
    {
        let q_row = &q[i * d..i * d + d];
        let o = &mut out[i * d..i * d + d];
        o.iter_mut().for_each(|x| *x = 0.0);
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;

        let t_eff = i + 1; // clés autorisées : 0..=i
        let mut j0 = 0;
        while j0 < t_eff
        {
            let bc = FLASH_BC.min(t_eff - j0);
            let mut block_max = f32::NEG_INFINITY;
            for (jj, sc) in scores[..bc].iter_mut().enumerate()
            {
                let k_row = &k[(j0 + jj) * d..(j0 + jj) * d + d];
                let val = scale * backend.sdot_f32(q_row, k_row);
                *sc = val;
                block_max = block_max.max(val);
            }

            let m_new = m.max(block_max);
            let corr = if m == f32::NEG_INFINITY
            {
                0.0
            }
            else
            {
                (m - m_new).exp()
            };

            for sc in scores[..bc].iter_mut()
            {
                *sc -= m_new;
            }
            exp_inplace(&mut scores[..bc]);

            l = l * corr + scores[..bc].iter().sum::<f32>();
            for x in o.iter_mut()
            {
                *x *= corr;
            }
            for (jj, &p) in scores[..bc].iter().enumerate()
            {
                let v_row = &v[(j0 + jj) * d..(j0 + jj) * d + d];
                backend.saxpy_f32(p, v_row, o);
            }

            m = m_new;
            j0 += bc;
        }

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

/// **Attention multi-tête** : `h` têtes indépendantes de dimension `d_head`
/// chacune, sur des tenseurs à têtes concaténées le long de l'axe des features
/// (layout usuel après la projection QKV) — `q` est `s×(h·d_head)`, `k` et `v`
/// sont `t×(h·d_head)`, `out` est `s×(h·d_head)`. Chaque tête `head` attend
/// indépendamment sur ses `d_head` colonnes `[head·d_head, (head+1)·d_head)`.
///
/// `causal = true` applique le masquage causal (exige `t == s`) via
/// [`flash_attention_causal`] ; sinon [`flash_attention`]. Les têtes sont
/// extraites/réinsérées en buffers contigus pour réutiliser les noyaux flash.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    q: &[f32],
    s: usize,
    t: usize,
    h: usize,
    d_head: usize,
    k: &[f32],
    v: &[f32],
    scale: f32,
    causal: bool,
    out: &mut [f32],
) {
    let dm = h * d_head; // dimension modèle (features concaténées)
    assert_eq!(q.len(), s * dm, "multi_head_attention: Q shape");
    assert_eq!(k.len(), t * dm, "multi_head_attention: K shape");
    assert_eq!(v.len(), t * dm, "multi_head_attention: V shape");
    assert_eq!(out.len(), s * dm, "multi_head_attention: out shape");
    if causal
    {
        assert_eq!(t, s, "multi_head_attention: causal exige t == s");
    }

    let mut qh = vec![0.0f32; s * d_head];
    let mut kh = vec![0.0f32; t * d_head];
    let mut vh = vec![0.0f32; t * d_head];
    let mut oh = vec![0.0f32; s * d_head];

    for head in 0..h
    {
        let off = head * d_head;
        // Extraction de la tête vers des buffers contigus.
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

        if causal
        {
            flash_attention_causal(&qh, s, d_head, &kh, &vh, scale, &mut oh);
        }
        else
        {
            flash_attention(&qh, s, d_head, &kh, t, &vh, scale, &mut oh);
        }

        // Réinsertion de la sortie de la tête.
        for r in 0..s
        {
            out[r * dm + off..r * dm + off + d_head]
                .copy_from_slice(&oh[r * d_head..r * d_head + d_head]);
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

    /// Référence causale : softmax sur les clés `0..=i` seulement.
    fn causal_ref(q: &[f32], s: usize, d: usize, k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; s * d];
        for i in 0..s
        {
            let t_eff = i + 1;
            let mut row = vec![0.0f32; t_eff];
            for (j, r) in row.iter_mut().enumerate()
            {
                let mut acc = 0.0f32;
                for e in 0..d
                {
                    acc += q[i * d + e] * k[j * d + e];
                }
                *r = scale * acc;
            }
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for r in row.iter_mut()
            {
                *r = (*r - m).exp();
                sum += *r;
            }
            for e in 0..d
            {
                let mut acc = 0.0f32;
                for (j, &p) in row.iter().enumerate()
                {
                    acc += p * v[j * d + e];
                }
                out[i * d + e] = acc / sum;
            }
        }
        out
    }

    #[test]
    fn flash_causal_matches_reference() {
        // s=100 > FLASH_BC exerce le balayage causal multi-bloc et le triangle.
        for &(s, d) in &[(1usize, 1usize), (4, 8), (17, 5), (64, 16), (100, 12)]
        {
            let q: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.07).sin()).collect();
            let k: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.05).cos() * 2.0).collect();
            let v: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.03) - 0.5).collect();
            let scale = 1.0 / (d as f32).sqrt();

            let want = causal_ref(&q, s, d, &k, &v, scale);
            let mut got = vec![0.0f32; s * d];
            flash_attention_causal(&q, s, d, &k, &v, scale, &mut got);
            for idx in 0..s * d
            {
                let tol = 1e-4 * (1.0 + want[idx].abs());
                assert!(
                    (got[idx] - want[idx]).abs() <= tol,
                    "causal s={s} d={d} idx={idx}: {} vs {}",
                    got[idx],
                    want[idx]
                );
            }
        }
    }

    #[test]
    fn causal_first_query_is_value_row_zero() {
        // Requête 0 ne voit que la clé 0 → softmax trivial → out[0] == v[0].
        let (s, d) = (4, 3);
        let q: Vec<f32> = (0..s * d).map(|i| (i as f32).sin()).collect();
        let k: Vec<f32> = (0..s * d).map(|i| (i as f32).cos()).collect();
        let v: Vec<f32> = (0..s * d).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; s * d];
        flash_attention_causal(&q, s, d, &k, &v, 0.3, &mut out);
        for e in 0..d
        {
            assert!((out[e] - v[e]).abs() <= 1e-5, "out[0][{e}] != v[0][{e}]");
        }
    }

    #[test]
    fn multi_head_matches_per_head() {
        for &(s, t, h, dh) in &[(3usize, 5usize, 2usize, 4usize), (8, 8, 3, 6)]
        {
            let dm = h * dh;
            let q: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.09).sin()).collect();
            let k: Vec<f32> = (0..t * dm).map(|i| (i as f32 * 0.04).cos()).collect();
            let v: Vec<f32> = (0..t * dm).map(|i| (i as f32 * 0.02) - 0.3).collect();
            let scale = 1.0 / (dh as f32).sqrt();

            let mut got = vec![0.0f32; s * dm];
            multi_head_attention(&q, s, t, h, dh, &k, &v, scale, false, &mut got);

            // Référence : chaque tête via attention_ref sur ses colonnes.
            for head in 0..h
            {
                let off = head * dh;
                let mut qh = vec![0.0f32; s * dh];
                let mut kh = vec![0.0f32; t * dh];
                let mut vh = vec![0.0f32; t * dh];
                for r in 0..s
                {
                    qh[r * dh..r * dh + dh].copy_from_slice(&q[r * dm + off..r * dm + off + dh]);
                }
                for r in 0..t
                {
                    kh[r * dh..r * dh + dh].copy_from_slice(&k[r * dm + off..r * dm + off + dh]);
                    vh[r * dh..r * dh + dh].copy_from_slice(&v[r * dm + off..r * dm + off + dh]);
                }
                let want_h = attention_ref(&qh, s, dh, &kh, t, &vh, scale);
                for r in 0..s
                {
                    for e in 0..dh
                    {
                        let g = got[r * dm + off + e];
                        let w = want_h[r * dh + e];
                        assert!(
                            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                            "head {head} r={r} e={e}: {g} vs {w}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn multi_head_causal_matches_per_head() {
        let (s, h, dh) = (6usize, 2usize, 4usize);
        let dm = h * dh;
        let q: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.11).sin()).collect();
        let k: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.06).cos()).collect();
        let v: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.03) - 0.4).collect();
        let scale = 1.0 / (dh as f32).sqrt();

        let mut got = vec![0.0f32; s * dm];
        multi_head_attention(&q, s, s, h, dh, &k, &v, scale, true, &mut got);

        for head in 0..h
        {
            let off = head * dh;
            let mut qh = vec![0.0f32; s * dh];
            let mut kh = vec![0.0f32; s * dh];
            let mut vh = vec![0.0f32; s * dh];
            for r in 0..s
            {
                qh[r * dh..r * dh + dh].copy_from_slice(&q[r * dm + off..r * dm + off + dh]);
                kh[r * dh..r * dh + dh].copy_from_slice(&k[r * dm + off..r * dm + off + dh]);
                vh[r * dh..r * dh + dh].copy_from_slice(&v[r * dm + off..r * dm + off + dh]);
            }
            let want_h = causal_ref(&qh, s, dh, &kh, &vh, scale);
            for r in 0..s
            {
                for e in 0..dh
                {
                    let g = got[r * dm + off + e];
                    let w = want_h[r * dh + e];
                    assert!(
                        (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                        "causal head {head} r={r} e={e}: {g} vs {w}"
                    );
                }
            }
        }
    }
}
