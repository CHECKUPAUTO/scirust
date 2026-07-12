//! # Cache KV **quantifié int8** — décodage autoregressif à mémoire réduite
//!
//! Variante de [`crate::kv_cache::KvCache`] qui stocke `K` et `V` en **int8**
//! (÷4 mémoire vs `f32`) — le cache KV est souvent le premier poste mémoire de
//! l'inférence LLM à long contexte, et sa relecture domine la bande passante à
//! chaque token.
//!
//! * **Stockage** : à l'[`QuantizedKvCache::append`], chaque ligne `K`/`V` est
//!   quantifiée **symétriquement par token** (`scale = max|·|/127`) → int8 +
//!   échelle. Deux octets deviennent un demi (int8 vs `f32`).
//! * **Scores `q·kᵀ`** : la requête est quantifiée int8 par tête, puis les
//!   produits scalaires passent par le **dot int8 matériel**
//!   ([`crate::quant::dot_i8i8_i32`] — AVX-512BW sur x86, `dotprod` SDOT sur ARM),
//!   déquantifiés `scale · scale_q · scale_k[j] · dot`.
//! * **Agrégation `p·V`** : `V` est déquantifié à la volée (`scale_v[j]·v_i8`)
//!   dans la somme pondérée par le softmax (`f32`, précision préservée).
//!
//! La sortie approche celle du cache `f32` / de l'attention causale en bloc à la
//! tolérance de quantification près (vérifié dans les tests), pour **¼ de la
//! mémoire** `K`/`V`.

use crate::activations::exp_inplace;
use crate::quant::dot_i8i8_i32;

/// Quantifie une ligne `f32` en int8 symétrique : `scale = max|row|/127`,
/// `out[i] = round(row[i]/scale)`. Renvoie `scale` (1.0 si la ligne est nulle).
fn quantize_row_i8(row: &[f32], out: &mut [i8]) -> f32 {
    debug_assert_eq!(row.len(), out.len());
    let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (o, &v) in out.iter_mut().zip(row)
    {
        let r = (v * inv).round();
        *o = if r >= 127.0
        {
            127
        }
        else if r <= -127.0
        {
            -127
        }
        else
        {
            r as i8
        };
    }
    scale
}

/// Cache clés/valeurs **int8** d'une couche d'attention (têtes concaténées).
pub struct QuantizedKvCache {
    dm: usize,
    cap: usize,
    len: usize,
    k_q: Vec<i8>,      // cap × dm (int8)
    v_q: Vec<i8>,      // cap × dm (int8)
    k_scale: Vec<f32>, // cap (échelle par token)
    v_scale: Vec<f32>, // cap
}

impl QuantizedKvCache {
    /// Nouveau cache int8 pour au plus `cap` positions, dimension modèle `dm`.
    pub fn new(cap: usize, dm: usize) -> Self {
        Self {
            dm,
            cap,
            len: 0,
            k_q: vec![0; cap * dm],
            v_q: vec![0; cap * dm],
            k_scale: vec![0.0; cap],
            v_scale: vec![0.0; cap],
        }
    }

    /// Nombre de positions mémorisées.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` si aucun token n'a été empilé.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacité (longueur de séquence maximale).
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Vide le cache (réutilisable pour une nouvelle séquence).
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Octets occupés par `K`+`V` (int8) — vs `2·cap·dm·4` pour un cache `f32`.
    pub fn kv_bytes(&self) -> usize {
        2 * self.cap * self.dm
    }

    /// Empile (en les quantifiant) les `K`/`V` (longueur `dm`) du nouveau token.
    pub fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        assert_eq!(k_row.len(), self.dm, "append: K row length != dm");
        assert_eq!(v_row.len(), self.dm, "append: V row length != dm");
        assert!(self.len < self.cap, "append: cache plein ({})", self.cap);
        let off = self.len * self.dm;
        self.k_scale[self.len] = quantize_row_i8(k_row, &mut self.k_q[off..off + self.dm]);
        self.v_scale[self.len] = quantize_row_i8(v_row, &mut self.v_q[off..off + self.dm]);
        self.len += 1;
    }

    /// Attention multi-tête du query `q` (longueur `dm`) sur tout le cache int8 :
    /// `out = softmax(scale · q·Kᵀ) · V` par tête. Analogue quantifié de
    /// [`crate::kv_cache::KvCache::decode_step`].
    pub fn decode_step(
        &self,
        q: &[f32],
        n_heads: usize,
        d_head: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        assert_eq!(q.len(), self.dm, "decode_step: q length != dm");
        assert_eq!(out.len(), self.dm, "decode_step: out length != dm");
        assert_eq!(
            n_heads * d_head,
            self.dm,
            "decode_step: n_heads·d_head != dm"
        );
        let len = self.len;
        let dm = self.dm;

        let mut scores = vec![0.0f32; len.max(1)];
        let mut q_i8 = vec![0i8; d_head];
        for hh in 0..n_heads
        {
            let off = hh * d_head;
            let q_head = &q[off..off + d_head];
            let o_head = &mut out[off..off + d_head];
            o_head.iter_mut().for_each(|x| *x = 0.0);
            if len == 0
            {
                continue;
            }

            // Quantifie la requête (par tête) puis scores int8 matériels.
            let sq = quantize_row_i8(q_head, &mut q_i8);
            for (j, sc) in scores[..len].iter_mut().enumerate()
            {
                let k_head = &self.k_q[j * dm + off..j * dm + off + d_head];
                let dot = dot_i8i8_i32(&q_i8, k_head);
                *sc = scale * sq * self.k_scale[j] * dot as f32;
            }

            // Softmax stable.
            let m = scores[..len]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            for sc in scores[..len].iter_mut()
            {
                *sc -= m;
            }
            exp_inplace(&mut scores[..len]);
            let inv = 1.0 / scores[..len].iter().sum::<f32>();

            // o = Σ_j (p_j/sum)·dequant(v_j). V déquantifié à la volée.
            for (j, &e) in scores[..len].iter().enumerate()
            {
                let coeff = e * inv * self.v_scale[j];
                let v_head = &self.v_q[j * dm + off..j * dm + off + d_head];
                for (oe, &vq) in o_head.iter_mut().zip(v_head)
                {
                    *oe += coeff * vq as f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::multi_head_attention;

    #[test]
    fn quantized_incremental_close_to_batched() {
        // Le décodage incrémental via le cache KV **int8** doit approcher (à la
        // tolérance de quantification) l'attention causale en bloc `f32`.
        for &(s, h, dh) in &[
            (1usize, 1usize, 4usize),
            (4, 2, 8),
            (12, 3, 16),
            (40, 4, 16),
        ]
        {
            let dm = h * dh;
            // Données décorrélées (produit scalaire non dégénéré).
            let mut seed = 0x1234_5678u64;
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0
            };
            let q: Vec<f32> = (0..s * dm).map(|_| rnd()).collect();
            let k: Vec<f32> = (0..s * dm).map(|_| rnd()).collect();
            let v: Vec<f32> = (0..s * dm).map(|_| rnd()).collect();
            let scale = 1.0 / (dh as f32).sqrt();

            let mut expected = vec![0.0f32; s * dm];
            multi_head_attention(&q, s, s, h, dh, &k, &v, scale, true, &mut expected);

            let mut cache = QuantizedKvCache::new(s, dm);
            let mut got = vec![0.0f32; s * dm];
            for i in 0..s
            {
                cache.append(&k[i * dm..i * dm + dm], &v[i * dm..i * dm + dm]);
                let mut o = vec![0.0f32; dm];
                cache.decode_step(&q[i * dm..i * dm + dm], h, dh, scale, &mut o);
                got[i * dm..i * dm + dm].copy_from_slice(&o);
            }

            // Erreur relative RMS globale < 5 % (int8 sur K, V et q).
            let mut num = 0f64;
            let mut den = 0f64;
            for idx in 0..s * dm
            {
                num += (got[idx] - expected[idx]).powi(2) as f64;
                den += (expected[idx] as f64).powi(2);
            }
            let rel = (num / den).sqrt();
            assert!(rel < 0.05, "s={s} h={h} dh={dh}: erreur RMS {rel}");
        }
    }

    #[test]
    fn memory_is_quarter_of_f32() {
        let (cap, dm) = (2048usize, 1024usize);
        let cache = QuantizedKvCache::new(cap, dm);
        let f32_bytes = 2 * cap * dm * 4;
        assert_eq!(cache.kv_bytes(), f32_bytes / 4);
    }

    #[test]
    fn len_clear_and_capacity() {
        let mut cache = QuantizedKvCache::new(3, 4);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 3);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
    }
}
