//! # Cache KV — décodage autoregressif incrémental (Transformers)
//!
//! En génération token-par-token, l'attention causale de la position `t` porte
//! sur **toutes** les clés/valeurs `0..=t`. Sans cache, chaque nouveau token
//! recalcule les `K`/`V` de tout le préfixe — coût `O(t·d)` par pas, `O(s²·d)`
//! au total. Le **cache KV** mémorise les `K`/`V` déjà projetés et n'ajoute que
//! la ligne du nouveau token, ramenant chaque pas à `O(t·d)` de simple
//! attention (pas de reprojection) : c'est l'optimisation clé de l'inférence
//! LLM.
//!
//! [`KvCache`] stocke `K` et `V` (`cap × d_model`, têtes concaténées) et expose
//! [`KvCache::append`] (empile un token) puis [`KvCache::decode_step`]
//! (attention multi-tête du nouveau query sur tout le cache). Les produits
//! scalaires et l'accumulation passent par les kernels dispatchés
//! (AVX-512/AVX2/NEON/scalaire) ; le softmax réutilise l'`exp` vectorisée.
//!
//! **Équivalence vérifiée** : nourrir le cache un token à la fois produit,
//! ligne par ligne, exactement la sortie de l'attention causale calculée en
//! bloc sur toute la séquence (test `incremental_matches_batched`).

use crate::activations::exp_inplace;
use crate::dispatch::runtime_backend;

/// Cache clés/valeurs d'une couche d'attention (têtes concaténées, row-major).
pub struct KvCache {
    dm: usize,   // dimension modèle = n_heads · d_head
    cap: usize,  // longueur de séquence maximale
    len: usize,  // nombre de positions mémorisées
    k: Vec<f32>, // cap × dm
    v: Vec<f32>, // cap × dm
}

impl KvCache {
    /// Nouveau cache pour au plus `cap` positions et une dimension modèle `dm`.
    pub fn new(cap: usize, dm: usize) -> Self {
        Self {
            dm,
            cap,
            len: 0,
            k: vec![0.0; cap * dm],
            v: vec![0.0; cap * dm],
        }
    }

    /// Nombre de positions actuellement mémorisées.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` si aucun token n'a encore été empilé.
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

    /// Empile les `K`/`V` (longueur `dm` chacun) du nouveau token.
    pub fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        assert_eq!(k_row.len(), self.dm, "append: K row length != dm");
        assert_eq!(v_row.len(), self.dm, "append: V row length != dm");
        assert!(
            self.len < self.cap,
            "append: cache plein ({} positions)",
            self.cap
        );
        let off = self.len * self.dm;
        self.k[off..off + self.dm].copy_from_slice(k_row);
        self.v[off..off + self.dm].copy_from_slice(v_row);
        self.len += 1;
    }

    /// Attention multi-tête du query `q` (longueur `dm`) sur **tout** le cache
    /// courant : `out = softmax(scale · q·Kᵀ) · V` par tête, écrit dans `out`
    /// (longueur `dm`). À appeler après [`KvCache::append`] du token courant
    /// (il s'attend donc lui-même — masquage causal implicite).
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
        let backend = runtime_backend();

        let mut scores = vec![0.0f32; len.max(1)];
        for hh in 0..n_heads
        {
            let off = hh * d_head;
            let q_head = &q[off..off + d_head];
            let o_head = &mut out[off..off + d_head];
            o_head.iter_mut().for_each(|x| *x = 0.0);

            if len == 0
            {
                continue; // rien à attendre (cache vide)
            }

            // scores[j] = scale · q·k_j sur les positions mémorisées.
            for (j, sc) in scores[..len].iter_mut().enumerate()
            {
                let k_head = &self.k[j * dm + off..j * dm + off + d_head];
                *sc = scale * backend.sdot_f32(q_head, k_head);
            }

            // softmax stable (max, exp vectorisée, normalisation).
            let m = scores[..len]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            for sc in scores[..len].iter_mut()
            {
                *sc -= m;
            }
            exp_inplace(&mut scores[..len]);
            let sum: f32 = scores[..len].iter().sum();
            let inv = 1.0 / sum;

            // o = Σ_j (p_j/sum) · v_j.
            for (j, &e) in scores[..len].iter().enumerate()
            {
                let v_head = &self.v[j * dm + off..j * dm + off + d_head];
                backend.saxpy_f32(e * inv, v_head, o_head);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::multi_head_attention;

    #[test]
    fn incremental_matches_batched() {
        // Le décodage incrémental via le cache KV doit reproduire, ligne par
        // ligne, l'attention causale multi-tête calculée en bloc.
        for &(s, h, dh) in &[(1usize, 1usize, 2usize), (4, 2, 4), (10, 3, 6), (33, 2, 8)]
        {
            let dm = h * dh;
            let q: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.07).sin()).collect();
            let k: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.05).cos() * 2.0).collect();
            let v: Vec<f32> = (0..s * dm).map(|i| (i as f32 * 0.03) - 0.5).collect();
            let scale = 1.0 / (dh as f32).sqrt();

            // Référence : attention causale en bloc.
            let mut expected = vec![0.0f32; s * dm];
            multi_head_attention(&q, s, s, h, dh, &k, &v, scale, true, &mut expected);

            // Incrémental : empile puis décode chaque token.
            let mut cache = KvCache::new(s, dm);
            let mut got = vec![0.0f32; s * dm];
            for i in 0..s
            {
                cache.append(&k[i * dm..i * dm + dm], &v[i * dm..i * dm + dm]);
                let mut o = vec![0.0f32; dm];
                cache.decode_step(&q[i * dm..i * dm + dm], h, dh, scale, &mut o);
                got[i * dm..i * dm + dm].copy_from_slice(&o);
            }

            for idx in 0..s * dm
            {
                let tol = 1e-4 * (1.0 + expected[idx].abs());
                assert!(
                    (got[idx] - expected[idx]).abs() <= tol,
                    "s={s} h={h} dh={dh} idx={idx}: incr {} vs batch {}",
                    got[idx],
                    expected[idx]
                );
            }
        }
    }

    #[test]
    fn first_token_attends_to_itself() {
        // Un seul token en cache → softmax trivial → out == v (par tête).
        let (h, dh) = (2usize, 3usize);
        let dm = h * dh;
        let q: Vec<f32> = (0..dm).map(|i| (i as f32).sin()).collect();
        let kk: Vec<f32> = (0..dm).map(|i| (i as f32).cos()).collect();
        let vv: Vec<f32> = (0..dm).map(|i| i as f32 * 0.5).collect();
        let mut cache = KvCache::new(4, dm);
        cache.append(&kk, &vv);
        let mut out = vec![0.0f32; dm];
        cache.decode_step(&q, h, dh, 0.3, &mut out);
        for e in 0..dm
        {
            assert!(
                (out[e] - vv[e]).abs() <= 1e-5,
                "out[{e}] {} != v {}",
                out[e],
                vv[e]
            );
        }
    }

    #[test]
    fn len_clear_and_capacity() {
        let mut cache = KvCache::new(3, 4);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 3);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
    }
}
