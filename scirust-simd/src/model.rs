//! # Modèle Transformer multi-bloc (prefill + génération)
//!
//! Empile plusieurs [`TransformerBlock`] et expose les deux régimes de
//! l'inférence :
//!
//! * **`prefill`** — passe la séquence entière à travers tous les blocs (mode
//!   « traitement du prompt »).
//! * **`decode_step`** — fait avancer **un** token à travers tous les blocs en
//!   s'appuyant sur un cache KV par bloc (mode « génération autoregressive »).
//!
//! Propriété garantie : décoder une séquence token-par-token (via `decode_step`
//! et les caches) reproduit exactement, ligne par ligne, le `prefill` de la
//! séquence entière — l'équivalence *prefill ≡ decode* propagée à travers toute
//! la pile (test `stack_decode_matches_prefill`). C'est ce qui autorise à
//! traiter le prompt en bloc puis à générer incrémentalement sans divergence.

use crate::kv_cache::KvCache;
use crate::transformer::TransformerBlock;

/// Pile de blocs décodeur partageant la même dimension modèle.
pub struct TransformerModel<'a> {
    blocks: Vec<TransformerBlock<'a>>,
    d_model: usize,
}

impl<'a> TransformerModel<'a> {
    /// Construit un modèle à partir de blocs (non vide, `d_model` homogène).
    pub fn new(blocks: Vec<TransformerBlock<'a>>) -> Self {
        assert!(
            !blocks.is_empty(),
            "TransformerModel: au moins un bloc requis"
        );
        let d_model = blocks[0].d_model;
        assert!(
            blocks.iter().all(|b| b.d_model == d_model),
            "TransformerModel: d_model doit être homogène"
        );
        Self { blocks, d_model }
    }

    /// Nombre de blocs (couches).
    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Dimension modèle.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// **Prefill** : passe `x` (`s×d_model`, row-major) à travers tous les blocs,
    /// en place.
    pub fn prefill(&self, x: &mut [f32], s: usize) {
        for b in &self.blocks
        {
            b.forward(x, s);
        }
    }

    /// Alloue un cache KV par bloc, dimensionné pour `cap` positions.
    pub fn new_caches(&self, cap: usize) -> Vec<KvCache> {
        self.blocks
            .iter()
            .map(|_| KvCache::new(cap, self.d_model))
            .collect()
    }

    /// **Pas de décodage** : fait avancer le token `x_t` (`d_model`) à la
    /// position `pos` à travers tous les blocs, chacun mettant à jour son cache.
    /// `caches` doit avoir exactement `n_layers()` entrées (voir [`Self::new_caches`]).
    pub fn decode_step(&self, x_t: &mut [f32], pos: usize, caches: &mut [KvCache]) {
        assert_eq!(
            caches.len(),
            self.blocks.len(),
            "decode_step: un cache par bloc requis"
        );
        for (b, c) in self.blocks.iter().zip(caches.iter_mut())
        {
            b.forward_decode(x_t, pos, c);
        }
    }

    /// Démonstration de **boucle de génération** sur l'état caché (sans vocab) :
    /// traite un prompt `prompt` (`prompt_len × d_model`) en prefill *via les
    /// caches* (un `decode_step` par position, pour peupler les caches), puis
    /// génère `n_new` états en réinjectant à chaque pas le dernier état produit
    /// comme entrée du token suivant. Renvoie les `n_new` états générés
    /// (`n_new × d_model`).
    ///
    /// C'est une boucle *jouet* (l'échantillonnage réel exige une tête de
    /// dé-projection + un vocabulaire, hors de ce crate de noyaux), mais elle
    /// exerce le vrai chemin `decode_step` de bout en bout.
    pub fn generate_hidden(&self, prompt: &[f32], prompt_len: usize, n_new: usize) -> Vec<f32> {
        let d = self.d_model;
        assert_eq!(
            prompt.len(),
            prompt_len * d,
            "generate_hidden: prompt shape"
        );
        let mut caches = self.new_caches(prompt_len + n_new);

        // Prefill via les caches, position par position.
        let mut last = vec![0.0f32; d];
        for t in 0..prompt_len
        {
            let mut row = prompt[t * d..t * d + d].to_vec();
            self.decode_step(&mut row, t, &mut caches);
            last = row;
        }

        // Génération : réinjecte le dernier état.
        let mut out = Vec::with_capacity(n_new * d);
        for i in 0..n_new
        {
            let pos = prompt_len + i;
            let mut row = last.clone();
            self.decode_step(&mut row, pos, &mut caches);
            out.extend_from_slice(&row);
            last = row;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construit un bloc à poids déterministes (les slices vivent dans `store`).
    struct Weights {
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        norm1: Vec<f32>,
        norm2: Vec<f32>,
    }

    fn make_weights(d: usize, dff: usize, seed: f32) -> Weights {
        let mk = |n: usize, s: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 + s) * 0.017).sin() * 0.4)
                .collect()
        };
        Weights {
            wq: mk(d * d, seed + 1.0),
            wk: mk(d * d, seed + 2.0),
            wv: mk(d * d, seed + 3.0),
            wo: mk(d * d, seed + 4.0),
            w1: mk(d * dff, seed + 5.0),
            b1: mk(dff, seed + 6.0),
            w2: mk(dff * d, seed + 7.0),
            norm1: (0..d).map(|i| 1.0 + i as f32 * 0.01).collect(),
            norm2: (0..d).map(|i| 0.9 + i as f32 * 0.02).collect(),
        }
    }

    fn block<'a>(w: &'a Weights, d: usize, h: usize, dff: usize) -> TransformerBlock<'a> {
        TransformerBlock {
            d_model: d,
            n_heads: h,
            d_ff: dff,
            wq: &w.wq,
            wk: &w.wk,
            wv: &w.wv,
            wo: &w.wo,
            w1: &w.w1,
            b1: &w.b1,
            w2: &w.w2,
            norm1: &w.norm1,
            norm2: &w.norm2,
            eps: 1e-5,
            rope_base: 10000.0,
            causal: true,
        }
    }

    #[test]
    fn stack_decode_matches_prefill() {
        let (s, d, h, dff, n_layers) = (6usize, 8usize, 2usize, 16usize, 3usize);
        let store: Vec<Weights> = (0..n_layers)
            .map(|l| make_weights(d, dff, l as f32 * 10.0))
            .collect();
        let model = TransformerModel::new(store.iter().map(|w| block(w, d, h, dff)).collect());
        assert_eq!(model.n_layers(), n_layers);

        let x0: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.05).cos()).collect();

        // Prefill de toute la séquence à travers la pile.
        let mut prefill = x0.clone();
        model.prefill(&mut prefill, s);

        // Décodage incrémental token par token à travers la pile.
        let mut caches = model.new_caches(s);
        let mut decoded = vec![0.0f32; s * d];
        for t in 0..s
        {
            let mut row = x0[t * d..t * d + d].to_vec();
            model.decode_step(&mut row, t, &mut caches);
            decoded[t * d..t * d + d].copy_from_slice(&row);
        }

        for i in 0..s * d
        {
            let tol = 5e-3 * (1.0 + prefill[i].abs());
            assert!(
                (decoded[i] - prefill[i]).abs() <= tol,
                "idx {i}: decode {} vs prefill {}",
                decoded[i],
                prefill[i]
            );
        }
    }

    #[test]
    fn generation_loop_runs_finite_and_deterministic() {
        let (d, h, dff, n_layers) = (8usize, 2usize, 16usize, 2usize);
        let store: Vec<Weights> = (0..n_layers)
            .map(|l| make_weights(d, dff, l as f32 * 5.0))
            .collect();
        let model = TransformerModel::new(store.iter().map(|w| block(w, d, h, dff)).collect());

        let prompt_len = 3;
        let n_new = 4;
        let prompt: Vec<f32> = (0..prompt_len * d)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let a = model.generate_hidden(&prompt, prompt_len, n_new);
        assert_eq!(a.len(), n_new * d);
        assert!(a.iter().all(|x| x.is_finite()), "sorties non finies");

        // Déterministe : deux exécutions identiques.
        let b = model.generate_hidden(&prompt, prompt_len, n_new);
        assert_eq!(a, b, "génération non déterministe");
    }
}
