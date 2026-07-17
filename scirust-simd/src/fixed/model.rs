// scirust-simd/src/fixed/model.rs
//
// # Modèle Transformer multi-couche, quantifié déterministe
//
// [`TransformerModel`] empile plusieurs [`super::transformer::TransformerBlock`]
// et expose les deux régimes de l'inférence — le pendant quantifié
// déterministe du module flottant [`crate::model`] (non déterministe) :
//
// * [`TransformerModel::prefill`] — passe la séquence entière à travers tous
//   les blocs (mode « traitement du prompt »), couche par couche.
// * [`TransformerModel::decode_step`] — fait avancer **un** token à travers
//   tous les blocs en s'appuyant sur un cache KV par bloc (mode « génération
//   autoregressive »).
// * [`TransformerModel::generate_hidden`] — boucle de génération *jouet* sur
//   l'état caché (sans vocabulaire/tête de sortie, hors du périmètre de ce
//   crate de noyaux) : préremplit les caches position par position, puis
//   réinjecte à chaque pas le dernier état produit. Exerce le vrai chemin
//   `decode_step` de bout en bout.
//
// ## Équivalence bit-à-bit propagée à toute la pile
//
// [`super::transformer`] établit que, pour **un** bloc, décoder token par
// token via [`super::kv_cache::KvCache`] reproduit bit-à-bit le préremplissage
// causal en bloc. Cette propriété se propage **sans perte** à l'empilement :
// à la position `t`, la couche `L` n'a besoin, pour son attention causale, que
// des sorties de la couche `L` aux positions `0..=t` — qu'elles aient été
// produites par un unique appel `prefill` sur toute la séquence, ou par des
// appels `decode_step` antérieurs (chacun faisant déjà traverser la position
// correspondante à **toute** la pile avant de passer à la suivante). Les deux
// ordres de calcul redonnent donc, couche après couche, exactement les mêmes
// bits (`stack_decode_matches_prefill_bit_exact`) — une garantie strictement
// plus forte que le module flottant [`crate::model`], dont le test équivalent
// (`stack_decode_matches_prefill`) nécessite une tolérance relative `5e-3`.
// Réservé au stockage `i32` (hérité de [`super::transformer`]).

use super::kv_cache::KvCache;
use super::transformer::TransformerBlock;
use super::types::Fixed;

/// Pile de blocs décodeur partageant la même dimension modèle.
pub struct TransformerModel<const FRAC: u32> {
    blocks: Vec<TransformerBlock<FRAC>>,
    d_model: usize,
}

impl<const FRAC: u32> TransformerModel<FRAC> {
    /// Construit un modèle à partir de blocs (non vide, `d_model` homogène).
    ///
    /// Panique si `blocks` est vide, ou si les blocs n'ont pas tous le même
    /// `d_model`.
    #[must_use]
    pub fn new(blocks: Vec<TransformerBlock<FRAC>>) -> Self {
        assert!(
            !blocks.is_empty(),
            "TransformerModel::new : au moins un bloc requis"
        );
        let d_model = blocks[0].d_model;
        assert!(
            blocks.iter().all(|b| b.d_model == d_model),
            "TransformerModel::new : d_model doit être homogène"
        );
        Self { blocks, d_model }
    }

    /// Nombre de blocs (couches).
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Dimension modèle.
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// **Préremplissage** : passe `x` (`s×d_model`, row-major) à travers tous
    /// les blocs, en place, couche par couche.
    ///
    /// `None` si une normalisation d'un bloc rencontre une ligne indéfinie
    /// (cf. [`TransformerBlock::forward`]) — propage le premier échec
    /// rencontré, couches restantes non exécutées. Panique si
    /// `x.len() != s·d_model`.
    #[must_use]
    pub fn prefill(&self, x: &mut [Fixed<i32, FRAC>], s: usize) -> Option<()> {
        for b in &self.blocks
        {
            b.forward(x, s)?;
        }
        Some(())
    }

    /// Alloue un cache KV par bloc, dimensionné pour `cap` positions.
    #[must_use]
    pub fn new_caches(&self, cap: usize) -> Vec<KvCache<FRAC>> {
        self.blocks
            .iter()
            .map(|_| KvCache::new(cap, self.d_model))
            .collect()
    }

    /// **Pas de décodage** : fait avancer le token `x_t` (`d_model`) à la
    /// position `pos` à travers tous les blocs, chacun mettant à jour son
    /// cache. `caches` doit avoir exactement [`Self::n_layers`] entrées.
    ///
    /// `None` dans les mêmes conditions que [`Self::prefill`]. Panique si
    /// `caches.len() != n_layers()`, ou si `x_t.len() != d_model` (cf.
    /// [`TransformerBlock::forward_decode`]).
    #[must_use]
    pub fn decode_step(
        &self,
        x_t: &mut [Fixed<i32, FRAC>],
        pos: usize,
        caches: &mut [KvCache<FRAC>],
    ) -> Option<()> {
        assert_eq!(
            caches.len(),
            self.blocks.len(),
            "TransformerModel::decode_step : {} cache(s) fourni(s) ≠ {} couche(s)",
            caches.len(),
            self.blocks.len()
        );
        for (b, c) in self.blocks.iter().zip(caches.iter_mut())
        {
            b.forward_decode(x_t, pos, c)?;
        }
        Some(())
    }

    /// Boucle de génération *jouet* sur l'état caché (sans vocabulaire) :
    /// traite `prompt` (`prompt_len × d_model`) en préremplissage *via les
    /// caches* (un [`Self::decode_step`] par position, pour peupler les
    /// caches), puis génère `n_new` états en réinjectant à chaque pas le
    /// dernier état produit comme entrée du token suivant. Renvoie les
    /// `n_new` états générés (`n_new × d_model`).
    ///
    /// L'échantillonnage réel exige une tête de dé-projection et un
    /// vocabulaire, hors du périmètre de ce crate de noyaux — cette boucle
    /// exerce le vrai chemin [`Self::decode_step`] de bout en bout.
    /// Déterministe par construction (aucun aléa) : deux appels identiques
    /// produisent des bits identiques.
    ///
    /// `None` dans les mêmes conditions que [`Self::decode_step`]. Panique si
    /// `prompt.len() != prompt_len·d_model`.
    #[must_use]
    pub fn generate_hidden(
        &self,
        prompt: &[Fixed<i32, FRAC>],
        prompt_len: usize,
        n_new: usize,
    ) -> Option<Vec<Fixed<i32, FRAC>>> {
        let d = self.d_model;
        assert_eq!(
            prompt.len(),
            prompt_len * d,
            "TransformerModel::generate_hidden : prompt de longueur {} ≠ {prompt_len}×{d}",
            prompt.len()
        );
        let mut caches = self.new_caches(prompt_len + n_new);

        // Préremplissage via les caches, position par position.
        let mut last = vec![Fixed::zero(); d];
        for t in 0..prompt_len
        {
            let mut row = prompt[t * d..t * d + d].to_vec();
            self.decode_step(&mut row, t, &mut caches)?;
            last = row;
        }

        // Génération : réinjecte le dernier état.
        let mut out = Vec::with_capacity(n_new * d);
        for i in 0..n_new
        {
            let pos = prompt_len + i;
            let mut row = last.clone();
            self.decode_step(&mut row, pos, &mut caches)?;
            out.extend_from_slice(&row);
            last = row;
        }
        Some(out)
    }
}
