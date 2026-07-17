// scirust-simd/src/fixed/kv_cache.rs
//
// # Cache KV quantifié déterministe — décodage autoregressif incrémental
//
// Le pendant **quantifié déterministe** du module flottant [`crate::kv_cache`]
// (non déterministe). En génération token-par-token, l'attention causale de
// la position `t` porte sur **toutes** les clés/valeurs `0..=t`. Sans cache,
// chaque nouveau token recalcule les `K`/`V` de tout le préfixe — coût
// `O(t·d)` par pas, `O(s²·d)` au total. [`KvCache`] mémorise les `K`/`V` déjà
// projetés et n'ajoute que la ligne du nouveau token, ramenant chaque pas à
// `O(t·d)` de simple attention (pas de reprojection).
//
// Construit directement sur [`super::attention::multi_head_attention`] : le
// masquage causal est **implicite**, pas calculé — la position `t` n'a
// simplement pas encore de clés/valeurs au-delà d'elle-même dans le cache
// tant que [`KvCache::append`] du token courant n'a pas eu lieu.
//
// **Équivalence vérifiée** : nourrir le cache un token à la fois produit,
// ligne par ligne, exactement la sortie de l'attention causale multi-tête
// calculée en bloc sur toute la séquence (test `incremental_matches_batched`).
// Réservé au stockage `i32` (hérité de [`super::attention`]).

use super::attention::multi_head_attention;
use super::types::Fixed;

/// Cache clés/valeurs d'une couche d'attention (têtes concaténées, row-major),
/// virgule fixe déterministe.
pub struct KvCache<const FRAC: u32> {
    dm: usize,                // dimension modèle = n_heads · d_head
    cap: usize,               // longueur de séquence maximale
    len: usize,               // nombre de positions mémorisées
    k: Vec<Fixed<i32, FRAC>>, // cap × dm
    v: Vec<Fixed<i32, FRAC>>, // cap × dm
}

impl<const FRAC: u32> KvCache<FRAC> {
    /// Nouveau cache pour au plus `cap` positions et une dimension modèle `dm`.
    #[must_use]
    pub fn new(cap: usize, dm: usize) -> Self {
        Self {
            dm,
            cap,
            len: 0,
            k: vec![Fixed::zero(); cap * dm],
            v: vec![Fixed::zero(); cap * dm],
        }
    }

    /// Nombre de positions actuellement mémorisées.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` si aucun token n'a encore été empilé.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacité (longueur de séquence maximale).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Vide le cache (réutilisable pour une nouvelle séquence).
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Empile les `K`/`V` (longueur `dm` chacun) du nouveau token.
    ///
    /// Panique si `k_row`/`v_row` n'ont pas la longueur `dm`, ou si le cache
    /// est plein (`len == capacity()`).
    pub fn append(&mut self, k_row: &[Fixed<i32, FRAC>], v_row: &[Fixed<i32, FRAC>]) {
        assert_eq!(
            k_row.len(),
            self.dm,
            "KvCache::append : K de longueur {} ≠ {}",
            k_row.len(),
            self.dm
        );
        assert_eq!(
            v_row.len(),
            self.dm,
            "KvCache::append : V de longueur {} ≠ {}",
            v_row.len(),
            self.dm
        );
        assert!(
            self.len < self.cap,
            "KvCache::append : cache plein ({} positions)",
            self.cap
        );
        let off = self.len * self.dm;
        self.k[off..off + self.dm].copy_from_slice(k_row);
        self.v[off..off + self.dm].copy_from_slice(v_row);
        self.len += 1;
    }

    /// Attention multi-tête, déterministe, du query `q` (longueur `dm`) sur
    /// **tout** le cache courant : `out = softmax(scale · q·Kᵀ) · V` par tête
    /// (longueur `dm`). À appeler après [`KvCache::append`] du token courant
    /// (il s'attend donc lui-même — masquage causal implicite, cf. en-tête de
    /// module). `dm` nulle-longueur si le cache est vide.
    ///
    /// Panique si `q.len() != dm` ou `n_heads·d_head != dm`.
    #[must_use]
    pub fn decode_step(
        &self,
        q: &[Fixed<i32, FRAC>],
        n_heads: usize,
        d_head: usize,
        scale: Fixed<i32, FRAC>,
    ) -> Vec<Fixed<i32, FRAC>> {
        assert_eq!(
            q.len(),
            self.dm,
            "KvCache::decode_step : q de longueur {} ≠ {}",
            q.len(),
            self.dm
        );
        assert_eq!(
            n_heads * d_head,
            self.dm,
            "KvCache::decode_step : n_heads·d_head ({}) ≠ dm ({})",
            n_heads * d_head,
            self.dm
        );
        if self.len == 0
        {
            return vec![Fixed::zero(); self.dm];
        }
        multi_head_attention(
            q,
            1,
            self.len,
            n_heads,
            d_head,
            &self.k[..self.len * self.dm],
            &self.v[..self.len * self.dm],
            scale,
            false,
        )
    }
}
