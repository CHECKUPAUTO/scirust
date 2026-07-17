// scirust-simd/src/fixed/transformer.rs
//
// # Bloc Transformer décodeur, quantifié déterministe
//
// [`TransformerBlock`] assemble **toutes** les briques `fixed::` construites
// pour les séquences/Transformers en un vrai bloc décodeur **pre-norm**
// (style LLaMA/GPT-NeoX), le pendant quantifié déterministe du module
// flottant [`crate::transformer`] (non déterministe) :
//
// ```text
// h  = RMSNorm(x, γ₁)
// q,k,v = h·Wqᵀ, h·Wkᵀ, h·Wvᵀ         (projections, Linear::forward_batch)
// q,k   = RoPE(q), RoPE(k)            (par tête)
// a  = MultiHeadAttention(q,k,v, causal)
// x  = x + a·Woᵀ                      (projection de sortie + résidu)
// h₂ = RMSNorm(x, γ₂)
// f  = SiLU(h₂·W₁ᵀ + b₁)·W₂ᵀ          (FFN, porte SiLU)
// x  = x + f                          (résidu)
// ```
//
// Rien de numériquement nouveau n'est introduit ici : chaque étape délègue à
// une brique déjà construite et testée dans une session précédente —
// [`super::layer::Linear`] (projections **et** FFN, biais compris),
// [`super::norm::rmsnorm`] (les deux normalisations), [`super::norm::rope_apply`]
// (encodage positionnel rotatif, via le petit adaptateur [`rope_apply_heads`]
// ci-dessous pour l'appliquer indépendamment à chaque tête),
// [`super::attention::multi_head_attention`] (attention, préremplissage par
// lot) et [`super::kv_cache::KvCache`] (attention, décodage incrémental). Ce
// module est un **assemblage**, pas un nouvel algorithme numérique.
//
// ## `rope_apply_heads` — RoPE par tête
//
// [`super::norm::rope_apply`] fait tourner les paires `(x[2i], x[2i+1])` sur
// **toute la largeur** de chaque ligne (angle dépendant de `i` relatif à `d`
// entier). Mais dans un bloc multi-tête, chaque tête doit tourner ses propres
// paires **indépendamment**, avec un angle dépendant de `i` relatif à
// `d_head` (pas `d_model`) — sinon la fréquence de rotation d'une tête serait
// fausse dès que `n_heads > 1`. [`rope_apply_heads`] extrait donc chaque tête
// dans un tampon contigu réutilisé (même technique d'extraction que
// [`super::attention::multi_head_attention`]), délègue à [`super::norm::rope_apply`]
// sur ce tampon (`d = d_head`), puis recopie — aucune nouvelle trigonométrie.
//
// ## Équivalence bit-à-bit préremplissage / décodage incrémental
//
// Comme pour [`super::kv_cache`] (dont c'est une conséquence directe), nourrir
// [`TransformerBlock::forward_decode`] un token à la fois reproduit, ligne par
// ligne, **exactement** (bit-à-bit, pas à une tolérance près) la sortie de
// [`TransformerBlock::forward`] causal sur toute la séquence
// (`decode_matches_prefill_bit_exact`) : chaque brique composée est elle-même
// **locale à sa ligne** — `rmsnorm`/`Linear::forward` d'une ligne ne dépend
// que de cette ligne, et `rope_apply`/l'attention via `KvCache` ne dépendent
// que de la position absolue, jamais du regroupement en lot. C'est une
// garantie strictement plus forte que le module flottant
// [`crate::transformer`], dont le test équivalent (`decode_matches_prefill`)
// nécessite une tolérance relative `2e-3` (somme flottante non associative).
// Réservé au stockage `i32` (hérité de [`super::attention`]/[`super::kv_cache`]).

use super::activation::silu;
use super::attention::multi_head_attention;
use super::kv_cache::KvCache;
use super::layer::Linear;
use super::norm::{rmsnorm, rope_apply};
use super::traits::{NumericScalar, RealScalar};
use super::types::Fixed;

/// Applique RoPE **indépendamment à chaque tête** de `x` (`s×(h·d_head)`,
/// têtes concaténées le long des colonnes), en place.
///
/// La ligne `r` est à la position `pos_offset + r`. Chaque tête (`d_head`
/// colonnes contiguës) est extraite dans un tampon réutilisé, tournée via
/// [`super::norm::rope_apply`] (avec `d = d_head`, donc la fréquence correcte
/// par tête), puis recopiée. Réservé au stockage `i32`. Panique si
/// `x.len() != s·h·d_head` ou si `d_head` est impair.
pub fn rope_apply_heads<const FRAC: u32>(
    x: &mut [Fixed<i32, FRAC>],
    s: usize,
    h: usize,
    d_head: usize,
    base: Fixed<i32, FRAC>,
    pos_offset: usize,
) {
    let dm = h * d_head;
    assert_eq!(
        x.len(),
        s * dm,
        "rope_apply_heads : x de longueur {} ≠ {s}×{dm}",
        x.len()
    );

    let mut head_buf = vec![Fixed::zero(); s * d_head];
    for head in 0..h
    {
        let off = head * d_head;
        for r in 0..s
        {
            head_buf[r * d_head..r * d_head + d_head]
                .copy_from_slice(&x[r * dm + off..r * dm + off + d_head]);
        }
        rope_apply(&mut head_buf, s, d_head, base, pos_offset);
        for r in 0..s
        {
            x[r * dm + off..r * dm + off + d_head]
                .copy_from_slice(&head_buf[r * d_head..r * d_head + d_head]);
        }
    }
}

/// Bloc décodeur Transformer **pre-norm**, quantifié déterministe (cf.
/// en-tête de module pour l'assemblage et les conventions de formes).
///
/// `wq`/`wk`/`wv`/`wo` sont des projections `d_model → d_model` ; `w1` est
/// `d_model → d_ff` (porte SiLU) et `w2` est `d_ff → d_model`. `norm1`/`norm2`
/// sont les gains RMSNorm (`d_model` éléments chacun).
pub struct TransformerBlock<const FRAC: u32> {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub wq: Linear<Fixed<i32, FRAC>>,
    pub wk: Linear<Fixed<i32, FRAC>>,
    pub wv: Linear<Fixed<i32, FRAC>>,
    pub wo: Linear<Fixed<i32, FRAC>>,
    pub w1: Linear<Fixed<i32, FRAC>>,
    pub w2: Linear<Fixed<i32, FRAC>>,
    pub norm1: Vec<Fixed<i32, FRAC>>,
    pub norm2: Vec<Fixed<i32, FRAC>>,
    pub eps: Fixed<i32, FRAC>,
    pub rope_base: Fixed<i32, FRAC>,
    pub causal: bool,
}

impl<const FRAC: u32> TransformerBlock<FRAC> {
    /// Construit un bloc, en validant la cohérence des formes des couches
    /// fournies (`in_features`/`out_features` de chaque [`Linear`], longueur
    /// des gains de normalisation) contre `d_model`/`d_ff`.
    ///
    /// Panique en cas d'incohérence.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        wq: Linear<Fixed<i32, FRAC>>,
        wk: Linear<Fixed<i32, FRAC>>,
        wv: Linear<Fixed<i32, FRAC>>,
        wo: Linear<Fixed<i32, FRAC>>,
        w1: Linear<Fixed<i32, FRAC>>,
        w2: Linear<Fixed<i32, FRAC>>,
        norm1: Vec<Fixed<i32, FRAC>>,
        norm2: Vec<Fixed<i32, FRAC>>,
        eps: Fixed<i32, FRAC>,
        rope_base: Fixed<i32, FRAC>,
        causal: bool,
    ) -> Self {
        assert_eq!(
            d_model % n_heads,
            0,
            "TransformerBlock::new : d_model ({d_model}) non divisible par n_heads ({n_heads})"
        );
        for (name, lin, want_in, want_out) in [
            ("wq", &wq, d_model, d_model),
            ("wk", &wk, d_model, d_model),
            ("wv", &wv, d_model, d_model),
            ("wo", &wo, d_model, d_model),
            ("w1", &w1, d_model, d_ff),
            ("w2", &w2, d_ff, d_model),
        ]
        {
            assert_eq!(
                lin.in_features(),
                want_in,
                "TransformerBlock::new : {name}.in_features() = {} ≠ {want_in}",
                lin.in_features()
            );
            assert_eq!(
                lin.out_features(),
                want_out,
                "TransformerBlock::new : {name}.out_features() = {} ≠ {want_out}",
                lin.out_features()
            );
        }
        assert_eq!(
            norm1.len(),
            d_model,
            "TransformerBlock::new : norm1 de longueur {} ≠ {d_model}",
            norm1.len()
        );
        assert_eq!(
            norm2.len(),
            d_model,
            "TransformerBlock::new : norm2 de longueur {} ≠ {d_model}",
            norm2.len()
        );
        Self {
            d_model,
            n_heads,
            d_ff,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            norm1,
            norm2,
            eps,
            rope_base,
            causal,
        }
    }

    /// Échelle `1/√d_head` de l'attention produit-scalaire.
    fn attention_scale(&self) -> Fixed<i32, FRAC> {
        let d_head = self.d_model / self.n_heads;
        Fixed::<i32, FRAC>::from_i32(d_head as i32).sqrt().recip()
    }

    /// Propagation avant, **préremplissage par lot** : `x` est le flux
    /// résiduel `s×d_model` (row-major), mis à jour en place.
    ///
    /// `None` si une normalisation rencontre une ligne indéfinie (rms ou
    /// écart-type nul, ou débordement de division — cf.
    /// [`super::norm::rmsnorm`]). Panique si `x.len() != s·d_model`.
    #[must_use]
    pub fn forward(&self, x: &mut [Fixed<i32, FRAC>], s: usize) -> Option<()> {
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h;
        assert_eq!(
            x.len(),
            s * d,
            "TransformerBlock::forward : x de longueur {} ≠ {s}×{d}",
            x.len()
        );

        // ---- Sous-bloc attention (pre-norm) ----
        let hn = rmsnorm(x, s, d, &self.norm1, self.eps)?;
        let mut q = self.wq.forward_batch(&hn, s);
        let mut k = self.wk.forward_batch(&hn, s);
        let v = self.wv.forward_batch(&hn, s);

        rope_apply_heads(&mut q, s, h, dh, self.rope_base, 0);
        rope_apply_heads(&mut k, s, h, dh, self.rope_base, 0);

        let attn =
            multi_head_attention(&q, s, s, h, dh, &k, &v, self.attention_scale(), self.causal);
        let o = self.wo.forward_batch(&attn, s);
        for (xi, oi) in x.iter_mut().zip(&o)
        {
            *xi += *oi; // résidu
        }

        // ---- Sous-bloc FFN (pre-norm) ----
        let hn2 = rmsnorm(x, s, d, &self.norm2, self.eps)?;
        let f1 = self.w1.forward_batch_activated(&hn2, s, silu);
        let f2 = self.w2.forward_batch(&f1, s);
        for (xi, fi) in x.iter_mut().zip(&f2)
        {
            *xi += *fi; // résidu
        }
        Some(())
    }

    /// **Pas de décodage incrémental** (génération autoregressive) : traite
    /// un unique nouveau token `x_t` (longueur `d_model`) à la position
    /// absolue `pos`, en s'appuyant sur `cache` pour l'attention causale sur
    /// tout le préfixe déjà vu. Met `x_t` à jour en place et empile ses
    /// `K`/`V` dans le cache.
    ///
    /// Équivaut, ligne par ligne et **bit-à-bit**, au chemin *préremplissage*
    /// [`Self::forward`] (causal) exécuté sur toute la séquence — mais en
    /// `O(pos·d)` par token au lieu de recalculer le préfixe (cf. en-tête de
    /// module, `decode_matches_prefill_bit_exact`).
    ///
    /// `None` dans les mêmes conditions que [`Self::forward`]. Panique si
    /// `x_t.len() != d_model`, ou si le cache est plein ou de dimension
    /// modèle différente (cf. [`super::kv_cache::KvCache::append`]).
    #[must_use]
    pub fn forward_decode(
        &self,
        x_t: &mut [Fixed<i32, FRAC>],
        pos: usize,
        cache: &mut KvCache<FRAC>,
    ) -> Option<()> {
        let d = self.d_model;
        let h = self.n_heads;
        let dh = d / h;
        assert_eq!(
            x_t.len(),
            d,
            "TransformerBlock::forward_decode : x_t de longueur {} ≠ {d}",
            x_t.len()
        );

        // ---- Attention (pre-norm), 1 token ----
        let hn = rmsnorm(x_t, 1, d, &self.norm1, self.eps)?;
        let mut q = self.wq.forward(&hn);
        let mut k = self.wk.forward(&hn);
        let v = self.wv.forward(&hn);

        rope_apply_heads(&mut q, 1, h, dh, self.rope_base, pos);
        rope_apply_heads(&mut k, 1, h, dh, self.rope_base, pos);

        cache.append(&k, &v);
        let attn = cache.decode_step(&q, h, dh, self.attention_scale());

        let o = self.wo.forward(&attn);
        for (xi, oi) in x_t.iter_mut().zip(&o)
        {
            *xi += *oi; // résidu
        }

        // ---- FFN (pre-norm), 1 token ----
        let hn2 = rmsnorm(x_t, 1, d, &self.norm2, self.eps)?;
        let f1 = self.w1.forward_activated(&hn2, silu);
        let f2 = self.w2.forward(&f1);
        for (xi, fi) in x_t.iter_mut().zip(&f2)
        {
            *xi += *fi; // résidu
        }
        Some(())
    }
}
