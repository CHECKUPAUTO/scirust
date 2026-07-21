// scirust-simd/src/dsp/wavelet.rs
//
// # Transformée en ondelettes discrète par schéma de lifting [`dwt_decompose`]/[`dwt_reconstruct`]
//
// Complète [`super::fft`]/[`super::stft`] (analyse temps-fréquence à
// résolution **fixe**) par une décomposition **multirésolution** : chaque
// niveau sépare le signal en une approximation (basse fréquence, moitié de
// la longueur) et un détail (haute fréquence, autre moitié), et l'on peut
// réitérer sur l'approximation pour affiner l'échelle — le complément
// standard de la FFT quand l'information pertinente n'est pas à une échelle
// unique (débruitage, compression, détection de discontinuités).
//
// ## Le schéma de lifting (Sweldens)
//
// Plutôt qu'un banc de filtres (convolution + sous-échantillonnage), chaque
// niveau se calcule en trois étapes **en place**, sans FFT ni filtre séparé :
//
// 1. **Séparation** : `x` (longueur paire) devient les échantillons pairs
//    `e[i] = x[2i]` et impairs `o[i] = x[2i+1]`.
// 2. **Prédiction** : le détail `d[i]` est l'écart entre `o[i]` et une
//    prédiction construite à partir des `e[i]` voisins.
// 3. **Mise à jour** : l'approximation `s[i]` corrige `e[i]` à partir des
//    `d[i]` voisins (pour que `s` reste une moyenne locale correcte, pas
//    seulement le sous-échantillon `e`).
//
// [`haar_forward`]/[`haar_inverse`] implémentent le cas le plus simple
// (moyenne/différence, un seul voisin de chaque côté) ; [`cdf53_forward`]/
// [`cdf53_inverse`] implémentent l'ondelette biorthogonale **CDF 5/3**
// (« LeGall », norme JPEG2000 sans perte), qui prédit/met à jour à partir de
// **deux** voisins et compacte mieux l'énergie sur un signal lisse par
// morceaux (moins de détail significatif, plus proche de zéro).
//
// ## Réversibilité — **exacte au bit près en virgule fixe**
//
// Chaque étape de mise à jour ajoute un terme `f(d[…])` à `e[i]` pour
// produire `s[i]` ; l'inverse **soustrait ce même terme, recalculé depuis les
// mêmes `d[…]`**, ce qui l'annule par télescopage — algébriquement,
// `s − f(d) = (e + f(d)) − f(d) = e`. Cette identité est exacte en arithmétique
// réelle, et le reste **au bit près en virgule fixe** (`FixedI32<FRAC>`) :
// l'addition/soustraction de deux `Fixed<I, FRAC>` de même format est exacte
// (pas d'arrondi, contrairement à la multiplication), donc seule la
// multiplication par `f` arrondit — et elle produit **le même résultat
// arrondi** à l'aller et au retour (mêmes opérandes, même règle
// déterministe), que les additions/soustractions exactes qui l'entourent
// annulent parfaitement. Pas besoin du `floor` du schéma « ondelette entière »
// de JPEG2000 (qui vise des coefficients entiers, pas la réversibilité en
// elle-même) : ici l'exactitude vient uniquement du fait que `+`/`−` ne
// perdent aucun bit en virgule fixe.
//
// En `f32`/`f64`, `+`/`−` **arrondissent** eux aussi (IEEE 754) : la même
// identité de télescopage ne s'annule alors plus tout à fait exactement
// (quelques ULP d'écart typiquement, cf. tests) — l'exactitude bit-à-bit est
// une propriété **spécifique à la virgule fixe déterministe**, pas une
// conséquence générale de la structure de lifting. C'est néanmoins la
// différence structurelle avec un banc de filtres flottant classique
// (convolution + arrondi de sous-échantillonnage), où rien ne garantit même
// une reconstruction à quelques ULP.
//
// ## Bord : extension symétrique
//
// [`cdf53_forward`]/[`cdf53_inverse`] ont besoin d'un voisin `e[i+1]` (pour
// prédire le dernier `o`) et d'un voisin `d[i−1]` (pour mettre à jour le
// premier `e`) qui n'existent pas aux extrémités : on les remplace par
// réflexion (`e[i+1] := e[i]` en fin de tableau, `d[i−1] := d[0]` en début),
// une convention identique en analyse et en synthèse — donc elle aussi
// exactement annulée à la reconstruction, par le même argument de
// télescopage.
//
// ## Pyramide multi-niveaux (Mallat)
//
// [`dwt_decompose`] applique un niveau, puis réapplique sur la **moitié
// approximation** (`x[..len/2]`) pour affiner l'échelle, `levels` fois :
// après `n` niveaux, `x` contient l'approximation la plus grossière suivie
// des détails du plus grossier au plus fin. [`dwt_reconstruct`] défait les
// niveaux dans l'ordre inverse (du plus fin au plus grossier).

use crate::fixed::RealScalar;

/// Ondelette disponible pour [`dwt_decompose`]/[`dwt_reconstruct`] (cf.
/// en-tête de module).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wavelet {
    /// Haar (moyenne/différence) : un seul voisin de chaque côté, la plus
    /// simple et la moins coûteuse.
    Haar,
    /// CDF 5/3 (« LeGall »), norme JPEG2000 sans perte : deux voisins de
    /// chaque côté, meilleure compaction d'énergie sur un signal lisse par
    /// morceaux que Haar.
    Cdf53,
}

/// Une étape de lifting **Haar** : `x` de longueur paire `n`, remplacé en
/// place par `n/2` coefficients d'approximation (moyenne des paires)
/// **suivis** de `n/2` coefficients de détail (différence des paires).
/// Réversible exactement par [`haar_inverse`] (cf. en-tête de module).
pub fn haar_forward<T: RealScalar>(x: &mut [T]) {
    let n = x.len();
    assert!(n.is_multiple_of(2), "haar_forward : longueur {n} impaire");
    let m = n / 2;
    let half = T::from_i32(2).recip(); // puissance de 2 : recip() exact.

    let mut out = vec![T::zero(); n];
    for i in 0..m
    {
        let e = x[2 * i];
        let o = x[2 * i + 1];
        let d = o - e;
        out[i] = e + d * half;
        out[m + i] = d;
    }
    x.copy_from_slice(&out);
}

/// Réciproque de [`haar_forward`] : `x` contient `n/2` coefficients
/// d'approximation suivis de `n/2` coefficients de détail, remplacés en
/// place par le signal original (reconstruction exacte, cf. en-tête de
/// module).
pub fn haar_inverse<T: RealScalar>(x: &mut [T]) {
    let n = x.len();
    assert!(n.is_multiple_of(2), "haar_inverse : longueur {n} impaire");
    let m = n / 2;
    let half = T::from_i32(2).recip();

    let mut out = vec![T::zero(); n];
    for i in 0..m
    {
        let s = x[i];
        let d = x[m + i];
        let e = s - d * half;
        out[2 * i] = e;
        out[2 * i + 1] = d + e;
    }
    x.copy_from_slice(&out);
}

/// Une étape de lifting **CDF 5/3** : `x` de longueur paire `n` (`≥ 2`),
/// remplacé en place par `n/2` coefficients d'approximation suivis de `n/2`
/// coefficients de détail. Réversible exactement par [`cdf53_inverse`]
/// (extension symétrique aux bords, cf. en-tête de module).
pub fn cdf53_forward<T: RealScalar>(x: &mut [T]) {
    let n = x.len();
    assert!(n.is_multiple_of(2), "cdf53_forward : longueur {n} impaire");
    let m = n / 2;
    assert!(
        m >= 1,
        "cdf53_forward : au moins un échantillon pair/impair"
    );
    let half = T::from_i32(2).recip();
    let quarter = T::from_i32(4).recip();

    let e: Vec<T> = (0..m).map(|i| x[2 * i]).collect();
    let o: Vec<T> = (0..m).map(|i| x[2 * i + 1]).collect();

    let mut d = vec![T::zero(); m];
    for i in 0..m
    {
        let next = if i + 1 < m { i + 1 } else { m - 1 };
        d[i] = o[i] - (e[i] + e[next]) * half;
    }
    let mut s = vec![T::zero(); m];
    for i in 0..m
    {
        let prev = i.saturating_sub(1);
        s[i] = e[i] + (d[prev] + d[i]) * quarter;
    }

    x[..m].copy_from_slice(&s);
    x[m..].copy_from_slice(&d);
}

/// Réciproque de [`cdf53_forward`] : `x` contient `n/2` coefficients
/// d'approximation suivis de `n/2` coefficients de détail, remplacés en
/// place par le signal original.
pub fn cdf53_inverse<T: RealScalar>(x: &mut [T]) {
    let n = x.len();
    assert!(n.is_multiple_of(2), "cdf53_inverse : longueur {n} impaire");
    let m = n / 2;
    assert!(
        m >= 1,
        "cdf53_inverse : au moins un coefficient de chaque côté"
    );
    let half = T::from_i32(2).recip();
    let quarter = T::from_i32(4).recip();

    let s = x[..m].to_vec();
    let d = x[m..].to_vec();

    // Défait la mise à jour d'abord (dernière étape à l'aller), puis la
    // prédiction — ordre inverse de `cdf53_forward` (cf. en-tête de module).
    let mut e = vec![T::zero(); m];
    for i in 0..m
    {
        let prev = i.saturating_sub(1);
        e[i] = s[i] - (d[prev] + d[i]) * quarter;
    }
    let mut o = vec![T::zero(); m];
    for i in 0..m
    {
        let next = if i + 1 < m { i + 1 } else { m - 1 };
        o[i] = d[i] + (e[i] + e[next]) * half;
    }

    let mut out = vec![T::zero(); n];
    for i in 0..m
    {
        out[2 * i] = e[i];
        out[2 * i + 1] = o[i];
    }
    x.copy_from_slice(&out);
}

#[inline]
fn one_level_forward<T: RealScalar>(x: &mut [T], wavelet: Wavelet) {
    match wavelet
    {
        Wavelet::Haar => haar_forward(x),
        Wavelet::Cdf53 => cdf53_forward(x),
    }
}

#[inline]
fn one_level_inverse<T: RealScalar>(x: &mut [T], wavelet: Wavelet) {
    match wavelet
    {
        Wavelet::Haar => haar_inverse(x),
        Wavelet::Cdf53 => cdf53_inverse(x),
    }
}

/// Décomposition en pyramide multi-niveaux (Mallat) : applique `levels` fois
/// une étape de lifting `wavelet`, en réappliquant à chaque niveau sur la
/// **moitié approximation** du niveau précédent (cf. en-tête de module).
/// Après l'appel, `x` contient l'approximation la plus grossière (longueur
/// `x.len() >> levels`) suivie des détails, du plus grossier au plus fin.
///
/// Panique si `x.len()` n'est pas un multiple de `2^levels`.
pub fn dwt_decompose<T: RealScalar>(x: &mut [T], levels: usize, wavelet: Wavelet) {
    for i in 0..levels
    {
        let len = x.len() >> i;
        assert!(
            len.is_multiple_of(2) && len >= 2,
            "dwt_decompose : longueur {} non divisible par 2^{levels}",
            x.len()
        );
        one_level_forward(&mut x[..len], wavelet);
    }
}

/// Réciproque de [`dwt_decompose`] : défait les `levels` niveaux dans
/// l'ordre inverse (du plus fin au plus grossier), reconstruisant le signal
/// original **exactement** (cf. en-tête de module).
pub fn dwt_reconstruct<T: RealScalar>(x: &mut [T], levels: usize, wavelet: Wavelet) {
    for i in (0..levels).rev()
    {
        let len = x.len() >> i;
        assert!(
            len.is_multiple_of(2) && len >= 2,
            "dwt_reconstruct : longueur {} non divisible par 2^{levels}",
            x.len()
        );
        one_level_inverse(&mut x[..len], wavelet);
    }
}
