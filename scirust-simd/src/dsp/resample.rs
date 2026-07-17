// scirust-simd/src/dsp/resample.rs
//
// # Ré-échantillonnage rationnel `L/M`, déterministe
//
// [`resample`] change la fréquence d'échantillonnage d'un signal selon un
// rapport rationnel `L/M` (`L` : facteur de suréchantillonnage, `M` : facteur
// de sous-échantillonnage), via un filtre passe-bas prototype (sinus
// cardinal fenêtré par une fenêtre de Kaiser, cf. [`super::window::kaiser_coeff`])
// décomposé en `L` sous-filtres **polyphase** — la technique standard pour
// éviter de matérialiser le signal suréchantillonné (`L−1` zéros insérés
// entre chaque échantillon, dont la plupart des produits seraient nuls).
//
// **Générique sur le scalaire** comme le reste de `dsp` : la même
// implémentation sert `f32`/`f64` et la virgule fixe déterministe
// (`FixedI32<FRAC>`), un pipeline de ré-échantillonnage reproductible
// bit-à-bit sur toute architecture.
//
// ## Filtre prototype
//
// Fréquence de coupure normalisée (à la fréquence suréchantillonnée `Fs·L`) :
// `fc = 1 / (2·max(L, M))` — la plus restrictive des deux contraintes anti-repliement
// (suréchantillonnage : garder en dessous de `Fs/2` ; sous-échantillonnage :
// garder en dessous de `Fs·L/(2M)`). Le prototype `h[k] = L·2fc·sinc(2fc·(k−centre))·w[k]`
// (`w` : fenêtre de Kaiser symétrique, `β = 6`) a un gain continu `L`, qui
// compense exactement l'atténuation moyenne de l'insertion de zéros —
// convention standard de ré-échantillonnage polyphase.
//
// ## Décomposition polyphase
//
// `h` (longueur `2·half_taps·L + 1`, centré) est réparti en `L` sous-filtres
// `poly[p][j] = h[j·L + r]`, `r = (L − p) mod L` (`p ∈ 0..L`, `j ∈
// 0..2·half_taps+1`, complété par des zéros si `h` est plus court pour une
// phase donnée) — **`r`, pas `p`** : les prises de `h` alignées sur un
// multiple de `L` une fois décalées du centre du filtre sont celles d'indice
// `≡ −p (mod L)`, pas `≡ p (mod L)` (les deux ne coïncident que pour `L ≤ 2`,
// cf. [`resample::polyphase_decompose`]). Pour l'échantillon de sortie `n`,
// la position suréchantillonnée est `pos = n·M`, la phase `p = pos mod L`
// sélectionne le sous-filtre et l'échantillon d'entrée central est
// `⌈pos/L⌉` (pas `⌊pos/L⌋`, cohérent avec le décalage ci-dessus) — aucun zéro
// n'est jamais multiplié ni sommé.
//
// **Vérifié par test** (`resample_polyphase_matches_naive_zero_stuff_reference`) :
// ce raccourci polyphase donne, en virgule fixe, un résultat **bit-exact**
// avec l'insertion de zéros explicite suivie d'une convolution complète puis
// d'une décimation — même principe que `matmul_bt` face à `matmul` (même
// somme de produits, réorganisée pour éviter le travail inutile).

use core::ops::Div;

use crate::fixed::RealScalar;

use super::window::kaiser_coeff;

/// Paramètre de forme de la fenêtre de Kaiser du filtre prototype (bon
/// compromis largeur de transition / atténuation de bande coupée, ~60 dB).
fn beta<T: RealScalar>() -> T {
    T::from_i32(6)
}

/// `sinc(x) = sin(πx)/(πx)`, `sinc(0) = 1`.
fn sinc<T: RealScalar + Div<Output = T>>(x: T) -> T {
    if x == T::zero()
    {
        return T::one();
    }
    let px = T::pi() * x;
    px.sin() / px
}

/// Filtre prototype passe-bas (sinus cardinal fenêtré), longueur
/// `2·half_taps·l + 1`, centré, gain continu `l` (cf. en-tête de module).
pub(crate) fn design_prototype<T: RealScalar + Div<Output = T>>(
    l: usize,
    m: usize,
    half_taps: usize,
) -> Vec<T> {
    let n_taps = 2 * half_taps * l + 1;
    let center = half_taps * l;
    let max_lm = T::from_i32(core::cmp::max(l, m) as i32);
    let two_fc = T::one() / max_lm; // 2·fc = 1/max(l,m)
    let l_t = T::from_i32(l as i32);
    let b = beta::<T>();

    let mut h = vec![T::zero(); n_taps];
    for (k, hk) in h.iter_mut().enumerate()
    {
        let offset = T::from_i32(k as i32 - center as i32);
        let w = kaiser_coeff::<T>(k, n_taps - 1, b);
        *hk = l_t * two_fc * sinc(two_fc * offset) * w;
    }
    h
}

/// Décompose le prototype `h` en `l` sous-filtres polyphase de `2·half_taps+1`
/// prises chacun : `poly[p][j] = h[j·l + r]`, `r = (l − p) mod l` (zéro si
/// hors bornes).
///
/// `r`, pas `p` directement : les prises de `h` qui contribuent réellement à
/// la phase `p` (celles alignées sur un multiple de `l` une fois décalées du
/// centre du filtre) sont celles d'indice `≡ −p (mod l)`, pas `≡ p (mod l)` —
/// les deux ne coïncident que pour `l ≤ 2`. Se combine avec [`resample`], qui
/// indexe l'entrée à `⌈pos/l⌉` (pas `⌊pos/l⌋`) pour rester cohérent avec ce
/// décalage (vérifié par `resample_polyphase_matches_naive_zero_stuff_reference`).
pub(crate) fn polyphase_decompose<T: RealScalar>(
    h: &[T],
    l: usize,
    half_taps: usize,
) -> Vec<Vec<T>> {
    let taps_per_phase = 2 * half_taps + 1;
    let mut poly = vec![vec![T::zero(); taps_per_phase]; l];
    for (p, branch) in poly.iter_mut().enumerate()
    {
        let r = (l - p) % l;
        for (j, tap) in branch.iter_mut().enumerate()
        {
            let idx = j * l + r;
            if idx < h.len()
            {
                *tap = h[idx];
            }
        }
    }
    poly
}

/// Ré-échantillonne `x` selon le rapport rationnel `l/m` (suréchantillonnage
/// `l`, sous-échantillonnage `m`) via un filtre passe-bas prototype
/// polyphase (cf. en-tête de module). `half_taps` est le nombre de prises de
/// chaque sous-filtre polyphase de part et d'autre du centre (support total
/// du filtre : `2·half_taps·l + 1` prises) — plus grand, meilleure
/// atténuation de bande coupée, coût `O(x.len()·l/m·half_taps)` plus élevé.
///
/// Longueur de sortie : `x.len()·l/m` (division entière). Zéro-complète aux
/// bords (les prises retombant hors de `x` contribuent une valeur nulle).
///
/// Panique si `l == 0`, `m == 0` ou `half_taps == 0`.
#[must_use]
pub fn resample<T: RealScalar + Div<Output = T>>(
    x: &[T],
    l: usize,
    m: usize,
    half_taps: usize,
) -> Vec<T> {
    assert!(l >= 1, "resample : l doit être ≥ 1");
    assert!(m >= 1, "resample : m doit être ≥ 1");
    assert!(half_taps >= 1, "resample : half_taps doit être ≥ 1");

    let h = design_prototype::<T>(l, m, half_taps);
    let poly = polyphase_decompose(&h, l, half_taps);
    let taps_per_phase = 2 * half_taps + 1;

    let out_len = x.len() * l / m;
    let mut y = Vec::with_capacity(out_len);
    for n in 0..out_len
    {
        let pos = n * m;
        let k0 = pos.div_ceil(l) as isize; // ⌈pos/l⌉ : cf. doc de `polyphase_decompose`.
        let p = pos % l;
        let filt = &poly[p];

        let mut acc = T::zero();
        for (j, &tap) in filt.iter().enumerate().take(taps_per_phase)
        {
            let idx = k0 - half_taps as isize + j as isize;
            if idx >= 0 && (idx as usize) < x.len()
            {
                acc = acc + tap * x[idx as usize];
            }
        }
        y.push(acc);
    }
    y
}
