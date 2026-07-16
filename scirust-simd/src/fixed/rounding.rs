// scirust-simd/src/fixed/rounding.rs
//
// # Politiques d'arrondi
//
// La multiplication et la division en virgule fixe produisent un accumulateur
// élargi qu'il faut **décaler à droite de `FRAC` bits** pour revenir au format
// `Q_FRAC`. Ce décalage jette `FRAC` bits de poids faible ; la façon dont on
// les traite est la politique d'arrondi, **jamais implicite** (cf. la
// philosophie du module).

use super::repr::WideInt;

/// Politique d'arrondi appliquée au décalage `>> FRAC` d'un accumulateur.
///
/// Toutes sont déterministes et indépendantes de l'architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum RoundingMode {
    /// Troncature vers zéro (le quotient perd sa partie fractionnaire).
    /// C'est l'arrondi le moins cher et le plus prévisible — **défaut** des
    /// opérateurs (`*`, `/`).
    #[default]
    TowardZero,
    /// Arrondi vers −∞ (plancher). Équivaut au décalage arithmétique brut.
    Floor,
    /// Arrondi vers +∞ (plafond).
    Ceil,
    /// Arrondi au plus proche, moitié au **pair** (arrondi du banquier). Sans
    /// biais statistique ; recommandé pour l'accumulation longue.
    NearestEven,
}

/// Calcule `round(value / 2^frac)` selon `mode`, en restant dans le type élargi.
///
/// `value` est l'accumulateur exact (produit ou dividende décalé). Le résultat
/// est encore élargi ; l'appelant le rétrécit ensuite selon la politique
/// d'overflow. Déterministe pour tout `WideInt`.
///
/// Invariant : le reste `value − floor·2^frac` appartient à `[0, 2^frac)` car
/// `floor` provient d'un décalage arithmétique (arrondi vers −∞).
#[inline]
#[must_use]
pub fn round_shift<W: WideInt>(value: W, frac: u32, mode: RoundingMode) -> W {
    if frac == 0
    {
        return value;
    }
    let floor = value.shr(frac); // arrondi vers −∞
    let rem = value.wrapping_sub(floor.shl(frac)); // ∈ [0, 2^frac)
    match mode
    {
        RoundingMode::Floor => floor,
        RoundingMode::TowardZero =>
        {
            // Vers zéro = plancher pour les positifs, plafond pour les négatifs.
            if value < W::ZERO && rem != W::ZERO
            {
                floor.wrapping_add(W::ONE)
            }
            else
            {
                floor
            }
        },
        RoundingMode::Ceil =>
        {
            if rem != W::ZERO
            {
                floor.wrapping_add(W::ONE)
            }
            else
            {
                floor
            }
        },
        RoundingMode::NearestEven =>
        {
            let half = W::ONE.shl(frac - 1); // 2^(frac−1)
            match rem.cmp(&half)
            {
                core::cmp::Ordering::Greater => floor.wrapping_add(W::ONE),
                core::cmp::Ordering::Less => floor,
                // Égalité exacte : arrondir vers le pair.
                core::cmp::Ordering::Equal =>
                {
                    if floor.is_odd()
                    {
                        floor.wrapping_add(W::ONE)
                    }
                    else
                    {
                        floor
                    }
                },
            }
        },
    }
}
