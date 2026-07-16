// scirust-simd/src/fixed/types.rs
//
// # `Fixed<I, FRAC>` — nombre en virgule fixe générique
//
// ## Représentation mémoire
//
// `#[repr(transparent)]` autour d'un entier signé `I` (`i32`, `i64`). La valeur
// réelle représentée est `raw / 2^FRAC` (format « Q(BITS−FRAC).FRAC »). Le type
// a exactement la taille et l'alignement de `I` : un `[Fixed<i32, F>; N]` a le
// même layout qu'un `[i32; N]`, ce qui permet le traitement SIMD sans copie.
//
// ## Plage et résolution (exemple `FixedI32<16>` = Q16.16)
//
// * plage    : `[-2^15, 2^15 − 2^-16]` ≈ `[-32768, 32767.99998]`
// * résolution : `2^-16` ≈ `1.526e-5` (pas constant sur toute la plage)
//
// La résolution est **absolue et constante**, contrairement au flottant dont
// la résolution est relative. C'est la propriété qui rend la virgule fixe
// déterministe et reproductible bit-à-bit.
//
// ## Coût des opérations
//
// * `+`, `−`, `neg`, `min`, `max`, `clamp`, comparaison : 1 instruction entière.
// * `*` : une multiplication élargie (`i32×i32→i64`, `i64×i64→i128`) + décalage
//   arrondi + rétrécissement. Exacte avant arrondi.
// * `/` : un décalage à gauche élargi + une division entière élargie.
//
// ## Overflow & arrondi
//
// Voir [`super::OverflowMode`] et [`super::RoundingMode`]. Les opérateurs
// enveloppent et tronquent par défaut ; les méthodes `checked_*`,
// `saturating_*`, `*_rounded` donnent le contrôle total.

use core::fmt;

use super::overflow::{OverflowMode, narrow};
use super::repr::{FixedStorage, WideInt};
use super::rounding::{RoundingMode, round_shift};

/// Nombre en virgule fixe : valeur réelle = `raw / 2^FRAC`.
///
/// Générique sur l'entier de stockage `I` ([`FixedStorage`]) et le nombre de
/// bits fractionnaires `FRAC` (const). Utiliser les alias
/// [`FixedI32`](super::FixedI32) / [`FixedI64`](super::FixedI64) et
/// [`Q16_16`](super::Q16_16) etc. plutôt que la forme brute.
///
/// **Invariant d'usage** : `0 ≤ FRAC < I::BITS`. Les alias fournis le
/// respectent ; construire un `FRAC` hors de cette plage est un abus.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Fixed<I: FixedStorage, const FRAC: u32>(pub I);

impl<I: FixedStorage, const FRAC: u32> Fixed<I, FRAC> {
    // -------------------------------------------------------------- //
    //  Constructeurs & accès brut                                    //
    // -------------------------------------------------------------- //

    /// Construit depuis l'entier brut (interprété comme `raw / 2^FRAC`).
    #[inline(always)]
    #[must_use]
    pub const fn from_raw(raw: I) -> Self {
        Self(raw)
    }

    /// Entier brut sous-jacent.
    #[inline(always)]
    #[must_use]
    pub const fn to_raw(self) -> I {
        self.0
    }

    /// Zéro (`0.0`).
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self(I::ZERO)
    }

    /// Un (`1.0`), soit le brut `2^FRAC`.
    #[inline(always)]
    #[must_use]
    pub fn one() -> Self {
        Self(I::ONE.wrapping_shl(FRAC))
    }

    /// Plus petite valeur strictement positive représentable : `2^-FRAC`.
    #[inline(always)]
    #[must_use]
    pub fn resolution() -> Self {
        Self(I::ONE)
    }

    /// Plus grande valeur représentable (`MAX / 2^FRAC`).
    #[inline(always)]
    #[must_use]
    pub fn max_value() -> Self {
        Self(I::MAX)
    }

    /// Plus petite valeur représentable (`MIN / 2^FRAC`).
    #[inline(always)]
    #[must_use]
    pub fn min_value() -> Self {
        Self(I::MIN)
    }

    // -------------------------------------------------------------- //
    //  Conversion entier → fixe                                      //
    // -------------------------------------------------------------- //

    /// Depuis un entier (valeur `v.0`), **saturant** si hors plage.
    #[inline]
    #[must_use]
    pub fn from_int_saturating(value: I) -> Self {
        Self(I::from_wide_saturating(value.to_wide().shl(FRAC)))
    }

    /// Depuis un entier, `None` si hors plage.
    #[inline]
    #[must_use]
    pub fn from_int_checked(value: I) -> Option<Self> {
        I::from_wide_checked(value.to_wide().shl(FRAC)).map(Self)
    }

    /// Depuis un entier, enveloppant si hors plage.
    #[inline]
    #[must_use]
    pub fn from_int_wrapping(value: I) -> Self {
        Self(value.wrapping_shl(FRAC))
    }

    // -------------------------------------------------------------- //
    //  Conversion flottant ↔ fixe                                    //
    // -------------------------------------------------------------- //

    /// Facteur d'échelle `2^FRAC` en `f64` (exact : puissance de deux).
    #[inline(always)]
    #[must_use]
    fn scale_f64() -> f64 {
        (2.0f64).powi(FRAC as i32)
    }

    /// Facteur d'échelle `2^FRAC` en `f32`.
    #[inline(always)]
    #[must_use]
    fn scale_f32() -> f32 {
        (2.0f32).powi(FRAC as i32)
    }

    /// Convertit en `f64`. **Sans perte** tant que `FRAC ≤ 52` et que la
    /// magnitude brute tient dans la mantisse (53 bits) ; sinon arrondi au
    /// plus proche `f64`.
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.0.to_f64() / Self::scale_f64()
    }

    /// Convertit en `f32`. Perte au-delà de 24 bits significatifs (arrondi
    /// au plus proche `f32`).
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        self.0.to_f32() / Self::scale_f32()
    }

    /// Depuis un `f64`, avec politique d'arrondi explicite. `None` si NaN,
    /// infini, ou hors plage représentable.
    ///
    /// L'arrondi porte sur `value · 2^FRAC` avant conversion en entier :
    /// `TowardZero`→troncature, `Floor`/`Ceil`→plancher/plafond,
    /// `NearestEven`→pair le plus proche.
    #[must_use]
    pub fn from_f64(value: f64, rounding: RoundingMode) -> Option<Self> {
        if !value.is_finite()
        {
            return None;
        }
        let scaled = value * Self::scale_f64();
        let rounded = match rounding
        {
            RoundingMode::TowardZero => scaled.trunc(),
            RoundingMode::Floor => scaled.floor(),
            RoundingMode::Ceil => scaled.ceil(),
            RoundingMode::NearestEven => scaled.round_ties_even(),
        };
        // Bornes en f64 : comparaison sûre (les bornes i32/i64 sont finies).
        if rounded < I::MIN.to_f64() || rounded > I::MAX.to_f64()
        {
            return None;
        }
        // `rounded` est un entier fini dans la plage → conversion saturante exacte.
        Some(Self(I::from_f64_saturating(rounded)))
    }

    // -------------------------------------------------------------- //
    //  Addition / soustraction / négation                            //
    // -------------------------------------------------------------- //

    /// Addition enveloppante (défaut de l'opérateur `+`).
    #[inline(always)]
    #[must_use]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
    /// Addition vérifiée : `None` en cas de débordement.
    #[inline(always)]
    #[must_use]
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        self.0.checked_add(rhs.0).map(Self)
    }
    /// Addition saturante.
    #[inline(always)]
    #[must_use]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Soustraction enveloppante (défaut de l'opérateur `−`).
    #[inline(always)]
    #[must_use]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
    /// Soustraction vérifiée.
    #[inline(always)]
    #[must_use]
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        self.0.checked_sub(rhs.0).map(Self)
    }
    /// Soustraction saturante.
    #[inline(always)]
    #[must_use]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Négation enveloppante (défaut de l'opérateur unaire `−`). `−MIN` ↦ `MIN`.
    #[inline(always)]
    #[must_use]
    pub fn wrapping_neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }
    /// Négation vérifiée : `None` pour `MIN`.
    #[inline(always)]
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        self.0.checked_neg().map(Self)
    }
    /// Négation saturante : `−MIN` ↦ `MAX`.
    #[inline(always)]
    #[must_use]
    pub fn saturating_neg(self) -> Self {
        Self(self.0.saturating_neg())
    }

    // -------------------------------------------------------------- //
    //  Multiplication                                                //
    // -------------------------------------------------------------- //

    /// Produit élargi exact `self·rhs` (avant arrondi/rétrécissement), en
    /// unités de `2^(2·FRAC)`. Interne aux variantes de multiplication.
    #[inline(always)]
    fn mul_wide(self, rhs: Self) -> I::Wide {
        self.0.to_wide().wrapping_mul(rhs.0.to_wide())
    }

    /// Multiplication paramétrée (arrondi + overflow explicites).
    /// `None` uniquement en overflow `Checked` débordant.
    #[inline]
    #[must_use]
    pub fn mul_rounded(
        self,
        rhs: Self,
        rounding: RoundingMode,
        overflow: OverflowMode,
    ) -> Option<Self> {
        let shifted = round_shift(self.mul_wide(rhs), FRAC, rounding);
        narrow::<I>(shifted, overflow).map(Self)
    }

    /// Multiplication enveloppante, troncature vers zéro (défaut de `*`).
    #[inline]
    #[must_use]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        let shifted = round_shift(self.mul_wide(rhs), FRAC, RoundingMode::TowardZero);
        Self(I::from_wide_wrapping(shifted))
    }
    /// Multiplication vérifiée (troncature vers zéro).
    #[inline]
    #[must_use]
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        self.mul_rounded(rhs, RoundingMode::TowardZero, OverflowMode::Checked)
    }
    /// Multiplication saturante (troncature vers zéro).
    #[inline]
    #[must_use]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        // `narrow` renvoie toujours `Some` en mode Saturate.
        self.mul_rounded(rhs, RoundingMode::TowardZero, OverflowMode::Saturate)
            .unwrap_or(Self(I::ZERO))
    }

    // -------------------------------------------------------------- //
    //  Division                                                      //
    // -------------------------------------------------------------- //

    /// Dividende élargi `self.raw << FRAC` (avant division par `rhs.raw`).
    #[inline(always)]
    fn div_wide_numerator(self) -> I::Wide {
        self.0.to_wide().shl(FRAC)
    }

    /// Division paramétrée (arrondi + overflow explicites).
    /// `None` si `rhs == 0` **ou** en overflow `Checked` débordant.
    #[inline]
    #[must_use]
    pub fn div_rounded(
        self,
        rhs: Self,
        rounding: RoundingMode,
        overflow: OverflowMode,
    ) -> Option<Self> {
        if rhs.0 == I::ZERO
        {
            return None;
        }
        // (self << FRAC) / rhs : quotient exact en Q_FRAC, puis arrondi du reste.
        let num = self.div_wide_numerator();
        let den = rhs.0.to_wide();
        let quotient = num.div_trunc(den);
        // Arrondit `num/den` (quotient tronqué + reste) selon la politique.
        let rounded = div_round(num, den, quotient, rounding);
        narrow::<I>(rounded, overflow).map(Self)
    }

    /// Division enveloppante, troncature vers zéro (défaut de `/`).
    ///
    /// # Panics
    /// Panique si `rhs == 0` (comme la division entière ; déterministe).
    #[inline]
    #[must_use]
    pub fn wrapping_div(self, rhs: Self) -> Self {
        self.div_rounded(rhs, RoundingMode::TowardZero, OverflowMode::Wrap)
            .expect("division virgule fixe par zéro")
    }
    /// Division vérifiée : `None` si `rhs == 0` ou débordement.
    #[inline]
    #[must_use]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        self.div_rounded(rhs, RoundingMode::TowardZero, OverflowMode::Checked)
    }
    /// Division saturante.
    ///
    /// # Panics
    /// Panique si `rhs == 0`.
    #[inline]
    #[must_use]
    pub fn saturating_div(self, rhs: Self) -> Self {
        self.div_rounded(rhs, RoundingMode::TowardZero, OverflowMode::Saturate)
            .expect("division virgule fixe par zéro")
    }

    // -------------------------------------------------------------- //
    //  min / max / clamp / abs / signe                               //
    // -------------------------------------------------------------- //

    /// Minimum (ordre total sur la valeur réelle).
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        if self.0 <= rhs.0 { self } else { rhs }
    }
    /// Maximum.
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 { self } else { rhs }
    }
    /// Restreint à `[lo, hi]`. Panique (debug) si `lo > hi`.
    #[inline(always)]
    #[must_use]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        debug_assert!(lo.0 <= hi.0, "clamp: lo > hi");
        self.max(lo).min(hi)
    }
    /// Valeur absolue **saturante** : `|MIN|` ↦ `MAX` (car `−MIN` déborde).
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }
    /// Valeur absolue vérifiée : `None` pour `MIN`.
    #[inline(always)]
    #[must_use]
    pub fn checked_abs(self) -> Option<Self> {
        self.0.checked_abs().map(Self)
    }

    /// Vrai si strictement négatif.
    #[inline(always)]
    #[must_use]
    pub fn is_negative(self) -> bool {
        self.0 < I::ZERO
    }
    /// Vrai si strictement positif.
    #[inline(always)]
    #[must_use]
    pub fn is_positive(self) -> bool {
        self.0 > I::ZERO
    }
    /// Vrai si nul.
    #[inline(always)]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == I::ZERO
    }
}

// ------------------------------------------------------------------ //
//  Aides internes (hors `impl` pour rester libres du const generic)  //
// ------------------------------------------------------------------ //

/// Arrondit `num/den` (quotient tronqué vers zéro déjà calculé) selon `rounding`.
///
/// `quotient_trunc = num.div_trunc(den)`. Le reste `num − q·den` a le signe de
/// `num` ; le signe du **vrai quotient** est `sign(num) ⊕ sign(den)`, seul
/// critère correct pour Floor/Ceil (utiliser `num` seul serait faux si
/// `den < 0`).
#[inline]
fn div_round<W: WideInt>(num: W, den: W, quotient_trunc: W, rounding: RoundingMode) -> W {
    let rem = num.wrapping_sub(quotient_trunc.wrapping_mul(den));
    if rem == W::ZERO
    {
        return quotient_trunc;
    }
    // Signe du vrai quotient (rem a le signe de num).
    let quotient_negative = (rem < W::ZERO) ^ (den < W::ZERO);
    let one = W::ONE;
    match rounding
    {
        RoundingMode::TowardZero => quotient_trunc, // `/` tronque déjà vers zéro
        RoundingMode::Floor =>
        {
            if quotient_negative
            {
                quotient_trunc.wrapping_sub(one)
            }
            else
            {
                quotient_trunc
            }
        },
        RoundingMode::Ceil =>
        {
            if quotient_negative
            {
                quotient_trunc
            }
            else
            {
                quotient_trunc.wrapping_add(one)
            }
        },
        RoundingMode::NearestEven =>
        {
            let twice_rem = wide_abs(rem).shl(1);
            let bump = match twice_rem.cmp(&wide_abs(den))
            {
                core::cmp::Ordering::Greater => true,
                core::cmp::Ordering::Less => false,
                core::cmp::Ordering::Equal => quotient_trunc.is_odd(),
            };
            if !bump
            {
                quotient_trunc
            }
            else if quotient_negative
            {
                quotient_trunc.wrapping_sub(one)
            }
            else
            {
                quotient_trunc.wrapping_add(one)
            }
        },
    }
}

#[inline]
fn wide_abs<W: WideInt>(v: W) -> W {
    if v < W::ZERO
    {
        W::ZERO.wrapping_sub(v)
    }
    else
    {
        v
    }
}

// ------------------------------------------------------------------ //
//  Debug / Display                                                    //
// ------------------------------------------------------------------ //

impl<I: FixedStorage, const FRAC: u32> fmt::Debug for Fixed<I, FRAC> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fixed<{}b,Q{}>({}, raw={:?})",
            I::BITS,
            FRAC,
            self,
            self.0
        )
    }
}

impl<I: FixedStorage, const FRAC: u32> fmt::Display for Fixed<I, FRAC> {
    /// Formatage décimal **exact** (les fractions binaires terminent en ≤ FRAC
    /// chiffres décimaux). Aucune allocation : écriture directe dans le
    /// formateur, chiffre par chiffre.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let raw = self.0;
        let neg = raw < I::ZERO;
        // Magnitude dans l'élargi pour gérer |MIN| sans débordement.
        let mag = wide_abs(raw.to_wide());
        let int_part = mag.shr(FRAC);
        let mut frac_raw = mag.wrapping_sub(int_part.shl(FRAC));

        if neg
        {
            write!(f, "-")?;
        }
        write!(f, "{}", int_part)?;

        if frac_raw != <I::Wide as WideInt>::ZERO
        {
            write!(f, ".")?;
            // fr·10 = (fr<<3)+(fr<<1) ; le chiffre est la partie entière après
            // remise à l'échelle, le reste reprend la boucle. Termine en ≤ FRAC
            // itérations (borne de sécurité incluse).
            let mut guard = 0u32;
            while frac_raw != <I::Wide as WideInt>::ZERO && guard <= FRAC
            {
                let scaled = frac_raw.shl(3).wrapping_add(frac_raw.shl(1));
                let digit = scaled.shr(FRAC);
                write!(f, "{}", digit)?;
                frac_raw = scaled.wrapping_sub(digit.shl(FRAC));
                guard += 1;
            }
        }
        Ok(())
    }
}
