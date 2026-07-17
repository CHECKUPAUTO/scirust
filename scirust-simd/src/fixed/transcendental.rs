// scirust-simd/src/fixed/transcendental.rs
//
// # Transcendantes en virgule fixe (`FixedI32<FRAC>`)
//
// `exp exp2 ln log2 sin cos tanh sigmoid atan atan2 asin acos bessel_i0 erf
// erfc` pour le stockage `i32`, avec des **bornes d'erreur ULP prouvées**
// (balayage dense sur tout le domaine actif — voir `transcendental_ulp_bounds`
// / `bessel_i0_ulp_bounds` / `erf_ulp_bounds` dans le module de tests).
//
// Bornes mesurées en Q16.16 (ULP = 2⁻¹⁶) :
//
// | Fonction | Domaine testé | Erreur max |
// |---|---|---|
// | `ln` / `log2` | `(0, 32000]` | 0.50 ULP |
// | `sin` / `cos` | `[-100, 100]` | 0.52 ULP |
// | `tanh` / `sigmoid` | `[-12, 16]` | 0.50 ULP |
// | `atan` | `[-128, 128]` | 0.50 ULP |
// | `asin` / `acos` | `[-0.999, 0.999]` | 0.51 ULP |
// | `erf` / `erfc` | `[-4.5, 4.5]` | 0.50 ULP |
// | `exp` | `[-10, 10]` | 3.24 ULP |
// | `exp2` | `[-14, 14.5]` | 5.01 ULP |
// | `bessel_i0` | `[0, 12]` | 132 ULP (≤ 2 ULP si `I₀(x) ≤ 1024`) |
//
// `exp`/`exp2`/`bessel_i0` sont les seules > 1 ULP : l'erreur *relative* du
// minimax (≈2e-9 pour `exp`/`exp2`, ≈1e-7 pour `bessel_i0`) se convertit en
// ULP *absolus* proportionnellement à la magnitude, donc le pire cas est au
// sommet de la plage (`eˣ`/`2ˣ`/`I₀(x)` y sont les plus grands). Pour
// `|résultat| ≤ 1024` l'erreur reste ≤ 1 ULP (`exp`/`exp2`) ou ≤ 2 ULP
// (`bessel_i0`). `erf`/`erfc` restent bien conditionnées (sortie bornée dans
// `[-1, 1]`, pas de combinaison avec `eˣ`) : ≤ 1 ULP partout, comme
// `sin`/`cos`/`atan`.
//
// ## Méthode
//
// 1. **Réduction d'argument** exacte en entier :
//    * `exp` : `x·log₂e = k + f`, `f ∈ [-½, ½]` → `2^k · 2^f`.
//    * `ln`  : `x = 2^k · m`, `m ∈ [1, 2)` → `k + log₂(m)`, puis `·ln2`.
//    * `sin/cos` : réduction mod 2π puis par octant vers `[-π/4, π/4]`.
// 2. **Polynôme minimax** (Remez) évalué par Horner. Coefficients calculés hors
//    ligne (erreur d'approximation ≤ 0.03 ULP Q16.16) et figés en `i64`.
// 3. Toute l'arithmétique interne se fait en **Q32** (`i64`, produits via `i128`)
//    — 16 bits de garde au-dessus de Q16.16 — puis arrondi/saturation vers la
//    résolution de sortie. `tanh`/`sigmoid` réutilisent `exp` + inverse exact.
//
// Aucune opération flottante dans le chemin de calcul : entièrement entier,
// donc **déterministe bit-à-bit** sur toute architecture.
//
// Domaine : `FixedI32<FRAC>` avec `FRAC ≤ 31`. Le chemin `i64` (`FixedI64`) et
// les transcendantes vectorisées relèvent d'un lot ultérieur.

use super::types::Fixed;

// ------------------------------------------------------------------ //
//  Représentation interne Q32                                         //
// ------------------------------------------------------------------ //

/// Bits fractionnaires de la représentation interne.
const Q: u32 = 32;
/// `1.0` en Q32.
const ONE_Q: i64 = 1 << Q;

// Coefficients minimax (Remez), figés en Q32. Ordre : c0 + c1·x + … + cn·xⁿ.
// Erreurs d'approximation f64 : exp2 1.9e-9, log2 4.2e-8, sin 2.6e-9, cos 3.7e-7.
const EXP2_C: [i64; 7] = [
    4294967296, 2977044584, 1031765026, 238384739, 41308972, 5755442, 664776,
];
const LOG2_C: [i64; 9] = [
    -14711951260,
    35024616741,
    -42980318503,
    39882996028,
    -25827496272,
    11333447405,
    -3214170635,
    531913241,
    -39036563,
];
const SIN_C: [i64; 8] = [2, 4294967230, 31, -715826377, -209, 35782944, 263, -834895];
const COS_C: [i64; 7] = [4294967296, 0, -2147483648, 0, 178956912, 0, -5899877];
// atan(x) = x · P(x²), P minimax en u = x² sur [0, 1] (erreur ≈ 0.006 ULP).
const ATAN_C: [i64; 8] = [
    4294966916,
    -1431602515,
    857764031,
    -602558991,
    427038257,
    -257288806,
    105472057,
    -20531903,
];

const LOG2E_Q32: i64 = 6196328019;
const LN2_Q32: i64 = 2977044472;
const TAU_Q32: i64 = 26986075409;
const INV_TAU_Q32: i64 = 683565276;
const HALF_PI_Q32: i64 = 6746518852;
const PI_Q32: i64 = 13493037704;
const TWO_OVER_PI_Q32: i64 = 2734261102;

/// Produit Q32 arrondi au plus proche (accumulateur `i128`).
#[inline(always)]
fn mul_q(a: i64, b: i64) -> i64 {
    (((a as i128) * (b as i128) + (1 << (Q - 1))) >> Q) as i64
}

/// Horner : évalue `Σ cᵢ·xⁱ` (coeffs bas→haut) en Q32.
#[inline]
fn poly_q(x: i64, coeffs: &[i64]) -> i64 {
    let mut acc = coeffs[coeffs.len() - 1];
    for &c in coeffs.iter().rev().skip(1)
    {
        acc = mul_q(acc, x).wrapping_add(c);
    }
    acc
}

/// `FixedI32<FRAC>` (raw à FRAC) → Q32 (`i64`). Exact pour `FRAC ≤ 32`.
#[inline(always)]
fn to_q32<const FRAC: u32>(x: Fixed<i32, FRAC>) -> i64 {
    (x.to_raw() as i64) << (Q - FRAC)
}

/// Q32 (`i128`) → `FixedI32<FRAC>`, arrondi au plus proche, **saturant**.
#[inline]
fn from_q32<const FRAC: u32>(v: i128) -> Fixed<i32, FRAC> {
    let shift = Q - FRAC;
    // `saturating_add` : `exp2_core` peut renvoyer `i128::MAX` (débordement de
    // l'exponentielle) ; un `+` nu déborderait à l'ajout du demi-ULP d'arrondi.
    let rounded = v.saturating_add(1i128 << (shift - 1)) >> shift;
    let raw = if rounded > i32::MAX as i128
    {
        i32::MAX
    }
    else if rounded < i32::MIN as i128
    {
        i32::MIN
    }
    else
    {
        rounded as i32
    };
    Fixed::from_raw(raw)
}

// ------------------------------------------------------------------ //
//  exp / exp2                                                         //
// ------------------------------------------------------------------ //

/// `2^y` interne : `y_q32 → 2^y` en Q32 (`i128`, peut être grand ou ~0).
#[inline]
fn exp2_core(y_q32: i64) -> i128 {
    // y = k + f, f ∈ [-½, ½] (k = arrondi au plus proche).
    let k = (y_q32 + (1 << (Q - 1))) >> Q;
    let f = y_q32 - (k << Q);
    let mantissa = poly_q(f, &EXP2_C) as i128; // 2^f ∈ [0.707, 1.414] Q32
    if k > 62
    {
        i128::MAX // débordera → saturation à from_q32
    }
    else if k < -62
    {
        0
    }
    else if k >= 0
    {
        mantissa << k
    }
    else
    {
        mantissa >> (-k)
    }
}

/// `2^x`.
#[inline]
#[must_use]
pub fn exp2<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(exp2_core(to_q32(x)))
}

/// `eˣ`.
#[inline]
#[must_use]
pub fn exp<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(exp2_core(mul_q(to_q32(x), LOG2E_Q32)))
}

// ------------------------------------------------------------------ //
//  ln / log2                                                          //
// ------------------------------------------------------------------ //

/// `log₂(x_q32)` en Q32 pour `x_q32 > 0`.
#[inline]
fn log2_core(x_q32: i64) -> i64 {
    // x = 2^k · m, m ∈ [1, 2) (soit m_q32 ∈ [2^32, 2^33)).
    let msb = 63 - x_q32.leading_zeros() as i32;
    let k = msb - Q as i32;
    let m_q32 = if k >= 0 { x_q32 >> k } else { x_q32 << (-k) };
    let log2_m = poly_q(m_q32, &LOG2_C); // ∈ [0, 1) Q32
    ((k as i64) << Q).wrapping_add(log2_m)
}

/// `log₂(x)`. Renvoie `min_value` pour `x ≤ 0` (indéfini, saturé, sans panique).
#[inline]
#[must_use]
pub fn log2<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    if x.to_raw() <= 0
    {
        return Fixed::min_value();
    }
    from_q32(log2_core(to_q32(x)) as i128)
}

/// `ln(x)`. Renvoie `min_value` pour `x ≤ 0`.
#[inline]
#[must_use]
pub fn ln<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    if x.to_raw() <= 0
    {
        return Fixed::min_value();
    }
    from_q32(mul_q(log2_core(to_q32(x)), LN2_Q32) as i128)
}

// ------------------------------------------------------------------ //
//  sin / cos                                                          //
// ------------------------------------------------------------------ //

/// `(sin, cos)` en Q32 pour un angle `x_q32` quelconque.
#[inline]
fn sincos_core(x_q32: i64) -> (i64, i64) {
    // Réduction mod 2π → r ∈ [-π, π].
    let turns = (mul_q(x_q32, INV_TAU_Q32) + (1 << (Q - 1))) >> Q; // arrondi
    let r = (x_q32 as i128) - (turns as i128) * (TAU_Q32 as i128);
    // Réduction par octant → r2 ∈ [-π/4, π/4], quadrant m ∈ ℤ.
    let m = ((mul_q(r as i64, TWO_OVER_PI_Q32) + (1 << (Q - 1))) >> Q) as i128;
    let r2 = (r - m * (HALF_PI_Q32 as i128)) as i64;
    let s = poly_q(r2, &SIN_C);
    let c = poly_q(r2, &COS_C);
    match ((m % 4) + 4) % 4
    {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// `sin(x)`.
#[inline]
#[must_use]
pub fn sin<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(sincos_core(to_q32(x)).0 as i128)
}

/// `cos(x)`.
#[inline]
#[must_use]
pub fn cos<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(sincos_core(to_q32(x)).1 as i128)
}

// ------------------------------------------------------------------ //
//  atan / atan2 / asin / acos                                         //
// ------------------------------------------------------------------ //

/// `atan(r)` en Q32 pour `|r| ≤ 1` : `r · P(r²)` (polynôme minimax).
#[inline]
fn atan_unit(r_q32: i64) -> i64 {
    let sq = mul_q(r_q32, r_q32); // r² ∈ [0, 1] Q32
    mul_q(r_q32, poly_q(sq, &ATAN_C))
}

/// `atan(num / den)` en Q32 pour `den > 0`, sans débordement.
///
/// Choisit le plus petit des deux ratios (`|num|≤den` → polynôme direct ; sinon
/// `atan(n/d) = sign·π/2 − atan(d/n)`), de sorte que l'argument du polynôme
/// reste dans `[-1, 1]` et que le quotient Q32 tienne dans un `i64`.
#[inline]
fn atan_of_ratio(num: i64, den: i64) -> i64 {
    if num.unsigned_abs() <= den as u64
    {
        let r = (((num as i128) << Q) / (den as i128)) as i64; // |r| ≤ 1
        atan_unit(r)
    }
    else
    {
        let r = (((den as i128) << Q) / (num as i128)) as i64; // |r| < 1
        let a = atan_unit(r);
        if num >= 0
        {
            HALF_PI_Q32 - a
        }
        else
        {
            -HALF_PI_Q32 - a
        }
    }
}

/// `atan(x_q32)` en Q32.
#[inline]
fn atan_core(x_q32: i64) -> i64 {
    atan_of_ratio(x_q32, ONE_Q)
}

/// `atan2(y_q32, x_q32)` en Q32, résultat dans `(-π, π]`.
#[inline]
fn atan2_core(y_q32: i64, x_q32: i64) -> i64 {
    if x_q32 == 0 && y_q32 == 0
    {
        return 0;
    }
    // Angle de base dans [0, π/2] à partir des magnitudes.
    let a = if x_q32 == 0
    {
        HALF_PI_Q32
    }
    else
    {
        atan_of_ratio(y_q32.unsigned_abs() as i64, x_q32.unsigned_abs() as i64)
    };
    // Placement par quadrant depuis les signes de (x, y).
    match (x_q32 >= 0, y_q32 >= 0)
    {
        (true, true) => a,
        (false, true) => PI_Q32 - a,
        (false, false) => a - PI_Q32,
        (true, false) => -a,
    }
}

/// `√v` en Q32 pour `v ≥ 0` (racine entière de `v·2³²`).
#[inline]
fn sqrt_q32(v_q32: i64) -> i64 {
    if v_q32 <= 0
    {
        return 0;
    }
    // r = ⌊√(v·2³²)⌋ ; comme v ≤ 1 (Q32), v·2³² ≤ 2⁶⁴ tient dans un i128.
    let n = (v_q32 as i128) << Q;
    let mut r = 1i128 << 32;
    // Newton entier (converge en ~6 itérations pour n ≤ 2⁶⁴).
    for _ in 0..8
    {
        r = (r + n / r) >> 1;
    }
    while r * r > n
    {
        r -= 1;
    }
    r as i64
}

/// `asin(x_q32)` en Q32 pour `x_q32 ∈ [-1, 1]` (sinon borné à ±π/2).
#[inline]
fn asin_core(x_q32: i64) -> i64 {
    let x = x_q32.clamp(-ONE_Q, ONE_Q);
    // asin(x) = atan2(x, √(1 − x²)).
    let c = ONE_Q - mul_q(x, x); // 1 − x² ∈ [0, 1]
    atan2_core(x, sqrt_q32(c))
}

/// `arctangente` `atan(x)`.
#[inline]
#[must_use]
pub fn atan<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(atan_core(to_q32(x)) as i128)
}

/// `atan2(y, x)` : angle du point `(x, y)` dans `(-π, π]`.
#[inline]
#[must_use]
pub fn atan2<const FRAC: u32>(y: Fixed<i32, FRAC>, x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(atan2_core(to_q32(y), to_q32(x)) as i128)
}

/// `arcsinus` `asin(x)`. Hors de `[-1, 1]`, saturé à `±π/2`.
#[inline]
#[must_use]
pub fn asin<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(asin_core(to_q32(x)) as i128)
}

/// `arccosinus` `acos(x) = π/2 − asin(x)`. Hors de `[-1, 1]`, saturé à `0`/`π`.
#[inline]
#[must_use]
pub fn acos<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32((HALF_PI_Q32 - asin_core(to_q32(x))) as i128)
}

// ------------------------------------------------------------------ //
//  sigmoid / tanh (via exp + inverse exact)                           //
// ------------------------------------------------------------------ //

/// `σ(x_q32) = 1/(1+e^{-x})` en Q32 (`i128`).
#[inline]
fn sigmoid_core(x_q32: i64) -> i128 {
    let e = exp2_core(mul_q(-x_q32, LOG2E_Q32)); // e^{-x} en Q32
    // `saturating_add` : pour `x` très négatif, `e` peut valoir `i128::MAX` ;
    // le dénominateur sature alors et la sigmoïde tend vers 0 (correct).
    let denom = (ONE_Q as i128).saturating_add(e); // ≥ 1 (Q32) > 0
    // 1/denom en Q32 = 2^64 / denom_q32.
    (1i128 << (2 * Q)) / denom
}

/// Sigmoïde logistique `σ(x) = 1/(1+e^{-x})`.
#[inline]
#[must_use]
pub fn sigmoid<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(sigmoid_core(to_q32(x)))
}

/// `tanh(x) = 2·σ(2x) − 1`.
#[inline]
#[must_use]
pub fn tanh<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    let s = sigmoid_core(to_q32(x).wrapping_mul(2));
    from_q32(2 * s - (ONE_Q as i128))
}

// ------------------------------------------------------------------ //
//  bessel_i0 (fonction de Bessel modifiée, première espèce, ordre 0)   //
// ------------------------------------------------------------------ //

/// `1/6` en Q32, pour la réduction `u = x/6 − 1` de [`bessel_i0_core`].
const INV_SIX_Q32: i64 = 715_827_883;

// `g(x) = I₀(x)·e⁻ˣ` sur `x ∈ [0, 12]`, ramené à `u = x/6 − 1 ∈ [-1, 1]` puis
// approximé par un polynôme (moindres carrés sur base de Chebyshev, converti
// en base de puissance de `u`, coefficients figés en Q32 bas→haut). Erreur
// mesurée par balayage dense f64 : ≤ 2.1e-8 en absolu sur `g` — cf.
// `bessel_i0_ulp_bounds` pour la borne ULP correspondante sur `I₀` lui-même.
// `g` est borné dans `(0, 1]` : aucun risque de dépassement pour les formats
// virgule fixe étroits (contrairement à `I₀(x)` directement, qui croît vite).
const BESSEL_I0_C: [i64; 21] = [
    715_788_226,
    -376_392_825,
    299_189_561,
    -267_060_324,
    253_460_958,
    -249_788_724,
    252_242_846,
    -260_744_624,
    261_332_089,
    -229_015_587,
    218_606_249,
    -285_902_485,
    232_410_556,
    3_625_489,
    16_653_717,
    -268_741_689,
    166_669_234,
    100_839_382,
    -53_352_819,
    -64_278_783,
    34_506_419,
];

/// `I₀(x_q32)` en Q32 (`i128`) : `I₀(x) = g(x)·eˣ`, `g` par le polynôme
/// [`BESSEL_I0_C`] (argument saturé à `[0, 12]`, `I₀` étant paire), `eˣ` par
/// [`exp2_core`] (argument réel, non saturé). Produit calculé en `i128` avec
/// un seul arrondi final (`from_q32`) — pas de perte intermédiaire.
#[inline]
fn bessel_i0_core(x_q32: i64) -> i128 {
    let x_abs = x_q32.abs();
    let x_dom = x_abs.min(12 * ONE_Q); // domaine du polynôme : [0, 12]
    let u = mul_q(x_dom, INV_SIX_Q32) - ONE_Q; // u = x/6 − 1 ∈ [-1, 1]
    let g = poly_q(u, &BESSEL_I0_C) as i128;
    let e = exp2_core(mul_q(x_abs, LOG2E_Q32)); // eˣ, x réel (pas saturé)
    g.saturating_mul(e) >> Q
}

/// Fonction de Bessel modifiée de première espèce, ordre 0 : `I₀(x) = (1/π)
/// ∫₀^π e^{x·cos θ} dθ`. Paire, `I₀(0) = 1`, croissance ≈ `eˣ/√(2πx)` pour
/// `x` grand — au cœur de la fenêtre de Kaiser
/// ([`crate::dsp::window::kaiser`]).
///
/// Domaine testé/garanti : `x ∈ [0, 12]` (`I₀(12) ≈ 18949 < 32768`, la plage
/// représentable de `Q16.16`). Au-delà, `I₀` sature vers la valeur maximale
/// représentable de `T` (croît trop vite pour y rester). Voir
/// `bessel_i0_ulp_bounds` dans le module de tests : erreur ULP Q16.16 ≤ 2 pour
/// `I₀(x) ≤ 1024` (`x ≲ 8.9`), croissant ensuite avec la magnitude du résultat
/// (même phénomène que `exp`/`exp2` : l'erreur *relative* du polynôme, ≈1e-7,
/// se convertit en ULP *absolus* proportionnellement à la sortie).
#[inline]
#[must_use]
pub fn bessel_i0<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(bessel_i0_core(to_q32(x)))
}

// ------------------------------------------------------------------ //
//  erf / erfc (fonction d'erreur)                                     //
// ------------------------------------------------------------------ //

// `erf(x)` sur `x ∈ [0, 4]`, ramené à `u = x/2 − 1 ∈ [-1, 1]` puis approximé
// par un polynôme (moindres carrés sur base de Chebyshev, degré 20, converti
// en base de puissance de `u`, coefficients figés en Q32 bas→haut). Erreur
// mesurée par balayage dense f64 : ≤ 0.52 ULP Q16.16/Q8.24 — cf.
// `erf_ulp_bounds`. Contrairement à `bessel_i0`, `erf` reste borné dans
// `[-1, 1]` : aucune combinaison avec `eˣ` n'est nécessaire, le polynôme
// tient seul en Q32 `i64` (pas de débordement possible).
const ERF_C: [i64; 21] = [
    4_274_876_578,
    177_528_012,
    -710_112_339,
    1_656_931_672,
    -2_367_035_341,
    1_798_876_738,
    126_155_743,
    -1_856_765_945,
    1_750_067_221,
    -114_967_717,
    -1_159_211_398,
    930_052_540,
    102_002_508,
    -612_039_641,
    259_475_803,
    197_304_778,
    -175_892_838,
    -30_690_111,
    54_451_721,
    1_253_280,
    -7_294_047,
];

/// `erf(x_q32)` en Q32 (`i64` — `erf ∈ [-1, 1]`, jamais de débordement).
/// Argument saturé à `[0, 4]` (`erf` étant impaire), signe réappliqué à la
/// fin.
#[inline]
fn erf_core(x_q32: i64) -> i64 {
    let x_abs = x_q32.abs();
    let x_dom = x_abs.min(4 * ONE_Q); // domaine du polynôme : [0, 4]
    let u = mul_q(x_dom, ONE_Q / 2) - ONE_Q; // u = x/2 − 1 ∈ [-1, 1]
    let e = poly_q(u, &ERF_C);
    if x_q32 < 0 { -e } else { e }
}

/// Fonction d'erreur `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`. Impaire, `erf(0) = 0`,
/// `erf(x) → ±1` pour `x → ±∞` — au cœur de [`crate::fixed::activation::gelu`]
/// (`Φ(x) = (1 + erf(x/√2))/2`, la fonction de répartition normale centrée
/// réduite).
///
/// Domaine du polynôme : `x ∈ [0, 4]` (`erf(4) ≈ 1 − 1.5·10⁻⁸`, indiscernable
/// de `1` à la résolution `Q16.16`) ; erreur ULP Q16.16 ≤ 1 prouvée sur `[-4.5,
/// 4.5]` par `erf_ulp_bounds` (fonction bien conditionnée, contrairement à
/// `bessel_i0`). Au-delà, saturation propre vers `±1`, vérifiée jusqu'à `±10`
/// par `erf_saturates_beyond_domain`.
#[inline]
#[must_use]
pub fn erf<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32(erf_core(to_q32(x)) as i128)
}

/// Fonction d'erreur complémentaire `erfc(x) = 1 − erf(x)`.
#[inline]
#[must_use]
pub fn erfc<const FRAC: u32>(x: Fixed<i32, FRAC>) -> Fixed<i32, FRAC> {
    from_q32((ONE_Q as i128) - (erf_core(to_q32(x)) as i128))
}

// ------------------------------------------------------------------ //
//  softmax (activation vectorielle déterministe)                      //
// ------------------------------------------------------------------ //

/// Softmax numériquement stable, écrit dans `out` (aucune allocation).
///
/// `out[i] = exp(x[i] − max) / Σⱼ exp(x[j] − max)`. La soustraction du maximum
/// évite tout débordement de l'exponentielle ; la somme est accumulée
/// **exactement en Q32/`i128`**, donc le résultat est **déterministe bit-à-bit**
/// et indépendant de l'ordre. Panique si `input.len() != out.len()`.
pub fn softmax_into<const FRAC: u32>(input: &[Fixed<i32, FRAC>], out: &mut [Fixed<i32, FRAC>]) {
    assert_eq!(
        input.len(),
        out.len(),
        "softmax_into: longueurs différentes"
    );
    if input.is_empty()
    {
        return;
    }
    // Maximum (stabilité), déterministe.
    let mut m = input[0];
    for &x in &input[1..]
    {
        if x > m
        {
            m = x;
        }
    }
    let m_q = to_q32(m);
    // Passe 1 : Σ exp(xᵢ − max) en Q32 (i128, exact). ≥ 2^32 (l'argmax donne 1).
    let mut sum: i128 = 0;
    for &x in input
    {
        sum += exp2_core(mul_q(to_q32(x) - m_q, LOG2E_Q32));
    }
    // Passe 2 : normalisation exᵢ / Σ.
    for (o, &x) in out.iter_mut().zip(input)
    {
        let e = exp2_core(mul_q(to_q32(x) - m_q, LOG2E_Q32));
        *o = from_q32((e << Q) / sum);
    }
}
