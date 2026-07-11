//! Basses précisions **reproductibles** : bf16, f16 (IEEE binary16) et FP8
//! (E4M3/E5M2, spécification publique OCP *8-bit Floating Point*) — le
//! troisième chantier long de la cartographie (volet 111), explicitement
//! hors périmètre de RepDL.
//!
//! Tout est **manipulation de bits entière** (aucune opération flottante
//! au-delà des comparaisons) ⇒ portable par construction : mêmes bits sur
//! toute plate-forme. Deux modes d'arrondi :
//!
//! - **RNE** (au plus près, pair en cas d'égalité) — le mode déterministe ;
//! - **arrondi stochastique** piloté par [`crate::philox`] : la probabilité
//!   de monter est proportionnelle au reste tronqué, et l'aléa est
//!   **contre-basé** — le tirage du rang i ne dépend que de (graine, flux,
//!   i), donc la quantification stochastique d'un tenseur est bit-identique
//!   quel que soit le découpage en threads. C'est l'« arrondi stochastique
//!   reproductible » identifié comme trou commun de RepDL et scirust.
//!
//! Les produits bf16×bf16 (mantisses 8 bits) et f16×f16 (11 bits) sont
//! **exacts en f32** (16 et 22 bits ≤ 24) : [`gemm_bf16_exact`] accumule en
//! f64 en ordre fixe — GEMM basse précision bit-exact inter-plates-formes.
//!
//! FP8 : formats de l'OCP 8-bit FP spec — E4M3 (biais 7, pas d'infini,
//! NaN = S.1111.111, max fini ±448) et E5M2 (biais 15, ±∞ et NaN IEEE,
//! max fini ±57344). Conversion **saturante** vers le max fini (le
//! comportement d'entraînement usuel), documentée ; NaN → NaN canonique du
//! format.

use crate::philox::Philox4x32;

// ================================================================== //
//  bf16 (bfloat16 : 1 + 8 + 7 bits — les 16 bits hauts d'un f32)      //
// ================================================================== //

/// f32 → bf16 (RNE) : troncature des 16 bits bas avec arrondi au plus près,
/// pair en cas d'égalité. NaN → NaN canonique bf16 (0x7FC0 signé).
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan()
    {
        return 0x7fc0 | (((bits >> 16) as u16) & 0x8000);
    }
    let lower = bits & 0xffff;
    let upper = (bits >> 16) as u16;
    let round_up = lower > 0x8000 || (lower == 0x8000 && (upper & 1) == 1);
    // +1 déborde proprement vers l'infini (0x7f80) par construction IEEE
    if round_up
    {
        upper.wrapping_add(1)
    }
    else
    {
        upper
    }
}

/// bf16 → f32 (exact).
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ================================================================== //
//  f16 (IEEE binary16 : 1 + 5 + 10 bits)                              //
// ================================================================== //

/// f32 → f16 (RNE), sous-normaux et saturations IEEE compris.
pub fn f32_to_f16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff
    {
        // inf / NaN
        return if mant == 0
        {
            sign | 0x7c00
        }
        else
        {
            sign | 0x7e00
        };
    }
    // exposant f16 : e16 = e32 − 127 + 15
    let e16 = exp - 127 + 15;
    if e16 >= 0x1f
    {
        return sign | 0x7c00; // dépasse le max f16 ⇒ ±inf (RNE)
    }
    if e16 <= 0
    {
        // sous-normal f16 (ou zéro) : mantisse avec bit implicite, décalée
        if e16 < -10
        {
            return sign; // < demi plus petit sous-normal ⇒ ±0
        }
        let full = mant | 0x0080_0000; // 24 bits significatifs
        let shift = (14 - e16) as u32; // position de troncature (≥ 14)
        let kept = (full >> shift) as u16;
        let rem = full & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round_up = rem > half || (rem == half && (kept & 1) == 1);
        return sign | if round_up { kept + 1 } else { kept };
    }
    // normal : tronque la mantisse 23 → 10 bits avec RNE
    let kept = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let round_up = rem > 0x1000 || (rem == 0x1000 && (kept & 1) == 1);
    let h = ((e16 as u16) << 10) | kept;
    sign | if round_up { h + 1 } else { h } // le report monte l'exposant, IEEE
}

/// f16 → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0x1f
    {
        sign | 0x7f80_0000 | (mant << 13) // inf / NaN (payload conservé)
    }
    else if exp == 0
    {
        if mant == 0
        {
            sign // ±0
        }
        else
        {
            // sous-normal f16 = mant · 2⁻²⁴ : normalise vers f32
            let h = 31 - mant.leading_zeros(); // position du bit de tête
            let e32 = h + 103; // biaisé : 2^(h−24) → h − 24 + 127
            let m32 = (mant << (23 - h)) & 0x007f_ffff;
            sign | (e32 << 23) | m32
        }
    }
    else
    {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ================================================================== //
//  FP8 (OCP : E4M3 et E5M2)                                           //
// ================================================================== //

/// Format FP8 (spec OCP publique).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// 1+4+3, biais 7, PAS d'infini, NaN = S.1111.111, max fini ±448.
    E4M3,
    /// 1+5+2, biais 15, ±inf et NaN IEEE, max fini ±57344.
    E5M2,
}

impl Fp8Format {
    fn params(self) -> (i32, u32, u8, f32) {
        // (biais, bits de mantisse, code NaN canonique, max fini)
        match self
        {
            Fp8Format::E4M3 => (7, 3, 0x7f, 448.0),
            Fp8Format::E5M2 => (15, 2, 0x7e, 57_344.0),
        }
    }
}

/// Décomposition commune f32 → FP8, partagée par RNE et l'arrondi
/// stochastique : `Ok(code)` couvre les cas déjà tranchés sans dépendre
/// d'aucune décision d'arrondi (NaN/inf/saturation/flush-to-zero) ; `Err`
/// porte (signe, mantisse tronquée `kept`, reste `rem`, largeur `shift`,
/// champ d'exposant FP8 `e_field`) pour les cas où RNE et arrondi
/// stochastique divergent — factoriser évite de dupliquer (et risquer de
/// désynchroniser) cette arithmétique de troncature délicate.
fn fp8_pre_round(x: f32, fmt: Fp8Format) -> Result<u8, (u8, u32, u32, u32, u8)> {
    let (bias, mbits, nan_code, max_fin) = fmt.params();
    let bits = x.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    if x.is_nan()
    {
        return Ok(sign | nan_code);
    }
    if x.is_infinite()
    {
        return Ok(match fmt
        {
            Fp8Format::E4M3 => sign | 0x7e, // sature au max fini (S.1111.110)
            Fp8Format::E5M2 => sign | 0x7c, // ±inf
        });
    }
    let ax = x.abs();
    if ax > max_fin
    {
        // saturation (comportement train-time usuel, documenté)
        return Ok(match fmt
        {
            Fp8Format::E4M3 => sign | 0x7e,
            Fp8Format::E5M2 => sign | 0x7b, // S.11110.11 = 57344
        });
    }
    if ax == 0.0
    {
        return Ok(sign);
    }
    // décompose |x| = m·2^e, m ∈ [2^23, 2^24)
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    let (m24, e_unb) = if exp == 0
    {
        // f32 sous-normal : bien sous le plus petit sous-normal FP8 ⇒ ±0
        (mant, -126 - 23)
    }
    else
    {
        (mant | 0x0080_0000, exp - 127 - 23)
    };
    // e8 = exposant biaisé FP8 du bit de tête
    let top = 23; // m24 a son bit de tête en position 23 (sauf sous-normal f32)
    let lead_exp = e_unb + top; // exposant réel du bit de tête
    let e8 = lead_exp + bias;
    let min_sub_exp = 1 - bias - mbits as i32; // exposant du plus petit sous-normal
    if lead_exp < min_sub_exp - 1
    {
        return Ok(sign); // sous le demi plus petit sous-normal ⇒ ±0
    }
    // nombre de bits de mantisse conservés : mbits (normal) ou moins (sous-normal)
    let (kept_bits, e_field): (i32, u8) = if e8 >= 1
    {
        (mbits as i32, e8 as u8)
    }
    else
    {
        ((mbits as i32 + e8 - 1).max(-1), 0)
    };
    // tronque m24 (24 bits significatifs) à 1+kept_bits bits
    let shift = (23 - kept_bits).max(0) as u32;
    let kept = m24 >> shift;
    let rem = m24 & ((1u32 << shift) - 1);
    Err((sign, kept, rem, shift, e_field))
}

/// Termine l'encodage FP8 : combine `kept`/`e_field` en code, applique le
/// report d'arrondi (`round_up`) et la saturation au max fini.
fn fp8_finish(sign: u8, kept: u32, e_field: u8, mbits: u32, round_up: bool, fmt: Fp8Format) -> u8 {
    let mut code = if e_field >= 1
    {
        (((e_field as u32) << mbits) | (kept & ((1 << mbits) - 1))) as u8
    }
    else
    {
        kept as u8 // sous-normal : e=0, mantisse = kept
    };
    if round_up
    {
        code += 1; // le report monte proprement exposant/format (codes contigus)
    }
    // le report peut dépasser le max fini
    match fmt
    {
        Fp8Format::E4M3 if code >= 0x7f => code = 0x7e,
        Fp8Format::E5M2 if code >= 0x7c => code = 0x7b,
        _ =>
        {},
    }
    sign | code
}

/// f32 → FP8 (RNE, **saturant** au max fini — convention d'entraînement).
/// NaN → NaN canonique du format ; ±inf → saturation (E4M3) ou ±inf (E5M2).
pub fn f32_to_fp8_rne(x: f32, fmt: Fp8Format) -> u8 {
    let (kept, rem, shift, e_field, sign) = match fp8_pre_round(x, fmt)
    {
        Ok(code) => return code,
        Err((sign, kept, rem, shift, e_field)) => (kept, rem, shift, e_field, sign),
    };
    let half = 1u32 << (shift - 1);
    let round_up = rem > half || (rem == half && (kept & 1) == 1);
    fp8_finish(sign, kept, e_field, fmt.params().1, round_up, fmt)
}

/// FP8 → f32 (exact).
pub fn fp8_to_f32(c: u8, fmt: Fp8Format) -> f32 {
    let (bias, mbits, _, _) = fmt.params();
    let sign = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((c >> mbits) & (0x7f >> mbits)) as i32;
    let m = (c & ((1 << mbits) - 1)) as i32;
    match fmt
    {
        Fp8Format::E4M3 if e == 0xf && m == 0x7 => return f32::NAN * sign,
        Fp8Format::E5M2 if e == 0x1f =>
        {
            return if m == 0
            {
                f32::INFINITY * sign
            }
            else
            {
                f32::NAN * sign
            };
        },
        _ =>
        {},
    }
    let (mant, exp) = if e == 0
    {
        (m as f32, 1 - bias - mbits as i32) // sous-normal : m·2^(1−biais−mbits)
    }
    else
    {
        ((m + (1 << mbits)) as f32, e - bias - mbits as i32)
    };
    sign * mant * 2f32.powi(exp)
}

// ================================================================== //
//  Arrondi stochastique reproductible (Philox)                        //
// ================================================================== //

/// f32 → bf16 par **arrondi stochastique** : monte avec probabilité
/// (reste/ulp), tirée du flux Philox au rang `index` — reproductible et
/// indépendante du découpage (contrairement à un RNG séquentiel).
/// Non biaisé : E[valeur] = x.
pub fn f32_to_bf16_stochastic(x: f32, rng: &Philox4x32, stream: u32, index: u64) -> u16 {
    let bits = x.to_bits();
    if x.is_nan()
    {
        return 0x7fc0 | (((bits >> 16) as u16) & 0x8000);
    }
    let lower = bits & 0xffff;
    let upper = (bits >> 16) as u16;
    if lower == 0 || (upper & 0x7f80) == 0x7f80
    {
        return upper; // exact, ou inf (pas d'arrondi)
    }
    // monte ssi tirage < reste (tirage uniforme sur 16 bits)
    let draw = rng.u32_at(stream, index) >> 16;
    if draw < lower
    {
        upper.wrapping_add(1)
    }
    else
    {
        upper
    }
}

/// f32 → FP8 par **arrondi stochastique** (Philox contre-basé) : monte avec
/// probabilité (reste/2^shift), non biaisé, reproductible et indépendant du
/// découpage — le pendant FP8 de [`f32_to_bf16_stochastic`], sur la
/// décomposition partagée [`fp8_pre_round`].
pub fn f32_to_fp8_stochastic(
    x: f32,
    fmt: Fp8Format,
    rng: &Philox4x32,
    stream: u32,
    index: u64,
) -> u8 {
    let (kept, rem, shift, e_field, sign) = match fp8_pre_round(x, fmt)
    {
        Ok(code) => return code,
        Err((sign, kept, rem, shift, e_field)) => (kept, rem, shift, e_field, sign),
    };
    // tirage uniforme sur `shift` bits (shift ∈ [~20,24] ⇒ tient dans un
    // tirage u32) ; monte ssi tirage < reste (reste nul ⇒ jamais monté).
    let draw = rng.u32_at(stream, index) >> (32 - shift);
    fp8_finish(sign, kept, e_field, fmt.params().1, draw < rem, fmt)
}

// ================================================================== //
//  GEMM bf16 exact                                                    //
// ================================================================== //

/// GEMM `C = A·B` sur entrées **bf16** (codes u16, row-major) avec
/// accumulation f64 en ordre fixe : les produits bf16×bf16 sont exacts en
/// f32 (8×8 bits de mantisse), la somme f64 est en ordre figé — résultat
/// f32 bit-exact inter-plates-formes. Le pendant basse précision de
/// `portable_f32::gemm_f32`.
pub fn gemm_bf16_exact(a: &[u16], b: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "gemm_bf16_exact: A doit être m×k");
    assert_eq!(b.len(), k * n, "gemm_bf16_exact: B doit être k×n");
    let mut c = vec![0.0f32; m * n];
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0.0f64;
            for l in 0..k
            {
                let x = bf16_to_f32(a[i * k + l]) as f64;
                let y = bf16_to_f32(b[l * n + j]) as f64;
                acc += x * y; // produit exact (16 bits de mantisse ≤ 53)
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;
    use crate::portable_f32::{fnv1a_fold_bits, fnv1a_init};

    /// bf16 RNE : équivaut exactement à l'arrondi f32→bf16 de référence
    /// (comparaison au demi-ulp via l'entier des 16 bits bas) sur un balayage
    /// dense de l'espace des bits, spéciaux compris.
    #[test]
    fn bf16_rne_matches_reference_dense_sweep() {
        let mut i = 0u64;
        while i <= u32::MAX as u64
        {
            let x = f32::from_bits(i as u32);
            let got = f32_to_bf16_rne(x);
            if x.is_nan()
            {
                assert_eq!(got & 0x7fff, 0x7fc0);
            }
            else
            {
                // référence : plus proche des deux bf16 encadrants
                let lo = (i as u32) >> 16;
                let rem = (i as u32) & 0xffff;
                let expected = if rem > 0x8000 || (rem == 0x8000 && lo & 1 == 1)
                {
                    (lo as u16).wrapping_add(1)
                }
                else
                {
                    lo as u16
                };
                assert_eq!(got, expected, "x={x} bits={i:08x}");
            }
            i += 65_537;
        }
    }

    /// f16 : aller-retour exact pour tous les codes f16 (les 65 536),
    /// sous-normaux, infinis et NaN compris.
    #[test]
    fn f16_roundtrip_all_codes() {
        for h in 0..=u16::MAX
        {
            let x = f16_to_f32(h);
            if x.is_nan()
            {
                assert!((h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0);
                continue;
            }
            let back = f32_to_f16_rne(x);
            assert_eq!(back, h, "code {h:04x} → {x} → {back:04x}");
        }
    }

    /// FP8 : aller-retour exact pour les 256 codes des deux formats.
    #[test]
    fn fp8_roundtrip_all_codes() {
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2]
        {
            for c in 0..=u8::MAX
            {
                let x = fp8_to_f32(c, fmt);
                if x.is_nan()
                {
                    continue; // NaN → NaN canonique (code différent possible)
                }
                let back = f32_to_fp8_rne(x, fmt);
                assert_eq!(back, c, "{fmt:?} code {c:02x} → {x} → {back:02x}");
            }
        }
    }

    /// FP8 : frontières d'arrondi exactes — pour chaque paire de codes
    /// consécutifs, le milieu arrondit au code PAIR, et milieu ± 1 ulp f32
    /// arrondit au bon côté.
    #[test]
    fn fp8_rounding_boundaries() {
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2]
        {
            for c in 0..0x7du8
            {
                let (a, b) = (fp8_to_f32(c, fmt), fp8_to_f32(c + 1, fmt));
                if !a.is_finite() || !b.is_finite() || b <= a
                {
                    continue;
                }
                let mid = (a as f64 + b as f64) * 0.5;
                let mid_f32 = mid as f32; // milieu exactement représentable
                assert_eq!(mid_f32 as f64, mid);
                let down = f32::from_bits(mid_f32.to_bits() - 1);
                let up = f32::from_bits(mid_f32.to_bits() + 1);
                assert_eq!(f32_to_fp8_rne(down, fmt), c, "{fmt:?} sous {mid}");
                assert_eq!(f32_to_fp8_rne(up, fmt), c + 1, "{fmt:?} sur {mid}");
                let at_mid = f32_to_fp8_rne(mid_f32, fmt);
                let even = if c & 1 == 0 { c } else { c + 1 };
                assert_eq!(at_mid, even, "{fmt:?} milieu {mid} → pair");
            }
        }
    }

    /// E4M3 : valeurs remarquables de la spec (max fini 448, pas d'infini).
    #[test]
    fn fp8_spec_landmarks() {
        assert_eq!(fp8_to_f32(0x7e, Fp8Format::E4M3), 448.0);
        assert!(fp8_to_f32(0x7f, Fp8Format::E4M3).is_nan());
        assert_eq!(f32_to_fp8_rne(1e9, Fp8Format::E4M3), 0x7e); // sature
        assert_eq!(fp8_to_f32(0x7b, Fp8Format::E5M2), 57_344.0);
        assert_eq!(fp8_to_f32(0x7c, Fp8Format::E5M2), f32::INFINITY);
        assert_eq!(f32_to_fp8_rne(f32::INFINITY, Fp8Format::E5M2), 0x7c);
        assert_eq!(f32_to_fp8_rne(1e9, Fp8Format::E5M2), 0x7b); // sature (fini)
        // plus petits sous-normaux : E4M3 = 2⁻⁹, E5M2 = 2⁻¹⁶
        assert_eq!(fp8_to_f32(0x01, Fp8Format::E4M3), 2f32.powi(-9));
        assert_eq!(fp8_to_f32(0x01, Fp8Format::E5M2), 2f32.powi(-16));
    }

    /// Arrondi stochastique : reproductible (mêmes index ⇒ mêmes bits, quel
    /// que soit l'ordre d'appel), non biaisé (moyenne ≈ x), et dégénère en
    /// identité sur les valeurs exactes.
    #[test]
    fn stochastic_rounding_is_reproducible_and_unbiased() {
        let rng = Philox4x32::new(2026);
        let x = 1.003_921_6f32; // entre deux bf16
        // ordre d'appel inversé ⇒ mêmes résultats aux mêmes index
        let fwd: Vec<u16> = (0..64)
            .map(|i| f32_to_bf16_stochastic(x, &rng, 0, i))
            .collect();
        let rev: Vec<u16> = (0..64)
            .rev()
            .map(|i| f32_to_bf16_stochastic(x, &rng, 0, i))
            .collect();
        for i in 0..64
        {
            assert_eq!(fwd[i], rev[63 - i]);
        }
        // non-biais : moyenne des décodés ≈ x (loi des grands nombres)
        let n = 100_000u64;
        let mean: f64 = (0..n)
            .map(|i| bf16_to_f32(f32_to_bf16_stochastic(x, &rng, 1, i)) as f64)
            .sum::<f64>()
            / n as f64;
        let (lo, hi) = (
            bf16_to_f32(f32_to_bf16_rne(x) - 1),
            bf16_to_f32(f32_to_bf16_rne(x)),
        );
        assert!(
            (mean - x as f64).abs() < (hi - lo) as f64 * 0.02,
            "biais : moyenne {mean} vs {x} (grille [{lo}, {hi}])"
        );
        // valeur exacte : jamais perturbée
        assert_eq!(
            f32_to_bf16_stochastic(1.0, &rng, 2, 0),
            f32_to_bf16_rne(1.0)
        );
    }

    /// FP8 arrondi stochastique : pour toute paire de codes consécutifs (les
    /// deux formats), un point pris strictement entre les deux atterrit
    /// TOUJOURS sur l'un des deux voisins (jamais un troisième code — la
    /// décomposition [`fp8_pre_round`] reste cohérente même en sous-normal),
    /// avec une fréquence de montée qui suit le reste (test au point 1/4 :
    /// nettement plus souvent le bas que le haut).
    #[test]
    fn fp8_stochastic_lands_on_rne_neighbors_and_tracks_remainder() {
        let rng = Philox4x32::new(0x05ee_df98);
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2]
        {
            for c in 0..0x7du8
            {
                let (a, b) = (fp8_to_f32(c, fmt), fp8_to_f32(c + 1, fmt));
                if !a.is_finite() || !b.is_finite() || b <= a
                {
                    continue;
                }
                // point au quart de l'intervalle : reste ≈ 1/4 côté bas
                let q1 = (a as f64 + (b as f64 - a as f64) * 0.25) as f32;
                let mut up = 0u32;
                let n = 400u64;
                for i in 0..n
                {
                    let code = f32_to_fp8_stochastic(q1, fmt, &rng, c as u32, i);
                    assert!(
                        code == c || code == c + 1,
                        "{fmt:?} code {c:02x}: tirage {code:02x} hors des voisins"
                    );
                    if code == c + 1
                    {
                        up += 1;
                    }
                }
                let freq = up as f64 / n as f64;
                assert!(
                    freq < 0.45,
                    "{fmt:?} code {c:02x}: fréquence de montée {freq} trop haute pour 1/4"
                );
            }
        }
    }

    /// FP8 arrondi stochastique : reproductible (mêmes index ⇒ mêmes bits),
    /// non biaisé (moyenne ≈ x), valeurs exactes jamais perturbées — le
    /// pendant FP8 de `stochastic_rounding_is_reproducible_and_unbiased`.
    #[test]
    fn fp8_stochastic_rounding_is_reproducible_and_unbiased() {
        let rng = Philox4x32::new(4_102_026);
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2]
        {
            let x = fp8_to_f32(0x30, fmt) * 1.3; // entre deux codes FP8
            let fwd: Vec<u8> = (0..64)
                .map(|i| f32_to_fp8_stochastic(x, fmt, &rng, 0, i))
                .collect();
            let rev: Vec<u8> = (0..64)
                .rev()
                .map(|i| f32_to_fp8_stochastic(x, fmt, &rng, 0, i))
                .collect();
            for i in 0..64
            {
                assert_eq!(fwd[i], rev[63 - i], "{fmt:?} non reproductible");
            }
            let n = 50_000u64;
            let mean: f64 = (0..n)
                .map(|i| fp8_to_f32(f32_to_fp8_stochastic(x, fmt, &rng, 1, i), fmt) as f64)
                .sum::<f64>()
                / n as f64;
            let rne = f32_to_fp8_rne(x, fmt);
            let (lo, hi) = (fp8_to_f32(rne - 1, fmt), fp8_to_f32(rne, fmt));
            let grid = (hi - lo).abs().max(fp8_to_f32(rne + 1, fmt) - hi);
            assert!(
                (mean - x as f64).abs() < grid as f64 * 0.05,
                "{fmt:?} biais : moyenne {mean} vs {x}"
            );
            // valeur exacte (code pile) : jamais perturbée
            let exact = fp8_to_f32(0x20, fmt);
            assert_eq!(
                f32_to_fp8_stochastic(exact, fmt, &rng, 2, 0),
                f32_to_fp8_rne(exact, fmt)
            );
        }
    }

    /// GEMM bf16 exact : petits entiers exacts + empreinte-contrat.
    #[test]
    fn gemm_bf16_exact_works_and_is_fingerprinted() {
        let a: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| f32_to_bf16_rne(x))
            .collect();
        let b: Vec<u16> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .map(|&x| f32_to_bf16_rne(x))
            .collect();
        assert_eq!(
            gemm_bf16_exact(&a, &b, 2, 2, 2),
            vec![19.0, 22.0, 43.0, 50.0]
        );

        let mut rng = PcgEngine::new(1113);
        let a: Vec<u16> = (0..9 * 7)
            .map(|_| f32_to_bf16_rne(rng.float() * 4.0 - 2.0))
            .collect();
        let b: Vec<u16> = (0..7 * 5)
            .map(|_| f32_to_bf16_rne(rng.float() * 4.0 - 2.0))
            .collect();
        let c = gemm_bf16_exact(&a, &b, 9, 7, 5);
        let mut fp = fnv1a_init();
        for &v in &c
        {
            fp = fnv1a_fold_bits(fp, v.to_bits());
        }
        assert_eq!(
            fp, 0xa655_1c8f_a3cd_f155,
            "empreinte gemm bf16 : 0x{fp:016x}"
        );
    }
}
