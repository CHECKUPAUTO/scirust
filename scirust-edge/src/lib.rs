#![no_std]
//! scirust-edge : inference int8 deterministe a partir d'un artefact QSR1,
//! no_std et sans allocation. Reproduit bit-pour-bit
//! scirust-runtime::quant::QModel::infer (memes maths entieres).

const QMAGIC: &[u8; 4] = b"QSR1";
pub const MAX_LAYERS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeError {
    BadMagic,
    Truncated,
    UnknownTag,
    TooManyLayers,
    EmptyModel,
    BufferTooSmall,
}

#[derive(Clone, Copy, Default)]
struct Meta {
    in_f: usize,
    out_f: usize,
    s_in: f32,
    relu: bool,
    off_scales: usize,
    off_w: usize,
    off_bias: usize,
}

fn rd_u32(b: &[u8], p: &mut usize) -> Result<u32, EdgeError> {
    if *p + 4 > b.len() { return Err(EdgeError::Truncated); }
    let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
    *p += 4; Ok(v)
}
fn rd_f32(b: &[u8], p: &mut usize) -> Result<f32, EdgeError> {
    if *p + 4 > b.len() { return Err(EdgeError::Truncated); }
    let v = f32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
    *p += 4; Ok(v)
}
fn rd_u8(b: &[u8], p: &mut usize) -> Result<u8, EdgeError> {
    if *p + 1 > b.len() { return Err(EdgeError::Truncated); }
    let v = b[*p]; *p += 1; Ok(v)
}
fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn i32_at(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

// maths entieres : copie exacte de scirust-runtime::quant
fn quantize_multiplier(m: f64) -> (i64, u32) {
    if m <= 0.0 { return (0, 0); }
    let mut frac = m;
    let mut shift: u32 = 0;
    while frac < 0.5 { frac *= 2.0; shift += 1; }
    let mut mult = libm::round(frac * (1i64 << 31) as f64) as i64;
    if mult >= (1i64 << 31) { mult = 1i64 << 30; if shift > 0 { shift -= 1; } }
    (mult, shift)
}
fn requant_i32(acc: i32, mult: i64, shift: u32) -> i64 {
    let total = 31 + shift;
    (acc as i64 * mult + (1i64 << (total - 1))) >> total
}

fn parse(model: &[u8], metas: &mut [Meta; MAX_LAYERS]) -> Result<usize, EdgeError> {
    if model.len() < 8 || &model[0..4] != QMAGIC { return Err(EdgeError::BadMagic); }
    let mut p = 4usize;
    let n = rd_u32(model, &mut p)? as usize;
    if n == 0 { return Err(EdgeError::EmptyModel); }
    if n > MAX_LAYERS { return Err(EdgeError::TooManyLayers); }
    for i in 0..n {
        if rd_u32(model, &mut p)? != 0 { return Err(EdgeError::UnknownTag); }
        let in_f = rd_u32(model, &mut p)? as usize;
        let out_f = rd_u32(model, &mut p)? as usize;
        let s_in = rd_f32(model, &mut p)?;
        let relu = rd_u8(model, &mut p)? == 1;
        let ns = rd_u32(model, &mut p)? as usize;
        let off_scales = p;
        if p + ns * 4 > model.len() { return Err(EdgeError::Truncated); }
        p += ns * 4;
        let nw = rd_u32(model, &mut p)? as usize;
        let off_w = p;
        if p + nw > model.len() { return Err(EdgeError::Truncated); }
        p += nw;
        let nb = rd_u32(model, &mut p)? as usize;
        let off_bias = p;
        if p + nb * 4 > model.len() { return Err(EdgeError::Truncated); }
        p += nb * 4;
        metas[i] = Meta { in_f, out_f, s_in, relu, off_scales, off_w, off_bias };
    }
    Ok(n)
}

/// Tailles requises (en elements) : (buffer activation i8 *chacun*, buffer acc i32, sortie f32).
pub fn buffer_requirements(model: &[u8], batch: usize) -> Result<(usize, usize, usize), EdgeError> {
    let mut metas = [Meta::default(); MAX_LAYERS];
    let n = parse(model, &mut metas)?;
    let mut max_w = 0usize;
    let mut max_out = 0usize;
    for m in &metas[..n] {
        if m.in_f > max_w { max_w = m.in_f; }
        if m.out_f > max_w { max_w = m.out_f; }
        if m.out_f > max_out { max_out = m.out_f; }
    }
    Ok((batch * max_w, batch * max_out, batch * metas[n - 1].out_f))
}

/// Inference deterministe sans allocation : ecrit les logits dans `out`, renvoie leur nombre.
#[allow(clippy::too_many_arguments)]
pub fn infer(
    model: &[u8],
    input: &[f32],
    batch: usize,
    act_a: &mut [i8],
    act_b: &mut [i8],
    acc: &mut [i32],
    out: &mut [f32],
) -> Result<usize, EdgeError> {
    let mut metas = [Meta::default(); MAX_LAYERS];
    let n = parse(model, &mut metas)?;
    let (na, nacc, nout) = buffer_requirements(model, batch)?;
    if act_a.len() < na || act_b.len() < na || acc.len() < nacc || out.len() < nout {
        return Err(EdgeError::BufferTooSmall);
    }
    let in0 = metas[0].in_f;
    if input.len() < batch * in0 { return Err(EdgeError::Truncated); }
    // quantize input -> act_a (cur), comme quantize_tensor(input, s_in0)
    let s0 = metas[0].s_in;
    for idx in 0..batch * in0 {
        let mut q = libm::roundf(input[idx] / s0);
        if q < -128.0 { q = -128.0; } else if q > 127.0 { q = 127.0; }
        act_a[idx] = q as i8;
    }
    let mut cur_in_a = true;
    for li in 0..n {
        let l = metas[li];
        for bi in 0..batch {
            for o in 0..l.out_f {
                let mut sum = 0i32;
                for k in 0..l.in_f {
                    let cv = if cur_in_a { act_a[bi * l.in_f + k] } else { act_b[bi * l.in_f + k] } as i32;
                    let wv = model[l.off_w + k * l.out_f + o] as i8 as i32;
                    sum += cv * wv;
                }
                acc[bi * l.out_f + o] = sum;
            }
        }
        if li + 1 == n {
            for bi in 0..batch {
                for o in 0..l.out_f {
                    let a = acc[bi * l.out_f + o] + i32_at(model, l.off_bias + o * 4);
                    out[bi * l.out_f + o] = a as f32 * l.s_in * f32_at(model, l.off_scales + o * 4);
                }
            }
            return Ok(batch * l.out_f);
        }
        let s_out = metas[li + 1].s_in;
        for bi in 0..batch {
            for o in 0..l.out_f {
                let a = acc[bi * l.out_f + o] + i32_at(model, l.off_bias + o * 4);
                let scale_o = f32_at(model, l.off_scales + o * 4) as f64;
                let (m, sh) = quantize_multiplier((l.s_in as f64 * scale_o) / s_out as f64);
                let mut r = requant_i32(a, m, sh);
                if l.relu && r < 0 { r = 0; }
                if r > 127 { r = 127; }
                if r < -128 { r = -128; }
                if cur_in_a { act_b[bi * l.out_f + o] = r as i8; } else { act_a[bi * l.out_f + o] = r as i8; }
            }
        }
        cur_in_a = !cur_in_a;
    }
    Err(EdgeError::EmptyModel)
}

/// Certificat de ressources calcule statiquement depuis l artefact QSR1,
/// sans executer l inference. Valeurs exactes et bornees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceCert {
    pub layers: usize,
    pub batch: usize,
    pub flash_artifact_bytes: usize,
    pub scratch_ram_bytes: usize,
    pub act_bytes_each: usize,
    pub acc_bytes: usize,
    pub out_bytes: usize,
    pub mac_count: u64,
    pub out_dim: usize,
}

pub fn resource_certificate(model: &[u8], batch: usize) -> Result<ResourceCert, EdgeError> {
    let mut metas = [Meta::default(); MAX_LAYERS];
    let n = parse(model, &mut metas)?;
    let (na, nacc, nout) = buffer_requirements(model, batch)?;
    let act_bytes_each = na;
    let acc_bytes = nacc * 4;
    let out_bytes = nout * 4;
    let scratch_ram_bytes = act_bytes_each * 2 + acc_bytes + out_bytes;
    let mut mac: u64 = 0;
    for m in &metas[..n] {
        mac += batch as u64 * m.in_f as u64 * m.out_f as u64;
    }
    Ok(ResourceCert {
        layers: n,
        batch,
        flash_artifact_bytes: model.len(),
        scratch_ram_bytes,
        act_bytes_each,
        acc_bytes,
        out_bytes,
        mac_count: mac,
        out_dim: metas[n - 1].out_f,
    })
}

// --- Preuves formelles bornees (Kani). Invisibles au build de production (cfg(kani)). ---
#[cfg(kani)]
mod proofs {
    use super::requant_i32;

    // 1) requant_i32 : aucun overflow ni decalage hors limite dans le domaine documente
    //    (mult issu de quantize_multiplier : 0 <= mult < 2^31 ; shift <= 32).
    #[kani::proof]
    fn requant_no_overflow_in_envelope() {
        let acc: i32 = kani::any();
        let mult: i64 = kani::any();
        let shift: u32 = kani::any();
        kani::assume(mult >= 0 && mult < (1i64 << 31));
        kani::assume(shift <= 32);
        let _ = requant_i32(acc, mult, shift);
    }

    // 2) Dents : hors enveloppe (shift >= 34 -> total-1 >= 64), requant_i32 DOIT paniquer.
    #[kani::proof]
    #[kani::should_panic]
    fn requant_panics_outside_envelope() {
        let shift: u32 = kani::any();
        kani::assume(shift >= 34 && shift <= 40);
        let _ = requant_i32(0, 0, shift);
    }

    // 3) Lemme de conception : pour une dimension de contraction K <= 131071,
    //    le pire-cas |accumulateur| (K * 128 * 128) tient dans un i32.
    #[kani::proof]
    fn accumulator_bound_fits_i32() {
        let k: u64 = kani::any();
        kani::assume(k <= 131_071);
        let worst: u64 = k * 16_384;
        assert!(worst <= i32::MAX as u64);
    }

    // 4) Code reel d accumulation i8->i32 a la largeur MNIST (256),
    //    sans overflow i32, sur TOUTES les paires d entrees i8.
    #[kani::proof]
    #[kani::unwind(257)]
    fn dot_i8_i32_no_overflow_256() {
        let mut sum: i32 = 0;
        for _ in 0..256 {
            let a: i8 = kani::any();
            let w: i8 = kani::any();
            sum += (a as i32) * (w as i32);
        }
        let _ = sum;
    }
}
