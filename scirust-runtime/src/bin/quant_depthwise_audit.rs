// AUDIT depthwise int8 : ref f32 (oracle = Conv2d(1,1) framework par canal) + int8 fidelite.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Conv2d, KaimingNormal, Module, Padding, PcgEngine, Zeros};
use scirust_runtime::{fnv_fold_f32, fnv_init};

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut d = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        d.push(((z >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0); // [-1,1)
    }
    d
}

#[allow(clippy::too_many_arguments)]
fn depthwise_f32(x: &[f32], batch: usize, c: usize, h: usize, w: usize, weight: &[f32], bias: &[f32], k: usize, s: usize, pad: usize) -> Vec<f32> {
    let ho = (h + 2 * pad - k) / s + 1;
    let wo = (w + 2 * pad - k) / s + 1;
    let kk = k * k;
    let (chw, cho) = (c * h * w, c * ho * wo);
    let mut out = vec![0.0f32; batch * cho];
    for b in 0..batch {
        for ch in 0..c {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut sum = bias[ch];
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = oh as isize * s as isize + kh as isize - pad as isize;
                            let iw = ow as isize * s as isize + kw as isize - pad as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                sum += x[b * chw + ch * h * w + ih as usize * w + iw as usize]
                                    * weight[ch * kk + kh * k + kw];
                            }
                        }
                    }
                    out[b * cho + ch * ho * wo + oh * wo + ow] = sum;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn depthwise_int8(x: &[f32], batch: usize, c: usize, h: usize, w: usize, weight: &[f32], bias: &[f32], k: usize, s: usize, pad: usize) -> Vec<f32> {
    let ho = (h + 2 * pad - k) / s + 1;
    let wo = (w + 2 * pad - k) / s + 1;
    let kk = k * k;
    let m = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s_act = if m == 0.0 { 1.0 } else { m / 127.0 };
    let x_q: Vec<i8> = x.iter().map(|&v| (v / s_act).round().clamp(-128.0, 127.0) as i8).collect();
    let mut s_w = vec![1.0f32; c];
    for ch in 0..c {
        let mut mm = 0.0f32;
        for j in 0..kk { mm = mm.max(weight[ch * kk + j].abs()); }
        s_w[ch] = if mm == 0.0 { 1.0 } else { mm / 127.0 };
    }
    let w_q: Vec<i8> = (0..c * kk).map(|idx| { let ch = idx / kk; (weight[idx] / s_w[ch]).round().clamp(-128.0, 127.0) as i8 }).collect();
    let bias_i32: Vec<i32> = (0..c).map(|ch| (bias[ch] / (s_act * s_w[ch])).round() as i32).collect();
    let (chw, cho) = (c * h * w, c * ho * wo);
    let mut out = vec![0.0f32; batch * cho];
    for b in 0..batch {
        for ch in 0..c {
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc: i32 = bias_i32[ch];
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = oh as isize * s as isize + kh as isize - pad as isize;
                            let iw = ow as isize * s as isize + kw as isize - pad as isize;
                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                acc += x_q[b * chw + ch * h * w + ih as usize * w + iw as usize] as i32
                                    * w_q[ch * kk + kh * k + kw] as i32;
                            }
                        }
                    }
                    out[b * cho + ch * ho * wo + oh * wo + ow] = acc as f32 * s_act * s_w[ch];
                }
            }
        }
    }
    out
}

fn main() {
    let (batch, c, h, w, k, s, pad) = (4usize, 8usize, 8usize, 8usize, 3usize, 1usize, 1usize);
    let kk = k * k;
    let (ho, wo) = ((h + 2 * pad - k) / s + 1, (w + 2 * pad - k) / s + 1);
    let x = synth(batch * c * h * w, 11);
    let weight = synth(c * kk, 22);
    let bias = synth(c, 33);

    let refr = depthwise_f32(&x, batch, c, h, w, &weight, &bias, k, s, pad);

    // oracle independant : Conv2d(1,1) du framework, canal par canal
    let mut rng = PcgEngine::new(1);
    let mut conv = Conv2d::new(1, 1, k, s, Padding::Same, &KaimingNormal, Some(&Zeros), &mut rng).input_dims(h, w);
    let mut oracle = vec![0.0f32; batch * c * ho * wo];
    for ch in 0..c {
        conv.weight = Tensor::from_vec(weight[ch * kk..ch * kk + kk].to_vec(), 1, kk);
        conv.bias = Some(Tensor::from_vec(vec![bias[ch]], 1, 1));
        let mut xc = vec![0.0f32; batch * h * w];
        for b in 0..batch {
            xc[b * h * w..(b + 1) * h * w].copy_from_slice(&x[b * c * h * w + ch * h * w..b * c * h * w + (ch + 1) * h * w]);
        }
        let tape = Tape::new();
        let v = tape.input(Tensor::from_vec(xc, batch, h * w));
        let o = conv.forward(&tape, v);
        let od = tape.value(o.idx()).data.clone();
        for b in 0..batch {
            for j in 0..ho * wo {
                oracle[b * c * ho * wo + ch * ho * wo + j] = od[b * ho * wo + j];
            }
        }
    }

    let mirror_match = refr.len() == oracle.len()
        && refr.iter().zip(oracle.iter()).all(|(a, b)| a.to_bits() == b.to_bits());

    let int8 = depthwise_int8(&x, batch, c, h, w, &weight, &bias, k, s, pad);
    let (mut max_abs, mut max_rel) = (0.0f32, 0.0f32);
    for (o, q) in refr.iter().zip(int8.iter()) {
        let ae = (o - q).abs();
        if ae > max_abs { max_abs = ae; }
        if o.abs() > 1e-3 { let re = ae / o.abs(); if re > max_rel { max_rel = re; } }
    }

    println!();
    println!("=== AUDIT depthwise int8 (C={} {}x{} k={} batch={}) ===", c, h, w, k, batch);
    println!("ref f32 == Conv2d(1,1) framework par canal : {}", if mirror_match { "BIT-EXACT (indexation correcte)" } else { "DIVERGE" });
    println!("int8 vs ref : max_abs={:.3e}  max_rel={:.3e}", max_abs, max_rel);
    println!("fp ref f32  : {:#018x}", fnv_fold_f32(fnv_init(), &refr));
    println!("fp int8     : {:#018x}", fnv_fold_f32(fnv_init(), &int8));
    println!("Filtres depthwise : f32 {} o -> int8 {} o (+{} scales)", c * kk * 4, c * kk, c);
}
