// AUDIT (b2) : vraie conv int8 (boucle directe entiere) vs oracle conv2d_forward.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Conv2d, KaimingNormal, Module, Padding, PcgEngine, Zeros};
use scirust_runtime::{fnv_fold_f32, fnv_init};

fn synth(n: usize, cols: usize, seed: u64) -> Tensor {
    let mut s = seed;
    let mut data = Vec::with_capacity(n * cols);
    for _ in 0..n * cols
    {
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        data.push(((z >> 40) as f32) / (1u32 << 24) as f32);
    }
    Tensor::from_vec(data, n, cols)
}

#[allow(clippy::too_many_arguments)]
fn conv_f32_mirror(
    x: &[f32],
    batch: usize,
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let h_out = (h + 2 * pad - k) / stride + 1;
    let w_out = (w + 2 * pad - k) / stride + 1;
    let kk = k * k;
    let (img_in, img_out) = (in_c * h * w, out_c * h_out * w_out);
    let mut out = vec![0.0f32; batch * img_out];
    for b in 0..batch
    {
        for oc in 0..out_c
        {
            for oh in 0..h_out
            {
                for ow in 0..w_out
                {
                    let mut sum = bias[oc];
                    for ic in 0..in_c
                    {
                        for kh in 0..k
                        {
                            for kw in 0..k
                            {
                                let ih = oh as isize * stride as isize + kh as isize - pad as isize;
                                let iw = ow as isize * stride as isize + kw as isize - pad as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize
                                {
                                    let in_idx =
                                        b * img_in + ic * h * w + ih as usize * w + iw as usize;
                                    let w_idx = oc * in_c * kk + ic * kk + kh * k + kw;
                                    sum += x[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    out[b * img_out + oc * h_out * w_out + oh * w_out + ow] = sum;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn conv_int8(
    x: &[f32],
    batch: usize,
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
    out_c: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let h_out = (h + 2 * pad - k) / stride + 1;
    let w_out = (w + 2 * pad - k) / stride + 1;
    let kk = k * k;
    let m = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s_act = if m == 0.0 { 1.0 } else { m / 127.0 };
    let x_q: Vec<i8> = x
        .iter()
        .map(|&v| (v / s_act).round().clamp(-128.0, 127.0) as i8)
        .collect();
    let mut s_w = vec![1.0f32; out_c];
    for oc in 0..out_c
    {
        let mut mm = 0.0f32;
        for j in 0..in_c * kk
        {
            mm = mm.max(weight[oc * in_c * kk + j].abs());
        }
        s_w[oc] = if mm == 0.0 { 1.0 } else { mm / 127.0 };
    }
    let w_q: Vec<i8> = (0..out_c * in_c * kk)
        .map(|idx| {
            let oc = idx / (in_c * kk);
            (weight[idx] / s_w[oc]).round().clamp(-128.0, 127.0) as i8
        })
        .collect();
    let bias_i32: Vec<i32> = (0..out_c)
        .map(|oc| (bias[oc] / (s_act * s_w[oc])).round() as i32)
        .collect();
    let (img_in, img_out) = (in_c * h * w, out_c * h_out * w_out);
    let mut out = vec![0.0f32; batch * img_out];
    for b in 0..batch
    {
        for oc in 0..out_c
        {
            for oh in 0..h_out
            {
                for ow in 0..w_out
                {
                    let mut acc: i32 = bias_i32[oc];
                    for ic in 0..in_c
                    {
                        for kh in 0..k
                        {
                            for kw in 0..k
                            {
                                let ih = oh as isize * stride as isize + kh as isize - pad as isize;
                                let iw = ow as isize * stride as isize + kw as isize - pad as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize
                                {
                                    let in_idx =
                                        b * img_in + ic * h * w + ih as usize * w + iw as usize;
                                    let w_idx = oc * in_c * kk + ic * kk + kh * k + kw;
                                    acc += x_q[in_idx] as i32 * w_q[w_idx] as i32;
                                }
                            }
                        }
                    }
                    out[b * img_out + oc * h_out * w_out + oh * w_out + ow] =
                        acc as f32 * s_act * s_w[oc];
                }
            }
        }
    }
    out
}

fn main() {
    let (n, in_c, hh, ww, out_c, k, s, pad) = (
        8usize, 3usize, 32usize, 32usize, 32usize, 3usize, 1usize, 1usize,
    );
    let x = synth(n, in_c * hh * ww, 12345);

    let mut rng = PcgEngine::new(42);
    let mut conv = Conv2d::new(
        in_c,
        out_c,
        k,
        s,
        Padding::Same,
        &KaimingNormal,
        Some(&Zeros),
        &mut rng,
    )
    .input_dims(hh, ww);
    let tape = Tape::new();
    let v = tape.input(x.clone());
    let out_v = conv.forward(&tape, v);
    let oracle = tape.value(out_v.idx()).data.clone();

    let bias_data = conv.bias.as_ref().unwrap().data.clone();
    let mirror = conv_f32_mirror(
        &x.data,
        n,
        in_c,
        hh,
        ww,
        &conv.weight.data,
        &bias_data,
        out_c,
        k,
        s,
        pad,
    );
    let int8 = conv_int8(
        &x.data,
        n,
        in_c,
        hh,
        ww,
        &conv.weight.data,
        &bias_data,
        out_c,
        k,
        s,
        pad,
    );

    // (1) miroir f32 == oracle ?
    let mirror_match = oracle.len() == mirror.len()
        && oracle
            .iter()
            .zip(mirror.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());

    // (2) int8 vs oracle
    let (mut max_abs, mut max_rel) = (0.0f32, 0.0f32);
    for (o, q) in oracle.iter().zip(int8.iter())
    {
        let ae = (o - q).abs();
        if ae > max_abs
        {
            max_abs = ae;
        }
        if o.abs() > 1e-3
        {
            let re = ae / o.abs();
            if re > max_rel
            {
                max_rel = re;
            }
        }
    }
    let fp_oracle = fnv_fold_f32(fnv_init(), &oracle);
    let fp_int8 = fnv_fold_f32(fnv_init(), &int8);

    let conv_params = out_c * in_c * k * k;
    println!();
    println!(
        "=== AUDIT b2 : conv int8 directe vs oracle (conv1 3->32, batch={}) ===",
        n
    );
    println!(
        "h_out x w_out  : {} x {}",
        (hh + 2 * pad - k) / s + 1,
        (ww + 2 * pad - k) / s + 1
    );
    println!(
        "miroir f32 == conv2d_forward : {}",
        if mirror_match
        {
            "BIT-EXACT (indexation correcte)"
        }
        else
        {
            "DIVERGE"
        }
    );
    println!(
        "int8 vs oracle : max_abs={:.3e}  max_rel={:.3e}",
        max_abs, max_rel
    );
    println!("fp oracle f32  : {:#018x}", fp_oracle);
    println!("fp conv int8   : {:#018x}", fp_int8);
    println!(
        "Filtres conv1  : f32 {} o -> int8 {} o (+{} scales)",
        conv_params * 4,
        conv_params,
        out_c
    );
}
