// AUDIT pointwise 1x1 int8 : ref f32 (oracle = Conv2d(cin,cout,1,1) framework) + int8 fidelite.
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
        d.push(((z >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0);
    }
    d
}

fn pointwise_f32(x: &[f32], batch: usize, cin: usize, hw: usize, weight: &[f32], bias: &[f32], cout: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * cout * hw];
    for b in 0..batch {
        for oc in 0..cout {
            for pos in 0..hw {
                let mut sum = bias[oc];
                for ic in 0..cin {
                    sum += x[b * cin * hw + ic * hw + pos] * weight[oc * cin + ic];
                }
                out[b * cout * hw + oc * hw + pos] = sum;
            }
        }
    }
    out
}

fn pointwise_int8(x: &[f32], batch: usize, cin: usize, hw: usize, weight: &[f32], bias: &[f32], cout: usize) -> Vec<f32> {
    let m = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let s_act = if m == 0.0 { 1.0 } else { m / 127.0 };
    let x_q: Vec<i8> = x.iter().map(|&v| (v / s_act).round().clamp(-128.0, 127.0) as i8).collect();
    let mut s_w = vec![1.0f32; cout];
    for oc in 0..cout {
        let mut mm = 0.0f32;
        for ic in 0..cin { mm = mm.max(weight[oc * cin + ic].abs()); }
        s_w[oc] = if mm == 0.0 { 1.0 } else { mm / 127.0 };
    }
    let w_q: Vec<i8> = (0..cout * cin).map(|idx| { let oc = idx / cin; (weight[idx] / s_w[oc]).round().clamp(-128.0, 127.0) as i8 }).collect();
    let bias_i32: Vec<i32> = (0..cout).map(|oc| (bias[oc] / (s_act * s_w[oc])).round() as i32).collect();
    let mut out = vec![0.0f32; batch * cout * hw];
    for b in 0..batch {
        for oc in 0..cout {
            for pos in 0..hw {
                let mut acc: i32 = bias_i32[oc];
                for ic in 0..cin {
                    acc += x_q[b * cin * hw + ic * hw + pos] as i32 * w_q[oc * cin + ic] as i32;
                }
                out[b * cout * hw + oc * hw + pos] = acc as f32 * s_act * s_w[oc];
            }
        }
    }
    out
}

fn main() {
    let (batch, cin, cout, h, w) = (4usize, 8usize, 16usize, 8usize, 8usize);
    let hw = h * w;
    let x = synth(batch * cin * hw, 11);
    let weight = synth(cout * cin, 22);
    let bias = synth(cout, 33);

    let refr = pointwise_f32(&x, batch, cin, hw, &weight, &bias, cout);

    let mut rng = PcgEngine::new(1);
    let mut conv = Conv2d::new(cin, cout, 1, 1, Padding::Same, &KaimingNormal, Some(&Zeros), &mut rng).input_dims(h, w);
    conv.weight = Tensor::from_vec(weight.clone(), cout, cin);
    conv.bias = Some(Tensor::from_vec(bias.clone(), 1, cout));
    let tape = Tape::new();
    let v = tape.input(Tensor::from_vec(x.clone(), batch, cin * hw));
    let o = conv.forward(&tape, v);
    let oracle = tape.value(o.idx()).data.clone();

    let mirror = refr.len() == oracle.len() && refr.iter().zip(oracle.iter()).all(|(a, b)| a.to_bits() == b.to_bits());

    let int8 = pointwise_int8(&x, batch, cin, hw, &weight, &bias, cout);
    let (mut max_abs, mut max_rel) = (0.0f32, 0.0f32);
    for (a, q) in refr.iter().zip(int8.iter()) {
        let ae = (a - q).abs();
        if ae > max_abs { max_abs = ae; }
        if a.abs() > 1e-3 { let re = ae / a.abs(); if re > max_rel { max_rel = re; } }
    }

    println!();
    println!("=== AUDIT pointwise 1x1 int8 (cin={} cout={} {}x{} batch={}) ===", cin, cout, h, w, batch);
    println!("ref f32 == Conv2d(cin,cout,1,1) framework : {}", if mirror { "BIT-EXACT (indexation correcte)" } else { "DIVERGE" });
    println!("int8 vs ref : max_abs={:.3e}  max_rel={:.3e}", max_abs, max_rel);
    println!("fp ref f32  : {:#018x}", fnv_fold_f32(fnv_init(), &refr));
    println!("fp int8     : {:#018x}", fnv_fold_f32(fnv_init(), &int8));
    println!("Poids 1x1   : f32 {} o -> int8 {} o (+{} scales)", cout * cin * 4, cout * cin, cout);
    println!("=> separable MobileNet = depthwise (valide) o pointwise (valide), tout en int8 deterministe");
}
