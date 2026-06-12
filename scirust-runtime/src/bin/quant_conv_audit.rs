// AUDIT (b1) : fake-quant per-channel des poids du CNN, fidelite vs oracle f32.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    Conv2d, KaimingNormal, Linear, MaxPool2d, Module, Padding, PcgEngine, ReLU, Sequential, Zeros,
};
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

fn build_cnn(seed: u64) -> Sequential {
    let mut rng = PcgEngine::new(seed);
    Sequential::new()
        .add(
            Conv2d::new(
                3,
                32,
                3,
                1,
                Padding::Same,
                &KaimingNormal,
                Some(&Zeros),
                &mut rng,
            )
            .input_dims(32, 32),
        )
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(32, 32, 32))
        .add(
            Conv2d::new(
                32,
                64,
                3,
                1,
                Padding::Same,
                &KaimingNormal,
                Some(&Zeros),
                &mut rng,
            )
            .input_dims(16, 16),
        )
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(64, 16, 16))
        .add(Linear::new(4096, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng))
}

fn forward_logits(model: &mut Sequential, x: &Tensor) -> Vec<f32> {
    let tape = Tape::new();
    let v = tape.input(x.clone());
    let logits = model.forward(&tape, v);
    tape.value(logits.idx()).data.clone()
}

fn argmax(row: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut best = row[0];
    for (i, &v) in row.iter().enumerate().skip(1)
    {
        if v > best
        {
            best = v;
            bi = i;
        }
    }
    bi
}

fn fakequant_per_row(t: &Tensor) -> Tensor {
    let (rows, cols) = t.shape();
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows
    {
        let mut ma = 0.0f32;
    #[allow(clippy::needless_range_loop)]
        for c in 0..cols
        {
            ma = ma.max(t.data[r * cols + c].abs());
        }
        let s = if ma == 0.0 { 1.0 } else { ma / 127.0 };
    #[allow(clippy::needless_range_loop)]
        for c in 0..cols
        {
            let q = (t.data[r * cols + c] / s).round().clamp(-128.0, 127.0);
            out[r * cols + c] = q * s;
        }
    }
    Tensor::from_vec(out, rows, cols)
}

fn fakequant_per_col(t: &Tensor) -> Tensor {
    let (rows, cols) = t.shape();
    let mut scales = vec![1.0f32; cols];
    #[allow(clippy::needless_range_loop)]
    for c in 0..cols
    {
        let mut ma = 0.0f32;
        for r in 0..rows
        {
            ma = ma.max(t.data[r * cols + c].abs());
        }
        scales[c] = if ma == 0.0 { 1.0 } else { ma / 127.0 };
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows
    {
    #[allow(clippy::needless_range_loop)]
        for c in 0..cols
        {
            let q = (t.data[r * cols + c] / scales[c])
                .round()
                .clamp(-128.0, 127.0);
            out[r * cols + c] = q * scales[c];
        }
    }
    Tensor::from_vec(out, rows, cols)
}

fn main() {
    let n = 32usize;
    let x = synth(n, 3 * 32 * 32, 12345);

    // oracle f32
    let mut a = build_cnn(42);
    let lo = forward_logits(&mut a, &x);
    let fp_orig = fnv_fold_f32(fnv_init(), &lo);

    // fake-quant de tous les poids (conv per-row, linear per-col)
    let mut sd = a.state_dict();
    for t in sd.values_mut()
    {
        *t = match t.shape()
        {
            (32, 27) | (64, 288) => fakequant_per_row(t), // filtres conv
            (4096, 256) | (256, 10) => fakequant_per_col(t), // poids linear
            _ => t.clone(),                               // biais : inchanges
        };
    }
    let mut b = build_cnn(999);
    b.load_state_dict(&sd).expect("load_state_dict");
    let lq = forward_logits(&mut b, &x);
    let fp_q = fnv_fold_f32(fnv_init(), &lq);

    // fidelite
    let (mut max_abs, mut max_rel) = (0.0f32, 0.0f32);
    for (o, q) in lo.iter().zip(lq.iter())
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
    let mut agree = 0usize;
    for i in 0..n
    {
        if argmax(&lo[i * 10..i * 10 + 10]) == argmax(&lq[i * 10..i * 10 + 10])
        {
            agree += 1;
        }
    }

    // taille
    let conv_params = 32 * 27 + 64 * 288;
    let lin_params = 4096 * 256 + 256 * 10;
    let total = conv_params + lin_params;
    let f32b = total * 4;
    let int8b = total + (32 + 64 + 256 + 10) * 4;
    let ratio = f32b as f64 / int8b as f64;

    println!();
    println!(
        "=== AUDIT b1 : fake-quant per-channel CNN (synthetique, batch={}) ===",
        n
    );
    println!(
        "fp f32 oracle  : {:#018x}  (attendu 0x1381e4b51d0eeba4)",
        fp_orig
    );
    println!("fp poids quant : {:#018x}", fp_q);
    println!(
        "Erreur logits  : max_abs={:.3e}  max_rel={:.3e}",
        max_abs, max_rel
    );
    println!("argmax conserve: {}/{}", agree, n);
    println!("Poids f32  : {} octets", f32b);
    println!("Poids int8 : {} octets (+ scales per-channel)", int8b);
    println!("Reduction  : {:.2}x", ratio);
}
