// AUDIT FULL-INT (a2) : requant point-fixe entier pour la couche cachee.
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::quantization::{matmul_int8, quantize_per_channel, quantize_tensor};
use scirust_runtime::{fnv_fold_f32, fnv_init, load_weights};

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

fn linear_f32(
    x: &[f32],
    batch: usize,
    in_f: usize,
    w: &[f32],
    b: &[f32],
    out_f: usize,
) -> Vec<f32> {
    let mut o = vec![0.0f32; batch * out_f];
    for bi in 0..batch
    {
        for oj in 0..out_f
        {
            let mut s = 0.0f32;
            for i in 0..in_f
            {
                s += x[bi * in_f + i] * w[i * out_f + oj];
            }
            o[bi * out_f + oj] = s + b[oj];
        }
    }
    o
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

// Normalise M>0 en (mult dans [2^30,2^31), shift>=0) : M ~= mult / 2^(31+shift).
fn quantize_multiplier(m: f64) -> (i64, u32) {
    if m <= 0.0
    {
        return (0, 0);
    }
    let mut frac = m;
    let mut shift: u32 = 0;
    while frac < 0.5
    {
        frac *= 2.0;
        shift += 1;
    }
    let mut mult = (frac * (1i64 << 31) as f64).round() as i64;
    if mult >= (1i64 << 31)
    {
        mult = 1i64 << 30;
        shift = shift.saturating_sub(1);
    }
    (mult, shift)
}

// requant entier : arrondi(acc * mult / 2^(31+shift)).
fn requant(acc: i32, mult: i64, shift: u32) -> i64 {
    let total = 31 + shift;
    let prod = acc as i64 * mult;
    (prod + (1i64 << (total - 1))) >> total
}

fn main() {
    let data_dir =
        std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", data_dir),
        format!("{}/t10k-labels-idx1-ubyte", data_dir),
    )
    .expect("test");
    let train = MnistDataset::load_idx(
        format!("{}/train-images-idx3-ubyte", data_dir),
        format!("{}/train-labels-idx1-ubyte", data_dir),
    )
    .expect("train");

    let sd = load_weights("mnist_mlp.srt").expect("load_weights");
    let (mut w1, mut b1, mut w2, mut b2) = (None, None, None, None);
    for t in sd.values()
    {
        match t.shape()
        {
            (784, 256) => w1 = Some(t.clone()),
            (1, 256) => b1 = Some(t.clone()),
            (256, 10) => w2 = Some(t.clone()),
            (1, 10) => b2 = Some(t.clone()),
            _ =>
            {},
        }
    }
    let (w1, b1, w2, b2) = (w1.unwrap(), b1.unwrap(), w2.unwrap(), b2.unwrap());
    let (w1q, w1s) = quantize_per_channel(&w1.data, 784, 256);
    let (w2q, w2s) = quantize_per_channel(&w2.data, 256, 10);

    let mut calib = DataLoader::new(train.subsample(2000), 64, false, 7);
    let (mut m_in, mut m_hid) = (0.0f32, 0.0f32);
    for (xb, _yb) in calib.iter()
    {
        let (bs, _) = xb.shape();
        m_in = m_in.max(max_abs(&xb.data));
        let h = linear_f32(&xb.data, bs, 784, &w1.data, &b1.data, 256);
        let hr: Vec<f32> = h.iter().map(|&x| x.max(0.0)).collect();
        m_hid = m_hid.max(max_abs(&hr));
    }
    let s_in = if m_in == 0.0 { 1.0 } else { m_in / 127.0 };
    let s_hid = if m_hid == 0.0 { 1.0 } else { m_hid / 127.0 };

    let b1_i32: Vec<i32> = (0..256)
        .map(|o| (b1.data[o] / (s_in * w1s[o])).round() as i32)
        .collect();
    let req1: Vec<(i64, u32)> = (0..256)
        .map(|o| quantize_multiplier((s_in as f64 * w1s[o] as f64) / s_hid as f64))
        .collect();
    let b2_i32: Vec<i32> = (0..10)
        .map(|o| (b2.data[o] / (s_hid * w2s[o])).round() as i32)
        .collect();

    let mut loader = DataLoader::new(test.subsample(test.n), 64, false, 42);
    let (mut correct, mut total) = (0usize, 0usize);
    let mut fp = fnv_init();
    for (xb, yb) in loader.iter()
    {
        let (bs, _) = xb.shape();
        let q_in = quantize_tensor(&xb.data, s_in);
        let acc1 = matmul_int8(&q_in, &w1q, bs, 784, 256);
        let mut q_hid = vec![0i8; bs * 256];
        for bi in 0..bs
        {
            for o in 0..256
            {
                let a = acc1[bi * 256 + o] + b1_i32[o];
                let (m, sh) = req1[o];
                let q = requant(a, m, sh).clamp(0, 127);
                q_hid[bi * 256 + o] = q as i8;
            }
        }
        let acc2 = matmul_int8(&q_hid, &w2q, bs, 256, 10);
        let mut logits = vec![0.0f32; bs * 10];
        for bi in 0..bs
        {
            for o in 0..10
            {
                let a = acc2[bi * 10 + o] + b2_i32[o];
                logits[bi * 10 + o] = a as f32 * s_hid * w2s[o];
            }
        }
        fp = fnv_fold_f32(fp, &logits);
        for i in 0..bs
        {
            let tc = argmax(&yb.data[i * 10..(i + 1) * 10]);
            if argmax(&logits[i * 10..(i + 1) * 10]) == tc
            {
                correct += 1;
            }
            total += 1;
        }
    }

    println!();
    println!(
        "=== AUDIT FULL-INT a2 (requant point-fixe entier, MNIST test {}) ===",
        total
    );
    println!(
        "Accuracy full-int : {:.2}% ({}/{})",
        correct as f32 / total as f32 * 100.0,
        correct,
        total
    );
    println!("  (f32 97.73% ; int8 dyn 97.74% ; int8 statique a1 97.71%)");
    println!("Empreinte logits full-int : {:#018x}", fp);
    println!(
        "  hot path inter-couches 100% entier (i64 mult+shift), zero flottant; dequant final f32 sur 10 sorties."
    );
}
