// AUDIT (c2) : QModel via l'API lib, sur le MLP MNIST reel.
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::quantization::quantize_per_channel;
use scirust_runtime::quant::{QLayer, QLinear, QModel};
use scirust_runtime::{fnv_bytes, fnv_fold_f32, fnv_init, load_weights};

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

fn main() {
    let dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", dir),
        format!("{}/t10k-labels-idx1-ubyte", dir),
    )
    .expect("test");
    let train = MnistDataset::load_idx(
        format!("{}/train-images-idx3-ubyte", dir),
        format!("{}/train-labels-idx1-ubyte", dir),
    )
    .expect("train");

    let sd = load_weights("mnist_mlp.srt").expect("load_weights");
    let (mut w1, mut b1, mut w2, mut b2) = (None, None, None, None);
    for (_k, t) in &sd
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
    for (xb, _y) in calib.iter()
    {
        let (bs, _) = xb.shape();
        m_in = m_in.max(max_abs(&xb.data));
        let h = linear_f32(&xb.data, bs, 784, &w1.data, &b1.data, 256);
        m_hid = m_hid.max(max_abs(&h.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>()));
    }
    let s_in = if m_in == 0.0 { 1.0 } else { m_in / 127.0 };
    let s_hid = if m_hid == 0.0 { 1.0 } else { m_hid / 127.0 };
    let b1_i32: Vec<i32> = (0..256)
        .map(|o| (b1.data[o] / (s_in * w1s[o])).round() as i32)
        .collect();
    let b2_i32: Vec<i32> = (0..10)
        .map(|o| (b2.data[o] / (s_hid * w2s[o])).round() as i32)
        .collect();

    let model = QModel {
        layers: vec![
            QLayer::Linear(QLinear {
                in_f: 784,
                out_f: 256,
                s_in,
                relu_after: true,
                scales: w1s,
                w_q: w1q,
                bias_i32: b1_i32,
            }),
            QLayer::Linear(QLinear {
                in_f: 256,
                out_f: 10,
                s_in: s_hid,
                relu_after: false,
                scales: w2s,
                w_q: w2q,
                bias_i32: b2_i32,
            }),
        ],
    };

    let path = "/tmp/mnist_mlp_lib.qsrt";
    let buf = model.save(path).expect("save");
    let hash = fnv_bytes(&buf);
    let srt1 = std::fs::metadata("mnist_mlp.srt")
        .map(|m| m.len())
        .unwrap_or(0);

    let loaded = QModel::load(path).expect("load");

    let mut loader = DataLoader::new(test.subsample(test.n), 64, false, 42);
    let (mut correct, mut total) = (0usize, 0usize);
    let mut fp = fnv_init();
    for (xb, yb) in loader.iter()
    {
        let (bs, _) = xb.shape();
        let logits = loaded.infer(&xb.data, bs);
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
        "=== AUDIT c2 : QModel via API lib (MNIST test {}) ===",
        total
    );
    println!(
        "Artefact QSR1 (lib) : {} octets  hash {:#018x}",
        buf.len(),
        hash
    );
    println!(
        "SRT1 f32            : {} octets  -> {:.2}x",
        srt1,
        srt1 as f64 / buf.len() as f64
    );
    println!(
        "Accuracy (QModel::infer depuis fichier) : {:.2}% ({}/{})",
        correct as f32 / total as f32 * 100.0,
        correct,
        total
    );
    println!(
        "Empreinte logits : {:#018x}  (attendu 0xa9b9a102c7cea67b)",
        fp
    );
    println!(
        "=> promotion lib fidele au prototype : {}",
        if fp == 0xa9b9a102c7cea67b
        {
            "OUI (bit-pour-bit)"
        }
        else
        {
            "NON"
        }
    );
}
