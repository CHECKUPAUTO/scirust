// AUDIT (c1) : artefact quantifie QSR1 (poids int8 + scales + biais i32) round-trip + rejeu full-int.
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::quantization::{matmul_int8, quantize_per_channel, quantize_tensor};
use scirust_runtime::{fnv_bytes, fnv_fold_f32, fnv_init, load_weights};

struct QLinear {
    in_f: usize,
    out_f: usize,
    s_in: f32,
    relu_after: bool,
    scales: Vec<f32>,
    w_q: Vec<i8>,
    bias_i32: Vec<i32>,
}

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
        if shift > 0
        {
            shift -= 1;
        }
    }
    (mult, shift)
}
fn requant(acc: i32, mult: i64, shift: u32) -> i64 {
    let total = 31 + shift;
    (acc as i64 * mult + (1i64 << (total - 1))) >> total
}

fn write_qsrt(layers: &[QLinear], path: &str) -> std::io::Result<Vec<u8>> {
    let mut b = Vec::new();
    b.extend_from_slice(b"QSR1");
    b.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for l in layers
    {
        b.extend_from_slice(&(l.in_f as u32).to_le_bytes());
        b.extend_from_slice(&(l.out_f as u32).to_le_bytes());
        b.extend_from_slice(&l.s_in.to_le_bytes());
        b.push(if l.relu_after { 1 } else { 0 });
        b.extend_from_slice(&(l.scales.len() as u32).to_le_bytes());
        for &s in &l.scales
        {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b.extend_from_slice(&(l.w_q.len() as u32).to_le_bytes());
        for &q in &l.w_q
        {
            b.push(q as u8);
        }
        b.extend_from_slice(&(l.bias_i32.len() as u32).to_le_bytes());
        for &x in &l.bias_i32
        {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, &b)?;
    Ok(b)
}

fn ru32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}
fn rf32(b: &[u8], p: &mut usize) -> f32 {
    let v = f32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}
fn ri32(b: &[u8], p: &mut usize) -> i32 {
    let v = i32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}

fn read_qsrt(path: &str) -> std::io::Result<Vec<QLinear>> {
    let b = std::fs::read(path)?;
    assert_eq!(&b[0..4], b"QSR1", "mauvais magic QSR1");
    let mut p = 4usize;
    let n = ru32(&b, &mut p) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n
    {
        let in_f = ru32(&b, &mut p) as usize;
        let out_f = ru32(&b, &mut p) as usize;
        let s_in = rf32(&b, &mut p);
        let relu_after = b[p] == 1;
        p += 1;
        let ns = ru32(&b, &mut p) as usize;
        let scales: Vec<f32> = (0..ns).map(|_| rf32(&b, &mut p)).collect();
        let nw = ru32(&b, &mut p) as usize;
        let w_q: Vec<i8> = (0..nw)
            .map(|_| {
                let v = b[p] as i8;
                p += 1;
                v
            })
            .collect();
        let nb = ru32(&b, &mut p) as usize;
        let bias_i32: Vec<i32> = (0..nb).map(|_| ri32(&b, &mut p)).collect();
        out.push(QLinear {
            in_f,
            out_f,
            s_in,
            relu_after,
            scales,
            w_q,
            bias_i32,
        });
    }
    Ok(out)
}

fn infer(layers: &[QLinear], input: &[f32], batch: usize) -> Vec<f32> {
    let mut cur_q = quantize_tensor(input, layers[0].s_in);
    for li in 0..layers.len()
    {
        let l = &layers[li];
        let acc = matmul_int8(&cur_q, &l.w_q, batch, l.in_f, l.out_f);
        if li + 1 == layers.len()
        {
            let mut out = vec![0.0f32; batch * l.out_f];
            for bi in 0..batch
            {
                for o in 0..l.out_f
                {
                    let a = acc[bi * l.out_f + o] + l.bias_i32[o];
                    out[bi * l.out_f + o] = a as f32 * l.s_in * l.scales[o];
                }
            }
            return out;
        }
        let s_out = layers[li + 1].s_in;
        let mut q = vec![0i8; batch * l.out_f];
        for bi in 0..batch
        {
            for o in 0..l.out_f
            {
                let a = acc[bi * l.out_f + o] + l.bias_i32[o];
                let (m, sh) =
                    quantize_multiplier((l.s_in as f64 * l.scales[o] as f64) / s_out as f64);
                let mut r = requant(a, m, sh);
                if l.relu_after && r < 0
                {
                    r = 0;
                }
                if r > 127
                {
                    r = 127;
                }
                if r < -128
                {
                    r = -128;
                }
                q[bi * l.out_f + o] = r as i8;
            }
        }
        cur_q = q;
    }
    unreachable!()
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

    let layers = vec![
        QLinear {
            in_f: 784,
            out_f: 256,
            s_in,
            relu_after: true,
            scales: w1s,
            w_q: w1q,
            bias_i32: b1_i32,
        },
        QLinear {
            in_f: 256,
            out_f: 10,
            s_in: s_hid,
            relu_after: false,
            scales: w2s,
            w_q: w2q,
            bias_i32: b2_i32,
        },
    ];

    let path = "/tmp/mnist_mlp.qsrt";
    let buf = write_qsrt(&layers, path).expect("write_qsrt");
    let hash = fnv_bytes(&buf);
    let srt1_size = std::fs::metadata("mnist_mlp.srt")
        .map(|m| m.len())
        .unwrap_or(0);

    // reload depuis le fichier (aucun poids en dur)
    let loaded = read_qsrt(path).expect("read_qsrt");

    let mut loader = DataLoader::new(test.subsample(test.n), 64, false, 42);
    let (mut correct, mut total) = (0usize, 0usize);
    let mut fp = fnv_init();
    for (xb, yb) in loader.iter()
    {
        let (bs, _) = xb.shape();
        let logits = infer(&loaded, &xb.data, bs);
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
        "=== AUDIT c1 : artefact quantifie QSR1 (MNIST test {}) ===",
        total
    );
    println!("Artefact QSR1 : {} octets  hash {:#018x}", buf.len(), hash);
    println!(
        "SRT1 f32      : {} octets  -> reduction {:.2}x",
        srt1_size,
        srt1_size as f64 / buf.len() as f64
    );
    println!(
        "Accuracy (rejeu depuis artefact) : {:.2}% ({}/{})",
        correct as f32 / total as f32 * 100.0,
        correct,
        total
    );
    println!(
        "Empreinte logits : {:#018x}  (attendu a2 0xa9b9a102c7cea67b)",
        fp
    );
}
