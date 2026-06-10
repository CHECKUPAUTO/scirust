//! Inference quantifiee int8 deterministe (artefact QSR1, auto-descriptif).
use scirust_core::quantization::{matmul_int8, quantize_tensor};
use std::io;

const QMAGIC: &[u8; 4] = b"QSR1";

#[derive(Clone, Debug)]
pub struct QLinear {
    pub in_f: usize,
    pub out_f: usize,
    pub s_in: f32,
    pub relu_after: bool,
    pub scales: Vec<f32>,
    pub w_q: Vec<i8>,
    pub bias_i32: Vec<i32>,
}

#[derive(Clone, Debug)]
pub enum QLayer {
    Linear(QLinear),
}

#[derive(Clone, Debug, Default)]
pub struct QModel {
    pub layers: Vec<QLayer>,
}

fn as_linear(layer: &QLayer) -> &QLinear {
    match layer
    {
        QLayer::Linear(l) => l,
    }
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
fn requant_i32(acc: i32, mult: i64, shift: u32) -> i64 {
    let total = 31 + shift;
    (acc as i64 * mult + (1i64 << (total - 1))) >> total
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

impl QModel {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(QMAGIC);
        b.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers
        {
            match layer
            {
                QLayer::Linear(l) =>
                {
                    b.extend_from_slice(&0u32.to_le_bytes()); // tag 0 = linear
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
                },
            }
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> io::Result<QModel> {
        if b.len() < 8 || &b[0..4] != QMAGIC
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mauvais magic QSR1",
            ));
        }
        let mut p = 4usize;
        let n = ru32(b, &mut p) as usize;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n
        {
            let tag = ru32(b, &mut p);
            match tag
            {
                0 =>
                {
                    let in_f = ru32(b, &mut p) as usize;
                    let out_f = ru32(b, &mut p) as usize;
                    let s_in = rf32(b, &mut p);
                    let relu_after = b[p] == 1;
                    p += 1;
                    let ns = ru32(b, &mut p) as usize;
                    let scales: Vec<f32> = (0..ns).map(|_| rf32(b, &mut p)).collect();
                    let nw = ru32(b, &mut p) as usize;
                    let w_q: Vec<i8> = (0..nw)
                        .map(|_| {
                            let v = b[p] as i8;
                            p += 1;
                            v
                        })
                        .collect();
                    let nb = ru32(b, &mut p) as usize;
                    let bias_i32: Vec<i32> = (0..nb).map(|_| ri32(b, &mut p)).collect();
                    layers.push(QLayer::Linear(QLinear {
                        in_f,
                        out_f,
                        s_in,
                        relu_after,
                        scales,
                        w_q,
                        bias_i32,
                    }));
                },
                other =>
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tag couche inconnu {other}"),
                    ));
                },
            }
        }
        Ok(QModel { layers })
    }

    pub fn save(&self, path: &str) -> io::Result<Vec<u8>> {
        let b = self.to_bytes();
        std::fs::write(path, &b)?;
        Ok(b)
    }
    pub fn load(path: &str) -> io::Result<QModel> {
        QModel::from_bytes(&std::fs::read(path)?)
    }

    /// Inference full-int deterministe : input (batch, in_f0) -> logits (batch, out_f_last).
    pub fn infer(&self, input: &[f32], batch: usize) -> Vec<f32> {
        assert!(!self.layers.is_empty(), "QModel vide");
        let mut cur_q = quantize_tensor(input, as_linear(&self.layers[0]).s_in);
        for li in 0..self.layers.len()
        {
            let l = as_linear(&self.layers[li]);
            let acc = matmul_int8(&cur_q, &l.w_q, batch, l.in_f, l.out_f);
            if li + 1 == self.layers.len()
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
            let s_out = as_linear(&self.layers[li + 1]).s_in;
            let mut q = vec![0i8; batch * l.out_f];
            for bi in 0..batch
            {
                for o in 0..l.out_f
                {
                    let a = acc[bi * l.out_f + o] + l.bias_i32[o];
                    let (m, sh) =
                        quantize_multiplier((l.s_in as f64 * l.scales[o] as f64) / s_out as f64);
                    let mut r = requant_i32(a, m, sh);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qmodel_roundtrip_and_deterministic() {
        let m = QModel {
            layers: vec![
                QLayer::Linear(QLinear {
                    in_f: 3,
                    out_f: 2,
                    s_in: 0.01,
                    relu_after: true,
                    scales: vec![0.002, 0.003],
                    w_q: vec![1, -2, 3, 4, -5, 6],
                    bias_i32: vec![10, -20],
                }),
                QLayer::Linear(QLinear {
                    in_f: 2,
                    out_f: 2,
                    s_in: 0.05,
                    relu_after: false,
                    scales: vec![0.004, 0.001],
                    w_q: vec![7, -8, 9, 10],
                    bias_i32: vec![0, 5],
                }),
            ],
        };
        let bytes = m.to_bytes();
        let m2 = QModel::from_bytes(&bytes).expect("from_bytes");
        assert_eq!(m2.to_bytes(), bytes, "octets non stables apres roundtrip");
        let x = vec![0.5f32, -0.3, 0.8, 0.1, 0.2, -0.4];
        let a = m.infer(&x, 2);
        let b = m2.infer(&x, 2);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "inference non identique apres roundtrip"
        );
    }
}
