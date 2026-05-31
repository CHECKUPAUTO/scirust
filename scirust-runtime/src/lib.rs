//! scirust-runtime : runtime d'inference deterministe.
//! #1 keystone : poids figes -> inference bit-exact rejouable.
//! SRT1 : cles triees => octets disque deterministes (artefact hashable).
//! Manifeste : reconstruction generique de n'importe quel Sequential.

use scirust_core::autodiff::reverse::Tensor;
use scirust_core::nn::{
    Conv2d, KaimingNormal, Linear, MaxPool2d, Padding, PcgEngine, ReLU, Sequential, Zeros,
};
use std::collections::HashMap;
use std::io;

const MAGIC: &[u8; 4] = b"SRT1";

pub fn save_weights(sd: &HashMap<String, Tensor>, path: &str) -> io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);
    let mut keys: Vec<&String> = sd.keys().collect();
    keys.sort();
    buf.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for k in keys {
        let t = &sd[k];
        let (rows, cols) = t.shape();
        let kb = k.as_bytes();
        buf.extend_from_slice(&(kb.len() as u32).to_le_bytes());
        buf.extend_from_slice(kb);
        buf.extend_from_slice(&(rows as u32).to_le_bytes());
        buf.extend_from_slice(&(cols as u32).to_le_bytes());
        buf.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
        for &x in &t.data {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, &buf)
}

fn rd_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}
fn rd_u64(b: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}

pub fn load_weights(path: &str) -> io::Result<HashMap<String, Tensor>> {
    let buf = std::fs::read(path)?;
    if buf.len() < 8 || &buf[0..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "mauvais magic SRT1"));
    }
    let mut p = 4usize;
    let n = rd_u32(&buf, &mut p) as usize;
    let mut map = HashMap::new();
    for _ in 0..n {
        let klen = rd_u32(&buf, &mut p) as usize;
        let key = String::from_utf8(buf[p..p + klen].to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "cle utf8"))?;
        p += klen;
        let rows = rd_u32(&buf, &mut p) as usize;
        let cols = rd_u32(&buf, &mut p) as usize;
        let dlen = rd_u64(&buf, &mut p) as usize;
        let mut data = Vec::with_capacity(dlen);
        for _ in 0..dlen {
            data.push(f32::from_le_bytes(buf[p..p + 4].try_into().unwrap()));
            p += 4;
        }
        map.insert(key, Tensor::from_vec(data, rows, cols));
    }
    Ok(map)
}

pub fn fnv_init() -> u64 {
    0xcbf29ce484222325
}
pub fn fnv_fold_f32(mut fp: u64, data: &[f32]) -> u64 {
    for &x in data {
        fp ^= x.to_bits() as u64;
        fp = fp.wrapping_mul(0x100000001b3);
    }
    fp
}
pub fn fnv_bytes(data: &[u8]) -> u64 {
    let mut fp = fnv_init();
    for &b in data {
        fp ^= b as u64;
        fp = fp.wrapping_mul(0x100000001b3);
    }
    fp
}

#[derive(Debug, Clone, Copy)]
pub enum LayerSpec {
    Linear { in_f: usize, out_f: usize },
    Relu,
    Conv2d { in_c: usize, out_c: usize, kernel: usize, stride: usize, same: bool, in_h: usize, in_w: usize },
    MaxPool2d { kernel: usize, stride: usize, c: usize, h: usize, w: usize },
}

pub fn build_model(specs: &[LayerSpec]) -> Sequential {
    let mut rng = PcgEngine::new(0); // poids ecrases par load_state_dict ; seul le shape compte
    let mut m = Sequential::new();
    for s in specs {
        m = match *s {
            LayerSpec::Linear { in_f, out_f } => {
                m.add(Linear::new(in_f, out_f, &KaimingNormal, &Zeros, &mut rng))
            }
            LayerSpec::Relu => m.add(ReLU::new()),
            LayerSpec::Conv2d { in_c, out_c, kernel, stride, same, in_h, in_w } => {
                let pad = if same { Padding::Same } else { Padding::Valid };
                m.add(
                    Conv2d::new(in_c, out_c, kernel, stride, pad, &KaimingNormal, Some(&Zeros), &mut rng)
                        .input_dims(in_h, in_w),
                )
            }
            LayerSpec::MaxPool2d { kernel, stride, c, h, w } => {
                m.add(MaxPool2d::new(kernel, stride).input_shape(c, h, w))
            }
        };
    }
    m
}

pub fn write_manifest(specs: &[LayerSpec]) -> String {
    let mut s = String::new();
    for sp in specs {
        match *sp {
            LayerSpec::Linear { in_f, out_f } => s.push_str(&format!("linear {in_f} {out_f}\n")),
            LayerSpec::Relu => s.push_str("relu\n"),
            LayerSpec::Conv2d { in_c, out_c, kernel, stride, same, in_h, in_w } => s.push_str(&format!(
                "conv2d {in_c} {out_c} {kernel} {stride} {} {in_h} {in_w}\n",
                if same { "same" } else { "valid" }
            )),
            LayerSpec::MaxPool2d { kernel, stride, c, h, w } => {
                s.push_str(&format!("maxpool2d {kernel} {stride} {c} {h} {w}\n"))
            }
        }
    }
    s
}

pub fn parse_manifest(text: &str) -> Result<Vec<LayerSpec>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let ln = i + 1;
        let num = |s: &str| s.parse::<usize>().map_err(|_| format!("ligne {ln}: nombre invalide '{s}'"));
        let spec = match t[0] {
            "linear" => LayerSpec::Linear { in_f: num(t[1])?, out_f: num(t[2])? },
            "relu" => LayerSpec::Relu,
            "conv2d" => LayerSpec::Conv2d {
                in_c: num(t[1])?,
                out_c: num(t[2])?,
                kernel: num(t[3])?,
                stride: num(t[4])?,
                same: match t[5] {
                    "same" => true,
                    "valid" => false,
                    x => return Err(format!("ligne {ln}: padding '{x}'")),
                },
                in_h: num(t[6])?,
                in_w: num(t[7])?,
            },
            "maxpool2d" => LayerSpec::MaxPool2d {
                kernel: num(t[1])?,
                stride: num(t[2])?,
                c: num(t[3])?,
                h: num(t[4])?,
                w: num(t[5])?,
            },
            other => return Err(format!("ligne {ln}: couche inconnue '{other}'")),
        };
        out.push(spec);
    }
    Ok(out)
}
