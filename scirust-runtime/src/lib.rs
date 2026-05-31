//! scirust-runtime : runtime d'inference deterministe.
//! Garantie #1 (keystone) : poids figes -> inference bit-exact rejouable.
//! Format SRT1 : cles triees => octets disque deterministes (artefact hashable, auditable).

use scirust_core::autodiff::reverse::Tensor;
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
