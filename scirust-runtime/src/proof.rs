//! Bundle de preuves verifiable pour un artefact QSR1.
//! Rien n est cru sur parole: verify() re-derive chaque champ depuis les octets.
use crate::quant::{QLayer, QModel};
use crate::{fnv_fold_f32, fnv_init};
use sha2::{Digest, Sha256};

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    for b in d.iter() { s.push_str(&format!("{:02x}", b)); }
    s
}
fn sha256_f32_hex(x: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in x { h.update(v.to_bits().to_le_bytes()); }
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    for b in d.iter() { s.push_str(&format!("{:02x}", b)); }
    s
}

/// Entree canonique reproductible (splitmix64 -> [-1,1)).
pub fn gen_input(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    (0..n).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect()
}

pub struct VectorClaim {
    pub seed: u64,
    pub input_len: usize,
    pub out_fp: u64,
    pub out_sha256: String,
}

pub struct ProofBundle {
    pub format: String,
    pub artifact_sha256: String,
    pub artifact_bytes: usize,
    pub batch: usize,
    pub layers: usize,
    pub out_dim: usize,
    pub ram_scratch_bytes: usize,
    pub mac_count: u64,
    pub vectors: Vec<VectorClaim>,
}

impl ProofBundle {
    pub fn build(artifact: &[u8], batch: usize, seeds: &[u64]) -> ProofBundle {
        let model = QModel::from_bytes(artifact).expect("artefact QSR1 invalide");
        let cert = scirust_edge::resource_certificate(artifact, batch).expect("cert");
        let in0 = match &model.layers[0] { QLayer::Linear(l) => l.in_f };
        let mut vectors = Vec::new();
        for &seed in seeds {
            let input_len = batch * in0;
            let input = gen_input(seed, input_len);
            let std_out = model.infer(&input, batch);
            let (na, nacc, nout) = scirust_edge::buffer_requirements(artifact, batch).unwrap();
            let mut aa = vec![0i8; na];
            let mut ab = vec![0i8; na];
            let mut ac = vec![0i32; nacc];
            let mut o = vec![0.0f32; nout];
            let m = scirust_edge::infer(artifact, &input, batch, &mut aa, &mut ab, &mut ac, &mut o).unwrap();
            let edge_out = &o[..m];
            assert!(std_out.len() == edge_out.len()
                && std_out.iter().zip(edge_out).all(|(x, y)| x.to_bits() == y.to_bits()),
                "std != no_std a la construction du bundle");
            vectors.push(VectorClaim {
                seed,
                input_len,
                out_fp: fnv_fold_f32(fnv_init(), &std_out),
                out_sha256: sha256_f32_hex(&std_out),
            });
        }
        ProofBundle {
            format: "SCIRUST-PROOF-1".into(),
            artifact_sha256: sha256_hex(artifact),
            artifact_bytes: artifact.len(),
            batch,
            layers: cert.layers,
            out_dim: cert.out_dim,
            ram_scratch_bytes: cert.scratch_ram_bytes,
            mac_count: cert.mac_count,
            vectors,
        }
    }

    fn body(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("format={}\n", self.format));
        s.push_str(&format!("artifact_sha256={}\n", self.artifact_sha256));
        s.push_str(&format!("artifact_bytes={}\n", self.artifact_bytes));
        s.push_str(&format!("batch={}\n", self.batch));
        s.push_str(&format!("layers={}\n", self.layers));
        s.push_str(&format!("out_dim={}\n", self.out_dim));
        s.push_str(&format!("ram_scratch_bytes={}\n", self.ram_scratch_bytes));
        s.push_str(&format!("mac_count={}\n", self.mac_count));
        s.push_str(&format!("vectors={}\n", self.vectors.len()));
        for (i, v) in self.vectors.iter().enumerate() {
            s.push_str(&format!("vec.{}.seed={}\n", i, v.seed));
            s.push_str(&format!("vec.{}.input_len={}\n", i, v.input_len));
            s.push_str(&format!("vec.{}.out_fp={:016x}\n", i, v.out_fp));
            s.push_str(&format!("vec.{}.out_sha256={}\n", i, v.out_sha256));
        }
        s
    }
    pub fn bundle_digest(&self) -> String { sha256_hex(self.body().as_bytes()) }
    pub fn to_canonical(&self) -> String {
        let mut s = self.body();
        s.push_str(&format!("bundle_sha256={}\n", self.bundle_digest()));
        s
    }
}

pub fn parse_canonical(text: &str) -> Result<ProofBundle, String> {
    use std::collections::HashMap;
    let mut map: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        if line.is_empty() { continue; }
        let (k, v) = line.split_once('=').ok_or_else(|| format!("ligne sans = : {line}"))?;
        map.insert(k.to_string(), v.to_string());
    }
    let g = |k: &str| -> Result<String, String> { map.get(k).cloned().ok_or_else(|| format!("champ manquant {k}")) };
    let gu = |k: &str| -> Result<u64, String> { map.get(k).ok_or_else(|| format!("champ manquant {k}"))?.parse().map_err(|_| format!("u64 invalide {k}")) };
    let gus = |k: &str| -> Result<usize, String> { map.get(k).ok_or_else(|| format!("champ manquant {k}"))?.parse().map_err(|_| format!("usize invalide {k}")) };
    let nv = gus("vectors")?;
    let mut vectors = Vec::with_capacity(nv);
    for i in 0..nv {
        let fp_s = g(&format!("vec.{i}.out_fp"))?;
        vectors.push(VectorClaim {
            seed: gu(&format!("vec.{i}.seed"))?,
            input_len: gus(&format!("vec.{i}.input_len"))?,
            out_fp: u64::from_str_radix(&fp_s, 16).map_err(|_| format!("fp invalide vec {i}"))?,
            out_sha256: g(&format!("vec.{i}.out_sha256"))?,
        });
    }
    Ok(ProofBundle {
        format: g("format")?,
        artifact_sha256: g("artifact_sha256")?,
        artifact_bytes: gus("artifact_bytes")?,
        batch: gus("batch")?,
        layers: gus("layers")?,
        out_dim: gus("out_dim")?,
        ram_scratch_bytes: gus("ram_scratch_bytes")?,
        mac_count: gu("mac_count")?,
        vectors,
    })
}

pub fn verify(b: &ProofBundle, artifact: &[u8]) -> Vec<(String, bool)> {
    let mut r = Vec::new();
    r.push(("sha256 artefact".into(), sha256_hex(artifact) == b.artifact_sha256));
    r.push(("taille artefact".into(), artifact.len() == b.artifact_bytes));
    match scirust_edge::resource_certificate(artifact, b.batch) {
        Ok(c) => {
            r.push(("RAM scratch (cert)".into(), c.scratch_ram_bytes == b.ram_scratch_bytes));
            r.push(("MAC (cert)".into(), c.mac_count == b.mac_count));
            r.push(("couches (cert)".into(), c.layers == b.layers));
            r.push(("dim sortie (cert)".into(), c.out_dim == b.out_dim));
        }
        Err(_) => r.push(("cert ressources (parse artefact)".into(), false)),
    }
    let model = QModel::from_bytes(artifact);
    for (i, v) in b.vectors.iter().enumerate() {
        let ok = (|| -> Result<bool, ()> {
            let model = model.as_ref().map_err(|_| ())?;
            let input = gen_input(v.seed, v.input_len);
            let std_out = model.infer(&input, b.batch);
            let (na, nacc, nout) = scirust_edge::buffer_requirements(artifact, b.batch).map_err(|_| ())?;
            let mut aa = vec![0i8; na];
            let mut ab = vec![0i8; na];
            let mut ac = vec![0i32; nacc];
            let mut o = vec![0.0f32; nout];
            let m = scirust_edge::infer(artifact, &input, b.batch, &mut aa, &mut ab, &mut ac, &mut o).map_err(|_| ())?;
            let edge_out = &o[..m];
            let bit_eq = std_out.len() == edge_out.len()
                && std_out.iter().zip(edge_out).all(|(x, y)| x.to_bits() == y.to_bits());
            let fp_std = fnv_fold_f32(fnv_init(), &std_out);
            let fp_edge = fnv_fold_f32(fnv_init(), edge_out);
            let sha_std = sha256_f32_hex(&std_out);
            Ok(bit_eq && fp_std == v.out_fp && fp_edge == v.out_fp && sha_std == v.out_sha256)
        })().unwrap_or(false);
        r.push((format!("conformite vec {i} (std==no_std + fp + sha)"), ok));
    }
    r
}

pub fn verify_file(text: &str, artifact: &[u8]) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let bundle = match parse_canonical(text) {
        Ok(b) => b,
        Err(e) => { out.push((format!("parse bundle: {e}"), false)); return out; }
    };
    let claimed = text.lines().find_map(|l| l.strip_prefix("bundle_sha256=")).unwrap_or("");
    out.push(("integrite bundle (sha256 du texte)".into(), claimed == bundle.bundle_digest()));
    out.extend(verify(&bundle, artifact));
    out
}
