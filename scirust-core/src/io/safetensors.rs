// scirust-core/src/io/safetensors.rs
//
// Sérialisation/désérialisation au format safetensors (Hugging Face).
// Implémentation minimaliste sans dépendance — JSON header + bytes f32.
//
// Format safetensors :
//   [u64 LE: header_size]
//   [header_size bytes: JSON UTF-8]
//   [data: bytes des tenseurs concaténés dans l'ordre du JSON]
//
// JSON header :
// {
//   "tensor_name": {
//     "dtype": "F32",
//     "shape": [rows, cols],
//     "data_offsets": [start, end]   // offsets DANS le data buffer
//   },
//   "__metadata__": { ... }          // optionnel
// }
//
// Cette implémentation supporte F32 uniquement et les tenseurs 2D.
// Compatible avec PyTorch/Hugging Face quand les shapes sont 2D row-major.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::fs::File;
use std::path::Path;
use crate::autodiff::reverse::Tensor;

// ================================================================== //
//  Sauvegarde                                                         //
// ================================================================== //

pub fn save_safetensors<P: AsRef<Path>>(
    tensors: &[(String, Tensor)],
    path: P,
) -> io::Result<()> {
    let bytes = serialize(tensors);
    let mut f = File::create(path)?;
    f.write_all(&bytes)?;
    Ok(())
}

pub fn serialize(tensors: &[(String, Tensor)]) -> Vec<u8> {
    // 1. Calculer les offsets et construire le JSON header
    let mut offset = 0usize;
    let mut entries: Vec<String> = Vec::with_capacity(tensors.len());

    for (name, t) in tensors {
        let n_bytes = t.data.len() * 4; // f32
        let entry = format!(
            r#""{}":{{"dtype":"F32","shape":[{},{}],"data_offsets":[{},{}]}}"#,
            escape_json(name),
            t.rows, t.cols,
            offset, offset + n_bytes,
        );
        entries.push(entry);
        offset += n_bytes;
    }

    let header = format!("{{{}}}", entries.join(","));
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    // 2. Assembler : [u64 header_size][header][data]
    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    for (_, t) in tensors {
        for &x in &t.data {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ================================================================== //
//  Chargement                                                         //
// ================================================================== //

pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
) -> io::Result<HashMap<String, Tensor>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    deserialize(&buf)
}

pub fn deserialize(bytes: &[u8]) -> io::Result<HashMap<String, Tensor>> {
    if bytes.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "fichier trop court"));
    }
    let header_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + header_size > bytes.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "header_size invalide"));
    }
    let header = std::str::from_utf8(&bytes[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let data = &bytes[8 + header_size..];

    parse_header(header, data)
}

// ------------------------------------------------------------------ //
//  Mini-parser JSON dédié au format safetensors                       //
//  Suffisant pour nos clés/types fixes — pas un parser général.       //
// ------------------------------------------------------------------ //

fn parse_header(header: &str, data: &[u8]) -> io::Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    // On cherche les motifs "name":{"dtype":"F32","shape":[r,c],"data_offsets":[s,e]}
    // Approche : itérer sur les ouvertures de "..." :{

    let bytes = header.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Trouver une clé string
        if bytes[i] != b'"' { i += 1; continue; }
        let key_start = i + 1;
        let key_end = find_unescaped_quote(&bytes[key_start..])
            .map(|p| key_start + p)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "string non terminée"))?;
        let key = &header[key_start..key_end];
        i = key_end + 1;

        // Sauter les espaces et le ':'
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':') { i += 1; }

        // Skip __metadata__
        if key == "__metadata__" {
            // Trouver l'objet correspondant et le sauter
            i = skip_balanced(bytes, i, b'{', b'}');
            continue;
        }

        // Doit être un objet { ... }
        if i >= bytes.len() || bytes[i] != b'{' {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                       format!("attendu '{{' après {key}")));
        }
        let obj_end = skip_balanced(bytes, i, b'{', b'}');
        let obj = &header[i..obj_end];
        i = obj_end;

        // Parser dtype, shape, data_offsets dans obj
        let dtype = extract_str_field(obj, "dtype")?;
        if dtype != "F32" {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                       format!("dtype non supporté : {dtype}")));
        }
        let shape = extract_array_field(obj, "shape")?;
        let offsets = extract_array_field(obj, "data_offsets")?;

        if shape.len() != 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                       format!("seuls les tenseurs 2D sont supportés, got shape len {}", shape.len())));
        }
        if offsets.len() != 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "data_offsets doit être [s, e]"));
        }

        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let (start, end) = (offsets[0] as usize, offsets[1] as usize);
        if end > data.len() || start > end {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "offsets hors bornes"));
        }
        let n = (end - start) / 4;
        if n != rows * cols {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                       format!("taille data inattendue : {n} vs {}", rows * cols)));
        }

        let mut floats = Vec::with_capacity(n);
        for k in 0..n {
            let off = start + k * 4;
            let f = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            floats.push(f);
        }

        out.insert(key.to_string(), Tensor::from_vec(floats, rows, cols));
    }

    Ok(out)
}

fn find_unescaped_quote(b: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' { i += 2; continue; }
        if b[i] == b'"' { return Some(i); }
        i += 1;
    }
    None
}

fn skip_balanced(bytes: &[u8], start: usize, open: u8, close: u8) -> usize {
    let mut depth = 0i32;
    let mut i = start;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            // sauter une string entière
            let p = find_unescaped_quote(&bytes[i + 1..]).unwrap_or(bytes.len() - i - 1);
            i = i + 1 + p + 1;
            continue;
        }
        if bytes[i] == open { depth += 1; }
        else if bytes[i] == close {
            depth -= 1;
            if depth == 0 { return i + 1; }
        }
        i += 1;
    }
    i
}

fn extract_str_field(obj: &str, name: &str) -> io::Result<String> {
    let pat = format!(r#""{}":""#, name);
    let start = obj.find(&pat)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("champ {name} absent")))?
        + pat.len();
    let end = obj[start..].find('"')
        .map(|p| start + p)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "string non terminée"))?;
    Ok(obj[start..end].to_string())
}

fn extract_array_field(obj: &str, name: &str) -> io::Result<Vec<i64>> {
    let pat = format!(r#""{}":["#, name);
    let start = obj.find(&pat)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("champ {name} absent")))?
        + pat.len();
    let end = obj[start..].find(']')
        .map(|p| start + p)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "array non terminé"))?;
    let inner = &obj[start..end];
    let nums: Result<Vec<i64>, _> = inner.split(',')
        .map(|s| s.trim().parse::<i64>())
        .collect();
    nums.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_single_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let bytes = serialize(&[("weight".into(), t.clone())]);
        let loaded = deserialize(&bytes).unwrap();
        let recovered = loaded.get("weight").unwrap();
        assert_eq!(recovered.shape(), (2, 3));
        assert_eq!(recovered.data, t.data);
    }

    #[test]
    fn round_trip_multi_tensor() {
        let tensors = vec![
            ("fc1.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2)),
            ("fc1.bias".to_string(),   Tensor::from_vec(vec![0.1, 0.2], 1, 2)),
            ("fc2.weight".to_string(), Tensor::from_vec(vec![5.0; 6], 2, 3)),
        ];
        let bytes = serialize(&tensors);
        let loaded = deserialize(&bytes).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded["fc1.weight"].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded["fc1.bias"].shape(), (1, 2));
        assert_eq!(loaded["fc2.weight"].data.len(), 6);
    }

    #[test]
    fn file_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_scirust_safetensors.safetensors");
        let tensors = vec![
            ("test".to_string(), Tensor::from_vec(vec![3.14, 2.71, 1.41, 1.73], 2, 2)),
        ];
        save_safetensors(&tensors, &path).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let t = &loaded["test"];
        assert!((t.data[0] - 3.14).abs() < 1e-6);
        assert!((t.data[3] - 1.73).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }
}
