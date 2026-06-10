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
//
// ## Limitations du parser JSON
//
// Ce module utilise un parser JSON ad-hoc maison pour éviter de tirer
// serde_json comme dépendance. Le parser est conçu pour gérer les fichiers
// produits par `serialize_state_dict` de cette même bibliothèque.
//
// Il ne garantit PAS le parsing correct de :
// - Fichiers safetensors HuggingFace arbitraires (metadata profondément
//   imbriquées ou inhabituelles)
// - Types non-F32 (BF16, F16, I8, etc.)
// - Tenseurs de rang ≠ 2
// - Headers > 16 MiB
//
// Pour une interopérabilité safetensors complète, envisager un feature
// flag `serde-json` utilisant un vrai parser JSON.
//
// Pour l'instant, ce module convient pour : sauvegarder des modèles
// SciRust, les recharger, les partager entre utilisateurs SciRust.
// Il ne convient PAS pour du chargement cross-framework (PyTorch,
// Candle, etc.) sans validation supplémentaire.

const MAX_HEADER_SIZE: usize = 16 * 1024 * 1024;

use crate::autodiff::reverse::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

// ================================================================== //
//  Sauvegarde                                                         //
// ================================================================== //

pub fn save_safetensors<P: AsRef<Path>>(tensors: &[(String, Tensor)], path: P) -> io::Result<()> {
    let bytes = serialize(tensors);
    let mut f = File::create(path)?;
    f.write_all(&bytes)?;
    Ok(())
}

pub fn serialize(tensors: &[(String, Tensor)]) -> Vec<u8> {
    let mut offset = 0usize;
    let mut entries: Vec<String> = Vec::with_capacity(tensors.len());

    for (name, t) in tensors
    {
        let n_bytes = t.data.len() * 4;
        let entry = format!(
            r#""{}":{{"dtype":"F32","shape":[{},{}],"data_offsets":[{},{}]}}"#,
            escape_json(name),
            t.rows,
            t.cols,
            offset,
            offset + n_bytes,
        );
        entries.push(entry);
        offset += n_bytes;
    }

    let header = format!("{{{}}}", entries.join(","));
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    for (_, t) in tensors
    {
        for &x in &t.data
        {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn unescape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next()
    {
        if c == '\\'
        {
            match chars.next()
            {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some(c) =>
                {
                    out.push('\\');
                    out.push(c);
                },
                None => out.push('\\'),
            }
        }
        else
        {
            out.push(c);
        }
    }
    out
}

// ================================================================== //
//  Chargement                                                         //
// ================================================================== //

pub fn load_safetensors<P: AsRef<Path>>(path: P) -> io::Result<HashMap<String, Tensor>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    deserialize(&buf)
}

pub fn deserialize(bytes: &[u8]) -> io::Result<HashMap<String, Tensor>> {
    if bytes.len() < 8
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fichier trop court",
        ));
    }
    let header_size_bytes: [u8; 8] = bytes[0..8].try_into().expect("header size slice");
    let header_size_u64 = u64::from_le_bytes(header_size_bytes);
    let header_size = usize::try_from(header_size_u64).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "header_size overflow sur cette plateforme",
        )
    })?;
    if header_size > MAX_HEADER_SIZE
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "header trop grand : {} bytes (max {})",
                header_size, MAX_HEADER_SIZE
            ),
        ));
    }
    if 8 + header_size > bytes.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "header_size invalide",
        ));
    }
    let header = std::str::from_utf8(&bytes[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let data = &bytes[8 + header_size..];

    parse_header(header, data)
}

fn parse_header(header: &str, data: &[u8]) -> io::Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    let bytes = header.as_bytes();
    let mut i = 0;
    while i < bytes.len()
    {
        if bytes[i] != b'"'
        {
            i += 1;
            continue;
        }
        let key_start = i + 1;
        let key_end = find_unescaped_quote(&bytes[key_start..])
            .map(|p| key_start + p)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "string non terminée"))?;
        let key = &header[key_start..key_end];
        i = key_end + 1;

        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':')
        {
            i += 1;
        }

        if key == "__metadata__"
        {
            i = skip_balanced(bytes, i, b'{', b'}');
            continue;
        }

        if i >= bytes.len() || bytes[i] != b'{'
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("attendu '{{' après {key}"),
            ));
        }
        let obj_end = skip_balanced(bytes, i, b'{', b'}');
        let obj = &header[i..obj_end];
        i = obj_end;

        let dtype = extract_str_field(obj, "dtype")?;
        if dtype != "F32"
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dtype non supporté : {dtype}"),
            ));
        }
        let shape = extract_array_field(obj, "shape")?;
        let offsets = extract_array_field(obj, "data_offsets")?;

        if shape.len() != 2
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "seuls les tenseurs 2D sont supportés, got shape len {}",
                    shape.len()
                ),
            ));
        }
        if offsets.len() != 2
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "data_offsets doit être [s, e]",
            ));
        }

        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let (start, end) = (offsets[0] as usize, offsets[1] as usize);
        if end > data.len() || start > end
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "offsets hors bornes",
            ));
        }
        let n = (end - start) / 4;
        if n != rows * cols
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("taille data inattendue : {n} vs {}", rows * cols),
            ));
        }

        let mut floats = Vec::with_capacity(n);
        for k in 0..n
        {
            let off = start + k * 4;
            let float_bytes: [u8; 4] = data[off..off + 4]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "offset data invalide"))?;
            let f = f32::from_le_bytes(float_bytes);
            floats.push(f);
        }

        out.insert(key.to_string(), Tensor::from_vec(floats, rows, cols));
    }

    Ok(out)
}

fn find_unescaped_quote(b: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < b.len()
    {
        if b[i] == b'\\'
        {
            i += 2;
            continue;
        }
        if b[i] == b'"'
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn skip_balanced(bytes: &[u8], start: usize, open: u8, close: u8) -> usize {
    let mut depth = 0i32;
    let mut i = start;
    while i < bytes.len()
    {
        if bytes[i] == b'"'
        {
            let p = find_unescaped_quote(&bytes[i + 1..]).unwrap_or(bytes.len() - i - 1);
            i = i + 1 + p + 1;
            continue;
        }
        if bytes[i] == open
        {
            depth += 1;
        }
        else if bytes[i] == close
        {
            depth -= 1;
            if depth == 0
            {
                return i + 1;
            }
        }
        i += 1;
    }
    i
}

fn extract_str_field(obj: &str, name: &str) -> io::Result<String> {
    let pat = format!(r#""{}":""#, name);
    let start = obj.find(&pat).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("champ {name} absent"))
    })? + pat.len();
    let end = obj[start..]
        .find('"')
        .map(|p| start + p)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "string non terminée"))?;
    Ok(obj[start..end].to_string())
}

fn extract_array_field(obj: &str, name: &str) -> io::Result<Vec<i64>> {
    let pat = format!(r#""{}":["#, name);
    let start = obj.find(&pat).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("champ {name} absent"))
    })? + pat.len();
    let end = obj[start..]
        .find(']')
        .map(|p| start + p)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "array non terminé"))?;
    let inner = &obj[start..end];
    let nums: Result<Vec<i64>, _> = inner.split(',').map(|s| s.trim().parse::<i64>()).collect();
    nums.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ================================================================== //
//  Metadata-aware serialization                                      //
// ================================================================== //

pub fn serialize_with_metadata(
    tensors: &[(String, Tensor)],
    metadata: &std::collections::HashMap<String, String>,
) -> Vec<u8> {
    let meta_entries: Vec<String> = metadata
        .iter()
        .map(|(k, v)| format!(r#""{}":"{}""#, escape_json(k), escape_json(v)))
        .collect();
    let meta_json = format!(r#""__metadata__":{{{}}}"#, meta_entries.join(","));

    let mut offset = 0usize;
    let mut entries: Vec<String> = Vec::with_capacity(tensors.len());

    for (name, t) in tensors
    {
        let n_bytes = t.data.len() * 4;
        let entry = format!(
            r#""{}":{{"dtype":"F32","shape":[{},{}],"data_offsets":[{},{}]}}"#,
            escape_json(name),
            t.rows,
            t.cols,
            offset,
            offset + n_bytes,
        );
        entries.push(entry);
        offset += n_bytes;
    }

    let header = format!("{{{},{}}}", meta_json, entries.join(","));
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    for (_, t) in tensors
    {
        for &x in &t.data
        {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

fn parse_header_with_metadata(
    header: &str,
    data: &[u8],
) -> io::Result<(
    std::collections::HashMap<String, Tensor>,
    std::collections::HashMap<String, String>,
)> {
    let tensors = parse_header(header, data)?;
    let metadata = extract_metadata(header);
    Ok((tensors, metadata))
}

pub fn deserialize_with_metadata(
    bytes: &[u8],
) -> io::Result<(
    std::collections::HashMap<String, Tensor>,
    std::collections::HashMap<String, String>,
)> {
    if bytes.len() < 8
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fichier trop court",
        ));
    }
    let header_size_bytes: [u8; 8] = bytes[0..8].try_into().expect("header size slice");
    let header_size_u64 = u64::from_le_bytes(header_size_bytes);
    let header_size = usize::try_from(header_size_u64).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "header_size overflow sur cette plateforme",
        )
    })?;
    if header_size > MAX_HEADER_SIZE
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "header trop grand : {} bytes (max {})",
                header_size, MAX_HEADER_SIZE
            ),
        ));
    }
    if 8 + header_size > bytes.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "header_size invalide",
        ));
    }
    let header = std::str::from_utf8(&bytes[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let data = &bytes[8 + header_size..];

    parse_header_with_metadata(header, data)
}

// ------------------------------------------------------------------ //
//  save_state_dict / load_state_dict  (Tensor state dict API)        //
// ------------------------------------------------------------------ //

pub fn serialize_state_dict(
    state: &std::collections::HashMap<String, Tensor>,
    metadata: &std::collections::HashMap<String, String>,
) -> Vec<u8> {
    let meta_entries: Vec<String> = metadata
        .iter()
        .map(|(k, v)| format!(r#""{}":"{}""#, escape_json(k), escape_json(v)))
        .collect();
    let meta_json = format!(r#""__metadata__":{{{}}}"#, meta_entries.join(","));

    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();

    let mut offset = 0usize;
    let mut entries: Vec<String> = Vec::with_capacity(state.len());
    let mut raw_data: Vec<u8> = Vec::new();

    for name in keys
    {
        let t = &state[name];
        let n_bytes = t.data.len() * 4;
        let entry = format!(
            r#""{}":{{"dtype":"F32","shape":[{},{}],"data_offsets":[{},{}]}}"#,
            escape_json(name),
            t.rows,
            t.cols,
            offset,
            offset + n_bytes,
        );
        entries.push(entry);
        offset += n_bytes;
        for &x in &t.data
        {
            raw_data.extend_from_slice(&x.to_le_bytes());
        }
    }

    let header = format!("{{{},{}}}", meta_json, entries.join(","));
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + raw_data.len());
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&raw_data);
    out
}

pub fn deserialize_state_dict(
    bytes: &[u8],
) -> io::Result<(
    std::collections::HashMap<String, Tensor>,
    std::collections::HashMap<String, String>,
)> {
    if bytes.len() < 8
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fichier trop court",
        ));
    }
    let header_size_bytes: [u8; 8] = bytes[0..8].try_into().expect("header size slice");
    let header_size_u64 = u64::from_le_bytes(header_size_bytes);
    let header_size = usize::try_from(header_size_u64).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "header_size overflow sur cette plateforme",
        )
    })?;
    if header_size > MAX_HEADER_SIZE
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "header trop grand : {} bytes (max {})",
                header_size, MAX_HEADER_SIZE
            ),
        ));
    }
    if 8 + header_size > bytes.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "header_size invalide",
        ));
    }
    let header = std::str::from_utf8(&bytes[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let data_buf = &bytes[8 + header_size..];

    let metadata = extract_metadata(header);
    let tensors = parse_header(header, data_buf)?;
    Ok((tensors, metadata))
}

pub fn save_state_dict<P: AsRef<Path>>(
    path: P,
    state: &std::collections::HashMap<String, Tensor>,
    metadata: Option<std::collections::HashMap<String, String>>,
) -> io::Result<()> {
    let meta = metadata.unwrap_or_default();
    let bytes = serialize_state_dict(state, &meta);
    let mut f = File::create(path.as_ref())?;
    f.write_all(&bytes)?;
    Ok(())
}

pub fn load_state_dict<P: AsRef<Path>>(
    path: P,
) -> io::Result<(
    std::collections::HashMap<String, Tensor>,
    std::collections::HashMap<String, String>,
)> {
    let mut f = File::open(path.as_ref())?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    deserialize_state_dict(&buf)
}

/// Extract the `__metadata__` section from a safetensors JSON header.
fn extract_metadata(header: &str) -> std::collections::HashMap<String, String> {
    let mut meta = std::collections::HashMap::new();
    let needle = r#""__metadata__":"#;
    if let Some(start) = header.find(needle)
    {
        let brace_start = start + needle.len();
        let bytes = header.as_bytes();
        let obj_end = skip_balanced(bytes, brace_start, b'{', b'}');
        let obj = &header[brace_start..obj_end];

        let b = obj.as_bytes();
        let mut j = 0;
        while j < b.len()
        {
            if b[j] != b'"'
            {
                j += 1;
                continue;
            }
            let ks = j + 1;
            let ke = match find_unescaped_quote(&b[ks..]).map(|p| ks + p)
            {
                Some(p) => p,
                None => break,
            };
            let k = &obj[ks..ke];
            j = ke + 1;

            while j < b.len() && (b[j] == b' ' || b[j] == b':')
            {
                j += 1;
            }

            if j >= b.len() || b[j] != b'"'
            {
                break;
            }
            let vs = j + 1;
            let ve = match find_unescaped_quote(&b[vs..]).map(|p| vs + p)
            {
                Some(p) => p,
                None => break,
            };
            let v = &obj[vs..ve];
            j = ve + 1;

            meta.insert(unescape_json(k), unescape_json(v));
        }
    }
    meta
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tape;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::rng::PcgEngine;
    use crate::nn::{Linear, Module, ReLU, Sequential};

    #[test]
    fn test_safetensors_header_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let bytes = serialize(&[("weight".into(), t)]);
        let loaded = deserialize(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded["weight"].shape(), (2, 2));
    }

    #[test]
    fn test_save_load_single_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let mut state = std::collections::HashMap::new();
        state.insert("weight".to_string(), t.clone());
        let bytes = serialize_state_dict(&state, &std::collections::HashMap::new());
        let (loaded, _) = deserialize_state_dict(&bytes).unwrap();
        let recovered = loaded.get("weight").unwrap();
        assert_eq!(recovered.shape(), (2, 3));
        assert_eq!(recovered.data, t.data);
    }

    #[test]
    fn test_save_load_state_dict_with_metadata() {
        let mut state = std::collections::HashMap::new();
        state.insert("w".to_string(), Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let mut meta = std::collections::HashMap::new();
        meta.insert("epoch".to_string(), "5".to_string());
        meta.insert("test_accuracy".to_string(), "0.977".to_string());

        let bytes = serialize_state_dict(&state, &meta);
        let (loaded, loaded_meta) = deserialize_state_dict(&bytes).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded_meta.get("epoch").unwrap(), "5");
        assert_eq!(loaded_meta.get("test_accuracy").unwrap(), "0.977");
    }

    #[test]
    fn test_load_corrupted_file_returns_error() {
        // header_size claims 1_000_000 bytes but file is tiny
        let mut bad = vec![0u8; 16];
        bad[0] = 0x40;
        bad[1] = 0x42;
        bad[2] = 0x0F;
        bad[3] = 0x00; // 1_000_000 in little-endian
        let res = deserialize_state_dict(&bad);
        assert!(res.is_err(), "corrupted file should return Err");
    }

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
            (
                "fc1.weight".to_string(),
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2),
            ),
            (
                "fc1.bias".to_string(),
                Tensor::from_vec(vec![0.1, 0.2], 1, 2),
            ),
            (
                "fc2.weight".to_string(),
                Tensor::from_vec(vec![5.0; 6], 2, 3),
            ),
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
        let tensors = vec![(
            "test".to_string(),
            Tensor::from_vec(vec![3.13, 2.71, 1.41, 1.73], 2, 2),
        )];
        save_safetensors(&tensors, &path).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let t = &loaded["test"];
        assert!((t.data[0] - 3.13).abs() < 1e-6);
        assert!((t.data[3] - 1.73).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trip_with_metadata() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut meta = std::collections::HashMap::new();
        meta.insert("model_name".to_string(), "test_model".to_string());
        meta.insert("format".to_string(), "safetensors".to_string());

        let bytes = serialize_with_metadata(&[("weight".into(), t.clone())], &meta);
        let (loaded, loaded_meta) = deserialize_with_metadata(&bytes).unwrap();

        assert_eq!(loaded.len(), 1);
        let recovered = loaded.get("weight").unwrap();
        assert_eq!(recovered.shape(), (2, 2));
        assert_eq!(recovered.data, vec![1.0, 2.0, 3.0, 4.0]);

        assert_eq!(loaded_meta.get("model_name").unwrap(), "test_model");
        assert_eq!(loaded_meta.get("format").unwrap(), "safetensors");
    }

    #[test]
    fn round_trip_with_metadata_escaped_values() {
        let t = Tensor::from_vec(vec![0.5], 1, 1);
        let mut meta = std::collections::HashMap::new();
        meta.insert(
            "description".to_string(),
            r#"quote "test" value"#.to_string(),
        );

        let bytes = serialize_with_metadata(&[("x".into(), t.clone())], &meta);
        let (loaded, loaded_meta) = deserialize_with_metadata(&bytes).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded_meta.get("description").unwrap(),
            r#"quote "test" value"#
        );
    }

    #[test]
    fn deserialize_without_metadata_returns_empty_map() {
        let t = Tensor::from_vec(vec![1.0, 2.0], 1, 2);
        let bytes = serialize(&[("x".into(), t)]);
        let (_tensors, meta) = deserialize_with_metadata(&bytes).unwrap();
        assert!(meta.is_empty());
    }

    fn make_test_state_dict() -> std::collections::HashMap<String, Tensor> {
        let mut state = std::collections::HashMap::new();
        state.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
        );
        state.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.1, 0.2, 0.3], 1, 3),
        );
        state
    }

    #[test]
    fn state_dict_round_trip() {
        let state = make_test_state_dict();
        let mut meta = std::collections::HashMap::new();
        meta.insert("arch".to_string(), "mlp".to_string());

        let bytes = serialize_state_dict(&state, &meta);
        let (loaded, loaded_meta) = deserialize_state_dict(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded_meta.get("arch").unwrap(), "mlp");

        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), (2, 3));
        assert!((w.data[0] - 1.0).abs() < 1e-6);
        assert!((w.data[5] - 6.0).abs() < 1e-6);

        let b = loaded.get("bias").unwrap();
        assert_eq!(b.shape(), (1, 3));
        assert!((b.data[0] - 0.1).abs() < 1e-6);
        assert!((b.data[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn state_dict_file_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_scirust_state_dict.safetensors");
        let state = make_test_state_dict();
        let mut meta = std::collections::HashMap::new();
        meta.insert("test".to_string(), "file_round_trip".to_string());

        save_state_dict(&path, &state, Some(meta)).unwrap();
        let (loaded, loaded_meta) = load_state_dict(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded_meta.get("test").unwrap(), "file_round_trip");

        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), (2, 3));
        assert!((w.data[0] - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn serialize_is_deterministic() {
        let mut state = std::collections::HashMap::new();
        state.insert("z.weight".to_string(), Tensor::from_vec(vec![1.0; 4], 2, 2));
        state.insert("a.weight".to_string(), Tensor::from_vec(vec![2.0; 4], 2, 2));
        state.insert("m.bias".to_string(), Tensor::from_vec(vec![3.0; 2], 1, 2));

        let bytes1 = serialize_state_dict(&state, &std::collections::HashMap::new());
        let bytes2 = serialize_state_dict(&state, &std::collections::HashMap::new());
        assert_eq!(
            bytes1, bytes2,
            "Saves consécutifs doivent produire des bytes identiques"
        );
    }

    #[test]
    fn test_mnist_save_then_load_then_inference() {
        let mut rng = PcgEngine::new(42);
        let mut model = Sequential::new()
            .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng))
            .add(ReLU::new())
            .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng));

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![0.5; 784], 1, 784));
        let y1 = model.forward(&tape, x);
        let out1 = tape.value(y1.idx()).data.clone();

        let sd = model.state_dict();
        let dir = std::env::temp_dir();
        let path = dir.join("test_mnist_save_load.safetensors");
        save_state_dict(&path, &sd, None).unwrap();

        let mut rng2 = PcgEngine::new(99);
        let mut model2 = Sequential::new()
            .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng2))
            .add(ReLU::new())
            .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng2));

        let (loaded_sd, _) = load_state_dict(&path).unwrap();
        model2.load_state_dict(&loaded_sd).unwrap();

        let tape2 = Tape::new();
        let x2 = tape2.input(Tensor::from_vec(vec![0.5; 784], 1, 784));
        let y2 = model2.forward(&tape2, x2);
        let out2 = tape2.value(y2.idx()).data;

        assert_eq!(out1.len(), out2.len());
        for i in 0..out1.len()
        {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-5,
                "inference mismatch at {}: {} vs {}",
                i,
                out1[i],
                out2[i]
            );
        }

        let _ = std::fs::remove_file(&path);
    }
}
