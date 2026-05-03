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

fn unescape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some(c) => { out.push('\\'); out.push(c); }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
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
    let header_size_bytes: [u8; 8] = bytes[0..8].try_into().expect("header size slice");
    let header_size_u64 = u64::from_le_bytes(header_size_bytes);
    let header_size = usize::try_from(header_size_u64)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "header_size overflow sur cette plateforme"))?;
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
            let float_bytes: [u8; 4] = data[off..off + 4].try_into()
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
//  Metadata-aware serialization                                      //
// ================================================================== //

/// Like `serialize`, but includes a `__metadata__` section in the header.
pub fn serialize_with_metadata(
    tensors: &[(String, Tensor)],
    metadata: &std::collections::HashMap<String, String>,
) -> Vec<u8> {
    // 1. Build the metadata JSON object string
    let meta_entries: Vec<String> = metadata.iter().map(|(k, v)| {
        format!(r#""{}":"{}""#, escape_json(k), escape_json(v))
    }).collect();
    let meta_json = format!(r#""__metadata__":{{{}}}"#, meta_entries.join(","));

    // 2. Compute tensor offsets and build tensor entries
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

    // 3. Assemble header: metadata + tensors
    let header = format!("{{{},{}}}", meta_json, entries.join(","));
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    // 4. Assemble final output
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

/// Like `parse_header` but also returns the `__metadata__` dict.
fn parse_header_with_metadata(header: &str, data: &[u8]) -> io::Result<(
    std::collections::HashMap<String, Tensor>,
    std::collections::HashMap<String, String>,
)> {
    let tensors = parse_header(header, data)?;
    let metadata = extract_metadata(header);
    Ok((tensors, metadata))
}

/// Like `deserialize`, but also returns the `__metadata__` dict.
pub fn deserialize_with_metadata(bytes: &[u8]) -> io::Result<(
    std::collections::HashMap<String, Tensor>,
    std::collections::HashMap<String, String>,
)> {
    if bytes.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "fichier trop court"));
    }
    let header_size_bytes: [u8; 8] = bytes[0..8].try_into().expect("header size slice");
    let header_size_u64 = u64::from_le_bytes(header_size_bytes);
    let header_size = usize::try_from(header_size_u64)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "header_size overflow sur cette plateforme"))?;
    if 8 + header_size > bytes.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "header_size invalide"));
    }
    let header = std::str::from_utf8(&bytes[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let data = &bytes[8 + header_size..];

    parse_header_with_metadata(header, data)
}

// ------------------------------------------------------------------ //
//  save_state_dict / load_state_dict  (ndarray f64 state dict API)   //
// ------------------------------------------------------------------ //

/// Convert an `ndarray::ArrayD<f64>` to a flat f32 buffer and shape vector.
fn ndarray_to_f32(arr: &ndarray::ArrayD<f64>) -> (Vec<f32>, Vec<usize>) {
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
    (data, shape)
}

/// Convert flat f32 data and shape back to an `ndarray::ArrayD<f64>`.
fn f32_to_ndarray(data: &[f32], shape: &[usize]) -> ndarray::ArrayD<f64> {
    let arr_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(shape),
        arr_f64,
    ).expect("f32_to_ndarray: invalid shape for data length")
}

/// Serialize a state dict (ndarray f64) + metadata into a safetensors-compatible byte buffer.
pub fn serialize_state_dict(
    state: &std::collections::HashMap<String, ndarray::ArrayD<f64>>,
    metadata: &std::collections::HashMap<String, String>,
) -> Vec<u8> {
    // Build metadata JSON
    let meta_entries: Vec<String> = metadata.iter().map(|(k, v)| {
        format!(r#""{}":"{}""#, escape_json(k), escape_json(v))
    }).collect();
    let meta_json = format!(r#""__metadata__":{{{}}}"#, meta_entries.join(","));

    // Compute offsets and tensor entries
    let mut offset = 0usize;
    let mut entries: Vec<String> = Vec::with_capacity(state.len());

    // We also need to keep the flattened data in order
    let mut raw_data: Vec<u8> = Vec::new();

    for (name, arr) in state {
        let (flat_f32, shape) = ndarray_to_f32(arr);
        let n_bytes = flat_f32.len() * 4;

        // Shape as comma-separated list
        let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let shape_json = shape_str.join(",");

        let entry = format!(
            r#""{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
            escape_json(name),
            shape_json,
            offset, offset + n_bytes,
        );
        entries.push(entry);
        offset += n_bytes;

        for &x in &flat_f32 {
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

/// Deserialize a safetensors buffer into a state dict (ndarray f64) + metadata.
pub fn deserialize_state_dict(bytes: &[u8]) -> io::Result<(
    std::collections::HashMap<String, ndarray::ArrayD<f64>>,
    std::collections::HashMap<String, String>,
)> {
    if bytes.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "fichier trop court"));
    }
    let header_size_bytes: [u8; 8] = bytes[0..8].try_into().expect("header size slice");
    let header_size_u64 = u64::from_le_bytes(header_size_bytes);
    let header_size = usize::try_from(header_size_u64)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "header_size overflow sur cette plateforme"))?;
    if 8 + header_size > bytes.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "header_size invalide"));
    }
    let header = std::str::from_utf8(&bytes[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let data_buf = &bytes[8 + header_size..];

    // Parse metadata
    let metadata = extract_metadata(header);

    // Parse tensor entries (skipping __metadata__)
    let bytes_h = header.as_bytes();
    let mut tensors: std::collections::HashMap<String, ndarray::ArrayD<f64>> = std::collections::HashMap::new();
    let mut i = 0;
    while i < bytes_h.len() {
        if bytes_h[i] != b'"' { i += 1; continue; }
        let key_start = i + 1;
        let key_end = find_unescaped_quote(&bytes_h[key_start..])
            .map(|p| key_start + p)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "string non terminée"))?;
        let key = &header[key_start..key_end];
        i = key_end + 1;

        while i < bytes_h.len() && (bytes_h[i] == b' ' || bytes_h[i] == b':') { i += 1; }

        if key == "__metadata__" {
            i = skip_balanced(bytes_h, i, b'{', b'}');
            continue;
        }

        if i >= bytes_h.len() || bytes_h[i] != b'{' {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                       format!("attendu '{{' après {key}")));
        }
        let obj_end = skip_balanced(bytes_h, i, b'{', b'}');
        let obj = &header[i..obj_end];
        i = obj_end;

        let dtype = extract_str_field(obj, "dtype")?;
        if dtype != "F32" {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                       format!("dtype non supporté : {dtype}")));
        }
        let shape = extract_array_field(obj, "shape")?;
        let offsets = extract_array_field(obj, "data_offsets")?;

        let (start, end) = (offsets[0] as usize, offsets[1] as usize);
        if end > data_buf.len() || start > end {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "offsets hors bornes"));
        }
        let n_floats = (end - start) / 4;
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let expected_len: usize = shape_usize.iter().product();
        if n_floats != expected_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                       format!("taille data inattendue : {n_floats} vs {expected_len}")));
        }

        let mut floats = Vec::with_capacity(n_floats);
        for k in 0..n_floats {
            let off = start + k * 4;
            let float_bytes: [u8; 4] = data_buf[off..off + 4].try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "offset data invalide"))?;
            let f = f32::from_le_bytes(float_bytes);
            floats.push(f);
        }

        let arr = f32_to_ndarray(&floats, &shape_usize);
        tensors.insert(key.to_string(), arr);
    }

    Ok((tensors, metadata))
}

/// Save a state dict + metadata to a safetensors file on disk.
pub fn save_state_dict<P: AsRef<Path>>(
    path: P,
    state: &std::collections::HashMap<String, ndarray::ArrayD<f64>>,
    metadata: &std::collections::HashMap<String, String>,
) -> io::Result<()> {
    let bytes = serialize_state_dict(state, metadata);
    let mut f = File::create(path.as_ref())?;
    f.write_all(&bytes)?;
    Ok(())
}

/// Load a state dict + metadata from a safetensors file on disk.
pub fn load_state_dict<P: AsRef<Path>>(
    path: P,
) -> io::Result<(
    std::collections::HashMap<String, ndarray::ArrayD<f64>>,
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
    // Look for "__metadata__":{ ... }
    let needle = r#""__metadata__":"#;
    if let Some(start) = header.find(needle) {
        let brace_start = start + needle.len();
        let bytes = header.as_bytes();
        let obj_end = skip_balanced(bytes, brace_start, b'{', b'}');
        let obj = &header[brace_start..obj_end];

        // Parse key-value pairs inside the metadata object
        let b = obj.as_bytes();
        let mut j = 0;
        while j < b.len() {
            if b[j] != b'"' { j += 1; continue; }
            let ks = j + 1;
            let ke = match find_unescaped_quote(&b[ks..]).map(|p| ks + p) {
                Some(p) => p,
                None => break,
            };
            let k = &obj[ks..ke];
            j = ke + 1;

            // Skip whitespace and ':'
            while j < b.len() && (b[j] == b' ' || b[j] == b':') { j += 1; }

            // Value must be a string
            if j >= b.len() || b[j] != b'"' { break; }
            let vs = j + 1;
            let ve = match find_unescaped_quote(&b[vs..]).map(|p| vs + p) {
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

    // -- Metadata tests -------------------------------------------------- //

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
        meta.insert("description".to_string(), r#"quote "test" value"#.to_string());

        let bytes = serialize_with_metadata(&[("x".into(), t.clone())], &meta);
        let (loaded, loaded_meta) = deserialize_with_metadata(&bytes).unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded_meta.get("description").unwrap(), r#"quote "test" value"#);
    }

    #[test]
    fn deserialize_without_metadata_returns_empty_map() {
        let t = Tensor::from_vec(vec![1.0, 2.0], 1, 2);
        let bytes = serialize(&[("x".into(), t)]);
        let (_tensors, meta) = deserialize_with_metadata(&bytes).unwrap();
        assert!(meta.is_empty());
    }

    // -- ndarray state_dict tests --------------------------------------- //

    fn make_test_state_dict() -> std::collections::HashMap<String, ndarray::ArrayD<f64>> {
        let mut state = std::collections::HashMap::new();
        // 2D weight: (2, 3)
        let w = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();
        state.insert("weight".to_string(), w);
        // 1D bias: (3,)
        let b = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3]),
            vec![0.1, 0.2, 0.3],
        ).unwrap();
        state.insert("bias".to_string(), b);
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

        // Check weight
        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 3]);
        assert!((w[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((w[[1, 2]] - 6.0).abs() < 1e-6);

        // Check bias
        let b = loaded.get("bias").unwrap();
        assert_eq!(b.shape(), &[3]);
        assert!((b[[0]] - 0.1).abs() < 1e-6);
        assert!((b[[2]] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn state_dict_file_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_scirust_state_dict.safetensors");
        let state = make_test_state_dict();
        let mut meta = std::collections::HashMap::new();
        meta.insert("test".to_string(), "file_round_trip".to_string());

        save_state_dict(&path, &state, &meta).unwrap();
        let (loaded, loaded_meta) = load_state_dict(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded_meta.get("test").unwrap(), "file_round_trip");

        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape(), &[2, 3]);
        assert!((w[[0, 0]] - 1.0).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ndarray_to_f32_and_back() {
        let arr = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 2]),
            vec![1.5, 2.5, 3.5, 4.5],
        ).unwrap();
        let (flat, shape) = ndarray_to_f32(&arr);
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(flat.len(), 4);
        assert!((flat[0] - 1.5).abs() < 1e-6);

        let restored = f32_to_ndarray(&flat, &shape);
        assert_eq!(restored.shape(), &[2, 2]);
        assert!((restored[[0, 0]] - 1.5).abs() < 1e-6);
        assert!((restored[[1, 1]] - 4.5).abs() < 1e-6);
    }
}
