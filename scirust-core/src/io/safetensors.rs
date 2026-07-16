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
// Cette implémentation supporte F32 uniquement. L'API historique
// (`serialize`/`deserialize`, `save_state_dict`/`load_state_dict`) est typée
// sur le `Tensor` 2D ; l'API `*_state_dict_nd` (plus bas) sérialise des
// `TensorND` de rang quelconque (le champ JSON `shape` porte naturellement la
// shape complète). Compatible avec PyTorch/Hugging Face pour du row-major.
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
use crate::tensor::tensor_nd::TensorND;
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

        // Reject negative dims/offsets from an untrusted header BEFORE casting to
        // usize: `-1i64 as usize` becomes usize::MAX and then overflows
        // `rows * cols` (a panic / DoS on a crafted file).
        if shape.iter().chain(offsets.iter()).any(|&v| v < 0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "valeur négative dans shape / data_offsets",
            ));
        }
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let (start, end) = (offsets[0] as usize, offsets[1] as usize);
        let numel = rows.checked_mul(cols).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "débordement de rows * cols")
        })?;
        if end > data.len() || start > end
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "offsets hors bornes",
            ));
        }
        if (end - start) % 4 != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "span data_offsets non multiple de 4 octets (f32)",
            ));
        }
        let n = (end - start) / 4;
        if n != numel
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("taille data inattendue : {n} vs {numel}"),
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
    if inner.trim().is_empty()
    {
        // "[]" — a rank-0 (scalar) shape is legal for the N-D API.
        return Ok(Vec::new());
    }
    let nums: Result<Vec<i64>, _> = inner.split(',').map(|s| s.trim().parse::<i64>()).collect();
    nums.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Split a safetensors buffer into (JSON header, data payload), validating the
/// `u64` header size (bounds, `MAX_HEADER_SIZE`, UTF-8).
fn split_header(bytes: &[u8]) -> io::Result<(&str, &[u8])> {
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
    Ok((header, &bytes[8 + header_size..]))
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

// ------------------------------------------------------------------ //
//  save_state_dict_nd / load_state_dict_nd  (TensorND, rang N)       //
// ------------------------------------------------------------------ //

/// Serialize an N-D state dict (`name → TensorND`) plus metadata into
/// safetensors bytes. The full N-D shape goes into the JSON `shape` field
/// (rank 0 → `[]`); data is written contiguously row-major, keys sorted, so
/// the output is deterministic. Round-trips bit-for-bit through
/// [`deserialize_state_dict_nd`].
pub fn serialize_state_dict_nd(
    state: &HashMap<String, TensorND>,
    metadata: &HashMap<String, String>,
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
        let t = state[name].to_contiguous();
        let n_bytes = t.data.len() * 4;
        let shape_json = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let entry = format!(
            r#""{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
            escape_json(name),
            shape_json,
            offset,
            offset + n_bytes,
        );
        entries.push(entry);
        offset += n_bytes;
        for &x in t.data.iter()
        {
            raw_data.extend_from_slice(&x.to_le_bytes());
        }
    }

    let header = if entries.is_empty()
    {
        format!("{{{meta_json}}}")
    }
    else
    {
        format!("{{{},{}}}", meta_json, entries.join(","))
    };
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + raw_data.len());
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&raw_data);
    out
}

/// Parse a safetensors JSON header into N-D tensors: like [`parse_header`] but
/// accepting any rank (including 0 — a scalar), with the same untrusted-input
/// validation (F32 only, no negative dims/offsets, checked `numel` overflow,
/// offsets in bounds and consistent with the shape).
fn parse_header_nd(header: &str, data: &[u8]) -> io::Result<HashMap<String, TensorND>> {
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

        if offsets.len() != 2
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "data_offsets doit être [s, e]",
            ));
        }
        // Reject negative dims/offsets from an untrusted header BEFORE casting
        // to usize (same DoS guard as the 2-D parser).
        if shape.iter().chain(offsets.iter()).any(|&v| v < 0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "valeur négative dans shape / data_offsets",
            ));
        }
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let numel = dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "débordement du produit de shape",
                )
            })?;
        let (start, end) = (offsets[0] as usize, offsets[1] as usize);
        if end > data.len() || start > end
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "offsets hors bornes",
            ));
        }
        if (end - start) % 4 != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "span data_offsets non multiple de 4 octets (f32)",
            ));
        }
        let n = (end - start) / 4;
        if n != numel
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("taille data inattendue : {n} vs {numel}"),
            ));
        }

        let mut floats = Vec::with_capacity(n);
        for k in 0..n
        {
            let off = start + k * 4;
            let float_bytes: [u8; 4] = data[off..off + 4]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "offset data invalide"))?;
            floats.push(f32::from_le_bytes(float_bytes));
        }

        out.insert(key.to_string(), TensorND::new(floats, dims));
    }

    Ok(out)
}

/// Deserialize safetensors bytes produced by [`serialize_state_dict_nd`] (any
/// rank) into `(state dict, metadata)`.
pub fn deserialize_state_dict_nd(
    bytes: &[u8],
) -> io::Result<(HashMap<String, TensorND>, HashMap<String, String>)> {
    let (header, data) = split_header(bytes)?;
    let metadata = extract_metadata(header);
    let tensors = parse_header_nd(header, data)?;
    Ok((tensors, metadata))
}

/// Save an N-D state dict (plus optional metadata) to a safetensors file —
/// the N-D counterpart of [`save_state_dict`]. Pairs with
/// `NdDecoderLM::state_dict`.
pub fn save_state_dict_nd<P: AsRef<Path>>(
    path: P,
    state: &HashMap<String, TensorND>,
    metadata: Option<HashMap<String, String>>,
) -> io::Result<()> {
    let meta = metadata.unwrap_or_default();
    let bytes = serialize_state_dict_nd(state, &meta);
    let mut f = File::create(path.as_ref())?;
    f.write_all(&bytes)?;
    Ok(())
}

/// Load an N-D state dict (plus metadata) from a safetensors file — the N-D
/// counterpart of [`load_state_dict`]. Feed the result to
/// `NdDecoderLM::load_state_dict`.
pub fn load_state_dict_nd<P: AsRef<Path>>(
    path: P,
) -> io::Result<(HashMap<String, TensorND>, HashMap<String, String>)> {
    let mut f = File::open(path.as_ref())?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    deserialize_state_dict_nd(&buf)
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

    // A crafted header with a negative dimension must be rejected with an error,
    // not cast to usize::MAX and panic on the rows*cols multiply (a DoS on an
    // untrusted safetensors file).
    #[test]
    fn deserialize_rejects_negative_shape_without_panicking() {
        let header = r#"{"t":{"dtype":"F32","shape":[-1,4],"data_offsets":[0,16]}}"#;
        let hdr = header.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(hdr);
        buf.extend_from_slice(&[0u8; 16]); // 4 f32 of payload
        let r = deserialize(&buf);
        assert!(r.is_err(), "negative shape must be an error");
    }

    // A valid round trip still works after the added validation.
    #[test]
    fn deserialize_valid_after_validation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let bytes = serialize(&[("w".to_string(), t.clone())]);
        let map = deserialize(&bytes).expect("valid buffer must deserialize");
        assert_eq!(map["w"].data, t.data);
        assert_eq!(map["w"].shape(), (2, 2));
    }

    // ---------------------------------------------------------------- //
    //  N-D state dicts (TensorND)                                      //
    // ---------------------------------------------------------------- //

    fn bits_equal(a: &TensorND, b: &TensorND) -> bool {
        a.shape == b.shape
            && a.data
                .iter()
                .zip(b.data.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Rank-3 and rank-4 tensors round-trip through the N-D serializer with
    /// their full shapes and bit-identical data (awkward values included:
    /// -0.0, subnormals).
    #[test]
    fn nd_state_dict_round_trip_rank3_rank4_bitexact() {
        let mut r3: Vec<f32> = (0..24).map(|i| (i as f32 - 11.5) * 0.37).collect();
        r3[0] = -0.0;
        r3[1] = 1.0e-38; // subnormal
        let r4: Vec<f32> = (0..120).map(|i| 1.0 / (i as f32 + 0.5)).collect();

        let mut state = HashMap::new();
        state.insert(
            "conv.kernel".to_string(),
            TensorND::new(r3.clone(), vec![2, 3, 4]),
        );
        state.insert(
            "attn.qkv".to_string(),
            TensorND::new(r4.clone(), vec![2, 3, 4, 5]),
        );

        let bytes = serialize_state_dict_nd(&state, &HashMap::new());
        let (loaded, meta) = deserialize_state_dict_nd(&bytes).unwrap();
        assert!(meta.is_empty());
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["conv.kernel"].shape(), &[2, 3, 4]);
        assert_eq!(loaded["attn.qkv"].shape(), &[2, 3, 4, 5]);
        assert!(bits_equal(&loaded["conv.kernel"], &state["conv.kernel"]));
        assert!(bits_equal(&loaded["attn.qkv"], &state["attn.qkv"]));

        // Deterministic output (sorted keys).
        let bytes2 = serialize_state_dict_nd(&state, &HashMap::new());
        assert_eq!(bytes, bytes2);
    }

    /// N-D file round trip via save_state_dict_nd / load_state_dict_nd,
    /// with metadata and mixed ranks (1, 2, 3) plus a rank-0 scalar.
    #[test]
    fn nd_state_dict_file_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_scirust_state_dict_nd.safetensors");

        let mut state = HashMap::new();
        state.insert("gamma".to_string(), TensorND::ones(&[7]));
        state.insert(
            "w".to_string(),
            TensorND::new((0..6).map(|i| i as f32).collect(), vec![2, 3]),
        );
        state.insert(
            "cube".to_string(),
            TensorND::new((0..8).map(|i| i as f32 * 0.5).collect(), vec![2, 2, 2]),
        );
        state.insert("scalar".to_string(), TensorND::new(vec![42.5], vec![]));
        let mut meta = HashMap::new();
        meta.insert("arch".to_string(), "nd_decoder".to_string());

        save_state_dict_nd(&path, &state, Some(meta)).unwrap();
        let (loaded, loaded_meta) = load_state_dict_nd(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded_meta.get("arch").unwrap(), "nd_decoder");
        assert_eq!(loaded.len(), 4);
        for (k, v) in &state
        {
            assert!(bits_equal(&loaded[k], v), "tensor {k} not bit-identical");
        }
        assert_eq!(loaded["scalar"].ndim(), 0);
        assert_eq!(loaded["scalar"].numel(), 1);
    }

    // The N-D parser keeps the untrusted-input guards of the 2-D one: a
    // negative dimension is an error, not a usize::MAX cast + overflow panic.
    #[test]
    fn nd_deserialize_rejects_negative_shape_without_panicking() {
        let header = r#"{"t":{"dtype":"F32","shape":[-1,4,2],"data_offsets":[0,32]}}"#;
        let hdr = header.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(hdr);
        buf.extend_from_slice(&[0u8; 32]);
        assert!(deserialize_state_dict_nd(&buf).is_err());
    }

    /// End-to-end: an `NdDecoderLM` saved to a safetensors file and loaded
    /// into a fresh same-config model restores **every** parameter
    /// bit-for-bit — the "save a trained ND transformer" capability.
    #[test]
    fn nd_decoder_lm_safetensors_save_load_params_identical() {
        use crate::nn::nd_decoder::{NdDecoderConfig, NdDecoderLM};

        let cfg = NdDecoderConfig {
            vocab: 6,
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_layers: 2,
            max_seq: 8,
        };
        let src = NdDecoderLM::new(cfg, &mut PcgEngine::new(21));
        let sd = src.state_dict();

        let dir = std::env::temp_dir();
        let path = dir.join("test_scirust_nd_decoder.safetensors");
        save_state_dict_nd(&path, &sd, None).unwrap();
        let (loaded_sd, _) = load_state_dict_nd(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let mut dst = NdDecoderLM::new(cfg, &mut PcgEngine::new(84));
        dst.load_state_dict(&loaded_sd).unwrap();

        let sd2 = dst.state_dict();
        assert_eq!(sd2.len(), sd.len());
        for (k, v) in &sd
        {
            assert!(
                bits_equal(&sd2[k], v),
                "param {k} not bit-identical after file round trip"
            );
        }
    }
}
