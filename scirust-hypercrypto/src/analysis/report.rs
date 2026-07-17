//! Deterministic machine-readable results: a minimal ordered-JSON writer and a
//! SHA-256 fingerprint for every result document (spec §"Machine-readable
//! results"). No `serde` dependency; field order is explicit and stable.

use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

/// A minimal JSON value with deterministic (insertion-order) serialization.
#[derive(Clone, Debug)]
pub enum Json {
    /// JSON null.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// Unsigned integer.
    U64(u64),
    /// Signed integer.
    I64(i64),
    /// String (escaped on output).
    Str(String),
    /// Array.
    Arr(Vec<Json>),
    /// Object with ordered keys.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Convenience constructor for an object.
    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    /// Convenience constructor for a string.
    pub fn s(v: impl Into<String>) -> Json {
        Json::Str(v.into())
    }

    /// Serialize with 2-space indentation and stable key order.
    pub fn to_pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out.push('\n');
        out
    }

    fn write(&self, out: &mut String, indent: usize) {
        match self
        {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::U64(v) => out.push_str(&v.to_string()),
            Json::I64(v) => out.push_str(&v.to_string()),
            Json::Str(s) =>
            {
                out.push('"');
                for ch in s.chars()
                {
                    match ch
                    {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\t' => out.push_str("\\t"),
                        '\r' => out.push_str("\\r"),
                        c => out.push(c),
                    }
                }
                out.push('"');
            },
            Json::Arr(items) =>
            {
                if items.is_empty()
                {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (i, it) in items.iter().enumerate()
                {
                    push_indent(out, indent + 1);
                    it.write(out, indent + 1);
                    if i + 1 < items.len()
                    {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_indent(out, indent);
                out.push(']');
            },
            Json::Obj(pairs) =>
            {
                if pairs.is_empty()
                {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (i, (k, v)) in pairs.iter().enumerate()
                {
                    push_indent(out, indent + 1);
                    out.push('"');
                    out.push_str(k);
                    out.push_str("\": ");
                    v.write(out, indent + 1);
                    if i + 1 < pairs.len()
                    {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_indent(out, indent);
                out.push('}');
            },
        }
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent
    {
        out.push_str("  ");
    }
}

/// A `[u64; 8]` as a JSON array.
pub fn u64x8(v: [u64; 8]) -> Json {
    Json::Arr(v.iter().map(|&x| Json::U64(x)).collect())
}

/// Lowercase-hex SHA-256 of a string (spec §14.4 hex convention).
pub fn sha256_hex(data: &str) -> String {
    let mut h = Sha256::new();
    h.update(data.as_bytes());
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    for b in d
    {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The spec's `GraphId` for the fixed `F-PROG` byte encoding (spec §12.2):
/// `SHA256("SCIRUST-HYPERCRYPTO-V0.1/GRAPH-ID" || bytes)`.
pub const F_PROG_BYTES: [u8; 18] = [
    0x02, 0x01, 0x01, 0x00, 0x02, 0x00, 0x04, 0x06, 0x02, 0x02, 0x06, 0x08, 0x00, 0x09, 0x00, 0x0a,
    0x00, 0x00,
];

/// Compute the `GraphId` of the fixed `F-PROG` (lowercase hex SHA-256).
pub fn f_prog_graph_id() -> String {
    let mut h = Sha256::new();
    h.update(b"SCIRUST-HYPERCRYPTO-V0.1/GRAPH-ID");
    h.update(F_PROG_BYTES);
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    for b in d
    {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Write a JSON document to `dir/name.json` plus a `dir/name.json.sha256`
/// sidecar containing its fingerprint. Returns `(path, fingerprint)`.
pub fn write_result_file(dir: &Path, name: &str, doc: &Json) -> std::io::Result<(PathBuf, String)> {
    std::fs::create_dir_all(dir)?;
    let body = doc.to_pretty();
    let fp = sha256_hex(&body);
    let path = dir.join(format!("{name}.json"));
    let mut f = std::fs::File::create(&path)?;
    f.write_all(body.as_bytes())?;
    let sidecar = dir.join(format!("{name}.json.sha256"));
    let mut sf = std::fs::File::create(&sidecar)?;
    writeln!(sf, "{fp}  {name}.json")?;
    Ok((path, fp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_deterministic() {
        let a = Json::obj(vec![
            ("b", Json::U64(2)),
            ("a", Json::Arr(vec![Json::Bool(true), Json::Null])),
        ]);
        let s1 = a.to_pretty();
        let s2 = a.to_pretty();
        assert_eq!(s1, s2);
        // insertion order is preserved (b before a)
        assert!(s1.find("\"b\"").unwrap() < s1.find("\"a\"").unwrap());
    }

    #[test]
    fn sha256_known_answer() {
        // SHA-256("") known digest
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn graph_id_is_stable() {
        let id = f_prog_graph_id();
        assert_eq!(id.len(), 64);
        // stable across calls
        assert_eq!(id, f_prog_graph_id());
    }
}
