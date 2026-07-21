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

/// Optional bridge to the workspace-wide CANR §9 benchmark schema
/// (`scirust-bench-schema`, feature `bench-schema`). This crate's own
/// [`Json`] stays serde-free by design (see the module docs); this bridge
/// only compiles in when a consumer opts into the feature, and converts
/// *after* a result document exists — it does not touch how `Json` itself
/// is built or serialized.
#[cfg(feature = "bench-schema")]
impl Json {
    /// Flatten every numeric leaf of this document into a
    /// [`scirust_bench_schema::BenchRecord`], one row per leaf, `metric`
    /// being the dot/bracket-joined path from the root (e.g.
    /// `"gf2_rank"`, `"differentials.0.prob_ppm"`). Non-numeric leaves
    /// (`Str`, `Null`) are skipped — they carry no measured quantity to
    /// certify or track over time. `Bool` leaves convert to `0.0`/`1.0`.
    ///
    /// `kernel`/`dataset`/`method`/`seed` are supplied by the caller — this
    /// document has no notion of them itself (it is a bag of measurements,
    /// not a benchmark record).
    #[must_use]
    pub fn to_bench_records(
        &self,
        kernel: impl Into<String>,
        dataset: impl Into<String>,
        method: impl Into<String>,
        seed: u64,
    ) -> Vec<scirust_bench_schema::BenchRecord> {
        let kernel = kernel.into();
        let dataset = dataset.into();
        let method = method.into();
        let mut out = Vec::new();
        self.flatten_into("", &kernel, &dataset, &method, seed, &mut out);
        out
    }

    fn flatten_into(
        &self,
        path: &str,
        kernel: &str,
        dataset: &str,
        method: &str,
        seed: u64,
        out: &mut Vec<scirust_bench_schema::BenchRecord>,
    ) {
        let leaf = |value: f64, out: &mut Vec<scirust_bench_schema::BenchRecord>| {
            out.push(scirust_bench_schema::BenchRecord::new(
                kernel.to_string(),
                dataset.to_string(),
                method.to_string(),
                seed,
                path.to_string(),
                value,
            ));
        };
        match self
        {
            Json::Null | Json::Str(_) =>
            {},
            Json::Bool(b) => leaf(f64::from(u8::from(*b)), out),
            // u64/i64 up to 2^53 round-trip exactly through f64; this
            // crate's measured quantities (ranks, counts, ppm fractions,
            // kernel-size logs) are far below that ceiling.
            Json::U64(v) => leaf(*v as f64, out),
            Json::I64(v) => leaf(*v as f64, out),
            Json::Arr(items) =>
            {
                for (i, item) in items.iter().enumerate()
                {
                    let child_path = if path.is_empty()
                    {
                        i.to_string()
                    }
                    else
                    {
                        format!("{path}.{i}")
                    };
                    item.flatten_into(&child_path, kernel, dataset, method, seed, out);
                }
            },
            Json::Obj(pairs) =>
            {
                for (k, v) in pairs
                {
                    let child_path = if path.is_empty()
                    {
                        k.clone()
                    }
                    else
                    {
                        format!("{path}.{k}")
                    };
                    v.flatten_into(&child_path, kernel, dataset, method, seed, out);
                }
            },
        }
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

    #[cfg(feature = "bench-schema")]
    #[test]
    fn to_bench_records_flattens_numeric_leaves_with_dotted_paths() {
        let doc = Json::obj(vec![
            ("gf2_rank", Json::U64(12)),
            ("invertible", Json::Bool(true)),
            ("note", Json::s("skipped: not numeric")),
            (
                "differentials",
                Json::Arr(vec![
                    Json::obj(vec![("prob_ppm", Json::U64(500))]),
                    Json::obj(vec![("prob_ppm", Json::U64(750))]),
                ]),
            ),
        ]);
        let records = doc.to_bench_records(
            "hypercrypto/phase1",
            "MINI-8/fixed",
            "matrix-lifting",
            0xA11CE,
        );

        // 4 numeric leaves; the Str leaf is skipped.
        assert_eq!(records.len(), 4);
        assert!(records.iter().all(|r| r.seed == 0xA11CE));
        assert!(records.iter().all(|r| r.method == "matrix-lifting"));

        let rank = records.iter().find(|r| r.metric == "gf2_rank").unwrap();
        assert_eq!(rank.value, 12.0);

        let inv = records.iter().find(|r| r.metric == "invertible").unwrap();
        assert_eq!(inv.value, 1.0);

        let d0 = records
            .iter()
            .find(|r| r.metric == "differentials.0.prob_ppm")
            .unwrap();
        assert_eq!(d0.value, 500.0);
        let d1 = records
            .iter()
            .find(|r| r.metric == "differentials.1.prob_ppm")
            .unwrap();
        assert_eq!(d1.value, 750.0);

        assert!(!records.iter().any(|r| r.metric == "note"));

        // Round trips as JSONL like any other BenchRecord set.
        let text = scirust_bench_schema::to_jsonl(&records);
        let back = scirust_bench_schema::parse_jsonl(&text).expect("round trip");
        assert_eq!(back, records);
    }
}
