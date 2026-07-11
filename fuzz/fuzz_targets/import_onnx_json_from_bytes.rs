#![no_main]
//! Fuzz target for the ONNX JSON model importer.
//!
//! `import_onnx_json` parses a *fully untrusted* JSON string (a model file
//! loaded from disk or the network) via `serde_json`. It must be total on
//! any input: for every possible string it either returns `Ok(graph)` or
//! `Err(_)` — it must never panic or exhaust the stack (e.g. via unbounded
//! recursion on deeply nested JSON arrays/objects). `data` is converted with
//! `from_utf8_lossy` rather than rejected on invalid UTF-8, so libFuzzer's
//! byte-level mutations keep exercising the JSON parser instead of mostly
//! being discarded for failing a UTF-8 check first.

use libfuzzer_sys::fuzz_target;
use scirust_onnx::import_onnx_json;

fuzz_target!(|data: &[u8]| {
    let json = String::from_utf8_lossy(data);
    let _ = import_onnx_json(&json);
});
