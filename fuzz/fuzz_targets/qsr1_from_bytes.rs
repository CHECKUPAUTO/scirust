#![no_main]
//! Fuzz target for the QSR1 quantized-model deserializer.
//!
//! `QModel::from_bytes` parses a *fully untrusted* byte buffer (a model file
//! loaded from disk or the network). It must be total on `&[u8]`: for every
//! possible input it either returns `Ok(model)` or `Err(io::Error)` — it must
//! never panic (out-of-bounds slice, arithmetic overflow, capacity overflow).
//! This property was hardened in PR #266 (bounds-checked readers,
//! self-consistency validation, `checked_mul`); the fuzzer guards against
//! regressions.

use libfuzzer_sys::fuzz_target;
use scirust_runtime::quant::QModel;

fuzz_target!(|data: &[u8]| {
    // The only contract under test: parsing arbitrary bytes never panics.
    // A well-formed buffer yields Ok; anything else must be a clean Err.
    let _ = QModel::from_bytes(data);
});
