#![no_main]
//! Fuzz target for the safetensors deserializer.
//!
//! `scirust_core::io::safetensors::deserialize` parses a *fully untrusted* byte
//! buffer (a weights file loaded from disk or the network). It must be total on
//! `&[u8]`: for every possible input it either returns `Ok(map)` or
//! `Err(io::Error)` — it must never panic (out-of-bounds slice, arithmetic
//! overflow, capacity overflow, or unbounded allocation). The parser was
//! hardened with a header-size cap, negative-dimension rejection, `checked_mul`
//! on `rows*cols`, and file-length-bounded capacities; this fuzzer guards
//! against regressions. `deserialize_with_metadata` and `deserialize_state_dict`
//! share the same header path and are exercised alongside it.

use libfuzzer_sys::fuzz_target;
use scirust_core::io::safetensors;

fuzz_target!(|data: &[u8]| {
    // The only contract under test: parsing arbitrary bytes never panics.
    let _ = safetensors::deserialize(data);
    let _ = safetensors::deserialize_with_metadata(data);
    let _ = safetensors::deserialize_state_dict(data);
});
