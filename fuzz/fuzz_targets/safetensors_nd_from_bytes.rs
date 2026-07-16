#![no_main]
//! Fuzz target for the N-D safetensors state-dict deserializer.
//!
//! `scirust_core::io::deserialize_state_dict_nd` parses a *fully untrusted*
//! byte buffer (an N-D weights file loaded from disk or the network). It must
//! be total on `&[u8]`: for every possible input it either returns
//! `Ok((state, metadata))` or `Err(io::Error)` — it must never panic
//! (out-of-bounds slice, arithmetic overflow, capacity overflow) and never
//! allocate absurdly. The N-D header path carries the same untrusted-input
//! guards as the 2-D parser (negative-dimension rejection, checked `numel`
//! product, offsets bounds-checked against the file length and required to be
//! consistent with the shape); this fuzzer proves those guards hold and guards
//! against regressions.

use libfuzzer_sys::fuzz_target;
use scirust_core::io;

fuzz_target!(|data: &[u8]| {
    // The only contract under test: parsing arbitrary bytes never panics.
    // A well-formed buffer yields Ok; anything else must be a clean Err.
    let _ = io::deserialize_state_dict_nd(data);
});
