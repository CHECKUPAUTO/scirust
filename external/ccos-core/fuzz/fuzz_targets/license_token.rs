#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
    if data.len() <= 1 << 20 {
        let _ = ccos_core::license::verify_token_blob(data, 0);
    }
});
