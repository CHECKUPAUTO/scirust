#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
    use base64::Engine;
    let _ = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(data);
});
