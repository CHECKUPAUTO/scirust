# Compatibility

The supported compiler baseline is Rust 1.89.0; CI also checks latest stable. Deterministic community profiles must remain replay-compatible. SIMD/full backends and REPLAY-RELAX features are opt-in and may be architecture-sensitive; they are not labeled bit-exact without canonical vector evidence.
