# scirust-cayley-filter

Deterministic experimental filtering in Rust based on sedenions.

## Safety rule

The Cayley filter is enabled only if:

`development_loss < 1.0`

Otherwise, the output remains identical to the input.

## Results

- successful on aligned synthetic noise;
- abstains on VoiceBank, MIT-BIH ECG, and CWRU;
- 71 tests passing;
- strict Clippy passing;
- unsafe code forbidden.

See `docs/ARCHITECTURE.md`.
