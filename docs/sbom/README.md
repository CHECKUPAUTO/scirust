# SBOM — Software Bill of Materials

[`scirust.cdx.json`](scirust.cdx.json) is the aggregated CycloneDX JSON SBOM
for **all Cargo workspace members** and their resolved dependency graphs. It is
not merely the dependency closure of the top-level façade crate.

## Provenance and verification

- `cargo cyclonedx` generates one source BOM beside every workspace member's
  manifest from the committed `Cargo.lock` (cargo-cyclonedx 0.5 has no
  `--workspace` flag because workspace traversal is already the default).
- `scripts/merge-cyclonedx.py` merges components and dependency edges, removes
  duplicate identities, sorts the result, and fails if any Cargo workspace
  package is absent.
- Absolute checkout paths emitted by `cargo-cyclonedx` are normalized
  recursively to `file://workspace` in component identities, nested targets,
  PURLs, properties, external references, and dependency edges. Two builds of
  the same commit at different filesystem locations therefore produce the same
  canonical JSON bytes.
- The aggregate omits random serial numbers and derives its timestamp from
  `SOURCE_DATE_EPOCH` (the current commit time by default).

## Regenerate

```sh
cargo install cargo-cyclonedx --locked
./scripts/generate-sbom.sh
```

The generator runs the checkout-independence regression test automatically.
It can also be run directly without Rust tooling:

```sh
python3 scripts/test_merge_cyclonedx.py
```

The generated file is intentionally ignored rather than committing a snapshot
that can silently become stale. CI regenerates it and uploads it as an
artifact. The release workflow regenerates it from the tagged commit and
publishes it together with a SHA-256 checksum; that attachment is authoritative
for the released version.
