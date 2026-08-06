# License Architecture

Tokens are versioned `ccoslic1` records with explicit algorithm and key ID. Ed25519 and SLH-DSA keyrings are separate and bounded; unknown IDs, algorithms, duplicate fields, malformed keys, expiry, machine mismatch, and oversized input fail closed. Offline revocation lists are signed `ccosrev1` records and are never trusted merely because a local file exists. Test keys are rejected for release builds.

This is an engineering description, not legal advice. The repository uses a
custom dual-license identifier and PolyForm Noncommercial text; commercial use
requires the rightsholder's separate terms. EU software copyright context is
Directive 2009/24/EC ([EUR-Lex](https://eur-lex.europa.eu/eli/dir/2009/24/oj)).
