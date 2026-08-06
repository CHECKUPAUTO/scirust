# Licensing

CCOS is dual-licensed.

## 1. Noncommercial and personal use — free

CCOS is available free of charge under the PolyForm Noncommercial License 1.0.0
(see [LICENSE.md](LICENSE.md)). This covers any noncommercial purpose, including
personal study, research, experimentation, hobby and amateur projects, and use by
charitable, educational, public research, public safety or health, and government
organizations.

## 2. Commercial use — paid license required

Any commercial use — use by or for a business with an anticipated commercial
application, including use in or as part of a product or service offered for a fee —
requires a separate commercial license.

To obtain a commercial license, contact: zekrititarek@gmail.com

## 2b. The Pro tier (runtime feature unlock) — distinct from the copyright license

Two different things are both called "license"; do not confuse them:

- **The copyright license** (§1/§2 above) governs your *right to use the source
  code at all*. It is a legal document, not software.
- **The Pro license token** is a *runtime unlock*: a signed, offline-verified
  token (`$CCOS_LICENSE`, `$CCOS_LICENSE_FILE`, or `~/.config/ccos/license`) that
  switches a running engine from the **community tier** to the **Pro tier**. It
  gates only the premium capabilities — nine runtime features, from
  custom-authority-weights up to the CCOS_EXTENDED fusion kernels
  (`slhav2-full-kernel`, `rsi-self-improvement`, `rsi-dgm`). The **core is never
  gated**: without a token, ingestion, the causal graph, recall and replay are
  fully functional, and every Pro refusal is explicit (never a silent downgrade).
  Verification is a pure offline signature check (ed25519 and/or post-quantum
  SLH-DSA) — no network, no telemetry, air-gap friendly, fail-closed when no
  vendor key is baked in.

A commercial engagement typically includes both: the commercial copyright license
(§2) and Pro tokens for your deployments. Issuing tokens (vendor side: keygen,
signing, key embedding) and installing them (customer side) are documented in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) §4–§4c.

## 3. Copyright

Copyright 2026 Tarek Zekriti. All rights reserved except as expressly granted by the
applicable license above.

## 4. Contributions

To preserve the dual-license model, external contributions are accepted only under a
Contributor License Agreement that licenses the contribution to the copyright holder
for use under both the noncommercial and the commercial license.
