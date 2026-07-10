# Licensing

scirust is dual-licensed.

## 1. Noncommercial and personal use — free

scirust is available free of charge under the PolyForm Noncommercial License 1.0.0
(see [LICENSE.md](LICENSE.md)). This covers any noncommercial purpose, including
personal study, research, experimentation, hobby and amateur projects, and use by
charitable, educational, public research, public safety or health, and government
organizations.

## 2. Commercial use — paid license required

Any commercial use — use by or for a business with an anticipated commercial
application, including use in or as part of a product or service offered for a fee —
requires a separate commercial license.

Commercial licenses are sold **per module**: a license unlocks exactly the domain
modules you need (navigation, control, functional-safety, …) rather than the whole
catalogue. The unlock is enforced cryptographically by the `scirust-license` crate —
each license is a signed file listing the entitled modules, the licensee and a
validity window; the runtime embeds only a public Merkle root and verifies the
signature before granting access. See [`docs/PLAQUETTE_COMMERCIALE.md`](docs/PLAQUETTE_COMMERCIALE.md)
for the module catalogue and bundles, and [`scirust-license/README.md`](scirust-license/README.md)
for the mechanism.

A license may additionally be **node-locked** to a single machine, which (together
with a monthly validity window) supports per-machine subscription pricing. The bind
stores only a domain-separated SHA-256 hash of a host-supplied machine identifier —
never the raw id — and is part of the signed payload, so it cannot be edited or
removed. See [`scirust-license/README.md`](scirust-license/README.md) for the
per-machine flow.

To obtain a commercial license, contact: zekrititarek@gmail.com

## 3. Copyright

Copyright 2026 Tarek Zekriti. All rights reserved except as expressly granted by the
applicable license above.

## 4. Contributions

To preserve the dual-license model, external contributions are accepted only under a
Contributor License Agreement that licenses the contribution to the copyright holder
for use under both the noncommercial and the commercial license.
