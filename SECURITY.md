# Politique de sécurité

## Signalement de vulnérabilités

Signalez toute vulnérabilité en privé à **contact@checkupauto.fr**
(mainteneur, cf. `paper/SciRust-technical-report.md`). N'ouvrez pas
d'issue publique pour une faille exploitable. Accusé de réception sous
7 jours.

## Surface et garanties

- **Pur Rust, zéro FFI** : pas de bibliothèque C/C++ embarquée ; la
  chaîne d'approvisionnement se limite aux crates listées dans
  `Cargo.lock` (committé) et auditées par `cargo deny check`
  (advisories RustSec, licences, sources) à chaque CI.
- **`unsafe` confiné** : intrinsics SIMD uniquement, documentés par des
  en-têtes de sûreté (`scirust-simd/src/dispatch.rs`) ; aucun `unsafe`
  dans les chemins d'API publics de haut niveau.
- **Déterminisme** : l'inférence est bit-exacte et rejouable (runtime
  SRT1) — propriété utile aux audits forensiques.

## SBOM (nomenclature logicielle)

- **Format CycloneDX 1.x (JSON)** — standard consommable par les scanners
  industriels (OWASP Dependency-Track, Grype, etc.).
- **Génération reproductible** : `./scripts/generate-sbom.sh` (s'appuie sur
  `cargo cyclonedx` + le `Cargo.lock` committé). Un instantané est versionné
  dans [`docs/sbom/`](docs/sbom/) pour visibilité immédiate.
- **CI/Release** : le job `sbom` régénère et publie le SBOM en artefact à
  chaque build ; le workflow de release l'attache à chaque tag `v*`
  (cf. `release v0.14`). Aucune divergence silencieuse : la source de
  vérité reste `Cargo.lock` + la régénération.

## Avis connus acceptés

- RUSTSEC-2024-0436 (`paste`, unmaintained — non-vulnérabilité) :
  dépendance transitive de nalgebra→simba, sans correctif amont ;
  ignoré avec justification dans `deny.toml`.
