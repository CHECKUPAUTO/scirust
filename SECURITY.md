# Politique de sécurité

## Signalement de vulnérabilités

Signalez toute vulnérabilité en privé à **contact@checkupauto.fr**
(mainteneur, cf. `paper/SciRust-technical-report.md`). N'ouvrez pas
d'issue publique pour une faille exploitable. Accusé de réception sous
7 jours.

## Surface et garanties

- **Pur Rust, aucune bibliothèque C/C++ embarquée** : le workspace actif
  n'inclut **aucune** dépendance FFI *consommée* (pas de liaison à une
  bibliothèque C/C++ tierce). La chaîne d'approvisionnement des *crates*
  se limite à celles listées dans `Cargo.lock` (committé) et auditées par
  `cargo deny check` (advisories RustSec, licences, sources) à chaque CI.

  > **Note FFI exportée.** `scirust-runtime/src/enclave.rs` **exporte** un
  > point d'entrée `extern "C"` (`safe_enclave_infer`) destiné à un
  > environnement TEE / TrustZone `#![no_std]`. Il s'agit d'une ABI
  > Rust→C *exportée* (le runtime est appelable depuis C), pas d'une
  > bibliothèque C embarquée. Les tailles des tampons (`dims`) sont
  > validées par rapport aux slices Rust **avant** le chemin `unsafe`
  > (`EnclaveRuntime::infer`), de sorte qu'un `dims` incohérent est
  > rejeté (`Err`) plutôt que de provoquer une lecture/écriture hors
  > bornes dans l'enclave.

  > **Note archive.** Le répertoire `archive/` contient du code plus
  > ancien (notamment `archive/scirust-gpu/{cublas.rs,cuda_backend.rs}`,
  > `archive/scirust-simd/sve.rs`) qui **utilise** une FFI C/CUDA. Ce code
  > n'est **pas** membre du workspace actif (hors `Cargo.toml`), n'est pas
  > compilé par la CI, et est conservé à titre historique. Il est donc
  > **hors périmètre** des garanties ci-dessus.

- **`unsafe` confiné et justifié** : le `unsafe` apparaît dans plusieurs
  modules (intrinsics SIMD `scirust-simd/src/{dispatch,complex}.rs`,
  alignement mémoire `scirust-arena/src/{slab,aligned,allocator}.rs` avec
  backing `AlignBlock(128)`, autodiff/tenseur/matrix dans `scirust-core`,
  et le point d'entrée enclave susmentionné). Chaque bloc est documenté par
  un en-tête de sûreté (alignement, invariants, version anti-use-after-free).
  Aucun `unsafe` n'est requis **de la part de l'appelant** sur les chemins
  d'API publics de haut niveau : l'`unsafe` est interne et encapsulé.

- **Déterminisme** : l'inférence est bit-exacte et rejouable (runtime
  SRT1) — propriété utile aux audits forensiques. Le bruit de la
  *campagne* d'injection de fautes (`scirust-func-safety/src/fault_injection.rs`,
  `NoiseInjection`) est lui-même déterministe (LCG à graine fixe dérivée
  de l'indice du neurone) afin de préserver la reproductibilité ;
  `rand::random` non seedé n'est utilisé dans aucun chemin d'inférence.

## SBOM (nomenclature logicielle)

- **Format CycloneDX 1.x (JSON)** — standard consommable par les scanners
  industriels (OWASP Dependency-Track, Grype, etc.).
- **Génération reproductible** : `./scripts/generate-sbom.sh` (s'appuie sur
  `cargo cyclonedx` + le `Cargo.lock` committé). Un instantané est versionné
  dans [`docs/sbom/`](docs/sbom/) pour visibilité immédiate.
- **CI/Release** : le job `sbom` régénère et publie le SBOM en artefact à
  chaque build ; le workflow de release l'attache à chaque tag `v*`
  (cf. `release v0.14`). La source de vérité reste `Cargo.lock` + la
  régénération ; l'instantané committé ne doit pas être traité comme source.

## Chaîne d'approvisionnement CI

- Les workflows GitHub utilisent des actions tierces. Le pinning se fait par
  tag de version (`@v2`, `@nightly`) ; pour durcir la chaîne d'approvisionnement,
  pinner ces actions par **SHA de commit** est recommandé
  (cf. audit `AUDIT_COMPLET.md`, finding S4).
- Aucun workflow n'utilise `pull_request_target` (le pattern d'élévation
  de privilèges dangereux). Le workflow `release.yml` restreint
  `permissions: contents: write` au seul besoin de créer la release.

## Avis connus acceptés

- RUSTSEC-2024-0436 (`paste`, unmaintained — non-vulnérabilité) :
  dépendance transitive de nalgebra→simba, sans correctif amont ;
  ignoré avec justification dans `deny.toml`.

## Intégrité des artefacts de certification

- Les dossiers d'evidence (`scirust-func-safety/src/evidence.rs`,
  chaîne FNV-1a) sont **tamper-evident mais non tamper-resistant** :
  ils détectent une édition naïve (champ modifié sans recalcul de la
  chaîne) mais ne résistent pas à un attaquant qui recalcule toute la
  chaîne (algorithme public, sans clé secrète). L'intégrité repose sur le
  contrôle d'accès en écriture au fichier et sur l'argument de preuve
  de la runtime (inférence vérifiable). Ne pas authentifier un dossier
  non fiable uniquement via `from_json().verify()`.