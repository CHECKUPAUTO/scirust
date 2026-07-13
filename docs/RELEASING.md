# Releasing and branch protection

Repository rules and environment approvals are GitHub settings; source files
cannot enforce them. An administrator must configure the settings below once.

## Prepare a release

1. Set the root `Cargo.toml` package version to an exact `MAJOR.MINOR.PATCH`.
2. Move the relevant entries from `## [Non publié]` in `CHANGELOG.md` to a
   `## [MAJOR.MINOR.PATCH] — YYYY-MM-DD` section.
3. Run the required CI commands with the committed `Cargo.lock`. Do not update
   the lockfile during a release build.
4. Merge the release commit to protected `master` and wait for every required
   check below to pass.
5. Create an annotated tag matching the Cargo version exactly, for example:

   ```sh
   git tag -a v0.14.0 -m "SciRust v0.14.0"
   git push origin v0.14.0
   ```

The release workflow verifies the exact SemVer tag, Cargo version, changelog
section, and that the tagged commit is reachable from `origin/master`. Its
read-only validation job reruns format, clippy, build, tests, and complete
workspace SBOM generation with `--locked`. A separate job then crosses the
artifact boundary, verifies the SHA-256 checksum, and creates a **draft**
GitHub Release.

Configure the `release` GitHub environment with required reviewers. A reviewer
must inspect and publish the draft. SciRust does not publish crates.io packages.

## Protect `master`

Create a branch ruleset for `master` and require these CI check contexts:

- `Format Check`
- `Clippy`
- `Build & Test (nightly, x86_64)`
- `Build & Test (portable-simd, nightly)`
- `Build & Test (nightly SIMD extensions)`
- `Build & Test (stable, x86_64)`
- `Check (MSRV 1.89.0)`
- `Check (windows-latest, stable)`
- `Check (macos-latest, stable)`
- `Check opt-in network features`
- `Cross-check (aarch64)`
- `License & Security Audit`
- `Miri (memory safety and numeric crates)`
- `GPU (wgpu / lavapipe)`
- `SBOM (CycloneDX)`

Also require:

- pull requests with at least one approval from someone other than the author;
- all review conversations resolved and the branch up to date;
- no administrator bypass, force-push, or deletion;
- signed commits/tags if the organization has a signing policy.

The bounded fuzz smoke test and external coverage upload remain informational.
Long fuzz campaigns and CUDA hardware tests run outside the hosted merge gate.

After changing a workflow job name, update the required check context in the
ruleset. Audit the ruleset and `release` environment periodically; repository
files cannot detect an administrator weakening those settings.
