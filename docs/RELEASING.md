# Releasing & branch protection

This is the runbook for the two governance items that require a human with
repository admin rights (they cannot be done from inside the repo).

## Cutting a release (e.g. v0.14.0)

Releasing is automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml),
which fires on any pushed `v*` tag. A tag is a deliberate human action; the
workflow only reacts to it.

1. **Bump the version.** Set the `scirust` facade version in the root
   `Cargo.toml` (e.g. `0.14.0` → `0.15.0`). Bump any sub-crate that actually
   changed if/when they are published; today all sub-crates are
   `publish = false`.
2. **Finalize the changelog.** Move the accumulated entries under
   `## [Non publié]` in [`CHANGELOG.md`](../CHANGELOG.md) into a new
   `## [<version>] — <date>` section.
3. **Refresh the SBOM snapshot.** `./scripts/generate-sbom.sh` then commit
   `docs/sbom/scirust.cdx.json`.
4. **Commit, then tag and push:**
   ```sh
   git commit -am "release: v0.14.0"
   git tag v0.14.0
   git push origin master --tags
   ```
5. The release workflow then: re-runs the workspace build+test gate
   (`-D warnings`), regenerates the CycloneDX SBOM, creates the GitHub Release
   with auto-generated notes, and attaches `scirust.cdx.json`.

> Publishing to crates.io is **not** wired (every crate is `publish = false`,
> path deps, non-commercial licence). Releases are tags + artifacts only.

## Branch protection (master)

Configure once, in **Settings → Branches → Add branch ruleset** (or the
classic *Branch protection rules*) for `master`. This makes the CI gates
mandatory, which is what turns "the gates are green locally" into an enforced
contract — and is what prevents a recurrence of the merge regression that
broke compilation on all architectures.

Require these status checks to pass before merging (job names from
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)):

- `Format Check`
- `Clippy`
- `Build & Test (nightly, x86_64)`
- `Build & Test (stable, x86_64)`
- `Cross-check (aarch64)`
- `License & Security Audit`

Recommended companion settings:

- Require a pull request before merging (≥ 1 approval).
- Require branches to be up to date before merging.
- Require conversation resolution before merging.
- Do not allow force-pushes or deletions of `master`.

The `SBOM (CycloneDX)` and `Code Coverage` jobs are intentionally
informational (`continue-on-error`) and should **not** be marked as required.

### Via the GitHub API (alternative)

```sh
gh api -X PUT repos/Memorithm/scirust/branches/master/protection \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Format Check", "Clippy",
      "Build & Test (nightly, x86_64)", "Build & Test (stable, x86_64)",
      "Cross-check (aarch64)", "License & Security Audit"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": { "required_approving_review_count": 1 },
  "restrictions": null
}
JSON
```

> The status-check *contexts* must match the job `name:` values exactly. If a
> job name changes in `ci.yml`, update the protection rule too.
