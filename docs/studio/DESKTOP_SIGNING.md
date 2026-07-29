# Code signing and the updater

**Status: the pipeline is complete and tested; the credentials are not in
this repository and cannot be.**

That distinction is the whole content of this document. Everything that can
be built without a certificate has been built. What is left is not code — it
is a purchase, a private key, and a URL that someone has to own.

## What is true today

Every artifact this repository produces is **unsigned**, and says so:

- the Windows installer raises SmartScreen, as documented in
  `WINDOWS_DESKTOP_ACCEPTANCE.md`;
- the CI artifact is named `scirust-studio-windows-preview-unsigned`;
- the build log carries two lines from `release-config` stating that the
  updater is disabled and the installer is unsigned.

None of that is an oversight to be tidied up later. A preview that *pretended*
to be signed would be worse than one that plainly is not, because the whole
value of a signature is that its absence is visible.

## What `release-config` does

`apps/scirust-studio/tools/release-config` reads the environment before every
release build and decides one of three things. It is a Rust binary with no
dependencies — anything it links is something that can affect a signed
artifact — and its decision logic has fourteen tests that need no certificate
to run.

| Environment | Outcome |
|---|---|
| No signing variables set | Build proceeds, unsigned, and says so. Patch is `{}`. |
| Every variable in a group set | Build proceeds signed, with the config patch that turns it on. |
| **Some** variables in a group set | **Build fails.** |

The third row is the reason the tool exists.

### Why a half-configuration is refused rather than warned about

Each partial state produces a build that completes, an artifact that runs, and
a security property that silently is not there:

- **A public key with no private key** ships an application that checks for
  updates it can never verify. It refuses every update it is offered, which
  looks like a broken server rather than a broken build.
- **A private key with no public key** produces update artifacts that are
  signed and an application that does not check the signature. That is
  strictly worse than shipping no updater at all.
- **A certificate thumbprint with no timestamp URL** produces a signature that
  stops verifying the day the certificate expires. The installer works for
  months and then, for no visible reason, does not.

None of those announce themselves at build time, so none of them can be left
to a warning somebody scrolls past.

A variable set to the **empty string counts as absent**, because CI systems
routinely expose an unset secret that way and treating it as present is how an
empty private key gets handed to a signer. But one real value beside two empty
ones is still partial, and still refused — there is a test for each.

## What a maintainer must supply

### The updater

Three values, and all three or none:

| Name | Kind | What it is |
|---|---|---|
| `TAURI_SIGNING_PRIVATE_KEY` | secret | The minisign private key update artifacts are signed with. Generate with `cargo tauri signer generate`. |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | secret | Its password, if you set one. |
| `SCIRUST_UPDATER_PUBKEY` | secret | The matching public key, which is compiled into the application. |
| `SCIRUST_UPDATER_ENDPOINTS` | variable | Comma-separated URLs the application asks for updates. **No default**, deliberately: an updater pointed at a URL nobody chose is a supply-chain hole with a plausible-looking config. |

The private key must never enter this repository, a build log, or an issue.
`release-config` reads it only to check that it is *present* and never echoes
it; a test asserts it does not appear in the emitted patch or in any of the
notes printed to stderr.

### Windows Authenticode

Two values, both or neither:

| Name | Kind | What it is |
|---|---|---|
| `SCIRUST_WINDOWS_CERT_THUMBPRINT` | secret | SHA-1 thumbprint of a code-signing certificate already installed in the runner's certificate store. |
| `SCIRUST_WINDOWS_TIMESTAMP_URL` | variable | An RFC 3161 timestamp authority. Required — see above. |

Getting the certificate *into* the runner's store is the part this repository
cannot help with: it depends on whether you use a hardware token, an
EV certificate held by a cloud signing service, or a PFX imported at the start
of the job. Whatever the route, the thumbprint is what Tauri needs afterwards.

## What is deliberately NOT wired

**The updater plugin is not registered in the application.** The build
pipeline can produce signed update artifacts and stamp the public key into the
config, but `tauri-plugin-updater` is not a dependency of
`scirust-studio-desktop` and no `updater:*` permission appears in the
capability file.

That is a decision, not an omission. Registering it would:

- add a runtime component that makes outbound network requests, into an
  application whose entire security posture is that the webview holds no
  general network, filesystem or process capability (see
  `DESKTOP_SECURITY.md`);
- add a capability grant that could not be exercised end-to-end in this
  repository, because there is no key to sign with and no endpoint to talk
  to — so it would be an untested grant, which is exactly the kind this
  project refuses elsewhere.

The order is therefore: obtain the signing material, configure the variables
above, confirm the pipeline produces signed artifacts, and *then* register the
plugin against a channel that actually exists and can be tested against. Doing
it in the other order means shipping a network capability on trust.

## Verifying a signed build

Once configured, three things should be checked on the artifact rather than in
the log:

1. `Get-AuthenticodeSignature .\scirust-studio.exe` reports `Valid`, and its
   `TimeStamperCertificate` is not null. A null timestamp is the failure the
   `SCIRUST_WINDOWS_TIMESTAMP_URL` rule exists to prevent.
2. The bundle directory contains a `.sig` file beside each updater artifact.
   Its absence means `createUpdaterArtifacts` did not take effect, and the
   updater has nothing to verify.
3. The application's own acceptance script
   (`scripts/studio/test-desktop-artifact.ps1`) still passes. A signature that
   broke the sidecar's resolution is a signature that broke the application.
