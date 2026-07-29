# Windows desktop acceptance

How to obtain, run and check the SciRust Studio Windows preview, and what its
limits are. Read the limits first.

## What this artifact is, and is not

It **is** a real native application: a Tauri 2 shell, the operating system's
WebView2, a WebAssembly interface, and the same worker, adapters and run store
the CLI uses. It runs actual simulations and records actual results.

It **is not**:

* **signed.** No code-signing certificate is involved. Windows SmartScreen
  will warn about it. See below.
* **a released product.** There is no updater, no installer publication and no
  licensing. It is a preview artifact produced by CI for review.
* **complete.** Five capabilities of sixteen are adapted; opening and saving
  scenario files needs a native file picker the shell does not yet expose.

## Getting it

`.github/workflows/studio-desktop.yml` publishes
`scirust-studio-windows-preview-unsigned` on every run of the workflow. It
contains:

| File | What it is |
|---|---|
| `scirust-studio.exe` | The application, run in place |
| `bundle/nsis/*.exe` | A per-user NSIS installer |
| `studio-acceptance.json` | The acceptance report CI produced for this build |

## The SmartScreen warning

On first run Windows shows **"Windows protected your PC"**. This is expected
and correct: the binary carries no Authenticode signature, so SmartScreen has
no publisher to check and no reputation to consult.

To run it anyway: *More info* → *Run anyway*.

Do not treat that click as routine. It is the correct response **only**
because you obtained this artifact from this repository's own CI run and can
see which commit produced it. An unsigned binary from anywhere else deserves
the warning it gets. Signing is deferred to a later phase, and this document
would be dishonest if it presented the warning as noise.

## Running it

```powershell
# In place, no installation.
.\scirust-studio.exe

# The scientific path end to end, with no window. Prints JSON, exits 0 on
# success.
.\scirust-studio.exe --smoke-test-backend

# Proves the WebView really loaded the bundled interface: the frontend must
# call back within 60 seconds or this exits non-zero.
.\scirust-studio.exe --smoke-test-window

.\scirust-studio.exe --version
```

The installer installs per-user (`installMode: currentUser`), so no elevation
prompt appears and nothing is written outside the user's profile.

## The automated acceptance check

`scripts/studio/test-desktop-artifact.ps1` runs against the built artifact —
not the source it came from — and writes `studio-acceptance.json` beside the
executable. CI runs it before publishing; run it yourself with:

```powershell
./scripts/studio/test-desktop-artifact.ps1 -Verbose
```

It checks, in order:

1. the application executable exists;
2. the sidecar worker is staged under its exact target-triple name, with
   `.exe` after the triple, and is a real binary rather than a stub;
3. the staged frontend has an index document **and** a WebAssembly module —
   the two halves of "the window will not be blank";
4. `--version` runs and prints a version;
5. `--smoke-test-backend` reports `ok: true`, with a worker that handshook,
   a populated catalogue, a result stored under **schema v2** with real axis
   coordinates, a store-integrity check that verified, at least one
   scientific check passed and none failed;
6. an NSIS installer exists and is a plausible size.

Step 5 is the one that matters. It is the only check that proves the
scientific path works *in the artifact* rather than in a developer's build
tree.

## Manual acceptance

What a reviewer should be able to do, in one sitting, with no documentation
open:

| # | Action | Expected |
|---|---|---|
| 1 | Launch the application | Window opens; status bar shows the calculation engine **running** with its version, and the run-store path |
| 2 | Read the status bar | Backend "Rust / Tauri", precision "f64" — claims the application can substantiate |
| 3 | Open **Catalogue** | Five capabilities from the real registry, each with its parameters, initial state, outputs, scientific checks and solvers |
| 4 | Note `sim.chemistry.robertson` | Marked "indeterminate progress" — it is honest about not being able to report a fraction |
| 5 | Open a tutorial from **Home** | The tested scenario loads into the editor and the view switches to Experiment |
| 6 | Press **Validate** | "The scenario is valid." |
| 7 | Break the scenario (rename `mass` to `mas`) and validate | A structured problem with a title, an explanation, a suggested fix, the field and a line number — not a formatted blob |
| 8 | Restore it and press **Run** | A determinate progress bar advances and reports the simulation time reached |
| 9 | Run again and press **Cancel** mid-run | Status becomes *Cancelling* then **Cancelled** — never "failed" |
| 10 | Run the Robertson tutorial | Indeterminate activity, never a percentage |
| 11 | Let a run complete | The chart appears, plotted against the coordinates the solver produced; the caption states how many of how many points are drawn |
| 12 | Open the chart's text view | An accessible summary and a table of plotted coordinates that match the picture |
| 13 | Read the inspector | Metrics, scientific checks and provenance — with store integrity reported **separately** from the physics checks |
| 14 | Open **Runs**, press **Verify** on a run | The activity log reports that the stored bytes still match their hashes |
| 15 | Press `Ctrl+Shift+P` | The command palette; unavailable actions are listed **disabled with a reason**, not hidden |
| 16 | Type `sh` or `rm -rf /` into the activity console | Refused: "is not a command. This console runs named actions only." |
| 17 | Switch language to Français | Every visible string changes; numbers in the chart and the stored data do not |
| 18 | Switch theme to High contrast | Pure black and white with a heavier focus ring |
| 19 | Resize the window narrow | The inspector folds under the content rather than being squeezed |
| 20 | Close the window while a run is active | Confirmation before the run is abandoned |

### A legacy result

If you have a run recorded before schema v2 (or you copy one from
`scirust-studio-store/tests/fixtures/`), open it and confirm:

* the run list marks it **v1 / Legacy result**;
* the chart's horizontal axis is labelled **Sample index**, never "time";
* a notice above the chart says the spacing shown is not the spacing the
  solver used.

The application will not invent an axis for a result that does not carry one.

## Known limits in this preview

* **Unsigned**, as above.
* **Open / Save scenario are disabled**, with the reason "Not available in
  this build: it needs a native file picker". The actions exist in the palette
  so a user learns why rather than hunting for a menu that was never there.
* **Five capabilities**, not sixteen.
* **No updater, no telemetry, no network access of any kind.** The application
  makes no outbound request; the content security policy forbids it.
* **WebView2** must be present. Windows 11 and recent Windows 10 include it;
  older installations may need the Evergreen runtime.

## If something fails

Attach `studio-acceptance.json` and the output of

```powershell
.\scirust-studio.exe --smoke-test-backend
```

to the report. That JSON names the stage that failed, which distinguishes a
packaging problem from a scientific one — and those get fixed in entirely
different places.
