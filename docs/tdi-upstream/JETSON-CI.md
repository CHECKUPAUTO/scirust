# Jetson self-hosted CI

The TDI repository uses four self-hosted GitHub Actions runners on a Jetson ARM64 host.

## Runner inventory

- `jetson-tdi-01`
- `jetson-tdi-02`
- `jetson-tdi-03`
- `jetson-tdi-04`

All runners expose these labels:

- `self-hosted`
- `Linux`
- `ARM64`
- `jetson`
- `tdi`

The workflow targets:

```yaml
runs-on: [self-hosted, Linux, ARM64, jetson, tdi]
```

## Parallel validation

The CI workflow dispatches four independent jobs:

- formatting;
- tests;
- Clippy;
- preregistration integrity.

With four online runners, these jobs execute concurrently.

## Paths

Runner installations:

```text
/mnt/nvme/github-runners/jetson-tdi-01
/mnt/nvme/github-runners/jetson-tdi-02
/mnt/nvme/github-runners/jetson-tdi-03
/mnt/nvme/github-runners/jetson-tdi-04
```

Shared Rust installation:

```text
/mnt/nvme/github-runners/home/.cargo
/mnt/nvme/github-runners/home/.rustup
```

The workflow uses this persistent installation directly. It does not
download a Rust toolchain during each job and does not depend on a remote
Cargo cache action.

Persistent build directories:

```text
/mnt/nvme/github-runners/cache/tdi/formatting
/mnt/nvme/github-runners/cache/tdi/tests
/mnt/nvme/github-runners/cache/tdi/clippy
```

The only external action retained by the workflow is
`actions/checkout@v5`.

## Services

```text
actions.runner.Memorithm-TDI.jetson-tdi-01.service
actions.runner.Memorithm-TDI.jetson-tdi-02.service
actions.runner.Memorithm-TDI.jetson-tdi-03.service
actions.runner.Memorithm-TDI.jetson-tdi-04.service
```

Each runner service is enabled at boot and configured with:

```text
Restart=always
RestartSec=10s
```

## Watchdog

A systemd timer checks runner health every five minutes:

```text
github-runner-watchdog.timer
github-runner-watchdog.service
```

The watchdog restarts a runner when:

- its systemd service is inactive; or
- GitHub reports it offline during two consecutive checks.

## Operational checks

List runners:

```bash
gh api repos/Memorithm/TDI/actions/runners \
  --jq '.runners[] | {name,status,busy,labels:[.labels[].name]}'
```

Check services:

```bash
for n in 01 02 03 04; do
  systemctl status \
    "actions.runner.Memorithm-TDI.jetson-tdi-${n}.service" \
    --no-pager
done
```

Check watchdog:

```bash
systemctl status github-runner-watchdog.timer --no-pager
systemctl list-timers github-runner-watchdog.timer --no-pager
journalctl -u github-runner-watchdog.service --no-pager -n 100
```

## Network dependencies

The Jetson must resolve and reach at least:

```text
github.com
api.github.com
codeload.github.com
objects.githubusercontent.com
pipelines.actions.githubusercontent.com
broker.actions.githubusercontent.com
static.rust-lang.org
sh.rustup.rs
```

The host uses systemd-resolved with explicit DNS and fallback resolvers.

## Validated state

The four-runner configuration has been validated by:

- a successful parallel TDI workflow;
- one job assigned per runner;
- a deliberate runner shutdown;
- automatic systemd/watchdog recovery;
- restoration to GitHub `online` status;
- successful execution using the persistent shared Rust toolchain;
- successful warm-cache execution of all four jobs in parallel, with every
  job completing in less than one minute.
