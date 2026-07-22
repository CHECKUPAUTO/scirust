# Industrial evaluation datasets (phase 728)

Three preregistered workloads across two real industrial families
(turbofan run-to-failure PdM, semiconductor process control). Nothing here
is fetched by `cargo test` or by any library code; the only download path is
`scripts/fetch_industrial_datasets.sh`, which verifies every archive and
extracted file against the pinned SHA-256 checksums below. When the full
data is absent, integration tests **skip loudly** — they never fail and
never download.

## 1–2. NASA C-MAPSS turbofan degradation — run-to-failure PdM (FD001, FD003)

- **Source:** NASA Prognostics Center of Excellence data repository
  (`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`).
- **License:** work of the U.S. Government — public domain.
- **Citation:** A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*,
  PHM 2008.
- **Subsets used:** FD001 and FD003 (each 100 training units run to failure,
  26 columns: unit, cycle, 3 operational settings, 21 sensors). FD001 has a
  single fault mode (HPC degradation); FD003 has two (HPC + fan
  degradation). They are the **replication pair** for the SRCC questions —
  same domain and format, genuinely different degradation physics, so a
  result that holds on both is a replication in the preregistered sense.
- **Semantics:** training targets are `RUL(row) = max_cycle(unit) − cycle`;
  each unit's last training cycle has RUL 0. Units are the grouping key;
  cycles are the temporal key.
- **Checksums (SHA-256):**
  - archive `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`
  - `train_FD001.txt` `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`
  - `test_FD001.txt` `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`
  - `RUL_FD001.txt` `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`
  - `train_FD003.txt` `2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2`
  - `test_FD003.txt` `299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790`
  - `RUL_FD003.txt` `df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb`
- **Committed fixtures:** `tests/data/cmapss_fd001_head.txt` and
  `tests/data/cmapss_fd003_head.txt` — the first 40 cycles of units 1–2
  (80 rows each), for loader and pipeline tests without the full data.
  Public domain; fixture RUL targets are relative to the truncated fixture,
  not the real failure times, and are never used as evidence.

## 2. SECOM semiconductor manufacturing — real process/yield data

- **Source:** UCI Machine Learning Repository, dataset 179
  (`https://archive.ics.uci.edu/static/public/179/secom.zip`).
- **License:** CC BY 4.0.
- **Citation:** M. McCann, A. Johnston, *SECOM* (UCI Machine Learning
  Repository, 2008). DOI: 10.24432/C54305.
- **Semantics:** 1567 chronologically ordered process snapshots × 591
  sensor channels with `NaN` for missing readings; labels are pass (−1 →
  `0.0`) / fail (1 → `1.0`) yield outcomes. No grouping key; row order is
  the temporal key. The missing-value policy (train-fitted, §`missing`
  module) is mandatory before any finiteness-requiring stage.
- **Checksums (SHA-256):**
  - archive `eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e`
  - `secom.data` `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c`
  - `secom_labels.data` `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03`
- **Committed fixture:** `tests/data/secom_head.data` +
  `tests/data/secom_labels_head.data` — the first 12 rows, redistributed
  under CC BY 4.0 with the citation above, for loader and policy tests.

## Scope note

An earlier draft used the in-repo Opel Corsa OBD2 telemetry
(`examples/obd2_diagnostic/data/`) as a third workload. It was removed: that
is consumer automotive driving telemetry, not industrial machinery, and both
required families (turbofan PdM, process control) are already covered by the
three subsets above. The SRCC replication requirement is met within the
turbofan domain by the FD001/FD003 pair rather than by crossing into an
unrelated domain.
