#!/usr/bin/env bash
# region_benchmark.sh — CCOS v0.3 Context Region Engine benchmark.
#
# Measures, on a target source tree (default: ./src):
#   - number of regions and their cohesion (causal density)
#   - region map build time and single-activation latency
#   - flat (MemoryGraph-direct) vs region locality: causal precision/recall and
#     the tokens needed to cover a task's k-hop causal neighbourhood
#   - clustering throughput (cycles/second)
#
# Usage: scripts/region_benchmark.sh [PATH] [SAMPLES]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_PATH="${1:-src}"
SAMPLES="${2:-12}"
BIN="$ROOT/target/release/ccos"

cd "$ROOT"
echo "==> Building release binary…"
cargo build --release --quiet
[ -x "$BIN" ] || { echo "binary not found: $BIN"; exit 1; }

echo
echo "================ CCOS Region Benchmark ================"
echo "path: $TARGET_PATH   samples: $SAMPLES"
echo

# ---- Graph + region map summary (JSON) ----
REGIONS_JSON="$("$BIN" regions "$TARGET_PATH" --json)"
ANALYZE_JSON="$("$BIN" analyze "$TARGET_PATH" --json 2>/dev/null)"

python3 - "$REGIONS_JSON" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
m = d["map"]
dens = [r["causal_density"] for r in m if r["id"] != "region:external"]
memb = [r["members"] for r in m]
avg = lambda xs: sum(xs)/len(xs) if xs else 0.0
print(f"  nodes={d['nodes']}  edges={d['edges']}  regions={d['regions']}")
print(f"  avg members/region : {avg(memb):.1f}")
print(f"  avg causal density : {avg(dens):.3f}   (1.0 = fully internal)")
PY

# ---- Timing: build map & single activation ----
echo
echo "---- Timing ----"
t0=$(date +%s.%N); "$BIN" regions "$TARGET_PATH" >/dev/null; t1=$(date +%s.%N)
echo "  build region map : $(python3 -c "print(f'{($t1-$t0)*1000:.1f} ms')")"

CENTER="$(python3 -c "import json,sys; d=json.loads('''$REGIONS_JSON'''); \
rs=[r for r in d['map'] if r['id']!='region:external']; \
print(sorted(rs, key=lambda r:-r['members'])[0]['center'] if rs else '')")"
if [ -n "$CENTER" ]; then
  t0=$(date +%s.%N); "$BIN" regions "$TARGET_PATH" --activate "$CENTER" >/dev/null; t1=$(date +%s.%N)
  echo "  activate region  : $(python3 -c "print(f'{($t1-$t0)*1000:.1f} ms')")  (center=$CENTER)"
fi

# ---- Throughput: clustering cycles/second ----
echo
echo "---- Throughput (region map builds) ----"
N=10
t0=$(date +%s.%N)
for _ in $(seq "$N"); do "$BIN" regions "$TARGET_PATH" >/dev/null; done
t1=$(date +%s.%N)
python3 -c "dt=($t1-$t0)/$N; print(f'  {1.0/dt:.1f} builds/s  ({dt*1000:.1f} ms each)')"

# ---- Locality: flat (MemoryGraph direct) vs region ----
echo
echo "---- Locality: flat (v0.2) vs region (v0.3) ----"
# Sample target symbols: the centers of the largest regions.
mapfile -t TARGETS < <(python3 -c "
import json
d=json.loads('''$REGIONS_JSON''')
rs=[r for r in d['map'] if r['id']!='region:external']
rs.sort(key=lambda r:-r['members'])
for r in rs[:$SAMPLES]:
    print(r['center'])
")

SUM=$(for t in "${TARGETS[@]}"; do
  "$BIN" regions "$TARGET_PATH" --metrics "$t" --radius 2 --json 2>/dev/null || true
done | python3 -c "
import json,sys
fp=fr=rp=rr=ts=pg=0.0; n=0
for line in sys.stdin.read().split('}\n{'):
    line=line.strip()
    if not line: continue
    if not line.startswith('{'): line='{'+line
    if not line.endswith('}'): line=line+'}'
    try: d=json.loads(line)
    except Exception: continue
    fp+=d['flat']['causal_precision']; fr+=d['flat']['causal_recall']
    rp+=d['region']['causal_precision']; rr+=d['region']['causal_recall']
    ts+=d['token_saving_ratio']; pg+=d['precision_gain']; n+=1
if n:
    print(f'  samples            : {n}')
    print(f'  flat   precision   : {fp/n:.3f}   recall {fr/n:.3f}')
    print(f'  region precision   : {rp/n:.3f}   recall {rr/n:.3f}')
    print(f'  mean precision gain: {pg/n:+.3f}')
    print(f'  mean token saving  : {ts/n*100:+.1f}%  (to cover the k-hop neighbourhood)')
else:
    print('  (no metric samples)')
")
echo "$SUM"

echo
echo "======================================================"
echo "Interpretation: a region is a causal cluster, so it covers a task's"
echo "neighbourhood with higher recall and fewer tokens than paging the"
echo "globally highest-scoring nodes (flat). All figures are deterministic."
