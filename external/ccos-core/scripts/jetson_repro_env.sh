#!/usr/bin/env bash
# jetson_repro_env.sh — pin a Jetson (or generic Linux) to a stable, max-clock
# state so CCOS *measurements* are reproducible run-to-run.
#
# FOR BENCHMARKING / EVALUATION ONLY — not a production setting (it runs the
# board at max power with no thermal headroom).
#
# This does NOT speed up the CCOS kernel, which is <1% of an agent loop dominated
# by LLM inference (see docs/PERFORMANCE.md). It removes CPU/EMC frequency and
# thermal jitter so the paper's token-count and latency numbers are stable across
# runs. On a Jetson AGX Thor (Tegra SoC, unified memory) the controls are
# nvpmodel/jetson_clocks — there is no NUMA and nvidia-smi clock-locking does not
# apply.
#
# Usage:
#   sudo bash scripts/jetson_repro_env.sh            # pin to max / performance
#   sudo bash scripts/jetson_repro_env.sh --show     # show current state
#   sudo bash scripts/jetson_repro_env.sh --restore  # back to a dynamic governor
set -u

have() { command -v "$1" >/dev/null 2>&1; }

show_governors() {
  if compgen -G "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor" >/dev/null 2>&1; then
    echo "  CPU governors: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
  else
    echo "  CPU cpufreq: not exposed by this kernel"
  fi
}

set_governor() {
  local want="$1" n=0
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -w "$g" ] || continue
    if grep -qw "$want" "${g%scaling_governor}scaling_available_governors" 2>/dev/null; then
      echo "$want" >"$g" 2>/dev/null && n=$((n + 1))
    fi
  done
  echo "$n"
}

case "${1:-apply}" in
--show)
  echo "Current state:"
  show_governors
  have nvpmodel && nvpmodel -q 2>/dev/null | sed 's/^/  /'
  exit 0
  ;;
--restore)
  echo "Restoring a dynamic CPU governor (benchmark teardown)..."
  for cand in schedutil ondemand powersave; do
    n=$(set_governor "$cand")
    [ "$n" -gt 0 ] && {
      echo "  governor → $cand on $n core(s)"
      break
    }
  done
  show_governors
  echo "Note: nvpmodel/jetson_clocks persist until reboot or a manual 'nvpmodel -m <id>'."
  exit 0
  ;;
esac

echo "Pinning for reproducible measurement (benchmark mode — max power)..."

# 1. Jetson (Tegra): max power model + lock CPU/GPU/EMC clocks to their ceiling.
if have nvpmodel; then
  echo "  nvpmodel -m 0 (max performance mode)"
  nvpmodel -m 0 || echo "    (failed — run as root?)"
else
  echo "  nvpmodel: not present (not a Jetson) — skipping"
fi
if have jetson_clocks; then
  echo "  jetson_clocks (lock clocks to max)"
  jetson_clocks || echo "    (failed — run as root?)"
else
  echo "  jetson_clocks: not present (not a Jetson) — skipping"
fi

# 2. Generic Linux: CPU governor → performance (kills frequency-scaling jitter).
n=$(set_governor performance)
[ "$n" -gt 0 ] && echo "  CPU governor → performance on $n core(s)" ||
  echo "  CPU governor: unchanged (no writable cpufreq / no 'performance')"

echo
show_governors
echo "Done. Run with --restore to return to a dynamic governor after benchmarking."
