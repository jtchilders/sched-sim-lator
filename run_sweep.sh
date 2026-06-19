#!/usr/bin/env bash
# Arrival rate sensitivity sweep for the capacity queue.
# Runs sim at 10%, 25%, 50%, 100% of empirical λ, capacity-pool=512 only.
# Results go to results/sweep_<scale>/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV=".venv/bin/python"
if [ ! -x "$VENV" ]; then
    VENV="python3"
fi

DURATION_DAYS=30
SEED=42
POOL=512

for SCALE in 0.10 0.25 0.50 1.00; do
    LABEL=$(printf "scale%.2f" "$SCALE" | tr '.' '_')
    OUTDIR="results/sweep_${LABEL}"
    echo ""
    echo "======================================================="
    echo " Running sweep: capacity arrival_rate × ${SCALE}"
    echo " Output: ${OUTDIR}"
    echo "======================================================="
    $VENV sim.py \
        --source fitted \
        --duration-days $DURATION_DAYS \
        --seed $SEED \
        --capacity-pool $POOL \
        --arrival-rate-scale "$SCALE" \
        --plot \
        --outdir "$OUTDIR" \
        --csv "${OUTDIR}/jobs.csv"
done

echo ""
echo "======================================================="
echo " Sweep complete. Results:"
echo "======================================================="
for SCALE in 0.10 0.25 0.50 1.00; do
    LABEL=$(printf "scale%.2f" "$SCALE" | tr '.' '_')
    OUTDIR="results/sweep_${LABEL}"
    echo ""
    echo "--- λ × ${SCALE} ---"
    grep -E "unstarted|Average util|max_wait|STARVATION|Jobs sub" "${OUTDIR}"/*.txt 2>/dev/null || true
done
