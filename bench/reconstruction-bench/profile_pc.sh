#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${NVBIT_ROOT:=/scratch/mindreia/qwentization/tools/nvbit_release_x86_64}"
python3 prepare_pc_tool.py "$NVBIT_ROOT"
cmake -S "$NVBIT_ROOT/tools/pc_hist" -B "$NVBIT_ROOT/tools/pc_hist/build" \
  -DNVBIT_TOOL_CUDA_ARCHITECTURES=90
cmake --build "$NVBIT_ROOT/tools/pc_hist/build" -j "${BUILD_JOBS:-16}"
tool="$NVBIT_ROOT/tools/pc_hist/build/opcode_hist.so"
test -f "$tool"
out="${RESULTS_ROOT:-$PWD/../../results}/reconstruction_pc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
out=$(cd "$out" && pwd)
for f in q4_0 q4_K iq4_xs; do
  # Independent CPU checks run without instrumentation before profile capture.
  ./build/reconstruct --format "$f" --m 16 --k 1024 --nmat 1 --validate-only \
    --out "$out/validate_$f.csv" > "$out/validate_$f.log" 2>&1
  for size in small stream; do
    if [[ "$size" == small ]]; then dims=(--m 256 --k 1024 --nmat 1)
    else dims=(--m 17408 --k 5120 --nmat 4); fi
    env LD_PRELOAD="$tool" COUNT_WARP_LEVEL=0 EXCLUDE_PRED_OFF=1 TOOL_VERBOSE=1 \
      ./build/reconstruct --format "$f" "${dims[@]}" --profile \
      --out "$out/${size}_$f.csv" \
      2>&1 | tee "$out/${size}_${f}_pc.log"
    if ! grep -Eq '\|pc=[0-9]+\|.* = [1-9][0-9]*' "$out/${size}_${f}_pc.log"; then
      echo "No per-PC instruction counts were recorded; stopping." >&2; exit 1
    fi
  done
done
python3 parse_pc.py "$out"
echo "PC RESULTS: $out"
