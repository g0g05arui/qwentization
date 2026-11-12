#!/usr/bin/env bash
# Per-PC counts for the ORIGINAL llama.cpp matvec_energy binary, not our kernels.
set -euo pipefail
cd "$(dirname "$0")"
: "${NVBIT_ROOT:=/scratch/mindreia/qwentization/tools/nvbit_release_x86_64}"
: "${EXISTING_BENCH:=/scratch/mindreia/qwentization/build-nvbit/matvec_energy}"
python3 prepare_pc_tool.py "$NVBIT_ROOT"
cmake -S "$NVBIT_ROOT/tools/pc_hist" -B "$NVBIT_ROOT/tools/pc_hist/build" \
  -DNVBIT_TOOL_CUDA_ARCHITECTURES=90
cmake --build "$NVBIT_ROOT/tools/pc_hist/build" -j "${BUILD_JOBS:-16}"
tool="$NVBIT_ROOT/tools/pc_hist/build/opcode_hist.so"
test -f "$tool"; test -x "$EXISTING_BENCH"
out="${RESULTS_ROOT:-$PWD/../../results}/llama_pc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out";out=$(cd "$out" && pwd)
for f in q4_0 q4_K iq4_xs; do
  env LD_PRELOAD="$tool" COUNT_WARP_LEVEL=0 EXCLUDE_PRED_OFF=1 TOOL_VERBOSE=1 \
    GGML_CUDA_DISABLE_FUSION=1 \
    "$EXISTING_BENCH" -d CUDA0 -t "$f" -m 17408 -k 5120 \
    --nmat 4 --reps 1 --warmup 0 --idle 0 --cooldown 0 --no-check -s 0 -r 1 \
    -o "$out/$f" 2>&1 | tee "$out/${f}_pc.log"
  if ! grep -Eq '\|pc=[0-9]+\|.* = [1-9][0-9]*' "$out/${f}_pc.log"; then
    echo "No per-PC instruction counts were recorded; stopping." >&2;exit 1
  fi
done
python3 parse_pc.py "$out"
echo "ORIGINAL KERNEL PC RESULTS: $out"
