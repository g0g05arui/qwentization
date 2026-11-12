#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n ${LD_PRELOAD:-} ]]; then
  echo "LD_PRELOAD is set. Start an uninstrumented shell before timing." >&2
  exit 1
fi
prefix=${GGML_PREFIX:-/scratch/mindreia/qwentization/ggml-nvbit-install}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$prefix" \
  -DCMAKE_CUDA_ARCHITECTURES=90 2>&1 | tee build-config.log
cmake --build build -j "${BUILD_JOBS:-16}" 2>&1 | tee build.log
run_dir="${RESULTS_ROOT:-$PWD/../../results}/reconstruction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
{
  hostname
  nvidia-smi
  nvcc --version
  cmake --version
  sha256sum reconstruct.cu CMakeLists.txt
  cat build/CMakeCache.txt
} > "$run_dir/metadata.txt" 2>&1

# Gate every timed experiment on an independent CPU-reference smoke check.
for f in q4_0 q4_K iq4_xs; do
  ./build/reconstruct --format "$f" --m 16 --k 1024 --nmat 1 \
    --validate-only --out "$run_dir/validate_$f.csv" \
    2>&1 | tee "$run_dir/validate_$f.log"
done

# Small input: 136–144 KiB packed, plus 512 KiB decoded output.
# Streaming input: about 190–200 MiB packed, exceeding the H100's L2.
# Equal weight counts across formats, NOT equal byte counts for IQ4_XS.
# Fixed seed, randomized format order in each outer pass.
for pass in 0 1 2; do
  order=$(python3 - "$pass" <<'PY'
import random,sys
formats=['q4_0','q4_K','iq4_xs']
random.Random(471+int(sys.argv[1])).shuffle(formats)
print(' '.join(formats))
PY
)
  for f in $order; do
    ./build/reconstruct --format "$f" --m 256 --k 1024 --nmat 1 \
      --seconds "${SECONDS_PER_CASE:-1}" --repeat 1 \
      --out "$run_dir/small_${f}_pass${pass}.csv" \
      2>&1 | tee "$run_dir/small_${f}_pass${pass}.log"
    ./build/reconstruct --format "$f" --m 17408 --k 5120 --nmat 4 \
      --seconds "${SECONDS_PER_CASE:-1}" --repeat 1 \
      --out "$run_dir/stream_${f}_pass${pass}.csv" \
      2>&1 | tee "$run_dir/stream_${f}_pass${pass}.log"
  done
done
python3 summarize.py "$run_dir"
echo "RESULTS: $run_dir"
