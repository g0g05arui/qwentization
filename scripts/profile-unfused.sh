#!/usr/bin/env bash
set -euo pipefail

# Run on a Kuma compute node in a Slurm GPU allocation.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Run this inside a GPU allocation, or use request-h100.sh from your Mac.' >&2
  exit 1
fi
module load gcc cmake cuda
cd /scratch/mindreia/qwen-bench
bench="./llama.cpp/build/bin/llama-bench"
model="models/Qwen3.8-27B-Q4_0.gguf"
[[ -x "$bench" ]] || { echo "Missing executable: $bench" >&2; exit 1; }
[[ -s "$model" ]] || { echo "Missing model: $model" >&2; exit 1; }
command -v nsys >/dev/null
nvidia-smi

# Verify that this checkout implements the runtime switch.
if ! grep -R -q --include='*.cu' --include='*.cuh' \
    'GGML_CUDA_DISABLE_FUSION' llama.cpp/ggml/src/ggml-cuda; then
  echo 'This checkout does not expose GGML_CUDA_DISABLE_FUSION; stopping.' >&2
  exit 1
fi

run_id="unfused-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID}"
result_dir="results/$run_id"
mkdir -p "$result_dir"
export GGML_CUDA_DISABLE_FUSION=1

{
  date -u
  hostname
  printf 'SLURM_JOB_ID=%s\nGGML_CUDA_DISABLE_FUSION=1\n' "$SLURM_JOB_ID"
  git -C llama.cpp rev-parse HEAD
  module list 2>&1
  nvcc --version
  nsys --version
  nvidia-smi -q
} > "$result_dir/environment.txt"
"$bench" --help > "$result_dir/bench-help.txt" 2>&1

common_args=(-m "$model" -ngl 999 -t 16 -b 512 -ub 512 -fa on -r 3)
profile_args=(--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node)

for phase in prefill decode; do
  if [[ "$phase" == prefill ]]; then
    phase_args=(-p 512 -n 0)
  else
    phase_args=(-p 0 -n 128)
  fi
  report="$result_dir/$phase-unfused"
  echo "Profiling $phase into $report.nsys-rep"
  nsys profile "${profile_args[@]}" -o "$report" \
    "$bench" "${common_args[@]}" "${phase_args[@]}" \
    2>&1 | tee "$result_dir/$phase-run.log"
  [[ -s "$report.nsys-rep" ]] || { echo "Missing report: $report.nsys-rep" >&2; exit 1; }
  nsys stats --report cuda_gpu_kern_sum "$report.nsys-rep" \
    2>&1 | tee "$result_dir/$phase-kernels.txt"
done

printf '\nCompleted: /scratch/mindreia/qwen-bench/%s\n' "$result_dir"
printf 'On your Mac, run: bash fetch-profiles.sh %s\n' "$run_id"
echo 'Weight unpacking/scaling remain inside matrix kernels. Captures include warmup.'
