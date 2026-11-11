#!/usr/bin/env bash
set -euo pipefail

# Run on your Mac. Upload the profiler script, then hold one H100 allocation
# until both profiles and their summaries finish. No build or model download.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cluster="mindreia@kuma.hpc.epfl.ch"
remote_script="/scratch/mindreia/qwen-bench/scripts/profile-unfused.sh"

ssh "$cluster" 'mkdir -p /scratch/mindreia/qwen-bench/scripts'
scp "$script_dir/profile-unfused.sh" "$cluster:$remote_script"
ssh -t "$cluster" \
  'salloc --account=cs471 --partition=h100 --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --time=01:00:00 srun --pty bash -l /scratch/mindreia/qwen-bench/scripts/profile-unfused.sh'
