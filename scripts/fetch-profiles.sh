#!/usr/bin/env bash
set -euo pipefail

# Run on your Mac with the run ID printed by profile-unfused.sh.
if [[ $# -ne 1 || ! "$1" =~ ^unfused-[0-9]{8}-[0-9]{6}-[0-9]+$ ]]; then
  echo "Usage: bash $0 unfused-YYYYMMDD-HHMMSS-JOBID" >&2
  exit 1
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
run_id="$1"
destination="$project_dir/results/$run_id"
mkdir -p "$destination"
remote="mindreia@kuma.hpc.epfl.ch:/scratch/mindreia/qwen-bench/results/$run_id"
# Only copy reports and small text logs; omit derived SQLite exports.
scp "$remote/*.nsys-rep" "$remote/*.txt" "$remote/*.log" "$destination/"
printf 'Downloaded profiles and logs to %s\n' "$destination"
