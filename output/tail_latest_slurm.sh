#!/usr/bin/env bash
set -euo pipefail

extension="out"

if [[ "${1:-}" == "--err" ]]; then
  extension="err"
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf 'Usage: %s [--err]\n' "${0##*/}"
  exit 0
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--err]\n' "${0##*/}" >&2
  exit 2
fi

shopt -s nullglob
latest_file=""
latest_job_id=-1

for file in slurm-*."$extension"; do
  [[ -e "$file" ]] || continue

  if [[ "$file" =~ -([0-9]+)\.($extension)$ ]]; then
    job_id="${BASH_REMATCH[1]}"
    if (( 10#$job_id > latest_job_id )); then
      latest_job_id=$((10#$job_id))
      latest_file="$file"
    fi
  fi
done

if [[ -z "$latest_file" ]]; then
  printf 'No slurm-*.%s files found in current directory\n' "$extension" >&2
  exit 1
fi

printf 'Tailing latest %s job log: %s\n' "$extension" "$latest_file" >&2
exec tail -f "$latest_file"

