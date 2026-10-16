#!/usr/bin/env bash
# Score every fit in fits.json, one row per fit, then rebuild the table.
#
#   tools/spiralfiteval/leaderboard/run_published.sh [id ...]
#
# No argument: all of them, in registry order (Paris 4 first, so a problem with the reference
# metrics shows in the first row). SKIP_EXISTING=1 keeps rows that are already there, so an
# interrupted run resumes. A row that fails is reported at the end and does not stop the rest.
# Each fit is scored in its own process, so the peak RSS in a row is that row's own.
# LOG=path (default run_published.log here) gets one line per row.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
LOG=${LOG:-$HERE/run_published.log}
mapfile -t IDS < <(if [ $# -gt 0 ]; then printf '%s\n' "$@"; else
  "$PY" -c 'import json,sys; print("\n".join(f["id"] for f in json.load(open(sys.argv[1]))["fits"]))' "$HERE/fits.json"
fi)
echo "$(date -Is)  ${#IDS[@]} rows, python $("$PY" -V)" | tee -a "$LOG"
failed=()
for id in "${IDS[@]}"; do
  if [ "${SKIP_EXISTING:-0}" = 1 ] && [ -f "$HERE/rows/$id.json" ]; then
    echo "$(date -Is)  skip $id (row exists)" | tee -a "$LOG"; continue
  fi
  echo "######## $(date -Is)  $id" | tee -a "$LOG"
  t0=$SECONDS
  if PYTHON="$PY" "$HERE/add_row.sh" --id "$id" --no-table >>"$LOG" 2>&1; then
    echo "$(date -Is)  done $id in $((SECONDS - t0)) s: $(tail -2 "$LOG" | head -1)" | tee -a "$LOG"
  else
    failed+=("$id")
    echo "$(date -Is)  FAILED $id after $((SECONDS - t0)) s (see $LOG)" | tee -a "$LOG"
  fi
done
"$PY" "$HERE/build_table.py" | tee -a "$LOG"
if [ ${#failed[@]} -gt 0 ]; then
  echo "$(date -Is)  ${#failed[@]} rows FAILED: ${failed[*]}" | tee -a "$LOG"
  exit 1
fi
echo "$(date -Is)  all ${#IDS[@]} rows scored" | tee -a "$LOG"
