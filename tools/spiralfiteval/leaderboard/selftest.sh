#!/usr/bin/env bash
# Check the "add your fit" command end to end on a synthetic fit, in a temporary rows folder,
# so nothing in the leaderboard is touched. About 10 s, no download, no CT.
#   tools/spiralfiteval/leaderboard/selftest.sh
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PY=${PYTHON:-python3}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
"$PY" -c "
import sys; sys.path.insert(0, '$REPO/tools/spiralfiteval/tests')
from synth import write_fit; write_fit('$TMP/fit')"
export LEADERBOARD_ROWS=$TMP/rows
PYTHON="$PY" "$HERE/add_row.sh" --fit "$TMP/fit" --id selftest --author "self-test" \
    --scroll-name "synthetic spiral" --no-table -- --floors 2 >/dev/null
"$PY" - "$TMP/rows/selftest.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
h = r["headline"]
assert r["id"] == "selftest" and r["author"] == "self-test", r["id"]
assert r["input_hash"]["kind"] == "manifest" and len(r["input_hash"]["sha256"]) == 64
assert r["cost"]["peak_rss_gb"] and r["cost"]["peak_rss_gb"] < 16, r["cost"]
assert h["H5_duplicate"] == 0.0, h          # a clean synthetic spiral has no duplicates
assert 19.5 <= h["H5_median_gap_vox"] <= 20.5, h   # the synthetic pitch is 20 voxels
assert r["floors"] == {}, r["floors"]       # no references: nothing that could sit at a floor
print("selftest: row written, hash, cost and H5 as expected")
PY
