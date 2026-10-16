#!/usr/bin/env bash
# One command to score a spiral fit and add its row. See README.md in this folder.
#   add_row.sh --id <id in fits.json>
#   add_row.sh --fit PATH --id my-id --author "Me" -- --references PHercParis4 --z 8400 9400 --floors 5
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$HERE/add_row.py" "$@"
