#!/usr/bin/env bash
# Fetch our latest code without discarding local edits or touching downloaded data.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ -f config/hpc.local.sh ]]; then
    source config/hpc.local.sh
fi
source scripts/hpc_bootstrap.sh
traffic_require_git || exit 2
for candidate in python3.11 python3.10 python3.12 python3.9 python3.8 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,8) <= sys.version_info[:2] <= (3,12)' 2>/dev/null; then
        exec "$candidate" -u scripts/update.py "$@"
    fi
done
echo 'Load Python 3.8–3.12 in config/hpc.local.sh and rerun.' >&2
exit 2
