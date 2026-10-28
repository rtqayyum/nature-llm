#!/usr/bin/env bash
# Step 1 only: CPU acquisition and audit. No training or GPU allocation.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ -f config/hpc.local.sh ]]; then
    # Optional site-specific module loads / TRAFFIC_WORK_ROOT configuration.
    source config/hpc.local.sh
fi
for candidate in python3.11 python3.10 python3.12 python3.9 python3.8 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,8) <= sys.version_info[:2] <= (3,12)' 2>/dev/null; then
        exec "$candidate" -u scripts/launch.py "$@"
    fi
done
echo 'Python 3.8–3.12 is required. Load an available Python module in config/hpc.local.sh and rerun.' >&2
exit 2
