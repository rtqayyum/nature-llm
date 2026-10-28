#!/usr/bin/env bash
# Compatibility entry point. The replacement bootstrap uses Python, not Git.
set -euo pipefail
TRAFFIC_PROJECT_DIR="${1:-${TRAFFIC_PROJECT_ROOT:-$PWD}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/bootstrap_hpc.py" ]]; then
    exec python3 "$SCRIPT_DIR/bootstrap_hpc.py" --project "$TRAFFIC_PROJECT_DIR"
fi
mkdir -p "$TRAFFIC_PROJECT_DIR/work/bootstrap"
python3 - "$TRAFFIC_PROJECT_DIR/work/bootstrap/entry.py" <<'PY'
import sys
from urllib.request import urlretrieve
urlretrieve('https://raw.githubusercontent.com/rtqayyum/nature-llm/main/scripts/bootstrap_hpc.py', sys.argv[1])
PY
exec python3 "$TRAFFIC_PROJECT_DIR/work/bootstrap/entry.py" --project "$TRAFFIC_PROJECT_DIR"
