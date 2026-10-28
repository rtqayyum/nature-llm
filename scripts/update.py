#!/usr/bin/env python3
"""Fast-forward code only, and refuse updates while the acquisition is running."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    if not shutil.which("git"):
        raise SystemExit("Git is not on PATH. Run bash update_and_run.sh to try the HPC Git module, or configure config/hpc.local.sh.")
    project = Path(__file__).resolve().parents[1]
    work = Path(os.environ.get("TRAFFIC_WORK_ROOT", str(project / "work"))).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / "acquisition.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An acquisition is running. Let it finish before updating code.")
        state = subprocess.check_output(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"], text=True)
        if state.strip():
            raise SystemExit("Tracked code has local edits; update stopped without changing them.")
        subprocess.run(["git", "-C", str(project), "pull", "--ff-only"], check=True)
    if sys.argv[1:] == ["--sync-only"]:
        return 0
    return subprocess.call(["bash", str(project / "run.sh")] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
