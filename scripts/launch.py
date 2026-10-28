#!/usr/bin/env python3
"""Launch a detached worker and stream its log; disconnection leaves it running."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    project = Path(__file__).resolve().parents[1]
    work = Path(os.environ.get("TRAFFIC_WORK_ROOT", str(project / "work"))).expanduser().resolve()
    if work == project or project in work.parents and work != project / "work":
        # Default work/ is ignored; arbitrary source subdirectories are not.
        if work != project / "work":
            raise SystemExit("TRAFFIC_WORK_ROOT must be outside the code tree, or its default work/ directory.")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())
    report = work / "reports" / run_id
    report.mkdir(parents=True)
    log = work / "logs" / (run_id + ".log")
    command = [sys.executable, "-u", str(project / "scripts" / "pipeline.py"),
               "--project", str(project), "--work", str(work), "--report", str(report)] + sys.argv[1:]
    with log.open("wb") as stream:
        child = subprocess.Popen(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    (report / "launch.json").write_text(json.dumps({"pid": child.pid, "log": str(log), "command": command}, indent=2))
    print("HPC work directory:", work, flush=True)
    print("Log:", log, flush=True)
    print("Report:", report / "SUMMARY.md", flush=True)
    print("Worker PID:", child.pid, "(continues if SSH disconnects)", flush=True)
    try:
        with log.open() as stream:
            while True:
                chunk = stream.read()
                if chunk:
                    print(chunk, end="", flush=True)
                result = child.poll()
                if result is not None:
                    print(stream.read(), end="", flush=True)
                    print("Worker exit status:", result, flush=True)
                    return result
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped viewing the log. The detached worker is still running.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
