#!/usr/bin/env python3
"""Update to a verified immutable GitHub snapshot without a Git executable."""
import os
from pathlib import Path
import sys

from bootstrap_hpc import cli


if __name__ == "__main__":
    project = os.environ.get("TRAFFIC_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    sys.exit(cli(["--project", project] + sys.argv[1:]))
