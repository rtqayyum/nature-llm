#!/usr/bin/env python3
"""Install verified GitHub source snapshots and run on HPC, without Git.

This file is deliberately self-contained so an older HPC checkout can bootstrap
it with Python's standard library. Code snapshots are immutable; datasets and
reports keep their existing work directory. No system software is installed.
"""
import argparse
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

REPOSITORY = "rtqayyum/nature-llm"
MARKER = ".source_snapshot.json"


def github_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "nature-llm-hpc/2", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def resolve_source(repository, revision="main"):
    result = github_json("https://api.github.com/repos/" + repository + "/commits/" + revision)
    commit = result["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("GitHub returned an invalid source commit")
    tree = github_json("https://api.github.com/repos/" + repository + "/git/trees/" + result["commit"]["tree"]["sha"] + "?recursive=1")
    if tree.get("truncated"):
        raise ValueError("GitHub file inventory is truncated; stopped without installing code")
    files = [{"path": item["path"], "size": item["size"], "sha": item["sha"], "mode": item["mode"]}
             for item in tree["tree"] if item["type"] == "blob"]
    if any(item["type"] not in ("blob", "tree") for item in tree["tree"]):
        raise ValueError("Source submodules are unsupported")
    return commit, files


def blob_sha(data):
    return hashlib.sha1(("blob " + str(len(data)) + "\0").encode() + data).hexdigest()


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts or "\\" in relative:
        raise ValueError("Unsafe archive member path")
    target = root.joinpath(*name.parts)
    target.resolve().relative_to(root.resolve())
    return target


def verify_snapshot(destination, repository, commit, files, mutable=()):
    """Check immutable source; payload verification is performed by acquire.py."""
    marker = json.loads((destination / MARKER).read_text())
    if marker["repository"] != repository or marker["commit"] != commit:
        raise ValueError("Existing source snapshot identity differs from the requested commit")
    expected = {item["path"]: item for item in files}
    recorded = {item["path"]: item for item in marker["files"]}
    if expected != recorded:
        raise ValueError("Existing snapshot inventory differs from pinned sources")
    for name, item in expected.items():
        target = safe_path(destination, name)
        if name in mutable:
            continue
        if not target.is_file() or target.is_symlink() or target.stat().st_size != item["size"]:
            raise ValueError("Missing or modified source file: " + name)
        if blob_sha(target.read_bytes()) != item["sha"]:
            raise ValueError("Source file digest differs: " + name)
    return marker


def install_snapshot(repository, commit, files, destination, mutable=(), audit_only=False, archive_bytes=None):
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("A full immutable commit SHA is required")
    if destination.exists():
        verify_snapshot(destination, repository, commit, files, mutable)
        return destination
    if audit_only:
        raise ValueError("Pinned reference snapshot is missing; run acquisition first")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if archive_bytes is None:
        request = urllib.request.Request("https://codeload.github.com/" + repository + "/zip/" + commit,
                                         headers={"User-Agent": "nature-llm-hpc/2"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive_bytes = response.read(64 * 2**20 + 1)
        if len(archive_bytes) > 64 * 2**20:
            raise ValueError("Source archive exceeds 64 MiB; no code was installed")
    expected = {item["path"]: item for item in files}
    with tempfile.TemporaryDirectory(prefix="snapshot-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        seen = set()
        roots = set()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            if sum(member.file_size for member in archive.infolist()) > 128 * 2**20:
                raise ValueError("Expanded source archive exceeds 128 MiB")
            for member in archive.infolist():
                parts = PurePosixPath(member.filename).parts
                if not parts or PurePosixPath(member.filename).is_absolute() or ".." in parts:
                    raise ValueError("Unsafe archive path")
                roots.add(parts[0])
                if member.is_dir():
                    continue
                if len(parts) < 2 or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Archive symlink or root-level file is unsupported")
                name = "/".join(parts[1:])
                if name in seen or name not in expected:
                    raise ValueError("Duplicate or unexpected archive member: " + name)
                seen.add(name)
                item = expected[name]
                if item.get("mode", "100644") not in ("100644", "100755") or member.file_size != item["size"]:
                    raise ValueError("Archive file mode/size differs from source inventory: " + name)
                data = archive.read(member)
                if blob_sha(data) != item["sha"]:
                    raise ValueError("Archive file digest differs: " + name)
                target = safe_path(staging, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o755 if item.get("mode") == "100755" else 0o644)
        if len(roots) != 1 or seen != set(expected):
            raise ValueError("Archive does not match the complete pinned source inventory")
        marker = {"repository": repository, "commit": commit, "files": files,
                  "installed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}
        (staging / MARKER).write_text(json.dumps(marker, indent=2) + "\n")
        staging.rename(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description="Update and run nature-llm on HPC without a Git executable")
    default_project = os.environ.get("TRAFFIC_PROJECT_ROOT", str(Path.home() / "nature-llm"))
    parser.add_argument("--project", type=Path, default=Path(default_project))
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("TRAFFIC_WORK_ROOT", str(project / "work"))).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / "acquisition.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Acquisition is running. Let it finish before updating code.")
        print("Checking GitHub for the latest committed code (no Git executable needed)...", flush=True)
        commit, files = resolve_source(REPOSITORY)
        snapshot = install_snapshot(REPOSITORY, commit, files, work / "code_versions" / commit)
        (work / "CURRENT_CODE.json").write_text(json.dumps({"commit": commit, "path": str(snapshot)}, indent=2) + "\n")
        print("Verified source commit:", commit, flush=True)
        print("Code snapshot:", snapshot, flush=True)
        print("Existing data/report directory:", work, flush=True)
    if args.sync_only:
        return 0
    env = {**os.environ, "TRAFFIC_PROJECT_ROOT": str(project), "TRAFFIC_WORK_ROOT": str(work)}
    # Newest code runs from its own verified version; the old checkout is preserved.
    command = ["bash", str(snapshot / "run.sh")]
    if args.audit_only:
        command.append("--audit-only")
    return subprocess.call(command, cwd=snapshot, env=env)


def cli(argv=None):
    try:
        return main(argv)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print("GitHub setup failed:", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(cli())
