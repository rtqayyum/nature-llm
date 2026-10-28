#!/usr/bin/env python3
"""Acquire the pinned author release without silently substituting benchmark data.

This is an acquisition audit, not a forecasting/reproduction pass. No upstream
Python or checkpoint pickle is executed. Downloaded bytes stay in --work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import threading
import urllib.error
import urllib.parse
import urllib.request

from bootstrap_hpc import blob_sha, install_snapshot

PUBLIC = {
    "beijing_metro": ("beijing.npy", "bjfine-tuning.npy", 276, 15),
    "shanghai_metro": ("shanghai.npy", "shfine-tuning.npy", 288, 15),
    "hangzhou_metro": ("hangzhou.npy", "hzfine-tuning.npy", 80, 15),
    "pems03": ("03.npy", "03fine-tuning.npy", 358, 5),
    "pems04": ("04.npy", "04fine-tunning.npy", 307, 5),
    "pems08": ("08.npy", "08fine-tuning.npy", 170, 5),
    "nyc_bike": ("bike.npy", "bikefine-tuning.npy", 64, 60),
    "nyc_taxi": ("taxi.npy", "taxifine-tuning.npy", 64, 60),
}
USER_AGENT = "traffic-research-reproduction-audit/1.0"
THREAD_LOG = threading.local()


def emit(*values, **kwargs):
    kwargs["file"] = getattr(THREAD_LOG, "stream", sys.stdout)
    kwargs["flush"] = True
    print(*values, **kwargs)



def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inside(root, relative):
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts:
        raise ValueError("Unsafe relative asset path")
    destination = root / item
    destination.resolve().relative_to(root.resolve())
    return destination


def fingerprint(path):
    sha256 = hashlib.sha256()
    # Git blob SHA1 authenticates small Hugging Face files without LFS hashes.
    sha1 = hashlib.sha1(("blob " + str(path.stat().st_size) + "\0").encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            sha1.update(block)
    return {"size": path.stat().st_size, "sha256": sha256.hexdigest(), "git_blob_sha1": sha1.hexdigest()}


def verify(path, asset):
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return None
    actual = fingerprint(path)
    if asset.get("sha256"):
        return actual if actual["sha256"] == asset["sha256"] else None
    if asset.get("git_blob_sha1"):
        return actual if actual["git_blob_sha1"] == asset["git_blob_sha1"] else None
    raise ValueError("A pinned digest is required")


def request_json(url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def lfs_locations(asset, commit):
    """Fresh signed actions when supported; GitHub media as a public fallback."""
    batch = "https://github.com/LLM-UTP/LLM-UTP.git/info/lfs/objects/batch"
    payload = {"operation": "download", "transfers": ["basic"], "ref": {"name": commit},
               "objects": [{"oid": asset["sha256"], "size": asset["size"]}]}
    locations = []
    try:
        result = request_json(batch, json.dumps(payload).encode(),
                              {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
        action = result["objects"][0].get("actions", {}).get("download")
        if action:
            locations.append((action["href"], action.get("header", {})))
    except (OSError, ValueError, KeyError, IndexError):
        pass
    path = urllib.parse.quote(asset["path"], safe="/")
    locations.append(("https://media.githubusercontent.com/media/LLM-UTP/LLM-UTP/" + commit + "/" + path, {}))
    return locations


def transfer(url, destination, expected_size, headers=None):
    """Resume a partial object; restart if the server ignores Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected_size:
        partial.unlink()
        offset = 0
    request_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity", **(headers or {})}
    if offset:
        request_headers["Range"] = "bytes=" + str(offset) + "-"
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        response_status = response.getcode()
        if response_status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                raise ValueError("Invalid partial response range")
            mode = "ab" if offset else "wb"
        elif response_status == 200:
            offset, mode = 0, "wb"
        else:
            raise ValueError("Unexpected download response")
        progress_at = time.monotonic()
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                offset += len(chunk)
                if offset > expected_size:
                    raise ValueError("Response is larger than pinned object size")
                if time.monotonic() - progress_at > 15:
                    emit("  downloaded %.1f / %.1f MiB" % (offset / 2**20, expected_size / 2**20), flush=True)
                    progress_at = time.monotonic()
    if offset != expected_size:
        raise OSError("Incomplete object; retained partial download for retry")
    return partial


def download_verified(asset, destination, locations, retries=4):
    actual = verify(destination, asset)
    if actual:
        return {**actual, "status": "VERIFIED_EXISTING"}
    for attempt in range(retries):
        for url, headers in locations():
            try:
                partial = transfer(url, destination, asset["size"], headers)
                actual = verify(partial, asset)
                if not actual:
                    partial.unlink(missing_ok=True)
                    raise ValueError("Downloaded object digest does not match the pinned release")
                partial.replace(destination)
                return {**actual, "status": "DOWNLOADED_VERIFIED"}
            except (OSError, ValueError) as exc:
                # Do not log signed URLs, action headers or credentials.
                if isinstance(exc, urllib.error.HTTPError):
                    reason = "HTTP " + str(exc.code)
                else:
                    reason = type(exc).__name__
                emit("  retryable download failure:", reason, flush=True)
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("All verified download attempts failed for " + asset["path"])


def prepare_reference(work, lock, audit_only=False):
    # Git object identifiers are pinned, so no Git executable or API lookup
    # is needed to verify the author reference source archive.
    commit = lock["reference_commit"]
    files = lock["reference_files"]
    by_path = {item["path"]: item for item in files}
    for asset in lock["lfs_files"]:
        pointer = ("version https://git-lfs.github.com/spec/v1\n"
                   "oid sha256:" + asset["sha256"] + "\nsize " + str(asset["size"]) + "\n").encode()
        if blob_sha(pointer) != by_path[asset["path"]]["sha"]:
            raise ValueError("LFS inventory differs from pinned original source: " + asset["path"])
    repo = work / "reference" / "snapshots" / commit
    return install_snapshot("LLM-UTP/LLM-UTP", commit, files, repo,
                            mutable={item["path"] for item in lock["lfs_files"]}, audit_only=audit_only)


def audit_array(path, expected_nodes):
    import numpy as np
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("Not a complete NumPy .npy file")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype.hasobject or values.dtype.kind not in "iuf":
        raise ValueError("Expected a numeric traffic array")
    if values.ndim != 3 or values.shape[0] != expected_nodes or values.shape[1] < 13 or values.shape[2] < 1:
        raise ValueError("Unexpected node/time/feature layout: " + str(values.shape))
    finite_count = zero_count = negative_count = 0
    lower, upper = None, None
    for start in range(0, values.shape[1], 1024):
        block = values[:, start:start + 1024, :]
        valid = np.isfinite(block)
        finite_count += int(valid.sum())
        zero_count += int((valid & (block == 0)).sum())
        negative_count += int((valid & (block < 0)).sum())
        if valid.any():
            block_min, block_max = float(block[valid].min()), float(block[valid].max())
            lower = block_min if lower is None else min(lower, block_min)
            upper = block_max if upper is None else max(upper, block_max)
    return {"shape": list(values.shape), "dtype": str(values.dtype), "axis_order": "node,time,feature (author loader)",
            "elements": int(values.size), "nonfinite_elements": int(values.size) - finite_count,
            "zero_elements": zero_count, "negative_elements": negative_count, "minimum": lower, "maximum": upper,
            "timestamp_status": "NOT_VERIFIED: no calendar is reconstructed from row indices",
            "feature_status": "Original channels retained; raw/denoised channel provenance requires review"}


def build_report(work, report, repo, lock, assets, failures):
    datasets = []
    for name, (stage1, stage2, nodes, minutes) in PUBLIC.items():
        for stage, filename in (("stage1", stage1), ("stage2", stage2)):
            relative = "data/" + stage + "/" + filename
            row = {"dataset": name, "stage": stage, "path": str(repo / relative), "expected_nodes": nodes,
                   "native_interval_minutes_from_protocol": minutes, "status": "UNAVAILABLE"}
            asset = assets.get(relative)
            if asset and asset["status"] in ("VERIFIED_EXISTING", "DOWNLOADED_VERIFIED"):
                row["sha256"] = asset["sha256"]
                try:
                    row.update(audit_array(repo / relative, nodes))
                    row["status"] = "ARRAY_AUDITED"
                except Exception as exc:
                    row["status"] = "ARRAY_AUDIT_FAILED"
                    row["error"] = str(exc)
            datasets.append(row)
    duplicates = []
    for stage in ("stage1", "stage2"):
        groups = {}
        for row in datasets:
            if row.get("sha256"):
                groups.setdefault(row["sha256"], []).append(row["dataset"])
        # Keep only distinct datasets within this stage, not source/fine-tune reuse.
        for sha, names in groups.items():
            members = [r["dataset"] for r in datasets if r["stage"] == stage and r.get("sha256") == sha]
            if len(members) > 1:
                duplicates.append({"stage": stage, "datasets": members, "sha256": sha})
    restricted = [{"dataset": name, "full_archive": "NOT_PUBLICLY_RELEASED", "example_only": example,
                   "eligible_as_full_training_domain": False}
                  for name, example in (("nanning_metro", "data/example_nn.npy"),
                                        ("beijing_bus", "data/example_bus.npy"),
                                        ("beijing_shanghai_railway", "data/example_hsr.npy"))]
    blockers = ["Dataset timestamps, channel semantics, service gaps and publication split origins require reconciliation.",
                "Checkpoint S1/S2/S3 identity is unresolved; directory names alone are insufficient.",
                "Original GPT-2 initialization identity must be reconciled with the official base weights.",
                "The original eleven-domain joint retrain requires three restricted full archives.",
                "No forecasting inference or training reproduction has been attempted."]
    if duplicates:
        blockers.insert(0, "Distinct public domains have identical released payloads; Public8 training is blocked pending resolution.")
    if failures:
        blockers.insert(0, "Some requested files failed acquisition or integrity verification.")
    bad_arrays = [r for r in datasets if r["status"] != "ARRAY_AUDITED" or r.get("nonfinite_elements", 0)]
    if bad_arrays:
        blockers.insert(0, "One or more required traffic arrays are missing, invalid or contain nonfinite values.")
    summary = {"created_utc": now(), "reference_commit": lock["reference_commit"],
               "download_status": "INCOMPLETE" if failures else "ALL_REQUESTED_ASSETS_VERIFIED",
               "scientific_status": "REPRODUCTION_NOT_ESTABLISHED", "datasets": datasets,
               "duplicate_domains": duplicates, "restricted_datasets": restricted,
               "blockers": blockers, "failures": failures, "assets": assets,
               "training_started": False, "gpu_required": False}
    atomic_json(report / "acquisition_audit.json", summary)
    lines = ["# Step 1 acquisition audit", "", "Download status: " + summary["download_status"],
             "Scientific status: REPRODUCTION_NOT_ESTABLISHED", "", "| Dataset | Stage | Shape | Status |",
             "|---|---|---|---|"]
    for row in datasets:
        lines.append("| %s | %s | %s | %s |" % (row["dataset"], row["stage"], row.get("shape", "unknown"), row["status"]))
    lines.extend(["", "## Findings and next-step blockers", ""] + ["- " + text for text in blockers])
    if duplicates:
        lines.extend(["", "## Identical payloads", ""] + ["- " + x["stage"] + ": " + ", ".join(x["datasets"]) for x in duplicates])
    if failures:
        lines.extend(["", "## Acquisition failures", ""] + ["- " + x["path"] + ": " + x["error"] for x in failures])
    lines.extend(["", "All data, checkpoints and GPT-2 files are on the HPC under: " + str(work),
                  "Restricted example arrays are not full datasets. No training was run.",
                  "Rerun the same command to verify existing files and resume failed downloads.",
                  "Next: resolve release discrepancies, identify PEMS08 checkpoint strategy and reproduce checkpoint inference.", ""])
    (report / "SUMMARY.md").write_text("\n".join(lines))
    # Exit 3 keeps successful download distinct from scientific readiness.
    return 2 if failures or bad_arrays else 3 if duplicates else 0


def main():
    parser = argparse.ArgumentParser()
    for name in ("project", "work", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("TRAFFIC_DOWNLOAD_WORKERS", "1")))
    args = parser.parse_args()
    lock = json.loads((args.project / "config" / "sources.lock.json").read_text())
    repo = prepare_reference(args.work, lock, args.audit_only)
    atomic_json(args.report / "sources.lock.json", lock)
    gpt_dir = args.work / "models" / "gpt2" / lock["gpt2"]["revision"]
    planned = [(a, inside(repo, a["path"]), "lfs") for a in lock["lfs_files"]]
    planned += [(a, inside(gpt_dir, a["path"]), "gpt2") for a in lock["gpt2"]["files"]]
    remaining = sum(a["size"] for a, path, _ in planned if not path.is_file() or path.stat().st_size != a["size"])
    if not args.audit_only and shutil.disk_usage(args.work).free < remaining + 512 * 2**20:
        raise RuntimeError("Insufficient free filesystem space; approximately %.2f GiB plus quota headroom is needed" % (remaining / 2**30))
    emit("Pinned files:", len(planned), "Total expected %.2f GiB" % (sum(a["size"] for a, _, _ in planned) / 2**30), flush=True)
    if not 1 <= args.workers <= 40:
        raise ValueError("Download worker count must be between 1 and 40")
    assets, failures = {}, []
    worker_dir = Path(os.environ.get("TRAFFIC_WORKER_LOG_DIR", str(args.report / "workers")))
    worker_dir.mkdir(parents=True, exist_ok=True)
    def perform(index, item):
        asset, destination, kind = item
        worker_log = worker_dir / ("asset-%02d.log" % index)
        with worker_log.open("a") as stream:
            THREAD_LOG.stream = stream
            try:
                label, outcome = perform_asset(asset, destination, kind)
                outcome["worker_log"] = str(worker_log)
                return label, outcome
            finally:
                del THREAD_LOG.stream

    def perform_asset(asset, destination, kind):
        label = asset["path"] if kind == "lfs" else "gpt2/" + asset["path"]
        emit("\nASSET", label, flush=True)
        try:
            if args.audit_only:
                actual = verify(destination, asset)
                if not actual:
                    raise ValueError("Missing or digest-mismatched object")
                result = {**actual, "status": "VERIFIED_EXISTING"}
            else:
                if kind == "lfs":
                    locations = lambda a=asset: lfs_locations(a, lock["reference_commit"])
                else:
                    url = "https://huggingface.co/" + lock["gpt2"]["repository"] + "/resolve/" + lock["gpt2"]["revision"] + "/" + urllib.parse.quote(asset["path"])
                    locations = lambda u=url: [(u, {})]
                result = download_verified(asset, destination, locations)
            outcome = {**result, "local_path": str(destination), "verified_utc": now(),
                             "checkpoint_strategy": "UNRESOLVED" if label.endswith(".pth") else None}
            emit(result["status"], flush=True)
        except Exception as exc:
            outcome = {"status": "FAILED", "local_path": str(destination), "error": str(exc)}
            emit("FAILED:", label, "reason:", str(exc), "(continuing with other requested files)", flush=True)
        return label, outcome

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(perform, index, item): item for index, item in enumerate(planned)}
        for future in as_completed(futures):
            label, outcome = future.result()
            assets[label] = outcome
            if outcome["status"] == "FAILED":
                failures.append({"path": label, "error": outcome["error"], "worker_log": outcome["worker_log"]})
            progress = {"assets": assets, "failures": failures, "total_assets": len(planned),
                        "completed_assets": len(assets), "verified_assets": len(assets) - len(failures),
                        "failed_assets": len(failures), "workers": args.workers,
                        "verified_bytes": sum(value.get("size", 0) for value in assets.values()),
                        "expected_bytes": sum(item[0]["size"] for item in planned)}
            atomic_json(args.report / "asset_progress.json", progress)
            emit("PROGRESS files=%d/%d verified=%d failed=%d latest=%s status=%s" %
                 (len(assets), len(planned), progress["verified_assets"], len(failures), label, outcome["status"]))
    # Preserve the archive metadata, but do not mistake its pointer-only ZIP for full payloads.
    if not args.audit_only:
        try:
            metadata = request_json("https://zenodo.org/api/records/" + lock["zenodo_record"])
            atomic_json(args.report / "zenodo_metadata.json", metadata)
        except Exception:
            (args.report / "zenodo_metadata_unavailable.txt").write_text("Metadata endpoint unavailable; exact Git commit and payload digests are retained.\n")
    return build_report(args.work, args.report, repo, lock, assets, failures)


if __name__ == "__main__":
    sys.exit(main())
