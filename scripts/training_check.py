"""Read live per-seed reports and write an aggregate without touching training."""
import datetime
import json
from pathlib import Path
import statistics
import tempfile

from slurm_submit import seconds


def summarize(project, group_id, submission, command):
    group = project / "results/slurm_jobs" / group_id
    print("Reference training array", group_id, "tasks=", len(submission["array_tasks"]))
    accounting = {}
    for arguments in (["squeue", "--jobs", group_id, "--array", "--noheader", "--format=%i|%T|%R"],
        ["sacct", "-j", group_id, "--parsable2", "--noheader", "--format=JobID%40,JobName,State%30,ExitCode,Elapsed,AllocCPUS,TotalCPU,MaxRSS,Reason"]):
        response = command(arguments)
        print("\n" + " ".join(arguments) + "\n" + (response["stdout"] or response["stderr"] or "No current records"))
        if arguments[0] == "sacct":
            for line in response["stdout"].splitlines():
                fields = line.split("|")
                if len(fields) >= 9 and "." not in fields[0]:
                    accounting[fields[0]] = {"state": fields[2], "exit_code": fields[3], "reason": fields[8]}
                    try:
                        elapsed, cpus = seconds(fields[4]), int(fields[5])
                        whole, _, fraction = fields[6].partition(".")
                        cpu = seconds(whole) + (float("0." + fraction) if fraction else 0)
                        if elapsed and cpus:
                            accounting[fields[0]]["cpu_capacity_used_percent"] = round(100 * cpu / (elapsed * cpus), 2)
                    except (ValueError, TypeError):
                        pass
    rows, results, failures = [], {}, 0
    for index, task in enumerate(submission["array_tasks"]):
        key = group_id + "_" + str(index)
        report = project / "results/slurm_jobs" / key / "report"
        path = report / "training_progress.json"
        progress = json.loads(path.read_text()) if path.exists() else {"phase": "QUEUED_OR_SETUP"}
        result_path = report / "training_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        scheduler = accounting.get(key, {})
        failed = progress["phase"] == "FAILED" or any(scheduler.get("state", "").startswith(value)
            for value in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"))
        failures += failed
        if result:
            results.setdefault(task["dataset"], []).append(result)
        row = {"task_id": key, **task, "progress": progress, "scheduler": scheduler, "result": result,
            "stdout": str(project / "logs" / (submission["job_name"] + "_" + key + ".out")),
            "stderr": str(project / "logs" / (submission["job_name"] + "_" + key + ".err"))}
        rows.append(row)
        started_path = report.parent / "started.json"
        if scheduler.get("state") == "RUNNING" and started_path.exists():
            raw_id = str(json.loads(started_path.read_text()).get("actual_slurm_job_id", ""))
            if raw_id.isdigit():
                usage = command(["sstat", "--allsteps", "-j", raw_id, "--format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize"])
                print("LIVE CPU/MEMORY", key, "\n" + (usage["stdout"] or usage["stderr"]))
        print("TASK", key, task["dataset"], "seed=", task["seed"], progress["phase"],
            "epoch=%s/%s batch=%s/%s updates=%s" % (progress.get("epoch", 0), progress.get("max_epochs", "?"),
                progress.get("batch", 0), progress.get("train_batches", "?"), progress.get("optimizer_steps", 0)),
            "scheduler=", scheduler, "best=", progress.get("best"), "last=", progress.get("latest_epoch", progress.get("running_train_mse")))
        print("  Log:", row["stdout"])
        if progress.get("output_store"):
            print("  Checkpoints/history:", progress["output_store"])
        if progress.get("error"):
            print("  FAILURE:", progress["error"], "traceback:", report / "failure_traceback.txt")
        if result:
            print("  TEST:", json.dumps(result["metrics_author"]))
        for suffix in ("out", "err"):
            log = Path(row["stdout" if suffix == "out" else "stderr"])
            if log.exists():
                print("  RECENT", suffix, "\n" + "\n".join(log.read_text(errors="replace").splitlines()[-3:]))
    aggregates = {}
    for dataset, values in results.items():
        aggregates[dataset] = {"completed_seeds": len(values), "expected_seeds": len(submission["array_tasks"]) // len({t["dataset"] for t in submission["array_tasks"]}),
            "metrics": {name: {"mean": statistics.mean(row["metrics_author"][name] for row in values),
                "sample_std": statistics.stdev(row["metrics_author"][name] for row in values) if len(values) > 1 else None}
                for name in ("MAE", "RMSE", "WMAPE_percent")}}
    summary = {"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "group_id": group_id, "completed_experiments": sum(map(len, results.values())), "total_experiments": len(rows),
        "reported_failures": failures, "tasks": rows, "aggregates": aggregates,
        "interpretation": "Partial seed aggregates are provisional; Slurm accounting is authoritative for timeout/cancellation failures"}
    output = group / "report"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output, prefix="summary-", suffix=".tmp", delete=False) as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
        temporary = Path(stream.name)
    temporary.replace(output / "TRAINING_SUMMARY.json")
    lines = ["# Three-seed single-domain reference training", "", "Completed %s/%s experiments; reported application failures %s." %
        (summary["completed_experiments"], len(rows), failures), "", "| Task | Dataset | Seed | Phase | Epoch | Updates |", "|---|---|---:|---|---:|---:|"]
    for row in rows:
        p = row["progress"]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (row["task_id"], row["dataset"], row["seed"], p["phase"], p.get("epoch", 0), p.get("optimizer_steps", 0)))
    lines += ["", "Provisional completed-seed aggregates:", "", "```json", json.dumps(aggregates, indent=2), "```", "", summary["interpretation"],
        "Released scaling and cross-window dependence retained. Exact published S3 settings unresolved."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\nEXPERIMENTS completed=%s/%s reported_failed=%s" % (summary["completed_experiments"], len(rows), failures))
    print("AGGREGATES:", json.dumps(aggregates, indent=2))
    print("Notifications:", json.dumps(submission.get("notifications", {})), "queued receipt:", json.dumps(submission.get("queued_email", {})))
    print("Reports:", output / "SUMMARY.md", output / "TRAINING_SUMMARY.json")
    return summary
