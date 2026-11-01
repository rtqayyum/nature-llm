#!/usr/bin/env python3
"""Consolidate registered HPC evidence without training, inference or rescoring.

Only the Python standard library is needed. Scientific limitations are not
execution failures; missing/corrupt required evidence is an execution failure.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import traceback

from reference_datasets import PROFILES, PUBLIC_ORDER

METRICS = ("MAE", "RMSE", "WMAPE_percent")
SUCCESS = ("COMPLETED", "COMPLETED_WITH_REVIEW")


def contained(root, value):
    path = Path(value).resolve()
    path.relative_to(root)
    return path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def metrics(value):
    if not all(type(value.get(key)) in (int, float) and math.isfinite(value[key]) and value[key] >= 0 for key in METRICS):
        raise ValueError("Missing, negative or nonfinite metrics")
    return {key: value[key] for key in METRICS}


def agrees(left, right):
    return all(math.isclose(left[key], right[key], rel_tol=1e-5, abs_tol=1e-6) for key in METRICS)


def compare(measured, published):
    return {key: {"published": published[key], "observed": measured[key],
        "observed_minus_published": measured[key] - published[key],
        "same_two_decimal_rounding": format(measured[key], ".2f") == format(published[key], ".2f")}
        for key in METRICS}


class Evidence:
    def __init__(self, root):
        self.root = root
        self.files = {}

    def path(self, value):
        return contained(self.root, value)

    def remember(self, value, expected=None):
        path = self.path(value)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError("Evidence hash mismatch: " + str(path))
        previous = self.files.get(str(path))
        if previous and previous["sha256"] != actual:
            raise ValueError("Evidence changed while being read: " + str(path))
        self.files[str(path)] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
        return path

    def read(self, value):
        path = self.remember(value)
        return json.loads(path.read_text())

    def job(self, job_id, stage, terminal=True):
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            raise ValueError("Invalid project job ID")
        directory = self.path(self.root / "results/slurm_jobs" / job_id)
        submission = self.read(directory / "submission.json")
        if submission["job_id"] != job_id or submission["stage"] != stage or self.path(submission["project"]) != self.root:
            raise ValueError("Job registration mismatch: " + job_id)
        if terminal:
            status = self.read(directory / "job_status.json")
            if status["job_id"] != job_id or status.get("stage") != stage or status["shell_exit_code"] != 0 or status["execution_state"] not in SUCCESS:
                raise ValueError("Required job did not finish successfully: " + job_id)
        return directory / "report", submission


def checkpoint_rows(evidence, report, submission, legacy, lock):
    summary = evidence.read(report / "public8_result.json")
    if summary["execution_status"] not in SUCCESS or summary["completed_datasets"] != 8 or summary["failed_datasets"] != 0:
        raise ValueError("Public checkpoint evaluation is incomplete")
    records = summary["datasets"]
    if len(records) != 8 or {row["dataset"] for row in records} != set(PUBLIC_ORDER):
        raise ValueError("Missing or duplicate public dataset results")
    by_key = {row["dataset"]: row for row in records}
    rows = []
    for key in PUBLIC_ORDER:
        profile = PROFILES[key]
        result = evidence.read(report / "datasets" / profile["slug"] / "inference_result.json")
        if result["dataset"] != key or result["execution_status"] not in SUCCESS or result["strict_load_performed"] is not True or result["training_performed"] is not False:
            raise ValueError("Checkpoint result identity/strict-load mismatch: " + key)
        measured = metrics(result["metrics_author"])
        if not agrees(measured, metrics(result["metrics_independent"])) or by_key[key]["metrics_author"] != result["metrics_author"]:
            raise ValueError("Metric implementation/aggregate mismatch: " + key)
        store = evidence.path(result["output_store"])
        identity = evidence.read(store / "identity.json")
        run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if run_hash != result["run_identity"]:
            raise ValueError("Checkpoint run identity mismatch")
        source_code = evidence.path(submission["code"])
        evidence.remember(source_code / "scripts/pems08_infer.py", identity["inference_source"])
        evidence.remember(source_code / "scripts/reference_contract.py", identity["contract_source"])
        legacy_schema = "evaluation_scope" not in result
        if legacy_schema:
            # This specific older worker has no variant arguments. Accept its
            # format only with its reviewed immutable code hash and snapshot.
            if source_code.name != legacy["source_commit"] or identity["inference_source"] != legacy["inference_source_sha256"]:
                raise ValueError("Unknown historical checkpoint schema/source")
        elif result["evaluation_scope"] != "FAITHFUL_RELEASE_DEFAULT" or result["input_stage"] != "stage2" or result["split_route"] != "finetune":
            raise ValueError("A diagnostic variant cannot replace the faithful checkpoint result")
        configuration = identity["config"]
        if (configuration["dataset"] != key or configuration["reference_commit"] != lock["reference_commit"]
                or configuration["gpt2_base_revision"] != lock["gpt2"]["revision"]):
            raise ValueError("Checkpoint source/dataset/base revision mismatch")
        for name in (profile["stage2"], profile["checkpoint"], "utils/metrics.py"):
            pinned = next(row["sha256"] for row in lock["lfs_files"] if row["path"] == name)
            if identity["assets"].get(name) != pinned:
                raise ValueError("Checkpoint input identity differs from pinned stage-two assets")
        evidence.remember(report / "datasets" / profile["slug"] / "reference_environment_freeze.txt", identity["environment_freeze"])
        if evidence.read(store / "completed.json") != result:
            raise ValueError("Checkpoint result differs between report and run store")
        evidence.remember(store / "forecasts.npz", result["forecast_file_sha256"])
        comparison = compare(measured, profile["anchors"]["S1"])
        rows.append({"dataset": key, "title": profile["title"], "metrics": measured,
            "scored_origins": result["scored_origins"], "comparison_to_published_S1": comparison,
            "rounding_matches": sum(row["same_two_decimal_rounding"] for row in comparison.values()),
            "legacy_schema_verified_by_pinned_worker": legacy_schema,
            "checkpoint_strategy": "UNRESOLVED", "full_reproduction_claim": False})
    return rows


def training_rows(evidence, array_id, submission, rescore_report):
    summary = evidence.read(rescore_report / "rescore_summary.json")
    tasks = submission["array_tasks"]
    expected = [(dataset, seed) for dataset in ("pems08", "bj") for seed in (11, 22, 33)]
    if [(row["dataset"], row["seed"]) for row in tasks] != expected or len(summary["tasks"]) != 6:
        raise ValueError("Expected the recorded six pilot experiments")
    if summary["source_array"] != array_id or summary["execution_status"] not in SUCCESS or summary["completed_tasks"] != 6 or summary["failed_tasks"] != 0:
        raise ValueError("Rescore does not certify all six source tasks")
    rescored = {row["source_task"]: row for row in summary["tasks"]}
    if set(rescored) != {array_id + "_" + str(i) for i in range(6)}:
        raise ValueError("Duplicate or missing corrected tasks")
    rows, contracts = [], []
    for index, task in enumerate(tasks):
        key = array_id + "_" + str(index)
        report, registered = evidence.job(key, "reference-train3")
        contract = evidence.read(report / "training_contract.json")
        original = evidence.read(report / "training_result.json")
        correction = evidence.read(rescore_report / "tasks" / key / "rescore_result.json")
        if correction != rescored[key] or correction != evidence.read(evidence.path(correction["output_store"]) / "rescore_result.json"):
            raise ValueError("Corrected result differs between report and run store")
        if correction["execution_status"] not in SUCCESS or correction["inverse_transform"] != "TRAINING_MINMAX_AFFINE_V1":
            raise ValueError("Missing verified affine correction")
        if any(row.get(field) != task[field] for row in (registered, contract, original, correction) for field in ("dataset", "seed")):
            raise ValueError("Training task dataset/seed mismatch")
        if contract["config"]["array"] != "stage1" or contract["config"]["route"] != "pretrain":
            raise ValueError("Training is not the recorded reference protocol")
        source = correction["source_inputs"]
        evidence.remember(report / "training_contract.json", source["source_contract_sha256"])
        evidence.remember(report / "training_result.json", source["source_result_sha256"])
        source_code = evidence.path(registered["code"])
        evidence.remember(source_code / "scripts/reference_train.py", contract["trainer_sha256"])
        for name, checksum in contract["helper_sha256"].items():
            evidence.remember(source_code / "scripts" / name, checksum)
        store = evidence.path(original["output_store"])
        if evidence.read(store / "identity.json") != contract or evidence.read(store / "training_result.json") != original:
            raise ValueError("Training result/contract differs between report and output store")
        evidence.remember(store / "training_result.json", evidence.read(store / "COMPLETED.json")["result_sha256"])
        evidence.remember(store / original["best"]["file"], source["selected_model_sha256"])
        for field in ("minimum", "maximum"):
            if correction[field] != contract["min" if field == "minimum" else "max"]:
                raise ValueError("Correction scaler differs from training")
        measured = metrics(correction["corrected_metrics"]["author"])
        if not agrees(measured, metrics(correction["corrected_metrics"]["independent"])):
            raise ValueError("Corrected metrics disagree")
        rows.append({"task": key, **task, "epochs": original["epochs"], "best_epoch": original["best"]["epoch"],
            "optimizer_steps": original["optimizer_steps"], "minimum": contract["min"], "maximum": contract["max"],
            "legacy_metrics": metrics(correction["legacy_metrics"]["author"]), "corrected_metrics": measured,
            "published_S3_diagnostic_only": compare(measured, PROFILES[task["dataset"]]["anchors"]["S3"]),
            "exact_S3_reproduction": False})
        contracts.append({"task": key, "config": contract["config"], "initialization": contract["initialization"],
            "reference_commit": contract["reference_commit"], "gpt2_revision": contract["gpt2_revision"],
            "splits": contract["splits"], "runtime": contract["runtime"]})
    aggregates = {}
    for dataset in ("pems08", "bj"):
        selected = [row for row in rows if row["dataset"] == dataset]
        aggregates[dataset] = {name: {"mean": statistics.mean(row["corrected_metrics"][name] for row in selected),
            "sample_std": statistics.stdev(row["corrected_metrics"][name] for row in selected)} for name in METRICS}
    return rows, contracts, aggregates


def protocol_findings():
    return [
        {"item": "S1", "paper": "Joint eleven-domain training, domain fine-tuning, evaluation",
         "evidence": "Eight released checkpoints evaluated; three restricted full datasets unavailable; stage-one bike/taxi payloads identical",
         "decision": "Full S1 retraining not established; rounded checkpoint agreement does not identify strategy"},
        {"item": "S2", "paper": "Joint training without the domain fine-tuning stage",
         "evidence": "No identified joint-stage checkpoint or reproduced eleven-domain training history",
         "decision": "Not reproduced; target checkpoint names cannot identify S2"},
        {"item": "S3", "paper": "Independent single-dataset training and testing",
         "evidence": "Six fresh stage1/pretrain fits, fixed three seeds, selected by validation; two domains only",
         "decision": "Independent released-code training; S3-specific array, route, freeze policy and original seed history unresolved"},
        {"item": "Parameter policy", "paper": "Joint stage opens LLM parameters; subsequent stage freezes attention",
         "evidence": "Released constructor retains six blocks and marks only their MLP parameters trainable; forward executes two blocks",
         "decision": "Source/paper stage-policy mismatch; do not silently unfreeze layers and call it faithful reproduction"},
        {"item": "Inverse scaling", "paper": "Physical-unit prediction metrics",
         "evidence": "Our historical evaluator used max-only inversion for a negative-minimum Beijing stage-one array; six saved runs rescored using recorded min/max",
         "decision": "Use corrected values for independent training, preserve historical reports; zero-minimum PEMS08 control unchanged"},
        {"item": "Causality and data", "paper": "One-step traffic forecasting",
         "evidence": "Full-array normalization and implicit fusion softmax retained in released behavior; supplied auxiliary-channel generation and absolute timestamps unresolved",
         "decision": "Document limitations; changing these creates a separate adaptation, not a paper reproduction"}
    ]


def write_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def run(project, code, report):
    project = project.resolve()
    code, report = contained(project, code), contained(project, report)
    report.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(project)
    progress = {"phase": "VERIFYING", "completed_sections": 0, "total_sections": 4,
        "training_performed": False, "model_inference_performed": False, "rescoring_performed": False}
    def update(**fields):
        progress.update(fields)
        atomic_json(report / "reproduction_progress.json", progress)
        print("REPRODUCTION_REVIEW", json.dumps(progress), flush=True)
    try:
        update()
        config = evidence.read(code / "config/reproduction_review.json")
        lock = evidence.read(code / "config/sources.lock.json")
        evidence.remember(Path(__file__))
        evidence.remember(code / "scripts/reference_datasets.py")
        sources = {}
        for name, job in config["required_jobs"].items():
            sources[name] = evidence.job(job["job_id"], job["stage"], terminal=name != "training")
        checkpoint = checkpoint_rows(evidence, *sources["checkpoints"], config["checkpoint_legacy_schema"], lock)
        update(completed_sections=1, latest="eight checkpoint results verified")
        diagnostic = evidence.read(sources["diagnostics"][0] / "diagnostic_result.json")
        if (diagnostic["execution_status"] not in SUCCESS or len(diagnostic["cases"]) != 10
                or any(row["state"] != "COMPLETED_WITH_REVIEW" for row in diagnostic["cases"].values())
                or any("error" in row for row in diagnostic["paired_comparisons"])):
            raise ValueError("Fixed diagnostic is incomplete")
        train, contracts, aggregates = training_rows(evidence, config["required_jobs"]["training"]["job_id"],
            sources["training"][1], sources["correction"][0])
        update(completed_sections=2, latest="six training and corrected result chains verified")
        # Verify the exact source revision rather than import or patch its model.
        snapshot = project / "work/reference/snapshots" / lock["reference_commit"]
        source_rows = []
        for name in ("main_run.py", "fine-tuning.py", "prediction.py", "model/LLM_UTP.py", "data/unified_dataset.py", "data/dataset_registry.py"):
            path = evidence.remember(snapshot / name)
            content = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            expected = next(row["sha"] for row in lock["reference_files"] if row["path"] == name)
            if blob != expected:
                raise ValueError("Pinned author source mismatch: " + name)
            source_rows.append({"path": name, "git_blob_sha1": blob, "sha256": digest(path)})
        for contract in contracts:
            if contract["reference_commit"] != lock["reference_commit"] or contract["gpt2_revision"] != lock["gpt2"]["revision"]:
                raise ValueError("Pilot source/base-model revision differs from the reviewed release")
        optional = {}
        for name, job in config["optional_jobs"].items():
            try:
                directory, _ = evidence.job(job["job_id"], job["stage"])
                item = evidence.read(directory / job["file"])
                if item["execution_status"] not in SUCCESS:
                    raise ValueError("Optional report did not complete")
                optional[name] = {"state": "REGISTERED_REPORT_AVAILABLE", "job_id": job["job_id"], "report": str(directory / job["file"]),
                    "scope": "Supporting prior report; numerical probes are not rerun by this consolidation"}
            except (OSError, ValueError, KeyError) as exc:
                optional[name] = {"state": "NOT_VERIFIED", "job_id": job["job_id"], "reason": str(exc)}
        update(completed_sections=3, latest="pinned source and optional evidence reviewed")
        findings = protocol_findings()
        summary = {"execution_status": "COMPLETED_WITH_REVIEW", "scientific_status": "PARTIAL_NUMERICAL_AGREEMENT_FULL_REPRODUCTION_NOT_ESTABLISHED",
            "training_performed": False, "model_inference_performed": False, "rescoring_performed": False,
            "paper": config["paper"], "paper_full_text": config["paper_full_text"], "sources": config["required_jobs"],
            "reference_commit": lock["reference_commit"], "source_files": source_rows,
            "checkpoint_datasets": checkpoint, "checkpoint_all_three_rounding_matches": sum(row["rounding_matches"] == 3 for row in checkpoint),
            "training_experiments": train, "training_contracts": contracts, "corrected_aggregates": aggregates,
            "fixed_diagnostics": {key: diagnostic[key] for key in ("cases", "aliases", "paired_comparisons", "selection_policy")},
            "protocol_findings": findings, "optional_evidence": optional,
            "verification_scope": "Registered completed jobs, metric cross-checks, linked result/contract/model/forecast hashes and pinned source. No numerical inference, training or rescore rerun. Historical reports are not independently attested Slurm accounting.",
            "diagnostic_policy": "All fixed input/split diagnostics retained in their source report; no test-score-based selection",
            "full_paper_reproduction": False, "causal_adaptation_on_hold": True, "proposed_method_started": False,
            "next_decision": "Review the reproducibility matrix. Resolve source identity through documented author evidence; otherwise freeze explicitly limited released-behavior reference results before separate method design. No blind accuracy search."}
        csv_rows = [{"dataset": row["dataset"], "metric": key, **value} for row in checkpoint for key, value in row["comparison_to_published_S1"].items()]
        write_csv(report / "checkpoint_comparison.csv", csv_rows)
        write_csv(report / "training_comparison.csv", [{"task": row["task"], "dataset": row["dataset"], "seed": row["seed"], "metric": key,
            "legacy": row["legacy_metrics"][key], "corrected": row["corrected_metrics"][key],
            "published_S3_diagnostic_only": row["published_S3_diagnostic_only"][key]["published"]} for row in train for key in METRICS])
        atomic_json(report / "reproduction_review.json", summary)
        lines = ["# Reference-paper reproducibility assessment", "", "Execution: COMPLETED_WITH_REVIEW. Full paper reproduction: NOT ESTABLISHED.",
            "", "This job consolidates historical real HPC results. It performs no training, inference, rescoring or protocol modification.",
            "", "## Released-checkpoint results versus published S1", "", "Numbers are MAE / RMSE / WMAPE %. Rounding matches do not identify training strategy.", "",
            "| Dataset | Published S1 | Observed checkpoint | Matches / 3 |", "|---|---|---|---:|"]
        for row in checkpoint:
            published = PROFILES[row["dataset"]]["anchors"]["S1"]
            lines.append("| %s | %s | %s | %d |" % (row["title"], " / ".join("%.2f" % published[k] for k in METRICS),
                " / ".join("%.6f" % row["metrics"][k] for k in METRICS), row["rounding_matches"]))
        lines += ["", "## Independent training: corrected three-seed results", "", "These are stage1/pretrain reference fits, not certified published S3. Mean +/- sample standard deviation.",
            "", "| Dataset | MAE | RMSE | WMAPE % |", "|---|---:|---:|---:|"]
        for dataset, measured in aggregates.items():
            lines.append("| %s | %s |" % (PROFILES[dataset]["title"], " | ".join("%.6f +/- %.6f" % (measured[k]["mean"], measured[k]["sample_std"]) for k in METRICS)))
        lines += ["", "## Protocol reconciliation", "", "| Item | Paper | Available evidence | Decision |", "|---|---|---|---|"]
        lines += ["| %s | %s | %s | %s |" % (row["item"], row["paper"], row["evidence"], row["decision"]) for row in findings]
        lines += ["", "## Evidence and next decision", "", summary["verification_scope"], "",
            "Optional supporting report availability:", "", "```json", json.dumps(optional, indent=2), "```", "",
            summary["next_decision"], "", "Causal adaptation and proposed-method training remain on hold. No existing job is cancelled.",
            "No restricted example arrays are substituted; no offsets, model layers or metric conventions are tuned to match the paper.",
            "Detailed per-seed comparisons: training_comparison.csv. Checkpoint metric differences: checkpoint_comparison.csv.",
            "Machine-readable assessment: reproduction_review.json. Read-input integrity ledger: evidence_manifest.json.",
            "Paper: " + config["paper_full_text"], "Source revision: " + lock["reference_commit"], ""]
        (report / "SUMMARY.md").write_text("\n".join(lines))
        # Detect modification of any source artifact during the complete review.
        for row in list(evidence.files.values()):
            evidence.remember(row["path"], row["sha256"])
        atomic_json(report / "evidence_manifest.json", {"files": list(evidence.files.values()), "source_files_preserved": True})
        update(phase="COMPLETED_WITH_REVIEW", completed_sections=4, latest="assessment saved", failed_sections=0)
        print("\n".join(lines), flush=True)
        return 3
    except Exception as exc:
        update(phase="FAILED", error=str(exc), failed_sections=1)
        atomic_json(report / "reproduction_review.json", {"execution_status": "FAILED", "error": str(exc),
            "training_performed": False, "model_inference_performed": False})
        atomic_json(report / "evidence_manifest.json", {"files": list(evidence.files.values()), "assessment_complete": False})
        (report / "failure_traceback.txt").write_text(traceback.format_exc())
        (report / "SUMMARY.md").write_text("# Reproducibility review failed\n\n" + str(exc) + "\n\nSource files preserved; no experiment was started.\n")
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("project", "code", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.project, args.code, args.report))
