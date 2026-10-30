"""Project-local email preferences and submission receipts; no credentials."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from email.message import EmailMessage

MAIL_TYPES = "BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT,TIME_LIMIT_80,ARRAY_TASKS"


def preferences(project):
    path = project / "work/notification_preferences.json"
    address = os.environ.get("TRAFFIC_MAIL_USER")
    if address is None and path.exists():
        address = json.loads(path.read_text())["mail_user"]
    if not address:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", address):
        raise ValueError("Invalid notification email address")
    data = {"mail_user": address, "mail_type": MAIL_TYPES}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)
    return data


def queued_receipt(project, submission, settings):
    """Slurm has no QUEUED mail event. Try the cluster's local mail service."""
    if not settings:
        return {"state": "NOT_CONFIGURED"}
    subject = "[nature-llm] QUEUED job " + submission["job_id"] + " " + submission["job_name"]
    body = (subject + "\n\nSubmission accepted by Slurm; this does not mean it has started.\n" +
        "Allocation per task: %s CPU(s), %s GPU(s), time %s\n" % (submission["cpus"], submission["gpus"], submission["time_limit"]) +
        "Check: bash " + str(project / "Check") + " " + submission["job_id"] + "\n" +
        "Reports: " + str(project / "results/slurm_jobs" / submission["job_id"]) + "\n")
    try:
        sendmail = shutil.which("sendmail")
        if sendmail:
            message = EmailMessage()
            message["To"] = settings["mail_user"]
            message["Subject"] = subject
            message.set_content(body)
            subprocess.run([sendmail, "-t", "-oi"], input=message.as_string(), text=True, capture_output=True, timeout=20, check=True)
            return {"state": "ACCEPTED_BY_LOCAL_MAIL_SERVICE", "backend": "sendmail", "delivery_verified": False}
        for name in ("mailx", "mail"):
            executable = shutil.which(name)
            if executable:
                subprocess.run([executable, "-s", subject, settings["mail_user"]], input=body,
                    text=True, capture_output=True, timeout=20, check=True)
                return {"state": "ACCEPTED_BY_LOCAL_MAIL_SERVICE", "backend": name, "delivery_verified": False}
        return {"state": "UNAVAILABLE", "reason": "No sendmail/mailx/mail on submission node; Slurm event emails remain requested"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"state": "FAILED", "reason": str(exc), "delivery_verified": False}
