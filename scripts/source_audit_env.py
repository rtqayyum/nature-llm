"""Project-local workbook/RAR inspection dependencies, separate from inference."""
from pathlib import Path
import subprocess
import sys

from acquire import atomic_json


def ensure(project, code, report):
    project, code, report = project.resolve(), code.resolve(), report.resolve()
    code.relative_to(project)
    report.relative_to(project)
    prefix = project / ".venv-source-audit"
    prefix.resolve().relative_to(project)
    python = prefix / "bin/python"
    report.mkdir(parents=True, exist_ok=True)
    status = {"phase": "environment_check", "prefix": str(prefix)}
    path = report / "source_environment_status.json"
    def execute(args, **kwargs):
        print("+", " ".join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), check=True, **kwargs)
    atomic_json(path, status)
    if not python.exists():
        execute([sys.executable, "-m", "venv", "--copies", prefix])
    execute([python, "-c", "import sys; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve()", prefix])
    check = "import openpyxl,rarfile,importlib.metadata as m; assert (openpyxl.__version__,rarfile.__version__,m.version('et_xmlfile'))==('3.1.5','4.2','2.0.0')"
    if subprocess.run([str(python), "-c", check], capture_output=True).returncode:
        status["phase"] = "install_inspection_dependencies"
        atomic_json(path, status)
        cache = project / ".cache/pip"
        cache.mkdir(parents=True, exist_ok=True)
        execute([python, "-m", "pip", "--isolated", "--cache-dir", cache, "install", "--no-user", "--disable-pip-version-check",
                 "-r", code / "config/source_audit_requirements.txt"])
    execute([python, "-c", check])
    execute([python, "-m", "pip", "check"])
    with (report / "source_environment_freeze.txt").open("w") as out:
        execute([python, "-m", "pip", "freeze"], stdout=out)
    status["phase"] = "READY"
    atomic_json(path, status)
    return python
