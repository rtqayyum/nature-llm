"""Resume the frozen local CPU extractor in a persistent work directory and bundle receipts."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from collect_text import MODEL, N_SHARDS, PROMPT_SHA, ROOT, batch, inputs
from read_extractions import read

REV = "bb5d59e06d9551d752d08b292a50eb208b07ab1f"
WEIGHTS = ("dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
           "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a")
RUNTIME_SHA = "ab56524ce899526e51c04e77f089a5792c59282b49736b239900970d3947be47"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def obtain(path, url, expected):
    if path.exists():
        if digest(path)!=expected:
            raise ValueError(f"Checksum mismatch for cached {path.name}")
        return
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        download = Path(tmp)/path.name
        subprocess.run(["curl","--fail","--location","--retry","3","--silent","--show-error",
                        "--output",str(download),url],check=True)
        if digest(download)!=expected:
            raise ValueError(f"Checksum mismatch for downloaded {path.name}")
        download.replace(path)


@contextmanager
def server(cache):
    names=[]
    for i, expected in enumerate(WEIGHTS,1):
        name=f"qwen2.5-7b-instruct-q4_k_m-0000{i}-of-00002.gguf"
        obtain(cache/name,f"https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/{REV}/{name}?download=true",expected)
        names.append(name)
    archive=cache/"runtime.tgz"
    obtain(archive,"https://github.com/ggml-org/llama.cpp/releases/download/b12721/llama-b12721-bin-ubuntu-x64.tar.gz",RUNTIME_SHA)
    # Always unpack the binary and its sibling libraries from the checked archive.
    with tempfile.TemporaryDirectory(dir=cache) as tmp, (cache/"server.log").open("w") as log:
        subprocess.run(["tar","xzf",str(archive),"-C",tmp],check=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0))
            port=sock.getsockname()[1]
        url=f"http://127.0.0.1:{port}"
        args=[str(Path(tmp)/"llama-b12721/llama-server"),"-m",str(cache/names[0]),"--alias","nyc-qwen",
              "--host","127.0.0.1","--port",str(port),"-ngl","0","-np","4","-c","8192","-t","6","-tb","6","--no-warmup"]
        bypass=",".join(filter(None,[os.environ.get("no_proxy",os.environ.get("NO_PROXY","")),"127.0.0.1"]))
        os.environ["no_proxy"]=os.environ["NO_PROXY"]=bypass
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}. See {log.name}")
                try:
                    with urlopen(url+"/health",timeout=2) as response:
                        if json.load(response).get("status")=="ok":
                            break
                except (URLError,TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f"Server readiness timed out. See {log.name}")
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def run(work, limit):
    cache, shards = work/"cache",work/"parser_shards"
    for folder in (cache,shards):
        folder.mkdir(parents=True,exist_ok=True)
    notices=json.loads((ROOT.parent/"role_screen/sources.json").read_text())
    notices["may_notices.json"]=json.loads((ROOT/"sources.json").read_text())["may_notices.json"]
    for name,info in notices.items():
        obtain(cache/name,info["url"],info["sha256"])
    cases=inputs(cache,cache/"may_notices.json")
    for path in (ROOT/"parser_shards").glob("[0-9][0-9].json.gz"):
        dest=shards/path.name
        if dest.exists():
            if digest(dest)!=digest(path):
                raise ValueError("Published and local extractions differ at "+path.name)
        else:
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                copied=Path(tmp)/path.name
                shutil.copyfile(path,copied)
                copied.replace(dest)
    prompt=ROOT.parent/"may_parser/prompt_examples.txt"
    if digest(prompt)!=PROMPT_SHA:
        raise ValueError("Prompt checksum mismatch")
    receipt_path=work/"progress.json"
    started=datetime.now(timezone.utc).isoformat()
    generated=[]
    history=work/"generation_receipts"
    history.mkdir(exist_ok=True)
    run_name=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json"
    def checkpoint():
        _,report=read(cases,cache/"stations",shards,require_complete=False)
        report.update(job_id=os.environ.get("SLURM_JOB_ID"),started_utc=started,
                      checkpoint_utc=datetime.now(timezone.utc).isoformat(),generated_this_invocation=generated.copy())
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            receipt=Path(tmp)/receipt_path.name
            receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
            if generated:
                run_record=dict(model=MODEL, prompt_sha256=PROMPT_SHA, runtime_sha256=RUNTIME_SHA,
                                weight_sha256=WEIGHTS, runner_sha256=digest(Path(__file__)),
                                collector_sha256=digest(ROOT/"collect_text.py"),
                                inference_sha256=digest(ROOT.parent/"may_parser/infer.py"), job_id=report["job_id"], started_utc=started,
                                checkpoint_utc=report["checkpoint_utc"],
                                generated_files={f"{n:02}.json.gz":report["files"][f"{n:02}.json.gz"] for n in generated})
                entry=Path(tmp)/run_name
                entry.write_text(json.dumps(run_record,indent=2,sort_keys=True)+"\n")
                entry.replace(history/run_name)
            bundle=Path(tmp)/"may_extractions.tar.gz"
            with tarfile.open(bundle,"w:gz") as tar:
                tar.add(receipt,arcname=receipt.name)
                for entry in sorted(history.glob("*.json")):
                    previous=json.loads(entry.read_text())
                    if any(report["files"].get(name)!=meta for name,meta in previous["generated_files"].items()):
                        raise ValueError("Generation receipt conflicts with saved shards")
                    tar.add(entry,arcname="generation_receipts/"+entry.name)
                for name in report["files"]:
                    tar.add(shards/name,arcname="parser_shards/"+name)
            bundle.replace(work/bundle.name)
            receipt.replace(receipt_path)
        print(f"Verified {report['verified_inputs']}/{report['total_inputs']} distinct inputs",flush=True)
        return report
    report=checkpoint()
    pending=[n for n in range(N_SHARDS) if f"{n:02}.json.gz" not in report["files"]]
    if limit is not None:
        pending=pending[:limit]
    if pending:
        with server(cache) as url:
            for n in pending:
                batch(cases,prompt.read_text().strip(),url,n,shards/f"{n:02}.json.gz")
                generated.append(n)
                checkpoint()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("--limit",type=int,help="Maximum new shards in this invocation; zero verifies and packages the checkpoint")
    a=p.parse_args()
    if a.limit is not None and a.limit<0:
        p.error("limit must be nonnegative")
    work=a.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    with (work/"worker.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(work,a.limit)
