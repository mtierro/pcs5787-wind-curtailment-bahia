"""Snapshot protected inputs, assert unchanged contents, and manifest outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTECTED = ["models/v2", "models/v3", "data/processed/v2", "configs/v2", ".scratch/v2",
             "reports/v2", "src", "referencias_codex/notebook_atual_20261122"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, obj):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def protected_paths():
    files = set()
    for directory in PROTECTED:
        files.update(p for p in (ROOT / directory).rglob("*") if p.is_file())
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files.update(ROOT / p for p in tracked if p and (ROOT / p).is_file() and not (ROOT / p).is_relative_to(HERE))
    return sorted(files)


def snapshot():
    path = HERE / "provenance/historical_inputs.csv"
    if path.exists():
        raise RuntimeError("Snapshot already exists; use verify, never replace baseline.")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in protected_paths()]
    pd.DataFrame(rows).to_csv(path, index=False)
    packages = {p: importlib.metadata.version(p) for p in
                ["lightgbm", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow", "matplotlib", "pytest"]}
    json_write(HERE / "provenance/baseline.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT), protected_directories=PROTECTED,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
        python=platform.python_version(), platform=platform.platform(), packages=packages,
        cpu_count=os.cpu_count(), protocol_sha256=sha(HERE / "protocol.json"),
        files=len(rows), bytes=sum(r["bytes"] for r in rows)))
    print(f"Protected {len(rows)} files, {sum(r['bytes'] for r in rows):,} bytes", flush=True)


def verify(write=True):
    saved = pd.read_csv(HERE / "provenance/historical_inputs.csv")
    expected = set(saved.path)
    current = {str(p.relative_to(ROOT)) for p in protected_paths()}
    failures = []
    for row in saved.itertuples():
        p = ROOT / row.path
        if not p.exists() or p.stat().st_size != row.bytes or sha(p) != row.sha256:
            failures.append(row.path)
    failures.extend(sorted(expected ^ current))
    baseline = json.loads((HERE / "provenance/baseline.json").read_text())
    assert sha(HERE / "protocol.json") == baseline["protocol_sha256"], "Protocol changed"
    result = dict(utc=datetime.now(timezone.utc).isoformat(), files_verified=len(saved),
                  unchanged=not failures, failures=failures,
                  git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode())
    if write:
        json_write(HERE / "provenance/preservation.json", result)
    assert not failures, failures
    print(f"UNCHANGED: {len(saved)} protected files", flush=True)
    return result


def manifest(check=False):
    target = HERE / "manifest.sha256"
    if check:
        lines = target.read_text().splitlines()
        for line in lines:
            digest, name = line.split("  ", 1)
            assert sha(HERE / name) == digest, name
        actual = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file() and p != target}
        assert actual == {line.split("  ", 1)[1] for line in lines}, "Unmanifested files"
        print(f"Manifest valid: {len(lines)} artifacts")
    else:
        files = sorted(p for p in HERE.rglob("*") if p.is_file() and p != target)
        target.write_text("".join(f"{sha(p)}  {p.relative_to(HERE)}\n" for p in files))
        print(f"Manifest: {len(files)} artifacts (excludes itself)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "verify", "manifest", "check-manifest"])
    action = parser.parse_args().action
    if action == "snapshot": snapshot()
    elif action == "verify": verify()
    else: manifest(action == "check-manifest")
