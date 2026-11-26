"""Read-only historical integrity checks; writes restricted to this package."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTECTED = ["models/v2", "models/v3", "data/processed/v2", "configs/v2", ".scratch/v2",
             "reports/v2", "src", "referencias_codex/notebook_atual_20261122", "reports/h2_feature_importance"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+"\n")


def protected_paths():
    paths = {p for name in PROTECTED for p in (ROOT/name).rglob("*") if p.is_file()}
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths.update(ROOT/name for name in tracked if name and (ROOT/name).is_file() and not (ROOT/name).is_relative_to(HERE))
    return sorted(paths)


def snapshot():
    path = HERE/"provenance/protected_files.csv"
    if path.exists(): raise RuntimeError("Snapshot already exists; do not replace.")
    path.parent.mkdir(exist_ok=True)
    rows = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in protected_paths()]
    pd.DataFrame(rows).to_csv(path, index=False)
    write_json(HERE/"provenance/registration.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(HERE/"protocol.json"),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip(),
        initial_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
        protected_count=len(rows), protected_bytes=sum(r["bytes"] for r in rows),
        python=platform.python_version(), platform=platform.platform(),
        dependencies={p:importlib.metadata.version(p) for p in ["lightgbm", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow", "matplotlib", "pytest"]}))
    print(f"Protocol registered; {len(rows)} historical files protected", flush=True)


def check_protocol():
    registration = json.loads((HERE/"provenance/registration.json").read_text())
    assert sha(HERE/"protocol.json") == registration["protocol_sha256"], "Protocol changed"
    return registration


def verify(write=True):
    check_protocol()
    saved = pd.read_csv(HERE/"provenance/protected_files.csv")
    failures = [r.path for r in saved.itertuples() if not (ROOT/r.path).exists() or sha(ROOT/r.path) != r.sha256]
    failures += sorted(set(saved.path) ^ {str(p.relative_to(ROOT)) for p in protected_paths()})
    result = dict(utc=datetime.now(timezone.utc).isoformat(), protected_files=len(saved),
                  unchanged=not failures, failures=failures,
                  git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode())
    if write: write_json(HERE/"provenance/preservation.json", result)
    assert not failures, failures
    print(f"UNCHANGED: {len(saved)} protected files", flush=True)
    return result


def manifest(check=False):
    path = HERE/"manifest.sha256"
    files = sorted(p for p in HERE.rglob("*") if p.is_file() and p != path)
    if not check:
        path.write_text("".join(f"{sha(p)}  {p.relative_to(HERE)}\n" for p in files))
    else:
        rows = [line.split("  ",1) for line in path.read_text().splitlines()]
        assert {name for _,name in rows} == {str(p.relative_to(HERE)) for p in files}
        for expected,name in rows: assert sha(HERE/name) == expected, name
    print(f"Manifest {'verified' if check else 'written'}: {len(files)} artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "verify", "manifest", "check-manifest"])
    action = parser.parse_args().action
    if action == "snapshot": snapshot()
    elif action == "verify": verify()
    else: manifest(action == "check-manifest")
